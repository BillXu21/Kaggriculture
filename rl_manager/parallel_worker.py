"""Spawn-safe CPU rollout worker entrypoint.

Importing this module must never initialize an accelerator.  The worker owns
only engines, openings, executors, and a local trajectory shard.  Policy
requests are sent to the parent owner through ``InferenceRequest`` messages.
"""

from __future__ import annotations

import os
import sys
import time
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rl_manager.parallel_protocol import (
    EpisodeAssignment,
    InferenceRequest,
    InferenceResponse,
    WorkerFailed,
    WorkerFinished,
    WorkerTask,
    policy_row_request_id,
)
from rl_manager.runner import EpisodeSpec, SelfPlayRunner
from rl_manager.types import PolicyIdentity, PolicyOutputs


FORBIDDEN_ACCELERATOR_MODULES = (
    "jax", "jaxlib", "torch_xla", "bc_manager_jax", "optax")


def assert_cpu_worker_imports(owner_pid: int) -> None:
    """Fail before rollout construction if a child owns accelerator modules."""
    loaded = sorted(
        name for name in FORBIDDEN_ACCELERATOR_MODULES
        if name in sys.modules or any(
            module_name.startswith(name + ".") for module_name in sys.modules))
    if os.getpid() == int(owner_pid):
        raise RuntimeError("parallel rollout worker started in TPU owner process")
    if loaded:
        raise RuntimeError(
            "CPU rollout worker imported accelerator modules before startup: "
            f"{loaded}; only the parent TPU owner may import JAX/libtpu")


class RemotePlanPolicy:
    """Worker-side synchronous proxy for one immutable policy identity."""

    def __init__(self, identity: PolicyIdentity, request_queue: Any,
                 response_queue: Any, worker_id: int) -> None:
        self.identity = identity
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._worker_id = int(worker_id)
        self._request_context: list[tuple[int, int, int]] | None = None

    def set_request_context(
            self, rows: Sequence[tuple[int, int, int]]) -> None:
        """Receive the runner's exact batch row identities before forwarding."""
        self._request_context = [tuple(map(int, row)) for row in rows]

    def plan_batch(self, inputs: Mapping[str, np.ndarray],
                   prng_id: str) -> PolicyOutputs:
        if self._request_context is None:
            raise RuntimeError(
                "remote policy request context missing; runner protocol drift")
        rows = self._request_context
        self._request_context = None
        batch_size = int(np.asarray(next(iter(inputs.values()))).shape[0])
        if len(rows) != batch_size:
            raise ValueError(
                f"remote policy context has {len(rows)} rows, expected "
                f"{batch_size}")
        requests: list[InferenceRequest] = []
        for row, (episode_index, seat, day) in enumerate(rows):
            request_id = policy_row_request_id(
                episode_index, seat, day, self.identity)
            row_inputs = {
                key: np.ascontiguousarray(np.asarray(value[row:row + 1]))
                for key, value in inputs.items()}
            requests.append(InferenceRequest(
                request_id=request_id,
                worker_id=self._worker_id,
                episode_index=episode_index,
                seat=seat,
                day=day,
                policy_identity=self.identity,
                prng_id=str(prng_id),
                inputs=row_inputs,
                queued_at=time.perf_counter()))
        for request in requests:
            self._request_queue.put(request)

        row_outputs: list[PolicyOutputs] = []
        expected = {request.request_id for request in requests}
        received: dict[str, PolicyOutputs] = {}
        while expected - received.keys():
            response = self._response_queue.get()
            if isinstance(response, WorkerFailed):
                raise RuntimeError(
                    f"inference owner failure: {response.error_message}")
            if not isinstance(response, InferenceResponse):
                raise RuntimeError(
                    f"worker {self._worker_id} received unexpected IPC message "
                    f"{type(response).__name__}")
            if response.request_id not in expected:
                raise RuntimeError(
                    f"worker {self._worker_id} received response for unknown "
                    f"request {response.request_id!r}")
            received[response.request_id] = response.outputs
        for request in requests:
            row_outputs.append(received[request.request_id])
        return _stack_row_outputs(row_outputs)


def _stack_row_outputs(outputs: Sequence[PolicyOutputs]) -> PolicyOutputs:
    if not outputs:
        raise ValueError("cannot stack empty policy output list")
    first = outputs[0]
    action_tensors = {
        key: np.concatenate([np.asarray(output.action_tensors[key])
                             for output in outputs], axis=0)
        for key in first.action_tensors}
    logprob_groups = {
        key: np.concatenate([np.asarray(output.logprob_groups[key]).reshape(1)
                             for output in outputs])
        for key in first.logprob_groups}
    return PolicyOutputs(
        action_tensors=action_tensors,
        logprob_groups=logprob_groups,
        logprob_total=np.asarray([output.logprob_total[0]
                                  for output in outputs], dtype=np.float32),
        value=np.asarray([output.value[0] for output in outputs],
                         dtype=np.float32),
        batch_size=len(outputs))


def _factory_from_wire(factory: Any, *, low_telemetry: bool = False) -> Any:
    if factory == "executor_v0@default":
        from rl_manager.executor_factory import make_default_executor_factory
        return make_default_executor_factory()
    if factory == "executor_v0@default-low-telemetry":
        from executor_v0.agent import AgentConfig
        from rl_manager.executor_factory import make_default_executor_factory

        return make_default_executor_factory(
            AgentConfig(strict=True, record_turn_snapshot=False))
    return factory


def _assignment_specs(
        assignments: Sequence[EpisodeAssignment], request_queue: Any,
        response_queue: Any, worker_id: int) -> list[EpisodeSpec]:
    policies: dict[PolicyIdentity, RemotePlanPolicy] = {}
    specs = []
    for assignment in assignments:
        seat_policies = []
        for identity in assignment.seat_policy_identities:
            policy = policies.get(identity)
            if policy is None:
                policy = RemotePlanPolicy(
                    identity, request_queue, response_queue, worker_id)
                policies[identity] = policy
            seat_policies.append(policy)
        specs.append(EpisodeSpec(
            episode_index=assignment.episode_index,
            seed=assignment.seed,
            composition=assignment.composition,
            policies=(seat_policies[0], seat_policies[1]),
            trainable_seats=assignment.trainable_seats,
            controlled_seat=assignment.controlled_seat))
    return specs


def worker_main(task_queue: Any, request_queue: Any, response_queue: Any,
                result_queue: Any) -> None:
    """Top-level ``spawn`` target; every exception becomes a failure message."""
    task: WorkerTask | None = None
    try:
        task = task_queue.get()
        assert_cpu_worker_imports(task.owner_pid)
        specs = _assignment_specs(
            task.episodes, request_queue, response_queue, task.worker_id)
        trajectory = None
        if task.trajectory_capacity is not None:
            from rl_manager.trajectory import TrajectoryBuffer, e_input_spec
            trajectory = TrajectoryBuffer(
                capacity=int(task.trajectory_capacity), input_spec=e_input_spec())
        runner = SelfPlayRunner(
            task.runner_config, trajectory_buffer=trajectory,
            executor_factory=_factory_from_wire(
                task.executor_factory,
                low_telemetry=task.runner_config.low_telemetry),
            master_seed=task.master_seed)
        results = tuple(runner.run(specs))
        result_queue.put(WorkerFinished(task.worker_id, results, trajectory))
    except BaseException as exc:  # noqa: BLE001 - transport all worker errors
        worker_id = int(task.worker_id) if task is not None else -1
        result_queue.put(WorkerFailed(
            worker_id=worker_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback_module.format_exc()))
