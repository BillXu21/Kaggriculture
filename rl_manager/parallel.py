"""Bounded spawned rollout topology with one parent inference owner.

The parent process owns the policy objects and is the only process allowed to
load JAX/libtpu.  Spawned workers receive only episode descriptors and local
CPU state.  Their manager-day NumPy requests are coalesced here, then routed
back by a stable ``episode/seat/day/policy`` identifier. The default scope is
policy/day; mixed-day policy scope and fixed physical padding are opt-ins.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import pickle
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from rl_manager.executor_factory import (
    EXECUTOR_FACTORY_VERSION,
    make_default_executor_factory,
)
from rl_manager.parallel_protocol import (
    EpisodeAssignment,
    InferenceRequest,
    InferenceResponse,
    WorkerFailed,
    WorkerFinished,
    WorkerTask,
)
from rl_manager.parallel_worker import worker_main
from rl_manager.provenance import backend_provenance, opening_provenance
from rl_manager.runner import (
    EpisodeResult,
    EpisodeSpec,
    RunnerConfig,
    SelfPlayRunner,
    TOTAL_MANAGER_DAYS,
    build_artifact_metadata,
)
from rl_manager.trajectory import TrajectoryBuffer, Transition
from rl_manager.types import BatchedPlanPolicy, PolicyIdentity, PolicyOutputs


class ParallelRolloutError(RuntimeError):
    """A worker or inference-owner protocol failure."""


BatchKey = PolicyIdentity | tuple[PolicyIdentity, int]


def _batch_key_sort_key(key: BatchKey) -> tuple[str, int]:
    if isinstance(key, PolicyIdentity):
        return (key.identity_id(), -1)
    return (key[0].identity_id(), int(key[1]))


def _factory_wire(factory: Any, *, low_telemetry: bool = False) -> Any:
    """Use a child-local default factory with its complete config."""
    if (getattr(factory, "name", None) == "executor_v0"
            and getattr(factory, "version", None) == EXECUTOR_FACTORY_VERSION):
        del low_telemetry
        config = getattr(factory, "agent_config", None)
        if config is None:
            raise ValueError(
                "registered executor_v0 factory is missing agent_config")
        return ("executor_v0@config", config)
    try:
        pickle.dumps(factory)
    except Exception as exc:  # noqa: BLE001 - turn pickle detail into API error
        raise ValueError(
            "parallel rollout executor factory must be pickleable under "
            "spawn, or be the registered executor_v0 default") from exc
    return factory


def _assignment(spec: EpisodeSpec) -> EpisodeAssignment:
    return EpisodeAssignment(
        episode_index=int(spec.episode_index), seed=int(spec.seed),
        composition=str(spec.composition),
        seat_policy_identities=(spec.policies[0].identity,
                                spec.policies[1].identity),
        trainable_seats=tuple(int(seat) for seat in spec.trainable_seats),
        controlled_seat=(None if spec.controlled_seat is None
                         else int(spec.controlled_seat)))


def _slice_outputs(outputs: PolicyOutputs, row: int) -> PolicyOutputs:
    return PolicyOutputs(
        action_tensors={key: np.ascontiguousarray(value[row:row + 1])
                        for key, value in outputs.action_tensors.items()},
        logprob_groups={key: np.asarray(value[row:row + 1]).copy()
                        for key, value in outputs.logprob_groups.items()},
        logprob_total=np.asarray(outputs.logprob_total[row:row + 1]).copy(),
        value=np.asarray(outputs.value[row:row + 1]).copy(),
        batch_size=1)


def _merge_shard(destination: TrajectoryBuffer, shard: TrajectoryBuffer,
                 seen: set[tuple[int, int, int]]) -> None:
    """Append a worker shard in canonical episode/seat/day order."""
    arrays = shard.finalize()
    order = sorted(
        range(len(shard)),
        key=lambda row: (int(arrays["episode_index"][row]),
                         int(arrays["seat"][row]), int(arrays["day"][row])))
    for source in order:
        key = (int(arrays["episode_index"][source]),
               int(arrays["seat"][source]), int(arrays["day"][source]))
        if key in seen:
            raise ParallelRolloutError(
                f"duplicate trajectory row received for episode/seat/day {key}")
        seen.add(key)
        inputs = {
            name: np.asarray(arrays[f"input_{name}"][source:source + 1])
            for name in shard.input_spec}
        actions = {
            name[len("action_"):]: np.asarray(
                arrays[name][source:source + 1])
            for name in arrays if name.startswith("action_")
        }
        logprob_groups = {
            name[len("logprob_"):]: float(arrays[name][source])
            for name in arrays if name.startswith("logprob_")
            and name != "logprob_total"}
        transition = Transition(
            episode_index=key[0], seed=int(arrays["seed"][source]),
            seat=key[1], day=key[2],
            trainable=bool(arrays["trainable"][source]), inputs=inputs,
            action_tensors=actions,
            logprob_groups=logprob_groups,
            logprob_total=float(arrays["logprob_total"][source]),
            value=float(arrays["value"][source]),
            trace_digest=bytes(np.asarray(
                arrays["trace_digest"][source], dtype=np.uint8).tolist()),
            truncated=bool(arrays["truncated"][source]))
        metadata = copy.deepcopy(shard.sidecar_records[source])
        destination.append(transition, metadata)
        target = len(destination) - 1
        destination.patch_trace_digest(target, transition.trace_digest)
        if bool(arrays["terminated"][source]) or float(arrays["reward"][source]):
            destination.patch_terminal(
                target, float(arrays["reward"][source]),
                bool(arrays["terminated"][source]))
        if bool(arrays["truncated"][source]):
            destination.patch_truncated(target)


class ParallelSelfPlayRunner:
    """Parent coordinator for CPU workers and one central policy owner."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        num_workers: int,
        trajectory_buffer: TrajectoryBuffer | None = None,
        executor_factory: Any | None = None,
        master_seed: int | None = None,
        request_queue_size: int | None = None,
        inference_batch_wait_seconds: float | None = None,
        max_inference_batch_size: int | None = None,
    ) -> None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) \
                or num_workers < 1:
            raise ValueError(f"num_workers must be a positive int, got {num_workers!r}")
        batch_wait = (config.inference_batch_wait_seconds
                      if inference_batch_wait_seconds is None
                      else inference_batch_wait_seconds)
        if not math.isfinite(batch_wait) or batch_wait < 0:
            raise ValueError(
                "inference_batch_wait_seconds must be finite and >= 0")
        if max_inference_batch_size is not None \
                and max_inference_batch_size < 1:
            raise ValueError("max_inference_batch_size must be >= 1")
        self.config = config
        self.num_workers = int(num_workers)
        self.buffer = trajectory_buffer
        if executor_factory is None:
            if config.low_telemetry:
                from executor_v0.agent import AgentConfig

                executor_factory = make_default_executor_factory(
                    AgentConfig(strict=True, record_turn_snapshot=False,
                                optional_spare_watering=True))
            else:
                executor_factory = make_default_executor_factory()
        self.executor_factory = executor_factory
        self.master_seed = master_seed
        self.request_queue_size = int(request_queue_size or max(4, num_workers * 4))
        fixed_batch = config.fixed_inference_batch_size
        if (fixed_batch is not None and max_inference_batch_size is not None
                and fixed_batch != max_inference_batch_size):
            raise ValueError(
                "max_inference_batch_size conflicts with configured "
                "fixed_inference_batch_size")
        self.batch_wait = float(batch_wait)
        self.max_batch = (fixed_batch if fixed_batch is not None
                          else max_inference_batch_size)
        self.fixed_batch = fixed_batch
        self.batch_scope = config.inference_batch_scope
        self.provenance: dict[str, Any] = {
            "opening": opening_provenance(config.opening),
            "backend": backend_provenance(config.backend_name,
                                           config.backend_configuration),
            "executor_factory": self.executor_factory,
            "executor_factory_version": getattr(
                self.executor_factory, "version", "unknown"),
            "master_seed": master_seed,
            "manager_start_day": config.manager_start_day,
            "inference_batch_scope": self.batch_scope,
            "fixed_inference_batch_size": self.fixed_batch,
            "inference_batch_wait_seconds": self.batch_wait,
        }
        self.inference_metrics: dict[str, Any] = {
            "requests": 0, "real_requests": 0,
            "batches": 0, "physical_inference_calls": 0,
            "batch_sizes": [], "real_batch_sizes": [],
            "physical_batch_sizes": [], "physical_rows": 0,
            "padding_rows": 0, "occupancy": 0.0,
            "queue_wait_seconds": 0.0, "inference_seconds": 0.0,
        }

    def run(self, specs: Sequence[EpisodeSpec]) -> list[EpisodeResult]:
        if not specs:
            return []
        if self.num_workers == 1:
            runner = SelfPlayRunner(
                self.config, trajectory_buffer=self.buffer,
                executor_factory=self.executor_factory,
                master_seed=self.master_seed)
            self.provenance = runner.provenance
            results = runner.run(specs)
            request_count = sum(result.transitions for result in results)
            self.inference_metrics["requests"] = request_count
            self.inference_metrics["real_requests"] = request_count
            return results

        if self.buffer is not None and len(self.buffer):
            raise ValueError("parallel trajectory destination must be empty")
        assignments = [_assignment(spec) for spec in specs]
        policy_by_identity: dict[PolicyIdentity, BatchedPlanPolicy] = {}
        for spec in specs:
            for policy in spec.policies:
                previous = policy_by_identity.get(policy.identity)
                if previous is not None and previous is not policy:
                    # Same identity is a fixed snapshot; using the first object
                    # prevents scheduling from selecting a different snapshot.
                    continue
                policy_by_identity[policy.identity] = policy
        factory_wire = _factory_wire(
            self.executor_factory, low_telemetry=self.config.low_telemetry)
        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue(maxsize=self.request_queue_size)
        result_queue = ctx.Queue()
        task_queues = [ctx.Queue(maxsize=1) for _ in range(self.num_workers)]
        response_queues = [ctx.Queue() for _ in range(self.num_workers)]
        processes = []
        shards: list[TrajectoryBuffer | None] = [None] * self.num_workers
        results_by_worker: dict[int, tuple[EpisodeResult, ...]] = {}
        groups: dict[int, list[EpisodeAssignment]] = defaultdict(list)
        for position, assignment in enumerate(assignments):
            groups[position % self.num_workers].append(assignment)
        shard_capacity = None
        if self.buffer is not None:
            shard_capacity = max(1, max(
                len(groups[worker]) * 2 * TOTAL_MANAGER_DAYS
                for worker in range(self.num_workers)))
        owner_pid = mp.current_process().pid
        for worker_id in range(self.num_workers):
            task = WorkerTask(
                worker_id=worker_id,
                episodes=tuple(groups[worker_id]),
                runner_config=self.config,
                executor_factory=factory_wire,
                master_seed=self.master_seed,
                trajectory_capacity=(shard_capacity
                                     if self.buffer is not None else None),
                owner_pid=owner_pid)
            process = ctx.Process(
                target=worker_main,
                args=(task_queues[worker_id], request_queue,
                      response_queues[worker_id], result_queue),
                name=f"kaggriculture-rollout-{worker_id}")
            processes.append(process)
            task_queues[worker_id].put(task)
            process.start()

        pending: dict[BatchKey, list[InferenceRequest]] = \
            defaultdict(list)
        pending_since: dict[BatchKey, float] = {}
        request_ids_seen: set[str] = set()
        try:
            while len(results_by_worker) < self.num_workers:
                try:
                    message = request_queue.get(timeout=0.01)
                except Empty:
                    message = None
                if isinstance(message, InferenceRequest):
                    if message.request_id in request_ids_seen:
                        raise ParallelRolloutError(
                            f"duplicate inference request {message.request_id!r}")
                    request_ids_seen.add(message.request_id)
                    key: BatchKey = (
                        message.policy_identity
                        if self.batch_scope == "policy"
                        else (message.policy_identity, int(message.day)))
                    pending[key].append(message)
                    pending_since.setdefault(key, time.perf_counter())
                elif message is not None:
                    raise ParallelRolloutError(
                        f"owner received unexpected request message "
                        f"{type(message).__name__}")

                now = time.perf_counter()
                for key in sorted(pending, key=_batch_key_sort_key):
                    if (now - pending_since[key] >= self.batch_wait
                            or (self.max_batch is not None
                                and len(pending[key]) >= self.max_batch)):
                        self._dispatch(key, pending.pop(key),
                                       pending_since.pop(key), policy_by_identity,
                                       response_queues)

                while True:
                    try:
                        result_message = result_queue.get_nowait()
                    except Empty:
                        break
                    if isinstance(result_message, WorkerFailed):
                        raise ParallelRolloutError(
                            f"rollout worker {result_message.worker_id} failed "
                            f"with {result_message.error_type}: "
                            f"{result_message.error_message}\n"
                            f"{result_message.traceback}")
                    if not isinstance(result_message, WorkerFinished):
                        raise ParallelRolloutError(
                            f"owner received unexpected result message "
                            f"{type(result_message).__name__}")
                    if result_message.worker_id in results_by_worker:
                        raise ParallelRolloutError(
                            f"duplicate completion from worker {result_message.worker_id}")
                    results_by_worker[result_message.worker_id] = \
                        result_message.results
                    shards[result_message.worker_id] = result_message.trajectory

                for worker_id, process in enumerate(processes):
                    if not process.is_alive() and process.exitcode not in (0, None) \
                            and worker_id not in results_by_worker:
                        raise ParallelRolloutError(
                            f"rollout worker {worker_id} exited with code "
                            f"{process.exitcode} without a failure message")

            # A worker cannot complete while it is waiting for an unserved
            # request, but force-drain defensively to make protocol failures
            # explicit instead of silently dropping a row.
            for key in list(pending):
                self._dispatch(key, pending.pop(key), pending_since.pop(key),
                               policy_by_identity, response_queues)
            if pending:
                raise ParallelRolloutError("inference requests remained pending")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            for queue in task_queues + response_queues:
                queue.close()
            request_queue.close()
            result_queue.close()

        if self.buffer is not None:
            seen: set[tuple[int, int, int]] = set()
            for shard in shards:
                if shard is None:
                    raise ParallelRolloutError("worker returned no trajectory shard")
                _merge_shard(self.buffer, shard, seen)
        results = [result for worker_id in sorted(results_by_worker)
                   for result in results_by_worker[worker_id]]
        by_index: dict[int, EpisodeResult] = {}
        for result in results:
            if result.episode_index in by_index:
                raise ParallelRolloutError(
                    f"duplicate episode result {result.episode_index}")
            by_index[result.episode_index] = result
        expected = {assignment.episode_index for assignment in assignments}
        if set(by_index) != expected:
            raise ParallelRolloutError(
                f"episode result set mismatch: expected {sorted(expected)}, "
                f"received {sorted(by_index)}")
        return [by_index[index] for index in sorted(by_index)]

    def _dispatch(
        self,
        key: BatchKey,
        requests: list[InferenceRequest],
        _first_queued: float,
        policy_by_identity: Mapping[PolicyIdentity, BatchedPlanPolicy],
        response_queues: Sequence[Any],
    ) -> None:
        if not requests:
            return
        # Sort before chunking. Queue arrival order is scheduler-dependent and
        # must never decide which rows share a physical policy call.
        requests = sorted(requests, key=lambda request: (
            request.episode_index, request.seat, request.day,
            request.request_id))
        if self.max_batch is not None and len(requests) > self.max_batch:
            for start in range(0, len(requests), self.max_batch):
                self._dispatch(key, requests[start:start + self.max_batch],
                               _first_queued, policy_by_identity,
                               response_queues)
            return
        identity = key if isinstance(key, PolicyIdentity) else key[0]
        policy = policy_by_identity.get(identity)
        if policy is None:
            raise ParallelRolloutError(f"no owner policy for identity {identity}")
        real_count = len(requests)
        physical_count = self.fixed_batch or real_count
        if physical_count < real_count:
            raise ParallelRolloutError(
                f"fixed inference batch size {physical_count} is smaller than "
                f"real request batch {real_count}")
        keys = sorted(requests[0].inputs)
        batch = {name: np.concatenate(
            [np.asarray(request.inputs[name]) for request in requests], axis=0)
                 for name in keys}
        row_ids = [request.request_id for request in requests]
        padding_count = physical_count - real_count
        if padding_count:
            for name in keys:
                source = np.asarray(batch[name])[0:1]
                padding = np.repeat(source, padding_count, axis=0)
                batch[name] = np.concatenate((batch[name], padding), axis=0)
            padding_prefix = "|".join(row_ids)
            row_ids.extend(
                f"padding/policy={identity.identity_id()}/"
                f"batch={padding_prefix}/slot={slot}"
                for slot in range(padding_count))
        # Row-aware policies derive stochasticity from each request ID. The
        # root remains snapshot-scoped so batch composition and padding do not
        # alter a real row's result.
        prng_id = f"parallel/policy={identity.identity_id()}"
        t0 = time.perf_counter()
        row_aware = getattr(policy, "plan_batch_with_row_ids", None)
        try:
            if callable(row_aware):
                outputs = row_aware(batch, row_ids, prng_id)
            else:
                outputs = policy.plan_batch(batch, prng_id)
        except Exception as exc:  # noqa: BLE001 - add owner-side context
            raise ParallelRolloutError(
                f"central policy {identity.identity_id()} failed: {exc}") from exc
        inference_seconds = time.perf_counter() - t0
        if outputs.batch_size != physical_count:
            raise ParallelRolloutError(
                f"policy {identity.identity_id()} returned batch "
                f"{outputs.batch_size}, expected {physical_count}")
        self.inference_metrics["requests"] += real_count
        self.inference_metrics["real_requests"] += real_count
        self.inference_metrics["batches"] += 1
        self.inference_metrics["physical_inference_calls"] += 1
        self.inference_metrics["batch_sizes"].append(real_count)
        self.inference_metrics["real_batch_sizes"].append(real_count)
        self.inference_metrics["physical_batch_sizes"].append(physical_count)
        self.inference_metrics["physical_rows"] += physical_count
        self.inference_metrics["padding_rows"] += padding_count
        self.inference_metrics["inference_seconds"] += inference_seconds
        physical_rows = self.inference_metrics["physical_rows"]
        self.inference_metrics["occupancy"] = (
            self.inference_metrics["real_requests"] / physical_rows
            if physical_rows else 0.0)
        self.inference_metrics["queue_wait_seconds"] += sum(
            max(0.0, time.perf_counter() - request.queued_at)
            for request in requests)
        for row, request in enumerate(requests):
            response_queues[request.worker_id].put(
                InferenceResponse(request.request_id, _slice_outputs(outputs, row)))

    def build_artifact_metadata(self, result: EpisodeResult) -> dict[str, Any]:
        return build_artifact_metadata(self.provenance, result)

    def save_trajectory_artifact(self, path: str | Path,
                                 buffer: TrajectoryBuffer,
                                 result: EpisodeResult) -> Path:
        return buffer.save(path, run_metadata=self.build_artifact_metadata(result))
