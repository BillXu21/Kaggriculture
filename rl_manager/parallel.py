"""Bounded spawned rollout topology with one parent inference owner.

The parent process owns the policy objects and is the only process allowed to
load JAX/libtpu.  Spawned workers receive only episode descriptors and local
CPU state.  Their manager-day NumPy requests are coalesced here, then routed
back by a stable ``episode/seat/day/policy`` identifier. The default scope is
policy/day; mixed-day policy scope and fixed physical padding are opt-ins.
"""

from __future__ import annotations

import copy
import dataclasses
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
    Stage25InferenceRequest,
    Stage25InferenceResponse,
    Stage25BootstrapRequest,
    Stage25BootstrapResponse,
    WorkerFailed,
    WorkerFinished,
    WorkerTask,
)
from rl_manager.parallel_worker import worker_main
from bc_manager.constants import TOTAL_DAYS
from rl_manager.provenance import backend_provenance, opening_provenance
from rl_manager.runner import (
    EpisodeResult,
    EpisodeSpec,
    RunnerConfig,
    SelfPlayRunner,
    build_artifact_metadata,
)
from rl_manager.trajectory import TrajectoryBuffer, Transition
from rl_manager.types import BatchedPlanPolicy, PolicyIdentity, PolicyOutputs
from rl_manager.stage25_types import (
    Stage25BehaviorIdentity, Stage25PolicyOutputs, stage25_rng_namespace)


class ParallelRolloutError(RuntimeError):
    """A worker or inference-owner protocol failure."""


BatchKey = (PolicyIdentity | Stage25BehaviorIdentity |
            tuple[PolicyIdentity, int])


def _batch_key_sort_key(key: BatchKey) -> tuple[str, int]:
    if hasattr(key, "identity_id"):
        return (key.identity_id(), -1)
    return (key[0].identity_id(), int(key[1]))


def pad_batch_to_physical(
        batch: Mapping[str, np.ndarray], physical_size: int
        ) -> tuple[dict[str, np.ndarray], int]:
    """Pad stacked `[N, ...]` arrays to one fixed physical batch.

    Single shared definition of rollout fixed-batch padding: short batches
    are extended to exactly `physical_size` rows by repeating row 0, the full
    physical batch is evaluated, and padded outputs are discarded by the
    caller. Returns the padded batch plus the padding-row count.
    """
    if (isinstance(physical_size, bool) or not isinstance(physical_size, int)
            or physical_size < 1):
        raise ValueError(
            "physical batch size must be a positive int, "
            f"got {physical_size!r}")
    arrays = {name: np.asarray(array) for name, array in batch.items()}
    if not arrays:
        raise ValueError("cannot pad an empty batch")
    real_count = int(next(iter(arrays.values())).shape[0])
    if real_count < 1:
        raise ValueError("cannot pad an empty batch")
    if real_count > physical_size:
        raise ValueError(
            f"real batch {real_count} exceeds physical batch size "
            f"{physical_size}")
    padding_count = physical_size - real_count
    if not padding_count:
        return dict(arrays), 0
    padded = {}
    for name, array in arrays.items():
        padding = np.repeat(array[0:1], padding_count, axis=0)
        padded[name] = np.concatenate((array, padding), axis=0)
    return padded, padding_count


def _factory_wire(factory: Any, *, low_telemetry: bool = False) -> Any:
    """Use a child-local default factory with its complete config."""
    if (getattr(factory, "name", None) == "stage25_executor"):
        return ("stage25_executor@config", factory.agent_config)
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
    curricula = []
    for policy in spec.policies:
        curriculum = getattr(policy, "curriculum", None)
        if curriculum is None:
            curriculum = getattr(getattr(policy, "config", None),
                                 "curriculum", None)
        if curriculum is None:
            curricula.append(None)
        elif isinstance(curriculum, Mapping):
            curricula.append(dict(curriculum))
        else:
            try:
                curricula.append(dataclasses.asdict(curriculum))
            except (TypeError, dataclasses.FrozenInstanceError) as exc:
                raise ValueError(
                    "Stage 2.5 curriculum must be an explicit dataclass or mapping"
                ) from exc
    return EpisodeAssignment(
        episode_index=int(spec.episode_index), seed=int(spec.seed),
        composition=str(spec.composition),
        seat_policy_identities=(spec.policies[0].identity,
                                spec.policies[1].identity),
        trainable_seats=tuple(int(seat) for seat in spec.trainable_seats),
        controlled_seat=(None if spec.controlled_seat is None
                         else int(spec.controlled_seat)),
        stage25_curricula=(curricula[0], curricula[1]))


def _slice_outputs(outputs: PolicyOutputs, row: int) -> PolicyOutputs:
    return PolicyOutputs(
        action_tensors={key: np.ascontiguousarray(value[row:row + 1])
                        for key, value in outputs.action_tensors.items()},
        logprob_groups={key: np.asarray(value[row:row + 1]).copy()
                        for key, value in outputs.logprob_groups.items()},
        logprob_total=np.asarray(outputs.logprob_total[row:row + 1]).copy(),
        value=np.asarray(outputs.value[row:row + 1]).copy(),
        batch_size=1)


def _slice_stage25_outputs(
        outputs: Stage25PolicyOutputs, row: int) -> Stage25PolicyOutputs:
    return Stage25PolicyOutputs(
        classes=outputs.classes[row:row + 1],
        component_logprobs=outputs.component_logprobs[row:row + 1],
        joint_logprob=outputs.joint_logprob[row:row + 1],
        value=outputs.value[row:row + 1],
        decoded_goals=outputs.decoded_goals[row:row + 1],
        valid=outputs.valid[row:row + 1],
        policy_identity=outputs.policy_identity, batch_size=1)


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
        stage25_trajectory_buffer: Any | None = None,
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
        self.stage25_trajectory = stage25_trajectory_buffer
        if executor_factory is None:
            if config.stage25_enabled:
                from rl_manager.executor_factory import make_stage25_executor_factory

                executor_factory = make_stage25_executor_factory()
            elif config.low_telemetry:
                from executor_v0.agent import AgentConfig

                executor_factory = make_default_executor_factory(
                    AgentConfig(strict=True, record_turn_snapshot=False,
                                optional_spare_watering=True))
            else:
                executor_factory = make_default_executor_factory()
        self.executor_factory = executor_factory
        self.master_seed = master_seed
        self.request_queue_size = int(request_queue_size or max(4, num_workers * 4))
        fixed_batch = (config.stage25_fixed_inference_batch_size
                       if config.stage25_enabled
                       else config.fixed_inference_batch_size)
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
            "e_history_version": config.e_history_version,
            "reward_config": config.reward_config.to_json_dict(),
            "inference_batch_scope": self.batch_scope,
            "fixed_inference_batch_size": self.fixed_batch,
            "inference_batch_wait_seconds": self.batch_wait,
        }
        self.inference_metrics: dict[str, Any] = {
            "requests": 0, "real_requests": 0, "logical_requests": 0,
            "bootstrap_requests": 0,
            "mixed_request_batches": 0,
            "batches": 0, "physical_inference_calls": 0,
            "batch_sizes": [], "real_batch_sizes": [],
            "physical_batch_sizes": [], "physical_rows": 0,
            "padding_rows": 0, "occupancy": 0.0,
            "animal_placement_rows": 0,
            "animal_placement_nonzero_rows": 0,
            "animal_placement_classes": 0,
            "animal_placement_nonzero_classes": 0,
            "queue_wait_seconds": 0.0, "inference_seconds": 0.0,
        }

    def run(self, specs: Sequence[EpisodeSpec]) -> list[EpisodeResult]:
        if not specs:
            return []
        if self.num_workers == 1:
            runner = SelfPlayRunner(
                self.config, trajectory_buffer=self.buffer,
                executor_factory=self.executor_factory,
                master_seed=self.master_seed,
                stage25_trajectory=self.stage25_trajectory)
            self.provenance = runner.provenance
            results = runner.run(specs)
            self.inference_metrics.update(runner.inference_metrics)
            return results

        if ((self.buffer is not None and len(self.buffer)) or
                (self.stage25_trajectory is not None and
                 len(self.stage25_trajectory))):
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
        stage25_shard_capacity = None
        if self.buffer is not None:
            shard_capacity = max(1, max(
                len(groups[worker]) * 2
                * (TOTAL_DAYS - self.config.manager_start_day)
                for worker in range(self.num_workers)))
        if self.stage25_trajectory is not None:
            stage25_shard_capacity = max(1, max(
                len(groups[worker]) * 2
                * (TOTAL_DAYS - self.config.manager_start_day)
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
                owner_pid=owner_pid,
                stage25_trajectory_capacity=(
                    stage25_shard_capacity
                    if self.stage25_trajectory is not None else None))
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
                elif isinstance(message, Stage25InferenceRequest):
                    if message.request_id in request_ids_seen:
                        raise ParallelRolloutError(
                            f"duplicate Stage 2.5 request {message.request_id!r}")
                    request_ids_seen.add(message.request_id)
                    # Stage 2.5 batches are grouped by immutable behavior
                    # identity; unlike legacy policy-day routing, day mixing
                    # is safe because K/context travel with every row.
                    key = message.identity.behavior_identity
                    pending[key].append(message)
                    pending_since.setdefault(key, time.perf_counter())
                elif isinstance(message, Stage25BootstrapRequest):
                    if message.request_id in request_ids_seen:
                        raise ParallelRolloutError(
                            f"duplicate Stage 2.5 bootstrap request {message.request_id!r}")
                    request_ids_seen.add(message.request_id)
                    key = message.identity.behavior_identity
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

        if self.stage25_trajectory is not None:
            for shard in shards:
                if shard is None:
                    raise ParallelRolloutError(
                        "worker returned no Stage 2.5 trajectory shard")
                for row in shard.rows:
                    self.stage25_trajectory.append(row)
        elif self.buffer is not None:
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
        # A queue can contain a decision and its truncation bootstrap at the
        # same boundary. Partition by request kind before dispatching; queue
        # position must never determine the response type.
        stage25_decisions = [
            request for request in requests
            if isinstance(request, Stage25InferenceRequest)]
        stage25_bootstraps = [
            request for request in requests
            if isinstance(request, Stage25BootstrapRequest)]
        legacy = [
            request for request in requests
            if isinstance(request, InferenceRequest)]
        if len(stage25_decisions) + len(stage25_bootstraps) + len(legacy) != len(requests):
            raise ParallelRolloutError("unknown inference request type")
        if stage25_decisions and stage25_bootstraps:
            self.inference_metrics["mixed_request_batches"] += 1
        if stage25_decisions:
            self._dispatch_stage25(
                key, stage25_decisions, _first_queued, policy_by_identity,
                response_queues)
        if stage25_bootstraps:
            self._dispatch_stage25_bootstrap(
                key, stage25_bootstraps, _first_queued, policy_by_identity,
                response_queues)
        if not legacy:
            return
        requests = legacy
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
        batch, padding_count = pad_batch_to_physical(batch, physical_count)
        row_ids = [request.request_id for request in requests]
        if padding_count:
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

    def _dispatch_stage25(
        self,
        key: BatchKey,
        requests: list[Stage25InferenceRequest],
        _first_queued: float,
        policy_by_identity: Mapping[Any, Any],
        response_queues: Sequence[Any],
    ) -> None:
        """Batch Stage 2.5 rows while preserving each row's K/support payload."""
        requests = sorted(requests, key=lambda request: (
            request.identity.episode_index, request.identity.seat,
            request.identity.day, request.request_id))
        if self.max_batch is not None and len(requests) > self.max_batch:
            for start in range(0, len(requests), self.max_batch):
                self._dispatch_stage25(
                    key, requests[start:start + self.max_batch], _first_queued,
                    policy_by_identity, response_queues)
            return
        policy = policy_by_identity.get(key)
        if policy is None:
            # A Stage 2.5 key may be represented by a PolicyIdentity while the
            # request carries the stronger behavior identity.
            policy = next((candidate for candidate in policy_by_identity.values()
                           if getattr(candidate, "behavior_identity", None) ==
                           requests[0].identity.behavior_identity or
                           getattr(candidate, "identity", None) ==
                           requests[0].identity.behavior_identity), None)
        if policy is None:
            raise ParallelRolloutError(
                "no owner policy for Stage 2.5 behavior identity "
                f"{requests[0].identity.behavior_identity.identity_id()}")
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
        batch, padding_count = pad_batch_to_physical(batch, physical_count)
        first = requests[0]
        padded = [first] * padding_count
        physical_requests = requests + padded
        capacities = np.concatenate([
            np.asarray(request.crop_capacity, dtype=np.int16)
            for request in physical_requests], axis=0)
        contexts = [request.physical_context for request in physical_requests]
        supports = [request.support for request in physical_requests]
        # The request identity is authoritative. The explicit rollout seed
        # is already part of the shared namespace; appending it here would
        # diverge from SelfPlayRunner's local row token.
        row_ids = [request.request_id for request in requests]
        row_ids.extend(
            f"padding/behavior={first.identity.behavior_identity.identity_id()}"
            f"/batch={'|'.join(request.request_id for request in requests)}"
            f"/slot={slot}" for slot in range(padding_count))
        prng_id = stage25_rng_namespace(
            first.identity.behavior_identity, getattr(policy, "seed", 0))
        t0 = time.perf_counter()
        outputs = SelfPlayRunner._stage25_policy_batch(
            policy, batch, capacities, contexts, supports, row_ids, prng_id)
        inference_seconds = time.perf_counter() - t0
        if outputs.batch_size != physical_count:
            raise ParallelRolloutError(
                "Stage 2.5 owner returned an unexpected physical batch size")
        expected_identity = first.identity.behavior_identity
        if outputs.policy_identity != expected_identity:
            raise ParallelRolloutError(
                "Stage 2.5 owner returned the wrong behavior identity")
        self.inference_metrics["requests"] += real_count
        self.inference_metrics["real_requests"] += real_count
        self.inference_metrics["logical_requests"] += real_count
        self.inference_metrics["batches"] += 1
        self.inference_metrics["physical_inference_calls"] += 1
        self.inference_metrics["batch_sizes"].append(real_count)
        self.inference_metrics["real_batch_sizes"].append(real_count)
        self.inference_metrics["physical_batch_sizes"].append(physical_count)
        self.inference_metrics["physical_rows"] += physical_count
        self.inference_metrics["padding_rows"] += padding_count
        real_classes = np.asarray(outputs.classes[:real_count])
        self.inference_metrics["animal_placement_rows"] += real_count
        self.inference_metrics["animal_placement_nonzero_rows"] += int(
            np.count_nonzero(np.any(real_classes[:, 1:4] > 0, axis=1)))
        self.inference_metrics["animal_placement_classes"] += real_count * 3
        self.inference_metrics["animal_placement_nonzero_classes"] += int(
            np.count_nonzero(real_classes[:, 1:4] > 0))
        self.inference_metrics["inference_seconds"] += inference_seconds
        physical_rows = self.inference_metrics["physical_rows"]
        self.inference_metrics["occupancy"] = (
            self.inference_metrics["logical_requests"] / physical_rows
            if physical_rows else 0.0)
        self.inference_metrics["queue_wait_seconds"] += sum(
            max(0.0, time.perf_counter() - request.queued_at)
            for request in requests)
        for row, request in enumerate(requests):
            response_queues[request.worker_id].put(
                Stage25InferenceResponse(
                    request.request_id, request.identity,
                    _slice_stage25_outputs(outputs, row)))

    def _dispatch_stage25_bootstrap(
        self,
        key: BatchKey,
        requests: list[Stage25BootstrapRequest],
        _first_queued: float,
        policy_by_identity: Mapping[Any, Any],
        response_queues: Sequence[Any],
    ) -> None:
        """Batch value-only truncation requests with fixed physical padding."""
        requests = sorted(requests, key=lambda request: (
            request.identity.episode_index, request.identity.seat,
            request.identity.day, request.request_id))
        if self.max_batch is not None and len(requests) > self.max_batch:
            for start in range(0, len(requests), self.max_batch):
                self._dispatch_stage25_bootstrap(
                    key, requests[start:start + self.max_batch], _first_queued,
                    policy_by_identity, response_queues)
            return
        policy = policy_by_identity.get(key)
        if policy is None:
            policy = next((candidate for candidate in policy_by_identity.values()
                           if getattr(candidate, "behavior_identity", None) == key or
                           getattr(candidate, "identity", None) == key), None)
        if policy is None:
            raise ParallelRolloutError(
                "no owner policy for Stage 2.5 bootstrap identity "
                f"{key.identity_id() if hasattr(key, 'identity_id') else key}")
        real_count = len(requests)
        physical_count = self.fixed_batch or real_count
        if physical_count < real_count:
            raise ParallelRolloutError(
                f"fixed inference batch size {physical_count} is smaller than "
                f"real bootstrap batch {real_count}")
        keys = sorted(requests[0].inputs)
        batch = {name: np.concatenate(
            [np.asarray(request.inputs[name]) for request in requests], axis=0)
                 for name in keys}
        batch, padding_count = pad_batch_to_physical(batch, physical_count)
        first = requests[0]
        physical_requests = requests + [first] * padding_count
        capacities = np.concatenate([
            np.asarray(request.crop_capacity, dtype=np.int16)
            for request in physical_requests], axis=0)
        contexts = [request.physical_context for request in physical_requests]
        row_ids = [request.request_id for request in requests]
        row_ids.extend(
            f"padding/bootstrap={key.identity_id()}/slot={slot}"
            for slot in range(padding_count))
        value_fn = getattr(policy, "bootstrap_value", None)
        if not callable(value_fn):
            raise ParallelRolloutError(
                "Stage 2.5 truncation requires parent value-only inference")
        t0 = time.perf_counter()
        raw = value_fn(
            inputs=batch, crop_capacity=capacities,
            physical_contexts=contexts, row_ids=row_ids)
        values = np.asarray(raw, dtype=np.float32)
        if values.shape != (physical_count,) or not np.all(np.isfinite(values)):
            raise ParallelRolloutError(
                "Stage 2.5 parent bootstrap must return finite float32 [B]")
        self.inference_metrics["batches"] += 1
        self.inference_metrics["physical_inference_calls"] += 1
        self.inference_metrics["logical_requests"] += real_count
        self.inference_metrics["bootstrap_requests"] += real_count
        self.inference_metrics["batch_sizes"].append(real_count)
        self.inference_metrics["real_batch_sizes"].append(real_count)
        self.inference_metrics["physical_batch_sizes"].append(physical_count)
        self.inference_metrics["physical_rows"] += physical_count
        self.inference_metrics["padding_rows"] += padding_count
        physical_rows = self.inference_metrics["physical_rows"]
        self.inference_metrics["occupancy"] = (
            self.inference_metrics["logical_requests"] / physical_rows
            if physical_rows else 0.0)
        self.inference_metrics["queue_wait_seconds"] += sum(
            max(0.0, time.perf_counter() - request.queued_at)
            for request in requests)
        self.inference_metrics["inference_seconds"] += (
            time.perf_counter() - t0)
        for row, request in enumerate(requests):
            response_queues[request.worker_id].put(
                Stage25BootstrapResponse(
                    request.request_id, request.identity,
                    np.asarray(values[row], dtype=np.float32), key))

    def build_artifact_metadata(self, result: EpisodeResult) -> dict[str, Any]:
        return build_artifact_metadata(self.provenance, result)

    def save_trajectory_artifact(self, path: str | Path,
                                 buffer: TrajectoryBuffer,
                                 result: EpisodeResult) -> Path:
        return buffer.save(path, run_metadata=self.build_artifact_metadata(result))
