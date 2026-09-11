"""Diagnostic wrapper for RL rollout/training throughput.

Run exactly like ``rl_manager.cli``::

    python -m tools.profile_rl_throughput train ...

The wrapper monkey-patches timing only; it does not change rollout decisions,
seeds, batching, or optimizer settings.  It prints machine-readable lines:

    THROUGHPUT_PROFILE {...}
    THROUGHPUT_PHASE {...}

Set ``KAGGRICULTURE_PROFILE_SYNC=1`` for a deep diagnostic pass that explicitly
waits for JAX output leaves inside ``PPOPolicy.act``.  That mode intentionally
serializes the measurement boundary and should be used only for short profiling
runs, not production throughput.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


_ACTIVE_RUNNER: Any | None = None
_SYNC_JAX = os.environ.get("KAGGRICULTURE_PROFILE_SYNC", "0") == "1"


def _emit(kind: str, payload: Mapping[str, Any]) -> None:
    print(f"{kind} {json.dumps(dict(payload), sort_keys=True, allow_nan=False)}",
          flush=True)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


class _SystemSampler:
    """Low-rate host sampler used only while a rollout is active."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = float(interval)
        self.cores_used: list[float] = []
        self.rss_gib: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore
        except ImportError:
            self.psutil = None
        else:
            self.psutil = psutil

    def start(self) -> None:
        if self.psutil is None:
            return
        self.psutil.cpu_percent(interval=None, percpu=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 4.0))

    def _loop(self) -> None:
        assert self.psutil is not None
        parent = self.psutil.Process(os.getpid())
        while not self._stop.wait(self.interval):
            try:
                percpu = self.psutil.cpu_percent(interval=None, percpu=True)
                self.cores_used.append(sum(float(v) for v in percpu) / 100.0)
                processes = [parent] + parent.children(recursive=True)
                rss = 0
                for process in processes:
                    try:
                        rss += int(process.memory_info().rss)
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        pass
                self.rss_gib.append(rss / (1024.0 ** 3))
            except Exception:
                # Profiling must never break a training run.
                pass

    def summary(self) -> dict[str, float | None]:
        return {
            "cpu_cores_mean": (statistics.fmean(self.cores_used)
                               if self.cores_used else None),
            "cpu_cores_p50": _percentile(self.cores_used, 0.50),
            "cpu_cores_p95": _percentile(self.cores_used, 0.95),
            "cpu_cores_max": max(self.cores_used) if self.cores_used else None,
            "rss_gib_mean": (statistics.fmean(self.rss_gib)
                             if self.rss_gib else None),
            "rss_gib_max": max(self.rss_gib) if self.rss_gib else None,
            "samples": len(self.cores_used),
        }


def _row_input_bytes(request: Any) -> int:
    return sum(int(np.asarray(value).nbytes)
               for value in request.inputs.values())


def _policy_output_bytes(outputs: Any) -> int:
    total = 0
    for value in outputs.action_tensors.values():
        total += int(np.asarray(value).nbytes)
    for value in outputs.logprob_groups.values():
        total += int(np.asarray(value).nbytes)
    total += int(np.asarray(outputs.logprob_total).nbytes)
    total += int(np.asarray(outputs.value).nbytes)
    return total


def _worker_timing_summary(results: Sequence[Any], specs: Sequence[Any],
                           num_workers: int) -> dict[str, Any]:
    """Recover each worker's final cumulative runner timers from results.

    ``SelfPlayRunner.timing_totals`` is monotonic within one worker and each
    EpisodeResult stores a snapshot.  Parallel assignment is round-robin by
    input position, so the per-key maximum for that worker is its final total.
    """
    position = {int(spec.episode_index): index for index, spec in enumerate(specs)}
    by_worker: dict[int, list[Any]] = defaultdict(list)
    for result in results:
        pos = position.get(int(result.episode_index))
        if pos is not None:
            by_worker[pos % num_workers].append(result)

    per_worker: dict[int, dict[str, float]] = {}
    keys: set[str] = set()
    for worker_id, worker_results in by_worker.items():
        keys.update(
            key for result in worker_results
            for key in getattr(result, "timing_seconds", {}).keys())
        per_worker[worker_id] = {}
        for key in keys:
            per_worker[worker_id][key] = max(
                (float(result.timing_seconds.get(key, 0.0))
                 for result in worker_results), default=0.0)

    summed = {
        key: sum(worker.get(key, 0.0) for worker in per_worker.values())
        for key in sorted(keys)
    }
    worker_total = {
        worker_id: sum(values.values())
        for worker_id, values in per_worker.items()
    }
    return {
        "workers_observed": len(per_worker),
        "sum_seconds": summed,
        "max_worker_accounted_seconds": (
            max(worker_total.values()) if worker_total else None),
        "mean_worker_accounted_seconds": (
            statistics.fmean(worker_total.values()) if worker_total else None),
    }


def _install_parallel_profile() -> None:
    import rl_manager.parallel as parallel

    original_slice = parallel._slice_outputs
    original_dispatch = parallel.ParallelSelfPlayRunner._dispatch
    original_run = parallel.ParallelSelfPlayRunner.run

    def profiled_slice(outputs: Any, row: int) -> Any:
        t0 = time.perf_counter()
        result = original_slice(outputs, row)
        elapsed = time.perf_counter() - t0
        runner = _ACTIVE_RUNNER
        if runner is not None:
            runner._profile_output_materialize_seconds += elapsed
            runner._profile_output_logical_bytes += _policy_output_bytes(result)
        return result

    def profiled_dispatch(self: Any, key: Any, requests: list[Any],
                          first_queued: float, policy_by_identity: Mapping[Any, Any],
                          response_queues: Sequence[Any]) -> Any:
        # The original method recursively chunks oversized request lists.  Only
        # count the physical chunks to avoid double-counting the outer wrapper.
        if (self.max_batch is not None and requests
                and len(requests) > self.max_batch):
            return original_dispatch(
                self, key, requests, first_queued, policy_by_identity,
                response_queues)

        logical_input_bytes = sum(_row_input_bytes(request) for request in requests)
        physical_count = self.fixed_batch or len(requests)
        row_bytes = _row_input_bytes(requests[0]) if requests else 0
        physical_input_bytes = row_bytes * int(physical_count)
        before_inference = float(self.inference_metrics["inference_seconds"])
        t0 = time.perf_counter()
        result = original_dispatch(
            self, key, requests, first_queued, policy_by_identity,
            response_queues)
        elapsed = time.perf_counter() - t0
        inference_delta = (
            float(self.inference_metrics["inference_seconds"]) - before_inference)
        self._profile_dispatch_seconds += elapsed
        self._profile_policy_call_return_seconds += inference_delta
        self._profile_input_logical_bytes += logical_input_bytes
        self._profile_input_physical_bytes += physical_input_bytes
        return result

    def profiled_run(self: Any, specs: Sequence[Any]) -> list[Any]:
        global _ACTIVE_RUNNER
        self._profile_dispatch_seconds = 0.0
        self._profile_policy_call_return_seconds = 0.0
        self._profile_output_materialize_seconds = 0.0
        self._profile_input_logical_bytes = 0
        self._profile_input_physical_bytes = 0
        self._profile_output_logical_bytes = 0
        self._profile_ppo_act_return_seconds = 0.0
        self._profile_ppo_act_sync_seconds = 0.0
        self._profile_ppo_act_calls = 0

        sampler = _SystemSampler()
        sampler.start()
        previous_runner = _ACTIVE_RUNNER
        _ACTIVE_RUNNER = self
        t0 = time.perf_counter()
        try:
            results = original_run(self, specs)
        finally:
            wall = time.perf_counter() - t0
            _ACTIVE_RUNNER = previous_runner
            sampler.stop()

        inference = dict(self.inference_metrics)
        real_requests = int(inference.get("real_requests", 0))
        physical_calls = int(inference.get("physical_inference_calls", 0))
        policy_return = float(self._profile_policy_call_return_seconds)
        materialize = float(self._profile_output_materialize_seconds)
        dispatch = float(self._profile_dispatch_seconds)
        parent_other = max(0.0, dispatch - policy_return - materialize)
        queue_wait = float(inference.get("queue_wait_seconds", 0.0))

        payload = {
            "kind": "rollout",
            "games": len(specs),
            "wall_seconds": wall,
            "games_per_second": (len(specs) / wall if wall else None),
            "manager_rows": real_requests,
            "manager_rows_per_second": (real_requests / wall if wall else None),
            "host": sampler.summary(),
            "worker_timing": _worker_timing_summary(
                results, specs, int(self.num_workers)),
            "parent": {
                "dispatch_wall_seconds": dispatch,
                "policy_call_return_seconds": policy_return,
                "output_materialize_seconds": materialize,
                "dispatch_other_python_ipc_seconds": parent_other,
                "queue_wait_sum_seconds": queue_wait,
                "queue_wait_mean_ms": (
                    1000.0 * queue_wait / real_requests if real_requests else None),
                "physical_inference_calls": physical_calls,
                "real_requests": real_requests,
                "physical_rows": int(inference.get("physical_rows", 0)),
                "padding_rows": int(inference.get("padding_rows", 0)),
                "occupancy": float(inference.get("occupancy", 0.0)),
                "input_logical_gib": self._profile_input_logical_bytes /
                                     (1024.0 ** 3),
                "input_physical_estimated_gib": self._profile_input_physical_bytes /
                                                (1024.0 ** 3),
                "output_logical_gib": self._profile_output_logical_bytes /
                                      (1024.0 ** 3),
                "ppo_act_calls": int(self._profile_ppo_act_calls),
                "ppo_act_return_seconds": float(
                    self._profile_ppo_act_return_seconds),
                "ppo_act_sync_seconds": float(self._profile_ppo_act_sync_seconds),
                "explicit_jax_sync": bool(_SYNC_JAX),
            },
        }
        _emit("THROUGHPUT_PROFILE", payload)
        return results

    parallel._slice_outputs = profiled_slice
    parallel.ParallelSelfPlayRunner._dispatch = profiled_dispatch
    parallel.ParallelSelfPlayRunner.run = profiled_run


def _install_ppo_act_profile() -> None:
    from rl_manager.ppo_policy import PPOPolicy

    original_act = PPOPolicy.act

    def profiled_act(self: Any, *args: Any, **kwargs: Any) -> Any:
        runner = _ACTIVE_RUNNER
        t0 = time.perf_counter()
        result = original_act(self, *args, **kwargs)
        returned = time.perf_counter() - t0
        sync_seconds = 0.0
        if _SYNC_JAX:
            import jax

            sync_start = time.perf_counter()
            for leaf in jax.tree_util.tree_leaves(result):
                block = getattr(leaf, "block_until_ready", None)
                if callable(block):
                    block()
            sync_seconds = time.perf_counter() - sync_start
        if runner is not None:
            runner._profile_ppo_act_calls += 1
            runner._profile_ppo_act_return_seconds += returned
            runner._profile_ppo_act_sync_seconds += sync_seconds
        return result

    PPOPolicy.act = profiled_act


def _install_phase_profiles() -> None:
    import rl_manager.ppo as ppo
    import rl_manager.ppo_adapter as adapter
    import rl_manager.ppo_checkpoint as checkpoint

    original_audit = adapter.recompute_stored_action_logprobs
    original_update = ppo.ppo_update
    original_save = checkpoint.save_ppo_checkpoint

    def timed_audit(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_audit(*args, **kwargs)
        _emit("THROUGHPUT_PHASE", {
            "phase": "logprob_audit",
            "seconds": time.perf_counter() - t0,
            "rows": int(np.asarray(result).shape[0]),
        })
        return result

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_update(*args, **kwargs)
        _emit("THROUGHPUT_PHASE", {
            "phase": "ppo_update",
            "seconds": time.perf_counter() - t0,
        })
        return result

    def timed_save(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_save(*args, **kwargs)
        _emit("THROUGHPUT_PHASE", {
            "phase": "checkpoint_save",
            "seconds": time.perf_counter() - t0,
            "path": str(result),
        })
        return result

    adapter.recompute_stored_action_logprobs = timed_audit
    ppo.ppo_update = timed_update
    checkpoint.save_ppo_checkpoint = timed_save


def install() -> None:
    _install_parallel_profile()
    _install_ppo_act_profile()
    _install_phase_profiles()


def main() -> int:
    install()
    from rl_manager.cli import main as cli_main

    return int(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
