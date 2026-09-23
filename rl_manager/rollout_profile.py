"""Lightweight, framework-neutral Stage 2.5 rollout profile helpers.

The profiler deliberately stores only a fixed set of aggregate counters.  It
must remain safe to import in spawned CPU workers and safe to serialize through
``multiprocessing`` and JSON without pulling in an accelerator framework.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ROLLOUT_PROFILE_SCHEMA_VERSION = 1

WORKER_TOP_LEVEL_SECONDS = (
    "worker_setup_seconds",
    "reset_setup_seconds",
    "manager_boundary_seconds",
    "agent_actions_seconds",
    "environment_seconds",
    "post_step_seconds",
    "finalization_seconds",
)

WORKER_DETAIL_SECONDS = (
    "stage25_identity_check_seconds",
    "stage25_daily_utilization_seconds",
    "stage25_provider_prepare_seconds",
    "stage25_manager_batch_build_seconds",
    "stage25_provider_accept_seconds",
    "stage25_trajectory_record_seconds",
    "remote_request_build_seconds",
    "remote_request_queue_put_seconds",
    "remote_response_wait_seconds",
    "remote_response_validate_stack_seconds",
    "observation_prepare_seconds",
    "action_agent_call_seconds",
    "action_hash_seconds",
    "action_trace_or_rollout_record_seconds",
    "fast_batch_step_seconds",
    "backend_slot_update_seconds",
    "observation_adapt_seconds",
    "canonical_state_seconds",
    "canonical_observations_seconds",
    "readonly_wrap_seconds",
    "observed_land_tracking_seconds",
    "track_post_step_seconds",
    "done_status_check_seconds",
    "finalize_total_seconds",
)

WORKER_COUNTS = (
    "episodes_assigned",
    "runner_chunks",
    "primitive_turns_processed",
    "active_env_turns",
    "manager_boundaries",
    "manager_rows",
    "executor_action_agent_calls",
    "native_batch_step_calls",
    "remote_requests",
    "remote_policy_batches",
    "remote_policy_rows",
)

PARENT_SECONDS = (
    "parallel_run_wall_seconds",
    "parent_dispatch_wall_seconds",
    "parent_result_merge_seconds",
    "owner_idle_request_wait_seconds",
    "parent_request_sort_seconds",
    "parent_input_concat_seconds",
    "parent_padding_seconds",
    "parent_capacity_context_build_seconds",
    "parent_policy_adapter_seconds",
    "parent_output_slice_seconds",
    "parent_response_queue_put_seconds",
    "parent_dispatch_total_seconds",
)

_WORKER_STATISTICS = (
    "worker_wall_seconds",
    "worker_process_cpu_seconds",
    "worker_cpu_wall_ratio",
    "manager_boundary_seconds",
    "remote_response_wait_seconds",
    "agent_actions_seconds",
    "environment_seconds",
    "finalization_seconds",
    "unclassified_residual_seconds",
    "episodes_assigned",
    "runner_chunks",
)


def new_worker_metrics() -> dict[str, float | int]:
    """Return one bounded worker accumulator with stable JSON-safe keys."""
    metrics: dict[str, float | int] = {
        "worker_id": -1,
        "worker_wall_seconds": 0.0,
        "worker_process_cpu_seconds": 0.0,
        "worker_cpu_wall_ratio": 0.0,
        "unclassified_residual_seconds": 0.0,
    }
    metrics.update({name: 0.0 for name in WORKER_TOP_LEVEL_SECONDS})
    metrics.update({name: 0.0 for name in WORKER_DETAIL_SECONDS})
    metrics.update({name: 0 for name in WORKER_COUNTS})
    return metrics


def new_parent_metrics() -> dict[str, float | int]:
    """Return the parent coordinator's fixed-size aggregate accumulator."""
    metrics: dict[str, float | int] = {name: 0.0 for name in PARENT_SECONDS}
    metrics.update({
        "num_workers": 0,
        "envs_per_worker": 0,
        "games_per_update": 0,
    })
    return metrics


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def metric_statistics(values: Sequence[float | int]) -> dict[str, float | int]:
    """Summarize one per-worker quantity without discarding imbalance."""
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0,
                "p90": 0.0, "max": 0.0}
    return {
        "count": len(numeric),
        "min": min(numeric),
        "median": percentile(numeric, 0.5),
        "mean": math.fsum(numeric) / len(numeric),
        "p90": percentile(numeric, 0.9),
        "max": max(numeric),
    }


def normalize_worker_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    """Validate and normalize a worker payload at the parent boundary."""
    expected = set(new_worker_metrics())
    unknown = set(metrics) - expected
    missing = expected - set(metrics)
    if unknown or missing:
        raise ValueError(
            f"worker timing metric schema mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
    normalized: dict[str, float | int] = {}
    integer_keys = {"worker_id", *WORKER_COUNTS}
    for name in expected:
        value = metrics[name]
        if name in integer_keys:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"worker timing metric {name} must be nonnegative int")
            normalized[name] = int(value)
        else:
            number = float(value)
            if not math.isfinite(number) or (
                    number < 0.0 and name != "unclassified_residual_seconds"):
                raise ValueError(
                    f"worker timing metric {name} must be finite and nonnegative")
            normalized[name] = number
    return normalized


def build_rollout_profile(
    worker_records: Sequence[Mapping[str, Any]],
    parent_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact machine-readable parent profile object."""
    records = [normalize_worker_metrics(record) for record in worker_records]
    records.sort(key=lambda record: int(record["worker_id"]))
    statistics = {
        name: metric_statistics([record[name] for record in records])
        for name in _WORKER_STATISTICS
    }
    summed = {
        name: math.fsum(float(record[name]) for record in records)
        for name in (*WORKER_TOP_LEVEL_SECONDS, *WORKER_DETAIL_SECONDS)
    }
    counts = {
        name: sum(int(record[name]) for record in records)
        for name in WORKER_COUNTS
    }
    def per_count(seconds: float, count_name: str, scale: float = 1000.0) -> float:
        count = int(counts[count_name])
        return seconds * scale / count if count else 0.0

    normalized = {
        "microseconds_per_active_env_turn": per_count(
            summed["environment_seconds"], "active_env_turns", 1_000_000.0),
        "milliseconds_per_executor_call": per_count(
            summed["action_agent_call_seconds"],
            "executor_action_agent_calls"),
        "milliseconds_per_batch_step": per_count(
            summed["fast_batch_step_seconds"],
            "native_batch_step_calls"),
        "milliseconds_per_manager_row": per_count(
            summed["manager_boundary_seconds"], "manager_rows"),
        "milliseconds_per_remote_policy_batch": per_count(
            summed["remote_response_wait_seconds"],
            "remote_policy_batches"),
    }
    if records:
        slowest = max(records, key=lambda record: float(record["worker_wall_seconds"]))
        fastest = min(records, key=lambda record: float(record["worker_wall_seconds"]))
        median_wall = float(statistics["worker_wall_seconds"]["median"])
        imbalance_ratio = (
            float(slowest["worker_wall_seconds"]) / median_wall
            if median_wall > 0.0 else 0.0)
        slowest_id = int(slowest["worker_id"])
        fastest_id = int(fastest["worker_id"])
    else:
        slowest_id = fastest_id = -1
        imbalance_ratio = 0.0
    parent = {name: float(parent_metrics.get(name, 0.0))
              for name in PARENT_SECONDS}
    parent.update({
        "num_workers": int(parent_metrics.get("num_workers", len(records))),
        "envs_per_worker": int(parent_metrics.get("envs_per_worker", 0)),
        "games_per_update": int(parent_metrics.get("games_per_update", 0)),
    })
    return {
        "schema_version": ROLLOUT_PROFILE_SCHEMA_VERSION,
        "worker": {
            "records": records,
            "statistics": statistics,
            "summed_seconds": summed,
            "slowest_worker_id": slowest_id,
            "fastest_worker_id": fastest_id,
            "critical_path_worker_id": slowest_id,
            "max_to_median_wall_ratio": imbalance_ratio,
        },
        "parent": parent,
        "counts": counts,
        "normalized": normalized,
    }
