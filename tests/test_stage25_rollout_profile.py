"""Focused contract tests for the opt-in Stage 2.5 rollout profiler."""

from __future__ import annotations

import json
import pickle

from rl_manager.parallel_protocol import WorkerFinished
from rl_manager.rollout_profile import (
    WORKER_TOP_LEVEL_SECONDS,
    build_rollout_profile,
    new_worker_metrics,
)
from rl_manager.runner import RunnerConfig, SelfPlayRunner
from rl_manager.stage25_ppo_cli import _parser


def _worker(worker_id: int, wall: float, cpu: float) -> dict[str, float | int]:
    record = new_worker_metrics()
    record["worker_id"] = worker_id
    record["worker_wall_seconds"] = wall
    record["worker_process_cpu_seconds"] = cpu
    record["worker_cpu_wall_ratio"] = cpu / wall
    record["worker_setup_seconds"] = 0.1
    record["reset_setup_seconds"] = 0.2
    record["manager_boundary_seconds"] = 0.3
    record["agent_actions_seconds"] = 0.4
    record["environment_seconds"] = 0.5
    record["post_step_seconds"] = 0.1
    record["finalization_seconds"] = 0.1
    accounted = sum(float(record[name]) for name in WORKER_TOP_LEVEL_SECONDS)
    record["unclassified_residual_seconds"] = wall - accounted
    return record


def test_profile_flag_defaults_off_and_can_be_enabled():
    args = _parser().parse_args([
        "--scratch", "--output-dir", "out",
    ])
    assert args.stage25_rollout_profile is False
    enabled = _parser().parse_args([
        "--scratch", "--stage25-rollout-profile", "--output-dir", "out",
    ])
    assert enabled.stage25_rollout_profile is True


def test_profile_off_has_no_accumulator_and_on_has_fixed_schema():
    assert RunnerConfig(stage25_enabled=True).stage25_rollout_profile is False
    assert SelfPlayRunner(RunnerConfig(stage25_enabled=True)).rollout_profile is None
    runner = SelfPlayRunner(RunnerConfig(
        stage25_enabled=True, stage25_rollout_profile=True))
    assert runner.rollout_profile is not None
    assert set(runner.rollout_profile) >= {
        "worker_wall_seconds", "remote_response_wait_seconds",
        "native_batch_step_calls", "finalize_total_seconds",
    }


def test_worker_finished_timing_payload_is_pickle_and_json_safe():
    metrics = _worker(2, 10.0, 6.0)
    message = WorkerFinished(2, (), None, metrics)
    restored = pickle.loads(pickle.dumps(message))
    assert restored.timing_metrics == metrics
    json.dumps(dict(restored.timing_metrics), allow_nan=False)


def test_parent_profile_aggregates_min_median_mean_p90_max_and_imbalance():
    profile = build_rollout_profile(
        [_worker(1, 4.0, 2.0), _worker(0, 2.0, 1.0), _worker(2, 6.0, 3.0)],
        {"num_workers": 3, "envs_per_worker": 4, "games_per_update": 12},
    )
    stats = profile["worker"]["statistics"]["worker_wall_seconds"]
    assert stats == {
        "count": 3, "min": 2.0, "median": 4.0, "mean": 4.0,
        "p90": 5.6, "max": 6.0,
    }
    assert profile["worker"]["slowest_worker_id"] == 2
    assert profile["worker"]["fastest_worker_id"] == 0
    assert profile["worker"]["max_to_median_wall_ratio"] == 1.5
    assert profile["parent"]["num_workers"] == 3


def test_top_level_worker_buckets_reconcile_with_explicit_residual():
    record = _worker(0, 10.0, 8.0)
    profile = build_rollout_profile([record], {})
    summed = profile["worker"]["summed_seconds"]
    assert sum(float(summed[name]) for name in WORKER_TOP_LEVEL_SECONDS) \
        + record["unclassified_residual_seconds"] == 10.0
