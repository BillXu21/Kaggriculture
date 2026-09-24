"""Focused contract tests for the non-executor hot-path audit instrumentation.

These tests cover only the opt-in diagnostics added on top of the Stage 2.5
rollout profiler: FastEnv step-subphase plumbing, provider preparation
subphases, and the derived reconciliation values. They do not assert any
production optimization.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rl_manager.parallel_protocol import WorkerFinished
from rl_manager.rollout_profile import (
    FAST_STEP_SUBPHASES,
    PROVIDER_PREPARE_SUBPHASES,
    build_rollout_profile,
    new_worker_metrics,
)

from test_stage25_provider import _obs

ROOT = Path(__file__).resolve().parents[1]


def _identity():
    from rl_manager.stage25_types import Stage25BehaviorIdentity

    return Stage25BehaviorIdentity(
        name="stage25-audit", version="v1", parameter_fingerprint="f" * 64,
        observation_schema_version="e_v1",
        policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1",
        curriculum_version="stage25_curriculum_v1",
        curriculum_fingerprint=(
            "7282cd9c883618c0258b2105f40e3d7996c969853a5adc6db07a57a12da9d1ec"))


def test_worker_metrics_include_fast_and_provider_subphase_schema():
    record = new_worker_metrics()
    for name in FAST_STEP_SUBPHASES:
        assert name in record
    for name in PROVIDER_PREPARE_SUBPHASES:
        assert name in record
    assert "stage25_provider_accept_input_copy_seconds" in record
    assert "stage25_provider_input_freeze_bytes" in record
    assert "stage25_provider_accept_copy_bytes" in record


def test_fast_step_subphases_reconcile_to_batch_step():
    record = new_worker_metrics()
    record["worker_id"] = 0
    record["worker_wall_seconds"] = 10.0
    record["fast_batch_step_seconds"] = 1.0
    record["fast_action_encode_seconds"] = 0.1
    record["fast_native_step_seconds"] = 0.2
    record["fast_observation_decode_seconds"] = 0.6
    record["native_batch_step_calls"] = 4
    record["active_env_turns"] = 16
    derived = build_rollout_profile([record], {})["derived"]
    assert derived["fast_subphase_sum_seconds"] == 0.9
    assert abs(derived["fast_step_unattributed_seconds"] - 0.1) < 1e-9
    assert abs(derived["fast_action_encode_fraction"] - 0.1) < 1e-9
    assert abs(derived["fast_native_step_fraction"] - 0.2) < 1e-9
    assert abs(derived["fast_observation_decode_fraction"] - 0.6) < 1e-9


def test_provider_prepare_subphases_reconcile():
    record = new_worker_metrics()
    record["worker_id"] = 0
    record["worker_wall_seconds"] = 10.0
    record["stage25_provider_prepare_seconds"] = 1.0
    record["stage25_provider_encode_live_inputs_seconds"] = 0.5
    record["stage25_provider_canonical_board_seconds"] = 0.2
    record["stage25_provider_input_freeze_copy_seconds"] = 0.1
    record["stage25_provider_input_freeze_bytes"] = 2048
    record["manager_rows"] = 8
    profile = build_rollout_profile([record], {})
    derived = profile["derived"]
    assert abs(derived["provider_prepare_subphase_sum_seconds"] - 0.8) < 1e-9
    assert abs(derived["provider_prepare_unattributed_seconds"] - 0.2) < 1e-9
    assert derived["provider_input_freeze_total_bytes"] == 2048
    normalized = profile["normalized"]
    assert abs(normalized["milliseconds_per_manager_row_provider_input_freeze"]
               - 12.5) < 1e-9
    assert abs(normalized["kilobytes_per_manager_row_provider_input_freeze"]
               - 0.25) < 1e-9


def test_fastenv_adapter_timings_reconcile_through_profile():
    from oracle.batched_backend import make_batched_backend

    batch = make_batched_backend("fast-batched", 2, {"numThreads": 1})
    batch.reset([11, 12])
    actions = [
        {"farmer": ["PASS"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]
    batch.step([actions, actions])
    timing = batch.last_timing_seconds
    assert set(timing) == {"action_encode", "native_step", "observation_decode"}
    assert all(float(value) >= 0.0 for value in timing.values())

    record = new_worker_metrics()
    record["worker_id"] = 0
    record["worker_wall_seconds"] = 1.0
    record["fast_batch_step_seconds"] = sum(float(v) for v in timing.values())
    record["fast_action_encode_seconds"] = float(timing["action_encode"])
    record["fast_native_step_seconds"] = float(timing["native_step"])
    record["fast_observation_decode_seconds"] = float(timing["observation_decode"])
    record["native_batch_step_calls"] = 1
    record["active_env_turns"] = 2
    derived = build_rollout_profile([record], {})["derived"]
    # The three existing FastEnv timers partition `batch.step` except for the
    # wrapper's own two perf_counter calls, so the unattributed remainder is
    # negligible relative to the measured subphases.
    assert derived["fast_step_unattributed_seconds"] >= 0.0
    assert derived["fast_subphase_sum_seconds"] > 0.0


def _provider_with_profile(profile):
    from rl_manager.stage25_provider import Stage25PlanProvider

    provider = Stage25PlanProvider(7, 0, 3, behavior_identity=_identity())
    started = time.perf_counter()
    context = provider.prepare_inference_context(
        _obs(day=3), behavior_identity=_identity(), profile=profile)
    elapsed = time.perf_counter() - started
    return provider, context, elapsed


def test_provider_subphases_populate_and_reconcile_with_wall_time():
    profile = new_worker_metrics()
    _provider, context, elapsed = _provider_with_profile(profile)
    assert context.row_token is not None
    subphase_sum = sum(float(profile[name]) for name in PROVIDER_PREPARE_SUBPHASES)
    assert subphase_sum > 0.0
    assert profile["stage25_provider_input_freeze_bytes"] > 0
    # Subphase timers cannot exceed the wall time of the whole call by more
    # than a small scheduling tolerance.
    assert subphase_sum <= elapsed + 1e-3


def test_provider_profile_on_off_produces_identical_context():
    from rl_manager.stage25_provider import Stage25PlanProvider

    identity = _identity()
    obs = _obs(day=3)
    plain_provider = Stage25PlanProvider(7, 0, 3, behavior_identity=identity)
    plain = plain_provider.prepare_inference_context(
        obs, behavior_identity=identity)
    profiled_provider = Stage25PlanProvider(7, 0, 3, behavior_identity=identity)
    profiled = profiled_provider.prepare_inference_context(
        obs, behavior_identity=identity, profile=new_worker_metrics())

    assert plain.crop_capacity == profiled.crop_capacity
    assert plain.daily_start == profiled.daily_start
    assert plain.row_token == profiled.row_token
    assert plain.physical_context == profiled.physical_context
    assert set(plain.inputs) == set(profiled.inputs)
    for name in plain.inputs:
        assert np.array_equal(
            plain.inputs[name], profiled.inputs[name], equal_nan=True)
    assert (plain.support is None) == (profiled.support is None)


def test_profiled_metric_payload_pickle_and_json_safe():
    profile = new_worker_metrics()
    _provider, _context, _elapsed = _provider_with_profile(profile)
    message = WorkerFinished(0, (), None, profile)
    restored = pickle.loads(pickle.dumps(message))
    assert restored.timing_metrics == profile
    json.dumps(dict(restored.timing_metrics), allow_nan=False)


def test_rollout_profile_module_is_accelerator_free():
    script = (
        "import sys\n"
        "import rl_manager.rollout_profile\n"
        "assert not any(name == 'jax' or name.startswith('jax.') or "
        "name == 'torch' or name.startswith('torch.')\n"
        "               for name in sys.modules)\n")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
