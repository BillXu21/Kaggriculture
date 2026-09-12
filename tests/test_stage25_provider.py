"""Focused tests for the persistent Stage 2.5 provider seam."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_CORRECTED_V1, E_HISTORY_LEGACY
from rl_manager.stage25_provider import (
    Stage25DecisionKey,
    Stage25DuplicateDecisionError,
    Stage25OutOfOrderError,
    Stage25PlanProvider,
    Stage25ProviderError,
    Stage25TerminalError,
)


ROOT = Path(__file__).resolve().parents[1]
HOLD = (0, 1, 0, 1, 100, 100, 100, 100, 100)


def _obs(day: int = 3, money: float = 3000.0) -> dict:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "yield_units": 0, "watered_today": True,
        "fertilized_until_day": -1, "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    tiles[0][1] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 0, "consecutive_unfed": 0, "fed_today": True,
    }
    tiles[0][2] = {
        "kind": "PASTURE", "animal": "SHEEP", "placed_day": 0,
        "yield_units": 0, "consecutive_unfed": 0, "fed_today": True,
    }
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0,
        "money": money, "tiles": tiles,
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day, "hour": 0, "step": day * 24,
        "farms": [farm, {**farm, "tiles": [row[:] for row in tiles]}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": []},
    }


class _FakeNativePolicy:
    def __init__(self, classes=HOLD):
        self.classes = classes
        self.calls: list[tuple[str, str, int]] = []

    def act(self, inputs, context, *, row_id, mode, seed):
        self.calls.append((row_id, mode, seed))
        assert inputs["crop_capacity"].shape == (1, 5)
        return self.classes


def test_first_boundary_initializes_observed_k_and_lowers_plan() -> None:
    provider = Stage25PlanProvider(episode_id="episode-1", seat=0,
                                   manager_start_day=3)
    plan = provider.accept_classes(_obs(), HOLD)

    assert provider.crop_capacity == (1, 0, 0, 0, 0)
    assert plan.crop_targets_dict["WHEAT"] == 1
    assert plan.land_count == 1
    assert all(value == 0 for value in plan.fertilizer_by_crop)
    assert all(value == 0 for value in plan.care_by_animal)
    assert all(value == 0 for row in plan.sell_quantities for value in row)
    assert provider.encoded_inputs is not None
    np.testing.assert_array_equal(
        provider.encoded_inputs["crop_capacity"], [[1, 0, 0, 0, 0]])
    assert provider.diagnostics["requested_classes"] == HOLD
    assert provider.diagnostics["requested_crop_goals"] == (1, 0, 0, 0, 0)


def test_same_day_reads_cache_but_duplicate_submission_is_rejected() -> None:
    native = _FakeNativePolicy()
    provider = Stage25PlanProvider(7, 0, 3, native_policy=native)
    first = provider.daily_plan(_obs(), 0)
    second = provider.daily_plan(_obs(money=999.0), 0)

    assert first is second
    assert len(native.calls) == 1
    before = provider.export_state()
    with pytest.raises(Stage25DuplicateDecisionError):
        provider.accept_classes(_obs(), HOLD)
    assert provider.export_state() == before


def test_sequential_crop_transition_and_corrected_e_history() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    provider.accept_classes(_obs(day=3, money=1000.0), HOLD)
    next_classes = (0, 1, 0, 1, 101, 100, 100, 100, 100)
    provider.accept_classes(_obs(day=4, money=1250.0), next_classes)

    assert provider.crop_capacity == (2, 0, 0, 0, 0)
    assert provider.e_history == (4, 1250.0)
    encoded = provider.encoded_inputs
    assert encoded is not None
    np.testing.assert_array_equal(encoded["crop_capacity"], [[1, 0, 0, 0, 0]])
    assert encoded["economic_context"][0, 13] == 1.0
    assert encoded["economic_context"][0, 12] > 0.0


def test_duplicate_out_of_order_terminal_and_invalid_delivery_are_atomic() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    before = provider.export_state()
    with pytest.raises(Stage25ProviderError):
        provider.accept_classes(_obs(), (0, 1, 0, 1, 0, 100, 100, 100, 100))
    assert provider.export_state() == before

    provider.accept_classes(_obs(), HOLD)
    before = provider.export_state()
    with pytest.raises(Stage25DuplicateDecisionError):
        provider.accept_classes(_obs(), HOLD,
                                decision_key=Stage25DecisionKey(7, 0, 3, "other"))
    assert provider.export_state() == before
    with pytest.raises(Stage25OutOfOrderError):
        provider.accept_classes(_obs(day=5), HOLD)
    assert provider.export_state() == before
    with pytest.raises(Stage25TerminalError):
        provider.accept_classes(_obs(day=4), HOLD, terminal=True)
    assert provider.export_state() == before


def test_reset_and_instance_isolation() -> None:
    left = Stage25PlanProvider("same", 0, 3)
    right = Stage25PlanProvider("same", 0, 3)
    left.accept_classes(_obs(), HOLD)
    assert right.crop_capacity is None
    assert right.cached_plan is None

    left.reset()
    assert left.crop_capacity is None
    assert left.cached_plan is None
    assert left.e_history is None
    left.accept_classes(_obs(), HOLD)
    assert left.crop_capacity == (1, 0, 0, 0, 0)


def test_export_import_is_strict_and_preserves_cached_plan() -> None:
    provider = Stage25PlanProvider(
        7, 0, 3, source_history_version=E_HISTORY_LEGACY)
    provider.accept_classes(_obs(), HOLD)
    encoded = provider.export_json()
    restored = Stage25PlanProvider(7, 0, 3)
    restored.import_state(encoded)

    assert restored.export_state() == provider.export_state()
    assert restored.cached_plan == provider.cached_plan
    assert restored.provenance["e_history_version"] == E_HISTORY_CORRECTED_V1
    assert restored.provenance["source_history_version"] == E_HISTORY_LEGACY
    assert "executor" not in json.loads(encoded)

    bad = json.loads(encoded)
    bad["version"] = "stage25_provider_state_v0"
    with pytest.raises(Stage25ProviderError, match="incompatible"):
        restored.import_state(bad)


def test_provider_derives_step_when_observation_omits_it() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    obs = _obs(day=3)
    obs.pop("step")  # official engine seat-1 observation shape
    provider.accept_classes(obs, HOLD)
    assert provider.crop_capacity == (1, 0, 0, 0, 0)


def test_provider_accepts_fast_engine_age_animal_tiles() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    obs = _obs(day=3)
    goose = obs["farms"][0]["tiles"][0][1]
    del goose["placed_day"]
    goose["age"] = 3
    provider.accept_classes(obs, HOLD)
    assert provider.crop_capacity == (1, 0, 0, 0, 0)


def test_age_and_placed_day_observations_encode_identically() -> None:
    canonical = Stage25PlanProvider(7, 0, 3)
    fast = Stage25PlanProvider(7, 0, 3)
    canonical.accept_classes(_obs(day=3, money=1000.0), HOLD)

    fast_obs = _obs(day=3, money=1000.0)
    goose = fast_obs["farms"][0]["tiles"][0][1]
    del goose["placed_day"]
    goose["age"] = 3
    sheep = fast_obs["farms"][0]["tiles"][0][2]
    del sheep["placed_day"]
    sheep["age"] = 3
    fast.accept_classes(fast_obs, HOLD)

    assert fast.diagnostics["requested_crop_goals"] == \
        canonical.diagnostics["requested_crop_goals"]
    canonical_inputs = canonical.encoded_inputs
    fast_inputs = fast.encoded_inputs
    assert canonical_inputs is not None and fast_inputs is not None
    for key in sorted(canonical_inputs):
        np.testing.assert_array_equal(canonical_inputs[key], fast_inputs[key], err_msg=key)
    # The caller-owned observation still carries the fast alias untouched.
    assert goose["age"] == 3 and "placed_day" not in goose


def test_external_provider_retains_explicit_curriculum() -> None:
    from rl_manager.stage25_config import Stage25CurriculumConfig

    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    provider = Stage25PlanProvider(7, 0, 3, curriculum=enabled)
    assert provider.effective_curriculum() == enabled
    provider.accept_classes(_obs(), HOLD)
    assert provider.provenance["curriculum_version"] == enabled.version
    assert provider.export_state()["curriculum"]["enabled"] is True


def test_native_checkpoint_curriculum_binds_and_rejects_mismatch(tmp_path) -> None:
    from rl_manager.stage25_checkpoint import save_stage25_inference_checkpoint
    from rl_manager.stage25_config import Stage25CurriculumConfig
    from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
    from rl_manager.stage25_provider import Stage25NativePolicy

    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    config = tiny_stage25_config(curriculum=enabled)
    params = init_stage25_params(config, seed=17)
    checkpoint = tmp_path / "enabled.npz"
    save_stage25_inference_checkpoint(checkpoint, params, config, seed=17)

    # An unspecified provider curriculum adopts the checkpoint curriculum and
    # uses exactly that support for sampling and validation.
    policy = Stage25NativePolicy(checkpoint, seed=17, mode="deterministic")
    provider = Stage25PlanProvider(7, 0, 3, native_policy=policy)
    assert provider.effective_curriculum() == enabled
    provider.daily_plan(_obs(), 0)
    assert provider.diagnostics["curriculum_version"] == enabled.version
    assert provider.provenance["curriculum"] == {
        "version": enabled.version, "enabled": True,
        "max_positive_crop_delta": 1,
        "max_land_expansion_per_decision": None,
        "max_animal_additions_per_species_per_decision": None,
    }

    # Explicit disagreement fails before sampling or state mutation.
    policy2 = Stage25NativePolicy(checkpoint, seed=17, mode="deterministic")
    policy2.act = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("native policy sampled despite curriculum mismatch"))
    mismatch = Stage25PlanProvider(
        7, 0, 3, native_policy=policy2,
        curriculum=Stage25CurriculumConfig())
    before = (mismatch.crop_capacity, mismatch.last_accepted_decision,
              mismatch.cached_plan, mismatch.e_history)
    with pytest.raises(Stage25ProviderError, match="curriculum"):
        mismatch.daily_plan(_obs(), 0)
    assert (mismatch.crop_capacity, mismatch.last_accepted_decision,
            mismatch.cached_plan, mismatch.e_history) == before

    # Explicit agreement is accepted.
    policy3 = Stage25NativePolicy(checkpoint, seed=17, mode="deterministic")
    agreed = Stage25PlanProvider(
        7, 0, 3, native_policy=policy3, curriculum=enabled)
    assert agreed.effective_curriculum() == enabled


def test_lightweight_provider_import_precedes_real_executor_import() -> None:
    script = """
import sys
import rl_manager.stage25_provider  # lightweight worker-side import first
assert 'torch' not in sys.modules, 'provider import pulled torch'
assert 'jax' not in sys.modules, 'provider import pulled jax'
import executor_v0
from executor_v0 import ExecutorAgent, PlanProvider, DailyPlan, generate_tasks
assert ExecutorAgent and PlanProvider and DailyPlan and generate_tasks
assert 'torch' in sys.modules, 'executor_v0 manager import did not run normally'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_external_import_and_acceptance_can_run_with_torch_and_jax_blocked() -> None:
    script = """
import builtins, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'jax' or name.startswith('jax.'):
        raise AssertionError('framework imported by external provider: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from rl_manager.stage25_provider import Stage25PlanProvider
tiles = [[None for _ in range(10)] for _ in range(10)]
tiles[0][0] = {'kind': 'PLANT', 'crop': 'WHEAT', 'planted_day': 0,
               'yield_units': 0, 'watered_today': True,
               'fertilized_until_day': -1, 'max_lifespan_step': -1,
               'consecutive_unwatered': 0}
tiles[0][1] = {'kind': 'COOP', 'animal': 'GOOSE', 'placed_day': 0,
               'yield_units': 0, 'consecutive_unfed': 0, 'fed_today': True}
tiles[0][2] = {'kind': 'PASTURE', 'animal': 'SHEEP', 'placed_day': 0,
               'yield_units': 0, 'consecutive_unfed': 0, 'fed_today': True}
farm = {'farmer': [0, 0], 'hands': [], 'hires_today': 0, 'money': 3000.0,
        'tiles': tiles, 'unlocked_quadrants': ['NW']}
obs = {'day': 3, 'hour': 0, 'step': 72, 'farms': [farm, farm],
       'market': {'inventory': {}, 'prices': {}}, 'town': {'unlocked_shops': []},
       'private': {'shed': {}, 'seeds': {}, 'inventories': []}}
p = Stage25PlanProvider(7, 0, 3)
p.accept_classes(obs, (0, 1, 0, 1, 100, 100, 100, 100, 100))
assert 'torch' not in sys.modules and 'jax' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
