"""Synthetic tests for tools/day_slice.py.

No real replay dependency: the fixture builds a synthetic replay whose
observations are produced by the compiled fast engine itself (all-PASS
play), guaranteeing exact boundary parity. Sibling modules
``tools.replay_io`` / ``tools.expert_plan`` are stubbed into ``sys.modules``
ONLY when the real modules are not importable, so these tests are
independent of the sibling worker's delivery timing.
"""

from __future__ import annotations

import copy
import json
import sys
import types

import pytest

import tools.day_slice as day_slice
from tools.day_slice import (
    SliceResult,
    first_diff,
    normalize_obs,
    run_day_slice,
)

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

SYNTHETIC_CONFIGURATION = {
    "episodeSteps": 720,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
    "seed": 7,
}

EPISODE_ID = 4242


def _make_synthetic_replay() -> dict:
    """Synthetic replay: steps 0..48, all-PASS, engine-generated observations.

    Steps 1..24 replay the day-0 prefix; steps 25..48 supply the recorded
    opponent actions for the 24 post-boundary turns of the day-1 slice.
    """
    from fast_env.api import FastKaggricultureEnv

    env = FastKaggricultureEnv(dict(SYNTHETIC_CONFIGURATION))
    observations = env.reset()
    current = [copy.deepcopy(observations[0]), copy.deepcopy(observations[1])]
    steps = []
    n_steps = 49
    for index in range(n_steps):
        step_entries = []
        for player in (0, 1):
            current[player]["day"] = index // 24
            current[player]["hour"] = index % 24
            current[player]["step"] = index
            step_entries.append({
                "action": dict(PASS_ACTION),
                "observation": copy.deepcopy(current[player]),
                "reward": 0,
                "status": "ACTIVE",
            })
        steps.append(step_entries)
        if index < n_steps - 1:
            current = env.step([dict(PASS_ACTION), dict(PASS_ACTION)])[0]
    return {
        "configuration": dict(SYNTHETIC_CONFIGURATION),
        "id": EPISODE_ID,
        "info": {"EpisodeId": EPISODE_ID, "seed": SYNTHETIC_CONFIGURATION["seed"]},
        "steps": steps,
    }


def _module_importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _tiny_all_zero_plan():
    from executor_v0.plan import SELL_BIN_ANCHORS, DailyPlan
    from replay_daily.constants import PRODUCTS

    return DailyPlan.create(
        crop_targets={},
        animal_targets={},
        land_count=1,
        fertilizer_by_crop={},
        care_by_animal={},
        sell_quantities={product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
                         for product in PRODUCTS},
    )


@pytest.fixture()
def synthetic_replay() -> dict:
    replay = _make_synthetic_replay()
    # Spec-mandated shape of the initial observation.
    obs0 = replay["steps"][0][0]["observation"]
    assert obs0["farms"][0]["money"] == 3000
    assert obs0["farms"][0]["farmer"] == [4, 4]
    assert obs0["farms"][0]["unlocked_quadrants"] == ["NW"]
    assert all(tile is None or tile == "LOCKED"
               for row in obs0["farms"][0]["tiles"] for tile in row)
    assert set(obs0["private"]["shed"]) and \
        all(v == 0 for v in obs0["private"]["shed"].values())
    assert all(v == 0 for v in obs0["private"]["seeds"].values())
    assert obs0["day"] == 0 and obs0["hour"] == 0 and obs0["step"] == 0
    return replay


@pytest.fixture()
def stub_sibling_modules(synthetic_replay):
    """Inject stub modules only for absent siblings; yield injected names."""
    injected: list[str] = []
    if not _module_importable("tools.expert_plan"):
        stub = types.ModuleType("tools.expert_plan")
        stub.extract_daily_plan = lambda replay, seat, day: _tiny_all_zero_plan()
        stub.boundary_observation = (
            lambda replay, seat, day:
            replay["steps"][day * 24][seat]["observation"])
        sys.modules["tools.expert_plan"] = stub
        injected.append("tools.expert_plan")
    if not _module_importable("tools.replay_io"):
        stub = types.ModuleType("tools.replay_io")
        stub.load_replay = lambda path: json.load(open(path, encoding="utf-8"))
        stub.episode_configuration = lambda replay: dict(replay["configuration"])
        stub.episode_id = lambda replay: int(
            replay.get("info", {}).get("EpisodeId", replay.get("id", 0)))
        sys.modules["tools.replay_io"] = stub
        injected.append("tools.replay_io")
    yield injected
    for name in injected:
        sys.modules.pop(name, None)


def _pass_agent_factory():
    return lambda seat: (lambda obs: {
        "farmer": ["PASS"], "hands": [], "market": []})


# ----------------------------------------------------------------- tests


def test_run_day_slice_synthetic_day1_seat0(synthetic_replay,
                                            stub_sibling_modules) -> None:
    result = run_day_slice(synthetic_replay, 1, 0, _pass_agent_factory())

    assert isinstance(result, SliceResult)
    assert result.error is None
    assert result.boundary_verified is True
    assert result.boundary_first_diff is None
    assert result.turns_executed == 24
    assert result.cash_start == 3000.0
    assert result.cash_end == 3000.0
    assert result.episode_id == EPISODE_ID
    assert result.seed == SYNTHETIC_CONFIGURATION["seed"]
    assert result.seat == 0 and result.day == 1
    assert result.foreman_action_families == {"PASS": 24}
    assert result.market_op_families == {}
    assert result.hires_today_max == 0
    assert result.hands_end == 0
    assert result.unlocked_start == 1 and result.unlocked_end == 1


def test_normalize_obs_strips_cosmetics_and_zero_equates() -> None:
    obs = {
        "step": 24,
        "remainingOverageTime": 60,
        "day": 1,
        "hour": 0,
        "player": 0,
        "farms": [{
            "money": 3000.0,
            "hires_today": 0,
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "tiles": [[{
                "kind": "PLANT", "crop": "WHEAT", "age": 3,
                "planted_day": 21, "yield_units": 1,
                "fertilized_until_day": -1,
            }, {"kind": "PASTURE", "animal": "COW", "age": 2,
                "placed_day": 22}]],
        }],
        "private": {"seeds": {"WHEAT": 0, "CARROT": 2},
                    "shed": {"EGG": 0}},
    }
    normalized = normalize_obs(obs)

    # Input untouched (deep copy).
    assert obs["step"] == 24 and obs["remainingOverageTime"] == 60
    assert "age" in obs["farms"][0]["tiles"][0][0]

    assert "step" not in normalized
    assert "remainingOverageTime" not in normalized
    plant = normalized["farms"][0]["tiles"][0][0]
    assert "age" not in plant
    assert plant["crop"] == "WHEAT" and plant["planted_day"] == 21
    pasture = normalized["farms"][0]["tiles"][0][1]
    assert "age" not in pasture and "placed_day" not in pasture
    assert pasture["animal"] == "COW"
    assert normalized["private"]["seeds"] == {"CARROT": 2}
    # All-zero inventory entries drop out; the emptied container stays.
    assert normalized["private"]["shed"] == {}

    # Missing-vs-present zero entries become equal forms.
    assert normalize_obs({"private": {"seeds": {"WHEAT": 0}}}) == \
        normalize_obs({"private": {"seeds": {}}})
    assert normalize_obs({"private": {"seeds": {"WHEAT": 0.0}}}) == \
        normalize_obs({"private": {"seeds": {}}})
    # Non-zero values and False booleans survive.
    kept = normalize_obs({"a": {"x": False, "y": 3}})
    assert kept == {"a": {"x": False, "y": 3}}


def test_first_diff_finds_money_difference() -> None:
    base = {"farms": [{"money": 100.0}], "market": {"prices": [1.0, 2.0]}}
    other = {"farms": [{"money": 200.0}], "market": {"prices": [1.0, 2.0]}}

    diff = first_diff(normalize_obs(base), normalize_obs(other))
    assert diff is not None
    assert "money" in diff

    assert first_diff(normalize_obs(base), normalize_obs(base)) is None
    # Float tolerance.
    near = {"farms": [{"money": 100.0 + 1e-9}], "market": {"prices": [1.0, 2.0]}}
    assert first_diff(normalize_obs(base), normalize_obs(near)) is None
    # Type mismatch is reported deterministically.
    typed = first_diff({"a": 1}, {"a": "1"})
    assert typed is not None and "type" in typed


def test_error_path_agent_raises(synthetic_replay,
                                 stub_sibling_modules) -> None:
    def bad_agent_factory(seat):
        def bad(obs):
            raise RuntimeError("boom")
        return bad

    result = run_day_slice(synthetic_replay, 1, 0, bad_agent_factory)
    assert result.error is not None
    assert "RuntimeError" in result.error and "boom" in result.error
    assert result.turns_executed == 0

    with pytest.raises(RuntimeError, match="boom"):
        run_day_slice(synthetic_replay, 1, 0, bad_agent_factory, strict=True)


def test_run_slices_and_summarize(synthetic_replay,
                                  stub_sibling_modules) -> None:
    results = day_slice.run_slices(
        [(synthetic_replay, 1, 0)], _pass_agent_factory())
    assert len(results) == 1 and results[0].error is None

    summary = day_slice.summarize(results)
    assert summary["slices"] == 1
    assert summary["boundary_verified_count"] == 1
    assert summary["total_turns"] == 24
    assert summary["cash_delta_mean"] == 0.0
    assert summary["per_slice"] == [
        {"episode_id": EPISODE_ID, "seat": 0, "day": 1, "cash_delta": 0.0}]
