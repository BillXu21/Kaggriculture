"""Focused real-fast-engine tests for the one-arm fixed-plan runner."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import json

import pytest

from executor_v0.plan import SELL_BIN_ANCHORS, DailyPlan
from replay_daily.constants import PRODUCTS
from tools.fixed_plan_tape import FixedPlanTape
import tools.multiday_fixed_plan as multiday_fixed_plan
from tools.multiday_fixed_plan import (
    BoundaryMismatchError,
    FixedPlanRunError,
    run_multiday_fixed_plan,
)


def _fast_engine_available() -> bool:
    try:
        from fast_env.api import FastKaggricultureEnv  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


requires_fast = pytest.mark.skipif(
    not _fast_engine_available(), reason="fast engine extension is unavailable"
)


CONFIGURATION = {
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
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _zero_plan() -> DailyPlan:
    return DailyPlan.create(
        crop_targets={name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        animal_targets={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        land_count=1,
        fertilizer_by_crop={name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        care_by_animal={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        sell_quantities={
            product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
            for product in PRODUCTS
        },
    )


def _replay_three_days() -> dict:
    from fast_env.api import FastKaggricultureEnv

    env = FastKaggricultureEnv(dict(CONFIGURATION))
    current = env.reset()
    steps = []
    last = 3 * 24
    for index in range(last + 1):
        entries = []
        for seat in (0, 1):
            observation = copy.deepcopy(current[seat])
            observation["day"] = index // 24
            observation["hour"] = index % 24
            observation["step"] = index
            entries.append({
                "action": copy.deepcopy(PASS_ACTION),
                "observation": observation,
                "reward": 0,
                "status": "ACTIVE",
            })
        steps.append(entries)
        if index < last:
            current = env.step([copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)])[0]
    return {
        "configuration": dict(CONFIGURATION),
        "id": 4242,
        "info": {"EpisodeId": 4242, "seed": CONFIGURATION["seed"]},
        "steps": steps,
    }


def _static_replay_three_days() -> dict:
    """Small backend-independent replay for stable trace plumbing tests."""
    base = {
        "day": 0,
        "hour": 0,
        "step": 0,
        "player": 0,
        "farms": [{
            "farmer": [0, 0],
            "hands": [],
            "hires_today": 0,
            "money": 3000.0,
            "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"],
        }] * 2,
        "market": {
            "inventory": {product: 10000 for product in PRODUCTS},
            "prices": {product: 100 for product in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {}, "seeds": {}, "inventories": [],
        },
    }
    steps = []
    for index in range(3 * 24 + 1):
        observations = []
        for seat in (0, 1):
            observation = copy.deepcopy(base)
            observation["player"] = seat
            observation["day"] = index // 24
            observation["hour"] = index % 24
            observation["step"] = index
            observations.append(observation)
        steps.append([
            {
                "action": copy.deepcopy(PASS_ACTION),
                "observation": observations[seat],
                "reward": 0,
                "status": "ACTIVE",
            }
            for seat in (0, 1)
        ])
    return {
        "configuration": dict(CONFIGURATION),
        "id": 4242,
        "info": {"EpisodeId": 4242, "seed": CONFIGURATION["seed"]},
        "steps": steps,
    }


class _StaticBackend:
    name = "fast"

    def __init__(self, configuration):
        del configuration
        self._step = 0
        self._replay = _static_replay_three_days()
        self._rewards = [0.0, 0.0]
        self._statuses = ["ACTIVE", "ACTIVE"]

    def reset(self):
        self._step = 0
        return [
            copy.deepcopy(self._replay["steps"][0][seat]["observation"])
            for seat in (0, 1)
        ]

    def step(self, actions):
        del actions
        self._step += 1
        return (
            [
                copy.deepcopy(self._replay["steps"][self._step][seat]["observation"])
                for seat in (0, 1)
            ],
            list(self._rewards),
            list(self._statuses),
        )

    def canonical_state(self):
        return {"step": self._step}

    @property
    def rewards(self):
        return list(self._rewards)

    @property
    def statuses(self):
        return list(self._statuses)


def _tape(seat: int = 0, length: int = 3) -> FixedPlanTape:
    end = length - 1
    provenance = {
        "manager": "synthetic-fixed-manager",
        "checkpoint": "none-for-synthetic-tape",
        "model_variant": "fixed",
        "seed": CONFIGURATION["seed"],
        "seat": seat,
        "opening_identity": "synthetic-pass",
        "source_repo_sha": "3726e373c65b8221c4062138174898f6cf756119",
        "backend": {"name": "fast", "version": "local"},
        "engine": {"name": "kaggriculture", "version": "1.32.7"},
        "recording_window": {"start_day": 0, "end_day": end},
    }
    plan = _zero_plan()
    return FixedPlanTape.create(
        plans=[(day, plan) for day in range(length)],
        provenance=provenance,
    )


def _run(replay, tape, **kwargs):
    return run_multiday_fixed_plan(
        replay,
        tape,
        0,
        3,
        label="synthetic-smoke",
        executor_provenance="executor-v07-test",
        **kwargs,
    )


@requires_fast
def test_three_day_fast_runner_is_deterministic_and_fully_fixed():
    replay = _replay_three_days()
    tape = _tape()
    first = _run(replay, tape)
    second = _run(replay, tape)

    assert first.to_json() == second.to_json()
    document = first.to_dict()
    assert document["schema_version"] == 1
    assert document["tape_fingerprint"] == tape.artifact_sha256
    assert document["boundary"]["verified"] is True
    assert document["window"] == {
        "start_day": 0,
        "end_day": 2,
        "length": 3,
        "turns_per_day": 24,
        "turns": 72,
    }
    assert document["totals"]["turns"] == 72
    assert len(document["days"]) == 3
    assert all(day["cash_start"] == 3000.0 for day in document["days"])
    assert all(day["cash_delta"] == 0.0 for day in document["days"])
    assert all(day["foreman_counts"]["pass"] == 24 for day in document["days"])
    assert document["strategy_inputs"]["fixed_plan_provider"] is True
    assert document["strategy_inputs"]["live_manager_invocations"] == 0
    assert document["strategy_inputs"]["opening_source"] == "replay_prefix"
    assert document["executor_diagnostics"]["fallback_errors"] == []
    assert document["artifact_sha256"] == hashlib.sha256(
        json.dumps(
            {key: value for key, value in document.items() if key != "artifact_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@requires_fast
def test_runner_writes_the_same_canonical_artifact(tmp_path):
    replay = _replay_three_days()
    output = tmp_path / "fixed-plan.json"
    result = _run(replay, _tape(), output_path=output)
    assert output.read_text(encoding="utf-8").rstrip("\n") == result.to_json()


@requires_fast
def test_runner_metric_helpers_receive_one_backend_observation(monkeypatch):
    original = multiday_fixed_plan._farm_metrics
    seen_types = []

    def checked(observation, seat):
        seen_types.append(type(observation).__name__)
        assert isinstance(observation, Mapping)
        return original(observation, seat)

    monkeypatch.setattr(multiday_fixed_plan, "_farm_metrics", checked)
    result = _run(_replay_three_days(), _tape())

    assert result.to_dict()["boundary"]["verified"] is True
    assert seen_types and set(seen_types) == {"dict"}


def test_runner_rejects_window_and_seat_mismatches():
    replay = {
        "configuration": dict(CONFIGURATION),
        "info": {"EpisodeId": 4242, "seed": CONFIGURATION["seed"]},
        "steps": [],
    }
    with pytest.raises(FixedPlanRunError, match="window_length"):
        run_multiday_fixed_plan(
            replay,
            _tape(),
            0,
            5,
            label="bad-window",
            executor_provenance="executor-v07-test",
        )
    with pytest.raises(FixedPlanRunError, match="seat mismatch"):
        run_multiday_fixed_plan(
            replay,
            _tape(seat=1),
            0,
            3,
            label="bad-seat",
            executor_provenance="executor-v07-test",
        )


@requires_fast
def test_runner_rejects_first_boundary_difference():
    replay = _replay_three_days()
    replay["steps"][0][0]["observation"]["farms"][0]["money"] += 1
    with pytest.raises(BoundaryMismatchError, match="boundary mismatch.*seat 0"):
        _run(replay, _tape())


def test_turn_trace_is_opt_in_and_preserves_actions_and_primary_metrics(monkeypatch):
    monkeypatch.setattr(
        multiday_fixed_plan,
        "make_backend",
        lambda backend, configuration: _StaticBackend(configuration),
    )
    replay = _static_replay_three_days()
    tape = _tape()
    disabled = _run(replay, tape, turn_trace=False).to_dict()
    enabled = _run(replay, tape, turn_trace=True).to_dict()

    assert disabled["run_config"] == {"turn_trace": False}
    assert enabled["run_config"] == {"turn_trace": True}
    assert disabled["strategy_inputs"]["tested_action_trace_sha256"] == \
        enabled["strategy_inputs"]["tested_action_trace_sha256"]
    assert disabled["strategy_inputs"]["opponent_trace_sha256"] == \
        enabled["strategy_inputs"]["opponent_trace_sha256"]
    for disabled_day, enabled_day in zip(disabled["days"], enabled["days"]):
        assert "turn_trace" not in disabled_day["diagnostics"]
        trace = enabled_day["diagnostics"]["turn_trace"]
        assert len(trace) == 24
        assert [entry["hour"] for entry in trace] == list(range(24))
        assert {
            key: value for key, value in disabled_day.items()
            if key != "diagnostics"
        } == {
            key: value for key, value in enabled_day.items()
            if key != "diagnostics"
        }
