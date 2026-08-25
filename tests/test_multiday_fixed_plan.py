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


def test_runner_writes_the_same_canonical_artifact(tmp_path):
    replay = _replay_three_days()
    output = tmp_path / "fixed-plan.json"
    result = _run(replay, _tape(), output_path=output)
    assert output.read_text(encoding="utf-8").rstrip("\n") == result.to_json()


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


def test_runner_rejects_first_boundary_difference():
    replay = _replay_three_days()
    replay["steps"][0][0]["observation"]["farms"][0]["money"] += 1
    with pytest.raises(BoundaryMismatchError, match="boundary mismatch.*seat 0"):
        _run(replay, _tape())
