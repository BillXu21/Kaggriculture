"""Focused recorder tests over an engine-generated all-PASS replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("fast_env._kaggriculture_env")

from executor_v0.agent import AgentConfig
from executor_v0.plan import SELL_BIN_ANCHORS, DailyPlan
from replay_daily.constants import PRODUCTS
from tools import record_fixed_plan_tape as recorder
from tools.fixed_plan_tape import FixedPlanTape
from tools.record_fixed_plan_tape import (
    RecorderConfig,
    ReplayBoundaryMismatch,
    record_fixed_plan_tape,
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
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def synthetic_replay() -> dict:
    from fast_env.api import FastKaggricultureEnv

    env = FastKaggricultureEnv(dict(CONFIGURATION))
    current = [copy.deepcopy(obs) for obs in env.reset()]
    steps = []
    for index in range(97):
        entries = []
        for seat in (0, 1):
            observation = copy.deepcopy(current[seat])
            observation.update({
                "day": index // 24,
                "hour": index % 24,
                "step": index,
            })
            entries.append({
                "action": copy.deepcopy(PASS),
                "observation": observation,
                "reward": 0,
                "status": "ACTIVE",
            })
        steps.append(entries)
        if index < 96:
            current = env.step([copy.deepcopy(PASS), copy.deepcopy(PASS)])[0]
    return {
        "configuration": dict(CONFIGURATION),
        "info": {"EpisodeId": 4242, "seed": 7},
        "steps": steps,
    }


def plan() -> DailyPlan:
    return DailyPlan.create(
        crop_targets={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        land_count=1,
        fertilizer_by_crop={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities={
            product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
            for product in PRODUCTS
        },
    )


class RecordingProvider:
    model_variant = "E"

    def __init__(self):
        self.calls = []

    def daily_plan(self, obs, seat, previous_execution=None):
        self.calls.append((int(obs["day"]), int(obs["step"]), seat,
                           copy.deepcopy(previous_execution)))
        return plan()


def _record_kwargs(replay, output):
    return dict(
        replay_path_or_replay=replay,
        seat=0,
        start_day=1,
        length=3,
        backend_name="fast",
        opening_identity="synthetic",
        opening_provenance_data={"name": "synthetic", "digest": "trace"},
        output_path=output,
        source_repo_sha="3726e37",
        reference_executor_provenance=RecorderConfig(
            name="executor-v0", version="test", configuration={"strict": True}
        ),
        reference_executor_config=AgentConfig(strict=True),
    )


def test_recording_is_once_per_day_deterministic_and_fixes_opponent(
    tmp_path: Path, monkeypatch
):
    replay = synthetic_replay()
    provider = RecordingProvider()
    real_make_backend = recorder.make_backend
    backend_actions = []

    class SpyBackend:
        def __init__(self, delegate):
            self.delegate = delegate
            self.name = delegate.name

        def reset(self):
            return self.delegate.reset()

        def step(self, actions):
            backend_actions.append(copy.deepcopy(actions))
            return self.delegate.step(actions)

        def canonical_state(self):
            return self.delegate.canonical_state()

        @property
        def statuses(self):
            return self.delegate.statuses

        @property
        def rewards(self):
            return self.delegate.rewards

    monkeypatch.setattr(
        recorder, "make_backend",
        lambda name, configuration: SpyBackend(real_make_backend(name, configuration)),
    )
    output = tmp_path / "plans.json"
    first = record_fixed_plan_tape(provider=provider, **_record_kwargs(replay, output))

    assert [call[:3] for call in provider.calls] == [
        (1, 24, 0), (2, 48, 0), (3, 72, 0)
    ]
    assert all(call[3] == {"workers_hired": 0, "hire_cost": 0}
               for call in provider.calls)
    assert [day for day, _ in first.plans] == [1, 2, 3]
    # Prefix steps and every recording-window opponent step use replay PASS.
    assert all(pair[1] == PASS for pair in backend_actions)

    second_provider = RecordingProvider()
    second = record_fixed_plan_tape(
        provider=second_provider, force=True, **_record_kwargs(replay, output)
    )
    assert first.artifact_sha256 == second.artifact_sha256
    assert output.read_text(encoding="utf-8").strip() == second.to_json()
    assert isinstance(FixedPlanTape.load(output), FixedPlanTape)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert "opponent_trace_sha256" in document["provenance"]
    assert all("age" not in json.dumps(entry["plan"])
               for entry in document["plans"])


def test_boundary_mismatch_and_missing_checkpoint_fail_loudly(tmp_path: Path):
    replay = synthetic_replay()
    replay["steps"][24][0]["observation"]["farms"][0]["money"] += 1
    with pytest.raises(ReplayBoundaryMismatch, match="boundary mismatch"):
        record_fixed_plan_tape(
            provider=RecordingProvider(),
            **_record_kwargs(replay, tmp_path / "bad.json")
        )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        record_fixed_plan_tape(
            provider=RecordingProvider(),
            checkpoint_path=tmp_path / "missing-best.pt",
            checkpoint_sha256="never-used",
            **_record_kwargs(synthetic_replay(), tmp_path / "missing.json"),
        )
