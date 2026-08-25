"""Focused tests for checkpoint-free expert-intent tape compilation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import expert_plan
from tools.build_expert_plan_tape import (
    ExpertPlanTapeBuildError,
    build_expert_plan_tape,
)
from tools.fixed_plan_tape import FixedPlanTape


SAMPLES_DIR = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\samples"
)
REAL_REPLAY = SAMPLES_DIR / "98093786.json"


def _farm() -> dict:
    return {
        "tiles": [["EMPTY"]],
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
    }


def _replay() -> dict:
    steps = []
    for index in range(73):
        day, hour = divmod(index, 24)
        observation = {
            "day": day,
            "hour": hour,
            "farms": [_farm(), _farm()],
        }
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        steps.append([
            {"action": action, "observation": observation},
            {"action": action, "observation": json.loads(json.dumps(observation))},
        ])
    return {
        "name": "kaggriculture",
        "module_version": "1.32.7",
        "configuration": {"turnsPerDay": 24, "episodeSteps": 720, "seed": 17},
        "info": {"EpisodeId": 4242, "seed": 17},
        "steps": steps,
    }


def _kwargs(output: Path) -> dict:
    return {
        "seat": 0,
        "start_day": 0,
        "length": 3,
        "backend_name": "fast",
        "source_repo_sha": "3726e373c65b8221c4062138174898f6cf756119",
        "opening_identity": "synthetic",
        "opening_provenance_data": {"name": "synthetic", "digest": "trace"},
        "output_path": output,
        "label": "checkpoint-free-test",
    }


def test_build_calls_extractor_once_per_absolute_day_and_is_deterministic(
    tmp_path: Path, monkeypatch
):
    replay = _replay()
    original = expert_plan.extract_daily_plan
    calls = []

    def spy(replay_arg, seat, day):
        calls.append((seat, day))
        return original(replay_arg, seat, day)

    monkeypatch.setattr(expert_plan, "extract_daily_plan", spy)
    output = tmp_path / "expert.json"
    first = build_expert_plan_tape(replay, **_kwargs(output))
    assert calls == [(0, 0), (0, 1), (0, 2)]
    calls.clear()
    second = build_expert_plan_tape(
        replay, force=True, **_kwargs(output)
    )

    assert calls == [(0, 0), (0, 1), (0, 2)]
    assert first.artifact_sha256 == second.artifact_sha256
    assert output.read_text(encoding="utf-8").strip() == second.to_json()
    assert isinstance(FixedPlanTape.load(output), FixedPlanTape)

    provenance = first.provenance
    assert provenance["manager"] == "expert-replay-intent"
    assert provenance["checkpoint"] == "none:checkpoint-free"
    assert provenance["model_variant"] == "expert-replay-v1"
    assert provenance["backend"] == {"name": "fast", "version": "1.32.7"}
    assert provenance["engine"] == {"name": "kaggriculture", "version": "1.32.7"}
    assert provenance["recording_window"] == {"start_day": 0, "end_day": 2}
    assert provenance["replay"]["episode_id"] == 4242
    assert provenance["replay"]["name"] is None
    assert provenance["opponent_trace_sha256"] == provenance["replay"]["opponent_trace_sha256"]
    assert "not BC-E promotion evidence" in " ".join(provenance["limitations"])
    assert "observations" not in first.to_json()
    assert "model_outputs" not in first.to_json()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda replay: replay.update(module_version="0.0.0"), "unsupported replay"),
        (lambda replay: replay["configuration"].update(turnsPerDay=12), "configuration"),
        (lambda replay: replay["steps"].pop(), "requested window"),
        (lambda replay: replay["steps"][25][0].pop("action"), "missing a primitive action"),
    ],
)
def test_replay_contract_rejects_unsupported_or_incomplete_inputs(
    tmp_path: Path, change, match
):
    replay = _replay()
    change(replay)
    with pytest.raises(ExpertPlanTapeBuildError, match=match):
        build_expert_plan_tape(replay, **_kwargs(tmp_path / "bad.json"))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("seat", 2, "seat"),
        ("start_day", -1, "start_day"),
        ("length", 4, "length"),
        ("backend_name", "", "backend_name"),
        ("opening_identity", "", "opening_identity"),
    ],
)
def test_window_and_identity_inputs_are_validated(tmp_path: Path, field, value, match):
    kwargs = _kwargs(tmp_path / "bad.json")
    kwargs[field] = value
    with pytest.raises(ExpertPlanTapeBuildError, match=match):
        build_expert_plan_tape(_replay(), **kwargs)


def test_output_overwrite_requires_force(tmp_path: Path):
    output = tmp_path / "expert.json"
    build_expert_plan_tape(_replay(), **_kwargs(output))
    with pytest.raises(FileExistsError, match="use --force"):
        build_expert_plan_tape(_replay(), **_kwargs(output))


@pytest.mark.skipif(not REAL_REPLAY.is_file(), reason="authorized samples replay unavailable")
def test_optional_authorized_real_replay_compiles(tmp_path: Path):
    output = tmp_path / "real-expert.json"
    tape = build_expert_plan_tape(
        REAL_REPLAY,
        seat=0,
        start_day=3,
        length=3,
        backend_name="fast",
        source_repo_sha="3726e373c65b8221c4062138174898f6cf756119",
        opening_identity="standard_mixed",
        opening_provenance_data={"name": "standard_mixed", "digest": "test"},
        output_path=output,
    )
    assert [day for day, _ in tape.plans] == [3, 4, 5]
    assert len(tape.artifact_sha256) == hashlib.sha256(b"").digest_size * 2
