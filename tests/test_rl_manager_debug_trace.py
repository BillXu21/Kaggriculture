"""Focused deterministic contract tests for the issue #11 debug trace."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rl_manager.debug_trace import (
    DEBUG_TRACE_SCHEMA_VERSION,
    DebugTraceError,
    TraceRecorder,
    build_trace,
    canonical_json_bytes,
    load_trace,
    save_trace,
    validate_trace,
)


def _tile(*, plant: bool = True) -> dict:
    if plant:
        return {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "max_lifespan_step": 100,
            "yield_units": 1,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "fertilized_until_day": 0,
        }
    return {
        "kind": "PASTURE",
        "animal": "SHEEP",
        "placed_day": 0,
        "yield_units": 1,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def _state(step: int, day: int = 0, hour: int = 0) -> dict:
    farm = {
        "money": 3000.0,
        "tiles": [[_tile(), "LOCKED"]],
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    return {
        "step": step,
        "day": day,
        "hour": hour,
        "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
        "privates": [
            {"shed": {"WHEAT": 0}, "seeds": {"WHEAT": 1}, "inventories": [{}]},
            {"shed": {"WHEAT": 0}, "seeds": {"WHEAT": 1}, "inventories": [{}]},
        ],
        "market": {"inventory": {"WHEAT": 10}, "prices": {"WHEAT": 5}},
        "town": {"unlocked_shops": []},
        "rewards": [0.0, 0.0],
        "statuses": ["ACTIVE", "ACTIVE"],
    }


def _recorder() -> TraceRecorder:
    recorder = TraceRecorder({
        "seed": 17,
        "seat": 0,
        "view": "joint",
        "backend": "fast",
        "engine": "1.32.7",
        "provenance": {"episode": 3},
    })
    recorder.append_turn(
        step=0,
        day=0,
        hour=0,
        canonical_state=_state(0),
        current_seat=0,
        joint_actions={"0": {"farmer": ["PASS"]}, "1": {"farmer": ["PASS"]}},
        executor_debug={"0": {"pending": 1}},
    )
    recorder.append_turn(step=1, day=0, hour=1, canonical_state=_state(1, hour=1))
    return recorder


def test_envelope_turn_contract_and_optional_sidecars():
    document = _recorder().build()
    assert set(document) == {"schema_version", "metadata", "turns"}
    assert document["schema_version"] == DEBUG_TRACE_SCHEMA_VERSION
    assert document["metadata"]["view"] == "joint"
    assert set(document["turns"][0]) == {
        "step", "day", "hour", "current_seat", "canonical_state",
        "joint_actions", "executor_debug",
    }
    assert document["turns"][0]["canonical_state"]["farms"]
    assert document["turns"][0]["canonical_state"]["privates"]
    assert "market" in document["turns"][0]["canonical_state"]
    assert "town" in document["turns"][0]["canonical_state"]


def test_equivalent_builds_have_exact_deterministic_bytes():
    first = canonical_json_bytes(_recorder().build())
    second = canonical_json_bytes(_recorder().build())
    assert first == second
    assert b"\n" not in first


def test_strict_json_rejects_nan_inf_and_non_json_values():
    recorder = TraceRecorder()
    bad_state = _state(0)
    bad_state["farms"][0]["money"] = float("nan")
    with pytest.raises(DebugTraceError, match="non-finite"):
        recorder.append_turn(step=0, day=0, hour=0, canonical_state=bad_state)

    bad_state = _state(0)
    bad_state["farms"][0]["money"] = object()
    with pytest.raises(DebugTraceError, match="non-JSON"):
        recorder.append_turn(step=0, day=0, hour=0, canonical_state=bad_state)


def test_recorder_and_build_return_defensive_copies():
    state = _state(0)
    actions = {"0": {"farmer": ["PASS"]}}
    debug = {"0": {"pending": 1}}
    recorder = TraceRecorder()
    recorder.append_turn(
        step=0, day=0, hour=0, canonical_state=state,
        joint_actions=actions, executor_debug=debug,
    )
    state["farms"][0]["money"] = 1.0
    actions["0"]["farmer"][0] = "HACKED"
    debug["0"]["pending"] = 99
    document = recorder.build()
    assert document["turns"][0]["canonical_state"]["farms"][0]["money"] == 3000.0
    assert document["turns"][0]["joint_actions"]["0"]["farmer"] == ["PASS"]
    assert document["turns"][0]["executor_debug"]["0"]["pending"] == 1
    document["turns"][0]["canonical_state"]["farms"][0]["money"] = 2.0
    assert recorder.build()["turns"][0]["canonical_state"]["farms"][0]["money"] == 3000.0


def test_save_load_round_trip_is_valid_and_byte_stable(tmp_path: Path):
    original = _recorder().build()
    path = save_trace(tmp_path / "debug-trace.json", original)
    loaded = load_trace(path)
    validate_trace(loaded)
    assert canonical_json_bytes(loaded) == path.read_bytes()
    loaded["turns"][0]["day"] = 99
    assert load_trace(path)["turns"][0]["day"] == 0


def test_build_trace_and_validation_reject_bad_schema_or_steps():
    document = build_trace(turns=[
        {"step": 0, "day": 0, "hour": 0, "canonical_state": _state(0)},
        {"step": 2, "day": 0, "hour": 2, "canonical_state": _state(2, hour=2)},
    ])
    validate_trace(document)

    bad_schema = copy.deepcopy(document)
    bad_schema["schema_version"] = 2
    with pytest.raises(DebugTraceError, match="schema_version"):
        validate_trace(bad_schema)

    duplicate = copy.deepcopy(document)
    duplicate["turns"][1]["step"] = 0
    with pytest.raises(DebugTraceError, match="strictly increasing"):
        validate_trace(duplicate)

    malformed = copy.deepcopy(document)
    del malformed["turns"][0]["canonical_state"]
    with pytest.raises(DebugTraceError, match="missing required fields"):
        validate_trace(malformed)

    boolean_schema = copy.deepcopy(document)
    boolean_schema["schema_version"] = True
    with pytest.raises(DebugTraceError, match="schema_version must be an int"):
        validate_trace(boolean_schema)


def test_raw_age_alias_is_rejected_but_canonical_lifecycle_fields_are_accepted():
    accepted = _state(0)
    accepted["farms"][0]["tiles"][0][0] = _tile(plant=False)
    validate_trace({
        "schema_version": 1,
        "metadata": {},
        "turns": [{"step": 0, "day": 0, "hour": 0, "canonical_state": accepted}],
    })

    rejected = _state(0)
    rejected["farms"][0]["tiles"][0][0] = {
        **_tile(), "age": 2,
    }
    with pytest.raises(DebugTraceError, match="raw fast tile alias 'age'"):
        validate_trace({
            "schema_version": 1,
            "metadata": {},
            "turns": [{"step": 0, "day": 0, "hour": 0, "canonical_state": rejected}],
        })


def test_json_text_is_strictly_parseable_without_nan():
    text = canonical_json_bytes(_recorder().build()).decode("utf-8")
    json.dumps(json.loads(text), allow_nan=False)


def test_metadata_rejects_unknown_and_wall_clock_fields():
    with pytest.raises(DebugTraceError, match="unsupported fields"):
        TraceRecorder({"timestamp": "2026-08-25T00:00:00Z"})
    with pytest.raises(DebugTraceError, match="wall-clock"):
        TraceRecorder({"provenance": {"generated_at": "2026-08-25T00:00:00Z"}})
