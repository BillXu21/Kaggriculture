"""Focused contract tests for the fixed-plan comparator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.compare_multiday_fixed_plan import (
    ArtifactValidationError,
    FixedInputMismatchError,
    FixedPlanCompareError,
    compare_multiday_fixed_plan,
)


def _day(day: int) -> dict:
    return {
        "day": day,
        "cash_start": 100.0,
        "cash_end": 110.0,
        "cash_delta": 10.0,
        "wealth_start": 200.0,
        "wealth_end": 210.0,
        "wealth_delta": 10.0,
        "weeds_start": 0,
        "weeds_end": 0,
        "weeds_created": 0,
        "crops_destroyed_observed": 0,
        "animal_loss_evidence": [],
        "animal_escape_evidence": [],
        "animal_count_decreases_observed": 0,
        "animal_count_decreases_without_stable_evidence": 0,
        "plan_targets": {"requested": {"WHEAT": 1}, "feasible": {"WHEAT": 1}},
        "projection_changes": {},
        "eod_work_debt": {
            "all": [], "survival": [], "maintenance": [], "productive": [], "manager": [],
        },
        "pending_task_turns": {},
        "action_turn_counts": {"movement": 1, "pickup": 0, "interaction": 2, "pass": 21},
        "water_interactions": {
            "weed_prevention": 1, "yield_useful": 0, "redundant_or_unjustified": 0,
        },
        "harvested_units": {
            "observed": 2, "unmeasured_actions": 0, "unavailable_reason": None,
        },
        "survival": {
            "starvation_preemption_turns": 0,
            "feed_shortage_turns": 0,
            "partial_feed_buys": 0,
        },
        "hires": {"requested": 0, "submitted": 0, "observed_max": 0},
        "unaffordable_orders": [],
        "diagnostics": {"previous_labor": {"workers_hired": 0, "hire_cost": 0}},
    }


def _artifact(executor: str = "executor-baseline") -> dict:
    days = [_day(0), _day(1)]
    return {
        "schema_version": 1,
        "label": executor,
        "executor_provenance": executor,
        "tape_fingerprint": "tape-sha",
        "tape": {"fingerprint": "tape-sha", "provenance": {"seat": 0}},
        "replay": {
            "kind": "object",
            "sha256": "replay-sha",
            "episode_id": 42,
            "seed": 7,
            "configuration_sha256": "config-sha",
            "configuration": {"seed": 7, "turnsPerDay": 24},
        },
        "seat": 0,
        "window": {
            "start_day": 0, "end_day": 1, "length": 2,
            "turns_per_day": 24, "turns": 48,
        },
        "backend": {
            "name": "fast", "engine": {"name": "kaggriculture", "version": "1.32.7"},
        },
        "strategy_inputs": {
            "fixed_plan_provider": True,
            "live_manager_invocations": 0,
            "opening_source": "replay_prefix",
            "opponent_trace_sha256": "opponent-sha",
        },
        "days": days,
        "totals": {
            "turns": 48, "days": 2, "cash_delta": 20.0, "wealth_delta": 20.0,
        },
        "executor_diagnostics": {
            "schema_version": 2,
            "fallback_errors": [],
            "illegal_actions": {"available": False, "reason": "not exposed"},
        },
    }


def _redigest(document: dict) -> dict:
    unsigned = {key: value for key, value in document.items() if key != "artifact_sha256"}
    document["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return document


def _compare(baseline: dict, candidate: dict):
    return compare_multiday_fixed_plan(
        _redigest(baseline),
        _redigest(candidate),
        label="focused-compare",
    )


def test_equal_inputs_emit_deterministic_per_day_and_window_artifact():
    baseline = _artifact()
    candidate = _artifact("executor-candidate")
    first = _compare(copy.deepcopy(baseline), copy.deepcopy(candidate))
    second = _compare(copy.deepcopy(baseline), copy.deepcopy(candidate))

    assert first.to_json() == second.to_json()
    document = first.to_dict()
    assert document["schema_version"] == 1
    assert document["frozen_inputs"]["identical"] is True
    assert len(document["days"]) == 2
    assert document["days"][0]["metrics"]["bank_wealth"]["cash_delta"]["delta"] == 0
    assert document["window"]["window"]["bank_wealth"]["wealth_delta"]["delta"] == 0
    assert document["safety"]["promotion_safe"] is True

    output = Path(".focused-compare-output.json")
    output.unlink(missing_ok=True)
    first_with_output = _compare(copy.deepcopy(baseline), copy.deepcopy(candidate))
    try:
        first_with_output.save(output)
        assert output.read_text(encoding="utf-8").rstrip("\n") == first_with_output.to_json()
        with pytest.raises(FixedPlanCompareError, match="overwrite"):
            first_with_output.save(output)
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (("tape_fingerprint",), "other-tape", "tape_fingerprint"),
        (("replay", "sha256"), "other-replay", "replay_sha256"),
        (("replay", "seed"), 8, "seed"),
        (("replay", "configuration_sha256"), "other-config", "configuration_sha256"),
        (("seat",), 1, "seat"),
        (("window", "start_day"), 1, "window"),
        (("backend", "engine", "version"), "other-engine", "engine"),
        (("backend", "name"), "other-backend", "backend"),
        (("strategy_inputs", "opponent_trace_sha256"), "other-opponent", "opponent_trace_sha256"),
        (("window", "turns_per_day"), 12, "window"),
    ],
)
def test_each_frozen_input_mismatch_fails_loudly(path, value, needle):
    baseline = _artifact()
    candidate = _artifact("executor-candidate")
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == ("tape_fingerprint",):
        candidate["tape"]["fingerprint"] = value
    elif path == ("window", "start_day"):
        candidate["window"]["end_day"] = 2
        candidate["days"][0]["day"] = 1
        candidate["days"][1]["day"] = 2
    elif path == ("window", "turns_per_day"):
        candidate["window"]["turns"] = 24
        candidate["totals"]["turns"] = 24
    with pytest.raises(FixedInputMismatchError, match=needle):
        _compare(baseline, candidate)


def test_digest_tampering_is_rejected_before_comparison():
    baseline = _artifact()
    candidate = _redigest(_artifact("executor-candidate"))
    candidate["days"][0]["cash_delta"] = 999.0
    with pytest.raises(ArtifactValidationError, match="artifact_sha256 mismatch"):
        compare_multiday_fixed_plan(
            _redigest(baseline), candidate, label="tampered"
        )


def test_nullable_metrics_are_not_coerced_to_zero():
    baseline = _artifact()
    candidate = _artifact("executor-candidate")
    baseline["days"][0]["wealth_delta"] = None
    candidate["days"][0]["wealth_delta"] = None
    result = _compare(baseline, candidate).to_dict()
    metric = result["days"][0]["metrics"]["bank_wealth"]["wealth_delta"]
    assert metric["baseline"] is None
    assert metric["candidate"] is None
    assert metric["delta"] is None
    assert any("wealth_delta" in warning for warning in result["warnings"])


def test_manager_debt_increase_is_reported_but_does_not_fail_safety():
    baseline = _artifact()
    candidate = _artifact("executor-candidate")
    candidate["days"][0]["eod_work_debt"]["manager"] = ["SELL:WHEAT"]
    result = _compare(baseline, candidate).to_dict()
    assert result["safety"]["promotion_safe"] is True
    assert result["safety"]["manager_debt_is_not_a_failure"] is True
    debt = result["days"][0]["metrics"]["end_of_day_debt"]["eod_work_debt"]
    assert debt["delta"]["manager"] is None


def test_animal_and_survival_regressions_are_safety_failures_in_priority_order():
    baseline = _artifact()
    candidate = _artifact("executor-candidate")
    evidence = {"coord": [1, 1], "species": "COW", "kind": "escape_evidence"}
    candidate["days"][0]["animal_loss_evidence"] = [evidence]
    candidate["days"][0]["animal_escape_evidence"] = [evidence]
    candidate["days"][0]["eod_work_debt"]["survival"] = ["FEED:COW:1:1"]
    result = _compare(baseline, candidate).to_dict()
    assert result["safety"]["promotion_safe"] is False
    assert result["safety"]["primary_failure"] == "avoidable_animal_escape_or_loss"
    flags = result["safety"]["flags"]
    assert flags[1]["active"] is True
    assert flags[2]["active"] is True


def test_same_executor_requires_explicit_test_only_escape_hatch():
    with pytest.raises(FixedPlanCompareError, match="executor provenance is identical"):
        _compare(_artifact(), _artifact())
    result = compare_multiday_fixed_plan(
        _redigest(_artifact()), _redigest(_artifact()),
        label="same-provenance-test", allow_same_executor_for_test=True,
    ).to_dict()
    assert result["executor_provenance"]["same_executor_allowed_for_test"] is True
    assert any("test-only" in warning for warning in result["warnings"])
