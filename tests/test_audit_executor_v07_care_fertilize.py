"""Focused tests for the issue 14A CARE/FERTILIZE audit postprocessor."""

from __future__ import annotations

from tools.audit_executor_v07_care_fertilize import build_audit


def _record(
    *, care_requested=1, care_observed=0, care_assigned=True,
    fert_requested=0, fert_observed=0, fert_assigned=False,
):
    assignments = []
    if care_assigned:
        assignments.append({"op_family": "CARE", "task_key": "CARE:COW:1,1"})
    if fert_assigned:
        assignments.append({"op_family": "FERTILIZE", "task_key": "FERTILIZE:STRAWBERRY:1,1"})
    return {
        "requested": {"care_by_animal": {"COW": care_requested}, "fertilizer_by_crop": {"STRAWBERRY": fert_requested}},
        "feasible": {"care_by_animal": {"COW": care_requested}, "fertilizer_by_crop": {"STRAWBERRY": fert_requested}},
        "projection_changes": {"care": {"COW": {"eligible": 1}}, "fertilizer": {"STRAWBERRY": {"eligible": 1}}},
        "care_completed_observed": {"COW": care_observed},
        "fertilizer_completed_observed": {"STRAWBERRY": fert_observed},
        "turn_trace": [{"assignments": assignments}],
        "pending_task_turns": {"CARE:COW:1,1": 1},
        "errors": [],
    }


def _document(records):
    games = []
    for seed in (1, 2):
        games.append({
            "seed": seed,
            "seat": 0,
            "status": "complete",
            "transitions": 719,
            "final": {"statuses": ["DONE", "DONE"], "bank": 10},
            "executor_diagnostics": {
                "fallback_errors": [],
                "illegal_actions": {"available": False, "reason": "not exposed"},
                "days": {str(day): record for day, record in records.items()},
            },
        })
    return {
        "schema_version": 2,
        "artifact_type": "executor_v07_full_game_evaluator",
        "source_provenance": {"repo_sha": "a" * 40},
        "checkpoint": {"sha256": "b" * 64},
        "opponent": {"identity": "PASS"},
        "request": {"backend": "fast", "opening": "standard_mixed", "seeds": [1, 2], "seats": [0], "opponent": "PASS"},
        "games": games,
    }


def test_audit_preserves_zero_rows_and_separates_assignment_from_observation():
    report = build_audit(
        _document({"4": _record()}),
        expected_repo_sha="a" * 40,
        expected_checkpoint_sha256="b" * 64,
        expected_seeds=[1, 2],
        expected_seats=[0],
        expected_day_range=(4, 4),
    )

    assert len(report["zero_completion_rows"]) == 2  # one positive COW row per game
    cow = [row for row in report["material_shortfall_rows"] if row["family"] == "CARE" and row["entity"] == "COW"]
    assert len(cow) == 2
    assert cow[0]["submitted_assigned"] == 1
    assert cow[0]["observed_completed"] == 0
    assert cow[0]["classification"]["classification"] == "unresolved"
    assert report["totals_by_entity"]["CARE"]["COW"]["requested"] == 2


def test_audit_retains_signed_fertilizer_state_difference_and_assignment_count():
    report = build_audit(
        _document({"4": _record(fert_requested=2, fert_observed=1, fert_assigned=True)}),
        expected_day_range=(4, 4),
    )
    rows = [row for row in report["material_shortfall_rows"] if row["family"] == "FERTILIZE"]
    assert len(rows) == 2
    assert rows[0]["requested"] == 2
    assert rows[0]["submitted_assigned"] == 1
    assert rows[0]["observed_completed"] == 1
    assert rows[0]["raw_shortfall_requested_minus_observed"] == 1
    assert report["totals_by_entity"]["FERTILIZE"]["STRAWBERRY"]["raw_shortfall_requested_minus_observed"] == 2


def test_audit_classifies_explicit_projection_infeasibility():
    record = _record(care_requested=3, care_observed=0, care_assigned=False)
    record["feasible"]["care_by_animal"]["COW"] = 1
    record["projection_changes"]["care"]["COW"] = {"eligible": 1}
    report = build_audit(_document({"4": record}), expected_day_range=(4, 4))
    row = next(row for row in report["material_shortfall_rows"] if row["family"] == "CARE" and row["entity"] == "COW")
    assert row["classification"]["classification"] == "manager requested infeasible work"
    assert row["classification"]["evidence"]["projection_shortfall"] == 2
