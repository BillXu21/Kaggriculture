"""Focused contract tests for the fixed DailyPlan evaluation tape."""

import copy
import json

import pytest

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER, PRODUCT_ORDER
from executor_v0.manager import PlanProvider
from executor_v0.plan import SELL_BIN_ANCHORS, DailyPlan
from tools.fixed_plan_tape import (
    FixedPlanTape,
    FixedPlanTapeProvider,
    PlanTapeError,
)


def make_plan(offset: int = 0) -> DailyPlan:
    return DailyPlan.create(
        crop_targets={
            crop: index + offset for index, crop in enumerate(CROP_ORDER)
        },
        animal_targets={
            animal: index + offset for index, animal in enumerate(ANIMAL_ORDER)
        },
        land_count=2,
        fertilizer_by_crop={crop: offset for crop in CROP_ORDER},
        care_by_animal={animal: offset for animal in ANIMAL_ORDER},
        sell_quantities={
            product: {anchor: offset for anchor in SELL_BIN_ANCHORS}
            for product in PRODUCT_ORDER
        },
    )


def provenance() -> dict:
    return {
        "manager": "bc-manager",
        "checkpoint": "bc-e-best.pt:sha256=abc123",
        "model_variant": "E",
        "seed": 17,
        "seat": 1,
        "opening_identity": "standard_mixed",
        "source_repo_sha": "3726e373c65b8221c4062138174898f6cf756119",
        "backend": {"name": "fast", "version": "local"},
        "engine": {"name": "kaggriculture", "version": "1.32.7"},
        "recording_window": {"start_day": 5, "end_day": 7},
    }


def make_tape() -> FixedPlanTape:
    return FixedPlanTape.create(
        # Input order is deliberately not canonical; serialization is.
        plans=[(7, make_plan(2)), (5, make_plan(1)), (6, make_plan(1))],
        provenance=provenance(),
    )


def test_tape_json_round_trip_is_canonical_and_daily_plan_exact():
    tape = make_tape()
    rebuilt = FixedPlanTape.from_json(tape.to_json())

    assert tape.to_json() == rebuilt.to_json()
    assert tape.artifact_sha256 == rebuilt.artifact_sha256
    assert [day for day, _ in tape.plans] == [5, 6, 7]
    assert [plan.to_json_dict() for _, plan in tape.plans] == [
        plan.to_json_dict() for _, plan in rebuilt.plans
    ]
    document = json.loads(tape.to_json())
    assert all(
        "age" not in json.dumps(entry["plan"])
        for entry in document["plans"]
    )
    assert document["plans"][0]["plan"] == tape.plans[0][1].to_json_dict()


def test_two_providers_share_strategy_even_with_diverged_observations():
    tape = make_tape()
    baseline = FixedPlanTapeProvider(tape)
    candidate = FixedPlanTapeProvider(tape)
    assert isinstance(baseline, PlanProvider)

    baseline_obs = {"day": 5, "money": 1, "farms": [], "private": {}}
    candidate_obs = {
        "day": "5",
        "money": 999999,
        "farms": [{"tiles": [["WEED"]]}],
        "private": {"seeds": {"WHEAT": 400}},
    }
    assert baseline.daily_plan(baseline_obs, 1) == candidate.daily_plan(
        candidate_obs, 1, {"workers_hired": 99, "hire_cost": 999}
    )


def test_repeated_same_day_lookup_is_idempotent_and_state_free():
    provider = FixedPlanTapeProvider(make_tape())
    first = provider.daily_plan({"day": 5, "hour": 0}, 1)
    second = provider.daily_plan(
        {"day": 5, "hour": 23, "money": -100}, 1, {"hire_cost": 123}
    )
    assert second is first


@pytest.mark.parametrize(
    ("obs", "seat", "match"),
    [
        ({}, 1, "missing required day"),
        ({"day": 4}, 1, "outside recording window"),
        ({"day": 8}, 1, "outside recording window"),
        ({"day": 5}, 0, "seat mismatch"),
        ({"day": 5}, 2, "seat must be 0 or 1"),
        ({"day": 1.5}, 1, "integral"),
    ],
)
def test_provider_fails_loudly_for_invalid_requests(obs, seat, match):
    with pytest.raises(PlanTapeError, match=match):
        FixedPlanTapeProvider(make_tape()).daily_plan(obs, seat)


def test_duplicate_days_are_rejected_in_memory_and_json():
    with pytest.raises(PlanTapeError, match="duplicate plan day 5"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (5, make_plan(1))], provenance=provenance()
        )

    document = make_tape().to_json_dict()
    document["plans"].append(copy.deepcopy(document["plans"][0]))
    with pytest.raises(PlanTapeError, match="duplicate plan day 5"):
        FixedPlanTape.from_json_dict(document)


def test_interior_gaps_are_rejected_at_construction_and_load():
    with pytest.raises(PlanTapeError, match=r"missing=\[6\]"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (7, make_plan(1))],
            provenance=provenance(),
        )

    document = make_tape().to_json_dict()
    document["plans"] = [entry for entry in document["plans"] if entry["day"] != 6]
    with pytest.raises(PlanTapeError, match=r"missing=\[6\]"):
        FixedPlanTape.from_json_dict(document)


def test_empty_inverted_and_extra_window_coverage_is_rejected():
    empty = provenance()
    with pytest.raises(PlanTapeError, match=r"missing=\[5, 6, 7\]"):
        FixedPlanTape.create(plans=[], provenance=empty)

    inverted = provenance()
    inverted["recording_window"] = {"start_day": 7, "end_day": 5}
    with pytest.raises(PlanTapeError, match="start_day"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (6, make_plan()), (7, make_plan())],
            provenance=inverted,
        )

    extra = provenance()
    with pytest.raises(PlanTapeError, match=r"extra=\[8\]"):
        FixedPlanTape.create(
            plans=[
                (5, make_plan()), (6, make_plan()), (7, make_plan()),
                (8, make_plan()),
            ],
            provenance=extra,
        )


@pytest.mark.parametrize("seat", [-1, 2, True])
def test_only_production_seats_are_accepted(seat):
    bad = provenance()
    bad["seat"] = seat
    with pytest.raises(PlanTapeError, match="provenance.seat"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (6, make_plan()), (7, make_plan())],
            provenance=bad,
        )


@pytest.mark.parametrize(
    "field",
    ["backend", "engine"],
)
def test_backend_and_engine_require_nonempty_name_and_version(field):
    for malformed in ({}, {"name": "", "version": "1"},
                      {"name": "fast"}, {"name": "fast", "version": ""}):
        bad = provenance()
        bad[field] = malformed
        with pytest.raises(PlanTapeError, match=field):
            FixedPlanTape.create(
                plans=[(5, make_plan()), (6, make_plan()), (7, make_plan())],
                provenance=bad,
            )
        document = make_tape().to_json_dict()
        document["provenance"][field] = malformed
        with pytest.raises(PlanTapeError, match=field):
            FixedPlanTape.from_json_dict(document)


def test_malformed_plan_and_unsupported_schema_are_rejected():
    malformed = make_tape().to_json_dict()
    malformed["plans"][0]["plan"]["land_count"] = 0
    with pytest.raises(PlanTapeError, match="malformed|land_count"):
        FixedPlanTape.from_json_dict(malformed)

    unsupported = make_tape().to_json_dict()
    unsupported["schema_version"] = 99
    with pytest.raises(PlanTapeError, match="unsupported.*schema_version"):
        FixedPlanTape.from_json_dict(unsupported)


def test_explicit_checkpoint_and_provenance_are_required():
    missing_checkpoint = provenance()
    missing_checkpoint["checkpoint"] = ""
    with pytest.raises(PlanTapeError, match="checkpoint"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (6, make_plan()), (7, make_plan())],
            provenance=missing_checkpoint,
        )

    missing_source_sha = provenance()
    del missing_source_sha["source_repo_sha"]
    with pytest.raises(PlanTapeError, match="source_repo_sha"):
        FixedPlanTape.create(
            plans=[(5, make_plan()), (6, make_plan()), (7, make_plan())],
            provenance=missing_source_sha,
        )


def test_fingerprint_changes_when_canonical_tape_content_changes():
    first = make_tape()
    second = FixedPlanTape.create(
        plans=[(5, make_plan(1)), (6, make_plan(2)), (7, make_plan(3))],
        provenance=provenance(),
    )
    assert first.artifact_sha256 != second.artifact_sha256
