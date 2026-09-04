"""Focused passive manager crop-intent economic diagnostics tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bc_manager.constants import CROP_ORDER
from bc_manager_jax.model import tiny_manager_config
from rl_manager.diagnostics import (aggregate_economic_diagnostics,
                                    build_economic_diagnostics)


def _result(rows):
    return SimpleNamespace(
        episode_index=0,
        seed=0,
        composition="candidate_vs_frozen",
        final_banks=[3000.0, 3000.0],
        rewards=[0.0, 0.0],
        policy_identities=(
            {"seat": 0, "trainable": True},
            {"seat": 1, "trainable": False},
        ),
        utilization_snapshots=[
            {"seat": seat, "boundary": "terminal",
             "land_quadrants_owned": 1, "productive_occupancy": 0,
             "crop_squares": 0, "animal_squares": 0,
             "productive_squares": 0}
            for seat in (0, 1)
        ],
        land_purchase_events=[],
        manager_crop_rows=rows,
    )


def _row(day, vector, *, episode=0, seat=0, unresolved=(), achieved=None,
         trainable=True):
    return {
        "episode_index": episode,
        "seat": seat,
        "day": day,
        "trainable": trainable,
        "requested_crop_targets": dict(zip(CROP_ORDER, vector)),
        "unresolved_generator": list(unresolved),
        "achieved_final_crops": (dict(zip(CROP_ORDER, achieved))
                                 if achieved is not None else None),
    }


def test_requested_totals_species_mix_and_authoritative_saturation():
    max_target = tiny_manager_config(count_max=7).count_max
    rows = [
        _row(4, (1, 2, 0, 0, 0)),
        _row(5, (0, 2, 4, 0, 0)),
        _row(6, (max_target,) * len(CROP_ORDER)),
    ]
    intent = aggregate_economic_diagnostics(
        [_result(rows)], crop_action_max=max_target)["manager_crop_intent"]

    assert intent["requested_total"]["count"] == 3
    assert intent["requested_total"]["mean"] == pytest.approx(44 / 3)
    assert intent["requested_total"]["median"] == 6.0
    assert intent["requested_total"]["p10"] == pytest.approx(3.6)
    assert intent["requested_total"]["p90"] == pytest.approx(29.2)
    assert intent["requested_total"]["min"] == 3.0
    assert intent["requested_total"]["max"] == 35.0
    assert intent["requested_by_species"]["WHEAT"]["mean"] == 8 / 3
    assert intent["requested_by_species"]["CARROT"]["fraction_nonzero"] == 1.0
    assert intent["mix"]["mean_distinct_species_requested"] == 3.0
    assert intent["mix"]["fraction_single_species"] == 0.0
    assert intent["mix"]["mean_dominant_species_fraction"] == pytest.approx(23 / 45)
    assert intent["saturation"]["component_count"] == 15
    assert intent["saturation"]["component_at_max_count"] == 5
    assert intent["saturation"]["fraction_crop_components_at_action_max"] == 1 / 3
    assert intent["saturation"]["all_components_at_max_row_count"] == 1
    assert intent["saturation"]["by_species"]["MELON"]["action_max"] == max_target


def test_vector_change_does_not_cross_episode_or_seat_boundaries():
    rows = [
        _row(4, (1, 0, 0, 0, 0)),
        _row(5, (1, 0, 0, 0, 0)),
        _row(6, (0, 1, 0, 0, 0)),
        _row(4, (9, 0, 0, 0, 0), episode=1),
        _row(5, (8, 0, 0, 0, 0), seat=1),
    ]
    intent = aggregate_economic_diagnostics([_result(rows)], crop_action_max=9)[
        "manager_crop_intent"]
    assert intent[
        "fraction_target_vector_changed_from_previous_manager_day"] == {
            "changed_count": 1, "comparable_count": 2, "fraction": 0.5}


def test_by_day_late_buckets_unresolved_entries_and_shortfall():
    rows = [
        _row(4, (1, 1, 0, 0, 0),
             unresolved=("crop_deficit_unresolved:WHEAT:2", "other:9"),
             achieved=(0, 1, 0, 0, 0)),
        _row(20, (2, 0, 0, 0, 0), achieved=(3, 0, 0, 0, 0)),
        _row(25, (3, 0, 0, 0, 0),
             unresolved=("crop_deficit_unresolved:CARROT:4",),
             achieved=(1, 0, 0, 0, 0)),
        _row(29, (4, 0, 0, 0, 0), achieved=(0, 0, 0, 0, 0)),
    ]
    intent = aggregate_economic_diagnostics([_result(rows)], crop_action_max=9)[
        "manager_crop_intent"]

    assert intent["by_manager_day"]["4"] == {
        "row_count": 1,
        "requested_total_mean": 2.0,
        "requested_by_species": {**{crop: 0.0 for crop in CROP_ORDER},
                                  "WHEAT": 1.0, "CARROT": 1.0},
    }
    assert intent["late_game"]["20-24"]["row_count"] == 1
    assert intent["late_game"]["25-27"]["requested_total_mean"] == 3.0
    assert intent["late_game"]["28-29"]["requested_by_species_mean"]["WHEAT"] == 4.0
    unresolved = intent["unresolved_crop_deficit"]
    assert unresolved["manager_row_count"] == 4
    assert unresolved["rows_with_unresolved_deficit"] == 2
    assert unresolved["total_units"] == 6
    assert unresolved["by_species"]["WHEAT"]["total_units"] == 2
    assert unresolved["by_species"]["CARROT"]["fraction_rows_nonzero"] == 1 / 4
    shortfall = intent["end_of_day_shortfall"]
    assert shortfall["day_count"] == 4
    assert shortfall["days_with_shortfall"] == 3
    assert shortfall["total_units"] == 7
    assert shortfall["by_species"]["WHEAT"]["total_units"] == 7
    assert shortfall["by_species"]["CARROT"]["total_units"] == 0
    assert shortfall["requested_total"]["count"] == 4
    assert shortfall["achieved_final_crop_total"]["min"] == 0


def test_low_telemetry_rows_need_no_turn_snapshots_or_observations():
    row = _row(4, (1, 0, 0, 0, 0), achieved=(0, 0, 0, 0, 0))
    result = _result([row])
    result.turn_trace = None
    result.observations = None
    intent = aggregate_economic_diagnostics([result], crop_action_max=9)[
        "manager_crop_intent"]
    assert intent["requested_total"]["count"] == 1
    assert intent["end_of_day_shortfall"]["total_units"] == 1


def test_persisted_economic_payload_contains_complete_manager_section():
    result = _result([_row(29, (1, 2, 0, 0, 0), achieved=(0, 1, 0, 0, 0))])
    payload = build_economic_diagnostics([result], crop_action_max=7)
    json.dumps(payload, allow_nan=False)
    intent = payload["aggregate"]["manager_crop_intent"]
    assert set(intent["by_manager_day"]) == {"29"}
    assert intent["saturation"]["by_species"]["WHEAT"]["action_max"] == 7
