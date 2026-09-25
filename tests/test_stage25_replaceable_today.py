"""Focused Stage 2.5 replaceable-today lifecycle and parity tests."""

from __future__ import annotations

import copy

import numpy as np

from replay_daily.lifecycle import canonical_board, replaceable_today
from rl_manager.stage25_adapter import _morning_crop_observation_arrays
from rl_manager.stage25_provider import Stage25PlanProvider


def _plant(
    crop: str, *, planted_day: int, yield_units: int,
    watered_today: bool = False, fertilized_until_day: int = -1,
    max_lifespan_step: int = -1,
) -> dict:
    return {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "yield_units": yield_units, "watered_today": watered_today,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": max_lifespan_step,
    }


def _board(*tiles: dict) -> list[list[object]]:
    board: list[list[object]] = [[None for _ in range(10)] for _ in range(10)]
    for index, tile in enumerate(tiles):
        board[0][index] = tile
    return board


def test_wheat_ready_or_reachable_by_ordinary_water_counts() -> None:
    board = _board(
        _plant("WHEAT", planted_day=0, yield_units=3),
        _plant("WHEAT", planted_day=0, yield_units=2),
    )
    assert replaceable_today(board, 2) == (2, 0, 0, 0, 0)


def test_wheat_that_cannot_reach_threshold_today_does_not_count() -> None:
    board = _board(
        _plant("WHEAT", planted_day=1, yield_units=2),
        _plant("WHEAT", planted_day=0, yield_units=2, watered_today=True),
    )
    assert replaceable_today(board, 2) == (0, 0, 0, 0, 0)


def test_day_29_threshold_is_two_without_a_yield_one_terminal_exception() -> None:
    board = _board(
        _plant("WHEAT", planted_day=0, yield_units=2, watered_today=True),
        _plant("WHEAT", planted_day=0, yield_units=1, watered_today=True),
    )
    assert replaceable_today(board, 29) == (1, 0, 0, 0, 0)


def test_final_strawberry_retirement_counts_but_younger_productive_does_not() -> None:
    board = _board(
        _plant("STRAWBERRY", planted_day=0, yield_units=1),
        _plant("STRAWBERRY", planted_day=2, yield_units=1),
    )
    assert replaceable_today(board, 16) == (0, 0, 0, 1, 0)


def test_carrot_and_melon_count_only_when_ordinary_harvest_is_mature() -> None:
    board = _board(
        _plant("CARROT", planted_day=0, yield_units=1),
        _plant("MELON", planted_day=0, yield_units=1),
    )
    assert replaceable_today(board, 10) == (0, 1, 0, 0, 1)
    assert replaceable_today(board, 2) == (0, 1, 0, 0, 0)


def test_tomato_uses_mechanics_derived_final_production_age() -> None:
    board = _board(
        _plant("TOMATO", planted_day=0, yield_units=1),
        _plant("TOMATO", planted_day=1, yield_units=1),
    )
    assert replaceable_today(board, 11) == (0, 0, 1, 0, 0)


def test_late_hour_cutoffs_are_h21_for_one_shot_and_h20_for_recurring() -> None:
    wheat = _board(_plant("WHEAT", planted_day=0, yield_units=3))
    strawberry = _board(_plant("STRAWBERRY", planted_day=0, yield_units=1))
    assert replaceable_today(wheat, 2, 2 * 24 + 21) == (1, 0, 0, 0, 0)
    assert replaceable_today(wheat, 2, 2 * 24 + 22) == (0, 0, 0, 0, 0)
    assert replaceable_today(strawberry, 16, 16 * 24 + 20) == (0, 0, 0, 1, 0)
    assert replaceable_today(strawberry, 16, 16 * 24 + 21) == (0, 0, 0, 0, 0)


def _live_observation(board: list[list[object]], day: int) -> dict:
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0,
        "money": 3000.0, "tiles": board,
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day, "hour": 0, "step": day * 24,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_offline_and_live_replaceable_today_are_identical() -> None:
    raw_board = _board(_plant("STRAWBERRY", planted_day=0, yield_units=1))
    canonical = canonical_board(raw_board, 16, 16 * 24)
    offline_record = {"start": {"self": {
        "board": canonical, "unlocked_quadrants": ["NW"],
    }}}
    offline = _morning_crop_observation_arrays([offline_record], [16])

    provider = Stage25PlanProvider(episode_id=9, seat=0, manager_start_day=16)
    live_inputs, _, _ = provider._stage_observation(
        _live_observation(raw_board, 16), None)

    assert offline["crop_capacity"].dtype == np.int16
    assert offline["replaceable_today"].dtype == np.int16
    assert offline["available_crop_slots"].dtype == np.int16
    np.testing.assert_array_equal(offline["crop_capacity"], live_inputs["crop_capacity"])
    np.testing.assert_array_equal(offline["replaceable_today"], live_inputs["replaceable_today"])
    np.testing.assert_array_equal(offline["available_crop_slots"], live_inputs["available_crop_slots"])
    assert offline["available_crop_slots"].tolist() == [24]
