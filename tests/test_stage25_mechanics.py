"""Focused Stage 2.5 Packet 1A mechanics tests."""

import subprocess
import sys

import pytest

from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
    ANIMAL_ORDER,
    CROP_ORDER,
    PhysicalContext,
    animal_acquisition_deficits,
    animal_prefix_is_feasible,
    animal_target_support_mask,
    crop_class_to_delta,
    crop_delta_support_mask,
    crop_delta_to_class,
    decode_supported_crop_goals,
    initialize_crop_ledger,
    land_class_to_target,
    land_target_support_mask,
    land_target_to_class,
    physical_context_from_board,
    physical_crop_capacity,
    required_new_housing_cells,
    transition_crop_goal,
    transition_crop_ledger,
    transition_crop_ledger_classes,
    unplaced_animal_counts,
)


def context(*, cells=(25, 50, 75, 100), placed=(0, 0, 0),
            coops=0, pastures=0, unplaced=(0, 0, 0)):
    return PhysicalContext(1, cells, placed, coops, pastures, unplaced)


def test_action_order_and_class_delta_round_trips():
    assert ACTION_ORDER == (
        "land", "goose", "cow", "sheep", "wheat", "carrot", "tomato",
        "strawberry", "melon")
    assert ACTION_CLASS_COUNTS == (4, 101, 101, 101, 201, 201, 201, 201, 201)
    assert [land_class_to_target(i) for i in range(4)] == [1, 2, 3, 4]
    assert [land_target_to_class(i) for i in range(1, 5)] == [0, 1, 2, 3]
    assert land_target_support_mask(3) == (False, False, True, True)
    assert [crop_class_to_delta(i) for i in (0, 100, 200)] == [-100, 0, 100]
    assert [crop_delta_to_class(i) for i in (-100, 0, 100)] == [0, 100, 200]


def test_crop_ledger_transition_round_trip_and_hold():
    previous = (60, 0, 25, 100, 1)
    deltas = (-60, 0, 10, -100, 99)
    classes = tuple(crop_delta_to_class(delta) for delta in deltas)
    assert transition_crop_ledger(previous, deltas) == (0, 0, 35, 0, 100)
    assert transition_crop_ledger_classes(previous, classes) == (0, 0, 35, 0, 100)
    assert crop_class_to_delta(100) == 0
    assert transition_crop_goal(60, 0) == 60
    assert initialize_crop_ledger((3, 4, 5, 6, 7)) == (3, 4, 5, 6, 7)


@pytest.mark.parametrize("previous,delta", [(0, -1), (100, 1), (40, 101)])
def test_crop_transition_rejects_without_clipping(previous, delta):
    with pytest.raises(ValueError):
        transition_crop_goal(previous, delta)


def test_physical_context_counts_sticky_structures_and_new_land():
    board = [["LOCKED"] * 10 for _ in range(10)]
    board[0][0] = None
    board[0][1] = {"kind": "PLANT", "crop": "WHEAT"}
    board[0][2] = {"kind": "COOP"}
    board[0][3] = {"kind": "PASTURE"}
    board[0][4] = {"kind": "COOP", "animal": "GOOSE"}
    result = physical_context_from_board(board, ("NW",))
    assert result.crop_build_cells_by_land == (2, 27, 52, 77)
    assert result.placed_animals == (1, 0, 0)
    assert result.reusable_empty_coops == 1
    assert result.reusable_empty_pastures == 1


def test_reusable_housing_and_shared_pasture_capacity():
    ctx = context(cells=(10, 35, 60, 85), coops=4, pastures=5)
    assert required_new_housing_cells(ctx, 1, (4, 3, 3)) == 1
    assert physical_crop_capacity(ctx, 1, (4, 3, 3)) == 9
    # Cow and sheep draw from one shared pasture pool, not two independent pools.
    assert required_new_housing_cells(ctx, 1, (0, 4, 4)) == 3


def test_animal_support_is_autoregressive_and_rejects_impossible_prefixes():
    ctx = context(cells=(3, 28, 53, 78), coops=0, pastures=0)
    goose = animal_target_support_mask(ctx, 1, 0)
    assert goose[0] and goose[3] and not goose[4]
    cow = animal_target_support_mask(ctx, 1, 1, (3,))
    assert cow[0] and not cow[1]
    assert not animal_prefix_is_feasible(ctx, 1, (4, 0, 0))


def test_owned_unplaced_animals_satisfy_acquisition_without_changing_housing():
    ctx = context(placed=(1, 0, 0), unplaced=(2, 3, 0))
    assert animal_acquisition_deficits(ctx, (4, 2, 1)) == (1, 0, 1)
    assert physical_crop_capacity(ctx, 1, (4, 2, 1)) == 19
    assert unplaced_animal_counts({"GOOSE": 2}, ({"COW": 3},)) == (2, 3, 0)
    with pytest.raises(ValueError):
        animal_acquisition_deficits(ctx, (0, 0, 0))


def test_crop_residual_support_and_no_future_reservation():
    mask = crop_delta_support_mask(60, 50)
    assert mask[crop_delta_to_class(-60)]
    assert mask[crop_delta_to_class(-10)]
    assert not mask[crop_delta_to_class(0)]
    # A later species' old goal is not reserved in an earlier head.
    assert crop_delta_support_mask(0, 100)[crop_delta_to_class(100)]


def test_crop_support_capacity_invariants_and_forced_contraction():
    previous = (60, 25, 10, 0, 0)
    classes = (crop_delta_to_class(-60), crop_delta_to_class(0),
               crop_delta_to_class(0), crop_delta_to_class(0),
               crop_delta_to_class(0))
    assert decode_supported_crop_goals(previous, classes, 35) == (0, 25, 10, 0, 0)
    assert sum(decode_supported_crop_goals(previous, classes, 35)) <= 35
    assert crop_delta_support_mask(80, 0)[crop_delta_to_class(-80)]
    assert sum(crop_delta_support_mask(80, 0)) == 1


def test_hold_unavailable_when_earlier_goals_consume_residual_capacity():
    # K=20, residual=0: only full contraction is valid; HOLD is not repaired.
    mask = crop_delta_support_mask(20, 0)
    assert not mask[crop_delta_to_class(0)]
    assert mask[crop_delta_to_class(-20)]


def test_economic_state_is_not_a_physical_input():
    left = context(cells=(5, 30, 55, 80), placed=(1, 1, 1), coops=1, pastures=1)
    right = context(cells=(5, 30, 55, 80), placed=(1, 1, 1), coops=1, pastures=1)
    assert physical_crop_capacity(left, 1, (2, 2, 2)) == physical_crop_capacity(
        right, 1, (2, 2, 2))


def test_lightweight_module_has_no_heavy_framework_imports():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import rl_manager.stage25_mechanics; "
            "assert not ({name.split('.')[0] for name in sys.modules} "
            "& {'torch', 'jax', 'pyarrow'})"
        )],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
