"""Behavioral tests for executor_v0 stage 3: layout and crop reconciliation.

Pure-logic coverage of issue #1 section 4: sacrifice ordering, sticky
structures, empty-first animal space, excess-only crop conversion, honest
unresolved deficits, deterministic ties. A real canonical board from the
ignored local sample proves determinism/no exceptions only.
"""

import copy
import json
from pathlib import Path

import pytest

from executor_v0.layout import (
    SacrificeConfig,
    plan_animal_layout,
    quadrant_of,
    reconcile_crops,
    sacrifice_score,
    tile_role,
)
from replay_daily.constants import ANIMALS

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"

ANCHOR = (0, 0)  # shed/farmer anchor for all distance math in these tests


# ------------------------------------------------------------------ helpers


def blank_board() -> list[list[None]]:
    return [[None] * 10 for _ in range(10)]


def filled_board(open_coords=()) -> list[list]:
    """NW quadrant fully covered by sticky occupied COOPs except open cells.

    Guarantees no unintended empty tile can win selection; tests open exactly
    the cells they reason about.
    """
    board: list[list] = blank_board()
    closed = set(open_coords)
    for y in range(5):
        for x in range(5):
            if (y, x) not in closed:
                board[y][x] = structure("COOP", "GOOSE")
    return board


def plant(crop: str = "WHEAT", *, planted_day: int = 1, yield_units: int = 0,
          watered_today: bool = True, fertilized_until_day: int = -1,
          harvestable: bool | None = None,
          fertilizer_active: bool | None = None,
          days_until_next_harvest: int | None | object = ...) -> dict:
    tile = {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "yield_units": yield_units, "watered_today": watered_today,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": -1, "consecutive_unwatered": 0,
        "derived": {
            "age_days": planted_day,
            "currently_harvestable":
                (yield_units > 0 and planted_day >= 2)
                if harvestable is None else harvestable,
            "days_until_next_harvest": (
                0 if (yield_units > 0 and planted_day >= 2) else 1)
            if days_until_next_harvest is ... else days_until_next_harvest,
            "fertilizer_active":
                (fertilized_until_day >= 5) if fertilizer_active is None
                else fertilizer_active,
            "past_lifespan": False,
            "starving": False,
            "days_until_next_product": None,
        },
    }
    return tile


def structure(kind: str, animal: str | None = None, **extra) -> dict:
    tile = {"kind": kind}
    if animal is not None:
        tile["animal"] = animal
        tile.setdefault("placed_day", 0)
        tile.setdefault("yield_units", 0)
        tile.setdefault("consecutive_unfed", 0)
        tile.setdefault("fed_today", True)
    return {**tile, **extra}


def coords(placements) -> list[tuple[int, int]]:
    return [p.coord for p in placements]


# ------------------------------------------------------- sacrifice ordering


def test_sacrifice_prefers_young_unfertilized_low_progress():
    young = plant("WHEAT", planted_day=0, yield_units=0,
                  fertilized_until_day=-1, harvestable=False)
    mature = plant("WHEAT", planted_day=6, yield_units=3,
                   fertilized_until_day=9, harvestable=True)
    cfg = SacrificeConfig()
    # Young far away still beats mature right next to the shed.
    assert sacrifice_score(young, (9, 9), anchor=ANCHOR, config=cfg) < \
        sacrifice_score(mature, (0, 0), anchor=ANCHOR, config=cfg)


def test_sacrifice_weights_are_configurable_and_change_ordering():
    # Distance zeroed so only the sunk-investment terms decide.
    a = plant("WHEAT", planted_day=8, yield_units=0, harvestable=False,
              fertilized_until_day=-1)   # old but bare
    b = plant("WHEAT", planted_day=0, yield_units=5, harvestable=False,
              fertilized_until_day=-1)   # young but yield-heavy
    coord_a, coord_b = (0, 1), (0, 3)
    default = SacrificeConfig(distance_weight=0.0)
    assert sacrifice_score(b, coord_b, anchor=ANCHOR, config=default) > \
        sacrifice_score(a, coord_a, anchor=ANCHOR, config=default)
    flipped = SacrificeConfig(distance_weight=0.0, age_weight=50.0,
                              yield_units_weight=0.0)
    assert sacrifice_score(b, coord_b, anchor=ANCHOR, config=flipped) < \
        sacrifice_score(a, coord_a, anchor=ANCHOR, config=flipped)


def test_sacrifice_tie_breaks_deterministically_on_yx():
    tile = plant("CARROT")
    cfg = SacrificeConfig(distance_weight=0.0)
    assert sacrifice_score(tile, (4, 2), anchor=ANCHOR, config=cfg) == \
        sacrifice_score(tile, (2, 4), anchor=ANCHOR, config=cfg)
    board = filled_board(open_coords={(2, 4), (4, 2)})
    board[4][2] = tile
    board[2][4] = copy.deepcopy(tile)
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=ANCHOR,
                                config=cfg)
    assert coords(result.placements) == [(2, 4)]  # lower (y, x) wins
    assert result.placements[0].source == "crop_sacrifice"


def test_missing_or_null_derived_is_penalized_conservatively():
    assessed = plant("WHEAT")  # full derived timing
    bare = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "yield_units": 0}  # no derived struct at all
    null_timing = plant("WHEAT", days_until_next_harvest=None)
    cfg = SacrificeConfig(distance_weight=0.0)
    assert sacrifice_score(assessed, (1, 1), anchor=ANCHOR, config=cfg) < \
        sacrifice_score(null_timing, (1, 2), anchor=ANCHOR, config=cfg)
    assert sacrifice_score(null_timing, (1, 2), anchor=ANCHOR, config=cfg) < \
        sacrifice_score(bare, (1, 3), anchor=ANCHOR, config=cfg)


def test_tile_role_classification():
    assert tile_role(None) == "empty"
    assert tile_role("LOCKED") == "locked"
    assert tile_role("WEED") == "weed"
    assert tile_role("MYSTERY") == "locked"
    assert tile_role(plant()) == "plant"
    assert tile_role(structure("COOP", "GOOSE")) == "animal_structure"
    assert tile_role(structure("PASTURE")) == "empty_structure"
    assert tile_role({"kind": "SOMETHING"}) == "other"


def test_quadrant_layout():
    assert quadrant_of(0, 0) == "NW"
    assert quadrant_of(4, 9) == "NE"
    assert quadrant_of(5, 0) == "SW"
    assert quadrant_of(9, 9) == "SE"


# ------------------------------------------------------------ crop reconcile


def test_reconcile_preserves_matching_crops_when_balanced():
    board = filled_board(open_coords={(1, 1), (1, 2), (2, 1)})
    board[1][1] = plant("WHEAT")
    board[1][2] = plant("WHEAT")
    board[2][1] = plant("CARROT")
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"WHEAT": 2, "CARROT": 1},
                             anchor=ANCHOR)
    assert result.digs == ()
    assert result.plants == ()
    assert result.unresolved_deficits == ()


def test_reconcile_reduction_keeps_most_valuable_matches():
    board = filled_board(open_coords={(0, 1), (0, 2), (0, 3)})
    board[0][1] = plant("WHEAT", planted_day=7, yield_units=3,
                        harvestable=True, fertilized_until_day=9)
    board[0][2] = plant("WHEAT", planted_day=0, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[0][3] = plant("WHEAT", planted_day=1, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    # No deficits anywhere: true excess is left alone (no pointless digs).
    idle = reconcile_crops(board, unlocked_quadrants=("NW",),
                           crop_targets={"WHEAT": 2}, anchor=ANCHOR)
    assert idle.digs == () and idle.plants == ()
    # A TOMATO deficit with no empty tiles consumes the cheapest released
    # WHEAT excess; the harvestable match is preserved.
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"WHEAT": 2, "TOMATO": 1},
                             anchor=ANCHOR)
    assert [(d.coord, d.crop) for d in result.digs] == [((0, 2), "WHEAT")]
    assert [(p.coord, p.crop) for p in result.plants] == [((0, 2), "TOMATO")]
    assert result.unresolved_deficits == ()


def test_reconcile_deficit_uses_empty_tiles_before_any_sacrifice():
    board = filled_board(open_coords={(0, 1), (0, 2), (0, 3)})
    board[0][1] = plant("WHEAT", planted_day=0, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[0][2] = plant("CARROT", planted_day=0, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[0][3] = None  # the only empty legal tile
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"WHEAT": 2, "CARROT": 1},
                             anchor=ANCHOR)
    # WHEAT deficit 1 filled from the EMPTY tile; CARROT excess untouched.
    assert result.digs == ()
    assert [(p.coord, p.crop) for p in result.plants] == [((0, 3), "WHEAT")]
    assert result.unresolved_deficits == ()


def test_reconcile_mixed_targets_deterministic_and_excess_only():
    board = filled_board(open_coords={(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)})
    # WHEAT: 3 present, target 1 -> two true excess.
    board[0][1] = plant("WHEAT", planted_day=0, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[0][2] = plant("WHEAT", planted_day=1, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[0][3] = plant("WHEAT", planted_day=7, yield_units=3,
                        harvestable=True, fertilized_until_day=9)
    # TOMATO: 1 present, target 3 -> deficit 2; one empty tile available.
    board[1][1] = plant("TOMATO", planted_day=0, yield_units=0,
                        harvestable=False, fertilized_until_day=-1)
    board[1][2] = None  # the single empty legal tile
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"WHEAT": 1, "TOMATO": 3},
                             anchor=ANCHOR)
    # Empty tile goes to TOMATO first (canonical order processing).
    assert (result.plants[0].coord, result.plants[0].crop) == ((1, 2), "TOMATO")
    # Remaining TOMATO deficit 1 converts the cheapest WHEAT excess.
    assert len(result.digs) == 1
    coord, crop = result.digs[0].coord, result.digs[0].crop
    assert crop == "WHEAT"
    assert coord == (0, 1)  # youngest excess; keeper (0,3) is harvestable
    replacement = [p for p in result.plants if p.coord == coord]
    assert replacement and replacement[0].crop == "TOMATO"
    assert result.unresolved_deficits == ()
    # Determinism: identical inputs -> identical outputs.
    again = reconcile_crops(copy.deepcopy(board), unlocked_quadrants=("NW",),
                            crop_targets={"WHEAT": 1, "TOMATO": 3},
                            anchor=ANCHOR)
    assert again == result


def test_reconcile_unresolved_deficit_is_honest():
    board = filled_board()  # nothing selectable anywhere in the quadrant
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"MELON": 4}, anchor=ANCHOR)
    assert result.digs == () and result.plants == ()
    assert result.unresolved_deficits == (("MELON", 4),)


def test_reconcile_never_touches_sticky_structures_or_locked_region():
    board = filled_board()
    board[9][9] = plant("WHEAT")  # SE quadrant, locked here
    result = reconcile_crops(board, unlocked_quadrants=("NW",),
                             crop_targets={"WHEAT": 1}, anchor=ANCHOR)
    assert result.digs == () and result.plants == ()
    assert result.unresolved_deficits == (("WHEAT", 1),)


def test_reconcile_validates_targets():
    board = filled_board()
    with pytest.raises(ValueError):
        reconcile_crops(board, unlocked_quadrants=("NW",),
                        crop_targets={"OATS": 1}, anchor=ANCHOR)
    with pytest.raises(ValueError):
        reconcile_crops(board, unlocked_quadrants=("NW",),
                        crop_targets={"WHEAT": -1}, anchor=ANCHOR)


# ------------------------------------------------------------- animal layout


def test_animal_layout_empty_nearby_first_no_reserved_zone():
    board = filled_board(open_coords={(0, 1), (2, 0)})
    board[0][1] = plant("WHEAT")  # nearest selectable tile is a crop
    board[2][0] = None            # farther empty tile
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=(0, 0))
    # No reservation logic: the empty tile is used and the nearer crop is
    # untouched because empties suffice.
    assert coords(result.placements) == [(2, 0)]
    assert result.placements[0].source == "new_build"
    assert result.unresolved == ()


def test_animal_layout_matching_empty_structure_before_new_build():
    board = filled_board(open_coords={(4, 4), (0, 1)})
    board[4][4] = structure("COOP")  # matching but far
    board[0][1] = None               # near empty build spot
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=(0, 0))
    assert coords(result.placements) == [(4, 4)]
    assert result.placements[0].source == "empty_structure"


def test_animal_layout_structure_compatibility():
    board = filled_board(open_coords={(0, 1), (2, 2)})
    board[0][1] = structure("PASTURE")  # wrong type for GOOSE
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=(0, 0))
    # Wrong-type empty structure is not converted; GOOSE builds elsewhere.
    assert coords(result.placements) == [(2, 2)]
    assert result.placements[0].source == "new_build"
    assert result.placements[0].structure == ANIMALS["GOOSE"]["structure"]

    board2 = filled_board(open_coords={(0, 1), (2, 2)})
    board2[0][1] = structure("COOP")    # wrong type for COW
    result2 = plan_animal_layout(board2, unlocked_quadrants=("NW",),
                                 animals_needed={"COW": 1}, anchor=(0, 0))
    assert coords(result2.placements) == [(2, 2)]
    assert result2.placements[0].structure == ANIMALS["COW"]["structure"]


def test_animal_layout_sacrifices_only_when_necessary_and_cheapest():
    board = filled_board(open_coords={(0, 1), (0, 2)})
    cheap = plant("WHEAT", planted_day=0, yield_units=0, harvestable=False,
                  fertilized_until_day=-1)
    dear = plant("WHEAT", planted_day=8, yield_units=4, harvestable=True,
                 fertilized_until_day=9)
    board[0][1] = cheap
    board[0][2] = dear
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=(0, 0))
    assert coords(result.placements) == [(0, 1)]
    assert result.placements[0].source == "crop_sacrifice"


def test_animal_layout_excludes_locked_weed_and_occupied_structures():
    board = filled_board(open_coords={(0, 1), (0, 2), (2, 2)})
    board[0][1] = "LOCKED"
    board[0][2] = "WEED"
    board[2][2] = None
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"GOOSE": 1}, anchor=(0, 0))
    assert coords(result.placements) == [(2, 2)]


def test_animal_layout_locked_quadrant_and_unresolved_honesty():
    board = filled_board()
    board[9][9] = None  # SE quadrant not unlocked
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"SHEEP": 2}, anchor=(0, 0))
    assert result.placements == ()
    assert result.unresolved == (("SHEEP", 2),)


def test_animal_layout_multiple_species_deterministic_order():
    board = filled_board(open_coords={(0, 1), (0, 2), (0, 3)})
    board[0][1] = structure("PASTURE")  # empty structure: SHEEP match
    board[0][2] = None
    board[0][3] = None
    result = plan_animal_layout(board, unlocked_quadrants=("NW",),
                                animals_needed={"SHEEP": 1, "GOOSE": 1},
                                anchor=(0, 0))
    # GOOSE processed first (canonical order); its empty PASTURE match is
    # NOT usable, so GOOSE builds at the nearest empty (0,2). SHEEP then
    # takes the empty PASTURE structure regardless of distance.
    by_animal = {p.animal: p for p in result.placements}
    assert by_animal["GOOSE"].coord == (0, 2)
    assert by_animal["GOOSE"].source == "new_build"
    assert by_animal["SHEEP"].coord == (0, 1)
    assert by_animal["SHEEP"].source == "empty_structure"
    assert list(by_animal) == ["GOOSE", "SHEEP"]  # deterministic ordering


def test_animal_layout_validates_requests():
    board = filled_board()
    with pytest.raises(ValueError):
        plan_animal_layout(board, unlocked_quadrants=("NW",),
                           animals_needed={"DRAGON": 1}, anchor=ANCHOR)
    with pytest.raises(ValueError):
        plan_animal_layout(board, unlocked_quadrants=("NW",),
                           animals_needed={"GOOSE": -1}, anchor=ANCHOR)


# ---------------------------------------------------------------- real smoke


@pytest.mark.skipif(not SAMPLE.exists(),
                    reason="local real sample not present")
def test_real_canonical_board_smoke_deterministic_no_exceptions():
    """Determinism/no-exception smoke on a real board; no quality claim."""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    from replay_daily.extractor import extract_replay

    record = next(r for r in extract_replay(raw, partition_date="2026-08-20")
                  if r["metadata"]["seat"] == 0 and r["day"] == 7)
    start = record["start"]
    board = start["self"]["board"]
    anchor = tuple(start["self"]["farmer"])
    unlocked = start["self"]["unlocked_quadrants"]

    targets = {"WHEAT": 6, "CARROT": 4, "TOMATO": 2, "STRAWBERRY": 1,
               "MELON": 1}
    animals = {"GOOSE": 2, "COW": 1, "SHEEP": 1}

    crop_one = reconcile_crops(board, unlocked_quadrants=unlocked,
                               crop_targets=targets, anchor=anchor)
    crop_two = reconcile_crops(copy.deepcopy(board),
                               unlocked_quadrants=unlocked,
                               crop_targets=targets, anchor=anchor)
    assert crop_one == crop_two

    animal_one = plan_animal_layout(board, unlocked_quadrants=unlocked,
                                    animals_needed=animals, anchor=anchor)
    animal_two = plan_animal_layout(copy.deepcopy(board),
                                    unlocked_quadrants=unlocked,
                                    animals_needed=animals, anchor=anchor)
    assert animal_one == animal_two

    # Every placement lands on a legally selectable tile class in an
    # unlocked quadrant with the correct structure kind.
    usable_roles = {"empty", "empty_structure", "plant"}
    for placement in animal_one.placements:
        y, x = placement.coord
        assert quadrant_of(y, x) in unlocked
        assert tile_role(board[y][x]) in usable_roles
        assert placement.structure == ANIMALS[placement.animal]["structure"]
