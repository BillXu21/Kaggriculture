"""Focused Stage 2.5 Packet 3 crop-removal policy tests."""

from __future__ import annotations

import pytest

from executor_v0.plan import DailyPlan
from executor_v0.layout import SacrificeConfig, plan_animal_layout
from executor_v0.strip_executor import StripExecutorConfig
from executor_v0.strip_work import build_strip_work_plan
from replay_daily.lifecycle import canonical_board, replaceable_today
from rl_manager.executor_factory import make_stage25_executor_factory


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")


def _plan(**crop_updates: int) -> DailyPlan:
    crops = {crop: 0 for crop in CROPS}
    crops.update(crop_updates)
    return DailyPlan.create(
        crop_targets=crops,
        animal_targets={animal: 0 for animal in ANIMALS},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={
            product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for product in PRODUCTS
        },
    )


def _plant(
    crop: str,
    *,
    planted_day: int,
    yield_units: int = 0,
    watered_today: bool = True,
    fertilized_until_day: int = -1,
) -> dict:
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": watered_today,
        "consecutive_unwatered": 0,
        "fertilized_until_day": fertilized_until_day,
        "max_lifespan_step": -1,
    }


def _obs(day: int, hour: int, tile: dict, *, seeds=()) -> dict:
    board = [["LOCKED"] * 10 for _ in range(10)]
    board[0][0] = tile
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "hires_today": 0,
        "money": 3000.0,
        "tiles": board,
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "player": 0,
        "farms": [farm, {**farm, "tiles": [row[:] for row in board]}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {},
            "seeds": {crop: 10 for crop in seeds},
            "inventories": [{}],
        },
    }


def _kinds(result, chain_kind: str) -> list[str]:
    chain = next(chain for chain in result.chains if chain.kind == chain_kind)
    by_id = {item.id: item for item in result.items}
    return [by_id[item_id].kind for item_id in chain.item_ids]


def test_immature_wheat_default_flags_wait_and_report_negative_unresolved():
    result = build_strip_work_plan(
        _obs(3, 0, _plant("WHEAT", planted_day=3, yield_units=1)),
        _plan(),
    )
    assert not any(item.kind == "DIG" for item in result.items)
    assert not any(item.kind == "HARVEST" for item in result.items)
    assert result.diagnostics.unresolved_crop_delta_dict["WHEAT"] == -1


def test_mature_wheat_contraction_harvests_without_unnecessary_dig():
    result = build_strip_work_plan(
        _obs(3, 0, _plant("WHEAT", planted_day=0, yield_units=3)),
        _plan(),
    )
    assert [(item.kind, item.source) for item in result.items
            if item.kind in ("HARVEST", "DIG")] == [
                ("HARVEST", "crop_reduction")
            ]


def test_day29_wheat_threshold_is_two_but_not_one():
    eligible = build_strip_work_plan(
        _obs(29, 0, _plant("WHEAT", planted_day=0, yield_units=2)),
        _plan(),
    )
    ineligible = build_strip_work_plan(
        _obs(29, 0, _plant("WHEAT", planted_day=0, yield_units=1)),
        _plan(),
    )
    assert any(item.kind == "HARVEST" for item in eligible.items)
    assert not any(item.kind == "HARVEST" for item in ineligible.items)
    assert ineligible.diagnostics.unresolved_crop_delta_dict["WHEAT"] == -1


def test_spent_strawberry_retires_and_replants_in_order():
    result = build_strip_work_plan(
        _obs(16, 0, _plant("STRAWBERRY", planted_day=0, yield_units=1),
             seeds=("STRAWBERRY",)),
        _plan(STRAWBERRY=1),
    )
    assert _kinds(result, "CROP_GROWTH") == [
        "HARVEST", "DIG", "PLANT", "WATER"
    ]


def test_young_one_shot_can_be_sacrificed_only_when_enabled():
    obs = _obs(3, 0, _plant("WHEAT", planted_day=3))
    disabled = build_strip_work_plan(obs, _plan())
    enabled = build_strip_work_plan(
        obs,
        _plan(),
        allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=False,
    )
    assert not any(item.kind == "DIG" for item in disabled.items)
    assert any(item.kind == "DIG" and item.source == "crop_reduction"
               for item in enabled.items)


def test_older_crop_toggle_controls_sacrifice_candidate():
    obs = _obs(3, 0, _plant("WHEAT", planted_day=1))
    off = build_strip_work_plan(
        obs, _plan(), allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=False)
    on = build_strip_work_plan(
        obs, _plan(), allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=True)
    assert not any(item.kind == "DIG" for item in off.items)
    assert any(item.kind == "DIG" for item in on.items)


@pytest.mark.parametrize("crop,planted_day", [("STRAWBERRY", 0), ("TOMATO", 0)])
def test_productive_recurring_toggle_controls_sacrifice(
    crop: str, planted_day: int,
):
    age = 10 if crop == "STRAWBERRY" else 8
    obs = _obs(age, 0, _plant(crop, planted_day=planted_day, yield_units=1))
    off = build_strip_work_plan(
        obs, _plan(), allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=True,
        allow_productive_recurring_crop_sacrifice=False)
    on = build_strip_work_plan(
        obs, _plan(), allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=True,
        allow_productive_recurring_crop_sacrifice=True)
    assert not any(item.kind == "DIG" for item in off.items)
    assert any(item.kind == "DIG" for item in on.items)


def test_active_fertilizer_is_never_generic_sacrifice_candidate():
    result = build_strip_work_plan(
        _obs(3, 0, _plant("WHEAT", planted_day=3, fertilized_until_day=3)),
        _plan(),
        allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=True,
    )
    assert not any(item.kind == "DIG" for item in result.items)
    assert result.diagnostics.unresolved_crop_delta_dict["WHEAT"] == -1


def test_animal_slot_uses_the_same_default_crop_sacrifice_gate():
    board = [["LOCKED"] * 10 for _ in range(10)]
    board[0][0] = _plant("WHEAT", planted_day=3)
    default = plan_animal_layout(
        board, unlocked_quadrants=("NW",), animals_needed={"COW": 1},
        anchor=(0, 0), current_day=3, current_step=72)
    enabled = plan_animal_layout(
        board, unlocked_quadrants=("NW",), animals_needed={"COW": 1},
        anchor=(0, 0), current_day=3, current_step=72,
        config=SacrificeConfig(allow_live_crop_sacrifice=True))
    assert default.placements == ()
    assert default.unresolved == (("COW", 1),)
    assert enabled.placements[0].source == "crop_sacrifice"


def test_replacement_cutoffs_are_h21_one_shot_and_h20_recurring():
    wheat_h21 = build_strip_work_plan(
        _obs(3, 21, _plant("WHEAT", planted_day=0, yield_units=3),
             seeds=("TOMATO",)),
        _plan(TOMATO=1),
    )
    wheat_h22 = build_strip_work_plan(
        _obs(3, 22, _plant("WHEAT", planted_day=0, yield_units=3),
             seeds=("TOMATO",)),
        _plan(TOMATO=1),
    )
    berry_h20 = build_strip_work_plan(
        _obs(16, 20, _plant("STRAWBERRY", planted_day=0, yield_units=1),
             seeds=("TOMATO",)),
        _plan(TOMATO=1),
    )
    berry_h21 = build_strip_work_plan(
        _obs(16, 21, _plant("STRAWBERRY", planted_day=0, yield_units=1),
             seeds=("TOMATO",)),
        _plan(TOMATO=1),
    )
    assert _kinds(wheat_h21, "CROP_GROWTH") == ["HARVEST", "PLANT", "WATER"]
    assert not any(chain.kind == "CROP_GROWTH" for chain in wheat_h22.chains)
    assert not any(
        item.kind == "PLANT" and item.crop == "TOMATO"
        and item.source == "crop_reconciliation"
        for item in wheat_h22.items
    )
    assert _kinds(berry_h20, "CROP_GROWTH") == [
        "HARVEST", "DIG", "PLANT", "WATER"
    ]
    assert not any(chain.kind == "CROP_GROWTH" for chain in berry_h21.chains)
    assert not any(
        item.kind == "PLANT" and item.crop == "TOMATO"
        and item.source == "crop_reconciliation"
        for item in berry_h21.items
    )


def test_executor_flags_are_false_and_recorded_in_stage25_profile():
    default = make_stage25_executor_factory()
    config = StripExecutorConfig(
        aggressive_sell_all=True,
        allow_live_crop_sacrifice=True,
        allow_productive_recurring_crop_sacrifice=True,
        allow_older_crop_sacrifice=True,
    )
    enabled = make_stage25_executor_factory(config)
    default_config = default.effective_profile["strip_config"]
    enabled_config = enabled.effective_profile["strip_config"]
    assert default_config["allow_live_crop_sacrifice"] is False
    assert default_config["allow_productive_recurring_crop_sacrifice"] is False
    assert default_config["allow_older_crop_sacrifice"] is False
    assert enabled_config["allow_live_crop_sacrifice"] is True
    assert enabled_config["allow_productive_recurring_crop_sacrifice"] is True
    assert enabled_config["allow_older_crop_sacrifice"] is True


# ---------------------------------------------------------------------------
# replaceable_today / day-start planner alignment for future-in-the-day
# one-shot removals.  A crop that becomes harvestable through one ordinary
# WATER must be representable by reconcile_crops/strip_work, not just counted
# by replaceable_today.

def _obs_two(
    day: int, hour: int, first: dict, second: dict, *, seeds=(),
) -> dict:
    board = [["LOCKED"] * 10 for _ in range(10)]
    board[0][0] = first
    board[0][1] = second
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0, "money": 3000.0,
        "tiles": board, "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day, "hour": hour, "step": day * 24 + hour, "player": 0,
        "farms": [farm, {**farm, "tiles": [row[:] for row in board]}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {c: 10 for c in seeds},
                    "inventories": [{}]},
    }


def test_future_water_harvest_replacement_is_represented():
    """Test 1: exact validation reproduction."""
    tile = _plant("WHEAT", planted_day=0, yield_units=2, watered_today=False)
    obs = _obs(3, 0, tile, seeds=("TOMATO",))
    board = canonical_board(obs["farms"][0]["tiles"], 3, 0)
    assert replaceable_today(board, 3)[0] == 1

    result = build_strip_work_plan(obs, _plan(WHEAT=0, TOMATO=1))
    assert _kinds(result, "CROP_GROWTH") == [
        "WATER", "HARVEST", "PLANT", "WATER"
    ]
    assert result.diagnostics.unresolved_crop_delta_dict.get("WHEAT", 0) == 0
    assert result.diagnostics.unresolved_crop_delta_dict.get("TOMATO", 0) == 0


def test_future_water_harvest_pure_contraction_without_replanting():
    """Test 2: harvest-only contraction via preparatory WATER."""
    obs = _obs(3, 0, _plant("WHEAT", planted_day=0, yield_units=2,
                            watered_today=False))
    result = build_strip_work_plan(obs, _plan(WHEAT=0))
    assert _kinds(result, "CROP_REMOVAL") == ["WATER", "HARVEST"]
    assert not any(item.kind == "PLANT" for item in result.items)
    assert result.diagnostics.unresolved_crop_delta_dict.get("WHEAT", 0) == 0


def test_ready_wheat_replacement_has_no_prerequisite_water():
    """Test 3: already-ready crop keeps HARVEST -> PLANT -> WATER."""
    obs = _obs(3, 0, _plant("WHEAT", planted_day=0, yield_units=3,
                            watered_today=False), seeds=("TOMATO",))
    result = build_strip_work_plan(obs, _plan(WHEAT=0, TOMATO=1))
    assert _kinds(result, "CROP_GROWTH") == ["HARVEST", "PLANT", "WATER"]
    waters = [item for item in result.items if item.kind == "WATER"]
    assert [item.id for item in waters] == ["WATER:0,0"]


def test_young_one_shot_that_cannot_become_ready_stays_unresolved():
    """Test 4: no DIG and no fake future HARVEST."""
    obs = _obs(3, 0, _plant("WHEAT", planted_day=3, yield_units=0,
                            watered_today=False))
    result = build_strip_work_plan(obs, _plan(WHEAT=0))
    assert not any(item.kind == "DIG" for item in result.items)
    assert not any(item.kind == "HARVEST" for item in result.items)
    assert result.diagnostics.unresolved_crop_delta_dict.get("WHEAT", 0) == -1


def test_late_hour_water_reachable_replacement_respects_horizon():
    """Test 5: h20 fits WATER->HARVEST->PLANT->WATER; h21 does not."""
    at_h20 = build_strip_work_plan(
        _obs(3, 20, _plant("WHEAT", planted_day=0, yield_units=2,
                           watered_today=False), seeds=("TOMATO",)),
        _plan(WHEAT=0, TOMATO=1))
    assert _kinds(at_h20, "CROP_GROWTH") == ["WATER", "HARVEST", "PLANT", "WATER"]

    at_h21 = build_strip_work_plan(
        _obs(3, 21, _plant("WHEAT", planted_day=0, yield_units=2,
                           watered_today=False), seeds=("TOMATO",)),
        _plan(WHEAT=0, TOMATO=1))
    assert not any(chain.kind == "CROP_GROWTH" for chain in at_h21.chains)
    assert not any(
        item.kind == "PLANT" and item.crop == "TOMATO"
        and item.source == "crop_reconciliation"
        for item in at_h21.items
    )
    # Harvest-only contraction still fits at h21 (WATER -> HARVEST).
    assert _kinds(at_h21, "CROP_REMOVAL") == ["WATER", "HARVEST"]

    at_h23 = build_strip_work_plan(
        _obs(3, 23, _plant("WHEAT", planted_day=0, yield_units=2,
                           watered_today=False), seeds=("TOMATO",)),
        _plan(WHEAT=0, TOMATO=1))
    assert not any(chain.kind == "CROP_GROWTH" for chain in at_h23.chains)
    assert not any(chain.kind == "CROP_REMOVAL" for chain in at_h23.chains)


def test_preparatory_water_is_not_duplicated_by_routine_upkeep():
    """Test 6: one prerequisite WATER plus one replacement WATER."""
    obs = _obs(3, 0, _plant("WHEAT", planted_day=0, yield_units=2,
                            watered_today=False), seeds=("TOMATO",))
    result = build_strip_work_plan(obs, _plan(WHEAT=0, TOMATO=1))
    waters = {item.id: item for item in result.items if item.kind == "WATER"}
    assert set(waters) == {"REMOVAL_WATER:0,0", "WATER:0,0"}
    assert waters["REMOVAL_WATER:0,0"].depends_on == ()
    assert waters["WATER:0,0"].depends_on == ("PLANT:TOMATO:0,0",)
    # No routine upkeep WATER item was added on top of the removal chain.
    assert not any(item.source.startswith("routine") for item in waters.values())


def test_clean_future_removal_is_preferred_over_premature_sacrifice():
    """Test 7: clean lifecycle removal beats destructive sacrifice."""
    clean = _plant("WHEAT", planted_day=0, yield_units=2, watered_today=False)
    # Age 1 is below WHEAT's first yield day, is not in the routine water ages,
    # and is watered today, so it is a premature-sacrifice-only candidate.
    premature = _plant("WHEAT", planted_day=2, yield_units=0, watered_today=True)
    obs = _obs_two(3, 0, clean, premature)
    result = build_strip_work_plan(
        obs, _plan(WHEAT=1),
        allow_live_crop_sacrifice=True,
        allow_older_crop_sacrifice=True,
        allow_productive_recurring_crop_sacrifice=True,
    )
    assert not any(item.kind == "DIG" for item in result.items)
    assert [item.tile for item in result.items if item.kind == "WATER"] == [(0, 0)]
    assert result.diagnostics.unresolved_crop_delta_dict.get("WHEAT", 0) == 0
