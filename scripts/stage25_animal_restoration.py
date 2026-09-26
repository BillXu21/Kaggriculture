"""Deterministic Stage 2.5 escaped-sheep restoration smoke scenario."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bc_manager.constants import PRODUCTS
from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController


def _plan() -> DailyPlan:
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 1},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        care_by_animal={animal: 0 for animal in ("GOOSE", "COW", "SHEEP")},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )


def _observation(*, day: int, step: int, money: float, shed=None, inventory=None, sheep=False):
    farm = {
        "money": money,
        "unlocked_quadrants": ["NW"],
        "farmer": [4, 4],
        "hands": [],
        "tiles": [[None] * 10 for _ in range(10)],
    }
    farm["tiles"][4][4] = {
        "kind": "PASTURE",
        **(
            {
                "animal": "SHEEP",
                "placed_day": 1,
                "yield_units": 0,
                "fed_today": False,
                "cared_today": False,
                "consecutive_unfed": 0,
            }
            if sheep
            else {}
        ),
    }
    return {
        "day": day,
        "hour": 0,
        "step": step,
        "farms": [farm, deepcopy(farm)],
        "private": {
            "shed": shed or {},
            "seeds": {crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
            "inventories": [inventory or {}, {}],
        },
        "market": {
            "inventory": {product: 10000 for product in PRODUCTS},
            "prices": {product: 25 for product in PRODUCTS},
        },
        "configuration": {"shedCapacity": 100},
    }


def run_scenario() -> dict[str, object]:
    controller = StripExecutorController()
    plan = _plan()
    controller.act(_observation(day=1, step=0, money=1000, sheep=True), plan)
    buy = controller.act(_observation(day=2, step=24, money=1000), plan)
    assert buy.market_actions == (("BUY_ANIMAL", "SHEEP", 1),)
    pickup = controller.act(
        _observation(day=2, step=25, money=500, shed={"SHEEP": 1}), plan
    )
    assert pickup.farmer_action == ("PICKUP", "SHEEP", 1)
    place = controller.act(
        _observation(day=2, step=26, money=500, inventory={"SHEEP": 1}), plan
    )
    assert place.farmer_action == ("PLACE", "SHEEP", 1)
    return {
        "buy": list(buy.market_actions[0]),
        "pickup": list(pickup.farmer_action),
        "place": list(place.farmer_action),
        "restored_physical_sheep": 1,
    }


if __name__ == "__main__":
    print(json.dumps(run_scenario(), sort_keys=True))
