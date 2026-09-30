"""Submission-safe canonical view of official Kaggriculture observations.

The field mapping mirrors :func:`oracle.closed_loop._executor_observation`
without importing the evaluation package at runtime. Keep the parity test
against that function whenever either implementation changes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from bc_manager.constants import CROP_ORDER, PRODUCT_ORDER

_CROPS = CROP_ORDER
_SHED_ITEMS = PRODUCT_ORDER + ("GOOSE", "COW", "SHEEP")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return {key: _plain(item) for key, item in value.items()}
    except AttributeError:
        return value


def _canonical_tile(tile: Any, day: int) -> Any:
    if not isinstance(tile, dict):
        return tile
    kind = tile.get("kind")
    if kind == "PLANT":
        return {
            "kind": "PLANT",
            "crop": tile["crop"],
            "planted_day": int(tile["planted_day"]),
            "max_lifespan_step": int(tile["max_lifespan_step"]),
            "yield_units": int(tile["yield_units"]),
            "watered_today": bool(tile["watered_today"]),
            "consecutive_unwatered": int(tile["consecutive_unwatered"]),
            "fertilized_until_day": int(tile["fertilized_until_day"]),
        }
    if kind in ("COOP", "PASTURE"):
        canonical: dict[str, Any] = {"kind": kind}
        if "animal" in tile:
            placed_day = tile.get("placed_day")
            if placed_day is None and "age" in tile:
                placed_day = int(day) - int(tile["age"])
            canonical.update({
                "animal": tile["animal"],
                "placed_day": int(placed_day),
                "yield_units": int(tile["yield_units"]),
                "consecutive_unfed": int(tile["consecutive_unfed"]),
                "fed_today": bool(tile["fed_today"]),
                "cared_today": bool(tile["cared_today"]),
                "fertilizer_available": bool(tile["fertilizer_available"]),
                "pending_care_bonus": int(tile["pending_care_bonus"]),
            })
        return canonical
    return {key: _plain(value) for key, value in sorted(tile.items())}


def _canonical_farm(farm: Mapping[str, Any], day: int) -> dict[str, Any]:
    tiles = farm["tiles"]
    return {
        "money": float(farm["money"]),
        "tiles": [[_canonical_tile(tiles[y][x], day)
                   for x in range(len(tiles[y]))]
                  for y in range(len(tiles))],
        "farmer": [int(farm["farmer"][0]), int(farm["farmer"][1])],
        "hands": [[int(hand[0]), int(hand[1])] for hand in farm["hands"]],
        "unlocked_quadrants": [str(name)
                               for name in farm["unlocked_quadrants"]],
        "hires_today": int(farm["hires_today"]),
    }


def executor_observation(
    observation: Mapping[str, Any], *, from_fast: bool
) -> dict[str, Any]:
    """Adapt official or fast-engine aliases to the evaluation executor view."""
    view = copy.deepcopy(dict(observation))
    view.setdefault(
        "step", int(view.get("day", 0)) * 24 + int(view.get("hour", 0))
    )
    day = int(view["day"])
    farms = []
    for farm in view.get("farms", []):
        if not from_fast:
            farms.append(_canonical_farm(farm, day))
            continue
        # The fast engine's age aliases are converted by the oracle helper.
        fast_farm = copy.deepcopy(farm)
        for row in fast_farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in (
                        "COOP", "PASTURE") and tile.get("placed_day") is None \
                        and "age" in tile:
                    tile["placed_day"] = day - int(tile["age"])
        farms.append(_canonical_farm(fast_farm, day))
    view["farms"] = farms
    private = view.get("private")
    if isinstance(private, dict):
        private["shed"] = {
            name: int(private.get("shed", {}).get(name, 0))
            for name in _SHED_ITEMS
        }
        private["seeds"] = {
            name: int(private.get("seeds", {}).get(name, 0))
            for name in _CROPS
        }
        private["inventories"] = [
            {name: int(inventory.get(name, 0)) for name in _SHED_ITEMS}
            for inventory in private.get("inventories", [])
        ]
    return view


def canonicalize_official_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize an ordinary competition observation for submission use."""
    return executor_observation(observation, from_fast=False)
