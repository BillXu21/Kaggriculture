"""Expert daily manager intent extraction from raw Kaggriculture replays.

Derives one ``executor_v0.plan.DailyPlan`` for a (seat, day) pair using
ONLY mechanically-attributed events (canonical intent extraction, not
primitive imitation):

- Action alignment (verified): ``steps[i][seat]["action"]`` transformed
  ``steps[i-1][seat]["observation"]`` into ``steps[i][seat]["observation"]``;
  every event is attributed to the day/hour of observation ``i-1``.
- Worker positions are official ``[x, y]``; board rows are ``tiles[y][x]``.
- CARE/FERTILIZE/HARVEST identity comes exclusively from the tile under the
  acting worker in the PRE observation; farmer uses ``farm["farmer"]`` and
  hands use ``farm["hands"]`` zipped in order with their ops.

Deterministic and pure; no caching.
"""

from __future__ import annotations

from typing import Any

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.plan import DailyPlan
from replay_daily.constants import ANIMALS, PRODUCTS, sell_bin

__all__ = [
    "ANIMAL_SPECIES",
    "SELL_BIN_ANCHORS",
    "board_counts",
    "boundary_observation",
    "collect_day_events",
    "end_of_day_observation",
    "extract_daily_plan",
]

ANIMAL_SPECIES = ("GOOSE", "COW", "SHEEP")  # == ANIMAL_ORDER
SELL_BIN_ANCHORS = (0, 4, 8, 12, 16, 20)
_LAND_MIN, _LAND_MAX = 1, 4


# ------------------------------------------------------------------ tiles


def _tile_at(tiles: list[list[Any]], pos: Any) -> dict[str, Any] | None:
    """Dict tile under worker position ``[x, y]`` (rows are tiles[y][x])."""
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
        return None
    tile = tiles[y][x]
    return tile if isinstance(tile, dict) else None


def _tile_animal(tiles: list[list[Any]], pos: Any) -> str | None:
    tile = _tile_at(tiles, pos)
    if tile is None:
        return None
    animal = tile.get("animal")
    return animal if animal in ANIMAL_SPECIES else None


def _tile_crop(tiles: list[list[Any]], pos: Any) -> str | None:
    tile = _tile_at(tiles, pos)
    if tile is None or tile.get("kind") != "PLANT":
        return None
    return tile.get("crop")


def _harvest_item(tiles: list[list[Any]], pos: Any) -> str | None:
    tile = _tile_at(tiles, pos)
    if tile is None:
        return None
    if tile.get("kind") == "PLANT":
        return tile.get("crop")
    animal = tile.get("animal")
    if animal in ANIMALS:
        return ANIMALS[animal]["product"]
    return None


# ------------------------------------------------------- event collection


def collect_day_events(
    replay: dict[str, Any], seat: int, day: int
) -> dict[str, Any]:
    """Mechanically-attributed events for one (seat, day).

    Iterates i in 1..len(steps)-1; ``steps[i][seat]["action"]`` is attributed
    to the day/hour of ``steps[i-1][seat]["observation"]``; only events whose
    pre-observation day equals ``day`` are collected. ``steps[0]`` holds
    default no-op actions and never yields events.
    """
    steps = replay["steps"]
    sells: list[tuple[Any, str, int]] = []
    care_counts: dict[str, int] = {}
    fert_counts: dict[str, int] = {}
    harvest_counts: dict[str, int] = {}
    hires_submitted = 0
    land_buys = 0

    for i in range(1, len(steps)):
        pre_obs = steps[i - 1][seat].get("observation") or {}
        if pre_obs.get("day") != day:
            continue
        action = steps[i][seat].get("action")
        if not action:
            continue
        hour = pre_obs.get("hour")
        farm = pre_obs["farms"][seat]
        tiles = farm["tiles"]

        positions: list[Any] = [farm["farmer"]]
        positions.extend(farm.get("hands") or [])
        worker_ops: list[Any] = [action.get("farmer")]
        worker_ops.extend(action.get("hands") or [])

        for pos, op in zip(positions, worker_ops):
            if not isinstance(op, list) or not op:
                continue
            name = op[0]
            if name == "CARE":
                animal = _tile_animal(tiles, pos)
                if animal is not None:
                    care_counts[animal] = care_counts.get(animal, 0) + 1
            elif name == "FERTILIZE":
                crop = _tile_crop(tiles, pos)
                if crop is not None:
                    fert_counts[crop] = fert_counts.get(crop, 0) + 1
            elif name == "HARVEST":
                item = _harvest_item(tiles, pos)
                if item is not None:
                    harvest_counts[item] = harvest_counts.get(item, 0) + 1

        for order in action.get("market") or []:
            if not isinstance(order, list) or not order:
                continue
            op = order[0]
            if op == "SELL" and len(order) > 2:
                sells.append((hour, order[1], order[2]))
            elif op == "HIRE":
                hires_submitted += 1
            elif op == "BUY_LAND":
                land_buys += 1

    return {
        "sells": sells,
        "care_counts": care_counts,
        "fert_counts": fert_counts,
        "harvest_counts": harvest_counts,
        "hires_submitted": hires_submitted,
        "land_buys": land_buys,
    }


# -------------------------------------------------------- day boundaries


def boundary_observation(
    replay: dict[str, Any], seat: int, day: int
) -> dict[str, Any]:
    """First observation at start-of-day state (day == day, hour == 0)."""
    for step in replay["steps"]:
        obs = step[seat].get("observation") or {}
        if obs.get("day") == day and obs.get("hour") == 0:
            return obs
    raise ValueError(
        f"no start-of-day observation for seat {seat}, day {day}"
    )


def end_of_day_observation(
    replay: dict[str, Any], seat: int, day: int
) -> dict[str, Any]:
    """First observation with (day+1, hour==0), else the terminal state."""
    target = day + 1
    for step in replay["steps"]:
        obs = step[seat].get("observation") or {}
        if obs.get("day") == target and obs.get("hour") == 0:
            return obs
    return replay["steps"][-1][seat]["observation"]


# ------------------------------------------------------------ board scan


def board_counts(
    obs: dict[str, Any], seat: int = 0
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """(crops_by_name, animals_by_name, weeds, empty_structures) of a board.

    Scans ``obs["farms"][seat]["tiles"]`` (rows are tiles[y][x]): PLANT dicts
    count by crop; dicts with an "animal" key count by species; weeds are
    tiles that are the string "WEED" OR dicts {"kind": "WEED"}; empty
    structures are COOP/PASTURE dicts with no "animal" key.
    """
    tiles = obs["farms"][seat]["tiles"]
    crops: dict[str, int] = {name: 0 for name in CROP_ORDER}
    animals: dict[str, int] = {name: 0 for name in ANIMAL_ORDER}
    weeds = 0
    empty_structures = 0
    for row in tiles:
        for tile in row:
            if isinstance(tile, str):
                if tile == "WEED":
                    weeds += 1
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                weeds += 1
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop is not None:
                    crops[crop] = crops.get(crop, 0) + 1
            elif "animal" in tile:
                species = tile["animal"]
                animals[species] = animals.get(species, 0) + 1
            elif kind in ("COOP", "PASTURE"):
                empty_structures += 1
    return crops, animals, weeds, empty_structures


# ----------------------------------------------------------- plan builder


def extract_daily_plan(
    replay: dict[str, Any], seat: int, day: int
) -> DailyPlan:
    """Expert daily manager intent for one (seat, day) as a DailyPlan."""
    events = collect_day_events(replay, seat, day)
    end_obs = end_of_day_observation(replay, seat, day)

    crops_by_name, animals_by_name, _, _ = board_counts(end_obs, seat)
    unlocked = end_obs["farms"][seat].get("unlocked_quadrants") or []
    land_count = max(_LAND_MIN, min(_LAND_MAX, len(unlocked)))

    # Fertilizer applications capped per crop at that crop's END-of-day plant
    # count; CARE capped per species at its END-of-day animal count.
    fertilizer_by_crop = {
        crop: min(events["fert_counts"].get(crop, 0),
                  crops_by_name.get(crop, 0))
        for crop in CROP_ORDER
    }
    care_by_animal = {
        animal: min(events["care_counts"].get(animal, 0),
                    animals_by_name.get(animal, 0))
        for animal in ANIMAL_ORDER
    }

    sell_quantities: dict[str, dict[int, int]] = {
        product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
        for product in PRODUCTS
    }
    for hour, product, quantity in events["sells"]:
        if product in sell_quantities:
            sell_quantities[product][sell_bin(int(hour))] += int(quantity)

    return DailyPlan.create(
        crop_targets=crops_by_name,
        animal_targets=animals_by_name,
        land_count=land_count,
        fertilizer_by_crop=fertilizer_by_crop,
        care_by_animal=care_by_animal,
        sell_quantities=sell_quantities,
    )
