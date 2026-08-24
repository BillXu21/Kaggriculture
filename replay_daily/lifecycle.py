"""Derived lifecycle timing for canonical board tiles.

Rules are transcribed from the pinned 1.32.7 engine source
(commit 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c), functions:
`_new_plant`, `WATER`, `HARVEST`, `_daily_refresh_plants`, `_decay_plants`,
`_new_animal`, `_daily_refresh_animals`.

Raw tile fields are never discarded; derived fields are added under "derived".
Values that are not deterministically derivable from observation state are null.
"""

from typing import Any

from .constants import ANIMALS, CROPS


def _next_production_day(
    current_day: int, origin_day: int, first_yield_day: int, interval: int,
    max_production_count: int | None,
) -> int | None:
    """First day d > current_day whose morning refresh produces, or None.

    Engine refresh into day d produces when
    dsf = d - origin_day - first_yield_day >= 0 and dsf % interval == 0.
    For crops the production count dsf // interval + 1 must not exceed max_yield.
    """
    if interval <= 0:
        return None
    # Candidate must be beyond today and at least the first production day.
    start = max(current_day + 1, origin_day + first_yield_day)
    base = start - origin_day - first_yield_day  # >= 0
    remainder = base % interval
    d = start + ((interval - remainder) % interval)
    if max_production_count is not None:
        count = (d - origin_day - first_yield_day) // interval + 1
        if count > max_production_count:
            return None
    return d


def derive_plant(tile: dict[str, Any], current_day: int, current_step: int) -> dict[str, Any]:
    """Derived timing for a PLANT tile (raw fields preserved by caller)."""
    crop_data = CROPS[tile["crop"]]
    age_days = current_day - tile["planted_day"]
    yield_units = tile.get("yield_units", 0)

    harvestable = yield_units > 0 and age_days >= crop_data["first_yield_day"]

    days_until_next_harvest: int | None
    if harvestable:
        days_until_next_harvest = 0
    elif yield_units > 0:
        # Has accumulated units but still too young to legally harvest.
        days_until_next_harvest = crop_data["first_yield_day"] - age_days
    elif crop_data["ongoing"] and tile.get("watered_today") is True:
        next_d = _next_production_day(
            current_day, tile["planted_day"], crop_data["first_yield_day"],
            crop_data["interval"], crop_data["max_yield"],
        )
        days_until_next_harvest = None if next_d is None else next_d - current_day
    elif crop_data["ongoing"]:
        # Production at the next morning is not determined until the current
        # day has been watered; two consecutive dry days turn the crop into a
        # WEED before it can produce.
        days_until_next_harvest = None
    else:
        # Non-ongoing crop with no units: future yield depends on watering actions
        # inside its growth window, which observations do not determine. Null by design.
        days_until_next_harvest = None

    mls = tile.get("max_lifespan_step", -1)
    return {
        "age_days": age_days,
        "currently_harvestable": harvestable,
        "days_until_next_harvest": days_until_next_harvest,
        "fertilizer_active": tile.get("fertilized_until_day", -1) >= current_day,
        "past_lifespan": mls >= 0 and current_step >= mls,
    }


def _animal_placed_day(tile: dict[str, Any], current_day: int) -> int:
    """Placement day from either observation shape.

    Official 1.32.7 observations carry `placed_day` on animal tiles; the
    fast-engine decoder emits `age` (days since placement) instead. Both
    encode the same fact; accept either and fail loudly when neither is
    present.
    """
    placed = tile.get("placed_day")
    if placed is not None:
        return int(placed)
    age = tile.get("age")
    if age is not None:
        return current_day - int(age)
    raise KeyError("animal tile needs 'placed_day' or 'age'")


def derive_animal(tile: dict[str, Any], current_day: int) -> dict[str, Any]:
    """Derived timing for an animal structure tile (COOP/PASTURE holding an animal)."""
    animal_data = ANIMALS[tile["animal"]]
    yield_units = tile.get("yield_units", 0)
    harvestable = yield_units > 0

    if harvestable:
        days_until_next_product: int | None = 0
    elif tile.get("consecutive_unfed", 0) < 1 or tile.get("fed_today") is True:
        next_d = _next_production_day(
            current_day, _animal_placed_day(tile, current_day), animal_data["first_yield_day"],
            animal_data["interval"], None,
        )
        days_until_next_product = None if next_d is None else next_d - current_day
    else:
        # A second consecutive unfed day causes escape at the next refresh;
        # no future product is then mechanically guaranteed.
        days_until_next_product = None

    return {
        "currently_harvestable": harvestable,
        "days_until_next_product": days_until_next_product,
        "starving": tile.get("consecutive_unfed", 0) >= 1,
    }


def canonical_tile(tile: Any, current_day: int, current_step: int) -> Any:
    """Return the raw tile with derived timing attached; raw values pass through.

    None / "LOCKED" / "WEED" stay exactly as observed so empty unlocked tiles,
    locked quadrants, and weeds remain trivially distinguishable.
    """
    if not isinstance(tile, dict):
        return tile
    kind = tile.get("kind")
    out = dict(tile)
    if kind == "PLANT":
        out["derived"] = derive_plant(tile, current_day, current_step)
    elif "animal" in tile:
        out["derived"] = derive_animal(tile, current_day)
    else:
        out["derived"] = None
    return out


def canonical_board(tiles: list[list[Any]], current_day: int, current_step: int) -> list[list[Any]]:
    return [
        [canonical_tile(t, current_day, current_step) for t in row]
        for row in tiles
    ]
