"""Derived lifecycle timing for canonical board tiles.

Rules are transcribed from the pinned 1.32.7 engine source
(commit 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c), functions:
`_new_plant`, `WATER`, `HARVEST`, `_daily_refresh_plants`, `_decay_plants`,
`_new_animal`, `_daily_refresh_animals`.

Raw tile fields are preserved except for the fast-engine ``age`` alias, which
is folded into the canonical ``planted_day``/``placed_day`` field inside
``canonical_tile``'s derived view. Derived fields are added under "derived".
Values that are not deterministically derivable from observation state are null.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from .constants import ANIMALS, CROPS

# Pinned 1.32.7 turns per day; the absolute step is day*STEPS_PER_DAY + hour.
STEPS_PER_DAY = 24

# A one-shot crop released by h21 leaves h22/h23 for PLANT -> WATER.  A
# recurring crop needs one extra retirement action, so HARVEST -> DIG must
# finish by h20 before that same replacement tail.
ONE_SHOT_REPLACEABLE_CUTOFF_HOUR = 21
RECURRING_REPLACEABLE_CUTOFF_HOUR = 20
REPLACEABLE_CROP_ORDER = tuple(CROPS)
WHEAT_HARVEST_THRESHOLD = 3
WHEAT_DAY29_HARVEST_THRESHOLD = 2
FINAL_ACTIONABLE_STEP = 718


def _nonnegative_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{what} must be nonnegative, got {result}")
    return result


def resolve_observation_step(
    obs: Mapping[str, Any], *, step: int | None = None,
) -> int:
    """Return the absolute lifecycle step for one observation.

    An explicit valid ``step`` (argument, else ``obs['step']``) is preserved.
    When it is absent, the pinned engine convention ``day*24 + hour`` is used;
    ``day`` and ``hour`` are required and must be nonnegative integers.
    """
    if step is not None:
        return _nonnegative_int(step, "step")
    if not isinstance(obs, Mapping):
        raise ValueError("obs must be a mapping to resolve a lifecycle step")
    raw = obs.get("step")
    if raw is not None:
        return _nonnegative_int(raw, "obs['step']")
    if "day" not in obs:
        raise ValueError("obs is missing required field 'day'")
    if "hour" not in obs:
        raise ValueError("obs is missing required field 'hour'")
    day = _nonnegative_int(obs["day"], "obs['day']")
    hour = _nonnegative_int(obs["hour"], "obs['hour']")
    return day * STEPS_PER_DAY + hour


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


def plant_age_days(tile: Mapping[str, Any], current_day: int) -> int | None:
    """Return canonical plant age, or ``None`` for an incomplete fixture."""
    planted_day = tile.get("planted_day")
    if planted_day is not None:
        return int(current_day) - int(planted_day)
    derived = tile.get("derived") or {}
    age = derived.get("age_days")
    return None if age is None else int(age)


def wheat_harvest_eligibility(
    tile: Mapping[str, Any], current_day: int, current_step: int,
) -> tuple[bool, str]:
    """Return preferred WHEAT harvest eligibility and an auditable reason.

    Tetsuya's ordinary threshold is three units and becomes two units on day
    29.  Yield one does not inherit that day-29 exception.  Mechanical expiry
    and the final actionable turn remain separate safeguards.
    """
    held = int(tile.get("yield_units", 0) or 0)
    threshold = (
        WHEAT_DAY29_HARVEST_THRESHOLD
        if int(current_day) == 29
        else WHEAT_HARVEST_THRESHOLD
    )
    if held >= threshold:
        return True, "threshold_met"
    if current_step >= FINAL_ACTIONABLE_STEP:
        return True, "terminal_horizon"
    raw_lifespan = tile.get("max_lifespan_step", -1)
    lifespan = int(-1 if raw_lifespan is None else raw_lifespan)
    if lifespan >= 0 and current_step >= lifespan - 1:
        return True, "expiry"
    if int(current_day) == 29:
        return False, "below_day29_threshold"

    age = plant_age_days(tile, current_day)
    if age is None:
        return False, "unknown_growth"
    data = CROPS["WHEAT"]
    room = held < int(data["max_yield"])
    useful_today = (
        room
        and tile.get("watered_today") is not True
        and (int(data["max_yield_day"]) + 1) // 2
        <= age
        <= int(data["max_yield_day"])
    )
    future_productive_day = room and age < int(data["max_yield_day"])
    if useful_today or future_productive_day:
        return False, "future_growth"
    return True, "no_further_growth"


def _yield_after_ordinary_water(
    tile: Mapping[str, Any], current_day: int, age_days: int,
) -> int:
    """Forecast same-day yield after at most one ordinary WATER action."""
    held = int(tile.get("yield_units", 0) or 0)
    crop = tile.get("crop")
    data = CROPS.get(crop)
    if data is None or data["ongoing"] or tile.get("watered_today") is True:
        return held
    window_start = (data["max_yield_day"] + 1) // 2
    if not window_start <= age_days <= data["max_yield_day"]:
        return held
    raw_fertilized_until = tile.get("fertilized_until_day", -1)
    fertilized_until = int(
        -1 if raw_fertilized_until is None else raw_fertilized_until
    )
    bonus = 2 if fertilized_until >= int(current_day) else 1
    return min(int(data["max_yield"]), held + bonus)


def clean_crop_removal_actions(
    tile: Mapping[str, Any],
    current_day: int,
    current_step: int,
    *,
    for_replacement: bool,
) -> tuple[str, ...]:
    """Return the ordered clean lifecycle actions that clear one plant.

    An empty tuple means the crop must remain in place.  Both modes admit the
    same ordinary lifecycle sequence, including one preparatory WATER when the
    crop is not yet harvest-ready but one ordinary water would make it so; a
    crop-to-crop replacement additionally reserves two tail actions for
    PLANT -> WATER, yielding the ordinary h21 one-shot and h20 recurring start
    cutoffs.  ``for_replacement`` therefore only controls that tail horizon,
    never whether the preparatory WATER is represented.  TOMATO retirement is
    mechanics-derived because no useful Tetsuya removal sample exists.

    This is the single authority behind both ``replaceable_today`` (which calls
    it with ``for_replacement=True``) and day-start crop reconciliation, so the
    manager forecast and the planner can never disagree about which ordinary
    same-day sequences clear a tile.
    """
    crop = tile.get("crop")
    data = CROPS.get(crop)
    age = plant_age_days(tile, current_day)
    if (
        tile.get("kind") != "PLANT"
        or data is None
        or age is None
        or age < int(data["first_yield_day"])
    ):
        return ()

    held = int(tile.get("yield_units", 0) or 0)
    if data["ongoing"]:
        final_production_age = int(data["first_yield_day"]) + (
            int(data["max_yield"]) - 1
        ) * int(data["interval"])
        if age < final_production_age:
            return ()
        actions = (("HARVEST",) if held > 0 else ()) + ("DIG",)
    else:
        actions: tuple[str, ...] = ()
        if held > 0:
            if crop == "WHEAT":
                eligible = wheat_harvest_eligibility(
                    tile, current_day, current_step
                )[0]
            else:
                eligible = True
            if eligible:
                actions = ("HARVEST",)

        if not actions:
            # Not harvest-ready yet, but one ordinary WATER may still make it
            # cleanly harvestable later today.  The same forecast serves both
            # pure contraction (no tail) and replacement (PLANT->WATER tail).
            forecast_yield = _yield_after_ordinary_water(tile, current_day, age)
            if forecast_yield > held:
                forecast = dict(
                    tile, yield_units=forecast_yield, watered_today=True
                )
                harvest_step = current_step + 1
                if crop == "WHEAT":
                    eligible = wheat_harvest_eligibility(
                        forecast, current_day, harvest_step
                    )[0]
                else:
                    eligible = forecast_yield > 0
                if eligible:
                    actions = ("WATER", "HARVEST")
        if not actions:
            return ()

    tail_actions = 2 if for_replacement else 0
    final_step = current_step + len(actions) + tail_actions - 1
    day_final_step = current_day * STEPS_PER_DAY + STEPS_PER_DAY - 1
    return actions if final_step <= day_final_step else ()


def _plant_is_replaceable_today(
    tile: Mapping[str, Any], current_day: int, current_step: int,
) -> bool:
    """Whether one canonical PLANT can follow the normal replacement path.

    TOMATO retirement is mechanics-derived (there is no useful Tetsuya
    removal sample): like STRAWBERRY, it becomes cleanly retireable only once
    the last scheduled production is available to harvest, after which DIG is
    the normal retirement action.
    """
    return bool(clean_crop_removal_actions(
        tile, current_day, current_step, for_replacement=True
    ))


def replaceable_today(
    board: Sequence[Sequence[Any]], current_day: int, current_step: int | None = None,
) -> tuple[int, ...]:
    """Count clean same-day replacement capacity in canonical crop order.

    This is the day-start forecast used by manager decisions.  A supplied
    same-day ``current_step`` is also accepted for value-bootstrap transport,
    where the remaining h20/h21 cutoff is applied.  One-shot crops may use
    one ordinary WATER before HARVEST and must release by h21.  Recurring
    crops must already hold their final scheduled production so HARVEST ->
    DIG can retire them by h20.  Worker contention is intentionally not
    simulated.
    """
    day = _nonnegative_int(current_day, "current_day")
    step = day * STEPS_PER_DAY if current_step is None else _nonnegative_int(
        current_step, "current_step")
    if not day * STEPS_PER_DAY <= step < (day + 1) * STEPS_PER_DAY:
        raise ValueError(
            "replaceable_today current_step must be within current_day")
    counts = [0] * len(REPLACEABLE_CROP_ORDER)
    for row in board:
        for tile in row:
            if not isinstance(tile, Mapping) or tile.get("kind") != "PLANT":
                continue
            crop = tile.get("crop")
            if crop in REPLACEABLE_CROP_ORDER and _plant_is_replaceable_today(
                    tile, day, step):
                counts[REPLACEABLE_CROP_ORDER.index(crop)] += 1
    return tuple(counts)


def animal_placed_day(tile: Mapping[str, Any], current_day: int) -> int:
    """Placement day from either observation shape.

    Official 1.32.7 observations carry `placed_day` on animal tiles; the
    fast-engine decoder emits `age` (days since placement) instead. Both
    encode the same fact. A tile carrying both must agree; neither present
    fails loudly rather than being guessed.
    """
    placed = tile.get("placed_day")
    age = tile.get("age")
    if placed is None and age is None:
        raise KeyError("animal tile needs 'placed_day' or 'age'")
    derived = None if age is None else (
        int(current_day) - _nonnegative_int(age, "animal tile age"))
    if placed is None:
        return derived  # type: ignore[return-value]
    placed_day = _nonnegative_int(placed, "animal tile placed_day")
    if derived is not None and placed_day != derived:
        raise ValueError(
            f"animal tile contradicts itself: placed_day={placed_day} but "
            f"age={age} at day={current_day}")
    return placed_day


def derive_animal(tile: dict[str, Any], current_day: int) -> dict[str, Any]:
    """Derived timing for an animal structure tile (COOP/PASTURE holding an animal)."""
    animal_data = ANIMALS[tile["animal"]]
    yield_units = tile.get("yield_units", 0)
    harvestable = yield_units > 0

    if harvestable:
        days_until_next_product: int | None = 0
    elif tile.get("consecutive_unfed", 0) < 1 or tile.get("fed_today") is True:
        next_d = _next_production_day(
            current_day, animal_placed_day(tile, current_day), animal_data["first_yield_day"],
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


def _fold_age_alias(
    tile: dict[str, Any], canonical_key: str, current_day: int,
) -> None:
    """Fold the fast-engine ``age`` alias into the canonical day field.

    Mutates the derived-view copy only. Recognized plant/animal tiles convert
    deterministically; a contradictory ``placed_day``/``planted_day`` raises
    instead of being guessed.
    """
    if "age" not in tile:
        return
    age = _nonnegative_int(tile.pop("age"), f"{canonical_key} age")
    derived = int(current_day) - age
    existing = tile.get(canonical_key)
    if existing is None:
        tile[canonical_key] = derived
        return
    if _nonnegative_int(existing, canonical_key) != derived:
        raise ValueError(
            f"tile contradicts itself: {canonical_key}={existing} but age={age} "
            f"at day={current_day}")


def canonical_tile(tile: Any, current_day: int, current_step: int) -> Any:
    """Return the canonical derived view of one observed tile.

    None / "LOCKED" / "WEED" stay exactly as observed so empty unlocked tiles,
    locked quadrants, and weeds remain trivially distinguishable. The
    fast-engine ``age`` alias is folded into ``planted_day``/``placed_day``
    here so downstream strict canonical validation never sees it; other raw
    fields pass through unchanged.
    """
    if not isinstance(tile, dict):
        return tile
    kind = tile.get("kind")
    out = dict(tile)
    if kind == "PLANT":
        _fold_age_alias(out, "planted_day", current_day)
        out["derived"] = derive_plant(out, current_day, current_step)
    elif "animal" in tile:
        _fold_age_alias(out, "placed_day", current_day)
        out["derived"] = derive_animal(out, current_day)
    else:
        out["derived"] = None
    return out


def canonical_board(tiles: list[list[Any]], current_day: int, current_step: int) -> list[list[Any]]:
    return [
        [canonical_tile(t, current_day, current_step) for t in row]
        for row in tiles
    ]
