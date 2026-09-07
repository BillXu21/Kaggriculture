"""Stage 2.5 upkeep experiments; independent of neural output dimensions.

Fertilizer value is a conservative short-horizon estimate under the existing
executor's first-available harvest behavior, not an optimized harvest policy.
"""
from collections.abc import Mapping
from replay_daily.constants import ANIMALS, CROPS

WHEAT_HARVEST_THRESHOLD = 3
FINAL_ACTIONABLE_STEP = 718


def wheat_harvest_eligibility(
    tile: Mapping, day: int, step: int,
) -> tuple[bool, str]:
    """Return threshold eligibility and its auditable reason for one wheat.

    Productive watering is possible on an unwatered day in wheat's exact
    single-harvest growth window. Expiry is read from the engine-provided
    lifespan step rather than inferred from age. The step immediately before
    expiry is the last action that can avoid the day-boundary decay.
    """
    held = int(tile.get("yield_units", 0) or 0)
    if held >= WHEAT_HARVEST_THRESHOLD:
        return True, "threshold_met"
    if step >= FINAL_ACTIONABLE_STEP:
        return True, "terminal_horizon"
    lifespan = int(tile.get("max_lifespan_step", -1) or -1)
    if lifespan >= 0 and step >= lifespan - 1:
        return True, "expiry"

    data = CROPS["WHEAT"]
    age = day - int(tile["planted_day"])
    room = held < data["max_yield"]
    useful_today = (
        room and tile.get("watered_today") is not True
        and (data["max_yield_day"] + 1) // 2 <= age <= data["max_yield_day"]
    )
    future_productive_day = room and age < data["max_yield_day"] and day < 29
    if useful_today or future_productive_day:
        return False, "future_growth"
    return True, "no_further_growth"


def care_has_payoff(tile: Mapping, day: int) -> bool:
    """Care banked tonight must reach a later production before day 30."""
    if not tile.get("fed_today") or tile.get("cared_today"):
        return False
    data = ANIMALS[tile["animal"]]
    first = int(tile["placed_day"]) + data["first_yield_day"]
    earliest = max(first, day + 2)  # production precedes banking tonight
    production = first + max(0, (earliest - first + data["interval"] - 1)
                             // data["interval"]) * data["interval"]
    return production < 30


def fertilizer_extra_units(
    tile: Mapping, day: int, *, wheat_harvest_threshold: bool = False,
) -> int:
    """Compare three-day yields with/without one application, fixed upkeep.

Do not fertilize a mature one-shot crop after watering: it is about to be
harvested. Ongoing crops are assumed collected between production events;
existing held yield limits the first event. Missing future water/collection
can reduce this estimate and is measured by the eventual game comparison.
    """
    data = CROPS[tile["crop"]]
    expiry = int(tile.get("fertilized_until_day", -1))
    if expiry >= day or day >= 29:
        return 0
    age = day - int(tile["planted_day"])
    held = int(tile.get("yield_units", 0))
    cap = data["max_yield"]
    if data["ongoing"]:
        extra = 0
        for offset in range(3):
            production_day = day + offset + 1
            a = age + offset + 1
            delta = a - data["first_yield_day"]
            if production_day >= 30 or delta < 0 or delta % data["interval"]:
                continue
            if delta // data["interval"] >= data["max_yield"]:
                continue
            extra += min(cap, held + 2) - min(cap, held + 1)
            held = 0
        return extra
    if wheat_harvest_threshold and tile["crop"] == "WHEAT":
        # Ideal-upkeep forecast. Actual collection may be later when scheduler
        # capacity delays an otherwise eligible harvest.
        def forecast(treated: bool) -> int:
            forecast_yield = held
            for offset in range(data["max_yield_day"] - age + 1):
                forecast_day = day + offset
                if forecast_day >= 30:
                    break
                forecast_age = age + offset
                already_watered = offset == 0 and bool(tile.get("watered_today"))
                if not already_watered and (data["max_yield_day"] + 1) // 2 \
                        <= forecast_age <= data["max_yield_day"]:
                    gain = 2 if treated and offset <= 2 else 1
                    forecast_yield = min(cap, forecast_yield + gain)
                forecast_tile = dict(tile, yield_units=forecast_yield,
                                     watered_today=True)
                forecast_step = min(FINAL_ACTIONABLE_STEP,
                                    forecast_day * 24)
                eligible, _ = wheat_harvest_eligibility(
                    forecast_tile, forecast_day, forecast_step)
                if eligible:
                    break
            return forecast_yield

        return forecast(True) - forecast(False)
    if age >= data["first_yield_day"] and tile.get("watered_today"):
        return 0
    baseline = treated = held
    start = int(bool(tile.get("watered_today")))
    for offset in range(start, 3):
        a = age + offset
        if day + offset >= 30:
            break
        if (data["max_yield_day"] + 1) // 2 <= a <= data["max_yield_day"]:
            baseline = min(cap, baseline + 1)
            treated = min(cap, treated + 2)
        if a >= data["first_yield_day"]:
            break  # existing executor harvests at its first opportunity
    return treated - baseline
