"""Stage 2.5 upkeep experiment; independent of neural output dimensions.

Fertilizer value is a conservative short-horizon estimate under the existing
executor's first-available harvest behavior, not an optimized harvest policy.
"""
from collections.abc import Mapping
from replay_daily.constants import ANIMALS, CROPS


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


def fertilizer_extra_units(tile: Mapping, day: int) -> int:
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
