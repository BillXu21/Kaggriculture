"""Verified 1.32.7 Kaggriculture constants (upstream commit 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c).

Sources:
- kaggle_environments/envs/kaggriculture/kaggriculture.py at the pinned commit;
- MECHANICS.md regression tables;
- actual downloaded replays (data/samples).
"""

ENGINE_VERSION = "1.32.7"
# v2: canonical `events.care` by-animal ledger + entries, derived
# `targets.care_by_animal`, and fail-loud processed-data version checks.
# v1 processed artifacts are rejected, never migrated; regenerate from raw.
SCHEMA_VERSION = 2

# Engine CROPS table (exact copy of upstream constants).
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

# Engine ANIMALS table (exact copy of upstream constants).
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
]

# Species eligible for CARE (zero-defaulted in the canonical care ledger).
CARE_SPECIES = ("GOOSE", "COW", "SHEEP")

# NW always unlocked; players unlock the rest in this order (upstream LAND_ORDER/LAND_PRICES).
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

FARM_HAND_COST_MULT_DEFAULT = 1


def fib(n: int) -> int:
    """Engine `_fib`: fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(n_already_today: int, mult: int = FARM_HAND_COST_MULT_DEFAULT) -> int:
    """Engine `_hire_cost`: cost of the next hire given hires already made today."""
    return mult * fib(n_already_today)


def total_hire_cost(count: int, mult: int = FARM_HAND_COST_MULT_DEFAULT) -> int:
    """Exact total cost of `count` successful hires on one day (sum of fib(0..count-1))."""
    return mult * sum(fib(i) for i in range(count))


SELL_BIN_ANCHORS = [0, 4, 8, 12, 16, 20]


def sell_bin(hour: int) -> int:
    """Six intraday windows anchored at 0,4,8,12,16,20 via floor(hour/4)*4."""
    return (hour // 4) * 4
