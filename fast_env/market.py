"""Exact 1.32.7 market helpers, kept independent of the Rust extension."""

from __future__ import annotations

import math
from typing import Any, Mapping

HINGE_GAIN = 8.0
PRICE_FLOOR = 1
MARKET_I0 = 10_000

PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT": {"base": 25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": MARKET_I0, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": MARKET_I0, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": MARKET_I0, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape(function: str, x: float, target: float | None = None) -> float:
    x = max(0.0, float(x))
    if function == "linear":
        return x
    if function == "sq":
        return x * x
    if function == "sqrt":
        return math.sqrt(x)
    if function == "log":
        return math.log(1.0 + x)
    if function == "log10":
        return math.log10(1.0 + x)
    if function == "hinge":
        if not target or target <= 0:
            return x
        u = x / target
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(
    item: str,
    inventory: float,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Return the official 1.32.7 price using Python bankers rounding."""
    resolved = {name: dict(value) for name, value in MARKET_PARAMS.items()}
    if params:
        for name, patch in params.items():
            if name in resolved:
                resolved[name].update(patch)
    product = resolved[item]
    base = float(product["base"])
    i0 = float(product["I0"])
    target = float(product["T"])
    if inventory < i0:
        function = str(product["below_func"])
        amount = _shape(function, target, target)
        price = base + float(product["below_target"]) * base / amount * _shape(function, i0 - inventory, target)
    else:
        function = str(product["above_func"])
        amount = _shape(function, target, target)
        price = base - float(product["above_target"]) * base / amount * _shape(function, inventory - i0, target)
    return max(PRICE_FLOOR, int(round(price)))
