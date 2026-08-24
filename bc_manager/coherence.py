"""Diagnostic-only plan-coherence metrics (issue #6, Stage 1).

Pure NumPy functions over decoded predictions (or expert targets) plus the
current-state arrays already present in the adapter inputs. These metrics
NEVER clip or rewrite predictions, never enter the loss, and never gate
training — they only expose the economically-impossible-plan failure mode.

Lower-bound implied acquisition cost (deliberately narrow):

- positive crop-count deltas x pinned seed costs;
- positive animal-count deltas x pinned animal purchase costs;
- ordered remaining land-quadrant costs for expansion from the current
  unlocked count to the requested count (canonical LAND_ORDER prefix).

Excluded by definition: routing, maintenance, feed, fertilizer, sells,
hires, and all future costs. The ratio-to-cash rule is deterministic:
0 when cost == 0 (even at cash <= 0); +inf when cost > 0 and cash <= 0;
otherwise cost / cash.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from replay_daily.constants import LAND_PRICES

from .constants import ANIMAL_ORDER, CROP_ORDER, QUADRANT_ORDER
from .economics import ANIMAL_COSTS, SEED_COSTS

__all__ = [
    "current_crop_counts",
    "current_animal_counts",
    "lower_bound_acquisition_cost",
    "land_expansion_cost",
    "cash_ratio",
    "coherence_metrics",
    "plan_coherence_record",
    "aggregate_plan_coherence",
]


def current_crop_counts(board_crop: np.ndarray) -> np.ndarray:
    """Currently planted crops [N, 5] from adapter board_crop ids (1..5)."""
    board_crop = np.asarray(board_crop)
    return np.stack(
        [(board_crop == crop_id).sum(axis=1) for crop_id in
         range(1, len(CROP_ORDER) + 1)], axis=1).astype(np.int64)


def current_animal_counts(board_animal: np.ndarray) -> np.ndarray:
    """Currently placed animals [N, 3] from adapter board_animal ids (1..3)."""
    board_animal = np.asarray(board_animal)
    return np.stack(
        [(board_animal == animal_id).sum(axis=1) for animal_id in
         range(1, len(ANIMAL_ORDER) + 1)], axis=1).astype(np.int64)


def land_expansion_cost(current_unlocked: int, requested_land: int) -> int:
    """Ordered remaining quadrant costs from `current_unlocked` to target.

    Deterministic conservative formula for multiple expansions: buying
    quadrants `LAND_ORDER[u-1 .. t-2]` costs `LAND_PRICES[u-1 .. t-2]`.
    Regression (`t < u`) costs 0 here; it is reported separately as the
    land-regression rate.
    """
    u, t = int(current_unlocked), int(requested_land)
    if not 1 <= u <= len(QUADRANT_ORDER):
        raise ValueError(f"current unlocked count must be in [1, 4], got {u}")
    if not 1 <= t <= len(QUADRANT_ORDER):
        raise ValueError(
            f"requested land count must be in [1, 4], got {t}")
    if t <= u:
        return 0
    return int(sum(LAND_PRICES[i] for i in range(u - 1, t - 1)))


def lower_bound_acquisition_cost(
    crop_current: np.ndarray, crop_target: np.ndarray,
    animal_current: np.ndarray, animal_target: np.ndarray,
    unlocked_count: np.ndarray, land_target: np.ndarray,
) -> np.ndarray:
    """Per-row lower-bound acquisition cost [N] (see module docstring)."""
    crop_current = np.asarray(crop_current, dtype=np.int64)
    crop_target = np.asarray(crop_target, dtype=np.int64)
    animal_current = np.asarray(animal_current, dtype=np.int64)
    animal_target = np.asarray(animal_target, dtype=np.int64)
    unlocked_count = np.asarray(unlocked_count, dtype=np.int64)
    land_target = np.asarray(land_target, dtype=np.int64)

    crop_cost = (
        np.clip(crop_target - crop_current, 0, None)
        * np.asarray(SEED_COSTS, dtype=np.int64)).sum(axis=1)
    animal_cost = (
        np.clip(animal_target - animal_current, 0, None)
        * np.asarray(ANIMAL_COSTS, dtype=np.int64)).sum(axis=1)
    land_cost = np.array(
        [land_expansion_cost(u, t)
         for u, t in zip(unlocked_count, land_target)], dtype=np.int64)
    return crop_cost + animal_cost + land_cost


def cash_ratio(cost: np.ndarray, cash: np.ndarray) -> np.ndarray:
    """Deterministic ratio with stable zero/negative-cash handling."""
    cost = np.asarray(cost, dtype=np.float64)
    cash = np.asarray(cash, dtype=np.float64)
    ratio = np.zeros(cost.shape, dtype=np.float64)
    positive = cost > 0
    has_cash = cash > 0
    ratio[positive & has_cash] = (
        cost[positive & has_cash] / cash[positive & has_cash])
    ratio[positive & ~has_cash] = np.inf  # cost > 0 with cash <= 0
    return ratio


def coherence_metrics(
    crop_target_counts: np.ndarray,
    animal_target_counts: np.ndarray,
    land_target: np.ndarray,
    crop_current: np.ndarray,
    animal_current: np.ndarray,
    unlocked_count: np.ndarray,
    cash: np.ndarray,
) -> Mapping[str, float]:
    """Compact coherence summary for one set of requests vs current state."""
    cost = lower_bound_acquisition_cost(
        crop_current, crop_target_counts,
        animal_current, animal_target_counts,
        unlocked_count, land_target)
    ratio = cash_ratio(cost, cash)
    finite_ratio = ratio[np.isfinite(ratio)]
    return {
        "lower_bound_cost_mean": float(cost.mean()),
        "lower_bound_cost_median": float(np.median(cost)),
        "ratio_median": float(np.median(ratio)),
        "ratio_finite_mean": (
            float(finite_ratio.mean()) if finite_ratio.size else 0.0),
        "ratio_gt_1x_rate": float(np.mean(ratio > 1.0)),
        "ratio_gt_2x_rate": float(np.mean(ratio > 2.0)),
        "land_regression_rate": float(
            np.mean(np.asarray(land_target, dtype=np.int64)
                    < np.asarray(unlocked_count, dtype=np.int64))),
    }


# ------------------------------------------------- closed-loop diagnostics
# Same lower-bound definitions as above, applied to one live manager plan
# against the current encoded state. Diagnostic-only: nothing here clips or
# rewrites the plan. Ratios are stored JSON-safely (finite number or null
# plus explicit flags) so artifacts never contain nonstandard Infinity.


def _json_safe_ratio(cost: float, cash: float) -> dict:
    if cost <= 0:
        return {"ratio": 0.0, "zero_cash": False,
                "over_1x": False, "over_2x": False}
    if cash <= 0:
        return {"ratio": None, "zero_cash": True,
                "over_1x": True, "over_2x": True}
    ratio = cost / cash
    return {"ratio": ratio, "zero_cash": False,
            "over_1x": bool(ratio > 1.0), "over_2x": bool(ratio > 2.0)}


def plan_coherence_record(
    day: int,
    cash: float,
    crop_target_counts,
    animal_target_counts,
    land_target: int,
    crop_current,
    animal_current,
    unlocked_count: int,
) -> dict:
    """One JSON-safe closed-loop coherence record for a single daily plan."""
    cost = int(lower_bound_acquisition_cost(
        np.asarray(crop_current, dtype=np.int64)[None, :],
        np.asarray(crop_target_counts, dtype=np.int64)[None, :],
        np.asarray(animal_current, dtype=np.int64)[None, :],
        np.asarray(animal_target_counts, dtype=np.int64)[None, :],
        np.asarray([int(unlocked_count)], dtype=np.int64),
        np.asarray([int(land_target)], dtype=np.int64),
    )[0])
    return {
        "day": int(day),
        "cash": float(cash),
        "lower_bound_cost": cost,
        **_json_safe_ratio(float(cost), float(cash)),
        "land_regression": bool(int(land_target) < int(unlocked_count)),
        "requested": {
            "crops": [int(v) for v in crop_target_counts],
            "animals": [int(v) for v in animal_target_counts],
            "land_count": int(land_target),
        },
        "current": {
            "crops": [int(v) for v in crop_current],
            "animals": [int(v) for v in animal_current],
            "unlocked_count": int(unlocked_count),
        },
    }


def aggregate_plan_coherence(records) -> dict:
    """Aggregate per-day closed-loop coherence records into game summaries."""
    records = list(records)
    if not records:
        return {"days": 0}
    costs = [float(r["lower_bound_cost"]) for r in records]
    finite_ratios = [r["ratio"] for r in records if r["ratio"] is not None]
    return {
        "days": len(records),
        "lower_bound_cost_mean": float(np.mean(costs)),
        "lower_bound_cost_median": float(np.median(costs)),
        "ratio_finite_mean": (
            float(np.mean(finite_ratios)) if finite_ratios else None),
        "ratio_finite_median": (
            float(np.median(finite_ratios)) if finite_ratios else None),
        "over_1x_rate": float(np.mean([r["over_1x"] for r in records])),
        "over_2x_rate": float(np.mean([r["over_2x"] for r in records])),
        "zero_cash_positive_cost_days": int(
            sum(1 for r in records if r["zero_cash"])),
        "land_regression_rate": float(
            np.mean([r["land_regression"] for r in records])),
    }
