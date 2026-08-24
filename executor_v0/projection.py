"""Mechanical requested -> feasible plan projection (issue #1 section 3).

Transparent, strategy-free clipping of a manager `DailyPlan` against the
current canonical own state. The requested plan is never modified; both plans
plus JSON-serializable diagnostics are returned so manager failures stay
distinguishable from logistics failures.

Rules (exactly the V0 minimum):

- land: feasible = max(current unlocked count, requested), bounded 1..4;
  land never decreases.
- animals: existing animals are never removed; feasible per species =
  max(current count, requested); positive buy/build deficits are exposed.
- fertilizer: clipped only by mechanically eligible targets — the current
  actual crop counts by type. Resource availability is not determinable here
  and is deliberately not guessed; shortfalls are logged, never hidden.
- CARE: clipped only by mechanically eligible animals — the current actual
  animal counts by species. Unknown/missing eligibility yields zero feasible
  completion with an explicit shortfall; nothing is fabricated.
- sells: the whole-day projection keeps the requested schedule unchanged
  (clipping it here would silently allocate the same inventory to multiple
  bins). Runtime execution must use `clip_sell` per product/bin against the
  currently available inventory and carry the remaining quantity forward.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER, PRODUCT_ORDER

from .plan import DailyPlan

__all__ = ["ProjectionResult", "project_plan", "clip_sell"]


def _count(mapping: Mapping[str, int], name: str) -> int:
    value = mapping.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"current {name} count must be an integer")
    if value < 0:
        raise ValueError(f"current {name} count must be nonnegative")
    return int(value)


@dataclass(frozen=True)
class ProjectionResult:
    """Requested plan kept verbatim next to its mechanical projection."""

    requested_plan: DailyPlan
    feasible_plan: DailyPlan
    diagnostics: dict[str, Any]


def project_plan(
    requested: DailyPlan,
    *,
    current_land_count: int,
    current_animals: Mapping[str, int],
    current_crops: Mapping[str, int],
) -> ProjectionResult:
    """Project one requested daily plan onto the current own state."""
    if isinstance(current_land_count, bool) \
            or not isinstance(current_land_count, (int, np.integer)) \
            or not 1 <= int(current_land_count) <= 4:
        raise ValueError(
            f"current_land_count must be an integer in [1, 4], got "
            f"{current_land_count!r}")
    for name, mapping in (("current_animals", current_animals),
                          ("current_crops", current_crops)):
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{name} must be a mapping")

    current_land = int(current_land_count)
    feasible_land = max(current_land, requested.land_count)

    animals_diag: dict[str, dict[str, int]] = {}
    animal_targets: dict[str, int] = {}
    for name in ANIMAL_ORDER:
        current = _count(current_animals, name)
        req = requested.animal_targets_dict[name]
        feasible = max(current, req)
        animal_targets[name] = feasible
        animals_diag[name] = {
            "current": current, "requested": req, "feasible": feasible,
            "buy_build_deficit": feasible - current,
        }

    # CARE/FERTILIZE eligibility counts assets the plan itself establishes
    # this day (max of current and requested targets), not just the
    # start-of-day snapshot: clipping CARE against current animals while the
    # same plan buys animals made every request permanently infeasible at
    # hour 0 (issue #7). Requests above the planned asset total still clip.
    fertilizer_diag: dict[str, dict[str, int]] = {}
    fertilizer_targets: dict[str, int] = {}
    for crop in CROP_ORDER:
        current = _count(current_crops, crop)
        planned_total = max(current, requested.crop_targets_dict[crop])
        req = requested.fertilizer_by_crop_dict[crop]
        feasible = min(req, planned_total)
        fertilizer_targets[crop] = feasible
        fertilizer_diag[crop] = {
            "requested": req, "eligible": planned_total,
            "feasible": feasible, "shortfall": req - feasible,
        }

    care_diag: dict[str, dict[str, int]] = {}
    care_targets: dict[str, int] = {}
    for name in ANIMAL_ORDER:
        current = _count(current_animals, name)
        planned_total = max(current, requested.animal_targets_dict[name])
        req = requested.care_by_animal_dict[name]
        feasible = min(req, planned_total)
        care_targets[name] = feasible
        care_diag[name] = {
            "requested": req, "eligible": planned_total,
            "feasible": feasible, "shortfall": req - feasible,
        }

    feasible_plan = DailyPlan.create(
        crop_targets=requested.crop_targets_dict,
        animal_targets=animal_targets,
        land_count=feasible_land,
        fertilizer_by_crop=fertilizer_targets,
        care_by_animal=care_targets,
        sell_quantities={
            product: {int(anchor): row[bin_index]
                      for bin_index, anchor in enumerate((0, 4, 8, 12, 16, 20))}
            for product, row in zip(PRODUCT_ORDER, requested.sell_quantities)
        },
    )

    diagnostics: dict[str, Any] = {
        "land": {
            "current": current_land,
            "requested": requested.land_count,
            "feasible": feasible_land,
        },
        "animals": animals_diag,
        "fertilizer": fertilizer_diag,
        "care": care_diag,
        "sells": {
            "policy": (
                "unclipped_at_projection; runtime must clip per product/bin "
                "with clip_sell and carry the remaining quantity forward"),
            "requested_total_by_product": {
                product: sum(row)
                for product, row in zip(PRODUCT_ORDER,
                                        requested.sell_quantities)
            },
        },
    }
    return ProjectionResult(requested_plan=requested,
                            feasible_plan=feasible_plan,
                            diagnostics=diagnostics)


def clip_sell(product: str, requested_remaining: int,
              available: int) -> tuple[int, int]:
    """Clip one sell intent to currently available inventory.

    Returns ``(executed_now, remaining_after)``. The caller owns the remaining
    ledger; future bins never consume inventory that is not yet available.
    """
    if product not in PRODUCT_ORDER:
        raise ValueError(
            f"unknown sell product {product!r}; expected one of "
            f"{list(PRODUCT_ORDER)}")
    for name, value in (("requested_remaining", requested_remaining),
                        ("available", available)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {int(value)}")
    executed = min(int(requested_remaining), int(available))
    return executed, int(requested_remaining) - executed
