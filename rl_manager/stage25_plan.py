"""Lower accepted Stage 2.5 actions into the legacy executor transport.

Validation of physical/curriculum support and the persistent crop-ledger
transition belong to the lifecycle provider.  This adapter deliberately only
translates an already accepted nine-class action and its resolved crop goals
into :class:`executor_v0.plan.DailyPlan`.
"""

from __future__ import annotations

from collections.abc import Sequence

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.plan import DailyPlan, SELL_BIN_ANCHORS
from replay_daily.constants import PRODUCTS
from rl_manager.stage25_mechanics import (
    ACTION_ORDER,
    animal_class_to_target,
    initialize_crop_ledger,
    land_class_to_target,
)

STAGE25_PLAN_ADAPTER_VERSION = "stage25_daily_plan_adapter_v1"


def lower_stage25_daily_plan(
    action_classes: Sequence[int],
    crop_goals: Sequence[int],
) -> DailyPlan:
    """Return executor transport for one fully validated Stage 2.5 decision.

    CARE, fertilizer, and sell values are zero-valued transport scaffolding;
    the required Stage 2.5 executor profile supplies those mechanics.
    """
    classes = tuple(action_classes)
    if len(classes) != len(ACTION_ORDER):
        raise ValueError(
            f"Stage 2.5 action must contain {len(ACTION_ORDER)} classes, "
            f"got {len(classes)}"
        )
    goals = initialize_crop_ledger(crop_goals)
    land_target = land_class_to_target(classes[0])
    animal_targets = tuple(
        animal_class_to_target(value) for value in classes[1:4]
    )
    # Decode crop classes as a boundary check without applying their deltas a
    # second time; ``goals`` is already the provider's committed K'.
    from rl_manager.stage25_mechanics import crop_class_to_delta

    for value in classes[4:]:
        crop_class_to_delta(value)

    zeros_by_crop = {crop: 0 for crop in CROP_ORDER}
    zeros_by_animal = {animal: 0 for animal in ANIMAL_ORDER}
    zero_sells = {
        product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
        for product in PRODUCTS
    }
    return DailyPlan.create(
        crop_targets=dict(zip(CROP_ORDER, goals)),
        animal_targets=dict(zip(ANIMAL_ORDER, animal_targets)),
        land_count=land_target,
        fertilizer_by_crop=zeros_by_crop,
        care_by_animal=zeros_by_animal,
        sell_quantities=zero_sells,
    )


__all__ = ["STAGE25_PLAN_ADAPTER_VERSION", "lower_stage25_daily_plan"]
