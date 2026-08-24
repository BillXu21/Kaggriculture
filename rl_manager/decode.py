"""Deterministic issue-#8 decode: JAX E outputs -> action tensors / DailyPlan.

Mirrors `executor_v0.manager.decode_daily_plan` exactly (argmax counts, land
argmax + 1, sigmoid presence > 0.5, round-half-up expm1 quantity forced to 0
when absent) but in pure NumPy so the RL path never needs torch. Parity with
the torch decoder is enforced by test.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from executor_v0.plan import DailyPlan
from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
)
from replay_daily.constants import PRODUCTS
from executor_v0.plan import SELL_BIN_ANCHORS

_SELL_PRESENCE_THRESHOLD = 0.5

#: Exact encoded action-tensor schema (PPO-ready; one row per manager day).
ACTION_TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "crop": (NUM_CROPS,),
    "animal": (NUM_ANIMALS,),
    "land": (),
    "fertilizer": (NUM_CROPS,),
    "care": (NUM_ANIMALS,),
    "sell_presence": (NUM_PRODUCTS, SELL_BIN_COUNT),
    "sell_quantity": (NUM_PRODUCTS, SELL_BIN_COUNT),
}

#: Stochastic PPO logprob group slots (issue #9 B2 list; sell_quantity stays
#: frozen regression output and therefore has no logprob slot).
LOGPROB_GROUPS = ("crop", "animal", "land", "fertilizer", "care",
                  "sell_presence")


def decode_outputs_to_action_tensors(
    outputs: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Decode raw head outputs [B, ...] into the exact action tensors."""
    crop = np.argmax(np.asarray(outputs["crop_logits"]), axis=-1)
    animal = np.argmax(np.asarray(outputs["animal_logits"]), axis=-1)
    land = np.argmax(np.asarray(outputs["land_logits"]), axis=-1) + 1
    fertilizer = np.argmax(np.asarray(outputs["fertilizer_logits"]), axis=-1)
    care = np.argmax(np.asarray(outputs["care_logits"]), axis=-1)

    presence_logits = np.asarray(outputs["sell_presence_logits"])
    quantity_log1p = np.asarray(outputs["sell_quantity_log1p"])
    presence = 1.0 / (1.0 + np.exp(-presence_logits))
    quantity = np.clip(
        np.expm1(np.clip(quantity_log1p, a_min=0.0, a_max=None)),
        a_min=0.0, a_max=None)
    present = presence > _SELL_PRESENCE_THRESHOLD
    sell_quantity = np.floor(quantity + 0.5).astype(np.int64)
    sell_quantity = np.where(present, sell_quantity, 0)
    sell_presence = present.astype(np.uint8)

    return {
        "crop": crop.astype(np.int16),
        "animal": animal.astype(np.int16),
        "land": land.astype(np.int16),
        "fertilizer": fertilizer.astype(np.int16),
        "care": care.astype(np.int16),
        "sell_presence": sell_presence.astype(np.uint8),
        "sell_quantity": sell_quantity.astype(np.int16),
    }


def plans_from_action_tensors(
    action_tensors: Mapping[str, np.ndarray],
) -> list[DailyPlan]:
    """Rebuild validated `DailyPlan`s from the exact encoded action tensors."""
    tensors = {name: np.asarray(array)
               for name, array in action_tensors.items()}
    plans: list[DailyPlan] = []
    for row in range(tensors["crop"].shape[0]):
        # Presence is already folded into the quantities (absent -> 0).
        quantity_rows = tensors["sell_quantity"][row]
        sells = {
            product: {int(anchor): int(quantity_rows[product_index][bin_index])
                      for bin_index, anchor in enumerate(SELL_BIN_ANCHORS)}
            for product_index, product in enumerate(PRODUCTS)
        }
        plans.append(DailyPlan.create(
            crop_targets=dict(zip(
                CROP_ORDER, (int(v) for v in tensors["crop"][row]))),
            animal_targets=dict(zip(
                ANIMAL_ORDER, (int(v) for v in tensors["animal"][row]))),
            land_count=int(tensors["land"][row]),
            fertilizer_by_crop=dict(zip(
                CROP_ORDER, (int(v) for v in tensors["fertilizer"][row]))),
            care_by_animal=dict(zip(
                ANIMAL_ORDER, (int(v) for v in tensors["care"][row]))),
            sell_quantities=sells,
        ))
    return plans


def decode_outputs_to_plans(
    outputs: Mapping[str, object],
) -> list[DailyPlan]:
    """Decode raw head outputs into validated `DailyPlan`s (one per row)."""
    tensors = decode_outputs_to_action_tensors(outputs)
    return plans_from_action_tensors(tensors)


def round_half_up(value: float) -> int:
    """Exposed for tests: the deterministic sell-quantity rounding rule."""
    return int(math.floor(float(value) + 0.5))
