"""Exact JAX mirror of `bc_manager.loss.manager_loss`.

Seven fixed-weight group losses (crop, animal, land, fertilizer, care,
sell_presence, sell_quantity) with identical reduction semantics:

- count groups: mean cross-entropy over the whole [B, K] cell grid;
- land: cross-entropy against `land_count - 1`;
- sell_presence: mean BCE-with-logits over all 54 cells;
- sell_quantity: presence-masked SmoothL1 (beta=1) in log1p space divided
  by max(mask sum, 1).

Targets are validated eagerly and fail loudly on shape/range violations —
no clipping or fabrication.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bc_manager.loss import GROUP_NAMES, ManagerLossConfig
from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_LAND_CLASSES,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
)

REQUIRED_TARGET_KEYS = frozenset({
    "crop_target", "animal_target", "land_count", "fertilizer_target",
    "care_target", "sell_presence", "sell_quantity_log1p",
})


def _require_key(targets: Mapping[str, object], key: str) -> object:
    if key not in targets:
        raise ValueError(
            f"missing required target {key!r}; schema-v3 targets are never "
            f"fabricated")
    return targets[key]


def _validate_count_target(name: str, target: object, batch: int,
                           width: int, count_classes: int) -> np.ndarray:
    array = np.asarray(target)
    if array.shape != (batch, width):
        raise ValueError(
            f"{name} must have shape ({batch}, {width}), got {array.shape}")
    long_target = array.astype(np.int64)
    if int(long_target.min()) < 0 or int(long_target.max()) >= count_classes:
        raise ValueError(
            f"{name} values must be within [0, {count_classes - 1}] "
            f"(count_max), got min={int(long_target.min())}, "
            f"max={int(long_target.max())}; refusing to clip")
    return long_target


def validate_target_shapes(targets: Mapping[str, object], batch: int,
                           count_classes: int) -> dict[str, np.ndarray]:
    """Validate target shapes/ranges without needing model outputs.

    Mirrors `bc_manager.loss` semantics; returns validated host arrays
    (integer labels with land already shifted to 0-based, float presence
    and quantity). Usable before any forward pass (e.g. train step).
    """
    crop_target = _validate_count_target(
        "crop_target", _require_key(targets, "crop_target"),
        batch, NUM_CROPS, count_classes)
    animal_target = _validate_count_target(
        "animal_target", _require_key(targets, "animal_target"),
        batch, NUM_ANIMALS, count_classes)
    fertilizer_target = _validate_count_target(
        "fertilizer_target", _require_key(targets, "fertilizer_target"),
        batch, NUM_CROPS, count_classes)
    care_target = _validate_count_target(
        "care_target", _require_key(targets, "care_target"),
        batch, NUM_ANIMALS, count_classes)

    land_array = np.asarray(_require_key(targets, "land_count"))
    if land_array.shape != (batch,):
        raise ValueError(
            f"land_count must have shape ({batch},), got {land_array.shape}")
    land_long = land_array.astype(np.int64)
    if int(land_long.min()) < 1 or int(land_long.max()) > NUM_LAND_CLASSES:
        raise ValueError(
            f"land_count values must be within [1, {NUM_LAND_CLASSES}], "
            f"got min={int(land_long.min())}, max={int(land_long.max())}; "
            f"refusing to clip")

    expected_presence = (batch, NUM_PRODUCTS, SELL_BIN_COUNT)
    presence = np.asarray(_require_key(targets, "sell_presence"),
                          dtype=np.float32)
    if presence.shape != expected_presence:
        raise ValueError(
            f"sell_presence must have shape {expected_presence}, got "
            f"{presence.shape}")
    if bool(((presence != 0) & (presence != 1)).any()):
        raise ValueError("sell_presence target must be binary 0/1")

    quantity = np.asarray(_require_key(targets, "sell_quantity_log1p"),
                          dtype=np.float32)
    if quantity.shape != expected_presence:
        raise ValueError(
            f"sell_quantity_log1p must have shape {expected_presence}, got "
            f"{quantity.shape}")
    if not bool(np.isfinite(quantity).all()):
        raise ValueError("sell_quantity_log1p target contains non-finite "
                         "values; refusing to fabricate")

    return {
        "crop_target": crop_target,
        "animal_target": animal_target,
        "fertilizer_target": fertilizer_target,
        "care_target": care_target,
        "land_target": land_long - 1,
        "presence": presence,
        "quantity": quantity,
    }


def validate_targets(outputs: Mapping[str, jax.Array],
                     targets: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Eager target validation mirroring `bc_manager.loss`; returns the
    validated integer/float host arrays used by `manager_loss`."""
    crop_logits = np.asarray(outputs["crop_logits"])
    b = crop_logits.shape[0]
    count_classes = crop_logits.shape[-1]
    return validate_target_shapes(targets, b, count_classes)


def manager_loss(
    outputs: Mapping[str, jax.Array],
    targets: Mapping[str, object],
    config: ManagerLossConfig | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Weighted sum of seven group means; returns (total, named groups)."""
    config = config if config is not None else ManagerLossConfig()
    validated = validate_targets(outputs, targets)
    _, groups = loss_from_validated(outputs, validated)
    weighted = sum(config.weight(name) * groups[name] for name in GROUP_NAMES)
    return weighted, groups


def loss_from_validated(
    outputs: Mapping[str, jax.Array],
    validated: Mapping[str, np.ndarray],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """JIT-safe core loss over pre-validated targets (unit weights).

    `validated` is the mapping returned by `validate_targets` /
    `validate_target_shapes`. No host-side validation happens here, so this
    is safe to call inside jitted train steps.
    """

    def count_group(logits_key: str, target: np.ndarray) -> jax.Array:
        logits = jnp.asarray(outputs[logits_key], dtype=jnp.float32)
        flat = logits.reshape(-1, logits.shape[-1])
        labels = jnp.asarray(target.reshape(-1), dtype=jnp.int32)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            flat, labels))

    groups: dict[str, jax.Array] = {
        "crop": count_group("crop_logits", validated["crop_target"]),
        "animal": count_group("animal_logits", validated["animal_target"]),
        "land": jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            jnp.asarray(outputs["land_logits"], dtype=jnp.float32),
            jnp.asarray(validated["land_target"], dtype=jnp.int32))),
        "fertilizer": count_group("fertilizer_logits",
                                  validated["fertilizer_target"]),
        "care": count_group("care_logits", validated["care_target"]),
        "sell_presence": jnp.mean(optax.sigmoid_binary_cross_entropy(
            jnp.asarray(outputs["sell_presence_logits"], dtype=jnp.float32),
            jnp.asarray(validated["presence"]))),
    }

    mask = validated["presence"] > 0
    pair = _smooth_l1_beta1(
        jnp.asarray(outputs["sell_quantity_log1p"], dtype=jnp.float32),
        jnp.asarray(validated["quantity"]))
    # Masked mean over positive cells; zero positives stay a differentiable
    # finite zero connected to the prediction graph (torch clamp(min=1)).
    groups["sell_quantity"] = jnp.sum(pair * mask.astype(pair.dtype)) / \
        jnp.maximum(jnp.sum(mask.astype(pair.dtype)), 1.0)

    total = sum(groups[name] for name in GROUP_NAMES)
    return total, groups


def _smooth_l1_beta1(prediction: jax.Array,
                     target: jax.Array) -> jax.Array:
    """torch SmoothL1Loss(beta=1.0, reduction='none')."""
    diff = prediction - target
    return jnp.where(jnp.abs(diff) < 1.0, 0.5 * diff * diff,
                     jnp.abs(diff) - 0.5)
