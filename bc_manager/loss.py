"""Group-balanced structured loss for the daily-manager Transformer.

Seven fixed-weight group losses (crop, animal, land, fertilizer, care,
sell_presence, sell_quantity). Count tasks average cross-entropy over the
whole [B, K] cell grid of their group; selling is one BCEWithLogits mean
over all 54 presence cells plus a presence-masked SmoothL1 in log1p space,
so 54 sell cells can never dominate the total.

Targets are validated and fail loudly on shape/range violations — no
clipping or fabrication. `sell_quantity_log1p` from the adapter may exceed
log1p(100) per cell (repeated same-bin events); it is consumed as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_LAND_CLASSES,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
)

GROUP_NAMES = ("crop", "animal", "land", "fertilizer", "care",
               "sell_presence", "sell_quantity")

REQUIRED_TARGET_KEYS = frozenset({
    "crop_target", "animal_target", "land_count", "fertilizer_target",
    "care_target", "sell_presence", "sell_quantity_log1p",
})


@dataclass(frozen=True)
class ManagerLossConfig:
    """Fixed task weights for the seven group losses."""

    crop_weight: float = 1.0
    animal_weight: float = 1.0
    land_weight: float = 1.0
    fertilizer_weight: float = 1.0
    care_weight: float = 1.0
    sell_presence_weight: float = 1.0
    sell_quantity_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in GROUP_NAMES:
            weight = getattr(self, f"{name}_weight")
            if not weight > 0.0:
                raise ValueError(
                    f"{name}_weight must be positive, got {weight}")

    def weight(self, group: str) -> float:
        return getattr(self, f"{group}_weight")


def _require_key(targets: Mapping[str, Tensor], key: str) -> Tensor:
    if key not in targets:
        raise ValueError(
            f"missing required target {key!r}; schema-v3 targets are never "
            f"fabricated")
    return targets[key]


def _validate_count_target(name: str, target: Tensor, batch: int,
                           width: int, count_classes: int) -> Tensor:
    if tuple(target.shape) != (batch, width):
        raise ValueError(
            f"{name} must have shape ({batch}, {width}), got "
            f"{tuple(target.shape)}")
    long_target = target.long()
    if int(long_target.min()) < 0 or int(long_target.max()) >= count_classes:
        raise ValueError(
            f"{name} values must be within [0, {count_classes - 1}] "
            f"(count_max), got min={int(long_target.min())}, "
            f"max={int(long_target.max())}; refusing to clip")
    return long_target


def manager_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    config: ManagerLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of seven group means; returns (total, named groups)."""
    config = config if config is not None else ManagerLossConfig()

    crop_logits = outputs["crop_logits"]
    b = crop_logits.shape[0]
    count_classes = crop_logits.shape[-1]

    crop_target = _validate_count_target(
        "crop_target", _require_key(targets, "crop_target"),
        b, NUM_CROPS, count_classes)
    animal_target = _validate_count_target(
        "animal_target", _require_key(targets, "animal_target"),
        b, NUM_ANIMALS, count_classes)
    fertilizer_target = _validate_count_target(
        "fertilizer_target", _require_key(targets, "fertilizer_target"),
        b, NUM_CROPS, count_classes)
    care_target = _validate_count_target(
        "care_target", _require_key(targets, "care_target"),
        b, NUM_ANIMALS, count_classes)

    land_count = _require_key(targets, "land_count")
    if tuple(land_count.shape) != (b,):
        raise ValueError(
            f"land_count must have shape ({b},), got "
            f"{tuple(land_count.shape)}")
    land_long = land_count.long()
    if int(land_long.min()) < 1 or int(land_long.max()) > NUM_LAND_CLASSES:
        raise ValueError(
            f"land_count values must be within [1, {NUM_LAND_CLASSES}], "
            f"got min={int(land_long.min())}, max={int(land_long.max())}; "
            f"refusing to clip")

    presence = _require_key(targets, "sell_presence").to(torch.float32)
    expected_presence = (b, NUM_PRODUCTS, SELL_BIN_COUNT)
    if tuple(presence.shape) != expected_presence:
        raise ValueError(
            f"sell_presence must have shape {expected_presence}, got "
            f"{tuple(presence.shape)}")
    if bool(((presence != 0) & (presence != 1)).any()):
        raise ValueError("sell_presence target must be binary 0/1")

    quantity = _require_key(targets, "sell_quantity_log1p").to(torch.float32)
    if tuple(quantity.shape) != expected_presence:
        raise ValueError(
            f"sell_quantity_log1p must have shape {expected_presence}, got "
            f"{tuple(quantity.shape)}")
    if not bool(torch.isfinite(quantity).all()):
        raise ValueError("sell_quantity_log1p target contains non-finite "
                         "values; refusing to fabricate")

    groups: dict[str, Tensor] = {
        "crop": F.cross_entropy(
            crop_logits.reshape(-1, count_classes), crop_target.reshape(-1)),
        "animal": F.cross_entropy(
            outputs["animal_logits"].reshape(-1, count_classes),
            animal_target.reshape(-1)),
        "land": F.cross_entropy(outputs["land_logits"], land_long - 1),
        "fertilizer": F.cross_entropy(
            outputs["fertilizer_logits"].reshape(-1, count_classes),
            fertilizer_target.reshape(-1)),
        "care": F.cross_entropy(
            outputs["care_logits"].reshape(-1, count_classes),
            care_target.reshape(-1)),
        "sell_presence": F.binary_cross_entropy_with_logits(
            outputs["sell_presence_logits"], presence),
    }

    mask = presence > 0
    pair = F.smooth_l1_loss(outputs["sell_quantity_log1p"], quantity,
                            reduction="none", beta=1.0)
    # Masked mean over positive cells; with zero positives this stays a
    # differentiable finite zero connected to the prediction graph.
    groups["sell_quantity"] = (pair * mask.to(pair.dtype)).sum() / \
        mask.sum().clamp(min=1.0).to(pair.dtype)

    total = sum(config.weight(name) * groups[name] for name in GROUP_NAMES)
    return total, groups


_CATEGORICAL_OUTPUTS = {
    "crop": "crop_logits",
    "animal": "animal_logits",
    "land": "land_logits",
    "fertilizer": "fertilizer_logits",
    "care": "care_logits",
}


def teacher_student_loss(
    student_outputs: Mapping[str, Tensor],
    teacher_outputs: Mapping[str, Tensor],
    *,
    temperature: float = 1.0,
    config: ManagerLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Group-balanced soft-target loss from an E teacher to a JE student.

    Categorical groups use Hinton-style KL with the ``temperature**2``
    correction. Sell presence uses the teacher sigmoid probability as a soft
    BCE target. Sell quantity uses teacher-presence-weighted SmoothL1 over
    the log1p grid, retaining the hard loss's positive-cell semantics.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError(
            f"distill_temperature must be positive, got {temperature}")
    config = config if config is not None else ManagerLossConfig()
    t = float(temperature)
    groups: dict[str, Tensor] = {}
    for group, key in _CATEGORICAL_OUTPUTS.items():
        student = student_outputs[key]
        teacher = teacher_outputs[key].detach()
        if student.shape != teacher.shape:
            raise ValueError(
                f"teacher/student {key} shapes differ: "
                f"{tuple(teacher.shape)} != {tuple(student.shape)}")
        per_cell = F.kl_div(
            F.log_softmax(student / t, dim=-1),
            F.softmax(teacher / t, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        groups[group] = per_cell.mean() * (t * t)

    student_presence = student_outputs["sell_presence_logits"]
    teacher_presence = teacher_outputs["sell_presence_logits"].detach()
    if student_presence.shape != teacher_presence.shape:
        raise ValueError(
            "teacher/student sell_presence_logits shapes differ: "
            f"{tuple(teacher_presence.shape)} != "
            f"{tuple(student_presence.shape)}")
    groups["sell_presence"] = F.binary_cross_entropy_with_logits(
        student_presence, torch.sigmoid(teacher_presence))

    student_quantity = student_outputs["sell_quantity_log1p"]
    teacher_quantity = teacher_outputs["sell_quantity_log1p"].detach()
    if student_quantity.shape != teacher_quantity.shape:
        raise ValueError(
            "teacher/student sell_quantity_log1p shapes differ: "
            f"{tuple(teacher_quantity.shape)} != "
            f"{tuple(student_quantity.shape)}")
    quantity_pair = F.smooth_l1_loss(
        student_quantity, teacher_quantity, reduction="none", beta=1.0)
    teacher_presence_probability = torch.sigmoid(teacher_presence).detach()
    groups["sell_quantity"] = (
        quantity_pair * teacher_presence_probability).sum() / \
        teacher_presence_probability.sum().clamp(min=1.0)

    total = sum(config.weight(name) * groups[name] for name in GROUP_NAMES)
    return total, groups


def mixed_manager_loss(hard_loss: Tensor, distill_loss: Tensor,
                       distill_weight: float) -> Tensor:
    """Return the configured hard-label/teacher objective mixture."""
    if not 0.0 <= distill_weight <= 1.0:
        raise ValueError(
            f"distill_weight must be within [0, 1], got {distill_weight}")
    return (1.0 - distill_weight) * hard_loss + \
        distill_weight * distill_loss
