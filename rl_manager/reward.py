"""Explicit terminal reward semantics for manager PPO rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

TERMINAL_WLT = "terminal_wlt"
TERMINAL_OWN_BANK = "terminal_own_bank"
TERMINAL_OWN_BANK_LINEAR = "terminal_own_bank_linear"
REWARD_MODES = (TERMINAL_WLT, TERMINAL_OWN_BANK, TERMINAL_OWN_BANK_LINEAR)

BEHAVIOR_SHAPING_FEATURES = (
    "goose", "cow", "sheep",
    "wheat", "carrot", "tomato", "strawberry", "melon",
)
MAX_TOTAL_SHAPING_WEIGHT = 0.25
# Stage 2.5 physical crop counts fit on the 100-cell board, and absolute
# animal action targets are bounded to 100 by stage25_mechanics.
_MAX_REALIZED_COUNT = 100


@dataclass(frozen=True)
class BehaviorShapingFeature:
    """One bounded, normalized realized-count potential."""

    target: int
    weight: float

    def __post_init__(self) -> None:
        if isinstance(self.target, bool) or not isinstance(self.target, Integral):
            raise TypeError("behavior-shaping target must be an integer, not bool")
        target = int(self.target)
        if not 1 <= target <= _MAX_REALIZED_COUNT:
            raise ValueError(
                f"behavior-shaping target must be in [1, {_MAX_REALIZED_COUNT}]")
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise TypeError("behavior-shaping weight must be a real number, not bool")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("behavior-shaping weight must be finite and >= 0")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "weight", weight)

    def to_json_dict(self) -> dict[str, int | float]:
        return {"target": self.target, "weight": self.weight}


@dataclass(frozen=True)
class BehaviorShapingConfig:
    """Temporary Stage 2.5 shaping for the eight approved strategy features."""

    goose: BehaviorShapingFeature | None = None
    cow: BehaviorShapingFeature | None = None
    sheep: BehaviorShapingFeature | None = None
    wheat: BehaviorShapingFeature | None = None
    carrot: BehaviorShapingFeature | None = None
    tomato: BehaviorShapingFeature | None = None
    strawberry: BehaviorShapingFeature | None = None
    melon: BehaviorShapingFeature | None = None

    def __post_init__(self) -> None:
        weights: list[float] = []
        for name in BEHAVIOR_SHAPING_FEATURES:
            feature = getattr(self, name)
            if feature is not None and not isinstance(
                    feature, BehaviorShapingFeature):
                raise TypeError(
                    f"behavior-shaping feature {name!r} must be a "
                    "BehaviorShapingFeature or None")
            # A paired zero-weight CLI feature has exactly the same canonical
            # representation as omission, so resume comparisons stay stable.
            if feature is not None and feature.weight == 0.0:
                object.__setattr__(self, name, None)
                continue
            if feature is not None:
                weights.append(feature.weight)
        total = math.fsum(weights)
        if total > MAX_TOTAL_SHAPING_WEIGHT:
            raise ValueError(
                "total behavior-shaping weight must be <= "
                f"{MAX_TOTAL_SHAPING_WEIGHT}, got {total}")

    @property
    def enabled(self) -> bool:
        return any(getattr(self, name) is not None
                   for name in BEHAVIOR_SHAPING_FEATURES)

    def active_features(self) -> tuple[tuple[str, BehaviorShapingFeature], ...]:
        return tuple(
            (name, feature)
            for name in BEHAVIOR_SHAPING_FEATURES
            if (feature := getattr(self, name)) is not None
        )

    def to_json_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            name: feature.to_json_dict()
            for name, feature in self.active_features()
        }

    @classmethod
    def from_json_dict(cls, value: object) -> "BehaviorShapingConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("behavior_shaping must be a JSON object")
        unknown = set(value) - set(BEHAVIOR_SHAPING_FEATURES)
        if unknown:
            raise ValueError(
                f"unknown behavior-shaping features: {sorted(unknown)!r}")
        kwargs: dict[str, BehaviorShapingFeature] = {}
        for name in BEHAVIOR_SHAPING_FEATURES:
            raw = value.get(name)
            if raw is None:
                continue
            if not isinstance(raw, Mapping) or set(raw) != {"target", "weight"}:
                raise ValueError(
                    f"behavior-shaping feature {name!r} must contain exactly "
                    "target and weight")
            kwargs[name] = BehaviorShapingFeature(
                target=raw["target"], weight=raw["weight"])
        return cls(**kwargs)

    def potential(self, counts: Mapping[str, int | float]) -> float:
        return float(sum(
            feature.weight * normalized_saturated_potential(
                counts[name], feature.target)
            for name, feature in self.active_features()
        ))

    def contributions(
        self,
        previous_counts: Mapping[str, int | float],
        next_counts: Mapping[str, int | float],
    ) -> dict[str, float]:
        return {
            name: feature.weight * (
                normalized_saturated_potential(
                    next_counts[name], feature.target)
                - normalized_saturated_potential(
                    previous_counts[name], feature.target))
            for name, feature in self.active_features()
        }


def normalized_saturated_potential(count: int | float, target: int) -> float:
    """Return ``clamp(count, 0, target) / target``."""
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 1:
        raise ValueError("potential target must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, Real):
        raise TypeError("realized count must be a real number, not bool")
    realized = float(count)
    if not math.isfinite(realized):
        raise ValueError("realized count must be finite")
    return min(max(realized, 0.0), float(target)) / float(target)


@dataclass(frozen=True)
class RewardConfig:
    mode: str = TERMINAL_WLT
    bank_baseline: float = 3000.0
    bank_scale: float = 50000.0
    behavior_shaping: BehaviorShapingConfig = field(
        default_factory=BehaviorShapingConfig)

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(
                f"reward mode must be one of {REWARD_MODES}, got {self.mode!r}")
        if not math.isfinite(self.bank_baseline):
            raise ValueError("bank reward baseline must be finite")
        if not math.isfinite(self.bank_scale) or self.bank_scale <= 0:
            raise ValueError("bank reward scale must be finite and > 0")
        if isinstance(self.behavior_shaping, Mapping):
            object.__setattr__(
                self, "behavior_shaping",
                BehaviorShapingConfig.from_json_dict(self.behavior_shaping))
        if not isinstance(self.behavior_shaping, BehaviorShapingConfig):
            raise TypeError("behavior_shaping must be a BehaviorShapingConfig")
        if self.behavior_shaping.enabled and self.mode != TERMINAL_WLT:
            raise ValueError(
                "behavior shaping requires reward mode terminal_wlt")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bank_baseline": float(self.bank_baseline),
            "bank_scale": float(self.bank_scale),
            "behavior_shaping": self.behavior_shaping.to_json_dict(),
        }


def terminal_rewards(final_banks: Sequence[float], config: RewardConfig) -> list[float]:
    """Return terminal-only rewards independently for both observed seats."""
    if len(final_banks) != 2:
        raise ValueError(f"expected two final banks, got {len(final_banks)}")
    banks = [float(bank) for bank in final_banks]
    if not all(math.isfinite(bank) for bank in banks):
        raise ValueError(f"final banks must be finite, got {banks!r}")
    if config.mode == TERMINAL_OWN_BANK:
        return [math.tanh((bank - config.bank_baseline) / config.bank_scale)
                for bank in banks]
    if config.mode == TERMINAL_OWN_BANK_LINEAR:
        return [(bank - config.bank_baseline) / config.bank_scale
                for bank in banks]
    margin = banks[0] - banks[1]
    if margin == 0:
        return [0.0, 0.0]
    return ([1.0, -1.0] if margin > 0 else [-1.0, 1.0])


__all__ = [
    "BEHAVIOR_SHAPING_FEATURES",
    "MAX_TOTAL_SHAPING_WEIGHT",
    "REWARD_MODES",
    "TERMINAL_OWN_BANK",
    "TERMINAL_OWN_BANK_LINEAR",
    "TERMINAL_WLT",
    "BehaviorShapingConfig",
    "BehaviorShapingFeature",
    "RewardConfig",
    "normalized_saturated_potential",
    "terminal_rewards",
]
