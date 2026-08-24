"""Economic-context derivation for BC V1 variant E (issue #6, Stage 1).

Single authoritative NumPy implementation of the audited 14-channel vector
(`research/BC_V1_ECONOMIC_CONTEXT.md`, commit 2f48564). The batch adapter
path and the live observation path both call these functions so the two
encodings cannot drift.

Hard rules inherited from the audit:

- schema-v3 sources only: current money, current unlocked quadrants, pinned
  engine cost tables, and the exact same-`(episode_id, seat)` day-1 row;
- submitted market intents and end-of-day snapshots are never read;
- previous-day net cash change is derived only from an exactly adjacent
  earlier daily-start row of the same episode/seat — never from actions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from replay_daily.constants import ANIMALS, CROPS, LAND_PRICES

from .constants import ANIMAL_ORDER, CROP_ORDER

__all__ = [
    "ECONOMIC_DIM",
    "ECONOMIC_CONTEXT_KEY",
    "MODEL_VARIANTS",
    "SEED_COSTS",
    "ANIMAL_COSTS",
    "normalize_model_variant",
    "signed_log_cash",
    "cash_linear",
    "affordability",
    "economic_context",
    "derive_economic_context",
    "EconomicHistory",
]

# Canonical channel order (float32 [N, 14]):
#   [0]  cash_log            signed log1p of money scaled by 1e-4
#   [1]  cash_lin            money scaled by 1e-4
#   [2:7] seed affordability WHEAT/CARROT/TOMATO/STRAWBERRY/MELON
#   [7:10] animal affordability GOOSE/COW/SHEEP
#   [10] next-land affordability (8.0 saturated when all unlocked)
#   [11] land_next_valid     1 iff another locked quadrant exists
#   [12] prev_net_cash_log   signed-log previous daily-start cash delta
#   [13] prev_net_cash_valid 1 iff an exact day-1 row was joined
ECONOMIC_DIM = 14
ECONOMIC_CONTEXT_KEY = "economic_context"

MODEL_VARIANTS = ("V0", "J", "E", "JE")

SEED_COSTS = tuple(int(CROPS[name]["seed"]) for name in CROP_ORDER)
ANIMAL_COSTS = tuple(int(ANIMALS[name]["cost"]) for name in ANIMAL_ORDER)
assert SEED_COSTS == (10, 20, 50, 100, 80)
assert ANIMAL_COSTS == (300, 400, 500)

_CASH_SCALE = 1e-4
_LOG_CLIP = 8.0


def normalize_model_variant(value: str) -> str:
    """Normalize/validate a model-variant name to V0/J/E/JE."""
    if not isinstance(value, str):
        raise ValueError(
            f"model_variant must be a string, got {type(value).__name__}")
    variant = value.strip().upper()
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown model_variant {value!r}; expected one of "
            f"{list(MODEL_VARIANTS)}")
    return variant


def signed_log_cash(money: float) -> float:
    """clamp(sign(m)*log1p(|m|*1e-4), -8, 8) — the shared cash transform."""
    value = float(money)
    encoded = np.sign(value) * np.log1p(abs(value) * _CASH_SCALE)
    return float(np.clip(encoded, -_LOG_CLIP, _LOG_CLIP))


def cash_linear(money: float) -> float:
    """clamp(m*1e-4, -8, 8)."""
    return float(np.clip(float(money) * _CASH_SCALE, -_LOG_CLIP, _LOG_CLIP))


def affordability(money: float, cost: float) -> float:
    """clamp(log1p(max(m, 0)/cost), 0, 8); zero/negative cash gives 0."""
    if cost <= 0:
        raise ValueError(f"affordability cost must be positive, got {cost}")
    return float(np.clip(np.log1p(max(float(money), 0.0) / float(cost)),
                         0.0, _LOG_CLIP))


def economic_context(
    money: float,
    unlocked_count: int,
    prev_net_cash: float | None,
) -> np.ndarray:
    """One 14-channel economic-context row (float32).

    `prev_net_cash=None` encodes the invalid day-0/gap/reset state
    (channels 12/13 = 0/0); otherwise it is the exact difference of the two
    daily-start money snapshots and channel 13 is 1.
    """
    if not np.isfinite(float(money)):
        raise ValueError(f"money must be finite, got {money!r}")
    unlocked_count = int(unlocked_count)
    if not 1 <= unlocked_count <= len(LAND_PRICES) + 1:
        raise ValueError(
            f"unlocked_count must be in [1, {len(LAND_PRICES) + 1}], "
            f"got {unlocked_count}")

    row = np.zeros(ECONOMIC_DIM, dtype=np.float32)
    row[0] = signed_log_cash(money)
    row[1] = cash_linear(money)
    for k, cost in enumerate(SEED_COSTS):
        row[2 + k] = affordability(money, cost)
    for k, cost in enumerate(ANIMAL_COSTS):
        row[7 + k] = affordability(money, cost)
    if unlocked_count < 4:
        next_price = LAND_PRICES[unlocked_count - 1]
        row[10] = affordability(money, next_price)
        row[11] = 1.0
    else:
        # All quadrants unlocked: deterministic saturation, flagged invalid.
        row[10] = _LOG_CLIP
        row[11] = 0.0
    if prev_net_cash is not None:
        if not np.isfinite(float(prev_net_cash)):
            raise ValueError(
                f"prev_net_cash must be finite, got {prev_net_cash!r}")
        row[12] = signed_log_cash(prev_net_cash)
        row[13] = 1.0
    return row


def derive_economic_context(
    episode_ids: Sequence,
    seats: Sequence[int],
    days: Sequence[int],
    money: Sequence[float],
    unlocked_counts: Sequence[int],
) -> np.ndarray:
    """Batch economic context [N, 14] from per-row canonical columns.

    Grouping is strictly by `(episode_id, seat)` with an exact `day - 1`
    dict lookup — never positional, never order-dependent, so rows may be
    split across files in any interleaving. Duplicate
    `(episode_id, seat, day)` rows fail loudly. Day 0, gaps, resets, and
    unknown episode ids never join across groups: channels 12/13 encode
    0/invalid. Only *earlier* rows are read; future/end data is untouched.
    """
    n = len(days)
    if not (len(episode_ids) == len(seats) == len(money)
            == len(unlocked_counts) == n):
        raise ValueError("economic-context column lengths differ")
    groups: dict[tuple[int, int], dict[int, int]] = {}
    for i in range(n):
        episode_id = episode_ids[i]
        if episode_id is None:
            raise ValueError(
                f"row {i}: economic context requires metadata.episode_id; "
                f"refusing to guess episode grouping")
        key = (int(episode_id), int(seats[i]))
        group = groups.setdefault(key, {})
        day = int(days[i])
        if day in group:
            raise ValueError(
                f"duplicate canonical row for episode {key[0]} seat "
                f"{key[1]} day {day}; refusing ambiguous history")
        group[day] = i

    out = np.zeros((n, ECONOMIC_DIM), dtype=np.float32)
    for group in groups.values():
        for day, idx in group.items():
            prev_idx = group.get(day - 1)  # exact key join; earlier day only
            delta = (
                float(money[idx]) - float(money[prev_idx])
                if prev_idx is not None else None
            )
            out[idx] = economic_context(
                float(money[idx]), int(unlocked_counts[idx]), delta)
    return out


class EconomicHistory:
    """Live tracker of daily-start (day, money) for ONE episode/seat.

    Records only observed daily-start state — never actions or intents.
    `observe(day, money)` returns `(delta, valid)` for the current decision
    and then records the observation. Valid iff the previously recorded
    observation was exactly `day - 1`; any gap, backwards step, or new
    episode (call `reset()`) yields `(0.0, False)`.
    """

    def __init__(self) -> None:
        self._day: int | None = None
        self._money: float | None = None

    def reset(self) -> None:
        self._day = None
        self._money = None

    def observe(self, day: int, money: float) -> tuple[float, bool]:
        day = int(day)
        money = float(money)
        if day < 0:
            raise ValueError(f"day must be nonnegative, got {day}")
        if not np.isfinite(money):
            raise ValueError(f"money must be finite, got {money!r}")
        valid = self._day is not None and self._day == day - 1
        delta = money - float(self._money) if valid else 0.0
        self._day = day
        self._money = money
        return delta, bool(valid)

    def context(self, day: int, money: float, unlocked_count: int) \
            -> np.ndarray:
        """Convenience: observe and return the full 14-channel row."""
        delta, valid = self.observe(day, money)
        return economic_context(money, unlocked_count,
                                delta if valid else None)
