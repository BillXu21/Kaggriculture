"""Explicit terminal reward semantics for manager PPO rollouts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

TERMINAL_WLT = "terminal_wlt"
TERMINAL_OWN_BANK = "terminal_own_bank"
REWARD_MODES = (TERMINAL_WLT, TERMINAL_OWN_BANK)


@dataclass(frozen=True)
class RewardConfig:
    mode: str = TERMINAL_WLT
    bank_baseline: float = 3000.0
    bank_scale: float = 50000.0

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(
                f"reward mode must be one of {REWARD_MODES}, got {self.mode!r}")
        if not math.isfinite(self.bank_baseline):
            raise ValueError("bank reward baseline must be finite")
        if not math.isfinite(self.bank_scale) or self.bank_scale <= 0:
            raise ValueError("bank reward scale must be finite and > 0")

    def to_json_dict(self) -> dict[str, float | str]:
        return asdict(self)


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
    margin = banks[0] - banks[1]
    if margin == 0:
        return [0.0, 0.0]
    return ([1.0, -1.0] if margin > 0 else [-1.0, 1.0])


__all__ = [
    "REWARD_MODES",
    "TERMINAL_OWN_BANK",
    "TERMINAL_WLT",
    "RewardConfig",
    "terminal_rewards",
]
