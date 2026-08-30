"""Small promotion-ratchet state holder for PPO rollout opponents."""

from __future__ import annotations

from dataclasses import dataclass

from rl_manager.types import BatchedPlanPolicy


@dataclass
class PromotionRatchet:
    """Retain BC-E and replace only the current opponent on a PASS."""

    original_opponent: BatchedPlanPolicy
    current_opponent: BatchedPlanPolicy | None = None
    promotions: int = 0

    def __post_init__(self) -> None:
        if self.current_opponent is None:
            self.current_opponent = self.original_opponent

    def apply(self, passed: bool, snapshot: BatchedPlanPolicy) -> bool:
        """Atomically install ``snapshot`` only when the gate passes."""
        if not passed:
            return False
        self.current_opponent = snapshot
        self.promotions += 1
        return True


__all__ = ["PromotionRatchet"]
