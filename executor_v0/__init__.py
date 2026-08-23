"""V0 deterministic executor package (issue #1).

Stage 2 surface only: typed daily plan, daily manager wrapper with injection
seam and once-per-day caching, and mechanical requested->feasible projection.
Layout/reconciliation, task generation, foreman, agent runtime, hiring,
purchasing, and primitive sell execution arrive in later stages.
"""

from .manager import (
    CachingPlanProvider,
    CheckpointPlanProvider,
    FixedPlanProvider,
    PlanProvider,
    decode_daily_plan,
)
from .plan import SELL_BIN_ANCHORS, DailyPlan
from .projection import ProjectionResult, clip_sell, project_plan

__all__ = [
    "DailyPlan",
    "SELL_BIN_ANCHORS",
    "PlanProvider",
    "FixedPlanProvider",
    "CheckpointPlanProvider",
    "CachingPlanProvider",
    "decode_daily_plan",
    "ProjectionResult",
    "project_plan",
    "clip_sell",
]
