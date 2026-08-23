"""V0 deterministic executor package (issue #1).

Complete closed-loop surface: typed daily plan, daily manager wrapper with
injection seam and once-per-day caching, mechanical requested->feasible
projection, layout/reconciliation, per-turn task generation, greedy foreman,
and the stateful `ExecutorAgent` with hiring, shortage purchasing, bin sells,
JSON diagnostics, deterministic PASS fallback, plus the optional engine smoke
harness (`python -m executor_v0.smoke`).
"""

from .agent import AgentConfig, ExecutorAgent, make_agent
from .manager import (
    CachingPlanProvider,
    CheckpointPlanProvider,
    FixedPlanProvider,
    PlanProvider,
    decode_daily_plan,
)
from .plan import SELL_BIN_ANCHORS, DailyPlan
from .projection import ProjectionResult, clip_sell, project_plan
from .tasks import GenerationResult, Priority, Task, generate_tasks

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
    "Task",
    "Priority",
    "GenerationResult",
    "generate_tasks",
    "AgentConfig",
    "ExecutorAgent",
    "make_agent",
]
