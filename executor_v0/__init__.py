"""V0 deterministic executor package (issue #1).

Complete closed-loop surface: typed daily plan, daily manager wrapper with
injection seam and once-per-day caching, mechanical requested->feasible
projection, layout/reconciliation, per-turn task generation, greedy foreman,
and the stateful `ExecutorAgent` with hiring, shortage purchasing, bin sells,
JSON diagnostics, deterministic PASS fallback, plus the optional engine smoke
harness (`python -m executor_v0.smoke`).

Imports are lazy (PEP 562): importing ``executor_v0`` or a lightweight
submodule such as ``executor_v0.plan`` must not import the Torch-owning
``executor_v0.manager``/``executor_v0.agent`` modules, so framework-neutral
workers can reuse the plan transport. Public names remain identical and are
resolved on first attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# public name -> owning submodule (must preserve the historical eager surface)
_EXPORTS: dict[str, str] = {
    "AgentConfig": "agent",
    "ExecutorAgent": "agent",
    "make_agent": "agent",
    "HiringRecommendation": "hiring",
    "ScheduleHiringPolicy": "hiring",
    "recommend_hires": "hiring",
    "PersistentTaskScheduler": "scheduler",
    "SchedulerConfig": "scheduler",
    "SchedulerResult": "scheduler",
    "CachingPlanProvider": "manager",
    "CheckpointPlanProvider": "manager",
    "FixedPlanProvider": "manager",
    "PlanProvider": "manager",
    "decode_daily_plan": "manager",
    "SELL_BIN_ANCHORS": "plan",
    "DailyPlan": "plan",
    "ProjectionResult": "projection",
    "clip_sell": "projection",
    "project_plan": "projection",
    "GenerationResult": "tasks",
    "Priority": "tasks",
    "Task": "tasks",
    "generate_tasks": "tasks",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") \
            from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
