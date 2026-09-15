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
    "BlockReason": "strip_work",
    "RowKey": "strip_work",
    "WorkStatus": "strip_work",
    "WorkItem": "strip_work",
    "WorkChain": "strip_work",
    "SupplyRequirement": "strip_work",
    "SupplyDemand": "strip_work",
    "SupplySnapshot": "strip_work",
    "RowSummary": "strip_work",
    "RowWorkload": "strip_work",
    "WorkDiagnostics": "strip_work",
    "StripWorkConfig": "strip_work",
    "StripWorkPlan": "strip_work",
    "StripWorkResult": "strip_work",
    "row_key_for_tile": "strip_work",
    "build_strip_work_plan": "strip_work",
    "HorizontalRouteCandidate": "strip_routes",
    "RouteAssignment": "strip_routes",
    "RoutePhase": "strip_routes",
    "StripRoute": "strip_routes",
    "WorkerId": "strip_routes",
    "assign_horizontal_routes": "strip_routes",
    "generate_horizontal_route_candidates": "strip_routes",
    "StripExecutorConfig": "strip_executor",
    "StripExecutorController": "strip_executor",
    "StripExecutorResult": "strip_executor",
    "HireStopReason": "strip_hiring",
    "RouteLaborEstimate": "strip_hiring",
    "StripHiringPlan": "strip_hiring",
    "plan_strip_hiring": "strip_hiring",
    "MarketBlockReason": "strip_market",
    "MarketBootstrapState": "strip_market",
    "MarketIntent": "strip_market",
    "MarketPendingOrder": "strip_market",
    "MarketTurnPlan": "strip_market",
    "build_market_turn_plan": "strip_market",
    "sell_bin_anchor": "strip_market",
    "PickupBatch": "strip_supply",
    "PendingPickup": "strip_supply",
    "RouteSupplyPlan": "strip_supply",
    "RouteSupplyState": "strip_supply",
    "build_route_supply_plans": "strip_supply",
    "extract_route_supply_demand": "strip_supply",
    "extract_tile_supply_demand": "strip_supply",
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
