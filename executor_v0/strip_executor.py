"""Opt-in Packet 2 executor for fixed five-tile strip ownership.

This controller is intentionally separate from :mod:`executor_v0.agent` and
does not use the persistent scheduler.  Ownership is assigned once at the
start of a day; only the immutable Packet 1 forecast is refreshed on later
turns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from executor_v0.plan import DailyPlan
from executor_v0.strip_routes import (
    RouteAssignment,
    RoutePhase,
    StripRoute,
    WorkerId,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
)
from executor_v0.strip_work import (
    BlockReason,
    StripWorkConfig,
    StripWorkPlan,
    WorkItem,
    WorkStatus,
    build_strip_work_plan,
)

__all__ = [
    "StripExecutorConfig",
    "StripExecutorController",
    "StripExecutorResult",
]


_LOCAL_PRIORITY = {
    # Maintenance that would otherwise be lost by a later action comes first.
    "FEED": 10,
    "FERTILIZE": 20,
    "WATER": 30,
    "CARE": 40,
    "COLLECT_FERTILIZER": 50,
    "HARVEST": 60,
    "DIG": 70,
    "BUILD_COOP": 80,
    "BUILD_PASTURE": 81,
    "PLACE": 90,
    "PLANT": 100,
}


@dataclass(frozen=True)
class StripExecutorConfig:
    """Packet 2 policy knobs; no hiring, pickup, or route optimization."""

    acting_seat: int = 0
    work_config: StripWorkConfig = field(default_factory=StripWorkConfig)


@dataclass(frozen=True)
class StripExecutorResult:
    """Actions and JSON-friendly diagnostics for one primitive turn."""

    farmer_action: tuple
    hands_actions: tuple[tuple, ...]
    market_actions: tuple[tuple, ...]
    diagnostics: dict[str, Any]

    def action_dict(self) -> dict[str, Any]:
        return {
            "farmer": list(self.farmer_action),
            "hands": [list(action) for action in self.hands_actions],
            "market": [list(action) for action in self.market_actions],
        }


WorkPlanBuilder = Callable[..., StripWorkPlan]


class StripExecutorController:
    """Fixed daily ownership and a one-pass deterministic route executor."""

    def __init__(
        self,
        *,
        config: StripExecutorConfig = StripExecutorConfig(),
        work_builder: WorkPlanBuilder = build_strip_work_plan,
    ) -> None:
        self.config = config
        self._work_builder = work_builder
        self._day: int | None = None
        self._routes: dict[WorkerId, StripRoute] = {}
        self._assignment: RouteAssignment | None = None
        self._unassigned_ids: tuple[str, ...] = ()
        self._plan: StripWorkPlan | None = None
        self._passed_work: dict[str, dict[str, str]] = {}
        self._daily: dict[str, Any] = {}

    @property
    def routes(self) -> tuple[StripRoute, ...]:
        return tuple(sorted(self._routes.values(), key=lambda route: route.route_id))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics()

    def reset_day(self, obs: Mapping[str, Any], plan: DailyPlan) -> StripWorkPlan:
        """Discard route state and assign the current fixed workforce."""

        day = int(obs.get("day", 0))
        work_plan = self._build_work_plan(obs, plan)
        positions = self._worker_positions(obs)
        candidates = generate_horizontal_route_candidates(work_plan)
        assignment = assign_horizontal_routes(
            candidates,
            positions,
            assignment_hour=int(obs.get("hour", 0)),
        )
        self._day = day
        self._plan = work_plan
        self._assignment = assignment
        self._routes = {route.owner: route for route in assignment.routes}
        self._unassigned_ids = tuple(route.route_id for route in assignment.unassigned)
        self._passed_work = {route.route_id: {} for route in assignment.routes}
        self._daily = {
            "day": day,
            "assignment_hour": int(obs.get("hour", 0)),
            "active_routes": len(candidates),
            "assigned_routes": len(assignment.routes),
            "unassigned_routes": len(assignment.unassigned),
            "workers": len(positions),
            "unassigned_active_routes": list(self._unassigned_ids),
            "route_workload": {
                candidate.route_id: candidate.workload_interactions
                for candidate in candidates
            },
            "tileless_unresolved_work": [
                item.id for item in work_plan.items if item.tile is None
            ],
        }
        return work_plan

    def act(self, obs: Mapping[str, Any], plan: DailyPlan) -> StripExecutorResult:
        """Return one action per current worker and refresh work once."""

        day = int(obs.get("day", 0))
        if self._day != day:
            work_plan = self.reset_day(obs, plan)
        else:
            work_plan = self._build_work_plan(obs, plan)
            self._plan = work_plan

        positions = self._worker_positions(obs)
        for worker, route in self._routes.items():
            if worker not in positions and route.phase not in (
                RoutePhase.DONE,
                RoutePhase.INVALID,
            ):
                route.phase = RoutePhase.INVALID
        actions: list[tuple] = []
        for worker in sorted(positions):
            route = self._routes.get(worker)
            if route is None:
                actions.append(("PASS",))
                continue
            actions.append(self._act_worker(route, positions[worker], work_plan, obs))

        farmer_action = actions[0] if actions else ("PASS",)
        hands_actions = tuple(actions[1:])
        return StripExecutorResult(
            farmer_action=farmer_action,
            hands_actions=hands_actions,
            market_actions=(),
            diagnostics=self._diagnostics(),
        )

    next_worker_actions = act

    def _build_work_plan(
        self, obs: Mapping[str, Any], plan: DailyPlan
    ) -> StripWorkPlan:
        return self._work_builder(
            obs,
            plan,
            config=self.config.work_config,
            acting_seat=self.config.acting_seat,
        )

    def _worker_positions(self, obs: Mapping[str, Any]) -> dict[WorkerId, tuple[int, int]]:
        farm = obs["farms"][self.config.acting_seat]
        raw_positions = [farm.get("farmer")]
        raw_positions.extend(farm.get("hands") or ())
        positions: dict[WorkerId, tuple[int, int]] = {}
        for index, raw in enumerate(raw_positions):
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                continue
            x, y = int(raw[0]), int(raw[1])
            positions[WorkerId(index)] = (y, x)
        return positions

    def _worker_inventory(
        self, obs: Mapping[str, Any], worker: WorkerId
    ) -> dict[str, int]:
        inventories = ((obs.get("private") or {}).get("inventories") or ())
        if worker.index >= len(inventories):
            return {}
        raw = inventories[worker.index]
        return {
            str(item): int(amount)
            for item, amount in raw.items()
            if int(amount) > 0
        } if isinstance(raw, Mapping) else {}

    def _act_worker(
        self,
        route: StripRoute,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> tuple:
        hour = int(obs.get("hour", 0))
        if route.phase == RoutePhase.INVALID:
            return ("PASS",)
        if route.phase == RoutePhase.DONE:
            route.pass_turns_after_completion += 1
            return ("PASS",)
        self._record_late_work(route, work_plan)

        if route.pending_cursor is not None:
            expected = route.traversal[route.pending_cursor]
            if position != expected:
                movement = _vertical_first_step(position, expected)
                if movement is None:
                    route.blocked_local_work["ROUTE_BLOCKED"] = (
                        route.blocked_local_work.get("ROUTE_BLOCKED", 0) + 1
                    )
                    return ("PASS",)
                route.movement_turns += 1
                return movement
            route.cursor = route.pending_cursor
            route.pending_cursor = None
            route.phase = RoutePhase.SWEEP

        target = route.current_tile
        if route.phase == RoutePhase.TRAVEL_TO_ENTRY or position != target:
            if route.phase == RoutePhase.TRAVEL_TO_ENTRY and position == target:
                route.phase = RoutePhase.SWEEP
            else:
                movement = _vertical_first_step(position, target)
                if movement is None:
                    route.blocked_local_work["ROUTE_BLOCKED"] = (
                        route.blocked_local_work.get("ROUTE_BLOCKED", 0) + 1
                    )
                    return ("PASS",)
                route.movement_turns += 1
                return movement

        inventory = self._worker_inventory(obs, route.owner)
        local_items = tuple(item for item in work_plan.items if item.tile == target)
        item = self._select_local_item(local_items, inventory)
        if item is not None:
            action = _interaction_action(item)
            if action is not None:
                route.actions_performed[item.kind] = (
                    route.actions_performed.get(item.kind, 0) + 1
                )
                route.interaction_turns += item.interaction_turns
                return action

        self._record_skipped_work(route, local_items, inventory)
        self._record_passed_tile(route, target, local_items)
        if route.cursor + 1 >= len(route.traversal):
            route.phase = RoutePhase.DONE
            route.completion_hour = hour
            return ("PASS",)
        route.pending_cursor = route.cursor + 1
        return _vertical_first_step(position, route.traversal[route.pending_cursor]) or (
            "PASS",
        )

    def _select_local_item(
        self, items: tuple[WorkItem, ...], inventory: Mapping[str, int]
    ) -> WorkItem | None:
        candidates = sorted(
            (item for item in items if _supported_kind(item.kind)),
            key=lambda item: (_LOCAL_PRIORITY.get(item.kind, 1000), item.id),
        )
        for item in candidates:
            if item.status == WorkStatus.READY and _has_worker_supplies(item, inventory):
                return item
            # A Packet 1 aggregate shortage is not a shortage for a worker
            # already carrying the exact resource. Dependencies remain strict.
            if (
                item.block_reason == BlockReason.MISSING_SUPPLY
                and not item.depends_on
                and _has_worker_supplies(item, inventory)
            ):
                return item
        return None

    def _record_skipped_work(
        self,
        route: StripRoute,
        items: tuple[WorkItem, ...],
        inventory: Mapping[str, int],
    ) -> None:
        for item in items:
            if not _supported_kind(item.kind):
                continue
            reason = None
            if item.status != WorkStatus.READY:
                reason = item.block_reason.value if item.block_reason else item.status.value
            elif not _has_worker_supplies(item, inventory):
                reason = "SUPPLY_TRIP_NEEDED"
                route.unavailable_supply_work[item.kind] = (
                    route.unavailable_supply_work.get(item.kind, 0) + 1
                )
            if reason:
                route.blocked_local_work[reason] = (
                    route.blocked_local_work.get(reason, 0) + 1
                )

    def _record_passed_tile(
        self, route: StripRoute, tile: tuple[int, int], items: tuple[WorkItem, ...]
    ) -> None:
        passed = self._passed_work.setdefault(route.route_id, {})
        for item in items:
            if item.tile == tile:
                passed[item.id] = item.status.value

    def _record_late_work(self, route: StripRoute, work_plan: StripWorkPlan) -> None:
        if route.cursor == 0:
            return
        passed = self._passed_work.setdefault(route.route_id, {})
        by_id = {item.id: item for item in work_plan.items}
        for item_id, previous_status in passed.items():
            item = by_id.get(item_id)
            if (
                item is not None
                and item.status == WorkStatus.READY
                and previous_status != WorkStatus.READY
            ):
                route.late_work_ids.add(item_id)
        for item in work_plan.items:
            if (
                item.tile in route.traversal[: route.cursor]
                and item.id not in passed
                and item.status == WorkStatus.READY
            ):
                route.late_work_ids.add(item.id)

    def _diagnostics(self) -> dict[str, Any]:
        assignment = self._assignment
        routes = self.routes
        completed = sum(route.phase == RoutePhase.DONE for route in routes)
        payload = dict(self._daily)
        payload.update(
            {
                "idle_workers": [
                    worker.label
                    for worker in (assignment.idle_workers if assignment else ())
                ],
                "completed_routes": completed,
                "unfinished_routes": len(routes) - completed,
                "route_diagnostics": [route.to_json_dict() for route in routes],
                "actual_interactions_completed": sum(
                    route.interaction_turns for route in routes
                ),
                "workload_from_packet1": {
                    route_id: workload
                    for route_id, workload in payload.get("route_workload", {}).items()
                },
            }
        )
        return payload


def _supported_kind(kind: str) -> bool:
    return kind in _LOCAL_PRIORITY


def _has_worker_supplies(item: WorkItem, inventory: Mapping[str, int]) -> bool:
    return all(
        requirement.scope != "inventory"
        or int(inventory.get(requirement.item, 0)) >= requirement.quantity
        for requirement in item.required_supplies
    )


def _interaction_action(item: WorkItem) -> tuple | None:
    if item.kind in {
        "WATER",
        "HARVEST",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "CARE",
        "FERTILIZE",
        "COLLECT_FERTILIZER",
    }:
        return (item.kind,)
    if item.kind == "PLANT":
        return ("PLANT", item.crop) if item.crop else None
    if item.kind == "PLACE":
        return (
            ("PLACE", item.animal, item.quantity)
            if item.animal and item.quantity > 0
            else None
        )
    return None


def _vertical_first_step(
    position: tuple[int, int], target: tuple[int, int]
) -> tuple | None:
    """One legal in-bounds step: y first, then x, with no path search."""

    y, x = position
    target_y, target_x = target
    if y != target_y:
        direction, destination = (
            ("SOUTH", (y + 1, x)) if target_y > y else ("NORTH", (y - 1, x))
        )
    elif x != target_x:
        direction, destination = (
            ("EAST", (y, x + 1)) if target_x > x else ("WEST", (y, x - 1))
        )
    else:
        return None
    if not (0 <= destination[0] < 10 and 0 <= destination[1] < 10):
        return None
    return (direction,)
