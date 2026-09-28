"""Opt-in Packet 2/3 executor for fixed five-tile strip ownership.

This controller is intentionally separate from :mod:`executor_v0.agent` and
does not use the persistent scheduler.  Procurement and coverage-driven
hiring are bootstrapped before ownership is assigned once at the start of a
day; only the Packet 1 forecast is refreshed on later turns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

from executor_v0.plan import DailyPlan
from replay_daily.constants import FARM_HAND_COST_MULT_DEFAULT, hire_cost
from executor_v0.strip_cost import (
    nearest_shed_access, ordered_inventory_demand, route_cost_segment_from_items,
    simulate_route_cost,
)
from executor_v0.strip_claim_board import (
    ClaimBoard, ClaimPhase, RowFragment, SchedulerMode, ServiceClass,
    UncoveredRequiredWork,
    build_claim_board, reconcile_claim_board,
)
from executor_v0.strip_claim_scheduler import (
    HypotheticalWorkerCoverage,
    claim_runtime_fragment,
    evaluate_hypothetical_worker,
    hypothetical_worker_route,
    schedule_claim_board,
)
from executor_v0.tasks import generate_optional_idle_cleanup_tasks
from executor_v0.strip_market import (
    MarketBootstrapState,
    build_market_turn_plan,
)
from executor_v0.strip_routes import (
    RouteAssignment,
    RoutePhase,
    StripRoute,
    WorkerId,
    _candidate_cost_segment,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
    remaining_day_action_slots,
    route_cursor_invariants_hold,
)
from executor_v0.strip_hiring import (
    StripHiringPlan,
    plan_strip_hiring,
    predict_hire_spawns,
    future_worker_actions,
)
from executor_v0.strip_supply import (
    LOCAL_ACTION_PRIORITY,
    PendingPickup,
    PickupBatch,
    RouteSupplyPlan,
    RouteSupplyState,
    build_route_supply_plans,
    extract_tile_supply_demand,
)
from executor_v0.strip_work import (
    BlockReason,
    StripWorkConfig,
    StripWorkPlan,
    WorkItem,
    WorkStatus,
    build_strip_work_plan,
    row_key_for_tile,
)


_RETAINED_ONE_SHOT_CROPS = frozenset(("WHEAT", "CARROT", "MELON"))

__all__ = [
    "StripExecutorConfig",
    "StripExecutorController",
    "StripExecutorResult",
]


_LOCAL_PRIORITY = LOCAL_ACTION_PRIORITY


@dataclass(frozen=True)
class StripExecutorConfig:
    """Packet 2 routing, Packet 3 removal, and strip bootstrap policy knobs."""

    acting_seat: int = 0
    work_config: StripWorkConfig = field(default_factory=StripWorkConfig)
    shed_capacity: int = 100
    max_market_orders: int = 10
    market_params: Mapping[str, Mapping[str, Any]] | None = None
    aggressive_sell_all: bool = False
    allow_live_crop_sacrifice: bool = False
    allow_productive_recurring_crop_sacrifice: bool = False
    allow_older_crop_sacrifice: bool = False
    enable_row_claim_board: bool = False


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


@dataclass(frozen=True)
class _ClaimHireRecord:
    worker: WorkerId
    spawn: tuple[int, int]
    coverage: HypotheticalWorkerCoverage
    route: StripRoute


class StripExecutorController:
    """Fixed daily ownership and a one-pass deterministic route executor."""

    def __init__(
        self,
        *,
        config: StripExecutorConfig = StripExecutorConfig(),
        work_builder: WorkPlanBuilder = build_strip_work_plan,
        low_telemetry: bool = False,
    ) -> None:
        self.config = config
        self._work_builder = work_builder
        self._low_telemetry = bool(low_telemetry)
        self._day: int | None = None
        self._routes: dict[WorkerId, StripRoute] = {}
        self._assignment: RouteAssignment | None = None
        self._unassigned_ids: tuple[str, ...] = ()
        self._plan: StripWorkPlan | None = None
        self._passed_work: dict[str, dict[str, str]] = {}
        self._supply_plans: dict[str, RouteSupplyPlan] = {}
        self._supply_states: dict[str, RouteSupplyState] = {}
        self._latest_inventories: dict[WorkerId, dict[str, int]] = {}
        self._initial_shed: dict[str, int] = {}
        self._daily: dict[str, Any] = {}
        self._daily_plan: DailyPlan | None = None
        self._market_state = MarketBootstrapState()
        self._routes_finalized = False
        self._bootstrap_stage = "PROCUREMENT"
        self._pending_hires: dict[str, int] | None = None
        self._hire_no_progress = 0
        self._hiring_blocked = False
        self._hire_plan: StripHiringPlan | None = None
        self._candidate_routes = ()
        self._worker_count_before_hiring = 0
        self._hire_submitted = 0
        self._hire_observed = 0
        self._hire_failures = 0
        self._animal_revisit_tiles: dict[str, tuple[int, int]] = {}
        self._observation_for_diagnostics: Mapping[str, Any] = {}
        self._claim_board: ClaimBoard | None = None
        self._claim_timings: dict[str, float] = {}
        self._claim_pass_reasons: dict[str, dict[str, Any]] = {}
        self._claim_hire_records: dict[WorkerId, _ClaimHireRecord] = {}
        self._pending_claim_hires: tuple[_ClaimHireRecord, ...] = ()
        self._claim_schedule_prepared = False
        self._claim_hiring_diagnostics: dict[str, Any] = {}

    @property
    def routes(self) -> tuple[StripRoute, ...]:
        return tuple(sorted(self._routes.values(), key=lambda route: route.route_id))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._result_diagnostics()

    @property
    def uncovered_required_work(self) -> UncoveredRequiredWork | None:
        if self._claim_board is None:
            return None
        return self._claim_board.uncovered_snapshot(
            remaining_day_action_slots(self._observation_for_diagnostics)
        )

    def _finalize_day(
        self,
        obs: Mapping[str, Any],
        plan: DailyPlan,
        work_plan: StripWorkPlan | None = None,
    ) -> StripWorkPlan:
        """Freeze Packet 2 ownership and Packet 3 reservations once."""

        day = int(obs.get("day", 0))
        bootstrap_diagnostics = self._daily.get("hiring_diagnostics")
        hire_stop_reason = self._daily.get("hire_stop_reason")
        if work_plan is None:
            work_plan = self._build_work_plan(obs, plan)
        positions = self._worker_positions(obs)
        inventories = {
            worker: self._worker_inventory(obs, worker) for worker in positions
        }
        if self.config.enable_row_claim_board:
            return self._finalize_claim_day(obs, work_plan, positions, inventories)
        candidates = generate_horizontal_route_candidates(work_plan)
        assignment = assign_horizontal_routes(
            candidates,
            positions,
            assignment_hour=int(obs.get("hour", 0)),
            remaining_action_slots=remaining_day_action_slots(obs),
            worker_action_slots={
                worker: remaining_day_action_slots(obs) for worker in positions
            },
            worker_inventories=inventories,
            shed_stock=((obs.get("private") or {}).get("shed") or {}),
            global_resources=((obs.get("private") or {}).get("seeds") or {}),
        )
        self._day = day
        self._plan = work_plan
        self._assignment = assignment
        self._routes = {route.owner: route for route in assignment.routes}
        self._unassigned_ids = tuple(route.route_id for route in assignment.unassigned)
        self._passed_work = {route.route_id: {} for route in assignment.routes}
        self._latest_inventories = inventories
        self._initial_shed = {
            str(item): max(0, int(amount))
            for item, amount in ((obs.get("private") or {}).get("shed") or {}).items()
        }
        self._supply_plans = {
            route.route_id: supply_plan
            for route, supply_plan in zip(
                assignment.routes,
                build_route_supply_plans(
                    assignment.routes,
                    work_plan,
                    self._latest_inventories,
                    ((obs.get("private") or {}).get("shed") or {}),
                    positions,
                ),
            )
        }
        self._supply_states = {
            route.route_id: RouteSupplyState() for route in assignment.routes
        }
        for route in assignment.routes:
            if not self._supply_plans[route.route_id].requires_pickup:
                route.phase = RoutePhase.TRAVEL_TO_ENTRY
            else:
                route.phase = RoutePhase.PREPARE_SUPPLIES
        if self._low_telemetry:
            self._daily = {}
            return work_plan
        self._daily = {
            "day": day,
            "assignment_hour": int(obs.get("hour", 0)),
            "remaining_slots_at_assignment": remaining_day_action_slots(obs),
            "active_routes": len(candidates),
            "useful_row_count": len(candidates),
            "assigned_routes": len(assignment.routes),
            "unassigned_routes": len(assignment.unassigned),
            "workers": len(positions),
            "large_route_assignment_mode": assignment.large_route_assignment_mode,
            "primary_rows_assigned": assignment.primary_rows_assigned,
            "overflow_rows_assigned": assignment.overflow_rows_assigned,
            "idle_workers_with_unassigned_feasible_rows": (
                assignment.idle_workers_with_unassigned_feasible_rows
            ),
            "unassigned_active_routes": list(self._unassigned_ids),
            "unassigned_supply_demand": {
                candidate.route_id: dict(
                    extract_tile_supply_demand(candidate.owned_tiles, work_plan)
                )
                for candidate in assignment.unassigned
            },
            "route_workload": {
                candidate.route_id: candidate.workload_interactions
                for candidate in candidates
            },
            "route_forecast_effective_interactions": {
                candidate.route_id: candidate.forecasted_workload_interactions
                for candidate in candidates
            },
            "route_forecast_known_continuation_interactions": {
                candidate.route_id: candidate.known_continuation_interactions
                for candidate in candidates
            },
            "packed_rows_per_worker": {
                route.owner.label: [segment.segment_id for segment in route.segments]
                for route in assignment.routes
            },
            "rows_expected_complete_with_n_workers": (
                self._hire_plan.rows_expected_complete_with_n_workers
                if self._hire_plan else len(candidates)
            ),
            "rows_expected_complete_with_n_plus_one_workers": (
                self._hire_plan.rows_expected_complete_with_n_plus_one_workers
                if self._hire_plan else len(candidates)
            ),
            "tileless_unresolved_work": [
                item.id for item in work_plan.items if item.tile is None
            ],
            "supply_diagnostics": self._supply_daily_diagnostics(
                self._initial_shed
            ),
        }
        self._daily.update(
            self._deadline_assignment_diagnostics(
                assignment,
                candidates,
                positions,
                obs,
            )
        )
        if bootstrap_diagnostics is not None:
            self._daily["hiring_diagnostics"] = bootstrap_diagnostics
        if hire_stop_reason is not None:
            self._daily["hire_stop_reason"] = hire_stop_reason
        self._daily["worker_count_before_hiring"] = self._worker_count_before_hiring
        self._daily["worker_count_final"] = len(positions)
        self._daily["candidate_routes"] = [candidate.route_id for candidate in candidates]
        estimates = {
            estimate.route_id: estimate
            for estimate in (self._hire_plan.route_estimates if self._hire_plan else ())
        }
        self._daily["unassigned_hire_driving_routes"] = [
            candidate.route_id
            for candidate in assignment.unassigned
            if estimates.get(candidate.route_id, None)
            and estimates[candidate.route_id].hire_driving
        ]
        self._daily["unassigned_fertilizer_only_routes"] = [
            candidate.route_id
            for candidate in assignment.unassigned
            if estimates.get(candidate.route_id, None)
            and estimates[candidate.route_id].fertilizer_only
        ]
        return work_plan

    def _claim_execution_plan(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan
    ) -> StripWorkPlan:
        """Add current safe cleanup only to opt-in execution work."""

        occupied = {item.tile for item in work_plan.items if item.tile is not None}
        cleanup = generate_optional_idle_cleanup_tasks(
            obs, self.config.acting_seat, mode="weed_water"
        )
        added = tuple(
            WorkItem(
                id=task.key, kind=task.kind, tile=task.tile,
                crop=task.crop, source=task.source,
                row_key=row_key_for_tile(task.tile),
            )
            for task in cleanup
            if task.tile is not None and task.tile not in occupied
        )
        return replace(work_plan, items=work_plan.items + added) if added else work_plan

    def _claim_supply_plan(
        self, route: StripRoute, position: tuple[int, int]
    ) -> RouteSupplyPlan:
        board = self._claim_board
        assert board is not None
        items = tuple(
            item for bundle_id, bundle in board.bundles.items()
            if board.owner_by_bundle.get(bundle_id) == route.owner
            and bundle.tile in route.owned_tiles
            for item in bundle.items
        )
        demand_pairs, order = ordered_inventory_demand(items)
        carried: dict[str, int] = {}
        reserved: dict[str, int] = {}
        for claim in board.reservations.values():
            if claim.worker != route.owner or not all(
                board.bundles[bundle_id].tile in route.owned_tiles
                for bundle_id in claim.bundle_ids
            ):
                continue
            for key, amount in claim.carried:
                carried[key] = carried.get(key, 0) + amount
            for key, amount in claim.shed:
                reserved[key] = reserved.get(key, 0) + amount
        demand = dict(demand_pairs)
        missing = {
            key: max(0, amount - carried.get(key, 0) - reserved.get(key, 0))
            for key, amount in demand_pairs
        }
        sequence = tuple(PickupBatch(key, reserved[key]) for key in order
                         if reserved.get(key, 0) > 0)
        return RouteSupplyPlan(
            route_id=route.route_id, owner=route.owner,
            pickup_tile=nearest_shed_access(position) if sequence else None,
            demand=demand_pairs,
            already_carried=tuple(sorted((k, min(v, demand.get(k, 0)))
                                         for k, v in carried.items() if v > 0)),
            reserved_from_shed=tuple(sorted((k, min(v, demand.get(k, 0)))
                                            for k, v in reserved.items() if v > 0)),
            missing_stock=tuple(sorted((k, v) for k, v in missing.items() if v > 0)),
            capacity_limited=(), pickup_sequence=sequence,
        )

    def _finalize_claim_day(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan,
        positions: Mapping[WorkerId, tuple[int, int]],
        inventories: Mapping[WorkerId, Mapping[str, int]],
    ) -> StripWorkPlan:
        start = perf_counter()
        private = obs.get("private") or {}
        execution_plan = self._claim_execution_plan(obs, work_plan)
        board = build_claim_board(
            execution_plan, inventories, private.get("shed") or {},
            private.get("seeds") or {}, epoch_id=f"DAY:{int(obs.get('day', 0))}",
        )
        built = perf_counter()
        result = schedule_claim_board(
            board, execution_plan, positions, remaining_day_action_slots(obs),
            int(obs.get("hour", 0)), mode=SchedulerMode.NORMAL,
        )
        self._claim_board = board
        self._claim_timings = {
            "board_build": (built - start) * 1000,
            **result.timings_ms,
        }
        self._day = int(obs.get("day", 0))
        self._plan = work_plan
        self._assignment = result.assignment
        self._routes = {route.owner: route for route in result.assignment.routes}
        self._unassigned_ids = ()
        self._passed_work = {route.route_id: {} for route in result.assignment.routes}
        self._latest_inventories = {worker: dict(stock)
                                    for worker, stock in inventories.items()}
        self._initial_shed = dict(board.observed_shed)
        self._supply_plans = {
            route.route_id: self._claim_supply_plan(route, positions[route.owner])
            for route in result.assignment.routes
        }
        self._supply_states = {route.route_id: RouteSupplyState()
                               for route in result.assignment.routes}
        for route in result.assignment.routes:
            route.phase = (RoutePhase.PREPARE_SUPPLIES
                           if self._supply_plans[route.route_id].requires_pickup
                           else RoutePhase.TRAVEL_TO_ENTRY)
        self._daily = {} if self._low_telemetry else {
            "day": self._day,
            "assignment_hour": int(obs.get("hour", 0)),
            "workers": len(positions),
            "assigned_routes": len(result.assignment.routes),
            "row_claim_board": board.diagnostics(),
            "row_claim_timing_ms": self._claim_timings,
            "uncovered_required_snapshot": board.uncovered_snapshot(
                remaining_day_action_slots(obs)
            ).to_json_dict(),
        }
        self._claim_timings["total_finalization"] = (perf_counter() - start) * 1000
        self._claim_schedule_prepared = True
        return work_plan

    def _plan_claim_hires(
        self,
        obs: Mapping[str, Any],
        work_plan: StripWorkPlan,
        positions: Mapping[WorkerId, tuple[int, int]],
    ) -> tuple[tuple[str], ...]:
        """Reserve affordable hires against only uncovered required fragments."""
        board = self._claim_board
        if board is None:
            return ()
        started = perf_counter()
        slots = future_worker_actions(obs)
        snapshot = board.uncovered_snapshot(slots)
        farm = (obs.get("farms") or ())[self.config.acting_seat]
        hands = tuple(farm.get("hands") or ())
        remaining_headcount = max(0, 240 - len(hands))
        order_cap = self._max_market_orders(obs)
        configuration = obs.get("configuration")
        config = configuration if isinstance(configuration, Mapping) else {}
        board_size = max(2, int(config.get("boardSize", 10)))
        current_worker_count = len(positions)
        previous_diagnostics = self._claim_hiring_diagnostics
        previous_workers = list(previous_diagnostics.get("planned_workers", ()))
        previous_costs = list(
            previous_diagnostics.get("sequential_hire_costs", ())
        )
        previous_considered = int(
            previous_diagnostics.get("hypothetical_workers_considered", 0)
        )
        previous_reserved = int(
            previous_diagnostics.get("required_interactions_reserved", 0)
        )
        next_worker_index = max(
            (worker.index for worker in positions), default=-1
        ) + 1
        hires_today = max(0, int(farm.get("hires_today", 0)))
        cost_mult = self._hire_cost_mult(obs)
        cash = float(farm.get("money", 0.0))
        remaining_cash = cash
        existing_positions = [positions[worker] for worker in sorted(positions)]
        planned: list[_ClaimHireRecord] = []
        costs: list[int] = []
        considered = 0
        stop_reason = "NO_REQUIRED_LEFTOVERS"

        limit = min(remaining_headcount, order_cap, len(snapshot.bundle_views))
        if snapshot.fragments:
            stop_reason = "NO_MEANINGFUL_REQUIRED_COVERAGE"
        for hire_index in range(limit):
            spawns = predict_hire_spawns(
                tuple(existing_positions), 1, board_size
            )
            if not spawns:
                stop_reason = "NO_LEGAL_SPAWN"
                break
            spawn = spawns[0]
            worker = WorkerId(next_worker_index + hire_index)
            considered += 1
            coverage = evaluate_hypothetical_worker(snapshot, spawn, slots)
            if not coverage.claimed_fragments or coverage.effective_interactions <= 0:
                stop_reason = (
                    "RESOURCE_BLOCKED_REQUIRED_LEFTOVERS"
                    if "RESOURCE_SHORTAGE" in coverage.rejection_reasons
                    else "NO_MEANINGFUL_REQUIRED_COVERAGE"
                )
                break

            cost = hire_cost(hires_today + len(planned), cost_mult)
            if remaining_cash < cost:
                stop_reason = "CASH"
                break

            route = hypothetical_worker_route(
                board, work_plan, worker, spawn, coverage, slots,
                int(obs.get("hour", 0)),
            )
            if route is None:
                stop_reason = "HORIZON_SHORTAGE"
                break
            route.route_id = f"CLAIM:{worker.label}"
            bundle_ids = tuple(
                bundle_id for fragment in coverage.claimed_fragments
                for bundle_id in fragment.bundle_ids
            )
            reservation = board.trial(worker, bundle_ids)
            if reservation is None:
                stop_reason = "RESOURCE_RESERVATION_MISMATCH"
                break
            if (dict(reservation.shed) != dict(coverage.reservation_shed)
                    or dict(reservation.global_resources)
                    != dict(coverage.reservation_global)):
                stop_reason = "RESOURCE_RESERVATION_MISMATCH"
                break
            if not board.claim(reservation):
                stop_reason = "RESOURCE_RESERVATION_MISMATCH"
                break
            board.record_claim_source(bundle_ids, "HIRE_RESERVATION")

            record = _ClaimHireRecord(worker, spawn, coverage, route)
            planned.append(record)
            costs.append(cost)
            self._claim_hire_records[worker] = record
            self._routes[worker] = route
            self._passed_work[route.route_id] = {}
            supply = self._claim_supply_plan(route, spawn)
            self._supply_plans[route.route_id] = supply
            self._supply_states[route.route_id] = RouteSupplyState()
            route.phase = (RoutePhase.PREPARE_SUPPLIES if supply.requires_pickup
                           else RoutePhase.TRAVEL_TO_ENTRY)
            remaining_cash -= cost
            existing_positions.append(spawn)
            snapshot = board.uncovered_snapshot(slots)
            stop_reason = "NO_REQUIRED_LEFTOVERS"
            if not snapshot.fragments:
                break

        if snapshot.fragments and len(planned) >= order_cap:
            stop_reason = "ORDER_CAP"
        elif snapshot.fragments and len(planned) >= remaining_headcount:
            stop_reason = "WORKER_LIMIT"

        self._pending_claim_hires = tuple(planned)
        planned_workers = [
            {
                "worker": record.worker.label,
                "spawn": list(record.spawn),
                "bundle_ids": [
                    bundle_id for fragment in record.coverage.claimed_fragments
                    for bundle_id in fragment.bundle_ids
                ],
                "effective_interactions": record.coverage.effective_interactions,
                "incremental_turns": record.coverage.incremental_turns,
                "reservation_shed": dict(record.coverage.reservation_shed),
                "reservation_global": dict(record.coverage.reservation_global),
            }
            for record in planned
        ]
        self._claim_hiring_diagnostics = {
            "schema_version": 1,
            "existing_workers": int(previous_diagnostics.get(
                "existing_workers", current_worker_count
            )),
            "hypothetical_workers_considered": previous_considered + considered,
            "wanted_hires": len(previous_workers) + len(planned),
            "submittable_hires": len(previous_workers) + len(planned),
            "submitted_this_round": len(planned),
            "sequential_hire_costs": previous_costs + costs,
            "cash_before_hiring": float(previous_diagnostics.get(
                "cash_before_hiring", cash
            )),
            "cash_after_planned_hiring": remaining_cash,
            "future_action_slots": slots,
            "required_interactions_reserved": previous_reserved + sum(
                record.coverage.effective_interactions for record in planned
            ),
            "hire_stop_reason": stop_reason,
            "planned_workers": previous_workers + planned_workers,
        }
        elapsed = (perf_counter() - started) * 1000
        self._claim_timings["hypothetical_hiring"] = (
            self._claim_timings.get("hypothetical_hiring", 0.0) + elapsed
        )
        self._claim_timings["hypothetical_hires_considered"] = (
            self._claim_timings.get("hypothetical_hires_considered", 0) + considered
        )
        self._claim_timings["total_finalization"] = (
            self._claim_timings.get("total_finalization", 0.0) + elapsed
        )
        if not self._low_telemetry:
            self._daily["claim_hiring"] = self._claim_hiring_diagnostics
            self._daily["row_claim_timing_ms"] = self._claim_timings
        return tuple(("HIRE",) for _ in planned)

    def _reconcile_prepared_claim_schedule(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan
    ) -> StripWorkPlan:
        """Apply an observed bootstrap delta while retaining committed owners."""
        board = self._claim_board
        if board is None:
            return work_plan
        started = perf_counter()
        private = obs.get("private") or {}
        execution_plan = self._claim_execution_plan(obs, work_plan)
        changed_owners = reconcile_claim_board(
            board, execution_plan,
            {worker: self._worker_inventory(obs, worker)
             for worker in self._worker_positions(obs)},
            private.get("shed") or {}, private.get("seeds") or {},
        )
        self._refresh_claim_route_supply(obs, changed_owners)
        self._claim_timings["bootstrap_reconciliation"] = (
            self._claim_timings.get("bootstrap_reconciliation", 0.0)
            + (perf_counter() - started) * 1000
        )
        return work_plan

    def _claim_bootstrap_result(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan,
        market_actions: tuple[tuple, ...],
    ) -> StripExecutorResult:
        """Keep observed workers productive while a hire awaits confirmation."""
        started = perf_counter()
        positions = self._worker_positions(obs)
        execution_plan = self._claim_execution_plan(obs, work_plan)
        self._claim_pass_reasons = {}
        actions: list[tuple] = []
        for worker in sorted(positions):
            route = self._routes.get(worker)
            if route is None:
                route = self._claim_refill(worker, positions[worker], execution_plan, obs)
                if route is None:
                    staging = self._claim_stage(worker, positions[worker], obs)
                    if staging is not None:
                        actions.append(staging)
                        continue
                    self._record_claim_pass(worker, positions[worker], obs, None)
                    actions.append(("PASS",))
                    continue
            action = self._act_worker(route, positions[worker], execution_plan, obs)
            if action == ("PASS",):
                self._record_claim_pass(worker, positions[worker], obs, route)
            actions.append(action)
        self._claim_timings["bootstrap_worker_dispatch"] = (
            self._claim_timings.get("bootstrap_worker_dispatch", 0.0)
            + (perf_counter() - started) * 1000
        )
        # The loop above already follows sorted WorkerId order; keep farmer first.
        ordered = actions
        return StripExecutorResult(
            farmer_action=ordered[0] if ordered else ("PASS",),
            hands_actions=tuple(ordered[1:]),
            market_actions=market_actions,
            diagnostics=self._result_diagnostics(),
        )

    def _finish_prepared_claim_schedule(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan
    ) -> StripWorkPlan:
        """Finalize the already planned claims after hire reconciliation."""
        started = perf_counter()
        positions = self._worker_positions(obs)
        self._latest_inventories = {
            worker: self._worker_inventory(obs, worker) for worker in positions
        }
        self._plan = work_plan
        self._day = int(obs.get("day", 0))
        routes = tuple(sorted(self._routes.values(), key=lambda route: route.route_id))
        self._assignment = RouteAssignment(
            routes, (), tuple(worker for worker in sorted(positions)
                              if worker not in self._routes),
            primary_rows_assigned=sum(bool(route.segments and
                                           ":PRIMARY:" in route.segments[0].segment_id)
                                      for route in routes),
            overflow_rows_assigned=sum(max(0, len(route.segments) - 1)
                                       for route in routes),
        )
        self._unassigned_ids = ()
        private = obs.get("private") or {}
        self._initial_shed = {
            str(item): max(0, int(amount))
            for item, amount in (private.get("shed") or {}).items()
        }
        if not self._low_telemetry:
            self._daily.update({
                "workers": len(positions),
                "assigned_routes": len(routes),
                "row_claim_board": self._claim_board.diagnostics()
                if self._claim_board is not None else {},
                "uncovered_required_snapshot": (
                    self._claim_board.uncovered_snapshot(
                        remaining_day_action_slots(obs)
                    ).to_json_dict()
                    if self._claim_board is not None else {}
                ),
                "claim_hiring": self._claim_hiring_diagnostics,
                "row_claim_pass_reasons": dict(self._claim_pass_reasons),
            })
            self._daily["row_claim_timing_ms"] = self._claim_timings
        elapsed = (perf_counter() - started) * 1000
        self._claim_timings["final_assignment"] = (
            self._claim_timings.get("final_assignment", 0.0) + elapsed
        )
        self._claim_timings["total_finalization"] = (
            self._claim_timings.get("total_finalization", 0.0) + elapsed
        )
        self._claim_schedule_prepared = False
        return work_plan

    def _claim_refill(
        self, worker: WorkerId, position: tuple[int, int],
        work_plan: StripWorkPlan, obs: Mapping[str, Any],
    ) -> StripRoute | None:
        board = self._claim_board
        if board is None:
            return None
        started = perf_counter()
        segment = claim_runtime_fragment(
            board, work_plan, worker, position, remaining_day_action_slots(obs),
            mode=SchedulerMode.NORMAL,
        )
        self._claim_timings["runtime_refill"] = (
            self._claim_timings.get("runtime_refill", 0.0)
            + (perf_counter() - started) * 1000
        )
        if segment is None:
            return None
        traversal = segment.traversal
        route_id = f"CLAIM:{worker.label}:REFILL:{board.observation_version}"
        route = StripRoute(
            route_id, traversal, traversal, worker, traversal[0],
            abs(position[0] - traversal[0][0]) + abs(position[1] - traversal[0][1]),
            int(obs.get("hour", 0)),
            workload_interactions=segment.represented_interactions,
            source_shape="horizontal_claim_segments", segments=(segment,),
        )
        self._routes[worker] = route
        self._passed_work[route_id] = {}
        supply = self._claim_supply_plan(route, position)
        self._supply_plans[route_id] = supply
        self._supply_states[route_id] = RouteSupplyState()
        route.phase = (RoutePhase.PREPARE_SUPPLIES if supply.requires_pickup
                       else RoutePhase.TRAVEL_TO_ENTRY)
        return route

    def _refresh_claim_route_supply(
        self, obs: Mapping[str, Any], changed_owners: set[WorkerId]
    ) -> None:
        positions = self._worker_positions(obs)
        for worker in sorted(changed_owners):
            route = self._routes.get(worker)
            if route is None or worker not in positions:
                continue
            refreshed = self._claim_supply_plan(route, positions[worker])
            old = self._supply_plans.get(route.route_id)
            if old == refreshed:
                continue
            self._supply_plans[route.route_id] = refreshed
            state = self._supply_states.setdefault(route.route_id, RouteSupplyState())
            if state.pending is not None:
                continue
            if refreshed.requires_pickup and (
                old is None or refreshed.reserved_from_shed != old.reserved_from_shed
            ):
                self._supply_states[route.route_id] = RouteSupplyState()
                route.phase = RoutePhase.PREPARE_SUPPLIES
                route.completion_hour = None
            elif not refreshed.requires_pickup and route.phase == RoutePhase.PREPARE_SUPPLIES:
                route.phase = RoutePhase.TRAVEL_TO_ENTRY

    def _claim_stage(
        self, worker: WorkerId, position: tuple[int, int],
        obs: Mapping[str, Any], route: StripRoute | None = None,
    ) -> tuple | None:
        started = perf_counter()
        try:
            return self._claim_stage_target(worker, position, obs, route)
        finally:
            self._claim_timings["staging"] = (
                self._claim_timings.get("staging", 0.0)
                + (perf_counter() - started) * 1000
            )

    def _claim_stage_target(
        self, worker: WorkerId, position: tuple[int, int],
        obs: Mapping[str, Any], route: StripRoute | None,
    ) -> tuple | None:
        board = self._claim_board
        slots = remaining_day_action_slots(obs)
        if board is None or slots <= 1:
            return None

        fragments = board.required_fragments()

        def required_tier(fragment: RowFragment) -> int:
            return int(not any(
                item.status == WorkStatus.READY
                for bundle_id in fragment.bundle_ids
                for item in board.bundles[bundle_id].items
            ))

        targets_by_tier = (
            tuple(fragment for fragment in fragments if required_tier(fragment) == 0),
            tuple(fragment for fragment in fragments if required_tier(fragment) == 1),
        )
        target: tuple[int, int] | None = None
        for candidates in targets_by_tier:
            feasible = []
            for fragment in candidates:
                entry = fragment.traversal[0]
                distance = abs(position[0] - entry[0]) + abs(position[1] - entry[1])
                action_turns = min(
                    board.bundles[bundle_id].effective_interactions
                    for bundle_id in fragment.bundle_ids
                )
                if distance + max(1, action_turns) <= slots:
                    feasible.append((distance, entry, fragment.row_key))
            if feasible:
                target = min(feasible)[1]
                break

        if target is not None and target == position:
            return None

        if target is None:
            supply_work = any(
                bundle.service_class != ServiceClass.OPTIONAL
                and board.phase_by_bundle[bundle.bundle_id] == ClaimPhase.UNCLAIMED
                and any(board.remaining_shed.get(item, 0) > 0
                        for item, _ in bundle.inventory_demand)
                for bundle in board.bundles.values()
            )
            if route is not None:
                supply_plan = self._supply_plans.get(route.route_id)
                supply_state = self._supply_states.get(route.route_id)
                if (supply_plan is not None and supply_plan.requires_pickup
                        and supply_state is not None and supply_state.pending is None):
                    supply_work = supply_work or any(
                        supply_state.acquired.get(item, 0) < quantity
                        for item, quantity in supply_plan.reserved_from_shed
                    )
            if supply_work:
                shed_target = nearest_shed_access(position)
                distance = abs(position[0] - shed_target[0]) + abs(position[1] - shed_target[1])
                if shed_target != position and distance + 1 <= slots:
                    target = shed_target

        if target is None:
            center = (4, 4)
            distance = abs(position[0] - center[0]) + abs(position[1] - center[1])
            if center != position and distance + 1 <= slots:
                target = center

        if target is None or target == position:
            return None
        movement = _vertical_first_step(position, target)
        if movement is not None:
            if route is not None:
                self._record_movement(route, obs)
            self._claim_pass_reasons.pop(worker.label, None)
        return movement

    def _record_claim_pass(
        self, worker: WorkerId, position: tuple[int, int],
        obs: Mapping[str, Any], route: StripRoute | None, *,
        submitted_market_action: bool = False,
    ) -> None:
        board = self._claim_board
        slots = remaining_day_action_slots(obs)
        if board is None:
            if self._pending_hires is not None:
                reason = "AWAITING_OBSERVATION_CONFIRMATION"
            elif self._market_state.pending_buys:
                reason = "PENDING_MARKET_OR_SUPPLY_EFFECT"
            elif submitted_market_action:
                reason = "AWAITING_OBSERVATION_CONFIRMATION"
            elif slots <= 1:
                reason = "TERMINAL_OR_DEADLINE_BOUNDARY"
            else:
                reason = "INVALID_OR_MISSING_WORKER_STATE"
            self._claim_pass_reasons[worker.label] = {
                "reason": reason,
                "owned_candidates": 0,
                "nearby_required_candidates": 0,
                "other_required_candidates": 0,
                "optional_candidates": 0,
                "staging_candidates": 0,
                "scheduler_miss": False,
            }
            return
        required = [
            bundle for bundle_id, bundle in board.bundles.items()
            if board.phase_by_bundle[bundle_id] == ClaimPhase.UNCLAIMED
            and bundle.claimable
            and bundle.service_class != ServiceClass.OPTIONAL
        ]
        nearby = sum(
            abs(position[0] - bundle.tile[0]) + abs(position[1] - bundle.tile[1]) <= 1
            for bundle in required
        )
        optional = len(board.unclaimed(ServiceClass.OPTIONAL))
        feasible = any(
            board.trial(worker, (bundle.bundle_id,)) is not None
            and (abs(position[0] - bundle.tile[0])
                 + abs(position[1] - bundle.tile[1])
                 + bundle.effective_interactions <= slots)
            for bundle in required
        )
        supply = self._supply_states.get(route.route_id) if route is not None else None
        if route is not None and route.phase == RoutePhase.INVALID:
            reason = "INVALID_OR_MISSING_WORKER_STATE"
        elif supply is not None and supply.pending is not None:
            reason = "AWAITING_OBSERVATION_CONFIRMATION"
        elif self._pending_hires is not None:
            reason = "AWAITING_OBSERVATION_CONFIRMATION"
        elif self._market_state.pending_buys:
            reason = "PENDING_MARKET_OR_SUPPLY_EFFECT"
        elif slots <= 1:
            reason = "TERMINAL_OR_DEADLINE_BOUNDARY"
        elif ((required and any(bundle.tile == position for bundle in required))
              or position == (4, 4)):
            reason = "ALREADY_AT_STAGING_TARGET"
        else:
            reason = "NO_LEGAL_REACHABLE_WORK"
        self._claim_pass_reasons[worker.label] = {
            "reason": reason,
            "owned_candidates": sum(owner == worker for owner in
                                    board.owner_by_bundle.values()),
            "nearby_required_candidates": nearby,
            "other_required_candidates": len(required) - nearby,
            "optional_candidates": optional,
            "staging_candidates": sum(bundle.tile != position for bundle in required),
            "scheduler_miss": bool(feasible and reason == "NO_LEGAL_REACHABLE_WORK"),
        }

    def _deadline_assignment_diagnostics(
        self,
        assignment: RouteAssignment,
        candidates,
        positions: Mapping[WorkerId, tuple[int, int]],
        obs: Mapping[str, Any],
    ) -> dict[str, Any]:
        slots = remaining_day_action_slots(obs)
        start_hour = int(obs.get("hour", 0))
        by_id = {candidate.route_id: candidate for candidate in candidates}
        per_worker: dict[str, list[dict[str, Any]]] = {}
        complete_segments = 0
        useful_completed = 0
        useful_missed = 0
        route_costs: dict[str, Any] = {}
        private = obs.get("private") or {}
        remaining_global = (
            {
                str(item): max(0, int(amount))
                for item, amount in private.get("seeds", {}).items()
            }
            if isinstance(private.get("seeds"), Mapping)
            else None
        )
        remaining_unassigned_shed = {
            str(item): max(0, int(amount))
            for item, amount in (private.get("shed") or {}).items()
        }
        for supply_plan in self._supply_plans.values():
            for item, quantity in supply_plan.reserved_from_shed:
                remaining_unassigned_shed[item] = max(
                    0, remaining_unassigned_shed.get(item, 0) - quantity
                )
        for route in assignment.routes:
            entries: list[dict[str, Any]] = []
            route_segments = []
            for segment in route.segments:
                candidate = by_id.get(segment.segment_id) or by_id.get(
                    segment.physical_row_id
                )
                if segment.cost_segment is not None:
                    route_segments.append(segment.cost_segment)
                elif candidate is not None:
                    route_segments.append(
                        _candidate_cost_segment(candidate, segment.traversal)
                    )
            supply_plan = self._supply_plans.get(route.route_id)
            reserved = (
                dict(supply_plan.reserved_from_shed)
                if supply_plan is not None
                else None
            )
            cost = simulate_route_cost(
                positions[route.owner],
                tuple(route_segments),
                carried_inventory=self._latest_inventories.get(route.owner, {}),
                remaining_action_slots=slots,
                assignment_turn=start_hour,
                reserved_supply=reserved,
                global_resources=remaining_global,
                pickup_tile=(
                    supply_plan.pickup_tile if supply_plan is not None else None
                ),
            )
            route_costs[route.route_id] = cost.to_json_dict()
            if remaining_global is not None:
                for item, quantity in cost.global_quantities_consumed:
                    remaining_global[item] = max(
                        0, remaining_global.get(item, 0) - quantity
                    )
            results = {value.segment_id: value for value in cost.segment_results}
            for segment in route.segments:
                segment_cost = results.get(segment.segment_id)
                if segment_cost is None:
                    continue
                complete_segments += int(segment_cost.complete_before_deadline)
                useful_completed += (
                    segment_cost.effective_interactions_completed_before_deadline
                )
                useful_missed += segment_cost.effective_interactions_missed
                entries.append(
                    {
                        "segment_id": segment.segment_id,
                        "estimated_arrival_turn": start_hour
                        + segment_cost.arrival_elapsed_turns,
                        "estimated_completion_turn": start_hour
                        + segment_cost.completion_elapsed_turns,
                        "expected_useful_interactions_completed_before_deadline": (
                            segment_cost.effective_interactions_completed_before_deadline
                        ),
                        "expected_useful_interactions_left_after_deadline": (
                            segment_cost.effective_interactions_missed
                        ),
                        "forecast_effective_interactions": (
                            segment_cost.effective_interaction_turns
                        ),
                        "forecast_known_continuation_interactions": (
                            segment_cost.known_continuation_turns
                        ),
                        "expected_complete_before_deadline": (
                            segment_cost.complete_before_deadline
                        ),
                        "canonical_cost": segment_cost.to_json_dict(),
                    }
                )
            per_worker[route.owner.label] = entries
        for candidate in assignment.unassigned:
            segment = _candidate_cost_segment(candidate, candidate.owned_tiles)
            cost = simulate_route_cost(
                candidate.owned_tiles[0],
                (segment,),
                remaining_action_slots=0,
                assignment_turn=start_hour,
                shed_stock=remaining_unassigned_shed,
                global_resources=remaining_global,
            )
            route_costs[f"UNASSIGNED:{candidate.route_id}"] = cost.to_json_dict()
            useful_missed += cost.effective_interactions_missed
            for item, quantity in cost.supply_quantities_requiring_pickup:
                remaining_unassigned_shed[item] = max(
                    0, remaining_unassigned_shed.get(item, 0) - quantity
                )
            if remaining_global is not None:
                for item, quantity in cost.global_quantities_consumed:
                    remaining_global[item] = max(
                        0, remaining_global.get(item, 0) - quantity
                    )
        return {
            "deadline_route_diagnostics": per_worker,
            "per_worker_segment_sequence": {
                worker: [entry["segment_id"] for entry in entries]
                for worker, entries in per_worker.items()
            },
            "estimated_segment_completion_turns": {
                worker: [entry["estimated_completion_turn"] for entry in entries]
                for worker, entries in per_worker.items()
            },
            "segments_expected_complete_before_deadline": complete_segments,
            "useful_interactions_expected_complete_before_deadline": useful_completed,
            "useful_interactions_expected_missed": useful_missed,
            "expected_useful_interactions_completed": useful_completed,
            "expected_useful_interactions_missed": useful_missed,
            "canonical_route_costs": route_costs,
            "overloaded_rows_detected": assignment.overloaded_rows_detected,
            "row_helpers_required": assignment.overloaded_rows_detected,
            "row_helpers_assigned": assignment.row_helpers_assigned,
            "unresolved_overloaded_rows": assignment.unresolved_overloaded_rows,
            "row_overload_diagnostics": list(assignment.row_diagnostics),
        }

    def reset_day(self, obs: Mapping[str, Any], plan: DailyPlan) -> StripWorkPlan:
        """Begin a two-phase day and return its preliminary Packet 1 forecast.

        This initializes daily strategic/market state and builds the preliminary
        work forecast.  Route ownership and Packet 3 reservations are finalized
        later by :meth:`_finalize_day`, once any bootstrap market procurement has
        been observed (or bounded out); callers that only need the forecast use
        this method, while :meth:`act` drives the full bootstrap.
        """

        self._start_day(obs, plan)
        self._plan = self._build_work_plan(obs, plan)
        return self._plan

    def act(self, obs: Mapping[str, Any], plan: DailyPlan) -> StripExecutorResult:
        """Reconcile bootstrap first, then execute frozen routes plus sell bins."""

        self._observation_for_diagnostics = obs
        day = int(obs.get("day", 0))
        if self._day != day:
            self._start_day(obs, plan)
        if self._daily_plan is None:
            self._daily_plan = plan
        active_plan = self._daily_plan
        work_plan = self._build_work_plan(obs, active_plan)
        self._plan = work_plan
        self._confirm_pending_supply_pickups(obs)
        execution_plan: StripWorkPlan | None = None

        if not self._routes_finalized:
            self._reconcile_market_observation(obs)
            if self._bootstrap_stage == "PROCUREMENT":
                protected = self._bootstrap_reservations(work_plan, obs)
                market_plan = build_market_turn_plan(
                    obs,
                    active_plan,
                    work_plan,
                    self._market_state,
                    acting_seat=self.config.acting_seat,
                    shed_capacity=self._shed_capacity(obs),
                    max_orders=self._max_market_orders(obs),
                    market_params=self._market_params(obs),
                    protected_reservations=protected,
                    aggressive_sell_all=self.config.aggressive_sell_all,
                )
                self._market_state.bootstrap_turns += bool(market_plan.orders)
                self._market_state.pending_buys = market_plan.pending_buys
                self._market_state.latest_diagnostics = market_plan.diagnostics
                if market_plan.orders:
                    return self._bootstrap_pass_result(obs, market_plan.orders)
                self._bootstrap_stage = "HIRING"
                self._worker_count_before_hiring = len(self._worker_positions(obs))

            if self._bootstrap_stage == "HIRING":
                if self.config.enable_row_claim_board:
                    if self._pending_hires is not None:
                        if int(obs.get("step", 0)) <= self._pending_hires["submitted_step"]:
                            return self._bootstrap_pass_result(obs, ())
                        self._reconcile_hire_observation(obs)
                        work_plan = self._reconcile_prepared_claim_schedule(
                            obs, work_plan
                        )
                    positions = self._worker_positions(obs)
                    if not self._claim_schedule_prepared:
                        inventories = {
                            worker: self._worker_inventory(obs, worker)
                            for worker in positions
                        }
                        self._finalize_claim_day(
                            obs, work_plan, positions, inventories
                        )
                    claim_orders: tuple[tuple, ...] = ()
                    if not self._hiring_blocked:
                        claim_orders = self._plan_claim_hires(
                            obs, work_plan, positions
                        )
                    else:
                        self._claim_hiring_diagnostics["hire_stop_reason"] = "FAILED"
                    if claim_orders:
                        farm = obs["farms"][self.config.acting_seat]
                        self._pending_hires = {
                            "submitted_step": int(obs.get("step", 0)),
                            "hands_before": len(farm.get("hands") or ()),
                            "hires_before": int(farm.get("hires_today", 0)),
                            "submitted": len(claim_orders),
                        }
                        self._hire_submitted += len(claim_orders)
                        return self._claim_bootstrap_result(
                            obs, work_plan, claim_orders
                        )
                elif self._pending_hires is not None:
                    if int(obs.get("step", 0)) <= self._pending_hires["submitted_step"]:
                        return self._bootstrap_pass_result(obs, ())
                    # Reconciliation always clears the pending record on this path.
                    self._reconcile_hire_observation(obs)
                if not self.config.enable_row_claim_board and self._hiring_blocked:
                    if not self._low_telemetry:
                        self._daily["hire_stop_reason"] = "FAILED"
                elif not self.config.enable_row_claim_board:
                    candidates = generate_horizontal_route_candidates(work_plan)
                    positions = self._worker_positions(obs)
                    inventories = {
                        worker: self._worker_inventory(obs, worker)
                        for worker in positions
                    }
                    self._candidate_routes = candidates
                    self._hire_plan = plan_strip_hiring(
                        obs,
                        work_plan,
                        candidates,
                        positions,
                        inventories,
                        acting_seat=self.config.acting_seat,
                        max_orders=self._max_market_orders(obs),
                        farm_hand_cost_mult=self._hire_cost_mult(obs),
                    )
                    if not self._low_telemetry:
                        self._daily["hiring_diagnostics"] = (
                            self._hire_plan.to_json_dict())
                        self._daily["hire_stop_reason"] = (
                            self._hire_plan.stop_reason.value)
                    if self._hire_plan.orders:
                        submitted = len(self._hire_plan.orders)
                        farm = obs["farms"][self.config.acting_seat]
                        self._pending_hires = {
                            "submitted_step": int(obs.get("step", 0)),
                            "hands_before": len(farm.get("hands") or ()),
                            "hires_before": int(farm.get("hires_today", 0)),
                            "submitted": submitted,
                        }
                        self._hire_submitted += submitted
                        return self._bootstrap_work_result(
                            obs, work_plan, self._hire_plan.orders
                        )
                self._bootstrap_stage = "FINALIZED"
            if self.config.enable_row_claim_board and self._claim_schedule_prepared:
                work_plan = self._finish_prepared_claim_schedule(obs, work_plan)
            else:
                work_plan = self._finalize_day(
                    obs, active_plan, work_plan=work_plan
                )
            self._routes_finalized = True
            self._market_state.finalized_hour = int(obs.get("hour", 0))
        else:
            # Already finalized: still reconcile each new observation exactly once.
            self._reconcile_market_observation(obs)
            if self.config.enable_row_claim_board and self._claim_board is not None:
                private = obs.get("private") or {}
                execution_plan = self._claim_execution_plan(obs, work_plan)
                changed_owners = reconcile_claim_board(
                    self._claim_board, execution_plan,
                    {worker: self._worker_inventory(obs, worker)
                     for worker in self._worker_positions(obs)},
                    private.get("shed") or {}, private.get("seeds") or {},
                )
                self._refresh_claim_route_supply(obs, changed_owners)
            else:
                self._refresh_route_supply_plans(obs, work_plan)
        protected = self._observed_reservations(
            obs, self._outstanding_reservations()
        )
        market_plan = build_market_turn_plan(
            obs,
            active_plan,
            work_plan,
            self._market_state,
            acting_seat=self.config.acting_seat,
            shed_capacity=self._shed_capacity(obs),
            max_orders=self._max_market_orders(obs),
            market_params=self._market_params(obs),
            protected_reservations=protected,
            purchases_enabled=False,
            retry_animal_purchases=True,
            retry_replacement_seed_purchases=True,
            aggressive_sell_all=self.config.aggressive_sell_all,
        )
        self._market_state.latest_diagnostics = market_plan.diagnostics

        if any(order and str(order[0]).startswith("BUY_") for order in market_plan.orders):
            # Wait for the authoritative purchase observation before allowing
            # routes to consume a newly refreshed PLACE successor.
            return self._bootstrap_pass_result(obs, market_plan.orders)

        positions = self._worker_positions(obs)
        self._latest_inventories = {
            worker: self._worker_inventory(obs, worker) for worker in positions
        }
        for worker, route in self._routes.items():
            if worker not in positions and route.phase not in (
                RoutePhase.DONE,
                RoutePhase.INVALID,
            ):
                route.phase = RoutePhase.INVALID
        actions: list[tuple] = []
        if self.config.enable_row_claim_board:
            self._claim_pass_reasons = {}
        if execution_plan is None:
            execution_plan = (
                self._claim_execution_plan(obs, work_plan)
                if self.config.enable_row_claim_board else work_plan
            )
        for worker in sorted(positions):
            route = self._routes.get(worker)
            if route is None:
                if self.config.enable_row_claim_board:
                    route = self._claim_refill(worker, positions[worker], execution_plan, obs)
                if route is None:
                    if self.config.enable_row_claim_board:
                        staging = self._claim_stage(worker, positions[worker], obs)
                        if staging is not None:
                            actions.append(staging)
                            continue
                        self._record_claim_pass(worker, positions[worker], obs, None)
                    actions.append(("PASS",))
                    continue
            action = self._act_worker(route, positions[worker], execution_plan, obs)
            if self.config.enable_row_claim_board and action == ("PASS",):
                self._record_claim_pass(worker, positions[worker], obs, route)
            actions.append(action)

        farmer_action = actions[0] if actions else ("PASS",)
        hands_actions = tuple(actions[1:])
        return StripExecutorResult(
            farmer_action=farmer_action,
            hands_actions=hands_actions,
            market_actions=market_plan.orders,
            diagnostics=self._result_diagnostics(),
        )

    next_worker_actions = act

    def _start_day(self, obs: Mapping[str, Any], plan: DailyPlan) -> None:
        self._day = int(obs.get("day", 0))
        self._daily_plan = plan
        self._routes = {}
        self._assignment = None
        self._unassigned_ids = ()
        self._passed_work = {}
        self._supply_plans = {}
        self._supply_states = {}
        self._latest_inventories = {}
        self._initial_shed = {}
        self._market_state = MarketBootstrapState()
        self._routes_finalized = False
        self._bootstrap_stage = "PROCUREMENT"
        self._pending_hires = None
        self._hire_no_progress = 0
        self._hiring_blocked = False
        self._hire_plan = None
        self._candidate_routes = ()
        self._worker_count_before_hiring = 0
        self._hire_submitted = 0
        self._hire_observed = 0
        self._hire_failures = 0
        self._animal_revisit_tiles = {}
        self._observation_for_diagnostics = obs
        self._claim_board = None
        self._claim_timings = {}
        self._claim_pass_reasons = {}
        self._claim_hire_records = {}
        self._pending_claim_hires = ()
        self._claim_schedule_prepared = False
        self._claim_hiring_diagnostics = {}
        self._daily = (
            {"day": self._day, "assignment_hour": int(obs.get("hour", 0))}
            if not self._low_telemetry else {}
        )

    def _shed_capacity(self, obs: Mapping[str, Any]) -> int:
        configuration = obs.get("configuration")
        if isinstance(configuration, Mapping):
            return max(1, int(configuration.get("shedCapacity", self.config.shed_capacity)))
        return max(1, self.config.shed_capacity)

    def _max_market_orders(self, obs: Mapping[str, Any]) -> int:
        configuration = obs.get("configuration")
        if isinstance(configuration, Mapping):
            return max(1, int(configuration.get("maxMarketOrdersPerTurn", self.config.max_market_orders)))
        return max(1, self.config.max_market_orders)

    def _hire_cost_mult(self, obs: Mapping[str, Any]) -> int:
        configuration = obs.get("configuration")
        if isinstance(configuration, Mapping):
            return max(0, int(configuration.get("farmHandCostMult", 1)))
        return FARM_HAND_COST_MULT_DEFAULT

    def _market_params(self, obs: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]] | None:
        if self.config.market_params is not None:
            return self.config.market_params
        market = obs.get("market")
        params = market.get("params") if isinstance(market, Mapping) else None
        return params if isinstance(params, Mapping) else None

    def _pass_result(self, obs: Mapping[str, Any]) -> StripExecutorResult:
        return self._bootstrap_pass_result(
            obs,
            tuple(
                tuple(order)
                for order in self._market_state.latest_diagnostics.get("market_orders", ())
            ),
        )

    def _bootstrap_pass_result(
        self, obs: Mapping[str, Any], market_actions: tuple[tuple, ...] | tuple
    ) -> StripExecutorResult:
        positions = self._worker_positions(obs)
        if self.config.enable_row_claim_board:
            self._claim_pass_reasons = {}
            for worker in sorted(positions):
                self._record_claim_pass(
                    worker, positions[worker], obs, self._routes.get(worker),
                    submitted_market_action=bool(market_actions),
                )
        actions = tuple(("PASS",) for _ in sorted(positions))
        return StripExecutorResult(
            farmer_action=actions[0] if actions else ("PASS",),
            hands_actions=actions[1:],
            market_actions=tuple(tuple(order) for order in market_actions),
            diagnostics=self._result_diagnostics(),
        )

    def _bootstrap_work_result(
        self,
        obs: Mapping[str, Any],
        work_plan: StripWorkPlan,
        market_actions: tuple[tuple, ...] | tuple,
    ) -> StripExecutorResult:
        """Let only currently observed workers make one safe provisional step."""

        positions = self._worker_positions(obs)
        assignment = assign_horizontal_routes(
            generate_horizontal_route_candidates(work_plan),
            positions,
            assignment_hour=int(obs.get("hour", 0)),
            remaining_action_slots=remaining_day_action_slots(obs),
            worker_action_slots={
                worker: remaining_day_action_slots(obs) for worker in positions
            },
            worker_inventories={
                worker: self._worker_inventory(obs, worker) for worker in positions
            },
            shed_stock=((obs.get("private") or {}).get("shed") or {}),
            global_resources=((obs.get("private") or {}).get("seeds") or {}),
        )
        routes = {route.owner: route for route in assignment.routes}
        items_by_tile: dict[tuple[int, int], list[WorkItem]] = {}
        for item in work_plan.items:
            if item.tile is not None and item.kind != "DIG":
                items_by_tile.setdefault(item.tile, []).append(item)
        local_items_by_tile = {
            tile: tuple(items) for tile, items in items_by_tile.items()
        }
        actions = tuple(
            self._bootstrap_worker_action(
                routes.get(worker),
                positions[worker],
                local_items_by_tile,
                work_plan,
                obs,
            )
            for worker in sorted(positions)
        )
        diagnostics = self._result_diagnostics()
        if not self._low_telemetry:
            diagnostics.update({
                "assigned_routes": len(assignment.routes),
                "unassigned_routes": len(assignment.unassigned),
                "idle_workers": [
                    worker.label for worker in assignment.idle_workers
                ],
                "packed_rows_per_worker": {
                    route.owner.label: [
                        segment.segment_id for segment in route.segments
                    ]
                    for route in assignment.routes
                },
                "large_route_assignment_mode": (
                    assignment.large_route_assignment_mode
                ),
                "primary_rows_assigned": assignment.primary_rows_assigned,
                "overflow_rows_assigned": assignment.overflow_rows_assigned,
                "idle_workers_with_unassigned_feasible_rows": (
                    assignment.idle_workers_with_unassigned_feasible_rows
                ),
                "row_overload_diagnostics": list(assignment.row_diagnostics),
                "overloaded_rows_detected": assignment.overloaded_rows_detected,
                "row_helpers_required": assignment.overloaded_rows_detected,
                "row_helpers_assigned": assignment.row_helpers_assigned,
                "unresolved_overloaded_rows": assignment.unresolved_overloaded_rows,
            })
        return StripExecutorResult(
            farmer_action=actions[0] if actions else ("PASS",),
            hands_actions=actions[1:],
            market_actions=tuple(tuple(order) for order in market_actions),
            diagnostics=diagnostics,
        )

    def _bootstrap_worker_action(
        self,
        route: StripRoute | None,
        position: tuple[int, int],
        items_by_tile: Mapping[tuple[int, int], tuple[WorkItem, ...]],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> tuple:
        """Perform at most one reachable bootstrap action without ownership.

        The distance check only guards this proposed action; it does not score
        the route's total duration, which is owned by the canonical simulator.
        """

        if route is None:
            return ("PASS",)
        inventory = self._worker_inventory(obs, route.owner)
        slots = remaining_day_action_slots(obs)
        for tile in route.traversal:
            item = self._select_local_item(items_by_tile.get(tile, ()), inventory)
            if item is None:
                continue
            if item.source == "optional_deferrable" and self._optional_would_starve_tail(
                route, tile, work_plan, obs
            ):
                continue
            distance = abs(position[0] - tile[0]) + abs(position[1] - tile[1])
            if distance + 1 > slots:
                return ("PASS",)
            if position != tile:
                return _vertical_first_step(position, tile) or ("PASS",)
            return _interaction_action(item) or ("PASS",)
        return ("PASS",)

    def _reconcile_hire_observation(self, obs: Mapping[str, Any]) -> None:
        pending = self._pending_hires
        if pending is None or int(obs.get("step", 0)) <= pending["submitted_step"]:
            return
        farm = obs["farms"][self.config.acting_seat]
        hands_delta = max(0, len(farm.get("hands") or ()) - pending["hands_before"])
        hires_delta = max(0, int(farm.get("hires_today", 0)) - pending["hires_before"])
        observed = min(pending["submitted"], hands_delta, hires_delta)
        self._hire_observed += observed
        self._hire_failures += max(0, pending["submitted"] - observed)
        if self.config.enable_row_claim_board and self._pending_claim_hires:
            positions = self._worker_positions(obs)
            spawn_mismatches = []
            rejected_workers = set()
            for index, record in enumerate(self._pending_claim_hires):
                confirmed_at_expected_spawn = (
                    index < observed and positions.get(record.worker) == record.spawn
                )
                if confirmed_at_expected_spawn:
                    continue
                rejected_workers.add(record.worker.label)
                if index < observed:
                    spawn_mismatches.append(record.worker.label)
                board = self._claim_board
                if board is not None:
                    for bundle_id in record.coverage.claimed_fragments:
                        for claimed_id in bundle_id.bundle_ids:
                            if board.owner_by_bundle.get(claimed_id) == record.worker:
                                board.release(claimed_id)
                self._routes.pop(record.worker, None)
                self._supply_plans.pop(record.route.route_id, None)
                self._supply_states.pop(record.route.route_id, None)
                self._passed_work.pop(record.route.route_id, None)
                self._claim_hire_records.pop(record.worker, None)
            self._pending_claim_hires = ()
            if rejected_workers:
                planned_workers = list(
                    self._claim_hiring_diagnostics.get("planned_workers", ())
                )
                costs = list(
                    self._claim_hiring_diagnostics.get("sequential_hire_costs", ())
                )
                rejected_interactions = 0
                retained_workers = []
                retained_costs = []
                for index, worker_record in enumerate(planned_workers):
                    if worker_record.get("worker") in rejected_workers:
                        rejected_interactions += int(
                            worker_record.get("effective_interactions", 0)
                        )
                        continue
                    retained_workers.append(worker_record)
                    if index < len(costs):
                        retained_costs.append(costs[index])
                self._claim_hiring_diagnostics["planned_workers"] = retained_workers
                self._claim_hiring_diagnostics["sequential_hire_costs"] = retained_costs
                self._claim_hiring_diagnostics["wanted_hires"] = len(retained_workers)
                self._claim_hiring_diagnostics["submittable_hires"] = len(
                    retained_workers
                )
                self._claim_hiring_diagnostics[
                    "required_interactions_reserved"
                ] = max(
                    0,
                    int(self._claim_hiring_diagnostics.get(
                        "required_interactions_reserved", 0
                    )) - rejected_interactions,
                )
            if spawn_mismatches:
                self._claim_hiring_diagnostics["spawn_mismatches"] = spawn_mismatches
                self._claim_hiring_diagnostics["runtime_refill_required"] = True
        self._pending_hires = None
        if observed:
            self._hire_no_progress = 0
        else:
            self._hire_no_progress += 1
            if self._hire_no_progress >= 2:
                self._hiring_blocked = True
        if not self._low_telemetry:
            self._daily["hires_observed"] = self._hire_observed
            self._daily["failed_hires"] = self._hire_failures

    def _reconcile_market_observation(self, obs: Mapping[str, Any]) -> None:
        """Confirm pending buys from observed deltas, with two no-progress tries."""

        if not self._market_state.pending_buys:
            return
        farm = (obs.get("farms") or ())[self.config.acting_seat]
        private = obs.get("private") or {}
        shed = private.get("shed") or {}
        seeds = private.get("seeds") or {}
        unlocked = set(farm.get("unlocked_quadrants") or ())
        for pending in self._market_state.pending_buys:
            if int(obs.get("step", 0)) <= pending.submitted_step:
                continue
            if pending.kind == "BUY_SEED":
                now = int(seeds.get(pending.item, 0))
            elif pending.kind == "BUY_LAND":
                now = int(pending.item in unlocked)
            else:
                now = int(shed.get(pending.item, 0))
            gained = max(0, now - pending.observed_before + pending.same_item_sold)
            realized = min(pending.quantity, gained)
            if realized:
                self._market_state.buy_observed[pending.key] = (
                    self._market_state.buy_observed.get(pending.key, 0) + realized
                )
                self._market_state.no_progress_counts.pop(pending.key, None)
                continue
            attempts = self._market_state.no_progress_counts.get(pending.key, 0) + 1
            self._market_state.no_progress_counts[pending.key] = attempts
            if attempts >= 2:
                self._market_state.failed_intents.add(pending.key)
        self._market_state.pending_buys = ()

    def _confirm_pending_supply_pickups(self, obs: Mapping[str, Any]) -> None:
        """Apply observation-confirmed Packet 3 pickup progress before SELL."""

        for route_id, supply_state in self._supply_states.items():
            pending = supply_state.pending
            if pending is None:
                continue
            inventory = self._worker_inventory(obs, self._supply_plans[route_id].owner)
            gained = max(0, int(inventory.get(pending.item, 0)) - pending.inventory_before)
            acquired = min(pending.quantity, gained)
            if acquired:
                supply_state.acquired[pending.item] = (
                    supply_state.acquired.get(pending.item, 0) + acquired
                )
                if self._claim_board is not None:
                    self._claim_board.confirm_pickup(
                        self._supply_plans[route_id].owner, pending.item, acquired
                    )
            remaining = pending.quantity - acquired
            supply_state.pending = None
            if remaining:
                attempts = supply_state.attempts.get(pending.item, 0)
                shed = (obs.get("private") or {}).get("shed") or {}
                if not acquired or attempts >= 2 or int(shed.get(pending.item, 0)) <= 0:
                    supply_state.failed_or_unfulfilled[pending.item] = (
                        supply_state.failed_or_unfulfilled.get(pending.item, 0) + remaining
                    )
                    if self._claim_board is not None:
                        self._claim_board.release_failed_pickup(
                            self._supply_plans[route_id].owner, pending.item, remaining
                        )

    def _bootstrap_reservations(
        self, work_plan: StripWorkPlan, obs: Mapping[str, Any]
    ) -> dict[str, int]:
        """Protect committed inventory demand before routes are finalized.

        Procurement can span observations.  Until the market order completes,
        route plans do not exist yet, so the observed purchase would otherwise
        be visible to aggressive selling without an owner.
        """

        required: dict[str, int] = {}
        for work_item in work_plan.items:
            for requirement in work_item.required_supplies:
                if requirement.scope != "inventory" or requirement.quantity <= 0:
                    continue
                required[requirement.item] = (
                    required.get(requirement.item, 0) + requirement.quantity
                )
        acquired: dict[str, int] = {}
        for worker in self._worker_positions(obs):
            for item, amount in self._worker_inventory(obs, worker).items():
                acquired[item] = acquired.get(item, 0) + amount
        return self._observed_reservations(
            obs,
            {
                item: max(0, amount - acquired.get(item, 0))
                for item, amount in required.items()
            },
        )

    def _observed_reservations(
        self, obs: Mapping[str, Any], reservations: Mapping[str, int]
    ) -> dict[str, int]:
        """Limit shed protection to stock observed on this observation."""

        shed = ((obs.get("private") or {}).get("shed") or {})
        protected: dict[str, int] = {}
        for item, amount in reservations.items():
            quantity = min(max(0, int(amount)), max(0, int(shed.get(item, 0))))
            if quantity:
                protected[item] = quantity
        return protected

    def _outstanding_reservations(self) -> dict[str, int]:
        protected: dict[str, int] = {}
        routes_by_id = {route.route_id: route for route in self._routes.values()}
        for route_id, plan in self._supply_plans.items():
            route = routes_by_id.get(route_id)
            if self.config.enable_row_claim_board and route is None:
                continue
            if route is not None and route.phase == RoutePhase.INVALID:
                continue
            state = self._supply_states.get(route_id, RouteSupplyState())
            carried = dict(plan.already_carried)
            for item, required in plan.demand:
                outstanding = max(
                    0,
                    required
                    - carried.get(item, 0)
                    - state.acquired.get(item, 0)
                    - state.failed_or_unfulfilled.get(item, 0),
                )
                if outstanding:
                    protected[item] = protected.get(item, 0) + outstanding
        return protected

    def _build_work_plan(
        self, obs: Mapping[str, Any], plan: DailyPlan
    ) -> StripWorkPlan:
        preferred_crop_slots: dict[str, tuple[tuple[int, int], ...]] = {}
        for route in self._routes.values():
            if (
                route.continuation_source == "retained_crop_maintenance"
                and route.continuation_crop in _RETAINED_ONE_SHOT_CROPS
                and route.continuation_tile is not None
                and route.continuation_next_kind == "PLANT"
            ):
                preferred_crop_slots.setdefault(route.continuation_crop, ())
                preferred_crop_slots[route.continuation_crop] += (
                    route.continuation_tile,
                )
        return self._work_builder(
            obs,
            plan,
            config=self.config.work_config,
            acting_seat=self.config.acting_seat,
            preferred_crop_slots=preferred_crop_slots,
            allow_live_crop_sacrifice=self.config.allow_live_crop_sacrifice,
            allow_productive_recurring_crop_sacrifice=(
                self.config.allow_productive_recurring_crop_sacrifice),
            allow_older_crop_sacrifice=self.config.allow_older_crop_sacrifice,
        )

    def _result_diagnostics(self) -> dict[str, Any]:
        if self._low_telemetry:
            payload = {
                "schema_version": 1,
                "telemetry_mode": "reduced",
                "diagnostics_reduced": True,
            }
            if self.config.enable_row_claim_board and self._claim_board is not None:
                payload["row_claim_board"] = self._claim_board.diagnostics()
                payload["row_claim_pass_reasons"] = dict(self._claim_pass_reasons)
            return payload
        return self._diagnostics()

    def _refresh_route_supply_plans(
        self, obs: Mapping[str, Any], work_plan: StripWorkPlan
    ) -> None:
        """Refresh reservations when observation creates new inventory work.

        A purchase is not part of the route forecast until it is observed in
        the shed.  The corresponding PLACE then gains an inventory
        requirement, so the frozen Packet 3 ledger must acquire that new
        demand without discarding already confirmed pickup progress.
        """
        if not self._routes:
            return
        positions = self._worker_positions(obs)
        inventories = {
            worker: self._worker_inventory(obs, worker) for worker in positions
        }
        plans = build_route_supply_plans(
            self._routes.values(),
            work_plan,
            inventories,
            ((obs.get("private") or {}).get("shed") or {}),
            positions,
        )
        for route, refreshed in zip(self._routes.values(), plans):
            old = self._supply_plans.get(route.route_id)
            old_demand = dict(old.demand) if old is not None else {}
            new_demand = dict(refreshed.demand)
            old_requires_pickup = old.requires_pickup if old is not None else False
            added = any(
                new_demand.get(item, 0) > old_demand.get(item, 0)
                for item in new_demand
            )
            pickup_became_available = (
                refreshed.requires_pickup and not old_requires_pickup
            )
            self._supply_plans[route.route_id] = refreshed
            if not (added or pickup_became_available) or not refreshed.requires_pickup:
                continue
            state = self._supply_states.setdefault(
                route.route_id, RouteSupplyState()
            )
            if state.pending is None:
                route.phase = RoutePhase.PREPARE_SUPPLIES
                route.completion_hour = None

    def _schedule_animal_revisit(
        self, route: StripRoute, work_plan: StripWorkPlan
    ) -> None:
        """Schedule at most one bounded revisit for a late PLACE successor."""
        if route.route_id in self._animal_revisit_tiles:
            return
        passed = self._passed_work.get(route.route_id, {})
        ready = sorted(
            (
                item
                for item in work_plan.items
                if item.kind == "PLACE"
                and item.status == WorkStatus.READY
                and item.id in route.late_work_ids
                and item.id in passed
            ),
            key=lambda item: item.id,
        )
        if ready:
            self._animal_revisit_tiles[route.route_id] = ready[0].tile

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
        supply_plan = self._supply_plans.get(route.route_id)
        supply_state = self._supply_states.setdefault(
            route.route_id, RouteSupplyState()
        )
        if route.phase == RoutePhase.PREPARE_SUPPLIES and supply_plan is not None:
            action = self._prepare_supplies(
                route, position, supply_plan, supply_state, obs, work_plan
            )
            if action is not None:
                return action
        # Late-work observation runs every turn, including after DONE, so work
        # missed by the one-pass sweep stays visible without reopening the route.
        self._record_late_work(route, work_plan)
        self._schedule_animal_revisit(route, work_plan)
        revisit_tile = self._animal_revisit_tiles.get(route.route_id)
        if revisit_tile is not None:
            if position != revisit_tile:
                movement = _vertical_first_step(position, revisit_tile)
                if movement is not None:
                    self._record_movement(route, obs)
                    return movement
                route.blocked_local_work["ANIMAL_REVISIT_BLOCKED"] = (
                    route.blocked_local_work.get("ANIMAL_REVISIT_BLOCKED", 0) + 1
                )
                return ("PASS",)
            action = self._try_local_action(route, revisit_tile, work_plan, obs)
            if action is not None and action[0] == "PLACE":
                route.late_work_ids.discard(
                    next(
                        item.id
                        for item in work_plan.items
                        if item.kind == "PLACE"
                        and item.tile == revisit_tile
                        and item.status == WorkStatus.READY
                    )
                )
                self._animal_revisit_tiles.pop(route.route_id, None)
                return action
            if action is not None:
                return action
        self._reopen_crop_continuation(route, position, work_plan)
        if (
            self.config.enable_row_claim_board
            and self._claim_board is not None
            and route.phase != RoutePhase.DONE
            and route.continuation_item_id is None
            and not any(
                self._claim_board.owner_by_bundle.get(
                    f"TILE:{tile[0]},{tile[1]}"
                ) == route.owner
                for tile in route.traversal[route.cursor:]
            )
        ):
            route.phase = RoutePhase.DONE
            route.completion_hour = hour
        if route.phase == RoutePhase.DONE:
            if self.config.enable_row_claim_board:
                if (
                    route.continuation_item_id is not None
                    and route.continuation_tile == position == route.current_tile
                ):
                    action = self._try_local_action(route, position, work_plan, obs)
                    if action is not None:
                        return action
                refill = self._claim_refill(route.owner, position, work_plan, obs)
                if refill is not None:
                    return self._act_worker(refill, position, work_plan, obs)
                staging = self._claim_stage(route.owner, position, obs, route)
                if staging is not None:
                    return staging
                route.pass_turns_after_completion += 1
                return ("PASS",)
            if self._help_untouched_segment(
                route, work_plan, position=position, obs=obs
            ):
                route.completion_hour = None
            else:
                if (
                    route.continuation_item_id is not None
                    and route.continuation_tile == position == route.current_tile
                ):
                    action = self._try_local_action(route, position, work_plan, obs)
                    if action is not None:
                        return action
                route.pass_turns_after_completion += 1
                return ("PASS",)

        if route.pending_cursor is not None:
            expected = route.traversal[route.pending_cursor]
            if position != expected:
                # Departure is not confirmed yet.  If we are still standing on
                # the tile we tried to leave, local work that appeared in the
                # meantime still belongs to this worker and must be handled
                # before moving on; the tile is not passed until we observe
                # ourselves elsewhere.
                if position == route.current_tile:
                    action = self._try_local_action(route, position, work_plan, obs)
                    if action is not None:
                        return action
                movement = _vertical_first_step(position, expected)
                if movement is None:
                    route.blocked_local_work["ROUTE_BLOCKED"] = (
                        route.blocked_local_work.get("ROUTE_BLOCKED", 0) + 1
                    )
                    return ("PASS",)
                if not self._route_can_reach_useful_action(
                    route, position, work_plan, obs, expected
                ):
                    route.blocked_local_work["DEADLINE_UNREACHABLE"] = (
                        route.blocked_local_work.get("DEADLINE_UNREACHABLE", 0) + 1
                    )
                    return ("PASS",)
                self._record_movement(route, obs)
                return movement
            departing = route.current_tile
            route.cursor = route.pending_cursor
            route.pending_cursor = None
            route.phase = RoutePhase.SWEEP
            self._mark_tile_passed(route, departing)

        # Departure confirmation advances the cursor first; only then can the
        # bounded one-hop repair inspect the immediately previous tile.
        self._reopen_crop_continuation(route, position, work_plan)

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
                if not self._route_can_reach_useful_action(
                    route, position, work_plan, obs, target
                ):
                    route.blocked_local_work["DEADLINE_UNREACHABLE"] = (
                        route.blocked_local_work.get("DEADLINE_UNREACHABLE", 0) + 1
                    )
                    return ("PASS",)
                self._record_movement(route, obs)
                return movement

        action = self._try_local_action(route, target, work_plan, obs)
        if action is not None:
            return action

        inventory = self._worker_inventory(obs, route.owner)
        local_items = tuple(item for item in work_plan.items if item.tile == target)
        self._record_skipped_work(route, local_items, inventory)
        self._snapshot_tile_work(route, target, local_items)
        if route.cursor + 1 >= len(route.traversal):
            # The final tile is processed: mark it passed so late work on it
            # stays diagnosable after completion.
            self._mark_tile_passed(route, target)
            route.phase = RoutePhase.DONE
            route.completion_hour = hour
            if supply_plan is not None:
                supply_state.remaining_at_completion = {
                    item: int(inventory.get(item, 0))
                    for item, _ in supply_plan.demand
                    if int(inventory.get(item, 0)) > 0
                }
            if self.config.enable_row_claim_board:
                refill = self._claim_refill(route.owner, position, work_plan, obs)
                if refill is not None:
                    return self._act_worker(refill, position, work_plan, obs)
                staging = self._claim_stage(route.owner, position, obs, route)
                if staging is not None:
                    return staging
            return ("PASS",)
        route.pending_cursor = route.cursor + 1
        return _vertical_first_step(position, route.traversal[route.pending_cursor]) or (
            "PASS",
        )

    def _reopen_crop_continuation(
        self,
        route: StripRoute,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
    ) -> None:
        """Reopen only an immediately previous tile for a crop successor.

        A one-pass route remains the default.  Reopening is permitted only
        when this worker performed DIG/HARVEST/PLANT, the next observation
        makes its direct PLANT/WATER successor READY, and the worker is still
        on the next tile of the same owned traversal.  This keeps late
        unrelated work diagnostic-only and prevents arbitrary backtracking.
        """
        item_id = route.continuation_item_id
        if item_id is None:
            return
        if route.phase == RoutePhase.DONE:
            if (
                route.continuation_tile is None
                or position != route.current_tile
                or route.continuation_tile != route.current_tile
            ):
                return
            previous_tile = route.current_tile
        else:
            if route.phase != RoutePhase.SWEEP:
                return
            if route.pending_cursor is not None or route.cursor <= 0:
                return
            if position != route.current_tile:
                return
            previous_tile = route.traversal[route.cursor - 1]
            if route.continuation_tile != previous_tile:
                return
            if previous_tile not in route.passed_tiles:
                return
        successors = tuple(
            item
            for item in work_plan.items
            if item.tile == previous_tile
            and item.kind == route.continuation_next_kind
        )
        ready_successors = tuple(
            item for item in successors if item.status == WorkStatus.READY
        )
        if not ready_successors:
            if successors:
                reason = successors[0].block_reason.value if successors[0].block_reason else successors[0].status.value
                route.continuation_blocked_reason = reason
                route.continuation_status = f"BLOCKED:{reason}"
                route.blocked_local_work[f"CONTINUATION_BLOCKED:{reason}"] = (
                    route.blocked_local_work.get(f"CONTINUATION_BLOCKED:{reason}", 0) + 1
                )
            elif route.continuation_status == "HARVEST_RETAINED":
                requested = work_plan.diagnostics.requested_crop_delta_dict
                if requested.get(route.continuation_crop or "", 0) <= 0:
                    route.continuation_status = "TARGET_REDUCED"
                    route.continuation_item_id = None
                    route.continuation_tile = None
                    route.continuation_next_kind = None
                else:
                    route.continuation_blocked_reason = "NO_LEGAL_SLOT"
                    route.continuation_status = "BLOCKED:NO_LEGAL_SLOT"
                    route.continuation_item_id = None
                    route.continuation_next_kind = None
            return
        if route.phase != RoutePhase.DONE:
            route.cursor -= 1
        else:
            route.phase = RoutePhase.SWEEP
        route.passed_tiles.discard(previous_tile)
        route.continuation_status = f"SUCCESSOR_{route.continuation_next_kind}"
        route.continuation_blocked_reason = None
        for successor in ready_successors:
            route.late_work_ids.discard(successor.id)
    def _prepare_supplies(
        self,
        route: StripRoute,
        position: tuple[int, int],
        supply_plan: RouteSupplyPlan,
        supply_state: RouteSupplyState,
        obs: Mapping[str, Any],
        work_plan: StripWorkPlan,
    ) -> tuple | None:
        """Advance one bounded pickup step for an unfinalized supply batch.

        Prior pickup confirmation is applied once per observation by
        :meth:`_confirm_pending_supply_pickups` before workers act, so this
        method only issues movement or the next batched ``PICKUP``.
        """

        inventory = self._worker_inventory(obs, route.owner)
        shed = ((obs.get("private") or {}).get("shed") or {})
        slots = remaining_day_action_slots(obs)
        self._schedule_animal_revisit(route, work_plan)
        items_by_tile: dict[tuple[int, int], list[WorkItem]] = {}
        for item in work_plan.items:
            if item.tile is not None:
                items_by_tile.setdefault(item.tile, []).append(item)

        def can_reach_first_interaction() -> bool:
            segments = []
            revisit_tile = self._animal_revisit_tiles.get(route.route_id)
            if revisit_tile is not None:
                segments.append(
                    route_cost_segment_from_items(
                        f"{route.route_id}:PICKUP_REVISIT:{revisit_tile[0]},{revisit_tile[1]}",
                        (revisit_tile,),
                        items_by_tile.get(revisit_tile, ()),
                    )
                )
            else:
                remaining_tiles = set(route.traversal[route.cursor :])
                for segment in route.segments:
                    traversal = tuple(
                        tile for tile in segment.traversal if tile in remaining_tiles
                    )
                    if not traversal:
                        continue
                    segments.append(
                        route_cost_segment_from_items(
                            segment.segment_id,
                            traversal,
                            (
                                item
                                for tile in traversal
                                for item in items_by_tile.get(tile, ())
                            ),
                            physical_row_id=segment.physical_row_id,
                        )
                    )
            if not segments:
                return False
            reserved = dict(supply_plan.reserved_from_shed)
            for item, quantity in supply_state.failed_or_unfulfilled.items():
                reserved[item] = max(0, reserved.get(item, 0) - quantity)
            estimate = simulate_route_cost(
                position,
                tuple(segments),
                carried_inventory=inventory,
                remaining_action_slots=slots,
                assignment_turn=int(obs.get("hour", 0)),
                reserved_supply=reserved,
                global_resources=((obs.get("private") or {}).get("seeds") or {}),
                pickup_tile=supply_plan.pickup_tile,
            )
            return (
                estimate.first_interaction_turn is not None
                and estimate.first_interaction_turn <= slots
            )

        while True:
            batch = next(
                (
                    candidate
                    for candidate in supply_plan.pickup_sequence
                    if (
                        candidate.quantity
                        - supply_state.acquired.get(candidate.item, 0)
                        - supply_state.failed_or_unfulfilled.get(candidate.item, 0)
                    )
                    > 0
                ),
                None,
            )
            if batch is None:
                route.phase = (
                    RoutePhase.SWEEP
                    if route.route_id in self._animal_revisit_tiles
                    else RoutePhase.TRAVEL_TO_ENTRY
                )
                return None
            remaining = (
                batch.quantity
                - supply_state.acquired.get(batch.item, 0)
                - supply_state.failed_or_unfulfilled.get(batch.item, 0)
            )
            attempts = supply_state.attempts.get(batch.item, 0)
            available = max(0, int(shed.get(batch.item, 0)))
            if attempts >= 2 or available <= 0:
                supply_state.failed_or_unfulfilled[batch.item] = (
                    supply_state.failed_or_unfulfilled.get(batch.item, 0) + remaining
                )
                continue
            if position != supply_plan.pickup_tile:
                if not can_reach_first_interaction():
                    route.blocked_local_work["DEADLINE_UNREACHABLE"] = (
                        route.blocked_local_work.get("DEADLINE_UNREACHABLE", 0) + 1
                    )
                    return ("PASS",)
                movement = _vertical_first_step(position, supply_plan.pickup_tile)
                if movement is None:
                    route.blocked_local_work["SUPPLY_ROUTE_BLOCKED"] = (
                        route.blocked_local_work.get("SUPPLY_ROUTE_BLOCKED", 0) + 1
                    )
                    supply_state.failed_or_unfulfilled[batch.item] = (
                        supply_state.failed_or_unfulfilled.get(batch.item, 0) + remaining
                    )
                    continue
                supply_state.travel_turns += 1
                return movement
            if not can_reach_first_interaction():
                route.blocked_local_work["DEADLINE_UNREACHABLE"] = (
                    route.blocked_local_work.get("DEADLINE_UNREACHABLE", 0) + 1
                )
                return ("PASS",)
            quantity = min(remaining, available)
            supply_state.pending = PendingPickup(
                batch.item, quantity, max(0, int(inventory.get(batch.item, 0)))
            )
            supply_state.attempts[batch.item] = attempts + 1
            supply_state.pickup_turns += 1
            return ("PICKUP", batch.item, quantity)

    def _record_movement(
        self, route: StripRoute, obs: Mapping[str, Any]
    ) -> None:
        route.movement_turns += 1
        last_action = route.last_useful_action_step
        if last_action is not None and int(obs.get("step", 0)) > last_action:
            route.movement_only_turns += 1

    def _segment_can_reach_useful_action(
        self,
        segment,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> bool:
        """Guard one transfer with a one-use reachability check.

        This does not estimate the segment's completion time; full assignment
        and overload decisions use the canonical route cost simulator.
        """

        slots = remaining_day_action_slots(obs)
        if slots <= 0:
            return False
        has_work = False
        for tile in segment.traversal:
            items = tuple(
                item
                for item in work_plan.items
                if item.tile == tile
                and item.kind in _LOCAL_PRIORITY
                and item.status == WorkStatus.READY
            )
            if not items:
                continue
            has_work = True
            distance = abs(position[0] - tile[0]) + abs(position[1] - tile[1])
            if distance + 1 <= slots:
                return True
        return not has_work

    def _route_can_reach_useful_action(
        self,
        route: StripRoute,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
        target: tuple[int, int],
    ) -> bool:
        """Guard the next movement turn against a futile departure.

        The predicate asks whether any currently ready action remains
        reachable (or whether one step toward a known continuation is useful).
        It is a local execution safety check, not a whole-route forecast.
        """

        slots = remaining_day_action_slots(obs)
        if slots <= 0:
            return False
        feasible_distances: list[int] = []
        for tile in route.traversal[route.cursor:]:
            items = tuple(
                item
                for item in work_plan.items
                if item.tile == tile
                and item.kind in _LOCAL_PRIORITY
                and item.status == WorkStatus.READY
            )
            if not items:
                continue
            distance = abs(position[0] - tile[0]) + abs(position[1] - tile[1])
            feasible_distances.append(distance)
        if feasible_distances:
            return min(feasible_distances) + 1 <= slots
        # A target with no currently READY item may be a continuation that the
        # next observation makes actionable; one arrival turn is still useful.
        return abs(position[0] - target[0]) + abs(position[1] - target[1]) + 1 <= slots

    def _help_untouched_segment(
        self,
        route: StripRoute,
        work_plan: StripWorkPlan,
        *,
        position: tuple[int, int] | None = None,
        obs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append one safe untouched segment to a completed worker.

        Helping is deliberately bounded: only a segment after another
        worker's currently active segment is eligible, and only when it adds
        no inventory-scoped demand.  This preserves frozen ownership and all
        observation-confirmed shed reservations while preventing a completed
        worker from idling beside useful work.
        """

        for donor in sorted(self._routes.values(), key=lambda value: value.owner):
            if donor is route or donor.phase in (RoutePhase.DONE, RoutePhase.INVALID):
                continue
            current_index = next(
                (
                    index
                    for index, segment in enumerate(donor.segments)
                    if donor.current_tile in segment.traversal
                ),
                None,
            )
            if current_index is None:
                continue
            # A donor that has already emitted movement toward its next segment
            # is no longer "untouched" from the executor-state perspective.
            # Stealing that segment truncates ``donor.traversal`` while
            # ``donor.pending_cursor`` still addresses a tile inside it, which
            # later surfaces as an IndexError.  Resolve the pending target once
            # and refuse to transfer the segment that owns it; ownership must
            # remain with the donor until that transition resolves.
            pending_tile = (
                donor.traversal[donor.pending_cursor]
                if donor.pending_cursor is not None
                and 0 <= donor.pending_cursor < len(donor.traversal)
                else None
            )
            for segment_index in range(current_index + 1, len(donor.segments)):
                segment = donor.segments[segment_index]
                if segment.segment_id in donor.completed_segment_ids:
                    continue
                if any(tile in donor.passed_tiles for tile in segment.traversal):
                    continue
                if pending_tile is not None and pending_tile in segment.traversal:
                    continue
                if extract_tile_supply_demand(segment.traversal, work_plan):
                    continue
                if (
                    position is not None
                    and obs is not None
                    and not self._segment_can_reach_useful_action(
                        segment, position, work_plan, obs
                    )
                ):
                    continue
                remaining_segments = (
                    donor.segments[:segment_index]
                    + donor.segments[segment_index + 1:]
                )
                donor.traversal = tuple(
                    tile for value in remaining_segments for tile in value.traversal
                )
                donor.owned_tiles = donor.traversal
                donor.segments = remaining_segments
                route.traversal = route.traversal + segment.traversal
                route.owned_tiles = route.owned_tiles + segment.traversal
                route.segments = route.segments + (segment,)
                route.transferred_segment_ids.add(segment.segment_id)
                route.phase = RoutePhase.SWEEP
                route.pending_cursor = None
                for mutated in (donor, route):
                    if not route_cursor_invariants_hold(mutated):
                        raise ValueError(
                            "helping transfer violated route cursor invariants "
                            f"for {mutated.owner.label}"
                        )
                return True
        return False

    def _try_local_action(
        self,
        route: StripRoute,
        tile: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> tuple | None:
        """Execute one supported local item on ``tile``, else return ``None``."""

        inventory = self._worker_inventory(obs, route.owner)
        local_items = tuple(item for item in work_plan.items if item.tile == tile)
        if self.config.enable_row_claim_board and self._claim_board is not None:
            bundle_id = f"TILE:{tile[0]},{tile[1]}"
            if self._claim_board.owner_by_bundle.get(bundle_id) != route.owner:
                local_items = ()
        continuation_items = tuple(
            item
            for item in local_items
            if item.kind == route.continuation_next_kind
        ) if route.continuation_item_id is not None and route.continuation_tile == tile else ()
        if route.continuation_item_id is not None and route.continuation_tile == tile:
            if not continuation_items and route.continuation_status in {
                "HARVEST_RETAINED",
                "BLOCKED:MISSING_GLOBAL_RESOURCE",
            }:
                requested = work_plan.diagnostics.requested_crop_delta_dict
                if requested.get(route.continuation_crop or "", 0) <= 0:
                    route.continuation_status = "TARGET_REDUCED"
                    route.continuation_item_id = None
                    route.continuation_next_kind = None
                else:
                    route.continuation_blocked_reason = "NO_LEGAL_SLOT"
                    route.continuation_status = "BLOCKED:NO_LEGAL_SLOT"
                    route.continuation_item_id = None
                    route.continuation_next_kind = None
                    route.blocked_local_work["CONTINUATION_BLOCKED:NO_LEGAL_SLOT"] = (
                        route.blocked_local_work.get(
                            "CONTINUATION_BLOCKED:NO_LEGAL_SLOT", 0
                        ) + 1
                    )
            elif continuation_items and not any(
                item.status == WorkStatus.READY for item in continuation_items
            ):
                reason = continuation_items[0].block_reason.value if continuation_items[0].block_reason else continuation_items[0].status.value
                route.continuation_blocked_reason = reason
                route.continuation_status = f"BLOCKED:{reason}"
        item = self._select_local_item(local_items, inventory)
        if item is None:
            return None
        if item.source == "optional_deferrable" and self._optional_would_starve_tail(
            route, tile, work_plan, obs
        ):
            route.blocked_local_work["OPTIONAL_DEFERRED_DEADLINE"] = (
                route.blocked_local_work.get("OPTIONAL_DEFERRED_DEADLINE", 0) + 1
            )
            return None
        action = _interaction_action(item)
        if action is None:
            return None
        if (
            item.kind == "PLANT"
            and route.continuation_item_id is not None
            and route.continuation_next_kind == "PLANT"
            and int(obs.get("hour", 0)) >= 22
        ):
            route.continuation_blocked_reason = "INSUFFICIENT_DAY_TIME"
            route.continuation_status = "BLOCKED:INSUFFICIENT_DAY_TIME"
            route.blocked_local_work["CONTINUATION_BLOCKED:INSUFFICIENT_DAY_TIME"] = (
                route.blocked_local_work.get(
                    "CONTINUATION_BLOCKED:INSUFFICIENT_DAY_TIME", 0
                ) + 1
            )
            return None
        route.actions_performed[item.kind] = (
            route.actions_performed.get(item.kind, 0) + 1
        )
        if self.config.enable_row_claim_board and self._claim_board is not None:
            bundle_id = f"TILE:{tile[0]},{tile[1]}"
            if self._claim_board.owner_by_bundle.get(bundle_id) == route.owner:
                self._claim_board.phase_by_bundle[bundle_id] = ClaimPhase.IN_PROGRESS
        route.interaction_turns += item.interaction_turns
        route.last_useful_action_step = int(obs.get("step", 0))
        successor = next(
            (
                candidate
                for candidate in work_plan.items
                if candidate.tile == tile
                and item.id in candidate.depends_on
                and _is_crop_continuation(item.kind, candidate.kind)
            ),
            None,
        )
        retained_harvest = (
            item.kind == "HARVEST"
            and item.source == "routine_harvest"
            and item.crop in _RETAINED_ONE_SHOT_CROPS
        )
        if successor is not None or retained_harvest:
            route.continuation_item_id = item.id
            route.continuation_tile = tile
            route.continuation_next_kind = (
                successor.kind if successor is not None else "PLANT"
            )
            route.continuation_crop = item.crop
            route.continuation_source = (
                "retained_crop_maintenance"
                if retained_harvest
                else route.continuation_source or item.source
            )
            route.continuation_status = (
                "HARVEST_RETAINED" if retained_harvest else f"SUCCESSOR_{successor.kind}"
            )
            route.continuation_blocked_reason = None
        elif item.kind == "WATER" and route.continuation_next_kind == "WATER":
            route.continuation_status = "COMPLETED"
            route.continuation_blocked_reason = None
            route.continuation_item_id = None
            route.continuation_tile = None
            route.continuation_next_kind = None
        else:
            route.continuation_item_id = None
            route.continuation_tile = None
            route.continuation_next_kind = None
            route.continuation_crop = None
            route.continuation_source = None
            route.continuation_blocked_reason = None
        return action

    def _optional_would_starve_tail(
        self,
        route: StripRoute,
        tile: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> bool:
        """Ask whether one optional action pushes required tail work past the boundary.

        This is a per-turn opportunity-cost guard for PASS-only cleanup.  It
        does not predict route completion; planner deadline estimates use the
        canonical route cost simulator.
        """

        slots = remaining_day_action_slots(obs)
        if slots <= 0:
            return True
        try:
            current_index = route.traversal.index(tile, route.cursor)
        except ValueError:
            return False
        for future_tile in route.traversal[current_index + 1 :]:
            required = any(
                item.tile == future_tile
                and item.kind in _LOCAL_PRIORITY
                and item.status == WorkStatus.READY
                and item.source != "optional_deferrable"
                for item in work_plan.items
            )
            if not required:
                continue
            distance = abs(tile[0] - future_tile[0]) + abs(tile[1] - future_tile[1])
            if 1 + distance + 1 > slots:
                return True
        return False

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

    def _snapshot_tile_work(
        self, route: StripRoute, tile: tuple[int, int], items: tuple[WorkItem, ...]
    ) -> None:
        """Record item statuses as the worker leaves/handles ``tile``.

        This is the baseline used to tell work that was already handled from
        work that became ``READY`` only after the worker moved on.  It does not
        by itself mean the tile is passed.
        """

        passed = self._passed_work.setdefault(route.route_id, {})
        for item in items:
            if item.tile == tile:
                passed[item.id] = item.status.value

    def _mark_tile_passed(self, route: StripRoute, tile: tuple[int, int]) -> None:
        route.passed_tiles.add(tile)
        for segment in route.segments:
            if tile == segment.traversal[-1]:
                route.completed_segment_ids.add(segment.segment_id)

    def _record_late_work(self, route: StripRoute, work_plan: StripWorkPlan) -> None:
        # Only confirmed-passed tiles can hold late work; the current tile and
        # future route tiles still belong to the worker.  A set keeps repeated
        # observations idempotent.
        if not route.passed_tiles:
            return
        passed = self._passed_work.setdefault(route.route_id, {})
        passed_tiles = route.passed_tiles
        by_id = {item.id: item for item in work_plan.items}
        for item_id, previous_status in passed.items():
            item = by_id.get(item_id)
            if (
                item is not None
                and item.tile in passed_tiles
                and item.status == WorkStatus.READY
                and previous_status != WorkStatus.READY
            ):
                route.late_work_ids.add(item_id)
        for item in work_plan.items:
            if (
                item.tile in passed_tiles
                and item.id not in passed
                and item.status == WorkStatus.READY
            ):
                route.late_work_ids.add(item.id)

    def _supply_daily_diagnostics(self, shed: Mapping[str, int]) -> dict[str, Any]:
        plans = tuple(self._supply_plans.values())
        initial = {
            str(item): max(0, int(amount))
            for item, amount in shed.items()
            if int(amount) > 0
        }
        reservations: dict[str, int] = {}
        shortage: dict[str, int] = {}
        for plan in plans:
            for item, amount in plan.reserved_from_shed:
                reservations[item] = reservations.get(item, 0) + amount
            for item, amount in plan.missing_stock:
                shortage[item] = shortage.get(item, 0) + amount
        fully = sum(plan.fully_supplied for plan in plans if plan.demand)
        partial = sum(
            bool(plan.demand)
            and not plan.fully_supplied
            and any(amount for _, amount in plan.already_carried + plan.reserved_from_shed)
            for plan in plans
        )
        zero = sum(
            bool(plan.demand)
            and not any(amount for _, amount in plan.already_carried + plan.reserved_from_shed)
            for plan in plans
        )
        return {
            "initial_reservable_shed_stock": dict(sorted(initial.items())),
            "total_reservations_by_item": dict(sorted(reservations.items())),
            "unreserved_shortage_by_item": dict(sorted(shortage.items())),
            "routes_requiring_pickup": sum(plan.requires_pickup for plan in plans),
            "routes_fully_supplied": fully,
            "routes_partially_supplied": partial,
            "routes_with_zero_supply_fulfillment": zero,
            "route_supply_plans": [plan.to_json_dict() for plan in plans],
        }

    def _diagnostics(self) -> dict[str, Any]:
        assignment = self._assignment
        routes = self.routes
        completed = sum(route.phase == RoutePhase.DONE for route in routes)
        payload = dict(self._daily)
        farm_values = self._observation_for_diagnostics.get("farms") or ()
        positions = (
            self._worker_positions(self._observation_for_diagnostics)
            if self.config.acting_seat < len(farm_values)
            else {}
        )
        farm = (
            farm_values[self.config.acting_seat]
            if self.config.acting_seat < len(farm_values)
            else {}
        )
        estimates = self._hire_plan.route_estimates if self._hire_plan else ()
        payload.update(
            {
                "idle_workers": [
                    worker.label
                    for worker in (assignment.idle_workers if assignment else ())
                ],
                "completed_routes": completed,
                "unfinished_routes": len(routes) - completed,
                "route_diagnostics": [
                    {
                        **route.to_json_dict(),
                        "supply_plan": self._supply_plans[route.route_id].to_json_dict(),
                        "supply_state": self._supply_states[route.route_id].to_json_dict(),
                    }
                    for route in routes
                ],
                "actual_interactions_completed": sum(
                    route.interaction_turns for route in routes
                ),
                "actual_final_movement_only_turns": sum(
                    route.movement_only_turns for route in routes
                ),
                "actual_pass_tail_turns": sum(
                    route.pass_turns_after_completion for route in routes
                ),
                "unfinished_useful_work": (
                    sum(
                        max(0, route.workload_interactions - route.interaction_turns)
                        for route in routes
                    )
                    + sum(
                        candidate.workload_interactions
                        for candidate in assignment.unassigned
                    )
                    if assignment
                    else 0
                ),
                "workload_from_packet1": {
                    route_id: workload
                    for route_id, workload in payload.get("route_workload", {}).items()
                },
                "bootstrap_stage": self._bootstrap_stage,
                "worker_count_before_hiring": self._worker_count_before_hiring,
                "worker_count_final": len(positions),
                "hire_driving_routes": [
                    estimate.route_id for estimate in estimates if estimate.hire_driving
                ],
                "fertilizer_only_routes": [
                    estimate.route_id for estimate in estimates if estimate.fertilizer_only
                ],
                "coverage_prefix": list(
                    self._hire_plan.coverage_prefix if self._hire_plan else ()
                ),
                "wanted_hires": (
                    int(self._claim_hiring_diagnostics.get("wanted_hires", 0))
                    if self.config.enable_row_claim_board
                    else self._hire_plan.wanted_hires if self._hire_plan else 0
                ),
                "affordable_hires": (
                    int(self._claim_hiring_diagnostics.get("submittable_hires", 0))
                    if self.config.enable_row_claim_board
                    else self._hire_plan.affordable_hires if self._hire_plan else 0
                ),
                "submitted_hires": self._hire_submitted,
                "observed_hires": self._hire_observed,
                "failed_hires": self._hire_failures,
                # When hiring is permanently blocked, the retained hiring plan
                # describes the last attempt rather than the current turn.
                "hiring_diagnostics_status": (
                    "LAST_ATTEMPT" if self._hiring_blocked else "CURRENT"
                ),
                "sequential_hire_costs": list(
                    self._claim_hiring_diagnostics.get("sequential_hire_costs", ())
                    if self.config.enable_row_claim_board
                    else self._hire_plan.sequential_hire_costs
                    if self._hire_plan else ()
                ),
                "cash_before_hiring": (
                    self._claim_hiring_diagnostics.get("cash_before_hiring")
                    if self.config.enable_row_claim_board
                    else self._hire_plan.cash_before_hiring
                    if self._hire_plan else None
                ),
                "cash_after_observed_hiring": float(farm.get("money", 0.0)),
                "future_action_slots": (
                    int(self._claim_hiring_diagnostics.get("future_action_slots", 0))
                    if self.config.enable_row_claim_board
                    else self._hire_plan.future_action_slots
                    if self._hire_plan else 0
                ),
            }
        )
        if self._supply_plans:
            payload["supply_diagnostics"] = self._supply_daily_diagnostics(
                self._initial_shed
            )
        payload["market_diagnostics"] = self._market_state.to_json_dict()
        payload["routes_finalized"] = self._routes_finalized
        if self.config.enable_row_claim_board and self._claim_board is not None:
            payload["row_claim_board"] = self._claim_board.diagnostics()
            payload["row_claim_pass_reasons"] = dict(self._claim_pass_reasons)
            payload["row_claim_timing_ms"] = dict(self._claim_timings)
        return payload


def _supported_kind(kind: str) -> bool:
    return kind in _LOCAL_PRIORITY


def _is_crop_continuation(completed: str | None, next_kind: str) -> bool:
    return (completed, next_kind) in {
        ("WATER", "HARVEST"),
        ("DIG", "PLANT"),
        ("HARVEST", "PLANT"),
        ("PLANT", "WATER"),
    }


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
