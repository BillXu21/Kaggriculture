"""Opt-in Packet 2/3 executor for fixed five-tile strip ownership.

This controller is intentionally separate from :mod:`executor_v0.agent` and
does not use the persistent scheduler.  Procurement and coverage-driven
hiring are bootstrapped before ownership is assigned once at the start of a
day; only the Packet 1 forecast is refreshed on later turns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from executor_v0.layout import quadrant_of, tile_role
from executor_v0.plan import DailyPlan
from executor_v0.strip_market import (
    MarketBootstrapState,
    build_market_turn_plan,
)
from executor_v0.strip_routes import (
    HorizontalRouteCandidate,
    RouteAssignment,
    RoutePhase,
    StripRoute,
    WorkerId,
    assign_horizontal_routes,
    build_horizontal_route,
    generate_horizontal_route_candidates,
)
from executor_v0.strip_hiring import StripHiringPlan, plan_strip_hiring
from executor_v0.strip_supply import (
    LOCAL_ACTION_PRIORITY,
    PendingPickup,
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
)

__all__ = [
    "StripExecutorConfig",
    "StripExecutorController",
    "StripExecutorResult",
]


_LOCAL_PRIORITY = LOCAL_ACTION_PRIORITY


@dataclass(frozen=True)
class StripExecutorConfig:
    """Packet 2 routing, Packet 3 prep, and strip bootstrap policy knobs."""

    acting_seat: int = 0
    work_config: StripWorkConfig = field(default_factory=StripWorkConfig)
    shed_capacity: int = 100
    max_market_orders: int = 10
    market_params: Mapping[str, Mapping[str, Any]] | None = None
    aggressive_sell_all: bool = False


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
        self._retired_routes: list[StripRoute] = []
        self._assignment: RouteAssignment | None = None
        self._unassigned_ids: tuple[str, ...] = ()
        self._queued_candidates: list[HorizontalRouteCandidate] = []
        self._helper_route_ids: set[str] = set()
        self._weed_cleanup_targets: dict[WorkerId, tuple[int, int]] = {}
        self._sharing_diagnostics: dict[str, int] = {}
        self._work_by_tile: dict[tuple[int, int], tuple[WorkItem, ...]] = {}
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
        self._observation_for_diagnostics: Mapping[str, Any] = {}

    @property
    def routes(self) -> tuple[StripRoute, ...]:
        return tuple(sorted(
            (*self._retired_routes, *self._routes.values()),
            key=lambda route: route.route_id,
        ))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics()

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
        candidates = generate_horizontal_route_candidates(work_plan)
        assignment = assign_horizontal_routes(
            candidates,
            positions,
            assignment_hour=int(obs.get("hour", 0)),
        )
        self._day = day
        self._plan = work_plan
        self._work_by_tile = _index_work_by_tile(work_plan)
        self._assignment = assignment
        self._routes = {route.owner: route for route in assignment.routes}
        self._retired_routes = []
        self._queued_candidates = list(assignment.unassigned)
        self._unassigned_ids = tuple(
            candidate.route_id for candidate in self._queued_candidates
        )
        self._helper_route_ids = set()
        self._weed_cleanup_targets = {}
        self._sharing_diagnostics = {
            "queued_rows_reassigned": 0,
            "helper_suffix_transfers": 0,
            "helper_tiles_transferred": 0,
            "productive_help_interactions": 0,
            "idle_weed_cleanup_assignments": 0,
            "idle_weed_digs": 0,
        }
        self._passed_work = {route.route_id: {} for route in assignment.routes}
        self._latest_inventories = {
            worker: self._worker_inventory(obs, worker) for worker in positions
        }
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
        self._daily = {
            "day": day,
            "assignment_hour": int(obs.get("hour", 0)),
            "active_routes": len(candidates),
            "assigned_routes": len(assignment.routes),
            "unassigned_routes": len(assignment.unassigned),
            "workers": len(positions),
            "unassigned_active_routes": list(self._unassigned_ids),
            "unassigned_supply_demand": {
                candidate.route_id: dict(
                    extract_tile_supply_demand(candidate.owned_tiles, work_plan)
                )
                for candidate in self._queued_candidates
            },
            "route_workload": {
                candidate.route_id: candidate.workload_interactions
                for candidate in candidates
            },
            "tileless_unresolved_work": [
                item.id for item in work_plan.items if item.tile is None
            ],
            "supply_diagnostics": self._supply_daily_diagnostics(
                self._initial_shed
            ),
        }
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
            for candidate in self._queued_candidates
            if estimates.get(candidate.route_id, None)
            and estimates[candidate.route_id].hire_driving
        ]
        self._daily["unassigned_fertilizer_only_routes"] = [
            candidate.route_id
            for candidate in self._queued_candidates
            if estimates.get(candidate.route_id, None)
            and estimates[candidate.route_id].fertilizer_only
        ]
        return work_plan

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
        self._work_by_tile = _index_work_by_tile(work_plan)
        self._confirm_pending_supply_pickups(obs)

        if not self._routes_finalized:
            self._reconcile_market_observation(obs)
            if self._bootstrap_stage == "PROCUREMENT":
                market_plan = build_market_turn_plan(
                    obs,
                    active_plan,
                    work_plan,
                    self._market_state,
                    acting_seat=self.config.acting_seat,
                    shed_capacity=self._shed_capacity(obs),
                    max_orders=self._max_market_orders(obs),
                    market_params=self._market_params(obs),
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
                if self._pending_hires is not None:
                    if int(obs.get("step", 0)) <= self._pending_hires["submitted_step"]:
                        return self._bootstrap_pass_result(obs, ())
                    # Reconciliation always clears the pending record on this path.
                    self._reconcile_hire_observation(obs)
                if self._hiring_blocked:
                    self._daily["hire_stop_reason"] = "FAILED"
                else:
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
                    self._daily["hiring_diagnostics"] = self._hire_plan.to_json_dict()
                    self._daily["hire_stop_reason"] = self._hire_plan.stop_reason.value
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
                        return self._bootstrap_pass_result(obs, self._hire_plan.orders)
                self._bootstrap_stage = "FINALIZED"
            work_plan = self._finalize_day(
                obs, active_plan, work_plan=work_plan
            )
            self._routes_finalized = True
            self._market_state.finalized_hour = int(obs.get("hour", 0))
        else:
            # Already finalized: still reconcile each new observation exactly once.
            self._reconcile_market_observation(obs)
        protected = self._outstanding_reservations()
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
            aggressive_sell_all=self.config.aggressive_sell_all,
        )
        self._market_state.latest_diagnostics = market_plan.diagnostics

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
        for worker in sorted(positions):
            route = self._routes.get(worker)
            if route is None:
                actions.append(self._act_idle_worker(
                    worker, positions[worker], work_plan, obs
                ))
                continue
            actions.append(self._act_worker(route, positions[worker], work_plan, obs))

        farmer_action = actions[0] if actions else ("PASS",)
        hands_actions = tuple(actions[1:])
        return StripExecutorResult(
            farmer_action=farmer_action,
            hands_actions=hands_actions,
            market_actions=market_plan.orders,
            diagnostics=self._diagnostics(),
        )

    next_worker_actions = act

    def _start_day(self, obs: Mapping[str, Any], plan: DailyPlan) -> None:
        self._day = int(obs.get("day", 0))
        self._daily_plan = plan
        self._routes = {}
        self._retired_routes = []
        self._assignment = None
        self._unassigned_ids = ()
        self._queued_candidates = []
        self._helper_route_ids = set()
        self._weed_cleanup_targets = {}
        self._sharing_diagnostics = {}
        self._work_by_tile = {}
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
        self._observation_for_diagnostics = obs
        self._daily = {"day": self._day, "assignment_hour": int(obs.get("hour", 0))}

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
        return 1

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
        actions = tuple(("PASS",) for _ in sorted(positions))
        return StripExecutorResult(
            farmer_action=actions[0] if actions else ("PASS",),
            hands_actions=actions[1:],
            market_actions=tuple(tuple(order) for order in market_actions),
            diagnostics=self._diagnostics(),
        )

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
        self._pending_hires = None
        if observed:
            self._hire_no_progress = 0
        else:
            self._hire_no_progress += 1
            if self._hire_no_progress >= 2:
                self._hiring_blocked = True
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
            remaining = pending.quantity - acquired
            supply_state.pending = None
            if remaining:
                attempts = supply_state.attempts.get(pending.item, 0)
                shed = (obs.get("private") or {}).get("shed") or {}
                if not acquired or attempts >= 2 or int(shed.get(pending.item, 0)) <= 0:
                    supply_state.failed_or_unfulfilled[pending.item] = (
                        supply_state.failed_or_unfulfilled.get(pending.item, 0) + remaining
                    )

    def _outstanding_reservations(self) -> dict[str, int]:
        protected: dict[str, int] = {}
        for route_id, plan in self._supply_plans.items():
            state = self._supply_states.get(route_id, RouteSupplyState())
            for item, reserved in plan.reserved_from_shed:
                outstanding = max(
                    0,
                    reserved
                    - state.acquired.get(item, 0)
                    - state.failed_or_unfulfilled.get(item, 0),
                )
                if outstanding:
                    protected[item] = protected.get(item, 0) + outstanding
        return protected

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
        supply_plan = self._supply_plans.get(route.route_id)
        supply_state = self._supply_states.setdefault(
            route.route_id, RouteSupplyState()
        )
        if route.phase == RoutePhase.PREPARE_SUPPLIES and supply_plan is not None:
            action = self._prepare_supplies(
                route, position, supply_plan, supply_state, obs
            )
            if action is not None:
                return action
        # Late-work observation runs every turn, including after DONE, so work
        # missed by the one-pass sweep stays visible without reopening the route.
        self._record_late_work(route, work_plan)
        if route.phase == RoutePhase.DONE:
            return self._act_idle_worker(route.owner, position, work_plan, obs, route)

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
                route.movement_turns += 1
                return movement
            departing = route.current_tile
            route.cursor = route.pending_cursor
            route.pending_cursor = None
            route.phase = RoutePhase.SWEEP
            self._mark_tile_passed(route, departing)

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

        action = self._try_local_action(route, target, work_plan, obs)
        if action is not None:
            return action

        inventory = self._worker_inventory(obs, route.owner)
        local_items = self._work_by_tile.get(target, ())
        self._record_skipped_work(route, local_items, inventory)
        self._snapshot_tile_work(route, target, local_items)
        if route.cursor + 1 >= len(route.traversal):
            # The final tile is processed: mark it passed so late work on it
            # stays diagnosable after completion.
            self._mark_tile_passed(route, target)
            route.phase = RoutePhase.DONE
            route.completion_hour = hour
            supply_state.remaining_at_completion = {
                item: int(inventory.get(item, 0))
                for item, _ in supply_plan.demand
                if int(inventory.get(item, 0)) > 0
            }
            return ("PASS",)
        route.pending_cursor = route.cursor + 1
        return _vertical_first_step(position, route.traversal[route.pending_cursor]) or (
            "PASS",
        )

    def _prepare_supplies(
        self,
        route: StripRoute,
        position: tuple[int, int],
        supply_plan: RouteSupplyPlan,
        supply_state: RouteSupplyState,
        obs: Mapping[str, Any],
    ) -> tuple | None:
        """Advance one bounded pickup step for an unfinalized supply batch.

        Prior pickup confirmation is applied once per observation by
        :meth:`_confirm_pending_supply_pickups` before workers act, so this
        method only issues movement or the next batched ``PICKUP``.
        """

        inventory = self._worker_inventory(obs, route.owner)
        shed = ((obs.get("private") or {}).get("shed") or {})
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
                route.phase = RoutePhase.TRAVEL_TO_ENTRY
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
            quantity = min(remaining, available)
            supply_state.pending = PendingPickup(
                batch.item, quantity, max(0, int(inventory.get(batch.item, 0)))
            )
            supply_state.attempts[batch.item] = attempts + 1
            supply_state.pickup_turns += 1
            return ("PICKUP", batch.item, quantity)

    def _try_local_action(
        self,
        route: StripRoute,
        tile: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> tuple | None:
        """Execute one supported local item on ``tile``, else return ``None``."""

        inventory = self._worker_inventory(obs, route.owner)
        local_items = self._work_by_tile.get(tile, ())
        item = self._select_local_item(local_items, inventory)
        if item is None:
            return None
        action = _interaction_action(item)
        if action is None:
            return None
        route.actions_performed[item.kind] = (
            route.actions_performed.get(item.kind, 0) + 1
        )
        route.interaction_turns += item.interaction_turns
        if route.route_id in self._helper_route_ids:
            self._sharing_diagnostics["productive_help_interactions"] += (
                item.interaction_turns
            )
        return action

    def _act_idle_worker(
        self,
        worker: WorkerId,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
        completed_route: StripRoute | None = None,
    ) -> tuple:
        """Give a completed or otherwise unassigned worker useful day-local work."""

        hour = int(obs.get("hour", 0))
        best_help = self._best_helper_suffix(worker, position, work_plan, hour)
        if best_help is not None and best_help[4] > 0:
            helper = self._transfer_helper_suffix(
                worker, position, work_plan, hour, best_help
            )
            if completed_route is not None:
                self._retired_routes.append(completed_route)
            self._routes[worker] = helper
            return self._act_worker(helper, position, work_plan, obs)

        queued = self._assign_queued_row(worker, position, work_plan, obs)
        if queued is not None:
            if completed_route is not None:
                self._retired_routes.append(completed_route)
            self._routes[worker] = queued
            return self._act_worker(queued, position, work_plan, obs)

        if best_help is not None:
            helper = self._transfer_helper_suffix(
                worker, position, work_plan, hour, best_help
            )
            if completed_route is not None:
                self._retired_routes.append(completed_route)
            self._routes[worker] = helper
            return self._act_worker(helper, position, work_plan, obs)

        cleanup = self._idle_weed_cleanup(worker, position, obs)
        if cleanup is not None:
            return cleanup
        if completed_route is not None:
            completed_route.pass_turns_after_completion += 1
        return ("PASS",)

    def _assign_queued_row(
        self,
        worker: WorkerId,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        obs: Mapping[str, Any],
    ) -> StripRoute | None:
        """Take the first queued row whose real work is still reachable today."""

        if not self._queued_candidates:
            return None
        private = obs.get("private") or {}
        shed = private.get("shed") or {}
        protected = self._outstanding_reservations()
        available_shed = {
            str(item): max(0, int(amount) - protected.get(str(item), 0))
            for item, amount in shed.items()
        }
        inventory = self._worker_inventory(obs, worker)
        hour = int(obs.get("hour", 0))
        for index, candidate in enumerate(self._queued_candidates):
            route = build_horizontal_route(
                candidate, worker, position, assignment_hour=hour
            )
            supply_plan = build_route_supply_plans(
                (route,), work_plan, {worker: inventory}, available_shed,
                {worker: position},
            )[0]
            if supply_plan.pickup_tile is not None:
                route = build_horizontal_route(
                    candidate, worker, supply_plan.pickup_tile,
                    assignment_hour=hour,
                )
                supply_plan = build_route_supply_plans(
                    (route,), work_plan, {worker: inventory}, available_shed,
                    {worker: position},
                )[0]
            if not _route_has_reachable_work(
                route, work_plan, position, hour, supply_plan,
                self._work_by_tile,
            ):
                continue
            self._queued_candidates.pop(index)
            self._unassigned_ids = tuple(
                item.route_id for item in self._queued_candidates
            )
            self._supply_plans[route.route_id] = supply_plan
            self._supply_states[route.route_id] = RouteSupplyState()
            route.phase = (
                RoutePhase.PREPARE_SUPPLIES
                if supply_plan.requires_pickup
                else RoutePhase.TRAVEL_TO_ENTRY
            )
            self._daily["unassigned_routes"] = len(self._queued_candidates)
            self._daily["assigned_routes"] = int(
                self._daily.get("assigned_routes", 0)
            ) + 1
            self._daily["unassigned_active_routes"] = list(self._unassigned_ids)
            self._daily["unassigned_supply_demand"] = {
                queued.route_id: dict(
                    extract_tile_supply_demand(queued.owned_tiles, work_plan)
                )
                for queued in self._queued_candidates
            }
            self._sharing_diagnostics["queued_rows_reassigned"] += 1
            return route
        return None

    def _best_helper_suffix(
        self,
        worker: WorkerId,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        hour: int,
    ) -> tuple[StripRoute, int, tuple[tuple[int, int], ...], int, int, int] | None:
        """Find the highest-priority reachable suffix without inventory work."""

        items_by_id = {item.id: item for item in work_plan.items}
        turns_left = max(0, 24 - hour)
        choices = []
        for donor in sorted(self._routes.values(), key=lambda route: route.route_id):
            if (
                donor.owner == worker
                or donor.phase in (RoutePhase.DONE, RoutePhase.INVALID)
                or donor.route_id in self._helper_route_ids
            ):
                continue
            boundary = max(
                donor.cursor,
                donor.pending_cursor
                if donor.pending_cursor is not None
                else donor.cursor,
            )
            passed_indexes = [
                index for index, tile in enumerate(donor.traversal)
                if tile in donor.passed_tiles
            ]
            if passed_indexes:
                boundary = max(boundary, max(passed_indexes))
            last_supply_index = max(
                (
                    index
                    for index, tile in enumerate(donor.traversal)
                    if any(
                        requirement.scope == "inventory"
                        and requirement.quantity > 0
                        for item in self._work_by_tile.get(tile, ())
                        for requirement in item.required_supplies
                    )
                ),
                default=-1,
            )
            start = max(boundary + 1, last_supply_index + 1)
            if start >= len(donor.traversal):
                continue
            suffix = donor.traversal[start:]
            help_items = [
                item
                for tile in suffix
                for item in self._work_by_tile.get(tile, ())
                if _is_helpable_item(item, items_by_id)
            ]
            interactions = sum(item.interaction_turns for item in help_items)
            if interactions <= 0:
                continue
            remaining_interactions = sum(
                item.interaction_turns
                for tile in donor.traversal
                if tile not in donor.passed_tiles
                for item in self._work_by_tile.get(tile, ())
                if _is_helpable_item(item, items_by_id)
            )
            urgency = sum(
                item.interaction_turns
                for item in help_items
                if item.kind == "WATER"
                and item.source == "survival_weed_prevention"
            )
            helper_traversal = tuple(reversed(suffix))
            entry = helper_traversal[0]
            useful_costs = [
                abs(position[0] - entry[0])
                + abs(position[1] - entry[1])
                + index
                + 1
                for index, tile in enumerate(helper_traversal)
                if any(
                    _is_helpable_item(item, items_by_id)
                    for item in self._work_by_tile.get(tile, ())
                )
            ]
            if not useful_costs or min(useful_costs) > turns_left:
                continue
            choices.append((
                -urgency, -remaining_interactions, donor.route_id, suffix[0],
                donor, start, suffix, interactions, urgency,
                remaining_interactions,
            ))
        if not choices:
            return None
        choice = min(choices, key=lambda item: item[:4])
        return choice[4], choice[5], choice[6], choice[7], choice[8], choice[9]

    def _transfer_helper_suffix(
        self,
        worker: WorkerId,
        position: tuple[int, int],
        work_plan: StripWorkPlan,
        hour: int,
        selection: tuple[
            StripRoute, int, tuple[tuple[int, int], ...], int, int, int
        ],
    ) -> StripRoute:
        """Move a donor's safe tail to one worker, preserving exclusive ownership.

        Inventory-scoped work is never transferred. Its original route reservation
        and acquired stock stay with the donor; water and other supply-free work
        can be shared without modifying the reservation ledger.
        """

        donor, start, suffix, interactions, _urgency, _remaining = selection
        helper_traversal = tuple(reversed(suffix))
        suffix_set = set(suffix)
        donor.traversal = donor.traversal[:start]
        donor.owned_tiles = tuple(
            tile for tile in donor.owned_tiles if tile not in suffix_set
        )
        donor.workload_interactions = max(
            0, donor.workload_interactions - interactions
        )
        helper = StripRoute(
            route_id=(
                f"HELP:{donor.route_id}:{worker.label}:"
                f"{helper_traversal[0][0]},{helper_traversal[0][1]}"
            ),
            owned_tiles=tuple(sorted(suffix_set)),
            traversal=helper_traversal,
            owner=worker,
            entry_tile=helper_traversal[0],
            entry_distance=(
                abs(position[0] - helper_traversal[0][0])
                + abs(position[1] - helper_traversal[0][1])
            ),
            assignment_hour=hour,
            workload_interactions=interactions,
            source_shape="helper_suffix",
            phase=RoutePhase.TRAVEL_TO_ENTRY,
        )
        helper_supply = build_route_supply_plans(
            (helper,), work_plan,
            {worker: self._worker_inventory(self._observation_for_diagnostics, worker)},
            {},
            {worker: position},
        )[0]
        # The suffix selector guarantees no inventory demand, so this assertion
        # documents the reservation-transfer invariant at the ownership boundary.
        if helper_supply.demand:
            raise AssertionError("helper suffix unexpectedly requires inventory")
        self._supply_plans[helper.route_id] = helper_supply
        self._supply_states[helper.route_id] = RouteSupplyState()
        self._helper_route_ids.add(helper.route_id)
        self._sharing_diagnostics["helper_suffix_transfers"] += 1
        self._sharing_diagnostics["helper_tiles_transferred"] += len(suffix)
        return helper

    def _idle_weed_cleanup(
        self,
        worker: WorkerId,
        position: tuple[int, int],
        obs: Mapping[str, Any],
    ) -> tuple | None:
        """Walk to and dig one visible unlocked weed after productive work."""

        farm = obs["farms"][self.config.acting_seat]
        board = farm.get("tiles") or ()
        unlocked = set(farm.get("unlocked_quadrants") or ())
        weeds = {
            (y, x)
            for y, row in enumerate(board[:10])
            for x, tile in enumerate(row[:10])
            if quadrant_of(y, x) in unlocked and tile_role(tile) == "weed"
        }
        for owner, tile in tuple(self._weed_cleanup_targets.items()):
            if tile not in weeds:
                del self._weed_cleanup_targets[owner]
        target = self._weed_cleanup_targets.get(worker)
        if target is None:
            claimed = set(self._weed_cleanup_targets.values())
            choices = sorted(
                weeds - claimed,
                key=lambda tile: (
                    abs(position[0] - tile[0]) + abs(position[1] - tile[1]),
                    tile,
                ),
            )
            turns_left = max(0, 24 - int(obs.get("hour", 0)))
            target = next(
                (
                    tile for tile in choices
                    if abs(position[0] - tile[0])
                    + abs(position[1] - tile[1]) + 1 <= turns_left
                ),
                None,
            )
            if target is None:
                return None
            self._weed_cleanup_targets[worker] = target
            self._sharing_diagnostics["idle_weed_cleanup_assignments"] += 1
        distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        if distance + 1 > max(0, 24 - int(obs.get("hour", 0))):
            self._weed_cleanup_targets.pop(worker, None)
            return None
        if position == target:
            self._sharing_diagnostics["idle_weed_digs"] += 1
            return ("DIG",)
        return _vertical_first_step(position, target)

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
                "wanted_hires": self._hire_plan.wanted_hires if self._hire_plan else 0,
                "affordable_hires": (
                    self._hire_plan.affordable_hires if self._hire_plan else 0
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
                    self._hire_plan.sequential_hire_costs if self._hire_plan else ()
                ),
                "cash_before_hiring": (
                    self._hire_plan.cash_before_hiring if self._hire_plan else None
                ),
                "cash_after_observed_hiring": float(farm.get("money", 0.0)),
                "future_action_slots": (
                    self._hire_plan.future_action_slots if self._hire_plan else 0
                ),
                **self._sharing_diagnostics,
                "queued_row_ids": [
                    candidate.route_id for candidate in self._queued_candidates
                ],
                "helper_route_ids": sorted(self._helper_route_ids),
                "idle_weed_targets": {
                    worker.label: list(tile)
                    for worker, tile in sorted(self._weed_cleanup_targets.items())
                },
            }
        )
        if self._supply_plans:
            payload["supply_diagnostics"] = self._supply_daily_diagnostics(
                self._initial_shed
            )
        payload["market_diagnostics"] = self._market_state.to_json_dict()
        payload["routes_finalized"] = self._routes_finalized
        return payload


def _supported_kind(kind: str) -> bool:
    return kind in _LOCAL_PRIORITY


def _index_work_by_tile(
    work_plan: StripWorkPlan,
) -> dict[tuple[int, int], tuple[WorkItem, ...]]:
    indexed: dict[tuple[int, int], list[WorkItem]] = {}
    for item in work_plan.items:
        if item.tile is not None:
            indexed.setdefault(item.tile, []).append(item)
    return {tile: tuple(items) for tile, items in indexed.items()}


def _is_helpable_item(
    item: WorkItem,
    items_by_id: Mapping[str, WorkItem],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    """Accept ready work and same-tile dependency chains that can become ready."""

    if (
        not _supported_kind(item.kind)
        or _interaction_action(item) is None
        or item.id in visiting
    ):
        return False
    if item.status == WorkStatus.READY:
        return True
    if (
        item.status != WorkStatus.BLOCKED
        or item.block_reason != BlockReason.DEPENDENCY_BLOCKED
        or not item.depends_on
    ):
        return False
    next_visiting = visiting | {item.id}
    return all(
        (dependency := items_by_id.get(dependency_id)) is not None
        and dependency.tile == item.tile
        and _is_helpable_item(dependency, items_by_id, next_visiting)
        for dependency_id in item.depends_on
    )


def _route_has_reachable_work(
    route: StripRoute,
    work_plan: StripWorkPlan,
    position: tuple[int, int],
    hour: int,
    supply_plan: RouteSupplyPlan,
    work_by_tile: Mapping[tuple[int, int], tuple[WorkItem, ...]],
) -> bool:
    """Check whether this row can reach one executable interaction today."""

    items_by_id = {item.id: item for item in work_plan.items}
    carried = dict(supply_plan.already_carried)
    reserved = dict(supply_plan.reserved_from_shed)
    available = {
        item: carried.get(item, 0) + reserved.get(item, 0)
        for item in set(carried) | set(reserved)
    }
    pickup_turns = len(supply_plan.pickup_sequence)
    if supply_plan.pickup_tile is None:
        supply_distance = 0
        travel_position = position
    else:
        supply_distance = (
            abs(position[0] - supply_plan.pickup_tile[0])
            + abs(position[1] - supply_plan.pickup_tile[1])
        )
        travel_position = supply_plan.pickup_tile
    entry_distance = (
        abs(travel_position[0] - route.entry_tile[0])
        + abs(travel_position[1] - route.entry_tile[1])
    )
    actions_before_route = supply_distance + pickup_turns + entry_distance
    turns_left = max(0, 24 - hour)

    for index, tile in enumerate(route.traversal):
        local_items = sorted(
            (
                work_plan_item
                for work_plan_item in work_by_tile.get(tile, ())
                if _is_helpable_item(work_plan_item, items_by_id)
            ),
            key=lambda item: (_LOCAL_PRIORITY.get(item.kind, 1000), item.id),
        )
        for item in local_items:
            if not all(
                requirement.scope != "inventory"
                or available.get(requirement.item, 0) >= requirement.quantity
                for requirement in item.required_supplies
            ):
                continue
            for requirement in item.required_supplies:
                if requirement.scope == "inventory":
                    available[requirement.item] -= requirement.quantity
            return actions_before_route + index + 1 <= turns_left
    return False


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
