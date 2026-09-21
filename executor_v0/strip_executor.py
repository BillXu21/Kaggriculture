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

from executor_v0.plan import DailyPlan
from executor_v0.strip_market import (
    MarketBootstrapState,
    build_market_turn_plan,
)
from executor_v0.strip_routes import (
    RouteAssignment,
    RoutePhase,
    StripRoute,
    WorkerId,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
    route_cursor_invariants_hold,
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


_RETAINED_ONE_SHOT_CROPS = frozenset(("WHEAT", "CARROT", "MELON"))

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

    @property
    def routes(self) -> tuple[StripRoute, ...]:
        return tuple(sorted(self._routes.values(), key=lambda route: route.route_id))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics()

    def _finalize_day(self, obs: Mapping[str, Any], plan: DailyPlan) -> StripWorkPlan:
        """Freeze Packet 2 ownership and Packet 3 reservations once."""

        day = int(obs.get("day", 0))
        bootstrap_diagnostics = self._daily.get("hiring_diagnostics")
        hire_stop_reason = self._daily.get("hire_stop_reason")
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
            "useful_row_count": len(candidates),
            "assigned_routes": len(assignment.routes),
            "unassigned_routes": len(assignment.unassigned),
            "workers": len(positions),
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
            work_plan = self._finalize_day(obs, active_plan)
            self._routes_finalized = True
            self._market_state.finalized_hour = int(obs.get("hour", 0))
        else:
            # Already finalized: still reconcile each new observation exactly once.
            self._reconcile_market_observation(obs)
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
            market_actions=market_plan.orders,
            diagnostics=self._diagnostics(),
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
        )

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
            added = any(
                new_demand.get(item, 0) > old_demand.get(item, 0)
                for item in new_demand
            )
            self._supply_plans[route.route_id] = refreshed
            if not added or not refreshed.requires_pickup:
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
                route, position, supply_plan, supply_state, obs
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
                    route.movement_turns += 1
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
        if route.phase == RoutePhase.DONE:
            if self._help_untouched_segment(route, work_plan):
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
                route.movement_turns += 1
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
                route.movement_turns += 1
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

    def _help_untouched_segment(
        self, route: StripRoute, work_plan: StripWorkPlan
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
        route.interaction_turns += item.interaction_turns
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


def _is_crop_continuation(completed: str | None, next_kind: str) -> bool:
    return (completed, next_kind) in {
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
