"""Coverage-driven labor planning for the experimental strip executor.

This module deliberately consumes Packet 1 work and Packet 2 route candidates
directly.  It has no dependency on the legacy task scheduler or hiring policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from replay_daily.constants import FARM_HAND_COST_MULT_DEFAULT, hire_cost

from executor_v0.strip_cost import (
    RouteCostResult,
    RouteCostSegment,
    SegmentCostResult,
    simulate_route_cost,
)
from executor_v0.strip_routes import (
    HorizontalRouteCandidate,
    WorkerId,
    _candidate_cost_segment,
    assign_horizontal_routes,
    remaining_day_action_slots,
)
from executor_v0.strip_work import StripWorkPlan

__all__ = [
    "HireStopReason",
    "RouteLaborEstimate",
    "StripHiringPlan",
    "plan_strip_hiring",
]


class HireStopReason(StrEnum):
    COVERED = "COVERED"
    CASH = "CASH"
    TIME = "TIME"
    ORDER_CAP = "ORDER_CAP"
    FAILED = "FAILED"
    NO_HIRE_DRIVING_WORK = "NO_HIRE_DRIVING_WORK"


@dataclass(frozen=True)
class RouteLaborEstimate:
    route_id: str
    route_index: int
    hire_driving: bool
    fertilizer_only: bool
    first_use_eta: int | None
    estimated_full_turns: int
    future_action_slots: int
    useful_before_deadline: bool
    route_overloaded: bool
    first_use_work_id: str | None
    reasons: tuple[str, ...] = ()
    # Time-accounting breakdown for the corrected first-use ETA.  ``movement``
    # includes the Packet 3 pickup detour when a pickup is required;
    # ``preceding_interaction_turns`` counts every executable local interaction
    # performed before the first hire-driving item.
    movement_turns: int = 0
    pickup_turns: int = 0
    preceding_interaction_turns: int = 0
    estimated_arrival_turn: int = 0
    estimated_completion_turn: int = 0
    expected_useful_interactions_completed_before_deadline: int = 0
    expected_useful_interactions_left_after_deadline: int = 0
    forecast_effective_interactions: int = 0
    forecast_known_continuation_interactions: int = 0
    resource_feasible: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StripHiringPlan:
    current_workers: int
    target_workers: int
    wanted_hires: int
    affordable_hires: int
    submittable_hires: int
    sequential_hire_costs: tuple[int, ...]
    coverage_prefix: tuple[str, ...]
    route_estimates: tuple[RouteLaborEstimate, ...]
    cash_before_hiring: float
    future_action_slots: int
    stop_reason: HireStopReason
    hire_reason: str = ""
    rows_expected_complete_with_n_workers: int = 0
    rows_expected_complete_with_n_plus_one_workers: int = 0
    packed_segment_groups: tuple[tuple[str, ...], ...] = ()
    overloaded_rows_detected: int = 0
    row_helpers_required: int = 0
    row_helpers_assigned: int = 0
    unresolved_overloaded_rows: int = 0

    @property
    def orders(self) -> tuple[tuple[str], ...]:
        return tuple(("HIRE",) for _ in range(self.submittable_hires))

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sequential_hire_costs"] = list(self.sequential_hire_costs)
        payload["coverage_prefix"] = list(self.coverage_prefix)
        payload["route_estimates"] = [
            estimate.to_json_dict() for estimate in self.route_estimates
        ]
        payload["stop_reason"] = self.stop_reason.value
        return payload


@dataclass
class _SupplyLedger:
    shed: dict[str, int]
    seeds: dict[str, int]

def _positive_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(item): max(0, int(amount))
        for item, amount in value.items()
        if int(amount) > 0
    }


def _future_worker_actions(obs: Mapping[str, Any]) -> int:
    # A HIRE is processed after this turn's unit actions, so the new worker
    # receives the shared horizon with the current turn excluded.
    return remaining_day_action_slots(obs, include_current_turn=False)


def _spawn_positions(
    observed: Sequence[tuple[int, int]], hires: int, board_size: int
) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    # Official engine order is NW, NE, SW, SE in [x,y].  Convert it to the
    # strip executor's canonical [y,x] coordinates without re-sorting.
    access = (
        (half - 1, half - 1),
        (half - 1, half),
        (half, half - 1),
        (half, half),
    )
    occupancy = {tile: sum(position == tile for position in observed) for tile in access}
    spawned: list[tuple[int, int]] = []
    for _ in range(max(0, hires)):
        tile = min(
            enumerate(access), key=lambda pair: (occupancy[pair[1]], pair[0])
        )[1]
        spawned.append(tile)
        occupancy[tile] += 1
    return tuple(spawned)




def _estimate_route(
    candidate: HorizontalRouteCandidate,
    route_index: int,
    segment_cost: SegmentCostResult,
    route_cost: RouteCostResult,
    action_slots: int,
) -> RouteLaborEstimate:
    driving_turns = segment_cost.feasible_hire_driving_interaction_turns
    represented_driving = candidate.hire_driving_interactions > 0
    fertilizer_only = (
        candidate.forecasted_workload_interactions > 0 and not represented_driving
    )
    useful = (
        driving_turns > 0
        and segment_cost.first_use_turn is not None
        and segment_cost.first_use_turn <= action_slots
    )
    reasons: list[str] = []
    if not represented_driving:
        reasons.append("FERTILIZER_ONLY" if fertilizer_only else "NO_HIRE_DRIVING_WORK")
    elif driving_turns <= 0:
        reasons.append("NO_MECHANICALLY_FEASIBLE_HIRE_DRIVING_WORK")
    elif not useful:
        reasons.append("FIRST_USE_AFTER_DEADLINE")
    movement_turns = segment_cost.first_use_movement_turns
    return RouteLaborEstimate(
        route_id=candidate.route_id,
        route_index=route_index,
        hire_driving=driving_turns > 0,
        fertilizer_only=fertilizer_only,
        first_use_eta=segment_cost.first_use_turn,
        estimated_full_turns=max(
            0, segment_cost.completion_elapsed_turns - segment_cost.start_elapsed_turns
        ),
        future_action_slots=max(
            0, action_slots - segment_cost.start_elapsed_turns
        ),
        useful_before_deadline=useful,
        route_overloaded=(
            driving_turns > 0
            and segment_cost.hire_driving_interactions_missed > 0
        ),
        first_use_work_id=segment_cost.first_use_work_id,
        reasons=tuple(reasons),
        movement_turns=movement_turns,
        pickup_turns=route_cost.pickup_action_turns,
        preceding_interaction_turns=(
            segment_cost.first_use_preceding_interaction_turns
        ),
        estimated_arrival_turn=segment_cost.arrival_elapsed_turns,
        estimated_completion_turn=segment_cost.completion_elapsed_turns,
        expected_useful_interactions_completed_before_deadline=(
            segment_cost.hire_driving_interactions_completed_before_deadline
        ),
        expected_useful_interactions_left_after_deadline=(
            segment_cost.hire_driving_interactions_missed
        ),
        forecast_effective_interactions=segment_cost.effective_interaction_turns,
        forecast_known_continuation_interactions=segment_cost.known_continuation_turns,
        resource_feasible=segment_cost.resource_feasible,
    )


def _estimate_packed_workers(
    candidates: Sequence[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    worker_inventories: Mapping[WorkerId, Mapping[str, int]],
    work_plan: StripWorkPlan,
    *,
    future_action_slots: int,
    current_workers: int,
    shed: Mapping[str, int],
    seeds: Mapping[str, int],
    fertilizer_item_ids: frozenset[str],
) -> tuple[tuple[RouteLaborEstimate, ...], int, int]:
    """Estimate candidate completion under one deterministic packed assignment."""

    if not candidates or not worker_positions:
        return (), 0, 0
    assignment = assign_horizontal_routes(
        candidates,
        worker_positions,
        assignment_hour=0,
        worker_action_slots={
            worker: future_action_slots
            + int(worker.index < current_workers)
            for worker in worker_positions
        },
        worker_inventories=worker_inventories,
        shed_stock=shed,
        global_resources=seeds,
        enable_row_helpers=False,
    )
    by_id = {candidate.route_id: candidate for candidate in candidates}
    index_by_id = {candidate.route_id: index for index, candidate in enumerate(candidates)}
    estimates: dict[str, RouteLaborEstimate] = {}
    completed_driving = 0
    driving_total = 0
    ledger = _SupplyLedger(_positive_counts(shed), _positive_counts(seeds))
    for route in assignment.routes:
        position = worker_positions[route.owner]
        carried = _positive_counts(worker_inventories.get(route.owner))
        slots = future_action_slots + (1 if route.owner.index < current_workers else 0)
        cost_segments: list[RouteCostSegment] = []
        for segment in route.segments:
            candidate = by_id.get(segment.segment_id)
            if segment.cost_segment is not None:
                cost_segments.append(segment.cost_segment)
            elif candidate is not None:
                cost_segments.append(_candidate_cost_segment(candidate, segment.traversal))
        route_cost = simulate_route_cost(
            position,
            tuple(cost_segments),
            carried_inventory=carried,
            remaining_action_slots=slots,
            shed_stock=ledger.shed,
            global_resources=ledger.seeds,
        )
        for item, quantity in route_cost.supply_quantities_requiring_pickup:
            ledger.shed[item] = max(0, ledger.shed.get(item, 0) - quantity)
        for item, quantity in route_cost.global_quantities_consumed:
            ledger.seeds[item] = max(0, ledger.seeds.get(item, 0) - quantity)
        segment_costs = {result.segment_id: result for result in route_cost.segment_results}
        for segment in route.segments:
            candidate = by_id.get(segment.segment_id)
            if candidate is None:
                continue
            estimate = _estimate_route(
                candidate,
                index_by_id[candidate.route_id],
                segment_costs[segment.segment_id],
                route_cost,
                slots,
            )
            estimates[candidate.route_id] = estimate
            driving_total += int(estimate.hire_driving)
            if estimate.hire_driving and not estimate.expected_useful_interactions_left_after_deadline:
                completed_driving += 1
    ordered = tuple(
        estimates[candidate.route_id]
        for candidate in candidates
        if candidate.route_id in estimates
    )
    return ordered, completed_driving, driving_total


def plan_strip_hiring(
    obs: Mapping[str, Any],
    work_plan: StripWorkPlan,
    candidates: Sequence[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    worker_inventories: Mapping[WorkerId, Mapping[str, int]],
    *,
    acting_seat: int = 0,
    max_orders: int = 10,
    farm_hand_cost_mult: int = FARM_HAND_COST_MULT_DEFAULT,
) -> StripHiringPlan:
    """Plan the affordable useful prefix of observation-confirmed HIRE orders."""

    farm = (obs.get("farms") or ())[acting_seat]
    private = obs.get("private") or {}
    observed_workers = tuple(sorted(worker_positions))
    current_workers = len(observed_workers)
    future_slots = _future_worker_actions(obs)
    configuration = obs.get("configuration")
    config = configuration if isinstance(configuration, Mapping) else {}
    board_size = max(2, int(config.get("boardSize", 10)))
    predicted_spawns = _spawn_positions(
        tuple(worker_positions[worker] for worker in observed_workers),
        max(0, 2 * len(candidates) - current_workers),
        board_size,
    )
    fertilizer_item_ids = frozenset(
        item_id
        for chain in work_plan.chains
        if chain.kind == "FERTILIZER_UPKEEP" or chain.source == "fertilizer_policy"
        for item_id in chain.item_ids
    )
    max_workers = max(current_workers, len(candidates))
    all_positions = dict(worker_positions)
    all_positions.update(
        {
            WorkerId(current_workers + index): position
            for index, position in enumerate(predicted_spawns)
        }
    )
    packed_results: dict[int, tuple[tuple[RouteLaborEstimate, ...], int, int]] = {}
    for worker_count in range(current_workers, max_workers + 1):
        packed_results[worker_count] = _estimate_packed_workers(
            candidates,
            {worker: all_positions[worker] for worker in sorted(all_positions)[:worker_count]},
            worker_inventories,
            work_plan,
            future_action_slots=future_slots,
            current_workers=current_workers,
            shed=private.get("shed"),
            seeds=private.get("seeds"),
            fertilizer_item_ids=fertilizer_item_ids,
        )
    driving_total = max((result[2] for result in packed_results.values()), default=0)
    target_workers = current_workers
    hire_reason = "no_useful_work"
    if driving_total:
        current_completed = packed_results[current_workers][1]
        best_completed = max(result[1] for result in packed_results.values())
        if len(candidates) > 8:
            target_workers = max(current_workers, len(candidates))
            hire_reason = "one_worker_per_useful_row_large_board"
        else:
            target_workers = min(
                worker_count
                for worker_count, result in packed_results.items()
                if result[1] == best_completed
            )
            if current_completed >= driving_total:
                hire_reason = "covered_by_existing_packed_capacity"
            elif current_completed == best_completed:
                hire_reason = "no_extra_worker_useful_before_deadline"
            else:
                hire_reason = "additional_workers_reach_best_packed_coverage"

    base_target_workers = target_workers
    overload_assignment = assign_horizontal_routes(
        candidates,
        {
            worker: all_positions[worker]
            for worker in sorted(all_positions)[:base_target_workers]
        },
        assignment_hour=int(obs.get("hour", 0)),
        worker_action_slots={
            worker: (
                remaining_day_action_slots(obs)
                if worker.index < current_workers
                else future_slots
            )
            for worker in sorted(all_positions)[:base_target_workers]
        },
        worker_inventories=worker_inventories,
        shed_stock=private.get("shed"),
        global_resources=private.get("seeds"),
        enable_row_helpers=False,
    )
    overloaded_rows = overload_assignment.overloaded_rows_detected
    row_helpers_assigned = 0
    helper_target_assignment = overload_assignment
    if overloaded_rows:
        # A dedicated extra position per forecast overload is the bounded
        # correctness target. The assignment may use fewer if an observed idle
        # worker is already available.
        helper_capacity = max(
            current_workers,
            len(candidates) + overloaded_rows,
            base_target_workers,
        )
        helper_positions = {
            worker: all_positions[worker]
            for worker in sorted(all_positions)[:helper_capacity]
        }
        helper_target_assignment = assign_horizontal_routes(
            candidates,
            helper_positions,
            assignment_hour=int(obs.get("hour", 0)),
            worker_action_slots={
                worker: (
                    remaining_day_action_slots(obs)
                    if worker.index < current_workers
                    else future_slots
                )
                for worker in helper_positions
            },
            worker_inventories=worker_inventories,
            shed_stock=private.get("shed"),
            global_resources=private.get("seeds"),
        )
        row_helpers_assigned = helper_target_assignment.row_helpers_assigned
        if row_helpers_assigned:
            target_workers = max(
                base_target_workers,
                len(candidates) + row_helpers_assigned,
            )
            hire_reason = "overloaded_rows_require_dedicated_helpers"
    final_estimates = packed_results.get(target_workers or current_workers, ((), 0, 0))[0]
    if target_workers > max_workers:
        final_estimates = packed_results.get(base_target_workers, ((), 0, 0))[0]
    estimates = list(final_estimates)
    rows_with_n = packed_results.get(current_workers, ((), 0, 0))[1]
    rows_with_n_plus_one = packed_results.get(
        current_workers + 1, ((), rows_with_n, 0)
    )[1]
    wanted = max(0, target_workers - current_workers)
    hires_today = max(0, int(farm.get("hires_today", 0)))
    costs = tuple(
        hire_cost(hires_today + index, farm_hand_cost_mult)
        for index in range(wanted)
    )
    cash = float(farm.get("money", 0.0))
    remaining = cash
    affordable = 0
    for cost in costs:
        if remaining < cost:
            break
        remaining -= cost
        affordable += 1
    submittable = min(affordable, max(0, int(max_orders)))

    if not driving_total:
        stop = HireStopReason.NO_HIRE_DRIVING_WORK
    elif wanted == 0:
        stop = HireStopReason.COVERED
    elif submittable < affordable:
        # The per-turn market-order cap, not cash, limits this submission.
        stop = HireStopReason.ORDER_CAP
    elif affordable < wanted:
        stop = HireStopReason.CASH
    else:
        stop = HireStopReason.COVERED
    final_worker_positions = {
        worker: all_positions[worker]
        for worker in sorted(all_positions)[: target_workers or current_workers]
    }
    final_assignment = assign_horizontal_routes(
        candidates,
        final_worker_positions,
        assignment_hour=0,
        worker_action_slots={
            worker: (
                remaining_day_action_slots(obs)
                if worker.index < current_workers
                else future_slots
            )
            for worker in final_worker_positions
        },
        worker_inventories=worker_inventories,
        shed_stock=private.get("shed"),
        global_resources=private.get("seeds"),
    )
    return StripHiringPlan(
        current_workers=current_workers,
        target_workers=target_workers,
        wanted_hires=wanted,
        affordable_hires=affordable,
        submittable_hires=submittable,
        sequential_hire_costs=costs,
        coverage_prefix=tuple(
            estimate.route_id for estimate in estimates if estimate.hire_driving
        ),
        route_estimates=tuple(estimates),
        cash_before_hiring=cash,
        future_action_slots=future_slots,
        stop_reason=stop,
        hire_reason=hire_reason,
        rows_expected_complete_with_n_workers=rows_with_n,
        rows_expected_complete_with_n_plus_one_workers=rows_with_n_plus_one,
        packed_segment_groups=tuple(
            tuple(segment.segment_id for segment in route.segments)
            for route in final_assignment.routes
        ),
        overloaded_rows_detected=overloaded_rows,
        row_helpers_required=overloaded_rows,
        row_helpers_assigned=final_assignment.row_helpers_assigned,
        unresolved_overloaded_rows=final_assignment.unresolved_overloaded_rows,
    )
