"""Bounded three-pass horizontal-row scheduler over one shared claim board."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Mapping

from executor_v0.strip_claim_board import (
    ClaimBoard, RowFragment, SchedulerMode, ServiceClass, UncoveredRequiredWork,
)
from executor_v0.strip_cost import (
    AppendCostEstimate, AppendCostState, estimate_append_cost,
    nearest_shed_access, route_cost_segment_from_items, simulate_route_cost,
)
from executor_v0.strip_routes import (
    RouteAssignment, RoutePhase, RouteSegment, StripRoute, WorkerId,
)
from executor_v0.strip_work import StripWorkPlan


@dataclass
class WorkerPlanningState:
    worker: WorkerId
    observed_position: tuple[int, int]
    prefix: AppendCostState
    segments: list[RouteSegment] = field(default_factory=list)
    claimed_bundle_ids: list[str] = field(default_factory=list)
    optional_on_route: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScheduleResult:
    assignment: RouteAssignment
    worker_states: Mapping[WorkerId, WorkerPlanningState]
    timings_ms: Mapping[str, float]


@dataclass(frozen=True)
class HypotheticalWorkerCoverage:
    claimed_fragments: tuple[RowFragment, ...]
    effective_interactions: int
    incremental_turns: int
    reservation_shed: tuple[tuple[str, int], ...]
    reservation_global: tuple[tuple[str, int], ...]
    endpoint: tuple[int, int]
    rejection_reasons: tuple[str, ...]


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _span_traversal(tiles: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    first, last = tiles[0], tiles[-1]
    if first[0] != last[0]:
        raise ValueError("claim fragment must stay on one horizontal row")
    step = 1 if first[1] <= last[1] else -1
    return tuple((first[0], x) for x in range(first[1], last[1] + step, step))


def _fragment_segment(board: ClaimBoard, fragment: RowFragment,
                      traversal: tuple[tuple[int, int], ...],
                      work_plan: StripWorkPlan, index: int):
    bundle_ids = set(fragment.bundle_ids)
    items = tuple(
        item for tile in traversal
        for item in (board.bundles.get(f"TILE:{tile[0]},{tile[1]}").items
                     if f"TILE:{tile[0]},{tile[1]}" in board.bundles else ())
        if f"TILE:{tile[0]},{tile[1]}" in bundle_ids
    )
    segment_id = (
        f"ROW:{fragment.row_key.quadrant}:{fragment.row_key.global_row}:"
        f"{fragment.row_key.x_start}-{fragment.row_key.x_end}:{fragment.role}:{index}"
    )
    cost = route_cost_segment_from_items(segment_id, traversal, items,
                                         physical_row_id=segment_id.split(":PRIMARY")[0])
    return segment_id, cost


def _candidate(board: ClaimBoard, worker: WorkerPlanningState,
               fragment: RowFragment, traversal: tuple[tuple[int, int], ...],
               work_plan: StripWorkPlan, index: int):
    reservation = board.trial(worker.worker, fragment.bundle_ids)
    if reservation is None:
        return None
    segment_id, cost = _fragment_segment(board, fragment, traversal, work_plan, index)
    estimate = estimate_append_cost(worker.prefix, cost)
    if estimate.completed_interactions <= 0 or not estimate.complete_before_deadline:
        return None
    return reservation, segment_id, cost, estimate


def _commit(board: ClaimBoard, worker: WorkerPlanningState,
            fragment: RowFragment, traversal: tuple[tuple[int, int], ...],
            candidate) -> None:
    reservation, segment_id, cost, estimate = candidate
    if not board.claim(reservation):
        raise AssertionError("claim changed between trial and commit")
    worker.segments.append(RouteSegment(
        segment_id, traversal, traversal[0], estimate.travel_turns,
        physical_row_id=(f"ROW:{fragment.row_key.quadrant}:"
                         f"{fragment.row_key.global_row}:"
                         f"{fragment.row_key.x_start}-{fragment.row_key.x_end}"),
        represented_interactions=sum(work.represented_turns for tile in cost.work_by_tile
                                     for work in tile),
        known_continuation_interactions=sum(work.continuation_turns
                                            for tile in cost.work_by_tile for work in tile),
        forecast_tile_interactions=tuple(sum(work.effective_turns for work in tile)
                                         for tile in cost.work_by_tile),
        cost_segment=cost,
    ))
    worker.prefix = estimate.next_state
    worker.claimed_bundle_ids.extend(fragment.bundle_ids)


def _refresh_segment(board: ClaimBoard, state: WorkerPlanningState,
                     index: int) -> None:
    old = state.segments[index]
    owned_items = tuple(
        item for tile in old.traversal
        for key, service in board.bundles.items()
        if service.tile == tile and board.owner_by_bundle.get(key) == state.worker
        for item in service.items
    )
    cost = route_cost_segment_from_items(
        old.segment_id, old.traversal, owned_items,
        physical_row_id=old.physical_row_id,
    )
    state.segments[index] = replace(
        old, cost_segment=cost,
        represented_interactions=sum(
            work.represented_turns for tile in cost.work_by_tile for work in tile
        ),
        known_continuation_interactions=sum(
            work.continuation_turns for tile in cost.work_by_tile for work in tile
        ),
        forecast_tile_interactions=tuple(
            sum(work.effective_turns for work in tile) for tile in cost.work_by_tile
        ),
    )


def _score(estimate: AppendCostEstimate, worker: WorkerId,
           traversal: tuple[tuple[int, int], ...], hard: bool = False):
    density = estimate.completed_interactions / max(1, estimate.incremental_turns)
    return (0 if hard else 1, -density, -estimate.completed_interactions,
            estimate.incremental_turns, estimate.travel_turns, worker.index, traversal)


def _required_pass(board: ClaimBoard, states: dict[WorkerId, WorkerPlanningState],
                   work_plan: StripWorkPlan) -> None:
    fragments = board.required_fragments()
    def fragment_order(fragment: RowFragment):
        best_density = 0.0
        for state in states.values():
            for traversal in (fragment.traversal, fragment.traversal[::-1]):
                candidate = _candidate(board, state, fragment, traversal,
                                       work_plan, len(state.segments))
                if candidate is not None:
                    estimate = candidate[3]
                    best_density = max(
                        best_density,
                        estimate.completed_interactions
                        / max(1, estimate.incremental_turns),
                    )
        return (
            0 if any(board.bundles[b].service_class == ServiceClass.HARD_REQUIRED
                     for b in fragment.bundle_ids) else 1,
            min(board.bundles[b].source_rank for b in fragment.bundle_ids),
            -best_density,
            -sum(board.bundles[b].effective_interactions for b in fragment.bundle_ids),
            fragment.row_key, fragment.traversal,
        )
    fragments = tuple(sorted(fragments, key=fragment_order))
    for original in fragments:
        while True:
            outstanding = tuple(bundle_id for bundle_id in original.bundle_ids
                                if bundle_id in board.unclaimed())
            if not outstanding:
                break
            choices = []
            # Largest feasible prefix or suffix from each end. A full row is
            # evaluated once per direction; every shorter trial is local.
            for state in states.values():
                for reverse in (False, True):
                    ordered = outstanding[::-1] if reverse else outstanding
                    for width in range(len(ordered), 0, -1):
                        selected = ordered[:width]
                        traversal = _span_traversal(
                            tuple(board.bundles[b].tile for b in selected)
                        )
                        fragment = RowFragment(original.row_key, selected, traversal,
                                               "REQUIRED_TAIL")
                        candidate = _candidate(board, state, fragment, traversal,
                                               work_plan, len(state.segments))
                        if candidate is None:
                            continue
                        estimate = candidate[3]
                        hard = any(board.bundles[b].service_class == ServiceClass.HARD_REQUIRED
                                   for b in selected)
                        choices.append((_score(estimate, state.worker, traversal, hard),
                                        state, fragment, traversal, candidate))
                        break
            if not choices:
                break
            _, state, fragment, traversal, candidate = min(choices, key=lambda value: value[0])
            _commit(board, state, fragment, traversal, candidate)


def _optional_pass(board: ClaimBoard, states: dict[WorkerId, WorkerPlanningState],
                   work_plan: StripWorkPlan, mode: SchedulerMode) -> None:
    for fragment in board.required_fragments():
        for state in states.values():
            for bundle_id in fragment.bundle_ids:
                tile = board.bundles[bundle_id].tile
                one = RowFragment(fragment.row_key, (bundle_id,), (tile,),
                                  "REQUIRED_TAIL")
                if _candidate(board, state, one, (tile,), work_plan,
                              len(state.segments)) is not None:
                    return
    optional_ids = sorted(board.unclaimed(ServiceClass.OPTIONAL), key=lambda value: (
        0 if any(item.kind == "DIG" and item.source == "dig_cleanup"
                 for item in board.bundles[value].items) else 1,
        board.bundles[value].tile,
    ))
    for bundle_id in optional_ids:
        bundle = board.bundles[bundle_id]
        if mode == SchedulerMode.LIQUIDATION and not any(
            item.kind == "DIG" and item.source == "dig_cleanup"
            for item in bundle.items
        ):
            continue
        on_route = next((
            state for state in states.values()
            if any(bundle.tile in segment.traversal for segment in state.segments)
            and state.prefix.elapsed_turns + bundle.effective_interactions
            <= state.prefix.remaining_action_slots
            and board.trial(state.worker, (bundle_id,)) is not None
        ), None)
        if on_route is not None:
            reservation = board.trial(on_route.worker, (bundle_id,))
            assert reservation is not None and board.claim(reservation)
            on_route.claimed_bundle_ids.append(bundle_id)
            on_route.optional_on_route.append(bundle_id)
            on_route.prefix = replace(
                on_route.prefix,
                elapsed_turns=on_route.prefix.elapsed_turns + bundle.effective_interactions,
            )
            for index, old in enumerate(on_route.segments):
                if bundle.tile not in old.traversal:
                    continue
                _refresh_segment(board, on_route, index)
                break
            continue
        choices = []
        fragment = RowFragment(bundle.row_key, (bundle_id,), (bundle.tile,),
                               "OPTIONAL_TAIL")
        for state in states.values():
            position = state.prefix.position
            distance = abs(position[0] - bundle.tile[0]) + abs(position[1] - bundle.tile[1])
            if bundle.items[0].source == "dig_cleanup" and distance > 1 and not any(
                bundle.tile in segment.traversal for segment in state.segments
            ):
                continue
            candidate = _candidate(board, state, fragment, fragment.traversal,
                                   work_plan, len(state.segments))
            if candidate is not None:
                estimate = candidate[3]
                choices.append((estimate.incremental_turns, distance, state.worker,
                                state, candidate))
        if choices:
            _, _, _, state, candidate = min(choices, key=lambda value: value[:3])
            _commit(board, state, fragment, fragment.traversal, candidate)


def _routes(states: Mapping[WorkerId, WorkerPlanningState], assignment_hour: int,
            work_plan: StripWorkPlan, board: ClaimBoard) -> RouteAssignment:
    routes: list[StripRoute] = []
    for state in states.values():
        while state.segments:
            segments = state.segments
            exact = simulate_route_cost(
                state.observed_position,
                tuple(segment.cost_segment for segment in segments
                      if segment.cost_segment is not None),
                carried_inventory=board.worker_carried.get(state.worker, {}),
                shed_stock=board.observed_shed,
                global_resources=board.observed_global,
                remaining_action_slots=state.prefix.remaining_action_slots,
                assignment_turn=assignment_hour,
            )
            if exact.route_complete_before_deadline:
                break
            if state.optional_on_route:
                optional_id = state.optional_on_route.pop()
                tile = board.bundles[optional_id].tile
                board.release(optional_id)
                for index, segment in enumerate(state.segments):
                    if tile in segment.traversal:
                        _refresh_segment(board, state, index)
                        break
                continue
            if len(segments) == 1:
                break
            removed = segments.pop()
            for tile in removed.traversal:
                bundle_id = f"TILE:{tile[0]},{tile[1]}"
                if board.owner_by_bundle.get(bundle_id) == state.worker:
                    board.release(bundle_id)
        if not state.segments:
            continue
        traversal = tuple(tile for segment in state.segments for tile in segment.traversal)
        route_id = f"CLAIM:{state.worker.label}"
        route = StripRoute(
            route_id, traversal, traversal, state.worker, traversal[0],
            abs(state.observed_position[0] - traversal[0][0])
            + abs(state.observed_position[1] - traversal[0][1]),
            assignment_hour,
            workload_interactions=sum(segment.represented_interactions
                                      for segment in state.segments),
            source_shape="horizontal_claim_segments",
            phase=RoutePhase.TRAVEL_TO_ENTRY,
            segments=tuple(state.segments),
        )
        routes.append(route)
    return RouteAssignment(tuple(routes), (),
                           tuple(worker for worker in states
                                 if worker not in {route.owner for route in routes}),
                           primary_rows_assigned=sum(bool(state.segments and
                                                          ":PRIMARY:" in state.segments[0].segment_id)
                                                     for state in states.values()),
                           overflow_rows_assigned=sum(max(0, len(state.segments) - 1)
                                                      for state in states.values()))


def schedule_claim_board(board: ClaimBoard, work_plan: StripWorkPlan,
                         positions: Mapping[WorkerId, tuple[int, int]],
                         slots: int, assignment_hour: int,
                         *, mode: SchedulerMode = SchedulerMode.NORMAL) -> ScheduleResult:
    started = perf_counter()
    states = {worker: WorkerPlanningState(
        worker, position,
        AppendCostState(position, 0, slots,
                        tuple(sorted(board.worker_carried.get(worker, {}).items()))),
    ) for worker, position in sorted(positions.items())}
    built = perf_counter()
    rows = []
    for row in sorted(board.rows):
        ids = tuple(bundle_id for bundle_id in board.rows[row].bundle_ids_by_tile
                    if bundle_id and board.bundles[bundle_id].claimable
                    and board.bundles[bundle_id].service_class != ServiceClass.OPTIONAL)
        if ids:
            rows.append((row, ids))
    def row_order(pair):
        row, ids = pair
        full = board.rows[row].tiles
        fragment = RowFragment(row, ids, full, "PRIMARY")
        best_density = 0.0
        for state in states.values():
            for traversal in (full, full[::-1]):
                candidate = _candidate(board, state, fragment, traversal,
                                       work_plan, 0)
                if candidate is not None:
                    estimate = candidate[3]
                    best_density = max(
                        best_density,
                        estimate.completed_interactions
                        / max(1, estimate.incremental_turns),
                    )
        return (
            0 if any(board.bundles[b].service_class == ServiceClass.HARD_REQUIRED
                     for b in ids) else 1,
            min(board.bundles[b].source_rank for b in ids),
            -best_density,
            -sum(board.bundles[b].effective_interactions for b in ids), row,
        )
    rows.sort(key=row_order)
    unused = set(states)
    for row, ids in rows:
        choices = []
        full = board.rows[row].tiles
        fragment = RowFragment(row, ids, full, "PRIMARY")
        for worker in sorted(unused):
            for traversal in (full, full[::-1]):
                candidate = _candidate(board, states[worker], fragment, traversal,
                                       work_plan, 0)
                if candidate is not None:
                    estimate = candidate[3]
                    choices.append((_score(estimate, worker, traversal), worker,
                                    traversal, candidate))
        if choices:
            _, worker, traversal, candidate = min(choices, key=lambda value: value[0])
            _commit(board, states[worker], fragment, traversal, candidate)
            unused.remove(worker)
        if not unused:
            break
    first = perf_counter()
    _required_pass(board, states, work_plan)
    second = perf_counter()
    _optional_pass(board, states, work_plan, mode)
    third = perf_counter()
    assignment = _routes(states, assignment_hour, work_plan, board)
    end = perf_counter()
    return ScheduleResult(assignment, states, {
        "planning_state_setup": (built - started) * 1000,
        "pass1": (first - built) * 1000,
        "pass2": (second - first) * 1000,
        "pass3": (third - second) * 1000,
        "finalization": (end - third) * 1000,
    })


def evaluate_hypothetical_worker(uncovered: UncoveredRequiredWork,
                                 spawn: tuple[int, int],
                                 future_slots: int) -> HypotheticalWorkerCoverage:
    """Run the local tail rule for one new worker over an immutable snapshot."""
    if future_slots < 0:
        raise ValueError("future_slots must be nonnegative")
    views = {view.bundle_id: view for view in uncovered.bundle_views}
    fragments = list(uncovered.fragments)
    shed = dict(uncovered.remaining_shed)
    global_stock = dict(uncovered.remaining_global)
    initial_shed = dict(shed)
    initial_global = dict(global_stock)
    position = spawn
    remaining = future_slots
    claimed: list[RowFragment] = []
    pickup_items: set[str] = set()
    reasons: set[str] = set()
    while fragments:
        choices = []
        for fragment_index, fragment in enumerate(fragments):
            for reverse in (False, True):
                ordered = (fragment.bundle_ids[::-1] if reverse
                           else fragment.bundle_ids)
                for width in range(len(ordered), 0, -1):
                    ids = ordered[:width]
                    traversed = _span_traversal(
                        tuple(views[bundle_id].tile for bundle_id in ids)
                    )
                    demand: dict[str, int] = {}
                    seed_demand: dict[str, int] = {}
                    for bundle_id in ids:
                        for key, amount in views[bundle_id].inventory_demand:
                            demand[key] = demand.get(key, 0) + amount
                        for key, amount in views[bundle_id].global_demand:
                            seed_demand[key] = seed_demand.get(key, 0) + amount
                    if (any(shed.get(key, 0) < amount for key, amount in demand.items())
                            or any(global_stock.get(key, 0) < amount
                                   for key, amount in seed_demand.items())):
                        reasons.add("RESOURCE_SHORTAGE")
                        continue
                    travel = _distance(position, traversed[0])
                    new_pickups = set(demand) - pickup_items
                    if new_pickups and not pickup_items:
                        shed_tile = nearest_shed_access(position)
                        travel = (_distance(position, shed_tile)
                                  + _distance(shed_tile, traversed[0]))
                    travel += sum(_distance(left, right) for left, right
                                  in zip(traversed, traversed[1:]))
                    interactions = sum(views[bundle_id].effective_interactions
                                       for bundle_id in ids)
                    cost = travel + len(new_pickups) + interactions
                    if cost > remaining:
                        reasons.add("HORIZON_SHORTAGE")
                        continue
                    hard = any(views[bundle_id].hard_required for bundle_id in ids)
                    density = interactions / max(1, cost)
                    part = RowFragment(fragment.row_key, ids, traversed,
                                       "REQUIRED_TAIL")
                    choices.append(((0 if hard else 1, -density, -interactions,
                                     cost, traversed), fragment_index, part,
                                    demand, seed_demand, cost, new_pickups))
                    break
        if not choices:
            break
        _, fragment_index, part, demand, seed_demand, cost, new_pickups = min(
            choices, key=lambda value: value[0]
        )
        claimed.append(part)
        remaining -= cost
        position = part.traversal[-1]
        pickup_items.update(new_pickups)
        for key, amount in demand.items():
            shed[key] -= amount
        for key, amount in seed_demand.items():
            global_stock[key] -= amount
        old = fragments.pop(fragment_index)
        used = set(part.bundle_ids)
        leftovers = tuple(bundle_id for bundle_id in old.bundle_ids
                          if bundle_id not in used)
        if leftovers:
            fragments.append(RowFragment(
                old.row_key, leftovers,
                _span_traversal(tuple(views[bundle_id].tile for bundle_id in leftovers)),
                "REQUIRED_TAIL",
            ))
    return HypotheticalWorkerCoverage(
        tuple(claimed),
        sum(views[bundle_id].effective_interactions
            for fragment in claimed for bundle_id in fragment.bundle_ids),
        future_slots - remaining,
        tuple(sorted((key, amount - shed.get(key, 0))
                     for key, amount in initial_shed.items()
                     if amount > shed.get(key, 0))),
        tuple(sorted((key, amount - global_stock.get(key, 0))
                     for key, amount in initial_global.items()
                     if amount > global_stock.get(key, 0))),
        position, tuple(sorted(reasons)),
    )


def claim_runtime_fragment(board: ClaimBoard, work_plan: StripWorkPlan,
                           worker: WorkerId, position: tuple[int, int],
                           slots: int, *, mode: SchedulerMode = SchedulerMode.NORMAL
                           ) -> RouteSegment | None:
    """Claim one local required fragment, then one convenient optional tile."""
    state = WorkerPlanningState(
        worker, position, AppendCostState(
            position, 0, slots,
            tuple(sorted(board.worker_carried.get(worker, {}).items())),
            pickup_confirmed=True,
        ),
    )
    def choices_for(fragments: tuple[RowFragment, ...]):
        choices = []
        for fragment in fragments:
            for reverse in (False, True):
                ordered = fragment.bundle_ids[::-1] if reverse else fragment.bundle_ids
                for width in range(len(ordered), 0, -1):
                    ids = ordered[:width]
                    traversal = _span_traversal(
                        tuple(board.bundles[b].tile for b in ids)
                    )
                    if fragment.role == "OPTIONAL_TAIL" and any(
                        board.bundles[b].items[0].source == "dig_cleanup" for b in ids
                    ) and _distance(position, traversal[0]) > 1:
                        continue
                    part = RowFragment(fragment.row_key, ids, traversal, fragment.role)
                    candidate = _candidate(board, state, part, traversal, work_plan, 0)
                    if candidate is not None:
                        estimate = candidate[3]
                        hard = any(
                            board.bundles[b].service_class == ServiceClass.HARD_REQUIRED
                            for b in ids
                        )
                        choices.append((_score(estimate, worker, traversal, hard),
                                        part, traversal, candidate))
                        break
        return choices

    choices = choices_for(board.required_fragments())
    if not choices:
        optional = tuple(
            RowFragment(board.bundles[b].row_key, (b,), (board.bundles[b].tile,),
                        "OPTIONAL_TAIL")
            for b in board.unclaimed(ServiceClass.OPTIONAL)
            if mode == SchedulerMode.NORMAL or any(
                item.kind == "DIG" and item.source == "dig_cleanup"
                for item in board.bundles[b].items
            )
        )
        choices = choices_for(optional)
    if not choices:
        return None
    _, fragment, traversal, candidate = min(choices, key=lambda value: value[0])
    _commit(board, state, fragment, traversal, candidate)
    return state.segments[0]
