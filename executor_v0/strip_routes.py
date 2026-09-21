"""Generic deterministic routes for the experimental strip executor.

Packet 2 generates only five-tile horizontal quadrant rows.  ``StripRoute``
itself deliberately represents an arbitrary ordered set of owned ``(y, x)``
tiles so later experiments can add route generators without replacing the
executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping

from executor_v0.strip_work import RowKey, StripWorkPlan

__all__ = [
    "HorizontalRouteCandidate",
    "RouteAssignment",
    "RoutePhase",
    "RouteSegment",
    "StripRoute",
    "WorkerId",
    "assign_horizontal_routes",
    "generate_horizontal_route_candidates",
    "route_cursor_invariants_hold",
]


class RoutePhase(StrEnum):
    PREPARE_SUPPLIES = "PREPARE_SUPPLIES"
    TRAVEL_TO_ENTRY = "TRAVEL_TO_ENTRY"
    SWEEP = "SWEEP"
    DONE = "DONE"
    INVALID = "INVALID"


@dataclass(frozen=True, order=True)
class WorkerId:
    """Stable farmer-first worker identity for one observation/day."""

    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError(f"worker index must be nonnegative, got {self.index!r}")

    @property
    def label(self) -> str:
        return "FARMER" if self.index == 0 else f"HAND:{self.index - 1}"


@dataclass(frozen=True)
class HorizontalRouteCandidate:
    route_id: str
    row_key: RowKey
    owned_tiles: tuple[tuple[int, int], ...]
    workload_interactions: int
    ready_interactions: int
    future_interactions: int
    source_shape: str = "horizontal_quadrant_row"

    def __post_init__(self) -> None:
        if len(self.owned_tiles) != 5:
            raise ValueError("Packet 2 horizontal routes must own exactly five tiles")


@dataclass(frozen=True)
class RouteSegment:
    """One deterministic horizontal row in a worker's day-local chain."""

    segment_id: str
    traversal: tuple[tuple[int, int], ...]
    entry_tile: tuple[int, int]
    entry_distance: int
    source_shape: str = "horizontal_quadrant_row"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "traversal": [list(tile) for tile in self.traversal],
            "entry_tile": list(self.entry_tile),
            "entry_distance": self.entry_distance,
            "source_shape": self.source_shape,
        }


@dataclass
class StripRoute:
    """One owned route and its small, day-local execution state."""

    route_id: str
    owned_tiles: tuple[tuple[int, int], ...]
    traversal: tuple[tuple[int, int], ...]
    owner: WorkerId
    entry_tile: tuple[int, int]
    entry_distance: int
    assignment_hour: int
    workload_interactions: int = 0
    source_shape: str | None = None
    cursor: int = 0
    pending_cursor: int | None = None
    phase: RoutePhase = RoutePhase.TRAVEL_TO_ENTRY
    completion_hour: int | None = None
    actions_performed: dict[str, int] = field(default_factory=dict)
    movement_turns: int = 0
    interaction_turns: int = 0
    pass_turns_after_completion: int = 0
    blocked_local_work: dict[str, int] = field(default_factory=dict)
    unavailable_supply_work: dict[str, int] = field(default_factory=dict)
    late_work_ids: set[str] = field(default_factory=set)
    # Tiles whose departure has been confirmed by a later observation, plus the
    # final tile once the route completes.  Used only for diagnostics.
    passed_tiles: set[tuple[int, int]] = field(default_factory=set)
    # A bounded crop-chain obligation.  This is intentionally one-hop and
    # route-local: only the immediately previous owned tile may be reopened.
    continuation_item_id: str | None = None
    continuation_tile: tuple[int, int] | None = None
    continuation_next_kind: str | None = None
    continuation_crop: str | None = None
    continuation_source: str | None = None
    continuation_status: str | None = None
    continuation_blocked_reason: str | None = None
    segments: tuple[RouteSegment, ...] = ()
    completed_segment_ids: set[str] = field(default_factory=set)
    transferred_segment_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.owned_tiles:
            raise ValueError("route must own at least one tile")
        if len(set(self.owned_tiles)) != len(self.owned_tiles):
            raise ValueError("owned route tiles must be unique")
        if len(self.traversal) != len(self.owned_tiles) or set(self.traversal) != set(
            self.owned_tiles
        ):
            raise ValueError("traversal must contain every owned tile exactly once")
        if self.entry_tile != self.traversal[0]:
            raise ValueError("entry_tile must be the first traversal tile")
        if not 0 <= self.cursor < len(self.traversal):
            raise ValueError("route cursor must address a traversal tile")
        if not self.segments:
            self.segments = (
                RouteSegment(
                    self.route_id,
                    self.traversal,
                    self.entry_tile,
                    self.entry_distance,
                    self.source_shape or "horizontal_quadrant_row",
                ),
            )
        if len({segment.segment_id for segment in self.segments}) != len(self.segments):
            raise ValueError("route segment ids must be unique")

    @property
    def completed(self) -> bool:
        return self.phase == RoutePhase.DONE

    @property
    def current_tile(self) -> tuple[int, int]:
        return self.traversal[self.cursor]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "owner": self.owner.label,
            "owned_tiles": [list(tile) for tile in self.owned_tiles],
            "traversal": [list(tile) for tile in self.traversal],
            "entry_tile": list(self.entry_tile),
            "entry_distance": self.entry_distance,
            "assignment_hour": self.assignment_hour,
            "completion_hour": self.completion_hour,
            "cursor": self.cursor,
            "pending_cursor": self.pending_cursor,
            "phase": self.phase.value,
            "completed": self.completed,
            "source_shape": self.source_shape,
            "segments": [segment.to_json_dict() for segment in self.segments],
            "assigned_segment_ids": [segment.segment_id for segment in self.segments],
            "completed_segment_ids": sorted(self.completed_segment_ids),
            "transferred_segment_ids": sorted(self.transferred_segment_ids),
            "workload_interactions": self.workload_interactions,
            "actions_performed": dict(sorted(self.actions_performed.items())),
            "movement_turns": self.movement_turns,
            "interaction_turns": self.interaction_turns,
            "pass_turns_after_completion": self.pass_turns_after_completion,
            "blocked_local_work": dict(sorted(self.blocked_local_work.items())),
            "unavailable_supply_work": dict(
                sorted(self.unavailable_supply_work.items())
            ),
            "late_work_ids": sorted(self.late_work_ids),
            "passed_tiles": [list(tile) for tile in sorted(self.passed_tiles)],
            "continuation": (
                {
                    "owner": self.owner.label,
                    "tile": list(self.continuation_tile)
                    if self.continuation_tile is not None
                    else None,
                    "completed_stage": self.continuation_item_id,
                    "next_expected_stage": self.continuation_next_kind,
                    "crop": self.continuation_crop,
                    "source": self.continuation_source,
                    "status": self.continuation_status,
                    "blocked_reason": self.continuation_blocked_reason,
                }
                if self.continuation_item_id is not None
                or self.continuation_status is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RouteAssignment:
    routes: tuple[StripRoute, ...]
    unassigned: tuple[HorizontalRouteCandidate, ...]
    idle_workers: tuple[WorkerId, ...]


def _manhattan_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Return the deterministic movement-turn estimate between two tiles."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def route_cursor_invariants_hold(route: StripRoute) -> bool:
    """True when ``route``'s cursor and pending cursor both address traversal.

    Helping transfers rewrite ``traversal``/``segments``.  Any transfer that
    leaves a cursor pointing past the end corrupts route state and would
    otherwise surface later as a raw ``IndexError`` in ``_act_worker``.
    """

    if not 0 <= route.cursor < len(route.traversal):
        return False
    if route.pending_cursor is None:
        return True
    return 0 <= route.pending_cursor < len(route.traversal)


def generate_horizontal_route_candidates(
    work_plan: StripWorkPlan,
) -> tuple[HorizontalRouteCandidate, ...]:
    """Generate one candidate for each row with represented spatial work."""

    candidates: list[HorizontalRouteCandidate] = []
    for summary in work_plan.row_summaries:
        interactions = summary.ready_interactions + summary.future_interactions
        if interactions <= 0:
            continue
        key = summary.row_key
        tiles = tuple((key.global_row, x) for x in range(key.x_start, key.x_end + 1))
        candidates.append(
            HorizontalRouteCandidate(
                route_id=(
                    f"ROW:{key.quadrant}:{key.global_row}:"
                    f"{key.x_start}-{key.x_end}"
                ),
                row_key=key,
                owned_tiles=tiles,
                workload_interactions=interactions,
                ready_interactions=summary.ready_interactions,
                future_interactions=summary.future_interactions,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.row_key))


@dataclass(frozen=True)
class _ChainPlan:
    movement_turns: int
    completion_turns: int
    assigned: tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]


def _chain_plan_for_mask(
    candidates: tuple[HorizontalRouteCandidate, ...],
    worker_position: tuple[int, int],
    mask: int,
) -> _ChainPlan:
    """Find the cheapest ordered/oriented chain for one candidate subset."""

    if not mask:
        return _ChainPlan(0, 0, ())

    # State is (owned mask, last candidate index, endpoint side).  Side 0
    # traverses left-to-right and side 1 right-to-left.  Keeping the path in
    # the value makes equal-cost choices stable without relying on hash order.
    states: dict[
        tuple[int, int, int], tuple[int, tuple[tuple[int, int, int], ...]]
    ] = {}
    for index, candidate in enumerate(candidates):
        bit = 1 << index
        if not mask & bit:
            continue
        for side in (0, 1):
            traversal = candidate.owned_tiles if side == 0 else tuple(reversed(candidate.owned_tiles))
            distance = _manhattan_distance(worker_position, traversal[0])
            states[(bit, index, side)] = (distance, ((index, side, distance),))

    for owned in range(1, mask + 1):
        if owned & ~mask:
            continue
        for (state_mask, last, side), (movement, path) in tuple(states.items()):
            if state_mask != owned:
                continue
            previous = candidates[last].owned_tiles
            previous_end = previous[-1] if side == 0 else previous[0]
            for index, candidate in enumerate(candidates):
                bit = 1 << index
                if mask & bit == 0 or owned & bit:
                    continue
                for next_side in (0, 1):
                    traversal = (
                        candidate.owned_tiles
                        if next_side == 0
                        else tuple(reversed(candidate.owned_tiles))
                    )
                    distance = _manhattan_distance(previous_end, traversal[0])
                    next_key = (owned | bit, index, next_side)
                    next_value = (movement + distance, path + ((index, next_side, distance),))
                    old_value = states.get(next_key)
                    if old_value is None or (next_value[0], next_value[1]) < (old_value[0], old_value[1]):
                        states[next_key] = next_value

    indices = tuple(index for index in range(len(candidates)) if mask & (1 << index))
    # Preserve the canonical west/east ordering for the two halves of one
    # physical row.  Whole vertical chains may still run in either direction
    # so workers starting at opposite ends get their nearest half.
    allowed_orders = {indices}
    if len({candidates[index].row_key.global_row for index in indices}) > 1:
        allowed_orders.add(tuple(reversed(indices)))
    choices = [
        value
        for (state_mask, _, _), value in states.items()
        if state_mask == mask
        and tuple(index for index, _, _ in value[1]) in allowed_orders
    ]
    movement, path = min(choices, key=lambda value: (value[0], value[1]))
    assigned: list[tuple[HorizontalRouteCandidate, RouteSegment]] = []
    interactions = 0
    sweep = 0
    for index, side, distance in path:
        candidate = candidates[index]
        traversal = candidate.owned_tiles if side == 0 else tuple(reversed(candidate.owned_tiles))
        assigned.append(
            (
                candidate,
                RouteSegment(
                    candidate.route_id,
                    traversal,
                    traversal[0],
                    distance,
                    candidate.source_shape,
                ),
            )
        )
        interactions += candidate.workload_interactions
        sweep += max(0, len(traversal) - 1)
    return _ChainPlan(movement, movement + sweep + interactions, tuple(assigned))


def _pareto_insert(
    values: list[tuple[int, int, int, tuple[int, ...]]],
    value: tuple[int, int, int, tuple[int, ...]],
) -> None:
    """Keep only packing states that can still win lexicographically."""

    if any(
        all(existing[index] <= value[index] for index in range(3))
        and (
            any(existing[index] < value[index] for index in range(3))
            or existing[3] <= value[3]
        )
        for existing in values
    ):
        return
    values[:] = [
        existing
        for existing in values
        if not (
            all(value[index] <= existing[index] for index in range(3))
            and (
                any(value[index] < existing[index] for index in range(3))
                or value[3] <= existing[3]
            )
        )
    ]
    values.append(value)


def _pack_small_route_sets(
    candidates: tuple[HorizontalRouteCandidate, ...],
    workers: tuple[WorkerId, ...],
    positions: Mapping[WorkerId, tuple[int, int]],
) -> dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]]:
    """Pack at most eight rows while retaining exact load/travel tradeoffs."""

    full_mask = (1 << len(candidates)) - 1
    plans = {
        worker: {
            mask: _chain_plan_for_mask(candidates, positions[worker], mask)
            for mask in range(full_mask + 1)
        }
        for worker in workers
    }
    states: dict[int, list[tuple[int, int, int, tuple[int, ...]]]] = {0: [(0, 0, 0, ())]}
    for worker in workers:
        next_states: dict[int, list[tuple[int, int, int, tuple[int, ...]]]] = {}
        for covered, values in states.items():
            remaining = full_mask ^ covered
            subset = remaining
            while True:
                plan = plans[worker][subset]
                for maximum, movement, total, signature in values:
                    candidate_value = (
                        max(maximum, plan.completion_turns),
                        movement + plan.movement_turns,
                        total + plan.completion_turns,
                        signature + (subset,),
                    )
                    _pareto_insert(
                        next_states.setdefault(covered | subset, []), candidate_value
                    )
                if subset == 0:
                    break
                subset = (subset - 1) & remaining
        states = next_states

    winning = min(states[full_mask], key=lambda value: value)
    grouped: dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]] = {}
    for worker, mask in zip(workers, winning[3]):
        grouped[worker] = plans[worker][mask].assigned
    return grouped


def _pack_large_route_set(
    candidates: tuple[HorizontalRouteCandidate, ...],
    workers: tuple[WorkerId, ...],
    positions: Mapping[WorkerId, tuple[int, int]],
    ) -> dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]]:
    """Bounded fallback for unusually dense forecasts."""

    grouped: dict[WorkerId, list[tuple[HorizontalRouteCandidate, RouteSegment]]] = {
        worker: [] for worker in workers
    }
    endpoints = dict(positions)
    loads = {worker: 0 for worker in workers}
    for candidate in candidates:
        choices = []
        for worker in workers:
            position = endpoints[worker]
            left = candidate.owned_tiles
            right = tuple(reversed(left))
            options = (
                (position, left),
                (position, right),
            )
            distance, traversal = min(
                (_manhattan_distance(position, option[1][0]), option[1])
                for option in options
            )
            projected = loads[worker] + distance + len(traversal) - 1 + candidate.workload_interactions
            choices.append((projected, distance, len(grouped[worker]), worker.index, worker, traversal))
        projected, distance, _, _, worker, traversal = min(choices)
        del projected
        grouped[worker].append(
            (
                candidate,
                RouteSegment(candidate.route_id, traversal, traversal[0], distance, candidate.source_shape),
            )
        )
        endpoints[worker] = traversal[-1]
        loads[worker] = loads[worker] + distance + len(traversal) - 1 + candidate.workload_interactions
    return {worker: tuple(value) for worker, value in grouped.items()}


def assign_horizontal_routes(
    candidates: Iterable[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    *,
    assignment_hour: int,
) -> RouteAssignment:
    """Pack row segments into deterministic, travel-efficient worker chains.

    The small-board path evaluates all worker/segment subsets.  A subset's
    best chain includes both segment order and endpoint orientation, and the
    packing minimizes the largest estimated completion time before total
    movement.  This makes adjacent rows stay together when workloads are
    equal, while allowing a worker that starts near the other end to keep its
    nearby cluster.  Larger inputs use a bounded nearest-endpoint fallback.
    """

    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.row_key,
            ),
        )
    )
    ordered_workers = tuple(sorted(worker_positions))
    grouped: dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]] = {
        worker: () for worker in ordered_workers
    }

    if ordered_workers and len(ordered_candidates) <= 8:
        grouped = _pack_small_route_sets(ordered_candidates, ordered_workers, worker_positions)
    elif ordered_workers:
        grouped = _pack_large_route_set(ordered_candidates, ordered_workers, worker_positions)

    routes: list[StripRoute] = []
    assigned_ids: set[str] = set()
    for worker in ordered_workers:
        assigned = grouped[worker]
        if not assigned:
            continue
        segments = tuple(segment for _, segment in assigned)
        traversal = tuple(tile for segment in segments for tile in segment.traversal)
        owned_tiles = tuple(tile for candidate, _ in assigned for tile in candidate.owned_tiles)
        route_id = segments[0].segment_id if len(segments) == 1 else (
            f"CHAIN:{worker.label}:" + ",".join(segment.segment_id for segment in segments)
        )
        assigned_ids.update(segment.segment_id for segment in segments)
        routes.append(
            StripRoute(
                route_id=route_id,
                owned_tiles=owned_tiles,
                traversal=traversal,
                owner=worker,
                entry_tile=segments[0].entry_tile,
                entry_distance=segments[0].entry_distance,
                assignment_hour=assignment_hour,
                workload_interactions=sum(
                    candidate.workload_interactions for candidate, _ in assigned
                ),
                source_shape=segments[0].source_shape,
                segments=segments,
            )
        )
    return RouteAssignment(
        routes=tuple(routes),
        unassigned=tuple(
            candidate
            for candidate in ordered_candidates
            if candidate.route_id not in assigned_ids
        ),
        idle_workers=tuple(worker for worker in ordered_workers if not grouped[worker]),
    )
