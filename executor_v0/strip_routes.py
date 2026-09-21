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


def assign_horizontal_routes(
    candidates: Iterable[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    *,
    assignment_hour: int,
) -> RouteAssignment:
    """Pack sorted row segments into stable worker-owned chains.

    Segment count is the primary load key, followed by travel from the
    worker's current/last endpoint, then worker index.  Manhattan ties choose
    the left endpoint (the lower canonical x value).
    """

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.row_key))
    ordered_workers = tuple(sorted(worker_positions))
    grouped: dict[WorkerId, list[tuple[HorizontalRouteCandidate, RouteSegment]]] = {
        worker: [] for worker in ordered_workers
    }
    endpoints = dict(worker_positions)
    for candidate in ordered_candidates:
        if not ordered_workers:
            break
        choices: list[tuple[int, int, int, WorkerId, RouteSegment]] = []
        for worker in ordered_workers:
            position = endpoints[worker]
            left_to_right = candidate.owned_tiles
            left, right = left_to_right[0], left_to_right[-1]
            left_distance = abs(position[0] - left[0]) + abs(position[1] - left[1])
            right_distance = abs(position[0] - right[0]) + abs(position[1] - right[1])
            traversal = left_to_right if left_distance <= right_distance else tuple(
                reversed(left_to_right)
            )
            segment = RouteSegment(
                candidate.route_id,
                traversal,
                traversal[0],
                min(left_distance, right_distance),
                candidate.source_shape,
            )
            choices.append((len(grouped[worker]), segment.entry_distance, worker.index, worker, segment))
        _, _, _, worker, segment = min(choices)
        candidate = next(item for item in ordered_candidates if item.route_id == segment.segment_id)
        grouped[worker].append((candidate, segment))
        endpoints[worker] = segment.traversal[-1]

    routes: list[StripRoute] = []
    assigned_ids: set[str] = set()
    for worker in ordered_workers:
        assigned = tuple(grouped[worker])
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
