"""Generic deterministic routes for the experimental strip executor.

Packet 2 generates only five-tile horizontal quadrant rows.  ``StripRoute``
itself deliberately represents an arbitrary ordered set of owned ``(y, x)``
tiles so later experiments can add route generators without replacing the
executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from typing import Any, Iterable, Mapping

from executor_v0.strip_cost import (
    RouteCostResult,
    RouteCostSegment,
    inventory_requirements,
    route_cost_segment_from_forecast,
    route_cost_segment_from_items,
    simulate_route_cost,
)
from executor_v0.strip_work import (
    RowKey,
    StripWorkPlan,
    WorkItem,
    forecast_effective_interactions,
)

__all__ = [
    "HorizontalRouteCandidate",
    "RouteAssignment",
    "RoutePhase",
    "RouteSegment",
    "StripRoute",
    "WorkerId",
    "assign_horizontal_routes",
    "forecast_row_overloads",
    "generate_horizontal_route_candidates",
    "remaining_day_action_slots",
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
    # Interaction counts in owned-tile order.  Older callers construct
    # candidates directly, so an empty value retains the conservative
    # whole-segment completion estimate.
    tile_interactions: tuple[int, ...] = ()
    tile_known_continuation_interactions: tuple[int, ...] = ()
    forecast_tile_interactions: tuple[int, ...] = ()
    inventory_items: tuple[str, ...] = ()
    tile_hire_driving_interactions: tuple[int, ...] = ()
    tile_inventory_items: tuple[tuple[str, ...], ...] = ()
    tile_work_items: tuple[tuple[WorkItem, ...], ...] = ()
    fertilizer_work_ids: frozenset[str] = frozenset()
    physical_row_id: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= len(self.owned_tiles) <= 5:
            raise ValueError("horizontal row fragments must own one to five tiles")

    @property
    def row_id(self) -> str:
        return self.physical_row_id or self.route_id

    @property
    def represented_interactions(self) -> int:
        if len(self.tile_interactions) == len(self.owned_tiles):
            return sum(self.tile_interactions)
        return self.workload_interactions

    @property
    def known_continuation_interactions(self) -> int:
        if len(self.tile_known_continuation_interactions) == len(self.owned_tiles):
            return sum(self.tile_known_continuation_interactions)
        return 0

    @property
    def forecasted_workload_interactions(self) -> int:
        if len(self.forecast_tile_interactions) == len(self.owned_tiles):
            return sum(self.forecast_tile_interactions)
        if len(self.tile_interactions) == len(self.owned_tiles):
            return sum(self.tile_interactions)
        return self.workload_interactions

    @property
    def forecasted_tile_interactions(self) -> tuple[int, ...]:
        if len(self.forecast_tile_interactions) == len(self.owned_tiles):
            return self.forecast_tile_interactions
        if len(self.tile_interactions) == len(self.owned_tiles):
            return self.tile_interactions
        return (0,) * (len(self.owned_tiles) - 1) + (self.workload_interactions,)

    @property
    def hire_driving_interactions(self) -> int:
        if len(self.tile_hire_driving_interactions) == len(self.owned_tiles):
            return sum(self.tile_hire_driving_interactions)
        return self.forecasted_workload_interactions


@dataclass(frozen=True)
class RouteSegment:
    """One deterministic horizontal row in a worker's day-local chain."""

    segment_id: str
    traversal: tuple[tuple[int, int], ...]
    entry_tile: tuple[int, int]
    entry_distance: int
    source_shape: str = "horizontal_quadrant_row"
    physical_row_id: str | None = None
    represented_interactions: int = 0
    known_continuation_interactions: int = 0
    forecast_tile_interactions: tuple[int, ...] = ()
    cost_segment: RouteCostSegment | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "traversal": [list(tile) for tile in self.traversal],
            "entry_tile": list(self.entry_tile),
            "entry_distance": self.entry_distance,
            "source_shape": self.source_shape,
            "physical_row_id": self.physical_row_id,
            "represented_interactions": self.represented_interactions,
            "known_continuation_interactions": self.known_continuation_interactions,
            "forecast_tile_interactions": list(self.forecast_tile_interactions),
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
    movement_only_turns: int = 0
    interaction_turns: int = 0
    last_useful_action_step: int | None = None
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
            "movement_only_turns": self.movement_only_turns,
            "interaction_turns": self.interaction_turns,
            "last_useful_action_step": self.last_useful_action_step,
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
    large_route_assignment_mode: bool = False
    primary_rows_assigned: int = 0
    overflow_rows_assigned: int = 0
    idle_workers_with_unassigned_feasible_rows: int = 0
    row_diagnostics: tuple[dict[str, Any], ...] = ()

    @property
    def overloaded_rows_detected(self) -> int:
        return sum(bool(row.get("helper_required")) for row in self.row_diagnostics)

    @property
    def row_helpers_assigned(self) -> int:
        return sum(row.get("helper_worker") is not None for row in self.row_diagnostics)

    @property
    def unresolved_overloaded_rows(self) -> int:
        return sum(
            bool(row.get("helper_required")) and not bool(row.get("row_overload_resolved"))
            for row in self.row_diagnostics
        )


def _manhattan_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Return the deterministic movement-turn estimate between two tiles."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


@lru_cache(maxsize=4096)
def _candidate_cost_segment(
    candidate: HorizontalRouteCandidate,
    traversal: tuple[tuple[int, int], ...],
) -> RouteCostSegment:
    if len(candidate.tile_work_items) == len(candidate.owned_tiles) and any(
        candidate.tile_work_items
    ):
        items_by_tile = dict(zip(candidate.owned_tiles, candidate.tile_work_items, strict=True))
        return route_cost_segment_from_items(
            candidate.route_id,
            traversal,
            (item for tile in traversal for item in items_by_tile.get(tile, ())),
            physical_row_id=candidate.row_id,
            fertilizer_item_ids=candidate.fertilizer_work_ids,
        )
    represented = _candidate_tile_values(
        candidate, candidate.tile_interactions, candidate.forecasted_tile_interactions
    )
    continuation = _candidate_tile_values(
        candidate, candidate.tile_known_continuation_interactions, (0,) * len(candidate.owned_tiles)
    )
    driving = _candidate_tile_values(
        candidate, candidate.tile_hire_driving_interactions, candidate.forecasted_tile_interactions
    )
    inventory = _candidate_inventory_by_tile(candidate)
    return route_cost_segment_from_forecast(
        candidate.route_id,
        traversal,
        tuple(represented.get(tile, 0) for tile in traversal),
        tuple(continuation.get(tile, 0) for tile in traversal),
        physical_row_id=candidate.row_id,
        hire_driving_interactions=tuple(driving.get(tile, 0) for tile in traversal),
        inventory_items_by_tile=tuple(inventory.get(tile, ()) for tile in traversal),
    )


def remaining_day_action_slots(
    obs: Mapping[str, Any], *, include_current_turn: bool = True
) -> int:
    """Return actionable worker turns before the next day/terminal boundary.

    Existing workers can act on the observation being assigned, while a hand
    submitted by a HIRE order first acts on the following observation.  The
    terminal observation itself is never actionable.
    """

    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    configuration = obs.get("configuration")
    config = configuration if isinstance(configuration, Mapping) else {}
    turns_per_day = max(1, int(config.get("turnsPerDay", 24)))
    episode_steps = max(1, int(config.get("episodeSteps", 720)))
    step = int(obs.get("step", day * turns_per_day + hour))
    next_boundary = min((day + 1) * turns_per_day, episode_steps - 1)
    slots = max(0, next_boundary - step)
    return slots if include_current_turn else max(0, slots - 1)


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

    routed_kinds = {
        "WATER", "HARVEST", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
        "CARE", "FERTILIZE", "COLLECT_FERTILIZER", "PLANT", "PLACE",
    }
    items_by_tile: dict[tuple[int, int], list[WorkItem]] = {}
    for item in work_plan.items:
        if item.tile is not None and item.kind in routed_kinds:
            items_by_tile.setdefault(item.tile, []).append(item)

    candidates: list[HorizontalRouteCandidate] = []
    fertilizer_work_ids = frozenset(
        item_id
        for chain in work_plan.chains
        if chain.kind == "FERTILIZER_UPKEEP" or chain.source == "fertilizer_policy"
        for item_id in chain.item_ids
    )
    for summary in work_plan.row_summaries:
        interactions = summary.ready_interactions + summary.future_interactions
        if interactions <= 0:
            continue
        key = summary.row_key
        tiles = tuple((key.global_row, x) for x in range(key.x_start, key.x_end + 1))
        tile_items = tuple(tuple(items_by_tile.get(tile, ())) for tile in tiles)
        forecasts = tuple(forecast_effective_interactions(values) for values in tile_items)
        inventory_items: set[str] = set()
        tile_inventory_items: list[tuple[str, ...]] = []
        driving_forecasts = []
        for values in tile_items:
            driving_items = tuple(
                item
                for item in values
                if item.id not in fertilizer_work_ids
                and item.kind != "FERTILIZE"
                and not (
                    item.kind == "WATER"
                    and item.source == "fertilizer_linked_productive"
                )
            )
            driving_forecasts.append(
                forecast_effective_interactions(driving_items).effective_interactions
            )
            tile_inventory: set[str] = set()
            for item in values:
                tile_inventory.update(
                    requirement.item for requirement in inventory_requirements(item)
                )
            tile_inventory_items.append(tuple(sorted(tile_inventory)))
            inventory_items.update(tile_inventory)
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
                tile_interactions=tuple(
                    forecast.represented_interactions for forecast in forecasts
                ),
                tile_known_continuation_interactions=tuple(
                    forecast.known_continuation_interactions
                    for forecast in forecasts
                ),
                forecast_tile_interactions=tuple(
                    forecast.effective_interactions for forecast in forecasts
                ),
                inventory_items=tuple(sorted(inventory_items)),
                tile_hire_driving_interactions=tuple(driving_forecasts),
                tile_inventory_items=tuple(tile_inventory_items),
                tile_work_items=tile_items,
                fertilizer_work_ids=fertilizer_work_ids,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.row_key))


@dataclass(frozen=True)
class _ChainPlan:
    movement_turns: int
    completion_turns: int
    useful_interactions: int
    useful_segments: int
    unfinished_interactions: int
    assigned: tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]


@dataclass(frozen=True)
class _LargeRoutePacking:
    grouped: dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]]
    primary_rows_assigned: int
    overflow_rows_assigned: int
    idle_workers_with_unassigned_feasible_rows: int


def _compute_chain_plan_for_mask(
    candidates: tuple[HorizontalRouteCandidate, ...],
    worker_position: tuple[int, int],
    mask: int,
    remaining_action_slots: int | None = None,
    worker_inventory: Mapping[str, int] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
) -> _ChainPlan:
    """Find the cheapest ordered/oriented chain for one candidate subset."""

    if not mask:
        return _ChainPlan(0, 0, 0, 0, 0, ())

    indices = tuple(index for index in range(len(candidates)) if mask & (1 << index))
    # Preserve the canonical west/east ordering for the two halves of one
    # physical row.  Whole vertical chains may still run in either direction
    # so workers starting at opposite ends get their nearest half.
    allowed_orders = [indices]
    if len({candidates[index].row_key.global_row for index in indices}) > 1:
        allowed_orders.append(tuple(reversed(indices)))

    choices: list[tuple[tuple[int, int, int], ...]] = []
    for order in allowed_orders:
        for orientation_bits in range(1 << len(order)):
            path: list[tuple[int, int, int]] = []
            previous_end = worker_position
            for offset, index in enumerate(order):
                side = (orientation_bits >> offset) & 1
                traversal = (
                    candidates[index].owned_tiles
                    if side == 0
                    else tuple(reversed(candidates[index].owned_tiles))
                )
                distance = _manhattan_distance(previous_end, traversal[0])
                path.append((index, side, distance))
                previous_end = traversal[-1]
            choices.append(tuple(path))

    def path_result(path_value):
        route_segments = tuple(
            _candidate_cost_segment(
                candidates[index],
                (
                    candidates[index].owned_tiles
                    if side == 0
                    else tuple(reversed(candidates[index].owned_tiles))
                ),
            )
            for index, side, _distance in path_value
        )
        budget = (
            1_000_000_000
            if remaining_action_slots is None
            else remaining_action_slots
        )
        return simulate_route_cost(
            worker_position,
            route_segments,
            carried_inventory=worker_inventory,
            remaining_action_slots=budget,
            shed_stock=shed_stock,
            global_resources=global_resources,
            include_segment_results=False,
        )

    evaluated = [(path_result(path), path) for path in choices]

    def approach_travel(result: RouteCostResult) -> int:
        return (
            result.setup_travel_turns
            + result.pickup_travel_turns
            + result.inter_segment_travel_turns
        )

    if remaining_action_slots is None:
        result, path = min(
            evaluated,
            key=lambda value: (value[0].total_turns, approach_travel(value[0]), value[1]),
        )
    else:
        result, path = min(
            evaluated,
            key=lambda value: (
                -value[0].effective_interactions_completed_before_deadline,
                -value[0].segments_completed_before_deadline,
                value[0].total_turns,
                approach_travel(value[0]),
                value[1],
            ),
        )
    assigned: list[tuple[HorizontalRouteCandidate, RouteSegment]] = []
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
                    candidate.route_id,
                    candidate.represented_interactions,
                    candidate.known_continuation_interactions,
                    (
                        candidate.forecasted_tile_interactions
                        if side == 0
                        else tuple(reversed(candidate.forecasted_tile_interactions))
                    ),
                    _candidate_cost_segment(candidate, traversal),
                ),
            )
        )
    completion = result.total_turns
    useful_interactions = result.effective_interactions_completed_before_deadline
    useful_segments = result.segments_completed_before_deadline
    movement = (
        result.setup_travel_turns
        + result.pickup_travel_turns
        + result.inter_segment_travel_turns
    )
    return _ChainPlan(
        movement,
        completion,
        useful_interactions,
        useful_segments,
        result.effective_interactions_missed,
        tuple(assigned),
    )


@lru_cache(maxsize=4096)
def _cached_chain_plan_for_mask(
    candidates: tuple[HorizontalRouteCandidate, ...],
    worker_position: tuple[int, int],
    mask: int,
    remaining_action_slots: int | None,
    worker_inventory: tuple[tuple[str, int], ...],
    shed_stock: tuple[tuple[str, int], ...] | None,
    global_resources: tuple[tuple[str, int], ...] | None,
) -> _ChainPlan:
    return _compute_chain_plan_for_mask(
        candidates,
        worker_position,
        mask,
        remaining_action_slots,
        dict(worker_inventory),
        None if shed_stock is None else dict(shed_stock),
        None if global_resources is None else dict(global_resources),
    )


def _chain_plan_for_mask(
    candidates: tuple[HorizontalRouteCandidate, ...],
    worker_position: tuple[int, int],
    mask: int,
    remaining_action_slots: int | None = None,
    worker_inventory: Mapping[str, int] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
) -> _ChainPlan:
    """Cache deterministic subset costs across repeated assignment forecasts."""

    return _cached_chain_plan_for_mask(
        candidates,
        worker_position,
        mask,
        remaining_action_slots,
        tuple(sorted((worker_inventory or {}).items())),
        None if shed_stock is None else tuple(sorted(shed_stock.items())),
        None if global_resources is None else tuple(sorted(global_resources.items())),
    )


def _pareto_insert(
    values: list[tuple[int, int, int, int, int, int, tuple[int, ...]]],
    value: tuple[int, int, int, int, int, int, tuple[int, ...]],
) -> None:
    """Keep only packing states that can still win lexicographically."""

    if any(
        all(existing[index] <= value[index] for index in range(6))
        and (
            any(existing[index] < value[index] for index in range(6))
            or existing[6] <= value[6]
        )
        for existing in values
    ):
        return
    values[:] = [
        existing
        for existing in values
        if not (
            all(value[index] <= existing[index] for index in range(6))
            and (
                any(value[index] < existing[index] for index in range(6))
                or value[6] <= existing[6]
            )
        )
    ]
    values.append(value)


def _pack_small_route_sets(
    candidates: tuple[HorizontalRouteCandidate, ...],
    workers: tuple[WorkerId, ...],
    positions: Mapping[WorkerId, tuple[int, int]],
    worker_action_slots: Mapping[WorkerId, int] | None = None,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
) -> dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]]:
    """Pack at most eight rows while retaining exact load/travel tradeoffs."""

    full_mask = (1 << len(candidates)) - 1
    plans = {
        worker: {
            mask: _chain_plan_for_mask(
                candidates,
                positions[worker],
                mask,
                None if worker_action_slots is None else worker_action_slots[worker],
                (worker_inventories or {}).get(worker),
                shed_stock,
                global_resources,
            )
            for mask in range(full_mask + 1)
        }
        for worker in workers
    }
    states: dict[int, list[tuple[int, int, int, int, int, int, tuple[int, ...]]]] = {
        0: [(0, 0, 0, 0, 0, 0, ())]
    }
    for worker in workers:
        next_states: dict[
            int, list[tuple[int, int, int, int, int, int, tuple[int, ...]]]
        ] = {}
        for covered, values in states.items():
            remaining = full_mask ^ covered
            subset = remaining
            while True:
                plan = plans[worker][subset]
                for useful, segments, unfinished, maximum, movement, total, signature in values:
                    if worker_action_slots is None:
                        candidate_value = (
                            0,
                            0,
                            0,
                            max(maximum, plan.completion_turns),
                            movement + plan.movement_turns,
                            total + plan.completion_turns,
                            signature + (subset,),
                        )
                    else:
                        candidate_value = (
                            useful - plan.useful_interactions,
                            segments - plan.useful_segments,
                            unfinished + plan.unfinished_interactions,
                            max(maximum, plan.completion_turns),
                            movement + plan.movement_turns,
                            0,
                            signature + (subset,),
                        )
                    _pareto_insert(
                        next_states.setdefault(covered | subset, []), candidate_value
                    )
                if subset == 0:
                    break
                subset = (subset - 1) & remaining
        states = next_states

    if worker_action_slots is None:
        winning = min(states[full_mask], key=lambda value: value[3:])
    else:
        winning = min(states[full_mask])
    grouped: dict[WorkerId, tuple[tuple[HorizontalRouteCandidate, RouteSegment], ...]] = {}
    for worker, mask in zip(workers, winning[6]):
        grouped[worker] = plans[worker][mask].assigned
    return grouped


def _pack_large_route_set(
    candidates: tuple[HorizontalRouteCandidate, ...],
    workers: tuple[WorkerId, ...],
    positions: Mapping[WorkerId, tuple[int, int]],
    worker_action_slots: Mapping[WorkerId, int] | None = None,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
    ) -> _LargeRoutePacking:
    """Spread feasible primary rows before greedily packing overflow rows."""

    masks = {worker: 0 for worker in workers}
    preferences: dict[WorkerId, tuple[int, ...]] = {}
    for worker in workers:
        slots = (
            None if worker_action_slots is None else worker_action_slots[worker]
        )
        choices = []
        for index, candidate in enumerate(candidates):
            plan = _chain_plan_for_mask(
                candidates,
                positions[worker],
                1 << index,
                slots,
                (worker_inventories or {}).get(worker),
                shed_stock,
                global_resources,
            )
            if slots is not None and plan.useful_interactions <= 0:
                continue
            choices.append(
                (
                    -plan.useful_interactions if slots is not None else 0,
                    -plan.useful_segments if slots is not None else 0,
                    plan.completion_turns,
                    plan.movement_turns,
                    candidate.row_key,
                    index,
                )
            )
        preferences[worker] = tuple(choice[-1] for choice in sorted(choices))

    primary_owner: dict[int, WorkerId] = {}

    def assign_primary(worker: WorkerId, seen: set[int]) -> bool:
        for index in preferences[worker]:
            if index in seen:
                continue
            seen.add(index)
            owner = primary_owner.get(index)
            if owner is None or assign_primary(owner, seen):
                primary_owner[index] = worker
                return True
        return False

    for worker in workers:
        assign_primary(worker, set())
    for index, worker in primary_owner.items():
        masks[worker] |= 1 << index

    primary_workers = tuple(worker for worker in workers if masks[worker])
    overflow_workers = primary_workers
    for index, candidate in enumerate(candidates):
        if index in primary_owner:
            continue
        choices = []
        for worker in overflow_workers:
            mask = masks[worker] | (1 << index)
            base = _chain_plan_for_mask(
                candidates,
                positions[worker],
                masks[worker],
                None if worker_action_slots is None else worker_action_slots[worker],
                (worker_inventories or {}).get(worker),
                shed_stock,
                global_resources,
            )
            plan = _chain_plan_for_mask(
                candidates,
                positions[worker],
                mask,
                None if worker_action_slots is None else worker_action_slots[worker],
                (worker_inventories or {}).get(worker),
                shed_stock,
                global_resources,
            )
            unfinished = plan.unfinished_interactions
            choices.append(
                (
                    -(
                        plan.useful_interactions - base.useful_interactions
                    ) if worker_action_slots is not None else 0,
                    -(
                        plan.useful_segments - base.useful_segments
                    ) if worker_action_slots is not None else 0,
                    -plan.useful_interactions if worker_action_slots is not None else 0,
                    -plan.useful_segments if worker_action_slots is not None else 0,
                    unfinished,
                    plan.completion_turns,
                    plan.movement_turns,
                    worker.index,
                    worker,
                )
            )
        if not choices or (
            worker_action_slots is not None
            and max(-choice[0] for choice in choices) <= 0
        ):
            continue
        worker = min(choices)[-1]
        masks[worker] |= 1 << index

    grouped = {
        worker: _chain_plan_for_mask(
            candidates,
            positions[worker],
            masks[worker],
            None if worker_action_slots is None else worker_action_slots[worker],
            (worker_inventories or {}).get(worker),
            shed_stock,
            global_resources,
        ).assigned
        for worker in workers
    }
    assigned_indices = {
        index
        for mask in masks.values()
        for index in range(len(candidates))
        if mask & (1 << index)
    }
    idle_workers = tuple(worker for worker in workers if not grouped[worker])
    unassigned_indices = set(range(len(candidates))) - assigned_indices
    idle_feasible = sum(
        any(
            worker_action_slots is None
            or _chain_plan_for_mask(
                candidates,
                positions[worker],
                1 << index,
                worker_action_slots[worker],
                (worker_inventories or {}).get(worker),
                shed_stock,
                global_resources,
            ).useful_interactions > 0
            for index in unassigned_indices
        )
        for worker in idle_workers
    )
    primary_count = len(primary_owner)
    assigned_count = len(assigned_indices)
    return _LargeRoutePacking(
        grouped=grouped,
        primary_rows_assigned=primary_count,
        overflow_rows_assigned=max(0, assigned_count - primary_count),
        idle_workers_with_unassigned_feasible_rows=idle_feasible,
    )


def _assign_horizontal_routes_unsplit(
    candidates: Iterable[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    *,
    assignment_hour: int,
    remaining_action_slots: int | None = None,
    worker_action_slots: Mapping[WorkerId, int] | None = None,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
) -> RouteAssignment:
    """Pack row segments into deterministic, deadline-aware worker chains.

    With a remaining-day budget, the exact small-board path scores useful
    interactions, useful completed segments, unfinished interactions, maximum
    completion, movement, and the stable subset signature.  Without a budget
    it preserves the historical makespan/movement score.  Larger inputs first
    match feasible primary rows to distinct workers, then use a bounded
    marginal-gain approximation for overflow rows.
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

    large_route_assignment_mode = len(ordered_candidates) > 8
    primary_rows_assigned = 0
    overflow_rows_assigned = 0
    idle_workers_with_unassigned_feasible_rows = 0
    if ordered_workers and not large_route_assignment_mode:
        slots = worker_action_slots
        if slots is None and remaining_action_slots is not None:
            slots = {worker: remaining_action_slots for worker in ordered_workers}
        grouped = _pack_small_route_sets(
            ordered_candidates,
            ordered_workers,
            worker_positions,
            slots,
            worker_inventories,
            shed_stock,
            global_resources,
        )
    elif ordered_workers:
        slots = worker_action_slots
        if slots is None and remaining_action_slots is not None:
            slots = {worker: remaining_action_slots for worker in ordered_workers}
        large_packing = _pack_large_route_set(
            ordered_candidates,
            ordered_workers,
            worker_positions,
            slots,
            worker_inventories,
            shed_stock,
            global_resources,
        )
        grouped = large_packing.grouped
        primary_rows_assigned = large_packing.primary_rows_assigned
        overflow_rows_assigned = large_packing.overflow_rows_assigned
        idle_workers_with_unassigned_feasible_rows = (
            large_packing.idle_workers_with_unassigned_feasible_rows
        )

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
        large_route_assignment_mode=large_route_assignment_mode,
        primary_rows_assigned=primary_rows_assigned,
        overflow_rows_assigned=overflow_rows_assigned,
        idle_workers_with_unassigned_feasible_rows=(
            idle_workers_with_unassigned_feasible_rows
        ),
    )


def _candidate_tile_values(
    candidate: HorizontalRouteCandidate,
    values: tuple[int, ...],
    fallback: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    selected = values if len(values) == len(candidate.owned_tiles) else fallback
    return dict(zip(candidate.owned_tiles, selected, strict=True))


def _candidate_inventory_by_tile(
    candidate: HorizontalRouteCandidate,
) -> dict[tuple[int, int], tuple[str, ...]]:
    if len(candidate.tile_inventory_items) == len(candidate.owned_tiles):
        return dict(zip(candidate.owned_tiles, candidate.tile_inventory_items, strict=True))
    return {tile: candidate.inventory_items for tile in candidate.owned_tiles}


def forecast_row_overloads(
    candidates: Iterable[HorizontalRouteCandidate],
    assignment: RouteAssignment,
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    *,
    assignment_hour: int,
    worker_action_slots: Mapping[WorkerId, int] | None = None,
    remaining_action_slots: int | None = None,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Forecast physical-row overloads from canonical chained route costs."""

    ordered = tuple(sorted(candidates, key=lambda item: item.row_key))
    by_segment = {candidate.route_id: candidate for candidate in ordered}
    slots_by_worker = worker_action_slots or {}
    inventories = worker_inventories or {}
    rows: dict[str, dict[str, Any]] = {}
    for candidate in ordered:
        rows[candidate.row_id] = {
            "physical_row_id": candidate.row_id,
            "forecast_effective_interactions": candidate.forecasted_workload_interactions,
            "forecast_known_continuation_interactions": candidate.known_continuation_interactions,
            "primary_worker": None,
            "primary_tiles": [],
            "primary_expected_completion_turn": None,
            "helper_required": False,
            "helper_worker": None,
            "helper_tiles": [],
            "helper_expected_completion_turn": None,
            "row_overload_resolved": True,
            "primary_entry_elapsed": None,
            "primary_entry_travel_turns": 0,
            "canonical_cost": None,
        }

    remaining_shed = (
        None
        if shed_stock is None
        else {str(item): max(0, int(amount)) for item, amount in shed_stock.items()}
    )
    remaining_global = (
        None
        if global_resources is None
        else {str(item): max(0, int(amount)) for item, amount in global_resources.items()}
    )
    for route in assignment.routes:
        if not route.segments or route.owner not in worker_positions:
            continue
        candidates_by_route_segment = {
            segment.segment_id: by_segment.get(segment.segment_id)
            or by_segment.get(segment.physical_row_id)
            for segment in route.segments
        }
        route_cost_segments = tuple(
            _candidate_cost_segment(candidate, segment.traversal)
            for segment in route.segments
            if (candidate := candidates_by_route_segment[segment.segment_id]) is not None
        )
        slots = (
            slots_by_worker.get(route.owner)
            if worker_action_slots is not None
            else remaining_action_slots
        )
        cost = simulate_route_cost(
            worker_positions[route.owner],
            route_cost_segments,
            carried_inventory=inventories.get(route.owner, {}),
            remaining_action_slots=1_000_000_000 if slots is None else slots,
            assignment_turn=assignment_hour,
            shed_stock=remaining_shed,
            global_resources=remaining_global,
        )
        if remaining_shed is not None:
            for item, quantity in cost.supply_quantities_requiring_pickup:
                remaining_shed[item] = max(0, remaining_shed.get(item, 0) - quantity)
        if remaining_global is not None:
            for item, quantity in cost.global_quantities_consumed:
                remaining_global[item] = max(0, remaining_global.get(item, 0) - quantity)
        costs_by_segment = {result.segment_id: result for result in cost.segment_results}
        for segment in route.segments:
            candidate = candidates_by_route_segment[segment.segment_id]
            if candidate is not None:
                segment_cost = costs_by_segment[segment.segment_id]
                diagnostic = rows[candidate.row_id]
                overloaded = (
                    segment_cost.hire_driving_interactions_missed > 0
                    and candidate.hire_driving_interactions > 0
                )
                diagnostic.update(
                    {
                        "primary_worker": route.owner.label,
                        "primary_tiles": [list(tile) for tile in segment.traversal],
                        "primary_expected_completion_turn": assignment_hour
                        + segment_cost.completion_elapsed_turns,
                        "helper_required": overloaded,
                        "row_overload_resolved": not overloaded,
                        "primary_entry_elapsed": segment_cost.arrival_elapsed_turns,
                        "primary_entry_travel_turns": (
                            segment_cost.arrival_elapsed_turns
                            - segment_cost.start_elapsed_turns
                        ),
                        "canonical_cost": {
                            "total_turns": cost.total_turns,
                            "completion_hour": cost.completion_hour,
                            "route_complete_before_deadline": (
                                cost.route_complete_before_deadline
                            ),
                            "effective_interactions_missed": (
                                cost.effective_interactions_missed
                            ),
                            "supply_quantities_required": dict(
                                cost.supply_quantities_required
                            ),
                            "supply_quantities_already_carried": dict(
                                cost.supply_quantities_already_carried
                            ),
                            "supply_quantities_requiring_pickup": dict(
                                cost.supply_quantities_requiring_pickup
                            ),
                            "supply_shortage": dict(cost.supply_shortage),
                            "pickup_tile": cost.pickup_tile,
                            "segment": segment_cost.to_json_dict(),
                            "available_slots": slots,
                        },
                    }
                )
    return tuple(rows[candidate.row_id] for candidate in ordered)


def _fragment_candidate(
    candidate: HorizontalRouteCandidate,
    tiles: tuple[tuple[int, int], ...],
    *,
    role: str,
    boundary_index: int,
) -> HorizontalRouteCandidate:
    represented = _candidate_tile_values(
        candidate,
        candidate.tile_interactions,
        candidate.forecasted_tile_interactions,
    )
    continuation = _candidate_tile_values(
        candidate,
        candidate.tile_known_continuation_interactions,
        (0,) * len(candidate.owned_tiles),
    )
    effective = _candidate_tile_values(
        candidate,
        candidate.forecasted_tile_interactions,
        candidate.forecasted_tile_interactions,
    )
    driving = _candidate_tile_values(
        candidate,
        candidate.tile_hire_driving_interactions,
        candidate.forecasted_tile_interactions,
    )
    inventory = _candidate_inventory_by_tile(candidate)
    work_items = (
        dict(zip(candidate.owned_tiles, candidate.tile_work_items, strict=True))
        if len(candidate.tile_work_items) == len(candidate.owned_tiles)
        else {}
    )
    tile_inventory = tuple(inventory.get(tile, ()) for tile in tiles)
    return replace(
        candidate,
        route_id=f"{candidate.row_id}:{role}:{boundary_index}",
        owned_tiles=tiles,
        workload_interactions=sum(represented.get(tile, 0) for tile in tiles),
        ready_interactions=sum(represented.get(tile, 0) for tile in tiles),
        future_interactions=0,
        source_shape=f"horizontal_row_{role.lower()}_fragment",
        tile_interactions=tuple(represented.get(tile, 0) for tile in tiles),
        tile_known_continuation_interactions=tuple(
            continuation.get(tile, 0) for tile in tiles
        ),
        forecast_tile_interactions=tuple(effective.get(tile, 0) for tile in tiles),
        inventory_items=tuple(sorted({item for values in tile_inventory for item in values})),
        tile_hire_driving_interactions=tuple(driving.get(tile, 0) for tile in tiles),
        tile_inventory_items=tile_inventory,
        tile_work_items=tuple(work_items.get(tile, ()) for tile in tiles),
        physical_row_id=candidate.row_id,
    )


def _fragment_segment(
    candidate: HorizontalRouteCandidate,
    traversal: tuple[tuple[int, int], ...],
    *,
    entry_distance: int,
) -> RouteSegment:
    by_tile = dict(zip(candidate.owned_tiles, candidate.forecasted_tile_interactions, strict=True))
    represented = dict(
        zip(
            candidate.owned_tiles,
            candidate.tile_interactions
            if len(candidate.tile_interactions) == len(candidate.owned_tiles)
            else candidate.forecasted_tile_interactions,
            strict=True,
        )
    )
    continuation = dict(
        zip(
            candidate.owned_tiles,
            candidate.tile_known_continuation_interactions
            if len(candidate.tile_known_continuation_interactions) == len(candidate.owned_tiles)
            else (0,) * len(candidate.owned_tiles),
            strict=True,
        )
    )
    return RouteSegment(
        candidate.route_id,
        traversal,
        traversal[0],
        entry_distance,
        candidate.source_shape,
        candidate.row_id,
        sum(represented.get(tile, 0) for tile in traversal),
        sum(continuation.get(tile, 0) for tile in traversal),
        tuple(by_tile.get(tile, 0) for tile in traversal),
        _candidate_cost_segment(candidate, traversal),
    )


def assign_horizontal_routes(
    candidates: Iterable[HorizontalRouteCandidate],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
    *,
    assignment_hour: int,
    remaining_action_slots: int | None = None,
    worker_action_slots: Mapping[WorkerId, int] | None = None,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]] | None = None,
    shed_stock: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
    enable_row_helpers: bool = True,
) -> RouteAssignment:
    """Pack rows and split any overloaded physical row across one helper."""

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.row_key))
    base = _assign_horizontal_routes_unsplit(
        ordered_candidates,
        worker_positions,
        assignment_hour=assignment_hour,
        remaining_action_slots=remaining_action_slots,
        worker_action_slots=worker_action_slots,
        worker_inventories=worker_inventories,
        shed_stock=shed_stock,
        global_resources=global_resources,
    )
    row_diagnostics = list(
        forecast_row_overloads(
            ordered_candidates,
            base,
            worker_positions,
            assignment_hour=assignment_hour,
            worker_action_slots=worker_action_slots,
            remaining_action_slots=remaining_action_slots,
            worker_inventories=worker_inventories,
            shed_stock=shed_stock,
            global_resources=global_resources,
        )
    )
    if not enable_row_helpers or not row_diagnostics:
        return replace(base, row_diagnostics=tuple(row_diagnostics))

    helper_workers = list(base.idle_workers)
    routes = list(base.routes)
    helper_routes: list[StripRoute] = []
    diagnostic_by_row = {row["physical_row_id"]: row for row in row_diagnostics}
    by_id = {candidate.route_id: candidate for candidate in ordered_candidates}

    def segments_for_route(route: StripRoute) -> tuple[RouteCostSegment, ...]:
        values = []
        for segment in route.segments:
            if segment.cost_segment is not None:
                values.append(segment.cost_segment)
                continue
            source = by_id.get(segment.segment_id) or by_id.get(
                segment.physical_row_id
            )
            if source is not None:
                values.append(_candidate_cost_segment(source, segment.traversal))
        return tuple(values)

    def simulate_split(
        primary_route: StripRoute,
        primary_segments: tuple[RouteCostSegment, ...],
        helper: WorkerId,
        helper_segment: RouteCostSegment,
    ):
        """Reallocate shared resources in the final worker execution order."""

        route_segments = {
            route.owner: segments_for_route(route)
            for route in (*routes, *helper_routes)
        }
        route_segments[primary_route.owner] = primary_segments
        route_segments[helper] = (helper_segment,)
        available_shed = (
            None
            if shed_stock is None
            else {
                str(item): max(0, int(amount))
                for item, amount in shed_stock.items()
            }
        )
        available_global = (
            None
            if global_resources is None
            else {
                str(item): max(0, int(amount))
                for item, amount in global_resources.items()
            }
        )
        costs = {}
        for owner in sorted(route_segments):
            worker_slots = (
                worker_action_slots.get(owner)
                if worker_action_slots is not None
                else remaining_action_slots
            )
            cost = simulate_route_cost(
                worker_positions[owner],
                route_segments[owner],
                carried_inventory=(worker_inventories or {}).get(owner, {}),
                remaining_action_slots=(
                    1_000_000_000 if worker_slots is None else worker_slots
                ),
                assignment_turn=assignment_hour,
                shed_stock=available_shed,
                global_resources=available_global,
            )
            costs[owner] = cost
            if available_shed is not None:
                for item, quantity in cost.supply_quantities_requiring_pickup:
                    available_shed[item] = max(
                        0, available_shed.get(item, 0) - quantity
                    )
            if available_global is not None:
                for item, quantity in cost.global_quantities_consumed:
                    available_global[item] = max(
                        0, available_global.get(item, 0) - quantity
                    )
        return costs[primary_route.owner], costs[helper]

    for candidate in ordered_candidates:
        diagnostic = diagnostic_by_row[candidate.row_id]
        if not diagnostic["helper_required"]:
            continue
        primary = next(
            (
                route
                for route in routes
                if any(
                    segment.physical_row_id == candidate.row_id
                    for segment in route.segments
                )
            ),
            None,
        )
        if primary is None or len(candidate.owned_tiles) != 5:
            diagnostic["row_overload_resolved"] = False
            continue
        segment_index = next(
            index
            for index, segment in enumerate(primary.segments)
            if segment.physical_row_id == candidate.row_id
        )
        original_segment = primary.segments[segment_index]
        traversal_options = (
            original_segment.traversal,
            tuple(reversed(original_segment.traversal)),
        )
        primary_slots = (
            worker_action_slots.get(primary.owner, 0)
            if worker_action_slots is not None
            else remaining_action_slots
        )
        if primary_slots is None:
            diagnostic["row_overload_resolved"] = True
            diagnostic["helper_required"] = False
            continue
        candidates_for_split = []
        for suffix_size, traversal in (
            (size, oriented)
            for size in range(1, len(original_segment.traversal))
            for oriented in traversal_options
        ):
            prefix = traversal[:-suffix_size]
            suffix = traversal[-suffix_size:]
            prefix_candidate = _fragment_candidate(
                candidate, prefix, role="PRIMARY", boundary_index=len(prefix)
            )
            helper_candidate = _fragment_candidate(
                candidate,
                tuple(reversed(suffix)),
                role="HELPER",
                boundary_index=len(prefix),
            )
            for helper in helper_workers:
                helper_slots = (
                    worker_action_slots.get(helper, 0)
                    if worker_action_slots is not None
                    else remaining_action_slots
                )
                if helper_slots is None:
                    continue
                helper_traversal = tuple(reversed(suffix))
                primary_cost_segments = []
                for index, segment in enumerate(primary.segments):
                    if index == segment_index:
                        primary_cost_segments.append(
                            _candidate_cost_segment(prefix_candidate, prefix)
                        )
                        continue
                    prior_candidate = by_id.get(segment.segment_id)
                    if prior_candidate is None:
                        prior_candidate = next(
                            (
                                item
                                for item in ordered_candidates
                                if item.row_id == segment.physical_row_id
                            ),
                            None,
                        )
                    if prior_candidate is not None:
                        primary_cost_segments.append(
                            _candidate_cost_segment(prior_candidate, segment.traversal)
                        )
                helper_cost_segment = _candidate_cost_segment(
                    helper_candidate, helper_traversal
                )
                primary_cost, helper_cost = simulate_split(
                    primary,
                    tuple(primary_cost_segments),
                    helper,
                    helper_cost_segment,
                )
                prefix_result = next(
                    result
                    for result in primary_cost.segment_results
                    if result.segment_id == prefix_candidate.route_id
                )
                if (
                    primary_cost.resource_feasible
                    and prefix_result.resource_feasible
                    and helper_cost.resource_feasible
                    and prefix_result.hire_driving_interactions_missed == 0
                    and prefix_result.effective_interactions_missed == 0
                    and helper_cost.hire_driving_interactions_missed == 0
                    and helper_cost.effective_interactions_missed == 0
                    and helper_cost.feasible_hire_driving_interaction_turns > 0
                ):
                    primary_movement = (
                        primary_cost.setup_travel_turns
                        + primary_cost.pickup_travel_turns
                        + primary_cost.pickup_action_turns
                        + primary_cost.inter_segment_travel_turns
                        + primary_cost.horizontal_sweep_turns
                    )
                    helper_movement = (
                        helper_cost.setup_travel_turns
                        + helper_cost.pickup_travel_turns
                        + helper_cost.pickup_action_turns
                        + helper_cost.horizontal_sweep_turns
                    )
                    candidates_for_split.append(
                        (
                            suffix_size,
                            prefix_result.completion_elapsed_turns + helper_cost.total_turns,
                            primary_movement + helper_movement,
                            helper.index,
                            suffix,
                            helper,
                            prefix,
                            prefix_candidate,
                            helper_candidate,
                            helper_traversal,
                            helper_cost,
                            prefix_result,
                            primary_cost,
                        )
                    )
            if candidates_for_split:
                break
        if not candidates_for_split:
            diagnostic["row_overload_resolved"] = False
            continue
        selected = min(candidates_for_split)
        (
            _suffix_size,
            _completion_score,
            _travel_score,
            _worker_index,
            suffix,
            helper,
            prefix,
            prefix_candidate,
            helper_candidate,
            helper_traversal,
            helper_cost,
            primary_segment_cost,
            primary_route_cost,
        ) = selected
        prefix_segment = _fragment_segment(
            prefix_candidate,
            prefix,
            entry_distance=original_segment.entry_distance,
        )
        helper_segment = _fragment_segment(
            helper_candidate,
            helper_traversal,
            entry_distance=_manhattan_distance(
                worker_positions[helper], helper_traversal[0]
            ),
        )
        segments = list(primary.segments)
        segments[segment_index] = prefix_segment
        previous_end = prefix[-1]
        for index in range(segment_index + 1, len(segments)):
            segments[index] = replace(
                segments[index],
                entry_distance=_manhattan_distance(
                    previous_end, segments[index].entry_tile
                ),
            )
            previous_end = segments[index].traversal[-1]
        primary.segments = tuple(segments)
        primary.traversal = tuple(tile for value in segments for tile in value.traversal)
        primary.owned_tiles = primary.traversal
        primary.workload_interactions = sum(
            segment.represented_interactions for segment in segments
        )
        helper_route = StripRoute(
            route_id=helper_segment.segment_id,
            owned_tiles=helper_traversal,
            traversal=helper_traversal,
            owner=helper,
            entry_tile=helper_traversal[0],
            entry_distance=helper_segment.entry_distance,
            assignment_hour=assignment_hour,
            workload_interactions=helper_segment.represented_interactions,
            source_shape=helper_segment.source_shape,
            segments=(helper_segment,),
        )
        helper_routes.append(helper_route)
        helper_workers.remove(helper)
        diagnostic.update(
            {
                "primary_tiles": [list(tile) for tile in prefix],
                "primary_expected_completion_turn": assignment_hour
                + primary_segment_cost.completion_elapsed_turns,
                "helper_required": True,
                "helper_worker": helper.label,
                "helper_tiles": [list(tile) for tile in helper_traversal],
                "helper_expected_completion_turn": assignment_hour + helper_cost.total_turns,
                "row_overload_resolved": True,
                "canonical_cost": {
                    **(diagnostic.get("canonical_cost") or {}),
                    "primary_after_split": {
                        "total_turns": primary_route_cost.total_turns,
                        "route_complete_before_deadline": (
                            primary_route_cost.route_complete_before_deadline
                        ),
                        "effective_interactions_missed": (
                            primary_route_cost.effective_interactions_missed
                        ),
                        "supply_quantities_required": dict(
                            primary_route_cost.supply_quantities_required
                        ),
                        "supply_quantities_requiring_pickup": dict(
                            primary_route_cost.supply_quantities_requiring_pickup
                        ),
                        "supply_shortage": dict(primary_route_cost.supply_shortage),
                    },
                    "primary_fragment": primary_segment_cost.to_json_dict(),
                    "helper_fragment": helper_cost.to_json_dict(),
                },
            }
        )

    routes.extend(helper_routes)
    routes.sort(key=lambda route: route.owner)
    assigned_workers = {route.owner for route in routes}
    return replace(
        base,
        routes=tuple(routes),
        idle_workers=tuple(worker for worker in sorted(worker_positions) if worker not in assigned_workers),
        row_diagnostics=tuple(row_diagnostics),
    )
