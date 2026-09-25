"""Pure deterministic cost simulation for strip routes.

``total_turns`` is the number of primitive worker turns consumed after the
assignment observation.  A route whose cost equals ``remaining_action_slots``
finishes; one whose cost is one greater does not.  Inventory pickup is planned
once, before the first segment, matching :mod:`executor_v0.strip_supply`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from executor_v0.foreman import SHED_ACCESS_TILES
from executor_v0.strip_work import (
    BlockReason,
    SupplyRequirement,
    WorkItem,
    WorkStatus,
    forecast_effective_interactions,
)

__all__ = [
    "LOCAL_ACTION_PRIORITY",
    "RouteCostResult",
    "RouteCostSegment",
    "RouteCostWork",
    "SegmentCostResult",
    "inventory_requirements",
    "nearest_shed_access",
    "ordered_inventory_demand",
    "ordered_route_items",
    "route_cost_segment_from_items",
    "simulate_route_cost",
]


LOCAL_ACTION_PRIORITY = {
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


def _pairs(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (str(item), int(quantity))
            for item, quantity in values.items()
            if int(quantity) > 0
        )
    )


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def nearest_shed_access(position: tuple[int, int]) -> tuple[int, int]:
    """Return the execution shed tile, with its stable tie break."""

    return min(SHED_ACCESS_TILES, key=lambda tile: (_distance(position, tile), tile))


def inventory_requirements(item: WorkItem) -> tuple[SupplyRequirement, ...]:
    """Return declared inventory demand plus retained legacy compatibility.

    Current strip work declares FEED and PLACE demand explicitly.  The fallback
    keeps direct/test-created ``WorkItem`` values compatible, and is shared by
    both the supply planner and simulator so it cannot create divergent demand.
    """

    requirements = [
        requirement
        for requirement in item.required_supplies
        if requirement.scope == "inventory" and requirement.quantity > 0
    ]
    declared = {requirement.item for requirement in requirements}
    if item.kind == "FEED" and "WHEAT" not in declared:
        requirements.append(SupplyRequirement("WHEAT", max(1, item.quantity)))
    if item.kind == "PLACE" and item.animal and item.animal not in declared:
        requirements.append(
            SupplyRequirement(item.animal, max(1, item.quantity), "inventory")
        )
    return tuple(requirements)


def ordered_route_items(
    items: Iterable[WorkItem], traversal: Sequence[tuple[int, int]]
) -> tuple[WorkItem, ...]:
    """Order authoritative work exactly as the strip sweep consumes it."""

    tile_rank = {tile: index for index, tile in enumerate(traversal)}
    return tuple(
        sorted(
            (
                item
                for item in items
                if item.tile in tile_rank
            ),
            key=lambda item: (
                tile_rank[item.tile],
                LOCAL_ACTION_PRIORITY.get(item.kind, 1000),
                item.id,
            ),
        )
    )


def ordered_inventory_demand(
    items: Iterable[WorkItem],
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Return quantity demand and stable first-use order for ordered work."""

    demand: dict[str, int] = defaultdict(int)
    first_use: list[str] = []
    seen: set[str] = set()
    for item in items:
        for requirement in sorted(
            inventory_requirements(item), key=lambda value: (value.item, value.quantity)
        ):
            demand[requirement.item] += int(requirement.quantity)
            if requirement.item not in seen:
                seen.add(requirement.item)
                first_use.append(requirement.item)
    return _pairs(demand), tuple(first_use)


@dataclass(frozen=True)
class RouteCostWork:
    work_id: str
    stage: str
    represented_turns: int
    continuation_turns: int
    hire_driving: bool
    inventory_requirements: tuple[tuple[str, int], ...]
    global_requirements: tuple[tuple[str, int], ...]
    continuation_global_requirements: tuple[tuple[str, int], ...]
    depends_on: tuple[str, ...]
    status: WorkStatus
    block_reason: BlockReason | None

    @property
    def effective_turns(self) -> int:
        return self.represented_turns + self.continuation_turns


@dataclass(frozen=True)
class RouteCostSegment:
    """Compact immutable forecast for one ordered physical route segment."""

    segment_id: str
    traversal: tuple[tuple[int, int], ...]
    work_by_tile: tuple[tuple[RouteCostWork, ...], ...]
    physical_row_id: str | None = None

    def __post_init__(self) -> None:
        if not self.traversal:
            raise ValueError("cost segment traversal must not be empty")
        if len(self.work_by_tile) != len(self.traversal):
            raise ValueError("cost segment work must align with traversal")


@dataclass(frozen=True)
class _CostSegmentSummary:
    work_items: tuple[RouteCostWork, ...]
    inventory_demand: tuple[tuple[str, int], ...]
    inventory_first_use_order: tuple[str, ...]
    global_demand: tuple[tuple[str, int], ...]
    represented_interactions: int
    known_continuation_interactions: int
    hire_driving_interactions: int
    statuses_resource_feasible: bool
    tile_movement_turns: tuple[int, ...]
    horizontal_sweep_turns: int


@lru_cache(maxsize=16384)
def _summarize_cost_segment(segment: RouteCostSegment) -> _CostSegmentSummary:
    work_items = tuple(
        work for tile_work in segment.work_by_tile for work in tile_work
    )
    inventory_demand: dict[str, int] = defaultdict(int)
    global_demand: dict[str, int] = defaultdict(int)
    inventory_first_use_order: list[str] = []
    seen_inventory: set[str] = set()
    for work in work_items:
        for item, quantity in work.inventory_requirements:
            inventory_demand[item] += quantity
            if item not in seen_inventory:
                seen_inventory.add(item)
                inventory_first_use_order.append(item)
        for item, quantity in work.global_requirements:
            global_demand[item] += quantity
        for item, quantity in work.continuation_global_requirements:
            global_demand[item] += quantity
    tile_movement_turns = (0,) + tuple(
        _distance(left, right)
        for left, right in zip(segment.traversal, segment.traversal[1:])
    )
    return _CostSegmentSummary(
        work_items=work_items,
        inventory_demand=_pairs(inventory_demand),
        inventory_first_use_order=tuple(inventory_first_use_order),
        global_demand=_pairs(global_demand),
        represented_interactions=sum(work.represented_turns for work in work_items),
        known_continuation_interactions=sum(
            work.continuation_turns for work in work_items
        ),
        hire_driving_interactions=sum(
            work.effective_turns for work in work_items if work.hire_driving
        ),
        statuses_resource_feasible=all(
            work.status == WorkStatus.READY
            or work.block_reason
            in {
                BlockReason.DEPENDENCY_BLOCKED,
                BlockReason.MISSING_SUPPLY,
                BlockReason.MISSING_GLOBAL_RESOURCE,
                BlockReason.MISSING_PURCHASE,
            }
            for work in work_items
        ),
        tile_movement_turns=tile_movement_turns,
        horizontal_sweep_turns=sum(tile_movement_turns),
    )


@dataclass(frozen=True)
class SegmentCostResult:
    segment_id: str
    physical_row_id: str | None
    start_position: tuple[int, int]
    end_position: tuple[int, int]
    start_elapsed_turns: int
    arrival_elapsed_turns: int
    completion_elapsed_turns: int
    setup_travel_turns: int
    pickup_travel_turns: int
    pickup_action_turns: int
    inter_segment_travel_turns: int
    horizontal_sweep_turns: int
    movement_turns: int
    represented_interaction_turns: int
    known_continuation_turns: int
    effective_interaction_turns: int
    feasible_effective_interaction_turns: int
    feasible_hire_driving_interaction_turns: int
    effective_interactions_completed_before_deadline: int
    effective_interactions_missed: int
    hire_driving_interactions_completed_before_deadline: int
    hire_driving_interactions_missed: int
    resource_feasible: bool
    complete_before_deadline: bool
    first_interaction_turn: int | None
    first_use_turn: int | None
    first_use_work_id: str | None
    first_use_stage: str | None
    first_use_preceding_interaction_turns: int
    first_use_movement_turns: int
    first_unfinished_tile: tuple[int, int] | None = None
    first_unfinished_work: str | None = None
    first_unfinished_stage: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteCostResult:
    """Authoritative deterministic timing and feasibility for one route."""

    total_turns: int
    completion_turn: int
    completion_hour: int
    end_position: tuple[int, int]
    setup_travel_turns: int
    pickup_travel_turns: int
    pickup_action_turns: int
    inter_segment_travel_turns: int
    horizontal_sweep_turns: int
    represented_interaction_turns: int
    known_continuation_turns: int
    effective_interaction_turns: int
    feasible_effective_interaction_turns: int
    hire_driving_interaction_turns: int
    feasible_hire_driving_interaction_turns: int
    effective_interactions_completed_before_deadline: int
    effective_interactions_missed: int
    hire_driving_interactions_completed_before_deadline: int
    hire_driving_interactions_missed: int
    segments_completed_before_deadline: int
    first_unfinished_segment: str | None
    first_unfinished_tile: tuple[int, int] | None
    first_unfinished_work: str | None
    first_unfinished_stage: str | None
    timing_complete_before_deadline: bool
    route_complete_before_deadline: bool
    supply_quantities_required: tuple[tuple[str, int], ...]
    supply_quantities_already_carried: tuple[tuple[str, int], ...]
    supply_quantities_requiring_pickup: tuple[tuple[str, int], ...]
    supply_shortage: tuple[tuple[str, int], ...]
    pickup_sequence: tuple[tuple[str, int], ...]
    pickup_tile: tuple[int, int] | None
    resource_feasible: bool
    first_interaction_turn: int | None
    first_use_turn: int | None
    first_hire_driving_work_id: str | None
    first_hire_driving_stage: str | None
    first_hire_driving_preceding_interaction_turns: int
    first_hire_driving_movement_turns: int
    global_quantities_required: tuple[tuple[str, int], ...]
    global_quantities_consumed: tuple[tuple[str, int], ...]
    global_shortage: tuple[tuple[str, int], ...]
    segment_results: tuple[SegmentCostResult, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "supply_quantities_required",
            "supply_quantities_already_carried",
            "supply_quantities_requiring_pickup",
            "supply_shortage",
        ):
            payload[key] = dict(getattr(self, key))
        payload["pickup_sequence"] = [
            {"item": item, "quantity": quantity}
            for item, quantity in self.pickup_sequence
        ]
        payload["segment_results"] = [
            result.to_json_dict() for result in self.segment_results
        ]
        return payload


def route_cost_segment_from_items(
    segment_id: str,
    traversal: Sequence[tuple[int, int]],
    items: Iterable[WorkItem],
    *,
    physical_row_id: str | None = None,
    fertilizer_item_ids: frozenset[str] = frozenset(),
) -> RouteCostSegment:
    """Prepare the compact simulator input from authoritative work items."""

    ordered_traversal = tuple(traversal)
    ordered_items = ordered_route_items(items, ordered_traversal)
    continuation_stages = dict(
        forecast_effective_interactions(ordered_items).continuation_stages_by_work_item
    )
    by_tile: dict[tuple[int, int], list[RouteCostWork]] = defaultdict(list)
    for item in ordered_items:
        if item.tile is None:
            continue
        implicit_continuation = continuation_stages.get(item.id, ())
        continuation = len(implicit_continuation)
        global_requirements = tuple(
            sorted(
                (requirement.item, int(requirement.quantity))
                for requirement in item.required_supplies
                if requirement.scope == "global_seed" and requirement.quantity > 0
            )
        )
        continuation_global = (
            ((item.crop, 1),)
            if "PLANT" in implicit_continuation and item.crop
            else ()
        )
        inventory = defaultdict(int)
        for requirement in inventory_requirements(item):
            inventory[requirement.item] += int(requirement.quantity)
        fertilizer_only = item.id in fertilizer_item_ids or item.kind == "FERTILIZE" or (
            item.kind == "WATER" and item.source == "fertilizer_linked_productive"
        )
        by_tile[item.tile].append(
            RouteCostWork(
                work_id=item.id,
                stage=item.kind,
                represented_turns=max(0, int(item.interaction_turns)),
                continuation_turns=max(0, continuation),
                hire_driving=not fertilizer_only,
                inventory_requirements=_pairs(inventory),
                global_requirements=global_requirements,
                continuation_global_requirements=continuation_global,
                depends_on=item.depends_on,
                status=item.status,
                block_reason=item.block_reason,
            )
        )
    return RouteCostSegment(
        segment_id=segment_id,
        traversal=ordered_traversal,
        work_by_tile=tuple(tuple(by_tile[tile]) for tile in ordered_traversal),
        physical_row_id=physical_row_id,
    )


def route_cost_segment_from_forecast(
    segment_id: str,
    traversal: Sequence[tuple[int, int]],
    represented_interactions: Sequence[int],
    continuation_interactions: Sequence[int] = (),
    *,
    physical_row_id: str | None = None,
    hire_driving_interactions: Sequence[int] = (),
    inventory_items_by_tile: Sequence[Sequence[str]] = (),
) -> RouteCostSegment:
    """Build the compatibility forecast form used by older route callers."""

    ordered_traversal = tuple(traversal)
    if len(represented_interactions) != len(ordered_traversal):
        raise ValueError("represented interactions must align with traversal")
    continuation = (
        tuple(continuation_interactions)
        if len(continuation_interactions) == len(ordered_traversal)
        else (0,) * len(ordered_traversal)
    )
    driving = (
        tuple(hire_driving_interactions)
        if len(hire_driving_interactions) == len(ordered_traversal)
        else tuple(represented_interactions)
    )
    inventory_by_tile = (
        tuple(tuple(values) for values in inventory_items_by_tile)
        if len(inventory_items_by_tile) == len(ordered_traversal)
        else ((),) * len(ordered_traversal)
    )
    work_by_tile = []
    for index, tile in enumerate(ordered_traversal):
        inventory = tuple((item, 1) for item in inventory_by_tile[index])
        count = max(0, int(represented_interactions[index]))
        tail = max(0, int(continuation[index]))
        driving_total = min(count + tail, max(0, int(driving[index])))
        driving_represented = min(count, driving_total)
        driving_continuation = min(tail, driving_total - driving_represented)
        other_represented = count - driving_represented
        other_continuation = tail - driving_continuation
        driving_turns = driving_represented + driving_continuation
        other_turns = other_represented + other_continuation
        tile_work = []
        if driving_turns or (inventory and not other_turns):
            tile_work.append(
                RouteCostWork(
                    work_id=f"{segment_id}:{tile[0]}:{tile[1]}:forecast:driving",
                    stage="FORECAST",
                    represented_turns=driving_represented,
                    continuation_turns=driving_continuation,
                    hire_driving=True,
                    inventory_requirements=inventory,
                    global_requirements=(),
                    continuation_global_requirements=(),
                    depends_on=(),
                    status=WorkStatus.READY,
                    block_reason=None,
                )
            )
            inventory = ()
        if other_turns or inventory:
            tile_work.append(
                RouteCostWork(
                    work_id=f"{segment_id}:{tile[0]}:{tile[1]}:forecast:other",
                    stage="FORECAST",
                    represented_turns=other_represented,
                    continuation_turns=other_continuation,
                    hire_driving=False,
                    inventory_requirements=inventory,
                    global_requirements=(),
                    continuation_global_requirements=(),
                    depends_on=(),
                    status=WorkStatus.READY,
                    block_reason=None,
                )
            )
        work_by_tile.append(tuple(tile_work))
    return RouteCostSegment(
        segment_id=segment_id,
        traversal=ordered_traversal,
        work_by_tile=tuple(work_by_tile),
        physical_row_id=physical_row_id,
    )


def _work_can_progress(
    work: RouteCostWork,
    feasible_ids: set[str],
    inventory: dict[str, int],
    global_resources: dict[str, int] | None,
) -> bool:
    if (
        work.status == WorkStatus.READY
        and not work.depends_on
        and not work.inventory_requirements
        and not work.global_requirements
    ):
        return True
    if work.depends_on and not all(value in feasible_ids for value in work.depends_on):
        return False
    if work.status != WorkStatus.READY and work.block_reason not in {
        BlockReason.DEPENDENCY_BLOCKED,
        BlockReason.MISSING_SUPPLY,
        BlockReason.MISSING_GLOBAL_RESOURCE,
        BlockReason.MISSING_PURCHASE,
    }:
        return False
    inventory_trial = dict(inventory)
    for item, quantity in work.inventory_requirements:
        if inventory_trial.get(item, 0) < quantity:
            return False
        inventory_trial[item] -= quantity
    global_trial = None if global_resources is None else dict(global_resources)
    if global_trial is not None:
        for item, quantity in work.global_requirements:
            if global_trial.get(item, 0) < quantity:
                return False
            global_trial[item] -= quantity
    inventory.clear()
    inventory.update(inventory_trial)
    if global_resources is not None and global_trial is not None:
        global_resources.clear()
        global_resources.update(global_trial)
    return True


def _simulate_unconstrained_route_cost(
    start_position: tuple[int, int],
    segments: Sequence[RouteCostSegment],
    summaries: Sequence[_CostSegmentSummary],
    *,
    remaining_action_slots: int,
    assignment_turn: int,
    include_segment_results: bool,
) -> RouteCostResult:
    """Fast exact path when all forecast work is resource-free and ready."""

    setup_travel = _distance(start_position, segments[0].traversal[0])
    elapsed = setup_travel
    movement_elapsed = setup_travel
    position = segments[0].traversal[0]
    represented = sum(summary.represented_interactions for summary in summaries)
    continuations = sum(
        summary.known_continuation_interactions for summary in summaries
    )
    effective = represented + continuations
    hire_driving = sum(summary.hire_driving_interactions for summary in summaries)
    completed = 0
    completed_hire_driving = 0
    inter_segment = 0
    sweep = 0
    segments_completed = 0
    first_use_turn: int | None = None
    first_interaction_turn: int | None = None
    first_use_work_id: str | None = None
    first_use_stage: str | None = None
    first_use_preceding_interactions = 0
    first_use_movement = 0
    first_unfinished: tuple[str, tuple[int, int], str, str] | None = None
    segment_results: list[SegmentCostResult] = []

    for segment_index, (segment, summary) in enumerate(
        zip(segments, summaries, strict=True)
    ):
        segment_start_position = start_position if segment_index == 0 else position
        segment_start_elapsed = 0 if segment_index == 0 else elapsed
        segment_movement_start = movement_elapsed
        travel = 0
        if segment_index:
            travel = _distance(position, segment.traversal[0])
            inter_segment += travel
            elapsed += travel
            movement_elapsed += travel
            position = segment.traversal[0]
        arrival = elapsed
        segment_completed = 0
        segment_hire_driving_completed = 0
        segment_first_interaction_turn: int | None = None
        segment_first_use_turn: int | None = None
        segment_first_use_work_id: str | None = None
        segment_first_use_stage: str | None = None
        segment_preceding_interactions = 0
        segment_first_use_movement = 0
        segment_first_unfinished: tuple[tuple[int, int], str, str] | None = None
        for tile_index, (tile, tile_work) in enumerate(
            zip(segment.traversal, segment.work_by_tile, strict=True)
        ):
            if tile_index:
                movement = summary.tile_movement_turns[tile_index]
                sweep += movement
                elapsed += movement
                movement_elapsed += movement
                position = tile
            for work in tile_work:
                work_turns = work.effective_turns
                if not work_turns:
                    continue
                work_start_elapsed = elapsed
                first_turn = work_start_elapsed + 1
                first_stage = (
                    work.stage
                    if work.represented_turns > 0
                    else "CONTINUATION"
                )
                if first_interaction_turn is None:
                    first_interaction_turn = first_turn
                if segment_first_interaction_turn is None:
                    segment_first_interaction_turn = first_turn
                if work.hire_driving and segment_first_use_turn is None:
                    segment_first_use_turn = first_turn
                    segment_first_use_work_id = work.work_id
                    segment_first_use_stage = first_stage
                    segment_first_use_movement = movement_elapsed
                    segment_preceding_interactions = max(
                        0, work_start_elapsed - movement_elapsed
                    )
                if first_use_turn is None and work.hire_driving:
                    first_use_turn = first_turn
                    first_use_work_id = work.work_id
                    first_use_stage = first_stage
                    first_use_movement = movement_elapsed
                    first_use_preceding_interactions = max(
                        0, work_start_elapsed - movement_elapsed
                    )
                completed_here = min(
                    work_turns,
                    max(0, remaining_action_slots - work_start_elapsed),
                )
                completed += completed_here
                segment_completed += completed_here
                if work.hire_driving:
                    completed_hire_driving += completed_here
                    segment_hire_driving_completed += completed_here
                if completed_here < work_turns:
                    unfinished_stage = (
                        work.stage
                        if completed_here < work.represented_turns
                        else "CONTINUATION"
                    )
                    unfinished = (tile, work.work_id, unfinished_stage)
                    if segment_first_unfinished is None:
                        segment_first_unfinished = unfinished
                    if first_unfinished is None:
                        first_unfinished = (segment.segment_id, *unfinished)
                elapsed += work_turns
        segment_complete = (
            segment_completed == summary.represented_interactions
            + summary.known_continuation_interactions
            and elapsed <= remaining_action_slots
        )
        segments_completed += int(segment_complete)
        if include_segment_results:
            segment_results.append(
                SegmentCostResult(
                segment_id=segment.segment_id,
                physical_row_id=segment.physical_row_id,
                start_position=segment_start_position,
                end_position=position,
                start_elapsed_turns=segment_start_elapsed,
                arrival_elapsed_turns=arrival,
                completion_elapsed_turns=elapsed,
                setup_travel_turns=setup_travel if segment_index == 0 else 0,
                pickup_travel_turns=0,
                pickup_action_turns=0,
                inter_segment_travel_turns=travel,
                horizontal_sweep_turns=summary.horizontal_sweep_turns,
                movement_turns=movement_elapsed - segment_movement_start,
                represented_interaction_turns=summary.represented_interactions,
                known_continuation_turns=summary.known_continuation_interactions,
                effective_interaction_turns=(
                    summary.represented_interactions
                    + summary.known_continuation_interactions
                ),
                feasible_effective_interaction_turns=(
                    summary.represented_interactions
                    + summary.known_continuation_interactions
                ),
                feasible_hire_driving_interaction_turns=(
                    summary.hire_driving_interactions
                ),
                effective_interactions_completed_before_deadline=segment_completed,
                effective_interactions_missed=max(
                    0,
                    summary.represented_interactions
                    + summary.known_continuation_interactions
                    - segment_completed,
                ),
                hire_driving_interactions_completed_before_deadline=(
                    segment_hire_driving_completed
                ),
                hire_driving_interactions_missed=max(
                    0,
                    summary.hire_driving_interactions
                    - segment_hire_driving_completed,
                ),
                resource_feasible=True,
                complete_before_deadline=segment_complete,
                first_interaction_turn=segment_first_interaction_turn,
                first_use_turn=segment_first_use_turn,
                first_use_work_id=segment_first_use_work_id,
                first_use_stage=segment_first_use_stage,
                first_use_preceding_interaction_turns=(
                    segment_preceding_interactions
                ),
                first_use_movement_turns=segment_first_use_movement,
                first_unfinished_tile=(
                    segment_first_unfinished[0]
                    if segment_first_unfinished is not None
                    else None
                ),
                first_unfinished_work=(
                    segment_first_unfinished[1]
                    if segment_first_unfinished is not None
                    else None
                ),
                first_unfinished_stage=(
                    segment_first_unfinished[2]
                    if segment_first_unfinished is not None
                    else None
                ),
                )
            )

    timing_complete = elapsed <= remaining_action_slots
    if first_unfinished is None and not timing_complete:
        last = segments[-1]
        first_unfinished = (
            last.segment_id,
            last.traversal[-1],
            "",
            "ROUTE_COMPLETION",
        )
    global_required: dict[str, int] = defaultdict(int)
    for summary in summaries:
        for item, quantity in summary.global_demand:
            global_required[item] += quantity
    return RouteCostResult(
        total_turns=elapsed,
        completion_turn=assignment_turn + elapsed,
        completion_hour=assignment_turn + elapsed,
        end_position=position,
        setup_travel_turns=setup_travel,
        pickup_travel_turns=0,
        pickup_action_turns=0,
        inter_segment_travel_turns=inter_segment,
        horizontal_sweep_turns=sweep,
        represented_interaction_turns=represented,
        known_continuation_turns=continuations,
        effective_interaction_turns=effective,
        feasible_effective_interaction_turns=effective,
        hire_driving_interaction_turns=hire_driving,
        feasible_hire_driving_interaction_turns=hire_driving,
        effective_interactions_completed_before_deadline=completed,
        effective_interactions_missed=max(0, effective - completed),
        hire_driving_interactions_completed_before_deadline=completed_hire_driving,
        hire_driving_interactions_missed=max(
            0, hire_driving - completed_hire_driving
        ),
        segments_completed_before_deadline=segments_completed,
        first_unfinished_segment=(first_unfinished[0] if first_unfinished else None),
        first_unfinished_tile=(first_unfinished[1] if first_unfinished else None),
        first_unfinished_work=(
            first_unfinished[2] or None if first_unfinished else None
        ),
        first_unfinished_stage=(first_unfinished[3] if first_unfinished else None),
        timing_complete_before_deadline=timing_complete,
        route_complete_before_deadline=timing_complete,
        supply_quantities_required=(),
        supply_quantities_already_carried=(),
        supply_quantities_requiring_pickup=(),
        supply_shortage=(),
        pickup_sequence=(),
        pickup_tile=None,
        resource_feasible=True,
        first_interaction_turn=first_interaction_turn,
        first_use_turn=first_use_turn,
        first_hire_driving_work_id=first_use_work_id,
        first_hire_driving_stage=first_use_stage,
        first_hire_driving_preceding_interaction_turns=(
            first_use_preceding_interactions
        ),
        first_hire_driving_movement_turns=first_use_movement,
        global_quantities_required=_pairs(global_required),
        global_quantities_consumed=(),
        global_shortage=(),
        segment_results=tuple(segment_results),
    )


def simulate_route_cost(
    start_position: tuple[int, int],
    segments: Sequence[RouteCostSegment],
    *,
    carried_inventory: Mapping[str, int] | None = None,
    remaining_action_slots: int,
    assignment_turn: int = 0,
    shed_stock: Mapping[str, int] | None = None,
    reserved_supply: Mapping[str, int] | None = None,
    global_resources: Mapping[str, int] | None = None,
    pickup_tile: tuple[int, int] | None = None,
    include_segment_results: bool = True,
) -> RouteCostResult:
    """Simulate one ordered route without mutating caller-owned state.

    ``shed_stock=None`` means stock is not authoritative for this comparison;
    all demand not already carried is assumed pickable, but its exact setup
    cost is still charged.  ``reserved_supply`` takes precedence when callers
    already have the authoritative non-overbooking result.

    ``include_segment_results=False`` keeps the same route-level accounting
    while omitting per-segment records for combinatorial packer scoring.
    """

    if remaining_action_slots < 0:
        raise ValueError("remaining_action_slots must be nonnegative")
    if not segments:
        return RouteCostResult(
            total_turns=0,
            completion_turn=assignment_turn,
            completion_hour=assignment_turn,
            end_position=start_position,
            setup_travel_turns=0,
            pickup_travel_turns=0,
            pickup_action_turns=0,
            inter_segment_travel_turns=0,
            horizontal_sweep_turns=0,
            represented_interaction_turns=0,
            known_continuation_turns=0,
            effective_interaction_turns=0,
            feasible_effective_interaction_turns=0,
            hire_driving_interaction_turns=0,
            feasible_hire_driving_interaction_turns=0,
            effective_interactions_completed_before_deadline=0,
            effective_interactions_missed=0,
            hire_driving_interactions_completed_before_deadline=0,
            hire_driving_interactions_missed=0,
            segments_completed_before_deadline=0,
            first_unfinished_segment=None,
            first_unfinished_tile=None,
            first_unfinished_work=None,
            first_unfinished_stage=None,
            timing_complete_before_deadline=True,
            route_complete_before_deadline=True,
            supply_quantities_required=(),
            supply_quantities_already_carried=(),
            supply_quantities_requiring_pickup=(),
            supply_shortage=(),
            pickup_sequence=(),
            pickup_tile=None,
            resource_feasible=True,
            first_interaction_turn=None,
            first_use_turn=None,
            first_hire_driving_work_id=None,
            first_hire_driving_stage=None,
            first_hire_driving_preceding_interaction_turns=0,
            first_hire_driving_movement_turns=0,
            global_quantities_required=(),
            global_quantities_consumed=(),
            global_shortage=(),
            segment_results=(),
        )

    summaries = tuple(_summarize_cost_segment(segment) for segment in segments)
    unconstrained = all(
        work.status == WorkStatus.READY
        and not work.depends_on
        and not work.inventory_requirements
        and not work.global_requirements
        and (global_resources is None or not work.continuation_global_requirements)
        for summary in summaries
        for work in summary.work_items
    ) and (
        global_resources is None
        or all(not summary.global_demand for summary in summaries)
    )
    if unconstrained:
        return _simulate_unconstrained_route_cost(
            start_position,
            segments,
            summaries,
            remaining_action_slots=remaining_action_slots,
            assignment_turn=assignment_turn,
            include_segment_results=include_segment_results,
        )
    demand: dict[str, int] = defaultdict(int)
    global_demand: dict[str, int] = defaultdict(int)
    first_use_order: list[str] = []
    seen_items: set[str] = set()
    for summary in summaries:
        for item, quantity in summary.inventory_demand:
            demand[item] += quantity
        for item in summary.inventory_first_use_order:
            if item not in seen_items:
                seen_items.add(item)
                first_use_order.append(item)
        for item, quantity in summary.global_demand:
            global_demand[item] += quantity
    carried_source = carried_inventory or {}
    carried = {
        item: min(quantity, max(0, int(carried_source.get(item, 0))))
        for item, quantity in demand.items()
    }
    pickup: dict[str, int] = {}
    shortage: dict[str, int] = {}
    for item in first_use_order:
        need = demand[item] - carried[item]
        if reserved_supply is not None:
            take = min(need, max(0, int(reserved_supply.get(item, 0))))
        elif shed_stock is None:
            take = need
        else:
            take = min(need, max(0, int(shed_stock.get(item, 0))))
        pickup[item] = take
        shortage[item] = need - take
    pickup_sequence = tuple(
        (item, pickup[item]) for item in first_use_order if pickup[item] > 0
    )
    selected_pickup_tile = pickup_tile
    if pickup_sequence and selected_pickup_tile is None:
        selected_pickup_tile = nearest_shed_access(start_position)

    elapsed = 0
    pickup_travel = 0
    setup_travel = 0
    movement_elapsed = 0
    position = start_position
    first_entry = segments[0].traversal[0]
    if pickup_sequence:
        assert selected_pickup_tile is not None
        pickup_travel = _distance(position, selected_pickup_tile)
        elapsed += pickup_travel + len(pickup_sequence)
        movement_elapsed += pickup_travel
        position = selected_pickup_tile
    setup_travel = _distance(position, first_entry)
    elapsed += setup_travel
    movement_elapsed += setup_travel
    position = first_entry

    available_inventory = {
        item: carried.get(item, 0) + pickup.get(item, 0) for item in demand
    }
    initial_global = (
        None
        if global_resources is None
        else {
            str(item): max(0, int(quantity))
            for item, quantity in global_resources.items()
        }
    )
    available_global = None if initial_global is None else dict(initial_global)
    feasible_ids: set[str] = set()
    represented = sum(summary.represented_interactions for summary in summaries)
    continuations = sum(
        summary.known_continuation_interactions for summary in summaries
    )
    effective = represented + continuations
    hire_driving = sum(summary.hire_driving_interactions for summary in summaries)
    feasible_effective = 0
    feasible_hire_driving = 0
    completed = 0
    completed_hire_driving = 0
    inter_segment = 0
    sweep = 0
    segments_completed = 0
    first_use_turn: int | None = None
    first_interaction_turn: int | None = None
    first_use_work_id: str | None = None
    first_use_stage: str | None = None
    first_use_preceding_interactions = 0
    first_use_movement = 0
    first_unfinished: tuple[str, tuple[int, int], str, str] | None = None
    segment_results: list[SegmentCostResult] = []

    for segment_index, segment in enumerate(segments):
        summary = summaries[segment_index]
        segment_start_position = start_position if segment_index == 0 else position
        segment_start_elapsed = 0 if segment_index == 0 else elapsed
        segment_setup_travel = setup_travel if segment_index == 0 else 0
        segment_pickup_travel = pickup_travel if segment_index == 0 else 0
        segment_pickup_actions = len(pickup_sequence) if segment_index == 0 else 0
        segment_movement_start = movement_elapsed
        travel = 0
        if segment_index:
            travel = _distance(position, segment.traversal[0])
            inter_segment += travel
            elapsed += travel
            movement_elapsed += travel
            position = segment.traversal[0]
        arrival = elapsed
        segment_represented = 0
        segment_continuations = 0
        segment_feasible = 0
        segment_completed = 0
        segment_hire_driving = 0
        segment_hire_driving_completed = 0
        segment_blocked = False
        segment_first_unfinished: tuple[tuple[int, int], str, str] | None = None
        segment_first_interaction_turn: int | None = None
        segment_first_use_turn: int | None = None
        segment_first_use_work_id: str | None = None
        segment_first_use_stage: str | None = None
        segment_preceding_interactions = 0
        segment_first_use_movement = 0
        for tile_index, (tile, tile_work) in enumerate(
            zip(segment.traversal, segment.work_by_tile, strict=True)
        ):
            if tile_index:
                movement = summary.tile_movement_turns[tile_index]
                sweep += movement
                elapsed += movement
                movement_elapsed += movement
                position = tile
            for work in tile_work:
                segment_represented += work.represented_turns
                segment_continuations += work.continuation_turns
                can_progress = _work_can_progress(
                    work, feasible_ids, available_inventory, available_global
                )
                if not can_progress:
                    segment_blocked = True
                    unfinished = (tile, work.work_id, work.stage)
                    if segment_first_unfinished is None:
                        segment_first_unfinished = unfinished
                    if first_unfinished is None:
                        first_unfinished = (segment.segment_id, *unfinished)
                    continue
                feasible_ids.add(work.work_id)
                feasible_turns = work.represented_turns
                continuation_feasible = True
                if (
                    available_global is not None
                    and work.continuation_global_requirements
                ):
                    continuation_feasible = all(
                        available_global.get(item, 0) >= quantity
                        for item, quantity in work.continuation_global_requirements
                    )
                if continuation_feasible:
                    if available_global is not None:
                        for item, quantity in work.continuation_global_requirements:
                            available_global[item] -= quantity
                    feasible_turns += work.continuation_turns
                elif work.continuation_turns:
                    segment_blocked = True
                    if segment_first_unfinished is None:
                        tile = segment.traversal[tile_index]
                        segment_first_unfinished = (tile, work.work_id, "PLANT")
                    if first_unfinished is None:
                        first_unfinished = (
                            segment.segment_id,
                            segment.traversal[tile_index],
                            work.work_id,
                            "PLANT",
                        )
                feasible_effective += feasible_turns
                segment_feasible += feasible_turns
                if work.hire_driving:
                    feasible_hire_driving += feasible_turns
                    segment_hire_driving += feasible_turns
                if feasible_turns:
                    work_start_elapsed = elapsed
                    first_turn = work_start_elapsed + 1
                    first_stage = (
                        work.stage
                        if work.represented_turns > 0
                        else "CONTINUATION"
                    )
                    if first_interaction_turn is None:
                        first_interaction_turn = first_turn
                    if segment_first_interaction_turn is None:
                        segment_first_interaction_turn = first_turn
                    if work.hire_driving and segment_first_use_turn is None:
                        segment_first_use_turn = first_turn
                        segment_first_use_work_id = work.work_id
                        segment_first_use_stage = first_stage
                        segment_first_use_movement = movement_elapsed
                        segment_preceding_interactions = max(
                            0,
                            work_start_elapsed
                            - movement_elapsed
                            - len(pickup_sequence),
                        )
                    if first_use_turn is None and work.hire_driving:
                        first_use_turn = first_turn
                        first_use_work_id = work.work_id
                        first_use_stage = first_stage
                        first_use_movement = movement_elapsed
                        first_use_preceding_interactions = max(
                            0,
                            work_start_elapsed
                            - movement_elapsed
                            - len(pickup_sequence),
                        )
                    completed_here = min(
                        feasible_turns,
                        max(0, remaining_action_slots - work_start_elapsed),
                    )
                    completed += completed_here
                    segment_completed += completed_here
                    if work.hire_driving:
                        completed_hire_driving += completed_here
                        segment_hire_driving_completed += completed_here
                    if completed_here < feasible_turns:
                        unfinished_stage = (
                            work.stage
                            if completed_here < work.represented_turns
                            else "CONTINUATION"
                        )
                        unfinished = (tile, work.work_id, unfinished_stage)
                        if segment_first_unfinished is None:
                            segment_first_unfinished = unfinished
                        if first_unfinished is None:
                            first_unfinished = (segment.segment_id, *unfinished)
                    elapsed += feasible_turns
        segment_effective = segment_represented + segment_continuations
        complete_before_deadline = (
            not segment_blocked
            and segment_completed == segment_feasible
            and elapsed <= remaining_action_slots
        )
        segments_completed += int(complete_before_deadline)
        if include_segment_results:
            segment_results.append(
                SegmentCostResult(
                segment_id=segment.segment_id,
                physical_row_id=segment.physical_row_id,
                start_position=segment_start_position,
                end_position=position,
                start_elapsed_turns=segment_start_elapsed,
                arrival_elapsed_turns=arrival,
                completion_elapsed_turns=elapsed,
                setup_travel_turns=segment_setup_travel,
                pickup_travel_turns=segment_pickup_travel,
                pickup_action_turns=segment_pickup_actions,
                inter_segment_travel_turns=travel,
                horizontal_sweep_turns=summary.horizontal_sweep_turns,
                movement_turns=(movement_elapsed - segment_movement_start),
                represented_interaction_turns=segment_represented,
                known_continuation_turns=segment_continuations,
                effective_interaction_turns=segment_effective,
                feasible_effective_interaction_turns=segment_feasible,
                feasible_hire_driving_interaction_turns=segment_hire_driving,
                effective_interactions_completed_before_deadline=segment_completed,
                effective_interactions_missed=max(0, segment_feasible - segment_completed),
                hire_driving_interactions_completed_before_deadline=(
                    segment_hire_driving_completed
                ),
                hire_driving_interactions_missed=max(
                    0, segment_hire_driving - segment_hire_driving_completed
                ),
                resource_feasible=not segment_blocked,
                complete_before_deadline=complete_before_deadline,
                first_interaction_turn=segment_first_interaction_turn,
                first_use_turn=segment_first_use_turn,
                first_use_work_id=segment_first_use_work_id,
                first_use_stage=segment_first_use_stage,
                first_use_preceding_interaction_turns=segment_preceding_interactions,
                first_use_movement_turns=segment_first_use_movement,
                first_unfinished_tile=(
                    segment_first_unfinished[0] if segment_first_unfinished else None
                ),
                first_unfinished_work=(
                    segment_first_unfinished[1] if segment_first_unfinished else None
                ),
                first_unfinished_stage=(
                    segment_first_unfinished[2] if segment_first_unfinished else None
                ),
                )
            )

    resource_feasible = (
        not any(shortage.values())
        and all(summary.statuses_resource_feasible for summary in summaries)
        and feasible_effective == effective
    )
    timing_complete = elapsed <= remaining_action_slots
    if first_unfinished is None and not timing_complete:
        last = segments[-1]
        first_unfinished = (
            last.segment_id,
            last.traversal[-1],
            "",
            "ROUTE_COMPLETION",
        )
    return RouteCostResult(
        total_turns=elapsed,
        completion_turn=assignment_turn + elapsed,
        completion_hour=assignment_turn + elapsed,
        end_position=position,
        setup_travel_turns=setup_travel,
        pickup_travel_turns=pickup_travel,
        pickup_action_turns=len(pickup_sequence),
        inter_segment_travel_turns=inter_segment,
        horizontal_sweep_turns=sweep,
        represented_interaction_turns=represented,
        known_continuation_turns=continuations,
        effective_interaction_turns=effective,
        feasible_effective_interaction_turns=feasible_effective,
        hire_driving_interaction_turns=hire_driving,
        feasible_hire_driving_interaction_turns=feasible_hire_driving,
        effective_interactions_completed_before_deadline=completed,
        effective_interactions_missed=max(0, feasible_effective - completed),
        hire_driving_interactions_completed_before_deadline=completed_hire_driving,
        hire_driving_interactions_missed=max(
            0, feasible_hire_driving - completed_hire_driving
        ),
        segments_completed_before_deadline=segments_completed,
        first_unfinished_segment=first_unfinished[0] if first_unfinished else None,
        first_unfinished_tile=first_unfinished[1] if first_unfinished else None,
        first_unfinished_work=first_unfinished[2] or None if first_unfinished else None,
        first_unfinished_stage=first_unfinished[3] if first_unfinished else None,
        timing_complete_before_deadline=timing_complete,
        route_complete_before_deadline=timing_complete and resource_feasible,
        supply_quantities_required=_pairs(demand),
        supply_quantities_already_carried=_pairs(carried),
        supply_quantities_requiring_pickup=_pairs(pickup),
        supply_shortage=_pairs(shortage),
        pickup_sequence=pickup_sequence,
        pickup_tile=(selected_pickup_tile if pickup_sequence else None),
        resource_feasible=resource_feasible,
        first_interaction_turn=first_interaction_turn,
        first_use_turn=first_use_turn,
        first_hire_driving_work_id=first_use_work_id,
        first_hire_driving_stage=first_use_stage,
        first_hire_driving_preceding_interaction_turns=first_use_preceding_interactions,
        first_hire_driving_movement_turns=first_use_movement,
        global_quantities_required=_pairs(global_demand),
        global_quantities_consumed=(
            _pairs(
                {
                    item: max(
                        0,
                        int(initial_global.get(item, 0))
                        - int(available_global.get(item, 0)),
                    )
                    for item in global_demand
                }
            )
            if initial_global is not None and available_global is not None
            else ()
        ),
        global_shortage=(
            _pairs(
                {
                    item: max(0, quantity - int(initial_global.get(item, 0)))
                    for item, quantity in global_demand.items()
                }
            )
            if initial_global is not None
            else ()
        ),
        segment_results=tuple(segment_results),
    )
