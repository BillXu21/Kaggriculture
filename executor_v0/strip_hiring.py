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

from executor_v0.strip_routes import HorizontalRouteCandidate, WorkerId
from executor_v0.strip_supply import LOCAL_ACTION_PRIORITY
from executor_v0.strip_work import (
    BlockReason,
    StripWorkPlan,
    SupplyRequirement,
    WorkItem,
    WorkStatus,
)

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

    def copy(self) -> _SupplyLedger:
        return _SupplyLedger(dict(self.shed), dict(self.seeds))


def _positive_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(item): max(0, int(amount))
        for item, amount in value.items()
        if int(amount) > 0
    }


def _future_worker_actions(obs: Mapping[str, Any]) -> int:
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    configuration = obs.get("configuration")
    config = configuration if isinstance(configuration, Mapping) else {}
    turns_per_day = max(1, int(config.get("turnsPerDay", 24)))
    episode_steps = max(1, int(config.get("episodeSteps", 720)))
    step = int(obs.get("step", day * turns_per_day + hour))
    # A HIRE is processed after this turn's unit actions.  The terminal state
    # is not actionable, so only turns strictly after this one and before the
    # next reset/terminal boundary are available to the new hand.
    next_boundary = min((day + 1) * turns_per_day, episode_steps - 1)
    return max(0, next_boundary - step - 1)


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


def _is_fertilizer_only_item(
    item: WorkItem, fertilizer_item_ids: frozenset[str] = frozenset()
) -> bool:
    return item.id in fertilizer_item_ids or item.kind == "FERTILIZE" or (
        item.kind == "WATER" and item.source == "fertilizer_linked_productive"
    )


def _ordered_route_items(
    candidate: HorizontalRouteCandidate,
    work_plan: StripWorkPlan,
    position: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], tuple[WorkItem, ...]]:
    left_to_right = candidate.owned_tiles
    left, right = left_to_right[0], left_to_right[-1]
    left_distance = abs(position[0] - left[0]) + abs(position[1] - left[1])
    right_distance = abs(position[0] - right[0]) + abs(position[1] - right[1])
    traversal = (
        left_to_right if left_distance <= right_distance else tuple(reversed(left_to_right))
    )
    tile_rank = {tile: index for index, tile in enumerate(traversal)}
    items = tuple(
        sorted(
            (
                item
                for item in work_plan.items
                if item.tile in tile_rank and item.kind in LOCAL_ACTION_PRIORITY
            ),
            key=lambda item: (
                tile_rank[item.tile],
                LOCAL_ACTION_PRIORITY[item.kind],
                item.id,
            ),
        )
    )
    return traversal, items


def _consume_requirements(
    item: WorkItem,
    ledger: _SupplyLedger,
    carried: dict[str, int],
) -> bool:
    trial = ledger.copy()
    trial_carried = dict(carried)
    requirements = list(item.required_supplies)
    declared = {(requirement.item, requirement.scope) for requirement in requirements}
    if item.kind == "PLANT" and item.crop and (item.crop, "global_seed") not in declared:
        requirements.append(SupplyRequirement(item.crop, 1, "global_seed"))
    if item.kind == "FEED" and ("WHEAT", "inventory") not in declared:
        requirements.append(SupplyRequirement("WHEAT", max(1, item.quantity), "inventory"))
    if item.kind == "PLACE" and item.animal and (item.animal, "inventory") not in declared:
        requirements.append(SupplyRequirement(item.animal, max(1, item.quantity), "inventory"))
    for requirement in requirements:
        need = max(0, int(requirement.quantity))
        if requirement.scope == "global_seed":
            available = trial.seeds.get(requirement.item, 0)
            if available < need:
                return False
            trial.seeds[requirement.item] = available - need
            continue
        from_worker = min(need, trial_carried.get(requirement.item, 0))
        trial_carried[requirement.item] = (
            trial_carried.get(requirement.item, 0) - from_worker
        )
        need -= from_worker
        available = trial.shed.get(requirement.item, 0)
        if available < need:
            return False
        trial.shed[requirement.item] = available - need
    ledger.shed, ledger.seeds = trial.shed, trial.seeds
    carried.clear()
    carried.update(trial_carried)
    return True


def _item_can_progress(
    item: WorkItem,
    feasible_ids: set[str],
    ledger: _SupplyLedger,
    carried: dict[str, int],
) -> bool:
    if item.depends_on and not all(dependency in feasible_ids for dependency in item.depends_on):
        return False
    if item.status != WorkStatus.READY and item.block_reason not in {
        BlockReason.DEPENDENCY_BLOCKED,
        BlockReason.MISSING_SUPPLY,
        BlockReason.MISSING_GLOBAL_RESOURCE,
        BlockReason.MISSING_PURCHASE,
    }:
        return False
    if item.block_reason == BlockReason.MISSING_PURCHASE:
        if item.kind != "PLACE" or not item.animal:
            return False
        # Packet 1 may omit PLACE's inventory requirement while waiting for a
        # purchase; _consume_requirements adds the exact implicit animal check.
    return _consume_requirements(item, ledger, carried)


def _estimate_route(
    candidate: HorizontalRouteCandidate,
    route_index: int,
    position: tuple[int, int],
    work_plan: StripWorkPlan,
    ledger: _SupplyLedger,
    carried: Mapping[str, int],
    action_slots: int,
    fertilizer_item_ids: frozenset[str],
) -> RouteLaborEstimate:
    traversal, items = _ordered_route_items(candidate, work_plan, position)
    route_carried = _positive_counts(carried)
    feasible_ids: set[str] = set()
    feasible: list[WorkItem] = []
    for item in items:
        if _item_can_progress(item, feasible_ids, ledger, route_carried):
            feasible_ids.add(item.id)
            feasible.append(item)

    driving = [
        item for item in feasible if not _is_fertilizer_only_item(item, fertilizer_item_ids)
    ]
    represented_driving = [
        item
        for item in items
        if not _is_fertilizer_only_item(item, fertilizer_item_ids)
    ]
    fertilizer_only = bool(items) and not represented_driving
    first = driving[0] if driving else None
    first_use_eta: int | None = None
    if first is not None and first.tile is not None:
        entry = traversal[0]
        entry_travel = abs(position[0] - entry[0]) + abs(position[1] - entry[1])
        sweep_travel = traversal.index(first.tile)
        pickup_items = {
            requirement.item
            for requirement in first.required_supplies
            if requirement.scope != "global_seed"
        }
        first_use_eta = entry_travel + sweep_travel + len(pickup_items) + 1

    inventory_items = {
        requirement.item
        for item in items
        for requirement in item.required_supplies
        if requirement.scope != "global_seed"
    }
    entry = traversal[0]
    entry_travel = abs(position[0] - entry[0]) + abs(position[1] - entry[1])
    estimated_full_turns = (
        entry_travel + max(0, len(traversal) - 1) + len(items) + len(inventory_items)
    )
    useful = first_use_eta is not None and first_use_eta <= action_slots
    reasons: list[str] = []
    if not represented_driving:
        reasons.append("FERTILIZER_ONLY" if fertilizer_only else "NO_HIRE_DRIVING_WORK")
    elif first is None:
        reasons.append("NO_MECHANICALLY_FEASIBLE_HIRE_DRIVING_WORK")
    elif not useful:
        reasons.append("FIRST_USE_AFTER_DEADLINE")
    return RouteLaborEstimate(
        route_id=candidate.route_id,
        route_index=route_index,
        hire_driving=bool(driving),
        fertilizer_only=fertilizer_only,
        first_use_eta=first_use_eta,
        estimated_full_turns=estimated_full_turns,
        future_action_slots=action_slots,
        useful_before_deadline=useful,
        route_overloaded=bool(items) and estimated_full_turns > action_slots,
        first_use_work_id=first.id if first else None,
        reasons=tuple(reasons),
    )


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
        max(0, len(candidates) - current_workers),
        board_size,
    )
    ledger = _SupplyLedger(
        _positive_counts(private.get("shed")), _positive_counts(private.get("seeds"))
    )
    fertilizer_item_ids = frozenset(
        item_id
        for chain in work_plan.chains
        if chain.kind == "FERTILIZER_UPKEEP" or chain.source == "fertilizer_policy"
        for item_id in chain.item_ids
    )
    estimates: list[RouteLaborEstimate] = []
    for index, candidate in enumerate(candidates):
        if index < current_workers:
            worker = observed_workers[index]
            position = worker_positions[worker]
            inventory = worker_inventories.get(worker, {})
            slots = future_slots + 1
        else:
            position = predicted_spawns[index - current_workers]
            inventory = {}
            slots = future_slots
        estimates.append(
            _estimate_route(
                candidate,
                index,
                position,
                work_plan,
                ledger,
                inventory,
                slots,
                fertilizer_item_ids,
            )
        )

    useful_indices = [
        estimate.route_index
        for estimate in estimates
        if estimate.hire_driving and estimate.useful_before_deadline
    ]
    target_workers = useful_indices[-1] + 1 if useful_indices else 0
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

    if not any(estimate.hire_driving for estimate in estimates):
        stop = HireStopReason.NO_HIRE_DRIVING_WORK
    elif wanted == 0:
        stop = HireStopReason.COVERED if useful_indices else HireStopReason.TIME
    elif affordable == 0 or affordable < wanted:
        stop = HireStopReason.CASH
    elif submittable < wanted:
        stop = HireStopReason.ORDER_CAP
    else:
        stop = HireStopReason.COVERED
    return StripHiringPlan(
        current_workers=current_workers,
        target_workers=target_workers,
        wanted_hires=wanted,
        affordable_hires=affordable,
        submittable_hires=submittable,
        sequential_hire_costs=costs,
        coverage_prefix=tuple(candidate.route_id for candidate in candidates[:target_workers]),
        route_estimates=tuple(estimates),
        cash_before_hiring=cash,
        future_action_slots=future_slots,
        stop_reason=stop,
    )
