"""Packet 3 route-local carried-supply planning.

The planner consumes Packet 1 ``WorkItem.required_supplies`` after Packet 2
has fixed route ownership.  It never changes assignment or traversal and it
never plans seeds or market purchases.  Plans are immutable for the day;
``RouteSupplyState`` contains only observation-confirmed execution progress.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from executor_v0.strip_cost import (
    LOCAL_ACTION_PRIORITY,
    nearest_shed_access,
    ordered_inventory_demand,
    ordered_route_items,
)
from executor_v0.strip_routes import StripRoute, WorkerId
from executor_v0.strip_work import StripWorkPlan, WorkItem

__all__ = [
    "LOCAL_ACTION_PRIORITY",
    "PickupBatch",
    "PendingPickup",
    "RouteSupplyPlan",
    "RouteSupplyState",
    "build_route_supply_plans",
    "extract_route_supply_demand",
    "extract_tile_supply_demand",
]


def _pairs(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (str(item), int(amount))
            for item, amount in values.items()
            if int(amount) != 0
        )
    )


@dataclass(frozen=True)
class PickupBatch:
    """One item-level quantity pickup, ordered by first route use."""

    item: str
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("pickup quantity must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteSupplyPlan:
    """Immutable daily reservation for one already-assigned route."""

    route_id: str
    owner: WorkerId
    pickup_tile: tuple[int, int] | None
    demand: tuple[tuple[str, int], ...]
    already_carried: tuple[tuple[str, int], ...]
    reserved_from_shed: tuple[tuple[str, int], ...]
    missing_stock: tuple[tuple[str, int], ...]
    capacity_limited: tuple[tuple[str, int], ...]
    pickup_sequence: tuple[PickupBatch, ...]

    def __post_init__(self) -> None:
        buckets = (
            self.demand,
            self.already_carried,
            self.reserved_from_shed,
            self.missing_stock,
            self.capacity_limited,
        )
        if any(amount < 0 for bucket in buckets for _, amount in bucket):
            raise ValueError("supply-plan quantities must be nonnegative")
        demand = dict(self.demand)
        carried = dict(self.already_carried)
        reserved = dict(self.reserved_from_shed)
        missing = dict(self.missing_stock)
        limited = dict(self.capacity_limited)
        for item, amount in demand.items():
            reconciled = (
                carried.get(item, 0)
                + reserved.get(item, 0)
                + missing.get(item, 0)
                + limited.get(item, 0)
            )
            if amount != reconciled:
                raise ValueError(
                    f"unreconciled supply plan for {item}: {amount} != {reconciled}"
                )
        if bool(self.pickup_sequence) != (self.pickup_tile is not None):
            raise ValueError("pickup tile must exist exactly when pickup is planned")

    @property
    def requires_pickup(self) -> bool:
        return bool(self.pickup_sequence)

    @property
    def fully_supplied(self) -> bool:
        return not any(amount for _, amount in self.missing_stock + self.capacity_limited)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "owner": self.owner.label,
            "pickup_tile": list(self.pickup_tile) if self.pickup_tile else None,
            "demand": dict(self.demand),
            "already_carried": dict(self.already_carried),
            "reserved_from_shed": dict(self.reserved_from_shed),
            "missing_stock": dict(self.missing_stock),
            "capacity_limited": dict(self.capacity_limited),
            "pickup_sequence": [batch.to_json_dict() for batch in self.pickup_sequence],
        }


@dataclass(frozen=True)
class PendingPickup:
    item: str
    quantity: int
    inventory_before: int


@dataclass
class RouteSupplyState:
    """Small mutable state; acquisition changes only after observation."""

    acquired: dict[str, int] = field(default_factory=dict)
    failed_or_unfulfilled: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    pending: PendingPickup | None = None
    pickup_turns: int = 0
    travel_turns: int = 0
    remaining_at_completion: dict[str, int] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "acquired": dict(sorted(self.acquired.items())),
            "failed_or_unfulfilled": dict(sorted(self.failed_or_unfulfilled.items())),
            "attempts": dict(sorted(self.attempts.items())),
            "pending_pickup": asdict(self.pending) if self.pending else None,
            "pickup_turns": self.pickup_turns,
            "travel_turns_to_supply_point": self.travel_turns,
            "remaining_at_completion": (
                dict(sorted(self.remaining_at_completion.items()))
                if self.remaining_at_completion is not None
                else None
            ),
        }


def _ordered_route_items(route: StripRoute, work_plan: StripWorkPlan) -> tuple[WorkItem, ...]:
    return ordered_route_items(work_plan.items, route.traversal)


def extract_route_supply_demand(
    route: StripRoute, work_plan: StripWorkPlan
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Return worker-carried demand and item order from authoritative work.

    Dependency-blocked items remain included.  Global seeds and any future
    non-inventory scopes are excluded rather than reinterpreted.
    """

    return ordered_inventory_demand(_ordered_route_items(route, work_plan))


def extract_tile_supply_demand(
    tiles: Iterable[tuple[int, int]], work_plan: StripWorkPlan
) -> tuple[tuple[str, int], ...]:
    """Summarize inventory-scoped demand for an unassigned tile set."""

    owned = set(tiles)
    ordered = ordered_route_items(work_plan.items, tuple(sorted(owned)))
    demand, _ = ordered_inventory_demand(ordered)
    return demand


def build_route_supply_plans(
    routes: Iterable[StripRoute],
    work_plan: StripWorkPlan,
    worker_inventories: Mapping[WorkerId, Mapping[str, int]],
    shed: Mapping[str, int],
    worker_positions: Mapping[WorkerId, tuple[int, int]],
) -> tuple[RouteSupplyPlan, ...]:
    """Reserve observed shed stock in the supplied (assignment) route order."""

    initial_shed = {str(item): max(0, int(amount)) for item, amount in shed.items()}
    remaining_shed = dict(initial_shed)
    plans: list[RouteSupplyPlan] = []
    reserved_totals: dict[str, int] = defaultdict(int)
    for route in routes:
        demand_pairs, item_order = extract_route_supply_demand(route, work_plan)
        demand = dict(demand_pairs)
        inventory = worker_inventories.get(route.owner, {})
        carried = {
            item: min(amount, max(0, int(inventory.get(item, 0))))
            for item, amount in demand.items()
        }
        reserved: dict[str, int] = {}
        missing: dict[str, int] = {}
        for item in item_order:
            need = demand[item] - carried[item]
            take = min(need, remaining_shed.get(item, 0))
            reserved[item] = take
            missing[item] = need - take
            remaining_shed[item] = remaining_shed.get(item, 0) - take
            reserved_totals[item] += take
        sequence = tuple(
            PickupBatch(item, reserved[item]) for item in item_order if reserved[item] > 0
        )
        position = worker_positions.get(route.owner)
        pickup_tile = nearest_shed_access(position) if sequence and position else None
        plans.append(
            RouteSupplyPlan(
                route_id=route.route_id,
                owner=route.owner,
                pickup_tile=pickup_tile,
                demand=demand_pairs,
                already_carried=_pairs(carried),
                reserved_from_shed=_pairs(reserved),
                missing_stock=_pairs(missing),
                # Pinned official 1.32.7 has no worker carrying capacity.
                capacity_limited=(),
                pickup_sequence=sequence,
            )
        )
    for item, amount in reserved_totals.items():
        if amount > initial_shed.get(item, 0):
            raise AssertionError(f"shed reservation overbooked for {item}")
    return tuple(plans)
