"""Focused Packet 3 supply planning and pickup tests."""

from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_routes import StripRoute, WorkerId
from executor_v0.strip_supply import (
    RouteSupplyState,
    build_route_supply_plans,
    extract_route_supply_demand,
)
from executor_v0.strip_work import (
    RowKey,
    RowSummary,
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    row_key_for_tile,
)


def route(route_id: str, owner: int, row: int = 0) -> StripRoute:
    tiles = tuple((row, x) for x in range(5))
    return StripRoute(
        route_id,
        tiles,
        tiles,
        WorkerId(owner),
        tiles[0],
        0,
        0,
    )


def item(kind: str, tile: tuple[int, int], *requirements: SupplyRequirement) -> WorkItem:
    return WorkItem(
        id=f"{kind}:{tile[0]},{tile[1]}",
        kind=kind,
        tile=tile,
        row_key=row_key_for_tile(tile),
        required_supplies=requirements,
    )


def work_plan(*items: WorkItem) -> StripWorkPlan:
    rows: dict[RowKey, list[WorkItem]] = {}
    for work in items:
        rows.setdefault(work.row_key, []).append(work)
    summaries = tuple(
        RowSummary(key, 5, len(values), 0, len(values))
        for key, values in sorted(rows.items())
    )
    return StripWorkPlan(
        items=items,
        chains=(),
        row_summaries=summaries,
        supply=SupplySnapshot(),
        diagnostics=WorkDiagnostics(),
        acting_seat=0,
    )


def plan() -> DailyPlan:
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    animals = ("GOOSE", "COW", "SHEEP")
    products = (*crops, "EGG", "MILK", "WOOL", "FERTILIZER")
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in crops},
        animal_targets={animal: 0 for animal in animals},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in crops},
        care_by_animal={animal: 0 for animal in animals},
        sell_quantities={
            product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for product in products
        },
    )


def observation(
    *, position=(0, 0), inventory=None, shed=None, hour=0, day=3
) -> dict:
    farm = {
        "farmer": [position[1], position[0]],
        "hands": [],
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": shed or {},
            "seeds": {"WHEAT": 10},
            "inventories": [inventory or {}, {}],
        },
    }


def test_demand_extraction_excludes_global_seeds_and_uses_work_items_once():
    current = work_plan(
        item("FEED", (0, 0), SupplyRequirement("WHEAT", 1)),
        item("FEED", (0, 1), SupplyRequirement("WHEAT", 1)),
        item("FERTILIZE", (0, 2), SupplyRequirement("FERTILIZER", 1)),
        item("PLANT", (0, 3), SupplyRequirement("WHEAT", 1, "global_seed")),
    )
    demand, order = extract_route_supply_demand(route("A", 0), current)
    assert dict(demand) == {"FERTILIZER": 1, "WHEAT": 2}
    assert order == ("WHEAT", "FERTILIZER")


def test_existing_inventory_is_worker_local_and_satisfies_first():
    current = work_plan(item("FERTILIZE", (0, 0), SupplyRequirement("FERTILIZER", 4)))
    plans = build_route_supply_plans(
        [route("A", 0)],
        current,
        {WorkerId(0): {"FERTILIZER": 2}, WorkerId(1): {"FERTILIZER": 9}},
        {"FERTILIZER": 3},
        {WorkerId(0): (0, 0)},
    )
    assert plans[0].demand == (("FERTILIZER", 4),)
    assert plans[0].already_carried == (("FERTILIZER", 2),)
    assert plans[0].reserved_from_shed == (("FERTILIZER", 2),)
    assert plans[0].missing_stock == ()


def test_late_observed_stock_is_reserved_from_route_demand():
    current = work_plan(item("FEED", (0, 0), SupplyRequirement("WHEAT", 2)))
    assigned = route("A", 0)
    supply_plan = build_route_supply_plans(
        [assigned],
        current,
        {WorkerId(0): {}},
        {},
        {WorkerId(0): (0, 0)},
    )[0]
    assert supply_plan.reserved_from_shed == ()
    assert supply_plan.missing_stock == (("WHEAT", 2),)

    controller = StripExecutorController()
    controller._routes = {assigned.owner: assigned}
    controller._supply_plans = {assigned.route_id: supply_plan}
    controller._supply_states = {assigned.route_id: RouteSupplyState()}
    assert controller._outstanding_reservations() == {"WHEAT": 2}


def test_shared_shed_reservation_is_assignment_ordered_and_non_overbooked():
    current = work_plan(
        item("FEED", (0, 0), SupplyRequirement("WHEAT", 2)),
        item("FEED", (1, 0), SupplyRequirement("WHEAT", 2)),
    )
    plans = build_route_supply_plans(
        [route("A", 0, 0), route("B", 1, 1)],
        current,
        {WorkerId(0): {}, WorkerId(1): {}},
        {"WHEAT": 3},
        {WorkerId(0): (0, 0), WorkerId(1): (1, 0)},
    )
    assert plans[0].reserved_from_shed == (("WHEAT", 2),)
    assert plans[1].reserved_from_shed == (("WHEAT", 1),)
    assert plans[1].missing_stock == (("WHEAT", 1),)
    assert sum(dict(plan.reserved_from_shed).get("WHEAT", 0) for plan in plans) == 3


def test_pickup_order_is_first_use_and_seed_demand_never_reserves():
    current = work_plan(
        item("FERTILIZE", (0, 1), SupplyRequirement("FERTILIZER", 2)),
        item("FEED", (0, 0), SupplyRequirement("WHEAT", 3)),
        item("PLANT", (0, 2), SupplyRequirement("CARROT", 1, "global_seed")),
    )
    plans = build_route_supply_plans(
        [route("A", 0)],
        current,
        {WorkerId(0): {}},
        {"WHEAT": 3, "FERTILIZER": 2, "CARROT": 9},
        {WorkerId(0): (0, 9)},
    )
    assert [batch.item for batch in plans[0].pickup_sequence] == [
        "WHEAT",
        "FERTILIZER",
    ]
    assert plans[0].pickup_tile == (4, 5)
    assert dict(plans[0].demand) == {"FERTILIZER": 2, "WHEAT": 3}


def test_fully_carried_route_has_no_pickup_tile_or_shed_trip():
    current = work_plan(item("FEED", (0, 0), SupplyRequirement("WHEAT", 3)))
    plans = build_route_supply_plans(
        [route("A", 0)],
        current,
        {WorkerId(0): {"WHEAT": 3}},
        {"WHEAT": 99},
        {WorkerId(0): (9, 9)},
    )
    assert plans[0].fully_supplied
    assert plans[0].pickup_tile is None
    assert plans[0].pickup_sequence == ()


def test_partial_observed_pickup_records_only_real_inventory_gain():
    current = work_plan(item("FEED", (0, 0), SupplyRequirement("WHEAT", 3)))

    def builder(obs, daily_plan, **kwargs):
        del obs, daily_plan, kwargs
        return current

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(position=(4, 4), shed={"WHEAT": 3}), plan())
    partial = controller.act(
        observation(position=(4, 4), inventory={"WHEAT": 1}, shed={}, hour=1),
        plan(),
    )
    state = partial.diagnostics["route_diagnostics"][0]["supply_state"]
    assert state["acquired"] == {"WHEAT": 1}
    assert state["failed_or_unfulfilled"] == {"WHEAT": 2}


def test_packed_route_reserves_chain_and_day_reset_rebuilds_ledger():
    current = work_plan(
        item("FEED", (0, 0), SupplyRequirement("WHEAT", 2)),
        item("FEED", (1, 0), SupplyRequirement("WHEAT", 2)),
    )

    def builder(obs, daily_plan, **kwargs):
        del obs, daily_plan, kwargs
        return current

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(
        observation(position=(0, 0), shed={"WHEAT": 2}), plan()
    )
    assert first.diagnostics["supply_diagnostics"]["total_reservations_by_item"] == {
        "WHEAT": 2
    }
    assert first.diagnostics["unassigned_supply_demand"] == {}
    assert first.diagnostics["route_diagnostics"][0]["supply_plan"]["demand"] == {
        "WHEAT": 4
    }

    next_day = controller.act(
        observation(position=(0, 0), shed={}, day=4), plan()
    )
    assert next_day.diagnostics["supply_diagnostics"]["total_reservations_by_item"] == {}
    assert next_day.diagnostics["route_diagnostics"][0]["supply_state"]["acquired"] == {}


def test_batched_pickup_is_confirmed_before_route_entry():
    current = work_plan(
        item("FEED", (0, 0), SupplyRequirement("WHEAT", 3))
    )

    def builder(obs, daily_plan, **kwargs):
        del obs, daily_plan, kwargs
        return current

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(
        observation(position=(4, 4), shed={"WHEAT": 3}), plan()
    )
    assert first.farmer_action == ("PICKUP", "WHEAT", 3)
    assert first.diagnostics["route_diagnostics"][0]["supply_state"]["acquired"] == {}

    # The command is not proof.  The next observation confirms the quantity.
    confirmed = controller.act(
        observation(position=(4, 4), inventory={"WHEAT": 3}, shed={}, hour=1),
        plan(),
    )
    assert confirmed.farmer_action == ("NORTH",)
    state = confirmed.diagnostics["route_diagnostics"][0]["supply_state"]
    assert state["acquired"] == {"WHEAT": 3}
    assert state["pickup_turns"] == 1
    assert controller._outstanding_reservations() == {}


def test_failed_pickup_has_no_phantom_acquisition_and_continues():
    current = work_plan(item("FEED", (0, 0), SupplyRequirement("WHEAT", 2)))

    def builder(obs, daily_plan, **kwargs):
        del obs, daily_plan, kwargs
        return current

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(position=(4, 4), shed={"WHEAT": 2}), plan())
    assert first.farmer_action == ("PICKUP", "WHEAT", 2)
    failed = controller.act(
        observation(position=(4, 4), shed={"WHEAT": 0}, hour=1), plan()
    )
    assert failed.farmer_action == ("NORTH",)
    state = failed.diagnostics["route_diagnostics"][0]["supply_state"]
    assert state["acquired"] == {}
    assert state["failed_or_unfulfilled"] == {"WHEAT": 2}
    assert controller._outstanding_reservations() == {}
