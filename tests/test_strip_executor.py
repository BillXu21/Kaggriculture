"""Cheap Packet 2 route/controller contract tests."""

from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_routes import (
    RoutePhase,
    StripRoute,
    WorkerId,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
)
from executor_v0.strip_work import (
    BlockReason,
    RowKey,
    RowSummary,
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    WorkStatus,
    row_key_for_tile,
)


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")


def plan() -> DailyPlan:
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={
            product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for product in PRODUCTS
        },
    )


def observation(*, hour=0, farmer=(0, 0), hands=(), inventories=None, day=3):
    farm = {
        "farmer": [farmer[0], farmer[1]],
        "hands": [[x, y] for x, y in hands],
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": {},
            "seeds": {"WHEAT": 10},
            "inventories": inventories if inventories is not None else [{}, {}],
        },
    }


def work_item(
    kind: str,
    tile: tuple[int, int],
    *,
    status=WorkStatus.READY,
    crop=None,
    required_supplies=(),
    depends_on=(),
    block_reason=None,
    item_id=None,
) -> WorkItem:
    return WorkItem(
        id=item_id or f"{kind}:{tile[0]},{tile[1]}",
        kind=kind,
        status=status,
        block_reason=block_reason,
        tile=tile,
        crop=crop,
        depends_on=tuple(depends_on),
        required_supplies=tuple(required_supplies),
        row_key=row_key_for_tile(tile),
    )


def fake_plan(items: tuple[WorkItem, ...]) -> StripWorkPlan:
    by_row: dict[RowKey, list[WorkItem]] = {}
    for item in items:
        if item.row_key is not None:
            by_row.setdefault(item.row_key, []).append(item)
    rows = []
    for key, row_items in sorted(by_row.items()):
        ready = sum(item.interaction_turns for item in row_items if item.ready)
        future = sum(item.interaction_turns for item in row_items if not item.ready)
        rows.append(RowSummary(key, 5, ready, future, ready + future))
    return StripWorkPlan(
        items=items,
        chains=(),
        row_summaries=tuple(rows),
        supply=SupplySnapshot(),
        diagnostics=WorkDiagnostics(),
        acting_seat=0,
    )


def test_generic_route_accepts_non_horizontal_traversal():
    route = StripRoute(
        "generic",
        ((0, 0), (1, 0), (1, 1)),
        ((1, 1), (1, 0), (0, 0)),
        WorkerId(0),
        (1, 1),
        2,
        0,
    )
    assert route.current_tile == (1, 1)
    assert route.traversal[1:] == ((1, 0), (0, 0))


def test_horizontal_generation_and_exclusive_row_assignment():
    key0 = RowKey("NW", 2, 2, 0, 4)
    key1 = RowKey("NW", 3, 3, 0, 4)
    work = fake_plan((work_item("WATER", (2, 0)), work_item("WATER", (3, 0))))
    candidates = generate_horizontal_route_candidates(work)
    assert candidates[0].row_key == key0
    assert candidates[0].owned_tiles == ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4))
    assert candidates[1].row_key == key1
    assigned = assign_horizontal_routes(
        candidates,
        {WorkerId(0): (2, 0), WorkerId(1): (3, 0)},
        assignment_hour=4,
    )
    assert {tile for route in assigned.routes for tile in route.owned_tiles} == {
        *candidates[0].owned_tiles,
        *candidates[1].owned_tiles,
    }
    assert not (
        set(assigned.routes[0].owned_tiles)
        & set(assigned.routes[1].owned_tiles)
    )


def test_assignment_endpoint_is_nearest_and_tie_is_left():
    work = fake_plan((work_item("WATER", (2, 0)),))
    candidate = generate_horizontal_route_candidates(work)
    right = assign_horizontal_routes(
        candidate, {WorkerId(0): (2, 4)}, assignment_hour=0
    ).routes[0]
    left = assign_horizontal_routes(
        candidate, {WorkerId(0): (2, 0)}, assignment_hour=0
    ).routes[0]
    tie = assign_horizontal_routes(
        candidate, {WorkerId(0): (0, 2)}, assignment_hour=0
    ).routes[0]
    assert right.traversal == tuple(reversed(candidate[0].owned_tiles))
    assert left.traversal == candidate[0].owned_tiles
    assert tie.traversal == candidate[0].owned_tiles


def test_assignment_is_repeatable_and_dependency_blocked_rows_are_active():
    blocked = work_item(
        "WATER",
        (2, 0),
        status=WorkStatus.BLOCKED,
        block_reason=BlockReason.DEPENDENCY_BLOCKED,
        depends_on=("PLANT:WHEAT:2,0",),
    )
    work = fake_plan((blocked,))
    candidates = generate_horizontal_route_candidates(work)
    assert len(candidates) == 1
    first = assign_horizontal_routes(
        candidates, {WorkerId(0): (0, 0)}, assignment_hour=0
    )
    second = assign_horizontal_routes(
        candidates, {WorkerId(0): (0, 0)}, assignment_hour=0
    )
    assert first.routes[0].to_json_dict() == second.routes[0].to_json_dict()


def test_fixed_assignment_counts_excess_routes_and_workers():
    work = fake_plan(
        tuple(work_item("WATER", (row, 0)) for row in (0, 1, 2))
    )
    candidates = generate_horizontal_route_candidates(work)
    assigned = assign_horizontal_routes(
        candidates,
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1)},
        assignment_hour=0,
    )
    assert len(assigned.routes) == 2
    assert [item.route_id for item in assigned.unassigned] == ["ROW:NW:2:0-4"]
    work_one = fake_plan((work_item("WATER", (0, 0)),))
    assigned_one = assign_horizontal_routes(
        generate_horizontal_route_candidates(work_one),
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1), WorkerId(2): (0, 2)},
        assignment_hour=0,
    )
    assert [worker.label for worker in assigned_one.idle_workers] == ["HAND:0", "HAND:1"]


def test_vertical_first_travel_then_monotonic_sweep():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan((work_item("WATER", (1, 4)),))

    controller = StripExecutorController(work_builder=builder)
    result = controller.act(observation(farmer=(0, 4)), plan())
    assert result.farmer_action == ("NORTH",)
    result = controller.act(observation(hour=1, farmer=(0, 3)), plan())
    assert result.farmer_action == ("NORTH",)
    # A real observation at the entry tile starts the sweep; the next move is
    # toward the second traversal tile and never reverses direction.
    result = controller.act(observation(hour=2, farmer=(0, 2)), plan())
    assert result.farmer_action == ("NORTH",)


def test_plant_water_and_harvest_replacement_stay_on_tile():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            item = work_item("PLANT", (0, 0), crop="WHEAT")
        elif hour == 1:
            item = work_item("WATER", (0, 0), crop="WHEAT")
        elif hour == 2:
            item = work_item("HARVEST", (0, 0), crop="WHEAT")
        elif hour == 3:
            item = work_item("PLANT", (0, 0), crop="WHEAT")
        elif hour == 4:
            item = work_item("WATER", (0, 0), crop="WHEAT")
        else:
            item = work_item("WATER", (0, 1))
        return fake_plan((item,))

    controller = StripExecutorController(work_builder=builder)
    actions = []
    for hour in range(6):
        result = controller.act(observation(hour=hour, farmer=(0, 0)), plan())
        actions.append(result.farmer_action)
    assert actions[:5] == [
        ("PLANT", "WHEAT"),
        ("WATER",),
        ("HARVEST",),
        ("PLANT", "WHEAT"),
        ("WATER",),
    ]


def test_worker_does_not_steal_work_while_crossing_another_route():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan(
            (work_item("WATER", (0, 0)), work_item("WATER", (1, 0)))
        )

    controller = StripExecutorController(work_builder=builder)
    result = controller.act(
        observation(farmer=(0, 1), hands=((0, 1),)),
        plan(),
    )
    assert result.farmer_action == ("NORTH",)
    assert result.hands_actions == (("WATER",),)


def test_useful_water_is_selected_before_harvest_on_same_tile():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan(
            (work_item("HARVEST", (0, 0)), work_item("WATER", (0, 0)))
        )

    controller = StripExecutorController(work_builder=builder)
    result = controller.act(observation(farmer=(0, 0)), plan())
    assert result.farmer_action == ("WATER",)


def test_missing_supply_does_not_pick_up_but_carried_supply_can_act():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan(
            (
                work_item(
                    "FERTILIZE",
                    (0, 0),
                    required_supplies=(SupplyRequirement("FERTILIZER", 1),),
                ),
            )
        )

    controller = StripExecutorController(work_builder=builder)
    missing = controller.act(observation(farmer=(0, 0)), plan())
    assert missing.farmer_action != ("FERTILIZE",)
    assert missing.diagnostics["route_diagnostics"][0]["unavailable_supply_work"] == {
        "FERTILIZE": 1
    }
    carried_controller = StripExecutorController(work_builder=builder)
    carried = carried_controller.act(
        observation(hour=1, farmer=(0, 0), inventories=[{"FERTILIZER": 1}, {}]),
        plan(),
    )
    assert carried.farmer_action == ("FERTILIZE",)


def test_completed_route_passes_and_late_work_is_not_revisited():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        item = work_item(
            "WATER", (0, 4) if int(obs["hour"]) == 0 else (0, 0)
        )
        return fake_plan((item,))

    controller = StripExecutorController(work_builder=builder)
    for hour in range(5):
        position = (hour, 0) if hour else (0, 0)
        controller.act(observation(hour=hour, farmer=position), plan())
    done = controller.act(observation(hour=5, farmer=(4, 0)), plan())
    assert done.farmer_action == ("PASS",)
    assert controller.routes[0].phase == RoutePhase.DONE
    assert "WATER:0,0" in controller.routes[0].late_work_ids
    later = controller.act(observation(hour=6, farmer=(4, 0)), plan())
    assert later.farmer_action == ("PASS",)


def test_day_boundary_discards_old_cursor_and_assignment():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        tile = (0, 0) if int(obs["hour"]) == 0 else (0, 1)
        return fake_plan((work_item("WATER", tile),))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0), day=3), plan())
    controller.act(observation(hour=1, farmer=(0, 0), day=3), plan())
    old_route = controller.routes[0]
    assert old_route.cursor == 0
    assert old_route.pending_cursor == 1
    controller.act(observation(hour=0, farmer=(0, 4), day=4), plan())
    new_route = controller.routes[0]
    assert new_route.assignment_hour == 0
    assert new_route.cursor == 0
    assert new_route.entry_tile == (0, 0)


def test_failed_departure_does_not_mark_tile_passed():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan((work_item("WATER", (0, 4)),))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    route = controller.routes[0]
    assert route.pending_cursor == 1

    # The engine reports no movement: departure is unconfirmed.
    result = controller.act(observation(hour=1, farmer=(0, 0)), plan())
    assert result.farmer_action == ("EAST",)
    assert route.cursor == 0
    assert route.pending_cursor == 1
    assert (0, 0) not in route.passed_tiles


def test_confirmed_departure_marks_prior_tile_passed():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return fake_plan((work_item("WATER", (0, 4)),))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    controller.act(observation(hour=1, farmer=(1, 0)), plan())
    route = controller.routes[0]
    assert route.cursor == 1
    assert route.pending_cursor == 2  # already committed to the next sweep step
    assert (0, 0) in route.passed_tiles
    assert (0, 1) not in route.passed_tiles


def test_work_appearing_before_confirmed_departure_is_executed():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            return fake_plan((work_item("WATER", (0, 4)),))
        if hour == 1:
            return fake_plan((work_item("WATER", (0, 0)),))
        return fake_plan((work_item("WATER", (0, 4)),))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    route = controller.routes[0]
    result = controller.act(observation(hour=1, farmer=(0, 0)), plan())
    assert result.farmer_action == ("WATER",)
    assert route.cursor == 0
    assert route.pending_cursor == 1
    assert (0, 0) not in route.passed_tiles
    assert "WATER:0,0" not in route.late_work_ids


def test_work_after_confirmed_departure_is_late_but_not_revisited():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            return fake_plan((work_item("WATER", (0, 4)),))
        if hour >= 2:
            return fake_plan((work_item("HARVEST", (0, 0), crop="WHEAT"),))
        return fake_plan(())

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    controller.act(observation(hour=1, farmer=(1, 0)), plan())
    route = controller.routes[0]
    assert (0, 0) in route.passed_tiles

    result = controller.act(observation(hour=2, farmer=(2, 0)), plan())
    assert result.farmer_action == ("EAST",)  # forward, never backward
    assert route.cursor == 2
    assert "HARVEST:0,0" in route.late_work_ids


def test_post_completion_late_work_is_recorded_once():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            return fake_plan((work_item("WATER", (0, 4)),))
        if hour >= 5:
            return fake_plan((work_item("HARVEST", (0, 0), crop="WHEAT"),))
        return fake_plan(())

    controller = StripExecutorController(work_builder=builder)
    for hour in range(4):
        controller.act(observation(hour=hour, farmer=(hour, 0)), plan())
    done = controller.act(observation(hour=4, farmer=(4, 0)), plan())
    route = controller.routes[0]
    assert done.farmer_action == ("PASS",)
    assert route.phase == RoutePhase.DONE
    assert route.completion_hour == 4
    assert route.late_work_ids == set()

    for hour in range(5, 8):
        later = controller.act(observation(hour=hour, farmer=(4, 0)), plan())
        assert later.farmer_action == ("PASS",)
    assert route.phase == RoutePhase.DONE
    assert route.cursor == 4
    assert route.late_work_ids == {"HARVEST:0,0"}


def test_post_completion_late_work_on_final_tile_is_recorded():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            return fake_plan((work_item("WATER", (0, 4)),))
        if hour >= 5:
            return fake_plan((work_item("HARVEST", (0, 4), crop="WHEAT"),))
        return fake_plan(())

    controller = StripExecutorController(work_builder=builder)
    for hour in range(4):
        controller.act(observation(hour=hour, farmer=(hour, 0)), plan())
    done = controller.act(observation(hour=4, farmer=(4, 0)), plan())
    route = controller.routes[0]
    assert done.farmer_action == ("PASS",)
    assert route.phase == RoutePhase.DONE
    assert (0, 4) in route.passed_tiles

    later = controller.act(observation(hour=5, farmer=(4, 0)), plan())
    assert later.farmer_action == ("PASS",)
    assert route.phase == RoutePhase.DONE
    assert "HARVEST:0,4" in route.late_work_ids
