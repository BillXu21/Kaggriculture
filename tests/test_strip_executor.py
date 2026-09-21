"""Cheap Packet 2 route/controller contract tests."""

from __future__ import annotations

import copy
from itertools import product

import pytest

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_routes import (
    HorizontalRouteCandidate,
    RoutePhase,
    RouteSegment,
    StripRoute,
    WorkerId,
    _chain_plan_for_mask,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
    route_cursor_invariants_hold,
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
    source="strip_forecast",
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
        source=source,
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


def test_packed_assignment_covers_excess_rows_without_extra_workers():
    work = fake_plan(
        tuple(work_item("WATER", (row, 0)) for row in (0, 1, 2, 3))
    )
    candidates = generate_horizontal_route_candidates(work)
    assigned = assign_horizontal_routes(
        candidates,
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1)},
        assignment_hour=0,
    )
    assert len(assigned.routes) == 2
    assert not assigned.unassigned
    assert sorted(len(route.segments) for route in assigned.routes) == [2, 2]
    work_one = fake_plan((work_item("WATER", (0, 0)),))
    assigned_one = assign_horizontal_routes(
        generate_horizontal_route_candidates(work_one),
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1), WorkerId(2): (0, 2)},
        assignment_hour=0,
    )
    assert [worker.label for worker in assigned_one.idle_workers] == ["HAND:0", "HAND:1"]


def _mechanical_rows(workloads=(1, 1, 1, 1)):
    return tuple(
        HorizontalRouteCandidate(
            f"ROW:{row}",
            RowKey("NW", row, row, 0, 4),
            tuple((row, x) for x in range(5)),
            workload,
            workload,
            0,
        )
        for row, workload in enumerate(workloads)
    )


def _route_rows(route):
    return tuple(segment.traversal[0][0] for segment in route.segments)


def _route_movement(route):
    return sum(
        segment.entry_distance + len(segment.traversal) - 1
        for segment in route.segments
    )


def _oracle_chain_plan(candidates, position, mask):
    indices = tuple(index for index in range(len(candidates)) if mask & (1 << index))
    orders = [indices]
    if len({candidates[index].row_key.global_row for index in indices}) > 1:
        orders.append(tuple(reversed(indices)))
    best = None
    for order in orders:
        for orientations in product((0, 1), repeat=len(order)):
            movement = 0
            path = []
            previous_end = position
            for index, side in zip(order, orientations):
                candidate = candidates[index]
                traversal = (
                    candidate.owned_tiles
                    if side == 0
                    else tuple(reversed(candidate.owned_tiles))
                )
                distance = abs(previous_end[0] - traversal[0][0]) + abs(
                    previous_end[1] - traversal[0][1]
                )
                movement += distance
                path.append((index, side, distance))
                previous_end = traversal[-1]
            key = (movement, tuple(path))
            if best is None or key < best[0]:
                best = (key, movement, tuple(path))
    assert best is not None
    movement, path = best[1:]
    interactions = sum(candidates[index].workload_interactions for index, _, _ in path)
    sweep = sum(len(candidates[index].owned_tiles) - 1 for index, _, _ in path)
    return movement, movement + sweep + interactions, path


def test_chain_plan_matches_allowed_order_orientation_oracle_for_all_small_masks():
    candidates = tuple(
        sorted(
            (
                HorizontalRouteCandidate(
                    route_id,
                    RowKey(quadrant, row % 5, row, x_start, x_start + 4),
                    tuple((row, x) for x in range(x_start, x_start + 5)),
                    workload,
                    workload,
                    0,
                )
                for route_id, quadrant, row, x_start, workload in (
                    ("NW2", "NW", 2, 0, 2),
                    ("NW3", "NW", 3, 0, 1),
                    ("NE2", "NE", 2, 5, 3),
                    ("NE3", "NE", 3, 5, 2),
                    ("SW7", "SW", 7, 0, 1),
                )
            ),
            key=lambda candidate: candidate.row_key,
        )
    )
    for position in ((0, 0), (4, 4), (5, 5), (9, 9), (4, 7)):
        for mask in range(1, 1 << len(candidates)):
            production = _chain_plan_for_mask(candidates, position, mask)
            movement, completion, path = _oracle_chain_plan(candidates, position, mask)
            assert (production.movement_turns, production.completion_turns) == (
                movement,
                completion,
            )
            assert tuple(
                (candidates.index(candidate), 0 if segment.traversal == candidate.owned_tiles else 1, segment.entry_distance)
                for candidate, segment in production.assigned
            ) == path


def test_four_candidate_route_packer_fixture_is_complete_and_repeatable():
    candidates = (
        HorizontalRouteCandidate(
            "NE3",
            RowKey("NE", 3, 3, 5, 9),
            tuple((3, x) for x in range(5, 10)),
            2,
            2,
            0,
        ),
        HorizontalRouteCandidate(
            "NE4",
            RowKey("NE", 4, 4, 5, 9),
            tuple((4, x) for x in range(5, 10)),
            1,
            1,
            0,
        ),
        HorizontalRouteCandidate(
            "NW2",
            RowKey("NW", 2, 2, 0, 4),
            tuple((2, x) for x in range(5)),
            2,
            2,
            0,
        ),
        HorizontalRouteCandidate(
            "NW3",
            RowKey("NW", 3, 3, 0, 4),
            tuple((3, x) for x in range(5)),
            3,
            3,
            0,
        ),
    )
    positions = {WorkerId(0): (4, 4)}
    first = assign_horizontal_routes(candidates, positions, assignment_hour=0)
    second = assign_horizontal_routes(candidates, positions, assignment_hour=0)
    assigned = [segment for route in first.routes for segment in route.segments]
    assert len(assigned) == len(candidates)
    assert {segment.segment_id for segment in assigned} == {
        candidate.route_id for candidate in candidates
    }
    assert first.unassigned == ()
    assert all(route_cursor_invariants_hold(route) for route in first.routes)
    assert [route.to_json_dict() for route in first.routes] == [
        route.to_json_dict() for route in second.routes
    ]


def test_four_adjacent_rows_pack_into_contiguous_two_row_chains():
    assigned = assign_horizontal_routes(
        _mechanical_rows(),
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1)},
        assignment_hour=0,
    )
    assert sorted(_route_rows(route) for route in assigned.routes) == [(0, 1), (2, 3)]
    assert sum(_route_movement(route) for route in assigned.routes) == 21


def test_opposite_worker_ends_receive_their_nearest_row_halves():
    assigned = assign_horizontal_routes(
        _mechanical_rows(),
        {WorkerId(0): (0, 0), WorkerId(1): (3, 0)},
        assignment_hour=0,
    )
    by_worker = {route.owner.index: _route_rows(route) for route in assigned.routes}
    assert by_worker == {0: (0, 1), 1: (3, 2)}


def test_four_rows_and_three_workers_leave_one_neighboring_pair():
    assigned = assign_horizontal_routes(
        _mechanical_rows(),
        {WorkerId(0): (0, 0), WorkerId(1): (0, 1), WorkerId(2): (0, 2)},
        assignment_hour=0,
    )
    assert sorted(len(route.segments) for route in assigned.routes) == [1, 1, 2]
    paired = next(route for route in assigned.routes if len(route.segments) == 2)
    assert _route_rows(paired) in ((0, 1), (1, 2), (2, 3))


def test_interaction_load_can_justify_longer_travel():
    assigned = assign_horizontal_routes(
        _mechanical_rows((8, 1, 1, 1)),
        {WorkerId(0): (0, 0), WorkerId(1): (3, 0)},
        assignment_hour=0,
    )
    by_worker = {route.owner.index: _route_rows(route) for route in assigned.routes}
    assert by_worker[0] == (0,)
    assert by_worker[1] == (3, 2, 1)


def test_geometric_assignment_ties_are_repeatable():
    candidates = _mechanical_rows()
    positions = {WorkerId(0): (1, 2), WorkerId(1): (2, 2)}
    first = assign_horizontal_routes(candidates, positions, assignment_hour=0)
    second = assign_horizontal_routes(candidates, positions, assignment_hour=0)
    assert [route.to_json_dict() for route in first.routes] == [
        route.to_json_dict() for route in second.routes
    ]


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


def test_controller_executes_retained_routine_harvest():
    obs = observation(day=8, farmer=(0, 0))
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": "TOMATO",
        "planted_day": 0,
        "yield_units": 1,
        "watered_today": True,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    retained_plan = DailyPlan.create(
        crop_targets={crop: (1 if crop == "TOMATO" else 0) for crop in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={
            product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for product in PRODUCTS
        },
    )

    controller = StripExecutorController()
    result = controller.act(obs, retained_plan)

    assert result.farmer_action == ("HARVEST",)
    assert any(
        item.id == "HARVEST:0,0" and item.source == "routine_harvest"
        for item in controller._plan.items
    )


@pytest.mark.parametrize("crop,day", [("WHEAT", 3), ("CARROT", 3), ("MELON", 10)])
def test_retained_one_shot_harvest_replants_and_waters_same_day(crop, day):
    initial = observation(day=day, farmer=(0, 0))
    initial["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": 0,
        "yield_units": 3 if crop == "WHEAT" else 1,
        "watered_today": True,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    target = plan()
    target = DailyPlan.create(
        crop_targets={name: int(name == crop) for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS},
        land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={
            product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
            for product in PRODUCTS
        },
    )
    initial["private"]["seeds"] = {crop: 1}
    controller = StripExecutorController()

    assert controller.act(initial, target).farmer_action == ("HARVEST",)

    after_harvest = copy.deepcopy(initial)
    after_harvest["hour"] = 1
    after_harvest["step"] += 1
    after_harvest["farms"][0]["tiles"][0][0] = None
    planted = controller.act(after_harvest, target)
    assert planted.farmer_action == ("PLANT", crop)
    assert planted.diagnostics["route_diagnostics"][0]["continuation"]["status"] == (
        "SUCCESSOR_WATER"
    )
    assert planted.diagnostics["route_diagnostics"][0]["continuation"]["source"] == (
        "retained_crop_maintenance"
    )

    after_plant = copy.deepcopy(after_harvest)
    after_plant["hour"] = 2
    after_plant["step"] += 1
    after_plant["private"]["seeds"][crop] = 0
    after_plant["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "yield_units": 0,
        "watered_today": False,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    watered = controller.act(after_plant, target)
    assert watered.farmer_action == ("WATER",)
    assert watered.diagnostics["route_diagnostics"][0]["continuation"]["status"] == (
        "COMPLETED"
    )


def test_retained_harvest_target_reduction_does_not_replant():
    initial = observation(day=3, farmer=(0, 0))
    initial["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "yield_units": 3, "watered_today": True,
        "fertilized_until_day": -1, "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    retained = plan()
    retained = DailyPlan.create(
        crop_targets={name: int(name == "WHEAT") for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS}, land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )
    reduced = DailyPlan.create(
        crop_targets={name: 0 for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS}, land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )
    controller = StripExecutorController()
    assert controller.act(initial, retained).farmer_action == ("HARVEST",)
    after = copy.deepcopy(initial)
    after["day"] = 4
    after["hour"] = 0
    after["step"] += 24
    after["farms"][0]["tiles"][0][0] = None
    result = controller.act(after, reduced)
    assert result.farmer_action != ("PLANT", "WHEAT")


def test_retained_harvest_seed_block_resumes_after_observed_seed():
    initial = observation(day=3, farmer=(0, 0))
    initial["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "yield_units": 3, "watered_today": True,
        "fertilized_until_day": -1, "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    target = DailyPlan.create(
        crop_targets={name: int(name == "WHEAT") for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS}, land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )
    initial["private"]["seeds"] = {"WHEAT": 0}
    controller = StripExecutorController()
    assert controller.act(initial, target).farmer_action == ("HARVEST",)

    blocked = copy.deepcopy(initial)
    blocked["hour"] = 1
    blocked["step"] += 1
    blocked["farms"][0]["tiles"][0][0] = None
    controller.act(blocked, target)
    assert controller.routes[0].continuation_blocked_reason == "MISSING_GLOBAL_RESOURCE"

    observed_seed = copy.deepcopy(blocked)
    observed_seed["hour"] = 2
    observed_seed["step"] += 1
    observed_seed["farmer"] = [1, 0]
    observed_seed["farms"][0]["farmer"] = [1, 0]
    observed_seed["private"]["seeds"] = {"WHEAT": 1}
    assert controller.act(observed_seed, target).farmer_action == ("WEST",)

    resumed = copy.deepcopy(observed_seed)
    resumed["hour"] = 3
    resumed["step"] += 1
    resumed["farmer"] = [0, 0]
    resumed["farms"][0]["farmer"] = [0, 0]
    assert controller.act(resumed, target).farmer_action == ("PLANT", "WHEAT")


def test_retained_harvest_does_not_start_two_step_chain_at_day_boundary():
    initial = observation(day=3, hour=22, farmer=(0, 0))
    initial["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "yield_units": 3, "watered_today": True,
        "fertilized_until_day": -1, "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    initial["private"]["seeds"] = {"WHEAT": 1}
    target = DailyPlan.create(
        crop_targets={name: int(name == "WHEAT") for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS}, land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )
    controller = StripExecutorController()
    assert controller.act(initial, target).farmer_action == ("HARVEST",)

    after = copy.deepcopy(initial)
    after["hour"] = 23
    after["step"] += 1
    after["farms"][0]["tiles"][0][0] = None
    result = controller.act(after, target)
    assert result.farmer_action != ("PLANT", "WHEAT")
    assert controller.routes[0].continuation_blocked_reason == "INSUFFICIENT_DAY_TIME"


def test_retained_continuation_reopens_completed_final_tile():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        if int(obs["hour"]) < 5:
            return fake_plan(
                (work_item("HARVEST", (0, 4), crop="WHEAT", source="routine_harvest"),)
            )
        if int(obs["hour"]) == 5:
            return fake_plan((work_item(
                "PLANT", (0, 4), crop="WHEAT",
                status=WorkStatus.BLOCKED,
                block_reason=BlockReason.DEPENDENCY_BLOCKED,
            ),))
        return fake_plan((work_item("PLANT", (0, 4), crop="WHEAT"),))

    controller = StripExecutorController(work_builder=builder)
    target = DailyPlan.create(
        crop_targets={name: int(name == "WHEAT") for name in CROPS},
        animal_targets={animal: 0 for animal in ANIMALS}, land_count=1,
        fertilizer_by_crop={name: 0 for name in CROPS},
        care_by_animal={animal: 0 for animal in ANIMALS},
        sell_quantities={product: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)} for product in PRODUCTS},
    )
    for hour in range(5):
        position = (hour, 0)
        result = controller.act(observation(hour=hour, farmer=position), target)
    assert result.farmer_action == ("HARVEST",)

    completed = controller.act(observation(hour=5, farmer=(4, 0)), target)
    assert completed.farmer_action == ("PASS",)
    assert controller.routes[0].phase == RoutePhase.DONE

    reopened = controller.act(observation(hour=6, farmer=(4, 0)), target)
    assert reopened.farmer_action == ("PLANT", "WHEAT")
    assert controller.routes[0].phase == RoutePhase.SWEEP


def test_retained_continuation_survives_worker_segment_change():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        items = [work_item("WATER", (1, 0))]
        if int(obs["hour"]) < 5:
            items.append(
                work_item(
                    "HARVEST", (0, 4), crop="WHEAT", source="routine_harvest"
                )
            )
        else:
            items.append(work_item("PLANT", (0, 4), crop="WHEAT"))
        return fake_plan(tuple(items))

    controller = StripExecutorController(work_builder=builder)
    for hour in range(5):
        position = (hour, 0)
        result = controller.act(observation(hour=hour, farmer=position), plan())
    assert result.farmer_action == ("HARVEST",)

    position = (4, 0)
    for hour in range(5, 10):
        result = controller.act(observation(hour=hour, farmer=position), plan())
        if result.farmer_action == ("PLANT", "WHEAT"):
            break
        if result.farmer_action == ("WEST",):
            position = (max(0, position[0] - 1), position[1])
        elif result.farmer_action == ("SOUTH",):
            position = (position[0], position[1] + 1)
        elif result.farmer_action == ("NORTH",):
            position = (position[0], position[1] - 1)
        elif result.farmer_action == ("EAST",):
            position = (position[0] + 1, position[1])
    else:
        pytest.fail("retained continuation was abandoned at a segment boundary")


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


def test_observed_animal_purchase_revisits_a_passed_place_once():
    place_id = "PLACE:SHEEP:0,0"

    def builder(obs, daily_plan, **kwargs):
        del daily_plan, kwargs
        private = obs.get("private") or {}
        shed = private.get("shed") or {}
        inventory = (private.get("inventories") or [{}])[0]
        observed = bool(shed.get("SHEEP") or inventory.get("SHEEP"))
        place = WorkItem(
            id=place_id,
            kind="PLACE",
            animal="SHEEP",
            tile=(0, 0),
            row_key=row_key_for_tile((0, 0)),
            required_supplies=((SupplyRequirement("SHEEP", 1),) if observed else ()),
            status=WorkStatus.READY if observed else WorkStatus.BLOCKED,
            block_reason=None if observed else BlockReason.MISSING_PURCHASE,
        )
        if observed:
            return fake_plan((place,))
        return fake_plan((
            WorkItem(id="BUY_ANIMAL:SHEEP:1", kind="BUY_ANIMAL", animal="SHEEP"),
            place,
        ))

    controller = StripExecutorController(work_builder=builder)
    position = (0, 0)
    shed: dict[str, int] = {}
    inventory: dict[str, int] = {}
    purchase_observed = False
    saw_late_place = False
    placed = False
    for step in range(24):
        current = observation(
            hour=step,
            farmer=(position[1], position[0]),
            inventories=[inventory, {}],
        )
        current["farms"][0]["money"] = 0 if step == 0 else 500
        current["private"]["shed"] = shed
        result = controller.act(current, plan())
        route = result.diagnostics["route_diagnostics"][0]
        saw_late_place |= place_id in route["late_work_ids"]
        if result.market_actions and not purchase_observed:
            purchase_observed = True
            shed = {"SHEEP": 1}
        if result.farmer_action == ("PICKUP", "SHEEP", 1):
            shed = {}
            inventory = {"SHEEP": 1}
        if result.farmer_action == ("PLACE", "SHEEP", 1):
            placed = True
            break
        y, x = position
        if result.farmer_action == ("NORTH",):
            y -= 1
        elif result.farmer_action == ("SOUTH",):
            y += 1
        elif result.farmer_action == ("EAST",):
            x += 1
        elif result.farmer_action == ("WEST",):
            x -= 1
        position = (y, x)

    assert saw_late_place
    assert placed


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


def test_crop_chain_continuation_runs_before_departure():
    """A newly ready local crop stage outranks advancing the frozen sweep."""

    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            local = work_item("HARVEST", (0, 0), crop="WHEAT")
        elif hour == 1:
            local = work_item("PLANT", (0, 0), crop="TOMATO")
        elif hour == 2:
            local = work_item("WATER", (0, 0), crop="TOMATO")
        else:
            local = None
        items = [work_item("WATER", (0, 4))]
        if local is not None:
            items.append(local)
        return fake_plan(tuple(items))

    controller = StripExecutorController(work_builder=builder)
    route = None
    for hour, expected in enumerate(("HARVEST", "PLANT", "WATER")):
        result = controller.act(observation(hour=hour, farmer=(0, 0)), plan())
        route = controller.routes[0]
        assert result.farmer_action[0] == expected
        assert route.cursor == 0
        assert (0, 0) not in route.passed_tiles

    result = controller.act(observation(hour=3, farmer=(0, 0)), plan())
    assert result.farmer_action == ("EAST",)
    assert route is not None
    assert (0, 0) not in route.passed_tiles


def test_late_crop_successor_reopens_only_the_previous_owned_tile():
    harvest_id = "HARVEST:0,0"
    plant_id = "PLANT:TOMATO:0,0"
    water_id = "WATER:0,0"

    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            items = (
                work_item("HARVEST", (0, 0), crop="WHEAT", item_id=harvest_id),
                work_item(
                    "PLANT",
                    (0, 0),
                    crop="TOMATO",
                    status=WorkStatus.BLOCKED,
                    depends_on=(harvest_id,),
                    block_reason=BlockReason.DEPENDENCY_BLOCKED,
                    item_id=plant_id,
                ),
            )
        elif hour == 1:
            items = (
                work_item(
                    "PLANT",
                    (0, 0),
                    crop="TOMATO",
                    status=WorkStatus.BLOCKED,
                    depends_on=(harvest_id,),
                    block_reason=BlockReason.DEPENDENCY_BLOCKED,
                    item_id=plant_id,
                ),
            )
        elif hour == 2:
            items = (
                work_item("PLANT", (0, 0), crop="TOMATO", item_id=plant_id),
                work_item(
                    "WATER",
                    (0, 0),
                    crop="TOMATO",
                    status=WorkStatus.BLOCKED,
                    depends_on=(plant_id,),
                    block_reason=BlockReason.DEPENDENCY_BLOCKED,
                    item_id=water_id,
                ),
            )
        else:
            items = (work_item("WATER", (0, 0), crop="TOMATO", item_id=water_id),)
        return fake_plan(items + (work_item("WATER", (0, 4)),))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    departure = controller.act(observation(hour=1, farmer=(0, 0)), plan())
    assert departure.farmer_action == ("EAST",)

    reopened = controller.act(observation(hour=2, farmer=(1, 0)), plan())
    route = controller.routes[0]
    assert reopened.farmer_action == ("WEST",)
    assert route.cursor == 0
    assert (0, 0) not in route.passed_tiles
    assert plant_id not in route.late_work_ids

    continuation = controller.act(observation(hour=3, farmer=(0, 0)), plan())
    assert continuation.farmer_action == ("WATER",)


def test_dig_replacement_chain_progresses_to_plant_and_water():
    dig_id = "DIG:0,0"
    plant_id = "PLANT:WHEAT:0,0"
    water_id = "WATER:0,0"

    def builder(obs, plan, **kwargs):
        del plan, kwargs
        hour = int(obs["hour"])
        if hour == 0:
            items = (
                work_item("DIG", (0, 0), item_id=dig_id),
                work_item(
                    "PLANT",
                    (0, 0),
                    crop="WHEAT",
                    status=WorkStatus.BLOCKED,
                    depends_on=(dig_id,),
                    block_reason=BlockReason.DEPENDENCY_BLOCKED,
                    item_id=plant_id,
                ),
            )
        elif hour == 1:
            items = (
                work_item("PLANT", (0, 0), crop="WHEAT", item_id=plant_id),
                work_item(
                    "WATER",
                    (0, 0),
                    crop="WHEAT",
                    status=WorkStatus.BLOCKED,
                    depends_on=(plant_id,),
                    block_reason=BlockReason.DEPENDENCY_BLOCKED,
                    item_id=water_id,
                ),
            )
        else:
            items = (work_item("WATER", (0, 0), crop="WHEAT", item_id=water_id),)
        return fake_plan(items + (work_item("WATER", (0, 4)),))

    controller = StripExecutorController(work_builder=builder)
    for hour, expected in enumerate(("DIG", "PLANT", "WATER")):
        result = controller.act(observation(hour=hour, farmer=(0, 0)), plan())
        assert result.farmer_action == ((expected, "WHEAT") if expected == "PLANT" else (expected,))


def test_crop_continuation_does_not_reopen_an_older_tile():
    harvest_id = "HARVEST:0,0"
    plant_id = "PLANT:TOMATO:0,0"

    def builder(obs, plan, **kwargs):
        del plan, kwargs
        if int(obs["hour"]) == 0:
            harvest = work_item("HARVEST", (0, 0), crop="WHEAT", item_id=harvest_id)
        else:
            harvest = work_item(
                "PLANT",
                (0, 0),
                crop="TOMATO",
                status=(
                    WorkStatus.BLOCKED
                    if int(obs["hour"]) < 3
                    else WorkStatus.READY
                ),
                depends_on=(harvest_id,) if int(obs["hour"]) < 3 else (),
                block_reason=(
                    BlockReason.DEPENDENCY_BLOCKED
                    if int(obs["hour"]) < 3
                    else None
                ),
                item_id=plant_id,
            )
        return fake_plan((harvest, work_item("WATER", (0, 4))))

    controller = StripExecutorController(work_builder=builder)
    controller.act(observation(hour=0, farmer=(0, 0)), plan())
    controller.act(observation(hour=1, farmer=(0, 0)), plan())
    controller.act(observation(hour=2, farmer=(1, 0)), plan())
    result = controller.act(observation(hour=3, farmer=(2, 0)), plan())
    assert result.farmer_action == ("EAST",)
    assert plant_id in controller.routes[0].late_work_ids


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


def _row_segment(segment_id: str, row: int) -> RouteSegment:
    traversal = tuple((row, col) for col in range(5))
    return RouteSegment(segment_id, traversal, traversal[0], 0)


def _chain_route(
    route_id: str,
    worker: WorkerId,
    segments: tuple[RouteSegment, ...],
    *,
    cursor: int,
    pending_cursor: int | None = None,
    phase: RoutePhase = RoutePhase.SWEEP,
) -> StripRoute:
    traversal = tuple(tile for segment in segments for tile in segment.traversal)
    return StripRoute(
        route_id,
        traversal,
        traversal,
        worker,
        traversal[0],
        0,
        0,
        cursor=cursor,
        pending_cursor=pending_cursor,
        phase=phase,
        segments=tuple(segments),
    )


def _register_helping_routes(
    controller: StripExecutorController, *routes: StripRoute
) -> None:
    controller._routes = {route.owner: route for route in routes}
    controller._supply_plans = {}
    controller._supply_states = {}


def test_helping_never_steals_segment_a_donor_is_already_departing_toward():
    """Regression: the official crash had donor.pending_cursor point into the
    segment a completed neighbor stole, indexing past the shortened traversal."""

    controller = StripExecutorController()
    a = _row_segment("A", 0)
    b = _row_segment("B", 1)
    # Donor sits on A's final tile and has already emitted movement toward B.
    donor = _chain_route(
        "donor", WorkerId(0), (a, b), cursor=4, pending_cursor=5
    )
    own = _row_segment("OWN", 2)
    receiver = _chain_route(
        "receiver", WorkerId(1), (own,), cursor=4, phase=RoutePhase.DONE
    )
    _register_helping_routes(controller, donor, receiver)

    result = controller._act_worker(
        receiver,
        own.traversal[-1],
        fake_plan(()),
        observation(hour=0, farmer=own.traversal[-1]),
    )
    assert result == ("PASS",)

    # B was not transferred: the donor still owns it and both cursors are valid.
    assert tuple(segment.segment_id for segment in donor.segments) == ("A", "B")
    assert donor.traversal == a.traversal + b.traversal
    assert donor.pending_cursor == 5
    assert route_cursor_invariants_hold(donor)
    assert not receiver.transferred_segment_ids

    # The donor's next turn must not raise and must keep heading toward B.
    follow = controller._act_worker(
        donor,
        a.traversal[-1],
        fake_plan(()),
        observation(hour=1, farmer=a.traversal[-1]),
    )
    assert follow == ("SOUTH",)
    assert donor.pending_cursor == 5
    assert route_cursor_invariants_hold(donor)


def test_helping_takes_a_later_segment_the_donor_has_not_committed_to():
    controller = StripExecutorController()
    a = _row_segment("A", 0)
    b = _row_segment("B", 1)
    c = _row_segment("C", 2)
    # pending_cursor points into B, so only the genuinely untouched C may move.
    donor = _chain_route(
        "donor", WorkerId(0), (a, b, c), cursor=4, pending_cursor=5
    )
    own = _row_segment("OWN", 3)
    receiver = _chain_route(
        "receiver", WorkerId(1), (own,), cursor=4, phase=RoutePhase.DONE
    )
    _register_helping_routes(controller, donor, receiver)

    controller._act_worker(
        receiver,
        own.traversal[-1],
        fake_plan(()),
        observation(hour=0, farmer=own.traversal[-1]),
    )

    assert tuple(segment.segment_id for segment in donor.segments) == ("A", "B")
    assert donor.traversal == a.traversal + b.traversal
    assert donor.pending_cursor == 5
    assert route_cursor_invariants_hold(donor)
    assert "C" in receiver.transferred_segment_ids
    assert c.traversal[0] in receiver.traversal
    assert route_cursor_invariants_hold(receiver)


def test_route_cursor_invariant_helper_flags_invalid_pending_cursor():
    a = _row_segment("A", 0)
    route = _chain_route("r", WorkerId(0), (a,), cursor=4)
    assert route_cursor_invariants_hold(route)
    route.pending_cursor = len(route.traversal)
    assert not route_cursor_invariants_hold(route)
    route.pending_cursor = -1
    assert not route_cursor_invariants_hold(route)
