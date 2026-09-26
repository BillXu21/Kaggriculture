from __future__ import annotations

from collections import defaultdict

from executor_v0.strip_cost import simulate_route_cost
from executor_v0.strip_routes import (
    WorkerId,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
)
from executor_v0.strip_work import (
    RowSummary,
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    row_key_for_tile,
)


def item(
    kind: str,
    tile: tuple[int, int],
    *,
    item_id: str | None = None,
    supplies: tuple[SupplyRequirement, ...] = (),
) -> WorkItem:
    return WorkItem(
        id=item_id or f"{kind}:{tile[0]}:{tile[1]}",
        kind=kind,
        tile=tile,
        animal="COW" if kind == "FEED" else None,
        required_supplies=supplies,
        row_key=row_key_for_tile(tile),
    )


def work_plan(*items: WorkItem) -> StripWorkPlan:
    rows: dict[object, list[WorkItem]] = defaultdict(list)
    for work in items:
        if work.row_key is not None:
            rows[work.row_key].append(work)
    summaries = tuple(
        RowSummary(
            key,
            5,
            sum(value.interaction_turns for value in values if value.ready),
            sum(value.interaction_turns for value in values if not value.ready),
            len(values),
        )
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


def test_near_deadline_shed_detour_triggers_canonical_row_overload():
    items = (
        item("FEED", (0, 0), supplies=(SupplyRequirement("WHEAT", 1),)),
        *(item("WATER", (0, x)) for x in range(1, 5)),
    )
    candidates = generate_horizontal_route_candidates(work_plan(*items))
    worker = WorkerId(0)
    position = {worker: (0, 0)}
    assignment = assign_horizontal_routes(
        candidates,
        position,
        assignment_hour=13,
        worker_action_slots={worker: 11},
        shed_stock={"WHEAT": 1},
        enable_row_helpers=False,
    )
    route = assignment.routes[0]
    cost = simulate_route_cost(
        position[worker],
        tuple(segment.cost_segment for segment in route.segments),
        assignment_turn=13,
        remaining_action_slots=11,
        shed_stock={"WHEAT": 1},
    )

    # Omitting shed travel leaves a misleading ten-turn route (hour 23 from
    # hour 13); the actual setup plus work cannot fit in eleven slots.
    assert cost.total_turns > 11
    assert cost.pickup_travel_turns > 0
    assert cost.route_complete_before_deadline is False
    assert cost.effective_interactions_missed > 0
    assert assignment.row_diagnostics[0]["helper_required"] is True


def test_row_overload_detection_carries_prior_segment_position_and_elapsed_time():
    items = tuple(
        item("WATER", (row, x), item_id=f"WATER:{row}:{x}")
        for row in (0, 1)
        for x in range(5)
    )
    candidates = generate_horizontal_route_candidates(work_plan(*items))
    worker = WorkerId(0)
    assignment = assign_horizontal_routes(
        candidates,
        {worker: (0, 0)},
        assignment_hour=0,
        worker_action_slots={worker: 14},
        enable_row_helpers=False,
    )

    route = assignment.routes[0]
    first, second = route.segments
    first_diagnostic = next(
        value
        for value in assignment.row_diagnostics
        if value["physical_row_id"] == first.physical_row_id
    )
    first_cost = first_diagnostic["canonical_cost"]["segment"]
    assert route.segments[0].physical_row_id == "ROW:NW:0:0-4"
    assert route.segments[1].physical_row_id == "ROW:NW:1:0-4"
    assert first_cost["start_position"] == (0, 0)
    assert first_cost["end_position"] == (0, 4)
    # The row diagnostic embeds its own canonical segment result and does not
    # restart this worker from the morning position for the second row.
    second_diagnostic = next(
        value
        for value in assignment.row_diagnostics
        if value["physical_row_id"] == second.physical_row_id
    )
    second_cost = second_diagnostic["canonical_cost"]["segment"]
    assert second_cost["start_position"] == (0, 4)
    assert second_cost["start_elapsed_turns"] == first_cost["completion_elapsed_turns"] == 9
    assert second_diagnostic["helper_required"] is True


def test_overload_split_uses_canonical_quantity_aware_helper_cost():
    items = tuple(
        item("CARE", (0, x), item_id=f"CARE:{x}:{index}")
        for x in range(3)
        for index in range(3)
    ) + (
        item("WATER", (0, 3)),
        item("FEED", (0, 4), supplies=(SupplyRequirement("WHEAT", 1),)),
    )
    candidates = generate_horizontal_route_candidates(work_plan(*items))
    primary = WorkerId(0)
    helper = WorkerId(1)
    positions = {primary: (0, 0), helper: (4, 4)}
    inventories = {primary: {"WHEAT": 1}, helper: {}}
    assignment = assign_horizontal_routes(
        candidates,
        positions,
        assignment_hour=10,
        worker_action_slots={primary: 14, helper: 14},
        worker_inventories=inventories,
        shed_stock={"WHEAT": 1},
    )

    row = assignment.row_diagnostics[0]
    assert assignment.overloaded_rows_detected == 1
    assert assignment.row_helpers_assigned == 1
    assert row["row_overload_resolved"] is True
    assert row["helper_tiles"] == [[0, 4]]
    helper_route = next(route for route in assignment.routes if route.owner == helper)
    canonical_helper = simulate_route_cost(
        positions[helper],
        tuple(segment.cost_segment for segment in helper_route.segments),
        remaining_action_slots=14,
        shed_stock={"WHEAT": 1},
    )
    diagnostic_helper = row["canonical_cost"]["helper_fragment"]

    assert canonical_helper.supply_quantities_required == (("WHEAT", 1),)
    assert canonical_helper.supply_quantities_requiring_pickup == (("WHEAT", 1),)
    assert canonical_helper.pickup_travel_turns == 0
    assert canonical_helper.pickup_action_turns == 1
    assert canonical_helper.setup_travel_turns == 4
    assert canonical_helper.total_turns == diagnostic_helper["total_turns"] == 6


def test_helper_split_does_not_reuse_supply_reserved_by_another_route():
    items = tuple(
        item("CARE", (0, x), item_id=f"CARE:{x}:{index}")
        for x in range(5)
        for index in range(5)
    ) + (
        item("FEED", (0, 4), supplies=(SupplyRequirement("WHEAT", 1),)),
        item("FEED", (1, 0), supplies=(SupplyRequirement("WHEAT", 1),)),
    )
    candidates = generate_horizontal_route_candidates(work_plan(*items))
    assignment = assign_horizontal_routes(
        candidates,
        {
            WorkerId(0): (0, 0),
            WorkerId(1): (1, 0),
            WorkerId(2): (0, 4),
        },
        assignment_hour=0,
        worker_action_slots={WorkerId(index): 24 for index in range(3)},
        shed_stock={"WHEAT": 1},
    )

    row = next(
        value
        for value in assignment.row_diagnostics
        if value["physical_row_id"] == candidates[0].row_id
    )
    assert row["helper_required"] is True
    assert row["row_overload_resolved"] is False
    assert assignment.row_helpers_assigned == 0


def test_missing_supply_does_not_trigger_a_hire_driving_helper():
    candidate = generate_horizontal_route_candidates(
        work_plan(
            item("FEED", (0, 0), supplies=(SupplyRequirement("WHEAT", 1),))
        )
    )
    worker = WorkerId(0)
    assignment = assign_horizontal_routes(
        candidate,
        {worker: (0, 0)},
        assignment_hour=8,
        worker_action_slots={worker: 16},
        shed_stock={},
        enable_row_helpers=False,
    )

    row = assignment.row_diagnostics[0]
    route = assignment.routes[0]
    cost = simulate_route_cost(
        (0, 0),
        tuple(segment.cost_segment for segment in route.segments),
        remaining_action_slots=16,
        shed_stock={},
    )
    assert cost.resource_feasible is False
    assert cost.hire_driving_interactions_missed == 0
    assert row["helper_required"] is False


def test_one_helper_insufficient_is_reported_without_a_third_worker():
    items = tuple(
        item("CARE", (0, x), item_id=f"CARE:{x}:{index}")
        for x in range(5)
        for index in range(40)
    )
    candidates = generate_horizontal_route_candidates(work_plan(*items))
    assignment = assign_horizontal_routes(
        candidates,
        {WorkerId(0): (0, 0), WorkerId(1): (0, 4)},
        assignment_hour=0,
        worker_action_slots={WorkerId(0): 24, WorkerId(1): 24},
    )

    assert assignment.row_diagnostics[0]["helper_required"] is True
    assert assignment.row_diagnostics[0]["row_overload_resolved"] is False
    assert assignment.unresolved_overloaded_rows == 1
    assert assignment.row_helpers_assigned == 0
    assert len(assignment.routes) == 1
