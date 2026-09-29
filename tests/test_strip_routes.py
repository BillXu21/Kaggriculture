from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from executor_v0.strip_cost import simulate_route_cost
from executor_v0.strip_routes import (
    WorkerId,
    assign_horizontal_routes,
    assign_horizontal_routes_frontier,
    generate_horizontal_route_candidates,
    route_assignment_fingerprint,
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


def _frontier_candidates(count: int, *, blocked: bool = False):
    values = []
    for row in range(count):
        if blocked and row == 1:
            values.append(
                item(
                    "FEED",
                    (row, 0),
                    supplies=(SupplyRequirement("WHEAT", 1),),
                )
            )
        else:
            values.append(item("WATER", (row, 0)))
    return generate_horizontal_route_candidates(work_plan(*values))


def _assert_frontier_matches_independent_calls(
    candidates,
    worker_positions,
    *,
    worker_action_slots=None,
    remaining_action_slots=None,
    worker_inventories=None,
    shed_stock=None,
    global_resources=None,
    enable_row_helpers=True,
):
    workers = tuple(sorted(worker_positions))
    counts = tuple(range(len(workers) + 1))
    actual = assign_horizontal_routes_frontier(
        candidates,
        worker_positions,
        worker_counts=counts,
        assignment_hour=7,
        remaining_action_slots=remaining_action_slots,
        worker_action_slots=worker_action_slots,
        worker_inventories=worker_inventories,
        shed_stock=shed_stock,
        global_resources=global_resources,
        enable_row_helpers=enable_row_helpers,
    )
    for count in counts:
        prefix = workers[:count]
        expected = assign_horizontal_routes(
            candidates,
            {worker: worker_positions[worker] for worker in prefix},
            assignment_hour=7,
            remaining_action_slots=remaining_action_slots,
            worker_action_slots=(
                None
                if worker_action_slots is None
                else {worker: worker_action_slots[worker] for worker in prefix}
            ),
            worker_inventories=(
                None
                if worker_inventories is None
                else {
                    worker: worker_inventories[worker]
                    for worker in prefix
                    if worker in worker_inventories
                }
            ),
            shed_stock=shed_stock,
            global_resources=global_resources,
            enable_row_helpers=enable_row_helpers,
        )
        assert actual[count] == expected
    return actual


def test_assignment_frontier_matches_each_exact_small_board_prefix():
    candidates = _frontier_candidates(4, blocked=True)
    positions = {
        WorkerId(0): (2, 2),
        WorkerId(1): (0, 4),
        WorkerId(2): (4, 0),
        WorkerId(3): (8, 8),
        WorkerId(4): (1, 9),
        WorkerId(5): (9, 1),
    }
    slots = {worker: 4 + worker.index for worker in positions}
    inventories = {worker: {} for worker in positions}
    actual = _assert_frontier_matches_independent_calls(
        candidates,
        positions,
        worker_action_slots=slots,
        worker_inventories=inventories,
        shed_stock={},
        global_resources={},
    )

    # The last two workers are surplus to this four-route exact-packer case.
    assert len(actual[6].idle_workers) >= 2
    assert len(actual[1].routes[0].segments) == len(candidates)


def test_assignment_frontier_preserves_orientation_ties_and_is_repeatable():
    candidates = _frontier_candidates(1)
    positions = {WorkerId(0): (2, 2), WorkerId(1): (8, 8), WorkerId(2): (2, 4)}
    first = _assert_frontier_matches_independent_calls(candidates, positions)
    second = assign_horizontal_routes_frontier(
        candidates,
        positions,
        worker_counts=range(len(positions) + 1),
        assignment_hour=7,
    )
    assert first == second
    assert first[1].routes[0].segments[0].traversal == candidates[0].owned_tiles


def test_assignment_frontier_matches_global_remaining_slot_budget():
    _assert_frontier_matches_independent_calls(
        _frontier_candidates(3),
        {WorkerId(0): (4, 0), WorkerId(1): (0, 4), WorkerId(2): (9, 9)},
        remaining_action_slots=11,
    )


def test_assignment_frontier_matches_each_large_board_prefix():
    candidates = _frontier_candidates(9, blocked=True)
    positions = {
        WorkerId(0): (0, 0),
        WorkerId(1): (8, 4),
        WorkerId(2): (4, 9),
        WorkerId(3): (9, 0),
    }
    slots = {WorkerId(0): 18, WorkerId(1): 18, WorkerId(2): 7, WorkerId(3): 7}
    inventories = {worker: {} for worker in positions}
    actual = _assert_frontier_matches_independent_calls(
        candidates,
        positions,
        worker_action_slots=slots,
        worker_inventories=inventories,
        shed_stock={},
        global_resources={},
    )
    repeated = assign_horizontal_routes_frontier(
        candidates,
        positions,
        worker_counts=range(len(positions) + 1),
        assignment_hour=7,
        worker_action_slots=slots,
        worker_inventories=inventories,
        shed_stock={},
        global_resources={},
    )
    assert actual == repeated
    assert actual[1].large_route_assignment_mode is True
    assert len(actual[1].unassigned) > 0
    assert len(actual[4].routes) <= 4


def test_assignment_fingerprint_uses_complete_values_not_object_identity():
    candidates = _frontier_candidates(2)
    positions = {WorkerId(0): (1, 1), WorkerId(1): (8, 8)}
    slots = {WorkerId(0): 12, WorkerId(1): 11}
    inventories = {WorkerId(0): {"WHEAT": 2}, WorkerId(1): {}}
    args = {
        "assignment_hour": 3,
        "remaining_action_slots": 12,
        "worker_action_slots": slots,
        "worker_inventories": inventories,
        "shed_stock": {"WHEAT": 4},
        "global_resources": {"WHEAT": 9},
        "enable_row_helpers": False,
    }

    first = route_assignment_fingerprint(candidates, positions, **args)
    equivalent = route_assignment_fingerprint(
        tuple(candidates), dict(positions), **{
            **args,
            "worker_action_slots": dict(slots),
            "worker_inventories": {
                worker: dict(inventory)
                for worker, inventory in inventories.items()
            },
            "shed_stock": dict(args["shed_stock"]),
            "global_resources": dict(args["global_resources"]),
        }
    )
    assert first == equivalent
    assert hash(first) == hash(equivalent)

    changed_inputs = (
        ((replace(
            candidates[0],
            workload_interactions=candidates[0].workload_interactions + 1,
        ), candidates[1]), positions, args),
        (candidates, {WorkerId(0): (1, 2), WorkerId(1): (8, 8)}, args),
        (candidates, positions, {**args, "assignment_hour": 4}),
        (candidates, positions, {**args, "remaining_action_slots": 11}),
        (candidates, positions, {
            **args,
            "worker_action_slots": {**slots, WorkerId(0): 13},
        }),
        (candidates, positions, {
            **args,
            "worker_inventories": {
                WorkerId(0): {"WHEAT": 3}, WorkerId(1): {},
            },
        }),
        (candidates, positions, {**args, "shed_stock": {"WHEAT": 5}}),
        (candidates, positions, {**args, "global_resources": {"WHEAT": 8}}),
        (candidates, positions, {**args, "enable_row_helpers": True}),
    )
    for changed_candidates, changed_positions, changed_args in changed_inputs:
        assert route_assignment_fingerprint(
            changed_candidates, changed_positions, **changed_args
        ) != first
