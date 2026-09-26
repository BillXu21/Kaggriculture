from __future__ import annotations

from dataclasses import replace

from executor_v0.strip_cost import (
    nearest_shed_access,
    route_cost_segment_from_forecast,
    route_cost_segment_from_items,
    simulate_route_cost,
)
from executor_v0.strip_work import SupplyRequirement, WorkItem, row_key_for_tile


def work(
    kind: str,
    tile: tuple[int, int],
    *,
    item_id: str | None = None,
    crop: str | None = None,
    source: str = "strip_forecast",
    quantity: int = 1,
    supplies: tuple[SupplyRequirement, ...] = (),
) -> WorkItem:
    return WorkItem(
        id=item_id or f"{kind}:{tile[0]}:{tile[1]}",
        kind=kind,
        tile=tile,
        crop=crop,
        source=source,
        quantity=quantity,
        required_supplies=supplies,
        row_key=row_key_for_tile(tile),
    )


def row_segment(segment_id: str, items: tuple[WorkItem, ...]):
    traversal = tuple((0, x) for x in range(5))
    return route_cost_segment_from_items(segment_id, traversal, items)


def distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def test_five_tile_water_row_cost_is_five_actions_plus_four_moves():
    result = simulate_route_cost(
        (0, 0),
        (row_segment("row", tuple(work("WATER", (0, x)) for x in range(5))),),
        remaining_action_slots=9,
    )

    assert result.total_turns == 9
    assert result.horizontal_sweep_turns == 4
    assert result.represented_interaction_turns == 5
    assert result.pickup_travel_turns == result.pickup_action_turns == 0
    assert result.route_complete_before_deadline is True


def test_entry_distance_is_charged_and_exact_boundary_is_inclusive():
    segment = row_segment("row", tuple(work("WATER", (0, x)) for x in range(5)))
    on_boundary = simulate_route_cost(
        (3, 0), (segment,), remaining_action_slots=12
    )
    short_by_one = simulate_route_cost(
        (3, 0), (segment,), remaining_action_slots=11
    )

    assert on_boundary.total_turns == 12
    assert on_boundary.setup_travel_turns == 3
    assert on_boundary.route_complete_before_deadline is True
    assert short_by_one.route_complete_before_deadline is False
    assert short_by_one.effective_interactions_missed == 1


def test_forecast_fallback_keeps_non_hire_driving_tail_separate():
    segment = route_cost_segment_from_forecast(
        "mixed",
        ((0, 0),),
        represented_interactions=(2,),
        hire_driving_interactions=(1,),
    )
    result = simulate_route_cost(
        (0, 0), (segment,), remaining_action_slots=1
    )

    assert result.effective_interaction_turns == 2
    assert result.hire_driving_interaction_turns == 1
    assert result.hire_driving_interactions_missed == 0
    assert result.effective_interactions_missed == 1


def test_packer_summary_and_detailed_cost_share_all_route_accounting():
    segment = route_cost_segment_from_items(
        "row",
        ((0, 0), (0, 1), (0, 2)),
        (
            work("FEED", (0, 0), supplies=(SupplyRequirement("WHEAT", 1),)),
            work("WATER", (0, 2)),
        ),
    )
    detailed = simulate_route_cost(
        (4, 4),
        (segment,),
        carried_inventory={},
        remaining_action_slots=24,
        shed_stock={"WHEAT": 1},
    )
    summary = simulate_route_cost(
        (4, 4),
        (segment,),
        carried_inventory={},
        remaining_action_slots=24,
        shed_stock={"WHEAT": 1},
        include_segment_results=False,
    )

    assert summary == replace(detailed, segment_results=())


def test_retained_harvest_continuations_are_charged_once():
    harvests = tuple(
        work(
            "HARVEST",
            (0, x),
            crop=crop,
            source="routine_harvest",
        )
        for x, crop in enumerate(("WHEAT", "CARROT", "MELON"))
    )
    segment = row_segment("row", harvests)
    result = simulate_route_cost(
        (0, 0),
        (segment,),
        remaining_action_slots=13,
        global_resources={"WHEAT": 1, "CARROT": 1, "MELON": 1},
    )

    assert result.represented_interaction_turns == 3
    assert result.known_continuation_turns == 6
    assert result.effective_interaction_turns == 9
    assert result.total_turns == 13  # four sweep moves plus all nine actions
    assert result.route_complete_before_deadline is True


def test_explicit_plant_and_water_suppress_duplicate_continuations():
    items = (
        work(
            "HARVEST",
            (0, 0),
            item_id="harvest",
            crop="WHEAT",
            source="routine_harvest",
        ),
        work(
            "PLANT",
            (0, 0),
            item_id="plant",
            crop="WHEAT",
            supplies=(SupplyRequirement("WHEAT", 1, "global_seed"),),
        ),
        work("WATER", (0, 0), item_id="water"),
    )
    result = simulate_route_cost(
        (0, 0),
        (row_segment("row", items),),
        remaining_action_slots=6,
        global_resources={"WHEAT": 1},
    )

    assert result.represented_interaction_turns == 3
    assert result.known_continuation_turns == 0
    assert result.effective_interaction_turns == 3


def test_continuation_resource_shortage_is_not_hire_driving_work():
    harvest = work(
        "HARVEST",
        (0, 0),
        crop="MELON",
        source="routine_harvest",
    )
    result = simulate_route_cost(
        (0, 0),
        (row_segment("row", (harvest,)),),
        remaining_action_slots=24,
        global_resources={},
    )

    assert result.effective_interaction_turns == 3
    assert result.feasible_effective_interaction_turns == 1
    assert result.resource_feasible is False
    assert result.hire_driving_interactions_missed == 0
    assert result.global_shortage == (("MELON", 1),)


def test_carried_quantity_satisfies_exact_inventory_demand():
    feed = work(
        "FEED",
        (0, 0),
        supplies=(SupplyRequirement("WHEAT", 3),),
    )
    result = simulate_route_cost(
        (0, 0),
        (row_segment("row", (feed,)),),
        carried_inventory={"WHEAT": 3},
        remaining_action_slots=1,
        shed_stock={},
    )

    assert result.supply_quantities_required == (("WHEAT", 3),)
    assert result.supply_quantities_already_carried == (("WHEAT", 3),)
    assert result.supply_quantities_requiring_pickup == ()
    assert result.pickup_travel_turns == result.pickup_action_turns == 0
    assert result.total_turns == 5  # one FEED action plus the row's four moves


def test_partial_carried_quantity_batches_missing_units_in_one_pickup():
    feed = work(
        "FEED",
        (0, 0),
        supplies=(SupplyRequirement("WHEAT", 3),),
    )
    start = (0, 0)
    access = nearest_shed_access(start)
    result = simulate_route_cost(
        start,
        (row_segment("row", (feed,)),),
        carried_inventory={"WHEAT": 1},
        remaining_action_slots=24,
        shed_stock={"WHEAT": 2},
    )

    assert result.supply_quantities_requiring_pickup == (("WHEAT", 2),)
    assert result.pickup_sequence == (("WHEAT", 2),)
    assert result.pickup_action_turns == 1
    assert result.pickup_travel_turns == distance(start, access)
    assert result.setup_travel_turns == distance(access, (0, 0))
    assert result.total_turns == (
        distance(start, access)
        + 1
        + distance(access, (0, 0))
        + 4
        + 1
    )


def test_two_inventory_types_share_one_shed_detour_but_need_two_pickups():
    items = (
        work(
            "FEED",
            (4, 4),
            supplies=(SupplyRequirement("WHEAT", 2),),
        ),
        work(
            "FERTILIZE",
            (4, 4),
            supplies=(SupplyRequirement("FERTILIZER", 3),),
        ),
    )
    segment = route_cost_segment_from_items("row", ((4, 4),), items)
    result = simulate_route_cost(
        (4, 4),
        (segment,),
        remaining_action_slots=4,
        shed_stock={"WHEAT": 2, "FERTILIZER": 3},
    )

    assert result.pickup_tile == (4, 4)
    assert result.pickup_travel_turns == 0
    assert result.setup_travel_turns == 0
    assert result.pickup_sequence == (("WHEAT", 2), ("FERTILIZER", 3))
    assert result.pickup_action_turns == 2
    assert result.total_turns == 4


def test_chained_segments_keep_position_elapsed_and_upfront_supply_batch():
    first = route_cost_segment_from_items(
        "row-a",
        ((0, 0), (0, 1)),
        (work("FEED", (0, 0), item_id="feed-a", supplies=(SupplyRequirement("WHEAT"),)),),
    )
    second = route_cost_segment_from_items(
        "row-b",
        ((1, 1), (1, 2)),
        (work("FEED", (1, 1), item_id="feed-b", supplies=(SupplyRequirement("WHEAT"),)),),
    )
    result = simulate_route_cost(
        (4, 4),
        (first, second),
        remaining_action_slots=24,
        shed_stock={"WHEAT": 2},
    )
    first_result, second_result = result.segment_results

    assert result.supply_quantities_required == (("WHEAT", 2),)
    assert result.pickup_sequence == (("WHEAT", 2),)
    assert result.pickup_action_turns == 1
    assert second_result.start_position == first_result.end_position == (0, 1)
    assert second_result.start_elapsed_turns == first_result.completion_elapsed_turns
    assert second_result.completion_elapsed_turns > first_result.completion_elapsed_turns


def test_near_deadline_exact_shed_detour_misses_work():
    items = tuple(
        work(
            "FEED" if x == 0 else "WATER",
            (0, x),
            supplies=(SupplyRequirement("WHEAT", 1),) if x == 0 else (),
        )
        for x in range(5)
    )
    segment = row_segment("row", items)
    result = simulate_route_cost(
        (0, 0),
        (segment,),
        assignment_turn=13,
        remaining_action_slots=11,
        shed_stock={"WHEAT": 1},
    )

    assert result.completion_hour == 13 + result.total_turns
    assert result.total_turns > 11
    assert result.route_complete_before_deadline is False
    assert result.effective_interactions_missed > 0
