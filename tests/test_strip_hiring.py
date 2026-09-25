"""Focused Packet 5 coverage-driven strip hiring tests."""

from __future__ import annotations

import copy

import executor_v0.strip_hiring as strip_hiring
from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_hiring import (
    HireStopReason,
    RouteLaborEstimate,
    StripHiringPlan,
    plan_strip_hiring,
)
from executor_v0.strip_routes import (
    WorkerId,
    assign_horizontal_routes,
    generate_horizontal_route_candidates,
    remaining_day_action_slots,
    route_cursor_invariants_hold,
)
from executor_v0.strip_work import (
    RowSummary,
    StripWorkPlan,
    SupplySnapshot,
    SupplyRequirement,
    WorkDiagnostics,
    WorkItem,
    WorkStatus,
    row_key_for_tile,
)


def daily_plan() -> DailyPlan:
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


def item(
    kind: str,
    row: int,
    *,
    source: str = "strip_forecast",
    status=WorkStatus.READY,
    requirements=(),
    x: int = 0,
    crop: str | None = None,
    animal: str | None = None,
    item_id: str | None = None,
) -> WorkItem:
    tile = (row, x)
    return WorkItem(
        id=item_id or f"{kind}:{row}:{x}",
        kind=kind,
        status=status,
        tile=tile,
        crop=crop,
        animal=animal,
        row_key=row_key_for_tile(tile),
        source=source,
        required_supplies=tuple(requirements),
    )


def work_plan(*items: WorkItem) -> StripWorkPlan:
    summaries = tuple(
        RowSummary(
            key,
            5,
            sum(value.interaction_turns for value in values if value.ready),
            sum(value.interaction_turns for value in values if not value.ready),
            len(values),
        )
        for key, values in sorted(
            ((key, [value for value in items if value.row_key == key])
             for key in sorted({value.row_key for value in items if value.row_key})),
        )
    )
    return StripWorkPlan(
        items=items,
        chains=(),
        row_summaries=summaries,
        supply=SupplySnapshot(),
        diagnostics=WorkDiagnostics(),
        acting_seat=0,
    )


def observation(*, hands=(), money=1000, hour=0, hires_today=0, shed=None) -> dict:
    farm = {
        "money": float(money),
        "hires_today": hires_today,
        "farmer": [0, 0],
        "hands": [[x, y] for x, y in hands],
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": 3,
        "hour": hour,
        "step": 3 * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": shed or {},
            "seeds": {"WHEAT": 10},
            "inventories": [{} for _ in range(len(hands) + 1)],
        },
        "configuration": {
            "boardSize": 10,
            "turnsPerDay": 24,
            "episodeSteps": 720,
            "maxMarketOrdersPerTurn": 10,
        },
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }


def hiring_plan(
    items,
    *,
    hands=(),
    money=1000,
    hour=0,
    hires_today=0,
    shed=None,
    inventories=None,
    max_orders=10,
    positions=None,
):
    forecast = work_plan(*items)
    candidates = generate_horizontal_route_candidates(forecast)
    if positions is None:
        positions = {WorkerId(0): (0, 0)}
        positions.update(
            {WorkerId(index): (0, 0) for index in range(1, len(hands) + 1)}
        )
    carried = {worker: {} for worker in positions}
    if inventories:
        carried.update(inventories)
    return plan_strip_hiring(
        observation(
            hands=hands,
            money=money,
            hour=hour,
            hires_today=hires_today,
            shed=shed,
        ),
        forecast,
        candidates,
        positions,
        carried,
        max_orders=max_orders,
    )


def _large_row_items(count):
    return [
        item("WATER", row, x=x)
        for row in range(8)
        for x in (0, 5)
    ][:count]


def _overloaded_row_items(row=0):
    animals = [
        item(
            kind,
            row,
            x=tile,
            source=f"routine_animal_{kind.lower()}",
            animal=animal,
            item_id=f"{kind}:{row}:{tile}",
        )
        for tile, animal in ((0, "COW"), (1, "SHEEP"))
        for kind in ("FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER")
    ]
    crops = [
        item(
            "HARVEST",
            row,
            x=tile,
            source="routine_harvest",
            crop=crop,
        )
        for tile, crop in zip((2, 3, 4), ("WHEAT", "CARROT", "MELON"), strict=True)
    ]
    return [*animals, *crops]


def hiring_plan_with_curve(monkeypatch, completed_by_workers, *, driving_total=None, **kwargs):
    """Keep the real planner boundary while fixing the packed estimator's curve."""

    def estimate(_candidates, worker_positions, *_args, **_kwargs):
        return (
            (),
            completed_by_workers[len(worker_positions)],
            driving_total if driving_total is not None else max(completed_by_workers.values()),
        )

    monkeypatch.setattr(strip_hiring, "_estimate_packed_workers", estimate)
    rows = [item("WATER", row) for row in range(len(completed_by_workers))]
    return hiring_plan(rows, **kwargs)


def test_multiple_useful_hires_reach_best_packed_coverage(monkeypatch):
    result = hiring_plan_with_curve(monkeypatch, {1: 1, 2: 3, 3: 5, 4: 5})
    assert result.target_workers == 3
    assert result.wanted_hires == 2
    assert result.orders == (("HIRE",), ("HIRE",))
    assert result.hire_reason == "additional_workers_reach_best_packed_coverage"


def test_existing_packed_capacity_wins_tie(monkeypatch):
    result = hiring_plan_with_curve(
        monkeypatch, {2: 5, 3: 5, 4: 5, 5: 5}, hands=((4, 4),)
    )
    assert result.target_workers == 2
    assert result.wanted_hires == 0
    assert result.orders == ()
    assert result.hire_reason == "covered_by_existing_packed_capacity"


def test_no_hire_driving_work_keeps_current_worker_target():
    result = hiring_plan([], hands=((4, 4),))
    assert result.target_workers == 2
    assert result.wanted_hires == 0
    assert result.stop_reason is HireStopReason.NO_HIRE_DRIVING_WORK


def test_incremental_coverage_hires_to_final_useful_count(monkeypatch):
    result = hiring_plan_with_curve(monkeypatch, {1: 2, 2: 3, 3: 4, 4: 5})
    assert result.target_workers == 4
    assert result.wanted_hires == 3
    assert result.orders == (("HIRE",),) * 3


def test_best_attainable_coverage_can_leave_work_unfinished(monkeypatch):
    result = hiring_plan_with_curve(
        monkeypatch, {1: 0, 2: 1, 3: 1}, driving_total=3
    )
    assert result.target_workers == 2
    assert result.wanted_hires == 1
    assert result.hire_reason == "additional_workers_reach_best_packed_coverage"


def test_no_extra_worker_useful_before_deadline(monkeypatch):
    result = hiring_plan_with_curve(
        monkeypatch, {1: 1, 2: 1, 3: 1}, driving_total=3
    )
    assert result.target_workers == 1
    assert result.orders == ()
    assert result.hire_reason == "no_extra_worker_useful_before_deadline"


def test_multi_hire_cash_uses_exact_sequential_fibonacci_prefix(monkeypatch):
    result = hiring_plan_with_curve(
        monkeypatch, {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        hires_today=1, money=3,
    )
    assert result.target_workers == 5
    assert result.wanted_hires == 4
    assert result.sequential_hire_costs == (1, 2, 3, 5)
    assert result.affordable_hires == result.submittable_hires == 2
    assert result.orders == (("HIRE",), ("HIRE",))
    assert result.stop_reason is HireStopReason.CASH


def test_multi_hire_order_cap_submits_legal_prefix(monkeypatch):
    result = hiring_plan_with_curve(
        monkeypatch, {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, max_orders=2
    )
    assert result.wanted_hires == result.affordable_hires == 4
    assert result.submittable_hires == 2
    assert result.orders == (("HIRE",), ("HIRE",))
    assert result.stop_reason is HireStopReason.ORDER_CAP


def test_large_board_hiring_targets_one_worker_per_useful_row():
    result = hiring_plan(_large_row_items(15), money=100_000, max_orders=10)

    assert result.target_workers == 15
    assert result.wanted_hires == 14
    assert result.affordable_hires == 14
    assert result.submittable_hires == 10
    assert result.orders == (("HIRE",),) * 10
    assert result.stop_reason is HireStopReason.ORDER_CAP


def test_large_board_hiring_adds_one_worker_for_an_overloaded_row():
    rows = [
        *_overloaded_row_items(8),
        *(item("WATER", row) for row in range(8)),
        item("WATER", 9),
    ]
    result = hiring_plan(rows, money=100_000, max_orders=10)

    assert result.target_workers == 11
    assert result.overloaded_rows_detected == 1
    assert result.row_helpers_required == 1
    assert result.row_helpers_assigned == 1
    assert result.unresolved_overloaded_rows == 0


def test_small_board_overload_adds_helper_without_changing_exact_no_overload_pack():
    result = hiring_plan(
        _overloaded_row_items(),
        money=100_000,
        positions={WorkerId(0): (5, 0)},
    )
    assert result.target_workers == 2
    assert result.row_helpers_required == 1
    assert result.row_helpers_assigned == 1


def test_two_helper_orders_continue_toward_target_after_cap_observation():
    rows = []
    overloaded_candidates = {(9, 0), (9, 5)}
    for index in range(15):
        if index < 2:
            row, x_start = 9, index * 5
        else:
            ordinary_index = index - 2
            row = ordinary_index // 2
            x_start = 0 if ordinary_index % 2 == 0 else 5
        overload = (row, x_start) in overloaded_candidates
        if overload:
            animal_tiles = (x_start, x_start + 1)
            rows.extend(
                item(
                    kind,
                    row,
                    x=tile,
                    source=f"routine_animal_{kind.lower()}",
                    animal=animal,
                    item_id=f"{kind}:{index}:{tile}",
                )
                for tile, animal in zip(animal_tiles, ("COW", "SHEEP"), strict=True)
                for kind in ("FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER")
            )
            rows.extend(
                item(
                    "HARVEST",
                    row,
                    x=x_start + tile,
                    source="routine_harvest",
                    crop=crop,
                    item_id=f"HARVEST:{index}:{tile}",
                )
                for tile, crop in zip(
                    (2, 3, 4), ("WHEAT", "CARROT", "MELON"), strict=True
                )
            )
        else:
            rows.append(item("WATER", row, x=x_start))
    forecast = work_plan(*rows)
    controller = StripExecutorController(
        work_builder=lambda obs, daily_plan, **kwargs: forecast
    )

    first = controller.act(observation(money=100_000), daily_plan())
    assert first.diagnostics["hiring_diagnostics"]["target_workers"] == 17
    assert first.diagnostics["hiring_diagnostics"]["overloaded_rows_detected"] == 2
    assert first.market_actions == (("HIRE",),) * 10

    confirmed = observation(
        hands=((0, 0),) * 10,
        money=100_000,
        hour=1,
        hires_today=10,
    )
    second = controller.act(confirmed, daily_plan())
    assert second.diagnostics["hiring_diagnostics"]["target_workers"] == 17
    assert second.diagnostics["hiring_diagnostics"]["wanted_hires"] == 6
    assert second.market_actions == (("HIRE",),) * 6


def test_large_board_hiring_keeps_target_but_submits_only_affordable_prefix():
    result = hiring_plan(_large_row_items(15), money=3, max_orders=10)

    assert result.target_workers == 15
    assert result.wanted_hires == 14
    assert result.sequential_hire_costs[:3] == (1, 1, 2)
    assert result.affordable_hires == result.submittable_hires == 2
    assert result.orders == (("HIRE",), ("HIRE",))
    assert result.stop_reason is HireStopReason.CASH


def test_controller_submits_multiple_hires_in_one_market_batch():
    rows = [item("WATER", row, x=x) for row in range(10) for x in range(5)]
    forecast = work_plan(*rows)
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    result = controller.act(observation(money=1000), daily_plan())
    assert result.market_actions == (("HIRE",),) * 9
    assert result.diagnostics["hiring_diagnostics"]["target_workers"] == 10
    assert result.farmer_action != ("PASS",)
    assert result.hands_actions == ()


def test_existing_hand_can_work_during_hire_submission():
    rows = [item("WATER", row, x=x) for row in range(10) for x in range(5)]
    forecast = work_plan(*rows)
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    result = controller.act(
        observation(hands=((1, 0),), money=1000), daily_plan()
    )

    assert result.market_actions
    assert all(order == ("HIRE",) for order in result.market_actions)
    assert result.farmer_action != ("PASS",)
    assert len(result.hands_actions) == 1
    assert result.hands_actions[0] != ("PASS",)


def test_thirteen_row_hiring_and_assignment_leave_no_feasible_worker_idle():
    rows = _large_row_items(13)
    forecast = work_plan(*rows)
    controller = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: forecast
    )
    result = controller.act(
        observation(hands=((0, 0),) * 10, money=100_000),
        daily_plan(),
    )
    packed_rows = result.diagnostics["packed_rows_per_worker"]

    assert result.market_actions == (("HIRE",), ("HIRE",))
    assert result.diagnostics["hiring_diagnostics"]["target_workers"] == 13
    assert result.diagnostics["hiring_diagnostics"]["wanted_hires"] == 2
    assert result.diagnostics["large_route_assignment_mode"] is True
    assert result.diagnostics["primary_rows_assigned"] == 11
    assert result.diagnostics["overflow_rows_assigned"] == 2
    assert result.diagnostics["idle_workers_with_unassigned_feasible_rows"] == 0
    assert not result.diagnostics["idle_workers"]
    assert len(result.hands_actions) == 10
    assert len(packed_rows) == 11
    assert sum(len(route_rows) for route_rows in packed_rows.values()) == 13
    assert any(len(route_rows) > 1 for route_rows in packed_rows.values())


def test_three_useful_rows_fit_one_packed_worker():
    result = hiring_plan([item("WATER", 0), item("WATER", 1), item("WATER", 2)])
    assert result.target_workers == 1
    assert result.wanted_hires == result.submittable_hires == 0
    assert result.orders == ()


def test_existing_coverage_needs_no_gratuitous_hire():
    result = hiring_plan(
        [item("WATER", 0), item("WATER", 1)], hands=((0, 0),), money=0
    )
    assert result.wanted_hires == 0
    assert result.orders == ()


def test_fertilizer_only_work_does_not_drive_hiring_or_fertilizer_water():
    result = hiring_plan(
        [
            item("FERTILIZE", 0, source="fertilizer_policy"),
            item("WATER", 0, source="fertilizer_linked_productive"),
            item("FERTILIZE", 1, source="fertilizer_policy"),
        ],
    )
    assert result.wanted_hires == 0
    assert all(estimate.fertilizer_only for estimate in result.route_estimates)


def test_fertilizer_tail_does_not_inflate_prefix_but_gap_does():
    tail = hiring_plan([
        item("WATER", 0), item("WATER", 1),
        item("FERTILIZE", 2, source="fertilizer_policy"),
        item("WATER", 2, source="fertilizer_linked_productive"),
    ])
    assert tail.target_workers == 1
    gap = hiring_plan([
        item("WATER", 0),
        item("FERTILIZE", 1, source="fertilizer_policy"),
        item("WATER", 1, source="fertilizer_linked_productive"),
        item("WATER", 2),
    ])
    assert gap.target_workers == 1


def test_impossible_route_does_not_poison_later_useful_route():
    result = hiring_plan([
        item("FEED", 0, status=WorkStatus.BLOCKED),
        item("WATER", 1),
    ])
    assert result.target_workers == 1
    assert result.route_estimates[0].hire_driving is False
    assert result.route_estimates[1].hire_driving is True


def test_useful_overloaded_route_can_drive_hire():
    result = hiring_plan([item("WATER", 0)], hour=22)
    estimate = result.route_estimates[0]
    assert estimate.hire_driving is True
    assert estimate.route_overloaded is True
    assert estimate.useful_before_deadline is True
    assert result.wanted_hires == 0  # farmer already owns the only route


def test_too_late_uncovered_route_does_not_drive_hiring():
    result = hiring_plan([item("WATER", 0), item("WATER", 1)], hour=23)
    assert result.future_action_slots == 0
    assert result.route_estimates[1].useful_before_deadline is False
    assert result.wanted_hires == 0


def test_sequential_fibonacci_affordability_and_no_economic_gate():
    rows = [
        item("WATER", row, x=x)
        for row in range(4)
        for x in range(5)
    ] + [
        item("WATER", row, x=0)
        for row in range(4)
        for _ in range(2)
    ]
    result = hiring_plan(
        rows,
        money=2,
        hires_today=1,
    )
    # Three workers reach the best packed coverage, but cash buys only one.
    assert result.target_workers == 3
    assert result.wanted_hires == 2
    assert result.sequential_hire_costs == (1, 2)
    assert result.affordable_hires == result.submittable_hires == 1
    assert result.stop_reason is HireStopReason.CASH


def test_four_rows_three_workers_need_not_be_four_workers():
    rows = [
        item("WATER", row, x=x)
        for row in range(4)
        for x in range(5)
    ] + [
        item("WATER", row, x=0)
        for row in range(4)
        for _ in range(2)
    ]
    result = hiring_plan(rows)
    assert result.target_workers == 3
    assert result.wanted_hires == 2
    assert result.hire_reason == "additional_workers_reach_best_packed_coverage"


def test_hiring_and_execution_expose_identical_packed_segment_groups():
    items = [item("WATER", row) for row in range(4)]
    positions = {WorkerId(0): (0, 0), WorkerId(1): (0, 1)}
    forecast = work_plan(*items)
    candidates = generate_horizontal_route_candidates(forecast)
    execution = assign_horizontal_routes(candidates, positions, assignment_hour=0)
    hiring = hiring_plan(items, hands=((0, 1),), positions=positions)
    assert hiring.packed_segment_groups == tuple(
        tuple(segment.segment_id for segment in route.segments)
        for route in execution.routes
    )


def test_hiring_uses_shared_deadline_slots_and_exposes_segment_estimates():
    result = hiring_plan([item("WATER", row) for row in range(4)], hour=22)
    expected = remaining_day_action_slots(observation(hour=22), include_current_turn=False)
    assert result.future_action_slots == expected
    assert result.route_estimates
    estimate = result.route_estimates[0]
    assert estimate.estimated_arrival_turn >= 0
    assert estimate.estimated_completion_turn >= estimate.estimated_arrival_turn
    assert (
        estimate.expected_useful_interactions_completed_before_deadline
        + estimate.expected_useful_interactions_left_after_deadline
        >= 1
    )


def test_controller_observes_hires_before_finalizing_routes():
    forecast = work_plan(item("WATER", 0), item("WATER", 1), item("WATER", 2))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == ()
    assert first.diagnostics["routes_finalized"] is True
    assert len(controller.routes) == 1

    confirmed = observation(
        hands=((4, 4), (5, 4)), money=998, hires_today=2, hour=1
    )
    second = controller.act(confirmed, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert len(controller.routes) == 1
    assert second.diagnostics["worker_count_final"] == 3


def test_controller_bounds_failed_hire_and_finalizes_with_real_workers():
    forecast = work_plan(item("WATER", 0), item("WATER", 1))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == ()
    unchanged1 = observation(money=1000, hour=1)
    second = controller.act(unchanged1, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert len(controller.routes) == 1


def test_controller_never_hires_after_finalization():
    calls = 0

    def builder(obs, plan, **kwargs):
        nonlocal calls
        calls += 1
        return work_plan(item("WATER", 0)) if calls == 1 else work_plan(
            item("WATER", 0), item("WATER", 1)
        )

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == ()
    later = controller.act(observation(money=1000, hour=1), daily_plan())
    assert later.market_actions == ()


def test_procurement_is_observed_before_hiring_uses_remaining_cash():
    forecast = work_plan(
        item("FEED", 0, requirements=(SupplyRequirement("WHEAT", 1),)),
        item("WATER", 1),
    )
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=26), daily_plan())
    assert first.market_actions == (("BUY_PRODUCT", "WHEAT", 1),)
    assert first.diagnostics["routes_finalized"] is False

    observed = observation(money=1, hour=1, shed={"WHEAT": 1})
    second = controller.act(observed, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert second.diagnostics["hiring_diagnostics"]["cash_before_hiring"] == 1.0


def test_partial_hire_realization_recomputes_target_without_phantom_worker():
    forecast = work_plan(item("WATER", 0), item("WATER", 1), item("WATER", 2))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    assert len(controller.act(observation(money=1000), daily_plan()).market_actions) == 0
    partial = observation(hands=((4, 4),), money=999, hires_today=1, hour=1)
    second = controller.act(partial, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert len(controller.routes) == 1


def test_hire_submission_reconciles_before_final_route_ownership():
    forecast = work_plan(
        *(item("WATER", row, x=x) for row in range(10) for x in range(5))
    )
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    submitted = controller.act(observation(money=1000), daily_plan())
    assert submitted.market_actions == (("HIRE",),) * 9
    assert submitted.farmer_action != ("PASS",)
    assert submitted.hands_actions == ()

    confirmed = observation(
        hands=tuple((index % 10, index // 10) for index in range(9)),
        money=912,
        hires_today=9,
        hour=1,
    )
    result = controller.act(confirmed, daily_plan())
    assert result.market_actions == ()
    assert result.diagnostics["routes_finalized"] is True
    assert result.diagnostics["observed_hires"] == 9
    assert controller._pending_hires is None
    assert result.diagnostics["worker_count_final"] == 10
    assert len(result.hands_actions) == 9

    route_ids = [
        segment.segment_id
        for route in controller.routes
        for segment in route.segments
    ]
    assert len(route_ids) == len(set(route_ids))
    assert all(route_cursor_invariants_hold(route) for route in controller.routes)
    observed_workers = set(controller._worker_positions(confirmed))
    assert {route.owner for route in controller.routes} <= observed_workers


def test_failed_hires_keep_progress_and_finalize_only_observed_workers():
    forecast = work_plan(
        *(item("WATER", row, x=x) for row in range(10) for x in range(5))
    )
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)

    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions
    assert first.farmer_action != ("PASS",)
    first_submitted = len(first.market_actions)

    second = controller.act(observation(money=1000, hour=1), daily_plan())
    assert second.market_actions
    assert controller._hire_failures == first_submitted
    second_submitted = len(second.market_actions)

    final = controller.act(observation(money=1000, hour=2), daily_plan())
    assert final.diagnostics["routes_finalized"] is True
    assert final.diagnostics["failed_hires"] == first_submitted + second_submitted
    assert final.diagnostics["worker_count_final"] == 1
    assert {route.owner for route in controller.routes} <= {WorkerId(0)}


def test_confirmed_hire_uses_real_endpoint_and_packet3_supply_plan():
    forecast = work_plan(
        item("WATER", 0),
        item("FEED", 1, requirements=(SupplyRequirement("WHEAT", 1),)),
    )
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == (("BUY_PRODUCT", "WHEAT", 1),)
    purchased = observation(hour=1, money=975, shed={"WHEAT": 1})
    second = controller.act(purchased, daily_plan())
    assert second.diagnostics["routes_finalized"] is True
    route_supply = next(
        value for value in second.diagnostics["route_diagnostics"]
        if value["owner"] == "FARMER"
    )
    assert route_supply["supply_plan"]["demand"] == {"WHEAT": 1}
    assert route_supply["supply_plan"]["pickup_sequence"] == [
        {"item": "WHEAT", "quantity": 1}
    ]


# --- Packet 5B first-use ETA correction ---------------------------------------


def _fertilizer_before_water_row():
    """NE row 0 (farmer) plus a prospective-hire NW row 0.

    The new hand sweeps (0,4)->(0,0): three feasible fertilizer applications
    precede the routine WATER on (0,0), the first hire-driving item.
    """

    fertilizer = (SupplyRequirement("FERTILIZER", 1),)
    return [
        item("WATER", 0, x=5),  # NE route sorts first -> farmer-owned
        item("FERTILIZE", 0, x=1, source="fertilizer_policy", requirements=fertilizer),
        item("FERTILIZE", 0, x=2, source="fertilizer_policy", requirements=fertilizer),
        item("FERTILIZE", 0, x=3, source="fertilizer_policy", requirements=fertilizer),
        item("WATER", 0, x=0, source="optional_deferrable"),
    ]


def test_preceding_fertilizer_delays_first_use_past_deadline():
    result = hiring_plan(_fertilizer_before_water_row(), shed={"FERTILIZER": 10}, hour=14)
    estimate = result.route_estimates[1]
    assert estimate.route_index == 1
    assert estimate.first_use_work_id == "WATER:0:0"
    # The packed single-worker chain approaches the second segment from the
    # first segment's endpoint: movement 13, sweep to (0,0) 4,
    # one batched FERTILIZER pickup, three fertilizer interactions, one water.
    assert estimate.movement_turns == 13
    assert estimate.preceding_interaction_turns == 3
    assert estimate.pickup_turns == 1
    assert estimate.first_use_eta == 13 + 4 + 1 + 3 + 1
    assert estimate.future_action_slots == 0
    assert estimate.useful_before_deadline is False
    # The late route must not independently raise the coverage target.
    assert result.target_workers == 1
    assert result.wanted_hires == 0
    assert result.orders == ()


def test_preceding_fertilizer_still_useful_earlier_in_day():
    result = hiring_plan(_fertilizer_before_water_row(), shed={"FERTILIZER": 10}, hour=8)
    estimate = result.route_estimates[1]
    # The improved packing gives the NW segment to the spawned worker, whose
    # predicted position is already on its entry tile.
    assert estimate.first_use_eta == 1
    assert estimate.future_action_slots == 16
    assert estimate.useful_before_deadline is True
    assert result.target_workers == 2
    assert result.wanted_hires == result.submittable_hires == 1


def test_first_use_eta_unchanged_when_no_preceding_work():
    result = hiring_plan([item("WATER", 0, x=5), item("WATER", 0, x=0)], hour=8)
    estimate = result.route_estimates[1]
    assert estimate.preceding_interaction_turns == 0
    assert estimate.pickup_turns == 0
    # Identical to the pre-Packet-5B value: entry travel + sweep + interaction.
    assert estimate.first_use_eta == 1


def test_preceding_fertilizer_pickup_is_batched():
    result = hiring_plan(_fertilizer_before_water_row(), shed={"FERTILIZER": 10}, hour=14)
    estimate = result.route_estimates[1]
    # Three fertilizer applications require one batched PICKUP, not three.
    assert estimate.preceding_interaction_turns == 3
    assert estimate.pickup_turns == 1


def test_carried_fertilizer_removes_pickup_but_keeps_interactions():
    result = hiring_plan(
        _fertilizer_before_water_row(),
        hands=((4, 4),),
        inventories={WorkerId(1): {"FERTILIZER": 3}},
        positions={WorkerId(0): (0, 0), WorkerId(1): (4, 4)},
        hour=14,
    )
    estimate = result.route_estimates[1]
    assert estimate.preceding_interaction_turns == 0
    assert estimate.pickup_turns == 0


def test_supplies_needed_after_first_driving_do_not_inflate_eta():
    result = hiring_plan(
        [
            item("WATER", 0, x=5),  # NE route -> farmer
            item("WATER", 0, x=4, source="optional_deferrable"),  # entry tile, first driving
            item("FEED", 0, x=3, requirements=(SupplyRequirement("WHEAT", 1),)),
        ],
        shed={"WHEAT": 5},
        hour=8,
    )
    estimate = result.route_estimates[1]
    assert estimate.first_use_work_id == "FEED:0:3"
    assert estimate.preceding_interaction_turns == 0
    # The endpoint-aware packing gives this segment to the spawned worker;
    # FEED is consequently the first driving item on its west-to-east sweep.
    assert estimate.pickup_turns == 1
    assert estimate.first_use_eta == 21


# --- Packet 5B diagnostics cleanups -------------------------------------------


def test_small_board_packed_capacity_avoids_order_cap_for_unneeded_workers():
    result = hiring_plan(
        [item("WATER", r) for r in range(8)],
        money=12,
        max_orders=3,
    )
    assert result.target_workers == 2
    assert result.wanted_hires == 1
    assert result.affordable_hires == 1
    assert result.submittable_hires == 1
    assert result.stop_reason is HireStopReason.COVERED


def test_controller_populates_top_level_stop_reason_for_covered_no_hire():
    forecast = work_plan(item("WATER", 0))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    result = controller.act(observation(money=1000), daily_plan())
    assert result.market_actions == ()
    assert result.diagnostics["routes_finalized"] is True
    assert result.diagnostics["hire_stop_reason"] == "COVERED"
    assert result.diagnostics["hiring_diagnostics_status"] == "CURRENT"


def test_strip_hiring_public_exports_are_lazy():
    import executor_v0

    assert executor_v0.StripHiringPlan is StripHiringPlan
    assert executor_v0.plan_strip_hiring is plan_strip_hiring
    assert executor_v0.HireStopReason is HireStopReason
    assert executor_v0.RouteLaborEstimate is RouteLaborEstimate
