"""Focused Packet 5 coverage-driven strip hiring tests."""

from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_hiring import plan_strip_hiring
from executor_v0.strip_routes import WorkerId, generate_horizontal_route_candidates
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
) -> WorkItem:
    tile = (row, 0)
    return WorkItem(
        id=f"{kind}:{row}",
        kind=kind,
        status=status,
        tile=tile,
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


def hiring_plan(items, *, hands=(), money=1000, hour=0, hires_today=0):
    forecast = work_plan(*items)
    candidates = generate_horizontal_route_candidates(forecast)
    positions = {WorkerId(0): (0, 0)}
    positions.update({WorkerId(index): (0, 0) for index in range(1, len(hands) + 1)})
    return plan_strip_hiring(
        observation(
            hands=hands,
            money=money,
            hour=hour,
            hires_today=hires_today,
        ),
        forecast,
        candidates,
        positions,
        {worker: {} for worker in positions},
        max_orders=10,
    )


def test_three_useful_rows_hire_to_fixed_prefix():
    result = hiring_plan([item("WATER", 0), item("WATER", 1), item("WATER", 2)])
    assert result.target_workers == 3
    assert result.wanted_hires == result.submittable_hires == 2
    assert result.orders == (("HIRE",), ("HIRE",))


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
    assert tail.target_workers == 2
    gap = hiring_plan([
        item("WATER", 0),
        item("FERTILIZE", 1, source="fertilizer_policy"),
        item("WATER", 1, source="fertilizer_linked_productive"),
        item("WATER", 2),
    ])
    assert gap.target_workers == 3


def test_impossible_route_does_not_poison_later_useful_route():
    result = hiring_plan([
        item("FEED", 0, status=WorkStatus.BLOCKED),
        item("WATER", 1),
    ])
    assert result.target_workers == 2
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
    result = hiring_plan(
        [item("WATER", 0), item("WATER", 1), item("WATER", 2), item("WATER", 3)],
        money=3,
        hires_today=1,
    )
    assert result.sequential_hire_costs == (1, 2, 3)
    assert result.affordable_hires == result.submittable_hires == 2
    assert result.stop_reason.value == "CASH"


def test_controller_observes_hires_before_finalizing_routes():
    forecast = work_plan(item("WATER", 0), item("WATER", 1), item("WATER", 2))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == (("HIRE",), ("HIRE",))
    assert controller.routes == ()

    confirmed = observation(
        hands=((4, 4), (5, 4)), money=998, hires_today=2, hour=1
    )
    second = controller.act(confirmed, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert len(controller.routes) == 3
    assert second.diagnostics["worker_count_final"] == 3


def test_controller_bounds_failed_hire_and_finalizes_with_real_workers():
    forecast = work_plan(item("WATER", 0), item("WATER", 1))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == (("HIRE",),)
    unchanged1 = observation(money=1000, hour=1)
    second = controller.act(unchanged1, daily_plan())
    assert second.market_actions == (("HIRE",),)
    unchanged2 = observation(money=1000, hour=2)
    third = controller.act(unchanged2, daily_plan())
    assert third.market_actions == ()
    assert third.diagnostics["routes_finalized"] is True
    assert third.diagnostics["hire_stop_reason"] == "FAILED"
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
    assert second.market_actions == (("HIRE",),)
    assert second.diagnostics["hiring_diagnostics"]["cash_before_hiring"] == 1.0


def test_partial_hire_realization_recomputes_target_without_phantom_worker():
    forecast = work_plan(item("WATER", 0), item("WATER", 1), item("WATER", 2))
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    assert len(controller.act(observation(money=1000), daily_plan()).market_actions) == 2
    partial = observation(hands=((4, 4),), money=999, hires_today=1, hour=1)
    second = controller.act(partial, daily_plan())
    assert second.market_actions == (("HIRE",),)
    assert second.diagnostics["hiring_diagnostics"]["current_workers"] == 2
    assert controller.routes == ()


def test_confirmed_hire_uses_real_endpoint_and_packet3_supply_plan():
    forecast = work_plan(
        item("WATER", 0),
        item("FEED", 1, requirements=(SupplyRequirement("WHEAT", 1),)),
    )
    controller = StripExecutorController(work_builder=lambda obs, plan, **kwargs: forecast)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == (("BUY_PRODUCT", "WHEAT", 1),)
    purchased = observation(hour=1, money=975, shed={"WHEAT": 1})
    assert controller.act(purchased, daily_plan()).market_actions == (("HIRE",),)
    confirmed = observation(
        hands=((9, 0),), money=974, hires_today=1, hour=2, shed={"WHEAT": 1}
    )
    second = controller.act(confirmed, daily_plan())
    assert second.diagnostics["routes_finalized"] is True
    hand_route = next(route for route in controller.routes if route.owner == WorkerId(1))
    assert hand_route.entry_tile == (1, 4)
    hand_supply = next(
        value for value in second.diagnostics["route_diagnostics"]
        if value["owner"] == "HAND:0"
    )
    assert hand_supply["supply_plan"]["demand"] == {"WHEAT": 1}
    assert hand_supply["supply_plan"]["pickup_sequence"] == [
        {"item": "WHEAT", "quantity": 1}
    ]
