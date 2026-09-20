"""Focused Packet 5 coverage-driven strip hiring tests."""

from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_hiring import (
    HireStopReason,
    RouteLaborEstimate,
    StripHiringPlan,
    plan_strip_hiring,
)
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
    x: int = 0,
) -> WorkItem:
    tile = (row, x)
    return WorkItem(
        id=f"{kind}:{row}:{x}",
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
    assert result.target_workers == 3
    assert result.wanted_hires == 2
    assert result.sequential_hire_costs == (1, 2)
    assert result.affordable_hires == result.submittable_hires == 1
    assert result.stop_reason.value == "CASH"


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
    assert result.hire_reason == "extra_worker_materially_completes_packed_work"


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
    assert estimate.first_use_eta == 13
    assert estimate.future_action_slots == 15
    assert estimate.useful_before_deadline is True
    assert result.target_workers == 2
    assert result.wanted_hires == result.submittable_hires == 1


def test_first_use_eta_unchanged_when_no_preceding_work():
    result = hiring_plan([item("WATER", 0, x=5), item("WATER", 0, x=0)], hour=8)
    estimate = result.route_estimates[1]
    assert estimate.preceding_interaction_turns == 0
    assert estimate.pickup_turns == 0
    # Identical to the pre-Packet-5B value: entry travel + sweep + interaction.
    assert estimate.first_use_eta == 4 + 4 + 0 + 0 + 1


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
    assert estimate.preceding_interaction_turns == 3
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
    assert estimate.first_use_work_id == "WATER:0:4"
    assert estimate.preceding_interaction_turns == 0
    # WHEAT is only needed by the later FEED and must not add a pickup turn.
    assert estimate.pickup_turns == 0
    assert estimate.first_use_eta == 4 + 0 + 0 + 0 + 1


# --- Packet 5B diagnostics cleanups -------------------------------------------


def test_packed_capacity_avoids_order_cap_for_unneeded_workers():
    result = hiring_plan(
        [item("WATER", r) for r in range(10)],
        money=12,
        max_orders=3,
    )
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
