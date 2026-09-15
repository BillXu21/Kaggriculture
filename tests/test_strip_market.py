from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorController
from executor_v0.strip_market import MarketBootstrapState, build_market_turn_plan
from executor_v0.strip_work import (
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    RowSummary,
    row_key_for_tile,
)
from replay_daily.constants import PRODUCTS


def daily_plan(*, sells: dict[str, dict[int, int]] | None = None) -> DailyPlan:
    quantities = {
        product: {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
        for product in PRODUCTS
    }
    for product, bins in (sells or {}).items():
        quantities[product].update(bins)
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        animal_targets={animal: 0 for animal in ("GOOSE", "COW", "SHEEP")},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        care_by_animal={animal: 0 for animal in ("GOOSE", "COW", "SHEEP")},
        sell_quantities=quantities,
    )


def work(*items: WorkItem) -> StripWorkPlan:
    by_row = {}
    for work_item in items:
        if work_item.tile is not None:
            by_row.setdefault(row_key_for_tile(work_item.tile), []).append(work_item)
    rows = tuple(
        RowSummary(
            key,
            5,
            sum(item.interaction_turns for item in row_items if item.ready),
            sum(item.interaction_turns for item in row_items if not item.ready),
            sum(item.interaction_turns for item in row_items),
        )
        for key, row_items in sorted(by_row.items())
    )
    return StripWorkPlan(
        items=items,
        chains=(),
        row_summaries=rows,
        supply=SupplySnapshot(),
        diagnostics=WorkDiagnostics(),
        acting_seat=0,
    )


def item(kind: str, *, crop=None, animal=None, product=None, quantity=1, supplies=(), tile=None):
    return WorkItem(
        id=f"{kind}:{crop or animal or product or quantity}",
        kind=kind,
        crop=crop,
        animal=animal,
        product=product,
        quantity=quantity,
        tile=tile,
        required_supplies=tuple(supplies),
    )


def observation(*, money=0, hour=0, step=None, shed=None, seeds=None, inventories=None, capacity=100, farmer=(0, 0)):
    farm = {
        "money": money,
        "unlocked_quadrants": ["NW"],
        "farmer": [farmer[0], farmer[1]],
        "hands": [],
        "tiles": [[None] * 10 for _ in range(10)],
    }
    prices = {product: 25 for product in PRODUCTS}
    inventory = {product: 10000 for product in PRODUCTS}
    return {
        "day": 1,
        "hour": hour,
        "step": hour if step is None else step,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": shed or {},
            "seeds": seeds or {crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
            "inventories": inventories or [{}],
        },
        "market": {"inventory": inventory, "prices": prices},
        "configuration": {"shedCapacity": capacity},
    }


def plan_market(obs, daily, work_plan, *, state=None, capacity=100, max_orders=10, protected=None):
    return build_market_turn_plan(
        obs,
        daily,
        work_plan,
        state or MarketBootstrapState(),
        shed_capacity=capacity,
        max_orders=max_orders,
        protected_reservations=protected,
    )


def test_shared_cash_does_not_reuse_original_bank_for_later_order():
    obs = observation(money=100)
    forecast = work(
        item("FEED", supplies=(SupplyRequirement("WHEAT", 4, "inventory"),)),
        item("PLANT", crop="WHEAT", tile=(0, 0), supplies=(SupplyRequirement("WHEAT", 3, "global_seed"),)),
    )
    result = plan_market(obs, daily_plan(), forecast)
    assert result.orders == (("BUY_PRODUCT", "WHEAT", 3), ("BUY_SEED", "WHEAT", 2))
    assert result.diagnostics["buy_demand"]["BUY_SEED:WHEAT"] == 3
    assert result.diagnostics["market_blocked"]["BUY_SEED:WHEAT"]["block_reason"] == "CASH"


def test_sell_proceeds_and_freed_capacity_fund_later_buy():
    obs = observation(money=0, shed={"WHEAT": 1})
    forecast = work(item("FEED", supplies=(SupplyRequirement("WHEAT", 2, "inventory"),)))
    result = plan_market(
        obs,
        daily_plan(sells={"WHEAT": {0: 1}}),
        forecast,
        capacity=1,
    )
    assert result.orders == (("SELL", "WHEAT", 1), ("BUY_PRODUCT", "WHEAT", 1))
    assert result.diagnostics["money_after_simulated_orders"] == 0


def test_multiple_buys_share_shed_capacity():
    obs = observation(money=1000, shed={"CARROT": 9})
    forecast = work(
        item("FEED", supplies=(SupplyRequirement("WHEAT", 2, "inventory"),)),
        item("BUY_ANIMAL", animal="COW"),
    )
    result = plan_market(obs, daily_plan(), forecast, capacity=10)
    assert result.orders == (("BUY_PRODUCT", "WHEAT", 1),)
    assert result.diagnostics["market_blocked"]["BUY_ANIMAL:COW"]["block_reason"] == "SHED_CAPACITY"


def test_feed_seed_and_animal_demands_are_exact_and_fertilizer_is_never_bought():
    obs = observation(money=5000, shed={"WHEAT": 2}, seeds={"WHEAT": 1})
    forecast = work(
        item("FEED", supplies=(SupplyRequirement("WHEAT", 5, "inventory"),)),
        item("PLANT", crop="WHEAT", tile=(0, 0), supplies=(SupplyRequirement("WHEAT", 3, "global_seed"),)),
        item("BUY_ANIMAL", animal="COW", quantity=2),
        item("FERTILIZE", product="FERTILIZER", supplies=(SupplyRequirement("FERTILIZER", 9, "inventory"),)),
    )
    result = plan_market(obs, daily_plan(), forecast)
    assert ("BUY_PRODUCT", "WHEAT", 3) in result.orders
    assert ("BUY_SEED", "WHEAT", 2) in result.orders
    assert ("BUY_ANIMAL", "COW", 2) in result.orders
    assert all(order[1] != "FERTILIZER" for order in result.orders if order[0] == "BUY_PRODUCT")


def test_buy_product_does_not_impose_a_market_inventory_floor():
    obs = observation(money=1000, shed={})
    obs["market"]["inventory"]["WHEAT"] = 0
    result = plan_market(
        obs,
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 1, "inventory"),))),
    )
    assert result.orders == (("BUY_PRODUCT", "WHEAT", 1),)


def test_missing_product_price_is_visible_and_not_guessed():
    obs = observation(money=100)
    del obs["market"]["prices"]["WHEAT"]
    result = plan_market(
        obs,
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 1, "inventory"),))),
    )
    assert result.orders == ()
    assert result.diagnostics["market_blocked"]["BUY_PRODUCT:WHEAT"]["block_reason"] == "MISSING_PRICE"


def test_tileless_unresolved_plant_does_not_create_speculative_seed_demand():
    result = plan_market(
        observation(money=1000),
        daily_plan(),
        work(item(
            "PLANT",
            crop="WHEAT",
            supplies=(SupplyRequirement("WHEAT", 5, "global_seed"),),
        )),
    )
    assert result.orders == ()


def test_sell_bins_are_independent_and_future_bins_are_not_early():
    daily = daily_plan(sells={"WHEAT": {0: 1, 4: 2, 8: 3}})
    for hour, expected in ((3, 1), (4, 2), (7, 2), (8, 3)):
        result = plan_market(observation(money=100, hour=hour, shed={"WHEAT": 10}), daily, work())
        assert result.orders == (("SELL", "WHEAT", expected),)


def test_carried_products_are_not_sellable_and_order_cap_preserves_prefix():
    daily = daily_plan(sells={product: {0: 1} for product in PRODUCTS})
    result = plan_market(
        observation(money=1000, shed={product: 1 for product in PRODUCTS}),
        daily,
        work(),
        max_orders=2,
    )
    assert result.orders == (("SELL", "WHEAT", 1), ("SELL", "CARROT", 1))

    carried = plan_market(
        observation(money=1000, inventories=[{"CARROT": 10}]),
        daily_plan(sells={"CARROT": {0: 10}}),
        work(),
    )
    assert carried.orders == ()
    assert carried.diagnostics["market_blocked"]["SELL:0:CARROT"]["block_reason"] == "NOT_IN_SHED"


def test_controller_keeps_workers_passed_until_animal_purchase_is_observed():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        if int((obs.get("private") or {}).get("shed", {}).get("COW", 0)):
            return work()
        return work(item("BUY_ANIMAL", animal="COW"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=500), daily_plan())
    assert first.market_actions == (("BUY_ANIMAL", "COW", 1),)
    assert first.farmer_action == ("PASS",)
    assert controller.routes == ()

    after = observation(money=100, step=1, shed={"COW": 1})
    second = controller.act(after, daily_plan())
    assert second.market_actions == ()
    assert second.diagnostics["routes_finalized"] is True
    assert second.diagnostics["market_diagnostics"]["buy_observed"]["BUY_ANIMAL:COW"] == 1


def test_controller_bounds_no_progress_market_retries_before_finalization():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return work(WorkItem(id="BUY_LAND:NE", kind="BUY_LAND", land="NE"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=1000, step=0), daily_plan())
    second = controller.act(observation(money=1000, step=1), daily_plan())
    third = controller.act(observation(money=1000, step=2), daily_plan())
    assert first.market_actions == second.market_actions == (("BUY_LAND",),)
    assert third.market_actions == ()
    assert third.diagnostics["routes_finalized"] is True
    assert "BUY_LAND:NE" in third.diagnostics["market_diagnostics"]["market_no_progress_failures"]


def test_later_sell_cash_does_not_reopen_procurement_after_finalization():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return work(WorkItem(id="BUY_ANIMAL:COW:1", kind="BUY_ANIMAL", animal="COW"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=0), daily_plan())
    assert first.market_actions == ()
    assert first.diagnostics["routes_finalized"] is True
    later = controller.act(observation(money=500, step=1), daily_plan())
    assert later.market_actions == ()


def test_land_unlock_is_an_observation_barrier_before_ne_route_generation():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        if "NE" in obs["farms"][0]["unlocked_quadrants"]:
            return work(item("WATER", tile=(0, 5)))
        return work(WorkItem(id="BUY_LAND:NE", kind="BUY_LAND", land="NE"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=1000), daily_plan())
    assert first.market_actions == (("BUY_LAND",),)
    assert controller.routes == ()

    unlocked = observation(money=0, step=1, farmer=(0, 5))
    unlocked["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    second = controller.act(unlocked, daily_plan())
    assert second.diagnostics["routes_finalized"] is True
    assert controller.routes
    assert controller.routes[0].owned_tiles == ((0, 5), (0, 6), (0, 7), (0, 8), (0, 9))


def test_purchased_feed_is_observed_then_reserved_for_packet3_pickup():
    def builder(obs, plan, **kwargs):
        del plan, kwargs
        return work(item(
            "FEED",
            supplies=(SupplyRequirement("WHEAT", 1, "inventory"),),
            quantity=1,
            tile=(0, 0),
        ))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=100), daily_plan())
    assert first.market_actions == (("BUY_PRODUCT", "WHEAT", 1),)
    assert first.farmer_action == ("PASS",)

    after = observation(money=75, step=1, shed={"WHEAT": 1}, farmer=(4, 4))
    second = controller.act(after, daily_plan())
    assert second.market_actions == ()
    assert second.farmer_action == ("PICKUP", "WHEAT", 1)


def test_confirmed_packet3_pickup_releases_sell_protection_before_next_bin():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return work(item(
            "FEED",
            supplies=(SupplyRequirement("WHEAT", 1, "inventory"),),
            tile=(0, 0),
        ))

    controller = StripExecutorController(work_builder=builder)
    fixed = daily_plan(sells={"WHEAT": {4: 1}})
    first = controller.act(
        observation(money=100, shed={"WHEAT": 2}, farmer=(4, 4)), fixed
    )
    assert first.market_actions == ()
    assert first.farmer_action == ("PICKUP", "WHEAT", 1)

    second = controller.act(
        observation(
            money=100,
            hour=4,
            step=4,
            shed={"WHEAT": 1},
            inventories=[{"WHEAT": 1}],
            farmer=(4, 4),
        ),
        fixed,
    )
    assert second.market_actions == (("SELL", "WHEAT", 1),)


def test_finalized_sell_protects_packet3_reservation_and_ignores_carried_stock():
    result = plan_market(
        observation(
            money=100,
            shed={"WHEAT": 6},
            inventories=[{"CARROT": 10}],
        ),
        daily_plan(sells={"WHEAT": {0: 6}, "CARROT": {0: 10}}),
        work(),
        protected={"WHEAT": 4},
    )
    # Supply protection is passed by the finalized controller; four units are
    # reserved, so only two of the six shed units are legal to sell.
    assert result.orders == (("SELL", "WHEAT", 2),)
    assert all(order[1] != "CARROT" for order in result.orders)
