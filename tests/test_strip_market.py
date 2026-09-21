from __future__ import annotations

import copy

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorConfig, StripExecutorController
from executor_v0.strip_market import MarketBootstrapState, build_market_turn_plan
from executor_v0.strip_work import (
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    RowSummary,
    build_strip_work_plan,
    row_key_for_tile,
)
from replay_daily.constants import PRODUCTS


def daily_plan(
    *,
    sells: dict[str, dict[int, int]] | None = None,
    crop_targets: dict[str, int] | None = None,
    animal_targets: dict[str, int] | None = None,
) -> DailyPlan:
    quantities = {
        product: {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
        for product in PRODUCTS
    }
    for product, bins in (sells or {}).items():
        quantities[product].update(bins)
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
        | dict(crop_targets or {}),
        animal_targets={animal: 0 for animal in ("GOOSE", "COW", "SHEEP")}
        | dict(animal_targets or {}),
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


def observation(*, money=0, hour=0, step=None, day=1, shed=None, seeds=None, inventories=None, capacity=100, farmer=(0, 0), tiles=None):
    farm = {
        "money": money,
        "unlocked_quadrants": ["NW"],
        "farmer": [farmer[0], farmer[1]],
        "hands": [],
        "tiles": [[None] * 10 for _ in range(10)],
    }
    for (y, x), tile in (tiles or {}).items():
        farm["tiles"][y][x] = tile
    prices = {product: 25 for product in PRODUCTS}
    inventory = {product: 10000 for product in PRODUCTS}
    return {
        "day": day,
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


def plan_market(
    obs,
    daily,
    work_plan,
    *,
    state=None,
    capacity=100,
    max_orders=10,
    protected=None,
    aggressive=False,
):
    return build_market_turn_plan(
        obs,
        daily,
        work_plan,
        state or MarketBootstrapState(),
        shed_capacity=capacity,
        max_orders=max_orders,
        protected_reservations=protected,
        aggressive_sell_all=aggressive,
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


def test_escaped_sheep_reuses_empty_pasture_and_places_after_confirmed_pickup():
    pasture = {(4, 4): {"kind": "PASTURE"}}
    targeted = daily_plan(animal_targets={"SHEEP": 1})
    controller = StripExecutorController()

    initial = controller.act(
        observation(money=0, farmer=(4, 4), step=0, tiles=pasture), targeted
    )
    assert initial.market_actions == ()
    forecast = build_strip_work_plan(
        observation(money=0, farmer=(4, 4), step=0, tiles=pasture), targeted
    )
    assert not any(item.kind.startswith("BUILD") for item in forecast.items)

    affordable = controller.act(
        observation(money=1000, farmer=(4, 4), step=1, tiles=pasture), targeted
    )
    assert affordable.market_actions == (("BUY_ANIMAL", "SHEEP", 1),)

    observed_purchase = controller.act(
        observation(
            money=900,
            shed={"SHEEP": 1},
            farmer=(4, 4),
            step=2,
            tiles=pasture,
        ),
        targeted,
    )
    assert observed_purchase.farmer_action == ("PICKUP", "SHEEP", 1)
    assert observed_purchase.diagnostics["route_diagnostics"][0]["supply_plan"][
        "demand"
    ] == {"SHEEP": 1}

    placed = controller.act(
        observation(
            money=900,
            farmer=(4, 4),
            inventories=[{"SHEEP": 1}],
            step=3,
            tiles=pasture,
        ),
        targeted,
    )
    assert placed.farmer_action == ("PLACE", "SHEEP", 1)


def test_controller_refreshes_blocked_plant_after_observed_seed_purchase():
    targeted = daily_plan(crop_targets={"WHEAT": 1})
    controller = StripExecutorController()
    first = controller.act(observation(money=16, farmer=(4, 4)), targeted)
    assert first.market_actions == (("BUY_SEED", "WHEAT", 1),)
    assert first.farmer_action == ("PASS",)

    second = controller.act(
        observation(money=6, step=1, seeds={"WHEAT": 1}, farmer=(4, 4)), targeted
    )
    assert second.market_actions == ()
    assert second.farmer_action == ("PLANT", "WHEAT")

    planted = observation(money=6, step=2, seeds={"WHEAT": 0}, farmer=(4, 4))
    planted["farms"][0]["tiles"][4][4] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 1,
        "yield_units": 0,
        "watered_today": False,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    third = controller.act(
        planted,
        targeted,
    )
    assert third.farmer_action == ("WATER",)


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


def test_later_cash_retries_persistent_animal_deficit_after_finalization():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return work(WorkItem(id="BUY_ANIMAL:COW:1", kind="BUY_ANIMAL", animal="COW"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(money=0), daily_plan())
    assert first.market_actions == ()
    assert first.diagnostics["routes_finalized"] is True
    later = controller.act(observation(money=500, step=1), daily_plan())
    assert later.market_actions == (("BUY_ANIMAL", "COW", 1),)


def test_failed_animal_purchase_does_not_poison_the_next_day():
    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return work(WorkItem(id="BUY_ANIMAL:COW:1", kind="BUY_ANIMAL", animal="COW"))

    controller = StripExecutorController(work_builder=builder)
    first = controller.act(observation(day=1, money=0, step=0), daily_plan())
    assert first.market_actions == ()
    later = controller.act(observation(day=2, money=500, step=24), daily_plan())
    assert later.market_actions == (("BUY_ANIMAL", "COW", 1),)


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


def test_partial_wheat_realization_keeps_remaining_feed_shortage_visible():
    state = MarketBootstrapState()
    # A previous turn already observed one realized unit; the shed below holds it.
    state.buy_observed["BUY_PRODUCT:WHEAT"] = 1
    result = plan_market(
        observation(money=1000, shed={"WHEAT": 1}, step=1),
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 3, "inventory"),))),
        state=state,
    )
    # 3 demand - 1 observed shed = 2, not 1.
    assert result.orders == (("BUY_PRODUCT", "WHEAT", 2),)


def test_partial_seed_realization_keeps_remaining_plant_shortage_visible():
    state = MarketBootstrapState()
    state.buy_observed["BUY_SEED:WHEAT"] = 1
    result = plan_market(
        observation(money=1000, seeds={"WHEAT": 1}, step=1),
        daily_plan(),
        work(item(
            "PLANT",
            crop="WHEAT",
            tile=(0, 0),
            supplies=(SupplyRequirement("WHEAT", 3, "global_seed"),),
        )),
        state=state,
    )
    # 3 demand - 1 observed seed = 2, not 1.
    assert result.orders == (("BUY_SEED", "WHEAT", 2),)


def test_fully_realized_purchase_produces_no_further_buy():
    state = MarketBootstrapState()
    state.buy_observed["BUY_PRODUCT:WHEAT"] = 3
    result = plan_market(
        observation(money=1000, shed={"WHEAT": 3}, step=1),
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 3, "inventory"),))),
        state=state,
    )
    assert result.orders == ()

    seed_state = MarketBootstrapState()
    seed_state.buy_observed["BUY_SEED:WHEAT"] = 3
    seed_result = plan_market(
        observation(money=1000, seeds={"WHEAT": 3}, step=1),
        daily_plan(),
        work(item(
            "PLANT",
            crop="WHEAT",
            tile=(0, 0),
            supplies=(SupplyRequirement("WHEAT", 3, "global_seed"),),
        )),
        state=seed_state,
    )
    assert seed_result.orders == ()


def test_manager_bins_remain_authoritative_when_aggressive_selling_is_off():
    result = plan_market(
        observation(money=0, shed={"MILK": 5}),
        daily_plan(sells={"MILK": {0: 2}}),
        work(),
    )
    assert result.orders == (("SELL", "MILK", 2),)
    assert result.diagnostics["sell_mode"] == "manager_bins"


def test_aggressive_selling_ignores_manager_sell_quantity_and_uses_observed_shed():
    result = plan_market(
        observation(money=0, shed={"MILK": 5}),
        daily_plan(sells={"MILK": {0: 1}}),
        work(),
        aggressive=True,
    )
    assert result.orders == (("SELL", "MILK", 5),)
    assert result.diagnostics["sell_mode"] == "aggressive_sell_all"
    assert result.diagnostics["aggressive_sell_observed"] == {"MILK": 5}
    assert result.diagnostics["aggressive_sell_submitted_this_turn"] == {"MILK": 5}


def test_aggressive_sell_protects_observed_feed_purchase_until_pickup():
    feed = work(
        item(
            "FEED",
            tile=(4, 4),
            supplies=(SupplyRequirement("WHEAT", 1, "inventory"),),
        )
    )

    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return feed

    controller = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True),
        work_builder=builder,
    )

    purchase = controller.act(
        observation(money=100, shed={}, farmer=(4, 4), hour=0),
        daily_plan(),
    )
    assert purchase.market_actions == (("BUY_PRODUCT", "WHEAT", 1),)

    observed = controller.act(
        observation(money=75, shed={"WHEAT": 1}, farmer=(4, 4), hour=1),
        daily_plan(),
    )
    assert observed.market_actions == ()
    assert observed.farmer_action == ("PICKUP", "WHEAT", 1)

    picked_up = controller.act(
        observation(
            money=75,
            shed={},
            inventories=[{"WHEAT": 1}],
            farmer=(4, 4),
            hour=2,
        ),
        daily_plan(),
    )
    assert picked_up.market_actions == ()
    assert picked_up.farmer_action == ("FEED",)


def test_bootstrap_protects_only_excess_over_feed_demand():
    feed = work(
        item(
            "FEED",
            tile=(4, 4),
            supplies=(SupplyRequirement("WHEAT", 2, "inventory"),),
        )
    )

    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return feed

    result = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True),
        work_builder=builder,
    ).act(
        observation(money=0, shed={"WHEAT": 5}, farmer=(4, 4)),
        daily_plan(),
    )
    assert result.market_actions == (("SELL", "WHEAT", 3),)
    assert result.diagnostics["market_diagnostics"]["aggressive_sell_protected"] == {
        "WHEAT": 2
    }


def test_repeated_bootstrap_observations_do_not_grow_feed_reservation():
    feed = work(
        item(
            "FEED",
            tile=(4, 4),
            supplies=(SupplyRequirement("WHEAT", 2, "inventory"),),
        )
    )

    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        return feed

    controller = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True),
        work_builder=builder,
    )
    for hour in (0, 1):
        result = controller.act(
            observation(money=0, shed={"WHEAT": 5}, farmer=(4, 4), hour=hour),
            daily_plan(),
        )
        assert result.diagnostics["market_diagnostics"]["aggressive_sell_protected"] == {
            "WHEAT": 2
        }


def test_aggressive_mode_sells_unreserved_wheat_even_when_feed_demand_exists():
    result = plan_market(
        observation(money=0, shed={"WHEAT": 5}, inventories=[{"WHEAT": 2}]),
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 4, "inventory"),))),
        aggressive=True,
    )
    assert result.orders == (("SELL", "WHEAT", 5),)
    assert result.diagnostics["aggressive_sell_observed"] == {"WHEAT": 5}


def test_aggressive_mode_sells_excess_fertilizer_and_protects_reservation():
    fertilizer = plan_market(
        observation(money=0, shed={"FERTILIZER": 5}),
        daily_plan(),
        work(),
        protected={"FERTILIZER": 2},
        aggressive=True,
    )
    assert fertilizer.orders == (("SELL", "FERTILIZER", 3),)

    wheat = plan_market(
        observation(money=0, shed={"WHEAT": 5}),
        daily_plan(),
        work(item("FEED", supplies=(SupplyRequirement("WHEAT", 3, "inventory"),))),
        protected={"WHEAT": 3},
        aggressive=True,
    )
    assert wheat.orders == (("SELL", "WHEAT", 2),)


def test_aggressive_mode_sells_all_canonical_products():
    result = plan_market(
        observation(
            money=0,
            shed={"WHEAT": 5, "FERTILIZER": 3, "CARROT": 2, "MELON": 4, "MILK": 1},
        ),
        daily_plan(sells={"WHEAT": {0: 5}, "FERTILIZER": {0: 3}}),
        work(),
        aggressive=True,
    )
    assert result.orders == (
        ("SELL", "WHEAT", 5),
        ("SELL", "CARROT", 2),
        ("SELL", "MELON", 4),
        ("SELL", "MILK", 1),
        ("SELL", "FERTILIZER", 3),
    )


def test_aggressive_sell_proceeds_fund_buy_land_in_the_same_turn():
    result = plan_market(
        observation(money=0, shed={"MILK": 7}),
        daily_plan(),
        work(WorkItem(id="BUY_LAND:NE", kind="BUY_LAND", land="NE")),
        aggressive=True,
    )
    assert result.orders[0] == ("SELL", "MILK", 7)
    assert result.orders[1] == ("BUY_LAND",)
    assert result.diagnostics["money_after_simulated_orders"] >= 0


def test_aggressive_sales_and_purchases_respect_market_order_cap():
    result = plan_market(
        observation(money=5000, shed={"WHEAT": 1, "CARROT": 1, "MILK": 1}),
        daily_plan(),
        work(WorkItem(id="BUY_LAND:NE", kind="BUY_LAND", land="NE")),
        max_orders=2,
        aggressive=True,
    )
    assert len(result.orders) == 2
    assert result.orders == (("SELL", "WHEAT", 1), ("SELL", "CARROT", 1))


def test_multiple_partial_observations_follow_current_stock_exactly():
    forecast = work(item("FEED", supplies=(SupplyRequirement("WHEAT", 5, "inventory"),)))
    # current stock 1 -> 4 left, 3 -> 2 left, 5 -> 0 left, regardless of history
    for stock, expected in ((1, 4), (3, 2), (5, 0)):
        state = MarketBootstrapState()
        state.buy_observed["BUY_PRODUCT:WHEAT"] = stock
        result = plan_market(
            observation(money=1000, shed={"WHEAT": stock}, step=stock),
            daily_plan(),
            forecast,
            state=state,
        )
        if expected:
            assert result.orders == (("BUY_PRODUCT", "WHEAT", expected),)
        else:
            assert result.orders == ()


def test_animal_demand_is_not_reduced_by_historical_observation():
    # Packet 1 still reports a concrete BUY_ANIMAL deficit, but the shed does not
    # yet hold the animal: the planner must still buy it even if history says so.
    state = MarketBootstrapState()
    state.buy_observed["BUY_ANIMAL:COW"] = 1
    still_needed = plan_market(
        observation(money=1000, shed={}, step=1),
        daily_plan(),
        work(item("BUY_ANIMAL", animal="COW")),
        state=state,
    )
    assert still_needed.orders == (("BUY_ANIMAL", "COW", 1),)

    # Once the animal is observed in the shed, rebuilt Packet 1 work drops the
    # BUY_ANIMAL item, so no order is emitted (removed exactly once).
    observed = plan_market(
        observation(money=1000, shed={"COW": 1}, step=1),
        daily_plan(),
        work(),
        state=state,
    )
    assert observed.orders == ()
