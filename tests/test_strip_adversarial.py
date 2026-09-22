"""Packet I adversarial invariant stress for the Stage 2.5 strip executor.

These tests are deliberately narrow: they assert *mechanical* invariants
(determinism, route ownership, supply accounting, action legality, forbidden
auto-sale) over small synthetic states and a bounded deterministic fuzz loop.

They do not attempt strategic optimization, do not add economic safeguards,
and do not require the native engine.  A strategically terrible plan is
allowed to fail economically; only illegal or inconsistent mechanics are
treated as defects.
"""

from __future__ import annotations

import copy
import json
from random import Random

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import (
    StripExecutorConfig,
    StripExecutorController,
)
from executor_v0.strip_work import (
    RowKey,
    RowSummary,
    StripWorkPlan,
    SupplyRequirement,
    SupplySnapshot,
    WorkDiagnostics,
    WorkItem,
    WorkStatus,
    build_strip_work_plan,
    row_key_for_tile,
)
from fast_env.api import UNIT_IDS
from replay_daily.constants import ANIMALS, CROPS, PRODUCTS


ANCHORS = (0, 4, 8, 12, 16, 20)
CROP_NAMES = frozenset(CROPS)
ANIMAL_NAMES = frozenset(ANIMALS)
PRODUCT_NAMES = frozenset(PRODUCTS)

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
UNIT_KINDS = frozenset(UNIT_IDS)
INTERACTION_KINDS = UNIT_KINDS - MOVES - {"PASS"}
MARKET_OPS = {"HIRE", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL", "BUY_LAND"}
SINGLE_TOKEN_MARKET_OPS = {"HIRE", "BUY_LAND"}
AGGRESSIVE_ALLOWED = {
    "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
}
AGGRESSIVE_FORBIDDEN = {"WHEAT", "FERTILIZER"}

_DELTAS = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}


# --------------------------------------------------------------------- helpers
def daily_plan(*, sells=None, crop_targets=None, land_count=1) -> DailyPlan:
    quantities = {product: {anchor: 0 for anchor in ANCHORS} for product in PRODUCT_NAMES}
    for product, bins in (sells or {}).items():
        quantities[str(product)].update({int(a): int(q) for a, q in bins.items()})
    return DailyPlan.create(
        crop_targets={crop: 0 for crop in CROP_NAMES} | dict(crop_targets or {}),
        animal_targets={animal: 0 for animal in ANIMAL_NAMES},
        land_count=land_count,
        fertilizer_by_crop={crop: 0 for crop in CROP_NAMES},
        care_by_animal={animal: 0 for animal in ANIMAL_NAMES},
        sell_quantities=quantities,
    )


def market(prices=None, inventory=None) -> dict:
    return {
        "prices": {product: 25 for product in PRODUCT_NAMES} | dict(prices or {}),
        "inventory": {product: 10000 for product in PRODUCT_NAMES}
        | dict(inventory or {}),
    }


def observation(
    positions,
    *,
    hour=0,
    day=3,
    tiles=None,
    shed=None,
    inventories=None,
    seeds=None,
    money=0.0,
    unlocked=("NW",),
    market_state=None,
    shed_capacity=100,
) -> dict:
    """Build one synthetic observation from board ``(y, x)`` worker positions."""

    board = tiles if tiles is not None else [[None] * 10 for _ in range(10)]

    def to_farm(position):
        return [int(position[1]), int(position[0])]

    farm = {
        "farmer": to_farm(positions[0]),
        "hands": [to_farm(position) for position in positions[1:]],
        "hires_today": len(positions) - 1,
        "money": float(money),
        "tiles": board,
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": int(day),
        "hour": int(hour),
        "step": int(day) * 24 + int(hour),
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "shed": dict(shed or {}),
            "seeds": dict(seeds or {}),
            "inventories": [
                dict(inventory)
                for inventory in (
                    inventories or [{} for _ in positions]
                )
            ],
        },
        "configuration": {"shedCapacity": int(shed_capacity)},
        "market": market() if market_state is None else market_state,
    }


def work_item(
    kind,
    tile,
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


def fake_plan(items) -> StripWorkPlan:
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
        items=tuple(items),
        chains=(),
        row_summaries=tuple(rows),
        supply=SupplySnapshot(),
        diagnostics=WorkDiagnostics(),
        acting_seat=0,
    )


def advance(position, action):
    delta = _DELTAS.get(action[0]) if action else None
    if delta is None:
        return position
    return (position[0] + delta[0], position[1] + delta[1])


def simulate(
    controller, daily, *, positions, hours, start_hour=0, on_turn=None, **obs_kwargs
):
    """Drive a controller for fixed turns, applying only confirmed movement."""

    moving = [tuple(position) for position in positions]
    for hour in range(start_hour, start_hour + hours):
        current = observation(moving, hour=hour, **obs_kwargs)
        result = controller.act(current, daily)
        if on_turn is not None:
            on_turn(hour, result, tuple(moving))
        actions = (result.farmer_action, *result.hands_actions)
        for index, action in enumerate(actions):
            if index < len(moving):
                moving[index] = advance(moving[index], action)
    return moving


def check_actions_are_legal(result, worker_count, context):
    """Assert one controller result only contains structurally legal actions."""

    actions = (result.farmer_action, *result.hands_actions)
    assert len(actions) == worker_count, (context, actions)

    def fail(message):
        raise AssertionError(f"{context}: {message}")

    for worker, action in enumerate(actions):
        action = tuple(action)
        if not action:
            fail(f"worker {worker} emitted an empty action")
        head = action[0]
        if head in UNIT_KINDS:
            if head == "PLANT":
                if len(action) != 2 or action[1] not in CROP_NAMES:
                    fail(f"worker {worker} emitted illegal PLANT {action!r}")
            elif head == "PLACE":
                if (
                    len(action) != 3
                    or action[1] not in ANIMAL_NAMES
                    or not isinstance(action[2], int)
                    or action[2] <= 0
                ):
                    fail(f"worker {worker} emitted illegal PLACE {action!r}")
            elif head == "PICKUP":
                if (
                    len(action) != 3
                    or action[1] not in PRODUCT_NAMES
                    or not isinstance(action[2], int)
                    or action[2] <= 0
                ):
                    fail(f"worker {worker} emitted illegal PICKUP {action!r}")
            elif len(action) != 1:
                fail(f"worker {worker} emitted illegal unit action {action!r}")
        elif head in MARKET_OPS:
            continue  # market actions are validated separately
        else:
            fail(f"worker {worker} emitted unknown action {action!r}")

    for order in result.market_actions:
        order = tuple(order)
        if not order or order[0] not in MARKET_OPS:
            fail(f"illegal market order {order!r}")
        if order[0] in SINGLE_TOKEN_MARKET_OPS:
            if len(order) != 1:
                fail(f"illegal single-token market order {order!r}")
            continue
        if len(order) != 3 or order[1] not in PRODUCT_NAMES:
            fail(f"illegal market order {order!r}")
        if not isinstance(order[2], int) or order[2] <= 0:
            fail(f"illegal market quantity {order!r}")


def _phases(controller):
    return {
        route.owner.index: (route.route_id, route.owned_tiles, route.traversal)
        for route in controller.routes
    }


def _route_for_worker(controller, worker_index):
    for route in controller.routes:
        if route.owner.index == worker_index:
            return route
    return None


# ------------------------------------------------------------------ invariants
def test_controller_is_deterministic_across_identical_runs():
    items = (
        work_item("WATER", (0, 0)),
        work_item("WATER", (0, 2)),
        work_item(
            "FEED",
            (1, 0),
            required_supplies=(SupplyRequirement("WHEAT", 2),),
        ),
        work_item("WATER", (1, 3)),
    )
    builder = lambda obs, plan, **kwargs: fake_plan(items)  # noqa: E731
    positions = [(0, 0), (1, 0)]
    daily = daily_plan(sells={"CARROT": {0: 3}})

    def run():
        controller = StripExecutorController(work_builder=builder)
        results = []
        simulate(
            controller,
            daily,
            positions=positions,
            hours=8,
            on_turn=lambda hour, result, _positions: results.append(
                (result.action_dict(), json.dumps(result.diagnostics, sort_keys=True))
            ),
            shed={"WHEAT": 2, "CARROT": 4},
            inventories=[{}, {}],
            money=0.0,
        )
        return results

    first = run()
    second = run()
    assert first == second


def test_frozen_routes_never_change_owner_or_geometry():
    items = tuple(
        work_item("WATER", tile) for tile in ((0, 0), (0, 3), (1, 1), (2, 4))
    )
    controller = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan(items)
    )
    frozen: dict[int, tuple] = {}

    def on_turn(_hour, result, _positions):
        if not result.diagnostics["routes_finalized"]:
            return
        current = _phases(controller)
        if not frozen:
            frozen.update(current)
        assert current == frozen, "frozen route geometry changed mid-day"
        tiles = [tile for _, owned, _ in current.values() for tile in owned]
        assert len(tiles) == len(set(tiles)), "two routes own the same tile"

    simulate(
        controller,
        daily_plan(),
        positions=[(0, 0), (1, 0), (2, 0), (3, 0)],
        hours=10,
        on_turn=on_turn,
        money=0.0,
    )
    assert frozen, "routes were never finalized"


def test_workers_never_act_outside_their_owned_route():
    items = (
        work_item("WATER", (0, 0)),
        work_item("WATER", (0, 1)),
        work_item("WATER", (1, 0)),
    )
    controller = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan(items)
    )
    violations = []

    def on_turn(_hour, result, positions):
        if not result.diagnostics["routes_finalized"]:
            return
        actions = (result.farmer_action, *result.hands_actions)
        for worker, action in enumerate(actions):
            if action and action[0] in INTERACTION_KINDS:
                route = _route_for_worker(controller, worker)
                if route is None or positions[worker] not in route.owned_tiles:
                    violations.append((worker, action, positions[worker]))

    simulate(
        controller,
        daily_plan(),
        positions=[(0, 0), (1, 0)],
        hours=8,
        on_turn=on_turn,
        money=0.0,
    )
    assert violations == [], violations


def test_deadline_feasible_rows_stay_assigned_to_packed_worker():
    rows = (
        work_item("WATER", (0, 0)),
        work_item("WATER", (1, 0)),
        work_item("WATER", (2, 0)),
    )
    understaffed = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan(rows)
    )
    observed = []

    def capture(_hour, result, _positions):
        if result.diagnostics["routes_finalized"]:
            observed.append(tuple(result.diagnostics["unassigned_active_routes"]))

    simulate(
        understaffed,
        daily_plan(),
        positions=[(0, 0)],
        hours=4,
        on_turn=capture,
        money=0.0,
    )
    # The deadline-aware multi-row route can finish all three one-action rows
    # before day end, so none is discarded merely because only one worker is
    # present.
    assert observed and all(not value for value in observed)
    assert len(set(observed)) == 1, "unassigned routes changed mid-day"

    overstaffed = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan((rows[0],))
    )

    def check_idle(_hour, result, _positions):
        if not result.diagnostics["routes_finalized"]:
            return
        assert sorted(result.diagnostics["idle_workers"]) == ["HAND:0", "HAND:1"]
        assert result.hands_actions == (("PASS",), ("PASS",))

    simulate(
        overstaffed,
        daily_plan(),
        positions=[(0, 0), (0, 1), (0, 2)],
        hours=4,
        on_turn=check_idle,
        money=0.0,
    )


def test_every_emitted_action_is_structurally_legal():
    items = (
        work_item("WATER", (0, 0)),
        work_item(
            "FEED",
            (0, 1),
            required_supplies=(SupplyRequirement("WHEAT", 1),),
        ),
        work_item(
            "FERTILIZE",
            (0, 2),
            required_supplies=(SupplyRequirement("FERTILIZER", 1),),
        ),
        work_item(
            "PLANT",
            (0, 3),
            crop="CARROT",
            required_supplies=(SupplyRequirement("CARROT", 1, "global_seed"),),
        ),
        work_item("HARVEST", (1, 0), crop="MELON"),
    )
    controller = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan(items)
    )
    worker_count = 2

    def on_turn(_hour, result, _positions):
        check_actions_are_legal(
            result, worker_count, context=f"hour={_hour}"
        )

    simulate(
        controller,
        daily_plan(sells={"MELON": {0: 1}}),
        positions=[(0, 0), (1, 0)],
        hours=10,
        on_turn=on_turn,
        shed={"MELON": 1, "WHEAT": 1, "FERTILIZER": 1},
        inventories=[{}, {"FERTILIZER": 1}],
        seeds={"CARROT": 1},
        money=1000.0,
    )


def test_failed_pickup_does_not_unlock_feed_and_route_continues():
    feed = work_item(
        "FEED", (0, 0), required_supplies=(SupplyRequirement("WHEAT", 2),)
    )
    completed = {"watered": False}

    def builder(obs, plan, **kwargs):
        del obs, plan, kwargs
        items = [feed]
        if not completed["watered"]:
            items.append(work_item("WATER", (0, 1)))
        return fake_plan(tuple(items))

    controller = StripExecutorController(work_builder=builder)
    emitted = []
    positions = [(4, 4)]

    for hour in range(14):
        result = controller.act(
            observation(
                positions,
                hour=hour,
                shed={"WHEAT": 2} if hour == 0 else {},
                inventories=[{}],
                money=0.0,
            ),
            daily_plan(),
        )
        emitted.append(result.farmer_action)
        # A real watering flips the tile's watered_today; mirror that so the
        # route can advance past the watered tile instead of retrying it.
        if result.farmer_action == ("WATER",):
            completed["watered"] = True
        if result.farmer_action:
            positions[0] = advance(positions[0], result.farmer_action)

    assert emitted[0] == ("PICKUP", "WHEAT", 2), emitted[:3]
    assert ("FEED",) not in emitted, emitted
    assert ("WATER",) in emitted, emitted
    route_diag = controller.diagnostics["route_diagnostics"][0]
    assert route_diag["supply_state"]["acquired"] == {}
    assert route_diag["supply_state"]["failed_or_unfulfilled"] == {"WHEAT": 2}
    assert route_diag["unavailable_supply_work"].get("FEED") == 1


def test_shed_reservations_are_bounded_and_never_negative():
    items = (
        work_item(
            "FERTILIZE",
            (0, 0),
            required_supplies=(SupplyRequirement("FERTILIZER", 2),),
        ),
        work_item(
            "FERTILIZE",
            (1, 0),
            required_supplies=(SupplyRequirement("FERTILIZER", 2),),
        ),
    )
    controller = StripExecutorController(
        work_builder=lambda obs, plan, **kwargs: fake_plan(items)
    )
    result = controller.act(
        observation(
            [(0, 0), (1, 0)],
            shed={"FERTILIZER": 3},
            inventories=[{}, {}],
            money=0.0,
        ),
        daily_plan(),
    )
    diagnostics = result.diagnostics
    assert diagnostics["routes_finalized"] is True
    reservations = diagnostics["supply_diagnostics"]["total_reservations_by_item"]
    assert reservations.get("FERTILIZER", 0) == 3
    plans = diagnostics["supply_diagnostics"]["route_supply_plans"]
    assert all(
        amount >= 0
        for plan in plans
        for bucket in ("already_carried", "reserved_from_shed", "missing_stock", "capacity_limited")
        for _, amount in plan[bucket].items()
    )
    assert reservations.get("FERTILIZER", 0) <= 3


def test_controller_aggressive_selling_never_liquidates_wheat_or_fertilizer():
    controller = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True)
    )
    result = controller.act(
        observation(
            [(4, 4)],
            shed={"WHEAT": 5, "FERTILIZER": 3, "CARROT": 2, "MILK": 4},
            inventories=[{}],
            money=100.0,
        ),
        daily_plan(),
    )
    sold = {order[1] for order in result.market_actions if order[0] == "SELL"}
    assert sold == {"CARROT", "MILK"}, result.market_actions
    assert sold.isdisjoint(AGGRESSIVE_FORBIDDEN)
    assert sold <= AGGRESSIVE_ALLOWED
    assert all(order[0] != "SELL" or order[1] not in AGGRESSIVE_FORBIDDEN
               for order in result.market_actions)


def test_bootstrap_proceeds_require_shed_location_not_carried():
    """Shed-located value can fund procurement; carried value cannot."""

    feed = work_item(
        "FEED", (0, 0), required_supplies=(SupplyRequirement("WHEAT", 2),)
    )

    def run(*, shed, inventories):
        fresh = StripExecutorController(
            config=StripExecutorConfig(aggressive_sell_all=True),
            work_builder=lambda obs, plan, **kwargs: fake_plan((feed,)),
        )
        return fresh.act(
            observation(
                [(4, 4)],
                shed=shed,
                inventories=inventories,
                money=0.0,
            ),
            daily_plan(),
        )

    shed_case = run(shed={"CARROT": 5}, inventories=[{}])
    shed_sales = {order[1] for order in shed_case.market_actions if order[0] == "SELL"}
    shed_buys = {order[1] for order in shed_case.market_actions if order[0] == "BUY_PRODUCT"}
    assert shed_sales == {"CARROT"}, shed_case.market_actions
    assert shed_buys == {"WHEAT"}, shed_case.market_actions

    carried_case = run(shed={}, inventories=[{"CARROT": 5}])
    carried_sales = {
        order[1] for order in carried_case.market_actions if order[0] == "SELL"
    }
    carried_buys = {
        order[1] for order in carried_case.market_actions if order[0] == "BUY_PRODUCT"
    }
    assert carried_sales == set(), carried_case.market_actions
    assert carried_buys == set(), carried_case.market_actions
    shed_money = shed_case.diagnostics["market_diagnostics"]
    assert (
        shed_money["money_after_simulated_orders"] > shed_money["money_before_plan"]
    )
    carried_money = carried_case.diagnostics["market_diagnostics"]
    assert (
        carried_money["money_after_simulated_orders"]
        == carried_money["money_before_plan"]
    )


def test_carried_product_delivery_seam_is_unreachable():
    """Known limitation: strip has no worker-inventory -> shed deposit route.

    ``build_strip_work_plan`` represents the deposit as a tile-less DELIVERY
    item, but ``LOCAL_ACTION_PRIORITY`` has no DELIVERY entry and the on-tile
    dispatcher only handles tile-local items, so aggressive selling cannot
    liquidate worker-carried product during the same day.
    """

    from executor_v0.strip_supply import LOCAL_ACTION_PRIORITY

    assert "DELIVERY" not in LOCAL_ACTION_PRIORITY

    plan = daily_plan(sells={"WHEAT": {0: 4}})
    obs = observation(
        [(0, 0)],
        tiles=_planted_board(),
        shed={"WHEAT": 1},
        inventories=[{"WHEAT": 3}],
        seeds={"WHEAT": 5},
        money=0.0,
    )
    work = build_strip_work_plan(obs, plan)
    delivery = [item for item in work.items if item.kind == "DELIVERY"]
    assert len(delivery) == 1
    assert delivery[0].tile is None
    sell = next(item for item in work.items if item.kind == "SELL")
    assert sell.depends_on == (delivery[0].id,)

    controller = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True)
    )
    emitted = []

    def on_turn(_hour, result, _positions):
        emitted.extend((result.farmer_action, *result.hands_actions))
        assert all(action[0] != "DROP" for action in emitted if action)

    simulate(
        controller,
        plan,
        positions=[(0, 0)],
        hours=8,
        on_turn=on_turn,
        tiles=_planted_board(),
        shed={"WHEAT": 1},
        inventories=[{"WHEAT": 3}],
        seeds={"WHEAT": 5},
        money=0.0,
    )
    # The executor never deposits, so carried WHEAT is never sold from shed.
    sold = {
        order[1] for order in controller.diagnostics["market_diagnostics"].get(
            "market_orders", []
        )
        if order[0] == "SELL"
    }
    assert "WHEAT" not in sold


def _planted_board():
    board = [[None] * 10 for _ in range(10)]
    board[0][2] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 0,
        "watered_today": False,
        "fertilized_until_day": -1,
        "max_lifespan_step": -1,
        "consecutive_unwatered": 0,
    }
    return board


def test_randomized_synthetic_cases_hold_strong_invariants():
    rng = Random(21001)
    for case in range(80):
        seed = rng.randrange(1_000_000)
        local = Random(seed)
        try:
            workers = local.randint(1, 3)
            positions = [
                (local.randint(0, 4), local.randint(0, 9)) for _ in range(workers)
            ]
            tiles = _random_board(local)
            plan = daily_plan(
                crop_targets={
                    crop: local.randint(0, 2) for crop in ("WHEAT", "CARROT", "MELON")
                },
                land_count=1,
            )
            shed = {
                product: local.randint(0, 4)
                for product in local.sample(sorted(PRODUCT_NAMES), k=local.randint(0, 3))
            }
            inventories = [
                {
                    product: local.randint(0, 2)
                    for product in local.sample(
                        sorted(PRODUCT_NAMES), k=local.randint(0, 2)
                    )
                }
                for _ in range(workers)
            ]
            seeds = {"WHEAT": local.randint(0, 3), "CARROT": local.randint(0, 2)}
            money = local.choice([0.0, 50.0, 500.0, 5000.0])
            start_hour = local.choice([0, 4, 12])
            kwargs = dict(
                tiles=tiles,
                shed=shed,
                inventories=inventories,
                seeds=seeds,
                money=money,
            )
            daily = plan
            first = _run_case(positions, daily, kwargs, hours=4, start_hour=start_hour)
            second = _run_case(positions, daily, kwargs, hours=4, start_hour=start_hour)
            assert first[0] == second[0], "actions differ between identical runs"
            assert first[1] == second[1], "diagnostics differ between identical runs"
            _assert_case_invariants(positions, first, workers)
        except Exception as exc:  # noqa: BLE001 - surface the failing seed
            raise AssertionError(
                f"adversarial fuzz seed={seed} case={case} failed: {exc!r}"
            ) from exc


def _run_case(positions, daily, kwargs, *, hours, start_hour):
    controller = StripExecutorController(
        config=StripExecutorConfig(aggressive_sell_all=True)
    )
    actions = []
    diagnostics = []
    legal = []

    def on_turn(hour, result, current):
        del hour, current
        actions.append((result.farmer_action, *result.hands_actions))
        diagnostics.append(json.dumps(result.diagnostics, sort_keys=True))
        legal.append(result)

    simulate(
        controller,
        daily,
        positions=positions,
        hours=hours,
        start_hour=start_hour,
        on_turn=on_turn,
        tiles=kwargs["tiles"],
        shed=kwargs["shed"],
        inventories=kwargs["inventories"],
        seeds=kwargs["seeds"],
        money=kwargs["money"],
    )
    return actions, diagnostics, legal, controller


def _assert_case_invariants(positions, run, workers):
    actions, _diagnostics, legal, controller = run
    for hour, result in enumerate(legal):
        check_actions_are_legal(result, workers, context=f"fuzz hour={hour}")
        sold = {order[1] for order in result.market_actions if order[0] == "SELL"}
        assert sold.isdisjoint(AGGRESSIVE_FORBIDDEN), sold
        for order in result.market_actions:
            if order[0] == "SELL":
                assert order[2] > 0
            if order[0] == "BUY_PRODUCT":
                assert order[1] != "FERTILIZER"
    frozen = _phases(controller)
    tiles = [tile for _, owned, _ in frozen.values() for tile in owned]
    assert len(tiles) == len(set(tiles))
    reservations = controller.diagnostics.get("supply_diagnostics", {}).get(
        "total_reservations_by_item", {}
    )
    assert all(amount >= 0 for amount in reservations.values())


def _random_board(rng):
    board = [[None] * 10 for _ in range(10)]
    for _ in range(rng.randint(0, 4)):
        y = rng.randint(0, 9)
        x = rng.randint(0, 9)
        if rng.random() < 0.15:
            board[y][x] = "LOCKED"
            continue
        board[y][x] = {
            "kind": "PLANT",
            "crop": rng.choice(sorted(CROP_NAMES)),
            "planted_day": rng.randint(0, 5),
            "yield_units": rng.randint(0, 3),
            "watered_today": rng.random() < 0.5,
            "fertilized_until_day": -1,
            "max_lifespan_step": -1,
            "consecutive_unwatered": rng.randint(0, 2),
        }
    return board
