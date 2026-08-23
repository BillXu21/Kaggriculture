"""Stage-2b focused mechanic parity probes: town shops/consumption/unlock,
global day-end RNG (weeds, shop draws), daily refresh ordering/reset,
reset determinism, and terminal lifecycle against the pinned official
1.32.7 engine.

Every scenario replays the exact same action pair through both engines and
compares the complete canonical state after every submitted pair
(``run_same_action_replay`` raises at the FIRST divergent field). Semantic
assertions pin the official mechanic itself (from the verified official
source ``kaggriculture.py`` @ 28b6d8af) so a jointly-wrong fast engine
cannot pass silently.

Skipped unless ``kaggle_environments`` passes the provenance guard.

Official mechanics pinned here (verified against the pinned source):

- Shop unlock: at end-of-day, when ``(day + 1) % townShopUnlockInterval == 0``
  and fewer than ``MAX_SHOP_INSTANCES == 8`` instances exist, ONE shop drawn
  with replacement via ``rng.choice(sorted(SHOPS))`` is appended. Duplicate
  instances are ordered multiplicity; each consumes independently.
- RNG: one ``random.Random((seed * 1_000_003) ^ day)`` per day boundary,
  shared across both farms in player order (weed draws over empty unlocked
  tiles row-major, farm 0 then farm 1), then the shop choice draw.
- Town consumption: every ``townShopSellInterval == 4`` steps each instance
  drains its product list (multiplier 2 for single-product shops); every
  ``townCenterSellInterval == 24`` steps the town center drains one unit of
  every non-FERTILIZER product (fires at step 0 too). Subtraction is
  UNCONDITIONAL: market stock can go negative, and prices keep tracking the
  scarcity branch down there.
- End-of-day order per farm: plant refresh, animal refresh, weed spawn,
  inventory drop to shed, farmer/hands/hires_today/inventories reset.
- Terminal: DONE + reward = final farm money at canonical step
  ``episodeSteps - 1``; earlier ERROR/INVALID statuses stay invalid even
  once DONE masks them (covered by the offline status-history suite); the
  official wrapper refuses any post-terminal step and the fast engine must
  not transition either.

Observed evidence this suite locks in (official 1.32.7, PASS-only, seed 3):
unlocks at steps 72/144/216/288/360/432/504/576 drawing
BRUNCH_SPOT, FARMERS_MARKET, PET_CAFE, YARN_STORE, PIZZA_SHOP,
FARMERS_MARKET, FARMERS_MARKET, BAKERY (three duplicate FARMERS_MARKET
instances; two single-product instances), cap held through day 27.
"""

from __future__ import annotations

import pytest

from oracle import DivergenceError, run_same_action_replay
from oracle.canonical import deep_diff
from oracle.provenance import ProvenanceError, verify_official_provenance

try:
    verify_official_provenance()
    OFFICIAL_AVAILABLE = True
    _SKIP_REASON = ""
except ProvenanceError as error:
    OFFICIAL_AVAILABLE = False
    _SKIP_REASON = str(error)

pytestmark = pytest.mark.skipif(not OFFICIAL_AVAILABLE, reason=_SKIP_REASON)


# ---------------------------------------------------------------------------
# Official rule tables (replicated verbatim from the pinned kaggriculture.py)
# ---------------------------------------------------------------------------

SHOP_PRODUCTS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
TOWN_CENTER_PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
)
MAX_SHOP_INSTANCES = 8
MARKET_I0 = 10_000

EXPECTED_UNLOCK_SEQUENCE = (
    "BRUNCH_SPOT", "FARMERS_MARKET", "PET_CAFE", "YARN_STORE",
    "PIZZA_SHOP", "FARMERS_MARKET", "FARMERS_MARKET", "BAKERY",
)
EXPECTED_UNLOCK_STEPS = (72, 144, 216, 288, 360, 432, 504, 576)

PAS = {"farmer": ["PASS"], "hands": [], "market": []}


def act(**overrides) -> dict[str, object]:
    action: dict[str, object] = {"farmer": ["PASS"], "hands": [], "market": []}
    action.update(overrides)
    return action


def replay(name: str, trace, configuration=None, max_turns=720):
    try:
        return run_same_action_replay(configuration or {"seed": 3}, trace, max_turns=max_turns)
    except DivergenceError as error:
        raise AssertionError(f"{name}: {error}") from error


def _shop_delta(shops: list[str]) -> dict[str, int]:
    """Market inventory delta one town-consumption step applies for `shops`."""
    delta = {name: 0 for name in (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "EGG", "MILK", "WOOL", "FERTILIZER",
    )}
    for shop_name in shops:
        products = SHOP_PRODUCTS[shop_name]
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            delta[item] -= multiplier
    return delta


# ---------------------------------------------------------------------------
# Town shops: unlock timing, duplicates, cap, consumption math
# ---------------------------------------------------------------------------


def test_shop_unlock_timing_duplicates_cap_and_consumption_trajectory() -> None:
    """27 PASS-only days, seed 3: exact unlock steps/order, duplicate
    multiplicity, the 8-instance cap, and a per-turn market-inventory
    trajectory recomputed from the official shop tables (covers the
    single-product x2 multiplier, the every-4-step shop cadence, the
    every-24-step town-center product set including its step-0 fire, and
    per-step price refresh visibility)."""
    from oracle.backend import make_backend

    configuration = {"seed": 3}
    official = make_backend("official", configuration)
    official.reset()

    expected = {name: MARKET_I0 for name in
                ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                 "EGG", "MILK", "WOOL", "FERTILIZER")}
    shops: list[str] = []
    unlock_events: list[tuple[int, str]] = []
    seen_trajectory_turns = 0

    for turn in range(648):
        step = turn  # interpreter step used by this transition
        if step % 4 == 0:
            for item, change in _shop_delta(shops).items():
                expected[item] += change
        if step % 24 == 0:
            for item in TOWN_CENTER_PRODUCTS:
                expected[item] -= 1

        official.step([dict(PAS), dict(PAS)])
        canonical = official.canonical_state()

        for item, value in expected.items():
            assert canonical["market"]["inventory"][item] == value, (
                f"inventory trajectory broke at turn {turn} for {item}"
            )
        seen_trajectory_turns += 1

        current_shops = canonical["town"]["unlocked_shops"]
        if len(current_shops) > len(shops):
            unlock_events.append((canonical["step"], current_shops[-1]))
        shops = list(current_shops)

    assert seen_trajectory_turns == 648
    assert [event[0] for event in unlock_events] == list(EXPECTED_UNLOCK_STEPS)
    assert [event[1] for event in unlock_events] == list(EXPECTED_UNLOCK_SEQUENCE)
    # Draw-with-replacement duplicates preserved as ordered multiplicity.
    assert shops.count("FARMERS_MARKET") == 3
    # Single-product instances exist so the x2 multiplier is exercised.
    assert "PET_CAFE" in shops and "YARN_STORE" in shops
    assert len(shops) == MAX_SHOP_INSTANCES
    # Step-0 town-center fire pinned explicitly (before any shop existed).
    assert unlock_events[0][0] == 72  # nothing unlocked before day 3

    # Full same-action parity over the identical trace.
    result = replay("shops_unlock_consume_648", [[dict(PAS), dict(PAS)]] * 648, configuration)
    assert result.turns_executed == 648


def test_town_center_fires_at_step_zero_with_empty_town() -> None:
    """The very first transition consumes one unit of every non-FERTILIZER
    product through the town center even though no shop is unlocked yet."""
    from oracle.backend import make_backend

    official = make_backend("official", {"seed": 3})
    fast = make_backend("fast", {"seed": 3})
    official.reset()
    fast.reset()
    official.step([dict(PAS), dict(PAS)])
    fast.step([dict(PAS), dict(PAS)])
    official_canonical = official.canonical_state()
    assert not official_canonical["town"]["unlocked_shops"]
    for item in TOWN_CENTER_PRODUCTS:
        assert official_canonical["market"]["inventory"][item] == MARKET_I0 - 1
    assert official_canonical["market"]["inventory"]["FERTILIZER"] == MARKET_I0
    assert deep_diff(official_canonical, fast.canonical_state()) == []


# ---------------------------------------------------------------------------
# Insufficient market stock: unconditional subtraction into negative stock
# ---------------------------------------------------------------------------


def test_market_stock_goes_negative_without_clamping() -> None:
    """BUY_PRODUCT has no stock check officially (money and shed room only),
    and town consumption subtracts unconditionally: huge configured
    money/shed capacity drive WHEAT stock far below zero while prices keep
    following the scarcity branch. Both engines must agree exactly."""
    configuration = {"seed": 3, "startingMoney": 4_000_000, "shedCapacity": 40_000}
    trace = [
        # Both players drain the same product in lockstep (per-unit quoting).
        [act(market=[["BUY_PRODUCT", "WHEAT", 15_000]]),
         act(market=[["BUY_PRODUCT", "WHEAT", 6_000]])],
        # Sell some back at scarcity pricing (supply returns above $1 only).
        [act(market=[["SELL", "WHEAT", 200]]), PAS],
        # Third buy pushes the stock deeper below zero past a day boundary.
        [act(market=[["BUY_PRODUCT", "WHEAT", 12_000]]), PAS],
        *[[dict(PAS), dict(PAS)] for _ in range(47)],
    ]
    from oracle.backend import make_backend

    official = make_backend("official", configuration)
    official.reset()
    min_stock = None
    for pair in trace:
        official.step([dict(pair[0]), dict(pair[1])])
        wheat = official.canonical_state()["market"]["inventory"]["WHEAT"]
        min_stock = wheat if min_stock is None else min(min_stock, wheat)
    final = official.canonical_state()
    assert min_stock is not None and min_stock < -20_000, min_stock
    assert final["market"]["inventory"]["WHEAT"] < 0
    # Prices never collapse below the official floor even at negative stock.
    assert min(final["market"]["prices"].values()) >= 1

    result = replay("negative_market_stock", trace, configuration)
    assert result.turns_executed == len(trace)


# ---------------------------------------------------------------------------
# Weed spawning: config handling, eligible tiles, seeded determinism
# ---------------------------------------------------------------------------


def test_zero_weed_chance_never_spawns_weeds() -> None:
    configuration = {"seed": 7, "weedSpawnChance": 0.0}
    from oracle.backend import make_backend

    official = make_backend("official", configuration)
    official.reset()
    for _ in range(96):
        official.step([dict(PAS), dict(PAS)])
    canonical = official.canonical_state()
    for farm in canonical["farms"]:
        for row in farm["tiles"]:
            for tile in row:
                assert tile != {"kind": "WEED"}

    result = replay("zero_weed_chance", [[dict(PAS), dict(PAS)]] * 96, configuration)
    assert result.turns_executed == 96


def test_high_weed_chance_spawns_only_on_empty_unlocked_tiles() -> None:
    """weedSpawnChance=0.5 floods the farms: parity must hold, every weed
    must sit inside an unlocked quadrant, and the shared per-day RNG stream
    (farm 0 row-major, then farm 1, then any shop draw) must produce the
    exact same board on both engines."""
    configuration = {"seed": 11, "weedSpawnChance": 0.5}
    from oracle.backend import make_backend

    official = make_backend("official", configuration)
    official.reset()
    for _ in range(96):
        official.step([dict(PAS), dict(PAS)])
    canonical = official.canonical_state()
    for farm_index, farm in enumerate(canonical["farms"]):
        weed_count = sum(
            1 for row in farm["tiles"] for tile in row if tile == {"kind": "WEED"}
        )
        assert weed_count > 20, (farm_index, weed_count)
        unlocked = set(farm["unlocked_quadrants"])
        half = 5
        quadrant_of = lambda x, y: ("N" if y < half else "S") + ("W" if x < half else "E")
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if tile == {"kind": "WEED"}:
                    assert quadrant_of(x, y) in unlocked

    result = replay("high_weed_chance", [[dict(PAS), dict(PAS)]] * 96, configuration)
    assert result.turns_executed == 96


def test_weed_rng_same_seed_repeatable_reset_repeatable_other_seed_distinct() -> None:
    """Same seed => bit-identical boards and shop draws across fresh runs
    (reset-after-run repeatability); a different seed must diverge somewhere
    a draw actually occurs (proves the draws are used, not ignored)."""
    from oracle.backend import make_backend

    def snapshots(seed: int) -> list[tuple[dict, dict]]:
        configuration = {"seed": seed, "weedSpawnChance": 0.5}
        official = make_backend("official", configuration)
        fast = make_backend("fast", configuration)
        official.reset()
        fast.reset()
        recorded = []
        for turn in range(96):
            official.step([dict(PAS), dict(PAS)])
            fast.step([dict(PAS), dict(PAS)])
            if turn in (23, 47, 71, 95):
                recorded.append((official.canonical_state(), fast.canonical_state()))
        return recorded

    run_a = snapshots(11)
    run_a_again = snapshots(11)  # fresh backends: reset-after-run repeatability
    run_b = snapshots(12)

    for (off_a, fast_a), (off_b, fast_b) in zip(run_a, run_a_again):
        assert deep_diff(off_a, off_b) == []
        assert deep_diff(fast_a, fast_b) == []
        assert deep_diff(off_a, fast_a) == []

    differences = sum(
        1 for (off_a, _), (off_b, _) in zip(run_a, run_b) if deep_diff(off_a, off_b)
    )
    assert differences > 0  # a different seed changed real RNG-driven state


# ---------------------------------------------------------------------------
# Daily refresh ordering / reset at day boundaries (with shop unlock)
# ---------------------------------------------------------------------------


def test_day_boundary_resets_hands_carry_and_hires_then_unlocks_shop() -> None:
    """Crossing into day 1 (plain boundary) and into day 3 (boundary that
    also fires the first shop unlock): hands removed, carried inventories
    dropped to the shed, hires_today reset, farmer back at spawn -- and the
    new shop appears exactly on the day-3 boundary."""
    from oracle.backend import make_backend

    configuration = {"seed": 3}
    trace = [
        [act(market=[["HIRE"], ["HIRE"], ["BUY_PRODUCT", "WHEAT", 6]]), PAS],
        [act(farmer=["PICKUP", "WHEAT", 4], hands=[["EAST"], ["WEST"]]), PAS],
        *[[dict(PAS), dict(PAS)] for _ in range(21)],
        [dict(PAS), dict(PAS)],  # turn 23: boundary into day 1
        [act(market=[["HIRE"]], hands=[["SOUTH"]]), PAS],  # Fibonacci reset => cost 1
        *[[dict(PAS), dict(PAS)] for _ in range(46)],
        [dict(PAS), dict(PAS)],  # turn 71: boundary into day 3 (+ shop unlock)
        [dict(PAS), dict(PAS)],
    ]

    official = make_backend("official", configuration)
    official.reset()
    for turn, pair in enumerate(trace):
        official.step([dict(pair[0]), dict(pair[1])])
        canonical = official.canonical_state()
        if turn == 23:
            farm = canonical["farms"][0]
            assert farm["hands"] == []
            assert farm["hires_today"] == 0
            assert farm["farmer"] == [4, 4]
            assert canonical["privates"][0]["inventories"] == [
                {name: 0 for name in (
                    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL", "FERTILIZER",
                    "GOOSE", "COW", "SHEEP")}
            ]
            assert canonical["privates"][0]["shed"]["WHEAT"] == 6
            assert canonical["town"]["unlocked_shops"] == []
        if turn == 71:
            assert canonical["town"]["unlocked_shops"] == ["BRUNCH_SPOT"]
            assert canonical["farms"][0]["hires_today"] == 0

    result = replay("day_boundary_reset_unlock", trace, configuration)
    assert result.turns_executed == len(trace)


def test_plant_lifecycle_shifts_weed_draws_before_shop_choice() -> None:
    """A planted tile removes one eligible weed-spawn slot from farm 0 from
    planting until it converts to a WEED, shifting every later draw in the
    shared per-day RNG stream (farm-0 weeds, farm-1 weeds, then the shop
    choice). Watering on the planting day resets consecutive_unwatered at
    the day-0 boundary; two unwatered days convert the tile at the day-2
    boundary. Exact board + shop parity proves the fast engine consumes the
    global stream in the official order even when crop state changes it."""
    configuration = {"seed": 3}
    trace = [
        [act(market=[["BUY_SEED", "WHEAT", 2]]), PAS],
        [act(farmer=["NORTH"]), PAS],
        [act(farmer=["NORTH"]), PAS],
        [act(farmer=["NORTH"]), PAS],
        [act(farmer=["NORTH"]), PAS],  # farmer now stands on tile (x=4, y=0)
        [act(farmer=["PLANT", "WHEAT"]), PAS],
        [act(farmer=["WATER"]), PAS],
        *[[dict(PAS), dict(PAS)] for _ in range(89)],
    ]
    from oracle.backend import make_backend

    official = make_backend("official", configuration)
    official.reset()
    for turn, pair in enumerate(trace):
        official.step([dict(pair[0]), dict(pair[1])])
        tile = official.canonical_state()["farms"][0]["tiles"][0][4]
        if turn == 5:
            assert tile["kind"] == "PLANT" and tile["consecutive_unwatered"] == 1
        if turn == 6:
            assert tile["watered_today"] is True
        if turn == 23:  # end of day 0: watering reset the counter
            assert tile["consecutive_unwatered"] == 0
        if turn == 47:  # end of day 1: one unwatered day
            assert tile["consecutive_unwatered"] == 1
        if turn >= 71:  # end of day 2: converted; slot stays occupied
            assert tile == {"kind": "WEED"}

    result = replay("plant_shifts_rng_draws", trace, configuration)
    assert result.turns_executed == len(trace)


# ---------------------------------------------------------------------------
# Terminal lifecycle
# ---------------------------------------------------------------------------


def test_terminal_step_day_hour_rewards_and_statuses_exact() -> None:
    """episodeSteps=50: DONE lands at canonical step 49 (day 2, hour 1) and
    rewards equal the FINAL farm money exactly (seat 0 spent 100 on seeds)."""
    configuration = {"seed": 3, "episodeSteps": 50}
    trace = [
        [act(market=[["BUY_SEED", "WHEAT", 10]]), PAS],
        *[[dict(PAS), dict(PAS)] for _ in range(48)],
    ]
    result = replay("terminal_rewards", trace, configuration)
    assert result.turns_executed == 49
    assert result.final_step == 49
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.official_rewards == result.fast_rewards == [2900.0, 3000.0]

    from oracle.backend import make_backend

    fast = make_backend("fast", configuration)
    fast.reset()
    for pair in trace:
        fast.step([dict(pair[0]), dict(pair[1])])
    money = [farm["money"] for farm in fast.canonical_state()["farms"]]
    assert money == [2900.0, 3000.0]
    assert fast.canonical_state()["step"] == 49
    assert fast.canonical_state()["day"] == 2
    assert fast.canonical_state()["hour"] == 1


def test_no_extra_transition_after_terminal() -> None:
    """After DONE the official wrapper refuses further steps outright; the
    fast engine must accept the call but leave every canonical field,
    reward, and status untouched."""
    import kaggle_environments.errors
    from oracle.backend import make_backend

    configuration = {"seed": 3, "episodeSteps": 50}
    official = make_backend("official", configuration)
    fast = make_backend("fast", configuration)
    official.reset()
    fast.reset()
    pas_pair = [dict(PAS), dict(PAS)]
    for _ in range(49):
        official.step(list(pas_pair))
        fast.step(list(pas_pair))
    assert official.statuses == ["DONE", "DONE"]

    official_before = official.canonical_state()
    fast_before = fast.canonical_state()

    with pytest.raises(kaggle_environments.errors.FailedPrecondition):
        official.step(list(pas_pair))

    observations, rewards, statuses = fast.step(list(pas_pair))
    assert deep_diff(official_before, fast.canonical_state()) == []
    assert deep_diff(fast_before, fast.canonical_state()) == []
    assert rewards == official_before["rewards"]
    assert statuses == ["DONE", "DONE"]
    assert observations[0]["step"] == 49
