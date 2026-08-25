"""Integration tests for the stateful executor_v0 V0 agent (issue #1 §7-9).

Covers: once-per-day manager caching with prior-labor feedback, multi-turn
legal-shaped actions across bins/days/hand changes, hour-0-only capped
workload hiring, shortage-only purchasing under the market-order cap,
six-bin sell clipping/decrement/refill/inactive-bin behavior, safe-mode
fallback plus strict mode, JSON diagnostics with requested/feasible/
achieved/submitted/observed distinctions, seat derivation, DI seams,
determinism, a real replay observation-sequence smoke, and smoke-harness
detection without requiring the engine.
"""

import copy
import json
from pathlib import Path

import pytest

from executor_v0.agent import AgentConfig, ExecutorAgent, make_agent
from executor_v0.manager import FixedPlanProvider
from executor_v0.plan import DailyPlan
from executor_v0.smoke import detect_engine
from replay_daily.constants import total_hire_cost

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"


# ------------------------------------------------------------------ helpers


def plant_tile(crop="WHEAT", **extra):
    tile = {"kind": "PLANT", "crop": crop, "planted_day": 0,
            "yield_units": 1, "max_lifespan_step": 120,
            "fertilized_until_day": -1, "consecutive_unwatered": 0,
            "watered_today": False}
    tile.update(extra)
    return tile


def pasture_tile(animal="GOOSE", **extra):
    tile = {"kind": "PASTURE", "animal": animal, "placed_day": 0,
            "yield_units": 0, "fed_today": True, "cared_today": True,
            "consecutive_unfed": 0, "pending_care_bonus": 0,
            "fertilizer_available": False}
    tile.update(extra)
    return tile


def empty_tiles():
    return [[None] * 10 for _ in range(10)]


def make_obs(day=3, hour=2, farmer=(0, 0), hands=(), hires_today=None,
             money=3000.0, shed=None, seeds=None, inventories=None,
             tiles=None, unlocked=("NW", "NE", "SW", "SE"), step=None):
    tiles = tiles if tiles is not None else empty_tiles()
    hires = len(hands) if hires_today is None else hires_today
    farm = {
        "farmer": list(farmer), "hands": [list(h) for h in hands],
        "hires_today": hires, "money": money, "tiles": tiles,
        "unlocked_quadrants": list(unlocked),
    }
    return {
        "day": day, "hour": hour,
        "step": step if step is not None else day * 24 + hour,
        "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": shed or {}, "seeds": seeds or {},
            "inventories": inventories
            if inventories is not None
            else [{} for _ in range(1 + len(hands))],
        },
    }


def recording_provider(plan):
    """FixedPlanProvider that records every daily_plan call."""

    class Recording(FixedPlanProvider):
        def __init__(self):
            super().__init__(plan)
            self.calls = []

        def daily_plan(self, obs, seat, previous_execution=None):
            self.calls.append((int(obs["day"]), seat,
                               dict(previous_execution or {})))
            return self.plan

    return Recording()


def simple_plan(**overrides):
    kwargs = dict(
        crop_targets={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        land_count=1,
        fertilizer_by_crop={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities={
            product: {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
            for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                            "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
        },
    )
    kwargs.update(overrides)
    return DailyPlan.create(**kwargs)


def assert_legal_shape(action, obs, seat=0):
    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list) and action["farmer"]
    assert all(isinstance(op, (str, int)) for op in action["farmer"])
    expected_hands = len(obs["farms"][seat].get("hands") or [])
    assert len(action["hands"]) == expected_hands
    for hand_action in action["hands"]:
        assert isinstance(hand_action, list) and hand_action
        assert all(isinstance(tok, (str, int)) for tok in hand_action)
    assert isinstance(action["market"], list)
    assert len(action["market"]) <= 10
    for order in action["market"]:
        assert isinstance(order, list) and order
        assert all(isinstance(tok, (str, int)) for tok in order)


# ------------------------------------------------------- manager once / days


def test_manager_called_once_per_day_and_prior_labor_carried():
    plan = simple_plan()
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)

    obs_d3_h0 = make_obs(day=3, hour=0)
    obs_d3_h1 = make_obs(day=3, hour=1)
    # Day 3 with two hired hands observed -> realized labor for day 3.
    obs_d3_h2 = make_obs(day=3, hour=2, hands=[(1, 1), (2, 2)],
                         hires_today=2)
    obs_d4_h0 = make_obs(day=4, hour=0)

    for obs in (obs_d3_h0, obs_d3_h1, obs_d3_h2, obs_d4_h0):
        assert_legal_shape(agent(obs), obs)

    assert [day for day, _, _ in provider.calls] == [3, 4]
    # Day 4's manager feedback mirrors day 3's observed hires (Fibonacci).
    assert provider.calls[0][2] == {"workers_hired": 0, "hire_cost": 0}
    assert provider.calls[1][2] == {
        "workers_hired": 2,
        "hire_cost": total_hire_cost(2, 1),
    }
    diag = agent.diagnostics_json()
    assert diag["days"]["4"]["previous_labor"] == {
        "workers_hired": 2, "hire_cost": total_hire_cost(2, 1)}
    assert diag["days"]["3"]["achieved_final"]["land_count"] == 4


def test_day0_starts_with_zero_labor_feedback():
    provider = recording_provider(simple_plan())
    agent = ExecutorAgent(provider, seat=0)
    agent(make_obs(day=0, hour=0))
    assert provider.calls == [(0, 0, {"workers_hired": 0, "hire_cost": 0})]


def test_same_turn_sequence_is_deterministic():
    def run_once():
        provider = recording_provider(simple_plan(
            sell_quantities={
                product: {anchor: (5 if product == "WHEAT" and anchor == 0
                                   else 0)
                          for anchor in (0, 4, 8, 12, 16, 20)}
                for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                                "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
            }))
        agent = ExecutorAgent(provider, seat=0)
        tiles = empty_tiles()
        tiles[2][2] = plant_tile()
        tiles[3][3] = plant_tile("CARROT")
        actions = []
        for hour, shed in ((0, {"WHEAT": 2}), (1, {"WHEAT": 9}),
                           (2, {"WHEAT": 9})):
            obs = make_obs(day=3, hour=hour, shed=shed, tiles=tiles)
            actions.append(agent(obs))
        return json.dumps(actions, sort_keys=True)

    assert run_once() == run_once()


# ------------------------------------------------------------------- selling


def sell_plan(wheat_bin0=5, wheat_bin1=1):
    rows = {}
    for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL", "FERTILIZER"):
        rows[product] = {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
    rows["WHEAT"][0] = wheat_bin0
    rows["WHEAT"][4] = wheat_bin1
    return simple_plan(sell_quantities=rows)


def test_sell_clips_decrements_refills_and_waits():
    provider = recording_provider(sell_plan())
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()

    obs0 = make_obs(day=3, hour=1, shed={"WHEAT": 0}, tiles=tiles)
    action0 = agent(obs0)
    assert action0["market"] == []  # nothing available: wait, no order

    obs1 = make_obs(day=3, hour=2, shed={"WHEAT": 2}, tiles=tiles)
    action1 = agent(obs1)
    assert action1["market"] == [["SELL", "WHEAT", 2]]

    # Refill later inside the SAME bin sells the remainder.
    obs2 = make_obs(day=3, hour=3, shed={"WHEAT": 9}, tiles=tiles)
    action2 = agent(obs2)
    assert action2["market"] == [["SELL", "WHEAT", 3]]

    # Ledger exhausted: no further sells this bin even with stock.
    obs3 = make_obs(day=3, hour=3, shed={"WHEAT": 9}, tiles=tiles)
    assert agent(obs3)["market"] == []

    diag = agent.diagnostics_json()
    bin_log = diag["days"]["3"]["sells"]["0"]["WHEAT"]
    assert bin_log == {"requested": 5, "submitted": 5, "remaining": 0}


def test_inactive_bin_never_sells_and_new_bin_resets_ledger():
    provider = recording_provider(sell_plan())
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()

    # Bin 0 exhausted.
    for hour, shed in ((0, {"WHEAT": 5}), (1, {"WHEAT": 5})):
        agent(make_obs(day=3, hour=hour, shed=shed, tiles=tiles))
    # Bin 1 (anchor 4): fresh requested quantity of 1.
    obs_bin1 = make_obs(day=3, hour=5, shed={"WHEAT": 7}, tiles=tiles)
    action = agent(obs_bin1)
    assert action["market"] == [["SELL", "WHEAT", 1]]
    diag = agent.diagnostics_json()
    assert diag["days"]["3"]["sells"]["4"]["WHEAT"] == \
        {"requested": 1, "submitted": 1, "remaining": 0}


# -------------------------------------------------------------------- hiring


def workload_plan():
    return simple_plan()


def test_hire_follows_workload_any_hour_within_affordability():
    """Workload-derived hiring at any hour; no arbitrary daily cap (issue #7).

    Hands are cleared at every engine day boundary, so a day that opens with
    zero hands must be able to hire mid-day whenever executable work and
    cash exist.
    """
    provider = recording_provider(workload_plan())
    config = AgentConfig(tasks_per_worker=10)
    agent = ExecutorAgent(provider, seat=0, config=config)
    tiles = empty_tiles()
    for y in range(5):          # 10 water targets -> desired 1 extra hand
        tiles[y][0] = plant_tile()

    obs_h0 = make_obs(day=3, hour=0, tiles=tiles, money=3000.0)
    action = agent(obs_h0)
    assert action["market"] == [["HIRE"]]

    # Hour 1 with the same workload: still understaffed (hands list is still
    # empty in this synthetic obs), so hiring remains available beyond h0.
    obs_h1 = make_obs(day=3, hour=1, tiles=tiles, money=3000.0)
    assert agent(obs_h1)["market"] == [["HIRE"]]
    assert agent.diagnostics_json()["days"]["3"]["hires"] == \
        {"requested": 1, "submitted": 2, "observed_max": 0}

    # Fully staffed: no more hires.
    obs_staffed = make_obs(day=3, hour=2, tiles=tiles, money=3000.0,
                           hands=[(4, 4)])
    assert agent(obs_staffed)["market"] == []


def test_hire_respects_workload_and_money_without_arbitrary_cap():
    provider = recording_provider(workload_plan())
    config = AgentConfig(tasks_per_worker=2)
    agent = ExecutorAgent(provider, seat=0, config=config)
    tiles = empty_tiles()
    for y in range(5):
        for x in range(4):      # 20 targets -> desired 10 hands
            tiles[y][x] = plant_tile()

    broke = make_obs(day=3, hour=0, tiles=tiles, money=0.0)
    assert agent(broke)["market"] == []          # cannot afford even fib(0)=1

    rich = make_obs(day=3, hour=0, tiles=tiles, money=3000.0)
    # fib costs 1+1+2+3+5+8+13+21+34+55 = 143 <= 3000: all 10 wanted hires
    # are submitted (the old hard cap of 5 is gone).
    assert agent(rich)["market"].count(["HIRE"]) == 10


def test_same_turn_sell_proceeds_fund_hires():
    """Sequential within-turn accounting: queued SELL revenue funds HIRE.

    Engine market orders apply in queue order, so hour-0 sells executed
    before hires make otherwise-unaffordable hiring mechanically possible
    (issue #7 pathology 1).
    """
    rows = {product: {anchor: (2 if anchor == 0 and product == "WHEAT"
                              else 0)
                      for anchor in (0, 4, 8, 12, 16, 20)}
            for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                            "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")}
    provider = recording_provider(simple_plan(
        crop_targets={"WHEAT": 10, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities=rows))
    config = AgentConfig(tasks_per_worker=10)
    agent = ExecutorAgent(provider, seat=0, config=config)
    tiles = empty_tiles()
    for i in range(10):         # 10 empty tiles -> desired ceil(10/10)=1 hand
        tiles[1][i] = None

    # No cash at all -- but shed holds WHEAT to sell this turn.
    obs = make_obs(day=3, hour=0, tiles=tiles, money=0.0,
                   shed={"WHEAT": 2})
    obs["market"]["prices"] = {"WHEAT": 25}
    action = agent(obs)
    assert ["SELL", "WHEAT", 2] in action["market"]
    assert ["HIRE"] in action["market"]


# -------------------------------------------------------- shortage purchases


def test_shortage_buys_are_exact_and_capped_deterministically():
    plan = simple_plan(crop_targets={"WHEAT": 6, "CARROT": 0, "TOMATO": 0,
                                     "STRAWBERRY": 0, "MELON": 0})
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()
    for i in range(6):          # six empty NW tiles -> 6 PLANT intents
        tiles[1][i] = None

    obs = make_obs(day=3, hour=2, seeds={}, shed={}, tiles=tiles)
    action = agent(obs)
    buy_orders = [o for o in action["market"] if o[0] == "BUY_SEED"]
    assert buy_orders == [["BUY_SEED", "WHEAT", 6]]
    assert_legal_shape(action, obs)


def test_carried_crop_product_does_not_reduce_seed_shortage():
    # Seeds are a global pool; carried WHEAT product must not count.
    plan = simple_plan(crop_targets={"WHEAT": 4, "CARROT": 0, "TOMATO": 0,
                                     "STRAWBERRY": 0, "MELON": 0})
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()
    for i in range(4):
        tiles[1][i] = None

    obs = make_obs(day=3, hour=2, seeds={}, shed={},
                   inventories=[{"WHEAT": 9}], tiles=tiles)
    action = agent(obs)
    assert [["BUY_SEED", "WHEAT", 4]] == \
        [o for o in action["market"] if o[0] == "BUY_SEED"]
    # And with global seeds present, no BUY_SEED is emitted at all.
    obs_seeded = make_obs(day=3, hour=2, seeds={"WHEAT": 4}, shed={},
                          inventories=[{}], tiles=tiles)
    action_seeded = agent(obs_seeded)
    assert [o for o in action_seeded["market"]
            if o[0] == "BUY_SEED"] == []


def test_over_cap_candidates_not_counted_or_decremented():
    # 9 sell products + 2 wanted hires + several buys > cap of 10: only the
    # first 10 candidates are submitted; ledgers/diagnostics stay honest.
    rows = {product: {anchor: (1 if anchor == 0 else 0)
                      for anchor in (0, 4, 8, 12, 16, 20)}
            for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                            "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")}
    plan = simple_plan(
        crop_targets={"WHEAT": 30, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0},
        sell_quantities=rows)
    provider = recording_provider(plan)
    config = AgentConfig(tasks_per_worker=5)
    agent = ExecutorAgent(provider, seat=0, config=config)
    tiles = empty_tiles()
    for y in range(5):
        for x in range(6):      # 30 empty tiles -> big seed shortage
            tiles[y][x] = None

    obs = make_obs(day=3, hour=0, shed={p: 3 for p in
                                        ("WHEAT", "CARROT", "TOMATO",
                                         "STRAWBERRY", "MELON", "EGG",
                                         "MILK", "WOOL", "FERTILIZER")},
                   seeds={}, tiles=tiles, money=3000.0)
    action = agent(obs)
    assert len(action["market"]) == 10
    sells_in_action = [o for o in action["market"] if o[0] == "SELL"]
    hires_in_action = [o for o in action["market"] if o[0] == "HIRE"]
    assert len(sells_in_action) == 9       # all sells fit
    assert len(hires_in_action) == 1       # only 1 of 2 wanted hires fits

    diag = agent.diagnostics_json()["days"]["3"]
    # Every submitted sell decremented exactly its own product ledger...
    for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL", "FERTILIZER"):
        entry = diag["sells"]["0"][product]
        assert entry == {"requested": 1, "submitted": 1, "remaining": 0}
    # ...and the dropped hires are requested but NOT submitted.
    assert diag["hires"] == {"requested": 7, "submitted": 1,
                             "observed_max": 0}


def test_market_order_cap_enforced():
    plan = simple_plan(
        crop_targets={"WHEAT": 30, "CARROT": 0, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        sell_quantities={
            product: {anchor: (5 if product == "WHEAT" else 0)
                      for anchor in (0, 4, 8, 12, 16, 20)}
            for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                            "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
        })
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()
    for y in range(5):
        for x in range(6):      # 30 plant intents -> big seed shortage
            tiles[y][x] = None

    obs = make_obs(day=3, hour=2, shed={"WHEAT": 50}, seeds={},
                   tiles=tiles)
    action = agent(obs)
    assert len(action["market"]) <= 10
    assert any(o[0] == "SELL" for o in action["market"])
    assert sum(1 for o in action["market"] if o[0] == "BUY_SEED") <= 9


# ------------------------------------------------------ fallback / strictness


def test_safe_mode_fallback_pass_shape_and_error_recorded():
    class ExplodingProvider(FixedPlanProvider):
        def daily_plan(self, obs, seat, previous_execution=None):
            raise RuntimeError("manager exploded")

    agent = ExecutorAgent(ExplodingProvider(simple_plan()), seat=0)
    obs = make_obs(day=3, hour=2, hands=[(1, 1)])
    action = agent(obs)
    assert action == {"farmer": ["PASS"], "hands": [["PASS"]],
                      "market": []}
    errors = agent.diagnostics_json()["fallback_errors"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "RuntimeError"

    strict = ExecutorAgent(ExplodingProvider(simple_plan()), seat=0,
                           config=AgentConfig(strict=True))
    with pytest.raises(RuntimeError):
        strict(obs)


def test_explicit_seat_contradiction_raises_into_fallback():
    agent = ExecutorAgent(recording_provider(simple_plan()), seat=0)
    obs = make_obs(day=3, hour=2)
    obs["player"] = 1
    action = agent(obs)  # safe mode: contradiction becomes PASS + error
    assert action["farmer"] == ["PASS"]
    assert agent.diagnostics_json()["fallback_errors"]


def test_seat_derived_from_obs_when_not_explicit():
    provider = recording_provider(simple_plan())
    agent = ExecutorAgent(provider)  # no explicit seat
    obs = make_obs(day=3, hour=2)
    obs["player"] = 1
    assert_legal_shape(agent(obs), obs, seat=1)
    assert provider.calls[0][1] == 1


# ---------------------------------------------------------------- diagnostics


def test_diagnostics_keys_json_requested_feasible_achieved():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT", watered_today=False)
    tiles[4][4] = pasture_tile("GOOSE", cared_today=True, fed_today=True)
    plan = simple_plan(crop_targets={"WHEAT": 3, "CARROT": 0, "TOMATO": 0,
                                     "STRAWBERRY": 0, "MELON": 0},
                       fertilizer_by_crop={
                           "WHEAT": 1, "CARROT": 0, "TOMATO": 0,
                           "STRAWBERRY": 0, "MELON": 0},
                       care_by_animal={"GOOSE": 1, "COW": 0, "SHEEP": 0})
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)
    agent(make_obs(day=3, hour=2, tiles=tiles))

    diag = json.loads(json.dumps(agent.diagnostics_json()))  # JSON round-trip
    day = diag["days"]["3"]
    for key in ("requested", "feasible", "projection_changes",
                "foreman_counts", "unfinished_tasks", "missed_maintenance",
                "sells", "hires", "previous_labor"):
        assert key in day
    assert day["requested"]["crop_targets"]["WHEAT"] == 3
    # Crop targets pass through projection unclipped; fertilizer/CARE are
    # clipped to mechanically eligible board counts.
    assert day["feasible"]["crop_targets"]["WHEAT"] == 3
    assert day["feasible"]["fertilizer_by_crop"]["WHEAT"] == 1
    assert day["feasible"]["care_by_animal"]["GOOSE"] == 1
    assert diag["illegal_actions"]["available"] is False
    assert diag["illegal_actions"]["count"] == 0
    assert "reason" in diag["illegal_actions"]

    # Advance the day so achieved_final / observed completions are finalized.
    agent(make_obs(day=4, hour=0, tiles=tiles))
    day3 = agent.diagnostics_json()["days"]["3"]
    assert day3["achieved_final"]["crops"]["WHEAT"] == 1
    assert day3["achieved_final"]["animals"]["GOOSE"] == 1
    assert day3["care_completed_observed"]["GOOSE"] == 1
    assert day3["foreman_counts"]["movement"] > 0


def test_projection_change_and_unfinished_tasks_recorded():
    plan = simple_plan(crop_targets={"WHEAT": 9, "CARROT": 0, "TOMATO": 0,
                                     "STRAWBERRY": 0, "MELON": 0})
    provider = recording_provider(plan)
    agent = ExecutorAgent(provider, seat=0)
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT")   # already watered -> no task
    agent(make_obs(day=3, hour=2, tiles=tiles))
    day = agent.diagnostics_json()["days"]["3"]
    assert day["projection_changes"]["land"]["requested"] == 1
    # Crop targets are not clipped at projection (reconciliation owns that).
    assert day["feasible"]["crop_targets"]["WHEAT"] == 9
    assert day["requested"]["crop_targets"]["WHEAT"] == 9


def test_turn_trace_is_opt_in_and_default_diagnostics_are_unchanged():
    obs = make_obs(day=3, hour=2)
    default_agent = ExecutorAgent(recording_provider(simple_plan()), seat=0)
    traced_agent = ExecutorAgent(
        recording_provider(simple_plan()), seat=0,
        config=AgentConfig(turn_trace=True))

    default_agent(obs)
    traced_agent(copy.deepcopy(obs))

    assert "turn_trace" not in default_agent.diagnostics_json()["days"]["3"]
    assert traced_agent.diagnostics_json()["days"]["3"]["turn_trace"]


def test_turn_trace_entry_shape_contains_survival_routing_and_counts():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile(
        consecutive_unwatered=1, watered_today=False)
    tiles[4][4] = pasture_tile(
        fed_today=False, consecutive_unfed=1)
    agent = ExecutorAgent(
        recording_provider(simple_plan()), seat=0,
        config=AgentConfig(turn_trace=True))

    agent(make_obs(day=3, hour=2, tiles=tiles, shed={"WHEAT": 2}))
    trace = agent.diagnostics_json()["days"]["3"]["turn_trace"]
    assert len(trace) == 1
    entry = trace[0]
    assert set(entry) == {
        "day", "hour", "feed", "expansion", "survival_tasks",
        "assignments", "unassigned_survival_task_keys",
        "pending_survival_task_keys", "counts", "market",
    }
    assert entry["day"] == 3
    assert entry["hour"] == 2
    assert entry["feed"] == {
        "starving": True, "shortage": 0, "unfed": 1, "reserve": 1,
    }
    assert entry["expansion"] == {
        "suppressed_current": True,
        "suppressed_from_prior": False,
        "reasons": ["current_starvation"],
    }
    assert entry["survival_tasks"]["feed"][0] == {
        "key": "FEED:4,4", "tile": [4, 4], "source": "mechanical",
        "priority": "MAINTENANCE",
    }
    assert entry["survival_tasks"]["water_must_weed_boundary"][0] == {
        "key": "WATER:2,2", "tile": [2, 2],
        "source": "water_must_weed_boundary", "priority": "MAINTENANCE",
        "consecutive_unwatered": 1, "watered_today": False,
    }
    assert set(entry["counts"]) == {"movement", "interaction", "pickup", "pass"}
    assert all(set(assignment) == {
        "worker", "worker_index", "position", "task_key", "action",
        "op_family",
    } for assignment in entry["assignments"])
    json.loads(json.dumps(agent.diagnostics_json(), sort_keys=True))


def test_turn_trace_replaces_repeated_hour_and_stays_ordered():
    agent = ExecutorAgent(
        recording_provider(simple_plan()), seat=0,
        config=AgentConfig(turn_trace=True))
    first = make_obs(day=3, hour=5)
    second_tiles = empty_tiles()
    second_tiles[1][1] = plant_tile(
        consecutive_unwatered=1, watered_today=False)
    second = make_obs(day=3, hour=5, tiles=second_tiles)
    agent(first)
    agent(make_obs(day=3, hour=2))
    agent(second)

    trace = agent.diagnostics_json()["days"]["3"]["turn_trace"]
    assert [entry["hour"] for entry in trace] == [2, 5]
    assert len([entry for entry in trace if entry["hour"] == 5]) == 1
    assert trace[1]["survival_tasks"]["water_must_weed_boundary"]


def test_turn_trace_does_not_change_actions_or_ordinary_diagnostics():
    tiles = empty_tiles()
    tiles[2][2] = plant_tile(consecutive_unwatered=1)
    tiles[4][4] = pasture_tile(fed_today=False)
    observations = [
        make_obs(day=3, hour=0, tiles=tiles, shed={"WHEAT": 2}),
        make_obs(day=3, hour=1, tiles=tiles, shed={"WHEAT": 2}),
        make_obs(day=3, hour=2, tiles=tiles, shed={"WHEAT": 2}),
    ]
    plain = ExecutorAgent(recording_provider(simple_plan()), seat=0)
    traced = ExecutorAgent(
        recording_provider(simple_plan()), seat=0,
        config=AgentConfig(turn_trace=True))

    plain_actions = [plain(copy.deepcopy(obs)) for obs in observations]
    traced_actions = [traced(copy.deepcopy(obs)) for obs in observations]
    assert plain_actions == traced_actions

    plain_diag = plain.diagnostics_json()
    traced_diag = traced.diagnostics_json()
    for record in traced_diag["days"].values():
        record.pop("turn_trace", None)
    assert json.dumps(plain_diag, sort_keys=True) == \
        json.dumps(traced_diag, sort_keys=True)


# ------------------------------------------------------------------ DI seams


def test_make_agent_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError):
        make_agent()
    with pytest.raises(FileNotFoundError):
        make_agent(checkpoint=tmp_path / "missing.pt")


def test_opponent_state_never_affects_own_actions_or_diagnostics():
    """Non-tautological isolation: two observations identical on own-seat
    data but wildly different in opponent public/private-like fields must
    produce byte-identical actions and diagnostics."""
    def build(opponent_variant):
        obs = make_obs(day=3, hour=2, hands=[(1, 1)],
                       inventories=[{"WHEAT": 1}, {}],
                       shed={"WHEAT": 2}, seeds={"WHEAT": 1})
        other = obs["farms"][1]
        other["farmer"] = list(opponent_variant["farmer"])
        other["hands"] = [list(h) for h in opponent_variant["hands"]]
        other["money"] = opponent_variant["money"]
        # Opponent-private-like sentinels planted where own private state
        # would live for seat 1; the agent must never read them.
        obs["farms"][1]["private"] = {
            "shed": {"WOOL": 999}, "seeds": {"MELON": 999},
            "inventories": [{"EGG": 999}],
        }
        obs["opponent_private_sentinel"] = opponent_variant["sentinel"]
        return obs

    variant_a = {"farmer": (0, 0), "hands": [], "money": 1.0,
                 "sentinel": {"shed": {"FERTILIZER": 1}}}
    variant_b = {"farmer": (9, 9), "hands": [(5, 5), (6, 6)],
                 "money": 99999.0,
                 "sentinel": {"shed": {}, "inventories": [{"MILK": 7}]}}

    def run(obs):
        agent = ExecutorAgent(recording_provider(simple_plan()), seat=0)
        action = agent(obs)
        return json.dumps({"action": action,
                           "diagnostics": agent.diagnostics_json()},
                          sort_keys=True)

    assert run(build(variant_a)) == run(build(variant_b))


def test_achieved_current_updated_every_turn_including_latest_day():
    """achieved diagnostics stay continuously current, so the latest day
    (e.g. day 29) has valid values without needing a following boundary."""
    tiles = empty_tiles()
    tiles[2][2] = plant_tile("WHEAT")
    tiles[4][4] = pasture_tile("GOOSE", cared_today=True)
    provider = recording_provider(simple_plan())
    agent = ExecutorAgent(provider, seat=0)
    agent(make_obs(day=29, hour=5, tiles=tiles))
    day = agent.diagnostics_json()["days"]["29"]
    assert day["achieved_current"] == {
        "crops": {name: (1 if name == "WHEAT" else 0)
                  for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                               "MELON")},
        "animals": {name: (1 if name == "GOOSE" else 0)
                    for name in ("GOOSE", "COW", "SHEEP")},
        "land_count": 4,
    }
    assert day["care_completed_observed"]["GOOSE"] == 1

    # Board change is reflected on the very next turn of the same day.
    tiles2 = empty_tiles()
    tiles2[0][0] = plant_tile("CARROT")
    agent(make_obs(day=29, hour=6, tiles=tiles2))
    day = agent.diagnostics_json()["days"]["29"]
    assert day["achieved_current"]["crops"]["CARROT"] == 1
    assert day["achieved_current"]["crops"]["WHEAT"] == 0


# ------------------------------------------------------------- replay smoke


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample episode missing")
def test_real_replay_observation_sequence_smoke():
    data = json.loads(SAMPLE.read_text())
    provider = recording_provider(simple_plan(
        crop_targets={"WHEAT": 4, "CARROT": 2, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0}))
    agent = ExecutorAgent(provider, seat=0)
    checked = 0
    for step in data["steps"][24:120]:  # skip day 0 hour 0 burst
        entry = step[0]
        obs = entry["observation"]
        if entry.get("status") not in (None, "ACTIVE"):
            continue
        action = agent(obs)
        assert_legal_shape(action, obs)
        checked += 1
    assert checked >= 50


# ------------------------------------------------------------- smoke harness


def test_detect_engine_reports_without_requiring_engine():
    report = detect_engine()
    assert set(report) == {"available", "version", "reason"}
    if not report["available"]:
        assert report["version"] is None
        assert "not installed" in report["reason"]


def test_smoke_cli_help_and_skip_path():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "executor_v0.smoke", "--help"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT))
    assert proc.returncode == 0
    assert "--manager" in proc.stdout
    assert "kaggriculture" in proc.stdout

    report = detect_engine()
    if not report["available"]:
        proc2 = subprocess.run(
            [sys.executable, "-m", "executor_v0.smoke", "--seed", "1"],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert proc2.returncode == 3
        assert proc2.stdout.startswith("SKIP:")
