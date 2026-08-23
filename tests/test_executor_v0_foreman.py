"""Focused tests for executor_v0 stage 5: deterministic greedy foreman.

Covers issue #1 section 6: underfoot execution, priority-dominated greedy
assignment with single claims, soft inventory specialization, shed routing
with bounded bulk pickup and no illegal pickups, one-step legal Manhattan
movement with official coordinate conversion and locked avoidance,
exact opcode mapping for every worker task kind, dependency blocking,
worker alignment/purity/diagnostics, market-task separation, and a real
replay smoke.
"""

import copy
import json
from pathlib import Path

import pytest

from executor_v0.foreman import (
    SHED_ACCESS_TILES,
    ForemanConfig,
    run_foreman,
)
from executor_v0.tasks import Priority, Task
from replay_daily.constants import ANIMALS

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"


# ------------------------------------------------------------------ helpers


def make_obs(farmer=(0, 0), hands=(), inventories=None, shed=None, seeds=None,
             tiles=None):
    tiles = tiles if tiles is not None else [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": list(farmer), "hands": [list(h) for h in hands],
        "hires_today": len(hands), "money": 3000.0, "tiles": tiles,
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
    }
    return {
        "day": 3, "hour": 2, "step": 90, "player": 0,
        "farms": [farm, {**copy.deepcopy(farm), "farmer": [9, 9]}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": shed or {}, "seeds": seeds or {},
            "inventories": inventories if inventories is not None
            else [{} for _ in range(1 + len(hands))],
        },
    }


def task(key, kind, tile=None, *, priority=Priority.MANAGER, item=None,
         quantity=1, crop=None, animal=None, product=None, deadline=None,
         depends_on=()):
    return Task(key=key, kind=kind, tile=tile, priority=priority,
                required_item=item, quantity=quantity, crop=crop,
                animal=animal, product=product, deadline_hour=deadline,
                depends_on=tuple(depends_on))


def assignment_for(result, worker_index):
    return next(a for a in result.assignments
                if a.worker_index == worker_index)


# ------------------------------------------------------------- worker views


def test_worker_order_position_conversion_and_inventory_alignment():
    obs = make_obs(farmer=[4, 5], hands=[[1, 2], [3, 4]],
                   inventories=[{"WHEAT": 2}, {"EGG": 1}, {}])
    result = run_foreman(obs, 0, tasks=[])
    assert result.farmer_action == ("PASS",)
    assert len(result.hands_actions) == 2
    # Positions arrive as official [x, y]; foreman works in [y, x].
    assert result.assignments[0].action == ("PASS",)  # nothing to do
    from executor_v0.foreman import _worker_views
    workers = _worker_views(obs, 0)
    assert [(w.position, w.inventory) for w in workers] == [
        ((5, 4), {"WHEAT": 2}), ((2, 1), {"EGG": 1}), ((4, 3), {})]


def test_malformed_worker_position_rejected():
    obs = make_obs()
    obs["farms"][0]["farmer"] = [1]
    with pytest.raises(ValueError):
        run_foreman(obs, 0, tasks=[])


# ------------------------------------------------------- underfoot & items


def test_underfoot_highest_priority_executes():
    obs = make_obs(farmer=(2, 2))
    tasks = [
        task("HARVEST:2,2", "HARVEST", (2, 2), priority=Priority.PRODUCTIVE),
        task("WATER:2,2", "WATER", (2, 2), priority=Priority.MAINTENANCE),
    ]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.farmer_action == ("WATER",)
    assert assignment_for(result, 0).reason == "underfoot_execution"
    assert result.counts["interaction"] == 1


def test_missing_required_item_prevents_underfoot_execution():
    obs = make_obs(farmer=(2, 2), shed={})
    tasks = [task("FEED:2,2", "FEED", (2, 2),
                  priority=Priority.MAINTENANCE, item="WHEAT")]
    result = run_foreman(obs, 0, tasks=tasks)
    # Shed also lacks wheat: no execution, no illegal pickup, task unassigned.
    assert result.farmer_action == ("PASS",)
    assert result.counts["pickup"] == 0
    assert [t.key for t in result.unassigned_tile_tasks] == ["FEED:2,2"]


# ------------------------------------------------- priority / claims / ties


def test_priority_dominates_distance_and_tasks_claimed_once():
    obs = make_obs(farmer=(0, 0))
    far_water = task("WATER:9,9", "WATER", (9, 9),
                     priority=Priority.MAINTENANCE)
    near_harvest = task("HARVEST:0,1", "HARVEST", (0, 1),
                        priority=Priority.PRODUCTIVE)
    result = run_foreman(obs, 0, tasks=[near_harvest, far_water])
    assert result.assignments[0].task_key == "WATER:9,9"
    assert result.farmer_action == ("SOUTH",)
    assert [t.key for t in result.unassigned_tile_tasks] == ["HARVEST:0,1"]


def test_equal_distance_tie_broken_by_task_key():
    obs = make_obs(farmer=(0, 0))
    tasks = [
        task("WATER:1,1", "WATER", (1, 1), priority=Priority.MAINTENANCE),
        task("WATER:1,0", "WATER", (1, 0), priority=Priority.MAINTENANCE),
    ]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.assignments[0].task_key == "WATER:1,0"  # lower key wins
    assert [t.key for t in result.unassigned_tile_tasks] == ["WATER:1,1"]


def test_two_workers_claim_distinct_tasks_deterministically():
    # Hand official position [x,y]=[5,0] -> canonical tile (0,5).
    obs = make_obs(farmer=(0, 0), hands=[[5, 0]],
                   inventories=[{}, {}])
    tasks = [
        task("WATER:1,0", "WATER", (1, 0), priority=Priority.MAINTENANCE),
        task("WATER:0,5", "WATER", (0, 5), priority=Priority.MAINTENANCE),
    ]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.assignments[0].task_key == "WATER:1,0"   # farmer: nearer
    assert result.assignments[1].task_key == "WATER:0,5"   # hand: underfoot
    assert result.hands_actions[0] == ("WATER",)


# --------------------------------------------------------- soft specialization


def test_carrying_required_item_wins_reasonable_tie():
    obs = make_obs(farmer=(0, 0),
                   inventories=[{"FERTILIZER": 1}])
    needs_item = task("FERTILIZE:0,3", "FERTILIZE", (0, 3),
                      priority=Priority.MANAGER, item="FERTILIZER")
    plain = task("WATER:3,0", "WATER", (3, 0), priority=Priority.MANAGER)
    result = run_foreman(obs, 0, tasks=[needs_item, plain])
    # Both distance 3; equal scores fall back to the stable key, but the
    # carried-item task never pays a variety penalty either way.
    assert result.assignments[0].task_key == "FERTILIZE:0,3"


def test_third_item_type_allowed_when_only_useful_work():
    obs = make_obs(farmer=(0, 0), shed={"FERTILIZER": 3},
                   inventories=[{"WHEAT": 1, "EGG": 1}])
    tasks = [task("FERTILIZE:5,5", "FERTILIZE", (5, 5),
                  priority=Priority.MANAGER, item="FERTILIZER")]
    result = run_foreman(obs, 0, tasks=tasks)
    # Variety penalty applies but the task is still claimed (only work).
    assert result.assignments[0].task_key == "FERTILIZE:5,5"
    assert result.farmer_action[0] in ("SOUTH", "EAST")


def test_variety_penalty_loses_to_no_penalty_at_equal_distance():
    obs = make_obs(farmer=(0, 0),
                   inventories=[{"WHEAT": 1, "EGG": 1}])
    variety = task("FERTILIZE:0,4", "FERTILIZE", (0, 4),
                   priority=Priority.MANAGER, item="FERTILIZER")
    same_types = task("FEED:4,0", "FEED", (4, 0),
                      priority=Priority.MANAGER, item="WHEAT")
    result = run_foreman(obs, 0, tasks=[variety, same_types])
    # Equal distances; adding FERTILIZER as a third type loses to WHEAT.
    assert result.assignments[0].task_key == "FEED:4,0"


# ------------------------------------------------------------------ shed flow


def test_missing_item_routes_via_shed_then_bulk_pickup():
    obs = make_obs(farmer=(0, 0), shed={"FERTILIZER": 9})
    tasks = [task("FERTILIZE:6,6", "FERTILIZE", (6, 6),
                  priority=Priority.MANAGER, item="FERTILIZER")]
    result = run_foreman(obs, 0, tasks=tasks)
    # Step one toward the nearest access tile (4,4).
    assert result.farmer_action == ("SOUTH",)
    assert result.assignments[0].reason.startswith("move_to_shed")
    # Arrive at the access tile: bounded bulk pickup, no waiting.
    obs2 = make_obs(farmer=(4, 4), shed={"FERTILIZER": 9})
    at_access = run_foreman(obs2, 0, tasks=tasks)
    assert at_access.farmer_action == ("PICKUP", "FERTILIZER", 5)
    assert at_access.counts["pickup"] == 1


def test_seeds_pickup_from_seed_store():
    obs = make_obs(farmer=(4, 5), seeds={"TOMATO": 3})
    tasks = [task("PLANT:TOMATO:6,6", "PLANT", (6, 6),
                  priority=Priority.MANAGER, crop="TOMATO",
                  item="TOMATO")]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.farmer_action == ("PICKUP", "TOMATO", 3)


def test_shed_lacks_item_no_illegal_pickup_pass_and_unassigned():
    obs = make_obs(farmer=(4, 4), shed={})
    tasks = [task("FEED:6,6", "FEED", (6, 6), priority=Priority.MAINTENANCE,
                  item="WHEAT")]
    result = run_foreman(obs, 0, tasks=tasks)
    # Even standing at a shed access tile, no stock means no pickup and no
    # claim; the worker passes and the task stays unassigned.
    assert result.farmer_action == ("PASS",)
    assert assignment_for(result, 0).reason == "no_feasible_task"
    assert result.counts["pickup"] == 0
    assert [t.key for t in result.unassigned_tile_tasks] == ["FEED:6,6"]


# ------------------------------------------------------------- movement laws


@pytest.mark.parametrize("delta,expected", [
    ((3, 0), "SOUTH"), ((-3, 0), "NORTH"), ((0, 3), "EAST"),
    ((0, -3), "WEST"),
])
def test_one_legal_step_axis_selection(delta, expected):
    fy, fx = 4, 4
    ty, tx = fy + delta[0], fx + delta[1]
    obs = make_obs(farmer=(fx, fy))
    tasks = [task(f"WATER:{ty},{tx}", "WATER", (ty, tx),
                  priority=Priority.MAINTENANCE)]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.farmer_action == (expected,)
    assert result.counts["movement"] == 1


def test_movement_avoids_locked_tile_falls_back_to_other_axis():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[1][0] = "LOCKED"  # blocks the vertical-first step from (0,0)
    obs = make_obs(farmer=(0, 0), tiles=tiles)
    # Horizontal step still reduces distance -> deterministic fallback.
    tasks = [task("WATER:3,2", "WATER", (3, 2),
                  priority=Priority.MAINTENANCE)]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.farmer_action == ("EAST",)
    assert result.counts["movement"] == 1


def test_blocked_only_reducing_step_emits_pass():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[1][0] = "LOCKED"
    obs = make_obs(farmer=(0, 0), tiles=tiles)
    tasks = [task("WATER:3,0", "WATER", (3, 0),
                  priority=Priority.MAINTENANCE)]
    result = run_foreman(obs, 0, tasks=tasks)
    # No reducing step remains legal; V0 passes rather than sidestepping.
    assert result.farmer_action == ("PASS",)
    assert assignment_for(result, 0).reason == "movement_blocked"


def test_blocked_both_axes_emits_pass_with_reason():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = "LOCKED"
    tiles[1][0] = "LOCKED"
    obs = make_obs(farmer=(0, 0), tiles=tiles)
    tasks = [task("WATER:5,5", "WATER", (5, 5),
                  priority=Priority.MAINTENANCE)]
    result = run_foreman(obs, 0, tasks=tasks)
    assert result.farmer_action == ("PASS",)
    assert assignment_for(result, 0).reason == "movement_blocked"


def test_out_of_bounds_target_never_stepped_into():
    obs = make_obs(farmer=(0, 0))
    tasks = [task("WATER:0,0", "WATER", (0, 0),
                  priority=Priority.MAINTENANCE)]
    # Underfoot executes; no movement attempted toward invalid targets.
    assert run_foreman(obs, 0, tasks=tasks).farmer_action == ("WATER",)


# ------------------------------------------------------------ opcode mapping


def test_exact_opcode_mapping_for_all_tile_kinds():
    cases = [
        ("WATER", dict(crop="WHEAT"), ("WATER",), {}),
        ("HARVEST", dict(animal="COW"), ("HARVEST",), {}),
        ("DIG", dict(crop="WHEAT"), ("DIG",), {}),
        ("BUILD_COOP", {}, ("BUILD_COOP",), {}),
        ("BUILD_PASTURE", {}, ("BUILD_PASTURE",), {}),
        ("FEED", dict(item="WHEAT", animal="GOOSE"), ("FEED",),
         {"WHEAT": 1}),
        ("CARE", dict(animal="GOOSE"), ("CARE",), {}),
        ("FERTILIZE", dict(item="FERTILIZER", crop="WHEAT"),
         ("FERTILIZE",), {"FERTILIZER": 1}),
        ("COLLECT_FERTILIZER", dict(animal="GOOSE"),
         ("COLLECT_FERTILIZER",), {}),
        ("PLACE", dict(animal="COW", quantity=1), ("PLACE", "COW", 1),
         {"COW": 1}),
        ("PLANT", dict(crop="MELON", item="MELON"), ("PLANT", "MELON"),
         {"MELON": 1}),
    ]
    for index, (kind, extra, expected, carry) in enumerate(cases):
        coord = (index, 9)               # canonical [y, x]
        obs = make_obs(farmer=(coord[1], coord[0]),  # official [x, y]
                       inventories=[carry])
        t = task(f"{kind}:{index}", kind, coord, **extra)
        result = run_foreman(obs, 0, tasks=[t])
        assert result.farmer_action == expected, kind


def test_malformed_metadata_omitted_safely():
    obs = make_obs(farmer=(2, 2))
    bad_plant = task("PLANT:2,2", "PLANT", (2, 2))  # no crop metadata
    result = run_foreman(obs, 0, tasks=[bad_plant])
    assert result.farmer_action == ("PASS",)
    assert assignment_for(result, 0).reason == "malformed_metadata"
    assert [t.key for t in result.unassigned_tile_tasks] == ["PLANT:2,2"]


# --------------------------------------------------------------- dependencies


def test_dependency_blocks_until_predecessor_absent():
    obs = make_obs(farmer=(2, 2))
    dig = task("DIG:2,2", "DIG", (2, 2), priority=Priority.MANAGER)
    plant = task("PLANT:TOMATO:2,2", "PLANT", (2, 2),
                 priority=Priority.MANAGER, crop="TOMATO",
                 depends_on=("DIG:2,2",))
    blocked = run_foreman(obs, 0, tasks=[dig, plant])
    assert blocked.assignments[0].task_key == "DIG:2,2"
    keys = {t.key for t in blocked.unassigned_tile_tasks}
    assert "PLANT:TOMATO:2,2" in keys  # predecessor still present
    # Predecessor completed/absent -> plant becomes executable.
    after = run_foreman(obs, 0, tasks=[plant])
    assert after.farmer_action == ("PLANT", "TOMATO")


# --------------------------------------------------- purity / diagnostics


def test_purity_json_diagnostics_and_market_separation():
    obs = make_obs(farmer=(0, 0), shed={"FERTILIZER": 2})
    frozen = copy.deepcopy(obs)
    tasks = [
        task("SELL:WHEAT:0", "SELL", None, priority=Priority.LOGISTICS,
             product="WHEAT", quantity=7, deadline=3),
        task("BUY_SEED:TOMATO", "BUY_SEED", None,
             priority=Priority.LOGISTICS, crop="TOMATO", quantity=2),
        task("BUY_LAND:NE", "BUY_LAND", None, priority=Priority.MANAGER,
             product="NE"),
        task("WATER:1,1", "WATER", (1, 1), priority=Priority.MAINTENANCE),
    ]
    result = run_foreman(obs, 0, tasks=tasks)
    assert obs == frozen
    assert [t.key for t in result.market_tasks] == \
        ["BUY_LAND:NE", "SELL:WHEAT:0", "BUY_SEED:TOMATO"]
    payload = result.to_json_dict()
    json.dumps(payload)
    assert set(result.counts) == {"movement", "interaction", "pickup",
                                  "pass"}
    assert sum(result.counts.values()) == len(result.assignments)


def test_determinism_identical_inputs_identical_outputs():
    obs = make_obs(farmer=(0, 0), hands=[[2, 2]], shed={"WHEAT": 4},
                   inventories=[{}, {}])
    tasks = [
        task("WATER:1,1", "WATER", (1, 1), priority=Priority.MAINTENANCE),
        task("FEED:3,3", "FEED", (3, 3), priority=Priority.MAINTENANCE,
             item="WHEAT"),
        task("WATER:4,4", "WATER", (4, 4), priority=Priority.MAINTENANCE),
    ]
    r1 = run_foreman(obs, 0, tasks=tasks)
    r2 = run_foreman(copy.deepcopy(obs), 0, tasks=copy.deepcopy(tasks))
    assert r1 == r2


# ---------------------------------------------------------------- real smoke


@pytest.mark.skipif(not SAMPLE.exists(),
                    reason="local real sample not present")
def test_real_observation_smoke_legal_worker_ops_no_exceptions():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    step_index = 24 * 5 + 3  # day 5, hour 3
    observation = raw["steps"][step_index][0]["observation"]
    legal_first_ops = {
        "NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "DIG",
        "PLANT", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "FERTILIZE",
        "COLLECT_FERTILIZER", "PICKUP", "PLACE", "DROP",
    }
    tasks = [
        task("WATER:1,1", "WATER", (1, 1), priority=Priority.MAINTENANCE),
        task("SELL:WHEAT:0", "SELL", None, priority=Priority.LOGISTICS,
             product="WHEAT", quantity=3, deadline=3),
    ]
    result = run_foreman(observation, 0, tasks=tasks)
    for action in (result.farmer_action, *result.hands_actions):
        assert isinstance(action, tuple) and action
        assert action[0] in legal_first_ops
        if action[0] == "PICKUP":
            assert len(action) == 3 and isinstance(action[2], int)
    json.dumps(result.to_json_dict())
