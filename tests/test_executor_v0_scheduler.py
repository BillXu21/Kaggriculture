import copy
import json

from executor_v0.scheduler import PersistentTaskScheduler
from executor_v0.foreman import run_foreman
from executor_v0.tasks import Priority, Task


def make_obs(*, day=3, hour=2, step=74, farmer=(0, 0), hands=(),
             inventories=None, shed=None, seeds=None, episode_id=None):
    farm = {
        "farmer": list(farmer), "hands": [list(p) for p in hands],
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
    }
    obs = {
        "day": day, "hour": hour, "step": step, "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "inventories": inventories if inventories is not None else [{} for _ in range(1 + len(hands))],
            "shed": shed or {}, "seeds": seeds or {},
        },
    }
    if episode_id is not None:
        obs["episode_id"] = episode_id
    return obs


def task(key, kind="WATER", tile=(1, 1), *, priority=Priority.MANAGER,
         item=None, quantity=1, crop=None, deadline=None, depends_on=()):
    return Task(key=key, kind=kind, tile=tile, priority=priority,
                required_item=item, quantity=quantity, crop=crop,
                deadline_hour=deadline, depends_on=tuple(depends_on))


def keys(result, worker=0):
    return [item.key for item in result.queues[worker]]


def test_persistent_target_then_interaction_at_arrival_and_failed_action_recovery():
    scheduler = PersistentTaskScheduler()
    target = task("WATER:2,2", tile=(2, 2), priority=Priority.MAINTENANCE)
    first = scheduler.schedule(make_obs(farmer=(0, 0)), 0, [target], 1)
    assert keys(first) == [target.key]
    arrived = make_obs(farmer=(2, 2))
    arrived["action_failed"] = True
    second = scheduler.schedule(arrived, 0, [target], 1)
    assert keys(second) == [target.key]
    assert any(e["event"] == "runtime" for e in second.events)


def test_queue_head_drives_each_route_step_and_interacts_on_arrival():
    scheduler = PersistentTaskScheduler()
    target = task("WATER:2,2", tile=(2, 2), priority=Priority.MAINTENANCE)

    observations = [
        make_obs(farmer=(0, 0), hour=2, step=74),
        make_obs(farmer=(0, 1), hour=3, step=75),
        make_obs(farmer=(1, 1), hour=4, step=76),
        make_obs(farmer=(2, 1), hour=5, step=77),
        make_obs(farmer=(2, 2), hour=6, step=78),
    ]
    actions = []
    for obs in observations:
        scheduled = scheduler.schedule(obs, 0, [target], 1)
        result = run_foreman(
            obs, 0, [target], worker_queues=scheduled.queues)
        actions.append(result.farmer_action)

    assert actions == [("SOUTH",), ("EAST",), ("SOUTH",), ("SOUTH",), ("WATER",)]


def test_exclusive_ownership_and_deterministic_least_extension():
    scheduler = PersistentTaskScheduler()
    a = task("A", tile=(1, 1), priority=Priority.MANAGER)
    b = task("B", tile=(8, 8), priority=Priority.MANAGER)
    result = scheduler.schedule(make_obs(farmer=(0, 0), hands=[[9, 9]]), 0, [a, b], 2)
    owners = [worker for worker, queue in result.queues.items() for item in queue if item.key in {"A", "B"}]
    assert len(owners) == 2
    assert sorted(item.key for queue in result.queues.values() for item in queue) == ["A", "B"]
    assert result.to_json_dict()["queues"]


def test_shared_inventory_and_global_seeds_are_reserved_once():
    scheduler = PersistentTaskScheduler()
    feed1 = task("FEED1", "FEED", (6, 6), priority=Priority.MAINTENANCE, item="WHEAT")
    feed2 = task("FEED2", "FEED", (7, 7), priority=Priority.MAINTENANCE, item="WHEAT")
    plant1 = task("P1", "PLANT", (1, 1), priority=Priority.MANAGER, crop="TOMATO")
    plant2 = task("P2", "PLANT", (2, 2), priority=Priority.MANAGER, crop="TOMATO")
    obs = make_obs(farmer=(4, 4), hands=[[4, 4]], shed={"WHEAT": 1}, seeds={"TOMATO": 1})
    result = scheduler.schedule(obs, 0, [feed1, feed2, plant1, plant2], 2)
    assigned = [item.key for queue in result.queues.values() for item in queue]
    assert assigned.count("FEED1") + assigned.count("FEED2") == 1
    assert assigned.count("P1") + assigned.count("P2") == 1


def test_resource_depletion_repairs_old_commitment_locally():
    scheduler = PersistentTaskScheduler()
    feed = task("FEED", "FEED", (1, 1), priority=Priority.MAINTENANCE, item="WHEAT")
    scheduler.schedule(make_obs(shed={"WHEAT": 1}), 0, [feed], 1)
    depleted = scheduler.schedule(make_obs(shed={}), 0, [feed], 1)
    assert keys(depleted) == []
    assert any(e.get("event") == "repair" and e.get("task_key") == "FEED"
               for e in depleted.events)


def test_dependencies_block_current_predecessor_then_release_when_absent():
    scheduler = PersistentTaskScheduler()
    dig = task("DIG", "DIG", (2, 2), priority=Priority.MANAGER)
    plant = task("PLANT", "PLANT", (2, 2), priority=Priority.MANAGER, crop="WHEAT", depends_on=("DIG",))
    blocked = scheduler.schedule(make_obs(seeds={"WHEAT": 1}), 0, [dig, plant], 1)
    assert keys(blocked) == ["DIG"]
    assert any(e.get("task_key") == "PLANT" and "dependency" in e.get("reason", "") for e in blocked.events)
    released = scheduler.schedule(make_obs(seeds={"WHEAT": 1}), 0, [plant], 1)
    assert keys(released) == ["PLANT"]


def test_urgent_maintenance_preempts_existing_productive_target():
    scheduler = PersistentTaskScheduler()
    productive = task("HARVEST", "HARVEST", (0, 0), priority=Priority.PRODUCTIVE)
    scheduler.schedule(make_obs(hour=23, step=95, farmer=(0, 0)), 0, [productive], 1)
    urgent = task("WATER", "WATER", (0, 0), priority=Priority.MAINTENANCE)
    result = scheduler.schedule(make_obs(hour=23, step=95, farmer=(0, 0)), 0, [productive, urgent], 1)
    assert keys(result)[0] == "WATER"
    assert any(e["event"] == "preempt" for e in result.events)


def test_day_episode_and_worker_count_resets_clear_queues():
    scheduler = PersistentTaskScheduler()
    target = task("A", tile=(1, 1))
    scheduler.schedule(make_obs(day=3, episode_id=1), 0, [target], 1)
    day_reset = scheduler.schedule(make_obs(day=4, episode_id=1), 0, [], 1)
    assert keys(day_reset) == []
    assert any(e.get("reason") == "day_change" for e in day_reset.events)
    scheduler.schedule(make_obs(day=4, episode_id=2), 0, [target], 1)
    episode_reset = scheduler.schedule(make_obs(day=4, episode_id=3), 0, [], 1)
    assert keys(episode_reset) == []
    assert any(e.get("reason") == "episode_change" for e in episode_reset.events)
    scheduler.schedule(make_obs(day=4, episode_id=3), 0, [target], 1)
    count_reset = scheduler.schedule(make_obs(day=4, episode_id=3, hands=[[1, 1]]), 0, [], 2)
    assert all(not queue for queue in count_reset.queues.values())
    assert any(e.get("reason") == "worker_count_mismatch" for e in count_reset.events)


def test_explicit_reset_and_json_safe_diagnostics():
    scheduler = PersistentTaskScheduler()
    target = task("A")
    scheduler.schedule(make_obs(), 0, [target], 1)
    scheduler.reset()
    result = scheduler.schedule(make_obs(), 0, [], 1)
    assert keys(result) == []
    json.dumps(result.to_json_dict())
    assert any(e.get("reason") == "explicit_reset" for e in result.events)
