"""Focused regressions for the opt-in persistent queue ownership contract."""

from executor_v0.foreman import run_foreman
from executor_v0.scheduler import PersistentTaskScheduler
from executor_v0.tasks import Priority

from test_executor_v0_scheduler import make_obs, task


def _assignment(result, worker_index):
    return next(
        assignment for assignment in result.assignments
        if assignment.worker_index == worker_index
    )


def _queue_keys(result):
    return {
        worker: [queued.key for queued in queue]
        for worker, queue in result.queues.items()
    }


def _advance_workers(obs, result):
    actions = [result.farmer_action, *result.hands_actions]
    positions = [obs["farms"][0]["farmer"], *obs["farms"][0]["hands"]]
    deltas = {
        "EAST": (1, 0),
        "WEST": (-1, 0),
        "SOUTH": (0, 1),
        "NORTH": (0, -1),
    }
    for position, action in zip(positions, actions):
        if action[0] in deltas:
            dx, dy = deltas[action[0]]
            position[0] += dx
            position[1] += dy


def test_underfoot_transfer_is_atomic_and_old_owner_cannot_route_to_target():
    obs = make_obs(farmer=(2, 2), hands=[[0, 0]])
    water = task("WATER:2,2", "WATER", (2, 2), priority=Priority.MAINTENANCE)
    alternative = task(
        "WATER:0,1", "WATER", (0, 1), priority=Priority.MANAGER)

    result = run_foreman(
        obs,
        0,
        [water, alternative],
        worker_queues={0: [], 1: [water]},
        queue_ownership_repair=True,
    )

    assert sum(
        assignment.task_key == water.key
        for assignment in result.assignments
    ) == 1
    assert _assignment(result, 0).task_key == water.key
    assert _assignment(result, 0).action == ("WATER",)
    assert _assignment(result, 1).task_key == alternative.key
    assert all(
        assignment.task_key != water.key
        for assignment in result.assignments
        if assignment.worker_index == 1
    )
    assert any(
        diagnostic["event"] == "queue_transfer"
        and diagnostic["task_key"] == water.key
        for diagnostic in result.diagnostics
    )


def test_greedy_fallback_cannot_steal_reserved_target_or_resource():
    obs = make_obs(farmer=(0, 0), hands=[[9, 9]], shed={"WHEAT": 1})
    reserved = task(
        "FEED:8,8", "FEED", (8, 8), priority=Priority.MANAGER,
        item="WHEAT")
    incidental = task(
        "WATER:0,1", "WATER", (0, 1), priority=Priority.MANAGER)
    scheduler = PersistentTaskScheduler()
    scheduled = scheduler.schedule(obs, 0, [reserved, incidental], 2)

    assert _queue_keys(scheduled) == {0: [incidental.key], 1: [reserved.key]}
    assert scheduled.reservations[reserved.key] == {
        "kind": "shed", "item": "WHEAT", "amount": 1,
    }

    result = run_foreman(
        obs,
        0,
        [reserved, incidental],
        worker_queues=scheduled.queues,
        queue_ownership_repair=True,
        scheduler_reservations=scheduled.reservations,
    )

    assert _assignment(result, 0).task_key == incidental.key
    assert _assignment(result, 1).task_key == reserved.key
    assert [
        assignment.task_key for assignment in result.assignments
    ].count(reserved.key) == 1


def test_equal_priority_incidental_work_does_not_break_valid_queue_route():
    obs = make_obs(farmer=(0, 0))
    queued = task("WATER:2,2", "WATER", (2, 2), priority=Priority.MANAGER)
    incidental = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)

    result = run_foreman(
        obs,
        0,
        [queued, incidental],
        worker_queues={0: [queued]},
        queue_ownership_repair=True,
    )

    assert _assignment(result, 0).task_key == queued.key
    assert _assignment(result, 0).action == ("SOUTH",)
    assert _assignment(result, 0).reason.startswith("move_to_task:")
    assert not any(
        diagnostic.get("event") == "queue_preemption"
        for diagnostic in result.diagnostics
    )


def test_strictly_higher_priority_underfoot_work_preempts_queue_head():
    obs = make_obs(farmer=(0, 0))
    queued = task("HARVEST:2,2", "HARVEST", (2, 2), priority=Priority.MANAGER)
    urgent = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MAINTENANCE)

    result = run_foreman(
        obs,
        0,
        [queued, urgent],
        worker_queues={0: [queued]},
        queue_ownership_repair=True,
    )

    assert _assignment(result, 0).task_key == urgent.key
    assert _assignment(result, 0).action == ("WATER",)
    assert _assignment(result, 0).reason == "urgent_preemption"
    assert any(
        diagnostic.get("event") == "queue_preemption"
        and diagnostic.get("task_key") == urgent.key
        for diagnostic in result.diagnostics
    )


def test_scheduler_releases_disappeared_tasks_and_rebuilds_dependency_queue():
    scheduler = PersistentTaskScheduler()
    target = task("WATER:2,2", tile=(2, 2))
    scheduler.schedule(make_obs(), 0, [target], 1)
    disappeared = scheduler.schedule(make_obs(), 0, [], 1)

    assert _queue_keys(disappeared) == {0: []}
    assert any(
        event.get("task_key") == target.key
        and event.get("reason") == "not_current"
        for event in disappeared.events
    )

    dig = task("DIG:2,2", "DIG", (2, 2), priority=Priority.MANAGER)
    plant = task(
        "PLANT:2,2", "PLANT", (2, 2), priority=Priority.MANAGER,
        crop="WHEAT", depends_on=(dig.key,))
    blocked = scheduler.schedule(
        make_obs(seeds={"WHEAT": 1}), 0, [dig, plant], 1)
    assert _queue_keys(blocked) == {0: [dig.key]}
    released = scheduler.schedule(
        make_obs(seeds={"WHEAT": 1}), 0, [plant], 1)
    assert _queue_keys(released) == {0: [plant.key]}


def test_scheduler_releases_commitment_when_resources_are_insufficient():
    scheduler = PersistentTaskScheduler()
    feed = task(
        "FEED:1,1", "FEED", (1, 1), priority=Priority.MAINTENANCE,
        item="WHEAT")
    scheduler.schedule(make_obs(shed={"WHEAT": 1}), 0, [feed], 1)
    depleted = scheduler.schedule(make_obs(shed={}), 0, [feed], 1)

    assert _queue_keys(depleted) == {0: []}
    assert any(
        event.get("event") == "repair"
        and event.get("task_key") == feed.key
        and "shed_lacks_item" in event.get("reason", "")
        for event in depleted.events
    )


def test_scheduler_rehomes_task_after_worker_disappearance():
    scheduler = PersistentTaskScheduler()
    target = task("WATER:8,8", tile=(8, 8))
    first = scheduler.schedule(
        make_obs(farmer=(0, 0), hands=[[9, 9]]), 0, [target], 2)
    assert _queue_keys(first) == {0: [], 1: [target.key]}

    one_worker = scheduler.schedule(
        make_obs(farmer=(0, 0), hands=[]), 0, [target], 1)
    assert _queue_keys(one_worker) == {0: [target.key]}
    assert any(
        event.get("reason") == "worker_count_mismatch"
        for event in one_worker.events
    )


def test_scheduler_day_episode_and_explicit_resets_clear_ownership():
    scheduler = PersistentTaskScheduler()
    target = task("WATER:1,1", tile=(1, 1))
    scheduler.schedule(make_obs(day=3, episode_id=1), 0, [target], 1)

    day_reset = scheduler.schedule(
        make_obs(day=4, episode_id=1), 0, [], 1)
    assert _queue_keys(day_reset) == {0: []}
    assert any(event.get("reason") == "day_change" for event in day_reset.events)

    scheduler.schedule(make_obs(day=4, episode_id=2), 0, [target], 1)
    episode_reset = scheduler.schedule(
        make_obs(day=4, episode_id=3), 0, [], 1)
    assert _queue_keys(episode_reset) == {0: []}
    assert any(
        event.get("reason") == "episode_change" for event in episode_reset.events
    )

    scheduler.schedule(make_obs(day=4, episode_id=3), 0, [target], 1)
    scheduler.reset()
    explicit_reset = scheduler.schedule(
        make_obs(day=4, episode_id=3), 0, [], 1)
    assert _queue_keys(explicit_reset) == {0: []}
    assert any(
        event.get("reason") == "explicit_reset"
        for event in explicit_reset.events
    )


def test_multi_turn_scheduler_foreman_sequence_has_no_duplicate_dispatch():
    scheduler = PersistentTaskScheduler()
    obs = make_obs(
        farmer=(0, 0), hands=[[9, 9]], hour=2, step=74)
    first = task("WATER:A", "WATER", (2, 2), priority=Priority.MANAGER)
    second = task("WATER:B", "WATER", (7, 7), priority=Priority.MANAGER)
    active = [first, second]
    dispatched = []

    for _ in range(6):
        scheduled = scheduler.schedule(obs, 0, active, 2)
        result = run_foreman(
            obs,
            0,
            active,
            worker_queues=scheduled.queues,
            queue_ownership_repair=True,
            scheduler_reservations=scheduled.reservations,
        )
        assigned = [
            assignment.task_key for assignment in result.assignments
            if assignment.task_key is not None
        ]
        assert len(assigned) == len(set(assigned))
        dispatched.extend(assigned)

        scheduler.reconcile_dispatch(0, result)
        completed = {
            assignment.task_key for assignment in result.assignments
            if assignment.task_key is not None
            and assignment.action == ("WATER",)
        }
        active = [item for item in active if item.key not in completed]
        _advance_workers(obs, result)
        obs["hour"] += 1
        obs["step"] += 1
        if not active:
            break

    assert dispatched.count(first.key) == 5
    assert dispatched.count(second.key) == 5
    assert not active
    final = scheduler.schedule(obs, 0, active, 2)
    assert _queue_keys(final) == {0: [], 1: []}


def test_queue_ownership_repair_flag_off_preserves_existing_dispatch_result():
    obs = make_obs(farmer=(0, 0), hands=[[2, 2]])
    queued = task("WATER:4,4", "WATER", (4, 4), priority=Priority.MANAGER)
    incidental = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)
    queues = {0: [queued], 1: []}

    default = run_foreman(obs, 0, [queued, incidental], worker_queues=queues)
    disabled = run_foreman(
        obs,
        0,
        [queued, incidental],
        worker_queues=queues,
        queue_ownership_repair=False,
    )

    assert default.to_json_dict() == disabled.to_json_dict()
