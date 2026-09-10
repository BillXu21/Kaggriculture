"""Focused regressions for repaired same-worker reserved-supply batching."""

from executor_v0.foreman import run_foreman
from executor_v0.scheduler import PersistentTaskScheduler
from executor_v0.tasks import Priority

from test_executor_v0_scheduler import make_obs, task


def _assignment(result, worker_index):
    return next(
        assignment for assignment in result.assignments
        if assignment.worker_index == worker_index
    )


def _feed(key, tile, *, quantity=1):
    return task(
        key, "FEED", tile, priority=Priority.MAINTENANCE,
        item="WHEAT", quantity=quantity)


def test_two_own_feed_reservations_are_picked_up_as_one_unit_batch():
    obs = make_obs(
        farmer=(4, 4), hands=[[9, 9]], inventories=[{}, {}],
        shed={"WHEAT": 2})
    first = _feed("FEED:A", (6, 6))
    second = _feed("FEED:B", (7, 7))

    result = run_foreman(
        obs, 0, [first, second],
        worker_queues={0: [first, second], 1: []},
        queue_ownership_repair=True,
        batch_reserved_supplies=True,
        scheduler_reservations={
            first.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
            second.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
        },
    )

    assert result.farmer_action == ("PICKUP", "WHEAT", 2)
    assert _assignment(result, 0).task_key == first.key
    assert result.counts["pickup"] == 1
    consumed = [
        event for event in result.diagnostics
        if event["event"] == "queue_reservation_consumed"
    ]
    assert {(event["task_key"], event["amount"]) for event in consumed} == {
        (first.key, 1), (second.key, 1)
    }


def test_batch_cannot_use_another_workers_reserved_shed_stock():
    obs = make_obs(
        farmer=(4, 4), hands=[[9, 9]], inventories=[{}, {}],
        shed={"WHEAT": 3})
    first = _feed("FEED:A", (6, 6))
    second = _feed("FEED:B", (7, 7))
    other = _feed("FEED:C", (8, 8))

    result = run_foreman(
        obs, 0, [first, second, other],
        worker_queues={0: [first, second], 1: [other]},
        queue_ownership_repair=True,
        batch_reserved_supplies=True,
        scheduler_reservations={
            first.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
            second.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
            other.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
        },
    )

    assert result.farmer_action == ("PICKUP", "WHEAT", 2)
    assert result.hands_actions[0][0] in {"NORTH", "SOUTH", "EAST", "WEST"}
    assert _assignment(result, 1).task_key == other.key
    assert all(
        assignment.task_key != other.key
        for assignment in result.assignments
        if assignment.worker_index == 0
    )


def test_batch_demand_subtracts_carried_inventory_once():
    obs = make_obs(
        farmer=(4, 4), inventories=[{"WHEAT": 1}], shed={"WHEAT": 1})
    first = _feed("FEED:A", (6, 6))
    second = _feed("FEED:B", (7, 7))
    scheduler = PersistentTaskScheduler()
    scheduled = scheduler.schedule(obs, 0, [first, second], 1)

    result = run_foreman(
        obs, 0, [first, second],
        worker_queues=scheduled.queues,
        queue_ownership_repair=True,
        batch_reserved_supplies=True,
        scheduler_reservations=scheduled.reservations,
    )

    # The carried unit satisfies the first FEED and is available to the
    # forthcoming queue as well; only the second task is shed-reserved.
    assert scheduled.reservations == {
        second.key: {"kind": "shed", "item": "WHEAT", "amount": 1}
    }
    assert result.farmer_action != ("PICKUP", "WHEAT", 2)

    # After the first FEED consumes the carried unit, the second task can use
    # its one reserved unit exactly once.
    scheduler.reconcile_dispatch(0, completed_task_keys=[first.key])
    next_obs = make_obs(farmer=(4, 4), inventories=[{}], shed={"WHEAT": 1})
    next_scheduled = scheduler.schedule(next_obs, 0, [second], 1)
    next_result = run_foreman(
        next_obs, 0, [second], worker_queues=next_scheduled.queues,
        queue_ownership_repair=True, batch_reserved_supplies=True,
        scheduler_reservations=next_scheduled.reservations)
    assert next_result.farmer_action == ("PICKUP", "WHEAT", 1)


def test_batch_pickup_reconciles_reservations_before_transfer_and_release():
    obs = make_obs(
        farmer=(4, 4), hands=[[0, 0]], inventories=[{}, {}],
        shed={"WHEAT": 2})
    first = _feed("FEED:A", (6, 6))
    second = _feed("FEED:B", (7, 7))
    scheduler = PersistentTaskScheduler()
    scheduled = scheduler.schedule(obs, 0, [first, second], 2)

    assert [item.key for item in scheduled.queues[0]] == [first.key, second.key]
    assert set(scheduled.reservations) == {first.key, second.key}
    result = run_foreman(
        obs, 0, [first, second], worker_queues=scheduled.queues,
        queue_ownership_repair=True, batch_reserved_supplies=True,
        scheduler_reservations=scheduled.reservations)
    assert result.farmer_action == ("PICKUP", "WHEAT", 2)

    reconciliation = scheduler.reconcile_dispatch(0, result)
    assert {event["reason"] for event in reconciliation} == {
        "reservation_to_carried"
    }
    assert scheduler.reconcile_dispatch(
        0, transfers=[{"task_key": second.key, "new_worker_index": 1}]
    )[-1]["reason"] == "transferred"
    scheduler.reconcile_dispatch(0, released_task_keys=[second.key])

    next_obs = make_obs(
        farmer=(4, 4), hands=[[0, 0]],
        inventories=[{"WHEAT": 2}, {}], shed={})
    next_result = scheduler.schedule(next_obs, 0, [first], 2)
    assert [item.key for item in next_result.queues[0]] == [first.key]
    assert not next_result.reservations


def test_batch_keeps_exclusive_claims_unique_and_flag_off_unchanged():
    obs = make_obs(
        farmer=(4, 4), hands=[[9, 9]], inventories=[{}, {}],
        shed={"WHEAT": 2})
    first = _feed("FEED:A", (6, 6))
    second = _feed("FEED:B", (7, 7))
    queues = {0: [first, second], 1: []}
    reservations = {
        first.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
        second.key: {"kind": "shed", "item": "WHEAT", "amount": 1},
    }

    result = run_foreman(
        obs, 0, [first, second], worker_queues=queues,
        queue_ownership_repair=True, batch_reserved_supplies=True,
        scheduler_reservations=reservations)
    assert [a.task_key for a in result.assignments].count(first.key) == 1
    assert [a.task_key for a in result.assignments].count(second.key) == 0

    legacy = run_foreman(
        obs, 0, [first, second], worker_queues=queues,
        queue_ownership_repair=True,
        scheduler_reservations=reservations)
    assert legacy.farmer_action == ("PICKUP", "WHEAT", 1)
