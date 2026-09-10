"""Focused regressions for bounded repaired-queue underfoot insertion."""

import json

from executor_v0.foreman import run_foreman
from executor_v0.tasks import Priority

from test_executor_v0_scheduler import make_obs, task


def _assignment(result, worker_index):
    return next(
        assignment for assignment in result.assignments
        if assignment.worker_index == worker_index
    )


def _insertion_diagnostics(result):
    return [
        diagnostic for diagnostic in result.diagnostics
        if diagnostic.get("event") == "queue_underfoot_insertion"
    ]


def test_safe_equal_priority_insertion_then_retained_route():
    obs = make_obs(farmer=(0, 0))
    retained = task("WATER:2,2", "WATER", (2, 2), priority=Priority.MANAGER)
    incidental = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)
    queues = {0: [retained], 1: []}

    result = run_foreman(
        obs, 0, [retained, incidental], worker_queues=queues,
        queue_ownership_repair=True, underfoot_queue_insertion=True)

    assert _assignment(result, 0).task_key == incidental.key
    assert result.farmer_action == ("WATER",)
    accepted = _insertion_diagnostics(result)
    assert len(accepted) == 1
    assert accepted[0]["accepted"] is True
    assert accepted[0]["task_key"] == incidental.key
    assert accepted[0]["retained_task_key"] == retained.key
    assert accepted[0]["retained_destination"] == [2, 2]
    assert queues == {0: [retained], 1: []}

    next_result = run_foreman(
        make_obs(farmer=(0, 0)), 0, [retained], worker_queues=queues,
        queue_ownership_repair=True, underfoot_queue_insertion=True)
    assert _assignment(next_result, 0).task_key == retained.key
    assert next_result.farmer_action == ("SOUTH",)


def test_deadline_breaking_insertion_is_rejected_and_route_resumes():
    obs = make_obs(farmer=(0, 0), hour=2, step=74)
    retained = task(
        "WATER:2,2", "WATER", (2, 2), priority=Priority.MANAGER, deadline=5)
    incidental = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)

    result = run_foreman(
        obs, 0, [retained, incidental], worker_queues={0: [retained]},
        queue_ownership_repair=True, underfoot_queue_insertion=True)

    assert _assignment(result, 0).task_key == retained.key
    assert result.farmer_action == ("SOUTH",)
    rejected = _insertion_diagnostics(result)
    assert len(rejected) == 1
    assert rejected[0]["accepted"] is False
    assert rejected[0]["task_key"] == incidental.key
    assert rejected[0]["retained_destination"] == [2, 2]
    assert rejected[0]["reason"].startswith("retained_route_deadline:")


def test_owned_underfoot_task_is_not_duplicated_by_insertion():
    obs = make_obs(farmer=(0, 0), hands=[[5, 5]])
    retained = task("WATER:2,2", "WATER", (2, 2), priority=Priority.MANAGER)
    exclusive = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)

    result = run_foreman(
        obs, 0, [retained, exclusive],
        worker_queues={0: [retained], 1: [exclusive]},
        queue_ownership_repair=True, underfoot_queue_insertion=True)

    assignments = [assignment for assignment in result.assignments
                   if assignment.task_key == exclusive.key]
    assert len(assignments) == 1
    assert assignments[0].worker_index == 0
    assert _assignment(result, 0).task_key == exclusive.key
    assert any(
        diagnostic.get("event") == "queue_transfer"
        and diagnostic.get("task_key") == exclusive.key
        for diagnostic in result.diagnostics
    )


def test_insertion_flag_off_matches_repaired_path():
    obs = make_obs(farmer=(0, 0))
    retained = task("WATER:2,2", "WATER", (2, 2), priority=Priority.MANAGER)
    incidental = task("WATER:0,0", "WATER", (0, 0), priority=Priority.MANAGER)
    queues = {0: [retained], 1: []}

    repaired = run_foreman(
        obs, 0, [retained, incidental], worker_queues=queues,
        queue_ownership_repair=True)
    explicitly_off = run_foreman(
        obs, 0, [retained, incidental], worker_queues=queues,
        queue_ownership_repair=True, underfoot_queue_insertion=False)

    assert json.dumps(repaired.to_json_dict(), sort_keys=True) == \
        json.dumps(explicitly_off.to_json_dict(), sort_keys=True)
