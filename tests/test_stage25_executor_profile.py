"""Stage 2.5 executor profile and persistent-goal maintenance tests."""

from dataclasses import replace
import json

import pytest

from executor_v0.agent import ExecutorAgent
from executor_v0.foreman import run_foreman
from executor_v0.scheduler import PersistentTaskScheduler
from executor_v0.tasks import Priority, Task, generate_tasks
from rl_manager.executor_factory import (
    EXECUTOR_FACTORY_VERSION,
    STAGE25_EXECUTOR_PROFILE_NAME,
    STAGE25_EXECUTOR_PROFILE_VERSION,
    make_default_executor_factory,
    make_stage25_executor_factory,
)
from test_executor_v0_agent import make_obs as agent_obs
from test_executor_v0_agent import recording_provider, simple_plan
from test_executor_v0_tasks import animal_tile, make_obs as task_obs, make_plan, plant_tile


def _task(key, kind, tile, *, crop=None, animal=None, depends_on=()):
    return Task(
        key=key,
        kind=kind,
        tile=tile,
        priority=Priority.MANAGER,
        crop=crop,
        animal=animal,
        depends_on=tuple(depends_on),
    )


def test_stage25_profile_is_explicit_and_guarded():
    factory = make_stage25_executor_factory()
    config = factory.agent_config

    assert factory.name == STAGE25_EXECUTOR_PROFILE_NAME
    assert factory.version == STAGE25_EXECUTOR_PROFILE_VERSION
    assert factory.version != EXECUTOR_FACTORY_VERSION
    assert config.strict is True
    assert config.heuristic_care is True
    assert config.heuristic_fertilizer is True
    assert config.aggressive_sell_all is True
    assert config.optional_spare_watering is True
    assert config.immediate_plant_water is True
    assert config.persistent_worker_queues is True
    assert config.queue_ownership_repair is True
    assert config.batch_reserved_supplies is True
    assert config.underfoot_queue_insertion is True
    assert config.suppress_expansion_from_prior_debt is True

    profile = factory.effective_profile
    assert profile["name"] == STAGE25_EXECUTOR_PROFILE_NAME
    assert profile["version"] == STAGE25_EXECUTOR_PROFILE_VERSION
    assert profile["strategic_protection"] == {
        "suppress_expansion_from_prior_debt": True,
        "current_survival_expansion_veto": "executor_enforced",
    }

    for field in profile["required_true"]:
        with pytest.raises(ValueError, match="requires.*enabled"):
            make_stage25_executor_factory(replace(config, **{field: False}))


def test_legacy_default_factory_and_config_are_unchanged():
    factory = make_default_executor_factory()

    assert factory.name == "executor_v0"
    assert factory.version == EXECUTOR_FACTORY_VERSION
    assert factory.agent_config.strict is True
    assert factory.agent_config.optional_spare_watering is True
    assert factory.agent_config.heuristic_care is False
    assert factory.agent_config.heuristic_fertilizer is False
    assert factory.agent_config.aggressive_sell_all is False


def test_profile_is_carried_into_actual_executor_diagnostics_and_upkeep():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant_tile(
        "STRAWBERRY", planted_day=1, yield_units=0,
        watered_today=True, fertilized_until_day=-1)
    board[0][1] = animal_tile("COW", fed_today=True, cared_today=False)
    obs = task_obs(day=9, step=218, tiles=board, shed={"STRAWBERRY": 2})
    obs["market"]["prices"] = {"STRAWBERRY": 120, "FERTILIZER": 10}

    factory = make_stage25_executor_factory()
    agent = factory.create(
        backend_name="fixture", seat=0, configuration={},
        provider=recording_provider(make_plan(
            crop_targets={"STRAWBERRY": 1},
            animal_targets={"COW": 1})),
    )
    agent(obs)

    assert agent.effective_profile == factory.effective_profile
    diagnostics = agent.diagnostics_json()
    assert diagnostics["effective_profile"] == factory.effective_profile
    json.dumps(diagnostics, allow_nan=False)
    kinds = {task["kind"] for task in agent.debug_trace_turn["tasks"]}
    assert "CARE" in kinds
    assert "FERTILIZE" in kinds
    assert any(order[:2] == ["SELL", "STRAWBERRY"]
               for order in agent.debug_trace_turn["market"]["submitted"])


def test_absolute_animal_targets_preserve_existing_animals_and_inventory():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = animal_tile("COW", cared_today=True)
    obs = task_obs(day=3, tiles=board, shed={"GOOSE": 1})
    plan = make_plan(animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0})

    result = generate_tasks(obs, 0, feasible_plan=plan, remaining_sells={})
    assert result.tasks
    assert not any(task.kind == "BUY_ANIMAL" and task.animal == "COW"
                   for task in result.tasks)
    assert not any(task.kind == "DIG" and task.tile == (0, 0)
                   for task in result.tasks)


def test_persistent_goal_survives_hold_vacancy_and_unfinished_planting():
    plan = simple_plan(crop_targets={
        "WHEAT": 3, "CARROT": 0, "TOMATO": 0,
        "STRAWBERRY": 0, "MELON": 0,
    })
    profile_factory = make_stage25_executor_factory()
    agent = ExecutorAgent(
        recording_provider(plan), seat=0,
        config=profile_factory.agent_config,
        profile=profile_factory.effective_profile,
    )

    first = agent(agent_obs(day=3, hour=2, seeds={"WHEAT": 1}))
    first_record = agent.diagnostics_json()["days"]["3"]
    assert first_record["requested"]["crop_targets"]["WHEAT"] == 3
    assert sum(task["kind"] == "PLANT" for task in agent.debug_trace_turn["tasks"]) == 3
    assert first["farmer"]

    # No crop is present at the next boundary: a vacancy and unfinished
    # planting do not rewrite the persistent absolute goal.
    agent(agent_obs(day=4, hour=0, seeds={"WHEAT": 1}))
    second_record = agent.diagnostics_json()["days"]["4"]
    assert second_record["requested"]["crop_targets"]["WHEAT"] == 3
    assert second_record["feasible"]["crop_targets"]["WHEAT"] == 3


def test_lowered_goal_uses_current_reconciliation_without_forced_destruction():
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant_tile("WHEAT", watered_today=True)
    plan = make_plan(crop_targets={
        "WHEAT": 1, "CARROT": 0, "TOMATO": 0,
        "STRAWBERRY": 0, "MELON": 0,
    })
    result = generate_tasks(
        task_obs(day=3, tiles=board), 0, feasible_plan=plan,
        remaining_sells={})
    assert not any(task.kind == "DIG" for task in result.tasks)
    assert not any(task.kind == "PLANT" for task in result.tasks)


def test_failed_expansion_keeps_requested_and_feasible_plans_distinct():
    plan = simple_plan(land_count=4)
    agent = make_stage25_executor_factory().create(
        backend_name="fixture", seat=0, configuration={},
        provider=recording_provider(plan),
    )
    agent(agent_obs(day=3, hour=2, unlocked=("NW",), money=0.0))
    record = agent.diagnostics_json()["days"]["3"]

    assert record["requested"]["land_count"] == 4
    assert record["feasible"]["land_count"] == 4
    assert record["land_purchase"]["requested"] is True
    assert record["land_purchase"]["submitted"] is False


def test_goal_change_releases_obsolete_plant_and_conversion_queue_work():
    scheduler = PersistentTaskScheduler()
    old_plant = _task(
        "PLANT:WHEAT:2,2", "PLANT", (2, 2), crop="WHEAT")
    old_dig = _task("DIG:3,3", "DIG", (3, 3), crop="WHEAT")
    old_build = _task("BUILD_COOP:3,3", "BUILD_COOP", (3, 3))
    old = scheduler.schedule(
        task_obs(day=3, farmer=(0, 0), seeds={"WHEAT": 1}), 0,
        [old_plant, old_dig, old_build], 1)
    assert old.queues[0]

    # A new persistent goal removes the old PLANT and conversion intent.  A
    # same-coordinate current task, when present, must be the new object.
    new_plant = _task(
        "PLANT:CARROT:2,2", "PLANT", (2, 2), crop="CARROT")
    new = scheduler.schedule(
        task_obs(day=3, farmer=(0, 0), seeds={"CARROT": 1}), 0,
        [new_plant], 1)
    assert [task.key for task in new.queues[0]] == [new_plant.key]
    assert any(event.get("event") == "release"
               and event.get("task_key") in {
                   old_plant.key, old_dig.key, old_build.key,
               }
               for event in new.events)

    dispatch = run_foreman(
        task_obs(day=3, farmer=(0, 0), seeds={"CARROT": 1}), 0,
        [new_plant], worker_queues=new.worker_queues,
        queue_ownership_repair=True,
        scheduler_reservations=new.reservations,
    )
    assert dispatch.assignments[0].task_key == new_plant.key
    assert dispatch.assignments[0].action != ("PLANT", "WHEAT")
