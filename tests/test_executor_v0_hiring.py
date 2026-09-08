import copy
import json

from executor_v0.hiring import ScheduleHiringPolicy, recommend_hires
from executor_v0.scheduler import SchedulerResult
from executor_v0.tasks import Priority, Task


def make_obs(*, day=3, hour=2, step=74, farmer=(0, 0), hands=(), shed=None,
             seeds=None, inventories=None):
    farm = {
        "farmer": list(farmer), "hands": [list(p) for p in hands],
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
        "hires_today": 0,
    }
    return {
        "day": day, "hour": hour, "step": step,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {
            "inventories": inventories if inventories is not None
            else [{} for _ in range(1 + len(hands))],
            "shed": shed or {}, "seeds": seeds or {},
        },
    }


def task(key, kind="WATER", tile=(4, 4), *, priority=Priority.MANAGER,
         item=None, quantity=1, crop=None, depends_on=(), source=""):
    return Task(key=key, kind=kind, tile=tile, priority=priority,
                required_item=item, quantity=quantity, crop=crop,
                depends_on=tuple(depends_on), source=source)


def test_hour_22_can_hire_for_one_future_action_but_hour_23_and_final_cannot():
    work = task("W", tile=(0, 0))
    ordinary = recommend_hires(make_obs(day=3, hour=22, step=94), 0, [work],
                               available_cash=10)
    assert ordinary.future_worker_actions == 1
    assert ordinary.wanted_hires == 0  # existing worker can cover one action

    overloaded = [task("W1", tile=(0, 0)), task("W2", tile=(4, 4))]
    at_22 = recommend_hires(make_obs(day=3, hour=22, step=94), 0, overloaded,
                            available_cash=10)
    assert at_22.wanted_hires == 1
    at_23 = recommend_hires(make_obs(day=3, hour=23, step=95), 0, overloaded,
                            available_cash=10)
    final = recommend_hires(make_obs(day=29, hour=22, step=718), 0, overloaded,
                            available_cash=10)
    assert at_23.submittable_hires == 0
    assert final.submittable_hires == 0
    assert any(d["reason"] == "no_future_worker_action_before_reset_or_terminal"
               for d in at_23.rejection_diagnostics)


def test_fibonacci_marginal_affordability_and_order_limit():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("W0", tile=(4, 4)), task("W1", tile=(5, 4)),
         task("W2", tile=(4, 5))],
        available_cash=6, hires_today=1, hire_cost_mult=1,
        market_order_limit=2,
        # Force the estimator to need workers by giving the current worker
        # only one future slot.
        policy=ScheduleHiringPolicy(),
    )
    # hire_cost(1), hire_cost(2), ... = 1, 2, 3, 5, ...; 6 pays the first
    # three marginal hires, but the market cap submits only two.
    assert result.affordable_hires == 3
    assert result.submittable_hires == 2
    assert result.hire_costs == (1, 2, 3)


def test_spawn_travel_pickup_and_plant_water_costs_are_inspectable():
    spawned = recommend_hires(
        make_obs(farmer=(0, 0), hour=22, step=94), 0,
        [task("SPAWNED", tile=(4, 4))], available_cash=10)
    spawned_diag = next(d for d in spawned.diagnostics
                        if d["task_key"] == "SPAWNED")
    assert spawned.wanted_hires == 1
    assert spawned_diag["worker_index"] == 1
    assert spawned_diag["cost"] == 1

    occupied_spawn = recommend_hires(
        make_obs(farmer=(4, 4), hour=22, step=94), 0,
        [task("SPAWN_ORDER", tile=(5, 4))], available_cash=10)
    spawn_order_diag = next(d for d in occupied_spawn.diagnostics
                            if d["task_key"] == "SPAWN_ORDER")
    assert spawn_order_diag["cost"] == 1

    pickup = recommend_hires(
        make_obs(farmer=(0, 0), hour=22, step=94, shed={"WHEAT": 1},
                 inventories=[{}]), 0,
        [task("FEED", "FEED", tile=(8, 8), priority=Priority.MAINTENANCE,
               item="WHEAT")], available_cash=10)
    feed_diag = next(d for d in pickup.diagnostics if d["task_key"] == "FEED")
    assert feed_diag["cost"] == 18  # 16 route + pickup + interaction
    assert pickup.submittable_hires == 0

    plant = recommend_hires(
        make_obs(farmer=(0, 0), hour=22, step=94, seeds={"WHEAT": 1}), 0,
        [task("PLANT", "PLANT", tile=(4, 4), crop="WHEAT")],
        available_cash=10)
    plant_diag = next(d for d in plant.diagnostics if d["task_key"] == "PLANT")
    assert plant_diag["cost"] == 10  # 8 Manhattan + PLANT + WATER


def test_dependency_dedup_categories_and_optional_only_rejection():
    dig = task("DIG", "DIG", tile=(4, 4), priority=Priority.MANAGER)
    plant = task("PLANT", "PLANT", tile=(4, 4), crop="WHEAT",
                 depends_on=("DIG",))
    optional = task("OPTIONAL", tile=(4, 4), priority=Priority.OPTIONAL)
    uncertain = task("MAYBE", tile=(4, 4), source="uncertain forecast")
    result = recommend_hires(make_obs(seeds={"WHEAT": 1}), 0,
                             [dig, plant, optional, uncertain], available_cash=100)
    assert result.category_counts["manager"] == 1
    assert result.category_counts["blocked"] == 1
    assert result.category_counts["optional"] == 1
    assert result.category_counts["uncertain"] == 1
    assert result.wanted_hires == 0
    assert any(d["reason"] == "optional_uncertain_or_blocked_only"
               for d in result.rejection_diagnostics) is False

    only_optional = recommend_hires(make_obs(), 0, [optional], available_cash=100)
    assert only_optional.wanted_hires == 0
    assert any(d["reason"] == "optional_uncertain_or_blocked_only"
               for d in only_optional.rejection_diagnostics)


def test_persistent_queue_is_counted_once_and_keeps_queue_order():
    first = task("FIRST", tile=(4, 4), priority=Priority.MAINTENANCE)
    second = task("SECOND", tile=(4, 5), priority=Priority.MANAGER)
    result = recommend_hires(
        make_obs(farmer=(4, 4), hour=2, step=74), 0,
        [first, second, first],
        scheduler_result=SchedulerResult(queues={0: [first, second]}),
        available_cash=0,
    )
    assert result.category_counts["maintenance"] == 1
    assert result.category_counts["manager"] == 1
    assert result.predicted_workload == 3  # 1 + (one step + interaction)


def test_deterministic_repeated_json_safe_output():
    tasks = [
        task("B", tile=(7, 7), priority=Priority.PRODUCTIVE),
        task("A", tile=(6, 6), priority=Priority.MAINTENANCE),
        task("SELL", "SELL", tile=None),
    ]
    first = recommend_hires(make_obs(), 0, tasks, available_cash=100)
    second = recommend_hires(make_obs(), 0, tasks, available_cash=100)
    assert first.to_json_dict() == second.to_json_dict()
    json.dumps(first.to_json_dict())
