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


def economic_candidates(result):
    return [item for item in result.diagnostics
            if item.get("event") == "candidate_outcome"]


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


def test_economic_repair_keeps_impossible_resources_blocked_but_rescues_water():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [
            task("NO_SEED", "PLANT", tile=(4, 4), crop="WHEAT"),
            task("NO_FERTILIZER", "FERTILIZE", tile=(4, 4),
                 item="FERTILIZER"),
            task("WATER_RESCUED", "WATER", tile=(4, 4),
                 priority=Priority.MAINTENANCE),
        ],
        available_cash=100,
        economic_repair=True,
    )

    assert result.wanted_hires == 1
    water = next(d for d in result.diagnostics
                 if d.get("task_key") == "WATER_RESCUED")
    plant = next(d for d in result.diagnostics
                 if d.get("task_key") == "NO_SEED")
    fertilizer = next(d for d in result.diagnostics
                      if d.get("task_key") == "NO_FERTILIZER")
    assert water["status"] == "scheduled"
    assert water["worker_index"] == 1
    assert plant["status"] == fertilizer["status"] == "blocked"
    assert plant["reason"].startswith("no_global_seeds")
    assert fertilizer["reason"] == "shed_lacks_item:FERTILIZER"


def test_economic_repair_accounts_for_feed_pickup_and_accepts_rescue():
    result = recommend_hires(
        make_obs(hour=11, step=83, shed={"WHEAT": 1}), 0,
        [task("FEED_RESCUED", "FEED", tile=(8, 8),
             priority=Priority.MAINTENANCE, item="WHEAT")],
        available_cash=100,
        economic_repair=True,
    )

    detail = next(d for d in result.diagnostics
                  if d.get("task_key") == "FEED_RESCUED")
    assert result.submittable_hires == 1
    assert detail["status"] == "scheduled"
    assert detail["cost"] == 10  # travel, pickup, and interaction


def test_economic_repair_shared_resources_do_not_create_phantom_completion():
    result = recommend_hires(
        make_obs(hour=21, step=93, shed={"WHEAT": 1}), 0,
        [
            task("FEED_A", "FEED", tile=(4, 4),
                 priority=Priority.MAINTENANCE, item="WHEAT"),
            task("FEED_B", "FEED", tile=(5, 4),
                 priority=Priority.MAINTENANCE, item="WHEAT"),
        ],
        available_cash=100,
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    assert [item["candidate_hires"] for item in candidates] == [0, 1, 2]
    assert len(candidates[0]["completed_task_keys"]) == 0
    assert all(len(item["completed_task_keys"]) == 1
               for item in candidates[1:])
    assert result.wanted_hires == 1


def test_economic_repair_rejects_no_added_work_and_accepts_cheap_useful_work():
    already_done = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("DONE", tile=(0, 0))], available_cash=100,
        economic_repair=True)
    cheap = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("CHEAP", tile=(4, 4))], available_cash=100,
        economic_repair=True)

    assert already_done.wanted_hires == 0
    assert any(d["reason"] == "no_added_feasible_work"
               for d in already_done.rejection_diagnostics)
    assert cheap.wanted_hires == cheap.submittable_hires == 1
    candidate = next(d for d in economic_candidates(cheap)
                     if d["candidate_hires"] == 1)
    assert candidate["newly_completed_task_keys"] == ["CHEAP"]
    assert candidate["marginal_cost"] == 1
    assert candidate["benefit"] > candidate["marginal_cost"]


def test_economic_repair_rejects_expensive_low_benefit_manager_work():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("MANAGER_REQUEST", tile=(4, 4))],
        available_cash=100,
        hire_cost_mult=10,
        economic_repair=True,
    )

    assert result.wanted_hires == result.submittable_hires == 0
    outcome = next(d for d in economic_candidates(result)
                   if d["candidate_hires"] == 1)
    assert outcome["evaluated"] is False
    assert outcome["status"] == "not_evaluated"
    assert outcome["rejected_reason"] == \
        "cumulative_cost_exceeds_benefit_upper_bound"
    assert any(d["reason"] == "cumulative_cost_exceeds_benefit_upper_bound"
               for d in result.rejection_diagnostics)


def test_economic_repair_plant_cost_includes_water_follow_up():
    result = recommend_hires(
        make_obs(hour=21, step=93, seeds={"WHEAT": 1}), 0,
        [task("PLANT_NOW", "PLANT", tile=(4, 4), crop="WHEAT")],
        available_cash=100,
        economic_repair=True,
    )

    detail = next(d for d in result.diagnostics
                  if d.get("task_key") == "PLANT_NOW")
    assert result.submittable_hires == 1
    assert detail["cost"] == 2
    assert detail["included_follow_up"]["kind"] == "WATER"


def test_economic_repair_respects_market_cap_and_authoritative_prices():
    tasks = [task("M0", tile=(4, 4)), task("M1", tile=(5, 4)),
             task("M2", tile=(4, 5))]
    result = recommend_hires(
        make_obs(hour=22, step=94), 0, tasks,
        available_cash=6, hires_today=1, hire_cost_mult=1,
        market_order_limit=1,
        policy=ScheduleHiringPolicy(manager_benefit=4),
        economic_repair=True,
    )

    assert result.wanted_hires == 3
    assert result.affordable_hires == 3
    assert result.submittable_hires == 1
    assert result.hire_costs == (1, 2, 3)
    assert any(d["reason"] == "market_order_limit"
               for d in result.rejection_diagnostics)


def test_economic_repair_affordability_cannot_create_manager_demand():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("CASHLESS", tile=(4, 4))],
        available_cash=0,
        economic_repair=True,
    )

    assert result.submittable_hires == 0
    assert result.affordable_hires == 0
    assert any(d["reason"] == "insufficient_cash_for_marginal_hire"
               for d in result.rejection_diagnostics)


def test_economic_repair_honors_final_and_terminal_horizons():
    work = [task("LAST", tile=(4, 4))]
    at_23 = recommend_hires(
        make_obs(hour=23, step=95), 0, work,
        available_cash=100, economic_repair=True)
    terminal = recommend_hires(
        make_obs(day=29, hour=22, step=718), 0, work,
        available_cash=100, economic_repair=True)

    assert at_23.submittable_hires == terminal.submittable_hires == 0
    assert all(item["remaining_capacity"] == 0
               for item in economic_candidates(at_23))
    assert any(d["reason"] == "no_future_worker_action_before_reset_or_terminal"
               for d in terminal.rejection_diagnostics)


def test_economic_repair_compares_all_candidates_against_one_baseline():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task(f"M{i}", tile=(4, 4)) for i in range(3)],
        available_cash=100,
        policy=ScheduleHiringPolicy(manager_benefit=4),
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    assert [item["candidate_hires"] for item in candidates] == [0, 1, 2, 3]
    baseline_keys = candidates[0]["completed_task_keys"]
    assert all(item["baseline_completed_task_keys"] == baseline_keys
               for item in candidates)
    assert any(item["rejected_reason"] == "not_best_net_benefit"
               for item in candidates[1:])


def test_economic_repair_upper_bound_includes_queue_only_tasks():
    queued = [task(f"Q{i}", tile=(4, 4)) for i in range(3)]
    result = recommend_hires(
        make_obs(hour=22, step=94), 0, [],
        scheduler_result=SchedulerResult(queues={0: queued}),
        available_cash=100,
        hires_today=1,
        policy=ScheduleHiringPolicy(manager_benefit=1, max_hires=3),
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    by_count = {item["candidate_hires"]: item for item in candidates}
    summary = next(item for item in result.diagnostics
                   if item.get("event") == "economic_repair")
    assert summary["benefit_upper_bound"] == 3
    assert by_count[2]["evaluated"] is True  # cumulative cost is exactly U
    assert by_count[3]["evaluated"] is False
    assert by_count[3]["status"] == "not_evaluated"
    assert "completed_task_keys" not in by_count[3]


def test_economic_repair_upper_bound_equality_is_evaluated():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task("M0", tile=(4, 4)), task("M1", tile=(4, 4))],
        available_cash=100,
        hires_today=1,
        policy=ScheduleHiringPolicy(manager_benefit=1.5),
        economic_repair=True,
    )

    outcome = next(item for item in economic_candidates(result)
                   if item["candidate_hires"] == 2)
    assert outcome["cumulative_cost"] == 3
    assert outcome["benefit_upper_bound"] == 3
    assert outcome["evaluated"] is True


def test_economic_repair_zero_benefit_model_skips_without_indexing_failures():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [task(f"M{i}", tile=(4, 4)) for i in range(3)],
        available_cash=100,
        policy=ScheduleHiringPolicy(
            max_hires=4,
            maintenance_benefit=0,
            productive_benefit=0,
            manager_benefit=0,
            logistics_benefit=0,
        ),
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    assert result.wanted_hires == result.submittable_hires == 0
    assert [item["evaluated"] for item in candidates] == [True, False, False, False]
    assert all(item["status"] == "not_evaluated"
               for item in candidates[1:])
    json.dumps(result.to_json_dict())


def test_economic_repair_skips_after_shared_resource_evaluation():
    result = recommend_hires(
        make_obs(hour=21, step=93, shed={"WHEAT": 1}), 0,
        [
            task("FEED_A", "FEED", tile=(4, 4),
                 priority=Priority.MAINTENANCE, item="WHEAT"),
            task("FEED_B", "FEED", tile=(5, 4),
                 priority=Priority.MAINTENANCE, item="WHEAT"),
        ],
        available_cash=100,
        hires_today=1,
        policy=ScheduleHiringPolicy(maintenance_benefit=1),
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    by_count = {item["candidate_hires"]: item for item in candidates}
    assert by_count[1]["evaluated"] is True
    assert len(by_count[1]["completed_task_keys"]) == 1
    assert by_count[2]["evaluated"] is False
    assert result.wanted_hires == 1


def test_economic_repair_sparse_candidates_preserve_order_clipping_schedule():
    result = recommend_hires(
        make_obs(hour=22, step=94), 0,
        [
            task("M0", tile=(4, 4)),
            task("M1", tile=(4, 4)),
            task("L0", tile=(4, 4), priority=Priority.LOGISTICS),
        ],
        available_cash=100,
        hires_today=1,
        market_order_limit=0,
        policy=ScheduleHiringPolicy(manager_benefit=2),
        economic_repair=True,
    )

    candidates = economic_candidates(result)
    by_count = {item["candidate_hires"]: item for item in candidates}
    assert result.wanted_hires == 1
    assert result.submittable_hires == 0
    assert by_count[1]["accepted"] is True
    assert by_count[3]["evaluated"] is False
    assert result.predicted_workload == 0  # market cap clips back to baseline


def test_economic_repair_false_is_legacy_output_equivalent():
    values = [
        task("A", tile=(6, 6), priority=Priority.MAINTENANCE),
        task("B", tile=(7, 7), priority=Priority.PRODUCTIVE),
        task("SELL", "SELL", tile=None),
    ]
    implicit = recommend_hires(make_obs(), 0, values, available_cash=100)
    explicit = recommend_hires(make_obs(), 0, values, available_cash=100,
                               economic_repair=False)

    assert implicit.to_json_dict() == explicit.to_json_dict()
    assert json.dumps(implicit.to_json_dict(), sort_keys=True) \
        == json.dumps(explicit.to_json_dict(), sort_keys=True)
