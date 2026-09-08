"""Deterministic, schedule-informed farm-hand hiring estimates.

This module is deliberately separate from :mod:`executor_v0.agent`.  It is a
pure, inspectable seam that an agent can call while assembling its market
orders.  It estimates future worker turns only: the worker action for the
current observation has already happened when a market HIRE is accepted.

Coordinates supplied by an observation are the engine's ``[x, y]`` pairs;
``Task.tile`` and shed access tiles use canonical ``(y, x)`` pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from executor_v0.foreman import SHED_ACCESS_TILES
from executor_v0.tasks import Priority, Task
from replay_daily.constants import FARM_HAND_COST_MULT_DEFAULT, hire_cost

__all__ = [
    "HiringRecommendation",
    "ScheduleHiringPolicy",
    "recommend",
    "recommend_hires",
]


_MARKET_KINDS = frozenset({
    "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND", "HIRE",
})
_WORKER_KINDS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})
_EPSILON = 1e-9


@dataclass(frozen=True)
class ScheduleHiringPolicy:
    """Mechanics and conservative choices used by :func:`recommend_hires`.

    The defaults mirror ``foreman``/``scheduler``.  ``max_hires`` is only an
    estimator bound; the caller's market-order cap remains authoritative.
    """

    shed_access_tiles: tuple[tuple[int, int], ...] = SHED_ACCESS_TILES
    pickup_batch: int = 5
    total_days: int = 30
    hours_per_day: int = 24
    final_actionable_step: int = 30 * 24 - 2
    max_hires: int = 64

    def __post_init__(self) -> None:
        if not self.shed_access_tiles:
            raise ValueError("shed_access_tiles must not be empty")
        if self.pickup_batch < 1:
            raise ValueError("pickup_batch must be positive")
        if self.total_days < 1 or self.hours_per_day < 1:
            raise ValueError("total_days and hours_per_day must be positive")
        if self.max_hires < 0:
            raise ValueError("max_hires must be non-negative")

    def recommend_hires(self, *args: Any, **kwargs: Any) -> "HiringRecommendation":
        """Convenience method for integrations that retain a policy object."""
        kwargs["policy"] = self
        return recommend_hires(*args, **kwargs)


@dataclass(frozen=True)
class HiringRecommendation:
    """A JSON-safe explanation of one hiring decision."""

    wanted_hires: int
    affordable_hires: int
    submittable_hires: int
    predicted_workload: int
    predicted_capacity: int
    predicted_capacity_with_hires: int
    future_worker_actions: int
    category_counts: dict[str, int] = field(default_factory=dict)
    diagnostics: tuple[dict[str, Any], ...] = ()
    rejection_diagnostics: tuple[dict[str, Any], ...] = ()
    hire_costs: tuple[int, ...] = ()
    worker_count: int = 1

    @property
    def wanted(self) -> int:
        return self.wanted_hires

    @property
    def affordable(self) -> int:
        return self.affordable_hires

    @property
    def submittable(self) -> int:
        return self.submittable_hires

    @property
    def recommended_hires(self) -> int:
        return self.submittable_hires

    @property
    def category_breakdown(self) -> dict[str, int]:
        return dict(self.category_counts)

    @property
    def orders(self) -> list[list[str]]:
        return [["HIRE"] for _ in range(self.submittable_hires)]

    @property
    def reasons(self) -> tuple[dict[str, Any], ...]:
        return self.rejection_diagnostics

    def to_json_dict(self) -> dict[str, Any]:
        """Return only ordinary JSON values, including all diagnostics."""
        return {
            "wanted_hires": self.wanted_hires,
            "affordable_hires": self.affordable_hires,
            "submittable_hires": self.submittable_hires,
            "wanted": self.wanted_hires,
            "affordable": self.affordable_hires,
            "submittable": self.submittable_hires,
            "orders": self.orders,
            "predicted_workload": self.predicted_workload,
            "predicted_capacity": self.predicted_capacity,
            "predicted_capacity_with_hires": self.predicted_capacity_with_hires,
            "future_worker_actions": self.future_worker_actions,
            "worker_count": self.worker_count,
            "category_counts": dict(self.category_counts),
            "hire_costs": list(self.hire_costs),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "rejection_diagnostics": [dict(item) for item in self.rejection_diagnostics],
        }


@dataclass
class _Worker:
    position: tuple[int, int]
    inventory: dict[str, int]
    remaining: int
    workload: int = 0

    def copy(self) -> "_Worker":
        return _Worker(self.position, dict(self.inventory), self.remaining,
                       self.workload)


@dataclass
class _Evaluation:
    capacity: int
    workload: int = 0
    required_workload: int = 0
    assigned: set[str] = field(default_factory=set)
    task_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    workers: list[_Worker] = field(default_factory=list)


def _task_sort(task: Task) -> tuple[int, int, str]:
    return (int(task.priority),
            task.deadline_hour if task.deadline_hour is not None else 1 << 30,
            str(task.key))


def _priority_name(task: Task) -> str:
    try:
        return Priority(int(task.priority)).name.lower()
    except (TypeError, ValueError):
        return str(task.priority).lower()


def _uncertain(task: Task) -> bool:
    text = f"{task.key} {task.source}".lower()
    return any(word in text for word in
               ("uncertain", "maybe", "speculative", "unknown", "forecast"))


def _engine_positions(obs: Mapping, seat: int) -> list[tuple[int, int]]:
    farm = (obs.get("farms") or [])[seat]
    raw = [farm.get("farmer", [0, 0])]
    raw.extend(farm.get("hands") or [])
    result = []
    for position in raw:
        if isinstance(position, (list, tuple)) and len(position) == 2:
            # Engine: [x, y].  Internal/task coordinates: (y, x).
            result.append((int(position[1]), int(position[0])))
    return result or [(0, 0)]


def _normalise_positions(
    obs: Mapping,
    seat: int,
    worker_count: int,
    worker_positions: Sequence[Sequence[int]] | None,
) -> list[tuple[int, int]]:
    if worker_positions is None:
        positions = _engine_positions(obs, seat)
    else:
        positions = []
        for position in worker_positions:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError(f"worker position must be a pair, got {position!r}")
            # Explicit positions are already canonical, matching Task.tile.
            positions.append((int(position[0]), int(position[1])))
    if not positions:
        positions = [(0, 0)]
    while len(positions) < worker_count:
        positions.append(positions[-1])
    return positions[:worker_count]


def _normalise_inventories(
    obs: Mapping,
    worker_count: int,
    worker_inventories: Sequence[Mapping[str, int]] | None,
) -> list[dict[str, int]]:
    if worker_inventories is None:
        values = ((obs.get("private") or {}).get("inventories") or [])
    else:
        values = worker_inventories
    result = []
    for value in values[:worker_count]:
        result.append({str(k): max(0, int(v)) for k, v in (value or {}).items()})
    while len(result) < worker_count:
        result.append({})
    return result


def _future_actions(obs: Mapping, policy: ScheduleHiringPolicy) -> int:
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", day * policy.hours_per_day + hour))
    day_boundary = (day + 1) * policy.hours_per_day
    # The current action is at ``step``.  The terminal step is a state, not a
    # worker action, hence the minus one in this inclusive range calculation.
    next_boundary = min(day_boundary, policy.final_actionable_step + 1)
    return max(0, next_boundary - step - 1)


def _spawn_positions(
    positions: Sequence[tuple[int, int]],
    hires: int,
    policy: ScheduleHiringPolicy,
) -> list[tuple[int, int]]:
    occupied = {tile: sum(position == tile for position in positions)
                for tile in policy.shed_access_tiles}
    result = []
    for _ in range(hires):
        # Preserve the fixed NW/NE/SW/SE preference on equal occupancy; do
        # not let tuple lexicographic order change the engine's tile order.
        tile = min(enumerate(policy.shed_access_tiles),
                   key=lambda pair: (occupied[pair[1]], pair[0]))[1]
        result.append(tile)
        occupied[tile] += 1
    return result


def _copy_task_map(
    tasks: Sequence[Task],
    queues: Mapping[Any, Sequence[Task]] | None,
) -> tuple[dict[str, Task], dict[str, int], dict[str, int], list[dict[str, Any]]]:
    """Merge current tasks and queues, retaining one deterministic task/key."""
    chosen: dict[str, Task] = {}
    queue_owner: dict[str, int] = {}
    queue_rank: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []

    def add(task: Task, owner: int | None = None) -> None:
        if not isinstance(task, Task):
            raise TypeError(f"tasks must contain Task values, got {task!r}")
        key = str(task.key)
        old = chosen.get(key)
        if old is None or _task_sort(task) < _task_sort(old):
            chosen[key] = task
            if owner is not None:
                queue_owner[key] = owner
        elif owner is not None and key not in queue_owner:
            queue_owner[key] = owner

    for task in tasks:
        add(task)
    if queues:
        for raw_worker, queue in sorted(queues.items(), key=lambda item: str(item[0])):
            try:
                worker = int(raw_worker)
            except (TypeError, ValueError):
                continue
            for rank, task in enumerate(queue or ()):
                add(task, worker)
                queue_rank.setdefault(str(task.key), rank)
    return chosen, queue_owner, queue_rank, diagnostics


def _queue_mapping(scheduler_result: Any) -> Mapping[Any, Sequence[Task]] | None:
    if scheduler_result is None:
        return None
    if isinstance(scheduler_result, Mapping):
        return (scheduler_result.get("queues")
                or scheduler_result.get("worker_queues")
                or scheduler_result.get("task_queues"))
    for name in ("queues", "worker_queues", "task_queues"):
        value = getattr(scheduler_result, name, None)
        if value is not None:
            return value
    return None


def _task_category(task: Task, current_keys: set[str]) -> tuple[str, str | None]:
    if any(str(dep) in current_keys for dep in task.depends_on):
        return "blocked", "dependency_remains_current"
    if _uncertain(task):
        return "uncertain", "uncertain_task"
    return _priority_name(task), None


def _route_task(
    worker: _Worker,
    task: Task,
    shared: dict[str, int],
    policy: ScheduleHiringPolicy,
) -> tuple[bool, int, str]:
    """Apply one task to a copied worker/resource state."""
    if task.tile is None:
        return False, 0, "non_worker_task"
    target = (int(task.tile[0]), int(task.tile[1]))
    cost = abs(worker.position[0] - target[0]) + abs(worker.position[1] - target[1])

    if task.kind == "PLANT":
        crop = str(task.crop or "")
        if not crop or shared.get(f"seed:{crop}", 0) < 1:
            return False, 0, f"no_global_seeds:{crop}" if crop else "malformed_plant"
        shared[f"seed:{crop}"] -= 1
        # Planting is immediately paired with a water obligation.
        cost += 2
    else:
        required = task.required_item
        if required is not None:
            item = str(required)
            demand = max(1, int(task.quantity))
            carried = max(0, int(worker.inventory.get(item, 0)))
            missing = max(0, demand - carried)
            if missing:
                if shared.get(f"shed:{item}", 0) < missing:
                    return False, 0, f"shed_lacks_item:{item}"
                access_distance = min(
                    abs(worker.position[0] - access[0])
                    + abs(worker.position[1] - access[1])
                    + abs(access[0] - target[0])
                    + abs(access[1] - target[1])
                    for access in policy.shed_access_tiles
                )
                # Route to access, perform one or more pickup actions, then
                # the ordinary interaction below.
                cost = access_distance + ceil(missing / policy.pickup_batch)
                shared[f"shed:{item}"] -= missing
            cost += 1
            worker.inventory[item] = max(0, carried + missing - demand)
        else:
            cost += 1

    worker.position = target
    return True, cost, ""


def _evaluate(
    task_map: Mapping[str, Task],
    queue_owner: Mapping[str, int],
    positions: Sequence[tuple[int, int]],
    inventories: Sequence[Mapping[str, int]],
    future_actions: int,
    hires: int,
    obs: Mapping,
    seat: int,
    policy: ScheduleHiringPolicy,
    category_by_key: Mapping[str, str],
    queue_rank: Mapping[str, int],
) -> _Evaluation:
    all_positions = list(positions) + _spawn_positions(positions, hires, policy)
    workers = [_Worker(position, dict(inventories[index]) if index < len(inventories) else {},
                       future_actions)
               for index, position in enumerate(all_positions)]
    private = obs.get("private") or {}
    shared: dict[str, int] = {}
    for item, amount in ((private.get("shed") or {}).items()):
        shared[f"shed:{item}"] = max(0, int(amount))
    for crop, amount in ((private.get("seeds") or {}).items()):
        shared[f"seed:{crop}"] = max(0, int(amount))

    evaluation = _Evaluation(capacity=len(workers) * future_actions, workers=workers)
    details: dict[str, dict[str, Any]] = {}

    # Queued tasks retain their ownership.  Their route cost is included once,
    # and the remaining tasks are assigned greedily by least feasible extension.
    ordered = sorted((task for task in task_map.values()
                      if task.key in category_by_key), key=_task_sort)
    queued = sorted(((task, queue_owner[task.key]) for task in ordered
                     if task.key in queue_owner),
                    key=lambda pair: (pair[1], queue_rank.get(str(pair[0].key), 1 << 30),
                                      _task_sort(pair[0])))
    assigned: set[str] = set()

    def schedule_on(task: Task, worker_index: int, reason: str) -> bool:
        if worker_index < 0 or worker_index >= len(workers):
            details[task.key] = {"status": "blocked", "category": "blocked",
                                  "reason": "queue_worker_missing"}
            return False
        candidate = workers[worker_index].copy()
        local_shared = shared.copy()
        ok, cost, failure = _route_task(candidate, task, local_shared, policy)
        if not ok or cost > candidate.remaining:
            details[task.key] = {"status": "blocked", "category": category_by_key[task.key],
                                  "reason": failure or "capacity_exhausted",
                                  "worker_index": worker_index, "cost": cost}
            return False
        candidate.remaining -= cost
        candidate.workload += cost
        workers[worker_index] = candidate
        shared.clear()
        shared.update(local_shared)
        evaluation.workload += cost
        evaluation.required_workload += cost
        assigned.add(task.key)
        details[task.key] = {"status": "scheduled", "category": category_by_key[task.key],
                             "worker_index": worker_index, "cost": cost, "reason": reason}
        return True

    for task, owner in queued:
        if category_by_key.get(task.key) in {"optional", "uncertain", "blocked"}:
            continue
        schedule_on(task, owner, "persistent_queue")

    for task in ordered:
        category = category_by_key[task.key]
        if category in {"optional", "uncertain", "blocked"} or task.key in assigned:
            continue
        choices: list[tuple[int, int, _Worker, dict[str, int]]] = []
        attempted_costs: list[int] = []
        for index, worker in enumerate(workers):
            candidate = worker.copy()
            local_shared = shared.copy()
            ok, cost, failure = _route_task(candidate, task, local_shared, policy)
            if cost > 0:
                attempted_costs.append(cost)
            if ok and cost <= candidate.remaining:
                choices.append((cost, index, candidate, local_shared))
            elif task.key not in details:
                details[task.key] = {"status": "blocked", "category": category,
                                      "reason": failure or "capacity_exhausted",
                                      "cost": cost}
        if choices:
            cost, index, candidate, local_shared = min(choices, key=lambda row: (row[0], row[1]))
            evaluation.required_workload += cost
            candidate.remaining -= cost
            candidate.workload += cost
            workers[index] = candidate
            shared = local_shared
            evaluation.workload += cost
            assigned.add(task.key)
            details[task.key] = {"status": "scheduled", "category": category,
                                 "worker_index": index, "cost": cost,
                                 "reason": "least_extension"}
        elif attempted_costs:
            # Keep the estimate inspectable even when a route is too long for
            # the remaining future slots.  Resource failures have cost zero
            # and are intentionally not treated as useful workload.
            evaluation.required_workload += min(attempted_costs)

    evaluation.assigned = assigned
    evaluation.task_details = details
    return evaluation


def recommend_hires(
    obs: Mapping,
    seat: int,
    tasks: Sequence[Task] | None = None,
    *,
    current_tasks: Sequence[Task] | None = None,
    worker_count: int | None = None,
    current_worker_count: int | None = None,
    worker_positions: Sequence[Sequence[int]] | None = None,
    current_worker_positions: Sequence[Sequence[int]] | None = None,
    worker_inventories: Sequence[Mapping[str, int]] | None = None,
    current_inventories: Sequence[Mapping[str, int]] | None = None,
    scheduler_result: Any = None,
    persistent_scheduler_result: Any = None,
    scheduler_queues: Mapping[Any, Sequence[Task]] | None = None,
    persistent_queues: Mapping[Any, Sequence[Task]] | None = None,
    available_cash: float = 0,
    hires_today: int | None = None,
    hire_cost_mult: int = FARM_HAND_COST_MULT_DEFAULT,
    market_order_limit: int = 10,
    policy: ScheduleHiringPolicy | None = None,
) -> HiringRecommendation:
    """Recommend a bounded number of future ``["HIRE"]`` market orders.

    ``worker_positions``/``current_worker_positions`` are canonical ``(y,x)``
    pairs when explicitly supplied.  Omitting them derives engine ``[x,y]``
    positions from ``obs``.  ``wanted_hires`` is the minimum number needed to
    schedule all known non-optional useful work; affordability and the market
    order limit are applied afterward.
    """
    if not isinstance(obs, Mapping):
        raise TypeError("obs must be a mapping")
    if not isinstance(seat, int) or isinstance(seat, bool) or seat < 0:
        raise ValueError("seat must be a non-negative integer")
    farms = obs.get("farms") or []
    if seat >= len(farms):
        raise ValueError(f"invalid seat {seat!r}")
    if policy is None:
        policy = ScheduleHiringPolicy()
    if tasks is not None and current_tasks is not None:
        raise ValueError("pass only one of tasks and current_tasks")
    task_values = list(tasks if tasks is not None else (current_tasks or ()))
    if worker_count is not None and current_worker_count is not None:
        raise ValueError("pass only one worker count")
    inferred_count = 1 + len(farms[seat].get("hands") or [])
    count = worker_count if worker_count is not None else current_worker_count
    count = inferred_count if count is None else int(count)
    if count < 1:
        raise ValueError("worker_count must be positive")
    explicit_positions = (worker_positions if worker_positions is not None
                          else current_worker_positions)
    explicit_inventories = (worker_inventories if worker_inventories is not None
                            else current_inventories)
    positions = _normalise_positions(obs, seat, count, explicit_positions)
    inventories = _normalise_inventories(obs, count, explicit_inventories)
    selected_scheduler = scheduler_result if scheduler_result is not None else persistent_scheduler_result
    queues = (_queue_mapping(selected_scheduler)
              if selected_scheduler is not None else
              (scheduler_queues if scheduler_queues is not None else persistent_queues))
    task_map, queue_owner, queue_rank, _ = _copy_task_map(task_values, queues)
    current_keys = {
        key for key, task in task_map.items()
        if task.kind not in _MARKET_KINDS and task.tile is not None
        and task.kind in _WORKER_KINDS
    }

    category_counts = {name: 0 for name in
                       ("maintenance", "urgent_maintenance", "productive", "manager", "logistics",
                        "optional", "uncertain", "blocked", "excluded_market",
                        "excluded_non_worker")}
    category_by_key: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    useful: dict[str, Task] = {}

    # Report exclusions from the input separately; task_map only contains
    # worker candidates and queue entries for deterministic deduplication.
    seen_input: set[str] = set()
    for task in task_values:
        if not isinstance(task, Task):
            raise TypeError(f"tasks must contain Task values, got {task!r}")
        if str(task.key) in seen_input:
            continue
        seen_input.add(str(task.key))
        # A duplicate key is represented by the single selected task below;
        # this also prevents a market/non-worker duplicate from inflating the
        # exclusion counts when a worker task wins deduplication.
        if task_map.get(str(task.key)) != task:
            continue
        if task.kind in _MARKET_KINDS:
            category_counts["excluded_market"] += 1
            diagnostics.append({"task_key": str(task.key), "kind": task.kind,
                                "status": "excluded", "category": "excluded_market",
                                "reason": "market_task"})
        elif task.tile is None or task.kind not in _WORKER_KINDS:
            category_counts["excluded_non_worker"] += 1
            diagnostics.append({"task_key": str(task.key), "kind": task.kind,
                                "status": "excluded", "category": "excluded_non_worker",
                                "reason": "non_worker_task"})

    for key, task in sorted(task_map.items(), key=lambda item: _task_sort(item[1])):
        if task.kind in _MARKET_KINDS or task.tile is None or task.kind not in _WORKER_KINDS:
            continue
        category, reason = _task_category(task, current_keys)
        category_by_key[key] = category
        category_counts[category] = category_counts.get(category, 0) + 1
        if category == "maintenance":
            category_counts["urgent_maintenance"] += 1
        diagnostic = {"task_key": key, "kind": task.kind, "priority": _priority_name(task),
                      "category": category, "status": "candidate",
                      "urgency": "urgent" if category == "maintenance" else "normal"}
        if reason:
            diagnostic.update(status="blocked", reason=reason)
        elif category in {"optional", "uncertain"}:
            diagnostic.update(status="ignored", reason=f"{category}_work_never_justifies_hire")
        else:
            useful[key] = task
        diagnostics.append(diagnostic)

    future = _future_actions(obs, policy)
    baseline = _evaluate(task_map, queue_owner, positions, inventories, future, 0,
                         obs, seat, policy, category_by_key, queue_rank)
    useful_keys = set(useful)
    # Resource-infeasible useful tasks are not a reason to hire.  Evaluate
    # once, then search the small monotone worker-count space for a feasible
    # assignment of all remaining useful work.
    blocked_useful = useful_keys - baseline.assigned
    for key in sorted(blocked_useful):
        detail = baseline.task_details.get(key, {})
        if detail.get("reason", "").startswith(("no_global_seeds", "shed_lacks_item")):
            category_counts["blocked"] += 1
            category_by_key[key] = "blocked"
            useful.pop(key, None)
            for item in diagnostics:
                if item.get("task_key") == key:
                    item.update(category="blocked", status="blocked",
                                reason=detail.get("reason"))
    useful_keys = set(useful)

    wanted = 0
    chosen_eval = baseline
    if future > 0 and useful_keys:
        for extra in range(0, min(policy.max_hires, len(useful_keys)) + 1):
            evaluation = _evaluate(task_map, queue_owner, positions, inventories, future,
                                   extra, obs, seat, policy, category_by_key, queue_rank)
            if useful_keys.issubset(evaluation.assigned):
                wanted = extra
                chosen_eval = evaluation
                break
            chosen_eval = evaluation
        else:
            # More workers cannot make one indivisible route fit into a
            # one-turn future slot.  In that case hiring is not justified;
            # report the capacity miss instead of emitting a doomed HIRE.
            wanted = 0
    elif useful_keys:
        # Explicitly explain the hour-23 and terminal-step boundary.
        wanted = 0

    rejection: list[dict[str, Any]] = []
    if useful_keys and future <= 0:
        rejection.append({"reason": "no_future_worker_action_before_reset_or_terminal",
                          "future_worker_actions": future,
                          "hour": int(obs.get("hour", 0)),
                          "step": int(obs.get("step", 0))})
    if not useful_keys and category_counts["optional"] + category_counts["uncertain"] + category_counts["blocked"]:
        rejection.append({"reason": "optional_uncertain_or_blocked_only",
                          "optional": category_counts["optional"],
                          "uncertain": category_counts["uncertain"],
                          "blocked": category_counts["blocked"]})
    if useful_keys and not useful_keys.issubset(chosen_eval.assigned):
        rejection.append({"reason": "useful_work_exceeds_per_worker_future_capacity",
                          "unassigned": sorted(useful_keys - chosen_eval.assigned)})

    # Add final schedule details in stable key order.
    for key in sorted(useful_keys):
        if key in chosen_eval.task_details:
            for item in diagnostics:
                if item.get("task_key") == key and item.get("status") in {"candidate", "blocked"}:
                    item.update(chosen_eval.task_details[key])
                    item["task_key"] = key
                    break

    already = int(farms[seat].get("hires_today", 0) if hires_today is None else hires_today)
    cash = float(available_cash)
    affordable = 0
    costs: list[int] = []
    for index in range(max(0, wanted)):
        cost = int(hire_cost(already + index, int(hire_cost_mult)))
        if cash + _EPSILON < cost:
            break
        cash -= cost
        affordable += 1
        costs.append(cost)
    submittable = min(wanted, affordable, max(0, int(market_order_limit)))
    if affordable < wanted:
        rejection.append({"reason": "insufficient_cash_for_marginal_hire",
                          "wanted": wanted, "affordable": affordable,
                          "available_cash": available_cash})
    if submittable < min(wanted, affordable):
        rejection.append({"reason": "market_order_limit", "wanted": wanted,
                          "affordable": affordable,
                          "market_order_limit": int(market_order_limit)})

    # Workload is the cost of the best evaluation we could make with the
    # recommended workforce; capacity includes only future action slots.
    predicted_capacity = baseline.capacity
    predicted_capacity_with = chosen_eval.capacity
    return HiringRecommendation(
        wanted_hires=int(wanted), affordable_hires=int(affordable),
        submittable_hires=int(submittable),
        predicted_workload=int(chosen_eval.required_workload),
        predicted_capacity=int(predicted_capacity),
        predicted_capacity_with_hires=int(predicted_capacity_with),
        future_worker_actions=int(future), category_counts=dict(category_counts),
        diagnostics=tuple(diagnostics), rejection_diagnostics=tuple(rejection),
        hire_costs=tuple(costs), worker_count=count,
    )


def recommend(*args: Any, **kwargs: Any) -> HiringRecommendation:
    """Backward-friendly alias kept out of ``__all__`` for integrations."""
    return recommend_hires(*args, **kwargs)
