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
from math import ceil, isfinite
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

    When ``economic_repair`` is enabled, the benefit fields are the explicit
    value model: maintenance is survival work, productive work is valuable but
    less urgent, and manager/logistics work is deliberately modest.  Values
    are per directly completed task, not per downstream consequence.  A task
    whose current dependency is also completed receives no second downstream
    credit, and a PLANT's required WATER follow-up is already included in its
    route cost.  Optional and uncertain work never justifies a hire.  These
    conservative defaults keep cash availability from turning an arbitrary
    manager request into a hire; callers may change the assumptions explicitly
    for an experiment.
    """

    shed_access_tiles: tuple[tuple[int, int], ...] = SHED_ACCESS_TILES
    pickup_batch: int = 5
    total_days: int = 30
    hours_per_day: int = 24
    final_actionable_step: int = 30 * 24 - 2
    max_hires: int = 64
    maintenance_benefit: float = 100.0
    productive_benefit: float = 8.0
    manager_benefit: float = 3.0
    logistics_benefit: float = 1.0

    def __post_init__(self) -> None:
        if not self.shed_access_tiles:
            raise ValueError("shed_access_tiles must not be empty")
        if self.pickup_batch < 1:
            raise ValueError("pickup_batch must be positive")
        if self.total_days < 1 or self.hours_per_day < 1:
            raise ValueError("total_days and hours_per_day must be positive")
        if self.max_hires < 0:
            raise ValueError("max_hires must be non-negative")
        for name in (
                "maintenance_benefit", "productive_benefit",
                "manager_benefit", "logistics_benefit"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

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


@dataclass
class _EconomicEvaluation:
    """One independent candidate schedule used only by the repair path."""

    capacity: int
    workload: int = 0
    assigned: set[str] = field(default_factory=set)
    task_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    workers: list[_Worker] = field(default_factory=list)
    remaining_resources: dict[str, int] = field(default_factory=dict)


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


def _economic_benefit(
    task: Task,
    category: str,
    current_keys: set[str],
    assigned: set[str],
    task_map: Mapping[str, Task],
    policy: ScheduleHiringPolicy,
) -> tuple[float, str, str | None]:
    """Return the conservative direct credit for one completed task."""
    if any(str(dep) in current_keys for dep in task.depends_on):
        return 0.0, "downstream", "dependency_credit_already_attributed"
    if task.kind == "WATER":
        # PLANT already reserves two interactions: PLANT and its required
        # same-tile WATER follow-up.  A separately supplied WATER task must
        # not make that same production opportunity look twice as valuable.
        for key in assigned:
            parent = task_map.get(key)
            if parent is not None and parent.kind == "PLANT" \
                    and parent.tile == task.tile:
                return 0.0, "downstream", "plant_water_follow_up_already_included"
    if category == "maintenance":
        return float(policy.maintenance_benefit), "survival", None
    if category == "productive":
        return float(policy.productive_benefit), "productive", None
    if category == "manager":
        return float(policy.manager_benefit), "manager", None
    if category == "logistics":
        return float(policy.logistics_benefit), "logistics", None
    return 0.0, category, "category_not_economically_valued"


def _economic_benefits(
    evaluation: _EconomicEvaluation,
    task_map: Mapping[str, Task],
    category_by_key: Mapping[str, str],
    current_keys: set[str],
    policy: ScheduleHiringPolicy,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in sorted(evaluation.assigned):
        task = task_map[key]
        value, benefit_class, reason = _economic_benefit(
            task, category_by_key[key], current_keys, evaluation.assigned,
            task_map, policy)
        values[key] = value
        detail = evaluation.task_details.get(key)
        if detail is not None:
            detail["benefit"] = value
            detail["benefit_class"] = benefit_class
            if reason:
                detail["benefit_reason"] = reason
    return values


def _economic_evaluate(
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
) -> _EconomicEvaluation:
    """Evaluate one worker-count candidate from a fresh resource ledger.

    This deliberately does not call the legacy evaluator.  The repair needs
    dependency readiness and resource failures to remain local to each
    candidate, so an impossible task cannot poison an otherwise useful
    partial schedule.
    """
    del seat  # The pure estimate only needs the already-normalised inputs.
    all_positions = list(positions) + _spawn_positions(positions, hires, policy)
    workers = [
        _Worker(
            position,
            dict(inventories[index]) if index < len(inventories) else {},
            future_actions,
        )
        for index, position in enumerate(all_positions)
    ]
    private = obs.get("private") or {}
    shared: dict[str, int] = {}
    for item, amount in (private.get("shed") or {}).items():
        shared[f"shed:{item}"] = max(0, int(amount))
    for crop, amount in (private.get("seeds") or {}).items():
        shared[f"seed:{crop}"] = max(0, int(amount))

    evaluation = _EconomicEvaluation(
        capacity=len(workers) * future_actions,
        workers=workers,
    )
    worker_tasks = {
        key: task for key, task in task_map.items()
        if key in category_by_key
    }
    current_worker_keys = set(worker_tasks)
    pending = {
        key: task for key, task in worker_tasks.items()
        if category_by_key[key] not in {"optional", "uncertain"}
    }
    assigned: set[str] = set()
    current_hour = int(obs.get("hour", 0))

    def dependencies(task: Task) -> tuple[str, ...]:
        return tuple(sorted(
            str(dep) for dep in task.depends_on
            if str(dep) in current_worker_keys
        ))

    def deadline_slots(task: Task) -> int | None:
        if task.deadline_hour is None:
            return None
        return max(0, int(task.deadline_hour) - current_hour)

    def mark_blocked(
        task: Task,
        reason: str,
        *,
        worker_index: int | None = None,
        cost: int = 0,
        **extra: Any,
    ) -> None:
        key = str(task.key)
        detail: dict[str, Any] = {
            "status": "blocked",
            "category": category_by_key[key],
            "reason": reason,
            "cost": int(cost),
        }
        if worker_index is not None:
            detail["worker_index"] = int(worker_index)
        detail.update(extra)
        evaluation.task_details[key] = detail

    def try_route(
        worker_index: int,
        task: Task,
    ) -> tuple[bool, int, str, _Worker, dict[str, int]]:
        candidate = workers[worker_index].copy()
        local_shared = shared.copy()
        ok, cost, failure = _route_task(candidate, task, local_shared, policy)
        cost = int(cost)
        if not ok:
            return False, cost, failure, candidate, local_shared
        if cost > workers[worker_index].remaining:
            return False, cost, "capacity_exhausted", candidate, local_shared
        allowed = deadline_slots(task)
        if allowed is not None and cost > allowed:
            return False, cost, "deadline_exceeded", candidate, local_shared
        return True, cost, "", candidate, local_shared

    def commit(
        task: Task,
        worker_index: int,
        cost: int,
        candidate: _Worker,
        local_shared: Mapping[str, int],
        reason: str,
    ) -> None:
        key = str(task.key)
        candidate.remaining -= cost
        candidate.workload += cost
        workers[worker_index] = candidate
        shared.clear()
        shared.update(local_shared)
        evaluation.workload += cost
        assigned.add(key)
        detail: dict[str, Any] = {
            "status": "scheduled",
            "category": category_by_key[key],
            "worker_index": int(worker_index),
            "cost": int(cost),
            "reason": reason,
        }
        if task.kind == "PLANT":
            detail["included_follow_up"] = {
                "kind": "WATER",
                "cost": 1,
                "reason": "required_plant_water_follow_up",
            }
        evaluation.task_details[key] = detail

    def failure_priority(reason: str) -> tuple[int, str]:
        if reason.startswith(("no_global_seeds", "shed_lacks_item")):
            return 0, reason
        if reason == "dependency_unresolved":
            return 1, reason
        if reason == "deadline_exceeded":
            return 2, reason
        if reason == "capacity_exhausted":
            return 3, reason
        return 4, reason

    def queue_sort(task: Task) -> tuple:
        key = str(task.key)
        if key in queue_owner:
            return (0, int(queue_owner[key]), queue_rank.get(key, 1 << 30),
                    _task_sort(task))
        return (1, _task_sort(task))

    while pending:
        ready = [
            task for task in pending.values()
            if all(dep in assigned or dep not in current_worker_keys
                   for dep in dependencies(task))
        ]
        if not ready:
            for task in sorted(pending.values(), key=_task_sort):
                missing = [dep for dep in dependencies(task) if dep not in assigned]
                mark_blocked(
                    task,
                    "dependency_unresolved",
                    dependencies=missing,
                )
            break

        task = min(ready, key=queue_sort)
        key = str(task.key)
        pending.pop(key, None)
        owner = queue_owner.get(key)
        if owner is not None:
            if owner < 0 or owner >= len(workers):
                mark_blocked(task, "queue_worker_missing", worker_index=owner)
                continue
            ok, cost, failure, candidate, local_shared = try_route(owner, task)
            if not ok:
                mark_blocked(task, failure or "capacity_exhausted",
                             worker_index=owner, cost=cost)
                continue
            commit(task, owner, cost, candidate, local_shared,
                   "persistent_queue")
            continue

        choices: list[tuple[int, int, _Worker, dict[str, int]]] = []
        failures: list[tuple[int, int, str]] = []
        for worker_index in range(len(workers)):
            ok, cost, failure, candidate, local_shared = try_route(
                worker_index, task)
            if ok:
                choices.append((cost, worker_index, candidate, local_shared))
            else:
                failures.append((worker_index, cost, failure or "capacity_exhausted"))
        if choices:
            cost, worker_index, candidate, local_shared = min(
                choices, key=lambda row: (row[0], row[1]))
            commit(task, worker_index, cost, candidate, local_shared,
                   "least_extension")
            continue

        if failures:
            _worker_index, cost, failure = min(
                failures, key=lambda row: (failure_priority(row[2]), row[0], row[1]))
            mark_blocked(task, failure, cost=cost)
        else:
            mark_blocked(task, "capacity_exhausted")

    evaluation.assigned = assigned
    evaluation.remaining_resources = {
        key: int(value) for key, value in sorted(shared.items())
    }
    return evaluation


def _economic_residuals(
    evaluation: _EconomicEvaluation,
    task_map: Mapping[str, Task],
    category_by_key: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    deadlines: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    for key, detail in sorted(evaluation.task_details.items()):
        if detail.get("status") != "blocked":
            continue
        task = task_map.get(key)
        if task is None or category_by_key.get(key) in {"optional", "uncertain"}:
            continue
        entry = {
            "task_key": key,
            "kind": task.kind,
            "category": category_by_key[key],
            "reason": detail.get("reason", "blocked"),
        }
        if "dependencies" in detail:
            entry["dependencies"] = list(detail["dependencies"])
        blockers.append(dict(entry))
        reason = str(detail.get("reason", ""))
        if task.deadline_hour is not None or reason == "deadline_exceeded":
            deadline_entry = dict(entry)
            deadline_entry["deadline_hour"] = task.deadline_hour
            deadlines.append(deadline_entry)
        if reason.startswith(("no_global_seeds", "shed_lacks_item")):
            resource_entry = dict(entry)
            if task.kind == "PLANT":
                resource_entry["resource"] = f"seed:{task.crop or ''}"
            else:
                resource_entry["resource"] = f"shed:{task.required_item or ''}"
            resources.append(resource_entry)
    return blockers, deadlines, resources


def _economic_resource_view(resources: Mapping[str, int]) -> dict[str, dict[str, int]]:
    view: dict[str, dict[str, int]] = {"shed": {}, "seeds": {}}
    for key, value in sorted(resources.items()):
        if key.startswith("shed:"):
            view["shed"][key.removeprefix("shed:")] = int(value)
        elif key.startswith("seed:"):
            view["seeds"][key.removeprefix("seed:")] = int(value)
    return view


def _recommend_economic_repair(
    obs: Mapping,
    seat: int,
    tasks: Sequence[Task] | None,
    *,
    current_tasks: Sequence[Task] | None,
    worker_count: int | None,
    current_worker_count: int | None,
    worker_positions: Sequence[Sequence[int]] | None,
    current_worker_positions: Sequence[Sequence[int]] | None,
    worker_inventories: Sequence[Mapping[str, int]] | None,
    current_inventories: Sequence[Mapping[str, int]] | None,
    scheduler_result: Any,
    persistent_scheduler_result: Any,
    scheduler_queues: Mapping[Any, Sequence[Task]] | None,
    persistent_queues: Mapping[Any, Sequence[Task]] | None,
    available_cash: float,
    hires_today: int | None,
    hire_cost_mult: int,
    market_order_limit: int,
    policy: ScheduleHiringPolicy | None,
) -> HiringRecommendation:
    """Opt-in economic repair; the ordinary path remains below unchanged."""
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
    selected_scheduler = (scheduler_result if scheduler_result is not None
                          else persistent_scheduler_result)
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
    seen_input: set[str] = set()
    for task in task_values:
        if not isinstance(task, Task):
            raise TypeError(f"tasks must contain Task values, got {task!r}")
        key = str(task.key)
        if key in seen_input:
            continue
        seen_input.add(key)
        if task_map.get(key) != task:
            continue
        if task.kind in _MARKET_KINDS:
            category_counts["excluded_market"] += 1
            diagnostics.append({"task_key": key, "kind": task.kind,
                                "status": "excluded", "category": "excluded_market",
                                "reason": "market_task"})
        elif task.tile is None or task.kind not in _WORKER_KINDS:
            category_counts["excluded_non_worker"] += 1
            diagnostics.append({"task_key": key, "kind": task.kind,
                                "status": "excluded", "category": "excluded_non_worker",
                                "reason": "non_worker_task"})

    for key, task in sorted(task_map.items(), key=lambda item: _task_sort(item[1])):
        if task.kind in _MARKET_KINDS or task.tile is None \
                or task.kind not in _WORKER_KINDS:
            continue
        raw_category = _priority_name(task)
        category_by_key[key] = raw_category
        category_counts[raw_category] = category_counts.get(raw_category, 0) + 1
        if raw_category == "maintenance":
            category_counts["urgent_maintenance"] += 1
        dependencies = [str(dep) for dep in task.depends_on
                        if str(dep) in current_keys]
        diagnostic: dict[str, Any] = {
            "task_key": key,
            "kind": task.kind,
            "priority": raw_category,
            "category": raw_category,
            "status": "candidate",
            "urgency": "urgent" if raw_category == "maintenance" else "normal",
        }
        if dependencies:
            diagnostic["dependencies"] = dependencies
        if raw_category in {"optional", "uncertain"}:
            diagnostic.update(
                status="ignored",
                reason=f"{raw_category}_work_never_justifies_hire")
        else:
            useful[key] = task
        diagnostics.append(diagnostic)

    future = _future_actions(obs, policy)
    useful_keys = set(useful)
    baseline = _economic_evaluate(
        task_map, queue_owner, positions, inventories, future, 0,
        obs, seat, policy, category_by_key, queue_rank)
    max_candidates = min(policy.max_hires, len(useful_keys))
    already = int(farms[seat].get("hires_today", 0) if hires_today is None else hires_today)
    already = max(0, already)
    candidate_costs = [
        int(hire_cost(already + index, int(hire_cost_mult)))
        for index in range(max_candidates)
    ]
    prefix_costs: list[int] = []
    running_cost = 0
    for cost in candidate_costs:
        running_cost += cost
        prefix_costs.append(running_cost)

    evaluations: list[_EconomicEvaluation] = []
    benefits: list[dict[str, float]] = []
    outcomes: list[dict[str, Any]] = []
    baseline_benefits = _economic_benefits(
        baseline, task_map, category_by_key, current_keys, policy)
    baseline_total_benefit = sum(baseline_benefits.values())
    previous_eval: _EconomicEvaluation | None = None
    previous_total_benefit = baseline_total_benefit
    for candidate_hires in range(max_candidates + 1):
        evaluation = (baseline if candidate_hires == 0 else _economic_evaluate(
            task_map, queue_owner, positions, inventories, future,
            candidate_hires, obs, seat, policy, category_by_key, queue_rank))
        if candidate_hires == 0:
            candidate_benefits = baseline_benefits
        else:
            candidate_benefits = _economic_benefits(
                evaluation, task_map, category_by_key, current_keys, policy)
        evaluations.append(evaluation)
        benefits.append(candidate_benefits)
        total_benefit = sum(candidate_benefits.values())
        newly_from_zero = sorted(evaluation.assigned - baseline.assigned)
        prior_assigned = baseline.assigned if previous_eval is None \
            else previous_eval.assigned
        marginal_newly = sorted(evaluation.assigned - prior_assigned)
        lost_from_zero = sorted(baseline.assigned - evaluation.assigned)
        previous_benefit = previous_total_benefit
        marginal_benefit = total_benefit - previous_benefit
        benefit_delta = total_benefit - baseline_total_benefit
        cumulative_cost = (prefix_costs[candidate_hires - 1]
                           if candidate_hires else 0)
        if candidate_hires == 0:
            economic_reason = "zero_hire_baseline"
            economically_viable = False
        elif future <= 0:
            economic_reason = "no_future_worker_action_before_reset_or_terminal"
            economically_viable = False
        elif not newly_from_zero:
            economic_reason = "no_added_feasible_work"
            economically_viable = False
        elif benefit_delta <= _EPSILON:
            economic_reason = "no_positive_new_benefit"
            economically_viable = False
        elif benefit_delta + _EPSILON < cumulative_cost:
            economic_reason = "marginal_benefit_below_cost"
            economically_viable = False
        else:
            economic_reason = "benefit_exceeds_cumulative_hire_cost"
            economically_viable = True
        blockers, deadlines, resources = _economic_residuals(
            evaluation, task_map, category_by_key)
        newly_categories = sorted({category_by_key[key] for key in newly_from_zero})
        marginal_categories = sorted({category_by_key[key] for key in marginal_newly})
        outcome: dict[str, Any] = {
            "event": "candidate_outcome",
            "candidate_hires": candidate_hires,
            "worker_count": count + candidate_hires,
            "marginal_index": candidate_hires - 1 if candidate_hires else None,
            "marginal_cost": (candidate_costs[candidate_hires - 1]
                              if candidate_hires else 0),
            "cumulative_cost": cumulative_cost,
            "baseline_completed_task_keys": sorted(baseline.assigned),
            "baseline_benefit": baseline_total_benefit,
            "completed_task_keys": sorted(evaluation.assigned),
            "newly_completed_task_keys": newly_from_zero,
            "newly_completed_categories": newly_categories,
            "marginal_newly_completed_task_keys": marginal_newly,
            "marginal_newly_completed_categories": marginal_categories,
            "lost_baseline_task_keys": lost_from_zero,
            "benefit": benefit_delta,
            "total_benefit": total_benefit,
            "marginal_benefit": marginal_benefit,
            "remaining_capacity": sum(worker.remaining for worker in evaluation.workers),
            "remaining_capacity_by_worker": [
                worker.remaining for worker in evaluation.workers
            ],
            "capacity": evaluation.capacity,
            "economically_viable": economically_viable,
            "accepted": False,
            "accepted_reason": None,
            "rejected_reason": economic_reason,
            "residual_blockers": blockers,
            "residual_deadlines": deadlines,
            "residual_resources": resources,
            "remaining_resources": _economic_resource_view(
                evaluation.remaining_resources),
        }
        outcomes.append(outcome)
        previous_eval = evaluation
        previous_total_benefit = total_benefit

    viable = [
        outcome for outcome in outcomes
        if outcome["candidate_hires"] > 0 and outcome["economically_viable"]
    ]

    cash = float(available_cash)
    if not isfinite(cash):
        cash = 0.0
    cash = max(0.0, cash)
    for outcome in outcomes:
        candidate_hires = int(outcome["candidate_hires"])
        remaining_cash = cash
        affordable_count = 0
        for cost in candidate_costs[:candidate_hires]:
            if remaining_cash + _EPSILON < cost:
                break
            remaining_cash -= cost
            affordable_count += 1
        outcome["affordable"] = affordable_count >= candidate_hires
        outcome["affordable_hires"] = affordable_count
        outcome["market_order_limit"] = max(0, int(market_order_limit))
        outcome["submission_possible"] = (
            affordable_count >= candidate_hires
            and candidate_hires <= max(0, int(market_order_limit)))

    affordable_viable = [outcome for outcome in viable if outcome["affordable"]]
    selection_pool = affordable_viable or viable
    if selection_pool:
        selected_outcome = max(
            selection_pool,
            key=lambda outcome: (
                float(outcome["benefit"]) - int(outcome["cumulative_cost"]),
                float(outcome["benefit"]),
                -int(outcome["candidate_hires"]),
            ),
        )
        selected_hires = int(selected_outcome["candidate_hires"])
    else:
        selected_outcome = None
        selected_hires = 0

    for outcome in outcomes:
        candidate_hires = int(outcome["candidate_hires"])
        if candidate_hires == selected_hires and selected_hires > 0:
            outcome["accepted"] = True
            outcome["accepted_reason"] = "best_positive_net_benefit"
            outcome["rejected_reason"] = None
        elif candidate_hires == 0:
            outcome["accepted_reason"] = "baseline_selected_no_hire" \
                if selected_hires == 0 else None
        elif outcome["economically_viable"]:
            outcome["rejected_reason"] = "not_best_net_benefit"

    affordable = 0
    costs: list[int] = []
    remaining_cash = cash
    for cost in candidate_costs[:selected_hires]:
        if remaining_cash + _EPSILON < cost:
            break
        remaining_cash -= cost
        affordable += 1
        costs.append(cost)
    submittable = min(selected_hires, affordable, max(0, int(market_order_limit)))
    actual_index = min(submittable, len(evaluations) - 1)
    actual_eval = evaluations[actual_index]
    actual_benefits = benefits[actual_index]
    for outcome in outcomes:
        candidate_hires = int(outcome["candidate_hires"])
        if candidate_hires != selected_hires:
            outcome["submission_accepted"] = False
            continue
        outcome["submission_accepted"] = submittable == selected_hires
        if outcome["submission_accepted"]:
            outcome["submission_rejected_reason"] = None
        elif affordable < selected_hires:
            outcome["submission_rejected_reason"] = "insufficient_cash_for_marginal_hire"
        else:
            outcome["submission_rejected_reason"] = "market_order_limit"

    blocked_actual = sorted(useful_keys - actual_eval.assigned)
    category_counts["blocked"] += len(blocked_actual)
    for key in sorted(useful_keys):
        detail = actual_eval.task_details.get(key)
        selected_detail = (evaluations[selected_hires].task_details.get(key)
                           if selected_hires < len(evaluations) else None)
        item = next((item for item in diagnostics
                     if item.get("task_key") == key), None)
        if item is None:
            continue
        item["baseline_status"] = (
            "scheduled" if key in baseline.assigned else "blocked")
        item["economic_benefit"] = actual_benefits.get(key, 0.0)
        if detail is not None:
            item.update(detail)
        else:
            item.update(status="blocked", reason="not_evaluated", cost=0)
        if key in blocked_actual:
            item["original_category"] = item.get("category", category_by_key[key])
            item["category"] = "blocked"
        if selected_hires > submittable and selected_detail is not None \
                and selected_detail.get("status") == "scheduled" \
                and key not in actual_eval.assigned:
            item["projected_status"] = "scheduled"
            item["status"] = "rejected_not_submittable"
            item["reason"] = "hire_count_not_affordable_or_capped"

    rejection: list[dict[str, Any]] = []
    if useful_keys and future <= 0:
        rejection.append({"reason": "no_future_worker_action_before_reset_or_terminal",
                          "future_worker_actions": future,
                          "hour": int(obs.get("hour", 0)),
                          "step": int(obs.get("step", 0))})
    if not useful_keys and category_counts["optional"] + category_counts["uncertain"]:
        rejection.append({"reason": "optional_uncertain_or_blocked_only",
                          "optional": category_counts["optional"],
                          "uncertain": category_counts["uncertain"],
                          "blocked": category_counts["blocked"]})
    if useful_keys and selected_hires == 0:
        if baseline.assigned.issuperset(useful_keys):
            rejection.append({"reason": "no_added_feasible_work",
                              "baseline_completed_task_keys": sorted(baseline.assigned)})
        elif not viable:
            rejection.append({"reason": "marginal_benefit_below_cost",
                              "useful_task_keys": sorted(useful_keys)})
    if affordable < selected_hires:
        rejection.append({"reason": "insufficient_cash_for_marginal_hire",
                          "wanted": selected_hires, "affordable": affordable,
                          "available_cash": available_cash})
    if submittable < min(selected_hires, affordable):
        rejection.append({"reason": "market_order_limit", "wanted": selected_hires,
                          "affordable": affordable,
                          "market_order_limit": int(market_order_limit)})
    for outcome in outcomes[1:]:
        if not outcome["accepted"]:
            rejection.append({
                "reason": outcome["rejected_reason"] or "not_selected",
                "candidate_hires": outcome["candidate_hires"],
                "marginal_cost": outcome["marginal_cost"],
                "benefit": outcome["benefit"],
                "newly_completed_task_keys": list(
                    outcome["newly_completed_task_keys"]),
                "residual_blockers": list(outcome["residual_blockers"]),
                "residual_deadlines": list(outcome["residual_deadlines"]),
                "residual_resources": list(outcome["residual_resources"]),
            })

    diagnostics.append({
        "event": "economic_repair",
        "baseline_completed_task_keys": sorted(baseline.assigned),
        "baseline_benefit": baseline_total_benefit,
        "selected_hires": selected_hires,
        "submitted_hires": submittable,
        "selected_candidate_benefit": (
            float(selected_outcome["benefit"]) if selected_outcome else 0.0),
        "candidate_count": len(outcomes),
        "assumptions": {
            "maintenance_benefit": policy.maintenance_benefit,
            "productive_benefit": policy.productive_benefit,
            "manager_benefit": policy.manager_benefit,
            "logistics_benefit": policy.logistics_benefit,
            "optional_and_uncertain_ignored": True,
            "downstream_credit_once": True,
            "plant_includes_water_follow_up": True,
        },
    })
    diagnostics.extend(outcomes)

    return HiringRecommendation(
        wanted_hires=int(selected_hires),
        affordable_hires=int(affordable),
        submittable_hires=int(submittable),
        predicted_workload=int(actual_eval.workload),
        predicted_capacity=int(baseline.capacity),
        predicted_capacity_with_hires=int(actual_eval.capacity),
        future_worker_actions=int(future),
        category_counts=dict(category_counts),
        diagnostics=tuple(diagnostics),
        rejection_diagnostics=tuple(rejection),
        hire_costs=tuple(costs),
        worker_count=count,
    )


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
    economic_repair: bool = False,
) -> HiringRecommendation:
    """Recommend a bounded number of future ``["HIRE"]`` market orders.

    ``worker_positions``/``current_worker_positions`` are canonical ``(y,x)``
    pairs when explicitly supplied.  Omitting them derives engine ``[x,y]``
    positions from ``obs``.  ``wanted_hires`` is the minimum number needed to
    schedule all known non-optional useful work; affordability and the market
    order limit are applied afterward.  With ``economic_repair=True``, the
    opt-in path instead selects the best positive net-benefit candidate and
    reports every candidate's independent schedule and residual blockers.
    """
    if economic_repair:
        return _recommend_economic_repair(
            obs, seat, tasks,
            current_tasks=current_tasks,
            worker_count=worker_count,
            current_worker_count=current_worker_count,
            worker_positions=worker_positions,
            current_worker_positions=current_worker_positions,
            worker_inventories=worker_inventories,
            current_inventories=current_inventories,
            scheduler_result=scheduler_result,
            persistent_scheduler_result=persistent_scheduler_result,
            scheduler_queues=scheduler_queues,
            persistent_queues=persistent_queues,
            available_cash=available_cash,
            hires_today=hires_today,
            hire_cost_mult=hire_cost_mult,
            market_order_limit=market_order_limit,
            policy=policy,
        )
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
