"""Stateful V0 executor agent (issue #1 sections 7-9).

One class, `ExecutorAgent`, closes the loop every primitive turn:

- once per new day: finalize the previous day's realized labor from the
  observed ``hires_today`` progression (exact Fibonacci hire cost), call the
  injected `PlanProvider` exactly once with that feedback, and mechanically
  project the requested plan via `project_plan`;
- every turn: regenerate the task set from the actual observation
  (`generate_tasks`), dispatch workers with the greedy foreman
  (`run_foreman`), and emit a bounded deterministic market queue:
  sells in the active four-hour bin only (clipped to actually available shed
  inventory via `clip_sell`, remainder carried within the bin), workload
  hiring, and exact-shortage BUY_SEED / BUY_PRODUCT / BUY_ANIMAL / BUY_LAND
  orders implied by the active task generator;
- hard survival guardrails keep existing animals ahead of discretionary
  expansion: current-day feed is protected from WHEAT sells, starvation
  boundary FEED work preempts non-survival tile work, feed-shortage purchases
  execute before discretionary buys, current survival pauses all expansion,
  and prior-day work debt pauses animal/housing commitments but not empty land;
- end-of-day work debt is measured from tasks still requiring work after the
  final primitive action, so temporary dependency/travel waiting that resolves
  during the day is not mislabeled as unfinished work;
- accumulate JSON-serializable per-day/game diagnostics distinguishing
  requested vs feasible vs achieved vs submitted vs observed completion;
- on any runtime failure in safe mode (default), return a legal-shaped all
  PASS action sized to the current hands and record the error. `strict=True`
  re-raises instead for tests/debugging.

No opponent private state is read; only ``obs["farms"][seat]`` and own
``obs["private"]`` are consumed. The manager is injectable; a checkpoint is
only loaded when an explicit path is supplied (never fabricated).
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.tasks import Priority
from replay_daily.constants import (
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT_DEFAULT,
    LAND_PRICES,
    PRODUCTS,
    hire_cost,
    total_hire_cost,
)
from replay_daily.lifecycle import canonical_board

from .foreman import ForemanConfig, apply_idle_cleanup, run_foreman
from .hiring import HiringRecommendation, recommend_hires
from .scheduler import PersistentTaskScheduler
from .tasks import (
    GenerationResult,
    Task,
    generate_optional_idle_cleanup_tasks,
)
from .manager import CheckpointPlanProvider, PlanProvider
from .plan import SELL_BIN_ANCHORS, DailyPlan
from .projection import clip_sell, project_plan
from .tasks import generate_tasks

__all__ = ["AgentConfig", "ExecutorAgent", "make_agent"]

_DIAGNOSTICS_SCHEMA_VERSION = 2
_MONEY_EPSILON = 1e-6
_INTERACTION_OPS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})
_CURRENT_SURVIVAL_SUPPRESSED_TASK_KINDS = frozenset({
    "BUILD_COOP", "BUILD_PASTURE", "BUY_ANIMAL", "BUY_LAND",
})
_PRIOR_DEBT_SUPPRESSED_TASK_KINDS = frozenset({
    "BUILD_COOP", "BUILD_PASTURE", "BUY_ANIMAL",
})


def _sell_bin_index(hour: int) -> int:
    return (int(hour) // 4) * 4


@dataclass(frozen=True)
class AgentConfig:
    tasks_per_worker: int = 10
    hire_cost_mult: int = FARM_HAND_COST_MULT_DEFAULT
    max_market_orders: int = 10
    shed_capacity: int = 100
    foreman: ForemanConfig = field(default_factory=ForemanConfig)
    strict: bool = False
    turn_trace: bool = False
    suppress_expansion_from_prior_debt: bool = True
    aggressive_sell_all: bool = False
    optional_idle_cleanup: bool = False
    optional_spare_watering: bool = False
    immediate_plant_water: bool = True
    deadline_safe_planting: bool = False
    deadline_safe_hiring: bool = False
    persistent_worker_queues: bool = False
    queue_ownership_repair: bool = False
    batch_reserved_supplies: bool = False
    underfoot_queue_insertion: bool = False
    schedule_informed_hiring: bool = False
    schedule_hiring_economic_repair: bool = False
    starvation_workload_visibility_repair: bool = False
    record_turn_snapshot: bool = True
    heuristic_care: bool = False
    heuristic_fertilizer: bool = False
    wheat_harvest_threshold: bool = False

    @property
    def idle_cleanup_enabled(self) -> bool:
        """Whether either PASS-only cleanup mode is enabled."""
        return self.optional_idle_cleanup or self.optional_spare_watering

    @property
    def cleanup_mode(self) -> str:
        """Resolve weed cleanup as the superset when both flags are enabled."""
        if self.optional_idle_cleanup:
            return "weed_water"
        if self.optional_spare_watering:
            return "water_only"
        return "none"


def _require_positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def _board_counts(board) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    crops = {name: 0 for name in CROP_ORDER}
    animals = {name: 0 for name in ANIMAL_ORDER}
    care_done = {name: 0 for name in ANIMAL_ORDER}
    fert_done = {name: 0 for name in CROP_ORDER}
    for row in board:
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            if "animal" in tile:
                species = tile["animal"]
                if species in animals:
                    animals[species] += 1
                    if tile.get("cared_today") is True:
                        care_done[species] += 1
            elif tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in crops:
                    crops[crop] += 1
                    derived = tile.get("derived") or {}
                    fertilized = derived.get("fertilizer_active") is True \
                        or (isinstance(tile.get("fertilized_until_day"), int)
                            and tile["fertilized_until_day"] >= 0)
                    if fertilized:
                        fert_done[crop] += 1
    return crops, animals, care_done, fert_done


def _animal_feed_state(
    obs: Mapping,
    seat: int,
    *,
    board: list[list[Any]] | None = None,
) -> dict[str, int]:
    farm = obs["farms"][seat]
    if board is None:
        board = canonical_board(
            farm["tiles"], int(obs["day"]), int(obs.get("step", 0)))
    unfed = 0
    starving = 0
    for row in board:
        for tile in row:
            if not isinstance(tile, Mapping) or "animal" not in tile:
                continue
            if tile.get("fed_today") is True:
                continue
            unfed += 1
            if int(tile.get("consecutive_unfed") or 0) >= 1:
                starving += 1
    private = obs.get("private") or {}
    carried_wheat = sum(
        int(inv.get("WHEAT", 0) or 0)
        for inv in (private.get("inventories") or [])
        if isinstance(inv, Mapping)
    )
    shed_wheat = int((private.get("shed") or {}).get("WHEAT", 0) or 0)
    available_wheat = carried_wheat + shed_wheat
    return {
        "unfed": unfed,
        "starving": starving,
        "carried_wheat": carried_wheat,
        "shed_wheat": shed_wheat,
        "available_wheat": available_wheat,
        "shed_reserve": max(0, unfed - carried_wheat),
        "shortage": max(0, unfed - available_wheat),
    }


def _snapshot_copy(value: Any) -> Any:
    """Copy the explicitly selected snapshot values into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _snapshot_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_copy(item) for item in value]
    return None


@dataclass(frozen=True)
class _PlantAttempt:
    worker_index: int
    tile: tuple[int, int]
    crop: str
    expected_step: int


@dataclass(frozen=True)
class _StarvationVisibility:
    """Flag-only partition between visible workload and safe dispatch work."""

    complete_tasks: tuple[Task, ...]
    eligible_tasks: tuple[Task, ...]
    hiring_tasks: tuple[Task, ...]
    critical_feeds: tuple[Task, ...] = ()
    safe_deferred: tuple[Task, ...] = ()
    deferred_tasks: tuple[Task, ...] = ()
    starving_animals: tuple[dict[str, Any], ...] = ()
    feed_coverage: tuple[dict[str, Any], ...] = ()
    reserved_worker_eta: tuple[dict[str, Any], ...] = ()
    deferred_reasons: dict[str, str] = field(default_factory=dict)
    forecast_before: dict[str, Any] = field(default_factory=dict)
    forecast_after: dict[str, Any] = field(default_factory=dict)


def _worker_turns_left(obs: Mapping) -> int:
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", int(obs.get("day", 0)) * 24 + hour))
    inclusive = max(0, min(24 - hour, 30 * 24 - 1 - step))
    return max(inclusive, 0)


def _canonical_worker_states(
    obs: Mapping,
    seat: int,
) -> tuple[list[tuple[int, int]], list[dict[str, int]]]:
    farm = obs["farms"][seat]
    positions = [farm.get("farmer") or [0, 0]]
    positions.extend(farm.get("hands") or [])
    canonical = [
        (int(position[1]), int(position[0]))
        for position in positions
        if isinstance(position, (list, tuple)) and len(position) == 2
    ] or [(0, 0)]
    inventories = list(((obs.get("private") or {}).get("inventories") or ()))
    states = [
        {str(item): max(0, int(quantity)) for item, quantity in
         (inventories[index] if index < len(inventories) else {}).items()}
        for index in range(len(canonical))
    ]
    return canonical, states


def _task_route_eta(
    position: tuple[int, int],
    inventory: Mapping[str, int],
    shed_wheat: int,
    task: Task,
    config: ForemanConfig,
) -> tuple[int | None, int, tuple[int, int], dict[str, int], int]:
    """Return ETA plus the copied worker/resource state after one task."""
    if task.tile is None:
        return None, 0, position, dict(inventory), shed_wheat
    target = (int(task.tile[0]), int(task.tile[1]))
    updated_inventory = dict(inventory)
    updated_shed = int(shed_wheat)
    if task.kind == "FEED" and task.required_item == "WHEAT":
        demand = max(1, int(task.quantity))
        carried = max(0, int(updated_inventory.get("WHEAT", 0)))
        missing = max(0, demand - carried)
        if missing > updated_shed:
            return None, missing, position, updated_inventory, updated_shed
        if missing:
            access = min(
                config.shed_access_tiles,
                key=lambda tile: (
                    abs(position[0] - tile[0]) + abs(position[1] - tile[1]),
                    tile,
                ),
            )
            eta = (
                abs(position[0] - access[0])
                + abs(position[1] - access[1])
                + math.ceil(missing / max(1, config.pickup_batch))
                + abs(access[0] - target[0])
                + abs(access[1] - target[1])
                + 1
            )
            updated_shed -= missing
            updated_inventory["WHEAT"] = carried + missing - demand
            return eta, missing, target, updated_inventory, updated_shed
        updated_inventory["WHEAT"] = carried - demand
    eta = abs(position[0] - target[0]) + abs(position[1] - target[1]) + 1
    return eta, 0, target, updated_inventory, updated_shed


class ExecutorAgent:
    def __init__(self, provider: PlanProvider, *, seat: int | None = None,
                 config: AgentConfig | None = None,
                 profile: Mapping[str, Any] | None = None) -> None:
        self.provider = provider
        self.seat = seat
        if seat is not None and seat not in (0, 1):
            raise ValueError(f"seat must be None, 0, or 1, got {seat!r}")
        self.config = config or AgentConfig()
        if profile is not None and not isinstance(profile, Mapping):
            raise TypeError("profile must be a mapping when provided")
        self._effective_profile = (
            copy.deepcopy(dict(profile)) if profile is not None else None)
        _require_positive_int(self.config.tasks_per_worker, "config.tasks_per_worker")
        _require_positive_int(self.config.max_market_orders, "config.max_market_orders")
        _require_positive_int(self.config.shed_capacity, "config.shed_capacity")
        self._day: int | None = None
        self._requested: DailyPlan | None = None
        self._feasible: DailyPlan | None = None
        self._projection_diagnostics: dict[str, Any] = {}
        self._bin_anchor: int | None = None
        self._remaining_sells: dict[str, int] = {}
        self._previous_execution: dict[str, int] = {"workers_hired": 0, "hire_cost": 0}
        self._max_hires_today: int = 0
        self._day_records: dict[int, dict[str, Any]] = {}
        self._errors: list[dict[str, Any]] = []
        self._suppress_expansion_today: bool = False
        self._debug_trace_turn: dict[str, Any] | None = None
        self._plant_attempts: dict[int, _PlantAttempt] = {}
        self._plant_water_continuations: dict[int, Task] = {}
        self._last_hire_rejections: list[dict[str, Any]] = []
        self._last_hiring_recommendation: HiringRecommendation | None = None
        self._scheduler = PersistentTaskScheduler()
        self._last_scheduler_result: Any | None = None
        self._last_step: int | None = None
        self._cleanup_metrics: dict[str, int] = {
            "baseline_pass_worker_actions": 0,
            "cleanup_replacements": 0,
            "weed_dig_cleanup_interactions": 0,
            "optional_water_cleanup_interactions": 0,
            "cleanup_movement_actions": 0,
            "remaining_pass_worker_actions": 0,
            "normal_non_pass_actions_changed": 0,
        }

    @staticmethod
    def _valid_unwatered_plant(
        board: list[list[Any]], tile: tuple[int, int], crop: str, day: int,
    ) -> bool:
        y, x = tile
        if not (0 <= y < len(board) and 0 <= x < len(board[y])):
            return False
        value = board[y][x]
        return isinstance(value, Mapping) \
            and value.get("kind") == "PLANT" \
            and value.get("crop") == crop \
            and value.get("planted_day") == day \
            and value.get("watered_today") is False

    def _refresh_plant_water_continuations(
        self,
        obs: Mapping,
        seat: int,
        board: list[list[Any]],
    ) -> None:
        if not self.config.immediate_plant_water:
            self._plant_attempts.clear()
            self._plant_water_continuations.clear()
            return

        step = int(obs.get("step", 0))
        for attempt in self._plant_attempts.values():
            if attempt.expected_step == step and self._valid_unwatered_plant(
                    board, attempt.tile, attempt.crop, int(obs["day"])):
                y, x = attempt.tile
                self._plant_water_continuations[attempt.worker_index] = Task(
                    key=f"WATER:{y},{x}", kind="WATER",
                    priority=Priority.MAINTENANCE, tile=attempt.tile,
                    crop=attempt.crop, source="plant_water_continuation",
                )
        self._plant_attempts.clear()

        worker_count = 1 + len(obs["farms"][seat].get("hands") or [])
        self._plant_water_continuations = {
            worker_index: task
            for worker_index, task in self._plant_water_continuations.items()
            if worker_index < worker_count
            and task.tile is not None
            and task.crop is not None
            and self._valid_unwatered_plant(
                board, task.tile, task.crop, int(obs["day"]))
        }

        if self._last_step is not None and int(obs.get("step", 0)) < self._last_step:
            self._scheduler.reset()

    def _record_plant_attempts(
        self,
        obs: Mapping,
        tasks: tuple[Task, ...],
        foreman_result: Any,
    ) -> None:
        self._plant_attempts.clear()
        if not self.config.immediate_plant_water:
            return
        task_by_key = {task.key: task for task in tasks}
        expected_step = int(obs.get("step", 0)) + 1
        for assignment in foreman_result.assignments:
            if not assignment.action or assignment.action[0] != "PLANT":
                continue
            task = task_by_key.get(assignment.task_key)
            if task is None or task.tile is None or task.crop is None:
                continue
            if tuple(assignment.action) != ("PLANT", task.crop):
                continue
            self._plant_attempts[assignment.worker_index] = _PlantAttempt(
                assignment.worker_index, task.tile, task.crop, expected_step)

    def _labor_forecast(
        self,
        obs: Mapping,
        seat: int,
        tasks: Sequence[Task],
    ) -> dict[str, Any]:
        """Make a diagnostic-only lower-bound route forecast."""
        positions, inventories = _canonical_worker_states(obs, seat)
        shed_wheat = int(((obs.get("private") or {}).get("shed") or {})
                         .get("WHEAT", 0) or 0)
        route_turns = 0
        feasible = 0
        unroutable: list[str] = []
        tile_tasks = [task for task in tasks if task.tile is not None]
        for task in sorted(tile_tasks, key=lambda item: item.sort_key):
            estimates = [
                _task_route_eta(position, inventory, shed_wheat, task,
                                self.config.foreman)[0]
                for position, inventory in zip(positions, inventories)
            ]
            eta = min((value for value in estimates if value is not None),
                      default=None)
            if eta is None:
                unroutable.append(task.key)
            else:
                feasible += 1
                route_turns += eta
        return {
            "task_count": len(tile_tasks),
            "feasible_task_count": feasible,
            "route_turns_lower_bound": route_turns,
            "unroutable_task_keys": unroutable,
            "worker_count": len(positions),
            "remaining_turns": _worker_turns_left(obs),
        }

    def _derive_starvation_visibility(
        self,
        obs: Mapping,
        seat: int,
        board: list[list[Any]],
        tasks: tuple[Task, ...],
        continuation_keys: set[str],
        feed: Mapping[str, int],
    ) -> _StarvationVisibility:
        """Partition starvation-boundary work without changing the flag-off path."""
        hiring_tasks = tuple(
            task for task in tasks if task.key not in continuation_keys)
        if not self.config.starvation_workload_visibility_repair:
            return _StarvationVisibility(
                complete_tasks=tasks,
                eligible_tasks=tasks,
                hiring_tasks=hiring_tasks,
            )
        before = self._labor_forecast(obs, seat, tasks)
        if not feed["starving"]:
            return _StarvationVisibility(
                complete_tasks=tasks,
                eligible_tasks=tasks,
                hiring_tasks=hiring_tasks,
                forecast_before=before,
                forecast_after=dict(before),
            )

        hour = int(obs["hour"])
        starving_animals: list[dict[str, Any]] = []
        starving_coords: set[tuple[int, int]] = set()
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if not isinstance(tile, Mapping) or "animal" not in tile \
                        or tile.get("fed_today") is True \
                        or int(tile.get("consecutive_unfed") or 0) < 1:
                    continue
                starving_coords.add((y, x))
                starving_animals.append({
                    "identity": f"{tile.get('animal')}:{y},{x}",
                    "animal": tile.get("animal"),
                    "tile": [y, x],
                    "consecutive_unfed": int(tile.get("consecutive_unfed") or 0),
                })

        critical_feeds: list[Task] = []
        critical_by_key: dict[str, Task] = {}
        for task in tasks:
            if task.kind != "FEED" or task.tile not in starving_coords:
                continue
            urgent = replace(task, deadline_hour=hour,
                             source="starvation_boundary")
            critical_feeds.append(urgent)
            critical_by_key[task.key] = urgent
        critical_feeds.sort(key=lambda task: task.sort_key)

        positions, inventories = _canonical_worker_states(obs, seat)
        shed_wheat = int(((obs.get("private") or {}).get("shed") or {})
                         .get("WHEAT", 0) or 0)
        remaining_turns = _worker_turns_left(obs)
        reserved_eta = [0] * len(positions)
        feed_coverage: list[dict[str, Any]] = []
        for task in critical_feeds:
            candidates: list[tuple[int, int, int, tuple[int, int], dict[str, int], int]] = []
            resource_candidates: list[int] = []
            for worker_index, (position, inventory) in enumerate(
                    zip(positions, inventories)):
                eta, missing, new_position, new_inventory, new_shed = \
                    _task_route_eta(position, inventory, shed_wheat, task,
                                    self.config.foreman)
                if eta is not None:
                    resource_candidates.append(worker_index)
                    if eta <= remaining_turns - reserved_eta[worker_index]:
                        candidates.append((
                            eta + reserved_eta[worker_index], eta, worker_index,
                            new_position, new_inventory, new_shed))
            coverage = {
                "task_key": task.key,
                "animal": task.animal,
                "tile": list(task.tile) if task.tile is not None else None,
                "feasible_workers": [item[2] for item in candidates],
                "resource_feasible_workers": resource_candidates,
                "reserved_worker": None,
                "reserved_eta": None,
                "status": "uncovered",
            }
            if candidates:
                _, eta, worker_index, new_position, new_inventory, new_shed = \
                    min(candidates, key=lambda item: (item[0], item[2]))
                positions[worker_index] = new_position
                inventories[worker_index] = new_inventory
                shed_wheat = new_shed
                reserved_eta[worker_index] += eta
                coverage.update(
                    reserved_worker=worker_index,
                    reserved_eta=reserved_eta[worker_index],
                    status="reserved_forecast",
                )
            elif not resource_candidates:
                coverage["reason"] = "resource_blocked"
            else:
                coverage["reason"] = "deadline_infeasible"
            feed_coverage.append(coverage)

        safe_deferred: list[Task] = []
        deferred_reasons: dict[str, str] = {}
        for task in tasks:
            if task.kind != "WATER" or task.key in continuation_keys \
                    or task.source != "water_must_weed_boundary":
                continue
            possible = any(
                (eta := _task_route_eta(
                    position, inventory, shed_wheat, task,
                    self.config.foreman)[0]) is not None
                and reserved_eta[index] + eta <= remaining_turns
                for index, (position, inventory) in enumerate(
                    zip(positions, inventories))
            )
            if possible:
                # Keep the existing maintenance task visible, but make the
                # released crop work yield to boundary FEED in the foreman.
                released = replace(task, priority=Priority.PRODUCTIVE)
                safe_deferred.append(released)
            else:
                deferred_reasons[task.key] = "feed_reservation_or_deadline"

        safe_continuations: list[Task] = []
        continuation_by_key = {task.key: task for task in tasks
                               if task.key in continuation_keys}
        for worker_index, task in self._plant_water_continuations.items():
            if task.key not in continuation_by_key or worker_index >= len(positions):
                continue
            eta = _task_route_eta(
                positions[worker_index], inventories[worker_index], shed_wheat,
                task, self.config.foreman)[0]
            if reserved_eta[worker_index] == 0 and eta is not None \
                    and eta <= remaining_turns:
                safe_continuations.append(task)
            else:
                deferred_reasons[task.key] = "feed_reservation_or_deadline"

        eligible_by_key = {task.key: task for task in critical_feeds}
        eligible_by_key.update({task.key: task for task in safe_deferred})
        eligible_by_key.update({task.key: task for task in safe_continuations})
        eligible = tuple(
            task for task in tasks
            if task.tile is None or task.key in eligible_by_key
        )
        deferred = tuple(
            task for task in tasks
            if task.tile is not None and task.key not in eligible_by_key
        )
        hiring = tuple(
            critical_by_key.get(task.key, task)
            for task in tasks if task.key not in continuation_keys
        )
        forecast_after = self._labor_forecast(obs, seat, eligible)
        for task in deferred:
            deferred_reasons.setdefault(
                task.key,
                "ordinary_daily_feed" if task.kind == "FEED"
                else "starvation_boundary_noncritical_work",
            )
        for coverage in feed_coverage:
            if coverage["status"] == "uncovered":
                deferred_reasons.setdefault(coverage["task_key"],
                                            coverage.get("reason", "uncovered"))
        return _StarvationVisibility(
            complete_tasks=tasks,
            eligible_tasks=eligible,
            hiring_tasks=hiring,
            critical_feeds=tuple(critical_feeds),
            safe_deferred=tuple((*safe_deferred, *safe_continuations)),
            deferred_tasks=deferred,
            starving_animals=tuple(starving_animals),
            feed_coverage=tuple(feed_coverage),
            deferred_reasons=dict(deferred_reasons),
            reserved_worker_eta=tuple({
                "worker_index": index,
                "reserved_feed_eta": value,
                "remaining_after_feed": max(0, remaining_turns - value),
            } for index, value in enumerate(reserved_eta)),
            forecast_before=before,
            forecast_after=forecast_after,
        )

    @staticmethod
    def _visibility_json(visibility: _StarvationVisibility) -> dict[str, Any]:
        return {
            "complete_workload": [task.to_json_dict()
                                   for task in visibility.complete_tasks],
            "eligible_workload": [task.to_json_dict()
                                   for task in visibility.eligible_tasks],
            "hiring_workload": [task.to_json_dict()
                                 for task in visibility.hiring_tasks],
            "critical_feed_keys": [task.key for task in visibility.critical_feeds],
            "starving_animals": list(visibility.starving_animals),
            "feed_coverage": [dict(item) for item in visibility.feed_coverage],
            "reserved_worker_eta": [dict(item)
                                     for item in visibility.reserved_worker_eta],
            "safe_deferred_work": [task.to_json_dict()
                                    for task in visibility.safe_deferred],
            "deferred_work": [
                {"key": task.key, "kind": task.kind,
                 "reason": visibility.deferred_reasons.get(
                     task.key,
                     "ordinary_daily_feed" if task.kind == "FEED"
                     else "starvation_boundary_noncritical_work")}
                for task in visibility.deferred_tasks
            ],
            "labor_forecast_before_filtering": dict(visibility.forecast_before),
            "labor_forecast_after_filtering": dict(visibility.forecast_after),
        }

    def __call__(self, obs: Mapping) -> dict[str, Any]:
        try:
            return self._act(obs)
        except Exception as exc:
            if self.config.strict:
                raise
            self._errors.append({
                "step": obs.get("step") if isinstance(obs, Mapping) else None,
                "error_type": type(exc).__name__, "message": str(exc),
            })
            return self._fallback_action(obs)

    def _resolve_seat(self, obs: Mapping) -> int:
        observed = obs.get("player")
        if self.seat is not None:
            if observed is not None and int(observed) != self.seat:
                raise ValueError(f"obs player {observed!r} contradicts explicit agent seat {self.seat}")
            return self.seat
        if observed is None:
            raise ValueError("obs carries no 'player' field; construct the agent with an explicit seat")
        seat = int(observed)
        if seat not in (0, 1):
            raise ValueError(f"obs player must be 0 or 1, got {seat!r}")
        return seat

    def _fallback_action(self, obs: Mapping) -> dict[str, Any]:
        hands = 0
        try:
            seat = self._resolve_seat(obs)
            hands = len(obs["farms"][seat].get("hands") or [])
        except Exception:
            hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}

    def _new_day(
        self,
        obs: Mapping,
        seat: int,
        *,
        board: list[list[Any]] | None = None,
    ) -> None:
        farm = obs["farms"][seat]
        if board is None:
            board = canonical_board(
                farm["tiles"], int(obs["day"]), int(obs.get("step", 0)))
        crops, animals, care_done, fert_done = _board_counts(board)
        prior_debt = False
        if self._day is not None:
            hires = self._max_hires_today
            self._previous_execution = {
                "workers_hired": hires,
                "hire_cost": total_hire_cost(hires, self.config.hire_cost_mult),
            }
            record = self._day_records[self._day]
            record["achieved_final"] = {
                "crops": crops, "animals": animals,
                "land_count": len(farm["unlocked_quadrants"]),
            }
            record["care_completed_observed"] = care_done
            record["fertilizer_completed_observed"] = fert_done
            debt = record.get("end_of_day_work_debt") or {}
            prior_debt = bool(debt.get("all"))
            prior_debt_suppressed = (
                prior_debt and self.config.suppress_expansion_from_prior_debt
            )
            record["next_day_expansion_suppressed"] = prior_debt_suppressed
        else:
            prior_debt_suppressed = False
        self._suppress_expansion_today = prior_debt_suppressed
        raw_hires = farm.get("hires_today", 0)
        self._max_hires_today = raw_hires if isinstance(raw_hires, int) and not isinstance(raw_hires, bool) else 0
        self._requested = self.provider.daily_plan(obs, seat, dict(self._previous_execution))
        result = project_plan(
            self._requested,
            current_land_count=len(farm["unlocked_quadrants"]),
            current_animals=animals,
            current_crops=crops,
        )
        self._feasible = result.feasible_plan
        self._projection_diagnostics = result.diagnostics
        self._bin_anchor = None
        self._remaining_sells = {}
        self._day_records[int(obs["day"])] = {
            "requested": self._requested.to_json_dict(),
            "feasible": self._feasible.to_json_dict(),
            "projection_changes": self._projection_diagnostics,
            "foreman_counts": {"movement": 0, "interaction": 0, "pickup": 0, "pass": 0},
            "unfinished_tasks": [],
            "missed_maintenance": [],
            "end_of_day_work_debt": {"all": [], "survival": [], "maintenance": [], "productive": [], "manager": []},
            "pending_task_turns": {},
            "pending_maintenance_turns": {},
            "sells": {},
            "hires": {"requested": 0, "submitted": 0, "observed_max": self._max_hires_today},
            "previous_labor": dict(self._previous_execution),
            "unresolved_generator": [],
            "survival": {
                "expansion_suppressed_from_prior_debt": self._suppress_expansion_today,
                "expansion_suppressed_current": False,
                "starvation_preemption_turns": 0,
                "feed_reserve_protected_units": 0,
                "feed_shortage_turns": 0,
                "partial_feed_buys": 0,
            },
            "land_purchase": {
                "requested": False,
                "task_present": False,
                "suppressed_prior_debt": False,
                "suppressed_current_survival": False,
                "affordable_before_hires": False,
                "unaffordable_before_hires": False,
                "submitted": False,
                "land_cost": None,
                "cash_before_land": None,
            },
            "errors": [],
        }
        if (self.config.schedule_informed_hiring
                and self.config.schedule_hiring_economic_repair):
            day_record = self._day_records[int(obs["day"])]
            day_record["hiring_decisions"] = []
        self._day = int(obs["day"])

    def _refresh_sell_ledger(self, obs: Mapping, bin_anchor: int) -> None:
        bin_index = SELL_BIN_ANCHORS.index(bin_anchor)
        self._remaining_sells = {
            product: self._feasible.sell_quantities[product_index][bin_index]
            for product_index, product in enumerate(PRODUCTS)
        }
        self._bin_anchor = bin_anchor
        record = self._day_records[int(obs["day"])]
        if self.config.aggressive_sell_all:
            record["sells"][str(bin_anchor)] = {
                product: {
                    "source": "aggressive_sell_all",
                    "requested": self._remaining_sells[product],
                    "submitted": 0,
                    "remaining": self._remaining_sells[product],
                    "override_requested": 0,
                    "override_submitted": 0,
                    "override_skipped": 0,
                }
                for product in PRODUCTS
            }
        else:
            record["sells"][str(bin_anchor)] = {
                product: {"requested": self._remaining_sells[product], "submitted": 0, "remaining": self._remaining_sells[product]}
                for product in PRODUCTS
            }

    def _sell_candidates(
        self,
        obs: Mapping,
        seat: int,
        *,
        feed: Mapping[str, int] | None = None,
    ) -> list[dict]:
        shed = (obs.get("private") or {}).get("shed") or {}
        if self.config.aggressive_sell_all:
            if feed is None:
                feed = _animal_feed_state(obs, seat)
            candidates = []
            bin_log = self._day_records[int(obs["day"])]
            bin_log = bin_log["sells"][str(self._bin_anchor)]
            for product in PRODUCTS:
                available = int(shed.get(product, 0))
                if product == "WHEAT" and available > 0:
                    protected = min(available, int(feed["shed_reserve"]))
                    if protected:
                        survival = self._day_records[int(obs["day"])]["survival"]
                        survival["feed_reserve_protected_units"] = max(
                            int(survival["feed_reserve_protected_units"]), protected)
                    available -= protected
                if available <= 0:
                    continue
                executed, _ = clip_sell(product, available, available)
                if executed > 0:
                    bin_log[product]["override_requested"] += executed
                    candidates.append({
                        "order": ["SELL", product, executed],
                        "product": product,
                        "executed": executed,
                        "source": "aggressive_sell_all",
                        "bc_requested": self._remaining_sells.get(product, 0),
                    })
            return candidates

        if feed is None:
            feed = _animal_feed_state(obs, seat)
        candidates = []
        for product in PRODUCTS:
            remaining = self._remaining_sells.get(product, 0)
            if remaining <= 0:
                continue
            available = int(shed.get(product, 0))
            if product == "WHEAT":
                protected = min(available, feed["shed_reserve"])
                available -= protected
                survival = self._day_records[int(obs["day"])]["survival"]
                survival["feed_reserve_protected_units"] = max(
                    int(survival["feed_reserve_protected_units"]), protected)
            executed, _ = clip_sell(product, remaining, available)
            if executed > 0:
                candidates.append({"order": ["SELL", product, executed], "product": product, "executed": executed})
        return candidates

    def _commit_sells(self, obs: Mapping, day: int, committed: list[dict]) -> None:
        record = self._day_records[day]
        bin_log = record["sells"][str(self._bin_anchor)]
        for item in committed:
            product = item["product"]
            executed = item["executed"]
            if self.config.aggressive_sell_all:
                record["sells"][str(self._bin_anchor)][product]["override_submitted"] += executed
                continue
            self._remaining_sells[product] = self._remaining_sells.get(product, 0) - executed
            entry = bin_log[product]
            entry["submitted"] += executed
            entry["remaining"] = self._remaining_sells[product]

    def _hire_orders(
        self,
        obs: Mapping,
        seat: int,
        tile_tasks: list[Task],
        available_cash: float,
        *,
        visibility: _StarvationVisibility | None = None,
        market_order_limit: int | None = None,
    ) -> tuple[list[list], int]:
        self._last_hiring_recommendation = None
        if visibility is not None and self.config.starvation_workload_visibility_repair:
            tile_tasks = [task for task in visibility.hiring_tasks
                          if task.tile is not None]
        if self.config.schedule_informed_hiring:
            recommendation = recommend_hires(
                obs, seat, tile_tasks,
                scheduler_result=(self._last_scheduler_result
                                  if self.config.persistent_worker_queues else None),
                available_cash=available_cash,
                hire_cost_mult=self.config.hire_cost_mult,
                market_order_limit=(
                    self.config.max_market_orders
                    if market_order_limit is None else max(0, market_order_limit)),
                economic_repair=self.config.schedule_hiring_economic_repair,
            )
            self._last_hiring_recommendation = recommendation
            self._last_hire_rejections = [
                dict(item) for item in recommendation.rejection_diagnostics
            ]
            return recommendation.orders, recommendation.wanted_hires

        farm = obs["farms"][seat]
        current_hands = len(farm.get("hands") or [])
        positions = [farm.get("farmer") or [0, 0]]
        positions.extend(farm.get("hands") or [])
        anchors = []
        for pos in positions:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                anchors.append((int(pos[1]), int(pos[0])))
        if not anchors:
            anchors = [(4, 4)]
        hour = int(obs["hour"])
        step = int(obs.get("step", int(obs["day"]) * 24 + hour))
        inclusive_turns_left = max(0, min(24 - hour, 30 * 24 - 1 - step))
        turns_left = max(inclusive_turns_left, 1)

        def turns_needed(tasks_):
            total = 0
            for t in tasks_:
                if t.tile is None:
                    continue
                travel = min(abs(t.tile[0] - ay) + abs(t.tile[1] - ax) for ay, ax in anchors)
                total += travel + 1
            return total

        maintenance = [t for t in tile_tasks if t.priority == Priority.MAINTENANCE or t.kind == "FEED"]
        crude = math.ceil(len(tile_tasks) / self.config.tasks_per_worker) if tile_tasks else 0
        desired = crude
        if maintenance:
            maint_workers = math.ceil(turns_needed(maintenance) / max(turns_left, 1))
            desired = max(desired, min(maint_workers, current_hands + 1 + len(maintenance)))
        wanted = max(desired - current_hands, 0)
        self._last_hire_rejections = []
        if self.config.deadline_safe_hiring and wanted > 0 \
                and inclusive_turns_left <= 1:
            self._last_hire_rejections.append({
                "reason": "no_future_worker_action_before_reset_or_terminal",
                "wanted": wanted,
                "inclusive_turns_left": inclusive_turns_left,
                "future_worker_actions": max(0, inclusive_turns_left - 1),
            })
            return [], wanted
        already_today = int(farm.get("hires_today", 0))
        cash = available_cash
        affordable = 0
        for k in range(wanted):
            cost = hire_cost(already_today + k, self.config.hire_cost_mult)
            if cash + _MONEY_EPSILON < cost:
                break
            cash -= cost
            affordable += 1
        return [["HIRE"]] * affordable, wanted

    @staticmethod
    def _sell_proceeds(obs: Mapping, committed_sells: list[dict]) -> float:
        prices = (obs.get("market") or {}).get("prices") or {}
        return sum(float(prices.get(item["product"], 0)) * float(item["executed"]) for item in committed_sells)

    @staticmethod
    def _buy_order_cost(obs: Mapping, task, unlocked_count: int, quantity: int | None = None) -> float | None:
        kind = task.kind
        quantity = int(task.quantity if quantity is None else quantity)
        if quantity <= 0:
            return None
        if kind == "BUY_SEED" and task.crop in CROPS:
            return float(CROPS[task.crop]["seed"] * quantity)
        if kind == "BUY_ANIMAL" and task.animal in ANIMALS:
            return float(ANIMALS[task.animal]["cost"] * quantity)
        if kind == "BUY_LAND":
            index = unlocked_count - 1
            if 0 <= index < len(LAND_PRICES):
                return float(LAND_PRICES[index])
            return None
        if kind == "BUY_PRODUCT" and task.product in PRODUCTS:
            from fast_env.market import market_price
            inventory = int(((obs.get("market") or {}).get("inventory") or {}).get(task.product, 0))
            total = 0.0
            for k in range(quantity):
                level = inventory - k - 1
                total += float(market_price(task.product, max(level, 0)))
            return total
        return None

    @staticmethod
    def _buy_op(task, quantity: int | None = None) -> list | None:
        quantity = int(task.quantity if quantity is None else quantity)
        if quantity <= 0:
            return None
        if task.kind == "BUY_SEED" and task.crop:
            return ["BUY_SEED", task.crop, quantity]
        if task.kind == "BUY_PRODUCT" and task.product:
            return ["BUY_PRODUCT", task.product, quantity]
        if task.kind == "BUY_ANIMAL" and task.animal:
            return ["BUY_ANIMAL", task.animal, quantity]
        if task.kind == "BUY_LAND":
            return ["BUY_LAND"]
        return None

    def _affordable_survival_feed_buy(self, obs: Mapping, task: Task, unlocked_count: int, available_cash: float):
        private = obs.get("private")
        shed = private.get("shed") if isinstance(private, Mapping) else None
        used = sum(
            quantity for quantity in shed.values()
            if isinstance(quantity, int)
            and not isinstance(quantity, bool)
            and quantity >= 0
        ) if isinstance(shed, Mapping) else 0
        room = max(0, self.config.shed_capacity - used)
        for quantity in range(min(int(task.quantity), room), 0, -1):
            cost = self._buy_order_cost(obs, task, unlocked_count, quantity=quantity)
            if cost is not None and available_cash + _MONEY_EPSILON >= cost:
                return self._buy_op(task, quantity=quantity), cost, quantity
        return None, 0.0, 0

    @staticmethod
    def _end_of_day_debt(tasks: tuple[Task, ...], foreman_result: Any) -> dict[str, list[str]]:
        completed_last_turn = {
            a.task_key for a in foreman_result.assignments
            if a.task_key is not None and a.action and a.action[0] in _INTERACTION_OPS
        }
        remaining = [t for t in tasks if t.tile is not None and t.key not in completed_last_turn]
        survival = [
            t.key for t in remaining
            if t.kind == "FEED" or (t.kind == "WATER" and t.source == "water_must_weed_boundary")
        ]
        maintenance = [t.key for t in remaining if t.priority == Priority.MAINTENANCE and t.key not in survival]
        productive = [t.key for t in remaining if t.priority == Priority.PRODUCTIVE]
        manager = [t.key for t in remaining if t.priority == Priority.MANAGER]
        return {
            "all": [t.key for t in remaining],
            "survival": survival,
            "maintenance": maintenance,
            "productive": productive,
            "manager": manager,
        }

    @staticmethod
    def _pending_task_keys(foreman_result: Any) -> list[str]:
        """Include assigned movement/pickup work in the debug churn metric."""
        pending: list[str] = []
        seen: set[str] = set()

        for task in foreman_result.unassigned_tile_tasks:
            if task.key not in seen:
                pending.append(task.key)
                seen.add(task.key)
        for assignment in foreman_result.assignments:
            task_key = assignment.task_key
            if task_key is None or task_key in seen:
                continue
            if not assignment.action \
                    or assignment.action[0] not in _INTERACTION_OPS:
                pending.append(task_key)
                seen.add(task_key)
        return pending

    @staticmethod
    def _market_snapshot_order(kind: str, payload: Any) -> list:
        if kind == "sell":
            return list(payload["order"])
        return list(payload)

    def _build_debug_trace_turn(
        self,
        *,
        day: int,
        hour: int,
        tasks: tuple[Task, ...],
        dispatch_tasks: tuple[Task, ...],
        generated_tasks: tuple[Task, ...],
        cleanup_tasks: tuple[Task, ...],
        generation: GenerationResult,
        foreman_result: Any,
        feed: Mapping[str, int],
        expansion_suppressed: bool,
        orders: list[list],
        candidates: list[tuple[str, Any]],
        unaffordable_orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_by_key = {
            task.key: task for task in (*tasks, *dispatch_tasks, *cleanup_tasks)
        }
        assignments = []
        for assignment in foreman_result.assignments:
            task = task_by_key.get(assignment.task_key)
            detail = {
                "worker_index": assignment.worker_index,
                "task_key": assignment.task_key,
                "reason": assignment.reason,
                "action": list(assignment.action),
                "target": list(task.tile) if task is not None and task.tile is not None else None,
            }
            if task is not None and task.source in ("water_optional_spare", "dig_cleanup"):
                detail["source"] = task.source
            assignments.append(detail)

        included_count = len(orders)
        skipped = [
            {
                "reason": "market_order_cap",
                "order": self._market_snapshot_order(kind, payload),
            }
            for kind, payload in candidates[included_count:]
        ]
        record = self._day_records[day]
        survival_record = record["survival"]
        eod_work_debt = record.get("end_of_day_work_debt") if hour == 23 else None

        snapshot = {
            "schema_version": 1,
            "day": day,
            "hour": hour,
            "actions": {
                "farmer": list(foreman_result.farmer_action),
                "hands": [list(action) for action in foreman_result.hands_actions],
            },
            "manager": {
                "requested": self._requested.to_json_dict() if self._requested is not None else None,
                "feasible": self._feasible.to_json_dict() if self._feasible is not None else None,
                "projection_changes": self._projection_diagnostics,
            },
            "tasks": [task.to_json_dict() for task in generated_tasks],
            "unresolved_tasks": list(generation.unresolved),
            "generation_diagnostics": list(generation.diagnostics),
            "assignments": assignments,
            "unassigned": {
                "task_keys": [task.key for task in foreman_result.unassigned_tile_tasks],
                "reasons": dict(sorted(foreman_result.unassigned_reasons.items())),
            },
            "market": {
                "submitted": [list(order) for order in orders],
                "unaffordable": [
                    {
                        "task": item.get("task"),
                        "cost": item.get("cost"),
                        "cash_available": item.get("cash_available"),
                        "survival": item.get("survival"),
                    }
                    for item in unaffordable_orders
                ],
                "skipped": skipped,
            },
            "survival": {
                "unfed_count": feed["unfed"],
                "starvation_boundary_count": feed["starving"],
                "shed_wheat": feed["shed_wheat"],
                "carried_wheat": feed["carried_wheat"],
                "protected_reserve": feed["shed_reserve"],
                "feed_reserve_protected_units": survival_record["feed_reserve_protected_units"],
                "shortage": feed["shortage"],
                "expansion_suppressed": bool(expansion_suppressed),
                "eod_work_debt": eod_work_debt,
            },
        }
        if self.config.starvation_workload_visibility_repair:
            snapshot["starvation_visibility"] = self._day_records[day].get(
                "starvation_visibility", {})
        if self.config.aggressive_sell_all:
            submitted_sells = []
            skipped_sells = []
            for index, (kind, payload) in enumerate(candidates):
                if kind != "sell":
                    continue
                detail = {
                    "source": payload["source"],
                    "product": payload["product"],
                    "quantity": payload["executed"],
                    "bc_requested": payload["bc_requested"],
                    "order": list(payload["order"]),
                }
                if index < included_count:
                    submitted_sells.append(detail)
                else:
                    detail["status"] = "skipped_market_order_cap"
                    skipped_sells.append(detail)
            snapshot["market"]["sell_mode"] = "aggressive_sell_all"
            snapshot["market"]["sell_submitted"] = submitted_sells
            snapshot["market"]["sell_skipped"] = skipped_sells
        return _snapshot_copy(snapshot)

    @staticmethod
    def _trace_task(task: Task, farm: Mapping) -> dict[str, Any]:
        """Return only causal, JSON-safe fields for one survival task."""
        entry: dict[str, Any] = {
            "key": task.key,
            "tile": list(task.tile) if task.tile is not None else None,
            "source": task.source,
            "priority": task.priority.name,
        }
        if task.kind == "WATER" and task.tile is not None:
            y, x = task.tile
            tiles = farm.get("tiles") or []
            tile = tiles[y][x] if 0 <= y < len(tiles) \
                and 0 <= x < len(tiles[y]) else None
            if isinstance(tile, Mapping):
                for name in ("consecutive_unwatered", "watered_today"):
                    value = tile.get(name)
                    if isinstance(value, bool):
                        entry[name] = value
                    elif name == "consecutive_unwatered" \
                            and isinstance(value, int):
                        entry[name] = value
        return entry

    @staticmethod
    def _trace_worker_position(farm: Mapping, worker_index: int) -> list[int] | None:
        positions = [farm.get("farmer")]
        positions.extend(farm.get("hands") or [])
        if not 0 <= worker_index < len(positions):
            return None
        position = positions[worker_index]
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
        try:
            # Foreman positions are canonical [y, x], while observations use
            # [x, y].  This is the position at the start of this turn.
            return [int(position[1]), int(position[0])]
        except (TypeError, ValueError):
            return None

    def _record_turn_trace(
        self,
        obs: Mapping,
        seat: int,
        day: int,
        hour: int,
        feed: Mapping[str, int],
        expansion_suppressed: bool,
        land_purchase: Mapping[str, Any],
        tasks: tuple[Task, ...],
        dispatch_tasks: tuple[Task, ...],
        cleanup_tasks: tuple[Task, ...],
        foreman_result: Any,
        pending: list[str],
        trace_market_by_payload: Mapping[int, tuple[str, str]],
        included: list[tuple[str, Any]],
        unaffordable_orders: list[dict[str, Any]],
    ) -> None:
        farm = obs["farms"][seat]
        survival_tasks = [
            task for task in tasks
            if task.kind == "FEED"
            or (task.kind == "WATER" and task.source == "water_must_weed_boundary")
        ]
        survival_keys = {task.key for task in survival_tasks}
        pending_keys = set(pending)
        dispatched_keys = {task.key for task in dispatch_tasks}
        # A starvation preemption removes non-FEED survival work from the
        # foreman's input; retain it as pending rather than misreporting it as
        # absent from the generated survival set.
        pending_survival = [
            task.key for task in survival_tasks
            if task.key in pending_keys or task.key not in dispatched_keys
        ]
        expansion_keys = {
            task.key for task in tasks
            if task.kind in ("BUY_ANIMAL", "BUY_LAND")
        }
        submitted = {"survival": [], "expansion": []}
        for payload_kind, payload in included:
            if payload_kind != "buy":
                continue
            category_and_key = trace_market_by_payload.get(id(payload))
            if category_and_key is not None:
                category, key = category_and_key
                submitted[category].append(key)
        unaffordable = {"survival": [], "expansion": []}
        for item in unaffordable_orders:
            key = item.get("task")
            if key in survival_keys:
                unaffordable["survival"].append(key)
            elif key in expansion_keys:
                unaffordable["expansion"].append(key)

        assignments = []
        task_by_key = {
            task.key: task for task in (*tasks, *dispatch_tasks, *cleanup_tasks)
        }
        for assignment in foreman_result.assignments:
            action = list(assignment.action)
            detail = {
                "worker": "farmer" if assignment.worker_index == 0
                else f"hand_{assignment.worker_index - 1}",
                "worker_index": assignment.worker_index,
                "position": self._trace_worker_position(
                    farm, assignment.worker_index),
                "task_key": assignment.task_key,
                "action": action,
                "op_family": action[0] if action else None,
            }
            task = task_by_key.get(assignment.task_key)
            if task is not None and task.tile is not None \
                    and task.source in ("water_optional_spare", "dig_cleanup"):
                detail["target"] = list(task.tile)
            if task is not None and task.source in ("water_optional_spare", "dig_cleanup"):
                detail["source"] = task.source
            assignments.append(detail)

        reasons = []
        if self._suppress_expansion_today:
            reasons.append("prior_day_work_debt")
        if feed["starving"]:
            reasons.append("current_starvation")
        if feed["shortage"]:
            reasons.append("current_feed_shortage")
        entry = {
            "day": day,
            "hour": hour,
            "feed": {
                "starving": bool(feed["starving"]),
                "shortage": int(feed["shortage"]),
                "unfed": int(feed["unfed"]),
                "reserve": int(feed["shed_reserve"]),
            },
            "expansion": {
                "suppressed_current": bool(expansion_suppressed),
                "suppressed_from_prior": bool(self._suppress_expansion_today),
                "reasons": reasons,
            },
            "land_purchase": dict(land_purchase),
            "survival_tasks": {
                "feed": [self._trace_task(task, farm) for task in survival_tasks
                         if task.kind == "FEED"],
                "water_must_weed_boundary": [
                    self._trace_task(task, farm) for task in survival_tasks
                    if task.kind == "WATER"
                    and task.source == "water_must_weed_boundary"],
            },
            "assignments": assignments,
            "unassigned_survival_task_keys": [
                task.key for task in foreman_result.unassigned_tile_tasks
                if task.key in survival_keys],
            "pending_survival_task_keys": pending_survival,
            "counts": dict(foreman_result.counts),
            "market": {
                category: {
                    "submitted_keys": values,
                    "unaffordable_keys": unaffordable[category],
                }
                for category, values in submitted.items()
            },
        }
        if self.config.starvation_workload_visibility_repair:
            entry["starvation_visibility"] = self._day_records[day].get(
                "starvation_visibility", {})
        trace = self._day_records[day].setdefault("turn_trace", [])
        for index, prior in enumerate(trace):
            if prior.get("hour") == hour:
                trace[index] = entry
                break
        else:
            trace.append(entry)
        trace.sort(key=lambda item: (int(item["day"]), int(item["hour"])))

    def _record_cleanup_telemetry(
        self,
        normal_foreman: Any,
        final_foreman: Any,
        cleanup_tasks: tuple[Task, ...],
    ) -> None:
        """Accumulate bounded action aggregates without retaining observations."""
        normal_actions = (
            normal_foreman.farmer_action, *normal_foreman.hands_actions)
        final_actions = (final_foreman.farmer_action, *final_foreman.hands_actions)
        if len(normal_actions) != len(final_actions) \
                or len(normal_actions) != len(normal_foreman.assignments):
            raise ValueError("foreman returned misaligned worker actions")
        cleanup_sources = {
            task.key: task.source for task in cleanup_tasks
        }
        for normal_action, final_action, assignment in zip(
                normal_actions, final_actions, final_foreman.assignments):
            if normal_action != ("PASS",):
                if final_action != normal_action:
                    self._cleanup_metrics["normal_non_pass_actions_changed"] += 1
                continue
            self._cleanup_metrics["baseline_pass_worker_actions"] += 1
            if final_action == ("PASS",):
                self._cleanup_metrics["remaining_pass_worker_actions"] += 1
                continue
            source = cleanup_sources.get(assignment.task_key)
            if source is None:
                continue
            self._cleanup_metrics["cleanup_replacements"] += 1
            if final_action[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
                self._cleanup_metrics["cleanup_movement_actions"] += 1
            elif source == "dig_cleanup" and final_action == ("DIG",):
                self._cleanup_metrics["weed_dig_cleanup_interactions"] += 1
            elif source == "water_optional_spare" \
                    and final_action == ("WATER",):
                self._cleanup_metrics["optional_water_cleanup_interactions"] += 1

        if self._cleanup_metrics["normal_non_pass_actions_changed"]:
            raise AssertionError("PASS-only cleanup changed a normal non-PASS action")

    def _cleanup_diagnostics(self) -> dict[str, Any]:
        metrics = dict(self._cleanup_metrics)
        baseline = metrics["baseline_pass_worker_actions"]
        metrics["cleanup_replacement_rate"] = (
            metrics["cleanup_replacements"] / baseline if baseline else 0.0
        )
        return metrics

    def _act(self, obs: Mapping) -> dict[str, Any]:
        seat = self._resolve_seat(obs)
        day, hour = int(obs["day"]), int(obs["hour"])
        board = canonical_board(
            obs["farms"][seat]["tiles"], day, int(obs.get("step", 0)))
        self._refresh_plant_water_continuations(obs, seat, board)
        if self._day != day:
            self._new_day(obs, seat, board=board)
        else:
            farm = obs["farms"][seat]
            raw_hires = farm.get("hires_today", 0)
            if isinstance(raw_hires, int) and not isinstance(raw_hires, bool):
                self._max_hires_today = max(self._max_hires_today, raw_hires)

        bin_anchor = _sell_bin_index(hour)
        if bin_anchor != self._bin_anchor:
            self._refresh_sell_ledger(obs, bin_anchor)

        generation = generate_tasks(
            obs, seat, feasible_plan=self._feasible,
            remaining_sells=self._remaining_sells,
            canonical_board_value=board,
            heuristic_care=self.config.heuristic_care,
            heuristic_fertilizer=self.config.heuristic_fertilizer,
            wheat_harvest_threshold=self.config.wheat_harvest_threshold)
        generated_tasks = generation.sorted_tasks()
        worker_positions = [
            (int(obs["farms"][seat]["farmer"][1]),
             int(obs["farms"][seat]["farmer"][0])),
            *((int(hand[1]), int(hand[0]))
              for hand in obs["farms"][seat].get("hands") or []),
        ]
        self._plant_water_continuations = {
            worker_index: task
            for worker_index, task in self._plant_water_continuations.items()
            if worker_index < len(worker_positions)
            and task.tile == worker_positions[worker_index]
        }
        generated_by_key = {task.key: task for task in generated_tasks}
        self._plant_water_continuations = {
            worker_index: generated_by_key[task.key]
            for worker_index, task in self._plant_water_continuations.items()
            if task.key in generated_by_key
            and generated_by_key[task.key].kind == "WATER"
            and generated_by_key[task.key].source == "water_must_weed_boundary"
        }
        continuation_keys = {
            task.key for task in self._plant_water_continuations.values()
        }
        tasks = tuple(sorted(
            (task for task in generated_tasks if task.key not in continuation_keys),
            key=lambda task: task.sort_key,
        )) + tuple(sorted(
            self._plant_water_continuations.values(),
            key=lambda task: task.sort_key,
        ))
        feed = _animal_feed_state(obs, seat, board=board)
        current_survival_pressure = bool(feed["starving"] or feed["shortage"])
        expansion_suppressed = self._suppress_expansion_today or current_survival_pressure
        if current_survival_pressure:
            suppressed_kinds = _CURRENT_SURVIVAL_SUPPRESSED_TASK_KINDS
        elif self._suppress_expansion_today:
            suppressed_kinds = _PRIOR_DEBT_SUPPRESSED_TASK_KINDS
        else:
            suppressed_kinds = frozenset()
        generated_land_tasks = tuple(
            task for task in tasks if task.kind == "BUY_LAND")
        land_purchase_turn = {
            "requested": bool(
                self._requested is not None
                and self._requested.land_count
                > len(obs["farms"][seat]["unlocked_quadrants"])
            ),
            "task_present": bool(generated_land_tasks),
            "suppressed_prior_debt": bool(
                generated_land_tasks and self._suppress_expansion_today
                and "BUY_LAND" in _PRIOR_DEBT_SUPPRESSED_TASK_KINDS
            ),
            "suppressed_current_survival": bool(
                generated_land_tasks and current_survival_pressure),
            "affordable_before_hires": False,
            "unaffordable_before_hires": False,
            "submitted": False,
            "land_cost": None,
            "cash_before_land": None,
        }
        if suppressed_kinds:
            suppressed_keys = {
                t.key for t in tasks if t.kind in suppressed_kinds
            }
            tasks = tuple(
                t for t in tasks
                if t.kind not in suppressed_kinds
                and not (
                    t.kind == "PLACE"
                    and any(dep in suppressed_keys for dep in t.depends_on)
                )
            )

        visibility = self._derive_starvation_visibility(
            obs, seat, board, tasks, continuation_keys, feed)
        dispatch_tasks = visibility.eligible_tasks
        if feed["starving"] and not self.config.starvation_workload_visibility_repair:
            dispatch_tasks = tuple(
                task for task in tasks if task.tile is None or task.kind == "FEED")
        normal_dispatch_tasks = tuple(
            task for task in dispatch_tasks if task.key not in continuation_keys)
        if self.config.starvation_workload_visibility_repair and feed["starving"]:
            worker_continuations = {
                worker_index: task
                for worker_index, task in self._plant_water_continuations.items()
                if task.key in {item.key for item in visibility.safe_deferred}
            }
        else:
            worker_continuations = (
                {} if feed["starving"] else self._plant_water_continuations
            )

        worker_queues: Mapping[int, Sequence[Task]] | None = None
        if self.config.persistent_worker_queues:
            scheduler_start = time.perf_counter_ns()
            self._last_scheduler_result = self._scheduler.schedule(
                obs, seat, normal_dispatch_tasks,
                worker_count=len(worker_positions),
            )
            scheduler_runtime_ms = (time.perf_counter_ns() - scheduler_start) / 1_000_000
            worker_queues = self._last_scheduler_result.worker_queues
            scheduler_record = self._day_records[day].setdefault(
                "scheduler", {"events": [], "runtime_ms": 0.0, "queue_lengths": {}})
            scheduler_record["events"].extend(self._last_scheduler_result.events)
            scheduler_record["runtime_ms"] += scheduler_runtime_ms
            scheduler_record["queue_lengths"] = {
                str(worker): len(queue)
                for worker, queue in sorted(worker_queues.items())
            }

        # Normal dispatch is deliberately completed in isolation.  Cleanup is
        # a second layer over only literal normal PASS actions.
        normal_foreman = run_foreman(obs, seat, normal_dispatch_tasks,
                                     config=self.config.foreman,
                                     worker_continuations=worker_continuations,
                                     deadline_safe_planting=(
                                          self.config.deadline_safe_planting),
                                      worker_queues=worker_queues,
                                      queue_ownership_repair=(
                                          self.config.queue_ownership_repair
                                          and self.config.persistent_worker_queues),
                                      batch_reserved_supplies=(
                                          self.config.batch_reserved_supplies
                                          and self.config.queue_ownership_repair
                                          and self.config.persistent_worker_queues),
                                      underfoot_queue_insertion=(
                                          self.config.underfoot_queue_insertion
                                          and self.config.queue_ownership_repair
                                          and self.config.persistent_worker_queues),
                                      scheduler_reservations=(
                                          self._last_scheduler_result.reservations
                                          if self.config.queue_ownership_repair
                                          and self.config.persistent_worker_queues
                                          and self._last_scheduler_result is not None
                                          else None))
        optional_tasks: tuple[Task, ...] = ()
        foreman_result = normal_foreman
        if self.config.idle_cleanup_enabled:
            optional_tasks = generate_optional_idle_cleanup_tasks(
                obs, seat, mode=self.config.cleanup_mode,
                canonical_board_value=board)
            foreman_result = apply_idle_cleanup(
                obs, seat, normal_foreman, optional_tasks)
        self._record_cleanup_telemetry(
            normal_foreman, foreman_result, optional_tasks)

        if self.config.queue_ownership_repair and self.config.persistent_worker_queues:
            reconciliation = self._scheduler.reconcile_dispatch(
                seat, foreman_result)
            scheduler_record = self._day_records[day].setdefault(
                "scheduler", {"events": [], "runtime_ms": 0.0, "queue_lengths": {}})
            scheduler_record["events"].extend(reconciliation)
            scheduler_record["events"].extend(
                dict(item) for item in (foreman_result.diagnostics or ()))

        record = self._day_records[day]
        survival_record = record["survival"]
        land_purchase_record = record["land_purchase"]
        for key in (
            "requested", "task_present", "suppressed_prior_debt",
            "suppressed_current_survival",
        ):
            land_purchase_record[key] = bool(
                land_purchase_record[key] or land_purchase_turn[key])
        survival_record["expansion_suppressed_current"] = bool(
            survival_record["expansion_suppressed_current"] or expansion_suppressed)
        if feed["starving"]:
            survival_record["starvation_preemption_turns"] += 1
        if feed["shortage"]:
            survival_record["feed_shortage_turns"] += 1

        sell_candidates = self._sell_candidates(obs, seat, feed=feed)
        money = float(obs["farms"][seat].get("money", 0.0))
        running_cash = money + self._sell_proceeds(obs, sell_candidates)
        unlocked_count = len(obs["farms"][seat]["unlocked_quadrants"])

        from executor_v0 import foreman as _foreman_mod
        all_market_tasks = sorted(
            (t for t in tasks if t.kind in _foreman_mod._MARKET_TASK_KINDS),
            key=lambda t: t.sort_key)
        survival_feed_buys = [
            t for t in all_market_tasks
            if t.kind == "BUY_PRODUCT" and t.product == "WHEAT" and feed["shortage"] > 0
        ]
        land_buys = [t for t in all_market_tasks if t.kind == "BUY_LAND"]
        other_buys = [
            t for t in all_market_tasks
            if t.kind in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")
            and t not in survival_feed_buys
        ]

        candidates = [("sell", c) for c in sell_candidates]
        trace_market_by_payload: dict[int, tuple[str, str]] = {}
        land_candidate_ids: set[int] = set()
        unaffordable_orders = []
        for task in survival_feed_buys:
            op, cost, quantity = self._affordable_survival_feed_buy(obs, task, unlocked_count, running_cash)
            if op is None:
                unaffordable_orders.append({
                    "task": task.key, "cost": self._buy_order_cost(obs, task, unlocked_count),
                    "cash_available": running_cash, "survival": True,
                })
                continue
            if quantity < int(task.quantity):
                survival_record["partial_feed_buys"] += 1
            running_cash -= cost
            candidates.append(("buy", op))
            if self.config.turn_trace:
                trace_market_by_payload[id(op)] = ("survival", task.key)

        for task in land_buys:
            op = self._buy_op(task)
            cost = self._buy_order_cost(obs, task, unlocked_count)
            if op is None or cost is None:
                continue
            land_purchase_turn["land_cost"] = cost
            land_purchase_turn["cash_before_land"] = running_cash
            if running_cash + _MONEY_EPSILON < cost:
                land_purchase_turn["unaffordable_before_hires"] = True
                unaffordable_orders.append({
                    "task": task.key, "cost": cost,
                    "cash_available": running_cash, "survival": False,
                })
                continue
            land_purchase_turn["affordable_before_hires"] = True
            running_cash -= cost
            candidates.append(("buy", op))
            land_candidate_ids.add(id(op))
            if self.config.turn_trace:
                trace_market_by_payload[id(op)] = ("expansion", task.key)

        hire_orders, hires_requested = self._hire_orders(
            obs, seat,
            [t for t in normal_dispatch_tasks
             if t.tile is not None and t.kind in _foreman_mod._TILE_TASK_KINDS],
            running_cash,
            visibility=visibility,
            market_order_limit=max(0, self.config.max_market_orders - len(candidates)),
        )
        already_today = int(obs["farms"][seat].get("hires_today", 0))
        for k, order in enumerate(hire_orders):
            running_cash -= hire_cost(already_today + k, self.config.hire_cost_mult)
            candidates.append(("hire", order))

        for task in other_buys:
            op = self._buy_op(task)
            if op is None:
                continue
            cost = self._buy_order_cost(obs, task, unlocked_count)
            if cost is None:
                continue
            if running_cash + _MONEY_EPSILON < cost:
                unaffordable_orders.append({
                    "task": task.key, "cost": cost, "cash_available": running_cash, "survival": False,
                })
                continue
            running_cash -= cost
            candidates.append(("buy", op))
            if self.config.turn_trace and task.kind in ("BUY_ANIMAL", "BUY_LAND"):
                trace_market_by_payload[id(op)] = ("expansion", task.key)

        included = candidates[:self.config.max_market_orders]
        orders = [payload["order"] if kind == "sell" else payload for kind, payload in included]
        land_purchase_turn["submitted"] = any(
            kind == "buy" and id(payload) in land_candidate_ids
            for kind, payload in included
        )
        for key in (
            "affordable_before_hires", "unaffordable_before_hires", "submitted",
        ):
            land_purchase_record[key] = bool(
                land_purchase_record[key] or land_purchase_turn[key])
        if land_purchase_turn["land_cost"] is not None:
            land_purchase_record["land_cost"] = land_purchase_turn["land_cost"]
            land_purchase_record["cash_before_land"] = land_purchase_turn[
                "cash_before_land"]

        self._commit_sells(obs, day, [payload for kind, payload in included if kind == "sell"])
        if self.config.aggressive_sell_all:
            included_sell_ids = {
                id(payload) for kind, payload in included if kind == "sell"
            }
            bin_log = record["sells"][str(self._bin_anchor)]
            for payload in sell_candidates:
                if id(payload) not in included_sell_ids:
                    bin_log[payload["product"]]["override_skipped"] += payload["executed"]
        submitted_hires = sum(1 for kind, _ in included if kind == "hire")

        for key in ("movement", "interaction", "pickup", "pass"):
            record["foreman_counts"][key] += foreman_result.counts[key]

        optional_keys = {task.key for task in optional_tasks}
        pending = [
            key for key in self._pending_task_keys(foreman_result)
            if key not in optional_keys
        ]
        for key in pending:
            record["pending_task_turns"][key] = record["pending_task_turns"].get(key, 0) + 1
        pending_maintenance = [key for key in pending if key.startswith(("WATER:", "FEED:", "COLLECT_FERTILIZER:"))]
        for key in pending_maintenance:
            record["pending_maintenance_turns"][key] = record["pending_maintenance_turns"].get(key, 0) + 1

        if hour == 23:
            debt = self._end_of_day_debt(tasks, foreman_result)
            record["end_of_day_work_debt"] = debt
            record["unfinished_tasks"] = list(debt["all"])
            record["missed_maintenance"] = list(debt["survival"]) + list(debt["maintenance"])
            record["unfinished_task_turns"] = {key: 1 for key in debt["all"]}
            record["missed_maintenance_turns"] = {key: 1 for key in record["missed_maintenance"]}

        if unaffordable_orders:
            record.setdefault("unaffordable_market_orders", []).extend(unaffordable_orders)
        record["hires"]["requested"] = max(record["hires"]["requested"], hires_requested)
        record["hires"]["submitted"] += submitted_hires
        record["hires"]["observed_max"] = self._max_hires_today
        if self._last_hiring_recommendation is not None:
            record["hiring_recommendation"] = (
                self._last_hiring_recommendation.to_json_dict())
            if (self.config.schedule_informed_hiring
                    and self.config.schedule_hiring_economic_repair):
                farm = obs["farms"][seat]
                raw_observed_hires = farm.get("hires_today", 0)
                observed_hires = (
                    int(raw_observed_hires)
                    if isinstance(raw_observed_hires, int)
                    and not isinstance(raw_observed_hires, bool) else 0)
                record["hiring_decisions"].append({
                    "day": day,
                    "hour": hour,
                    "requested_hires": int(hires_requested),
                    "orders_submitted": int(submitted_hires),
                    "observed_hires_today": observed_hires,
                    "recommendation": (
                        self._last_hiring_recommendation.to_json_dict()),
                })
        if self._last_hire_rejections:
            record.setdefault("hire_rejections", []).extend(
                self._last_hire_rejections)
        if self.config.starvation_workload_visibility_repair:
            visibility_record = self._visibility_json(visibility)
            already_today = int(obs["farms"][seat].get("hires_today", 0))
            visibility_record["hire_hourly_marginal_costs"] = [
                int(hire_cost(already_today + index, self.config.hire_cost_mult))
                for index in range(max(0, hires_requested))
            ]
            completed_interactions = {
                assignment.task_key
                for assignment in foreman_result.assignments
                if assignment.task_key is not None
                and assignment.action
                and assignment.action[0] in _INTERACTION_OPS
            }
            boundary_turn = (
                int(obs.get("hour", 0)) == 23
                or int(obs.get("step", int(day) * 24 + int(hour))) >= 718
            )
            visibility_record["missed_survival_work_at_boundary"] = [
                {
                    "task_key": task.key,
                    "kind": task.kind,
                    "reason": foreman_result.unassigned_reasons.get(
                        task.key, "not_completed_at_boundary"),
                }
                for task in (*visibility.critical_feeds,
                             *visibility.safe_deferred)
                if boundary_turn and task.key not in completed_interactions
            ]
            record["starvation_visibility"] = visibility_record
        record["unresolved_generator"] = list(generation.unresolved)

        crops, animals, care_done, fert_done = _board_counts(board)
        record["achieved_current"] = {
            "crops": crops, "animals": animals,
            "land_count": len(obs["farms"][seat]["unlocked_quadrants"]),
        }
        record["care_completed_observed"] = care_done
        record["fertilizer_completed_observed"] = fert_done

        if self.config.turn_trace:
            try:
                self._record_turn_trace(
                    obs, seat, day, hour, feed, expansion_suppressed,
                    land_purchase_turn, tasks,
                    dispatch_tasks, optional_tasks, foreman_result, pending,
                    trace_market_by_payload, included, unaffordable_orders)
            except Exception:
                # Trace capture is strictly diagnostic and must never turn a
                # successful executor decision into a fallback action.
                pass

        if self.config.record_turn_snapshot:
            try:
                self._debug_trace_turn = self._build_debug_trace_turn(
                    day=day,
                    hour=hour,
                    tasks=tasks,
                    dispatch_tasks=dispatch_tasks,
                    cleanup_tasks=optional_tasks,
                    generated_tasks=generated_tasks,
                    generation=generation,
                    foreman_result=foreman_result,
                    feed=feed,
                    expansion_suppressed=expansion_suppressed,
                    orders=orders,
                    candidates=candidates,
                    unaffordable_orders=unaffordable_orders,
                )
            except Exception:
                # Diagnostics must remain passive even if an unexpected optional
                # value cannot be rendered; the already-decided action is stable.
                self._debug_trace_turn = None
        else:
            self._debug_trace_turn = None

        self._record_plant_attempts(obs, tasks, foreman_result)
        self._last_step = int(obs.get("step", day * 24 + hour))

        return {
            "farmer": list(foreman_result.farmer_action),
            "hands": [list(a) for a in foreman_result.hands_actions],
            "market": orders[:self.config.max_market_orders],
        }

    @property
    def debug_trace_turn(self) -> dict[str, Any] | None:
        """Return a defensive copy of the latest primitive-turn snapshot."""
        return copy.deepcopy(self._debug_trace_turn)

    @property
    def effective_profile(self) -> dict[str, Any] | None:
        """Return the immutable-at-construction executor profile, if any."""
        return copy.deepcopy(self._effective_profile)

    def finalize_diagnostics(self, obs: Mapping, seat: int) -> None:
        """Complete the current day's realized-state diagnostic at terminal."""
        if self._day is None or int(obs["day"]) != self._day:
            return
        farm = obs["farms"][seat]
        board = canonical_board(
            farm["tiles"], int(obs["day"]), int(obs.get("step", 0)))
        crops, animals, care_done, fert_done = _board_counts(board)
        record = self._day_records[self._day]
        record["achieved_final"] = {
            "crops": crops,
            "animals": animals,
            "land_count": len(farm["unlocked_quadrants"]),
        }
        record["care_completed_observed"] = care_done
        record["fertilizer_completed_observed"] = fert_done

    def diagnostics_json(self) -> dict[str, Any]:
        diagnostics = {
            "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
            "seat": self.seat,
            "config": {
                "suppress_expansion_from_prior_debt": (
                    self.config.suppress_expansion_from_prior_debt
                ),
                "aggressive_sell_all": self.config.aggressive_sell_all,
                "optional_idle_cleanup": self.config.idle_cleanup_enabled,
                "optional_spare_watering": self.config.optional_spare_watering,
                "heuristic_care": self.config.heuristic_care,
                "heuristic_fertilizer": self.config.heuristic_fertilizer,
                "wheat_harvest_threshold": self.config.wheat_harvest_threshold,
                "optional_idle_cleanup_mode": self.config.cleanup_mode,
                "cleanup_mode": self.config.cleanup_mode,
                "immediate_plant_water": self.config.immediate_plant_water,
                "deadline_safe_planting": self.config.deadline_safe_planting,
                "deadline_safe_hiring": self.config.deadline_safe_hiring,
                "persistent_worker_queues": self.config.persistent_worker_queues,
                "queue_ownership_repair": self.config.queue_ownership_repair,
                "batch_reserved_supplies": self.config.batch_reserved_supplies,
                "underfoot_queue_insertion": self.config.underfoot_queue_insertion,
                "schedule_informed_hiring": self.config.schedule_informed_hiring,
                "schedule_hiring_economic_repair": (
                    self.config.schedule_hiring_economic_repair),
                "record_turn_snapshot": self.config.record_turn_snapshot,
            },
            "cleanup_metrics": self._cleanup_diagnostics(),
            "days": {str(day): record for day, record in sorted(self._day_records.items())},
            "illegal_actions": {
                "available": False,
                "reason": "the 1.32.7 observation does not expose per-action validity; illegal/ineffective detection requires engine-source instrumentation",
                "count": 0,
            },
            "fallback_errors": [dict(e) for e in self._errors],
        }
        if self._effective_profile is not None:
            diagnostics["effective_profile"] = copy.deepcopy(
                self._effective_profile)
        if self.config.starvation_workload_visibility_repair:
            diagnostics["config"]["starvation_workload_visibility_repair"] = True
        provider_diagnostics = getattr(self.provider, "diagnostics_json", None)
        if callable(provider_diagnostics):
            diagnostics["provider_diagnostics"] = provider_diagnostics()
        return diagnostics


def make_agent(*, provider: PlanProvider | None = None,
               checkpoint: str | None = None, device: str = "cpu",
               seat: int | None = None,
               config: AgentConfig | None = None,
               profile: Mapping[str, Any] | None = None) -> ExecutorAgent:
    if (provider is None) == (checkpoint is None):
        raise ValueError("provide exactly one of provider= or checkpoint=")
    if checkpoint is not None:
        provider = CheckpointPlanProvider(checkpoint, device=device)
    return ExecutorAgent(provider, seat=seat, config=config, profile=profile)


AgentCallable = Callable[[Mapping], dict[str, Any]]
