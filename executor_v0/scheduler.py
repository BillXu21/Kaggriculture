"""Persistent, deterministic task queue scheduling for :class:`Task`.

The foreman is deliberately turn-local.  This module is the small stateful
layer above it: it remembers which worker owns which still-observed task, but
does not execute actions, mutate observations, or perform path finding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from executor_v0.foreman import SHED_ACCESS_TILES
from executor_v0.tasks import Priority, Task

__all__ = [
    "PersistentTaskScheduler",
    "SchedulerConfig",
    "SchedulerResult",
]


_DAYS = 30
_HOURS = 24
_FINAL_ACTIONABLE_STEP = _DAYS * _HOURS - 2
_TILE_KINDS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})
_MARKET_KINDS = frozenset({"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND"})


@dataclass(frozen=True)
class SchedulerConfig:
    """Bounds and mechanics used by the scheduler's cost model."""

    shed_access_tiles: tuple[tuple[int, int], ...] = SHED_ACCESS_TILES
    pickup_batch: int = 5
    max_improvement_pass: int = 1


@dataclass
class SchedulerResult(Mapping[str, Any]):
    """One schedule, with task queues plus JSON-safe diagnostics."""

    queues: dict[int, list[Task]]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    seat: int = 0
    day: int = 0
    reservations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def worker_queues(self) -> dict[int, list[Task]]:
        return self.queues

    @property
    def task_queues(self) -> dict[int, list[Task]]:
        return self.queues

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "seat": self.seat,
            "day": self.day,
            "queues": {
                str(worker): [task.to_json_dict() for task in queue]
                for worker, queue in sorted(self.queues.items())
            },
            "diagnostics": list(self.diagnostics),
            "events": list(self.events),
        }
        if self.reservations:
            payload["reservations"] = {
                str(key): dict(value)
                for key, value in sorted(self.reservations.items())
            }
        return payload

    # Mapping compatibility makes integration code pleasantly unopinionated.
    def __getitem__(self, key: str) -> Any:
        if key == "queues":
            return self.queues
        if key in ("worker_queues", "task_queues"):
            return self.queues
        if key == "diagnostics":
            return self.diagnostics
        if key == "events":
            return self.events
        if key == "seat":
            return self.seat
        if key == "day":
            return self.day
        raise KeyError(key)

    def __iter__(self):
        return iter(("queues", "diagnostics", "events", "seat", "day"))

    def __len__(self) -> int:
        return 5


@dataclass
class _SeatState:
    day: int
    episode: Any
    worker_count: int
    queues: dict[int, list[Task]]
    reservations: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_hour: int = -1


class _Ledger:
    """A copy-on-write-ish resource ledger used during one schedule call."""

    def __init__(self, inventories: list[dict[str, int]], private: Mapping,
                 reservations: Mapping[str, Mapping[str, Any]],
                 pickup_batch: int):
        self.start_inventories = [dict(inv) for inv in inventories]
        self.inventories = [dict(inv) for inv in inventories]
        self.shed = {str(k): max(0, int(v))
                     for k, v in (private.get("shed") or {}).items()}
        self.seeds = {str(k): max(0, int(v))
                      for k, v in (private.get("seeds") or {}).items()}
        self.observed_shed = dict(self.shed)
        self.observed_seeds = dict(self.seeds)
        self.reservations = {str(k): dict(v) for k, v in reservations.items()}
        self.pickup_batch = max(1, int(pickup_batch))
        # Existing commitments reserve observed stock for this call as well.
        for reservation in self.reservations.values():
            if reservation.get("kind") == "seed":
                crop = str(reservation.get("item", ""))
                self.seeds[crop] = self.seeds.get(crop, 0) - int(reservation.get("amount", 0))
            else:
                item = str(reservation.get("item", ""))
                self.shed[item] = self.shed.get(item, 0) - int(reservation.get("amount", 0))

    def clone(self) -> "_Ledger":
        other = object.__new__(_Ledger)
        other.start_inventories = [dict(inv) for inv in self.start_inventories]
        other.inventories = [dict(inv) for inv in self.inventories]
        other.shed = dict(self.shed)
        other.seeds = dict(self.seeds)
        other.observed_shed = dict(self.observed_shed)
        other.observed_seeds = dict(self.observed_seeds)
        other.reservations = {k: dict(v) for k, v in self.reservations.items()}
        other.pickup_batch = self.pickup_batch
        return other

    def release(self, key: str) -> None:
        reservation = self.reservations.pop(key, None)
        if not reservation:
            return
        bucket = self.seeds if reservation.get("kind") == "seed" else self.shed
        item = str(reservation.get("item", ""))
        bucket[item] = bucket.get(item, 0) + int(reservation.get("amount", 0))

    def consume(self, task: Task, worker: int, *, allow_reservation: bool) -> tuple[bool, str]:
        """Reserve resources for one task and update simulated carried stock."""
        if task.kind == "PLANT":
            if not task.crop:
                return False, "malformed_metadata"
            existing = self.reservations.get(task.key)
            if existing and (existing.get("kind") != "seed"
                              or str(existing.get("item")) != str(task.crop)):
                self.release(task.key)
                existing = None
            if allow_reservation and existing and existing.get("kind") == "seed":
                if self.observed_seeds.get(str(task.crop), 0) < int(existing.get("amount", 0)):
                    return False, "no_global_seeds"
                return True, ""
            crop = str(task.crop)
            if self.seeds.get(crop, 0) < 1:
                return False, "no_global_seeds"
            self.seeds[crop] -= 1
            self.reservations[task.key] = {"kind": "seed", "item": crop, "amount": 1}
            return True, ""

        if task.required_item is None:
            if task.key in self.reservations:
                self.release(task.key)
            return True, ""
        item = str(task.required_item)
        demand = max(1, int(task.quantity))
        inventory = self.inventories[worker]
        carried = max(0, int(inventory.get(item, 0)))
        missing = max(0, demand - carried)
        existing = self.reservations.get(task.key)
        if existing and (existing.get("kind") == "seed"
                         or str(existing.get("item")) != item):
            self.release(task.key)
            existing = None
        reserved = (int(existing.get("amount", 0)) if allow_reservation and existing
                    and existing.get("kind") != "seed" and existing.get("item") == item else 0)
        if allow_reservation and existing and reserved != min(missing, reserved):
            self.release(task.key)
            existing = None
            reserved = 0
        if allow_reservation and reserved and self.observed_shed.get(item, 0) < reserved:
            if missing == 0:
                self.release(task.key)
            return False, f"shed_lacks_item:{item}"
        need_from_shed = max(0, missing - reserved)
        if need_from_shed > 0:
            if self.shed.get(item, 0) < need_from_shed:
                return False, f"shed_lacks_item:{item}"
            self.shed[item] -= need_from_shed
        if missing:
            self.reservations[task.key] = {
                "kind": "shed", "item": item, "amount": missing,
            }
        elif existing and allow_reservation:
            self.release(task.key)
        inventory[item] = max(0, carried + missing - demand)
        return True, ""


class PersistentTaskScheduler:
    """Inspectable persistent worker-queue scheduler.

    State is scoped by seat.  A call only observes and plans; it never edits
    ``obs`` or ``tasks``.  ``reset()`` is the explicit episode-reset seam.
    """

    def __init__(self, config: SchedulerConfig | None = None, *,
                 shed_access_tiles: Sequence[Sequence[int]] | None = None,
                 pickup_batch: int | None = None,
                 max_improvement_pass: int | None = None):
        base = config or SchedulerConfig()
        self.config = SchedulerConfig(
            shed_access_tiles=tuple(tuple(int(v) for v in tile)
                                    for tile in (shed_access_tiles or base.shed_access_tiles)),
            pickup_batch=(base.pickup_batch if pickup_batch is None else pickup_batch),
            max_improvement_pass=(base.max_improvement_pass if max_improvement_pass is None
                                  else max_improvement_pass),
        )
        self._states: dict[int, _SeatState] = {}
        self._reset_pending = False

    def reset(self) -> None:
        """Forget all ownership and resource reservations for a new episode."""
        self._states.clear()
        self._reset_pending = True

    @staticmethod
    def _episode(obs: Mapping) -> Any:
        for key in ("episode_id", "episode", "episode_index", "game_id", "reset_id"):
            if key in obs:
                return obs[key]
        return None

    @staticmethod
    def _positions(obs: Mapping, seat: int, count: int) -> list[tuple[int, int]]:
        farm = (obs.get("farms") or [])[seat]
        raw = [farm.get("farmer", [0, 0])]
        raw.extend(farm.get("hands") or [])
        result = []
        for pos in raw[:count]:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                # Engine observations are [x, y]; tasks use [y, x].
                result.append((int(pos[1]), int(pos[0])))
        if not result:
            result = [(0, 0)]
        while len(result) < count:
            result.append(result[-1])
        return result

    @staticmethod
    def _inventories(obs: Mapping, seat: int, count: int) -> list[dict[str, int]]:
        private = obs.get("private") or {}
        values = private.get("inventories") or []
        result = []
        for value in values[:count]:
            result.append({str(k): max(0, int(v)) for k, v in (value or {}).items()})
        while len(result) < count:
            result.append({})
        return result

    @staticmethod
    def _task_map(tasks: Sequence[Task]) -> dict[str, Task]:
        chosen: dict[str, Task] = {}
        for task in tasks:
            if not isinstance(task, Task):
                raise TypeError(f"tasks must contain Task values, got {task!r}")
            # Market orders are assembled after worker dispatch and never
            # belong in a worker's persistent route.
            if task.kind in _MARKET_KINDS:
                continue
            old = chosen.get(task.key)
            if old is None or PersistentTaskScheduler._sort_key(task) < PersistentTaskScheduler._sort_key(old):
                chosen[task.key] = task
        return chosen

    @staticmethod
    def _sort_key(task: Task) -> tuple[int, int, str]:
        return (int(task.priority),
                task.deadline_hour if task.deadline_hour is not None else 1 << 30,
                str(task.key))

    @staticmethod
    def _priority(task: Task) -> int:
        return int(task.priority)

    def _cost(self, position: tuple[int, int], task: Task,
              inventory: Mapping[str, int]) -> int:
        if task.tile is None:
            return 1
        target = (int(task.tile[0]), int(task.tile[1]))
        missing = 0
        if task.required_item is not None:
            missing = max(0, max(1, int(task.quantity)) - int(inventory.get(task.required_item, 0)))
        if missing:
            travel = min(
                abs(position[0] - access[0]) + abs(position[1] - access[1])
                + abs(access[0] - target[0]) + abs(access[1] - target[1])
                for access in self.config.shed_access_tiles
            )
            return travel + ceil(missing / max(1, self.config.pickup_batch)) + 1
        return abs(position[0] - target[0]) + abs(position[1] - target[1]) + 1

    def _queue_cost(self, queue: Sequence[Task], worker: int,
                    positions: Sequence[tuple[int, int]],
                    ledger: _Ledger) -> int:
        position = positions[worker]
        inventory = dict(ledger.start_inventories[worker])
        total = 0
        for task in queue:
            total += self._cost(position, task, inventory)
            if task.tile is not None:
                position = (int(task.tile[0]), int(task.tile[1]))
            if task.required_item is not None:
                item = str(task.required_item)
                demand = max(1, int(task.quantity))
                inventory[item] = max(0, int(inventory.get(item, 0)) - demand)
        return total

    @staticmethod
    def _capacity(obs: Mapping) -> tuple[int, int]:
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", day * _HOURS + hour))
        daily = max(0, _HOURS - hour)
        terminal = max(0, _FINAL_ACTIONABLE_STEP + 1 - step)
        return daily, terminal

    @staticmethod
    def _runtime_failure(obs: Mapping) -> bool:
        for key in ("last_action_result", "action_result", "runtime", "last_runtime"):
            value = obs.get(key)
            if isinstance(value, Mapping):
                if value.get("failed") is True or value.get("success") is False or value.get("ok") is False:
                    return True
            elif isinstance(value, str) and value.lower() in {"failed", "failure", "error"}:
                return True
        return bool(obs.get("action_failed") is True)

    def schedule(self, obs: Mapping, seat: int, tasks: Sequence[Task],
                 worker_count: int) -> SchedulerResult:
        if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        if not isinstance(obs, Mapping) or "farms" not in obs:
            raise ValueError("obs must contain farms")
        if not 0 <= int(seat) < len(obs["farms"]):
            raise ValueError(f"invalid seat {seat!r}")
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        task_map = self._task_map(tasks)
        positions = self._positions(obs, seat, worker_count)
        inventories = self._inventories(obs, seat, worker_count)
        actual_count = 1 + len((obs["farms"][seat].get("hands") or []))
        episode = self._episode(obs)
        events: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []

        old = self._states.get(int(seat))
        reset_reason = None
        if self._reset_pending:
            reset_reason = "explicit_reset"
            self._reset_pending = False
        elif old is not None:
            if old.day != day:
                reset_reason = "day_change"
            elif old.worker_count != worker_count or actual_count != old.worker_count:
                reset_reason = "worker_count_mismatch"
            elif old.episode != episode and episode is not None:
                reset_reason = "episode_change"
            elif old.last_hour >= 0 and hour < old.last_hour:
                reset_reason = "hour_regression_episode_reset"
        if reset_reason:
            events.append({"event": "runtime", "reason": reset_reason,
                           "seat": int(seat), "day": day})
            old = None

        state = _SeatState(day=day, episode=episode, worker_count=worker_count,
                           queues={i: [] for i in range(worker_count)},
                           reservations=(dict(old.reservations) if old else {}),
                           last_hour=hour)
        ledger = _Ledger(inventories, obs.get("private") or {}, state.reservations,
                         self.config.pickup_batch)
        seen: set[str] = set()
        active_keys = set(task_map)
        daily_capacity, terminal_capacity = self._capacity(obs)
        capacity = min(daily_capacity, terminal_capacity)

        def emit(event: str, task: Task | None = None, **extra: Any) -> None:
            payload: dict[str, Any] = {"event": event}
            if task is not None:
                payload["task_key"] = str(task.key)
            payload.update(extra)
            events.append(payload)

        # Reconcile old queues in ownership order.  A task that vanishes is
        # completed/disappeared; a still-current task retains its worker.
        for worker in range(worker_count):
            for task in ((old.queues.get(worker, []) if old else [])):
                if task.key in seen:
                    emit("release", task, worker_index=worker,
                         reason="duplicate_key",
                         prevented_conflicting_claim=True)
                    continue
                seen.add(task.key)
                current = task_map.get(task.key)
                if current is None:
                    emit("release", task, worker_index=worker, reason="not_current")
                    ledger.release(task.key)
                    continue
                blocked = next((dep for dep in current.depends_on if dep in active_keys), None)
                if blocked is not None:
                    emit("repair", current, worker_index=worker,
                         reason=f"dependency_current:{blocked}")
                    ledger.release(current.key)
                    seen.discard(current.key)
                    continue
                if current.tile is None and current.kind not in _MARKET_KINDS:
                    emit("repair", current, worker_index=worker, reason="no_actionable_target")
                    ledger.release(current.key)
                    seen.discard(current.key)
                    continue
                resource_ok, resource_reason = ledger.consume(
                    current, worker, allow_reservation=True)
                if not resource_ok:
                    reason = resource_reason
                    emit("repair", current, worker_index=worker, reason=reason)
                    ledger.release(current.key)
                    seen.discard(current.key)
                    continue
                tentative = state.queues[worker] + [current]
                if self._queue_cost(tentative, worker, positions, ledger) > capacity:
                    emit("repair", current, worker_index=worker,
                         reason="predicted_completion_failure", phase="completion")
                    ledger.release(current.key)
                    seen.discard(current.key)
                    emit("predicted_failure", current, worker_index=worker,
                         phase="completion", required=self._queue_cost(tentative, worker, positions, ledger),
                         remaining=capacity)
                    continue
                state.queues[worker].append(current)
                emit("retain", current, worker_index=worker, reason="current_and_feasible")
                emit("ownership", current, worker_index=worker, reason="preserved")

        if self._runtime_failure(obs):
            for worker, queue in state.queues.items():
                if queue:
                    emit("runtime", queue[0], worker_index=worker,
                         reason="failed_action_still_current")

        def dependency(task: Task) -> str | None:
            return next((dep for dep in task.depends_on if dep in active_keys), None)

        def try_assignment(task: Task, worker: int, *, urgent: bool) -> tuple[bool, _Ledger, int, str]:
            clone = ledger.clone()
            queue = state.queues[worker]
            prepend = urgent and bool(queue) and self._priority(queue[0]) > self._priority(task)
            if prepend:
                # The task has not consumed anything yet in this call; use the
                # worker's actual carried stock when preempting its current head.
                clone.inventories[worker] = dict(clone.start_inventories[worker])
            ok, reason = clone.consume(task, worker, allow_reservation=False)
            if not ok:
                return False, clone, 0, reason
            candidate_queue = ([task] + queue) if prepend else (queue + [task])
            cost = self._queue_cost(candidate_queue, worker, positions, clone)
            if cost > capacity:
                phase = "arrival" if task.tile is not None and self._cost(positions[worker], task, clone.start_inventories[worker]) > capacity else "completion"
                return False, clone, cost, f"predicted_{phase}_failure"
            if task.deadline_hour is not None and day == int(obs.get("day", day)):
                deadline_budget = max(0, int(task.deadline_hour) - hour + 1)
                task_cost = self._cost(positions[worker], task, clone.start_inventories[worker])
                if task_cost > deadline_budget:
                    return False, clone, task_cost, "predicted_deadline_failure"
            extension = self._cost(
                positions[worker] if not queue else
                ((queue[-1].tile[0], queue[-1].tile[1]) if queue[-1].tile is not None else positions[worker]),
                task, clone.inventories[worker])
            return True, clone, extension, ""

        pending = [task for task in sorted(task_map.values(), key=self._sort_key)
                   if task.key not in seen]
        deferred: list[Task] = []
        newly_assigned: set[str] = set()

        def assign(task: Task, urgent: bool) -> bool:
            blocked = dependency(task)
            if blocked is not None:
                emit("release", task, reason=f"dependency_current:{blocked}")
                return False
            if task.tile is None and task.kind not in _MARKET_KINDS:
                emit("release", task, reason="no_actionable_target")
                return False
            choices = []
            for worker in range(worker_count):
                ok, clone, score, reason = try_assignment(task, worker, urgent=urgent)
                if ok:
                    choices.append((score, worker, clone))
                elif reason.startswith("predicted_"):
                    phase = reason.removeprefix("predicted_").removesuffix("_failure")
                    emit("predicted_failure", task, worker_index=worker, phase=phase,
                         remaining=capacity, required=score)
            if not choices:
                if urgent:
                    # Make room locally for urgent maintenance.  The
                    # displaced tail remains eligible for the ordinary pass.
                    victims = [
                        (self._priority(queue[-1]), worker, queue[-1])
                        for worker, queue in state.queues.items()
                        if queue and self._priority(queue[-1]) > self._priority(task)
                    ]
                    if victims:
                        _priority, victim_worker, victim = max(
                            victims, key=lambda value: (value[0], -value[1], str(value[2].key)))
                        state.queues[victim_worker].pop()
                        seen.discard(victim.key)
                        deferred.append(victim)
                        ledger.release(victim.key)
                        emit("preempt", task, worker_index=victim_worker,
                             displaced_task_key=str(victim.key), reason="urgent_maintenance")
                        return assign(task, urgent=True)
                # Resource failures are honest diagnostics; the task is not
                # silently converted into a queue commitment.
                reason = "no_feasible_worker"
                for worker in range(worker_count):
                    ok, _clone, _score, candidate_reason = try_assignment(task, worker, urgent=urgent)
                    if not ok and candidate_reason.startswith(("no_", "shed_")):
                        reason = candidate_reason
                        break
                emit("release", task, reason=reason)
                return False
            _score, worker, chosen_ledger = min(choices, key=lambda value: (value[0], value[1]))
            existing = bool(state.queues[worker])
            if urgent and existing and self._priority(state.queues[worker][0]) > self._priority(task):
                state.queues[worker].insert(0, task)
                emit("preempt", task, worker_index=worker,
                     displaced_task_key=str(state.queues[worker][1].key), reason="urgent_maintenance")
            else:
                state.queues[worker].append(task)
            ledger.inventories = chosen_ledger.inventories
            ledger.shed = chosen_ledger.shed
            ledger.seeds = chosen_ledger.seeds
            ledger.reservations = chosen_ledger.reservations
            newly_assigned.add(task.key)
            emit("ownership", task, worker_index=worker, reason="greedy_least_extension")
            return True

        # Maintenance is its own pass so an urgent task can move ahead of a
        # persistent productive/manager commitment.
        for task in [t for t in pending if self._priority(t) == int(Priority.MAINTENANCE)]:
            if not assign(task, urgent=True):
                continue
            seen.add(task.key)
        for task in [t for t in pending + deferred
                     if self._priority(t) != int(Priority.MAINTENANCE)]:
            if task.key in seen:
                continue
            if assign(task, urgent=False):
                seen.add(task.key)

        # One deliberately bounded local improvement pass.  Only tasks first
        # assigned during this call are eligible, so valid persistent owners
        # remain stable across calls.  Resource-bearing tasks stay put because
        # moving them would require replaying the shed ledger.
        for _ in range(max(0, min(1, int(self.config.max_improvement_pass)))):
            moves = []
            for source, source_queue in state.queues.items():
                for index, candidate in enumerate(source_queue):
                    if candidate.key not in newly_assigned:
                        continue
                    if candidate.required_item is not None or candidate.kind == "PLANT":
                        continue
                    if self._priority(candidate) == int(Priority.MAINTENANCE):
                        continue
                    without = source_queue[:index] + source_queue[index + 1:]
                    before = self._queue_cost(source_queue, source, positions, ledger)
                    for destination in range(worker_count):
                        if destination == source:
                            continue
                        destination_queue = state.queues[destination]
                        insert_at = next(
                            (i for i, other in enumerate(destination_queue)
                             if self._sort_key(other) > self._sort_key(candidate)),
                            len(destination_queue),
                        )
                        with_candidate = (destination_queue[:insert_at]
                                          + [candidate]
                                          + destination_queue[insert_at:])
                        after = self._queue_cost(without, source, positions, ledger) \
                            + self._queue_cost(with_candidate, destination, positions, ledger)
                        if after < before + self._queue_cost(destination_queue, destination, positions, ledger) \
                                and self._queue_cost(with_candidate, destination, positions, ledger) <= capacity:
                            moves.append((before + self._queue_cost(destination_queue, destination, positions, ledger)
                                          - after, str(candidate.key), source, destination, index, insert_at))
            if not moves:
                break
            _gain, _key, source, destination, index, insert_at = min(
                moves, key=lambda move: (-move[0], move[1], move[2], move[3]))
            candidate = state.queues[source].pop(index)
            state.queues[destination].insert(insert_at, candidate)
            emit("ownership", candidate, worker_index=destination,
                 previous_worker_index=source, reason="bounded_improvement")

        # Report current tasks that were kept out of queues, including useful
        # prediction diagnostics for callers that want to explain a PASS.
        assigned = {task.key for queue in state.queues.values() for task in queue}
        for task in sorted(task_map.values(), key=self._sort_key):
            if task.key not in assigned and not any(e.get("task_key") == task.key for e in events):
                emit("release", task, reason="not_selected")
        state.reservations = ledger.reservations
        state.last_hour = hour
        self._states[int(seat)] = state
        diagnostics.extend(dict(event) for event in events)
        return SchedulerResult(
            queues={worker: list(queue) for worker, queue in sorted(state.queues.items())},
            diagnostics=diagnostics,
            events=list(events), seat=int(seat), day=day,
            reservations={key: dict(value)
                          for key, value in sorted(state.reservations.items())},
        )

    def reconcile_dispatch(
        self,
        seat: int,
        foreman_result: Any | None = None,
        *,
        completed_task_keys: Sequence[str] = (),
        released_task_keys: Sequence[str] = (),
        transfers: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Reconcile persistent ownership after one actual foreman turn.

        ``schedule`` is intentionally observational, while the engine applies
        the returned actions afterward.  The caller should invoke this seam
        with the actual :class:`~executor_v0.foreman.ForemanResult` so an
        interaction removes its completed head and an opt-in foreman transfer
        moves the still-live task to its new owner before the next schedule.
        Explicit key lists are accepted for integrations that do not retain a
        ForemanResult.  The method mutates only scheduler state and returns a
        bounded, JSON-safe event list.
        """
        state = self._states.get(int(seat))
        if state is None:
            return []

        events: list[dict[str, Any]] = []
        completed = {str(key) for key in completed_task_keys}
        released = {str(key) for key in released_task_keys}
        transfer_events: list[Mapping[str, Any]] = [
            value for value in transfers if isinstance(value, Mapping)
        ]

        if foreman_result is not None:
            assignments = getattr(foreman_result, "assignments", ())
            for assignment in assignments:
                action = tuple(getattr(assignment, "action", ()) or ())
                task_key = getattr(assignment, "task_key", None)
                if task_key is not None and action and action[0] in _TILE_KINDS:
                    completed.add(str(task_key))
            for diagnostic in getattr(foreman_result, "diagnostics", ()) or ():
                if not isinstance(diagnostic, Mapping):
                    continue
                event = str(diagnostic.get("event", ""))
                task_key = diagnostic.get("task_key")
                if task_key is None:
                    continue
                if event in {"queue_release", "release"}:
                    released.add(str(task_key))
                elif event in {"queue_transfer", "transfer"}:
                    transfer_events.append(diagnostic)

        def remove_key(task_key: str) -> list[int]:
            owners: list[int] = []
            for worker, queue in state.queues.items():
                kept = []
                for task in queue:
                    if str(task.key) == task_key:
                        owners.append(worker)
                    else:
                        kept.append(task)
                state.queues[worker] = kept
            return owners

        for task_key in sorted(completed | released):
            owners = remove_key(task_key)
            if task_key in state.reservations:
                state.reservations.pop(task_key, None)
            events.append({
                "event": "reconcile",
                "task_key": task_key,
                "reason": "completed" if task_key in completed else "released",
                "worker_indices": owners,
            })

        for transfer in transfer_events:
            task_key = transfer.get("task_key")
            new_worker = transfer.get("new_worker_index",
                                      transfer.get("worker_index"))
            if task_key is None or new_worker is None:
                continue
            task_key = str(task_key)
            try:
                new_worker = int(new_worker)
            except (TypeError, ValueError):
                continue
            if not 0 <= new_worker < state.worker_count or task_key in completed:
                continue
            found = next(
                (task for queue in state.queues.values() for task in queue
                 if str(task.key) == task_key),
                None,
            )
            remove_key(task_key)
            if found is None:
                # A transferred task is normally still present in the queue
                # snapshot.  Keep this event auditable even if a caller also
                # supplied an explicit completion/release for it.
                events.append({
                    "event": "reconcile",
                    "task_key": task_key,
                    "new_worker_index": new_worker,
                    "reason": "transfer_without_live_task",
                })
                continue
            state.queues[new_worker].insert(0, found)
            events.append({
                "event": "reconcile",
                "task_key": task_key,
                "new_worker_index": new_worker,
                "reason": "transferred",
            })

        state.reservations = {
            str(key): dict(value) for key, value in state.reservations.items()
        }
        return events[:128]
