"""Deterministic greedy foreman for worker task execution (issue #1 section 6).

One primitive turn per call: underfoot execution first, then greedy
assignment, soft inventory specialization, and exactly one legal Manhattan
step or interaction per worker. No search/VRP/pathfinder; routes are
recomputed by the caller every turn.

Shed access mechanic — replay-derived evidence: in elite 1.32.7 replays
(local sample episode 94735084), all 269 observed PICKUP/DROP events (both
seats) occur at one of the four center tiles ``[x,y] in
{(4,4),(5,4),(4,5),(5,5)}``, i.e. ``[y,x]`` coords
``{(4,4),(4,5),(5,4),(5,5)}``. These tiles are therefore overwhelmingly
observed to be valid pickup/drop locations; the sample does not prove they
are the *only* legal ones. They are exposed as `SHED_ACCESS_TILES` and
configurable via `ForemanConfig.shed_access_tiles`.

Worker op encoding — replay-verified: every worker action is a list;
bare ops are single-element lists (`["WATER"]`, `["NORTH"]`, `["PASS"]`);
argument ops append strings/ints (`["PLANT","WHEAT"]`,
`["PICKUP","FERTILIZER",3]`, `["PLACE","COW",1]`). Movement deltas:
NORTH dy=-1, SOUTH dy=+1, EAST dx=+1, WEST dx=-1.

Seed mechanic (1.32.7): `PLANT <crop>` consumes the GLOBAL own
``private.seeds[crop]`` pool atomically at the engine; seeds are never
picked up or carried, and crop items in worker inventories are products,
not seeds. The foreman therefore never routes PLANT work through the shed:
it checks and reserves global seeds per crop within the turn (deterministic
across workers), and blocked PLANT tasks stay unassigned with an honest
``no_global_seeds`` reason. Seed shortages surface as BUY_SEED market tasks
generated upstream.

Movement legality is exact: every in-bounds tile is enterable (engine
movement ops are unconditional), so a Manhattan-reducing step always exists
toward any in-bounds target and workers never dead-end into PASS en route.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from executor_v0.tasks import Task

__all__ = [
    "SHED_ACCESS_TILES",
    "ForemanConfig",
    "WorkerView",
    "Assignment",
    "ForemanResult",
    "apply_idle_cleanup",
    "run_foreman",
]

# [y, x] center tiles adjacent to the central shed; see module docstring.
SHED_ACCESS_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))

_TILE_TASK_KINDS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})
_MARKET_TASK_KINDS = frozenset({
    "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND",
})

_PRIORITY_STEP = 100_000     # priority dominates any distance
_VARIETY_PENALTY = 50_000    # adding a new carried item type past the cap
_CARRY_AFFINITY_BONUS = 10   # soft bonus when the required item is carried
_TOTAL_ACTIONS = 30 * 24 - 1  # step 718 is the final actionable step


@dataclass(frozen=True)
class ForemanConfig:
    max_carried_item_types: int = 2
    pickup_batch: int = 5
    shed_access_tiles: tuple[tuple[int, int], ...] = SHED_ACCESS_TILES
    underfoot_first: bool = False


@dataclass(frozen=True)
class WorkerView:
    index: int              # 0 = farmer, then hands in observation order
    position: tuple[int, int]          # canonical [y, x]
    inventory: dict[str, int]
    item_types: tuple[str, ...]        # sorted carried item names

    @property
    def item_type_count(self) -> int:
        return len(self.item_types)


@dataclass(frozen=True)
class Assignment:
    worker_index: int
    task_key: str | None       # None for PASS with no claim
    action: tuple             # engine op list, e.g. ("WATER",)
    reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "task_key": self.task_key,
            "action": list(self.action),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ForemanResult:
    farmer_action: tuple
    hands_actions: tuple[tuple, ...]
    assignments: tuple[Assignment, ...]
    unassigned_tile_tasks: tuple[Task, ...]
    market_tasks: tuple[Task, ...]
    counts: dict[str, int]
    unassigned_reasons: dict[str, str] = field(default_factory=dict)
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "farmer_action": list(self.farmer_action),
            "hands_actions": [list(a) for a in self.hands_actions],
            "assignments": [a.to_json_dict() for a in self.assignments],
            "unassigned_tile_tasks": [t.key
                                      for t in self.unassigned_tile_tasks],
            "market_tasks": [t.to_json_dict() for t in self.market_tasks],
            "counts": dict(self.counts),
            "unassigned_reasons": dict(self.unassigned_reasons),
        }
        if self.diagnostics:
            payload["diagnostics"] = [dict(item) for item in self.diagnostics]
        return payload


# ------------------------------------------------------------------ helpers


def _worker_views(obs: Mapping, seat: int) -> list[WorkerView]:
    """Farmer index 0 then hands in observation order; positions [x,y]->[y,x]."""
    farm = obs["farms"][seat]
    private = obs.get("private") or {}
    inventories = [dict(inv) for inv in (private.get("inventories") or [])]
    positions = [list(farm["farmer"])]
    positions.extend(list(h) for h in (farm.get("hands") or []))
    workers: list[WorkerView] = []
    for index, pos in enumerate(positions):
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError(
                f"worker {index} position must be [x, y], got {pos!r}")
        x, y = int(pos[0]), int(pos[1])
        inventory = inventories[index] if index < len(inventories) else {}
        workers.append(WorkerView(
            index=index, position=(y, x),
            inventory={str(k): int(v) for k, v in inventory.items()},
            item_types=tuple(sorted(str(k) for k, v in inventory.items()
                                    if int(v) > 0))))
    return workers


def _carried(worker: WorkerView, item: str | None) -> bool:
    return item is None or worker.inventory.get(item, 0) > 0


def _shed_available(obs: Mapping, seat: int, item: str) -> int:
    """Pickupable shed stock for one carried item (never seeds: planting
    consumes the global ``private.seeds`` pool atomically at the engine)."""
    private = obs.get("private") or {}
    return max(0, int((private.get("shed") or {}).get(item, 0)))


def _legal_step(board, unlocked_quadrants, frm: tuple[int, int],
                to: tuple[int, int]) -> bool:
    """Exact 1.32.7 movement legality: any in-bounds tile is enterable.

    Engine evidence (rust/kaggriculture_env/src/lib.rs ``apply_unit_action``):
    movement operations 1..=4 are exempt from the unlocked-quadrant guard that
    silences other operations on locked tiles, and nothing else gates
    ``move_position`` -- weeds, structures, locked quadrants, and other
    workers never block walking; only the board edge stops a step. The
    previous conservative refusal to path through locked quadrants was not an
    engine rule and made legal routes look impossible (issue #7).
    """
    y, x = to
    return 0 <= y < 10 and 0 <= x < 10


def _step_toward(board, unlocked_quadrants, pos: tuple[int, int],
                 target: tuple[int, int]) -> tuple[str, tuple[int, int]] | None:
    """One legal cardinal step reducing Manhattan distance.

    Deterministic axis order: reduce the larger delta first; exact ties
    reduce y (vertical) first. Returns None when both candidate steps are
    illegal (caller emits PASS).
    """
    dy = target[0] - pos[0]
    dx = target[1] - pos[1]
    vertical_first = abs(dy) >= abs(dx)
    candidates: list[tuple[str, tuple[int, int]]] = []
    if dy > 0:
        candidates.append(("SOUTH", (pos[0] + 1, pos[1])))
    elif dy < 0:
        candidates.append(("NORTH", (pos[0] - 1, pos[1])))
    if dx > 0:
        candidates.append(("EAST", (pos[0], pos[1] + 1)))
    elif dx < 0:
        candidates.append(("WEST", (pos[0], pos[1] - 1)))
    candidates.sort(key=lambda c: 0 if (
        (c[0] in ("SOUTH", "NORTH")) == vertical_first) else 1)
    for name, dest in candidates:
        if _legal_step(board, unlocked_quadrants, pos, dest):
            return name, dest
    return None


def _interaction_op(task: Task) -> tuple | None:
    """Exact engine op list for one tile task; None on malformed metadata."""
    kind = task.kind
    if kind in ("WATER", "HARVEST", "DIG", "BUILD_COOP", "BUILD_PASTURE",
                "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"):
        return (kind,)
    if kind == "PLANT":
        return (kind, task.crop) if task.crop else None
    if kind == "PLACE":
        return (kind, task.animal, task.quantity) \
            if task.animal and task.quantity > 0 else None
    return None


def _route_distance(pos: tuple[int, int], tile: tuple[int, int],
                    config: ForemanConfig) -> int:
    best = None
    for access in config.shed_access_tiles:
        total = abs(pos[0] - access[0]) + abs(pos[1] - access[1]) \
            + abs(access[0] - tile[0]) + abs(access[1] - tile[1])
        best = total if best is None else min(best, total)
    return best if best is not None else 0


def _nearest_access(pos: tuple[int, int],
                    config: ForemanConfig) -> tuple[int, int]:
    return min(config.shed_access_tiles,
               key=lambda a: (abs(pos[0] - a[0]) + abs(pos[1] - a[1]), a))


def _remaining_actionable_turns(obs: Mapping) -> int:
    """Return inclusive worker actions before reset or terminal state."""
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", int(obs.get("day", 0)) * 24 + hour))
    return max(0, min(24 - hour, _TOTAL_ACTIONS - step))


def _plant_deadline_feasible(
    obs: Mapping,
    worker: WorkerView,
    task: Task,
    *,
    enabled: bool,
    remaining_turns: int,
) -> tuple[bool, str]:
    """Check travel + PLANT + protected same-worker WATER budget."""
    if not enabled or task.kind != "PLANT" or task.tile is None:
        return True, ""
    travel = abs(worker.position[0] - task.tile[0]) \
        + abs(worker.position[1] - task.tile[1])
    required = travel + 2
    if required <= remaining_turns:
        return True, ""
    return False, (
        f"plant_deadline_insufficient_turns:required={required}:"
        f"remaining={remaining_turns}:travel={travel}"
    )


# ------------------------------------------------------------------- foreman


def run_foreman(
    obs: Mapping,
    seat: int,
    tasks: Sequence[Task],
    *,
    config: ForemanConfig = ForemanConfig(),
    worker_continuations: Mapping[int, Task] | None = None,
    deadline_safe_planting: bool = False,
    worker_queues: Mapping[int, Sequence[Task]] | None = None,
    queue_ownership_repair: bool = False,
    scheduler_reservations: Mapping[str, Mapping[str, Any]] | None = None,
) -> ForemanResult:
    """Run one greedy dispatch turn. Pure; inputs never mutated."""
    farm = obs["farms"][seat]
    board = farm["tiles"]
    unlocked = list(farm["unlocked_quadrants"])
    workers = _worker_views(obs, seat)
    remaining_turns = _remaining_actionable_turns(obs)

    all_tasks = list(tasks)
    dep_keys = {t.key for t in all_tasks}
    tile_tasks = []
    market_tasks = []
    unassigned: list[Task] = []
    unassigned_reasons: dict[str, str] = {}
    for t in all_tasks:
        if t.kind in _MARKET_TASK_KINDS:
            market_tasks.append(t)
        elif t.kind in _TILE_TASK_KINDS:
            # A dependency still present in this turn's set blocks the task;
            # it is surfaced as unassigned for diagnostics.
            blocked_by = next((dep for dep in t.depends_on
                               if dep in dep_keys), None)
            if blocked_by is not None:
                unassigned.append(t)
                unassigned_reasons.setdefault(
                    t.key, f"dependency_blocked:{blocked_by}")
                continue
            tile_tasks.append(t)
        else:
            continue  # unknown kinds are never dispatched
    tile_tasks.sort(key=lambda t: t.sort_key)
    market_tasks.sort(key=lambda t: t.sort_key)
    tile_by_key = {task.key: task for task in tile_tasks}

    # The repair is deliberately a separate opt-in layer.  In particular,
    # the legacy path below keeps its iteration order, claims, and RNG-free
    # decisions unchanged when no persistent queues (or no flag) are present.
    repair = bool(queue_ownership_repair and worker_queues is not None)
    queue_diagnostics: list[dict[str, Any]] = []
    queue_by_worker: dict[int, list[Task]] = {
        worker.index: [] for worker in workers
    }
    queue_owner: dict[str, int] = {}
    queue_task_by_key: dict[str, Task] = {}
    queue_blocked: set[str] = set()
    transferred_keys: set[str] = set()

    def queue_diag(event: str, *, task_key: str | None = None,
                   **extra: Any) -> None:
        if len(queue_diagnostics) >= 128:
            return
        payload: dict[str, Any] = {"event": event}
        if task_key is not None:
            payload["task_key"] = str(task_key)
        payload.update(extra)
        queue_diagnostics.append(payload)

    if repair:
        for worker in workers:
            for queued in (worker_queues or {}).get(worker.index, ()):
                key = str(getattr(queued, "key", ""))
                current = tile_by_key.get(getattr(queued, "key", None))
                if current is None:
                    queue_diag("queue_release", task_key=key,
                               worker_index=worker.index, reason="not_current")
                    continue
                if key in queue_owner:
                    queue_diag(
                        "prevented_conflicting_claim", task_key=key,
                        old_worker_index=queue_owner[key],
                        new_worker_index=worker.index,
                        reason="duplicate_queue_reservation",
                    )
                    continue
                blocked_by = next((dep for dep in current.depends_on
                                   if dep in dep_keys), None)
                if blocked_by is not None:
                    queue_blocked.add(key)
                    queue_diag("queue_release", task_key=key,
                               worker_index=worker.index,
                               reason=f"dependency_current:{blocked_by}")
                    continue
                queue_owner[key] = worker.index
                queue_task_by_key[key] = current
                queue_by_worker[worker.index].append(current)

    # Global own seed pool: planting consumes `private.seeds[crop]`
    # atomically at the engine. Workers never PICKUP or carry seeds; this
    # per-turn budget reserves seeds deterministically across workers so we
    # never emit more PLANT actions for a crop than the global stock.
    private = obs.get("private") or {}
    seed_budget = {str(k): int(v)
                   for k, v in (private.get("seeds") or {}).items()}
    shed_budget = {str(k): max(0, int(v))
                   for k, v in (private.get("shed") or {}).items()
                   if int(v) > 0}

    # Scheduler reservations are part of the repaired ownership contract:
    # incidental pickups may use only stock left after exclusive queue work.
    # A queued task receives its own reservation back when it is dispatched.
    reserved_shed: dict[str, int] = {}
    reserved_seeds: dict[str, int] = {}
    reservation_by_key: dict[str, dict[str, Any]] = {}
    if repair and scheduler_reservations:
        for raw_key, raw_reservation in scheduler_reservations.items():
            key = str(raw_key)
            if key not in queue_owner or not isinstance(raw_reservation, Mapping):
                continue
            reservation = dict(raw_reservation)
            reservation_by_key[key] = reservation
            amount = max(0, int(reservation.get("amount", 0)))
            item = str(reservation.get("item", ""))
            if reservation.get("kind") == "seed":
                reserved_seeds[item] = reserved_seeds.get(item, 0) + amount
            else:
                reserved_shed[item] = reserved_shed.get(item, 0) + amount
        for item, amount in reserved_shed.items():
            shed_budget[item] = max(0, shed_budget.get(item, 0) - amount)
        for crop, amount in reserved_seeds.items():
            seed_budget[crop] = max(0, seed_budget.get(crop, 0) - amount)

    released_reservations: set[str] = set()

    def release_queue_reservation(task_: Task) -> None:
        key = str(task_.key)
        if key in released_reservations:
            return
        reservation = reservation_by_key.get(key)
        if reservation is None:
            return
        amount = max(0, int(reservation.get("amount", 0)))
        item = str(reservation.get("item", ""))
        if reservation.get("kind") == "seed":
            seed_budget[item] = seed_budget.get(item, 0) + amount
        else:
            shed_budget[item] = shed_budget.get(item, 0) + amount
        released_reservations.add(key)

    def queue_shed_available(task_: Task) -> int:
        item = task_.required_item
        if item is None:
            return 0
        available = shed_budget.get(str(item), 0)
        if str(task_.key) in queue_owner and str(task_.key) not in released_reservations:
            reservation = reservation_by_key.get(str(task_.key))
            if reservation and reservation.get("kind") != "seed":
                available += max(0, int(reservation.get("amount", 0)))
        return available

    def queue_seeds_available(task_: Task) -> bool:
        if task_.kind != "PLANT" or not task_.crop:
            return True
        available = seed_budget.get(str(task_.crop), 0)
        if str(task_.key) in queue_owner and str(task_.key) not in released_reservations:
            reservation = reservation_by_key.get(str(task_.key))
            if reservation and reservation.get("kind") == "seed":
                available += max(0, int(reservation.get("amount", 0)))
        return available > 0

    def seeds_available(task_: Task) -> bool:
        if task_.kind != "PLANT":
            return True
        if not task_.crop:
            return True  # malformed metadata is handled at execution time
        return seed_budget.get(task_.crop, 0) > 0

    def reserve_seeds(task_: Task) -> None:
        if task_.kind == "PLANT" and task_.crop:
            seed_budget[task_.crop] = seed_budget.get(task_.crop, 0) - 1

    def refund_seeds(task_: Task) -> None:
        if task_.kind == "PLANT" and task_.crop:
            seed_budget[task_.crop] = seed_budget.get(task_.crop, 0) + 1

    claimed: set[str] = set()
    assignments: list[Assignment] = []
    counts = {"movement": 0, "interaction": 0, "pickup": 0, "pass": 0}
    actions: list[tuple] = []

    def release(task_: Task) -> None:
        """Return a claimed task to the unassigned pool exactly once."""
        refund_seeds(task_)
        claimed.discard(task_.key)
        if all(u.key != task_.key for u in unassigned):
            unassigned.append(task_)

    def repaired_underfoot_ok(worker: WorkerView, task: Task) -> bool:
        """Apply the ordinary underfoot/resource/deadline gates."""
        if task.key in claimed or task.tile != worker.position:
            return False
        if _interaction_op(task) is None or not queue_seeds_available(task):
            return False
        deadline_ok, deadline_reason = _plant_deadline_feasible(
            obs, worker, task, enabled=deadline_safe_planting,
            remaining_turns=remaining_turns)
        if not deadline_ok:
            unassigned_reasons.setdefault(task.key, deadline_reason)
            return False
        feasible_priorities = [
            int(other.priority) for other in tile_tasks
            if other.key not in claimed
            and other.tile is not None
            and _carried(worker, other.required_item)
            and _interaction_op(other) is not None
        ]
        if task.kind == "PLACE" and feasible_priorities \
                and int(task.priority) > min(feasible_priorities):
            return False
        if not _carried(worker, task.required_item):
            return False
        return True

    if repair:
        # Resolve transfers before any worker dispatch.  This is the crucial
        # atomicity point: an earlier old owner cannot route toward a task
        # that a later worker is executing from underfoot.
        for worker in workers:
            candidate = next(
                (task for task in tile_tasks
                 if repaired_underfoot_ok(worker, task)),
                None,
            )
            if candidate is None:
                continue
            key = str(candidate.key)
            old_worker = queue_owner.get(key)
            if old_worker is None or old_worker == worker.index \
                    or key in transferred_keys:
                continue
            queue_by_worker[old_worker] = [
                task for task in queue_by_worker[old_worker]
                if str(task.key) != key
            ]
            queue_by_worker[worker.index].insert(0, candidate)
            queue_owner[key] = worker.index
            transferred_keys.add(key)
            queue_diag(
                "queue_transfer", task_key=key,
                old_worker_index=old_worker,
                new_worker_index=worker.index,
                reason="queued_underfoot_atomic_transfer",
            )

    def repaired_queue_valid(worker: WorkerView, task: Task) -> tuple[bool, str]:
        if task.key in claimed or task.key in queue_blocked:
            return False, "already_claimed"
        if task.tile is None or _interaction_op(task) is None:
            return False, "no_actionable_target"
        blocked_by = next((dep for dep in task.depends_on if dep in dep_keys), None)
        if blocked_by is not None:
            return False, f"dependency_current:{blocked_by}"
        if not queue_seeds_available(task):
            return False, "no_global_seeds"
        deadline_ok, deadline_reason = _plant_deadline_feasible(
            obs, worker, task, enabled=deadline_safe_planting,
            remaining_turns=remaining_turns)
        if not deadline_ok:
            return False, deadline_reason
        if not _carried(worker, task.required_item) \
                and queue_shed_available(task) <= 0:
            return False, f"shed_lacks_item:{task.required_item}"
        return True, ""

    # Underfoot-first is the experimentally captured baseline contract:
    # reserve local work for every worker before allowing distant claims.
    # The disabled path intentionally retains legacy worker-order behavior.
    dispatch_order = (
        [(worker, True) for worker in workers]
        + [(worker, False) for worker in workers]
        if config.underfoot_first else [(worker, False) for worker in workers]
    )
    assigned_workers: set[int] = set()
    for worker, underfoot_only in dispatch_order:
        if worker.index in assigned_workers:
            continue
        chosen: Task | None = None
        reason = ""
        feasible_priorities = [
            int(task.priority) for task in tile_tasks
            if task.key not in claimed
            and task.tile is not None
            and _carried(worker, task.required_item)
            and _interaction_op(task) is not None
        ]
        best_feasible_priority = min(feasible_priorities, default=None)

        continuation = (worker_continuations or {}).get(worker.index)
        if continuation is not None \
                and continuation.kind == "WATER" \
                and continuation.tile is not None \
                and continuation.tile == worker.position \
                and _interaction_op(continuation) is not None \
                and (not repair or (
                    str(continuation.key) not in claimed
                    and queue_owner.get(str(continuation.key), worker.index)
                    == worker.index)):
            chosen, reason = continuation, "same_worker_plant_continuation"

        if repair and chosen is None and not underfoot_only:
            queue = queue_by_worker[worker.index]
            queued = None
            while queue:
                candidate = queue[0]
                key = str(candidate.key)
                if queue_owner.get(key) != worker.index or candidate.key in claimed:
                    queue.pop(0)
                    continue
                valid, invalid_reason = repaired_queue_valid(worker, candidate)
                if valid:
                    queued = candidate
                    break
                queue.pop(0)
                queue_owner.pop(key, None)
                queue_task_by_key.pop(key, None)
                queue_blocked.add(key)
                release_queue_reservation(candidate)
                queue_diag("queue_release", task_key=key,
                           worker_index=worker.index, reason=invalid_reason)

            if queued is not None:
                higher = next(
                    (task for task in tile_tasks
                     if int(task.priority) < int(queued.priority)
                     and repaired_underfoot_ok(worker, task)
                     and queue_owner.get(str(task.key)) in (None, worker.index)),
                    None,
                )
                if higher is not None:
                    chosen, reason = higher, "urgent_preemption"
                    queue_diag(
                        "queue_preemption", task_key=str(higher.key),
                        worker_index=worker.index,
                        displaced_task_key=str(queued.key),
                        reason="strictly_higher_priority_underfoot",
                    )
                else:
                    chosen, reason = queued, (
                        "queue_transfer" if str(queued.key) in transferred_keys
                        else "persistent_queue"
                    )
                    queue_diag("queue_retention", task_key=str(queued.key),
                               worker_index=worker.index,
                               reason="valid_queue_head")
            else:
                chosen = next(
                    (task for task in tile_tasks
                     if repaired_underfoot_ok(worker, task)
                     and queue_owner.get(str(task.key)) in (None, worker.index)),
                    None,
                )
                if chosen is not None:
                    reason = "underfoot_execution"

        if repair and chosen is None:
            queue_head = None
            if underfoot_only:
                queue = queue_by_worker[worker.index]
                while queue:
                    candidate_head = queue[0]
                    key = str(candidate_head.key)
                    if queue_owner.get(key) != worker.index or candidate_head.key in claimed:
                        queue.pop(0)
                        continue
                    valid, invalid_reason = repaired_queue_valid(worker, candidate_head)
                    if valid:
                        queue_head = candidate_head
                        break
                    queue.pop(0)
                    queue_owner.pop(key, None)
                    queue_task_by_key.pop(key, None)
                    queue_blocked.add(key)
                    release_queue_reservation(candidate_head)
                    queue_diag("queue_release", task_key=key,
                               worker_index=worker.index, reason=invalid_reason)
            chosen = next(
                (task for task in tile_tasks
                 if repaired_underfoot_ok(worker, task)
                 and queue_owner.get(str(task.key)) in (None, worker.index)),
                None,
            )
            if queue_head is not None and chosen is not None \
                    and chosen.key != queue_head.key \
                    and int(chosen.priority) >= int(queue_head.priority):
                chosen = None
            if chosen is not None:
                reason = "underfoot_execution"

        # A persistent queue owns its head until it completes or becomes
        # invalid.  Only a strictly more urgent underfoot task may preempt it;
        # equal/lower-priority incidental work must wait for the route.
        if chosen is None and not underfoot_only and not repair:
            queue = (worker_queues or {}).get(worker.index, ())
            queued = queue[0] if queue else None
            candidate = tile_by_key.get(queued.key) if queued is not None else None
            if candidate is not None:
                deadline_ok, deadline_reason = _plant_deadline_feasible(
                    obs, worker, candidate, enabled=deadline_safe_planting,
                    remaining_turns=remaining_turns)
                if (deadline_ok and seeds_available(candidate)
                        and _interaction_op(candidate) is not None
                        and (_carried(worker, candidate.required_item)
                             or _shed_available(obs, seat, candidate.required_item) > 0)):
                    urgent = next((task for task in tile_tasks
                                    if task.tile == worker.position
                                    and task.priority < candidate.priority
                                    and task.key not in claimed
                                    and seeds_available(task)
                                    and _interaction_op(task) is not None
                                    and _carried(worker, task.required_item)), None)
                    if urgent is not None:
                        chosen, reason = urgent, "urgent_preemption"
                    else:
                        chosen, reason = candidate, "persistent_queue"
                elif not deadline_ok:
                    unassigned_reasons.setdefault(candidate.key, deadline_reason)

        # 1. Underfoot: highest-priority actionable task at our tile.
        for task in tile_tasks if chosen is None and not repair else ():
            if task.key in claimed or task.tile != worker.position:
                continue
            if not _carried(worker, task.required_item):
                continue
            deadline_ok, deadline_reason = _plant_deadline_feasible(
                obs, worker, task, enabled=deadline_safe_planting,
                remaining_turns=remaining_turns)
            if not deadline_ok:
                unassigned_reasons.setdefault(task.key, deadline_reason)
                continue
            if task.kind == "PLACE" and best_feasible_priority is not None \
                    and int(task.priority) > best_feasible_priority:
                continue
            if not seeds_available(task):
                unassigned_reasons.setdefault(task.key, "no_global_seeds")
                continue
            op = _interaction_op(task)
            if op is None:
                continue
            chosen, reason = task, "underfoot_execution"
            break

        if underfoot_only and chosen is None:
            continue

        # 2. Greedy assignment minimizing priority-dominated score.
        if chosen is None:
            best_score = None
            for task in tile_tasks:
                if task.key in claimed or task.tile is None:
                    continue
                if repair and (str(task.key) in queue_owner
                               or str(task.key) in queue_blocked):
                    # A valid queue owns the task even when its head is not
                    # currently executable; a released/blocked head cannot
                    # be silently reclaimed by greedy fallback.
                    continue
                if not (queue_seeds_available(task) if repair
                        else seeds_available(task)):
                    unassigned_reasons.setdefault(task.key,
                                                  "no_global_seeds")
                    continue
                deadline_ok, deadline_reason = _plant_deadline_feasible(
                    obs, worker, task, enabled=deadline_safe_planting,
                    remaining_turns=remaining_turns)
                if not deadline_ok:
                    unassigned_reasons.setdefault(task.key, deadline_reason)
                    continue
                carried_ok = _carried(worker, task.required_item)
                if not carried_ok:
                    # Executable only if the shed can supply the item;
                    # otherwise the task stays unassigned this turn.
                    available_shed = (queue_shed_available(task) if repair
                                      else _shed_available(obs, seat,
                                                           task.required_item))
                    if available_shed <= 0:
                        continue
                distance = (
                    abs(worker.position[0] - task.tile[0])
                    + abs(worker.position[1] - task.tile[1])
                    if carried_ok
                    else _route_distance(worker.position, task.tile, config))
                item = task.required_item
                if carried_ok and item is not None:
                    # Soft specialization: already holding the item is worth
                    # a small deterministic discount.
                    distance = max(0, distance - _CARRY_AFFINITY_BONUS)
                variety_penalty = 0
                if item is not None and item not in worker.inventory \
                        and worker.item_type_count >= \
                        config.max_carried_item_types:
                    variety_penalty = _VARIETY_PENALTY
                score = int(task.priority) * _PRIORITY_STEP + distance \
                    + variety_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    chosen = task
                    reason = ("greedy_dispatch"
                              if carried_ok else "greedy_dispatch_via_shed")
        if chosen is None:
            actions.append(("PASS",))
            assignments.append(Assignment(worker.index, None, ("PASS",),
                                          "no_feasible_task"))
            counts["pass"] += 1
            continue

        claimed.add(chosen.key)
        assigned_workers.add(worker.index)
        needs_pickup = (chosen.required_item is not None
                        and not _carried(worker, chosen.required_item))
        if repair and chosen.kind == "PLANT":
            release_queue_reservation(chosen)
        reserve_seeds(chosen)

        if chosen.tile == worker.position and not needs_pickup:
            op = _interaction_op(chosen)
            if op is None:
                actions.append(("PASS",))
                assignments.append(Assignment(
                    worker.index, chosen.key, ("PASS",), "malformed_metadata"))
                counts["pass"] += 1
                release(chosen)
                continue
            actions.append(op)
            assignments.append(Assignment(worker.index, chosen.key, op, reason))
            counts["interaction"] += 1
            if repair and str(chosen.key) in queue_owner:
                queue_diag("queue_completion", task_key=str(chosen.key),
                           worker_index=worker.index, reason="interaction_dispatched")
            continue

        if needs_pickup:
            access = _nearest_access(worker.position, config)
            if worker.position == access:
                if repair:
                    # Consume this queued task's exclusive shed reservation
                    # atomically at the pickup point.  A traveling owner must
                    # retain the reservation until it actually reaches the
                    # shed; releasing it during route selection would let a
                    # separate greedy task steal the stock this turn.
                    release_queue_reservation(chosen)
                remaining = shed_budget.get(chosen.required_item, 0)
                if remaining <= 0:
                    # Unreachable when the greedy filter already dropped
                    # unstockable tasks; kept as an honest safety net.
                    actions.append(("PASS",))
                    assignments.append(Assignment(
                        worker.index, chosen.key, ("PASS",),
                        "shed_lacks_item"))
                    counts["pass"] += 1
                    release(chosen)
                    continue
                quantity = min(config.pickup_batch, remaining)
                if quantity <= 0:
                    actions.append(("PASS",))
                    assignments.append(Assignment(
                        worker.index, chosen.key, ("PASS",),
                        "shed_lacks_item"))
                    counts["pass"] += 1
                    release(chosen)
                    continue
                op = ("PICKUP", chosen.required_item, quantity)
                shed_budget[chosen.required_item] = remaining - quantity
                actions.append(op)
                assignments.append(Assignment(worker.index, chosen.key, op,
                                              "shed_bulk_pickup"))
                counts["pickup"] += 1
                continue
            waypoint = access
            reason_prefix = "move_to_shed"
        else:
            waypoint = chosen.tile
            reason_prefix = "move_to_task"

        step = _step_toward(board, unlocked, worker.position, waypoint)
        if step is None:
            actions.append(("PASS",))
            assignments.append(Assignment(worker.index, chosen.key,
                                          ("PASS",), "movement_blocked"))
            counts["pass"] += 1
            continue
        actions.append((step[0],))
        assignments.append(Assignment(worker.index, chosen.key, (step[0],),
                                      f"{reason_prefix}:{step[1]}"))
        counts["movement"] += 1

    for task in tile_tasks:
        if task.key not in claimed \
                and all(u.key != task.key for u in unassigned):
            unassigned.append(task)
        if (task.key in {u.key for u in unassigned}
                and task.kind == "PLANT" and deadline_safe_planting
                and task.key not in unassigned_reasons):
            unassigned_reasons[task.key] = (
                "plant_deadline_infeasible_for_all_workers:"
                f"remaining={remaining_turns}"
            )

    if config.underfoot_first:
        assignments.sort(key=lambda assignment: assignment.worker_index)
        actions = [assignment.action for assignment in assignments]

    return ForemanResult(
        farmer_action=actions[0],
        hands_actions=tuple(actions[1:]),
        assignments=tuple(assignments),
        unassigned_tile_tasks=tuple(unassigned),
        market_tasks=tuple(market_tasks),
        counts=counts,
        unassigned_reasons=unassigned_reasons,
        diagnostics=tuple(queue_diagnostics[:128]),
    )


def apply_idle_cleanup(
    obs: Mapping,
    seat: int,
    normal_result: ForemanResult,
    cleanup_tasks: Sequence[Task],
) -> ForemanResult:
    """Assign weed-first cleanup only to workers normal dispatch left PASSing.

    ``normal_result`` is authoritative for every non-PASS worker.  Cleanup
    claims are made atomically in worker order and are discarded by the caller
    after this primitive turn; no task is added to normal accounting.
    """
    if not cleanup_tasks:
        return normal_result

    farm = obs["farms"][seat]
    board = farm["tiles"]
    unlocked = list(farm["unlocked_quadrants"])
    workers = _worker_views(obs, seat)
    actions = [normal_result.farmer_action, *normal_result.hands_actions]
    assignments = list(normal_result.assignments)
    claimed: set[str] = set()
    counts = dict(normal_result.counts)

    candidates = [
        task for task in cleanup_tasks
        if task.kind in ("DIG", "WATER") and task.tile is not None
    ]
    remaining_turns = max(0, 23 - int(obs.get("hour", 23)))

    for worker in workers:
        if worker.index >= len(assignments) or actions[worker.index] != ("PASS",):
            continue
        available = [task for task in candidates if task.key not in claimed]
        if not available:
            continue
        weed_available = [task for task in available if task.kind == "DIG"]
        if weed_available:
            available = weed_available
        chosen = min(
            available,
            key=lambda task: (
                abs(worker.position[0] - task.tile[0])
                + abs(worker.position[1] - task.tile[1]),
                task.key,
            ),
        )
        distance = abs(worker.position[0] - chosen.tile[0]) \
            + abs(worker.position[1] - chosen.tile[1])
        if distance > remaining_turns:
            continue
        if chosen.tile == worker.position:
            action = _interaction_op(chosen)
            reason = f"{chosen.source}_underfoot"
        else:
            step = _step_toward(board, unlocked, worker.position, chosen.tile)
            if step is None:
                continue
            action = (step[0],)
            reason = f"{chosen.source}_move:{step[1]}"
        if action is None:
            continue
        claimed.add(chosen.key)
        actions[worker.index] = action
        assignments[worker.index] = Assignment(
            worker.index, chosen.key, action, reason)
        counts["pass"] -= 1
        counts["interaction" if action[0] in _TILE_TASK_KINDS else "movement"] += 1

    return ForemanResult(
        farmer_action=actions[0],
        hands_actions=tuple(actions[1:]),
        assignments=tuple(assignments),
        unassigned_tile_tasks=normal_result.unassigned_tile_tasks,
        market_tasks=normal_result.market_tasks,
        counts=counts,
        unassigned_reasons=dict(normal_result.unassigned_reasons),
        diagnostics=normal_result.diagnostics,
    )
