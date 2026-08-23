"""Deterministic greedy foreman for worker task execution (issue #1 section 6).

One primitive turn per call: underfoot execution first, then greedy
assignment, soft inventory specialization, and exactly one legal Manhattan
step or interaction per worker. No search/VRP/pathfinder; routes are
recomputed by the caller every turn.

Shed access mechanic — replay-verified evidence: in elite 1.32.7 replays
(local sample episode 94735084), every observed PICKUP/DROP (269 events,
both seats) occurs at one of the four center tiles ``[x,y] in
{(4,4),(5,4),(4,5),(5,5)}``, i.e. ``[y,x]`` coords
``{(4,4),(4,5),(5,4),(5,5)}``. These are exposed as `SHED_ACCESS_TILES` and
configurable via `ForemanConfig.shed_access_tiles`.

Worker op encoding — replay-verified: every worker action is a list;
bare ops are single-element lists (`["WATER"]`, `["NORTH"]`, `["PASS"]`);
argument ops append strings/ints (`["PLANT","WHEAT"]`,
`["PICKUP","FERTILIZER",3]`, `["PLACE","COW",1]`). Movement deltas:
NORTH dy=-1, SOUTH dy=+1, EAST dx=+1, WEST dx=-1.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bc_manager.constants import CROP_ORDER
from executor_v0.layout import tile_role
from executor_v0.tasks import Task

__all__ = [
    "SHED_ACCESS_TILES",
    "ForemanConfig",
    "WorkerView",
    "Assignment",
    "ForemanResult",
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


@dataclass(frozen=True)
class ForemanConfig:
    max_carried_item_types: int = 2
    pickup_batch: int = 5
    shed_access_tiles: tuple[tuple[int, int], ...] = SHED_ACCESS_TILES


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

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "farmer_action": list(self.farmer_action),
            "hands_actions": [list(a) for a in self.hands_actions],
            "assignments": [a.to_json_dict() for a in self.assignments],
            "unassigned_tile_tasks": [t.key
                                      for t in self.unassigned_tile_tasks],
            "market_tasks": [t.to_json_dict() for t in self.market_tasks],
            "counts": dict(self.counts),
        }


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
    """Pickupable stock for one item; seeds live in their own shed store."""
    private = obs.get("private") or {}
    available = int((private.get("shed") or {}).get(item, 0))
    if item in CROP_ORDER:
        available += int((private.get("seeds") or {}).get(item, 0))
    return max(0, available)


def _legal_step(board, unlocked_quadrants, frm: tuple[int, int],
                to: tuple[int, int]) -> bool:
    y, x = to
    if not (0 <= y < 10 and 0 <= x < 10):
        return False
    from executor_v0.layout import quadrant_of
    if quadrant_of(y, x) not in set(unlocked_quadrants):
        return False
    return tile_role(board[y][x]) != "locked"


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


# ------------------------------------------------------------------- foreman


def run_foreman(
    obs: Mapping,
    seat: int,
    tasks: Sequence[Task],
    *,
    config: ForemanConfig = ForemanConfig(),
) -> ForemanResult:
    """Run one greedy dispatch turn. Pure; inputs never mutated."""
    farm = obs["farms"][seat]
    board = farm["tiles"]
    unlocked = list(farm["unlocked_quadrants"])
    workers = _worker_views(obs, seat)

    all_tasks = list(tasks)
    dep_keys = {t.key for t in all_tasks}
    tile_tasks = []
    market_tasks = []
    unassigned: list[Task] = []
    for t in all_tasks:
        if t.kind in _MARKET_TASK_KINDS:
            market_tasks.append(t)
        elif t.kind in _TILE_TASK_KINDS:
            # A dependency still present in this turn's set blocks the task;
            # it is surfaced as unassigned for diagnostics.
            if any(dep in dep_keys for dep in t.depends_on):
                unassigned.append(t)
                continue
            tile_tasks.append(t)
        else:
            continue  # unknown kinds are never dispatched
    tile_tasks.sort(key=lambda t: t.sort_key)
    market_tasks.sort(key=lambda t: t.sort_key)

    claimed: set[str] = set()
    assignments: list[Assignment] = []
    counts = {"movement": 0, "interaction": 0, "pickup": 0, "pass": 0}
    actions: list[tuple] = []

    def release(task_: Task) -> None:
        """Return a claimed task to the unassigned pool exactly once."""
        claimed.discard(task_.key)
        if all(u.key != task_.key for u in unassigned):
            unassigned.append(task_)

    for worker in workers:
        chosen: Task | None = None
        reason = ""

        # 1. Underfoot: highest-priority actionable task at our tile.
        for task in tile_tasks:
            if task.key in claimed or task.tile != worker.position:
                continue
            if not _carried(worker, task.required_item):
                continue
            op = _interaction_op(task)
            if op is None:
                continue
            chosen, reason = task, "underfoot_execution"
            break

        # 2. Greedy assignment minimizing priority-dominated score.
        if chosen is None:
            best_score = None
            for task in tile_tasks:
                if task.key in claimed or task.tile is None:
                    continue
                carried_ok = _carried(worker, task.required_item)
                if not carried_ok:
                    # Executable only if the shed can supply the item;
                    # otherwise the task stays unassigned this turn.
                    if _shed_available(obs, seat, task.required_item) <= 0:
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
        needs_pickup = (chosen.required_item is not None
                        and not _carried(worker, chosen.required_item))

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
            continue

        if needs_pickup:
            access = _nearest_access(worker.position, config)
            if worker.position == access:
                stock = _shed_available(obs, seat, chosen.required_item)
                if stock <= 0:
                    # Unreachable when the greedy filter already dropped
                    # unstockable tasks; kept as an honest safety net.
                    actions.append(("PASS",))
                    assignments.append(Assignment(
                        worker.index, chosen.key, ("PASS",),
                        "shed_lacks_item"))
                    counts["pass"] += 1
                    release(chosen)
                    continue
                quantity = min(config.pickup_batch, stock)
                op = ("PICKUP", chosen.required_item, quantity)
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

    return ForemanResult(
        farmer_action=actions[0],
        hands_actions=tuple(actions[1:]),
        assignments=tuple(assignments),
        unassigned_tile_tasks=tuple(unassigned),
        market_tasks=tuple(market_tasks),
        counts=counts,
    )
