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
  execute before hiring/discretionary buys, and new animal/land commitments
  pause while current survival or prior-day work debt is unresolved;
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

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
import math
from typing import Any, Callable

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.tasks import Priority
from replay_daily.constants import (
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT_DEFAULT,
    LAND_ORDER,
    LAND_PRICES,
    PRODUCTS,
    hire_cost,
    total_hire_cost,
)
from replay_daily.lifecycle import canonical_board

from .foreman import ForemanConfig, run_foreman
from .tasks import Task
from .manager import CheckpointPlanProvider, PlanProvider
from .plan import SELL_BIN_ANCHORS, DailyPlan
from .projection import clip_sell, project_plan
from .tasks import GenerationResult, generate_tasks

__all__ = ["AgentConfig", "ExecutorAgent", "make_agent"]

_DIAGNOSTICS_SCHEMA_VERSION = 2
_MONEY_EPSILON = 1e-6
_INTERACTION_OPS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})
_EXPANSION_TASK_KINDS = frozenset({
    "BUILD_COOP", "BUILD_PASTURE", "PLACE", "BUY_ANIMAL", "BUY_LAND",
})


def _sell_bin_index(hour: int) -> int:
    return (int(hour) // 4) * 4


@dataclass(frozen=True)
class AgentConfig:
    tasks_per_worker: int = 10
    hire_cost_mult: int = FARM_HAND_COST_MULT_DEFAULT
    max_market_orders: int = 10
    foreman: ForemanConfig = field(default_factory=ForemanConfig)
    strict: bool = False


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


def _animal_feed_state(obs: Mapping, seat: int) -> dict[str, int]:
    farm = obs["farms"][seat]
    board = canonical_board(farm["tiles"], int(obs["day"]), int(obs.get("step", 0)))
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


class ExecutorAgent:
    def __init__(self, provider: PlanProvider, *, seat: int | None = None,
                 config: AgentConfig | None = None) -> None:
        self.provider = provider
        self.seat = seat
        if seat is not None and seat not in (0, 1):
            raise ValueError(f"seat must be None, 0, or 1, got {seat!r}")
        self.config = config or AgentConfig()
        _require_positive_int(self.config.tasks_per_worker, "config.tasks_per_worker")
        _require_positive_int(self.config.max_market_orders, "config.max_market_orders")
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

    def _new_day(self, obs: Mapping, seat: int) -> None:
        farm = obs["farms"][seat]
        board = canonical_board(farm["tiles"], int(obs["day"]), int(obs.get("step", 0)))
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
            record["next_day_expansion_suppressed"] = prior_debt
        self._suppress_expansion_today = prior_debt
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
            "errors": [],
        }
        self._day = int(obs["day"])

    def _refresh_sell_ledger(self, obs: Mapping, bin_anchor: int) -> None:
        bin_index = SELL_BIN_ANCHORS.index(bin_anchor)
        self._remaining_sells = {
            product: self._feasible.sell_quantities[product_index][bin_index]
            for product_index, product in enumerate(PRODUCTS)
        }
        self._bin_anchor = bin_anchor
        record = self._day_records[int(obs["day"])]
        record["sells"][str(bin_anchor)] = {
            product: {"requested": self._remaining_sells[product], "submitted": 0, "remaining": self._remaining_sells[product]}
            for product in PRODUCTS
        }

    def _sell_candidates(self, obs: Mapping, seat: int) -> list[dict]:
        shed = (obs.get("private") or {}).get("shed") or {}
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
            self._remaining_sells[product] = self._remaining_sells.get(product, 0) - executed
            entry = bin_log[product]
            entry["submitted"] += executed
            entry["remaining"] = self._remaining_sells[product]

    def _hire_orders(self, obs: Mapping, seat: int, tile_tasks: list[Task], available_cash: float) -> tuple[list[list], int]:
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
        turns_left = max(24 - int(obs["hour"]), 1)

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
        for quantity in range(int(task.quantity), 0, -1):
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
        generated_tasks: tuple[Task, ...],
        generation: GenerationResult,
        foreman_result: Any,
        feed: Mapping[str, int],
        expansion_suppressed: bool,
        orders: list[list],
        candidates: list[tuple[str, Any]],
        unaffordable_orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_by_key = {task.key: task for task in tasks}
        assignments = []
        for assignment in foreman_result.assignments:
            task = task_by_key.get(assignment.task_key)
            assignments.append({
                "worker_index": assignment.worker_index,
                "task_key": assignment.task_key,
                "reason": assignment.reason,
                "action": list(assignment.action),
                "target": list(task.tile) if task is not None and task.tile is not None else None,
            })

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
        return _snapshot_copy(snapshot)

    def _act(self, obs: Mapping) -> dict[str, Any]:
        seat = self._resolve_seat(obs)
        day, hour = int(obs["day"]), int(obs["hour"])
        if self._day != day:
            self._new_day(obs, seat)
        else:
            farm = obs["farms"][seat]
            raw_hires = farm.get("hires_today", 0)
            if isinstance(raw_hires, int) and not isinstance(raw_hires, bool):
                self._max_hires_today = max(self._max_hires_today, raw_hires)

        bin_anchor = _sell_bin_index(hour)
        if bin_anchor != self._bin_anchor:
            self._refresh_sell_ledger(obs, bin_anchor)

        generation = generate_tasks(obs, seat, feasible_plan=self._feasible, remaining_sells=self._remaining_sells)
        generated_tasks = generation.sorted_tasks()
        tasks = generated_tasks
        feed = _animal_feed_state(obs, seat)
        current_survival_pressure = bool(feed["starving"] or feed["shortage"])
        expansion_suppressed = self._suppress_expansion_today or current_survival_pressure
        if expansion_suppressed:
            tasks = tuple(t for t in tasks if t.kind not in _EXPANSION_TASK_KINDS)

        dispatch_tasks = tasks
        if feed["starving"]:
            dispatch_tasks = tuple(t for t in tasks if t.tile is None or t.kind == "FEED")

        foreman_result = run_foreman(obs, seat, dispatch_tasks, config=self.config.foreman)

        record = self._day_records[day]
        survival_record = record["survival"]
        survival_record["expansion_suppressed_current"] = bool(
            survival_record["expansion_suppressed_current"] or expansion_suppressed)
        if feed["starving"]:
            survival_record["starvation_preemption_turns"] += 1
        if feed["shortage"]:
            survival_record["feed_shortage_turns"] += 1

        sell_candidates = self._sell_candidates(obs, seat)
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
        other_buys = [
            t for t in all_market_tasks
            if t.kind in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND")
            and t not in survival_feed_buys
        ]

        candidates = [("sell", c) for c in sell_candidates]
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

        hire_orders, hires_requested = self._hire_orders(
            obs, seat,
            [t for t in dispatch_tasks if t.tile is not None and t.kind in _foreman_mod._TILE_TASK_KINDS],
            running_cash,
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

        included = candidates[:self.config.max_market_orders]
        orders = [payload["order"] if kind == "sell" else payload for kind, payload in included]

        self._commit_sells(obs, day, [payload for kind, payload in included if kind == "sell"])
        submitted_hires = sum(1 for kind, _ in included if kind == "hire")

        for key in ("movement", "interaction", "pickup", "pass"):
            record["foreman_counts"][key] += foreman_result.counts[key]

        pending = self._pending_task_keys(foreman_result)
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
        record["unresolved_generator"] = list(generation.unresolved)

        crops, animals, care_done, fert_done = _board_counts(
            canonical_board(obs["farms"][seat]["tiles"], day, int(obs.get("step", 0))))
        record["achieved_current"] = {
            "crops": crops, "animals": animals,
            "land_count": len(obs["farms"][seat]["unlocked_quadrants"]),
        }
        record["care_completed_observed"] = care_done
        record["fertilizer_completed_observed"] = fert_done

        try:
            self._debug_trace_turn = self._build_debug_trace_turn(
                day=day,
                hour=hour,
                tasks=tasks,
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

        return {
            "farmer": list(foreman_result.farmer_action),
            "hands": [list(a) for a in foreman_result.hands_actions],
            "market": orders[:self.config.max_market_orders],
        }

    @property
    def debug_trace_turn(self) -> dict[str, Any] | None:
        """Return a defensive copy of the latest primitive-turn snapshot."""
        return copy.deepcopy(self._debug_trace_turn)

    def diagnostics_json(self) -> dict[str, Any]:
        diagnostics = {
            "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
            "seat": self.seat,
            "days": {str(day): record for day, record in sorted(self._day_records.items())},
            "illegal_actions": {
                "available": False,
                "reason": "the 1.32.7 observation does not expose per-action validity; illegal/ineffective detection requires engine-source instrumentation",
                "count": 0,
            },
            "fallback_errors": [dict(e) for e in self._errors],
        }
        provider_diagnostics = getattr(self.provider, "diagnostics_json", None)
        if callable(provider_diagnostics):
            diagnostics["provider_diagnostics"] = provider_diagnostics()
        return diagnostics


def make_agent(*, provider: PlanProvider | None = None,
               checkpoint: str | None = None, device: str = "cpu",
               seat: int | None = None,
               config: AgentConfig | None = None) -> ExecutorAgent:
    if (provider is None) == (checkpoint is None):
        raise ValueError("provide exactly one of provider= or checkpoint=")
    if checkpoint is not None:
        provider = CheckpointPlanProvider(checkpoint, device=device)
    return ExecutorAgent(provider, seat=seat, config=config)


AgentCallable = Callable[[Mapping], dict[str, Any]]
