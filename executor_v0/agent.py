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
  inventory via `clip_sell`, remainder carried within the bin), hour-0-only
  crude workload hiring, and exact-shortage BUY_SEED / BUY_PRODUCT /
  BUY_ANIMAL / BUY_LAND orders implied by the active task generator;
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
from dataclasses import dataclass, field
import math
from typing import Any, Callable

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
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
from .manager import CheckpointPlanProvider, PlanProvider
from .plan import SELL_BIN_ANCHORS, DailyPlan
from .projection import clip_sell, project_plan
from .tasks import GenerationResult, generate_tasks

__all__ = ["AgentConfig", "ExecutorAgent", "make_agent"]

_DIAGNOSTICS_SCHEMA_VERSION = 1
# The engine accepts an order when money + epsilon covers it
# (``money[player] + MONEY_EPSILON >= cost`` in the rust core).
_MONEY_EPSILON = 1e-6


def _sell_bin_index(hour: int) -> int:
    """Active four-hour sell bin anchor: floor(hour/4)*4."""
    return (int(hour) // 4) * 4


@dataclass(frozen=True)
class AgentConfig:
    """Provisional V0 constants; calibrate later from closed-loop traces."""

    tasks_per_worker: int = 10          # crude workload divisor for hiring
    hire_cost_mult: int = FARM_HAND_COST_MULT_DEFAULT
    max_market_orders: int = 10         # engine maxMarketOrdersPerTurn
    foreman: ForemanConfig = field(default_factory=ForemanConfig)
    strict: bool = False                # True re-raises instead of PASSing


def _require_positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def _board_counts(board) -> tuple[dict[str, int], dict[str, int],
                                 dict[str, int], dict[str, int]]:
    """Current crops/animals by type plus observed CARE/FERTILIZER completions."""
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


class ExecutorAgent:
    """Kaggle-compatible callable closing the issue #1 V0 loop."""

    def __init__(self, provider: PlanProvider, *, seat: int | None = None,
                 config: AgentConfig | None = None) -> None:
        self.provider = provider
        self.seat = seat
        if seat is not None and seat not in (0, 1):
            raise ValueError(f"seat must be None, 0, or 1, got {seat!r}")
        self.config = config or AgentConfig()
        _require_positive_int(self.config.tasks_per_worker,
                              "config.tasks_per_worker")
        _require_positive_int(self.config.max_market_orders,
                              "config.max_market_orders")
        # ---- per-game state -------------------------------------------
        self._day: int | None = None
        self._requested: DailyPlan | None = None
        self._feasible: DailyPlan | None = None
        self._projection_diagnostics: dict[str, Any] = {}
        self._bin_anchor: int | None = None
        self._remaining_sells: dict[str, int] = {}
        self._previous_execution: dict[str, int] = {
            "workers_hired": 0, "hire_cost": 0}
        self._max_hires_today: int = 0
        self._day_records: dict[int, dict[str, Any]] = {}
        self._errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------- kaggle
    def __call__(self, obs: Mapping) -> dict[str, Any]:
        try:
            return self._act(obs)
        except Exception as exc:  # noqa: BLE001 - deliberate safe mode
            if self.config.strict:
                raise
            self._errors.append({
                "step": obs.get("step") if isinstance(obs, Mapping) else None,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            return self._fallback_action(obs)

    # -------------------------------------------------------------- core
    def _resolve_seat(self, obs: Mapping) -> int:
        observed = obs.get("player")
        if self.seat is not None:
            if observed is not None and int(observed) != self.seat:
                raise ValueError(
                    f"obs player {observed!r} contradicts explicit agent seat "
                    f"{self.seat}")
            return self.seat
        if observed is None:
            raise ValueError(
                "obs carries no 'player' field; construct the agent with an "
                "explicit seat")
        seat = int(observed)
        if seat not in (0, 1):
            raise ValueError(f"obs player must be 0 or 1, got {seat!r}")
        return seat

    def _fallback_action(self, obs: Mapping) -> dict[str, Any]:
        """Legal-shaped all-PASS action; never raises."""
        hands: int = 0
        try:
            seat = self._resolve_seat(obs)
            hands = len(obs["farms"][seat].get("hands") or [])
        except Exception:  # noqa: BLE001 - fallback must not raise
            hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * hands,
                "market": []}

    def _new_day(self, obs: Mapping, seat: int) -> None:
        """Finalize prior labor, call the manager once, project the plan."""
        farm = obs["farms"][seat]
        board = canonical_board(farm["tiles"], int(obs["day"]),
                                int(obs.get("step", 0)))
        crops, animals, care_done, fert_done = _board_counts(board)
        if self._day is not None:
            # Realized hires of the finished day come from the observed
            # hires_today progression, never from submitted HIRE intents.
            hires = self._max_hires_today
            self._previous_execution = {
                "workers_hired": hires,
                "hire_cost": total_hire_cost(hires,
                                             self.config.hire_cost_mult),
            }
            record = self._day_records[self._day]
            record["achieved_final"] = {
                "crops": crops, "animals": animals,
                "land_count": len(farm["unlocked_quadrants"]),
            }
            record["care_completed_observed"] = care_done
            record["fertilizer_completed_observed"] = fert_done
        # The engine resets hires_today at the day boundary; start fresh.
        raw_hires = farm.get("hires_today", 0)
        self._max_hires_today = raw_hires \
            if isinstance(raw_hires, int) and not isinstance(raw_hires, bool) \
            else 0
        self._requested = self.provider.daily_plan(
            obs, seat, dict(self._previous_execution))
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
            "foreman_counts": {"movement": 0, "interaction": 0,
                               "pickup": 0, "pass": 0},
            "unfinished_tasks": [],
            "missed_maintenance": [],
            "sells": {},
            "hires": {"requested": 0, "submitted": 0,
                      "observed_max": self._max_hires_today},
            "previous_labor": dict(self._previous_execution),
            "unresolved_generator": [],
            "errors": [],
        }
        self._day = int(obs["day"])

    def _refresh_sell_ledger(self, obs: Mapping, bin_anchor: int) -> None:
        """Reset remaining sells at each new four-hour bin from the plan."""
        bin_index = SELL_BIN_ANCHORS.index(bin_anchor)
        self._remaining_sells = {
            product: self._feasible.sell_quantities[product_index][bin_index]
            for product_index, product in enumerate(PRODUCTS)
        }
        self._bin_anchor = bin_anchor
        record = self._day_records[int(obs["day"])]
        record["sells"][str(bin_anchor)] = {
            product: {"requested": self._remaining_sells[product],
                      "submitted": 0,
                      "remaining": self._remaining_sells[product]}
            for product in PRODUCTS
        }

    def _sell_candidates(self, obs: Mapping, seat: int) -> list[dict]:
        """Clip sells to available shed inventory WITHOUT mutating state.

        Each candidate is ``{"order": [...], "product": p, "executed": n}``;
        the caller commits only candidates that survive the market-order
        cap via `_commit_sells`, so dropped orders never decrement the
        remaining ledger or inflate submitted diagnostics.
        """
        shed = (obs.get("private") or {}).get("shed") or {}
        candidates: list[dict] = []
        for product in PRODUCTS:
            remaining = self._remaining_sells.get(product, 0)
            if remaining <= 0:
                continue
            available = int(shed.get(product, 0))
            executed, _remaining_after = clip_sell(product, remaining,
                                                   available)
            if executed > 0:
                candidates.append({"order": ["SELL", product, executed],
                                   "product": product,
                                   "executed": executed})
        return candidates

    def _commit_sells(self, obs: Mapping, day: int,
                      committed: list[dict]) -> None:
        """Apply only actually-submitted sell orders to ledgers/logs."""
        record = self._day_records[day]
        bin_log = record["sells"][str(self._bin_anchor)]
        for item in committed:
            product = item["product"]
            executed = item["executed"]
            self._remaining_sells[product] = \
                self._remaining_sells.get(product, 0) - executed
            entry = bin_log[product]
            entry["submitted"] += executed
            entry["remaining"] = self._remaining_sells[product]

    def _hire_orders(self, obs: Mapping, seat: int, tile_task_count: int,
                     available_cash: float) -> tuple[list[list], int]:
        """Workload-derived hiring; returns (orders, requested).

        Exact mechanics (issue #7):

        - Hiring is useful at ANY hour (hands are cleared at every day
          boundary and cost only a tiny Fibonacci schedule), so the old
          hour-0-only gate no longer starves days that open with sells.
        - The old hard ``max 3 hires/day`` cap is replaced by the executable
          workload estimate plus exact sequential affordability.
        - ``available_cash`` is current money PLUS the revenue of sell orders
          already queued earlier this same turn: the engine applies market
          orders sequentially, so same-turn SELL -> HIRE funding is real and
          was previously ignored.
        """
        farm = obs["farms"][seat]
        current_hands = len(farm.get("hands") or [])
        desired = 0
        if tile_task_count > 0:
            desired = int(math.ceil(tile_task_count /
                                    self.config.tasks_per_worker))
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
        """Revenue of the sell candidates already queued this turn.

        The engine's interleaved market path pays the current per-unit price
        for each SELL order; observation prices are exactly those prices.
        """
        prices = (obs.get("market") or {}).get("prices") or {}
        total = 0.0
        for item in committed_sells:
            total += float(prices.get(item["product"], 0)) * \
                float(item["executed"])
        return total

    @staticmethod
    def _buy_order_cost(obs: Mapping, task, unlocked_count: int) \
            -> float | None:
        """Conservative whole-order cost of one shortage buy; None = malformed.

        BUY_SEED / BUY_ANIMAL / BUY_LAND have exact fixed costs. BUY_PRODUCT
        prices rise as the shared market depletes, so the whole-order cost is
        estimated unit-by-unit from the official price model at depleting
        inventory levels (the engine accepts an order only if money covers
        the entire quantity at once).
        """
        kind = task.kind
        quantity = int(task.quantity)
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
            inventory = int(((obs.get("market") or {}).get("inventory")
                             or {}).get(task.product, 0))
            total = 0.0
            for k in range(quantity):
                level = inventory - k - 1
                total += float(market_price(task.product, max(level, 0)))
            return total
        return None

    @staticmethod
    def _buy_op(task) -> list | None:
        """Exact market op list for one shortage task; None if malformed."""
        quantity = int(task.quantity)
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

        generation: GenerationResult = generate_tasks(
            obs, seat, feasible_plan=self._feasible,
            remaining_sells=self._remaining_sells)
        tasks = generation.sorted_tasks()
        foreman_result = run_foreman(obs, seat, tasks,
                                     config=self.config.foreman)

        # ---- market queue: sells -> hires -> shortage buys --------------
        # Candidates are built WITHOUT mutating any ledger or diagnostic;
        # only orders that survive the engine cap are committed below, so
        # dropped lower-priority candidates (deferred/recomputed next turn)
        # are never counted as submitted.
        #
        # Sequential within-turn accounting (issue #7): the engine applies
        # market orders in queue order, so earlier SELL revenue funds later
        # HIRE/BUY orders. Every candidate is gated against the running cash
        # it would see at execution time; unaffordable candidates are skipped
        # this turn instead of being resubmitted unchanged every hour.
        sell_candidates = self._sell_candidates(obs, seat)
        tile_task_count = sum(1 for t in tasks if t.tile is not None)
        money = float(obs["farms"][seat].get("money", 0.0))
        cash_after_sells = money + self._sell_proceeds(obs, sell_candidates)
        hire_orders, hires_requested = self._hire_orders(
            obs, seat, tile_task_count, cash_after_sells)
        unlocked_count = len(obs["farms"][seat]["unlocked_quadrants"])
        buy_tasks = sorted(
            (t for t in foreman_result.market_tasks
             if t.kind in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
                           "BUY_LAND")),
            key=lambda t: t.sort_key)

        candidates: list[tuple[str, object]] = \
            [("sell", c) for c in sell_candidates] + \
            [("hire", o) for o in hire_orders]
        running_cash = cash_after_sells
        unaffordable_orders: list[dict[str, Any]] = []
        for task in buy_tasks:
            op = self._buy_op(task)
            if op is None:
                continue
            cost = self._buy_order_cost(obs, task, unlocked_count)
            if cost is None:
                continue
            if running_cash + _MONEY_EPSILON < cost:
                unaffordable_orders.append({
                    "task": task.key, "cost": cost,
                    "cash_available": running_cash})
                continue
            running_cash -= cost
            candidates.append(("buy", op))
        included = candidates[:self.config.max_market_orders]
        orders = [payload["order"] if kind == "sell" else payload
                  for kind, payload in included]

        record = self._day_records[day]
        self._commit_sells(obs, day,
                           [payload for kind, payload in included
                            if kind == "sell"])
        submitted_hires = sum(1 for kind, _ in included if kind == "hire")

        for key in ("movement", "interaction", "pickup", "pass"):
            record["foreman_counts"][key] += foreman_result.counts[key]
        unfinished = [t.key for t in foreman_result.unassigned_tile_tasks]
        # Accumulate per-key turn counts across the day instead of keeping
        # only the last turn's snapshot: a task that churned/unassigned all
        # day must stay visible even when the final turn looked clean
        # (issue #7 diagnostics overwrite bug).
        unfinished_turns: dict[str, int] = record.setdefault(
            "unfinished_task_turns", {})
        for key in unfinished:
            unfinished_turns[key] = unfinished_turns.get(key, 0) + 1
        missed_now = [
            key for key in unfinished if key.startswith(
                ("WATER:", "FEED:", "COLLECT_FERTILIZER:"))]
        missed_turns: dict[str, int] = record.setdefault(
            "missed_maintenance_turns", {})
        for key in missed_now:
            missed_turns[key] = missed_turns.get(key, 0) + 1
        record["unfinished_tasks"] = unfinished
        record["missed_maintenance"] = missed_now
        if unaffordable_orders:
            record.setdefault("unaffordable_market_orders", []).extend(
                unaffordable_orders)
        record["hires"]["requested"] = max(record["hires"]["requested"],
                                           hires_requested)
        record["hires"]["submitted"] += submitted_hires
        record["hires"]["observed_max"] = self._max_hires_today
        record["unresolved_generator"] = list(generation.unresolved)

        # Continuously current achieved state from every processed
        # observation, so the latest day (e.g. day 29) always carries valid
        # current/final-seen values even without a following day boundary.
        crops, animals, care_done, fert_done = _board_counts(
            canonical_board(obs["farms"][seat]["tiles"], day,
                            int(obs.get("step", 0))))
        record["achieved_current"] = {
            "crops": crops, "animals": animals,
            "land_count": len(obs["farms"][seat]["unlocked_quadrants"]),
        }
        record["care_completed_observed"] = care_done
        record["fertilizer_completed_observed"] = fert_done

        return {
            "farmer": list(foreman_result.farmer_action),
            "hands": [list(a) for a in foreman_result.hands_actions],
            "market": orders[:self.config.max_market_orders],
        }

    # ------------------------------------------------------------ diagnostics
    def diagnostics_json(self) -> dict[str, Any]:
        """JSON-serializable per-day/game diagnostics accumulated so far.

        Additive issue #6 key: when the injected provider exposes its own
        ``diagnostics_json`` (e.g. CheckpointPlanProvider closed-loop
        coherence), it is embedded under ``provider_diagnostics``. Purely
        additive — action generation never consults this method.
        """
        diagnostics: dict[str, Any] = {
            "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
            "seat": self.seat,
            "days": {str(day): record
                     for day, record in sorted(self._day_records.items())},
            "illegal_actions": {
                "available": False,
                "reason": ("the 1.32.7 observation does not expose per-action "
                           "validity; illegal/ineffective detection requires "
                           "engine-source instrumentation"),
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
    """Build an `ExecutorAgent` from an injected provider or explicit path.

    Exactly one of ``provider`` / ``checkpoint`` must be given; a missing or
    invalid checkpoint file raises immediately (nothing is ever fabricated).
    """
    if (provider is None) == (checkpoint is None):
        raise ValueError(
            "provide exactly one of provider= or checkpoint=")
    if checkpoint is not None:
        provider = CheckpointPlanProvider(checkpoint, device=device)
    return ExecutorAgent(provider, seat=seat, config=config)


# Type alias documenting the Kaggle callable shape.
AgentCallable = Callable[[Mapping], dict[str, Any]]
