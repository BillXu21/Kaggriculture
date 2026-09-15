"""Packet 4 deterministic market planning for the experimental strip executor.

The planner in this module is deliberately independent of route ownership.  It
reconciles one observed state and one Packet 1 forecast through a single
sequential money/shed ledger; the caller must wait for a later observation
before treating submitted purchases as realized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.plan import DailyPlan
from fast_env.market import market_price
from replay_daily.constants import ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, PRODUCTS
from executor_v0.strip_work import StripWorkPlan

__all__ = [
    "MarketBlockReason",
    "MarketBootstrapState",
    "MarketIntent",
    "MarketPendingOrder",
    "MarketTurnPlan",
    "build_market_turn_plan",
    "sell_bin_anchor",
]


class MarketBlockReason(StrEnum):
    """Mechanical reason that an observed market intent was not submitted."""

    CASH = "CASH"
    SHED_CAPACITY = "SHED_CAPACITY"
    MISSING_PRICE = "MISSING_PRICE"
    FAILED = "FAILED"
    NOT_IN_SHED = "NOT_IN_SHED"
    RESERVATION_PROTECTED = "RESERVATION_PROTECTED"
    ORDER_CAP = "ORDER_CAP"


@dataclass(frozen=True)
class MarketIntent:
    """One stable logical market demand derived from plan plus work forecast."""

    key: str
    kind: str
    item: str | None
    requested: int
    anchor: int | None = None

    def __post_init__(self) -> None:
        if self.requested <= 0:
            raise ValueError("market intent quantity must be positive")


@dataclass(frozen=True)
class MarketPendingOrder:
    """Submitted buy awaiting confirmation from a later real observation."""

    key: str
    kind: str
    item: str | None
    quantity: int
    observed_before: int
    submitted_step: int
    same_item_sold: int = 0


@dataclass
class MarketBootstrapState:
    """Bounded cross-turn state; never substitutes for authoritative game state."""

    bootstrap_turns: int = 0
    finalized_hour: int | None = None
    sell_submitted: dict[tuple[int, str], int] = field(default_factory=dict)
    sell_requested: dict[tuple[int, str], int] = field(default_factory=dict)
    buy_submitted: dict[str, int] = field(default_factory=dict)
    buy_observed: dict[str, int] = field(default_factory=dict)
    no_progress_counts: dict[str, int] = field(default_factory=dict)
    failed_intents: set[str] = field(default_factory=set)
    pending_buys: tuple[MarketPendingOrder, ...] = ()
    closed_sell_bins: dict[tuple[int, str], int] = field(default_factory=dict)
    latest_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def finalized(self) -> bool:
        return self.finalized_hour is not None

    def to_json_dict(self) -> dict[str, Any]:
        def keyed(values: Mapping[tuple[int, str], int]) -> dict[str, int]:
            return {
                f"{anchor}:{item}": quantity
                for (anchor, item), quantity in sorted(values.items())
            }

        return {
            "bootstrap_turns": self.bootstrap_turns,
            "bootstrap_finalized_hour": self.finalized_hour,
            "buy_submitted": dict(sorted(self.buy_submitted.items())),
            "buy_observed": dict(sorted(self.buy_observed.items())),
            "sell_requested": keyed(self.sell_requested),
            "sell_submitted": keyed(self.sell_submitted),
            "sell_unfulfilled_at_bin_close": keyed(self.closed_sell_bins),
            "market_no_progress_failures": sorted(self.failed_intents),
            "pending_buys": [asdict(order) for order in self.pending_buys],
            **self.latest_diagnostics,
        }


@dataclass(frozen=True)
class MarketTurnPlan:
    """Legal deterministic market prefix simulated against one shared ledger."""

    orders: tuple[tuple, ...]
    intents: tuple[MarketIntent, ...]
    pending_buys: tuple[MarketPendingOrder, ...]
    diagnostics: dict[str, Any]

    @property
    def has_orders(self) -> bool:
        return bool(self.orders)


def sell_bin_anchor(hour: int) -> int:
    """Return the active independent four-hour manager sell bin."""

    return (max(0, min(23, int(hour))) // 4) * 4


def build_market_turn_plan(
    obs: Mapping[str, Any],
    daily_plan: DailyPlan,
    work_plan: StripWorkPlan,
    state: MarketBootstrapState,
    *,
    acting_seat: int = 0,
    shed_capacity: int = 100,
    max_orders: int = 10,
    market_params: Mapping[str, Mapping[str, Any]] | None = None,
    protected_reservations: Mapping[str, int] | None = None,
    purchases_enabled: bool = True,
) -> MarketTurnPlan:
    """Build one deterministic market prefix from the observed state.

    The private ledger is intentionally local to this call.  It is a legality
    forecast for the emitted list, never a mutation of ``obs`` or a substitute
    for the next engine observation.
    """

    if shed_capacity <= 0 or max_orders <= 0:
        raise ValueError("shed_capacity and max_orders must be positive")
    farm = (obs.get("farms") or ())[acting_seat]
    private = obs.get("private") or {}
    shed = _positive_counts(private.get("shed"))
    inventories = private.get("inventories") or ()
    carried = _sum_inventories(inventories)
    seeds = _positive_counts(private.get("seeds"))
    money = _number(farm.get("money"), 0.0)
    unlocked = tuple(str(q) for q in (farm.get("unlocked_quadrants") or ()))
    observed_shed = dict(shed)
    observed_seeds = dict(seeds)
    observed_unlocked = tuple(unlocked)
    market = obs.get("market") or {}
    market_inventory = _numeric_counts(market.get("inventory"))
    displayed_prices = market.get("prices") if isinstance(market, Mapping) else {}
    displayed_prices = displayed_prices if isinstance(displayed_prices, Mapping) else {}
    params = market_params
    protected = _positive_counts(protected_reservations)

    ledger = _Ledger(
        money=money,
        shed=shed,
        seeds=seeds,
        market_inventory=market_inventory,
        unlocked=unlocked,
        shed_capacity=shed_capacity,
        displayed_prices=displayed_prices,
        market_params=params,
    )
    anchor = sell_bin_anchor(int(obs.get("hour", 0)))
    for (old_anchor, product), requested in tuple(state.sell_requested.items()):
        if old_anchor < anchor and (old_anchor, product) not in state.closed_sell_bins:
            remaining = max(
                0, requested - state.sell_submitted.get((old_anchor, product), 0)
            )
            if remaining:
                state.closed_sell_bins[(old_anchor, product)] = remaining
    intents: list[MarketIntent] = []
    blocked: dict[str, dict[str, Any]] = {}

    # Manager sales are always the first class, and only the active bin is
    # eligible.  There is deliberately no economic retention policy here.
    sell_quantities = daily_plan.sell_quantities_dict
    active_bin = sell_quantities.get(str(anchor), {})
    for product in PRODUCTS:
        requested = max(0, int(active_bin.get(product, 0)))
        key = f"SELL:{anchor}:{product}"
        state.sell_requested[(anchor, product)] = requested
        remaining = requested - state.sell_submitted.get((anchor, product), 0)
        if remaining <= 0:
            continue
        intents.append(MarketIntent(key, "SELL", product, remaining, anchor))

    # Packet 1 is the authority for all represented executor deficits.
    feed_demand = sum(
        requirement.quantity
        for item in work_plan.items
        if item.kind == "FEED"
        for requirement in item.required_supplies
        if requirement.scope == "inventory" and requirement.item == "WHEAT"
    )
    # The current observation is authoritative: observed shed already contains any
    # realized purchase, so historical ``buy_observed`` must not be netted again.
    wheat_shortage = max(0, feed_demand - carried.get("WHEAT", 0) - shed.get("WHEAT", 0))
    if wheat_shortage:
        intents.append(MarketIntent("BUY_PRODUCT:WHEAT", "BUY_PRODUCT", "WHEAT", wheat_shortage))

    seed_demand = {crop: 0 for crop in CROP_ORDER}
    for item in work_plan.items:
        # Tile-less unresolved crop targets intentionally carry no speculative
        # seed demand; only spatially represented PLANT work can trigger a
        # global-seed purchase.
        if item.kind != "PLANT" or item.tile is None:
            continue
        for requirement in item.required_supplies:
            if requirement.scope == "global_seed" and requirement.item in seed_demand:
                seed_demand[requirement.item] += requirement.quantity
    for crop in CROP_ORDER:
        # Observed seeds are authoritative and already include realized buys.
        shortage = max(0, seed_demand[crop] - seeds.get(crop, 0))
        if shortage:
            intents.append(MarketIntent(f"BUY_SEED:{crop}", "BUY_SEED", crop, shortage))

    animal_demand = {animal: 0 for animal in ANIMAL_ORDER}
    for item in work_plan.items:
        if item.kind == "BUY_ANIMAL" and item.animal in animal_demand:
            animal_demand[item.animal] += max(1, int(item.quantity))
    for animal in ANIMAL_ORDER:
        # Packet 1 already drops concrete BUY_ANIMAL work once the animal is in
        # the observed shed/carried inventory, so no historical netting here.
        if animal_demand[animal]:
            intents.append(MarketIntent(
                f"BUY_ANIMAL:{animal}", "BUY_ANIMAL", animal, animal_demand[animal]
            ))

    current_land = len(unlocked)
    next_land = LAND_ORDER[current_land - 1] if current_land - 1 < len(LAND_ORDER) else None
    if purchases_enabled and next_land and any(
        item.kind == "BUY_LAND" and item.land == next_land for item in work_plan.items
    ):
        intents.append(MarketIntent(f"BUY_LAND:{next_land}", "BUY_LAND", next_land, 1))

    orders: list[tuple] = []
    pending: list[MarketPendingOrder] = []
    submitted_buy_this_turn: dict[str, int] = {}
    submitted_sell_by_product: dict[str, int] = {}
    for intent in intents:
        if len(orders) >= max_orders:
            blocked[intent.key] = _blocked(MarketBlockReason.ORDER_CAP, intent.requested)
            continue
        if intent.key in state.failed_intents:
            blocked[intent.key] = _blocked(MarketBlockReason.FAILED, intent.requested)
            continue
        if intent.kind == "SELL":
            available = max(0, ledger.shed.get(intent.item or "", 0) - protected.get(intent.item or "", 0))
            quantity = min(intent.requested, available)
            if quantity <= 0:
                reason = (MarketBlockReason.RESERVATION_PROTECTED
                          if ledger.shed.get(intent.item or "", 0) > 0 else MarketBlockReason.NOT_IN_SHED)
                blocked[intent.key] = _blocked(reason, intent.requested)
                continue
            price = ledger.price_for_sell(intent.item)
            if price is None:
                blocked[intent.key] = _blocked(MarketBlockReason.MISSING_PRICE, intent.requested)
                continue
            orders.append(("SELL", intent.item, quantity))
            state.sell_submitted[(intent.anchor or 0, intent.item or "")] = (
                state.sell_submitted.get((intent.anchor or 0, intent.item or ""), 0) + quantity
            )
            ledger.sell(intent.item, quantity)
            submitted_sell_by_product[intent.item or ""] = quantity
            continue

        if not purchases_enabled:
            blocked[intent.key] = _blocked(MarketBlockReason.FAILED, intent.requested)
            continue
        quantity, reason = ledger.affordable_quantity(intent)
        if quantity <= 0:
            blocked[intent.key] = _blocked(reason or MarketBlockReason.CASH, intent.requested)
            continue
        if intent.kind == "BUY_LAND":
            orders.append(("BUY_LAND",))
            ledger.buy_land(intent.item)
            quantity = 1
        else:
            orders.append((intent.kind, intent.item, quantity))
            ledger.buy(intent.kind, intent.item, quantity)
            if quantity < intent.requested:
                remaining_room = max(0, shed_capacity - sum(ledger.shed.values()))
                blocked[intent.key] = {
                    "requested": intent.requested,
                    "submitted": quantity,
                    "unfulfilled": intent.requested - quantity,
                    "block_reason": (
                        MarketBlockReason.SHED_CAPACITY.value
                        if remaining_room == 0
                        else MarketBlockReason.CASH.value
                    ),
                }
        state.buy_submitted[intent.key] = state.buy_submitted.get(intent.key, 0) + quantity
        submitted_buy_this_turn[intent.key] = quantity
        before = _observed_quantity(intent, observed_shed, observed_seeds, observed_unlocked)
        pending.append(MarketPendingOrder(
            intent.key, intent.kind, intent.item, quantity, before,
            int(obs.get("step", 0)), submitted_sell_by_product.get(intent.item or "", 0),
        ))

    diagnostics = {
        "market_orders": [list(order) for order in orders],
        "market_order_count": len(orders),
        "market_intents": [asdict(intent) for intent in intents],
        "market_blocked": blocked,
        "money_before_plan": money,
        "money_after_simulated_orders": ledger.money,
        "shed_occupancy_before_plan": sum(shed.values()),
        "shed_occupancy_after_simulated_orders": sum(ledger.shed.values()),
        "shed_reservation_protected": dict(sorted(protected.items())),
        "buy_demand": {intent.key: intent.requested for intent in intents if intent.kind.startswith("BUY_")},
        "buy_submitted_this_turn": submitted_buy_this_turn,
        "sell_requested_by_bin": {
            f"{bin_anchor}:{product}": quantity
            for (bin_anchor, product), quantity in state.sell_requested.items()
        },
        "sell_submitted_by_bin": {
            f"{bin_anchor}:{product}": quantity
            for (bin_anchor, product), quantity in state.sell_submitted.items()
        },
        "active_sell_bin": anchor,
    }
    return MarketTurnPlan(tuple(orders), tuple(intents), tuple(pending), diagnostics)


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _positive_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(item): max(0, int(amount)) for item, amount in value.items()}


def _numeric_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(item): int(amount) for item, amount in value.items()}


def _sum_inventories(inventories: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(inventories, (list, tuple)):
        return result
    for inventory in inventories:
        for item, amount in _positive_counts(inventory).items():
            result[item] = result.get(item, 0) + amount
    return result


def _blocked(reason: MarketBlockReason, requested: int) -> dict[str, Any]:
    return {"requested": requested, "submitted": 0, "unfulfilled": requested, "block_reason": reason.value}


def _observed_quantity(intent: MarketIntent, shed: Mapping[str, int], seeds: Mapping[str, int], unlocked: tuple[str, ...]) -> int:
    if intent.kind == "BUY_SEED":
        return int(seeds.get(intent.item or "", 0))
    if intent.kind == "BUY_LAND":
        return int(intent.item in unlocked)
    return int(shed.get(intent.item or "", 0))


@dataclass
class _Ledger:
    money: float
    shed: dict[str, int]
    seeds: dict[str, int]
    market_inventory: dict[str, int]
    unlocked: tuple[str, ...]
    shed_capacity: int
    displayed_prices: Mapping[str, Any]
    market_params: Mapping[str, Mapping[str, Any]] | None

    def price_for_sell(self, item: str | None) -> int | None:
        return self._price(item, buy=False)

    def _price(self, item: str | None, *, buy: bool) -> int | None:
        if not item or item not in PRODUCTS:
            return None
        displayed = self.displayed_prices.get(item)
        try:
            if int(displayed) <= 0:
                return None
        except (TypeError, ValueError):
            return None
        inventory = self.market_inventory.get(item)
        if inventory is None:
            return None
        return market_price(item, inventory - (1 if buy else 0), self.market_params)

    def affordable_quantity(self, intent: MarketIntent) -> tuple[int, MarketBlockReason | None]:
        desired = intent.requested
        if intent.kind == "BUY_LAND":
            index = len(self.unlocked) - 1
            cost = LAND_PRICES[index] if 0 <= index < len(LAND_PRICES) else None
            if cost is None or self.money < cost:
                return 0, MarketBlockReason.CASH
            return 1, None
        if intent.kind == "BUY_SEED":
            cost = int(CROPS[intent.item or ""]["seed"])
            return min(desired, int(self.money // cost)), MarketBlockReason.CASH
        room = max(0, self.shed_capacity - sum(self.shed.values()))
        if room <= 0:
            return 0, MarketBlockReason.SHED_CAPACITY
        if intent.kind == "BUY_ANIMAL":
            cost = int(ANIMALS[intent.item or ""]["cost"])
            return min(desired, room, int(self.money // cost)), MarketBlockReason.CASH
        if intent.kind == "BUY_PRODUCT":
            quantity = 0
            for _ in range(min(desired, room)):
                price = self._price(intent.item, buy=True)
                if price is None or self.money < price:
                    break
                self.money -= price
                quantity += 1
                self.market_inventory[intent.item or ""] = self.market_inventory.get(intent.item or "", 0) - 1
            # The speculative units are already charged above; ``buy`` only
            # updates shed and the caller does not charge them a second time.
            if quantity:
                return quantity, None
            if self._price(intent.item, buy=True) is None:
                return 0, MarketBlockReason.MISSING_PRICE
            return 0, MarketBlockReason.CASH
        return 0, MarketBlockReason.CASH

    def buy(self, kind: str, item: str | None, quantity: int) -> None:
        if kind == "BUY_SEED":
            cost = int(CROPS[item or ""]["seed"]) * quantity
            self.money -= cost
            self.seeds[item or ""] = self.seeds.get(item or "", 0) + quantity
        elif kind == "BUY_ANIMAL":
            self.money -= int(ANIMALS[item or ""]["cost"]) * quantity
            self.shed[item or ""] = self.shed.get(item or "", 0) + quantity
        elif kind == "BUY_PRODUCT":
            # affordable_quantity already simulated the dynamic cash quote.
            self.shed[item or ""] = self.shed.get(item or "", 0) + quantity

    def buy_land(self, quadrant: str | None) -> None:
        index = len(self.unlocked) - 1
        self.money -= LAND_PRICES[index]
        self.unlocked = (*self.unlocked, quadrant or LAND_ORDER[index])

    def sell(self, item: str | None, quantity: int) -> None:
        if not item:
            return
        for _ in range(quantity):
            price = self._price(item, buy=False)
            if price is None:
                return
            self.shed[item] = self.shed.get(item, 0) - 1
            self.money += price
            if price > 1:
                self.market_inventory[item] = self.market_inventory.get(item, 0) + 1
