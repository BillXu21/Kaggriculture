"""Pure, experimental strip work representation and forecasting.

This module describes mechanical work implied by a :class:`DailyPlan`.  It
does not assign workers, route them, buy anything, dispatch actions, or mutate
the observation or plan.  Coordinates are always canonical ``(y, x)`` board
coordinates and the board is read from ``obs['farms'][seat]['tiles']``.

Manager/executor boundary for the strip experiment: the manager owns crop
targets, animal targets, the land target, and strategic sell intent.  The
strip executor owns watering, harvest mechanics, feeding, care, and
wheat/strawberry fertilizer timing under configured permissions.  In
particular this builder ignores ``plan.care_by_animal`` and
``plan.fertilizer_by_crop``; CARE and fertilizer work is generated from
mechanical eligibility, not legacy count fields.

Status contract: ``READY`` means the primitive interaction is mechanically
executable from the current represented state (subject only to
worker/location assignment, which Packet 1 does not model).  Any item with a
non-empty ``depends_on`` refers to future sequential work that has not yet
occurred, so it is ``BLOCKED`` with ``DEPENDENCY_BLOCKED`` unless a more
specific own reason (missing supply, missing purchase, locked land, ...)
applies.  Chains may therefore be foreseeable/feasible as a whole while only
their first primitive is ``READY``; workload totals still include every
member.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil
from typing import Any, Iterable

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.layout import (
    SacrificeConfig,
    SHED_HUB_ANCHOR,
    plan_day_layouts,
    quadrant_of,
    tile_role,
)
from executor_v0.plan import DailyPlan
from executor_v0.upkeep import (
    care_has_payoff,
    fertilizer_extra_units,
    wheat_harvest_eligibility,
)
from replay_daily.constants import ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, PRODUCTS
from replay_daily.lifecycle import canonical_board, resolve_observation_step

__all__ = [
    "BlockReason",
    "RowKey",
    "WorkStatus",
    "WorkItem",
    "EffectiveInteractionForecast",
    "WorkChain",
    "SupplyRequirement",
    "SupplyDemand",
    "SupplySnapshot",
    "RowSummary",
    "RowWorkload",
    "WorkDiagnostics",
    "StripWorkConfig",
    "StripWorkPlan",
    "StripWorkResult",
    "forecast_effective_interactions",
    "row_key_for_tile",
    "build_strip_work_plan",
]


_RETAINED_ONE_SHOT_CROPS = frozenset(("WHEAT", "CARROT", "MELON"))


class WorkStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNRESOLVED = "UNRESOLVED"


class BlockReason(StrEnum):
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    MISSING_SUPPLY = "MISSING_SUPPLY"
    MISSING_GLOBAL_RESOURCE = "MISSING_GLOBAL_RESOURCE"
    MISSING_PURCHASE = "MISSING_PURCHASE"
    LOCKED_LAND = "LOCKED_LAND"
    NO_SPATIAL_SLOT = "NO_SPATIAL_SLOT"
    DEADLINE_INFEASIBLE = "DEADLINE_INFEASIBLE"
    OTHER_MECHANICAL_BLOCK = "OTHER_MECHANICAL_BLOCK"


@dataclass(frozen=True, order=True)
class RowKey:
    quadrant: str
    local_row: int
    global_row: int
    x_start: int
    x_end: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def row_key_for_tile(tile: tuple[int, int]) -> RowKey:
    """Return the stable five-tile row segment containing ``(y, x)``."""
    y, x = tile
    if not (0 <= y < 10 and 0 <= x < 10):
        raise ValueError(f"tile must be a board (y, x) coordinate, got {tile!r}")
    return RowKey(quadrant_of(y, x), y % 5, y, 0 if x < 5 else 5, 4 if x < 5 else 9)


@dataclass(frozen=True, order=True)
class SupplyRequirement:
    item: str
    quantity: int = 1
    scope: str = "inventory"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class SupplyDemand:
    item: str
    requested: int
    available: int
    missing: int
    scope: str = "inventory"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pairs(
    value: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted((str(k), int(v)) for k, v in dict(value).items() if int(v) != 0)
    )


def _dict(pairs: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(pairs)


@dataclass(frozen=True)
class SupplySnapshot:
    """Immutable supply view; seeds are deliberately separate from inventory."""

    shed: tuple[tuple[str, int], ...] = ()
    carried: tuple[tuple[str, int], ...] = ()
    seeds: tuple[tuple[str, int], ...] = ()

    @property
    def shed_dict(self) -> dict[str, int]:
        return _dict(self.shed)

    @property
    def carried_dict(self) -> dict[str, int]:
        return _dict(self.carried)

    @property
    def seeds_dict(self) -> dict[str, int]:
        return _dict(self.seeds)

    def inventory_amount(self, item: str) -> int:
        return self.shed_dict.get(item, 0) + self.carried_dict.get(item, 0)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "shed": self.shed_dict,
            "carried": self.carried_dict,
            "seeds": self.seeds_dict,
        }


@dataclass(frozen=True)
class StripWorkConfig:
    """Small policy surface for this experiment, not executor configuration."""

    acting_seat: int | None = None
    allow_wheat_fertilizer: bool = True
    allow_strawberry_fertilizer: bool = True
    wheat_fertilizer_ages: tuple[int, ...] = (2,)
    strawberry_fertilizer_ages: tuple[int, ...] = (9, 11, 13, 15)
    anchor: tuple[int, int] = SHED_HUB_ANCHOR
    pickup_batch: int = 1
    wheat_harvest_threshold: bool = True


@dataclass(frozen=True)
class WorkItem:
    """One mechanical interaction, with no worker or route attached."""

    id: str
    kind: str
    status: WorkStatus = WorkStatus.READY
    block_reason: BlockReason | None = None
    tile: tuple[int, int] | None = None
    crop: str | None = None
    animal: str | None = None
    product: str | None = None
    quantity: int = 1
    depends_on: tuple[str, ...] = ()
    required_supplies: tuple[SupplyRequirement, ...] = ()
    interaction_turns: int = 1
    pickup_turns: int = 0
    deposit_turns: int = 0
    known_dependency_turns: int = 0
    travel_turns: int = 0
    row_key: RowKey | None = None
    source: str = "strip_forecast"
    land: str | None = None

    @property
    def key(self) -> str:
        return self.id

    @property
    def required_supply_dict(self) -> dict[str, int]:
        return {r.item: r.quantity for r in self.required_supplies}

    @property
    def ready(self) -> bool:
        return self.status == WorkStatus.READY

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        out["block_reason"] = self.block_reason.value if self.block_reason else None
        out["tile"] = list(self.tile) if self.tile is not None else None
        out["required_supplies"] = [r.to_json_dict() for r in self.required_supplies]
        out["row_key"] = self.row_key.to_json_dict() if self.row_key else None
        return out


@dataclass(frozen=True)
class EffectiveInteractionForecast:
    """Represented work plus deterministic same-day continuation work."""

    represented_interactions: int
    known_continuation_interactions: int
    continuation_stages_by_work_item: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def effective_interactions(self) -> int:
        return self.represented_interactions + self.known_continuation_interactions


def forecast_effective_interactions(
    items: Iterable[WorkItem],
) -> EffectiveInteractionForecast:
    """Return the shared deadline/hiring interaction forecast for ``items``.

    Retained one-shot routine harvests are known to continue through PLANT and
    WATER on the same day.  Those two interactions are forecast only when the
    corresponding stages are not already represented on that tile.
    """

    represented = tuple(items)
    kinds_by_tile: dict[tuple[int, int], set[str]] = defaultdict(set)
    retained_harvest_by_tile: dict[tuple[int, int], str] = {}
    for item in represented:
        if item.tile is None:
            continue
        kinds_by_tile[item.tile].add(item.kind)
        if (
            item.kind == "HARVEST"
            and item.source == "routine_harvest"
            and item.crop in _RETAINED_ONE_SHOT_CROPS
        ):
            retained_harvest_by_tile.setdefault(item.tile, item.id)

    continuation_stages = tuple(
        (
            work_id,
            tuple(
                stage
                for stage in ("PLANT", "WATER")
                if stage not in kinds_by_tile[tile]
            ),
        )
        for tile, work_id in sorted(retained_harvest_by_tile.items())
        if any(
            stage not in kinds_by_tile[tile] for stage in ("PLANT", "WATER")
        )
    )
    return EffectiveInteractionForecast(
        represented_interactions=sum(
            max(0, int(item.interaction_turns)) for item in represented
        ),
        known_continuation_interactions=sum(
            len(stages) for _, stages in continuation_stages
        ),
        continuation_stages_by_work_item=continuation_stages,
    )


@dataclass(frozen=True)
class WorkChain:
    id: str
    kind: str
    item_ids: tuple[str, ...]
    status: WorkStatus
    block_reason: BlockReason | None = None
    tile: tuple[int, int] | None = None
    crop: str | None = None
    animal: str | None = None
    product: str | None = None
    interaction_turns: int = 0
    pickup_turns: int = 0
    deposit_turns: int = 0
    known_dependency_turns: int = 0
    travel_turns: int = 0
    row_key: RowKey | None = None
    source: str = "strip_forecast"
    land: str | None = None

    @property
    def key(self) -> str:
        return self.id

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        out["block_reason"] = self.block_reason.value if self.block_reason else None
        out["tile"] = list(self.tile) if self.tile is not None else None
        out["row_key"] = self.row_key.to_json_dict() if self.row_key else None
        return out


@dataclass(frozen=True)
class RowSummary:
    row_key: RowKey
    tile_count: int = 0
    ready_interactions: int = 0
    future_interactions: int = 0
    nontravel_turns: int = 0
    feed_quantity: int = 0
    fertilizer_quantity: int = 0
    animal_quantity: int = 0
    seed_quantity: int = 0
    active_chains: int = 0
    blocked_by_reason: tuple[tuple[str, int], ...] = ()

    @property
    def blocked_by_reason_dict(self) -> dict[str, int]:
        return _dict(self.blocked_by_reason)

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["row_key"] = self.row_key.to_json_dict()
        out["blocked_by_reason"] = self.blocked_by_reason_dict
        return out


@dataclass(frozen=True)
class RowWorkload:
    row_key: RowKey
    interactions: int
    nontravel_turns: int
    ready_interactions: int
    blocked_interactions: int

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["row_key"] = self.row_key.to_json_dict()
        return out


@dataclass(frozen=True)
class WorkDiagnostics:
    requested_crop_delta: tuple[tuple[str, int], ...] = ()
    represented_crop_delta: tuple[tuple[str, int], ...] = ()
    unresolved_crop_delta: tuple[tuple[str, int], ...] = ()
    requested_animal_delta: tuple[tuple[str, int], ...] = ()
    represented_animal_delta: tuple[tuple[str, int], ...] = ()
    unresolved_animal_delta: tuple[tuple[str, int], ...] = ()
    requested_land_count: int = 0
    current_land_count: int = 0
    unresolved_land_delta: int = 0
    work_counts_by_kind: tuple[tuple[str, int], ...] = ()
    work_counts_by_status: tuple[tuple[str, int], ...] = ()
    work_counts_by_reason: tuple[tuple[str, int], ...] = ()
    supply_demand: tuple[SupplyDemand, ...] = ()
    row_workload: tuple[RowWorkload, ...] = ()

    @property
    def requested_crop_delta_dict(self) -> dict[str, int]:
        return _dict(self.requested_crop_delta)

    @property
    def represented_crop_delta_dict(self) -> dict[str, int]:
        return _dict(self.represented_crop_delta)

    @property
    def unresolved_crop_delta_dict(self) -> dict[str, int]:
        return _dict(self.unresolved_crop_delta)

    @property
    def requested_animal_delta_dict(self) -> dict[str, int]:
        return _dict(self.requested_animal_delta)

    @property
    def represented_animal_delta_dict(self) -> dict[str, int]:
        return _dict(self.represented_animal_delta)

    @property
    def unresolved_animal_delta_dict(self) -> dict[str, int]:
        return _dict(self.unresolved_animal_delta)

    @property
    def work_counts_kind(self) -> dict[str, int]:
        return _dict(self.work_counts_by_kind)

    @property
    def work_counts_status(self) -> dict[str, int]:
        return _dict(self.work_counts_by_status)

    @property
    def work_counts_reason(self) -> dict[str, int]:
        return _dict(self.work_counts_by_reason)

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for name in (
            "requested_crop_delta",
            "represented_crop_delta",
            "unresolved_crop_delta",
            "requested_animal_delta",
            "represented_animal_delta",
            "unresolved_animal_delta",
            "work_counts_by_kind",
            "work_counts_by_status",
            "work_counts_by_reason",
        ):
            out[name] = dict(getattr(self, name))
        out["supply_demand"] = [d.to_json_dict() for d in self.supply_demand]
        out["row_workload"] = [r.to_json_dict() for r in self.row_workload]
        return out


@dataclass(frozen=True)
class StripWorkPlan:
    items: tuple[WorkItem, ...]
    chains: tuple[WorkChain, ...]
    row_summaries: tuple[RowSummary, ...]
    supply: SupplySnapshot
    diagnostics: WorkDiagnostics
    acting_seat: int

    @property
    def work_items(self) -> tuple[WorkItem, ...]:
        return self.items

    @property
    def row_summary_by_key(self) -> dict[RowKey, RowSummary]:
        return {r.row_key: r for r in self.row_summaries}

    @property
    def rows_by_key(self) -> dict[RowKey, RowSummary]:
        return self.row_summary_by_key

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "acting_seat": self.acting_seat,
            "items": [i.to_json_dict() for i in self.items],
            "chains": [c.to_json_dict() for c in self.chains],
            "row_summaries": [r.to_json_dict() for r in self.row_summaries],
            "supply": self.supply.to_json_dict(),
            "diagnostics": self.diagnostics.to_json_dict(),
        }


StripWorkResult = StripWorkPlan


def _as_config(config: StripWorkConfig | Mapping[str, Any]) -> StripWorkConfig:
    if isinstance(config, StripWorkConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be StripWorkConfig or a mapping")
    return StripWorkConfig(**dict(config))


def _seat_for(
    obs: Mapping[str, Any],
    config: StripWorkConfig,
    acting_seat: int | None,
    seat: int | None,
) -> int:
    chosen = acting_seat if acting_seat is not None else seat
    if chosen is None:
        chosen = config.acting_seat
    if chosen is None:
        chosen = obs.get("player", 0)
    if isinstance(chosen, bool) or not isinstance(chosen, int):
        raise ValueError(f"acting seat must be an integer, got {chosen!r}")
    farms = obs.get("farms")
    if not isinstance(farms, (list, tuple)) or not 0 <= chosen < len(farms):
        raise ValueError(f"acting seat {chosen} is not present in obs['farms']")
    return chosen


def _state(
    obs: Mapping[str, Any], seat: int
) -> tuple[list[list[Any]], tuple[str, ...], SupplySnapshot, int, float]:
    farm = obs["farms"][seat]
    raw_tiles = farm.get("tiles")
    if (
        not isinstance(raw_tiles, list)
        or len(raw_tiles) != 10
        or any(not isinstance(row, list) or len(row) != 10 for row in raw_tiles)
    ):
        raise ValueError("acting farm tiles must be a 10x10 list")
    day = int(obs.get("day", 0))
    step = resolve_observation_step(obs)
    board = canonical_board(raw_tiles, day, step)
    unlocked = tuple(
        sorted(str(q) for q in (farm.get("unlocked_quadrants") or ("NW",)))
    )
    private = obs.get("private") or {}
    shed = {str(k): int(v) for k, v in (private.get("shed") or {}).items()}
    seeds = {str(k): int(v) for k, v in (private.get("seeds") or {}).items()}
    carried: dict[str, int] = defaultdict(int)
    inventories = private.get("inventories") or ()
    if isinstance(inventories, Mapping):
        inventories = (inventories,)
    for inventory in inventories:
        if isinstance(inventory, Mapping):
            for item, amount in inventory.items():
                carried[str(item)] += int(amount)
    return (
        board,
        unlocked,
        SupplySnapshot(_pairs(shed), _pairs(carried), _pairs(seeds)),
        day,
        float(farm.get("money", 0.0)),
    )


def _tile_harvestable(tile: Mapping[str, Any], day: int, step: int) -> bool:
    if tile_role(tile) != "plant":
        return False
    crop = tile.get("crop")
    if crop == "WHEAT":
        return wheat_harvest_eligibility(tile, day, step)[0]
    if crop not in CROPS or int(tile.get("yield_units", 0) or 0) <= 0:
        return False
    planted = tile.get("planted_day")
    if planted is None:
        return bool((tile.get("derived") or {}).get("currently_harvestable"))
    return day - int(planted) >= CROPS[crop]["first_yield_day"]


_ROUTINE_WATER_AGES: dict[str, frozenset[int]] = {
    "WHEAT": frozenset((0, 2, 3, 4)),
    "CARROT": frozenset((0, 2, 3)),
    "MELON": frozenset((0, 2, 4, 6, 7, 8, 9, 10, 11, 12)),
    "STRAWBERRY": frozenset((0, 2, 4, 6, 8, 9, 11, 13, 15)),
    "TOMATO": frozenset((0, 2, 4, 6)),
}


def _removal_action_id(action: str, coord: tuple[int, int]) -> str:
    """Stable id for one crop-removal lifecycle action on ``coord``.

    A preparatory WATER that makes a not-yet-harvestable crop harvestable and
    the later replacement WATER on the same tile are two distinct operations,
    so the removal-owned WATER must not share the coordinate-only
    ``WATER:y,x`` id used by routine upkeep and the planting continuation:
    sharing it would merge the two into a single item and form a dependency
    cycle (WATER depends on PLANT while PLANT depends on WATER).  Every other
    removal action keeps the coordinate id so it dedupes with routine harvest
    work exactly as before.
    """
    if action == "WATER":
        return f"REMOVAL_WATER:{coord[0]},{coord[1]}"
    return f"{action}:{coord[0]},{coord[1]}"


def _routine_water_source(
    tile: Mapping[str, Any], crop: str, day: int, step: int
) -> str | None:
    """Return the stable reason for routine watering, if due today.

    The age sets mirror the executor's default routine-water candidates.  The
    two late MELON ages and WHEAT age four are retained only when the observed
    lifecycle still has yield room.  Actual derived lifecycle state can also
    suppress watering past expiry or after a harvestable crop has no useful
    room left.
    """
    if tile.get("watered_today") is True:
        return None
    derived = tile.get("derived") or {}
    if derived.get("past_lifespan"):
        return None
    if int(tile.get("consecutive_unwatered", 0) or 0) >= 1:
        return "survival_weed_prevention"
    planted = tile.get("planted_day")
    age_value = derived.get("age_days")
    if age_value is None and planted is not None:
        age_value = day - int(planted)
    if age_value is None:
        return None
    age = int(age_value)
    if crop == "WHEAT" and wheat_harvest_eligibility(tile, day, step)[0]:
        return None
    if age not in _ROUTINE_WATER_AGES[crop]:
        if crop == "TOMATO" and age >= CROPS[crop]["first_yield_day"]:
            held = int(tile.get("yield_units", 0) or 0)
            if held < CROPS[crop]["max_yield"]:
                return "yield_improving"
        return None
    held = int(tile.get("yield_units", 0) or 0)
    room = held < CROPS[crop]["max_yield"]
    if age in (11, 12) and crop == "MELON" and not room:
        return None
    if crop == "WHEAT" and age == 4:
        if not room:
            return None
    if age == 0:
        return "planting_continuation"
    if age >= CROPS[crop]["first_yield_day"] and room:
        return "yield_improving"
    return "optional_deferrable"


def _unlocked_counts(
    board: list[list[Any]], unlocked: tuple[str, ...]
) -> tuple[dict[str, int], dict[str, int]]:
    crops = {c: 0 for c in CROP_ORDER}
    animals = {a: 0 for a in ANIMAL_ORDER}
    allowed = set(unlocked)
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if quadrant_of(y, x) not in allowed or not isinstance(tile, Mapping):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[tile["crop"]] += 1
            if "animal" in tile and tile.get("animal") in animals:
                animals[tile["animal"]] += 1
    return crops, animals


class _Builder:
    def __init__(self, supply: SupplySnapshot, config: StripWorkConfig):
        self.supply, self.config = supply, config
        self.seeds = supply.seeds_dict
        self.shed = supply.shed_dict
        self.carried = supply.carried_dict
        self.items: dict[str, WorkItem] = {}
        self.chains: list[WorkChain] = []
        self.seed_remaining = self.seeds.copy()
        self.inventory_remaining = self.shed.copy()
        for item, amount in self.carried.items():
            self.inventory_remaining[item] = (
                self.inventory_remaining.get(item, 0) + amount
            )
        self.demands: Counter[tuple[str, str]] = Counter()

    def _requirements(
        self, requirements: tuple[SupplyRequirement, ...]
    ) -> None | BlockReason:
        reason = None
        for requirement in requirements:
            pool = (
                self.seed_remaining
                if requirement.scope == "global_seed"
                else self.inventory_remaining
            )
            available = pool.get(requirement.item, 0)
            pool[requirement.item] = max(0, available - requirement.quantity)
            self.demands[(requirement.item, requirement.scope)] += requirement.quantity
            if available < requirement.quantity:
                candidate = (
                    BlockReason.MISSING_GLOBAL_RESOURCE
                    if requirement.scope == "global_seed"
                    else BlockReason.MISSING_SUPPLY
                )
                reason = reason or candidate
        return reason

    def add(
        self,
        *,
        id: str,
        kind: str,
        tile: tuple[int, int] | None = None,
        crop: str | None = None,
        animal: str | None = None,
        product: str | None = None,
        quantity: int = 1,
        depends_on: Iterable[str] = (),
        requirements: tuple[SupplyRequirement, ...] = (),
        block_reason: BlockReason | None = None,
        pickup_turns: int = 0,
        deposit_turns: int = 0,
        land: str | None = None,
        source: str = "strip_forecast",
    ) -> WorkItem:
        """Register one primitive interaction.

        ``READY`` means executable now from the represented state.  A
        non-empty ``depends_on`` always names represented future work that has
        not occurred yet, so the item is ``BLOCKED``/``DEPENDENCY_BLOCKED``
        unless its own mechanics or supplies already fail with a more specific
        reason (which takes precedence).
        """
        deps = tuple(sorted(set(depends_on)))
        supply_reason = self._requirements(requirements)
        own_reason = block_reason or supply_reason
        if own_reason is None and deps:
            own_reason = BlockReason.DEPENDENCY_BLOCKED
        status = WorkStatus.BLOCKED if own_reason else WorkStatus.READY
        if id in self.items:
            old = self.items[id]
            deps = tuple(sorted(set(old.depends_on) | set(deps)))
            own_reason = old.block_reason or own_reason
            if own_reason is None and deps:
                own_reason = BlockReason.DEPENDENCY_BLOCKED
            status = WorkStatus.BLOCKED if own_reason else WorkStatus.READY
            requirements = old.required_supplies + tuple(
                r for r in requirements if r not in old.required_supplies
            )
            pickup_turns, deposit_turns = (
                max(old.pickup_turns, pickup_turns),
                max(old.deposit_turns, deposit_turns),
            )
        item = WorkItem(
            id,
            kind,
            status,
            own_reason,
            tile,
            crop,
            animal,
            product,
            quantity,
            deps,
            requirements,
            1,
            pickup_turns,
            deposit_turns,
            len(deps),
            0,
            row_key_for_tile(tile) if tile is not None else None,
            source,
            land,
        )
        self.items[id] = item
        return item

    def chain(
        self,
        id: str,
        kind: str,
        item_ids: Iterable[str],
        *,
        tile=None,
        crop=None,
        animal=None,
        product=None,
        land=None,
        source="strip_forecast",
    ) -> None:
        ids = tuple(item_ids)
        values = tuple(self.items[i] for i in ids)
        blocked = next(
            (i.block_reason for i in values if i.status != WorkStatus.READY), None
        )
        self.chains.append(
            WorkChain(
                id,
                kind,
                ids,
                WorkStatus.BLOCKED if blocked else WorkStatus.READY,
                blocked,
                tile,
                crop,
                animal,
                product,
                sum(i.interaction_turns for i in values),
                sum(i.pickup_turns for i in values),
                sum(i.deposit_turns for i in values),
                sum(i.known_dependency_turns for i in values),
                0,
                row_key_for_tile(tile) if tile is not None else None,
                source,
                land,
            )
        )

    def resolve_dependencies(self) -> None:
        # Enforce the READY-means-executable invariant: an item with declared
        # dependencies is future work awaiting those steps, so it stays
        # BLOCKED unless it already carries a more specific own reason.
        # Packet 1 never simulates execution, so dependencies are never
        # "already satisfied" here; this pass is idempotent with add().
        for _ in range(len(self.items) + 1):
            changed = False
            for item_id, item in list(self.items.items()):
                reason = item.block_reason or (
                    BlockReason.DEPENDENCY_BLOCKED if item.depends_on else None
                )
                replacement = WorkItem(
                    item.id,
                    item.kind,
                    WorkStatus.BLOCKED if reason else WorkStatus.READY,
                    reason,
                    item.tile,
                    item.crop,
                    item.animal,
                    item.product,
                    item.quantity,
                    item.depends_on,
                    item.required_supplies,
                    item.interaction_turns,
                    item.pickup_turns,
                    item.deposit_turns,
                    len(item.depends_on),
                    item.travel_turns,
                    item.row_key,
                    item.source,
                    item.land,
                )
                if replacement != item:
                    self.items[item_id], changed = replacement, True
            if not changed:
                break
        for index, chain in enumerate(self.chains):
            values = tuple(self.items[i] for i in chain.item_ids)
            blocked = next(
                (i.block_reason for i in values if i.status != WorkStatus.READY), None
            )
            self.chains[index] = WorkChain(
                chain.id,
                chain.kind,
                chain.item_ids,
                WorkStatus.BLOCKED if blocked else WorkStatus.READY,
                blocked,
                chain.tile,
                chain.crop,
                chain.animal,
                chain.product,
                sum(i.interaction_turns for i in values),
                sum(i.pickup_turns for i in values),
                sum(i.deposit_turns for i in values),
                sum(i.known_dependency_turns for i in values),
                0,
                chain.row_key,
                chain.source,
                chain.land,
            )


def _supply_item(
    builder: _Builder,
    item_id: str,
    kind: str,
    *,
    tile=None,
    crop=None,
    animal=None,
    product=None,
    quantity=1,
    depends_on=(),
    requirements=(),
    block_reason=None,
    source="strip_forecast",
) -> WorkItem:
    carried = builder.carried
    pickup = sum(
        ceil(
            max(0, r.quantity - carried.get(r.item, 0))
            / max(1, builder.config.pickup_batch)
        )
        for r in requirements
        if r.scope == "inventory"
    )
    return builder.add(
        id=item_id,
        kind=kind,
        tile=tile,
        crop=crop,
        animal=animal,
        product=product,
        quantity=quantity,
        depends_on=depends_on,
        requirements=requirements,
        block_reason=block_reason,
        pickup_turns=pickup,
        source=source,
    )


def build_strip_work_plan(
    obs: Mapping[str, Any],
    plan: DailyPlan,
    config: StripWorkConfig | Mapping[str, Any] = StripWorkConfig(),
    *,
    acting_seat: int | None = None,
    seat: int | None = None,
    preferred_crop_slots: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    allow_live_crop_sacrifice: bool = False,
    allow_productive_recurring_crop_sacrifice: bool = False,
    allow_older_crop_sacrifice: bool = False,
) -> StripWorkPlan:
    """Build a deterministic pure work forecast for one acting seat.

    The manager contributes crop/animal/land targets and sell intent; CARE,
    feeding, watering, and fertilizer timing are executor mechanics derived
    from observed state (legacy ``care_by_animal``/``fertilizer_by_crop``
    counts are ignored).  ``READY`` means executable now; anything awaiting a
    represented prerequisite is ``BLOCKED``/``DEPENDENCY_BLOCKED``.
    """
    if not isinstance(obs, Mapping):
        raise TypeError("obs must be a mapping")
    if not isinstance(plan, DailyPlan):
        raise TypeError("plan must be an executor_v0.plan.DailyPlan")
    cfg = _as_config(config)
    chosen_seat = _seat_for(obs, cfg, acting_seat, seat)
    board, unlocked, supply, day, money = _state(obs, chosen_seat)
    preferred_slots = {
        str(crop): tuple((int(coord[0]), int(coord[1])) for coord in coords)
        for crop, coords in (preferred_crop_slots or {}).items()
    }
    step = resolve_observation_step(obs)
    current_crops, current_animals = _unlocked_counts(board, unlocked)
    target_crops, target_animals = plan.crop_targets_dict, plan.animal_targets_dict
    crop_need = {c: max(0, target_crops[c] - current_crops[c]) for c in CROP_ORDER}
    animal_need = {
        a: max(0, target_animals[a] - current_animals[a]) for a in ANIMAL_ORDER
    }
    layouts = plan_day_layouts(
        board,
        unlocked_quadrants=unlocked,
        crop_targets=target_crops,
        animals_needed=animal_need,
        anchor=cfg.anchor,
        config=SacrificeConfig(
            allow_live_crop_sacrifice=allow_live_crop_sacrifice,
            allow_productive_recurring_crop_sacrifice=(
                allow_productive_recurring_crop_sacrifice),
            allow_older_crop_sacrifice=allow_older_crop_sacrifice),
        preferred_crop_slots=preferred_slots,
        current_day=day,
        current_step=step,
    )
    builder = _Builder(supply, cfg)
    current_land = len(unlocked)
    land_item_ids: tuple[str, ...] = ()
    if plan.land_count > current_land:
        # Sequential ordered acquisition: plot N+1 must be bought before plot
        # N+2 is legal, so each purchase depends on the previous one.  Only
        # the next legal purchase can be READY; later ones wait on it.
        # Affordability is charged sequentially against authoritative land
        # prices.  Downstream plot-less work stays LOCKED_LAND on these ids.
        quadrants = LAND_ORDER[max(0, current_land - 1) : plan.land_count - 1]
        ids = []
        remaining_land_money = money
        prev_land_id: str | None = None
        for offset, quadrant in enumerate(quadrants):
            price_index = (current_land - 1) + offset
            cost = (
                float(LAND_PRICES[price_index])
                if 0 <= price_index < len(LAND_PRICES)
                else None
            )
            affordable = cost is not None and remaining_land_money >= cost
            land_id = f"BUY_LAND:{quadrant}"
            builder.add(
                id=land_id,
                kind="BUY_LAND",
                land=quadrant,
                depends_on=(prev_land_id,) if prev_land_id is not None else (),
                block_reason=(
                    None if affordable else BlockReason.MISSING_GLOBAL_RESOURCE
                ),
                source="land_intent",
            )
            builder.chain(
                f"LAND:{quadrant}",
                "LAND_EXPANSION",
                (land_id,),
                land=quadrant,
            )
            ids.append(land_id)
            if affordable and cost is not None:
                remaining_land_money -= cost
            prev_land_id = land_id
        land_item_ids = tuple(ids)
    remaining_money = money
    represented_animals: Counter[str] = Counter()
    for slot in layouts.animals.placements:
        y, x = slot.coord
        tile = board[y][x]
        ids: list[str] = []
        if slot.source == "weed_reclaim":
            dig_id = f"DIG:{y},{x}"
            builder.add(id=dig_id, kind="DIG", tile=slot.coord, source="animal_layout")
            ids.append(dig_id)
        elif slot.source != "empty_structure":
            build_kind = "BUILD_COOP" if slot.structure == "COOP" else "BUILD_PASTURE"
            deps: tuple[str, ...] = ()
            if slot.source in ("crop_release", "crop_sacrifice"):
                removal_actions = slot.removal_actions or (
                    ("DIG",) if slot.source == "crop_sacrifice" else ()
                )
                previous: tuple[str, ...] = ()
                for action in removal_actions:
                    removal_id = _removal_action_id(action, slot.coord)
                    builder.add(
                        id=removal_id,
                        kind=action,
                        tile=slot.coord,
                        crop=tile.get("crop") if isinstance(tile, Mapping) else None,
                        depends_on=previous,
                        source=(
                            "animal_crop_release"
                            if slot.source == "crop_release"
                            else "animal_layout"
                        ),
                    )
                    ids.append(removal_id)
                    previous = (removal_id,)
                deps = previous
            build_id = f"{build_kind}:{y},{x}"
            builder.add(
                id=build_id,
                kind=build_kind,
                tile=slot.coord,
                depends_on=deps,
                source="animal_layout",
            )
            ids.append(build_id)
        place_id = f"PLACE:{slot.animal}:{y},{x}"
        have_animal = builder.inventory_remaining.get(slot.animal, 0)
        purchase_id = None
        purchase_ready = False
        if have_animal <= 0:
            purchase_id = (
                f"BUY_ANIMAL:{slot.animal}:{represented_animals[slot.animal] + 1}"
            )
            animal_cost = float(ANIMALS[slot.animal]["cost"])
            purchase_ready = remaining_money >= animal_cost
            builder.add(
                id=purchase_id,
                kind="BUY_ANIMAL",
                animal=slot.animal,
                block_reason=(
                    None if purchase_ready else BlockReason.MISSING_GLOBAL_RESOURCE
                ),
                source="animal_purchase",
            )
            if purchase_ready:
                remaining_money -= animal_cost
        reason = (
            BlockReason.MISSING_PURCHASE
            if have_animal <= 0 and purchase_ready
            else None
        )
        place_deps = tuple(ids[-1:] + ([purchase_id] if purchase_id else []))
        builder.add(
            id=place_id,
            kind="PLACE",
            tile=slot.coord,
            animal=slot.animal,
            depends_on=place_deps,
            requirements=(SupplyRequirement(slot.animal, 1, "inventory"),)
            if have_animal > 0
            else (),
            block_reason=reason,
            source="animal_layout",
        )
        if purchase_id:
            ids.append(purchase_id)
        ids.append(place_id)
        represented_animals[slot.animal] += 1
        chain_ids = (
            ([purchase_id] if purchase_id else [])
            + [item_id for item_id in ids[:-1] if item_id != purchase_id]
            + [place_id]
        )
        builder.chain(
            f"ANIMAL:{slot.animal}:{y},{x}",
            "ANIMAL_EXPANSION",
            chain_ids,
            tile=slot.coord,
            animal=slot.animal,
        )

    represented_crops: Counter[str] = Counter()
    for slot in layouts.animals.placements:
        y, x = slot.coord
        tile = board[y][x]
        if (
            slot.source in ("crop_release", "crop_sacrifice")
            and isinstance(tile, Mapping)
            and tile_role(tile) == "plant"
        ):
            represented_crops[str(tile.get("crop"))] -= 1
    replacement_harvest_coords: set[tuple[int, int]] = set()
    reduction_harvest_coords: set[tuple[int, int]] = set()
    removal_coords: set[tuple[int, int]] = set()
    handled_removal_coords: set[tuple[int, int]] = set()
    digs_by_coord = {d.coord: d for d in layouts.crops.digs}
    removals_by_coord = {r.coord: r for r in layouts.crops.removals}
    for slot in layouts.animals.placements:
        if slot.source not in ("crop_release", "crop_sacrifice"):
            continue
        removal_coords.add(slot.coord)
        if "HARVEST" in slot.removal_actions:
            reduction_harvest_coords.add(slot.coord)

    def add_removal_actions(
        coord: tuple[int, int],
        crop: str,
        actions: tuple[str, ...],
        *,
        source: str,
        replacement: bool,
    ) -> list[str]:
        ids: list[str] = []
        previous: tuple[str, ...] = ()
        for action in actions:
            action_id = _removal_action_id(action, coord)
            builder.add(
                id=action_id,
                kind=action,
                tile=coord,
                crop=crop,
                depends_on=previous,
                source=source,
            )
            ids.append(action_id)
            previous = (action_id,)
            removal_coords.add(coord)
            if action == "HARVEST":
                (replacement_harvest_coords if replacement else reduction_harvest_coords).add(coord)
        return ids

    for intent in layouts.crops.plants:
        y, x = intent.coord
        old = board[y][x]
        ids: list[str] = []
        removal = removals_by_coord.get(intent.coord)
        if isinstance(old, Mapping) and old.get("kind") == "PLANT":
            actions = removal.actions if removal is not None else ()
            if not actions and intent.coord in digs_by_coord:
                actions = ("DIG",)
            if not actions and _tile_harvestable(old, day, step):
                actions = ("HARVEST",)
            if actions:
                ids.extend(add_removal_actions(
                    intent.coord, str(old.get("crop")), actions,
                    source="crop_replacement", replacement=True))
                represented_crops[str(old.get("crop"))] -= 1
                handled_removal_coords.add(intent.coord)
        elif tile_role(old) == "weed":
            ids.extend(add_removal_actions(
                intent.coord, "WEED", ("DIG",),
                source="crop_replacement", replacement=True))
        plant_id = f"PLANT:{intent.crop}:{y},{x}"
        builder.add(
            id=plant_id,
            kind="PLANT",
            tile=intent.coord,
            crop=intent.crop,
            depends_on=ids,
            requirements=(SupplyRequirement(intent.crop, 1, "global_seed"),),
            source=(
                "retained_crop_maintenance"
                if intent.crop in _RETAINED_ONE_SHOT_CROPS
                and intent.coord in preferred_slots.get(intent.crop, ())
                else "crop_reconciliation"
            ),
        )
        ids.append(plant_id)
        water_id = f"WATER:{y},{x}"
        builder.add(
            id=water_id,
            kind="WATER",
            tile=intent.coord,
            crop=intent.crop,
            depends_on=(plant_id,),
            source="planting_continuation",
        )
        ids.append(water_id)
        builder.chain(
            f"CROP:{intent.crop}:{y},{x}",
            "CROP_GROWTH",
            ids,
            tile=intent.coord,
            crop=intent.crop,
        )
        represented_crops[intent.crop] += 1

    for removal in layouts.crops.removals:
        if removal.coord in handled_removal_coords:
            continue
        ids = add_removal_actions(
            removal.coord, removal.crop, removal.actions,
            source="crop_reduction", replacement=False)
        builder.chain(
            f"CROP_REMOVE:{removal.crop}:{removal.coord[0]},{removal.coord[1]}",
            "CROP_REMOVAL",
            ids,
            tile=removal.coord,
            crop=removal.crop,
            source="crop_reduction",
        )
        represented_crops[removal.crop] -= 1

    # Compatibility for callers that still provide only the legacy ``digs``
    # field in a hand-built reconciliation result.
    for dig in layouts.crops.digs:
        if dig.coord in handled_removal_coords or dig.coord in removal_coords:
            continue
        ids = add_removal_actions(
            dig.coord, dig.crop, ("DIG",),
            source="crop_reduction", replacement=False)
        builder.chain(
            f"CROP_REMOVE:{dig.crop}:{dig.coord[0]},{dig.coord[1]}",
            "CROP_REMOVAL",
            ids,
            tile=dig.coord,
            crop=dig.crop,
            source="crop_reduction",
        )
        represented_crops[dig.crop] -= 1

    claimed_harvest_coords = replacement_harvest_coords | reduction_harvest_coords
    allowed_quadrants = set(unlocked)
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            coord = (y, x)
            if quadrant_of(y, x) not in allowed_quadrants:
                continue
            if coord in claimed_harvest_coords:
                continue
            if not isinstance(tile, Mapping) or tile_role(tile) != "plant":
                continue
            crop = tile.get("crop")
            if crop not in CROPS or not _tile_harvestable(tile, day, step):
                continue
            builder.add(
                id=f"HARVEST:{y},{x}",
                kind="HARVEST",
                tile=coord,
                crop=str(crop),
                source="routine_harvest",
            )

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if quadrant_of(y, x) not in allowed_quadrants:
                continue
            if not isinstance(tile, Mapping) or tile.get("animal") not in ANIMALS:
                continue
            coord = (y, x)
            animal = str(tile["animal"])
            if tile.get("yield_units", 0) > 0:
                builder.add(
                    id=f"HARVEST:{y},{x}",
                    kind="HARVEST",
                    tile=coord,
                    animal=animal,
                    product=str(ANIMALS[animal]["product"]),
                    source="routine_animal_harvest",
                )
            if tile.get("fertilizer_available") is True:
                builder.add(
                    id=f"COLLECT_FERTILIZER:{y},{x}",
                    kind="COLLECT_FERTILIZER",
                    tile=coord,
                    animal=animal,
                    product="FERTILIZER",
                    source="routine_animal_fertilizer_collection",
                )

    unresolved_space_reason = (
        BlockReason.LOCKED_LAND
        if plan.land_count > len(unlocked)
        else BlockReason.NO_SPATIAL_SLOT
    )
    for crop, quantity in layouts.crops.unresolved_deficits:
        builder.add(
            id=f"UNRESOLVED_PLANT:{crop}",
            kind="PLANT",
            crop=crop,
            quantity=quantity,
            depends_on=land_item_ids,
            block_reason=unresolved_space_reason,
            source="crop_unresolved",
        )

    # CARE is owned by the strip executor, not the manager: every animal that
    # is mechanically care-worthwhile gets a service chain.  Legacy
    # plan.care_by_animal counts are deliberately ignored.  An unfed animal
    # needs FEED (supply-checked) before CARE; an already-fed animal gets a
    # standalone READY CARE.  Already-cared or no-payoff animals get nothing.
    for animal in ANIMAL_ORDER:
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if quadrant_of(y, x) not in allowed_quadrants:
                    continue
                if not (isinstance(tile, Mapping) and tile.get("animal") == animal):
                    continue
                if tile.get("cared_today") is True:
                    continue
                care_tile = (
                    tile
                    if tile.get("fed_today") is True
                    else dict(tile, fed_today=True)
                )
                if not care_has_payoff(care_tile, day):
                    continue
                ids: list[str] = []
                if tile.get("fed_today") is not True:
                    feed_id = f"FEED:{animal}:{y},{x}"
                    _supply_item(
                        builder,
                        feed_id,
                        "FEED",
                        tile=(y, x),
                        animal=animal,
                        requirements=(SupplyRequirement("WHEAT", 1, "inventory"),),
                        source="strip_care",
                    )
                    ids.append(feed_id)
                care_id = f"CARE:{animal}:{y},{x}"
                builder.add(
                    id=care_id,
                    kind="CARE",
                    tile=(y, x),
                    animal=animal,
                    depends_on=ids,
                    source="strip_care",
                )
                ids.append(care_id)
                builder.chain(
                    f"CARE:{animal}:{y},{x}",
                    "ANIMAL_CARE",
                    ids,
                    tile=(y, x),
                    animal=animal,
                )

    for animal, quantity in layouts.animals.unresolved:
        builder.add(
            id=f"UNRESOLVED_PLACE:{animal}",
            kind="PLACE",
            animal=animal,
            quantity=quantity,
            depends_on=land_item_ids,
            block_reason=unresolved_space_reason,
            source="animal_unresolved",
        )

    # Fertilizer timing is owned by the strip experiment, not the manager:
    # legacy plan.fertilizer_by_crop counts are deliberately ignored.  When a
    # crop permission is ON, every mechanically eligible plant is represented.
    # The supply ledger marks the first affordable applications READY (stable
    # y,x order) and the excess MISSING_SUPPLY, so total useful demand stays
    # visible under scarcity.  No purchase/top-up is ever created.
    for crop, enabled, ages in (
        ("WHEAT", cfg.allow_wheat_fertilizer, cfg.wheat_fertilizer_ages),
        ("STRAWBERRY", cfg.allow_strawberry_fertilizer, cfg.strawberry_fertilizer_ages),
    ):
        if not enabled:
            continue
        candidates: list[tuple[int, int]] = []
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if quadrant_of(y, x) not in allowed_quadrants:
                    continue
                if (y, x) in removal_coords:
                    # The removal chain already owns this tile (and any
                    # preparatory WATER); do not fertilize or water a crop
                    # that is being released today.
                    continue
                if not (
                    isinstance(tile, Mapping)
                    and tile.get("kind") == "PLANT"
                    and tile.get("crop") == crop
                ):
                    continue
                if crop == "WHEAT" and wheat_harvest_eligibility(tile, day, step)[0]:
                    continue
                if int(tile.get("fertilized_until_day", -1) or -1) >= day:
                    continue
                planted = tile.get("planted_day")
                if planted is None or day - int(planted) not in ages:
                    continue
                if (
                    fertilizer_extra_units(
                        tile, day, wheat_harvest_threshold=cfg.wheat_harvest_threshold
                    )
                    <= 0
                ):
                    continue
                candidates.append((y, x))
        for y, x in candidates:
            fert_id = f"FERTILIZE:{crop}:{y},{x}"
            _supply_item(
                builder,
                fert_id,
                "FERTILIZE",
                tile=(y, x),
                crop=crop,
                requirements=(SupplyRequirement("FERTILIZER", 1, "inventory"),),
                source="fertilizer_policy",
            )
            ids = [fert_id]
            if board[y][x].get("watered_today") is not True:
                water_id = f"WATER:{y},{x}"
                builder.add(
                    id=water_id,
                    kind="WATER",
                    tile=(y, x),
                    crop=crop,
                    depends_on=(fert_id,),
                    source="fertilizer_linked_productive",
                )
                ids.append(water_id)
            builder.chain(
                f"FERTILIZER:{crop}:{y},{x}",
                "FERTILIZER_UPKEEP",
                ids,
                tile=(y, x),
                crop=crop,
            )

    # Routine watering is represented for standing crops only.  Fertilizer
    # watering was already inserted above; the stable item id makes this scan
    # merge-safe and keeps FERTILIZE before WATER.
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if quadrant_of(y, x) not in allowed_quadrants:
                continue
            if (y, x) in removal_coords:
                continue
            if not (
                isinstance(tile, Mapping)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") in _ROUTINE_WATER_AGES
            ):
                continue
            crop = str(tile["crop"])
            source = _routine_water_source(tile, crop, day, step)
            if source is None:
                continue
            water_id = f"WATER:{y},{x}"
            if water_id in builder.items:
                continue
            builder.add(
                id=water_id,
                kind="WATER",
                tile=(y, x),
                crop=crop,
                source=source,
            )
            builder.chain(
                f"ROUTINE_WATER:{crop}:{y},{x}",
                "ROUTINE_WATERING",
                (water_id,),
                tile=(y, x),
                crop=crop,
                source=source,
            )

    sell_quantities = plan.sell_quantities_dict
    sell_totals = {
        p: sum(
            int(sell_quantities[str(anchor)].get(p, 0))
            for anchor in (0, 4, 8, 12, 16, 20)
        )
        for p in PRODUCTS
    }
    for product in PRODUCTS:
        quantity = sell_totals[product]
        if not quantity:
            continue
        shed_amount, carried_amount = (
            builder.shed.get(product, 0),
            builder.carried.get(product, 0),
        )
        delivery_quantity = min(carried_amount, max(0, quantity - shed_amount))
        deps: tuple[str, ...] = ()
        if delivery_quantity:
            delivery_id = f"DELIVERY:{product}"
            _supply_item(
                builder,
                delivery_id,
                "DELIVERY",
                product=product,
                quantity=delivery_quantity,
                requirements=(
                    SupplyRequirement(product, delivery_quantity, "inventory"),
                ),
                source="sell_delivery",
            )
            # Deposit is represented separately from the underlying inventory.
            old = builder.items[delivery_id]
            builder.items[delivery_id] = WorkItem(
                old.id,
                old.kind,
                old.status,
                old.block_reason,
                old.tile,
                old.crop,
                old.animal,
                old.product,
                old.quantity,
                old.depends_on,
                old.required_supplies,
                old.interaction_turns,
                old.pickup_turns,
                1,
                old.known_dependency_turns,
                old.travel_turns,
                old.row_key,
                old.source,
            )
            deps = (delivery_id,)
        reason = (
            None
            if shed_amount + carried_amount >= quantity
            else BlockReason.MISSING_SUPPLY
        )
        sell_id = f"SELL:{product}"
        builder.add(
            id=sell_id,
            kind="SELL",
            product=product,
            quantity=quantity,
            depends_on=deps,
            block_reason=reason,
            source="daily_sell_intent",
        )
        builder.chain(
            f"SELL_CHAIN:{product}", "SELL_INTENT", (*deps, sell_id), product=product
        )

    builder.resolve_dependencies()
    items = tuple(sorted(builder.items.values(), key=lambda i: i.id))
    chains = tuple(sorted(builder.chains, key=lambda c: c.id))
    rows = _row_summaries(board, items, chains)
    diagnostics = _diagnostics(
        plan,
        current_crops,
        current_animals,
        current_land,
        crop_need,
        animal_need,
        layouts,
        represented_crops,
        represented_animals,
        items,
        supply,
        builder,
        rows,
    )
    return StripWorkPlan(items, chains, rows, supply, diagnostics, chosen_seat)


def _row_summaries(
    board, items: tuple[WorkItem, ...], chains: tuple[WorkChain, ...]
) -> tuple[RowSummary, ...]:
    rows: dict[RowKey, dict[str, Any]] = {}
    for y, row in enumerate(board):
        for x, _ in enumerate(row):
            key = row_key_for_tile((y, x))
            rows.setdefault(
                key,
                {
                    "tile_count": 0,
                    "ready": 0,
                    "future": 0,
                    "turns": 0,
                    "feed": 0,
                    "fert": 0,
                    "animal": 0,
                    "seed": 0,
                    "chains": 0,
                    "reasons": Counter(),
                },
            )["tile_count"] += 1
    for chain in chains:
        if chain.row_key is not None:
            rows[chain.row_key]["chains"] += 1
    for item in items:
        if item.row_key is None:
            continue
        data = rows[item.row_key]
        if item.status == WorkStatus.READY:
            data["ready"] += item.interaction_turns
        else:
            data["future"] += item.interaction_turns
            data["reasons"][
                item.block_reason.value
                if item.block_reason
                else WorkStatus.UNRESOLVED.value
            ] += 1
        data["turns"] += item.interaction_turns + item.pickup_turns + item.deposit_turns
        if item.kind == "FEED":
            data["feed"] += item.quantity
        if item.kind == "FERTILIZE":
            data["fert"] += item.quantity
        if item.kind == "PLACE":
            data["animal"] += item.quantity
        data["seed"] += sum(
            r.quantity for r in item.required_supplies if r.scope == "global_seed"
        )
    return tuple(
        RowSummary(
            key,
            data["tile_count"],
            data["ready"],
            data["future"],
            data["turns"],
            data["feed"],
            data["fert"],
            data["animal"],
            data["seed"],
            data["chains"],
            tuple(sorted(data["reasons"].items())),
        )
        for key, data in sorted(rows.items())
    )


def _diagnostics(
    plan,
    current_crops,
    current_animals,
    current_land,
    crop_need,
    animal_need,
    layouts,
    represented_crops,
    represented_animals,
    items,
    supply,
    builder,
    row_summaries,
):
    target_crops = plan.crop_targets_dict
    target_animals = plan.animal_targets_dict
    seeds, shed, carried = builder.seeds, builder.shed, builder.carried
    unresolved_crops, unresolved_animals = (
        Counter(), Counter(dict(layouts.animals.unresolved))
    )
    represented_crops, represented_animals = (
        Counter(represented_crops),
        Counter(represented_animals),
    )
    for crop in CROP_ORDER:
        residual = (
            plan.crop_targets_dict[crop]
            - current_crops[crop]
            - represented_crops[crop]
        )
        if residual:
            unresolved_crops[crop] = residual
    for animal, need in animal_need.items():
        unresolved_animals[animal] += max(
            0, need - represented_animals[animal] - unresolved_animals[animal]
        )
    counts_kind, counts_status = (
        Counter(i.kind for i in items),
        Counter(i.status.value for i in items),
    )
    counts_reason = Counter(i.block_reason.value for i in items if i.block_reason)
    demand_rows = []
    for (item, scope), amount in sorted(builder.demands.items()):
        available = (
            seeds.get(item, 0)
            if scope == "global_seed"
            else shed.get(item, 0) + carried.get(item, 0)
        )
        demand_rows.append(
            SupplyDemand(item, amount, available, max(0, amount - available), scope)
        )
    demand = tuple(demand_rows)
    workload = tuple(
        RowWorkload(
            r.row_key,
            r.ready_interactions + r.future_interactions,
            r.nontravel_turns,
            r.ready_interactions,
            r.future_interactions,
        )
        for r in row_summaries
        if r.ready_interactions or r.future_interactions
    )
    return WorkDiagnostics(
        _pairs({c: target_crops[c] - current_crops[c] for c in CROP_ORDER}),
        _pairs(represented_crops),
        _pairs(unresolved_crops),
        _pairs(
            {a: target_animals[a] - current_animals[a] for a in ANIMAL_ORDER}
        ),
        _pairs(represented_animals),
        _pairs(unresolved_animals),
        plan.land_count,
        current_land,
        max(0, plan.land_count - current_land),
        tuple(sorted(counts_kind.items())),
        tuple(sorted(counts_status.items())),
        tuple(sorted(counts_reason.items())),
        demand,
        workload,
    )
