"""V0 layout and crop reconciliation (issue #1 section 4).

Small deterministic pure functions over the canonical schema-v3 board
(`replay_daily.lifecycle.canonical_board` output: logical tiles indexed
``tiles[y][x]``, coordinates ``[y, x]``). No search, no facility-location,
no future-animal prediction, no product rankings.

Deliberate V0 simplifications preserved here (see EXECUTOR_V0_PLAN.md):

- no reserved near-shed zone for hypothetical future animals;
- existing livestock structures (occupied OR empty) are sticky: occupied
  structures are never touched, empty structures are only reused for a
  matching animal;
- optional live-crop sacrifice is explicitly configured and ordered by
  lifecycle value (age, yield, watering state, then route position);
- one-step decisions only; navigation/routing belongs to later stages.

All returned intents are deterministic: ties break on ``(y, x)`` after the
primary key (score or Manhattan distance to the explicit anchor).

Issue #7 additions: ``tile_role`` recognizes both observed WEED shapes;
``SHED_HUB_ANCHOR`` is the persistent central logistics hub used as the
default layout anchor (stable across turns/days -- never the moving farmer);
``plan_animal_layout`` / ``reconcile_crops`` may reclaim WEED tiles as a
last-resort slot pool (DIG prerequisite emitted upstream); and
``plan_day_layouts`` runs both planners over one shared set of tile claims so
crop and animal layouts can never reserve the same tile.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from replay_daily.constants import ANIMALS, CROPS
from replay_daily.lifecycle import clean_crop_removal_actions, plant_age_days

__all__ = [
    "SacrificeConfig",
    "sacrifice_score",
    "manhattan",
    "quadrant_of",
    "tile_role",
    "SHED_HUB_ANCHOR",
    "AnimalSlotPlan",
    "AnimalLayoutResult",
    "plan_animal_layout",
    "PlantIntent",
    "DigIntent",
    "CropRemovalIntent",
    "CropReconciliationResult",
    "reconcile_crops",
    "DayLayoutResult",
    "plan_day_layouts",
]

# Persistent central logistics hub: the shed sits at the board center and all
# PICKUP/DROP traffic passes its four access tiles. Layout anchors minimize
# ongoing service distance from here; unlike the farmer position this anchor
# never moves, so compiled layouts stay stable within and across days.
SHED_HUB_ANCHOR = (4, 4)


# ------------------------------------------------------------------ config


@dataclass(frozen=True)
class SacrificeConfig:
    """Intentionally crude sunk-investment weights. Lower score = cheaper.

    The absolute scale is meaningless; only relative order matters. Weights
    are configuration, not tuned mechanics.
    """

    distance_weight: float = 1.0        # per Manhattan step to the anchor
    age_weight: float = 1.0             # per derived age_days
    yield_units_weight: float = 2.0     # per accumulated yield unit
    fertilizer_active_penalty: float = 10.0  # fertilizer currently active
    harvestable_penalty: float = 25.0        # currently harvestable
    null_timing_penalty: float = 5.0         # nullable timing is null
    missing_derived_penalty: float = 15.0    # derived struct entirely absent
    allow_live_crop_sacrifice: bool = False
    allow_productive_recurring_crop_sacrifice: bool = False
    allow_older_crop_sacrifice: bool = False


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quadrant_of(y: int, x: int) -> str:
    """Official quadrant layout: NW/NE top half, SW/SE bottom half."""
    if y < 5:
        return "NW" if x < 5 else "NE"
    return "SW" if x < 5 else "SE"


def tile_role(tile: Any) -> str:
    """Conservative classification of one canonical tile value.

    One of ``empty``, ``locked`` (incl. unrecognized bare strings),
    ``weed``, ``plant``, ``animal_structure`` (occupied COOP/PASTURE),
    ``empty_structure`` (COOP/PASTURE without an animal), ``other``
    (unknown dict shapes; treated as unusable).
    """
    if tile is None:
        return "empty"
    if isinstance(tile, str):
        if tile == "WEED":
            return "weed"
        return "locked"  # "LOCKED" and any unrecognized sentinel
    if isinstance(tile, Mapping):
        if tile.get("kind") == "WEED":
            # fast-engine decoder shape; official replays use the bare string
            return "weed"
        if "animal" in tile:
            return "animal_structure"
        kind = tile.get("kind")
        if kind in ("COOP", "PASTURE"):
            return "empty_structure"
        if kind == "PLANT":
            return "plant"
    return "other"


def _in_unlocked(coord: tuple[int, int], unlocked_quadrants) -> bool:
    return quadrant_of(*coord) in set(unlocked_quadrants)


def sacrifice_score(
    tile: Any,
    coord: tuple[int, int],
    *,
    anchor: tuple[int, int],
    config: SacrificeConfig = SacrificeConfig(),
) -> float:
    """Crude sunk-investment score for destroying one PLANT tile.

    Uses canonical raw/derived fields only. Missing or null lifecycle
    information is penalized (conservative: do not preferentially destroy
    plants we cannot assess). Never invented mechanics.
    """
    if tile_role(tile) != "plant":
        raise ValueError(
            f"sacrifice_score expects a PLANT tile at {coord}, got "
            f"{tile!r:.60}")
    score = config.distance_weight * manhattan(coord, anchor)
    derived = tile.get("derived")
    if isinstance(derived, Mapping):
        score += config.age_weight * float(derived.get("age_days") or 0)
        if derived.get("fertilizer_active"):
            score += config.fertilizer_active_penalty
        if derived.get("currently_harvestable"):
            score += config.harvestable_penalty
        if derived.get("days_until_next_harvest") is None:
            score += config.null_timing_penalty
    else:
        score += config.missing_derived_penalty
    score += config.yield_units_weight * float(tile.get("yield_units") or 0)
    return score


def _sacrifice_order_key(
    tile: Mapping[str, Any],
    coord: tuple[int, int],
    current_day: int,
    anchor: tuple[int, int],
) -> tuple[Any, ...]:
    age = plant_age_days(tile, current_day)
    return (
        int(age) if age is not None else 10**9,
        int(tile.get("yield_units", 0) or 0),
        int(tile.get("watered_today") is True),
        manhattan(coord, anchor),
        coord[0],
        coord[1],
    )


def _sacrifice_candidate(
    tile: Mapping[str, Any],
    coord: tuple[int, int],
    *,
    current_day: int,
    config: SacrificeConfig,
) -> bool:
    """Whether a plant may enter the explicitly enabled sacrifice escape hatch."""
    if not config.allow_live_crop_sacrifice or tile_role(tile) != "plant":
        return False
    derived = tile.get("derived")
    raw_fertilized_until = tile.get("fertilized_until_day", -1)
    raw_fertilizer_active = (
        raw_fertilized_until is not None
        and int(raw_fertilized_until) >= current_day
    )
    if (
        isinstance(derived, Mapping) and derived.get("fertilizer_active")
    ) or raw_fertilizer_active:
        return False
    age = plant_age_days(tile, current_day)
    if age is None or age < 0:
        return False
    if not config.allow_older_crop_sacrifice and age > 1:
        return False
    crop = tile.get("crop")
    data = CROPS.get(crop)
    if (
        data is not None
        and data["ongoing"]
        and not config.allow_productive_recurring_crop_sacrifice
    ):
        return False
    return True


# ------------------------------------------------------------- animal layout


@dataclass(frozen=True)
class AnimalSlotPlan:
    animal: str                     # GOOSE/COW/SHEEP
    structure: str                  # COOP/PASTURE
    coord: tuple[int, int]          # [y, x]
    source: str                     # empty_structure | new_build | crop_release |
                                   # crop_sacrifice | weed_reclaim
    removal_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnimalLayoutResult:
    placements: tuple[AnimalSlotPlan, ...]
    unresolved: tuple[tuple[str, int], ...]  # (animal, unmet count)


def _sorted_coords(coords: list[tuple[int, int]], anchor) \
        -> list[tuple[int, int]]:
    return sorted(coords, key=lambda c: (manhattan(c, anchor), c[0], c[1]))


PhysicalCropRowKey = tuple[int, int]  # (global y, quadrant x start)


def _physical_crop_row_key(coord: tuple[int, int]) -> PhysicalCropRowKey:
    """Return the horizontal five-tile strip containing ``coord``."""
    y, x = coord
    return y, 0 if x < 5 else 5


def _physical_crop_row_distance(
    row_key: PhysicalCropRowKey,
    anchor: tuple[int, int],
) -> int:
    """Minimum Manhattan distance from ``anchor`` to a tile in the row."""
    y, x_start = row_key
    nearest_x = min(max(anchor[1], x_start), x_start + 4)
    return abs(anchor[0] - y) + abs(anchor[1] - nearest_x)


def _select_row_compact_coord(
    candidates: Iterable[tuple[int, int]],
    *,
    crop_occupancy: Mapping[PhysicalCropRowKey, int],
    anchor: tuple[int, int],
) -> tuple[int, int]:
    """Select one candidate while packing crop work into physical rows."""
    candidates_by_row: dict[PhysicalCropRowKey, list[tuple[int, int]]] = {}
    for coord in candidates:
        candidates_by_row.setdefault(_physical_crop_row_key(coord), []).append(coord)
    if not candidates_by_row:
        raise ValueError("row-compact selection requires at least one candidate")

    row_key = min(
        candidates_by_row,
        key=lambda key: (
            0 if crop_occupancy.get(key, 0) > 0 else 1,
            -crop_occupancy.get(key, 0),
            _physical_crop_row_distance(key, anchor),
            key,
        ),
    )
    return min(
        candidates_by_row[row_key],
        key=lambda coord: (manhattan(coord, anchor), coord[0], coord[1]),
    )


def plan_animal_layout(
    board: list[list[Any]],
    *,
    unlocked_quadrants,
    animals_needed: Mapping[str, int],
    anchor: tuple[int, int],
    config: SacrificeConfig = SacrificeConfig(),
    current_day: int = 0,
    current_step: int | None = None,
) -> AnimalLayoutResult:
    """Choose slots for positive animal deficits. Pure; board never mutated.

    Order per species (canonical ANIMAL_ORDER): reuse an empty matching
    structure first, then use an empty tile, reclaim a WEED, release a cleanly
    removable crop, and finally use the explicitly enabled crop-sacrifice
    escape hatch. Locked tiles, occupied structures, wrong-type empty
    structures, active-fertilizer crops, and anything outside the unlocked
    quadrants are never selected. No tiles are reserved for hypothetical
    future animals.
    """
    for name in animals_needed:
        if name not in ANIMAL_ORDER:
            raise ValueError(f"unknown animal {name!r}")
        value = animals_needed[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"animals_needed[{name!r}] must be a "
                             f"nonnegative integer, got {value!r}")

    empty_structures: dict[str, list[tuple[int, int]]] = {
        "COOP": [], "PASTURE": []}
    empty_tiles: list[tuple[int, int]] = []
    weed_tiles: list[tuple[int, int]] = []
    clean_crop_tiles: list[tuple[tuple[int, int], tuple[int, int], str, tuple[str, ...]]] = []
    sacrifice_crop_tiles: list[tuple[tuple[Any, ...], tuple[int, int], str]] = []
    step = current_day * 24 if current_step is None else int(current_step)

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            coord = (y, x)
            if not _in_unlocked(coord, unlocked_quadrants):
                continue
            role = tile_role(tile)
            if role == "empty":
                empty_tiles.append(coord)
            elif role == "weed":
                weed_tiles.append(coord)
            elif role == "empty_structure":
                empty_structures[tile["kind"]].append(coord)
            elif role == "plant":
                clean_actions = clean_crop_removal_actions(
                    tile, current_day, step, for_replacement=False
                )
                if clean_actions:
                    clean_crop_tiles.append(
                        ((manhattan(coord, anchor), coord[0], coord[1]),
                         coord, str(tile["crop"]), clean_actions)
                    )
                elif _sacrifice_candidate(
                    tile, coord, current_day=current_day, config=config
                ):
                    sacrifice_crop_tiles.append(
                        (_sacrifice_order_key(tile, coord, current_day, anchor),
                         coord, str(tile["crop"]))
                    )
    clean_crop_tiles.sort(key=lambda item: item[0])
    sacrifice_crop_tiles.sort(key=lambda item: item[0])

    empty_tiles = _sorted_coords(empty_tiles, anchor)
    weed_tiles = _sorted_coords(weed_tiles, anchor)
    for kind in empty_structures:
        empty_structures[kind] = _sorted_coords(empty_structures[kind], anchor)
    placements: list[AnimalSlotPlan] = []
    unresolved: list[tuple[str, int]] = []
    for animal in ANIMAL_ORDER:
        need = int(animals_needed.get(animal, 0))
        structure = ANIMALS[animal]["structure"]
        while need > 0 and empty_structures[structure]:
            coord = empty_structures[structure].pop(0)
            placements.append(AnimalSlotPlan(animal, structure, coord,
                                             "empty_structure"))
            need -= 1
        while need > 0 and empty_tiles:
            coord = empty_tiles.pop(0)
            placements.append(AnimalSlotPlan(animal, structure, coord,
                                             "new_build"))
            need -= 1
        while need > 0 and weed_tiles:
            # Weeds block BUILD exactly like occupied tiles; reclaiming one
            # costs a DIG turn but destroys no sunk investment, so weeds are
            # strictly preferred over sacrificing crops (issue #7).
            coord = weed_tiles.pop(0)
            placements.append(AnimalSlotPlan(animal, structure, coord,
                                             "weed_reclaim"))
            need -= 1
        while need > 0 and clean_crop_tiles:
            _, coord, crop, actions = clean_crop_tiles.pop(0)
            placements.append(AnimalSlotPlan(
                animal, structure, coord, "crop_release", actions
            ))
            need -= 1
        while need > 0 and sacrifice_crop_tiles:
            _, coord, crop = sacrifice_crop_tiles.pop(0)
            placements.append(AnimalSlotPlan(
                animal, structure, coord, "crop_sacrifice", ("DIG",)
            ))
            need -= 1
        if need > 0:
            unresolved.append((animal, need))
    return AnimalLayoutResult(placements=tuple(placements),
                              unresolved=tuple(unresolved))


# -------------------------------------------------------- crop reconciliation


@dataclass(frozen=True)
class PlantIntent:
    coord: tuple[int, int]
    crop: str


@dataclass(frozen=True)
class DigIntent:
    coord: tuple[int, int]
    crop: str  # crop being removed


@dataclass(frozen=True)
class CropRemovalIntent:
    coord: tuple[int, int]
    crop: str
    actions: tuple[str, ...]
    sacrifice: bool = False


@dataclass(frozen=True)
class CropReconciliationResult:
    digs: tuple[DigIntent, ...]
    plants: tuple[PlantIntent, ...]
    unresolved_deficits: tuple[tuple[str, int], ...]
    removals: tuple[CropRemovalIntent, ...] = ()
    unresolved_reductions: tuple[tuple[str, int], ...] = ()


def reconcile_crops(
    board: list[list[Any]],
    *,
    unlocked_quadrants,
    crop_targets: Mapping[str, int],
    anchor: tuple[int, int],
    config: SacrificeConfig = SacrificeConfig(),
    preferred_crop_slots: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    current_day: int = 0,
    current_step: int | None = None,
) -> CropReconciliationResult:
    """Reconcile current crops toward requested target counts. Pure.

    Empty tiles and WEEDs are consumed before any crop release. A requested
    reduction first selects clean lifecycle removals from the authoritative
    timing helper; only the explicitly enabled sacrifice policy can add a
    premature live-crop DIG. A clean release may satisfy another crop deficit,
    while an otherwise unrepresentable contraction is returned as a signed
    ``unresolved_reductions`` entry rather than fabricated work.
    """
    for name in crop_targets:
        if name not in CROP_ORDER:
            raise ValueError(f"unknown crop {name!r}")
        value = crop_targets[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"crop_targets[{name!r}] must be a nonnegative "
                             f"integer, got {value!r}")

    step = current_day * 24 if current_step is None else int(current_step)

    scored: dict[str, list[tuple[float, tuple[int, int]]]] = {
        crop: [] for crop in CROP_ORDER}
    tiles_by_coord: dict[tuple[int, int], Mapping[str, Any]] = {}
    empty_tiles: list[tuple[int, int]] = []
    weed_tiles: list[tuple[int, int]] = []
    crop_row_occupancy: dict[PhysicalCropRowKey, int] = {}

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            coord = (y, x)
            if not _in_unlocked(coord, unlocked_quadrants):
                continue
            role = tile_role(tile)
            if role == "plant":
                crop = tile["crop"]
                if crop in scored:
                    tiles_by_coord[coord] = tile
                    scored[crop].append(
                        (sacrifice_score(tile, coord, anchor=anchor,
                                         config=config), coord))
                    row_key = _physical_crop_row_key(coord)
                    crop_row_occupancy[row_key] = (
                        crop_row_occupancy.get(row_key, 0) + 1
                    )
            elif role == "empty":
                empty_tiles.append(coord)
            elif role == "weed":
                weed_tiles.append(coord)

    # A retained one-shot harvest may ask reconciliation to preserve its
    # freshly emptied tile.  Reserve only as many authoritative empty slots as
    # the current manager target still needs; stale/reduced targets therefore
    # cannot create an independent crop obligation.
    preferred = preferred_crop_slots or {}
    reserved_empty: dict[str, list[tuple[int, int]]] = {
        crop: [] for crop in CROP_ORDER
    }
    available_empty = set(empty_tiles)
    for crop in CROP_ORDER:
        deficit = max(0, int(crop_targets.get(crop, 0)) - len(scored[crop]))
        for raw_coord in preferred.get(crop, ()):
            coord = (int(raw_coord[0]), int(raw_coord[1]))
            if deficit <= 0:
                break
            if coord not in available_empty:
                continue
            reserved_empty[crop].append(coord)
            available_empty.remove(coord)
            deficit -= 1
    empty_tiles = [coord for coord in empty_tiles if coord in available_empty]

    digs: list[DigIntent] = []
    removals: list[CropRemovalIntent] = []
    plants: list[PlantIntent] = []
    empty_filled: dict[str, int] = {crop: 0 for crop in CROP_ORDER}
    weed_filled: dict[str, int] = {crop: 0 for crop in CROP_ORDER}
    selected_releases: list[tuple[
        tuple[Any, ...], tuple[int, int], str, tuple[str, ...],
        tuple[str, ...], bool
    ]] = []
    maintenance_replacements: list[tuple[tuple[int, int], str, tuple[str, ...]]] = []
    unresolved_reductions: list[tuple[str, int]] = []

    for crop in CROP_ORDER:
        target = int(crop_targets.get(crop, 0))
        entries = sorted(scored[crop],
                         key=lambda item: (-item[0], item[1][0], item[1][1]))
        deficit = target - len(entries)
        while deficit > 0 and reserved_empty[crop]:
            coord = reserved_empty[crop].pop(0)
            plants.append(PlantIntent(coord, crop))
            row_key = _physical_crop_row_key(coord)
            crop_row_occupancy[row_key] = (
                crop_row_occupancy.get(row_key, 0) + 1
            )
            empty_filled[crop] += 1
            deficit -= 1
        while deficit > 0 and empty_tiles:
            coord = _select_row_compact_coord(
                empty_tiles, crop_occupancy=crop_row_occupancy,
                anchor=anchor)
            empty_tiles.remove(coord)
            plants.append(PlantIntent(coord, crop))
            row_key = _physical_crop_row_key(coord)
            crop_row_occupancy[row_key] = (
                crop_row_occupancy.get(row_key, 0) + 1
            )
            empty_filled[crop] += 1
            deficit -= 1
        while deficit > 0 and weed_tiles:
            # Reclaim a WEED tile: DIG then PLANT. Cheaper in sunk investment
            # than digging a living crop of another type (issue #7).
            coord = _select_row_compact_coord(
                weed_tiles, crop_occupancy=crop_row_occupancy,
                anchor=anchor)
            weed_tiles.remove(coord)
            digs.append(DigIntent(coord, "WEED"))
            plants.append(PlantIntent(coord, crop))
            row_key = _physical_crop_row_key(coord)
            crop_row_occupancy[row_key] = (
                crop_row_occupancy.get(row_key, 0) + 1
            )
            weed_filled[crop] += 1
            deficit -= 1

        excess = max(0, len(entries) - target)
        clean_candidates: list[tuple[
            tuple[Any, ...], tuple[int, int], str, tuple[str, ...], tuple[str, ...], bool
        ]] = []
        sacrifice_candidates: list[tuple[
            tuple[Any, ...], tuple[int, int], str, tuple[str, ...], tuple[str, ...], bool
        ]] = []
        for _, coord in entries:
            tile = tiles_by_coord[coord]
            contraction_actions = clean_crop_removal_actions(
                tile, current_day, step, for_replacement=False
            )
            replacement_actions = clean_crop_removal_actions(
                tile, current_day, step, for_replacement=True
            )
            if contraction_actions:
                clean_candidates.append((
                    (0, 0 if replacement_actions else 1,
                     manhattan(coord, anchor), coord[0], coord[1]),
                    coord, crop, replacement_actions, contraction_actions, False
                ))
            elif _sacrifice_candidate(
                tile, coord, current_day=current_day, config=config
            ):
                sacrifice_candidates.append((
                    (1, *_sacrifice_order_key(tile, coord, current_day, anchor)),
                    coord, crop, ("DIG",), ("DIG",), True
                ))
        clean_candidates.sort(key=lambda item: item[0])
        sacrifice_candidates.sort(key=lambda item: item[0])
        chosen = (clean_candidates + sacrifice_candidates)[:excess]
        selected_releases.extend(chosen)
        if len(chosen) < excess:
            unresolved_reductions.append((crop, excess - len(chosen)))

    # A spent recurring crop is retired and replanted in place when its
    # requested count is retained.  One-shot crops are re-planted only after
    # their observed HARVEST releases the tile on a later regeneration.
    selected_coords = {item[1] for item in selected_releases}
    for crop in CROP_ORDER:
        if len(scored[crop]) > int(crop_targets.get(crop, 0)):
            continue
        for _, coord in scored[crop]:
            if coord in selected_coords:
                continue
            tile = tiles_by_coord[coord]
            actions = clean_crop_removal_actions(
                tile, current_day, step, for_replacement=True
            )
            if actions and CROPS[crop]["ongoing"]:
                maintenance_replacements.append((coord, crop, actions))
                selected_coords.add(coord)

    selected_releases.sort(key=lambda item: item[0])
    replacement_releases = [
        item for item in selected_releases if item[3]
    ]
    contraction_releases = [
        item for item in selected_releases if not item[3]
    ]
    unresolved: list[tuple[str, int]] = []
    for crop in CROP_ORDER:
        target = int(crop_targets.get(crop, 0))
        current = len(scored[crop])
        deficit = target - current - empty_filled[crop] - weed_filled[crop]
        while deficit > 0 and replacement_releases:
            _, coord, old_crop, replacement_actions, contraction_actions, sacrifice = replacement_releases.pop(0)
            removals.append(CropRemovalIntent(
                coord, old_crop, replacement_actions, sacrifice
            ))
            plants.append(PlantIntent(coord, crop))
            deficit -= 1
        if deficit > 0:
            unresolved.append((crop, deficit))

    for _, coord, old_crop, _, contraction_actions, sacrifice in (
        contraction_releases + replacement_releases
    ):
        removals.append(CropRemovalIntent(
            coord, old_crop, contraction_actions, sacrifice
        ))

    for coord, crop, actions in maintenance_replacements:
        removals.append(CropRemovalIntent(coord, crop, actions, False))
        plants.append(PlantIntent(coord, crop))

    for removal in removals:
        if "DIG" in removal.actions:
            digs.append(DigIntent(removal.coord, removal.crop))

    return CropReconciliationResult(digs=tuple(digs), plants=tuple(plants),
                                    unresolved_deficits=tuple(unresolved),
                                    removals=tuple(removals),
                                    unresolved_reductions=tuple(unresolved_reductions))


# ------------------------------------------------------- coordinated layouts

_CLAIMED = object()  # internal sentinel: tile already claimed by the other planner


@dataclass(frozen=True)
class DayLayoutResult:
    """Both planners' results over one shared set of tile claims."""

    crops: CropReconciliationResult
    animals: AnimalLayoutResult


def plan_day_layouts(
    board: list[list[Any]],
    *,
    unlocked_quadrants,
    crop_targets: Mapping[str, int],
    animals_needed: Mapping[str, int],
    anchor: tuple[int, int] = SHED_HUB_ANCHOR,
    config: SacrificeConfig = SacrificeConfig(),
    preferred_crop_slots: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    current_day: int = 0,
    current_step: int | None = None,
) -> DayLayoutResult:
    """Plan animal and crop layouts once over a shared set of tile claims.

    Running ``plan_animal_layout`` and ``reconcile_crops`` independently lets
    both claim the same empty tile; the resulting PLANT/BUILD task collision
    wastes labor and scatters the layout (issue #7). Animals plan first
    (structures are sticky and their slots are scarcer), every tile they
    claim is masked out, and crops reconcile over the remainder. The default
    anchor is the persistent shed hub, so compiled targets do not churn as
    workers move during the day.
    """
    animal_result = plan_animal_layout(
        board, unlocked_quadrants=unlocked_quadrants,
        animals_needed=animals_needed, anchor=anchor, config=config,
        current_day=current_day, current_step=current_step)

    claimed = {slot.coord for slot in animal_result.placements}
    if not claimed:
        return DayLayoutResult(
            crops=reconcile_crops(
                board, unlocked_quadrants=unlocked_quadrants,
                crop_targets=crop_targets, anchor=anchor, config=config,
                preferred_crop_slots=preferred_crop_slots,
                current_day=current_day, current_step=current_step),
            animals=animal_result)

    masked = [row[:] for row in board]
    for y, x in claimed:
        masked[y][x] = _CLAIMED  # tile_role -> "other": ignored by reconcile
    crop_result = reconcile_crops(
        masked, unlocked_quadrants=unlocked_quadrants,
        crop_targets=crop_targets, anchor=anchor, config=config,
        preferred_crop_slots=preferred_crop_slots,
        current_day=current_day, current_step=current_step)
    return DayLayoutResult(crops=crop_result, animals=animal_result)
