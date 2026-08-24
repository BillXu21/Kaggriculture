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
- crop sacrifice scoring is intentionally crude and centralized in
  `SacrificeConfig`;
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
from typing import Any

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from replay_daily.constants import ANIMALS

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


# ------------------------------------------------------------- animal layout


@dataclass(frozen=True)
class AnimalSlotPlan:
    animal: str                     # GOOSE/COW/SHEEP
    structure: str                  # COOP/PASTURE
    coord: tuple[int, int]          # [y, x]
    source: str                     # empty_structure | new_build | crop_sacrifice


@dataclass(frozen=True)
class AnimalLayoutResult:
    placements: tuple[AnimalSlotPlan, ...]
    unresolved: tuple[tuple[str, int], ...]  # (animal, unmet count)


def _sorted_coords(coords: list[tuple[int, int]], anchor) \
        -> list[tuple[int, int]]:
    return sorted(coords, key=lambda c: (manhattan(c, anchor), c[0], c[1]))


def plan_animal_layout(
    board: list[list[Any]],
    *,
    unlocked_quadrants,
    animals_needed: Mapping[str, int],
    anchor: tuple[int, int],
    config: SacrificeConfig = SacrificeConfig(),
) -> AnimalLayoutResult:
    """Choose slots for positive animal deficits. Pure; board never mutated.

    Order per species (canonical ANIMAL_ORDER): reuse an empty matching
    structure first (mechanically free), then build on the nearest empty
    legal tile, then reclaim the nearest WEED tile (DIG prerequisite), then
    convert the cheapest crop tile. Locked tiles, occupied structures,
    wrong-type empty structures, and anything outside the unlocked quadrants
    are never selected. No tiles are reserved for hypothetical future animals.
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
    crop_tiles: list[tuple[float, tuple[int, int], str]] = []

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
                crop_tiles.append(
                    (sacrifice_score(tile, coord, anchor=anchor,
                                     config=config), coord))

    empty_tiles = _sorted_coords(empty_tiles, anchor)
    weed_tiles = _sorted_coords(weed_tiles, anchor)
    for kind in empty_structures:
        empty_structures[kind] = _sorted_coords(empty_structures[kind], anchor)
    crop_tiles.sort(key=lambda item: (item[0], item[1][0], item[1][1]))

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
        while need > 0 and crop_tiles:
            _, coord = crop_tiles.pop(0)
            placements.append(AnimalSlotPlan(animal, structure, coord,
                                             "crop_sacrifice"))
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
class CropReconciliationResult:
    digs: tuple[DigIntent, ...]
    plants: tuple[PlantIntent, ...]
    unresolved_deficits: tuple[tuple[str, int], ...]


def reconcile_crops(
    board: list[list[Any]],
    *,
    unlocked_quadrants,
    crop_targets: Mapping[str, int],
    anchor: tuple[int, int],
    config: SacrificeConfig = SacrificeConfig(),
) -> CropReconciliationResult:
    """Reconcile current crops toward requested target counts. Pure.

    Per canonical crop order: count matching PLANT tiles; when above target,
    retain the highest-sunk-investment matches and release only the true
    excess (cheapest first); when below target, fill from nearest empty
    legal tiles first, then reclaim nearest WEED tiles (DIG prerequisite),
    and only then from the cheapest released excess of other types. Tiles
    needed for their own target are never sacrificed. Sticky structures are
    never touched. Unmet deficits are reported honestly instead of
    over-planning.
    """
    for name in crop_targets:
        if name not in CROP_ORDER:
            raise ValueError(f"unknown crop {name!r}")
        value = crop_targets[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"crop_targets[{name!r}] must be a nonnegative "
                             f"integer, got {value!r}")

    scored: dict[str, list[tuple[float, tuple[int, int]]]] = {
        crop: [] for crop in CROP_ORDER}
    empty_tiles: list[tuple[int, int]] = []
    weed_tiles: list[tuple[int, int]] = []

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            coord = (y, x)
            if not _in_unlocked(coord, unlocked_quadrants):
                continue
            role = tile_role(tile)
            if role == "plant":
                crop = tile["crop"]
                if crop in scored:
                    scored[crop].append(
                        (sacrifice_score(tile, coord, anchor=anchor,
                                         config=config), coord))
            elif role == "empty":
                empty_tiles.append(coord)
            elif role == "weed":
                weed_tiles.append(coord)

    empty_tiles = _sorted_coords(empty_tiles, anchor)
    weed_tiles = _sorted_coords(weed_tiles, anchor)

    digs: list[DigIntent] = []
    plants: list[PlantIntent] = []
    empty_filled: dict[str, int] = {crop: 0 for crop in CROP_ORDER}
    weed_filled: dict[str, int] = {crop: 0 for crop in CROP_ORDER}
    released: list[tuple[float, tuple[int, int], str]] = []

    for crop in CROP_ORDER:
        target = int(crop_targets.get(crop, 0))
        entries = sorted(scored[crop],
                         key=lambda item: (-item[0], item[1][0], item[1][1]))
        # Retain the most invested matches; release only the true excess.
        for score_value, coord in entries[target:]:
            released.append((score_value, coord, crop))
        deficit = target - len(entries)
        while deficit > 0 and empty_tiles:
            coord = empty_tiles.pop(0)
            plants.append(PlantIntent(coord, crop))
            empty_filled[crop] += 1
            deficit -= 1
        while deficit > 0 and weed_tiles:
            # Reclaim a WEED tile: DIG then PLANT. Cheaper in sunk investment
            # than digging a living crop of another type (issue #7).
            coord = weed_tiles.pop(0)
            digs.append(DigIntent(coord, "WEED"))
            plants.append(PlantIntent(coord, crop))
            weed_filled[crop] += 1
            deficit -= 1

    released.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    unresolved: list[tuple[str, int]] = []
    for crop in CROP_ORDER:
        target = int(crop_targets.get(crop, 0))
        current = len(scored[crop])
        deficit = target - current - empty_filled[crop] - weed_filled[crop]
        while deficit > 0 and released:
            _, coord, old_crop = released.pop(0)
            digs.append(DigIntent(coord, old_crop))
            plants.append(PlantIntent(coord, crop))
            deficit -= 1
        if deficit > 0:
            unresolved.append((crop, deficit))

    return CropReconciliationResult(digs=tuple(digs), plants=tuple(plants),
                                    unresolved_deficits=tuple(unresolved))


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
        animals_needed=animals_needed, anchor=anchor, config=config)

    claimed = {slot.coord for slot in animal_result.placements}
    if not claimed:
        return DayLayoutResult(
            crops=reconcile_crops(
                board, unlocked_quadrants=unlocked_quadrants,
                crop_targets=crop_targets, anchor=anchor, config=config),
            animals=animal_result)

    masked = [row[:] for row in board]
    for y, x in claimed:
        masked[y][x] = _CLAIMED  # tile_role -> "other": ignored by reconcile
    crop_result = reconcile_crops(
        masked, unlocked_quadrants=unlocked_quadrants,
        crop_targets=crop_targets, anchor=anchor, config=config)
    return DayLayoutResult(crops=crop_result, animals=animal_result)
