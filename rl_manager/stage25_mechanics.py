"""Framework-free Stage 2.5 action and physical-support contract.

This module deliberately depends only on the Python standard library.  It is
shared by later rollout and JAX packets, so sampled class indices, signed crop
deltas, and decoded persistent goals remain separate at every public boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


ACTION_SCHEMA_VERSION = "stage25_physical_v1"

LAND_ACTION = "land"
ANIMAL_ORDER = ("GOOSE", "COW", "SHEEP")
CROP_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ACTION_ORDER = (LAND_ACTION,) + tuple(name.lower() for name in ANIMAL_ORDER + CROP_ORDER)
ACTION_CLASS_COUNTS = (4, 101, 101, 101, 201, 201, 201, 201, 201)

CROP_HOLD_CLASS = 100
CROP_DELTA_MIN = -100
CROP_DELTA_MAX = 100
CROP_GOAL_MIN = 0
CROP_GOAL_MAX = 100

__all__ = [
    "ACTION_SCHEMA_VERSION", "LAND_ACTION", "ANIMAL_ORDER", "CROP_ORDER",
    "ACTION_ORDER", "ACTION_CLASS_COUNTS", "CROP_HOLD_CLASS",
    "PhysicalContext", "land_class_to_target", "land_target_to_class",
    "animal_class_to_target", "animal_target_to_class",
    "crop_class_to_delta", "crop_delta_to_class", "initialize_crop_ledger",
    "transition_crop_goal", "transition_crop_ledger",
    "transition_crop_ledger_classes", "physical_context_from_board",
    "required_new_housing_cells", "physical_crop_capacity",
    "animal_prefix_is_feasible", "animal_target_support_mask",
    "animal_acquisition_deficits", "unplaced_animal_counts",
    "crop_delta_support_mask",
    "decode_supported_crop_goals", "land_target_support_mask",
]


def _require_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return value


def _require_range(value: object, what: str, low: int, high: int) -> int:
    parsed = _require_int(value, what)
    if not low <= parsed <= high:
        raise ValueError(f"{what} must be in [{low}, {high}], got {parsed}")
    return parsed


def land_class_to_target(class_index: int) -> int:
    """Decode a land class index to an absolute unlocked-land target."""
    return _require_range(class_index, "land class index", 0, 3) + 1


def land_target_to_class(target: int) -> int:
    """Encode an absolute unlocked-land target as a land class index."""
    return _require_range(target, "land target", 1, 4) - 1


def land_target_support_mask(observed_land: int) -> tuple[bool, ...]:
    """Return the four land classes that do not shrink observed land."""
    observed = _require_range(observed_land, "observed land", 1, 4)
    return tuple(target >= observed for target in range(1, 5))


def animal_class_to_target(class_index: int) -> int:
    """Decode an animal class index to an absolute placed-animal target."""
    return _require_range(class_index, "animal class index", 0, 100)


def animal_target_to_class(target: int) -> int:
    """Encode an absolute placed-animal target as an animal class index."""
    return _require_range(target, "animal target", 0, 100)


def crop_class_to_delta(class_index: int) -> int:
    """Decode a crop class index to its signed persistent-goal delta."""
    return _require_range(class_index, "crop class index", 0, 200) - 100


def crop_delta_to_class(delta: int) -> int:
    """Encode a signed persistent-goal delta as a crop class index."""
    return _require_range(delta, "crop delta", CROP_DELTA_MIN,
                          CROP_DELTA_MAX) + 100


def initialize_crop_ledger(observed_crop_counts: Sequence[int]) \
        -> tuple[int, ...]:
    """Initialize one seat's persistent goals from first-boundary occupancy."""
    return _crop_goal_vector(observed_crop_counts, "observed crop count")


def transition_crop_goal(previous_goal: int, sampled_delta: int) -> int:
    """Apply one sampled delta exactly once, rejecting rather than repairing."""
    previous = _require_range(previous_goal, "previous crop goal",
                              CROP_GOAL_MIN, CROP_GOAL_MAX)
    delta = _require_range(sampled_delta, "sampled crop delta",
                           CROP_DELTA_MIN, CROP_DELTA_MAX)
    updated = previous + delta
    if not CROP_GOAL_MIN <= updated <= CROP_GOAL_MAX:
        raise ValueError(
            f"sampled crop delta {delta} makes goal {updated} outside "
            f"[{CROP_GOAL_MIN}, {CROP_GOAL_MAX}]")
    return updated


def transition_crop_ledger(
    previous_goals: Sequence[int],
    sampled_deltas: Sequence[int],
) -> tuple[int, ...]:
    """Apply five sampled deltas exactly once to a persistent crop ledger."""
    previous = _crop_goal_vector(previous_goals, "previous crop goal")
    deltas = tuple(sampled_deltas)
    if len(deltas) != len(CROP_ORDER):
        raise ValueError(
            f"sampled crop deltas must contain {len(CROP_ORDER)} values, "
            f"got {len(deltas)}")
    return tuple(
        transition_crop_goal(goal, delta)
        for goal, delta in zip(previous, deltas)
    )


def transition_crop_ledger_classes(
    previous_goals: Sequence[int],
    sampled_class_indices: Sequence[int],
) -> tuple[int, ...]:
    """Decode crop classes, then apply their deltas without clipping."""
    classes = tuple(sampled_class_indices)
    if len(classes) != len(CROP_ORDER):
        raise ValueError(
            f"sampled crop classes must contain {len(CROP_ORDER)} values, "
            f"got {len(classes)}")
    return transition_crop_ledger(
        previous_goals, tuple(crop_class_to_delta(value) for value in classes))


def _crop_goal_vector(values: Iterable[int], what: str) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != len(CROP_ORDER):
        raise ValueError(
            f"{what}s must contain {len(CROP_ORDER)} values, got {len(result)}")
    return tuple(
        _require_range(value, f"{what}[{CROP_ORDER[index]}]",
                       CROP_GOAL_MIN, CROP_GOAL_MAX)
        for index, value in enumerate(result)
    )


# ---------------------------------------------------------------------------
# Physical context and support

QUADRANT_ORDER = ("NW", "NE", "SW", "SE")
LAND_TARGET_MIN = 1
LAND_TARGET_MAX = len(QUADRANT_ORDER)
ANIMAL_TARGET_MIN = 0
ANIMAL_TARGET_MAX = 100


def _animal_vector(values: Sequence[int], what: str) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != len(ANIMAL_ORDER):
        raise ValueError(
            f"{what} must contain {len(ANIMAL_ORDER)} values, got "
            f"{len(result)}")
    return tuple(
        _require_range(value, f"{what}[{ANIMAL_ORDER[index]}]",
                       ANIMAL_TARGET_MIN, ANIMAL_TARGET_MAX)
        for index, value in enumerate(result)
    )


@dataclass(frozen=True)
class PhysicalContext:
    """Observed physical state needed by the Stage 2.5 support masks.

    ``crop_build_cells_by_land[target - 1]`` is the number of cells that are
    crop/build-compatible under that hypothetical land footprint. It includes
    empty cells, weeds and planted crops, and excludes occupied or empty
    sticky structures, locked cells outside the footprint, and unknown tile
    shapes. The tuple is intentionally supplied as physical context rather
    than recomputed from policy/economic state.
    """

    observed_land: int
    crop_build_cells_by_land: tuple[int, ...]
    placed_animals: tuple[int, ...]
    reusable_empty_coops: int = 0
    reusable_empty_pastures: int = 0
    unplaced_animals: tuple[int, ...] = (0, 0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_land",
            _require_range(self.observed_land, "observed land",
                           LAND_TARGET_MIN, LAND_TARGET_MAX))
        cells = tuple(self.crop_build_cells_by_land)
        if len(cells) != LAND_TARGET_MAX:
            raise ValueError(
                f"crop_build_cells_by_land must contain {LAND_TARGET_MAX} "
                f"values, got {len(cells)}")
        if any(_require_int(value, "crop/build-compatible cell count") < 0
               for value in cells):
            raise ValueError("crop/build-compatible cell counts must be nonnegative")
        object.__setattr__(self, "crop_build_cells_by_land", cells)
        object.__setattr__(
            self, "placed_animals", _animal_vector(
                self.placed_animals, "placed animal counts"))
        object.__setattr__(
            self, "unplaced_animals", _animal_vector(
                self.unplaced_animals, "unplaced animal counts"))
        for name in ("reusable_empty_coops", "reusable_empty_pastures"):
            value = _require_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

    def crop_build_cells(self, land_target: int) -> int:
        """Return ``B`` for one absolute hypothetical land target."""
        target = _require_range(land_target, "land target",
                                LAND_TARGET_MIN, LAND_TARGET_MAX)
        if target < self.observed_land:
            raise ValueError(
                f"land target {target} is below observed land "
                f"{self.observed_land}")
        return self.crop_build_cells_by_land[target - 1]


def _quadrant_of(y: int, x: int) -> str:
    if not 0 <= y < 10 or not 0 <= x < 10:
        raise ValueError(f"board coordinate out of range: {(y, x)!r}")
    return ("NW" if x < 5 else "NE") if y < 5 else ("SW" if x < 5 else "SE")


def _tile_physical_role(tile: object) -> str:
    """Classify only shapes relevant to physical capacity."""
    if tile is None:
        return "compatible"
    if tile == "WEED":
        return "compatible"
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind in ("PLANT", "WEED"):
            return "compatible"
        if kind in ("COOP", "PASTURE"):
            return "empty_structure" if "animal" not in tile else "occupied_structure"
    return "unusable"


def physical_context_from_board(
    board: Sequence[Sequence[object]],
    unlocked_quadrants: Iterable[str],
    *,
    unplaced_animals: Sequence[int] = (0, 0, 0),
) -> PhysicalContext:
    """Derive physical context from the canonical 10x10 board.

    The engine exposes locked quadrants as locked sentinels. When a requested
    land target includes such a quadrant, those cells become part of the
    hypothetical footprint and are treated as newly available empty land;
    this is the physical-land premise required by the support contract.
    """
    rows = tuple(tuple(row) for row in board)
    if len(rows) != 10 or any(len(row) != 10 for row in rows):
        raise ValueError("board must be a 10x10 sequence")
    unlocked = tuple(unlocked_quadrants)
    if len(set(unlocked)) != len(unlocked) or any(
            quadrant not in QUADRANT_ORDER for quadrant in unlocked):
        raise ValueError(f"unlocked_quadrants must use {QUADRANT_ORDER}")
    if set(unlocked) != set(QUADRANT_ORDER[:len(unlocked)]):
        raise ValueError("unlocked_quadrants must be the canonical land prefix")

    role_by_coord = {
        (y, x): _tile_physical_role(rows[y][x])
        for y in range(10) for x in range(10)
    }
    cells_by_land: list[int] = []
    for target in range(1, LAND_TARGET_MAX + 1):
        footprint = set(QUADRANT_ORDER[:target])
        count = 0
        for (y, x), role in role_by_coord.items():
            quadrant = _quadrant_of(y, x)
            if quadrant not in footprint:
                continue
            # Locked cells in a newly unlocked quadrant are the engine's
            # representation of cells that become available with the land.
            if quadrant not in unlocked and role == "unusable" \
                    and rows[y][x] == "LOCKED":
                count += 1
            elif quadrant in unlocked and role == "compatible":
                count += 1
            elif quadrant not in unlocked and rows[y][x] == "LOCKED":
                count += 1
        cells_by_land.append(count)

    placed = [0] * len(ANIMAL_ORDER)
    empty_coops = empty_pastures = 0
    for (y, x), role in role_by_coord.items():
        if _quadrant_of(y, x) not in unlocked:
            continue
        tile = rows[y][x]
        if not isinstance(tile, dict):
            continue
        kind = tile.get("kind")
        if role == "occupied_structure":
            animal = tile.get("animal")
            if animal in ANIMAL_ORDER:
                placed[ANIMAL_ORDER.index(animal)] += 1
        elif role == "empty_structure":
            if kind == "COOP":
                empty_coops += 1
            elif kind == "PASTURE":
                empty_pastures += 1
    return PhysicalContext(
        observed_land=len(unlocked),
        crop_build_cells_by_land=tuple(cells_by_land),
        placed_animals=tuple(placed),
        reusable_empty_coops=empty_coops,
        reusable_empty_pastures=empty_pastures,
        unplaced_animals=unplaced_animals,
    )


def required_new_housing_cells(
    context: PhysicalContext,
    land_target: int,
    animal_targets: Sequence[int],
) -> int:
    """Return required newly built housing cells; never clamps ``C``."""
    target = _animal_vector(animal_targets, "animal targets")
    for index, (value, placed) in enumerate(zip(target, context.placed_animals)):
        if value < placed:
            raise ValueError(
                f"animal target {ANIMAL_ORDER[index]}={value} is below "
                f"observed placed count {placed}")
    context.crop_build_cells(land_target)
    goose_deficit = target[0] - context.placed_animals[0]
    pasture_deficit = (
        target[1] - context.placed_animals[1]
        + target[2] - context.placed_animals[2])
    new_coops = max(0, goose_deficit - context.reusable_empty_coops)
    new_pastures = max(0, pasture_deficit - context.reusable_empty_pastures)
    return new_coops + new_pastures


def physical_crop_capacity(
    context: PhysicalContext,
    land_target: int,
    animal_targets: Sequence[int],
) -> int:
    """Return ``C = B - newly-required housing cells`` for one prefix."""
    return context.crop_build_cells(land_target) - required_new_housing_cells(
        context, land_target, animal_targets)


def animal_prefix_is_feasible(
    context: PhysicalContext,
    land_target: int,
    animal_targets: Sequence[int],
) -> bool:
    """Whether an absolute animal prefix is physically feasible."""
    targets = _animal_vector(animal_targets, "animal targets")
    try:
        context.crop_build_cells(land_target)
    except ValueError:
        return False
    if any(value < placed for value, placed in zip(targets, context.placed_animals)):
        return False
    return physical_crop_capacity(context, land_target, targets) >= 0


def animal_target_support_mask(
    context: PhysicalContext,
    land_target: int,
    species_index: int,
    previous_targets: Sequence[int] = (),
) -> tuple[bool, ...]:
    """Return the 101-class support mask for one autoregressive animal head."""
    index = _require_range(species_index, "animal species index", 0,
                           len(ANIMAL_ORDER) - 1)
    previous = tuple(previous_targets)
    if len(previous) != index:
        raise ValueError(
            f"previous animal prefix must contain {index} targets, got "
            f"{len(previous)}")
    base = list(context.placed_animals)
    for prior, value in enumerate(previous):
        base[prior] = _require_range(value,
                                     f"previous animal target {ANIMAL_ORDER[prior]}",
                                     ANIMAL_TARGET_MIN, ANIMAL_TARGET_MAX)
    mask: list[bool] = []
    for candidate in range(ANIMAL_TARGET_MAX + 1):
        targets = base[:]
        targets[index] = candidate
        mask.append(animal_prefix_is_feasible(context, land_target, targets))
    return tuple(mask)


def animal_acquisition_deficits(
    context: PhysicalContext,
    requested_targets: Sequence[int],
) -> tuple[int, ...]:
    """Return purchases still required after shed/carried animals are reused."""
    targets = _animal_vector(requested_targets, "requested animal targets")
    for index, (target, placed) in enumerate(zip(targets, context.placed_animals)):
        if target < placed:
            raise ValueError(
                f"requested animal target {ANIMAL_ORDER[index]}={target} is "
                f"below observed placed count {placed}")
    return tuple(max(0, target - placed - owned)
                   for target, placed, owned in zip(
                       targets, context.placed_animals, context.unplaced_animals))


def unplaced_animal_counts(
    shed: Mapping[str, int],
    carried_inventories: Iterable[Mapping[str, int]] = (),
) -> tuple[int, ...]:
    """Count owned, not-yet-placed animals across shed and carried hands."""
    if not isinstance(shed, Mapping):
        raise ValueError("shed must be a mapping")
    totals = [0] * len(ANIMAL_ORDER)
    sources = (shed, *tuple(carried_inventories))
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("carried inventories must be mappings")
        for index, animal in enumerate(ANIMAL_ORDER):
            value = source.get(animal, 0)
            parsed = _require_int(value, f"inventory count {animal}")
            if parsed < 0:
                raise ValueError(f"inventory count {animal} must be nonnegative")
            totals[index] += parsed
    return tuple(totals)


def crop_delta_support_mask(previous_goal: int, residual_capacity: int) \
        -> tuple[bool, ...]:
    """Return the 201-class mask for one crop head.

    The arithmetic intentionally leaves negative residual capacity negative;
    no capacity is hidden by clamping or post-sampling repair.
    """
    goal = _require_range(previous_goal, "previous crop goal",
                          CROP_GOAL_MIN, CROP_GOAL_MAX)
    residual = _require_int(residual_capacity, "residual crop capacity")
    upper_delta = min(CROP_DELTA_MAX, residual) - goal
    lower_delta = max(CROP_DELTA_MIN, -goal)
    return tuple(lower_delta <= delta <= upper_delta
                 for delta in range(CROP_DELTA_MIN, CROP_DELTA_MAX + 1))


def decode_supported_crop_goals(
    previous_goals: Sequence[int],
    sampled_class_indices: Sequence[int],
    total_capacity: int,
) -> tuple[int, ...]:
    """Decode one supported fixed-order crop prefix, rejecting bad samples."""
    previous = _crop_goal_vector(previous_goals, "previous crop goal")
    classes = tuple(sampled_class_indices)
    if len(classes) != len(CROP_ORDER):
        raise ValueError(
            f"sampled crop classes must contain {len(CROP_ORDER)} values, "
            f"got {len(classes)}")
    capacity = _require_int(total_capacity, "total crop capacity")
    decoded: list[int] = []
    for index, (goal, class_index) in enumerate(zip(previous, classes)):
        delta = crop_class_to_delta(class_index)
        residual = capacity - sum(decoded)
        if not crop_delta_support_mask(goal, residual)[class_index]:
            raise ValueError(
                f"unsupported {CROP_ORDER[index]} class {class_index} "
                f"(delta={delta}, residual={residual})")
        decoded.append(transition_crop_goal(goal, delta))
    return tuple(decoded)
