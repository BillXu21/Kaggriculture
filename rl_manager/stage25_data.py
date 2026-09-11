"""Pure logical-row outcome proxies for the Stage 2.5 manager contract.

This module intentionally stops at a small, immutable data contract.  It does
not read Parquet, import the replay adapter, or depend on a training stack.
The physical checks are delegated to :mod:`rl_manager.stage25_mechanics`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .stage25_mechanics import (
    ACTION_ORDER,
    ANIMAL_ORDER,
    CROP_ORDER,
    PhysicalContext,
    animal_target_support_mask,
    crop_delta_support_mask,
    crop_delta_to_class,
    land_target_support_mask,
    physical_context_from_board,
    physical_crop_capacity,
    unplaced_animal_counts,
)


OUTCOME_PROXY_SCHEMA_VERSION = "stage25_outcome_proxy_v1"
# Short aliases make the persisted contract discoverable without introducing
# a second version value.
STAGE25_DATA_VERSION = OUTCOME_PROXY_SCHEMA_VERSION
OUTCOME_PROXY_VERSION = OUTCOME_PROXY_SCHEMA_VERSION

_CROP_DELTA_MIN = -100
_CROP_DELTA_MAX = 100
_LAND_COMPONENT = "land"
_ANIMAL_COMPONENTS = tuple(name.lower() for name in ANIMAL_ORDER)
_CROP_COMPONENTS = tuple(name.lower() for name in CROP_ORDER)
_QUADRANT_ORDER = ("NW", "NE", "SW", "SE")
# Logical records are replay_daily records.  Keep this literal here so the
# data contract remains framework-free and does not import the replay stack.
_CANONICAL_RECORD_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class OutcomeProxyProvenance:
    """How the pre-decision crop goals for one row were established."""

    prior_source: str
    prior_row_index: int | None
    prior_day: int | None
    prior_crop_goals: tuple[int, ...]
    gap_days: int
    history_reset: bool


@dataclass(frozen=True)
class OutcomeProxyLabel:
    """Immutable labels and provenance for one returned logical row.

    A component with no valid label is represented by ``None``.  This is an
    intentional exclusion, not a repaired or clipped target.
    """

    row_index: int
    episode_id: Any
    seat: Any
    day: int
    date: str | None
    land_label: int | None
    land_class: int | None
    animal_labels: tuple[int | None, ...]
    animal_classes: tuple[int | None, ...]
    crop_goals: tuple[int, ...]
    crop_deltas: tuple[int | None, ...]
    crop_classes: tuple[int | None, ...]
    provenance: OutcomeProxyProvenance
    valid_components: tuple[str, ...]
    invalid_components: tuple[str, ...]
    # A complete nine-action teacher-forcing example requires every one of the
    # nine ordered steps to have a populated class that is physically supported
    # under the exact preceding observed class prefix.  A partial chain may be
    # retained for diagnostics, but it is never a complete trainable example.
    complete_ar_chain: bool = False

    @property
    def labels(self) -> Mapping[str, Any]:
        """A read-only logical label view for callers that prefer mappings."""
        return MappingProxyType({
            "land": self.land_label,
            "animals": self.animal_labels,
            "crops": self.crop_goals,
            "crop_deltas": self.crop_deltas,
            "crop_classes": self.crop_classes,
        })


@dataclass(frozen=True)
class InvalidCounters:
    """Stable diagnostics for invalid components and excluded rows.

    The diagnostic fields are deliberately explicit about whether they count
    rows or components; these names are part of the data contract.
    """

    crop_delta_outside_vocabulary_components: int = 0
    crop_physical_incompatibility_components: int = 0
    land_invalidity_rows: int = 0
    animal_acquisition_invalidity_components: int = 0
    history_reset_gap_rows: int = 0
    animal_loss_ambiguity_components: int = 0
    animal_loss_ambiguity_rows: int = 0
    excluded_rows: int = 0
    invalid_rows: int = 0
    selection_excluded_rows: int = 0
    component_excluded_rows: int = 0
    target_invalidity_components: int = 0
    target_invalidity_rows: int = 0
    # Rows with at least one valid component but an incomplete nine-action
    # autoregressive chain.  They are retained only as partial diagnostics and
    # are excluded from the complete trainable output.  This is distinct from
    # ``component_excluded_rows``/``excluded_rows``, which cover rows with no
    # usable component at all.
    incomplete_ar_chain_rows: int = 0

    # Compact aliases retain the same values while keeping the canonical
    # fields above suitable for JSON/dataclass serialization.
    @property
    def crop_delta_out_of_range(self) -> int:
        return self.crop_delta_outside_vocabulary_components

    @property
    def crop_delta_outside_vocabulary(self) -> int:
        return self.crop_delta_outside_vocabulary_components

    @property
    def crop_physical_invalid(self) -> int:
        return self.crop_physical_incompatibility_components

    @property
    def crop_physical_incompatibility(self) -> int:
        return self.crop_physical_incompatibility_components

    @property
    def land_invalid(self) -> int:
        return self.land_invalidity_rows

    @property
    def land_invalidity(self) -> int:
        return self.land_invalidity_rows

    @property
    def animal_invalid(self) -> int:
        return self.animal_acquisition_invalidity_components

    @property
    def animal_acquisition_invalidity(self) -> int:
        return self.animal_acquisition_invalidity_components

    @property
    def history_reset(self) -> int:
        return self.history_reset_gap_rows

    @property
    def history_reset_gap(self) -> int:
        return self.history_reset_gap_rows

    @property
    def animal_loss_ambiguous(self) -> int:
        return self.animal_loss_ambiguity_components

    @property
    def animal_loss_ambiguity(self) -> int:
        return self.animal_loss_ambiguity_components

    @property
    def crop_label_physical_support_components(self) -> int:
        return self.crop_physical_incompatibility_components

    @property
    def land_label_invalid_rows(self) -> int:
        return self.land_invalidity_rows

    @property
    def animal_acquisition_label_invalid_components(self) -> int:
        return self.animal_acquisition_invalidity_components

    @property
    def history_gap_reset_rows(self) -> int:
        return self.history_reset_gap_rows

    @property
    def aggregate_excluded_rows(self) -> int:
        return self.excluded_rows

    @property
    def aggregate_invalid_rows(self) -> int:
        return self.invalid_rows

    @property
    def target_invalid_components(self) -> int:
        return self.target_invalidity_components

    @property
    def target_invalid_rows_count(self) -> int:
        return self.target_invalidity_rows


@dataclass(frozen=True)
class OutcomeProxyBuild:
    """Immutable builder output, preserving selected input order.

    ``rows`` (aliased by ``labels``) contains only complete nine-action
    teacher-forcing examples: every step has a populated class supported under
    the exact preceding observed prefix.  ``partial_rows`` retains selected
    rows with an incomplete chain for diagnostics/accounting only; those rows
    are never trainable examples.
    """

    schema_version: str
    rows: tuple[OutcomeProxyLabel, ...]
    counters: InvalidCounters
    partial_rows: tuple[OutcomeProxyLabel, ...] = ()

    @property
    def labels(self) -> tuple[OutcomeProxyLabel, ...]:
        return self.rows

    @property
    def complete_rows(self) -> tuple[OutcomeProxyLabel, ...]:
        return self.rows

    @property
    def trainable_rows(self) -> tuple[OutcomeProxyLabel, ...]:
        return self.rows

    @property
    def partial_labels(self) -> tuple[OutcomeProxyLabel, ...]:
        return self.partial_rows

    @property
    def report(self) -> InvalidCounters:
        return self.counters

    @property
    def invalid_counters(self) -> InvalidCounters:
        return self.counters

    @property
    def invalid_counts(self) -> InvalidCounters:
        return self.counters

    @property
    def counts(self) -> InvalidCounters:
        return self.counters

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> OutcomeProxyLabel:
        return self.rows[index]


OutcomeProxyRow = OutcomeProxyLabel
OutcomeProxyResult = OutcomeProxyBuild
BuildCounters = InvalidCounters


__all__ = [
    "OUTCOME_PROXY_SCHEMA_VERSION",
    "STAGE25_DATA_VERSION",
    "OUTCOME_PROXY_VERSION",
    "OutcomeProxyProvenance",
    "OutcomeProxyLabel",
    "OutcomeProxyRow",
    "InvalidCounters",
    "BuildCounters",
    "OutcomeProxyBuild",
    "OutcomeProxyResult",
    "build_outcome_proxy_labels",
]


def _as_mapping(value: object, path: str) -> Mapping[Any, Any]:
    """Normalize a dict or an Arrow ``map`` list of key/value pairs."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a mapping or list of pairs")
    result: dict[Any, Any] = {}
    for index, pair in enumerate(value):
        if isinstance(pair, Mapping):
            if "key" not in pair or "value" not in pair:
                raise ValueError(f"{path}[{index}] must contain key and value")
            key, item = pair["key"], pair["value"]
        elif isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) \
                and len(pair) == 2:
            key, item = pair
        else:
            raise ValueError(f"{path}[{index}] must be a key/value pair")
        if key in result:
            raise ValueError(f"duplicate map key at {path}: {key!r}")
        result[key] = item
    return result


def _field(value: object, name: str, path: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"missing {path}.{name}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"missing {path}.{name}") from exc


def _optional_field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _present_field(value: object, name: str) -> tuple[bool, Any]:
    if isinstance(value, Mapping):
        return name in value, value.get(name)
    if hasattr(value, name):
        return True, getattr(value, name)
    return False, None


def _first_present(record: object, metadata: Mapping[Any, Any],
                   names: Sequence[str]) -> tuple[bool, Any]:
    for name in names:
        present, value = _present_field(record, name)
        if present:
            return True, value
        if name in metadata:
            return True, metadata[name]
    return False, None


def _int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer, got {value!r}")
    return value


def _tile(tile: object) -> object:
    """Turn an Arrow tagged tile back into the logical mechanics shape."""
    if not isinstance(tile, Mapping) or "tile_kind" not in tile:
        return tile
    if tile.get("bare_string"):
        return tile["tile_kind"]
    if tile["tile_kind"] == "EMPTY":
        return None
    out = {key: value for key, value in tile.items()
           if key not in {"tile_kind", "bare_string", "present_mask", "derived"}
           and value is not None}
    out["kind"] = tile["tile_kind"]
    return out


def _board(value: object, path: str) -> list[list[object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a board sequence")
    rows = []
    for y, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"{path}[{y}] must be a row sequence")
        rows.append([_tile(tile) for tile in row])
    return rows


def _unlocked(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    result = tuple(value)
    if len(result) > len(_QUADRANT_ORDER) or any(
            not isinstance(quadrant, str) or quadrant not in _QUADRANT_ORDER
            for quadrant in result):
        raise ValueError(f"{path} must use {_QUADRANT_ORDER}")
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicate quadrants")
    if result != _QUADRANT_ORDER[:len(result)]:
        raise ValueError(f"{path} must be the canonical land prefix")
    return result


def _target_component_issues(targets: Mapping[Any, Any], field_name: str,
                             order: Sequence[str], observed: Sequence[int],
                             path: str) -> tuple[str, ...]:
    """Validate optional derived targets without making them authoritative."""
    if field_name not in targets:
        return ()
    raw = targets[field_name]
    components = tuple(name.lower() for name in order)
    try:
        values = _as_mapping(raw, path)
    except ValueError:
        return components
    if any(key not in order for key in values):
        return components
    issues: list[str] = []
    for index, name in enumerate(order):
        component = name.lower()
        # Canonical replay_daily maps are sparse: an omitted crop/animal is
        # the observed zero count, not a missing label.
        raw_value = values.get(name, 0)
        try:
            value = _int(raw_value, f"{path}.{name}")
        except ValueError:
            issues.append(component)
            continue
        if value != observed[index]:
            issues.append(component)
    return tuple(issues)


def _target_land_issue(targets: Mapping[Any, Any], observed: Sequence[str],
                       path: str) -> bool:
    raw = targets.get("unlocked_quadrants_end")
    if raw is None:
        return False
    try:
        target = _unlocked(raw, path)
    except ValueError:
        return True
    return target != tuple(observed)


def _state(section: object, path: str) -> Mapping[Any, Any]:
    state = _field(section, "self", path)
    if not isinstance(state, Mapping):
        raise ValueError(f"{path}.self must be a mapping")
    return state


def _crop_counts(board: Sequence[Sequence[object]], path: str) -> tuple[int, ...]:
    counts = [0] * len(CROP_ORDER)
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if not isinstance(tile, Mapping) or tile.get("kind") != "PLANT":
                continue
            crop = tile.get("crop")
            if crop in CROP_ORDER:
                counts[CROP_ORDER.index(crop)] += 1
            elif crop is not None:
                raise ValueError(f"unknown crop {crop!r} at {path}[{y}][{x}]")
    return tuple(counts)


def _animal_counts(board: Sequence[Sequence[object]], path: str) -> tuple[int, ...]:
    counts = [0] * len(ANIMAL_ORDER)
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if not isinstance(tile, Mapping) or tile.get("kind") != "COOP" \
                    and tile.get("kind") != "PASTURE":
                continue
            animal = tile.get("animal")
            if animal in ANIMAL_ORDER:
                counts[ANIMAL_ORDER.index(animal)] += 1
            elif animal is not None:
                raise ValueError(f"unknown animal {animal!r} at {path}[{y}][{x}]")
    return tuple(counts)


def _target_counts(value: object, order: Sequence[str], path: str) -> tuple[int, ...]:
    values = _as_mapping(value, path)
    result: list[int] = []
    for name in order:
        raw = values.get(name, 0)
        result.append(_int(raw, f"{path}.{name}"))
    return tuple(result)


def _date(record: object, metadata: Mapping[Any, Any]) -> str | None:
    value = metadata.get("partition_date", metadata.get("date"))
    if value is None:
        value = _optional_field(record, "partition_date")
    return None if value is None else str(value)


def _score(record: object, metadata: Mapping[Any, Any]) -> float | None:
    value = _optional_field(record, "score")
    if value is None:
        value = metadata.get("score", metadata.get("min_score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata.min_score must be numeric, got {value!r}") from exc


def _selected(record: object, metadata: Mapping[Any, Any],
              selected_dates: frozenset[str] | None,
              min_score: float | None) -> bool:
    if selected_dates is not None and _date(record, metadata) not in selected_dates:
        return False
    if min_score is not None:
        score = _score(record, metadata)
        if score is None or score < min_score:
            return False
    return True


@dataclass(frozen=True)
class _HistoryRow:
    index: int
    record: object
    metadata: Mapping[Any, Any]
    episode_id: Any
    seat: Any
    day: int
    date: str | None
    start_self: Mapping[Any, Any]
    end_self: Mapping[Any, Any]
    start_board: list[list[object]]
    end_board: list[list[object]]
    start_land: int
    end_land: int
    start_crops: tuple[int, ...]
    end_crops: tuple[int, ...]
    start_animals: tuple[int, ...]
    end_animals: tuple[int, ...]
    has_schema_version: bool
    schema_version: Any
    has_boundary_id: bool
    boundary_id: Any
    reset_requested: bool
    reset_identity: Any
    target_invalid_components: tuple[str, ...]


def _parse_row(index: int, record: object) -> _HistoryRow:
    metadata_raw = _field(record, "metadata", "record")
    metadata = _as_mapping(metadata_raw, "record.metadata")
    episode_id = metadata.get("episode_id")
    seat = metadata.get("seat")
    if episode_id is None or seat is None:
        raise ValueError(f"row {index}: metadata.episode_id and metadata.seat are required")
    for name, expected in (("episode_id", episode_id), ("seat", seat)):
        present, value = _present_field(record, name)
        if present and value != expected:
            raise ValueError(
                f"row {index}: top-level {name} disagrees with metadata")
    day = _int(_field(record, "day", f"row {index}"), f"row {index}.day")
    start = _field(record, "start", f"row {index}")
    end = _field(record, "end", f"row {index}")
    start_self = _state(start, f"row {index}.start")
    end_self = _state(end, f"row {index}.end")
    start_board = _board(_field(start_self, "board", f"row {index}.start.self"),
                         f"row {index}.start.self.board")
    end_board = _board(_field(end_self, "board", f"row {index}.end.self"),
                       f"row {index}.end.self.board")
    start_unlocked = _unlocked(
        _field(start_self, "unlocked_quadrants", f"row {index}.start.self"),
        f"row {index}.start.self.unlocked_quadrants")
    end_unlocked = _unlocked(
        _field(end_self, "unlocked_quadrants", f"row {index}.end.self"),
        f"row {index}.end.self.unlocked_quadrants")

    has_schema_version, schema_version = _first_present(
        record, metadata, ("schema_version",))
    if has_schema_version and schema_version != _CANONICAL_RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"row {index}: unsupported schema_version {schema_version!r}; "
            f"expected {_CANONICAL_RECORD_SCHEMA_VERSION}")

    has_boundary_id, boundary_id = _first_present(
        record, metadata, ("boundary_id", "manager_boundary_id", "boundary_index"))
    reset_present, reset_value = _first_present(
        record, metadata, ("reset", "reset_marker", "environment_reset",
                           "episode_reset", "is_reset"))
    reset_id_present, reset_identity = _first_present(
        record, metadata, ("reset_id", "environment_reset_id", "episode_reset_id"))
    reset_requested = bool(reset_value) if reset_present else False
    if reset_id_present and reset_identity is not None:
        # A reset id identifies a new environment sequence when it changes;
        # the comparison is performed while constructing history below.
        reset_requested = reset_requested or False

    targets_raw = _optional_field(record, "targets")
    target_container_invalid = False
    if targets_raw is None:
        targets = {}
    else:
        try:
            targets = _as_mapping(targets_raw, f"row {index}.targets")
        except ValueError:
            # Targets are derived conveniences.  A malformed convenience
            # must not make us lose a structurally usable observed state.
            targets = {}
            target_container_invalid = True
    end_crops = _crop_counts(end_board, f"row {index}.end.self.board")
    end_animals = _animal_counts(end_board, f"row {index}.end.self.board")
    target_invalid = list(_target_component_issues(
        targets, "crop_composition_end", CROP_ORDER, end_crops,
        f"row {index}.targets.crop_composition_end"))
    target_invalid.extend(_target_component_issues(
        targets, "animal_counts_end", ANIMAL_ORDER, end_animals,
        f"row {index}.targets.animal_counts_end"))
    if _target_land_issue(
            targets, end_unlocked,
            f"row {index}.targets.unlocked_quadrants_end"):
        target_invalid.append(_LAND_COMPONENT)
    if target_container_invalid:
        target_invalid.extend(_CROP_COMPONENTS)
        target_invalid.extend(_ANIMAL_COMPONENTS)
        target_invalid.append(_LAND_COMPONENT)
    return _HistoryRow(
        index=index,
        record=record,
        metadata=metadata,
        episode_id=episode_id,
        seat=seat,
        day=day,
        date=_date(record, metadata),
        start_self=start_self,
        end_self=end_self,
        start_board=start_board,
        end_board=end_board,
        start_land=len(start_unlocked),
        end_land=len(end_unlocked),
        start_crops=_crop_counts(start_board, f"row {index}.start.self.board"),
        end_crops=end_crops,
        start_animals=_animal_counts(start_board, f"row {index}.start.self.board"),
        end_animals=end_animals,
        has_schema_version=has_schema_version,
        schema_version=schema_version,
        has_boundary_id=has_boundary_id,
        boundary_id=boundary_id,
        reset_requested=reset_requested,
        reset_identity=reset_identity if reset_id_present else None,
        target_invalid_components=tuple(dict.fromkeys(target_invalid)),
    )


def _physical_context(row: _HistoryRow) -> PhysicalContext:
    shed = _as_mapping(row.start_self.get("shed", {}),
                       f"row {row.index}.start.self.shed")
    inventories_raw = row.start_self.get("inventories", ())
    if isinstance(inventories_raw, (str, bytes)) or not isinstance(inventories_raw, Sequence):
        raise ValueError(f"row {row.index}.start.self.inventories must be a sequence")
    inventories = tuple(_as_mapping(item, f"row {row.index}.start.self.inventories[{i}]")
                        for i, item in enumerate(inventories_raw))
    owned = unplaced_animal_counts(shed, inventories)
    unlocked = row.start_self["unlocked_quadrants"]
    return physical_context_from_board(row.start_board, unlocked,
                                       unplaced_animals=owned)


def _component_name(kind: str, index: int) -> str:
    return (_ANIMAL_COMPONENTS if kind == "animal" else _CROP_COMPONENTS)[index]


def _make_label(row: _HistoryRow, prior: OutcomeProxyProvenance,
                counters: dict[str, int]) -> OutcomeProxyLabel:
    valid: list[str] = []
    target_invalid = set(row.target_invalid_components)
    # Target mirrors never replace observed outcomes.  A bad mirror excludes
    # only that derived proxy component while leaving the observed value in
    # the corresponding ``*_label``/``*_goals`` field.
    invalid: list[str] = list(row.target_invalid_components)
    if target_invalid:
        counters["target_invalidity_components"] += len(target_invalid)
        counters["target_invalidity_rows"] += 1

    # Step 1 -- land.  This is the first action of the nine-step autoregressive
    # chain, so an invalid land outcome truncates every later step.
    land_label: int | None = row.end_land
    land_class: int | None = None
    land_valid = False
    try:
        if _LAND_COMPONENT in target_invalid:
            raise ValueError
        land_mask = land_target_support_mask(row.start_land)
        if not 1 <= row.end_land <= len(land_mask) or not land_mask[row.end_land - 1]:
            raise ValueError
        land_class = row.end_land - 1
        land_valid = True
        valid.append(_LAND_COMPONENT)
    except (ValueError, IndexError):
        counters["land_invalidity_rows"] += 1
        land_label = None
        invalid.append(_LAND_COMPONENT)

    # Steps 2-4 -- animals.  A component is only usable when its own class is
    # populated and every earlier action is itself valid.  Once a step is
    # invalid, later steps are not reconstructed from a fabricated prefix.
    animal_labels: list[int | None] = list(row.end_animals)
    animal_classes: list[int | None] = [None] * len(ANIMAL_ORDER)
    context: PhysicalContext | None = None
    try:
        context = _physical_context(row)
    except (ValueError, KeyError, TypeError):
        context = None

    loss_indices = {
        index for index, (start, end) in enumerate(
            zip(row.start_animals, row.end_animals)) if end < start
    }
    for index in loss_indices:
        component = _component_name("animal", index)
        counters["animal_loss_ambiguity_components"] += 1
        counters["animal_acquisition_invalidity_components"] += 1
        animal_labels[index] = None
        if component not in invalid:
            invalid.append(component)
    if loss_indices:
        counters["animal_loss_ambiguity_rows"] += 1

    animal_prefix: list[int] = []
    prefix_ok = land_valid
    for index, component in enumerate(_ANIMAL_COMPONENTS):
        target = row.end_animals[index]
        if component in target_invalid:
            # The count remains the observed end-board count; only the
            # target-derived class is withheld.  The observed prefix breaks
            # here because this step has no populated class.
            prefix_ok = False
            if component not in invalid:
                invalid.append(component)
            continue
        if index in loss_indices:
            # A loss is an ambiguous outcome, never a purchase target, so the
            # observed autoregressive prefix cannot continue through it.
            prefix_ok = False
            continue
        if not prefix_ok:
            counters["animal_acquisition_invalidity_components"] += 1
            animal_labels[index] = None
            if component not in invalid:
                invalid.append(component)
            continue
        try:
            # Positive deficits are legitimate purchases; a target below
            # observed placement is never repaired.
            if (context is None or land_label is None
                    or isinstance(target, bool) or not isinstance(target, int)
                    or not 0 <= target <= 100
                    or target < context.placed_animals[index]):
                raise ValueError
            mask = animal_target_support_mask(
                context, land_label, index, tuple(animal_prefix))
            if not mask[target]:
                raise ValueError
        except (ValueError, IndexError):
            counters["animal_acquisition_invalidity_components"] += 1
            animal_labels[index] = None
            prefix_ok = False
            if component not in invalid:
                invalid.append(component)
            continue
        animal_classes[index] = target
        valid.append(component)
        animal_prefix.append(target)
    animal_valid = prefix_ok

    # Steps 5-9 -- crops.  Crops depend on the complete land+animal prefix and
    # only on the decoded goals of earlier crops; capacity is never reserved
    # for later crop heads.
    crop_deltas: list[int | None] = [None] * len(CROP_ORDER)
    crop_classes: list[int | None] = [None] * len(CROP_ORDER)
    if context is not None and land_label is not None and animal_valid:
        try:
            capacity = physical_crop_capacity(
                context, land_label, tuple(animal_classes))
        except (ValueError, TypeError):
            capacity = -1
        if capacity >= 0:
            decoded_goals: list[int] = []
            crop_ok = True
            for index, (prior_goal, desired) in enumerate(
                     zip(prior.prior_crop_goals, row.end_crops)):
                component = _component_name("crop", index)
                if component in target_invalid:
                    crop_ok = False
                    if component not in invalid:
                        invalid.append(component)
                    continue
                if not crop_ok:
                    if component not in invalid:
                        invalid.append(component)
                    continue
                delta = desired - prior_goal
                if not _CROP_DELTA_MIN <= delta <= _CROP_DELTA_MAX:
                    counters["crop_delta_outside_vocabulary_components"] += 1
                    crop_ok = False
                    if component not in invalid:
                        invalid.append(component)
                    continue
                if not 0 <= prior_goal <= 100 or not 0 <= desired <= 100:
                    counters["crop_physical_incompatibility_components"] += 1
                    crop_ok = False
                    if component not in invalid:
                        invalid.append(component)
                    continue
                residual = capacity - sum(decoded_goals)
                try:
                    class_index = crop_delta_to_class(delta)
                    if not crop_delta_support_mask(prior_goal, residual)[class_index]:
                        raise ValueError
                except (ValueError, IndexError):
                    counters["crop_physical_incompatibility_components"] += 1
                    crop_ok = False
                    if component not in invalid:
                        invalid.append(component)
                    continue
                crop_deltas[index] = delta
                crop_classes[index] = class_index
                decoded_goals.append(prior_goal + delta)
                valid.append(component)
        else:
            # Capacity is a shared prerequisite, so it invalidates all crop
            # proxies.  It still never changes the observed end counts.
            for index in range(len(CROP_ORDER)):
                component = _component_name("crop", index)
                if component not in invalid:
                    invalid.append(component)
    else:
        for index in range(len(CROP_ORDER)):
            component = _component_name("crop", index)
            if component not in invalid:
                invalid.append(component)

    # Remove duplicate component names while retaining canonical order.
    valid = list(dict.fromkeys(valid))
    invalid = [name for name in dict.fromkeys(invalid) if name not in valid]
    if invalid:
        counters["invalid_rows"] += 1
        counters["component_excluded_rows"] += 1
    # A complete teacher-forcing example requires all nine ordered steps to be
    # present, populated, and physically supported under the exact prefix.
    complete_ar_chain = len(valid) == len(ACTION_ORDER)

    return OutcomeProxyLabel(
        row_index=row.index,
        episode_id=row.episode_id,
        seat=row.seat,
        day=row.day,
        date=row.date,
        land_label=land_label,
        land_class=land_class,
        animal_labels=tuple(animal_labels),
        animal_classes=tuple(animal_classes),
        crop_goals=row.end_crops,
        crop_deltas=tuple(crop_deltas),
        crop_classes=tuple(crop_classes),
        provenance=prior,
        valid_components=tuple(valid),
        invalid_components=tuple(invalid),
        complete_ar_chain=complete_ar_chain,
    )


def _boundary_adjacent(previous: _HistoryRow, current: _HistoryRow) -> bool:
    """Require a checkable boundary identity when either row supplies one."""
    if previous.has_boundary_id != current.has_boundary_id:
        return False
    if not previous.has_boundary_id:
        return True
    left, right = previous.boundary_id, current.boundary_id
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, int) and isinstance(right, int):
        return right == left + 1
    # Non-numeric identities cannot establish the expected next boundary.
    return False


def _sequence_identity_adjacent(previous: _HistoryRow,
                                current: _HistoryRow) -> bool:
    if previous.has_schema_version != current.has_schema_version:
        return False
    if (previous.has_schema_version and
            previous.schema_version != current.schema_version):
        return False
    if not _boundary_adjacent(previous, current):
        return False
    if ((previous.reset_identity is None) != (current.reset_identity is None)
            or (previous.reset_identity is not None
                and previous.reset_identity != current.reset_identity)):
        return False
    return True


def build_outcome_proxy_labels(
    records: Iterable[Mapping[str, Any]],
    selected_dates: Iterable[object] | None = None,
    min_score: float | None = None,
) -> OutcomeProxyBuild:
    """Build canonical Stage 2.5 labels from logical daily records.

    Histories are built in input order before date/score selection.  Thus an
    unselected row can still establish the prior synthetic crop goal for a
    selected adjacent row, while out-of-order input cannot fabricate one.
    """
    source = tuple(records)
    dates = None if selected_dates is None else frozenset(str(value) for value in selected_dates)
    threshold = None if min_score is None else float(min_score)
    parsed = tuple(_parse_row(index, record) for index, record in enumerate(source))
    groups: dict[tuple[Any, Any], list[_HistoryRow]] = {}
    for row in parsed:
        groups.setdefault((row.episode_id, row.seat), []).append(row)

    counters = {
        "crop_delta_outside_vocabulary_components": 0,
        "crop_physical_incompatibility_components": 0,
        "land_invalidity_rows": 0,
        "animal_acquisition_invalidity_components": 0,
        "history_reset_gap_rows": 0,
        "animal_loss_ambiguity_components": 0,
        "animal_loss_ambiguity_rows": 0,
        "invalid_rows": 0,
        "component_excluded_rows": 0,
        "selection_excluded_rows": 0,
        "target_invalidity_components": 0,
        "target_invalidity_rows": 0,
        "incomplete_ar_chain_rows": 0,
    }
    built: list[OutcomeProxyLabel] = []
    partial: list[OutcomeProxyLabel] = []
    for history in groups.values():
        previous_goals: tuple[int, ...] | None = None
        previous_row: _HistoryRow | None = None
        for row in history:
            if previous_row is None:
                reset = row.reset_requested
                prior_source = ("history_reset_start_occupancy" if reset
                                else "first_boundary_start_occupancy")
                prior_goals = row.start_crops
                prior_index = None
                prior_day = None
                gap_days = 0
                if reset:
                    counters["history_reset_gap_rows"] += 1
            elif (row.day == previous_row.day + 1
                  and _sequence_identity_adjacent(previous_row, row)
                  and not row.reset_requested):
                prior_source = "previous_synthetic_desired_end_goal"
                prior_goals = previous_goals if previous_goals is not None \
                    else previous_row.end_crops
                prior_index = previous_row.index
                prior_day = previous_row.day
                gap_days = 0
                reset = False
            else:
                prior_source = "history_reset_gap_start_occupancy"
                prior_goals = row.start_crops
                prior_index = None
                prior_day = None
                gap_days = max(0, row.day - previous_row.day - 1)
                reset = True
                counters["history_reset_gap_rows"] += 1
            provenance = OutcomeProxyProvenance(
                prior_source=prior_source,
                prior_row_index=prior_index,
                prior_day=prior_day,
                prior_crop_goals=tuple(prior_goals),
                gap_days=gap_days,
                history_reset=reset,
            )
            label = _make_label(row, provenance, counters)
            previous_goals = row.end_crops
            previous_row = row
            if _selected(row.record, row.metadata, dates, threshold):
                if label.complete_ar_chain:
                    built.append(label)
                elif label.valid_components:
                    # Retained only as a partial diagnostic; never a complete
                    # teacher-forcing training example.
                    partial.append(label)
                    counters["incomplete_ar_chain_rows"] += 1
            else:
                counters["selection_excluded_rows"] += 1

    # Preserves the original exclusion meaning: rows excluded by selection or
    # by having no usable component.  Partial diagnostic rows are neither
    # trainable nor counted as fully excluded here.
    excluded_rows = len(parsed) - len(built) - len(partial)
    final_counters = InvalidCounters(
        crop_delta_outside_vocabulary_components=counters[
            "crop_delta_outside_vocabulary_components"],
        crop_physical_incompatibility_components=counters[
            "crop_physical_incompatibility_components"],
        land_invalidity_rows=counters["land_invalidity_rows"],
        animal_acquisition_invalidity_components=counters[
            "animal_acquisition_invalidity_components"],
        history_reset_gap_rows=counters["history_reset_gap_rows"],
        animal_loss_ambiguity_components=counters[
            "animal_loss_ambiguity_components"],
        animal_loss_ambiguity_rows=counters["animal_loss_ambiguity_rows"],
        excluded_rows=excluded_rows,
        invalid_rows=counters["invalid_rows"],
        selection_excluded_rows=counters["selection_excluded_rows"],
        component_excluded_rows=counters["component_excluded_rows"],
        target_invalidity_components=counters[
            "target_invalidity_components"],
        target_invalidity_rows=counters["target_invalidity_rows"],
        incomplete_ar_chain_rows=counters["incomplete_ar_chain_rows"],
    )
    return OutcomeProxyBuild(
        schema_version=OUTCOME_PROXY_SCHEMA_VERSION,
        rows=tuple(sorted(built, key=lambda label: label.row_index)),
        counters=final_counters,
        partial_rows=tuple(sorted(partial, key=lambda label: label.row_index)),
    )
