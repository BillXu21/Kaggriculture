"""Projected canonical-Parquet loader for the Stage 2.5 manager.

The loader deliberately reads only the state and outcome fields needed by the
Stage 2.5 contract.  In particular, it does not project or materialize the
rich ``events`` struct.  Arrow rows are converted back to the logical shape
expected by :func:`rl_manager.stage25_data.build_outcome_proxy_labels`, while
the shared BC adapter encodes the start state into compact arrays.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bc_manager.adapter import _eval_metadata, _input_arrays_from_starts
from bc_manager.economics import (
    ECONOMIC_CONTEXT_KEY,
    E_HISTORY_CORRECTED_V1,
    derive_economic_context,
)
from replay_daily.constants import SCHEMA_VERSION
from replay_daily.lifecycle import canonical_board, replaceable_today
from replay_daily.storage import (
    _denorm_map,
    _denorm_public,
    _denorm_self,
    _denorm_shared,
    _denorm_targets,
)

from .stage25_data import (
    OutcomeProxyBuild,
    OutcomeProxyLabel,
    OutcomeProxyBuilder,
    _date as _data_date,
    _score as _data_score,
)
from .stage25_mechanics import (
    available_crop_slots,
    physical_context_from_board,
)

PROJECTED_COLUMNS = ("schema_version", "metadata", "day", "start", "targets", "end")
DEFAULT_TRAIN_DATES = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20")
DEFAULT_VAL_DATES = ("2026-08-21",)
DEFAULT_MIN_SCORE = 2950.0
DEFAULT_READ_BATCH_SIZE = 256
DEFAULT_PARQUET_PATHS = tuple(
    Path(f"canonical-{date}.parquet")
    for date in DEFAULT_TRAIN_DATES + DEFAULT_VAL_DATES
)
ACTION_STEPS = ("land", "goose", "cow", "sheep", "wheat", "carrot",
                "tomato", "strawberry", "melon")
CROP_STEPS = ACTION_STEPS[4:]


class SchemaVersionError(ValueError):
    """Raised when a projected file is missing, old, or mixed-version."""


@dataclass(frozen=True)
class Stage25AdapterConfig:
    """Configurable input paths and fixed date/score split policy."""

    paths: tuple[str | Path, ...] = DEFAULT_PARQUET_PATHS
    train_dates: tuple[str, ...] = DEFAULT_TRAIN_DATES
    val_dates: tuple[str, ...] = DEFAULT_VAL_DATES
    min_score: float = DEFAULT_MIN_SCORE
    e_history_version: str = E_HISTORY_CORRECTED_V1
    manager_start_day: int | None = None
    read_batch_size: int = DEFAULT_READ_BATCH_SIZE


@dataclass(frozen=True)
class _SourceRow:
    row_index: int
    source_path: str
    source_row: int


@dataclass(frozen=True, slots=True)
class _RowInfo:
    """Compact metadata retained after one canonical row is consumed."""

    source: _SourceRow
    metadata: dict[str, Any]
    date: str | None
    score: float | None
    audit_record: dict[str, Any]

    @property
    def day(self) -> int:
        return int(self.metadata["day"])

    @property
    def episode_id(self) -> Any:
        return self.metadata.get("episode_id")

    @property
    def seat(self) -> Any:
        return self.metadata.get("seat")


def _paths(value: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if value is None:
        return [Path(path) for path in DEFAULT_PARQUET_PATHS]
    if isinstance(value, (str, Path)):
        value = [value]
    result: list[Path] = []
    for item in value:
        candidate = Path(item)
        if candidate.is_dir():
            discovered = sorted(
                (path for path in candidate.rglob("*.parquet") if path.is_file()),
                key=lambda path: path.as_posix(),
            )
            if not discovered:
                raise ValueError(
                    f"{candidate}: canonical data directory contains no Parquet files")
            result.extend(discovered)
        else:
            result.append(candidate)
    if not result:
        raise ValueError("at least one canonical Parquet path is required")
    return list(dict.fromkeys(result))


def _require_schema_version(table: pa.Table | pa.RecordBatch, path: Path) -> None:
    if "schema_version" not in table.column_names:
        raise SchemaVersionError(f"{path}: schema_version column missing")
    versions = table.column("schema_version").to_pylist()
    if any(value is None for value in versions):
        raise SchemaVersionError(f"{path}: schema_version contains null")
    seen = sorted({int(value) for value in versions})
    if seen != [SCHEMA_VERSION]:
        raise SchemaVersionError(
            f"{path}: unsupported or mixed schema_version values {seen!r}; "
            f"expected only {SCHEMA_VERSION}"
        )


def _open_projected(path: Path) -> pq.ParquetFile:
    try:
        parquet = pq.ParquetFile(path)
        if "schema_version" not in parquet.schema_arrow.names:
            raise SchemaVersionError(f"{path}: schema_version column missing")
        missing = [name for name in PROJECTED_COLUMNS
                   if name not in parquet.schema_arrow.names]
        if missing:
            raise ValueError(
                f"{path}: not a canonical Stage 2.5 Parquet file; missing "
                f"projected columns {missing!r}"
            )
        return parquet
    except SchemaVersionError:
        raise
    except (KeyError, ValueError, pa.ArrowException) as exc:
        raise ValueError(
            f"{path}: not a canonical Stage 2.5 Parquet file ({exc}); "
            f"expected projected columns {list(PROJECTED_COLUMNS)!r}"
        ) from exc


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(_denorm_map(value))


def _logical_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one projected Arrow row to the logical builder shape."""
    start = row["start"]
    end = row["end"]
    targets = row.get("targets")

    def start_or_end(section: Mapping[str, Any], *, is_start: bool) -> dict[str, Any]:
        result = {
            "self": _denorm_self(section["self"]),
            "opponent_public": _denorm_public(section["opponent_public"]),
            **_denorm_shared(section),
        }
        if is_start:
            result.update({
                "day": section["day"],
                "hour": section["hour"],
                "previous_execution": {
                    "workers_hired": section["previous_execution"]["workers_hired"],
                    "hire_cost": section["previous_execution"]["hire_cost"],
                },
            })
        else:
            result.update({
                "boundary": section["boundary"],
                "day": section["day"],
                "hour": section["hour"],
            })
        return result

    return {
        "schema_version": row["schema_version"],
        "metadata": _metadata(row.get("metadata")),
        "day": row["day"],
        "start": start_or_end(start, is_start=True),
        "targets": None if targets is None else _denorm_targets(targets),
        "end": start_or_end(end, is_start=False),
    }


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    crop_names = {name.upper() for name in CROP_STEPS}
    board = state.get("board", ())
    for board_row in board if isinstance(board, (list, tuple)) else ():
        for tile in board_row if isinstance(board_row, (list, tuple)) else ():
            if not isinstance(tile, Mapping):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop in crop_names:
                crops[str(crop)] += 1
            if animal in ("GOOSE", "COW", "SHEEP"):
                animals[str(animal)] += 1
    unlocked = state.get("unlocked_quadrants", ())
    return {
        "unlocked_land": len(unlocked) if isinstance(unlocked, (list, tuple))
        else None,
        "crop_counts": dict(crops),
        "animal_counts": dict(animals),
        "money": state.get("money"),
    }


def _audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a small audit view, never a retained canonical board."""
    return {
        "start": {"self": _state_summary(record["start"]["self"])},
        "end": {"self": _state_summary(record["end"]["self"])},
    }


def _row_info(source: _SourceRow, record: Mapping[str, Any]) -> _RowInfo:
    metadata = record["metadata"]
    compact_metadata = _eval_metadata(metadata, int(record["day"]))
    return _RowInfo(
        source=source,
        metadata=compact_metadata,
        date=_data_date(record, metadata),
        score=_data_score(record, metadata),
        audit_record=_audit_record(record),
    )


def _iter_projected_batches(
    paths: Sequence[Path],
    *,
    read_batch_size: int,
):
    """Yield bounded projected batches in deterministic file/global order."""
    if read_batch_size <= 0:
        raise ValueError("read_batch_size must be positive")
    global_row = 0
    for path in paths:
        parquet = _open_projected(path)
        source_row = 0
        saw_rows = False
        try:
            batches = parquet.iter_batches(
                batch_size=read_batch_size, columns=list(PROJECTED_COLUMNS))
            for batch in batches:
                _require_schema_version(batch, path)
                saw_rows = saw_rows or bool(batch.num_rows)
                yield path, global_row, source_row, batch
                global_row += batch.num_rows
                source_row += batch.num_rows
            if not saw_rows:
                raise SchemaVersionError(
                    f"{path}: schema_version contains no rows; expected {SCHEMA_VERSION}")
        except SchemaVersionError:
            raise
        except (KeyError, ValueError, pa.ArrowException) as exc:
            raise ValueError(
                f"{path}: not a canonical Stage 2.5 Parquet file ({exc}); "
                f"expected projected columns {list(PROJECTED_COLUMNS)!r}"
            ) from exc


def _take_inputs(inputs: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    return {
        name: np.ascontiguousarray(np.asarray(value)[selected])
        for name, value in inputs.items()
    }


def _morning_crop_observation_arrays(
    records: Sequence[Mapping[str, Any]], days: Sequence[int],
) -> dict[str, np.ndarray]:
    """Materialize physical morning crop features from one canonical board scan."""
    if len(records) != len(days):
        raise ValueError("records and days must have equal lengths")
    if not records:
        return {
            "crop_capacity": np.empty((0, len(CROP_STEPS)), dtype=np.int16),
            "replaceable_today": np.empty((0, len(CROP_STEPS)), dtype=np.int16),
            "available_crop_slots": np.empty((0,), dtype=np.int16),
        }
    baselines = []
    replaceable = []
    available = []
    for record, day in zip(records, days):
        start_self = record["start"]["self"]
        step = int(day) * 24
        board = canonical_board(start_self["board"], int(day), step)
        context = physical_context_from_board(
            board, start_self["unlocked_quadrants"])
        baseline = context.observed_crop_counts
        baselines.append(baseline)
        replaceable.append(replaceable_today(board, int(day), step))
        available.append(available_crop_slots(context))
    return {
        "crop_capacity": np.asarray(baselines, dtype=np.int16),
        "replaceable_today": np.asarray(replaceable, dtype=np.int16),
        "available_crop_slots": np.asarray(available, dtype=np.int16),
    }


def _replaceable_today_arrays(
    records: Sequence[Mapping[str, Any]], days: Sequence[int],
) -> np.ndarray:
    """Backward-compatible focused seam for the lifecycle feature tests."""
    return _morning_crop_observation_arrays(records, days)["replaceable_today"]


def _label_actions(label: OutcomeProxyLabel) -> tuple[int, ...]:
    values = (label.land_class, *label.animal_classes, *label.crop_classes)
    if any(value is None for value in values):
        raise ValueError("complete Stage 2.5 label contains a missing class")
    return tuple(int(value) for value in values)


def _identity(info: _RowInfo) -> dict[str, Any]:
    metadata = info.metadata
    source = info.source
    row_id = f"{source.source_path}::row={source.source_row}"
    return {
        "row_id": row_id,
        "row_index": source.row_index,
        "source_path": source.source_path,
        "source_row": source.source_row,
        "episode_id": metadata.get("episode_id"),
        "seat": metadata.get("seat"),
        "day": info.day,
        "partition_date": metadata.get("partition_date"),
    }


def _selected(record: Mapping[str, Any], dates: set[str], min_score: float) -> tuple[bool, str | None]:
    # Share the Packet 1B selection semantics so the adapter can never filter a
    # row differently from the authoritative outcome-proxy builder.
    metadata = record["metadata"]
    if str(_data_date(record, metadata)) not in dates:
        return False, "date"
    score = _data_score(record, metadata)
    if score is None or float(score) < float(min_score):
        return False, "score"
    return True, None


class _CorpusAccumulator:
    """Streaming corpus diagnostics with no retained canonical rows."""

    def __init__(self, min_score: float) -> None:
        self.min_score = float(min_score)
        self.date_counts: Counter[str] = Counter()
        self.score_counts: Counter[str] = Counter()
        self.episode_ids: set[Any] = set()
        self.episode_seats: set[tuple[Any, Any]] = set()
        self.logical_rows: Counter[tuple[Any, Any, int]] = Counter()
        self.files: Counter[str] = Counter()
        self.schema_versions: Counter[str] = Counter()
        self.rows_read = 0

    def add(self, info: _RowInfo, schema_version: Any) -> None:
        date = info.date
        self.date_counts["<missing>" if date is None else str(date)] += 1
        if info.score is None:
            self.score_counts["missing"] += 1
        elif info.score < self.min_score:
            self.score_counts["below_min_score"] += 1
        else:
            self.score_counts["at_or_above_min_score"] += 1
        self.episode_ids.add(info.episode_id)
        self.episode_seats.add((info.episode_id, info.seat))
        self.logical_rows[(info.episode_id, info.seat, info.day)] += 1
        self.files[info.source.source_path] += 1
        self.schema_versions[str(schema_version)] += 1
        self.rows_read += 1

    def finish(self) -> dict[str, Any]:
        return {
            "files": dict(sorted(self.files.items())),
            "schema_versions": dict(sorted(self.schema_versions.items())),
            "rows_read": int(self.rows_read),
            "unique_episode_count": len(self.episode_ids),
            "unique_episode_seat_count": len(self.episode_seats),
            "duplicate_logical_row_count": sum(
                count - 1 for count in self.logical_rows.values() if count > 1),
            "date_counts": dict(sorted(self.date_counts.items())),
            "score_filter_counts": {
                **{key: int(self.score_counts.get(key, 0)) for key in
                   ("missing", "below_min_score", "at_or_above_min_score")},
                "minimum_score": self.min_score,
            },
        }


def _selected_info(info: _RowInfo, dates: set[str], min_score: float) -> tuple[bool, str | None]:
    if str(info.date) not in dates:
        return False, "date"
    if info.score is None or info.score < float(min_score):
        return False, "score"
    return True, None


def _diagnostics(
    build: OutcomeProxyBuild,
    labels: Sequence[OutcomeProxyLabel],
    partial: Sequence[OutcomeProxyLabel],
    selected_count: int,
    selection_reasons: Mapping[str, int],
) -> dict[str, Any]:
    all_labels = (*labels, *partial)
    valid = np.zeros((len(all_labels), len(ACTION_STEPS)), dtype=bool)
    for index, label in enumerate(all_labels):
        valid[index] = [step in label.valid_components for step in ACTION_STEPS]

    support = {
        step: {
            "valid": int(valid[:, index].sum()) if len(valid) else 0,
            "invalid": int(len(valid) - valid[:, index].sum()),
            "total": int(len(valid)),
        }
        for index, step in enumerate(ACTION_STEPS)
    }
    hold_change: dict[str, dict[str, int | float]] = {}
    for crop_index, crop in enumerate(CROP_STEPS):
        changes = [label.crop_deltas[crop_index]
                   for label in all_labels
                   if label.crop_deltas[crop_index] is not None]
        hold = sum(delta == 0 for delta in changes)
        change = len(changes) - hold
        hold_change[crop.upper()] = {
            "hold": int(hold), "change": int(change), "total": len(changes),
            "hold_fraction": float(hold / len(changes)) if changes else 0.0,
        }

    counters = asdict(build.counters)
    return {
        "counters": counters,
        "selection_reasons": dict(selection_reasons),
        "selected_rows": int(selected_count),
        "complete_rows": int(len(labels)),
        "partial_rows": int(len(partial)),
        "support_validity": support,
        "support_validity_array": valid,
        "hold_change_distributions": hold_change,
        "hold_change": hold_change,
    }


def _materialize_split(
    row_infos: Sequence[_RowInfo],
    inputs_all: Mapping[str, np.ndarray],
    build: OutcomeProxyBuild,
    *,
    dates: Sequence[str],
    min_score: float,
) -> dict[str, Any]:
    date_set = {str(value) for value in dates}
    complete = [label for label in build.rows
                if _selected_info(row_infos[label.row_index], date_set,
                                  min_score)[0]]
    partial = [label for label in build.partial_rows
               if _selected_info(row_infos[label.row_index], date_set,
                                 min_score)[0]]
    selected_indices = [info.source.row_index for info in row_infos
                        if _selected_info(info, date_set, min_score)[0]]
    complete_indices = [label.row_index for label in complete]
    identities = [_identity(row_infos[index]) for index in complete_indices]
    row_ids = [item["row_id"] for item in identities]
    actions = np.asarray([_label_actions(label) for label in complete], dtype=np.int16)
    if actions.size == 0:
        actions = np.empty((0, len(ACTION_STEPS)), dtype=np.int16)
    inputs = _take_inputs(inputs_all, complete_indices)
    # `crop_capacity` is the persisted legacy key for physical morning board
    # counts B_t. It is never the historical synthetic crop-goal ledger.
    meta = []
    for index, identity in zip(complete_indices, identities):
        item = dict(row_infos[index].metadata)
        item.update(identity)
        meta.append(item)

    selection_reasons = {"date": 0, "score": 0, "incomplete_ar_chain": 0,
                         "invalid_or_no_valid_component": 0}
    complete_set = set(complete_indices)
    partial_set = {label.row_index for label in partial}
    for index, info in enumerate(row_infos):
        selected, reason = _selected_info(info, date_set, min_score)
        if not selected:
            selection_reasons[reason or "date"] += 1
        elif index in partial_set:
            selection_reasons["incomplete_ar_chain"] += 1
        elif index not in complete_set:
            selection_reasons["invalid_or_no_valid_component"] += 1

    diagnostics = _diagnostics(
        build, complete, partial, len(selected_indices), selection_reasons)
    report = {
        "rows_read": len(row_infos),
        "rows_selected": len(selected_indices),
        "rows_excluded": len(row_infos) - len(selected_indices),
        "rows_complete": len(complete),
        "rows_partial": len(partial),
        "rows_not_trainable": len(selected_indices) - len(complete),
        "selection_excluded_rows": len(row_infos) - len(selected_indices),
        "min_score": float(min_score),
        "dates": list(dates),
        "exclusion_reasons": dict(selection_reasons),
    }
    target_arrays = {
        "action_classes": actions,
        "land_class": actions[:, 0],
        "animal_classes": actions[:, 1:4],
        "crop_classes": actions[:, 4:9],
    }
    logical_records = [row_infos[index].audit_record for index in complete_indices]
    return {
        "inputs": inputs,
        "targets": target_arrays,
        "actions": actions,
        "classes": actions,
        "labels": tuple(complete),
        "records": logical_records,
        "meta": meta,
        "row_ids": tuple(row_ids),
        "row_identities": identities,
        "partial_rows": tuple(partial),
        "partial_records": [row_infos[label.row_index].audit_record
                            for label in partial],
        "counters": build.counters,
        "diagnostics": diagnostics,
        "report": report,
    }


def _load_streamed(
    paths: Sequence[Path],
    *,
    selected_dates: Sequence[str],
    min_score: float,
    e_history_version: str,
    manager_start_day: int | None,
    read_batch_size: int,
) -> tuple[list[_RowInfo], dict[str, np.ndarray], OutcomeProxyBuild,
           dict[str, Any]]:
    """Scan canonical Parquet with bounded nested materialization.

    Only ``batch.to_pylist()`` is allowed to contain nested canonical objects.
    Each batch is converted to compact model arrays, labels, metadata, and
    audit summaries before its Python rows are released.
    """
    builder = OutcomeProxyBuilder(selected_dates, min_score)
    corpus_builder = _CorpusAccumulator(min_score)
    row_infos: list[_RowInfo] = []
    input_chunks: dict[str, list[np.ndarray]] = {}
    morning_baselines: list[Any] = []
    morning_replaceable: list[Any] = []
    morning_available: list[Any] = []
    for path, global_start, source_start, batch in _iter_projected_batches(
            paths, read_batch_size=read_batch_size):
        arrow_rows = batch.to_pylist()
        starts = [row["start"] for row in arrow_rows]
        days = [int(row["day"]) for row in arrow_rows]
        batch_inputs = _input_arrays_from_starts(
            starts, days, include_opponent=False)
        for name, value in batch_inputs.items():
            input_chunks.setdefault(name, []).append(value)
        for offset, arrow_row in enumerate(arrow_rows):
            logical = _logical_record(arrow_row)
            source = _SourceRow(
                global_start + offset, str(path), source_start + offset)
            info = _row_info(source, logical)
            row_infos.append(info)
            corpus_builder.add(info, logical["schema_version"])
            builder.consume(logical)
            day = int(arrow_row["day"])
            start_self = logical["start"]["self"]
            step = day * 24
            board = canonical_board(start_self["board"], day, step)
            context = physical_context_from_board(
                board, start_self["unlocked_quadrants"])
            morning_baselines.append(context.observed_crop_counts)
            morning_replaceable.append(replaceable_today(board, day, step))
            morning_available.append(available_crop_slots(context))
        del starts, days, batch_inputs, arrow_rows
        del logical, arrow_row, source, info, batch

    if input_chunks:
        inputs_all = {
            name: np.ascontiguousarray(np.concatenate(chunks, axis=0))
            for name, chunks in input_chunks.items()
        }
    else:
        inputs_all = _input_arrays_from_starts([], [], include_opponent=False)
    # `crop_capacity` is the persisted legacy key for physical morning board
    # counts B_t. It is never the historical synthetic crop-goal ledger.
    inputs_all["crop_capacity"] = np.asarray(
        morning_baselines, dtype=np.int16).reshape(len(row_infos), -1)
    inputs_all["replaceable_today"] = np.asarray(
        morning_replaceable, dtype=np.int16).reshape(len(row_infos), -1)
    inputs_all["available_crop_slots"] = np.asarray(
        morning_available, dtype=np.int16).reshape((len(row_infos),))
    inputs_all[ECONOMIC_CONTEXT_KEY] = derive_economic_context(
        [info.episode_id for info in row_infos],
        [info.seat for info in row_infos],
        [info.day for info in row_infos],
        inputs_all["scalars"][:, 0],
        inputs_all["unlocked"].sum(axis=1),
        e_history_version=e_history_version,
        manager_start_day=manager_start_day,
    )
    return row_infos, inputs_all, builder.finish(), corpus_builder.finish()


def load_dataset(
    paths: str | Path | Sequence[str | Path] | None = None,
    *,
    dates: Sequence[str] = DEFAULT_TRAIN_DATES,
    min_score: float = DEFAULT_MIN_SCORE,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    manager_start_day: int | None = None,
    read_batch_size: int = DEFAULT_READ_BATCH_SIZE,
    config: Stage25AdapterConfig | None = None,
) -> dict[str, Any]:
    """Load one complete Stage 2.5 split from projected canonical Parquet."""
    if config is not None:
        if paths is None:
            paths = config.paths
        if tuple(dates) == DEFAULT_TRAIN_DATES:
            dates = config.train_dates
        if min_score == DEFAULT_MIN_SCORE:
            min_score = config.min_score
        if e_history_version == E_HISTORY_CORRECTED_V1:
            e_history_version = config.e_history_version
        if manager_start_day is None:
            manager_start_day = config.manager_start_day
        if read_batch_size == DEFAULT_READ_BATCH_SIZE:
            read_batch_size = config.read_batch_size
    path_list = _paths(paths)
    row_infos, inputs_all, build, corpus = _load_streamed(
        path_list, selected_dates=dates, min_score=min_score,
        e_history_version=e_history_version,
        manager_start_day=manager_start_day, read_batch_size=read_batch_size)
    result = _materialize_split(
        row_infos, inputs_all, build,
        dates=dates, min_score=min_score)
    result["corpus"] = corpus
    return result


def load_train_val(
    paths: str | Path | Sequence[str | Path] | None = None,
    *,
    train_dates: Sequence[str] = DEFAULT_TRAIN_DATES,
    val_dates: Sequence[str] = DEFAULT_VAL_DATES,
    min_score: float = DEFAULT_MIN_SCORE,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    manager_start_day: int | None = None,
    read_batch_size: int = DEFAULT_READ_BATCH_SIZE,
    config: Stage25AdapterConfig | None = None,
) -> dict[str, Any]:
    """Load the date-held-out train/validation Stage 2.5 batches."""
    if config is not None:
        if paths is None:
            paths = config.paths
        if tuple(train_dates) == DEFAULT_TRAIN_DATES:
            train_dates = config.train_dates
        if tuple(val_dates) == DEFAULT_VAL_DATES:
            val_dates = config.val_dates
        if min_score == DEFAULT_MIN_SCORE:
            min_score = config.min_score
        if e_history_version == E_HISTORY_CORRECTED_V1:
            e_history_version = config.e_history_version
        if manager_start_day is None:
            manager_start_day = config.manager_start_day
        if read_batch_size == DEFAULT_READ_BATCH_SIZE:
            read_batch_size = config.read_batch_size
    overlap = set(train_dates) & set(val_dates)
    if overlap:
        raise ValueError(f"train/val date lists overlap: {sorted(overlap)}")
    path_list = _paths(paths)
    all_dates = tuple(train_dates) + tuple(val_dates)
    row_infos, inputs_all, build, corpus = _load_streamed(
        path_list, selected_dates=all_dates, min_score=min_score,
        e_history_version=e_history_version,
        manager_start_day=manager_start_day, read_batch_size=read_batch_size)
    train = _materialize_split(
        row_infos, inputs_all, build,
        dates=train_dates, min_score=min_score)
    val = _materialize_split(
        row_infos, inputs_all, build,
        dates=val_dates, min_score=min_score)
    return {
        "train": train,
        "val": val,
        "report": {
            "rows_read": len(row_infos),
            "rows_selected": train["report"]["rows_selected"] + val["report"]["rows_selected"],
            "train_rows": len(train["labels"]),
            "val_rows": len(val["labels"]),
            "train_dates": list(train_dates),
            "val_dates": list(val_dates),
            "min_score": float(min_score),
        },
        "corpus": corpus,
    }


load_stage25_dataset = load_dataset
load_stage25_train_val = load_train_val


__all__ = [
    "ACTION_STEPS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_READ_BATCH_SIZE",
    "DEFAULT_PARQUET_PATHS",
    "DEFAULT_TRAIN_DATES",
    "DEFAULT_VAL_DATES",
    "PROJECTED_COLUMNS",
    "SchemaVersionError",
    "Stage25AdapterConfig",
    "load_dataset",
    "load_stage25_dataset",
    "load_stage25_train_val",
    "load_train_val",
]
