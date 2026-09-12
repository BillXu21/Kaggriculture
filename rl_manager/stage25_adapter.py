"""Projected canonical-Parquet loader for the Stage 2.5 manager.

The loader deliberately reads only the state and outcome fields needed by the
Stage 2.5 contract.  In particular, it does not project or materialize the
rich ``events`` struct.  Arrow rows are converted back to the logical shape
expected by :func:`rl_manager.stage25_data.build_outcome_proxy_labels`, while
the shared BC adapter encodes the start state into compact arrays.
"""

from __future__ import annotations

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
    _date as _data_date,
    _score as _data_score,
    build_outcome_proxy_labels,
)

PROJECTED_COLUMNS = ("schema_version", "metadata", "day", "start", "targets", "end")
DEFAULT_TRAIN_DATES = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20")
DEFAULT_VAL_DATES = ("2026-08-21",)
DEFAULT_MIN_SCORE = 2950.0
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


@dataclass(frozen=True)
class _SourceRow:
    row_index: int
    source_path: str
    source_row: int


def _paths(value: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if value is None:
        return [Path(path) for path in DEFAULT_PARQUET_PATHS]
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        if candidate.is_dir():
            return [candidate / path.name for path in DEFAULT_PARQUET_PATHS]
        value = [value]
    result = [Path(path) for path in value]
    if not result:
        raise ValueError("at least one canonical Parquet path is required")
    return result


def _require_schema_version(table: pa.Table, path: Path) -> None:
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


def _read_projected(path: Path) -> pa.Table:
    try:
        table = pq.read_table(path, columns=list(PROJECTED_COLUMNS))
    except (KeyError, ValueError, pa.ArrowException) as exc:
        raise ValueError(
            f"{path}: not a canonical Stage 2.5 Parquet file ({exc}); "
            f"expected projected columns {list(PROJECTED_COLUMNS)!r}"
        ) from exc
    _require_schema_version(table, path)
    return table


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


def _read_all(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]], list[int], list[_SourceRow]]:
    records: list[dict[str, Any]] = []
    starts: list[Mapping[str, Any]] = []
    days: list[int] = []
    source_rows: list[_SourceRow] = []
    for path in paths:
        table = _read_projected(path)
        arrow_rows = table.to_pylist()
        for source_row, row in enumerate(arrow_rows):
            records.append(_logical_record(row))
            starts.append(row["start"])
            days.append(int(row["day"]))
            source_rows.append(_SourceRow(len(records) - 1, str(path), source_row))
    return records, starts, days, source_rows


def _take_inputs(inputs: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    return {
        name: np.ascontiguousarray(np.asarray(value)[selected])
        for name, value in inputs.items()
    }


def _label_actions(label: OutcomeProxyLabel) -> tuple[int, ...]:
    values = (label.land_class, *label.animal_classes, *label.crop_classes)
    if any(value is None for value in values):
        raise ValueError("complete Stage 2.5 label contains a missing class")
    return tuple(int(value) for value in values)


def _identity(source: _SourceRow, record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    row_id = f"{source.source_path}::row={source.source_row}"
    return {
        "row_id": row_id,
        "row_index": source.row_index,
        "source_path": source.source_path,
        "source_row": source.source_row,
        "episode_id": metadata.get("episode_id"),
        "seat": metadata.get("seat"),
        "day": record["day"],
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
    records: Sequence[dict[str, Any]],
    source_rows: Sequence[_SourceRow],
    inputs_all: Mapping[str, np.ndarray],
    build: OutcomeProxyBuild,
    *,
    dates: Sequence[str],
    min_score: float,
) -> dict[str, Any]:
    date_set = {str(value) for value in dates}
    complete = [label for label in build.rows
                if label.row_index < len(records)
                and _selected(records[label.row_index], date_set, min_score)[0]]
    partial = [label for label in build.partial_rows
               if label.row_index < len(records)
               and _selected(records[label.row_index], date_set, min_score)[0]]
    selected_indices = [index for index, record in enumerate(records)
                        if _selected(record, date_set, min_score)[0]]
    complete_indices = [label.row_index for label in complete]
    identities = [_identity(source_rows[index], records[index])
                  for index in complete_indices]
    row_ids = [item["row_id"] for item in identities]
    actions = np.asarray([_label_actions(label) for label in complete], dtype=np.int16)
    if actions.size == 0:
        actions = np.empty((0, len(ACTION_STEPS)), dtype=np.int16)
    crop_capacity = np.asarray(
        [label.provenance.prior_crop_goals for label in complete], dtype=np.int16)
    if crop_capacity.size == 0:
        crop_capacity = np.empty((0, len(CROP_STEPS)), dtype=np.int16)

    inputs = _take_inputs(inputs_all, complete_indices)
    inputs["crop_capacity"] = crop_capacity
    meta = []
    for index, identity in zip(complete_indices, identities):
        item = _eval_metadata(records[index]["metadata"], records[index]["day"])
        item.update(identity)
        meta.append(item)

    selection_reasons = {"date": 0, "score": 0, "incomplete_ar_chain": 0,
                         "invalid_or_no_valid_component": 0}
    complete_set = set(complete_indices)
    partial_set = {label.row_index for label in partial}
    for index, record in enumerate(records):
        selected, reason = _selected(record, date_set, min_score)
        if not selected:
            selection_reasons[reason or "date"] += 1
        elif index in partial_set:
            selection_reasons["incomplete_ar_chain"] += 1
        elif index not in complete_set:
            selection_reasons["invalid_or_no_valid_component"] += 1

    diagnostics = _diagnostics(
        build, complete, partial, len(selected_indices), selection_reasons)
    report = {
        "rows_read": len(records),
        "rows_selected": len(selected_indices),
        "rows_excluded": len(records) - len(selected_indices),
        "rows_complete": len(complete),
        "rows_partial": len(partial),
        "rows_not_trainable": len(selected_indices) - len(complete),
        "selection_excluded_rows": len(records) - len(selected_indices),
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
    logical_records = [records[index] for index in complete_indices]
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
        "partial_records": [records[label.row_index] for label in partial],
        "counters": build.counters,
        "diagnostics": diagnostics,
        "report": report,
    }


def load_dataset(
    paths: str | Path | Sequence[str | Path] | None = None,
    *,
    dates: Sequence[str] = DEFAULT_TRAIN_DATES,
    min_score: float = DEFAULT_MIN_SCORE,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    manager_start_day: int | None = None,
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
    path_list = _paths(paths)
    records, starts, days, source_rows = _read_all(path_list)
    metadata = [record["metadata"] for record in records]
    inputs_all = _input_arrays_from_starts(starts, days, include_opponent=False)
    inputs_all[ECONOMIC_CONTEXT_KEY] = derive_economic_context(
        [item.get("episode_id") for item in metadata],
        [item.get("seat") for item in metadata],
        days,
        inputs_all["scalars"][:, 0],
        inputs_all["unlocked"].sum(axis=1),
        e_history_version=e_history_version,
        manager_start_day=manager_start_day,
    )
    build = build_outcome_proxy_labels(
        records, selected_dates=dates, min_score=min_score)
    return _materialize_split(
        records, source_rows, inputs_all, build,
        dates=dates, min_score=min_score)


def load_train_val(
    paths: str | Path | Sequence[str | Path] | None = None,
    *,
    train_dates: Sequence[str] = DEFAULT_TRAIN_DATES,
    val_dates: Sequence[str] = DEFAULT_VAL_DATES,
    min_score: float = DEFAULT_MIN_SCORE,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    manager_start_day: int | None = None,
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
    overlap = set(train_dates) & set(val_dates)
    if overlap:
        raise ValueError(f"train/val date lists overlap: {sorted(overlap)}")
    path_list = _paths(paths)
    records, starts, days, source_rows = _read_all(path_list)
    metadata = [record["metadata"] for record in records]
    inputs_all = _input_arrays_from_starts(starts, days, include_opponent=False)
    inputs_all[ECONOMIC_CONTEXT_KEY] = derive_economic_context(
        [item.get("episode_id") for item in metadata],
        [item.get("seat") for item in metadata],
        days,
        inputs_all["scalars"][:, 0],
        inputs_all["unlocked"].sum(axis=1),
        e_history_version=e_history_version,
        manager_start_day=manager_start_day,
    )
    all_dates = tuple(train_dates) + tuple(val_dates)
    build = build_outcome_proxy_labels(
        records, selected_dates=all_dates, min_score=min_score)
    train = _materialize_split(
        records, source_rows, inputs_all, build,
        dates=train_dates, min_score=min_score)
    val = _materialize_split(
        records, source_rows, inputs_all, build,
        dates=val_dates, min_score=min_score)
    return {
        "train": train,
        "val": val,
        "report": {
            "rows_read": len(records),
            "rows_selected": train["report"]["rows_selected"] + val["report"]["rows_selected"],
            "train_rows": len(train["labels"]),
            "val_rows": len(val["labels"]),
            "train_dates": list(train_dates),
            "val_dates": list(val_dates),
            "min_score": float(min_score),
        },
    }


load_stage25_dataset = load_dataset
load_stage25_train_val = load_train_val


__all__ = [
    "ACTION_STEPS",
    "DEFAULT_MIN_SCORE",
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
