"""Focused real-Parquet tests for the projected Stage 2.5 adapter."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from replay_daily.constants import SCHEMA_VERSION
from replay_daily.storage import records_to_table, write_parquet
from rl_manager.stage25_adapter import (
    PROJECTED_COLUMNS,
    SchemaVersionError,
    load_dataset,
    load_train_val,
)


def _board(*, wheat: int = 0, goose: bool = False) -> list[list[object]]:
    board: list[list[object]] = [
        [None for _ in range(10)] for _ in range(10)
    ]
    for y in range(10):
        for x in range(10):
            if y >= 5 or x >= 5:
                board[y][x] = "LOCKED"
    if wheat:
        cells = [(y, x) for y in range(5) for x in range(5)]
        for y, x in cells[:wheat]:
            board[y][x] = {"kind": "PLANT", "crop": "WHEAT"}
    elif goose:
        board[0][0] = {"kind": "COOP", "animal": "GOOSE"}
    return board


def _state(board: list[list[object]], money: float = 3000.0) -> dict:
    return {
        "money": money,
        "board": board,
        "unlocked_quadrants": ["NW"],
        "farmer": [0, 0],
        "hands": [],
        "hires_today": 0,
        "shed": {},
        "seeds": {},
        "inventories": [{}],
    }


def _public(board: list[list[object]]) -> dict:
    return {
        "money": 3000.0,
        "board": board,
        "unlocked_quadrants": ["NW"],
        "farmer": [0, 0],
        "hands": [],
        "hires_today": 0,
    }


def _events() -> dict:
    return {
        "plants": {},
        "digs": {"total": 0, "replaced": {}},
        "fertilizer_applications": {"by_crop": {}, "entries": []},
        "harvests": {"by_item": {}, "entries": []},
        "care": {"by_animal": {}, "entries": []},
        "buys": {"seeds": {}, "products": {}, "animals": {}},
        "land_purchases": [],
        "hires": {"submitted": 0, "realized": {"workers_hired": 0, "hire_cost": 0}},
        "sells": [],
        "market_events_ordered": [],
        "worker_ops_other": {},
    }


def _record(
    day: int,
    date: str,
    *,
    score: float = 3000.0,
    start_wheat: int = 0,
    end_wheat: int = 0,
    start_goose: bool = False,
    end_goose: bool = False,
    money: float = 3000.0,
    episode_id: int = 7,
    seat: int = 0,
) -> dict:
    metadata = {
        "episode_id": episode_id,
        "source_dataset": "test", "partition_date": date,
        "source_path": f"source-{date}.json", "seat": seat,
        "player": "p0", "opponent": "p1", "seed": 1,
        "module_version": "test", "avg_score": score, "min_score": score,
        "max_score": score, "sum_score": score, "final_rewards": [score],
        "final_bank_self": score, "final_bank_opponent": score,
    }
    start_board = _board(wheat=start_wheat, goose=start_goose)
    end_board = _board(wheat=end_wheat, goose=end_goose)
    start = {
        "day": day, "hour": 0, "self": _state(start_board, money),
        "opponent_public": _public(_board()),
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": [], "shop_counts": {}},
        "previous_execution": {"workers_hired": 0, "hire_cost": 0},
    }
    end = {
        "boundary": f"d{day}", "day": day + 1, "hour": 0,
        "self": _state(end_board, money),
        "opponent_public": _public(_board()),
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": [], "shop_counts": {}},
    }
    return {
        "schema_version": SCHEMA_VERSION, "metadata": metadata, "day": day,
        "start": start, "events": _events(),
        "targets": {
            "crop_composition_end": {"WHEAT": end_wheat},
            "animal_counts_end": {"GOOSE": int(end_goose)},
            "unlocked_quadrants_end": ["NW"],
            "land_expansion": {"expanded": False, "new_quadrants": []},
            "fertilizer_by_crop": {}, "care_by_animal": {},
            "sell_quantity": {},
        },
        "end": end,
    }


def _write(path: Path, records: list[dict]) -> None:
    write_parquet(records, path)


def test_real_projected_parquet_returns_fixed_bc_arrays_and_diagnostics(tmp_path):
    path = tmp_path / "canonical-2026-08-17.parquet"
    _write(path, [_record(0, "2026-08-17", end_wheat=1)])

    result = load_dataset(path, dates=("2026-08-17",))

    assert result["actions"].shape == (1, 9)
    assert result["actions"].dtype == np.int16
    assert result["inputs"]["crop_capacity"].tolist() == [[0, 0, 0, 0, 0]]
    assert result["actions"][0].tolist() == [0, 0, 0, 0, 101, 100, 100, 100, 100]
    assert result["row_identities"][0]["source_row"] == 0
    assert result["diagnostics"]["support_validity"]["wheat"]["valid"] == 1
    assert result["diagnostics"]["hold_change_distributions"]["WHEAT"]["change"] == 1


def test_history_is_built_before_date_and_score_filtering(tmp_path):
    path = tmp_path / "canonical.parquet"
    _write(path, [
        _record(0, "2026-08-16", score=1000, end_wheat=1),
        _record(1, "2026-08-17", start_wheat=0, end_wheat=2),
    ])

    result = load_dataset(path, dates=("2026-08-17",), min_score=2950)

    assert [label.row_index for label in result["labels"]] == [1]
    assert result["inputs"]["crop_capacity"].tolist() == [[1, 0, 0, 0, 0]]
    assert result["labels"][0].provenance.prior_source == "previous_synthetic_desired_end_goal"


def test_current_end_never_enters_input_or_corrected_e_context(tmp_path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    first = _record(0, "2026-08-17", end_wheat=0, money=3000)
    second = _record(1, "2026-08-18", start_wheat=0, end_wheat=0, money=3200)
    changed = copy.deepcopy(second)
    changed["end"] = copy.deepcopy(first["end"])
    changed["end"]["self"]["board"] = _board(wheat=5)
    changed["targets"]["crop_composition_end"]["WHEAT"] = 5
    _write(left, [first, second])
    _write(right, [first, changed])

    a = load_dataset(left, dates=("2026-08-17", "2026-08-18"))
    b = load_dataset(right, dates=("2026-08-17", "2026-08-18"))

    assert "end" not in a["inputs"]
    np.testing.assert_array_equal(a["inputs"]["economic_context"], b["inputs"]["economic_context"])
    np.testing.assert_array_equal(a["inputs"]["board_kind"], b["inputs"]["board_kind"])


def test_incomplete_animal_row_is_partial_diagnostic_not_training_row(tmp_path):
    path = tmp_path / "canonical.parquet"
    _write(path, [_record(0, "2026-08-17", start_goose=True, end_goose=False)])

    result = load_dataset(path, dates=("2026-08-17",))

    assert result["actions"].shape == (0, 9)
    assert [label.row_index for label in result["partial_rows"]] == [0]
    assert result["report"]["rows_partial"] == 1
    assert result["diagnostics"]["counters"]["incomplete_ar_chain_rows"] == 1


def test_configurable_paths_and_exact_projection(tmp_path, monkeypatch):
    path = tmp_path / "custom.parquet"
    _write(path, [_record(0, "custom-date")])
    seen: list[list[str]] = []
    original = pq.read_table

    def spy(*args, **kwargs):
        seen.append(list(kwargs["columns"]))
        return original(*args, **kwargs)

    monkeypatch.setattr("rl_manager.stage25_adapter.pq.read_table", spy)
    result = load_train_val(
        [path], train_dates=("custom-date",), val_dates=(), min_score=2950)

    assert len(result["train"]["labels"]) == 1
    assert seen == [list(PROJECTED_COLUMNS)]
    assert "events" not in seen[0]


def test_mixed_schema_version_is_rejected(tmp_path):
    path = tmp_path / "mixed.parquet"
    table = records_to_table([_record(0, "2026-08-17")])
    version_index = table.schema.get_field_index("schema_version")
    mixed = table.set_column(version_index, "schema_version", pa.array([2]))
    pq.write_table(mixed, path)

    try:
        load_dataset(path, dates=("2026-08-17",))
    except SchemaVersionError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("mixed schema version was accepted")
