"""Focused contract tests for the compact schema-v3 BC data layer."""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bc_manager.adapter import (
    SchemaVersionError,
    aggregate_sells,
    build_targets,
    load_dataset,
    load_train_val,
    table_to_arrays,
)
from bc_manager.baseline import DayBaseline
from bc_manager.constants import (
    CROP_UNKNOWN_ID,
    PRODUCT_ORDER,
)
from bc_manager.metrics import group_metrics, nonzero_recall, sell_metrics
from replay_daily.constants import SCHEMA_VERSION
from replay_daily.extractor import extract_replay
from replay_daily.storage import RECORD_SCHEMA, records_to_table, write_parquet


# The small storage fixture is intentionally widened here: the BC adapter's
# stable contract is the engine's 10x10 board, not the 2x2 storage fixture.
sys.path.insert(0, str(Path(__file__).parent))
from test_replay_daily import make_replay  # noqa: E402
from test_replay_daily_storage import rich_specs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[dict]:
    records = extract_replay(make_replay(rich_specs()))
    dates = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-21")
    for i, record in enumerate(records):
        record["metadata"]["partition_date"] = dates[i]
        record["metadata"]["min_score"] = 2950.0
        for section in (record["start"], record["end"]):
            for state_name in ("self", "opponent_public"):
                board = section[state_name]["board"]
                section[state_name]["board"] = [
                    list(row) + [None] * (10 - len(row)) for row in board
                ] + [[None] * 10 for _ in range(10 - len(board))]
    return records


def test_v3_arrow_acceptance_shapes_targets_and_metadata_separation(tmp_path):
    path = tmp_path / "v3.parquet"
    write_parquet(_records(), path)
    data = load_dataset(path, dates=["2026-08-17"], min_score=2950)

    assert len(data["meta"]) == 1
    assert data["inputs"]["board_kind"].shape == (1, 100)
    assert data["inputs"]["board_numeric"].shape == (1, 100, 11)
    assert data["inputs"]["board_numeric"].dtype == np.float32
    assert data["inputs"]["board_mask"].shape == (1, 100, 4)
    assert data["targets"]["crop_target"].shape == (1, 5)
    assert data["targets"]["animal_target"].shape == (1, 3)
    assert data["targets"]["land_count"].shape == (1,)
    assert data["targets"]["fertilizer_target"].shape == (1, 5)
    assert data["targets"]["care_target"].shape == (1, 3)
    assert data["targets"]["sell_presence"].shape == (1, 9, 6)
    assert data["targets"]["sell_quantity_bounded"].dtype == np.int32
    assert data["targets"]["sell_quantity_log1p"].dtype == np.float32
    assert not any("score" in key or "bank" in key or "path" in key
                   for key in data["inputs"])
    assert {"min_score", "final_bank_self", "source_path"} <= set(data["meta"][0])


def test_v1_v2_and_mixed_arrow_rejected(tmp_path):
    rows = records_to_table(_records()).to_pylist()
    cases = (
        ("v1", [1] * len(rows)),
        ("v2", [2] * len(rows)),
        ("mixed-v1-v2", [1] + [2] * (len(rows) - 1)),
        ("mixed-v2-current", [2] + [SCHEMA_VERSION] * (len(rows) - 1)),
    )
    for name, versions in cases:
        bad = copy.deepcopy(rows)
        for row, version in zip(bad, versions):
            row["schema_version"] = version
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(bad, schema=RECORD_SCHEMA), path)
        with pytest.raises(SchemaVersionError, match="schema_version"):
            load_dataset(path, dates=["2026-08-17"])


def test_equal_cutoff_and_date_split_are_deterministic(tmp_path):
    records = _records()
    records[1]["metadata"]["min_score"] = 2949.0
    path = tmp_path / "split.parquet"
    write_parquet(records, path)
    split = load_train_val(
        path,
        train_dates=("2026-08-17", "2026-08-18", "2026-08-19"),
        val_dates=("2026-08-21",),
        min_score=2950,
    )
    assert len(split["train"]["meta"]) == 2
    assert len(split["val"]["meta"]) == 1
    assert split["report"]["rows_excluded"] == 1
    assert split["report"]["min_score"] == 2950
    assert {m["partition_date"] for m in split["train"]["meta"]} <= {
        "2026-08-17", "2026-08-18", "2026-08-19"
    }


def test_nullable_lifecycle_presence_and_unknown_crop_are_explicit():
    record = _records()[0]
    tile = record["end"]["self"]["board"][0][1]
    assert tile["kind"] == "PLANT"
    tile["crop"] = "FUTURE_CROP"
    record["start"]["self"]["board"][0][1] = copy.deepcopy(tile)
    table = records_to_table([record])
    inputs, _, _ = table_to_arrays(table)
    # The fixture's dry CARROT has a present derived timing key with null
    # value; the adapter preserves that fact as NaN plus derived presence.
    timing = inputs["board_numeric"][0, 1, -2]
    assert math.isnan(float(timing))
    assert inputs["board_mask"][0, 1, 3] == 1
    assert inputs["board_crop"][0, 1] == CROP_UNKNOWN_ID


def test_sell_transform_caps_each_event_bins_six_windows_and_repeats():
    sells = []
    for index, hour in enumerate((0, 4, 8, 12, 16, 20)):
        sells.append({"product": PRODUCT_ORDER[index], "quantity": 999,
                      "hour": hour})
    sells.extend([
        {"product": "WHEAT", "quantity": 1_000_000, "hour": 0},
        {"product": "WHEAT", "quantity": -12, "hour": 8},
        {"product": "WHEAT", "quantity": 2, "hour": 0},
    ])
    aggregate = aggregate_sells(sells)
    assert aggregate.shape == (9, 6)
    assert aggregate[0, 0] == 202  # 100 + 100 + 2; repeated events add
    assert aggregate[0, 2] == 0
    assert np.all(aggregate[range(1, 6), range(1, 6)] == 100)


def test_missing_care_is_rejected_instead_of_fabricated():
    target = {
        "crop_composition_end": {}, "animal_counts_end": {},
        "unlocked_quadrants_end": ["NW"], "fertilizer_by_crop": {},
        "care_by_animal": None,
    }
    with pytest.raises(ValueError, match="care_by_animal"):
        build_targets([target], [[]])


def _target_rows(n: int) -> dict[str, np.ndarray]:
    return {
        "crop_target": np.zeros((n, 5), dtype=np.int32),
        "animal_target": np.zeros((n, 3), dtype=np.int32),
        "land_count": np.ones(n, dtype=np.int32),
        "fertilizer_target": np.zeros((n, 5), dtype=np.int32),
        "care_target": np.zeros((n, 3), dtype=np.int32),
        "sell_presence": np.zeros((n, 9, 6), dtype=bool),
        "sell_quantity_bounded": np.zeros((n, 9, 6), dtype=np.int32),
    }


def test_sparse_metrics_zero_denominators_and_nonzero_recall():
    assert nonzero_recall(np.zeros((2, 3)), np.ones((2, 3))) == 0.0
    assert nonzero_recall(np.array([1, 2]), np.array([9, 0])) == 0.5
    metrics = group_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
    assert metrics["mae"] == 0.0
    assert metrics["true_nonzero_rate"] == 0.0
    assert metrics["nonzero_recall"] == 0.0
    assert sell_metrics(np.zeros((1, 9, 6)), np.zeros((1, 9, 6)))[
        "positive_cell_mae"] == 0.0


def test_day_baseline_fits_train_only_and_falls_back_for_unseen_days():
    train = _target_rows(3)
    train["crop_target"][:, 0] = [3, 3, 8]
    train["land_count"][:] = [1, 1, 2]
    train["sell_presence"][0, 0, 0] = True
    train["sell_quantity_bounded"][0, 0, 0] = 10
    train["sell_presence"][1, 0, 0] = True
    train["sell_quantity_bounded"][1, 0, 0] = 20
    train["sell_presence"][2, 1, 1] = True
    train["sell_quantity_bounded"][2, 1, 1] = 30

    baseline = DayBaseline().fit([4, 4, 5], train)
    validation = _target_rows(2)
    validation["crop_target"][:, 0] = 999  # sentinel never passed to fit
    prediction = baseline.predict([4, 5, 99])
    assert prediction["crop_target"][:, 0].tolist() == [3, 8, 3]
    assert prediction["land_count"].tolist() == [1, 2, 1]
    assert prediction["sell_quantity_bounded"].dtype == np.int32
    assert prediction["sell_quantity_log1p"].dtype == np.float32
    assert validation["crop_target"][0, 0] == 999
