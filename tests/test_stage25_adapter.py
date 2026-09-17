"""Focused real-Parquet tests for the projected Stage 2.5 adapter."""

from __future__ import annotations

import copy
from dataclasses import asdict
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


def test_streamed_adapter_matches_in_memory_reference_semantics(tmp_path):
    from bc_manager.adapter import _input_arrays_from_starts
    from rl_manager.stage25_data import build_outcome_proxy_labels

    records = [
        _record(0, "2026-08-17", end_wheat=1),
        _record(1, "2026-08-18", start_wheat=1, end_wheat=2),
        _record(2, "2026-08-19", start_wheat=2, end_wheat=1, score=1000),
    ]
    path = tmp_path / "canonical.parquet"
    _write(path, records)
    dates = ("2026-08-17", "2026-08-18")
    result = load_dataset(path, dates=dates, read_batch_size=1)
    expected = build_outcome_proxy_labels(
        records, selected_dates=dates, min_score=2950)

    assert result["labels"] == expected.rows
    assert result["partial_rows"] == expected.partial_rows
    assert result["diagnostics"]["counters"] == {
        key: value for key, value in asdict(expected.counters).items()
    }
    physical_starts = records_to_table(records).column("start").to_pylist()
    expected_inputs = _input_arrays_from_starts(
        physical_starts,
        [record["day"] for record in records],
        include_opponent=False)
    for name, value in expected_inputs.items():
        np.testing.assert_array_equal(result["inputs"][name][[0, 1]], value[[0, 1]])
    assert result["row_ids"] == tuple(
        f"{path}::row={index}" for index in (0, 1))


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


def test_train_val_split_keeps_history_through_row_excluded_from_both_splits(tmp_path):
    path = tmp_path / "canonical.parquet"
    _write(path, [
        _record(0, "2026-08-17", end_wheat=1),
        _record(1, "2026-08-18", score=1000, end_wheat=2),
        _record(2, "2026-08-19", start_wheat=0, end_wheat=3),
    ])

    result = load_train_val(
        path, train_dates=("2026-08-17",), val_dates=("2026-08-19",))

    assert [label.row_index for label in result["train"]["labels"]] == [0]
    assert [label.row_index for label in result["val"]["labels"]] == [2]
    assert result["val"]["labels"][0].provenance.prior_crop_goals == (
        2, 0, 0, 0, 0)
    assert result["val"]["labels"][0].provenance.prior_row_index == 1


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
    original = pq.ParquetFile.iter_batches

    def fail_full_table(*args, **kwargs):
        raise AssertionError("canonical loader must not call pq.read_table")

    def spy(self, *args, **kwargs):
        seen.append(list(kwargs["columns"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr("rl_manager.stage25_adapter.pq.ParquetFile.iter_batches", spy)
    monkeypatch.setattr("rl_manager.stage25_adapter.pq.read_table", fail_full_table)
    result = load_train_val(
        [path], train_dates=("custom-date",), val_dates=(), min_score=2950)

    assert len(result["train"]["labels"]) == 1
    assert seen == [list(PROJECTED_COLUMNS)]
    assert "events" not in seen[0]


def test_canonical_batches_are_invariant_at_tiny_boundaries(tmp_path):
    path = tmp_path / "canonical.parquet"
    _write(path, [
        _record(day, f"2026-08-{17 + day}", start_wheat=day,
                end_wheat=day + 1)
        for day in range(8)
    ])

    baseline = load_dataset(
        path, dates=tuple(f"2026-08-{17 + day}" for day in range(8)),
        read_batch_size=64)
    for batch_size in (1, 2, 7, 64):
        got = load_dataset(
            path, dates=tuple(f"2026-08-{17 + day}" for day in range(8)),
            read_batch_size=batch_size)
        assert got["row_ids"] == baseline["row_ids"]
        np.testing.assert_array_equal(got["actions"], baseline["actions"])
        for name in baseline["inputs"]:
            np.testing.assert_array_equal(
                got["inputs"][name], baseline["inputs"][name])
        assert [label.provenance for label in got["labels"]] == [
            label.provenance for label in baseline["labels"]]
        got_diagnostics = dict(got["diagnostics"])
        baseline_diagnostics = dict(baseline["diagnostics"])
        np.testing.assert_array_equal(
            got_diagnostics.pop("support_validity_array"),
            baseline_diagnostics.pop("support_validity_array"))
        assert got_diagnostics == baseline_diagnostics


def test_history_crosses_file_boundary_without_reset(tmp_path):
    left = tmp_path / "a.parquet"
    right = tmp_path / "b.parquet"
    _write(left, [_record(0, "2026-08-17", end_wheat=2)])
    _write(right, [_record(1, "2026-08-18", start_wheat=0, end_wheat=3)])

    result = load_dataset(
        [left, right], dates=("2026-08-17", "2026-08-18"), read_batch_size=1)

    assert [label.provenance.prior_crop_goals for label in result["labels"]] == [
        (0, 0, 0, 0, 0), (2, 0, 0, 0, 0)]
    assert result["row_identities"][1]["source_row"] == 0
    assert result["row_identities"][1]["row_index"] == 1


def test_bounded_reader_consumes_multiple_batches(tmp_path, monkeypatch):
    path = tmp_path / "canonical.parquet"
    _write(path, [_record(day, f"2026-08-{17 + day}") for day in range(5)])
    calls: list[int] = []
    original = pq.ParquetFile.iter_batches

    def spy(self, *args, **kwargs):
        calls.append(kwargs["batch_size"])
        batches = original(self, *args, **kwargs)

        def counted():
            for batch in batches:
                calls.append(-1)
                yield batch
        return counted()

    monkeypatch.setattr("rl_manager.stage25_adapter.pq.ParquetFile.iter_batches", spy)
    load_dataset(path, dates=tuple(f"2026-08-{17 + day}" for day in range(5)),
                 read_batch_size=2)
    assert calls == [2, -1, -1, -1]


def test_audit_invalid_crop_component_uses_authoritative_crop_goals(tmp_path):
    from tools.audit_stage25_canonical import _examples

    path = tmp_path / "canonical.parquet"
    record = _record(0, "2026-08-17", end_wheat=1)
    record["targets"]["crop_composition_end"]["WHEAT"] = 999
    _write(path, [record])
    result = load_dataset(path, dates=("2026-08-17",))

    examples = _examples(result, {
        "labels": (), "records": (), "partial_rows": result["partial_rows"],
        "partial_records": result["partial_records"],
    })
    assert examples
    assert "wheat:prefix_or_physical_capacity_support" in \
        examples[0]["physical_support_reasons"]


def test_canonical_bounded_arrays_complete_one_bc_checkpoint_roundtrip(tmp_path):
    import jax

    from rl_manager.stage25_bc import (
        Stage25BCConfig,
        init_opt_state,
        load_checkpoint,
        loss_and_metrics,
        make_fixed_batch,
        save_checkpoint,
        train_step,
    )
    from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params

    path = tmp_path / "canonical.parquet"
    _write(path, [_record(0, "2026-08-17", end_wheat=1)])
    loaded = load_dataset(path, dates=("2026-08-17",))
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=1)
    params = init_stage25_params(config.model, seed=19)
    opt_state = init_opt_state(params, config)
    rng = jax.random.PRNGKey(19)
    batch = make_fixed_batch(loaded["inputs"], loaded["actions"], 1,
                             row_ids=loaded["row_ids"])
    before = loss_and_metrics(params, batch, config)
    params, opt_state, rng, _ = train_step(params, opt_state, rng, batch, config)
    after = loss_and_metrics(params, batch, config)
    checkpoint = tmp_path / "stage25-bc.npz"
    save_checkpoint(checkpoint, params, opt_state, rng, config=config, step=1)
    restored = load_checkpoint(checkpoint, config=config)
    replayed = loss_and_metrics(restored[0], batch, config)

    assert np.isfinite(float(before["loss"]))
    assert np.isfinite(float(after["loss"]))
    assert np.isfinite(float(replayed["loss"]))
    np.testing.assert_allclose(after["loss"], replayed["loss"])


def test_directory_discovers_arbitrary_nested_parquet_names(tmp_path):
    corpus = tmp_path / "canonical-corpus"
    nested = corpus / "partitions"
    nested.mkdir(parents=True)
    _write(nested / "day-a.parquet", [_record(0, "2026-08-17", end_wheat=1)])
    _write(nested / "day-b.parquet", [
        _record(1, "2026-08-18", start_wheat=0, end_wheat=1)])

    result = load_dataset(
        corpus, dates=("2026-08-17", "2026-08-18"), min_score=2950)

    assert result["actions"].shape == (2, 9)
    assert result["corpus"]["rows_read"] == 2
    assert result["corpus"]["date_counts"] == {
        "2026-08-17": 1, "2026-08-18": 1}
    assert len(result["corpus"]["files"]) == 2


def test_adapter_selection_shares_builder_score_semantics():
    from rl_manager.stage25_adapter import _selected

    # The Packet 1B builder prefers a present ``score`` over ``min_score``; the
    # adapter must reuse that predicate instead of choosing its own precedence.
    high = {"metadata": {"partition_date": "2026-08-17", "score": 3000.0,
                         "min_score": 1000.0}}
    assert _selected(high, {"2026-08-17"}, 2950.0) == (True, None)
    low = {"metadata": {"partition_date": "2026-08-17", "score": 1000.0,
                        "min_score": 3000.0}}
    assert _selected(low, {"2026-08-17"}, 2950.0) == (False, "score")
    wrong_date = {"metadata": {"partition_date": "2026-08-16",
                               "min_score": 3000.0}}
    assert _selected(wrong_date, {"2026-08-17"}, 2950.0) == (False, "date")


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
