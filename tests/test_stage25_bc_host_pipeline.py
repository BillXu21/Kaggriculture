"""Behavior and lifecycle tests for the Stage 2.5 host batch pipeline."""

from __future__ import annotations

import numpy as np
import pytest

import rl_manager.stage25_bc as bc
from rl_manager.stage25_bc_cli import build_parser
from rl_manager.stage25_bc import iter_fixed_batches


def _inputs(rows: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)
    return {
        "board_kind": rng.integers(0, 3, (rows, 100), dtype=np.int16),
        "board_crop": rng.integers(0, 4, (rows, 100), dtype=np.int8),
        "board_animal": rng.integers(0, 3, (rows, 100), dtype=np.int8),
        "board_numeric": rng.normal(size=(rows, 100, 11)).astype(np.float32),
        "board_bool": rng.integers(0, 2, (rows, 100, 8), dtype=np.uint8).astype(bool),
        "board_mask": rng.integers(0, 4, (rows, 100, 4), dtype=np.uint8),
        "scalars": rng.normal(size=(rows, 4)).astype(np.float32),
        "shed_counts": rng.integers(0, 20, (rows, 12), dtype=np.int32),
        "seed_counts": rng.integers(0, 20, (rows, 5), dtype=np.int32),
        "carried_counts": rng.integers(0, 20, (rows, 12), dtype=np.int32),
        "unlocked": np.tile([[1, 0, 0, 0]], (rows, 1)).astype(np.uint8),
        "market_inventory": rng.integers(0, 20, (rows, 9), dtype=np.int32),
        "market_prices": rng.normal(size=(rows, 9)).astype(np.float32),
        "shop_counts": rng.integers(0, 20, (rows, 9), dtype=np.int32),
        "day": np.arange(rows, dtype=np.int16),
        "days_remaining": np.full((rows,), 29, dtype=np.int16),
        "economic_context": rng.normal(size=(rows, 14)).astype(np.float32),
        "crop_capacity": rng.integers(0, 20, (rows, 5), dtype=np.int16),
    }


def _actions(rows: int) -> np.ndarray:
    actions = np.zeros((rows, 9), dtype=np.int64)
    actions[:, 4:] = 100
    actions[:, 0] = np.arange(rows) % 4
    return actions


def _reference_batches(inputs, actions, batch_size, *, seed, epoch,
                       shuffle, start_batch, row_ids, contexts):
    n = len(inputs["board_kind"])
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        rng.shuffle(order)
    shuffled = {key: np.array(value[order], copy=True)
                for key, value in inputs.items()}
    labels = np.asarray(actions).astype(np.int32, copy=True)[order]
    ids = np.asarray(row_ids, dtype=np.int64)[order]
    shuffled_contexts = tuple(contexts[index] for index in order)
    result = []
    for batch_index, begin in enumerate(range(0, n, batch_size)):
        if batch_index < start_batch:
            continue
        end = min(begin + batch_size, n)
        source = np.arange(begin, end, dtype=np.int64)
        real = len(source)
        if real < batch_size:
            source = np.pad(source, (0, batch_size - real), mode="edge")
        mask = np.zeros(batch_size, dtype=bool)
        mask[:real] = True
        result.append((
            {key: np.array(value[source], copy=True)
             for key, value in shuffled.items()},
            np.array(labels[source], copy=True), mask,
            tuple(shuffled_contexts[index] for index in source),
            np.array(ids[source], copy=True)))
    return result


def _snapshot(batches):
    return [(
        {key: np.array(value, copy=True) for key, value in batch.inputs.items()},
        np.array(batch.actions, copy=True),
        np.array(batch.real_row_mask, copy=True),
        tuple(batch.physical_contexts) if batch.physical_contexts is not None else None,
        None if batch.row_ids is None else np.array(batch.row_ids, copy=True),
    ) for batch in batches]


def _assert_same(left, right):
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert expected[0].keys() == actual[0].keys()
        for key in expected[0]:
            assert np.array_equal(expected[0][key], actual[0][key])
        assert np.array_equal(expected[1], actual[1])
        assert np.array_equal(expected[2], actual[2])
        assert expected[3] == actual[3]
        assert np.array_equal(expected[4], actual[4])


def test_index_only_iterator_matches_reference_including_cursor_and_padding():
    inputs = _inputs()
    actions = _actions(len(inputs["board_kind"]))
    ids = np.arange(1000, 1000 + len(actions), dtype=np.int64)
    contexts = tuple({"row": int(row)} for row in range(len(actions)))
    kwargs = dict(seed=17, epoch=3, shuffle=True, start_batch=1,
                  row_ids=ids, contexts=contexts)
    expected = _reference_batches(inputs, actions, 4, **kwargs)
    actual = _snapshot(iter_fixed_batches(
        inputs, actions, 4, seed=17, epoch=3, shuffle=True, start_batch=1,
        row_ids=ids, physical_contexts=contexts))
    _assert_same(expected, actual)


def test_shuffle_is_deterministic_per_epoch_and_changes_between_epochs():
    inputs = _inputs(20)
    actions = _actions(20)

    def rows(epoch):
        return np.concatenate([
            batch.row_ids[batch.real_row_mask]
            for batch in iter_fixed_batches(
                inputs, actions, 6, seed=42, epoch=epoch, row_ids=None)])

    first = rows(0)
    assert np.array_equal(first, rows(0))
    assert not np.array_equal(first, rows(1))


def test_prefetch_is_bit_identical_for_all_worker_counts():
    inputs = _inputs(13)
    actions = _actions(13)
    kwargs = dict(seed=9, epoch=2, shuffle=True, start_batch=0,
                  row_ids=np.arange(13, dtype=np.int64))
    expected = _snapshot(iter_fixed_batches(inputs, actions, 5, **kwargs))
    for workers in (1, 2, 3):
        actual = _snapshot(iter_fixed_batches(
            inputs, actions, 5, host_workers=workers, prefetch_batches=2,
            **kwargs))
        _assert_same(expected, actual)


def test_source_preparation_keeps_original_arrays_shared():
    inputs = _inputs(4)
    source = bc._prepare_batch_source(
        inputs, _actions(4), physical_contexts=None, row_ids=None)
    assert source.inputs["board_numeric"] is inputs["board_numeric"]
    assert source.inputs["board_kind"] is inputs["board_kind"]


def test_worker_exception_propagates_and_shutdown_is_reached(monkeypatch):
    inputs = _inputs(8)
    actions = _actions(8)
    original = bc._make_indexed_batch
    calls = {"count": 0}

    def fail_once(source, indices, real_rows, batch_size):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic host worker failure")
        return original(source, indices, real_rows, batch_size)

    monkeypatch.setattr(bc, "_make_indexed_batch", fail_once)
    with pytest.raises(RuntimeError, match="synthetic host worker failure"):
        list(iter_fixed_batches(inputs, actions, 2, host_workers=2,
                                prefetch_batches=3))


def test_cli_exposes_bounded_host_pipeline_defaults():
    args = build_parser().parse_args([])
    assert args.host_workers == 3
    assert args.prefetch_batches == 6
