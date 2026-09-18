"""Focused tests for the epoch-oriented native Stage 2.5 BC trainer."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_LEGACY,
)
from rl_manager.stage25_bc import (
    CompiledEvalStep,
    CompiledTrainStep,
    Stage25BCConfig,
    init_opt_state,
    iter_fixed_batches,
    load_array_dataset,
    load_checkpoint,
    make_fixed_batch,
    train_step,
    validation_metrics,
)
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


def _inputs(rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "board_kind": rng.integers(0, 3, (rows, 100)).astype(np.int16),
        "board_crop": np.zeros((rows, 100), np.int8),
        "board_animal": np.zeros((rows, 100), np.int8),
        "board_numeric": np.zeros((rows, 100, 11), np.float32),
        "board_bool": np.zeros((rows, 100, 8), bool),
        "board_mask": np.zeros((rows, 100, 4), np.uint8),
        "scalars": np.zeros((rows, 4), np.float32),
        "shed_counts": np.zeros((rows, 12), np.int32),
        "seed_counts": np.zeros((rows, 5), np.int32),
        "carried_counts": np.zeros((rows, 12), np.int32),
        "unlocked": np.tile([[1, 0, 0, 0]], (rows, 1)).astype(np.uint8),
        "market_inventory": np.zeros((rows, 9), np.int32),
        "market_prices": np.zeros((rows, 9), np.float32),
        "shop_counts": np.zeros((rows, 9), np.int32),
        "day": np.zeros((rows,), np.int16),
        "days_remaining": np.full((rows,), 29, np.int16),
        "economic_context": np.zeros((rows, 14), np.float32),
        "crop_capacity": np.zeros((rows, 5), np.int16),
    }


def _actions(rows: int, seed: int = 0):
    rng = np.random.default_rng(seed + 17)
    actions = np.zeros((rows, 9), np.int64)
    actions[:, 4:] = 100
    actions[:, 0] = rng.integers(0, 4, rows)
    return actions


def _batch(rows: int = 2, batch_size: int = 2, seed: int = 0):
    return make_fixed_batch(_inputs(rows, seed), _actions(rows, seed), batch_size)


def _write_dataset(path: Path, rows: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, actions=_actions(rows, seed),
        **{f"input_{key}": value
           for key, value in _inputs(rows, seed).items()})


def _flat_left_right_maxdiff(left, right) -> float:
    return max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(jax.tree_util.tree_leaves(left),
                        jax.tree_util.tree_leaves(right)))


def test_load_array_dataset_strips_prefix_and_accepts_compressed_npz(tmp_path):
    path = tmp_path / "data.npz"
    _write_dataset(path, 5, 1)
    inputs, actions = load_array_dataset(path)
    assert actions.shape == (5, 9)
    assert "board_kind" in inputs and "input_board_kind" not in inputs
    assert inputs["board_kind"].shape == (5, 100)


def test_compiled_train_step_matches_reference_math():
    model = Stage25ModelConfig.tiny()
    config = Stage25BCConfig(model=model, batch_size=2, lr=1e-3,
                             weight_decay=0.0)
    params = init_stage25_params(model, seed=11)
    state = init_opt_state(params, config)
    batch = _batch()
    key = jax.random.PRNGKey(12)

    reference = train_step(params, state, key, batch, config)
    compiled = CompiledTrainStep(params, config)(params, state, key, batch)

    assert _flat_left_right_maxdiff(reference[0], compiled[0]) < 1e-5
    assert _flat_left_right_maxdiff(reference[1], compiled[1]) < 1e-5
    assert np.array_equal(np.asarray(reference[2]), np.asarray(compiled[2]))
    assert np.isclose(reference[3]["loss"], compiled[3]["loss"], rtol=1e-5)
    assert np.allclose(reference[3]["per_step_nll"],
                       compiled[3]["per_step_nll"], rtol=1e-4, atol=1e-5)


def test_compiled_train_step_dropout_matches_reference_and_rng():
    model = Stage25ModelConfig.tiny(dropout=0.15)
    config = Stage25BCConfig(model=model, batch_size=2, weight_decay=0.0)
    params = init_stage25_params(model, seed=13)
    state = init_opt_state(params, config)
    batch = _batch()
    key = jax.random.PRNGKey(14)

    reference = train_step(params, state, key, batch, config)
    compiled = CompiledTrainStep(params, config)(params, state, key, batch)

    assert _flat_left_right_maxdiff(reference[0], compiled[0]) < 1e-5
    assert np.array_equal(np.asarray(reference[2]), np.asarray(compiled[2]))
    assert np.isclose(reference[3]["loss"], compiled[3]["loss"], rtol=1e-5)


def test_compiled_update_freezes_value_head():
    model = Stage25ModelConfig.tiny()
    config = Stage25BCConfig(model=model, batch_size=2, lr=1e-2,
                             weight_decay=1e-1)
    params = init_stage25_params(model, seed=15)
    state = init_opt_state(params, config)
    updated, _, _, _ = CompiledTrainStep(params, config)(
        params, state, jax.random.PRNGKey(16), _batch())
    assert _flat_left_right_maxdiff(params["value_head"],
                                    updated["value_head"]) == 0.0
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(params["output_projections"]),
                        jax.tree_util.tree_leaves(updated["output_projections"])))


def test_compiled_invalid_action_fails_without_committing():
    model = Stage25ModelConfig.tiny()
    config = Stage25BCConfig(model=model, batch_size=2)
    params = init_stage25_params(model, seed=17)
    state = init_opt_state(params, config)
    key = jax.random.PRNGKey(18)
    step = CompiledTrainStep(params, config)
    bad = _actions(2, 0)
    bad[0, 0] = 4  # outside the step-0 vocabulary
    invalid = make_fixed_batch(_inputs(2, 0), bad, 2)
    with pytest.raises(ValueError, match="outside its vocabulary"):
        step(params, state, key, invalid)
    valid = _batch()
    direct = step(params, state, key, valid)
    again = step(params, state, key, valid)
    assert _flat_left_right_maxdiff(direct[0], again[0]) == 0.0
    assert np.array_equal(np.asarray(direct[2]), np.asarray(again[2]))


def test_validation_weighted_aggregation_and_padded_final_batch():
    model = Stage25ModelConfig.tiny()
    config = Stage25BCConfig(model=model, batch_size=2)
    params = init_stage25_params(model, seed=19)

    two_batches = validation_metrics(
        params, list(iter_fixed_batches(_inputs(3, 3), _actions(3, 3), 2)),
        config)
    padded = make_fixed_batch(_inputs(3, 3), _actions(3, 3), 4)
    single = validation_metrics(
        params, [padded], config, eval_step=CompiledEvalStep(config))
    reference = validation_metrics(
        params,
        list(iter_fixed_batches(_inputs(3, 3), _actions(3, 3), 4,
                                shuffle=False)),
        config)
    assert two_batches["valid_rows"] == reference["valid_rows"] == 3
    assert np.isclose(two_batches["joint_nll"], reference["joint_nll"], rtol=1e-5)
    assert np.allclose(two_batches["per_step_nll"],
                       reference["per_step_nll"], rtol=1e-4, atol=1e-5)
    assert np.allclose(two_batches["per_step_accuracy"],
                       reference["per_step_accuracy"], rtol=1e-4, atol=1e-5)
    assert np.isclose(single["joint_nll"], reference["joint_nll"], rtol=1e-5)


def _run(tmp_path: Path, train_rows: int = 6, val_rows: int = 4, **overrides):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train, train_rows, 21)
    _write_dataset(val, val_rows, 22)
    checkpoint_dir = tmp_path / "run"
    arguments = [
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(checkpoint_dir), "--batch-size", "2",
        "--model-size", "tiny", "--seed", "0", "--lr", "1e-3",
        "--weight-decay", "0.0",
    ]
    arguments.extend(overrides.pop("argv", []))
    for key, value in overrides.items():
        arguments.extend([f"--{key.replace('_', '-')}", str(value)])
    import rl_manager.stage25_bc_cli as cli
    assert cli.main(arguments) == 0
    return checkpoint_dir, train, val


def test_cli_writes_epoch_last_best_and_metrics(tmp_path):
    checkpoint_dir, _, _ = _run(tmp_path, epochs=2)
    for name in ("epoch_001.npz", "epoch_002.npz", "last.npz", "best.npz",
                 "metrics.jsonl"):
        assert (checkpoint_dir / name).exists(), name

    records = [json.loads(line) for line in
               (checkpoint_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]
    required = {
        "epoch", "global_step", "train_joint_nll", "val_joint_nll",
        "train_per_step_nll", "val_per_step_nll", "train_per_step_accuracy",
        "val_per_step_accuracy", "train_rows", "validation_rows",
        "train_wall_seconds", "validation_wall_seconds",
        "checkpoint_wall_seconds", "epoch_total_wall_seconds",
        "learning_rate", "best",
    }
    for record in records:
        assert required <= set(record)
        assert record["train_rows"] == 6
        assert record["validation_rows"] == 4
        assert record["learning_rate"] == 1e-3
        assert len(record["train_per_step_nll"]) == 9
        assert len(record["val_per_step_accuracy"]) == 9


def test_cli_best_and_last_checkpoint_selection(tmp_path):
    checkpoint_dir, _, _ = _run(tmp_path, epochs=2)
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    records = [json.loads(line) for line in
               (checkpoint_dir / "metrics.jsonl").read_text().splitlines()]
    best_record = min(records, key=lambda item: item["val_joint_nll"])
    assert best_record["best"] is True
    best_params = load_checkpoint(checkpoint_dir / "best.npz", config=config)[0]
    epoch_params = load_checkpoint(
        checkpoint_dir / f"epoch_{best_record['epoch']:03d}.npz", config=config)[0]
    assert _flat_left_right_maxdiff(best_params, epoch_params) == 0.0
    last_params = load_checkpoint(checkpoint_dir / "last.npz", config=config)[0]
    final_params = load_checkpoint(checkpoint_dir / "epoch_002.npz",
                                   config=config)[0]
    assert _flat_left_right_maxdiff(last_params, final_params) == 0.0


def test_cli_epoch_boundary_cursor_resumes_at_next_epoch(tmp_path):
    checkpoint_dir, _, _ = _run(tmp_path, epochs=1)
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    meta = load_checkpoint(checkpoint_dir / "last.npz", config=config)[3]
    assert meta["data_order_position"] == {"epoch": 1, "batch": 0, "seed": 0}
    assert meta["epoch"] == 1


def test_cli_resumed_second_epoch_matches_uninterrupted(tmp_path):
    # Uninterrupted two-epoch run.
    run_a, train, val = _run(tmp_path / "a", epochs=2)
    # One epoch, then resume into the second epoch.
    run_b, _, _ = _run(tmp_path / "b", epochs=1)
    import rl_manager.stage25_bc_cli as cli
    assert cli.main([
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(run_b), "--epochs", "1",
        "--resume", str(run_b / "last.npz"), "--batch-size", "2",
        "--model-size", "tiny", "--seed", "0", "--lr", "1e-3",
        "--weight-decay", "0.0",
    ]) == 0
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    params_a = load_checkpoint(run_a / "epoch_002.npz", config=config)[0]
    params_b = load_checkpoint(run_b / "epoch_002.npz", config=config)[0]
    assert _flat_left_right_maxdiff(params_a, params_b) == 0.0


def test_cli_historical_encoder_provenance_survives_epoch_checkpoints(tmp_path):
    from bc_manager_jax.checkpoint import save_native
    from bc_manager_jax.model import init_train_params

    model = Stage25ModelConfig.tiny()
    source = tmp_path / "legacy-source.npz"
    save_native(source, init_train_params(model.manager_config, seed=5,
                                          model_variant="E"),
                model.manager_config, model_variant="E",
                e_history_version=E_HISTORY_LEGACY)
    checkpoint_dir, _, _ = _run(
        tmp_path, epochs=2,
        argv=["--import", str(source), "--allow-legacy-e"])
    config = Stage25BCConfig(model=model, batch_size=2, lr=1e-3,
                             weight_decay=0.0)
    for name in ("epoch_001.npz", "epoch_002.npz", "last.npz", "best.npz"):
        meta = load_checkpoint(checkpoint_dir / name, config=config)[3]
        assert meta["e_history_version"] == E_HISTORY_CORRECTED_V1
        assert meta["source_e_identity"]["history_version"] == E_HISTORY_LEGACY
        assert meta["source_identity"]
        assert meta["provenance"]["historical_import"] == "encoder_only"


def test_cli_patience_stops_after_no_improvement(tmp_path, monkeypatch):
    import rl_manager.stage25_bc_cli as cli

    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train, 6, 31)
    _write_dataset(val, 4, 32)
    calls = {"count": 0}

    def plateau(params, batches, config, *, eval_step=None):
        calls["count"] += 1
        return {"joint_nll": 100.0 + calls["count"],
                "per_step_nll": np.zeros(9),
                "per_step_accuracy": np.zeros(9),
                "valid_rows": 4}

    monkeypatch.setattr(cli, "validation_metrics", plateau)
    checkpoint_dir = tmp_path / "run"
    assert cli.main([
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "5",
        "--patience", "1", "--batch-size", "2", "--model-size", "tiny",
        "--seed", "0", "--lr", "1e-3", "--weight-decay", "0.0",
    ]) == 0
    records = [json.loads(line) for line in
               (checkpoint_dir / "metrics.jsonl").read_text().splitlines()]
    assert calls["count"] == 2
    assert [record["epoch"] for record in records] == [1, 2]
    assert (checkpoint_dir / "epoch_002.npz").exists()
    assert not (checkpoint_dir / "epoch_003.npz").exists()


def test_cli_rejects_conflicting_train_sources(tmp_path):
    import rl_manager.stage25_bc_cli as cli

    data = tmp_path / "data.npz"
    _write_dataset(data, 2, 41)
    with pytest.raises(ValueError, match="either --train-data or --data"):
        cli.main(["--train-data", str(data), "--data", str(data),
                  "--model-size", "tiny", "--output", str(tmp_path / "out.npz")])


def test_cli_bounded_steps_can_stop_mid_epoch_and_resume(tmp_path):
    import rl_manager.stage25_bc_cli as cli

    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train, 6, 51)
    _write_dataset(val, 4, 52)
    checkpoint_dir = tmp_path / "run"
    assert cli.main([
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "3",
        "--steps", "1", "--batch-size", "2", "--model-size", "tiny",
        "--seed", "0", "--lr", "1e-3", "--weight-decay", "0.0",
    ]) == 0
    # A mid-epoch bound leaves no completed-epoch file but a resumable last.
    assert (checkpoint_dir / "last.npz").exists()
    assert not (checkpoint_dir / "epoch_001.npz").exists()
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    meta = load_checkpoint(checkpoint_dir / "last.npz", config=config)[3]
    assert meta["data_order_position"] == {"epoch": 0, "batch": 1, "seed": 0}
    assert meta["step"] == 1


def test_cli_step_budget_at_epoch_boundary_does_not_overrun(tmp_path):
    import rl_manager.stage25_bc_cli as cli

    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train, 6, 61)
    _write_dataset(val, 4, 62)
    checkpoint_dir = tmp_path / "run"
    assert cli.main([
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "3",
        "--steps", "3", "--batch-size", "2", "--model-size", "tiny",
        "--seed", "0", "--lr", "1e-3", "--weight-decay", "0.0",
    ]) == 0
    records = [json.loads(line) for line in
               (checkpoint_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1]
    assert (checkpoint_dir / "epoch_001.npz").exists()
    assert not (checkpoint_dir / "epoch_002.npz").exists()
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    meta = load_checkpoint(checkpoint_dir / "last.npz", config=config)[3]
    assert meta["step"] == 3
    assert meta["data_order_position"] == {"epoch": 1, "batch": 0, "seed": 0}


def test_cli_zero_steps_writes_no_epoch_checkpoint(tmp_path):
    import rl_manager.stage25_bc_cli as cli

    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train, 6, 71)
    _write_dataset(val, 4, 72)
    checkpoint_dir = tmp_path / "run"
    assert cli.main([
        "--train-data", str(train), "--val-data", str(val),
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "2",
        "--steps", "0", "--batch-size", "2", "--model-size", "tiny",
        "--seed", "0", "--lr", "1e-3", "--weight-decay", "0.0",
    ]) == 0
    assert (checkpoint_dir / "last.npz").exists()
    assert not (checkpoint_dir / "epoch_001.npz").exists()
    assert not (checkpoint_dir / "metrics.jsonl").exists()
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    meta = load_checkpoint(checkpoint_dir / "last.npz", config=config)[3]
    assert meta["step"] == 0
    assert meta["data_order_position"] == {"epoch": 0, "batch": 0, "seed": 0}


def test_cli_epoch_checkpoints_initialize_ppo(tmp_path):
    from rl_manager.stage25_checkpoint import initialize_stage25_ppo_from_checkpoint

    checkpoint_dir, _, _ = _run(tmp_path, epochs=1)
    model = Stage25ModelConfig.tiny()
    for name in ("epoch_001.npz", "last.npz", "best.npz"):
        params, meta = initialize_stage25_ppo_from_checkpoint(
            checkpoint_dir / name, config=model, seed=None,
            expected_e_history_version=E_HISTORY_CORRECTED_V1)
        assert meta["payload_kind"] == "stage25_bc_training_state_v1"
        assert meta["e_history_version"] == E_HISTORY_CORRECTED_V1
        assert "encoder" in params
