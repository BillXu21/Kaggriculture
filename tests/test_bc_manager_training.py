"""Focused tests for the BC training/evaluation CLI stack.

Covers: in-RAM dataset exact keys/shapes without metadata leakage (opponent
false/true), config resolution + parameter print, single CPU train/validation
steps with the clipping path, metric report contents with explicit all-zero
nonzero-recall behavior, train-only day baseline reporting, clear v1/empty-
split failures, checkpoint payload fields, save/load eval equivalence, a
synthetic schema-v3 CLI end-to-end smoke writing best/last checkpoints under
ignored temp, and early stopping. No network access, no full-corpus work.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from test_bc_manager import _records  # noqa: E402

from bc_manager.adapter import SchemaVersionError, load_train_val  # noqa: E402
from bc_manager.baseline import DayBaseline, evaluate_baseline  # noqa: E402
from bc_manager.cli import (  # noqa: E402
    build_parser,
    resolve_model_config,
    resolve_training_config,
)
from bc_manager.loss import GROUP_NAMES, ManagerLossConfig  # noqa: E402
from bc_manager.metrics import nonzero_recall  # noqa: E402
from bc_manager.model import DailyManagerTransformer  # noqa: E402
from bc_manager.model import tiny_manager_config  # noqa: E402
from bc_manager.training import (  # noqa: E402
    ManagerTorchDataset,
    TrainingConfig,
    arrays_to_tensors,
    evaluate,
    load_checkpoint,
    load_model_from_checkpoint,
    resolve_device,
    run_training,
    train_one_epoch,
)
from replay_daily.storage import write_parquet  # noqa: E402

TRAIN_DATES = ("2026-08-17", "2026-08-18", "2026-08-19")
VAL_DATES = ("2026-08-21",)
MIN_SCORE = 2950.0

TARGET_KEYS = {"crop_target", "animal_target", "land_count",
               "fertilizer_target", "care_target", "sell_presence",
               "sell_quantity_log1p"}


def write_v2(tmp_path, name="v2.parquet"):
    path = tmp_path / name
    write_parquet(_records(), path)
    return path


def load_splits(path, include_opponent=False):
    return load_train_val(path, train_dates=TRAIN_DATES, val_dates=VAL_DATES,
                          min_score=MIN_SCORE,
                          include_opponent=include_opponent)


def tiny_training(**overrides) -> TrainingConfig:
    params = dict(batch_size=2, epochs=1, checkpoint_dir=None)
    params.update(overrides)
    return TrainingConfig(**params)


# ------------------------------------------------- in-RAM dataset contract


def test_dataset_exact_keys_shapes_and_no_metadata_leakage(tmp_path):
    path = write_v2(tmp_path)
    data = load_splits(path)
    inputs_t, targets_t = arrays_to_tensors(
        data["train"]["inputs"], data["train"]["targets"],
        include_opponent_board=False)
    loader = DataLoader(ManagerTorchDataset(inputs_t, targets_t),
                        batch_size=2, shuffle=False)
    batch_inputs, batch_targets = next(iter(loader))

    assert not any(key.startswith("opp_") for key in batch_inputs)
    assert set(batch_targets) == TARGET_KEYS
    forbidden = ("score", "bank", "path", "meta", "episode", "player")
    assert all(not any(word in key for word in forbidden)
               for key in list(batch_inputs) + list(batch_targets))
    assert batch_inputs["board_kind"].shape == (2, 100)
    assert batch_inputs["board_numeric"].shape == (2, 100, 11)
    assert batch_inputs["scalars"].shape == (2, 4)
    assert batch_targets["crop_target"].shape == (2, 5)
    assert batch_targets["land_count"].shape == (2,)
    assert batch_targets["sell_presence"].shape == (2, 9, 6)

    # Opponent PUBLIC arrays are included only when requested.
    data_opp = load_splits(path, include_opponent=True)
    opp_inputs, _ = arrays_to_tensors(
        data_opp["train"]["inputs"], data_opp["train"]["targets"],
        include_opponent_board=True)
    assert not any(k.startswith("opp_") for k in inputs_t)
    assert any(k.startswith("opp_") for k in opp_inputs)


def test_cli_config_resolution_and_parameter_print(capsys):
    args = build_parser().parse_args(
        ["x.parquet", "--tiny", "--checkpoint-dir", "data/temp/x"])
    model_config = resolve_model_config(args)
    training_config = resolve_training_config(args)
    assert (model_config.d_model, model_config.num_layers,
            model_config.num_heads, model_config.ffn_dim,
            model_config.dropout) == (16, 1, 1, 32, 0.0)
    assert training_config.lr == 3e-4
    assert training_config.weight_decay == 1e-2
    assert training_config.batch_size == 256
    assert training_config.gradient_clip == 1.0
    assert training_config.num_workers == 0
    assert training_config.use_amp is None  # auto: CUDA only

    default_args = build_parser().parse_args(["x.parquet", "--checkpoint-dir",
                                              "data/temp/y"])
    default_config = resolve_model_config(default_args)
    assert (default_config.d_model, default_config.num_layers,
            default_config.num_heads, default_config.ffn_dim,
            default_config.dropout) == (128, 4, 4, 384, 0.1)
    model = DailyManagerTransformer(default_config)
    print(f"resolved default trainable parameters: "
          f"{model.trainable_parameters}")
    print(f"resolved device: {resolve_device('auto')}")
    assert model.trainable_parameters == 1_071_040  # regression pin


def test_single_cpu_train_and_validation_steps_finite_with_clip(tmp_path):
    path = write_v2(tmp_path)
    data = load_splits(path)
    train_in, train_tgt = arrays_to_tensors(
        data["train"]["inputs"], data["train"]["targets"],
        include_opponent_board=False)
    val_in, val_tgt = arrays_to_tensors(
        data["val"]["inputs"], data["val"]["targets"],
        include_opponent_board=False)
    device = torch.device("cpu")
    torch.manual_seed(3)
    model = DailyManagerTransformer(tiny_manager_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_config = tiny_training()

    train_report = train_one_epoch(
        model, DataLoader(ManagerTorchDataset(train_in, train_tgt),
                          batch_size=3),
        optimizer, device, training_config, scaler=None)
    assert set(GROUP_NAMES) <= {k.removeprefix("group.")
                                for k in train_report}
    assert np.isfinite(train_report["total"])
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    val_report = evaluate(model,
                          DataLoader(ManagerTorchDataset(val_in, val_tgt),
                                     batch_size=1),
                          device, ManagerLossConfig())
    assert np.isfinite(val_report["total"])
    for name in GROUP_NAMES:
        assert f"group.{name}" in val_report
    for task in ("crop", "animal", "fertilizer", "care"):
        assert f"{task}_exact_accuracy" in val_report
        assert f"{task}_mae" in val_report
        assert f"{task}_nonzero_recall" in val_report
    assert "animal_goose_nonzero_recall" in val_report
    assert "animal_goose_pred_nonzero_rate" in val_report
    assert "land_exact_accuracy" in val_report
    assert "land_mae" in val_report
    assert "sell_presence_accuracy" in val_report
    assert "sell_presence_nonzero_recall" in val_report
    assert "sell_positive_quantity_log_mae" in val_report


def test_nonzero_recall_all_zero_behavior_is_explicit():
    zeros = np.zeros((4,), dtype=np.int32)
    ones = np.ones((4,), dtype=np.int32)
    assert nonzero_recall(zeros, ones) == 0.0  # defined, no division error
    assert nonzero_recall(zeros, ones, zero_denominator_value=1.0) == 1.0
    assert nonzero_recall(ones, ones) == 1.0
    assert nonzero_recall(ones, zeros) == 0.0


def test_day_baseline_fit_train_only_and_reported_beside_model(tmp_path):
    path = write_v2(tmp_path)
    data = load_splits(path)
    baseline = DayBaseline().fit(data["train"]["inputs"]["day"],
                                 data["train"]["targets"])
    report = evaluate_baseline(baseline, data["val"]["inputs"]["day"],
                               data["val"]["targets"])
    assert {"crop_target", "animal_target", "fertilizer_target",
            "care_target", "land_count", "sells"} <= set(report)

    result = run_training(path, model_config=tiny_manager_config(),
                          training_config=tiny_training(), log=lambda _: None)
    reported = result["baseline_validation"]
    # The run's baseline report matches an explicitly train-only fit.
    assert (reported["crop_target"]["exact_accuracy"]
            == report["crop_target"]["exact_accuracy"])
    assert (reported["sells"]["presence_accuracy"]
            == report["sells"]["presence_accuracy"])


def test_v1_schema_and_empty_split_fail_clearly(tmp_path):
    # Storage refuses to *write* v1; craft the file through raw Arrow rows
    # exactly like the adapter-level rejection test does.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from replay_daily.storage import RECORD_SCHEMA, records_to_table

    rows = records_to_table(_records()).to_pylist()
    for row in rows:
        row["schema_version"] = 1
    v1_path = tmp_path / "v1.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=RECORD_SCHEMA), v1_path)
    with pytest.raises(SchemaVersionError, match="schema_version"):
        run_training(v1_path, model_config=tiny_manager_config(),
                     training_config=tiny_training(), log=lambda _: None)

    v2_path = write_v2(tmp_path)
    with pytest.raises(ValueError, match="empty train split"):
        run_training(v2_path, model_config=tiny_manager_config(),
                     training_config=tiny_training(),
                     train_dates=("2030-01-01",), log=lambda _: None)
    with pytest.raises(ValueError, match="empty validation split"):
        run_training(v2_path, model_config=tiny_manager_config(),
                     training_config=tiny_training(),
                     val_dates=("2030-01-01",), log=lambda _: None)


def test_checkpoint_best_and_last_payload_fields(tmp_path):
    path = write_v2(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    result = run_training(
        path, model_config=tiny_manager_config(),
        training_config=tiny_training(checkpoint_dir=str(ckpt_dir)),
        log=lambda _: None)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    assert best_path.exists() and last_path.exists()
    assert result["best_checkpoint"] == str(best_path)
    best = load_checkpoint(best_path)
    last = load_checkpoint(last_path)
    for payload in (best, last):
        assert payload["format"] == "bc_manager_checkpoint_v1"
        assert payload["kind"] in ("best", "last")
        assert isinstance(payload["epoch"], int)
        assert "model_state_dict" in payload
        assert payload["model_config"]["d_model"] == 16
        assert payload["training_config"]["lr"] == 3e-4
        assert "validation_metrics" in payload
    assert best["epoch"] == result["best_epoch"]
    assert last["epoch"] == len(result["history"])


def test_checkpoint_save_load_eval_output_equivalence(tmp_path):
    path = write_v2(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    run_training(path, model_config=tiny_manager_config(),
                 training_config=tiny_training(checkpoint_dir=str(ckpt_dir)),
                 log=lambda _: None)
    loaded_model, payload = load_model_from_checkpoint(ckpt_dir / "best.pt")

    data = load_splits(path)
    val_in, val_tgt = arrays_to_tensors(
        data["val"]["inputs"], data["val"]["targets"],
        include_opponent_board=False)
    loader = DataLoader(ManagerTorchDataset(val_in, val_tgt), batch_size=4)
    reference = DailyManagerTransformer(tiny_manager_config())
    # Same-batch equivalence is checked against the reloaded weights below;
    # the freshly initialized reference only guards that outputs differ.
    with torch.no_grad():
        fresh_outputs = reference(next(iter(loader))[0])
        reloaded_outputs = loaded_model(next(iter(loader))[0])
    assert set(reloaded_outputs) == set(fresh_outputs)
    assert any(not torch.equal(reloaded_outputs[k], fresh_outputs[k])
               for k in reloaded_outputs)
    val_report = evaluate(loaded_model, loader, torch.device("cpu"),
                          ManagerLossConfig())
    assert np.isfinite(val_report["total"])
    stored_total = payload["validation_metrics"]["total"]
    assert abs(val_report["total"] - stored_total) < 1e-6


def test_cli_end_to_end_synthetic_smoke_two_epochs(tmp_path, capsys):
    path = write_v2(tmp_path)
    ckpt_dir = tmp_path / "cli-ckpt"
    code = pytest.importorskip("bc_manager.cli").main([
        str(path), "--tiny", "--epochs", "2", "--batch-size", "2",
        "--train-dates", ",".join(TRAIN_DATES),
        "--val-dates", ",".join(VAL_DATES),
        "--min-score", str(MIN_SCORE), "--seed", "5",
        "--checkpoint-dir", str(ckpt_dir),
    ])
    captured = capsys.readouterr()
    assert code == 0
    assert "params=" in captured.out
    assert "epoch=2/2" in captured.out
    assert "best_epoch=" in captured.out
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()


def test_early_stopping_stops_before_epoch_budget(tmp_path):
    path = write_v2(tmp_path)
    # High LR converges fast on the 3-row synthetic split, so validation
    # total stops improving and patience triggers well before epoch 60.
    result = run_training(
        path, model_config=tiny_manager_config(),
        training_config=tiny_training(epochs=60, seed=11, lr=0.03,
                                      early_stopping_patience=3),
        log=lambda _: None)
    assert result["stopped_early"] is True
    assert len(result["history"]) < 60
    assert result["best_epoch"] >= 1


def test_no_network_or_full_corpus_surface():
    parser = build_parser()
    flags = {action.dest for action in parser._actions}
    assert "parquet" in flags  # local paths only; no URL/download argument
    assert not any("download" in dest or "url" in dest for dest in flags)
