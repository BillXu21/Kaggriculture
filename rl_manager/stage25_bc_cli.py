"""Native Stage 2.5 teacher-forced BC trainer.

The runner is epoch-oriented: a normal invocation completes whole epochs over
the train dataset, validates on a separate dataset, and writes resumable
checkpoints plus one JSONL metrics record per completed epoch.  ``--steps``
stays available as a bounded smoke/debug mode.

Resume-boundary convention
--------------------------
A checkpoint is written only after a committed optimizer update.  Its
``data_order_position`` names the *next* fixed batch to execute as
``(epoch, batch)``.  When an epoch finishes, that cursor is normalised to
``(epoch + 1, 0)`` so resuming a completed epoch begins at batch 0 of the
following epoch instead of at the exhausted end of the previous one.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    normalize_e_history_version,
)
from rl_manager.stage25_bc import (
    CompiledEvalStep,
    CompiledTrainStep,
    Stage25BCConfig,
    import_encoder_checkpoint,
    init_opt_state,
    iter_fixed_batches,
    load_array_dataset,
    load_checkpoint,
    save_checkpoint,
    validation_metrics,
)
from rl_manager.stage25_checkpoint import Stage25CheckpointError
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, action="append",
        help="canonical Parquet file/directory (repeat for multiple files), "
             "or one NPZ array dataset; alias for --train-data")
    parser.add_argument(
        "--train-data", type=Path,
        help="pickle-free NPZ array training dataset")
    parser.add_argument(
        "--val-data", type=Path,
        help="pickle-free NPZ array validation dataset")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model-size", choices=("tiny", "small", "large"), default="tiny")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--ffn", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="bounded optimizer-step budget for smoke/debug runs; "
             "omit for a full epoch-oriented run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="directory for per-epoch, last, and best resumable checkpoints "
             "plus metrics.jsonl; requires --val-data")
    parser.add_argument(
        "--patience", type=int, default=0,
        help="early-stop after this many epochs without validation improvement; "
             "0 disables early stopping")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--import", dest="import_path", type=Path)
    parser.add_argument(
        "--allow-legacy-e", action="store_true",
        help="allow a legacy E-history source encoder import (source only; "
             "Stage 2.5 operating history stays corrected)")
    parser.add_argument("--date", dest="dates", action="append")
    parser.add_argument("--min-score", type=float, default=2950.0)
    return parser


def _model(args: argparse.Namespace) -> Stage25ModelConfig:
    factory = {"tiny": Stage25ModelConfig.tiny,
               "small": Stage25ModelConfig.small,
               "large": Stage25ModelConfig.large}[args.model_size]
    overrides: dict[str, Any] = {}
    if args.d_model is not None:
        overrides["d_model"] = args.d_model
    if args.layers is not None:
        overrides["num_layers"] = args.layers
    if args.heads is not None:
        overrides["num_heads"] = args.heads
    if args.ffn is not None:
        overrides["ffn_dim"] = args.ffn
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    return factory(**overrides)


def _load_train(args: argparse.Namespace):
    if args.train_data is not None:
        if args.data:
            raise ValueError("provide either --train-data or --data, not both")
        inputs, actions = load_array_dataset(args.train_data)
        return inputs, actions, None
    if args.data:
        data_paths = tuple(args.data)
        canonical = any(path.is_dir() or path.suffix.lower() == ".parquet"
                        for path in data_paths)
        if canonical:
            if any(not (path.is_dir() or path.suffix.lower() == ".parquet")
                   for path in data_paths):
                raise ValueError(
                    "--data cannot mix canonical Parquet paths with an NPZ path")
            from rl_manager.stage25_adapter import load_dataset
            kwargs = {"min_score": args.min_score}
            if args.dates:
                kwargs["dates"] = tuple(args.dates)
            loaded = load_dataset(data_paths, **kwargs)
            return loaded["inputs"], loaded["actions"], loaded.get("row_ids")
        if len(data_paths) != 1:
            raise ValueError("NPZ array datasets accept exactly one --data path")
        inputs, actions = load_array_dataset(data_paths[0])
        return inputs, actions, None
    if not args.inputs or not args.labels:
        raise ValueError(
            "provide --train-data, --data, or both --inputs and --labels")
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {key: np.array(archive[key], copy=True) for key in archive.files}
    labels = np.load(args.labels, allow_pickle=False)
    if isinstance(labels, np.lib.npyio.NpzFile):
        with labels as archive:
            labels = np.array(
                archive["actions" if "actions" in archive else "labels"], copy=True)
    return inputs, labels, None


def _load_val(args: argparse.Namespace):
    if args.val_data is None:
        return None
    inputs, actions = load_array_dataset(args.val_data)
    return inputs, actions


def _store(path: Path, *, params: Any, opt_state: Any, rng: Any,
           config: Stage25BCConfig, step: int, epoch: int, cursor: dict[str, Any],
           seed: int, e_history_version: str, source_history_version: Any,
           source_identity: Any, provenance: Any, executor: Any, kind: str,
           cli_args: dict[str, Any]) -> None:
    save_checkpoint(
        path, params, opt_state, rng, config=config, step=step, epoch=epoch,
        seed=seed, shuffle_state=cursor,
        metadata={"run": {"cli": "rl_manager.stage25_bc_cli", "kind": kind},
                  "cli_args": cli_args},
        e_history_version=e_history_version,
        source_history_version=source_history_version,
        source_identity=source_identity, provenance=provenance,
        executor=executor)


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()


def _summary(record: dict[str, Any]) -> str:
    value = record["val_joint_nll"]
    val_text = "n/a" if value is None else f"{value:.6f}"
    return (f"epoch={record['epoch']} step={record['global_step']} "
            f"train_nll={record['train_joint_nll']:.6f} "
            f"val_nll={val_text} best={record['best']} "
            f"train_rows={record['train_rows']} "
            f"val_rows={record['validation_rows']} "
            f"time={record['epoch_total_wall_seconds']:.1f}s")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.steps is not None and args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if args.epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if args.patience < 0:
        raise ValueError("patience must be nonnegative")
    if args.checkpoint_dir is not None and args.val_data is None:
        raise ValueError("--checkpoint-dir requires --val-data for best selection")
    if args.train_data is not None and args.data:
        raise ValueError("provide either --train-data or --data, not both")

    train_inputs, train_actions, train_row_ids = _load_train(args)
    val_loaded = _load_val(args)
    model = _model(args)
    config = Stage25BCConfig(model=model, batch_size=args.batch_size,
                             lr=args.lr, weight_decay=args.weight_decay)
    rows = int(np.asarray(train_inputs["board_kind"]).shape[0])
    if rows < 1:
        raise ValueError("training dataset must be nonempty")
    num_batches = (rows + config.batch_size - 1) // config.batch_size

    operating_history = E_HISTORY_CORRECTED_V1
    source_history = None
    source_identity = None
    provenance = None
    executor = None
    if args.import_path:
        params, import_meta = import_encoder_checkpoint(
            args.import_path, model, seed=args.seed,
            allow_legacy_e=args.allow_legacy_e, return_metadata=True)
        source_history = import_meta.get("e_history_version")
        source_identity = dict(import_meta.get("source_identity") or {})
        source_identity["e_identity"] = import_meta.get("e_identity")
        source_identity["transfer"] = import_meta.get("imported")
        provenance = {"historical_import": import_meta.get("imported"),
                      "source_e_history_version": source_history}
    else:
        params = init_stage25_params(model, seed=args.seed)
    rng = jax.random.PRNGKey(args.seed)
    opt_state = init_opt_state(params, config)

    step = 0
    start_epoch = 0
    start_batch = 0
    if args.resume:
        try:
            params, opt_state, rng, meta = load_checkpoint(
                args.resume, config=config, params=params, seed=args.seed,
                expected_e_history_version=E_HISTORY_CORRECTED_V1,
                allow_legacy_e=False)
        except Stage25CheckpointError as exc:
            raise ValueError(
                "the BC CLI supports corrected-E operating history only; "
                f"cannot resume {args.resume}: {exc}") from exc
        step = int(meta.get("step", 0))
        cursor = meta.get("data_order_position") or {}
        start_epoch = int(cursor.get("epoch", meta.get("epoch", 0)))
        start_batch = int(cursor.get("batch", 0))
        if start_batch >= num_batches:
            # Tolerate a checkpoint written at the exhausted end of an epoch:
            # the next executable batch is batch 0 of the following epoch.
            start_epoch += 1
            start_batch = 0
        operating_history = normalize_e_history_version(
            meta.get("e_history_version", E_HISTORY_CORRECTED_V1))
        if operating_history != E_HISTORY_CORRECTED_V1:
            raise ValueError(
                "the BC CLI supports corrected-E operating history only; "
                f"checkpoint operating history is {operating_history!r}")
        stored_source = meta.get("source_e_identity") or {}
        if source_history is None:
            source_history = stored_source.get("history_version")
        if source_identity is None:
            source_identity = dict(meta.get("source_identity") or {})
        if provenance is None:
            provenance = dict(meta.get("provenance") or {})
        executor = dict(meta.get("executor") or {})

    train_step_fn = CompiledTrainStep(params, config)
    eval_step = CompiledEvalStep(config) if val_loaded is not None else None

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (checkpoint_dir / "metrics.jsonl"
                    if checkpoint_dir is not None else None)
    cli_args = dict(vars(args))

    def store(path: Path, *, kind: str, epoch: int,
              cursor: dict[str, Any]) -> None:
        _store(path, params=params, opt_state=opt_state, rng=rng,
               config=config, step=step, epoch=epoch, cursor=cursor,
               seed=args.seed, e_history_version=operating_history,
               source_history_version=source_history,
               source_identity=source_identity, provenance=provenance,
               executor=executor, kind=kind, cli_args=cli_args)

    remaining = args.steps
    cursor = {"epoch": start_epoch, "batch": start_batch, "seed": args.seed}
    best_val = math.inf
    best_epoch: int | None = None
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []

    for epoch_offset in range(args.epochs):
        if remaining is not None and remaining <= 0:
            break
        current_epoch = start_epoch + epoch_offset
        begin_batch = start_batch if epoch_offset == 0 else 0
        epoch_completed = True
        partial_batch: int | None = None
        train_rows = 0.0
        train_joint_sum = 0.0
        train_step_nll_sum: np.ndarray | None = None
        train_step_accuracy_sum: np.ndarray | None = None
        train_start = time.perf_counter()
        batches = iter_fixed_batches(
            train_inputs, train_actions, config.batch_size, seed=args.seed,
            epoch=current_epoch, shuffle=True, start_batch=begin_batch,
            row_ids=train_row_ids)
        for batch_index, batch in enumerate(batches):
            params, opt_state, rng, metrics = train_step_fn(
                params, opt_state, rng, batch)
            step += 1
            real = float(metrics["valid_rows"])
            train_rows += real
            train_joint_sum += float(metrics["loss"]) * real
            step_nll = np.asarray(metrics["per_step_nll"], np.float64) * real
            step_accuracy = np.asarray(
                metrics["per_step_accuracy"], np.float64) * real
            train_step_nll_sum = (step_nll if train_step_nll_sum is None
                                  else train_step_nll_sum + step_nll)
            train_step_accuracy_sum = (
                step_accuracy if train_step_accuracy_sum is None
                else train_step_accuracy_sum + step_accuracy)
            if args.steps is not None:
                print(
                    f"step={step} epoch={current_epoch + 1} "
                    f"batch={batch_index} loss={metrics['loss']:.6f} "
                    f"valid_rows={int(real)}", flush=True)
            if remaining is not None:
                remaining -= 1
                if remaining == 0:
                    partial_batch = batch_index + 1
                    epoch_completed = partial_batch >= num_batches
                    break
        train_wall = time.perf_counter() - train_start

        if not epoch_completed:
            cursor = {"epoch": current_epoch, "batch": int(partial_batch),
                      "seed": args.seed}
            break

        completed_epoch = current_epoch + 1
        if train_rows <= 0.0:
            raise ValueError("an epoch produced no real training rows")
        train_metrics = {
            "joint_nll": train_joint_sum / train_rows,
            "per_step_nll": (train_step_nll_sum / train_rows).tolist(),
            "per_step_accuracy": (
                train_step_accuracy_sum / train_rows).tolist(),
            "valid_rows": int(train_rows),
        }
        val_metrics = None
        val_wall = 0.0
        if val_loaded is not None:
            val_start = time.perf_counter()
            val_batches = iter_fixed_batches(
                val_loaded[0], val_loaded[1], config.batch_size, seed=args.seed,
                epoch=0, shuffle=False, row_ids=None)
            val_metrics = validation_metrics(
                params, val_batches, config, eval_step=eval_step)
            val_wall = time.perf_counter() - val_start
        became_best = False
        if val_metrics is not None:
            if val_metrics["joint_nll"] < best_val:
                best_val = val_metrics["joint_nll"]
                best_epoch = completed_epoch
                became_best = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        cursor = {"epoch": current_epoch + 1, "batch": 0, "seed": args.seed}
        checkpoint_start = time.perf_counter()
        if checkpoint_dir is not None:
            store(checkpoint_dir / f"epoch_{completed_epoch:03d}.npz",
                  kind="epoch", epoch=completed_epoch, cursor=cursor)
            store(checkpoint_dir / "last.npz",
                  kind="last", epoch=completed_epoch, cursor=cursor)
            if became_best:
                store(checkpoint_dir / "best.npz",
                      kind="best", epoch=completed_epoch, cursor=cursor)
        checkpoint_wall = time.perf_counter() - checkpoint_start

        epoch_total = train_wall + val_wall + checkpoint_wall
        record = {
            "version": "stage25_bc_epoch_metrics_v1",
            "epoch": completed_epoch,
            "epoch_index": current_epoch,
            "global_step": step,
            "train_joint_nll": train_metrics["joint_nll"],
            "val_joint_nll": (None if val_metrics is None
                              else val_metrics["joint_nll"]),
            "train_per_step_nll": train_metrics["per_step_nll"],
            "val_per_step_nll": (None if val_metrics is None
                                 else val_metrics["per_step_nll"].tolist()),
            "train_per_step_accuracy": train_metrics["per_step_accuracy"],
            "val_per_step_accuracy": (None if val_metrics is None
                                      else val_metrics["per_step_accuracy"].tolist()),
            "train_rows": train_metrics["valid_rows"],
            "validation_rows": (None if val_metrics is None
                                else val_metrics["valid_rows"]),
            "train_wall_seconds": train_wall,
            "validation_wall_seconds": val_wall,
            "checkpoint_wall_seconds": checkpoint_wall,
            "epoch_total_wall_seconds": epoch_total,
            "learning_rate": float(config.lr),
            "best": became_best,
            "best_epoch": best_epoch,
            "checkpoint": (None if checkpoint_dir is None
                           else str(checkpoint_dir
                                    / f"epoch_{completed_epoch:03d}.npz")),
        }
        records.append(record)
        if metrics_path is not None:
            _append_metrics(metrics_path, record)
        print(_summary(record), flush=True)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    if checkpoint_dir is not None:
        store(checkpoint_dir / "last.npz", kind="last",
              epoch=int(cursor["epoch"]), cursor=cursor)
    output = args.output
    if output is None and checkpoint_dir is None:
        output = Path("stage25_bc.npz")
    if output is not None:
        store(output, kind="final", epoch=int(cursor["epoch"]), cursor=cursor)
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
