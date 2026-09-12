"""Small bounded CLI for native Stage 2.5 teacher-forced BC."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    normalize_e_history_version,
)
from rl_manager.stage25_bc import (
    Stage25BCConfig,
    import_encoder_checkpoint,
    init_opt_state,
    iter_fixed_batches,
    load_array_dataset,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from rl_manager.stage25_checkpoint import Stage25CheckpointError
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model-size", choices=("tiny", "small", "large"), default="tiny")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("stage25_bc.npz"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--import", dest="import_path", type=Path)
    parser.add_argument(
        "--allow-legacy-e", action="store_true",
        help="allow a legacy E-history source encoder import (source only; "
             "Stage 2.5 operating history stays corrected)")
    parser.add_argument("--date", dest="dates", action="append")
    parser.add_argument("--min-score", type=float, default=2950.0)
    return parser


def _model(size: str) -> Stage25ModelConfig:
    return {"tiny": Stage25ModelConfig.tiny,
            "small": Stage25ModelConfig.small,
            "large": Stage25ModelConfig.large}[size]()


def _load(args: argparse.Namespace):
    if args.data:
        if args.data.suffix.lower() == ".parquet":
            from rl_manager.stage25_adapter import load_dataset
            kwargs = {"min_score": args.min_score}
            if args.dates:
                kwargs["dates"] = tuple(args.dates)
            loaded = load_dataset(args.data, **kwargs)
            return loaded["inputs"], loaded["actions"], loaded.get("row_ids")
        inputs, actions = load_array_dataset(args.data)
        return inputs, actions, None
    if not args.inputs or not args.labels:
        raise ValueError("provide --data or both --inputs and --labels")
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {key: np.array(archive[key], copy=True) for key in archive.files}
    labels = np.load(args.labels, allow_pickle=False)
    if isinstance(labels, np.lib.npyio.NpzFile):
        with labels as archive:
            labels = np.array(archive["actions" if "actions" in archive else "labels"], copy=True)
    return inputs, labels, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 0 or args.epochs < 0:
        raise ValueError("steps and epochs must be nonnegative")
    inputs, actions, row_ids = _load(args)
    model = _model(args.model_size)
    config = Stage25BCConfig(model=model, batch_size=args.batch_size,
                             lr=args.lr, weight_decay=args.weight_decay)
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
    epoch = 0
    resume_batch = 0
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
        epoch = int(meta.get("epoch", 0))
        cursor = meta.get("data_order_position", {})
        if isinstance(cursor, dict):
            epoch = int(cursor.get("epoch", epoch))
            resume_batch = int(cursor.get("batch", 0))
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
    remaining = args.steps
    position = {"epoch": epoch, "batch": resume_batch}
    for epoch_offset in range(args.epochs):
        current_epoch = epoch + epoch_offset
        start_batch = resume_batch if epoch_offset == 0 else 0
        epoch = current_epoch
        for batch_index, batch in enumerate(iter_fixed_batches(
                inputs, actions, config.batch_size, seed=args.seed,
                epoch=current_epoch, start_batch=start_batch,
                row_ids=row_ids)):
            if remaining == 0:
                break
            params, opt_state, rng, metrics = train_step(
                params, opt_state, rng, batch, config)
            step += 1
            remaining -= 1
            position = {"epoch": current_epoch,
                        "batch": start_batch + batch_index + 1}
            print(f"step={step} epoch={current_epoch} loss={metrics['loss']:.6f} "
                  f"valid_rows={int(metrics['valid_rows'])}", flush=True)
        if remaining == 0:
            break
    save_checkpoint(args.output, params, opt_state, rng, config=config,
                    step=step, epoch=epoch, seed=args.seed,
                    shuffle_state=position,
                    e_history_version=operating_history,
                    source_history_version=source_history,
                    source_identity=source_identity,
                    provenance=provenance,
                    executor=executor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
