"""Command-line entry point for BC training/evaluation.

Example (tiny CPU validation run on synthetic data):

    python -m bc_manager.cli data.parquet --tiny --epochs 2 \
        --train-dates 2026-08-17,2026-08-18 --val-dates 2026-08-21 \
        --checkpoint-dir data/temp/bc-checkpoints
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .adapter import SchemaVersionError
from .constants import MIN_SCORE_DEFAULT, TRAIN_DATES_DEFAULT, VAL_DATES_DEFAULT
from .economics import E_HISTORY_CORRECTED_V1, E_HISTORY_VERSIONS
from .model import ManagerConfig, tiny_manager_config
from .training import TrainingConfig, run_training


def _dates(value: str) -> tuple[str, ...]:
    return tuple(d.strip() for d in value.split(",") if d.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bc_manager.cli",
        description="Train/evaluate the daily-manager BC Transformer on "
                    "schema-v3 canonical daily Parquet.")
    parser.add_argument("parquet", nargs="+", help="schema-v3 Parquet path(s)")
    parser.add_argument("--train-dates", type=_dates,
                        default=tuple(TRAIN_DATES_DEFAULT),
                        help="comma-separated train partition dates")
    parser.add_argument("--val-dates", type=_dates,
                        default=tuple(VAL_DATES_DEFAULT),
                        help="comma-separated validation partition dates")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE_DEFAULT)
    # model configuration
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--count-max", type=int, default=100)
    parser.add_argument("--include-opponent-board", action="store_true")
    parser.add_argument("--variant", choices=("V0", "J", "E", "JE"),
                        default="V0",
                        help="model variant: V0 (current inputs), J (joint "
                             "plan decoder), E (economic context), or JE "
                             "(both; issue #6)")
    parser.add_argument("--e-history-version", choices=E_HISTORY_VERSIONS,
                        default=E_HISTORY_CORRECTED_V1,
                        help="E history semantics; legacy is compatibility-only")
    parser.add_argument("--tiny", action="store_true",
                        help="16/1/1/32/dropout=0 CPU validation config")
    # training configuration
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=None,
                        help="early-stopping patience on validation total")
    parser.add_argument("--amp", choices=("auto", "on", "off"),
                        default="auto",
                        help="CUDA AMP: auto enables only with CUDA device")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--no-last", action="store_true",
                        help="skip the optional last checkpoint")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="directory for best.pt / last.pt "
                             "(use an ignored path such as data/temp)")
    return parser


def resolve_model_config(args: argparse.Namespace) -> ManagerConfig:
    if args.tiny:
        return tiny_manager_config(
            count_max=args.count_max,
            include_opponent_board=args.include_opponent_board)
    return ManagerConfig(
        d_model=args.d_model, num_layers=args.layers, num_heads=args.heads,
        ffn_dim=args.ffn, dropout=args.dropout, count_max=args.count_max,
        include_opponent_board=args.include_opponent_board)


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    use_amp = {"auto": None, "on": True, "off": False}[args.amp]
    return TrainingConfig(
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        gradient_clip=args.gradient_clip, seed=args.seed,
        num_workers=args.num_workers, use_amp=use_amp,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir, save_last=not args.no_last)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model_config = resolve_model_config(args)
        training_config = resolve_training_config(args)
        result = run_training(
            args.parquet, model_config=model_config,
            training_config=training_config, train_dates=args.train_dates,
            val_dates=args.val_dates, min_score=args.min_score,
            device_spec=args.device, model_variant=args.variant,
            e_history_version=args.e_history_version)
    except SchemaVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if result["best_epoch"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
