"""Honest throughput benchmark / Kaggle-ready CLI for the JAX manager.

Measures inference and train-step throughput for the 106-token (own-only)
and 206-token (opponent PUBLIC board) regimes, on 1..N local devices via
the replicated data-parallel path (`bc_manager_jax.sharding`).

Honesty rules enforced here:

- every timed result is synchronized with `.block_until_ready()` on all
  output leaves before `perf_counter` stops;
- compilation cost is reported separately from steady state;
- unavailable device counts, non-divisible batches, and OOM/failures are
  recorded as explicit `skipped` entries — never invented;
- local CPU numbers are plumbing smoke values, NOT TPU throughput;
- f32 is the default and parity-preserving mode; optional bf16 casts
  floating parameter/input leaves and is labeled as such in every row.

Checkpoint mode strictly converts an existing PyTorch
`bc_manager_checkpoint_v1` file; its stored config determines the token
regime and its stored top-level `model_variant` (V0/E only; J/JE are
rejected loudly) determines the input contract. Every row records
`model_variant` additively. A missing checkpoint is a hard error, never
silently ignored. Random-model mode benchmarks both regimes separately.

Example Kaggle TPU command (8 devices, f32, real promoted BC-E checkpoint):

    python -m bc_manager_jax.benchmark \
        --device-counts 8 --dtype f32 \
        --checkpoint /kaggle/working/bc-v1-E/best.pt \
        --batch-sizes 256,512,1024,2048,4096 \
        --warmup 3 --iterations 10 \
        --output-json /kaggle/working/bc_e_jax_benchmark.json \
        --output-csv /kaggle/working/bc_e_jax_benchmark.csv

Because a checkpoint fixes one regime, cover both token regimes with
random models via a second run:

    python -m bc_manager_jax.benchmark --device-counts 8 \
        --model-config default --regimes both \
        --output-json /kaggle/working/bc_jax_benchmark_random.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from bc_manager.economics import ECONOMIC_DIM, economic_context
from bc_manager_jax.checkpoint import load_torch_checkpoint
from bc_manager_jax.model import (
    ECONOMIC_CONTEXT_KEY,
    forward,
    init_params,
    ManagerConfig,
    resolve_model_variant,
    tiny_manager_config,
)
from bc_manager_jax.train import TrainConfig, init_opt_state, train_step
from bc_manager_jax import sharding

DEFAULT_BATCH_SIZES = (256, 512, 1024, 2048, 4096)


# ------------------------------------------------------------- metadata


def param_count(params) -> int:
    return int(sum(int(np.asarray(leaf).size)
                   for leaf in jax.tree_util.tree_leaves(params)))


def describe_environment(requested_device_counts: list[Any]) -> dict[str, Any]:
    devices = jax.devices()
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "platform_name": platform.platform(),
        "process_count": jax.process_count(),
        "device_count_visible": len(devices),
        "device_descriptions": [d.device_kind for d in devices],
        "requested_device_counts": [str(n) for n in requested_device_counts],
        "sync_policy": "block_until_ready on all output leaves per timed "
                       "iteration; perf_counter timing",
        "reduction_policy": "full-global-batch jnp.mean reductions; GSPMD "
                            "cross-replica sums verified by "
                            "tests/test_bc_manager_jax_train.py",
        "honesty_note": "local CPU results are plumbing smoke values, not "
                        "TPU throughput",
    }


# ------------------------------------------------------ synthetic batches


def synthetic_batch(config: ManagerConfig, global_batch: int,
                    seed: int = 0,
                    model_variant: str = "V0") -> tuple[dict[str, np.ndarray],
                                                        dict[str, np.ndarray]]:
    """Deterministic shape-correct NumPy batch (adapter-array layout).

    Variant E rows carry an `economic_context` [B, 14] block built from the
    authoritative `bc_manager.economics.economic_context` function (never a
    re-derived formula).
    """
    variant = resolve_model_variant(model_variant)
    rng = np.random.default_rng(seed)
    b = global_batch

    def ints(shape, low, high, dtype=np.int32):
        return rng.integers(low, high, size=shape).astype(dtype)

    day = ints((b,), 0, 30, np.int16)
    inputs = {
        "board_kind": ints((b, 100), 0, 8, np.int16),
        "board_crop": ints((b, 100), 0, 7, np.int8),
        "board_animal": ints((b, 100), 0, 5, np.int8),
        "board_numeric":
            (rng.standard_normal((b, 100, 11)) * 3).astype(np.float32),
        "board_bool": ints((b, 100, 8), 0, 2, np.uint8),
        "board_mask": ints((b, 100, 4), 0, 2, np.uint8),
        "scalars": (rng.random((b, 4)) * 1000 + 10).astype(np.float32),
        "shed_counts": ints((b, 12), 0, 10),
        "seed_counts": ints((b, 5), 0, 10),
        "carried_counts": ints((b, 12), 0, 10),
        "unlocked": ints((b, 4), 0, 2, np.uint8),
        "market_inventory": ints((b, 9), 0, 20),
        "market_prices": (rng.random((b, 9)) * 50).astype(np.float32),
        "shop_counts": ints((b, 9), 0, 5),
        "day": day,
        "days_remaining": (29 - day.astype(np.int64)).astype(np.int16),
    }
    if config.include_opponent_board:
        for key in ("board_kind", "board_crop", "board_animal",
                    "board_numeric", "board_bool", "board_mask"):
            inputs[f"opp_{key}"] = {
                "board_kind": ints((b, 100), 0, 8, np.int16),
                "board_crop": ints((b, 100), 0, 7, np.int8),
                "board_animal": ints((b, 100), 0, 5, np.int8),
                "board_numeric":
                    (rng.standard_normal((b, 100, 11)) * 3).astype(np.float32),
                "board_bool": ints((b, 100, 8), 0, 2, np.uint8),
                "board_mask": ints((b, 100, 4), 0, 2, np.uint8),
            }[key]
        inputs["opp_scalars"] = (rng.random((b, 2)) * 500).astype(np.float32)
        inputs["opp_unlocked"] = ints((b, 4), 0, 2, np.uint8)

    if variant == "E":
        money = inputs["scalars"][:, 0].astype(np.float64)
        unlocked_counts = np.clip(
            inputs["unlocked"].sum(axis=1), 1, 4).astype(int)
        rows = np.zeros((b, ECONOMIC_DIM), dtype=np.float32)
        for i in range(b):
            # Deterministic mix of valid/invalid previous-cash channels.
            prev = float(money[i] - 100.0) if i % 2 == 0 else None
            rows[i] = economic_context(float(money[i]),
                                       int(unlocked_counts[i]), prev)
        inputs[ECONOMIC_CONTEXT_KEY] = rows

    c = config.count_classes
    presence = np.zeros((b, 9, 6), dtype=np.float32)
    flat = presence.reshape(-1)
    flat[:6] = 1.0
    targets = {
        "crop_target": ints((b, 5), 0, c),
        "animal_target": ints((b, 3), 0, c),
        "land_count": ints((b,), 1, 5),
        "fertilizer_target": ints((b, 5), 0, c),
        "care_target": ints((b, 3), 0, c),
        "sell_presence": presence,
        "sell_quantity_log1p":
            np.log1p(presence * rng.integers(1, 60, flat.shape)
                     .reshape(presence.shape).astype(np.float32)),
    }
    return inputs, targets


# ---------------------------------------------------------------- timing


def _block_all(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()


def _cast_float_leaves(tree, dtype):
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(dtype)
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(
            leaf.dtype, jax.numpy.floating) else leaf,
        tree)


def time_callable(fn, *, warmup: int, iterations: int) -> tuple[float, list[float]]:
    """Compile-time (first blocking call) then steady-state per-call times."""
    started = time.perf_counter()
    _block_all(fn())
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        _block_all(fn())
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        _block_all(fn())
        times.append(time.perf_counter() - started)
    return compile_seconds, times


# ------------------------------------------------------------- benchmark


def _resolve_regime_configs(args) -> list[tuple[str, ManagerConfig, Any, str, str]]:
    """Returns [(regime_label, config, params_or_None, source, variant)]."""
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise ValueError(
                f"--checkpoint {args.checkpoint} does not exist; refusing "
                f"to benchmark anything else silently")
        params, meta = load_torch_checkpoint(path)
        config = ManagerConfig(**meta["model_config"])
        variant = meta.get("model_variant", "V0")
        regime = "opponent" if config.include_opponent_board else "own"
        return [(regime, config, params,
                 f"torch checkpoint {path} (bc_manager_checkpoint_v1, "
                 f"variant={variant}, epoch={meta.get('epoch')})", variant)]
    base = tiny_manager_config() if args.model_config == "tiny" \
        else ManagerConfig()
    variant = resolve_model_variant(getattr(args, "variant", "V0"))
    configs = []
    regimes = {"own": ("own", False), "opponent": ("opponent", True)}
    if args.regimes == "own":
        regimes = {"own": ("own", False)}
    elif args.regimes == "opponent":
        regimes = {"opponent": ("opponent", True)}
    for label, include_opp in regimes.values():
        config = dataclasses.replace(base, include_opponent_board=include_opp)
        configs.append((label, config, None,
                        f"random init ({args.model_config})", variant))
    return configs


def run_case(regime: str, config: ManagerConfig, params, source: str,
             device_count: int, global_batch: int, args,
             model_variant: str = "V0") -> dict[str, Any]:
    """One (regime, device-count, batch) cell; never raises.

    Successful rows carry BOTH metric families under stable, unambiguous
    field names (`inference_*` and `train_*`); failed/skipped rows keep
    them as nulls plus an honest `reason`.
    """
    variant = resolve_model_variant(model_variant)
    row: dict[str, Any] = {
        "regime": regime,
        "model_variant": variant,
        "token_count": config.token_count,
        "param_count": None,
        "source": source,
        "model_config": asdict(config),
        "device_count": device_count,
        "global_batch": global_batch,
        "per_device_batch": global_batch // device_count
        if global_batch % device_count == 0 else None,
        "dtype_mode": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "status": "ok",
        "reason": None,
        "inference_compile_seconds": None,
        "inference_examples_per_second_mean": None,
        "inference_examples_per_second_best": None,
        "train_compile_seconds": None,
        "train_examples_per_second_mean": None,
        "train_examples_per_second_best": None,
    }
    try:
        if global_batch % device_count != 0:
            raise ValueError(
                f"global batch {global_batch} not divisible by "
                f"{device_count} devices")
        mesh = sharding.create_data_mesh(device_count)
        base_params = params if params is not None \
            else init_params(config, seed=args.seed, model_variant=variant)
        row["param_count"] = param_count(base_params)

        dtype = jax.numpy.bfloat16 if args.dtype == "bf16" \
            else jax.numpy.float32
        if args.dtype == "bf16":
            base_params = _cast_float_leaves(base_params, dtype)
        live_params = sharding.replicate_tree(base_params, mesh)

        inputs_np, targets_np = synthetic_batch(config, global_batch,
                                                seed=args.seed,
                                                model_variant=variant)
        if args.dtype == "bf16":
            inputs_np = {k: (v.astype(np.dtype(dtype))
                             if v.dtype.kind == "f" else v)
                         for k, v in inputs_np.items()}
            targets_np = {k: (v.astype(np.dtype(dtype))
                              if v.dtype.kind == "f" else v)
                          for k, v in targets_np.items()}
        sharded_inputs = sharding.shard_batch(inputs_np, mesh)
        sharded_targets = sharding.shard_batch(targets_np, mesh)

        rng = jax.random.PRNGKey(args.seed)
        train_config = TrainConfig()

        # ---- inference
        def infer():
            return forward(live_params, sharded_inputs, config,
                           model_variant=variant)

        compile_s, times = time_callable(infer, warmup=args.warmup,
                                         iterations=args.iterations)
        row["inference_compile_seconds"] = compile_s
        row["inference_examples_per_second_mean"] = float(
            global_batch / float(np.mean(times)))
        row["inference_examples_per_second_best"] = float(
            global_batch / min(times))

        # ---- train step
        opt_state = sharding.replicate_tree(
            init_opt_state(base_params, train_config), mesh)

        def step():
            return train_step(live_params, opt_state, rng, sharded_inputs,
                              sharded_targets, config, train_config,
                              variant)

        compile_s, times = time_callable(step, warmup=args.warmup,
                                         iterations=args.iterations)
        row["train_compile_seconds"] = compile_s
        row["train_examples_per_second_mean"] = float(
            global_batch / float(np.mean(times)))
        row["train_examples_per_second_best"] = float(
            global_batch / min(times))
    except Exception as error:  # noqa: BLE001 - honesty: record, don't invent
        row["status"] = "skipped"
        row["reason"] = f"{type(error).__name__}: {error}"
    return row


def run_benchmark(args) -> dict[str, Any]:
    visible = len(jax.devices())
    requested = (["all"] if n == "all" else n
                 for n in args.device_counts.split(",") if n.strip())
    requested_list: list[Any] = []
    device_counts: list[int] = []
    for item in requested:
        if item == "all":
            requested_list.append("all")
            device_counts.append(visible)
        else:
            n = int(item)
            requested_list.append(n)
            device_counts.append(n)

    report: dict[str, Any] = {
        "metadata": describe_environment(requested_list),
        "results": [],
    }
    cases = _resolve_regime_configs(args)
    for regime, config, params, source, variant in cases:
        for device_count in sorted(set(device_counts)):
            for global_batch in args.batch_sizes:
                print(f"[benchmark] regime={regime} variant={variant} "
                      f"devices={device_count} "
                      f"batch={global_batch} ...", flush=True)
                row = run_case(regime, config, params, source, device_count,
                               global_batch, args, model_variant=variant)
                report["results"].append(row)
                if row["status"] == "ok":
                    print(f"    ok: inference "
                          f"{row['inference_examples_per_second_mean']:.1f} "
                          f"ex/s (compile "
                          f"{row['inference_compile_seconds']:.2f}s) | "
                          f"train "
                          f"{row['train_examples_per_second_mean']:.1f} "
                          f"ex/s (compile "
                          f"{row['train_compile_seconds']:.2f}s)")
                else:
                    print(f"    skipped: {row['reason']}")
    return report


def _write_outputs(report: dict[str, Any], json_path: str | None,
                   csv_path: str | None) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(report, indent=2),
                                   encoding="utf-8")
        print(f"[benchmark] JSON written to {json_path}")
    if csv_path:
        fieldnames = ["regime", "model_variant", "token_count",
                      "param_count",
                      "device_count", "global_batch", "per_device_batch",
                      "dtype_mode", "status",
                      "inference_compile_seconds",
                      "inference_examples_per_second_mean",
                      "inference_examples_per_second_best",
                      "train_compile_seconds",
                      "train_examples_per_second_mean",
                      "train_examples_per_second_best", "reason"]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["results"])
        print(f"[benchmark] CSV written to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_manager_jax.benchmark",
        description="Honest inference/train throughput benchmark for the "
                    "JAX daily-manager Transformer (106-token own-only and "
                    "206-token opponent regimes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Kaggle TPU examples (f32, 8 devices):

  # Real promoted BC-E checkpoint (issue #8 target; its stored config fixes
  # ONE token regime and its stored variant must be E or V0):
  python -m bc_manager_jax.benchmark --device-counts 8 --dtype f32 \\
      --checkpoint /kaggle/working/bc-v1-E/best.pt \\
      --batch-sizes 256,512,1024,2048,4096 --warmup 3 --iterations 10 \\
      --output-json /kaggle/working/bc_e_jax_benchmark.json \\
      --output-csv /kaggle/working/bc_e_jax_benchmark.csv

  # Random-model coverage of BOTH regimes separately:
  python -m bc_manager_jax.benchmark --device-counts 8 \\
      --model-config default --regimes both \\
      --output-json /kaggle/working/bc_jax_benchmark_random.json

Local CPU smoke (plumbing only, NOT representative throughput):
  python -m bc_manager_jax.benchmark --model-config tiny \\
      --device-counts 1 --batch-sizes 32 --warmup 1 --iterations 3
""")
    parser.add_argument("--model-config", choices=("tiny", "default"),
                        default="default",
                        help="random-model architecture (default: %(default)s)")
    parser.add_argument("--variant", choices=("V0", "E"), default="V0",
                        help="random-model model_variant; checkpoint mode "
                             "always uses the stored variant instead "
                             "(default: %(default)s)")
    parser.add_argument("--regimes", choices=("own", "opponent", "both"),
                        default="both",
                        help="token regimes for random models "
                             "(default: %(default)s)")
    parser.add_argument("--checkpoint", default=None,
                        help="PyTorch bc_manager_checkpoint_v1 path; strictly "
                             "converted, its config fixes the regime; missing "
                             "file is a hard error")
    parser.add_argument("--device-counts", default="all",
                        help="comma list, e.g. '1,8' or 'all' "
                             "(default: %(default)s)")
    parser.add_argument("--batch-sizes", default=",".join(
        map(str, DEFAULT_BATCH_SIZES)),
        help="comma list of global batches, each divisible by every "
             "requested device count (default: %(default)s)")
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32",
                        help="f32 preserves parity; bf16 casts floating "
                             "leaves and is labeled per row "
                             "(default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")
                        if x.strip()]
    if not args.batch_sizes:
        raise ValueError("--batch-sizes produced no batches")
    report = run_benchmark(args)
    report["metadata"]["cli_args"] = {
        "model_config": args.model_config,
        "variant": args.variant,
        "regimes": args.regimes,
        "checkpoint": args.checkpoint,
        "batch_sizes": args.batch_sizes,
        "device_counts": args.device_counts,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    _write_outputs(report, args.output_json, args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
