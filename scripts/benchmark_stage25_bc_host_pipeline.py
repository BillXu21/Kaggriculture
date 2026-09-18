"""CPU-only benchmark for Stage 2.5 BC batch preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Make direct ``python scripts/...py`` execution use the repository packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_manager.stage25_bc import Stage25BCBatch, iter_fixed_batches

try:
    import psutil
except ImportError:  # pragma: no cover - optional benchmark enhancement
    psutil = None


def _inputs(rows: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
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


def _reference_batches(inputs, actions, batch_size, seed, epoch):
    n = len(inputs["board_kind"])
    order = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    rng.shuffle(order)
    shuffled = {key: value[order] for key, value in inputs.items()}
    labels = actions.astype(np.int32, copy=False)[order]
    for begin in range(0, n, batch_size):
        end = min(begin + batch_size, n)
        source = np.arange(begin, end, dtype=np.int64)
        real = len(source)
        if real < batch_size:
            source = np.pad(source, (0, batch_size - real), mode="edge")
        mask = np.zeros(batch_size, dtype=bool)
        mask[:real] = True
        yield Stage25BCBatch(
            {key: np.array(value[source], copy=True)
             for key, value in shuffled.items()},
            labels[source], mask, row_ids=order[source])


def _consume(batches) -> tuple[int, str]:
    rows = 0
    digest = hashlib.blake2b(digest_size=16)
    for batch in batches:
        real_ids = np.asarray(batch.row_ids)[batch.real_row_mask]
        rows += len(real_ids)
        digest.update(np.asarray(real_ids, dtype=np.int64).tobytes())
    return rows, digest.hexdigest()


def _measure(factory: Callable[[], object], rows: int, batch_size: int,
             repeats: int) -> dict:
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = _consume(factory())
        times.append(time.perf_counter() - start)
    wall = float(np.median(times))
    batch_count = (rows + batch_size - 1) // batch_size
    rss_mb = None
    peak_rss_mb = None
    if psutil is not None:
        memory = psutil.Process().memory_info()
        rss_mb = memory.rss / (1024 * 1024)
        peak = getattr(memory, "peak_wset", None)
        peak_rss_mb = (peak / (1024 * 1024)
                       if peak is not None else rss_mb)
    return {
        "wall_seconds": wall,
        "batches_per_sec": batch_count / wall,
        "rows_per_sec": rows / wall,
        "rows": result[0],
        "row_digest": result[1],
        "rss_after_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    inputs = _inputs(args.rows, args.seed)
    actions = _actions(args.rows)
    common = dict(seed=7, epoch=2, batch_size=args.batch_size)
    results = {
        "reference_full_reorder": _measure(
            lambda: _reference_batches(inputs, actions, **common),
            args.rows, args.batch_size, args.repeats),
        "index_only_sync": _measure(
            lambda: iter_fixed_batches(inputs, actions, host_workers=0,
                                       **common), args.rows, args.batch_size,
            args.repeats),
    }
    for workers in (1, 2, 3):
        results[f"index_only_prefetch_{workers}"] = _measure(
            lambda workers=workers: iter_fixed_batches(
                inputs, actions, host_workers=workers, prefetch_batches=6,
                **common), args.rows, args.batch_size, args.repeats)
    parity = results["reference_full_reorder"]["row_digest"] == results[
        "index_only_sync"]["row_digest"]
    print(json.dumps({
        "config": {"rows": args.rows, "batch_size": args.batch_size,
                   "repeats": args.repeats, "seed": args.seed},
        "exact_row_order_digest_parity": parity,
        "results": results,
    }, indent=2, sort_keys=True))
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
