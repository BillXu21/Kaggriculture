"""Measure Stage 2.5 row-token and RNG preparation before/after P2.

The standalone measurements use the P1-equivalent primitives explicitly so
they remain useful even though the optimized adapter no longer exposes the
old dispatch boundary.  Adapter timings warm both compiled paths first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np

from rl_manager.stage25_inference import (
    Stage25InferenceAdapter, _normalise_row_ids, _normalise_row_tokens,
    _root_key, _row_rng_keys,
)
from rl_manager.stage25_mechanics import PhysicalContext
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_types import stage25_row_token


def _inputs(batch: int) -> dict[str, np.ndarray]:
    return {
        "board_kind": np.zeros((batch, 100), dtype=np.int16),
        "board_crop": np.zeros((batch, 100), dtype=np.int8),
        "board_animal": np.zeros((batch, 100), dtype=np.int8),
        "board_numeric": np.zeros((batch, 100, 11), dtype=np.float32),
        "board_bool": np.zeros((batch, 100, 8), dtype=bool),
        "board_mask": np.zeros((batch, 100, 4), dtype=np.uint8),
        "scalars": np.zeros((batch, 4), dtype=np.float32),
        "shed_counts": np.zeros((batch, 12), dtype=np.int32),
        "seed_counts": np.zeros((batch, 5), dtype=np.int32),
        "carried_counts": np.zeros((batch, 12), dtype=np.int32),
        "unlocked": np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8),
                             (batch, 1)),
        "market_inventory": np.zeros((batch, 9), dtype=np.int32),
        "market_prices": np.zeros((batch, 9), dtype=np.float32),
        "shop_counts": np.zeros((batch, 9), dtype=np.int32),
        "day": np.zeros((batch,), dtype=np.int16),
        "days_remaining": np.full((batch,), 29, dtype=np.int16),
        "economic_context": np.zeros((batch, 14), dtype=np.float32),
        "crop_capacity": np.zeros((batch, 5), dtype=np.int16),
    }


CONTEXT = PhysicalContext(1, (25, 25, 25, 25), (0, 0, 0), 0, 0, (0, 0, 0))


def _elapsed(calls: int, function) -> float:
    started = time.perf_counter()
    for _ in range(calls):
        function()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--rows", type=int, default=26624)
    args = parser.parse_args()

    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=7)
    adapter = Stage25InferenceAdapter(params=params, config=config, seed=31)
    identity = adapter.identity.identity_id()
    ids = tuple(
        f"episode={index}/seat={index % 2}/day={4 + index % 28}/"
        f"behavior={identity}" for index in range(32))
    tokens = np.asarray([stage25_row_token(value) for value in ids],
                        dtype=np.int32)
    root = _root_key("bench", adapter.identity, adapter.seed)
    jax.block_until_ready(_row_rng_keys(root, tokens))

    real_token_seconds = _elapsed(
        1, lambda: [stage25_row_token(
            f"episode={index}/seat={index % 2}/day={4 + index % 28}/"
            f"behavior={identity}") for index in range(args.rows)])
    old_token_physical_seconds = _elapsed(
        1300, lambda: _normalise_row_ids(ids, 32))
    old_root_seconds = _elapsed(
        1300, lambda: _root_key("bench", adapter.identity, adapter.seed))
    old_fold_seconds = _elapsed(
        1300, lambda: jax.block_until_ready(_row_rng_keys(root, tokens)))

    canonical = adapter._cached_root_key("bench")
    del canonical
    new_prep_seconds = _elapsed(
        1300, lambda: (
            _normalise_row_tokens(tokens.tolist(), 32),
            adapter._cached_root_key("bench")))

    padding = {}
    for real in (8, 20, 32):
        old_ids = tuple(
            ids[:real] + tuple(f"padding/slot={slot}"
                               for slot in range(32 - real)))
        padded_tokens = np.concatenate(
            (tokens[:real], np.repeat(tokens[0], 32 - real)))
        old = _elapsed(args.calls, lambda: _normalise_row_ids(old_ids, 32))
        new = _elapsed(
            args.calls,
            lambda: _normalise_row_tokens(padded_tokens.tolist(), 32))
        padding[str(real)] = {
            "old_equivalent_seconds": old,
            "new_seconds": new,
            "speedup": old / new if new else float("inf"),
        }

    inputs = _inputs(32)
    contexts = (CONTEXT,) * 32
    # Warm compilation for the optimized root+token path and for the direct
    # compatibility path.  Neither timing below includes these calls.
    adapter.infer_batch(inputs, physical_contexts=contexts, row_ids=ids,
                        prng_id="bench", row_tokens=tokens)
    adapter.infer_batch(inputs, physical_contexts=contexts, row_ids=ids,
                        prng_id="bench", row_tokens=tokens)
    before = dict(adapter.inference_phase_seconds)
    for _ in range(args.calls):
        adapter.infer_batch(inputs, physical_contexts=contexts, row_ids=ids,
                            prng_id="bench", row_tokens=tokens)
    phases = {
        name: adapter.inference_phase_seconds[name] - before[name]
        for name in before
    }
    adapter_total = phases["adapter_total_seconds"]
    print(json.dumps({
        "calls": args.calls,
        "physical_batch": 32,
        "old_equivalent": {
            "real_token_seconds": real_token_seconds,
            "physical_token_1300_seconds": old_token_physical_seconds,
            "root_1300_seconds": old_root_seconds,
            "standalone_fold_1300_seconds": old_fold_seconds,
            "row_rng_prepare_seconds": (
                old_token_physical_seconds + old_root_seconds + old_fold_seconds),
        },
        "optimized": {
            "token_root_prepare_1300_seconds": new_prep_seconds,
            "adapter_phases": phases,
            "fold_in_location": "compiled_policy_call",
        },
        "speedups": {
            "standalone_preparation": (
                (old_token_physical_seconds + old_root_seconds + old_fold_seconds) /
                new_prep_seconds),
            "adapter_calls_per_second": (
                args.calls / adapter_total if adapter_total else float("inf")),
        },
        "padding": padding,
    }, indent=2))


if __name__ == "__main__":
    main()
