"""Warm B32 benchmark for Stage 2.5 parent validation modes.

This intentionally uses fixed synthetic inputs and excludes the first call for
each adapter from the reported sample so JAX compilation is not presented as
steady-state inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_mechanics import (
    PhysicalContext, animal_target_support_mask, crop_delta_support_mask,
    land_target_support_mask, physical_crop_capacity,
)
from rl_manager.stage25_policy import (
    init_stage25_params, large_stage25_config, small_stage25_config,
    tiny_stage25_config,
)


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


def _context() -> PhysicalContext:
    return PhysicalContext(1, (25, 25, 25, 25), (0, 0, 0), 0, 0, (0, 0, 0))


def _support(context: PhysicalContext) -> dict[str, object]:
    capacity = max(physical_crop_capacity(
        context, context.observed_land, context.placed_animals), 0)
    return {
        "land": list(land_target_support_mask(context.observed_land)),
        "animals": [list(animal_target_support_mask(
            context, context.observed_land, species,
            context.placed_animals[:species])) for species in range(3)],
        "crops": [list(crop_delta_support_mask(0, capacity)) for _ in range(5)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("tiny", "small", "large"),
                        default="tiny")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 1:
        raise SystemExit("batch-size, warmup, and repeats must be positive")

    config = {"tiny": tiny_stage25_config,
              "small": small_stage25_config,
              "large": large_stage25_config}[args.model_size]()
    params = init_stage25_params(config, seed=17)
    inputs = _inputs(args.batch_size)
    contexts = tuple(_context() for _ in range(args.batch_size))
    supports = tuple(_support(contexts[0]) for _ in range(args.batch_size))
    row_ids = tuple(f"benchmark-{index}" for index in range(args.batch_size))
    for mode in ("strict", "fast", "none"):
        adapter = Stage25InferenceAdapter(
            params=params, config=config, validation_mode=mode, seed=23)
        for _ in range(args.warmup):
            adapter.infer_batch(
                inputs, physical_contexts=contexts, supports=supports,
                row_ids=row_ids, prng_id="benchmark")
        before = dict(adapter.inference_phase_seconds)
        wall_started = time.perf_counter()
        for _ in range(args.repeats):
            adapter.infer_batch(
                inputs, physical_contexts=contexts, supports=supports,
                row_ids=row_ids, prng_id="benchmark")
        wall_ms = (time.perf_counter() - wall_started) * 1000.0 / args.repeats
        phases = {
            name: (adapter.inference_phase_seconds[name] - before[name])
            * 1000.0 / args.repeats
            for name in adapter.inference_phase_seconds
        }
        other = phases["adapter_total_seconds"] - phases[
            "support_validation_seconds"] - phases["policy_call_seconds"]
        print(
            f"{mode:6s} total_ms/call={wall_ms:9.3f} "
            f"support_ms={phases['support_validation_seconds']:9.3f} "
            f"policy_ms={phases['policy_call_seconds']:9.3f} "
            f"other_host_ms={other:9.3f} "
            f"adapter_ms={phases['adapter_total_seconds']:9.3f}")


if __name__ == "__main__":
    main()
