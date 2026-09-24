"""Benchmark the complete Python FastEnv observation decode operation.

Examples:
    python tools/benchmark_fastenv_decode.py
    python tools/benchmark_fastenv_decode.py --iterations 500
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_env import BatchedFastEnv
from fast_env._reference import decode_observation_pair as reference_decode
from fast_env.api import _decode_observation_pair as optimized_decode


def _action(farmer=("PASS",), market=()):
    return {
        "farmer": list(farmer),
        "hands": [],
        "market": [list(order) for order in market],
    }


def _pair(farmer=("PASS",), market=()):
    return [_action(farmer, market), _action(farmer, market)]


def _move(current: tuple[int, int], target: tuple[int, int]):
    x, y = current
    target_x, target_y = target
    if x > target_x:
        return ("WEST",), (x - 1, y)
    if x < target_x:
        return ("EAST",), (x + 1, y)
    if y > target_y:
        return ("NORTH",), (x, y - 1)
    if y < target_y:
        return ("SOUTH",), (x, y + 1)
    return ("PASS",), current


def _populated_batch(num_envs: int) -> BatchedFastEnv:
    batch = BatchedFastEnv(
        num_envs,
        {
            "numThreads": 1,
            "startingMoney": 100000,
            "weedSpawnChance": 0.0,
        },
        canonical_observations=True,
    )
    batch.reset([7 + index * 13 for index in range(num_envs)])
    first_orders = [
        ["BUY_SEED", crop, 10]
        for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    ]
    first_orders.extend([
        ["BUY_ANIMAL", "GOOSE", 1],
        ["BUY_ANIMAL", "SHEEP", 1],
        ["HIRE"],
        ["HIRE"],
        ["BUY_PRODUCT", "WHEAT", 10],
    ])
    batch.step([_pair(("BUILD_COOP",), first_orders) for _ in range(num_envs)])
    batch.step([_pair(("PICKUP", "GOOSE", 1)) for _ in range(num_envs)])
    batch.step([_pair(("PLACE", "GOOSE", 1)) for _ in range(num_envs)])
    batch.step([_pair(("WEST",)) for _ in range(num_envs)])
    batch.step([_pair(("BUILD_PASTURE",)) for _ in range(num_envs)])
    batch.step([_pair(("EAST",)) for _ in range(num_envs)])
    batch.step([
        _pair(("PICKUP", "SHEEP", 1)) for _ in range(num_envs)
    ])
    batch.step([_pair(("WEST",)) for _ in range(num_envs)])
    batch.step([
        _pair(("PLACE", "SHEEP", 1)) for _ in range(num_envs)
    ])

    current = (3, 4)
    positions = [(2, 4), (1, 4), (0, 4), (0, 3)]
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    for index, target in enumerate(positions):
        while current != target:
            farmer, current = _move(current, target)
            batch.step([_pair(farmer) for _ in range(num_envs)])
        batch.step([
            _pair(("PLANT", crops[index])) for _ in range(num_envs)
        ])
        batch.step([_pair(("WATER",)) for _ in range(num_envs)])
    observation = batch.observations(0)[0]
    tiles = [
        tile
        for row in observation["farms"][0]["tiles"]
        for tile in row
    ]
    assert sum(isinstance(tile, dict) and tile.get("kind") == "PLANT" for tile in tiles) >= 4
    assert sum(isinstance(tile, dict) and "animal" in tile for tile in tiles) == 2
    assert len(observation["farms"][0]["hands"]) == 2
    assert observation["private"]["shed"]["WHEAT"] > 0
    return batch


def _measure_pair(
    raws: Sequence,
    configuration,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    decoders = {
        "reference": reference_decode,
        "optimized": optimized_decode,
    }
    for _ in range(50):
        for decoder in decoders.values():
            for raw in raws:
                decoder(raw, configuration, canonical_farms=True)
    samples = {name: [] for name in decoders}
    gc.disable()
    try:
        for sample_index in range(9):
            order = list(decoders)
            if sample_index % 2:
                order.reverse()
            for name in order:
                decoder = decoders[name]
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    for raw in raws:
                        decoder(raw, configuration, canonical_farms=True)
                samples[name].append(
                    (time.perf_counter_ns() - started)
                    / (iterations * len(raws) * 1000.0)
                )
    finally:
        gc.enable()
    return {
        name: (statistics.median(values), statistics.mean(values))
        for name, values in samples.items()
    }


def _measure_full_step(
    batch: BatchedFastEnv,
    decoder,
    iterations: int,
) -> tuple[float, float]:
    actions = [[_action(), _action()] for _ in range(batch.num_envs)]
    original_decode = batch._decode
    if decoder is reference_decode:
        def reference_batch_decode(environment):
            environment._observations = [
                reference_decode(
                    environment.observation_buffer[index],
                    environment.configuration,
                    canonical_farms=environment.canonical_observations,
                )
                for index in range(environment.num_envs)
            ]
            return environment._observations

        batch._decode = MethodType(reference_batch_decode, batch)
    try:
        for _ in range(25):
            batch.step(actions)
        samples = []
        gc.disable()
        try:
            for _ in range(9):
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    batch.step(actions)
                samples.append(
                    (time.perf_counter_ns() - started)
                    / (iterations * 1000.0)
                )
        finally:
            gc.enable()
    finally:
        batch._decode = original_decode
    return statistics.median(samples), statistics.mean(samples)


def run(iterations: int) -> None:
    for state_name, factory in (
        ("sparse", lambda n: BatchedFastEnv(
            n, {"numThreads": 1}, canonical_observations=True
        )),
        ("populated", _populated_batch),
    ):
        for batch_size in (1, 4):
            batch = factory(batch_size)
            if state_name == "sparse":
                batch.reset([7 + index * 13 for index in range(batch_size)])
            raws = [batch.observation_buffer[index] for index in range(batch_size)]
            measured = _measure_pair(raws, batch.configuration, iterations)
            reference_median, reference_mean = measured["reference"]
            optimized_median, optimized_mean = measured["optimized"]
            print(
                f"{state_name:9} B{batch_size}: "
                f"reference median={reference_median:.2f} us/env-turn "
                f"mean={reference_mean:.2f}; "
                f"optimized median={optimized_median:.2f} "
                f"mean={optimized_mean:.2f}; "
                f"speedup={reference_median / optimized_median:.2f}x"
            )

    reference_batch = BatchedFastEnv(
        4, {"numThreads": 1}, canonical_observations=True
    )
    optimized_batch = BatchedFastEnv(
        4, {"numThreads": 1}, canonical_observations=True
    )
    reference_batch.reset([7, 19, 42, 123])
    optimized_batch.reset([7, 19, 42, 123])
    reference_median, reference_mean = _measure_full_step(
        reference_batch, reference_decode, iterations
    )
    optimized_median, optimized_mean = _measure_full_step(
        optimized_batch, optimized_decode, iterations
    )
    print(
        "full BatchedFastEnv.step B4: "
        f"reference median={reference_median:.2f} us/batch-step "
        f"mean={reference_mean:.2f}; optimized median={optimized_median:.2f} "
        f"mean={optimized_mean:.2f}; "
        f"speedup={reference_median / optimized_median:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    run(args.iterations)


if __name__ == "__main__":
    main()
