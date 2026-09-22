"""Measure Stage 2.5 strict diagnostic versus compact request transport.

This benchmark intentionally stops at Python object construction and pickle
transport.  It does not initialize a policy or consume RNG, so it isolates the
support payload removed by the hot-path packet.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rl_manager.parallel_protocol import Stage25InferenceRequest, Stage25RequestIdentity
from rl_manager.stage25_inference import (
    _validate_context_consistency,
    _validate_support_shapes,
)
from rl_manager.stage25_mechanics import PhysicalContext
from rl_manager.stage25_provider import Stage25PlanProvider
from rl_manager.stage25_types import Stage25BehaviorIdentity


SUPPORT_ENTRIES_PER_ROW = 4 + 3 * 101 + 5 * 201


def _identity() -> Stage25BehaviorIdentity:
    return Stage25BehaviorIdentity(
        name="stage25-benchmark", version="v1",
        parameter_fingerprint="f" * 64,
        observation_schema_version="e_v1",
        policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1",
        curriculum_version="stage25_curriculum_v1",
        curriculum_fingerprint="c" * 64)


def _context() -> PhysicalContext:
    return PhysicalContext(
        observed_land=1, crop_build_cells_by_land=(25, 25, 25, 25),
        placed_animals=(0, 0, 0), unplaced_animals=(0, 0, 0))


def _inputs(batch_size: int) -> dict[str, np.ndarray]:
    return {
        "unlocked": np.ones((batch_size, 4), dtype=np.uint8),
        "board_animal": np.zeros((batch_size, 100), dtype=np.int8),
    }


def _requests(batch_size: int, mode: str) -> tuple[tuple[Stage25InferenceRequest, ...], tuple[dict | None, ...], float]:
    context = _context()
    capacity = (0, 0, 0, 0, 0)
    started = time.perf_counter()
    if mode == "strict":
        provider = Stage25PlanProvider(0, 0, 4)
        supports = tuple(
            provider._support_payload(context, capacity)
            for _ in range(batch_size))
    else:
        supports = (None,) * batch_size
    support_seconds = time.perf_counter() - started
    requests = []
    identity = _identity()
    for row in range(batch_size):
        row_support = supports[row]
        requests.append(Stage25InferenceRequest(
            identity=Stage25RequestIdentity(row, 0, 4, identity),
            worker_id=0, prng_id="benchmark", inputs={
                "unlocked": _inputs(1)["unlocked"],
                "board_animal": _inputs(1)["board_animal"],
            }, crop_capacity=np.asarray([capacity], dtype=np.int16),
            physical_context=context, queued_at=0.0,
            support=row_support))
    return tuple(requests), supports, support_seconds


def _measure(args: argparse.Namespace, mode: str) -> dict[str, float | int | str]:
    construction_samples = []
    serialization_samples = []
    deserialization_samples = []
    validation_samples = []
    for _ in range(args.repeats):
        requests, supports, support_seconds = _requests(args.batch_size, mode)
        construction_samples.append(support_seconds)

        started = time.perf_counter()
        payload = pickle.dumps(requests, protocol=5)
        serialization_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        pickle.loads(payload)
        deserialization_samples.append(time.perf_counter() - started)

        if mode == "strict":
            inputs = _inputs(args.batch_size)
            contexts = tuple(_context() for _ in range(args.batch_size))
            capacities = np.zeros((args.batch_size, 5), dtype=np.int16)
            started = time.perf_counter()
            _validate_support_shapes(supports, args.batch_size)
            _validate_context_consistency(
                inputs, contexts, capacities, supports,
                check_observation=False)
            validation_samples.append(time.perf_counter() - started)
        else:
            validation_samples.append(0.0)

    return {
        "mode": mode,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "support_entries_per_row": SUPPORT_ENTRIES_PER_ROW if mode == "strict" else 0,
        "support_entries_per_batch": SUPPORT_ENTRIES_PER_ROW * args.batch_size if mode == "strict" else 0,
        "pickle_batch_bytes": len(payload),
        "pickle_bytes_per_row": len(payload) / args.batch_size,
        "support_construction_ms_per_batch": statistics.mean(construction_samples) * 1000,
        "pickle_serialize_ms_per_batch": statistics.mean(serialization_samples) * 1000,
        "pickle_deserialize_ms_per_batch": statistics.mean(deserialization_samples) * 1000,
        "parent_support_validation_ms_per_batch": statistics.mean(validation_samples) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        parser.error("batch-size and repeats must be positive")
    print(json.dumps({mode: _measure(args, mode)
                      for mode in ("strict", "fast")}, indent=2,
                 sort_keys=True))


if __name__ == "__main__":
    main()
