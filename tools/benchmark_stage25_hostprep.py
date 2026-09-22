"""Profile Stage 2.5 parent-side host preparation with representative batches.

The detailed P2 decomposition deliberately mirrors the operations in
``Stage25InferenceAdapter._prepare`` and ``stage25_policy._host_inputs`` without
adding production timers.  All reported calls are warmed and synchronized.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

_REPOSITORY_ROOT = Path(os.environ.get(
    "KAGGRICULTURE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(_REPOSITORY_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bc_manager_jax.model import validate_inputs as validate_encoder_inputs  # noqa: E402
from rl_manager.stage25_inference import Stage25InferenceAdapter, _validate_inputs  # noqa: E402
from rl_manager.stage25_mechanics import PhysicalContext  # noqa: E402
from rl_manager.stage25_policy import (  # noqa: E402
    _call_prepared_policy,
    _host_contexts,
    _host_inputs,
    init_stage25_params,
    tiny_stage25_config,
)

try:
    from rl_manager.stage25_policy import (
        _prepare_stage25_inputs_validated,
        _validate_stage25_inputs,
    )
except ImportError:
    _prepare_stage25_inputs_validated = None
    _validate_stage25_inputs = None


INTEGER_FIELDS = frozenset({
    "board_kind", "board_crop", "board_animal", "board_mask", "shed_counts",
    "carried_counts", "unlocked", "seed_counts", "market_inventory",
    "shop_counts", "day", "days_remaining",
})
CALLS_PER_UPDATE = 1_300


def representative_inputs(batch: int) -> dict[str, np.ndarray]:
    """Use the compact dtypes emitted by the Stage 2.5 batching path."""
    return {
        "board_kind": np.zeros((batch, 100), dtype=np.int16),
        "board_crop": np.zeros((batch, 100), dtype=np.int8),
        "board_animal": np.zeros((batch, 100), dtype=np.int8),
        "board_numeric": np.zeros((batch, 100, 11), dtype=np.float32),
        "board_bool": np.zeros((batch, 100, 8), dtype=np.bool_),
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


def representative_contexts(batch: int) -> tuple[PhysicalContext, ...]:
    context = PhysicalContext(
        1, (25, 25, 25, 25), (0, 0, 0), 0, 0, (0, 0, 0))
    return (context,) * batch


def _synchronize(value: Any) -> None:
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _measure(function: Callable[[], Any], *, warmup: int,
             repeats: int) -> float:
    for _ in range(warmup):
        _synchronize(function())
    started = time.perf_counter()
    for _ in range(repeats):
        _synchronize(function())
    return (time.perf_counter() - started) / repeats


def _base(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value for key, value in inputs.items()
            if key not in ("crop_capacity", "row_ids")}


def _shape_checks(base: dict[str, np.ndarray]) -> int:
    board = np.asarray(base["board_kind"])
    batch = int(board.shape[0])
    if board.shape != (batch, 100):
        raise ValueError
    if np.asarray(base["board_animal"]).shape != board.shape:
        raise ValueError
    if np.asarray(base["board_mask"]).shape != (batch, 100, 4):
        raise ValueError
    return batch


def _unlocked_checks(base: dict[str, np.ndarray]) -> None:
    unlocked = np.asarray(base["unlocked"])
    if unlocked.shape[1:] != (4,) or not np.all(np.isin(unlocked, (0, 1))):
        raise ValueError
    prefix = np.asarray(unlocked, dtype=np.int32)
    if np.any(prefix[:, 1:] > prefix[:, :-1]) or np.any(prefix.sum(1) < 1):
        raise ValueError


def _ledger_checks(inputs: dict[str, np.ndarray]) -> None:
    ledger = np.asarray(inputs["crop_capacity"])
    if ledger.shape[1:] != (5,):
        raise ValueError
    if not np.issubdtype(ledger.dtype, np.integer):
        if (not np.issubdtype(ledger.dtype, np.floating)
                or not np.all(np.isfinite(ledger))
                or not np.all(ledger == np.floor(ledger))):
            raise ValueError
    if np.any(ledger < 0) or np.any(ledger > 100):
        raise ValueError


def _numpy_normalize(base: dict[str, np.ndarray],
                     ledger: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    prepared = {
        key: np.asarray(value, dtype=(np.int32 if key in INTEGER_FIELDS
                                     else np.float32))
        for key, value in base.items()
    }
    return prepared, np.asarray(ledger, dtype=np.int32)


def _jax_convert(base: dict[str, np.ndarray],
                 ledger: np.ndarray) -> tuple[dict[str, jax.Array], jax.Array]:
    prepared = {
        key: jnp.asarray(value, dtype=(jnp.int32 if key in INTEGER_FIELDS
                                      else jnp.float32))
        for key, value in base.items()
    }
    return prepared, jnp.asarray(ledger, dtype=jnp.int32)


def profile_batch(batch: int, *, warmup: int, repeats: int) -> dict[str, Any]:
    inputs = representative_inputs(batch)
    base = _base(inputs)
    ledger = inputs["crop_capacity"]
    contexts = representative_contexts(batch)
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=17)
    row_tokens = np.arange(batch, dtype=np.int32)
    adapter = Stage25InferenceAdapter(
        params=params, config=config, validation_mode="fast", seed=23)

    operations: dict[str, Callable[[], Any]] = {
        "outer_validation": lambda: _validate_inputs(inputs),
        "base_mapping": lambda: _base(inputs),
        "encoder_validation": lambda: validate_encoder_inputs(
            base, config.manager_config, model_variant="E"),
        "board_shape_checks": lambda: _shape_checks(base),
        "unlocked_validation": lambda: _unlocked_checks(base),
        "crop_ledger_validation": lambda: _ledger_checks(inputs),
        "numpy_dtype_normalization": lambda: _numpy_normalize(base, ledger),
        "numpy_to_jax_conversion": lambda: _jax_convert(base, ledger),
        "p2_host_inputs_total": lambda: _host_inputs(inputs, config),
        "physical_context_packing": lambda: _host_contexts(contexts, batch),
    }
    if _validate_stage25_inputs is not None:
        operations["p3_single_validation"] = lambda: _validate_stage25_inputs(
            inputs, config)
        validated = _validate_stage25_inputs(inputs, config)
        operations["p3_single_jax_preparation"] = lambda: (
            _prepare_stage25_inputs_validated(validated))
    seconds = {
        name: _measure(function, warmup=warmup, repeats=repeats)
        for name, function in operations.items()
    }

    # Warm the complete stochastic adapter call separately from microbenchmarks.
    def adapter_call() -> Any:
        return adapter.infer_batch(
            inputs,
            physical_contexts=contexts,
            row_ids=tuple(f"bench-{index}" for index in range(batch)),
            row_tokens=row_tokens,
            prng_id="hostprep-benchmark",
        )

    canonical_numpy, canonical_capacity = _numpy_normalize(base, ledger)
    canonical_contexts = _host_contexts(contexts, batch)
    root = adapter._cached_root_key("hostprep-benchmark")
    p2_style_prepared = {
        key: jnp.asarray(value, dtype=(jnp.int32 if key in INTEGER_FIELDS
                                      else jnp.float32))
        for key, value in base.items()
    }
    p2_style_capacity = jnp.asarray(ledger.astype(np.int32))

    def p2_style_policy_call() -> Any:
        result = _call_prepared_policy(
            adapter.params, p2_style_prepared, p2_style_capacity, batch, config,
            mode="sample", rng_root=root, physical_contexts=contexts,
            row_ids=row_tokens, reject_invalid=False)
        _synchronize(result)
        return result

    def direct_numpy_policy_call() -> Any:
        result = _call_prepared_policy(
            adapter.params, canonical_numpy, canonical_capacity, batch, config,
            mode="sample", rng_root=root, physical_contexts=contexts,
            row_ids=row_tokens, reject_invalid=False)
        _synchronize(result)
        return result

    direct_numpy_seconds = _measure(
        direct_numpy_policy_call, warmup=max(warmup, 2), repeats=repeats)
    p2_style_seconds = _measure(
        p2_style_policy_call, warmup=max(warmup, 2), repeats=repeats)

    full_seconds = _measure(adapter_call, warmup=max(warmup, 2), repeats=repeats)
    phases_before = dict(adapter.inference_phase_seconds)
    for _ in range(repeats):
        adapter_call()
    phase_seconds = {
        name: (adapter.inference_phase_seconds[name] - phases_before[name]) / repeats
        for name in phases_before
    }
    seconds["full_adapter_wall"] = full_seconds
    seconds["direct_numpy_policy_call"] = direct_numpy_seconds
    seconds["p2_style_policy_call"] = p2_style_seconds
    seconds["canonical_numpy_context_packing"] = _measure(
        lambda: (canonical_numpy, canonical_capacity, canonical_contexts),
        warmup=warmup, repeats=repeats)
    seconds.update({f"adapter_{name}": value
                    for name, value in phase_seconds.items()})
    return {
        "batch": batch,
        "repeats": repeats,
        "seconds_per_call": seconds,
        "seconds_per_1300_calls": {
            name: value * CALLS_PER_UPDATE for name, value in seconds.items()
        },
        "input_dtypes": {name: str(value.dtype) for name, value in inputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=(8, 20, 32))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches):
        raise SystemExit("batch sizes must be positive")
    if args.warmup < 1 or args.repeats < 1:
        raise SystemExit("warmup and repeats must be positive")
    print(json.dumps({
        "calls_per_update": CALLS_PER_UPDATE,
        "profiles": [profile_batch(
            batch, warmup=args.warmup, repeats=args.repeats)
            for batch in args.batches],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
