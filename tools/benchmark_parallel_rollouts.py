"""CPU/mock-policy benchmark for the issue #17 rollout topology.

Example::

    python tools/benchmark_parallel_rollouts.py --episodes 8 --num-workers 4

This measures process/IPC topology with a deterministic scalar policy.  It
does not claim TPU performance; use the Kaggle command in the issue #17
runbook for the real BC-E/JAX measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_manager.decode import ACTION_TENSOR_SHAPES
from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.runner import GAME_TURNS, RunnerConfig, build_episode_spec
from rl_manager.seeds import SeedStream
from rl_manager.types import E_VS_E, PolicyIdentity, PolicyOutputs


class BenchmarkPolicy:
    def __init__(self) -> None:
        self.identity = PolicyIdentity(
            "benchmark-pass-plan", "issue-17-v1", "0" * 64)

    def plan_batch(self, inputs, prng_id):
        del prng_id
        batch = int(np.asarray(inputs["day"]).shape[0])
        actions = {
            name: np.zeros((batch,) + shape, dtype=np.int16)
            for name, shape in ACTION_TENSOR_SHAPES.items()}
        actions["land"] = np.ones(batch, dtype=np.int16)
        zeros = np.zeros(batch, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=actions,
            logprob_groups={name: zeros.copy() for name in (
                "crop", "animal", "land", "fertilizer", "care",
                "sell_presence")},
            logprob_total=zeros.copy(), value=zeros.copy(), batch_size=batch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=GAME_TURNS)
    parser.add_argument("--master-seed", type=int, default=17)
    parser.add_argument("--low-telemetry", action="store_true")
    parser.add_argument("--read-only-agent-observations", action="store_true")
    parser.add_argument("--batch-backend", action="store_true")
    parser.add_argument("--inference-batch-scope", choices=("policy_day", "policy"),
                        default="policy_day")
    parser.add_argument("--fixed-inference-batch-size", type=int)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=20.0)
    parser.add_argument(
        "--sweep-workers",
        help="Comma-separated worker counts; overrides --num-workers.")
    parser.add_argument(
        "--sweep-fixed-batch-sizes",
        help="Comma-separated physical batch sizes; overrides the fixed size.")
    parser.add_argument(
        "--sweep-batch-waits-ms",
        help="Comma-separated batch waits in milliseconds; overrides the wait.")
    parser.add_argument(
        "--sweep-batch-scopes",
        help="Comma-separated policy_day,policy scopes; overrides the scope.")
    parser.add_argument("--output-json", type=Path)
    return parser


def _csv_values(value: str | None, *, cast, label: str) -> list:
    if value is None:
        return []
    try:
        values = [cast(item.strip()) for item in value.split(",")
                  if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid {label} sweep: {value!r}") from exc
    if not values:
        raise SystemExit(f"{label} sweep must not be empty")
    return values


def _run_case(args, *, num_workers: int, fixed_batch_size: int | None,
              wait_ms: float, scope: str) -> dict:
    policy = BenchmarkPolicy()
    seeds = SeedStream(args.master_seed)
    specs = [build_episode_spec(
        index, seeds.episode_seed(index), E_VS_E, policy, policy)
        for index in range(args.episodes)]
    config = RunnerConfig(
        backend_name="fast",
        backend_configuration={"seed": 0, "numThreads": args.num_threads},
        num_envs=args.num_envs, max_turns=args.max_turns,
        low_telemetry=args.low_telemetry,
        read_only_agent_observations=args.read_only_agent_observations,
        batch_backend=args.batch_backend,
        inference_batch_scope=scope,
        fixed_inference_batch_size=fixed_batch_size,
        inference_batch_wait_seconds=wait_ms / 1000.0)
    runner = ParallelSelfPlayRunner(
        config, num_workers=num_workers, master_seed=args.master_seed)
    started = time.perf_counter()
    results = runner.run(specs)
    elapsed = time.perf_counter() - started
    inference = runner.inference_metrics
    trace_signature = hashlib.sha256(
        "|".join(result.trace_digest for result in results).encode("ascii")
    ).hexdigest()
    real_batches = inference.get("real_batch_sizes", [])
    return {
        "num_workers": num_workers,
        "fixed_inference_batch_size": fixed_batch_size,
        "inference_batch_wait_ms": wait_ms,
        "inference_batch_scope": scope,
        "episodes": len(results),
        "max_turns": args.max_turns,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "master_seed": args.master_seed,
        "policy": policy.identity.to_json_dict(),
        "elapsed_seconds": elapsed,
        "games_per_second": len(results) / elapsed if elapsed else None,
        "primitive_turns_per_second": (
            sum(result.transitions for result in results) / elapsed
            if elapsed else None),
        "manager_requests_per_second": (
            sum(result.transitions for result in results) / elapsed
            if elapsed else None),
        "inference_seconds": inference["inference_seconds"],
        "inference_percent": (
            100.0 * inference["inference_seconds"] / elapsed
            if elapsed else None),
        "call_count": inference.get("physical_inference_calls", 0),
        "real_batch_histogram": {
            str(size): real_batches.count(size)
            for size in sorted(set(real_batches))},
        "occupancy": inference.get("occupancy", 0.0),
        "inference": inference,
        "deterministic_signature": trace_signature,
        "result_seeds": [result.seed for result in results],
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.num_workers < 1 or args.num_envs < 1 \
            or args.num_threads < 1 or args.max_turns < 0 \
            or args.inference_batch_wait_ms < 0:
        raise SystemExit("episode/worker/env/thread/turn values must be valid")
    if args.fixed_inference_batch_size is not None \
            and args.fixed_inference_batch_size < 1:
        raise SystemExit("--fixed-inference-batch-size must be >= 1")
    workers = _csv_values(args.sweep_workers, cast=int, label="workers") \
        or [args.num_workers]
    batch_sizes = (_csv_values(args.sweep_fixed_batch_sizes, cast=int,
                               label="fixed batch sizes")
                   or [args.fixed_inference_batch_size])
    waits = (_csv_values(args.sweep_batch_waits_ms, cast=float,
                         label="batch waits") or [args.inference_batch_wait_ms])
    scopes = (_csv_values(args.sweep_batch_scopes, cast=str,
                          label="batch scopes") or [args.inference_batch_scope])
    if any(value < 1 for value in workers) or any(
            value is not None and value < 1 for value in batch_sizes):
        raise SystemExit("worker and fixed batch sweep values must be >= 1")
    if any(value < 0 for value in waits):
        raise SystemExit("batch wait sweep values must be >= 0")
    if any(value not in ("policy_day", "policy") for value in scopes):
        raise SystemExit("batch scopes must be policy_day or policy")
    cases = [_run_case(
        args, num_workers=workers_value, fixed_batch_size=batch_size,
        wait_ms=wait, scope=scope)
        for workers_value in workers for batch_size in batch_sizes
        for wait in waits for scope in scopes]
    record = {"schema_version": 2, "backend": "fast", "cases": cases}
    if len(cases) == 1:
        record.update(cases[0])
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
