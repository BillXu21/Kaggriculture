"""CPU/mock-policy benchmark for the issue #17 rollout topology.

Example::

    python tools/benchmark_parallel_rollouts.py --episodes 8 --num-workers 4

This measures process/IPC topology with a deterministic scalar policy.  It
does not claim TPU performance; use the Kaggle command in the issue #17
runbook for the real BC-E/JAX measurement.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.num_workers < 1 or args.num_envs < 1 \
            or args.num_threads < 1 or args.max_turns < 0:
        raise SystemExit("episode/worker/env/thread/turn values must be valid")
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
        batch_backend=args.batch_backend)
    runner = ParallelSelfPlayRunner(
        config, num_workers=args.num_workers, master_seed=args.master_seed)
    started = time.perf_counter()
    results = runner.run(specs)
    elapsed = time.perf_counter() - started
    record = {
        "schema_version": 1,
        "backend": "fast",
        "policy": policy.identity.to_json_dict(),
        "episodes": len(results),
        "max_turns": args.max_turns,
        "num_workers": args.num_workers,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "master_seed": args.master_seed,
        "elapsed_seconds": elapsed,
        "games_per_second": len(results) / elapsed if elapsed else None,
        "primitive_turns_per_second": (
            len(results) * args.max_turns / elapsed if elapsed else None),
        "manager_requests_per_second": (
            sum(result.transitions for result in results) / elapsed
            if elapsed else None),
        "inference": runner.inference_metrics,
        "result_seeds": [result.seed for result in results],
    }
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
