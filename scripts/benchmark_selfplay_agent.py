"""Steady end-to-end self-play benchmark for the executor hot path.

This benchmark intentionally does not instrument individual functions. Use
``profile_selfplay_agent.py`` for attribution, then use this command against
the base worktree and the optimized worktree for wall-clock comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SEEDS = (17, 42, 2026, 7, 123, 1013, 1022, 1003)


def run_case(
    repo_root: Path,
    checkpoint: Path,
    *,
    num_envs: int,
    low_telemetry: bool,
    read_only_observations: bool,
    record_rollout: bool,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    root = str(repo_root.resolve())
    sys.path.insert(0, root)
    try:
        from bc_manager_jax.checkpoint import load_torch_checkpoint
        from bc_manager_jax.model import ManagerConfig
        from rl_manager.policy import JaxEPlanPolicy
        from rl_manager.runner import (
            RunnerConfig,
            SelfPlayRunner,
            build_episode_spec,
        )
        from rl_manager.types import E_VS_E

        params, metadata = load_torch_checkpoint(checkpoint)
        policy = JaxEPlanPolicy(
            params, ManagerConfig(**metadata["model_config"]), name="benchmark")
        samples = []
        fingerprints = []
        for iteration in range(warmups + repeats):
            runner_config = {
                "backend_name": "fast",
                "backend_configuration": {"seed": 0, "numThreads": 1},
                "num_envs": num_envs,
            }
            if low_telemetry:
                runner_config["low_telemetry"] = True
            if read_only_observations:
                runner_config["read_only_agent_observations"] = True
            if record_rollout:
                runner_config["record_rollout"] = True
            runner = SelfPlayRunner(RunnerConfig(
                **runner_config))
            specs = [
                build_episode_spec(index, seed, E_VS_E, policy, policy)
                for index, seed in enumerate(SEEDS[:num_envs])
            ]
            start = time.perf_counter()
            results = runner.run(specs)
            elapsed = time.perf_counter() - start
            if iteration >= warmups:
                samples.append({
                    "seconds": elapsed,
                    "games_per_second": len(results) / elapsed,
                    "primitive_turns": sum(719 for _ in results),
                    "primitive_turns_per_second": (719 * len(results)) / elapsed,
                    "agent_actions_seconds": runner.timing_totals["agent_actions"],
                    "agent_actions_share": runner.timing_totals["agent_actions"] / elapsed,
                })
                fingerprints.append({
                    "final_banks": [result.final_banks for result in results],
                    "statuses": [result.statuses for result in results],
                    "trace_digests": [result.trace_digest for result in results],
                    "joint_action_fingerprints": [
                        _joint_action_fingerprint(result.rollout)
                        if record_rollout and result.rollout is not None
                        else None for result in results
                    ],
                })
        steady = sorted(sample["seconds"] for sample in samples)[len(samples) // 2]
        median_sample = next(sample for sample in samples if sample["seconds"] == steady)
        return {
            "repo_root": root,
            "checkpoint": str(checkpoint.resolve()),
            "num_envs": num_envs,
            "low_telemetry": low_telemetry,
            "read_only_observations": read_only_observations,
            "warmups": warmups,
            "repeats": repeats,
            "samples": samples,
            "median": median_sample,
            "fingerprints": fingerprints,
        }
    finally:
        # This process is one benchmark invocation; remove the temporary root
        # so repeated cases cannot accidentally import a previous worktree.
        sys.path.remove(root)


def _joint_action_fingerprint(rollout: Any) -> dict[str, Any]:
    payload = json.dumps(
        rollout.joint_actions, sort_keys=True, separators=(",", ":"))
    return {
        "count": len(rollout.joint_actions),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--low-telemetry", action="store_true")
    parser.add_argument("--read-only-observations", action="store_true")
    parser.add_argument("--record-rollout", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.num_envs):
        parser.error("--num-envs values must be >= 1")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    if max(args.num_envs) > len(SEEDS):
        parser.error(f"supported benchmark size is at most {len(SEEDS)}")

    payload = {
        "schema_version": 1,
        "cases": [run_case(
            args.repo_root, args.checkpoint, num_envs=num_envs,
            low_telemetry=args.low_telemetry, warmups=args.warmups,
            repeats=args.repeats,
            read_only_observations=args.read_only_observations,
            record_rollout=args.record_rollout)
            for num_envs in args.num_envs],
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
