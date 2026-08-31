"""Run the bounded Python-vs-Rust executor differential panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bc_manager_jax.checkpoint import load_torch_checkpoint  # noqa: E402
from bc_manager_jax.model import ManagerConfig  # noqa: E402
from executor_v0.differential import compare_episode_results  # noqa: E402
from rl_manager.executor_factory import (  # noqa: E402
    make_default_executor_factory,
    make_rust_executor_factory,
)
from rl_manager.policy import JaxEPlanPolicy  # noqa: E402
from rl_manager.runner import (  # noqa: E402
    RunnerConfig,
    SelfPlayRunner,
    build_episode_spec,
)
from rl_manager.types import E_VS_E  # noqa: E402


def run_case(policy: Any, seed: int, *, max_turns: int) -> dict[str, Any]:
    config = RunnerConfig(
        backend_name="fast",
        backend_configuration={"seed": 0, "numThreads": 1},
        max_turns=max_turns,
        num_envs=1,
        low_telemetry=True,
        read_only_agent_observations=True,
        record_rollout=True,
    )
    spec = build_episode_spec(0, seed, E_VS_E, policy, policy)
    python_result = SelfPlayRunner(
        config, executor_factory=make_default_executor_factory()).run([spec])[0]
    rust_result = SelfPlayRunner(
        config, executor_factory=make_rust_executor_factory()).run([spec])[0]
    return {
        "seed": seed,
        "max_turns": max_turns,
        "result": compare_episode_results(python_result, rust_result).to_json_dict(),
        "python_timing_seconds": python_result.timing_seconds,
        "rust_timing_seconds": rust_result.timing_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--max-turns", type=int, default=719)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params, metadata = load_torch_checkpoint(args.checkpoint)
    policy = JaxEPlanPolicy(
        params, ManagerConfig(**metadata["model_config"]), name="issue36-parity")
    cases = [run_case(policy, seed, max_turns=args.max_turns)
             for seed in args.seeds]
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "cases": cases,
        "all_equal": all(case["result"]["equal"] for case in cases),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
