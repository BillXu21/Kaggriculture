"""Run the bounded stateful official-vs-fast A/B gate and write JSON evidence.

The default run is three complete default episodes.  It uses the existing
deterministic ``executor_v0`` with a fixed plan, not a fabricated checkpoint.
The real BC checkpoint path is intentionally reported separately because no
repo-local ``*.pt`` artifact is searched for or loaded by this smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from oracle import make_checkpoint_executor_factory, run_closed_loop
from oracle.provenance import ProvenanceError, verify_official_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (0, 7, 42)
DEFAULT_OUTPUT = REPO_ROOT / "research" / "closed_loop_ab_report.json"


def _checkpoint_evidence() -> dict[str, object]:
    checkpoints = sorted(REPO_ROOT.rglob("*.pt"))
    return {
        "repo_local_pt_files": [str(path.relative_to(REPO_ROOT)) for path in checkpoints],
        "available": bool(checkpoints),
        "status": (
            "repo-local checkpoint exists; this deterministic-plan report does "
            "not substitute for its real BC run"
            if checkpoints
            else "no repo-local *.pt checkpoint found; real BC/executor A/B is unavailable"
        ),
    }


def run_report(seeds: tuple[int, ...]) -> dict[str, object]:
    started = time.perf_counter()
    episodes = []
    action_families: dict[str, int] = {}
    for seed in seeds:
        result = run_closed_loop({"seed": seed}, max_steps=719)
        episodes.append(result.to_dict())
        for family, count in result.action_families.items():
            action_families[family] = action_families.get(family, 0) + count
    checkpoint_evidence = _checkpoint_evidence()
    checkpoint_run: dict[str, object]
    checkpoint_files = checkpoint_evidence["repo_local_pt_files"]
    if checkpoint_files:
        checkpoint_path = REPO_ROOT / Path(str(checkpoint_files[0]))
        try:
            checkpoint_factory = make_checkpoint_executor_factory(
                str(checkpoint_path), device="cpu"
            )
            checkpoint_result = run_closed_loop(
                {"seed": 0}, max_steps=719,
                agent_factories={
                    "official": checkpoint_factory,
                    "fast": checkpoint_factory,
                },
            )
            checkpoint_run = {
                "status": "passed",
                "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                "result": checkpoint_result.to_dict(),
            }
        except Exception as error:  # noqa: BLE001 - report exact incompatibility
            checkpoint_run = {
                "status": "incompatible",
                "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        checkpoint_run = {
            "status": "unavailable",
            "reason": "no repo-local *.pt checkpoint found",
        }
    return {
        "schema_version": 1,
        "runner": "oracle.run_closed_loop",
        "engine_pair": {
            "official": "kaggle_environments 1.32.7",
            "fast": "fast_env Rust engine",
        },
        "agent": "executor_v0.ExecutorAgent + FixedPlanProvider",
        "accounting": {
            "reset_observation": 1,
            "accepted_steps_per_episode": 719,
            "canonical_terminal_step": 719,
        },
        "seeds": list(seeds),
        "episodes": episodes,
        "action_families_total": dict(sorted(action_families.items())),
        "terminal_outcomes": [
            {
                "seed": episode["seed"],
                "steps": episode["steps_executed"],
                "terminal_step": episode["terminal_step"],
                "official_statuses": episode["official_statuses"],
                "fast_statuses": episode["fast_statuses"],
                "official_rewards": episode["official_rewards"],
                "fast_rewards": episode["fast_rewards"],
            }
            for episode in episodes
        ],
        "checkpoint_evidence": checkpoint_evidence,
        "real_checkpoint_executor_ab": checkpoint_run,
        "wall_time_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        verify_official_provenance()
    except ProvenanceError as error:
        print(f"SKIP: official provenance unavailable: {error}")
        return 3
    report = run_report(tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "seeds": report["seeds"],
        "episodes": len(report["episodes"]),
        "wall_time_seconds": report["wall_time_seconds"],
        "checkpoint_evidence": report["checkpoint_evidence"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
