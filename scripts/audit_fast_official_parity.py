"""Audit an official-captured current-control trace against the fast engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from oracle import (
    ClosedLoopDivergenceError,
    DivergenceError,
    deep_diff,
    make_backend,
    make_current_control_factory,
    run_closed_loop,
    run_same_action_replay,
    verify_official_provenance,
)
from oracle.closed_loop import AgentFactory, _normal_observation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "local" / "fast_official_parity"
DEFAULT_RESET_SEEDS = (7, 17, 42, 123)
PASS = ["PASS"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass_action(observation: Mapping[str, Any], seat: int) -> dict[str, Any]:
    hands = len(observation["farms"][seat].get("hands") or [])
    return {"farmer": PASS, "hands": [PASS] * hands, "market": []}


def _source_identity() -> dict[str, Any]:
    import fast_env
    import fast_env._kaggriculture_env as native

    native_path = Path(native.__file__).resolve()
    files = [
        REPO_ROOT / "rust" / "kaggriculture_env" / "src" / "lib.rs",
        REPO_ROOT / "rust" / "kaggriculture_env" / "src" / "generated_protocol.rs",
        REPO_ROOT / "fast_env" / "api.py",
        REPO_ROOT / "fast_env" / "batch.py",
        Path(fast_env.__file__).resolve(),
    ]
    return {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "rust_source_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "rust/kaggriculture_env"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(path)
            for path in files
        },
        "native_path": str(native_path),
        "native_sha256": _sha256(native_path),
        "native_size": native_path.stat().st_size,
        "python": sys.executable,
    }


def _reset_checks(seeds: Sequence[int]) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        official = make_backend("official", {"seed": seed})
        fast = make_backend("fast", {"seed": seed})
        official_observations = official.reset()
        fast_observations = fast.reset()
        state_diffs = deep_diff(official.canonical_state(), fast.canonical_state())
        observation_diffs = []
        for seat in range(2):
            observation_diffs.extend(deep_diff(
                _normal_observation(
                    official_observations[seat], from_fast=False, default_step=0
                ),
                _normal_observation(
                    fast_observations[seat], from_fast=True, default_step=0
                ),
                path=f"observation[{seat}]",
            ))
        results.append({
            "seed": seed,
            "underlying_state_equal": not state_diffs,
            "policy_observations_equal": not observation_diffs,
            "state_first_difference": (
                state_diffs[0].render() if state_diffs else None
            ),
            "observation_first_difference": (
                observation_diffs[0].render() if observation_diffs else None
            ),
        })
    return results


def capture_official_trace(
    *,
    seed: int,
    candidate_seat: int,
    checkpoint: Path,
    opponent: str = "pass",
    max_steps: int = 719,
) -> dict[str, Any]:
    """Capture actions from the current control and a fixed opponent."""

    if opponent not in {"pass", "current_control"}:
        raise ValueError(f"unsupported opponent {opponent!r}")

    configuration = {"seed": seed}
    backend = make_backend("official", configuration)
    observations = backend.reset()
    factory = make_current_control_factory(str(checkpoint), device="cpu")
    candidate = factory("official", candidate_seat, configuration)
    opponent_agent = (
        factory("official", 1 - candidate_seat, configuration)
        if opponent == "current_control" else None
    )
    actions: list[list[dict[str, Any]]] = []
    started = time.perf_counter()
    for _ in range(max_steps):
        pair = []
        for seat in range(2):
            if seat == candidate_seat:
                action = candidate(copy.deepcopy(observations[seat]))
                pair.append(copy.deepcopy(dict(action)))
            elif opponent_agent is not None:
                action = opponent_agent(copy.deepcopy(observations[seat]))
                pair.append(copy.deepcopy(dict(action)))
            else:
                pair.append(_pass_action(observations[seat], seat))
        actions.append(pair)
        observations, _, statuses = backend.step(copy.deepcopy(pair))
        backend.validate_status_history()
        if statuses == ["DONE", "DONE"]:
            break
    return {
        "schema_version": 1,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "configuration": configuration,
        "policy": {
            "manager": "BC-E",
            "opening": "standard_mixed d0-d3",
            "executor": "current repaired deterministic executor",
            "opponent": opponent,
            "aggressive_sell_all": True,
            "optional_spare_watering": True,
            "immediate_plant_water": True,
            "suppress_expansion_from_prior_debt": True,
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "reset_observation_count": 1,
        "accepted_steps": len(actions),
        "final_step": backend.canonical_state()["step"],
        "statuses": backend.statuses,
        "rewards": backend.rewards,
        "capture_wall_time_seconds": time.perf_counter() - started,
        "joint_actions": actions,
    }


class _TraceAgent:
    def __init__(self, actions: Sequence[Sequence[Mapping[str, Any]]], seat: int):
        self._actions = actions
        self._seat = seat
        self._index = 0

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        action = self._actions[self._index][self._seat]
        self._index += 1
        return copy.deepcopy(action)


def _trace_agent_factory(
    actions: Sequence[Sequence[Mapping[str, Any]]],
) -> AgentFactory:
    def factory(
        backend_name: str, seat: int, configuration: Mapping[str, Any]
    ) -> _TraceAgent:
        del backend_name, configuration
        return _TraceAgent(actions, seat)

    return factory


def _divergence_payload(error: Exception) -> dict[str, Any]:
    report = getattr(error, "report")
    payload = report.to_dict()
    phase = payload.get("phase")
    if "p0_action" in payload:
        pre_state_equal = phase == "turn"
        pre_policy_observations_equal = phase == "turn"
    else:
        pre_state_equal = (
            None if phase == "reset_observation"
            else phase in {"observation", "action", "next_state"}
        )
        pre_policy_observations_equal = phase in {"action", "next_state"}
    payload.update({
        "error_type": type(error).__name__,
        "pre_state_equal": pre_state_equal,
        "pre_policy_observations_equal": pre_policy_observations_equal,
        "comparison_hierarchy": phase,
    })
    if "p0_action" in payload:
        payload["joint_action"] = [payload["p0_action"], payload["p1_action"]]
    elif "official_action" in payload:
        payload["joint_action"] = payload["official_action"]
    return payload


def run_audit(
    *,
    seed: int,
    candidate_seat: int,
    checkpoint: Path,
    reset_seeds: Sequence[int],
    output_dir: Path,
    opponent: str = "pass",
) -> dict[str, Any]:
    official_identity = verify_official_provenance()
    fast_identity = _source_identity()
    resets = _reset_checks(reset_seeds)
    trace = capture_official_trace(
        seed=seed, candidate_seat=candidate_seat, checkpoint=checkpoint,
        opponent=opponent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"trace_seed_{seed}_seat_{candidate_seat}.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")

    actions = trace["joint_actions"]
    divergence = None
    same_action_result = None
    policy_result = None
    try:
        same_action_result = run_same_action_replay(
            trace["configuration"], actions, max_turns=len(actions)
        )
        trace_factory = _trace_agent_factory(actions)
        policy_result = run_closed_loop(
            trace["configuration"],
            max_steps=len(actions),
            agent_factories={"official": trace_factory, "fast": trace_factory},
        )
    except (DivergenceError, ClosedLoopDivergenceError) as error:
        divergence = _divergence_payload(error)
        divergence_path = output_dir / f"first_divergence_seed_{seed}.json"
        divergence_path.write_text(
            json.dumps(divergence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema_version": 1,
        "scope": "issue-33 bounded fast-vs-official closed-loop parity audit",
        "official_identity": official_identity,
        "fast_identity": fast_identity,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "reset_checks": resets,
        "trace": {
            "path": str(trace_path),
            "seed": seed,
            "candidate_seat": candidate_seat,
            "opponent": opponent,
            "accepted_steps": trace["accepted_steps"],
            "action_sha256": hashlib.sha256(
                json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "same_action_replay": (
            vars(same_action_result) if same_action_result is not None else None
        ),
        "policy_visible_replay": (
            policy_result.to_dict() if policy_result is not None else None
        ),
        "first_divergence": divergence,
        "classification": (
            "provisionally training-safe for tested paths"
            if divergence is None
            and all(
                row["underlying_state_equal"] and row["policy_observations_equal"]
                for row in resets
            )
            else "NOT training-safe"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--opponent", choices=("pass", "current_control"), default="pass",
        help="official trace opponent used for escalation",
    )
    parser.add_argument(
        "--reset-seeds", nargs="+", type=int, default=list(DEFAULT_RESET_SEEDS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    summary = run_audit(
        seed=args.seed,
        candidate_seat=args.candidate_seat,
        checkpoint=args.checkpoint.resolve(),
        reset_seeds=args.reset_seeds,
        output_dir=args.output_dir.resolve(),
        opponent=args.opponent,
    )
    print(json.dumps({
        "summary": str(args.output_dir.resolve() / "summary.json"),
        "reset_parity": all(
            row["underlying_state_equal"] for row in summary["reset_checks"]
        ),
        "policy_observation_parity": all(
            row["policy_observations_equal"] for row in summary["reset_checks"]
        ),
        "trace_steps": summary["trace"]["accepted_steps"],
        "first_divergence": summary["first_divergence"],
        "classification": summary["classification"],
    }, sort_keys=True))
    return 1 if summary["first_divergence"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
