"""Guarded stationary-opponent PPO train/eval CLIs (issue #9 Stage B).

These commands exist so the eventual Kaggle run is one explicit, fully
parameterized invocation — they are NEVER executed by tests (tests cover
only parsing/planning/aggregation) and are NOT runnable on this laptop
until the real BC-E checkpoint exists and issue #7 freezes the executor.

Design rules enforced here:

- the real E checkpoint path is REQUIRED and must exist (fail loud);
- the executor factory selection is REQUIRED explicitly (no implicit
  default executor behind an RL run);
- worker/process/thread knobs are configurable and default to the safe
  value 1 everywhere; nothing hard-codes a many-core host;
- evaluation uses only the fixed named seed sets (smoke/dev/holdout),
  always plays BOTH seat orientations, prints the planned game count, and
  refuses dev/holdout without an explicit confirmation flag;
- evaluation output follows one fixed schema (W/L/T, paired margins,
  median/mean banks, per-seat split, anomalies/diagnostics, worst seeds);
  infrastructure only — no fabricated values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Fixed evaluation seed sets (issue #9 Evaluation section).
SMOKE_SEEDS: tuple[int, ...] = (17, 42, 2026)
DEV_SEEDS: tuple[int, ...] = tuple(range(200, 264))
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(5000, 5032))
SEED_SETS: dict[str, tuple[int, ...]] = {
    "smoke": SMOKE_SEEDS,
    "dev": DEV_SEEDS,
    "holdout": HOLDOUT_SEEDS,
}

#: Backends accepted by the rollout harness (`oracle.backend`).
KNOWN_BACKENDS = ("fast", "official")

#: Explicit executor factory registry (issue #7 swaps/adds entries here).
EXECUTOR_FACTORIES: Mapping[str, str] = {
    "executor_v0@stage-a-v1": "rl_manager.executor_factory:"
                              "make_default_executor_factory",
}

CONFIRM_FLAG = "--confirm-expensive"


# --------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_manager.cli",
        description="Stationary-opponent PPO V0 train/eval (issue #9).")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="PPO training against frozen E.")
    train.add_argument("--e-checkpoint", required=True,
                       help="Path to the REAL trained BC-E torch checkpoint "
                            "(never committed).")
    train.add_argument("--executor-factory", required=True,
                       choices=sorted(EXECUTOR_FACTORIES),
                       help="Explicit executor factory selection.")
    train.add_argument("--backend", default="fast", choices=KNOWN_BACKENDS)
    train.add_argument("--master-seed", type=int, required=True)
    train.add_argument("--num-workers", type=int, default=1,
                       help="Rollout worker processes (default 1; >1 not "
                            "implemented yet and fails loud).")
    train.add_argument("--num-envs", type=int, default=1,
                       help="Lockstep envs per worker chunk (default 1).")
    train.add_argument("--num-threads", type=int, default=1,
                       help="Engine Rayon threads per env (default 1).")
    train.add_argument("--episodes-per-update", type=int, default=8)
    train.add_argument("--updates", type=int, default=1)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--kl-to-frozen-coef", type=float, default=0.0)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--checkpoint", required=True,
                       help="Output RL PPO checkpoint path (.npz).")

    ev = sub.add_parser("eval",
                        help="Fixed-seed paired evaluation vs frozen E.")
    ev.add_argument("--checkpoint", required=True,
                    help="RL PPO checkpoint (.npz) to evaluate.")
    ev.add_argument("--e-checkpoint", required=True,
                    help="Frozen BC-E torch checkpoint for the opponent.")
    ev.add_argument("--executor-factory", required=True,
                    choices=sorted(EXECUTOR_FACTORIES))
    ev.add_argument("--backend", default="fast", choices=KNOWN_BACKENDS)
    ev.add_argument("--num-workers", type=int, default=1)
    ev.add_argument("--num-envs", type=int, default=1)
    ev.add_argument("--num-threads", type=int, default=1)
    ev.add_argument("--seed-set", required=True, choices=sorted(SEED_SETS))
    ev.add_argument("--output-json", required=True)
    ev.add_argument(CONFIRM_FLAG, action="store_true",
                    help="Required for the expensive dev/holdout panels.")
    return parser


# ------------------------------------------------------------ planning


def _validate_common(args: argparse.Namespace) -> dict[str, Any]:
    if args.executor_factory not in EXECUTOR_FACTORIES:
        raise ValueError(
            f"--executor-factory must be one of "
            f"{sorted(EXECUTOR_FACTORIES)}; refusing to guess the executor "
            f"(issue #7 selection must be explicit)")
    if args.backend not in KNOWN_BACKENDS:
        raise ValueError(f"--backend must be one of {KNOWN_BACKENDS}")
    knobs: dict[str, int] = {}
    for arg_name, label in (("num_workers", "workers"),
                            ("num_envs", "envs"),
                            ("num_threads", "threads")):
        value = int(getattr(args, arg_name))
        if value < 1:
            raise ValueError(f"--{label.replace('_', '-')} must be >= 1")
        knobs[arg_name] = value
    if knobs["num_workers"] > 1:
        raise NotImplementedError(
            "multi-process rollout workers are design-only until the Kaggle "
            "run; keep --num-workers 1 (fail loud instead of silently "
            "running single-process)")
    e_checkpoint = Path(args.e_checkpoint)
    if not e_checkpoint.is_file():
        raise FileNotFoundError(
            f"--e-checkpoint {e_checkpoint} does not exist; the real BC-E "
            f"checkpoint is required and is never committed to the repository")
    return {"knobs": knobs,
            "executor_factory": args.executor_factory,
            "backend": args.backend}


def plan_training(args: argparse.Namespace) -> dict[str, Any]:
    """Validate a train invocation into an explicit plan dict (no side
    effects beyond validation)."""
    plan = _validate_common(args)
    if args.master_seed < 0:
        raise ValueError("--master-seed must be nonnegative")
    for name, value in (("episodes_per_update", args.episodes_per_update),
                        ("updates", args.updates),
                        ("epochs", args.epochs),
                        ("minibatch_size", args.minibatch_size)):
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    plan.update({
        "mode": "train",
        "e_checkpoint": str(Path(args.e_checkpoint)),
        "master_seed": int(args.master_seed),
        "episodes_per_update": int(args.episodes_per_update),
        "updates": int(args.updates),
        "ppo": {"epochs": int(args.epochs),
                "minibatch_size": int(args.minibatch_size),
                "lr": float(args.lr),
                "kl_to_frozen_coef": float(args.kl_to_frozen_coef)},
        "output_dir": str(Path(args.output_dir)),
        "checkpoint": str(Path(args.checkpoint)),
    })
    return plan


def plan_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an eval invocation into an explicit plan dict (no games)."""
    plan = _validate_common(args)
    seeds = SEED_SETS[args.seed_set]
    expensive = args.seed_set in ("dev", "holdout")
    if expensive and not args.confirm_expensive:
        raise SystemExit(
            f"seed set {args.seed_set!r} plans {len(seeds) * 2} games; pass "
            f"{CONFIRM_FLAG} to acknowledge the cost")
    plan.update({
        "mode": "eval",
        "seed_set": args.seed_set,
        "seeds": list(seeds),
        "seat_orientations": ["candidate_vs_frozen", "frozen_vs_candidate"],
        "planned_games": len(seeds) * 2,
        "e_checkpoint": str(Path(args.e_checkpoint)),
        "checkpoint": str(Path(args.checkpoint)),
        "output_json": str(Path(args.output_json)),
    })
    print(f"planned evaluation: {plan['planned_games']} games "
          f"(seed set {args.seed_set!r}, both seat orientations)")
    return plan


# ----------------------------------------------------------- execution
# Never invoked by tests; requires the real checkpoints that are absent
# locally. Kept as thin compositions of the tested rl_manager primitives.


def execute_training(plan: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig

    from rl_manager.ppo import build_ppo_batch, init_train_state, ppo_update
    from rl_manager.ppo_adapter import ppo_batched_policy_from_state
    from rl_manager.ppo_policy import PPOConfig
    from rl_manager.policy import JaxEPlanPolicy
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
        build_episode_spec
    from rl_manager.seeds import SeedStream
    from rl_manager.trajectory import TrajectoryBuffer, e_input_spec

    frozen_params, metadata = load_torch_checkpoint(plan["e_checkpoint"])
    config = ManagerConfig(**metadata["model_config"])
    ppo_config = PPOConfig(**plan["ppo"])
    state = init_train_state(frozen_params, config,
                             seed=plan["master_seed"], ppo_config=ppo_config)
    candidate = ppo_batched_policy_from_state(
        state, config, ppo_config=ppo_config, name="ppo_candidate")
    frozen_policy = JaxEPlanPolicy(frozen_params, config, name="frozen_e")
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_stream = SeedStream(plan["master_seed"])
    history = []
    for update_index in range(plan["updates"]):
        buffer = TrajectoryBuffer(
            capacity=plan["episodes_per_update"] * 2 * 26,
            input_spec=e_input_spec())
        runner = SelfPlayRunner(
            RunnerConfig(
                backend_name=plan["backend"],
                backend_configuration={"seed": 0,
                                       "numThreads": plan["knobs"]["num_threads"]},
                num_envs=plan["knobs"]["num_envs"]),
            trajectory_buffer=buffer, master_seed=plan["master_seed"])
        # Seat randomization: alternate orientation by episode parity while
        # evaluation below always pairs both seats explicitly.
        specs = []
        for episode in range(plan["episodes_per_update"]):
            composition = ("candidate_vs_frozen" if episode % 2 == 0
                           else "frozen_vs_candidate")
            specs.append(build_episode_spec(
                update_index * plan["episodes_per_update"] + episode,
                seed_stream.episode_seed(
                    update_index * plan["episodes_per_update"] + episode),
                composition, candidate, frozen_policy))
        runner.run(specs)
        batch = build_ppo_batch(buffer.finalize(), gamma=ppo_config.gamma,
                                gae_lambda=ppo_config.gae_lambda)
        state, metrics = ppo_update(state, batch, config, ppo_config)
        candidate.refresh_identity()
        from rl_manager.ppo_checkpoint import save_ppo_checkpoint

        path = save_ppo_checkpoint(
            output_dir / f"ppo_update_{update_index:06d}.npz", state, config,
            ppo_config, provenance={"plan": dict(plan)})
        history.append({"update": update_index, "metrics": metrics,
                        "checkpoint": str(path)})
    return {"history": history, "final_checkpoint": history[-1]["checkpoint"]
            if history else None}


def execute_evaluation(plan: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig

    from rl_manager.ppo_adapter import ppo_batched_policy_from_state
    from rl_manager.ppo_checkpoint import load_ppo_checkpoint
    from rl_manager.policy import JaxEPlanPolicy
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
        build_episode_spec

    frozen_params, metadata = load_torch_checkpoint(plan["e_checkpoint"])
    config = ManagerConfig(**metadata["model_config"])
    state, _meta = load_ppo_checkpoint(plan["checkpoint"], config=config)
    candidate = ppo_batched_policy_from_state(
        state, config, name="ppo_candidate", deterministic=True)
    frozen_policy = JaxEPlanPolicy(frozen_params, config, name="frozen_e")
    runner = SelfPlayRunner(
        RunnerConfig(
            backend_name=plan["backend"],
            backend_configuration={"seed": 0,
                                   "numThreads": plan["knobs"]["num_threads"]},
            num_envs=plan["knobs"]["num_envs"]))
    specs = []
    for seed in plan["seeds"]:
        for orientation in plan["seat_orientations"]:
            specs.append(build_episode_spec(len(specs), seed, orientation,
                                            candidate, frozen_policy))
    results = runner.run(specs)
    summary = summarize_evaluation(results)
    from rl_manager.diagnostics import write_diagnostics

    write_diagnostics(plan["output_json"], summary)
    return summary


# ---------------------------------------------------------- aggregation


def summarize_evaluation(results: Sequence[Any]) -> dict[str, Any]:
    """Fixed evaluation output schema over EpisodeResult-like records.

    Pure aggregation — infrastructure only; callers supply real games.
    """
    wins = losses = ties = 0
    margins: list[float] = []
    candidate_banks: list[float] = []
    opponent_banks: list[float] = []
    per_orientation: dict[str, dict[str, int]] = {}
    anomalies: list[dict[str, Any]] = []
    seed_margins: list[tuple[int, str, float]] = []

    for result in results:
        composition = str(result.composition)
        candidate_seat = 0 if composition == "candidate_vs_frozen" else 1
        banks = [float(bank) for bank in result.final_banks]
        margin = banks[candidate_seat] - banks[1 - candidate_seat]
        margins.append(margin)
        candidate_banks.append(banks[candidate_seat])
        opponent_banks.append(banks[1 - candidate_seat])
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
        bucket = per_orientation.setdefault(
            composition, {"games": 0, "W": 0, "L": 0, "T": 0})
        bucket["games"] += 1
        bucket["W" if margin > 0 else "L" if margin < 0 else "T"] += 1
        statuses = [str(status) for status in result.statuses]
        if statuses != ["DONE", "DONE"]:
            anomalies.append({"seed": int(result.seed),
                              "kind": "statuses", "detail": statuses})
        for record in result.opening_diagnostics or []:
            if any(bool(value) for key, value in record.items()
                   if any(tag in key for tag in
                          ("fallback", "guard", "error", "delegat"))):
                anomalies.append({"seed": int(result.seed),
                                  "kind": "opening", "detail": record})
        seed_margins.append((int(result.seed), composition, margin))

    ordered = sorted(margins)
    n = len(ordered)
    median_margin = (ordered[n // 2] if n % 2
                     else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])) \
        if n else None
    worst = sorted(seed_margins, key=lambda entry: entry[2])[:5]

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "evaluation_schema_version": 1,
        "games": len(results),
        "wlt": {"W": wins, "L": losses, "T": ties},
        "win_rate": (wins / len(results)) if results else None,
        "paired_margins": margins,
        "median_margin": median_margin,
        "mean_margin": _mean(margins),
        "banks": {
            "candidate_median": _median(candidate_banks),
            "candidate_mean": _mean(candidate_banks),
            "opponent_median": _median(opponent_banks),
            "opponent_mean": _mean(opponent_banks),
        },
        "per_orientation": per_orientation,
        "anomalies": anomalies,
        "worst_seeds": [{"seed": seed, "orientation": orientation,
                         "margin": margin} for seed, orientation, margin
                        in worst],
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 \
        else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        plan = plan_training(args)
        execute_training(plan)  # pragma: no cover
        return 0
    plan = plan_evaluation(args)
    execute_evaluation(plan)  # pragma: no cover
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
