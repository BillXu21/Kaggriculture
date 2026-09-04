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
import dataclasses
import json
import math
from importlib import import_module
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from rl_manager.evaluation import evaluate_promotion
from rl_manager.evaluation import format_promotion_result
from rl_manager.evaluation import summarize_evaluation
from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_VERSIONS,
    normalize_e_history_version,
)
from bc_manager.constants import TOTAL_DAYS
from rl_manager.ratchet import PromotionRatchet
from rl_manager.runner import GAME_TURNS
from rl_manager.runner import INFERENCE_BATCH_SCOPES
from rl_manager.types import E_VS_E, E_VS_PASS
from rl_manager.types import (CANDIDATE_VS_FROZEN,
                               CURRENT_VS_CURRENT_ECONOMIC)
from rl_manager.reward import (REWARD_MODES, RewardConfig,
                               TERMINAL_OWN_BANK)
if TYPE_CHECKING:  # pragma: no cover - import-time accelerator safety
    from rl_manager.ppo_policy import (CurriculumMaskConfig,
                                       TargetedExplorationConfig)

#: Fixed evaluation seed sets (issue #9 Evaluation section).
SMOKE_SEEDS: tuple[int, ...] = (17, 42, 2026)
DEV_SEEDS: tuple[int, ...] = tuple(range(200, 264))
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(5000, 5032))
PROMOTION_SEEDS: tuple[int, ...] = tuple(range(3000, 3032))
SEED_SETS: dict[str, tuple[int, ...]] = {
    "smoke": SMOKE_SEEDS,
    "dev": DEV_SEEDS,
    "holdout": HOLDOUT_SEEDS,
}

#: Backends accepted by the rollout harness (`oracle.backend`).
KNOWN_BACKENDS = ("fast", "official")
DEBUG_TRACE_COMPOSITIONS = (E_VS_E, E_VS_PASS)
TRAINING_COMPOSITIONS = (CANDIDATE_VS_FROZEN, CURRENT_VS_CURRENT_ECONOMIC)

#: Explicit executor factory registry (issue #7 swaps/adds entries here).
EXECUTOR_FACTORIES: Mapping[str, str] = {
    "executor_v0@stage-a-v1": "rl_manager.executor_factory:"
                              "make_default_executor_factory",
}

CONFIRM_FLAG = "--confirm-expensive"


def _add_inference_batch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-batch-scope", choices=INFERENCE_BATCH_SCOPES,
                        default="policy_day",
                        help="Group central requests by policy/day (default) or "
                             "policy across days.")
    parser.add_argument("--fixed-inference-batch-size", type=int, default=None,
                        help="Physical central batch size; pad valid rows to B.")
    parser.add_argument("--inference-batch-wait-ms", type=float, default=20.0,
                        help="Maximum central batch wait before dispatch (ms).")


def _add_curriculum_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--curriculum-max-land", type=int, default=None,
                        help="Maximum decoded land target (1..4).")
    parser.add_argument("--curriculum-max-goose", type=int, default=None,
                        help="Maximum absolute GOOSE target (0..count_max).")
    parser.add_argument("--curriculum-max-cow", type=int, default=None,
                        help="Maximum absolute COW target (0..count_max).")
    parser.add_argument("--curriculum-max-sheep", type=int, default=None,
                        help="Maximum absolute SHEEP target (0..count_max).")


def _add_unlock_exploration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unlock-exploration-epsilon", type=float, default=0.0)
    parser.add_argument("--unlock-exploration-updates", type=int, default=0)
    parser.add_argument("--unlock-exploration-land-target", type=int, default=None)
    parser.add_argument("--unlock-exploration-goose-target", type=int, default=None)
    parser.add_argument("--unlock-exploration-cow-target", type=int, default=None)
    parser.add_argument("--unlock-exploration-sheep-target", type=int, default=None)


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
    train.add_argument("--opening", default="standard_mixed")
    train.add_argument("--manager-start-day", type=int, default=4)
    train.add_argument("--master-seed", type=int, required=True)
    train.add_argument("--init-mode", choices=("bc", "scratch"),
                       default="bc",
                       help="Mutable manager initialization (default: bc).")
    train.add_argument("--resume-checkpoint", default=None,
                       help="Resume PPO state at this run/update boundary.")
    train.add_argument("--training-composition", choices=TRAINING_COMPOSITIONS,
                       default=CANDIDATE_VS_FROZEN)
    train.add_argument("--reward-mode", choices=REWARD_MODES,
                       default="terminal_wlt")
    train.add_argument("--bank-reward-baseline", type=float, default=3000.0)
    train.add_argument("--bank-reward-scale", type=float, default=50000.0)
    train.add_argument("--num-workers", type=int, default=1,
                       help="Rollout worker processes (default 1).")
    train.add_argument("--num-envs", type=int, default=1,
                       help="Lockstep envs per worker chunk (default 1).")
    train.add_argument("--num-threads", type=int, default=1,
                       help="Engine Rayon threads per env (default 1).")
    train.add_argument("--low-telemetry", action="store_true",
                       help="Disable per-turn executor snapshots.")
    train.add_argument("--read-only-agent-observations", action="store_true",
                       help="Use safe read-only executor observation views.")
    train.add_argument("--batch-backend", action="store_true",
                       help="Use the native batched fast backend.")
    _add_inference_batch_options(train)
    train.add_argument("--episodes-per-update", type=int, default=8)
    train.add_argument("--updates", type=int, default=1)
    train.add_argument(
        "--promotion-every", type=int, default=0,
        help="Check a deterministic candidate every N updates (0 disables).")
    train.add_argument(
        "--max-promotions", type=int, default=None,
        help="Stop after this many accepted promotions.")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=8,
                        help="Must divide the expected complete-game row "
                             "count; checked at "
                             "plan time before any rollout.")
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--kl-to-frozen-coef", type=float, default=0.0)
    train.add_argument("--target-kl", type=float, default=None,
                       help="Stop remaining PPO epochs after this KL.")
    train.add_argument("--reject-update-kl", type=float, default=None,
                       help="Reject the whole update on nonfinite metrics or "
                            "post-epoch KL above this ceiling.")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--checkpoint", required=True,
                       help="Output RL PPO checkpoint path (.npz).")
    train.add_argument("--e-history-version", choices=E_HISTORY_VERSIONS,
                       default=E_HISTORY_CORRECTED_V1,
                       help="Explicit E input semantics; legacy is compatibility-only.")
    _add_curriculum_options(train)
    _add_unlock_exploration_options(train)

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
    ev.add_argument("--low-telemetry", action="store_true")
    ev.add_argument("--read-only-agent-observations", action="store_true")
    ev.add_argument("--batch-backend", action="store_true")
    _add_inference_batch_options(ev)
    ev.add_argument("--seed-set", required=True, choices=sorted(SEED_SETS))
    ev.add_argument("--output-json", required=True)
    ev.add_argument("--e-history-version", choices=E_HISTORY_VERSIONS,
                     default=E_HISTORY_CORRECTED_V1)
    _add_curriculum_options(ev)
    ev.add_argument(CONFIRM_FLAG, action="store_true",
                    help="Required for the expensive dev/holdout panels.")

    trace = sub.add_parser(
        "debug-trace",
        help="Generate complete canonical issue #11 trace JSON artifacts.",
    )
    trace.add_argument(
        "--case", action="append", metavar="SEED:SEAT",
        help="Trace one case; repeat for multiple cases (for example 17:0).",
    )
    trace.add_argument("--seed", type=int,
                       help="Single-case seed alternative to --case.")
    trace.add_argument("--seat", type=int, choices=(0, 1),
                       help="Single-case requested seat with --seed.")
    trace.add_argument("--backend", default="fast", choices=KNOWN_BACKENDS)
    trace.add_argument(
        "--composition", choices=DEBUG_TRACE_COMPOSITIONS, default=E_VS_E,
        help="Debug composition (default: e_vs_e).",
    )
    trace.add_argument(
        "--e-checkpoint", required=True,
        help="Path to the REAL trained BC-E torch checkpoint.",
    )
    trace.add_argument("--policy-seed", type=int, default=11,
                       help="Reserved deterministic policy seed (default 11).")
    trace.add_argument("--num-threads", type=int, default=1)
    trace.add_argument(
        "--max-turns", type=int, default=GAME_TURNS,
        help=f"Primitive transitions to run (default {GAME_TURNS}; "
             "trace contains the reset plus observed states).",
    )
    trace.add_argument("--output-dir", default="artifacts/debug_traces",
                       help="Output directory for trace JSON files.")
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
    history_version = normalize_e_history_version(
        getattr(args, "e_history_version", E_HISTORY_CORRECTED_V1))
    knobs: dict[str, int] = {}
    for arg_name, label in (("num_workers", "workers"),
                            ("num_envs", "envs"),
                            ("num_threads", "threads")):
        value = int(getattr(args, arg_name))
        if value < 1:
            raise ValueError(f"--{label.replace('_', '-')} must be >= 1")
        knobs[arg_name] = value
    e_checkpoint = Path(args.e_checkpoint)
    if not e_checkpoint.is_file():
        raise FileNotFoundError(
            f"--e-checkpoint {e_checkpoint} does not exist; the real BC-E "
            f"checkpoint is required and is never committed to the repository")
    scope = str(getattr(args, "inference_batch_scope", "policy_day"))
    if scope not in INFERENCE_BATCH_SCOPES:
        raise ValueError(
            f"--inference-batch-scope must be one of {INFERENCE_BATCH_SCOPES}")
    fixed_size = getattr(args, "fixed_inference_batch_size", None)
    if (fixed_size is not None
            and (isinstance(fixed_size, bool)
                 or not isinstance(fixed_size, int) or fixed_size < 1)):
        raise ValueError("--fixed-inference-batch-size must be >= 1")
    wait_ms = float(getattr(args, "inference_batch_wait_ms", 20.0))
    if not math.isfinite(wait_ms) or wait_ms < 0:
        raise ValueError("--inference-batch-wait-ms must be finite and >= 0")
    return {"knobs": knobs,
            "e_history_version": history_version,
            "runner_options": {
                "low_telemetry": bool(getattr(args, "low_telemetry", False)),
                "read_only_agent_observations": bool(
                    getattr(args, "read_only_agent_observations", False)),
                "batch_backend": bool(getattr(args, "batch_backend", False)),
                "inference_batch_scope": scope,
                "fixed_inference_batch_size": (
                    None if fixed_size is None else int(fixed_size)),
                "inference_batch_wait_seconds": wait_ms / 1000.0,
            },
            "executor_factory": args.executor_factory,
            "backend": args.backend}


def _curriculum_from_args(args: argparse.Namespace) -> CurriculumMaskConfig:
    """Build the explicit static strategic support for one run boundary."""
    from rl_manager.ppo_policy import CurriculumMaskConfig  # parent-side only

    return CurriculumMaskConfig(
        max_land=getattr(args, "curriculum_max_land", None),
        max_goose=getattr(args, "curriculum_max_goose", None),
        max_cow=getattr(args, "curriculum_max_cow", None),
        max_sheep=getattr(args, "curriculum_max_sheep", None),
    )


def _exploration_from_args(args: argparse.Namespace) -> TargetedExplorationConfig:
    """Build the explicit temporary rollout behavior config."""
    from rl_manager.ppo_policy import TargetedExplorationConfig  # parent-side only

    return TargetedExplorationConfig(
        epsilon=getattr(args, "unlock_exploration_epsilon", 0.0),
        land_target=getattr(args, "unlock_exploration_land_target", None),
        goose_target=getattr(args, "unlock_exploration_goose_target", None),
        cow_target=getattr(args, "unlock_exploration_cow_target", None),
        sheep_target=getattr(args, "unlock_exploration_sheep_target", None),
    )


def _resolve_executor_factory(
        identifier: str, *, low_telemetry: bool = False) -> Any:
    """Resolve the explicit registry entry in the owner before spawning."""
    try:
        target = EXECUTOR_FACTORIES[identifier]
        module_name, attribute = target.split(":", 1)
        builder = getattr(import_module(module_name), attribute)
    except (KeyError, ImportError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"cannot resolve executor factory {identifier!r} from registry") \
            from exc
    if low_telemetry and identifier == "executor_v0@stage-a-v1":
        default = builder()
        return builder(dataclasses.replace(
            default.agent_config, record_turn_snapshot=False))
    return builder()


def plan_training(args: argparse.Namespace) -> dict[str, Any]:
    """Validate a train invocation into an explicit plan dict (no side
    effects beyond validation)."""
    plan = _validate_common(args)
    curriculum = _curriculum_from_args(args)
    exploration = _exploration_from_args(args)
    unlock_updates = getattr(args, "unlock_exploration_updates", 0)
    if (isinstance(unlock_updates, bool) or not isinstance(unlock_updates, int)
            or unlock_updates < 0):
        raise ValueError("--unlock-exploration-updates must be an integer >= 0")
    targets = (exploration.land_target, exploration.goose_target,
               exploration.cow_target, exploration.sheep_target)
    if exploration.land_target is not None and curriculum.max_land is not None \
            and exploration.land_target > curriculum.max_land:
        raise ValueError(
            f"land exploration target {exploration.land_target} is outside "
            f"current curriculum max_land={curriculum.max_land}")
    for name, target, maximum in zip(
            ("goose", "cow", "sheep"),
            targets[1:],
            (curriculum.max_goose, curriculum.max_cow, curriculum.max_sheep)):
        if target is not None and maximum is not None and target > maximum:
            raise ValueError(
                f"{name} exploration target {target} is outside current "
                f"curriculum max_{name}={maximum}")
    has_target = any(target is not None for target in targets)
    if (float(exploration.epsilon) > 0.0 or has_target or unlock_updates > 0) \
            and not (float(exploration.epsilon) > 0.0 and unlock_updates > 0
                     and has_target):
        raise ValueError(
            "active unlock exploration requires epsilon > 0, updates > 0, "
            "and at least one explicit target")
    if args.master_seed < 0:
        raise ValueError("--master-seed must be nonnegative")
    opening = str(getattr(args, "opening", "standard_mixed"))
    if not opening:
        raise ValueError("--opening must be non-empty")
    manager_start_day = int(getattr(args, "manager_start_day", 4))
    if not 0 <= manager_start_day < TOTAL_DAYS:
        raise ValueError(
            f"--manager-start-day must be in [0, {TOTAL_DAYS - 1}]")
    init_mode = str(getattr(args, "init_mode", "bc"))
    if init_mode not in ("bc", "scratch"):
        raise ValueError("--init-mode must be one of ('bc', 'scratch')")
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint is not None and init_mode != "bc":
        raise ValueError("--resume-checkpoint cannot be combined with "
                         "--init-mode scratch")
    if resume_checkpoint is not None and not Path(resume_checkpoint).is_file():
        raise FileNotFoundError(
            f"--resume-checkpoint {resume_checkpoint} does not exist")
    composition = str(getattr(args, "training_composition",
                              CANDIDATE_VS_FROZEN))
    if composition not in TRAINING_COMPOSITIONS:
        raise ValueError(
            f"--training-composition must be one of {TRAINING_COMPOSITIONS}")
    reward_config = RewardConfig(
        mode=str(getattr(args, "reward_mode", "terminal_wlt")),
        bank_baseline=float(getattr(args, "bank_reward_baseline", 3000.0)),
        bank_scale=float(getattr(args, "bank_reward_scale", 50000.0)))
    if (composition == CURRENT_VS_CURRENT_ECONOMIC
            and reward_config.mode != TERMINAL_OWN_BANK):
        raise ValueError(
            "current_vs_current_economic requires --reward-mode "
            "terminal_own_bank; symmetric W/L training is rejected")
    for name, value in (("episodes_per_update", args.episodes_per_update),
                        ("updates", args.updates),
                        ("epochs", args.epochs),
                        ("minibatch_size", args.minibatch_size)):
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    promotion_every = int(getattr(args, "promotion_every", 0))
    max_promotions = getattr(args, "max_promotions", None)
    if promotion_every < 0:
        raise ValueError("--promotion-every must be >= 0")
    if max_promotions is not None and int(max_promotions) < 1:
        raise ValueError("--max-promotions must be >= 1")
    if max_promotions is not None and promotion_every == 0:
        raise ValueError("--max-promotions requires --promotion-every")
    # Plan-time divisibility check for complete manager horizons. The runtime
    # `ppo_update` strict check remains authoritative for truncations and
    # actual row counts; this only catches incompatible plans BEFORE any
    # env/checkpoint-heavy work.
    decisions_per_seat = TOTAL_DAYS - manager_start_day
    rows_per_game = decisions_per_seat * (
        2 if composition == CURRENT_VS_CURRENT_ECONOMIC else 1)
    expected_rows = int(args.episodes_per_update) * rows_per_game
    if expected_rows % int(args.minibatch_size) != 0:
        raise ValueError(
            f"--minibatch-size {args.minibatch_size} must divide the "
            f"expected complete-game row count {expected_rows} "
            f"(episodes_per_update {args.episodes_per_update} * "
            f"{rows_per_game}); "
            f"runtime ppo_update would fail loud after rollout")
    plan.update({
        "mode": "train",
        "e_checkpoint": str(Path(args.e_checkpoint)),
        "master_seed": int(args.master_seed),
        "opening": opening,
        "manager_start_day": manager_start_day,
        "manager_decisions_per_seat": decisions_per_seat,
        "init_mode": init_mode,
        "resume_checkpoint": (None if resume_checkpoint is None else
                               str(Path(resume_checkpoint))),
        "curriculum": curriculum.to_json_dict(),
        "unlock_exploration": exploration.to_json_dict(),
        "unlock_exploration_updates": int(unlock_updates),
        "training_composition": composition,
        "reward": reward_config.to_json_dict(),
        "expected_trainable_rows": expected_rows,
        "expected_trajectory_rows": (
            int(args.episodes_per_update) * 2 * decisions_per_seat),
        "rows_per_complete_game": rows_per_game,
        "episodes_per_update": int(args.episodes_per_update),
        "updates": int(args.updates),
        "promotion": {
            "every": promotion_every,
            "max_promotions": (None if max_promotions is None
                                else int(max_promotions)),
            "seed_set": "promotion",
            "seeds": list(PROMOTION_SEEDS),
        },
        "ppo": {"epochs": int(args.epochs),
                "minibatch_size": int(args.minibatch_size),
                "lr": float(args.lr),
                "kl_to_frozen_coef": float(args.kl_to_frozen_coef),
                "target_kl": (None if args.target_kl is None
                              else float(args.target_kl)),
                "reject_update_kl": (None if args.reject_update_kl is None
                                     else float(args.reject_update_kl))},
        "output_dir": str(Path(args.output_dir)),
        "checkpoint": str(Path(args.checkpoint)),
    })
    return plan


def _exploration_for_update(plan: Mapping[str, Any], update_index: int
                            ) -> TargetedExplorationConfig:
    """Resolve invocation-local first-N rollout behavior without PPO-step state."""

    from rl_manager.ppo_policy import TargetedExplorationConfig
    configured = TargetedExplorationConfig.from_json_dict(
        plan.get("unlock_exploration"))
    updates = int(plan.get("unlock_exploration_updates", 0))
    return configured if update_index < updates else TargetedExplorationConfig()


def _target_action_rates(arrays: Mapping[str, np.ndarray],
                         exploration: TargetedExplorationConfig
                         ) -> dict[str, dict[str, float | int]]:
    """Measure requested target action frequency on valid trainable rows."""
    rows = ((np.asarray(arrays["valid"]) == 1)
            & (np.asarray(arrays["trainable"]) == 1))
    count = int(rows.sum())
    rates: dict[str, dict[str, float | int]] = {}
    targets = {
        "land": (exploration.land_target, "action_land", None),
        "goose": (exploration.goose_target, "action_animal", 0),
        "cow": (exploration.cow_target, "action_animal", 1),
        "sheep": (exploration.sheep_target, "action_animal", 2),
    }
    for name, (target, field, species_index) in targets.items():
        if target is None:
            continue
        values = np.asarray(arrays[field])[rows]
        if species_index is not None:
            values = values[:, species_index]
        sampled = int(np.count_nonzero(values == target))
        rates[name] = {
            "target": int(target),
            "sampled_count": sampled,
            "trainable_row_count": count,
            "sampled_fraction": (sampled / count if count else None),
        }
    return rates


def plan_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an eval invocation into an explicit plan dict (no games)."""
    plan = _validate_common(args)
    curriculum = _curriculum_from_args(args)
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
        "curriculum": curriculum.to_json_dict(),
        "output_json": str(Path(args.output_json)),
    })
    print(f"planned evaluation: {plan['planned_games']} games "
          f"(seed set {args.seed_set!r}, both seat orientations)")
    return plan


def _parse_debug_case(value: str) -> tuple[int, int]:
    try:
        seed_text, seat_text = value.split(":", 1)
        seed, seat = int(seed_text), int(seat_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"--case must use SEED:SEAT, got {value!r} (for example 17:0)") \
            from exc
    if seed < 0:
        raise ValueError(f"--case seed must be nonnegative, got {seed}")
    if seat not in (0, 1):
        raise ValueError(f"--case seat must be 0 or 1, got {seat}")
    return seed, seat


def plan_debug_trace(args: argparse.Namespace) -> dict[str, Any]:
    """Validate a debug-trace invocation without constructing an environment."""
    cases: list[tuple[int, int]] = []
    if args.case:
        if args.seed is not None or args.seat is not None:
            raise ValueError(
                "use either repeated --case or --seed with --seat, not both")
        cases = [_parse_debug_case(value) for value in args.case]
    elif args.seed is not None or args.seat is not None:
        if args.seed is None or args.seat is None:
            raise ValueError("--seed and --seat must be supplied together")
        if args.seed < 0:
            raise ValueError(f"--seed must be nonnegative, got {args.seed}")
        if args.seat not in (0, 1):
            raise ValueError(f"--seat must be 0 or 1, got {args.seat}")
        cases = [(int(args.seed), int(args.seat))]
    else:
        raise ValueError("provide --case SEED:SEAT or --seed SEED --seat SEAT")
    if len(set(cases)) != len(cases):
        raise ValueError(f"duplicate debug-trace cases are not allowed: {cases}")
    if args.backend not in KNOWN_BACKENDS:
        raise ValueError(f"--backend must be one of {KNOWN_BACKENDS}")
    if args.policy_seed < 0:
        raise ValueError("--policy-seed must be nonnegative")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if args.max_turns < 0:
        raise ValueError("--max-turns must be >= 0")
    checkpoint = Path(args.e_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"--e-checkpoint {checkpoint} does not exist; the real BC-E "
            "checkpoint is required and is never committed to the repository")
    return {
        "mode": "debug-trace",
        "cases": cases,
        "backend": str(args.backend),
        "composition": str(args.composition),
        "e_checkpoint": str(checkpoint),
        "policy_seed": int(args.policy_seed),
        "num_threads": int(args.num_threads),
        "max_turns": int(args.max_turns),
        "output_dir": str(Path(args.output_dir)),
    }


# ----------------------------------------------------------- execution
# Never invoked by tests; requires the real checkpoints that are absent
# locally. Kept as thin compositions of the tested rl_manager primitives.


def _rollout_candidate_from_state(
    state: Any,
    config: Any,
    ppo_config: Any,
    previous: Any | None = None,
    deterministic: bool | None = None,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    curriculum: CurriculumMaskConfig | None = None,
    exploration: TargetedExplorationConfig | None = None,
) -> Any:
    """Build a rollout adapter bound to the exact returned train state."""
    from rl_manager.ppo_adapter import ppo_batched_policy_from_state

    if deterministic is None:
        deterministic = (previous.deterministic if previous is not None
                          else False)
    if curriculum is None and previous is not None:
        curriculum = previous.curriculum
    if exploration is None and previous is not None and not deterministic:
        exploration = previous.exploration
    if deterministic:
        from rl_manager.ppo_policy import TargetedExplorationConfig
        exploration = TargetedExplorationConfig()
    return ppo_batched_policy_from_state(
        state,
        config,
        ppo_config=ppo_config,
        name=(previous.identity.name if previous is not None
              else "ppo_candidate"),
        version=(previous.identity.version if previous is not None
                 else "ppo-v0"),
        deterministic=deterministic,
        e_history_version=(previous.identity.e_history_version
                           if previous is not None
                           and previous.identity.e_history_version is not None
                            else e_history_version),
        curriculum=curriculum,
        exploration=exploration,
    )


def execute_training(plan: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig, init_train_params

    from rl_manager.ppo import build_ppo_batch, init_train_state, ppo_update
    from rl_manager.ppo_adapter import ppo_snapshot_from_state
    from rl_manager.ppo_checkpoint import load_ppo_checkpoint
    from rl_manager.ppo_checkpoint import save_ppo_snapshot
    from rl_manager.ppo_policy import PPOConfig
    from rl_manager.ppo_policy import CurriculumMaskConfig  # parent-side only
    from rl_manager.policy import JaxEPlanPolicy
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
        build_episode_spec
    from rl_manager.seeds import SeedStream
    from rl_manager.trajectory import TrajectoryBuffer, e_input_spec
    from rl_manager.parallel import ParallelSelfPlayRunner

    frozen_params, metadata = load_torch_checkpoint(
        plan["e_checkpoint"],
        expected_e_history_version=plan["e_history_version"])
    config = ManagerConfig(**metadata["model_config"])
    curriculum = CurriculumMaskConfig.from_json_dict(plan["curriculum"])
    ppo_config = PPOConfig(**plan["ppo"])
    initial_base_params = None
    initialization_seed = None
    parent_meta: Mapping[str, Any] | None = None
    if plan.get("resume_checkpoint") is not None:
        state, parent_meta = load_ppo_checkpoint(
            plan["resume_checkpoint"], config=config,
            ppo_config=ppo_config,
            expected_e_history_version=plan["e_history_version"])
    else:
        if plan["init_mode"] == "scratch":
            initialization_seed = SeedStream(
                plan["master_seed"]).initialization_seed()
            initial_base_params = init_train_params(
                config, seed=initialization_seed, model_variant="E")
        state = init_train_state(
            frozen_params, config, seed=plan["master_seed"],
            ppo_config=ppo_config, initial_base_params=initial_base_params,
            curriculum=curriculum)
    candidate = _rollout_candidate_from_state(
        state, config, ppo_config,
        e_history_version=plan["e_history_version"], curriculum=curriculum,
        exploration=_exploration_for_update(plan, 0))
    print(f"curriculum={json.dumps(curriculum.to_json_dict(), sort_keys=True)}")
    original_bc_e = JaxEPlanPolicy(
        frozen_params, config, name="frozen_e",
        e_history_version=plan["e_history_version"])
    ratchet = PromotionRatchet(original_bc_e)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    promotion_dir = output_dir / "promotions"

    seed_stream = SeedStream(plan["master_seed"])
    history = []
    promotion_checks: list[dict[str, Any]] = []
    stop_after_promotion = False
    for update_index in range(plan["updates"]):
        exploration = _exploration_for_update(plan, update_index)
        candidate = _rollout_candidate_from_state(
            state, config, ppo_config, previous=candidate,
            e_history_version=plan["e_history_version"], curriculum=curriculum,
            exploration=exploration)
        updates_remaining = max(
            0, int(plan["unlock_exploration_updates"]) - update_index - 1)
        print(
            "unlock_exploration "
            f"active={str(not exploration.inactive).lower()} "
            f"epsilon={float(exploration.epsilon):g} "
            f"updates_remaining={updates_remaining} "
            f"land_target={exploration.land_target} "
            f"goose_target={exploration.goose_target} "
            f"cow_target={exploration.cow_target} "
            f"sheep_target={exploration.sheep_target}")
        buffer = TrajectoryBuffer(
            capacity=plan["expected_trajectory_rows"],
            input_spec=e_input_spec(),
            e_history_version=plan["e_history_version"])
        runner_config = RunnerConfig(
            backend_name=plan["backend"],
            backend_configuration={"seed": 0,
                                   "numThreads": plan["knobs"]["num_threads"]},
            num_envs=plan["knobs"]["num_envs"],
            opening=plan["opening"],
            manager_start_day=plan["manager_start_day"],
            e_history_version=plan["e_history_version"],
            reward_config=RewardConfig(**plan["reward"]),
            **plan["runner_options"])
        executor_factory = _resolve_executor_factory(
            plan["executor_factory"],
            low_telemetry=plan["runner_options"]["low_telemetry"])
        runner = (ParallelSelfPlayRunner(
            runner_config, num_workers=plan["knobs"]["num_workers"],
            trajectory_buffer=buffer, executor_factory=executor_factory,
            master_seed=plan["master_seed"])
            if plan["knobs"]["num_workers"] > 1 else SelfPlayRunner(
                runner_config, trajectory_buffer=buffer,
                executor_factory=executor_factory,
                master_seed=plan["master_seed"]))
        # Candidate-v-frozen keeps its historical alternating seat orientation;
        # economic self-play intentionally uses the same live policy twice.
        specs = []
        for episode in range(plan["episodes_per_update"]):
            composition = (plan["training_composition"]
                           if plan["training_composition"] ==
                           CURRENT_VS_CURRENT_ECONOMIC else
                           ("candidate_vs_frozen" if episode % 2 == 0
                            else "frozen_vs_candidate"))
            specs.append(build_episode_spec(
                update_index * plan["episodes_per_update"] + episode,
                seed_stream.episode_seed(
                    update_index * plan["episodes_per_update"] + episode),
                composition, candidate, ratchet.current_opponent))
        results = runner.run(specs)
        update_number = update_index + 1
        # Persist rollout economics before the logprob audit / PPO update so
        # a later failure cannot lose the completed rollout evidence.
        from rl_manager.diagnostics import (build_economic_diagnostics,
                                            write_diagnostics)
        economic = build_economic_diagnostics(
            results, crop_action_max=config.count_max)
        rollout_arrays = buffer.finalize()
        target_rates = _target_action_rates(rollout_arrays, exploration)
        economic["unlock_target_rates"] = target_rates
        economic["unlock_exploration"] = exploration.to_json_dict()
        economic["update"] = update_number
        economic["reward_config"] = dict(plan["reward"])
        write_diagnostics(
            output_dir / f"economic_update_{update_number:06d}.json", economic)
        batch = build_ppo_batch(
            rollout_arrays, gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            sidecar_records=buffer.sidecar_records)
        from rl_manager.ppo_adapter import recompute_stored_action_logprobs
        physical_batch_size = plan["runner_options"][
            "fixed_inference_batch_size"]
        recomputed_logprob = recompute_stored_action_logprobs(
            candidate._policy, batch.inputs, batch.action_tensors,
            physical_batch_size=physical_batch_size)
        max_abs_error = float(np.max(np.abs(
            recomputed_logprob.astype(np.float64)
            - np.asarray(batch.old_logprob, dtype=np.float64))))
        print(f"logprob_audit physical_batch_size={physical_batch_size} "
              f"rows={batch.size} max_abs_error={max_abs_error}")
        if not np.allclose(recomputed_logprob, batch.old_logprob,
                           rtol=0.0, atol=1e-5):
            raise ValueError(
                "rollout stored logprobs do not match active PPO policy "
                f"recomputation (max_abs_error={max_abs_error})")
        state, metrics = ppo_update(
            state, batch, config, ppo_config, curriculum=curriculum,
            exploration=exploration)
        if metrics["accepted"]:
            candidate = _rollout_candidate_from_state(
                state, config, ppo_config, previous=candidate,
                e_history_version=plan["e_history_version"],
                curriculum=curriculum, exploration=exploration)
        print(f"UPDATE {update_number}")
        rollout_opponent = (ratchet.current_opponent.identity.fingerprint
                            if plan["training_composition"] !=
                            CURRENT_VS_CURRENT_ECONOMIC else "not_used")
        print(f"learner={candidate.identity.fingerprint} "
              f"rollout_opponent={rollout_opponent} ppo_step={state.step}")
        console_aggregate = dict(economic["aggregate"])
        console_intent = dict(console_aggregate.get("manager_crop_intent", {}))
        console_intent.pop("by_manager_day", None)
        console_aggregate["manager_crop_intent"] = console_intent
        print(f"economic={json.dumps(console_aggregate, sort_keys=True)}")
        intent = economic["aggregate"]["manager_crop_intent"]
        print(
            "manager_crop_intent "
            f"requested_total_mean={intent['requested_total']['mean']} "
            f"distinct_species_mean={intent['mix']['mean_distinct_species_requested']} "
            "target_vector_change_fraction="
            f"{intent['fraction_target_vector_changed_from_previous_manager_day']['fraction']} "
            "component_at_max_fraction="
            f"{intent['saturation']['fraction_crop_components_at_action_max']} "
            "all_crop_heads_at_max_fraction="
            f"{intent['saturation']['fraction_manager_rows_all_crop_components_at_action_max']} "
            "unresolved_deficit_fraction="
            f"{intent['unresolved_crop_deficit']['fraction_rows_with_unresolved_deficit']} "
            "eod_shortfall_mean="
            f"{intent['end_of_day_shortfall']['mean_units_per_day']} "
            "late_crop_request_mean_d28_29="
            f"{intent['late_game']['28-29']['requested_total_mean']}"
        )
        for name, rate in target_rates.items():
            print(f"unlock_target_rate {name}={rate['target']} "
                  f"sampled_fraction={rate['sampled_fraction']}")

        promotion = plan["promotion"]
        if promotion["every"] and update_number % promotion["every"] == 0:
            eval_candidate = _rollout_candidate_from_state(
                state, config, ppo_config, previous=candidate,
                deterministic=True,
                e_history_version=plan["e_history_version"],
                curriculum=curriculum)
            eval_runner = (ParallelSelfPlayRunner(
                runner_config, num_workers=plan["knobs"]["num_workers"],
                executor_factory=executor_factory)
                           if plan["knobs"]["num_workers"] > 1 else
                           SelfPlayRunner(runner_config,
                                          executor_factory=executor_factory))
            eval_specs = []
            for seed in promotion["seeds"]:
                for orientation in ("candidate_vs_frozen",
                                    "frozen_vs_candidate"):
                    eval_specs.append(build_episode_spec(
                        len(eval_specs), seed, orientation, eval_candidate,
                        ratchet.current_opponent))
            eval_results = eval_runner.run(eval_specs)
            summary = summarize_evaluation(
                eval_results, expected_seeds=promotion["seeds"],
                provenance={
                    "seed_set": promotion["seed_set"],
                    "candidate_identity": eval_candidate.identity.to_json_dict(),
                    "opponent_identity": (
                        ratchet.current_opponent.identity.to_json_dict()),
                    "original_bc_e_identity": (
                        original_bc_e.identity.to_json_dict()),
                    "update": update_number,
                     "ppo_step": int(state.step),
                     "init_mode": plan["init_mode"],
                     "curriculum": curriculum.to_json_dict(),
                     "unlock_exploration": plan["unlock_exploration"],
                     "unlock_exploration_updates": plan[
                         "unlock_exploration_updates"],
                 })
            decision = evaluate_promotion(summary)
            summary["promotion"] = decision.to_dict()
            snapshot = ppo_snapshot_from_state(
                state, config, ppo_config=ppo_config,
                name=f"promotion_{ratchet.promotions + 1:03d}",
                version="ratchet-v1",
                e_history_version=plan["e_history_version"],
                curriculum=curriculum)
            gate = "PASS" if decision.passed else "HOLD"
            print(
                f"RATCHET update={update_number} "
                f"candidate={eval_candidate.identity.fingerprint} "
                f"snapshot={snapshot.identity.fingerprint} "
                f"W-L-T={summary['wlt']['W']}-{summary['wlt']['L']}-"
                f"{summary['wlt']['T']} "
                f"mean_margin={summary['mean_margin']} "
                f"median_margin={summary['median_margin']} "
                f"gate={gate} reasons={list(decision.failed_reasons)!r}")
            promotion_checks.append({"update": update_number,
                                     "summary": summary})
            if decision.passed:
                old_snapshot = ratchet.current_opponent
                promotion_number = ratchet.promotions + 1
                snapshot_path = save_ppo_snapshot(
                    promotion_dir / f"promotion_{promotion_number:03d}.npz",
                    state, config, ppo_config,
                    snapshot_identity=snapshot.identity.to_json_dict(),
                    provenance={
                        "update": update_number,
                        "ppo_step": int(state.step),
                        "original_bc_e": original_bc_e.identity.to_json_dict(),
                        "evaluation_seed_set": promotion["seed_set"],
                        "init_mode": plan["init_mode"],
                        "curriculum": curriculum.to_json_dict(),
                        "unlock_exploration": plan["unlock_exploration"],
                        "unlock_exploration_updates": plan[
                            "unlock_exploration_updates"],
                    },
                    e_history_version=plan["e_history_version"],
                    curriculum=curriculum)
                eval_path = promotion_dir / (
                    f"promotion_{promotion_number:03d}_eval.json")
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                eval_path.write_text(
                    json.dumps({
                        "evaluation": summary,
                        "promotion": decision.to_dict(),
                        "snapshot": snapshot.identity.to_json_dict(),
                        "snapshot_path": str(snapshot_path),
                    }, sort_keys=True, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")
                ratchet.apply(True, snapshot)
                print(f"PROMOTION #{ratchet.promotions} "
                      f"old_snapshot={old_snapshot.identity.fingerprint} "
                      f"new_snapshot={snapshot.identity.fingerprint} "
                      f"checkpoint={snapshot_path}")
                if (promotion["max_promotions"] is not None
                        and ratchet.promotions >= promotion["max_promotions"]):
                    stop_after_promotion = True
        from rl_manager.ppo_checkpoint import save_ppo_checkpoint

        path = save_ppo_checkpoint(
            output_dir / f"ppo_update_{update_index:06d}.npz", state, config,
            ppo_config, e_history_version=plan["e_history_version"],
            provenance={"plan": dict(plan),
                        "training_composition": plan["training_composition"],
                        "reward_config": dict(plan["reward"]),
                        "init_mode": plan["init_mode"],
                        "initialization_seed": initialization_seed,
                        "curriculum": curriculum.to_json_dict(),
                        "parent_checkpoint": plan.get("resume_checkpoint"),
                        "parent_curriculum": (None if parent_meta is None else
                                               parent_meta.get("curriculum"))},
            curriculum=curriculum)
        history.append({"update": update_index, "metrics": metrics,
                        "checkpoint": str(path),
                        "init_mode": plan["init_mode"],
                        "learner": candidate.identity.to_json_dict(),
                        "opponent": (None if plan["training_composition"] ==
                                      CURRENT_VS_CURRENT_ECONOMIC else
                                      ratchet.current_opponent.identity.to_json_dict()),
                        "frozen_reference": (
                            ratchet.current_opponent.identity.to_json_dict()),
                        "promotions": ratchet.promotions})
        if stop_after_promotion:
            break
    final_path = save_ppo_checkpoint(
        plan["checkpoint"], state, config, ppo_config,
        e_history_version=plan["e_history_version"],
        provenance={"plan": dict(plan),
                    "training_composition": plan["training_composition"],
                    "reward_config": dict(plan["reward"]),
                    "init_mode": plan["init_mode"],
                    "initialization_seed": initialization_seed,
                    "curriculum": curriculum.to_json_dict(),
                    "parent_checkpoint": plan.get("resume_checkpoint"),
                    "parent_curriculum": (None if parent_meta is None else
                                           parent_meta.get("curriculum"))},
        curriculum=curriculum)
    return {"history": history, "promotion_checks": promotion_checks,
            "init_mode": plan["init_mode"],
            "promotions": ratchet.promotions,
            "original_bc_e": original_bc_e.identity.to_json_dict(),
            "final_opponent": ratchet.current_opponent.identity.to_json_dict(),
            "final_checkpoint": str(final_path)}


def execute_evaluation(plan: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig

    from rl_manager.ppo_adapter import ppo_batched_policy_from_state
    from rl_manager.ppo_checkpoint import load_ppo_checkpoint
    from rl_manager.ppo_policy import CurriculumMaskConfig  # parent-side only
    from rl_manager.policy import JaxEPlanPolicy
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
        build_episode_spec
    from rl_manager.parallel import ParallelSelfPlayRunner

    frozen_params, metadata = load_torch_checkpoint(
        plan["e_checkpoint"],
        expected_e_history_version=plan["e_history_version"])
    config = ManagerConfig(**metadata["model_config"])
    curriculum = CurriculumMaskConfig.from_json_dict(plan["curriculum"])
    state, checkpoint_meta = load_ppo_checkpoint(
        plan["checkpoint"], config=config,
        expected_e_history_version=plan["e_history_version"])
    candidate = ppo_batched_policy_from_state(
        state, config, name="ppo_candidate", deterministic=True,
        e_history_version=plan["e_history_version"],
        curriculum=curriculum)
    print(f"curriculum={json.dumps(curriculum.to_json_dict(), sort_keys=True)}")
    frozen_policy = JaxEPlanPolicy(
        frozen_params, config, name="frozen_e",
        e_history_version=plan["e_history_version"])
    runner_config = RunnerConfig(
        backend_name=plan["backend"],
        backend_configuration={"seed": 0,
                               "numThreads": plan["knobs"]["num_threads"]},
        num_envs=plan["knobs"]["num_envs"],
        e_history_version=plan["e_history_version"],
        **plan["runner_options"])
    executor_factory = _resolve_executor_factory(
        plan["executor_factory"],
        low_telemetry=plan["runner_options"]["low_telemetry"])
    runner = (ParallelSelfPlayRunner(
        runner_config, num_workers=plan["knobs"]["num_workers"],
        executor_factory=executor_factory)
        if plan["knobs"]["num_workers"] > 1 else SelfPlayRunner(
            runner_config, executor_factory=executor_factory))
    specs = []
    for seed in plan["seeds"]:
        for orientation in plan["seat_orientations"]:
            specs.append(build_episode_spec(len(specs), seed, orientation,
                                            candidate, frozen_policy))
    results = runner.run(specs)
    summary = summarize_evaluation(
        results,
        expected_seeds=plan["seeds"],
        provenance={
            "seed_set": plan["seed_set"],
            "candidate_identity": candidate.identity.to_json_dict(),
            "opponent_identity": frozen_policy.identity.to_json_dict(),
            "curriculum": curriculum.to_json_dict(),
            "checkpoint_curriculum": checkpoint_meta.get("curriculum"),
            "init_mode": checkpoint_meta.get("provenance", {}).get(
                "init_mode"),
        },
    )
    decision = evaluate_promotion(summary)
    summary["promotion"] = decision.to_dict()
    print(format_promotion_result(summary, decision))
    from rl_manager.diagnostics import write_diagnostics

    write_diagnostics(plan["output_json"], summary)
    return summary


def _make_debug_trace_policy(plan: Mapping[str, Any]) -> Any:
    """Load the explicit frozen BC-E checkpoint through the JAX policy seam."""
    from bc_manager_jax.checkpoint import load_torch_checkpoint
    from bc_manager_jax.model import ManagerConfig
    from rl_manager.policy import JaxEPlanPolicy

    checkpoint = plan["e_checkpoint"]
    params, metadata = load_torch_checkpoint(
        checkpoint, expected_e_history_version=E_HISTORY_CORRECTED_V1)
    try:
        config = ManagerConfig(**metadata["model_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"BC-E checkpoint {checkpoint} lacks metadata.model_config; "
            "supply a compatible committed BC-E checkpoint") from exc
    return JaxEPlanPolicy(
        params, config, name="trace_e",
        e_history_version=metadata["e_history_version"])


def execute_debug_trace(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run selected runner episodes and write validated canonical traces."""
    from rl_manager.debug_trace import load_trace, save_trace, validate_trace
    from rl_manager.runner import (
        RunnerConfig,
        SelfPlayRunner,
        build_episode_spec,
    )
    from rl_manager.policy import PassPlanPolicy

    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = _make_debug_trace_policy(plan)
    opponent_policy = (
        PassPlanPolicy() if plan["composition"] == E_VS_PASS else policy)
    summaries: list[dict[str, Any]] = []
    for episode_index, (seed, seat) in enumerate(plan["cases"]):
        runner = SelfPlayRunner(
            RunnerConfig(
                backend_name=plan["backend"],
                backend_configuration={
                    "seed": 0, "numThreads": int(plan["num_threads"])},
                opening="standard_mixed",
                max_turns=int(plan["max_turns"]),
                record_debug_trace=True,
                debug_trace_seat=seat,
                debug_trace_view="joint",
            ),
            master_seed=seed,
        )
        spec = build_episode_spec(
            episode_index, seed, plan["composition"], policy,
            opponent_policy, controlled_seat=seat)
        try:
            result = runner.run([spec])[0]
        except ModuleNotFoundError as exc:
            if exc.name == "fast_env._kaggriculture_env":
                raise RuntimeError(
                    "the fast backend native module is unavailable; build or "
                    "install the project native wheel before running debug-trace "
                    "generation, or pass --backend official when the official "
                    "engine dependency is installed"
                ) from exc
            raise
        if result.debug_trace is None:  # pragma: no cover - runner seam guard
            raise RuntimeError("runner returned no debug trace after opt-in capture")
        path = output_dir / f"seed_{seed}_seat_{seat}.json"
        # save_trace intentionally replaces an existing same-case file with the
        # deterministic canonical representation; callers may rerun a case.
        save_trace(path, result.debug_trace)
        loaded = load_trace(path)
        validate_trace(loaded)
        size = path.stat().st_size
        summary = {
            "seed": seed,
            "seat": seat,
            "composition": plan["composition"],
            "turns": len(loaded["turns"]),
            "path": str(path),
            "bytes": size,
            "winner_seat": result.winner_seat,
            "terminated": result.terminated,
        }
        print(
            f"trace seed={seed} seat={seat} composition={summary['composition']} "
            f"turns={summary['turns']} "
            f"path={path} bytes={size} winner_seat={result.winner_seat} "
            f"terminated={result.terminated}"
        )
        summaries.append(summary)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        plan = plan_training(args)
        execute_training(plan)  # pragma: no cover
        return 0
    if args.command == "debug-trace":
        plan = plan_debug_trace(args)
        execute_debug_trace(plan)
        return 0
    plan = plan_evaluation(args)
    execute_evaluation(plan)  # pragma: no cover
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
