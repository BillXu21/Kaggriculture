"""Two-policy native Stage 2.5 PPO self-play trainer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.reward import (
    BEHAVIOR_SHAPING_FEATURES,
    TERMINAL_OWN_BANK,
    TERMINAL_WLT,
    BehaviorShapingConfig,
)
from rl_manager.runner import build_episode_spec
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer
from rl_manager.types import DUAL_POLICY_SELF_PLAY


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init", type=Path,
                        help="native Stage 2.5 BC/inference checkpoint")
    source.add_argument("--resume", type=Path,
                        help="dual-policy Stage 2.5 PPO checkpoint")
    parser.add_argument("--migrate-physical-baseline-bc", action="store_true",
                        help="use the strict existing physical-baseline BC migration seam")
    parser.add_argument("--model-size", choices=("tiny", "small", "large"),
                        default="tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=("fast", "official"), default="fast")
    parser.add_argument("--executor", choices=("strip", "legacy"), default="strip")
    parser.add_argument("--opening", default="standard_mixed")
    parser.add_argument("--reward-mode", choices=(TERMINAL_WLT, TERMINAL_OWN_BANK),
                        default=TERMINAL_WLT)
    parser.add_argument("--bank-reward-baseline", type=float, default=3000.0)
    parser.add_argument("--bank-reward-scale", type=float, default=50000.0)
    for feature in BEHAVIOR_SHAPING_FEATURES:
        parser.add_argument(f"--shape-{feature}-target", type=int, default=None)
        parser.add_argument(f"--shape-{feature}-weight", type=float, default=None)
    parser.add_argument("--allow-shaping-change-on-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--batch-backend", action="store_true")
    parser.add_argument("--stage25-rollout-profile", action="store_true")
    parser.add_argument("--inference-batch-wait-ms", type=float, default=20.0)
    parser.add_argument("--rollout-size", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=144)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.add_argument("--stage25-inference-validation",
                        choices=("strict", "fast", "none"), default="strict")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--jax-compilation-cache-dir", type=Path)
    return parser


def _reward_config(args: argparse.Namespace) -> Any:
    from rl_manager.stage25_ppo_cli import _reward_config as single_reward_config

    # The shared helper performs the existing shaping and terminal reward
    # validation. Composition has no bearing on the reward calculation here.
    args.training_composition = DUAL_POLICY_SELF_PLAY
    return single_reward_config(args)


def _training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_composition": DUAL_POLICY_SELF_PLAY,
        "reward": _reward_config(args).to_json_dict(),
    }


def _validate_args(args: argparse.Namespace) -> None:
    from rl_manager.stage25_ppo_cli import _validate_rollout_controls

    args.scratch = False
    args.scratch_hold_prior_tau = None
    args.training_composition = DUAL_POLICY_SELF_PLAY
    if args.migrate_physical_baseline_bc and args.init is None:
        raise ValueError("--migrate-physical-baseline-bc requires --init")
    if args.updates < 1 or args.rollout_size < 1 or args.workers < 1:
        raise ValueError("updates, rollout-size, and workers must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    _validate_rollout_controls(args)


def _new_or_resume_state(args: argparse.Namespace, config: Any) -> tuple[Any, dict[str, Any]]:
    from rl_manager.stage25_checkpoint import (
        _source_identity,
        initialize_stage25_ppo_from_checkpoint,
        load_stage25_dual_ppo_checkpoint,
        migrate_stage25_bc_checkpoint_for_ppo,
    )
    from rl_manager.stage25_ppo import init_stage25_dual_ppo_state

    if args.init is not None:
        if args.migrate_physical_baseline_bc:
            params, source_meta = migrate_stage25_bc_checkpoint_for_ppo(
                args.init, config=config.model)
        else:
            params, source_meta = initialize_stage25_ppo_from_checkpoint(
                args.init, config=config.model, seed=None)
        source_meta = dict(source_meta)
        source_meta["dual_source_identity"] = _source_identity(
            args.init, source_meta)
        return init_stage25_dual_ppo_state(
            config, seed=args.seed, params=params), source_meta

    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_ppo_cli import (
        _contract_without_behavior_shaping, _physical_contract,
        _resolve_executor_factory)

    fresh = init_stage25_dual_ppo_state(config, seed=args.seed)
    state, metadata = load_stage25_dual_ppo_checkpoint(
        args.resume, ppo_config=config,
        optimizer_state_template_a=fresh.policy_a.optimizer_state,
        optimizer_state_template_b=fresh.policy_b.optimizer_state,
        expected_training_contract=None,
        expected_physical_contract=_physical_contract(config),
        expected_executor=_executor_factory_provenance(
            _resolve_executor_factory(args.executor)))
    stored_contract = metadata["dual_state"].get("training_contract", {})
    requested_contract = _training_contract(args)
    contract_matches = stored_contract == requested_contract
    if (not contract_matches and args.allow_shaping_change_on_resume
            and _contract_without_behavior_shaping(stored_contract)
            == _contract_without_behavior_shaping(requested_contract)):
        contract_matches = True
    if not contract_matches:
        raise ValueError(
            "dual PPO checkpoint training/reward contract does not match: "
            f"{stored_contract!r} != {requested_contract!r}")
    return state, metadata


def _collection(state: Any, config: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_ppo import build_stage25_dual_ppo_batches
    from rl_manager.stage25_ppo_cli import (
        _resolve_executor_factory, _runner_config)

    started = time.perf_counter()
    policy_a = Stage25InferenceAdapter(
        params=state.policy_a.params, config=config.model,
        name=state.policy_a.behavior_identity.name,
        version=state.policy_a.behavior_identity.version,
        seed=state.policy_a.rollout_seed, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
    policy_b = Stage25InferenceAdapter(
        params=state.policy_b.params, config=config.model,
        name=state.policy_b.behavior_identity.name,
        version=state.policy_b.behavior_identity.version,
        seed=state.policy_b.rollout_seed, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
    if policy_a.identity != state.policy_a.behavior_identity:
        raise ValueError("Policy A identity does not match its frozen parameters")
    if policy_b.identity != state.policy_b.behavior_identity:
        raise ValueError("Policy B identity does not match its frozen parameters")

    trajectory = Stage25TrajectoryBuffer(
        max(1, args.rollout_size * 2 * 26))
    reward_config = _reward_config(args)
    runner_config = _runner_config(
        args, seed=state.rollout_seed, reward_config=reward_config)
    runner = ParallelSelfPlayRunner(
        runner_config, num_workers=args.workers,
        master_seed=state.rollout_seed,
        executor_factory=_resolve_executor_factory(args.executor),
        stage25_trajectory_buffer=trajectory)
    episode_start = int(state.rollout_progression.get("next_episode_index", 0))
    specs = tuple(build_episode_spec(
        episode_start + offset, state.rollout_seed + offset,
        DUAL_POLICY_SELF_PLAY, policy_a, policy_b)
        for offset in range(args.rollout_size))
    rollout_started = time.perf_counter()
    results = runner.run(specs)
    rollout_seconds = time.perf_counter() - rollout_started

    batch_started = time.perf_counter()
    batch_a, batch_b = build_stage25_dual_ppo_batches(
        trajectory, learner_identity_a=policy_a.identity,
        learner_identity_b=policy_b.identity, gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        normalize_advantages=config.normalize_advantages)
    batch_seconds = time.perf_counter() - batch_started
    executor = _executor_factory_provenance(runner.provenance["executor_factory"])
    trajectory.validate_executor_provenance(executor)
    return {
        "trajectory": trajectory,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "batch_a": batch_a,
        "batch_b": batch_b,
        "results": results,
        "executor_provenance": executor,
        "opening_provenance": runner.provenance.get("opening"),
        "inference_metrics": runner.inference_metrics,
        "rollout_profile": (getattr(runner, "rollout_profile", None)
                            if args.stage25_rollout_profile else None),
        "timing": {
            "rollout_seconds": rollout_seconds,
            "batch_construction_seconds": batch_seconds,
            "collection_seconds": time.perf_counter() - started,
        },
    }


def _head_to_head(results: list[Any], identity_a: Any,
                  identity_b: Any) -> dict[str, Any]:
    counts = {"A_wins": 0, "B_wins": 0, "ties": 0}
    banks_a: list[float] = []
    banks_b: list[float] = []
    for result in results:
        a_seat = next(
            int(record["seat"]) for record in result.policy_identities
            if record["policy"].get("identity_id") == identity_a.identity_id())
        b_seat = 1 - a_seat
        banks_a.append(float(result.final_banks[a_seat]))
        banks_b.append(float(result.final_banks[b_seat]))
        if result.winner_seat < 0:
            counts["ties"] += 1
        elif result.winner_seat == a_seat:
            counts["A_wins"] += 1
        else:
            counts["B_wins"] += 1
    games = len(results)
    mean_a = math.fsum(banks_a) / games if games else 0.0
    mean_b = math.fsum(banks_b) / games if games else 0.0
    return {
        **counts,
        "A_win_fraction": counts["A_wins"] / games if games else 0.0,
        "mean_bank_A": mean_a,
        "mean_bank_B": mean_b,
        "mean_bank_margin_A_minus_B": mean_a - mean_b,
    }


def _shaping_by_policy(results: list[Any], shaping: BehaviorShapingConfig,
                       identity_a: Any, identity_b: Any) -> dict[str, Any]:
    all_records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for result in results:
        for seat, record in (result.behavior_shaping or {}).items():
            policy_identity = result.policy_identities[int(seat)]["policy"]
            label = ("A" if policy_identity.get("identity_id")
                     == identity_a.identity_id() else "B")
            all_records[label].append(record)
    output = {}
    for label, records in all_records.items():
        total = math.fsum(float(item["total"]) for item in records)
        features: dict[str, Any] = {}
        for name, _ in shaping.active_features():
            rows = [item["features"][name] for item in records]
            initial = [float(row["initial_count"]) for row in rows
                       if row["initial_count"] is not None]
            final = [float(row["final_count"]) for row in rows
                     if row["final_count"] is not None]
            reached = [float(bool(row["target_reached"])) for row in rows
                       if row["target_reached"] is not None]
            features[name] = {
                "mean_episode_contribution": (
                    math.fsum(float(row["reward"]) for row in rows)
                    / len(records) if records else 0.0),
                "mean_initial_count": math.fsum(initial) / len(initial)
                if initial else 0.0,
                "mean_final_count": math.fsum(final) / len(final)
                if final else 0.0,
                "target_reached_fraction": math.fsum(reached) / len(reached)
                if reached else 0.0,
            }
        output[label] = {
            "total": total,
            "mean_per_episode": total / len(records) if records else 0.0,
            "learner_episodes": len(records),
            "features": features,
        }
    return output


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_checkpoint import save_stage25_dual_ppo_checkpoint
    from rl_manager.stage25_ppo import ppo_update_dual
    from rl_manager.stage25_ppo_cli import (
        _append_jsonl, _configure_jax_compilation_cache, _config,
        _physical_contract, _resolve_executor_factory)

    _validate_args(args)
    _configure_jax_compilation_cache(args.jax_compilation_cache_dir)
    if args.workers == 1 and args.physical_batch_size != 1:
        raise ValueError(
            "single-process collection requires --physical-batch-size 1; "
            "use --workers 2+ for central fixed-size owner batching")
    config = _config(args)
    runtime_executor = _executor_factory_provenance(
        _resolve_executor_factory(args.executor))
    state, source_meta = _new_or_resume_state(args, config)
    requested_contract = _training_contract(args)
    if args.resume is not None and source_meta.get("executor") != runtime_executor:
        raise ValueError("dual PPO checkpoint executor provenance does not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    latest_checkpoint = str(args.resume) if args.resume else None
    initial_info = None
    if args.init is not None:
        from rl_manager.stage25_inference import parameter_fingerprint

        initial_info = {
            "A_source_checkpoint_identity": source_meta.get(
                "dual_source_identity", source_meta.get("source_identity", {})),
            "B_source_checkpoint_identity": source_meta.get(
                "dual_source_identity", source_meta.get("source_identity", {})),
            "A_initial_parameter_fingerprint": parameter_fingerprint(
                state.policy_a.params),
            "B_initial_parameter_fingerprint": parameter_fingerprint(
                state.policy_b.params),
            "A_B_initial_params_equal": _trees_equal(
                state.policy_a.params, state.policy_b.params),
            "A_behavior_identity": state.policy_a.behavior_identity.to_json_dict(),
            "B_behavior_identity": state.policy_b.behavior_identity.to_json_dict(),
        }
    all_metrics = []
    for local_generation in range(args.updates):
        started = time.perf_counter()
        rollout = _collection(state, config, args=args)
        if rollout["executor_provenance"] != runtime_executor:
            raise ValueError("rollout executor provenance is not configured factory")
        next_state, update = ppo_update_dual(
            state, rollout["batch_a"], rollout["batch_b"], config,
            rollout_size=args.rollout_size)
        head_to_head = _head_to_head(
            rollout["results"], state.policy_a.behavior_identity,
            state.policy_b.behavior_identity)
        reward_config = _reward_config(args)
        ppo_a = dict(update["ppo_A"])
        ppo_b = dict(update["ppo_B"])
        ppo_a["entropy"] = ppo_a.get("entropy_surrogate", 0.0)
        ppo_b["entropy"] = ppo_b.get("entropy_surrogate", 0.0)
        ppo_a["audit"] = update["audit_A"]
        ppo_b["audit"] = update["audit_B"]
        record: dict[str, Any] = {
            "generation": next_state.generation,
            "games": len(rollout["results"]),
            "rows_A": len(rollout["batch_a"].classes),
            "rows_B": len(rollout["batch_b"].classes),
            "head_to_head": head_to_head,
            "ppo_A": ppo_a,
            "ppo_B": ppo_b,
            "audits": {"A": update["audit_A"], "B": update["audit_B"]},
            "reward": reward_config.to_json_dict(),
            "training_contract": requested_contract,
            "behavior_identity_A": next_state.policy_a.behavior_identity.to_json_dict(),
            "behavior_identity_B": next_state.policy_b.behavior_identity.to_json_dict(),
            "inference_metrics": rollout["inference_metrics"],
            "trajectory": rollout["trajectory"].diagnostic_summary(),
            "timing": {
                **rollout["timing"],
                "ppo_A_seconds": update["timing"]["ppo_A_seconds"],
                "ppo_B_seconds": update["timing"]["ppo_B_seconds"],
                "checkpoint_seconds": 0.0,
                "total_seconds": 0.0,
            },
            "checkpoint": latest_checkpoint,
            "checkpoint_saved": False,
        }
        if initial_info is not None and local_generation == 0 \
                and state.generation == 0:
            record["initialization"] = initial_info
        if reward_config.behavior_shaping.enabled:
            record["behavior_shaping"] = _shaping_by_policy(
                rollout["results"], reward_config.behavior_shaping,
                state.policy_a.behavior_identity,
                state.policy_b.behavior_identity)
        if rollout["rollout_profile"] is not None:
            record["rollout_profile"] = rollout["rollout_profile"]

        checkpoint_saved = (
            next_state.generation % args.checkpoint_every == 0
            or local_generation == args.updates - 1)
        checkpoint_seconds = 0.0
        if checkpoint_saved:
            checkpoint = args.output_dir / "latest.npz"
            checkpoint_started = time.perf_counter()
            save_stage25_dual_ppo_checkpoint(
                checkpoint, next_state, config,
                seed=int(source_meta.get("init_params", {}).get("seed", args.seed)),
                training_contract=requested_contract,
                physical_contract=_physical_contract(config),
                executor=runtime_executor,
                metadata={
                    "cli": "rl_manager.stage25_dual_ppo_cli",
                    "cli_args": vars(args),
                    "source_checkpoint": str(args.init) if args.init else None,
                    "initialization": initial_info,
                    "resume_from": (None if not args.resume else {
                        "path": str(args.resume),
                        "generation": source_meta.get("dual_state", {}).get(
                            "generation"),
                    }),
                },
                provenance={
                    "run": {
                        "opening": rollout["opening_provenance"],
                        "training_contract": requested_contract,
                    },
                },
                source_identity=(source_meta.get("dual_source_identity")
                                 or source_meta.get("source_identity") or None),
                source_history_version=(
                    (source_meta.get("source_e_identity") or {}).get(
                        "history_version")),
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            latest_checkpoint = str(checkpoint)
        state = next_state
        record["checkpoint"] = latest_checkpoint
        record["checkpoint_saved"] = checkpoint_saved
        record["timing"]["checkpoint_seconds"] = checkpoint_seconds
        record["timing"]["total_seconds"] = time.perf_counter() - started
        _append_jsonl(metrics_path, record)
        if args.json_stdout:
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        else:
            print(
                f"generation={record['generation']} games={record['games']} "
                f"rows_A={record['rows_A']} rows_B={record['rows_B']} "
                f"A_wins={head_to_head['A_wins']} "
                f"B_wins={head_to_head['B_wins']} ties={head_to_head['ties']} "
                f"bank_A={head_to_head['mean_bank_A']:.1f} "
                f"bank_B={head_to_head['mean_bank_B']:.1f} "
                f"checkpoint_saved={checkpoint_saved}", flush=True)
        all_metrics.append(record)
    return all_metrics


def _trees_equal(left: Any, right: Any) -> bool:
    import jax
    import numpy as np

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return (len(left_leaves) == len(right_leaves)
            and all(np.array_equal(np.asarray(a), np.asarray(b))
                    for a, b in zip(left_leaves, right_leaves)))


def main(argv: list[str] | None = None) -> int:
    try:
        run(_parser().parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - actionable CLI boundary
        raise SystemExit(
            f"stage25 dual PPO failed before a committed generation: {exc}") from exc
    return 0


if __name__ == "__main__":
    main()
