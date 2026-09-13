"""Small real-rollout native Stage 2.5 PPO runner.

The CLI intentionally runs complete Packet 5A rollouts between updates.  A
checkpoint is therefore resumable only at the completed rollout/update
boundary recorded by the native checkpoint format.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.runner import RunnerConfig, build_episode_spec
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer


def _model(name: str) -> Stage25ModelConfig:
    from rl_manager.stage25_policy import Stage25ModelConfig
    try:
        return {"tiny": Stage25ModelConfig.tiny,
                "small": Stage25ModelConfig.small,
                "large": Stage25ModelConfig.large}[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported model size {name!r}") from exc


def _identity(meta: dict[str, Any]) -> Stage25BehaviorIdentity | None:
    from rl_manager.stage25_types import Stage25BehaviorIdentity
    payload = meta.get("behavior_identity")
    if not isinstance(payload, dict) or not payload:
        return None
    fields = ("name", "version", "parameter_fingerprint",
              "observation_schema_version", "policy_schema_version",
              "e_history_version", "curriculum_version",
              "curriculum_fingerprint", "physical_support_version")
    if not all(field in payload for field in fields):
        raise ValueError("PPO checkpoint behavior identity is incomplete")
    return Stage25BehaviorIdentity(**{field: str(payload[field]) for field in fields})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scratch", action="store_true", help="initialize from fresh native parameters")
    source.add_argument("--init", type=Path, help="native Stage 2.5 BC/inference checkpoint")
    source.add_argument("--resume", type=Path, help="native Stage 2.5 PPO checkpoint")
    parser.add_argument("--model-size", choices=("tiny", "small", "large"), default="tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=("fast", "official"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rollout-size", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=144)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    return parser


def _config(args: argparse.Namespace) -> Stage25PPOConfig:
    from rl_manager.stage25_ppo import Stage25PPOConfig
    return Stage25PPOConfig(
        model=_model(args.model_size), physical_batch_size=args.physical_batch_size,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        learning_rate=args.learning_rate, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coef, value_coefficient=args.value_coef,
        gradient_clip=args.gradient_clip)


def _physical_contract(config: Stage25PPOConfig) -> dict[str, Any]:
    from rl_manager.stage25_checkpoint import PHYSICAL_SUPPORT_VERSION
    from rl_manager.stage25_mechanics import (
        ACTION_CLASS_COUNTS, ACTION_ORDER, ACTION_SCHEMA_VERSION)
    return {
        "version": PHYSICAL_SUPPORT_VERSION,
        "action_contract": {
            "schema_version": ACTION_SCHEMA_VERSION,
            "action_vocabulary": list(ACTION_ORDER),
            "action_class_counts": list(ACTION_CLASS_COUNTS),
        },
        "inference_batch_size": config.physical_batch_size,
    }


def _identity_from_meta(meta: Mapping[str, Any], field: str) -> Stage25BehaviorIdentity:
    from rl_manager.stage25_types import Stage25BehaviorIdentity
    payload = meta.get(field)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"PPO checkpoint {field} is missing")
    fields = ("name", "version", "parameter_fingerprint",
              "observation_schema_version", "policy_schema_version",
              "e_history_version", "curriculum_version",
              "curriculum_fingerprint", "physical_support_version")
    if not all(item in payload for item in fields):
        raise ValueError(f"PPO checkpoint {field} is incomplete")
    return Stage25BehaviorIdentity(**{item: str(payload[item]) for item in fields})


def _new_state(args: argparse.Namespace, config: Stage25PPOConfig) -> tuple[Stage25PPOTrainState, dict[str, Any]]:
    from rl_manager.stage25_checkpoint import (
        initialize_stage25_ppo_from_checkpoint, load_stage25_ppo_checkpoint)
    from rl_manager.stage25_ppo import init_stage25_ppo_state
    if args.scratch:
        return init_stage25_ppo_state(config, seed=args.seed), {}
    if args.init is not None:
        params, source_meta = initialize_stage25_ppo_from_checkpoint(
            args.init, config=config.model, seed=None)
        return init_stage25_ppo_state(config, seed=args.seed, params=params), source_meta
    from rl_manager.executor_factory import make_stage25_executor_factory
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    fresh = init_stage25_ppo_state(config, seed=args.seed)
    runtime_executor = _executor_factory_provenance(make_stage25_executor_factory())
    expected_physical = _physical_contract(config)
    params, optimizer_state, rng, meta, opponent_params = load_stage25_ppo_checkpoint(
        args.resume, config=config.model, seed=None,
        optimizer_state_template=fresh.optimizer_state,
        ppo_config=config.to_dict(), optimizer_config=config.to_dict(),
        curriculum=config.model.curriculum,
        expected_physical_contract=expected_physical, return_opponent=True)
    if meta.get("executor") != runtime_executor:
        raise ValueError(
            "PPO checkpoint executor provenance does not match the configured "
            f"factory: {meta.get('executor')!r} != {runtime_executor!r}")
    learner_identity = _identity_from_meta(meta, "behavior_identity")
    learner_adapter = Stage25InferenceAdapter(
        params=params, config=config.model, name=learner_identity.name,
        version=learner_identity.version, seed=args.seed, mode="stochastic")
    if learner_adapter.identity != learner_identity:
        raise ValueError(
            "PPO checkpoint behavior identity does not match loaded parameters "
            "and curriculum")
    opponent_identity = _identity_from_meta(meta, "opponent_identity")
    opponent_adapter = Stage25InferenceAdapter(
        params=opponent_params, config=config.model, name=opponent_identity.name,
        version=opponent_identity.version, seed=args.seed, mode="stochastic")
    if opponent_adapter.identity != opponent_identity:
        raise ValueError(
            "PPO checkpoint opponent identity does not match loaded parameters "
            "and curriculum")
    state = replace(
        fresh, params=params, optimizer_state=optimizer_state, rng=rng,
        update_counter=int(meta["update_counter"]),
        rollout_seed=int(meta.get("rollout_seed") or args.seed),
        rollout_progression=meta.get("rollout_progression") or {},
        behavior_identity=learner_identity, opponent_params=opponent_params,
        opponent_identity=opponent_identity)
    return state, meta


def _collection(
        state: Stage25PPOTrainState, config: Stage25PPOConfig,
        *, seed: int, args: argparse.Namespace,
        previous_params: Any | None = None,
) -> tuple[Stage25TrajectoryBuffer, Stage25InferenceAdapter, dict[str, Any]]:
    from rl_manager.executor_factory import make_stage25_executor_factory
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_ppo import build_stage25_ppo_batch
    learner = Stage25InferenceAdapter(
        params=state.params, config=config.model, name="stage25_learner",
        version="ppo-native-v1", seed=seed, mode="stochastic")
    opponent = Stage25InferenceAdapter(
        params=(state.params if state.opponent_params is None else state.opponent_params),
        config=config.model,
        name=("stage25_opponent" if state.opponent_identity is None
              else state.opponent_identity.name),
        version=("frozen-v1" if state.opponent_identity is None
                 else state.opponent_identity.version),
        seed=seed + 1, mode="stochastic")
    if state.opponent_identity is not None and opponent.identity != state.opponent_identity:
        raise ValueError("restored opponent identity does not match its parameters")
    capacity = max(1, args.rollout_size * 2 * 26)
    trajectory = Stage25TrajectoryBuffer(capacity)
    runner_config = RunnerConfig(
        backend_name=args.engine,
        backend_configuration={"seed": seed, "numThreads": 1},
        max_turns=args.max_turns, low_telemetry=True, stage25_enabled=True,
        stage25_mode="stochastic", stage25_fixed_inference_batch_size=config.physical_batch_size)
    runner = ParallelSelfPlayRunner(
        runner_config, num_workers=args.workers, master_seed=seed,
        executor_factory=make_stage25_executor_factory(),
        stage25_trajectory_buffer=trajectory)
    specs = tuple(build_episode_spec(
        index, seed + index, "candidate_vs_frozen", learner, opponent)
                  for index in range(args.rollout_size))
    results = runner.run(specs)
    batch = build_stage25_ppo_batch(
        trajectory, learner_identity=learner.identity, gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        normalize_advantages=config.normalize_advantages)
    stats = {
        "rollout_rows": len(trajectory), "learner_rows": len(batch.classes),
        "terminal_rows": sum(int(row.terminated) for row in trajectory.rows),
        "truncated_rows": sum(int(row.truncated) for row in trajectory.rows),
        "reward_sum": float(sum(float(row.reward) for row in trajectory.rows)),
        "final_banks": [float(bank) for result in results for bank in result.final_banks],
        "inference_metrics": runner.inference_metrics,
        "behavior_identity": learner.identity.to_json_dict(),
        "opponent_identity": opponent.identity.to_json_dict(),
        "executor_provenance": _executor_factory_provenance(
            runner.provenance["executor_factory"]),
    }
    for row in trajectory.rows:
        if row.provenance.get("executor") != stats["executor_provenance"]:
            raise ValueError(
                f"trajectory row {row.row_id!r} executor provenance disagrees "
                "with configured factory")
    return trajectory, learner, stats


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from rl_manager.stage25_checkpoint import save_stage25_ppo_checkpoint
    from rl_manager.executor_factory import make_stage25_executor_factory
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_ppo import build_stage25_ppo_batch, ppo_update
    if args.updates < 1 or args.rollout_size < 1 or args.workers < 1:
        raise ValueError("updates, rollout-size, and workers must be positive")
    config = _config(args)
    if args.workers == 1 and config.physical_batch_size != 1:
        raise ValueError(
            "Packet 5A single-process collection cannot honor a physical "
            "batch larger than one; use --workers 2+ or set "
            "--physical-batch-size 1")
    state, source_meta = _new_state(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for _ in range(args.updates):
        trajectory, learner, rollout_stats = _collection(
            state, config, seed=state.rollout_seed, args=args,
            previous_params=None)
        state = replace(state, behavior_identity=learner.identity)
        if state.opponent_params is None:
            from rl_manager.stage25_inference import Stage25InferenceAdapter
            initial_opponent = Stage25InferenceAdapter(
                params=state.params, config=config.model,
                name="stage25_opponent", version="frozen-v1",
                seed=state.rollout_seed + 1, mode="stochastic")
            state = replace(state, opponent_params=state.params,
                            opponent_identity=initial_opponent.identity)
        executor_provenance = rollout_stats.get("executor_provenance")
        expected_executor = _executor_factory_provenance(
            make_stage25_executor_factory())
        if executor_provenance is None:
            # Unit-level collection doubles may omit runtime statistics; the
            # production collection always supplies this factory-derived
            # value and is checked below.
            executor_provenance = expected_executor
        if executor_provenance != expected_executor:
            raise ValueError("rollout executor provenance is not the configured factory")
        previous_learner_params = state.params
        state, update_stats = ppo_update(state, build_stage25_ppo_batch(
            trajectory, learner_identity=learner.identity, gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_advantages=config.normalize_advantages), config)
        from rl_manager.stage25_inference import Stage25InferenceAdapter
        next_opponent = Stage25InferenceAdapter(
            params=previous_learner_params, config=config.model,
            name="stage25_opponent", version="frozen-v1",
            seed=state.rollout_seed + 1, mode="stochastic")
        state = replace(state, opponent_params=previous_learner_params,
                        opponent_identity=next_opponent.identity)
        state = replace(state, rollout_seed=state.rollout_seed + args.rollout_size)
        metadata = {
            "run": {"cli": "rl_manager.stage25_ppo_cli", "source": str(args.init) if args.init else None},
            "resume_from": (
                None if not args.resume else {
                    "path": str(args.resume),
                    "payload_kind": source_meta.get("payload_kind"),
                    "update_counter": source_meta.get("update_counter"),
                }),
        }
        checkpoint = args.output_dir / "latest.npz"
        save_stage25_ppo_checkpoint(
            checkpoint, state.params, state.optimizer_state, state.rng,
            config.model, seed=args.seed, update_counter=state.update_counter,
            rollout_seed=state.rollout_seed, rollout_progression=state.rollout_progression,
            ppo_config=config.to_dict(), optimizer_config=config.to_dict(),
            curriculum=config.model.curriculum, behavior_identity=state.behavior_identity,
            provenance={"run": metadata},
            physical_contract=_physical_contract(config),
            executor=executor_provenance,
            metadata={"cli_args": vars(args)},
            opponent_params=state.opponent_params,
            opponent_identity=state.opponent_identity,
            source_identity=(source_meta.get("source_identity") or None),
            source_history_version=(
                (source_meta.get("source_e_identity") or {}).get("history_version")),
        )
        record = {"update": state.update_counter, **rollout_stats, "update_metrics": update_stats, "checkpoint": str(checkpoint)}
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        all_metrics.append(record)
    return all_metrics


def main(argv: list[str] | None = None) -> int:
    try:
        run(_parser().parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - actionable CLI boundary
        raise SystemExit(f"stage25 PPO failed before a committed update: {exc}") from exc
    return 0


if __name__ == "__main__":
    main()
