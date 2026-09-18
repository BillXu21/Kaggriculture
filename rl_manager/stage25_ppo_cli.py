"""Small real-rollout native Stage 2.5 PPO runner.

The CLI intentionally runs complete Packet 5A rollouts between updates.  A
checkpoint is therefore resumable only at the completed rollout/update
boundary recorded by the native checkpoint format. Update 1 is normally a
warmup/compilation update; use update 2 and later for steady-state throughput.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Mapping

from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.reward import (
    TERMINAL_OWN_BANK,
    TERMINAL_WLT,
    RewardConfig,
)
from rl_manager.runner import RunnerConfig, build_episode_spec
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer
from rl_manager.types import CANDIDATE_VS_FROZEN, CURRENT_VS_CURRENT_ECONOMIC

_STAGE25_PHASE_METRICS = (
    "input_validation_seconds", "host_input_prepare_seconds",
    "context_validation_seconds", "support_validation_seconds",
    "row_rng_prepare_seconds", "policy_call_seconds",
    "output_conversion_seconds", "adapter_total_seconds",
)

if TYPE_CHECKING:
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_policy import Stage25ModelConfig
    from rl_manager.stage25_ppo import (
        Stage25PPOBatch,
        Stage25PPOConfig,
        Stage25PPOTrainState,
    )
    from rl_manager.stage25_types import Stage25BehaviorIdentity


def _percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bank_statistics(final_banks: list[float]) -> dict[str, float | int]:
    banks = [float(bank) for bank in final_banks]
    if not banks:
        return {
            "count": 0, "mean": 0.0, "median": 0.0, "min": 0.0,
            "bottom_decile_mean": 0.0, "p10": 0.0, "p25": 0.0,
            "p75": 0.0, "p90": 0.0, "max": 0.0,
            "zero_bank_fraction": 0.0,
        }
    ordered = sorted(banks)
    bottom_count = math.ceil(len(ordered) * 0.10)
    summary = {
        "count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.5),
        "min": ordered[0],
        "bottom_decile_mean": math.fsum(ordered[:bottom_count]) / bottom_count,
        "p10": _percentile(ordered, 0.10),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
        "zero_bank_fraction": sum(bank == 0.0 for bank in ordered) / len(ordered),
    }
    return summary


def _inference_summary(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    real_batches = [float(value) for value in metrics.get("real_batch_sizes", ())]
    physical_rows = int(metrics.get("physical_rows", 0))
    padding_rows = int(metrics.get("padding_rows", 0))
    logical_requests = int(metrics.get(
        "logical_requests", metrics.get("real_requests", metrics.get("requests", 0))))
    summary = {
        "physical_calls": int(metrics.get(
            "physical_inference_calls", metrics.get("batches", 0))),
        "real_requests": int(metrics.get("real_requests", 0)),
        "logical_requests": logical_requests,
        "physical_rows": physical_rows,
        "padding_rows": padding_rows,
        "padding_fraction": padding_rows / physical_rows if physical_rows else 0.0,
        "occupancy": float(metrics.get(
            "occupancy", logical_requests / physical_rows if physical_rows else 0.0)),
        "mean_real_batch_size": (
            math.fsum(real_batches) / len(real_batches) if real_batches else 0.0),
        "min_real_batch_size": min(real_batches, default=0.0),
        "median_real_batch_size": _percentile(real_batches, 0.5),
        "p10_real_batch_size": _percentile(real_batches, 0.10),
        "p90_real_batch_size": _percentile(real_batches, 0.90),
        "max_real_batch_size": max(real_batches, default=0.0),
        "aggregate_inference_seconds": float(metrics.get("inference_seconds", 0.0)),
        "aggregate_queue_wait_seconds": float(metrics.get("queue_wait_seconds", 0.0)),
    }
    summary.update({name: float(metrics.get(name, 0.0))
                    for name in _STAGE25_PHASE_METRICS})
    return summary


def _rate(numerator: float, seconds: float) -> float:
    return float(numerator) / seconds if seconds > 0.0 else 0.0


def _configure_jax_compilation_cache(path: Path | None) -> None:
    if path is None:
        return
    try:
        import jax
        from jax.experimental.compilation_cache import compilation_cache
        set_cache_dir = getattr(compilation_cache, "set_cache_dir")
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        path.mkdir(parents=True, exist_ok=True)
        set_cache_dir(str(path))
    except (AttributeError, ImportError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "--jax-compilation-cache-dir is unsupported by the installed "
            f"JAX persistent compilation-cache API: {exc}") from exc


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()


def _format_report(record: Mapping[str, Any]) -> str:
    timing = record["timing"]
    throughput = record["throughput"]
    inference = record["inference_summary"]
    bank = record["bank_summary"]
    ppo = record["update_metrics"]

    def value(key: str, digits: int = 2) -> str:
        return f"{float(timing[key]):.{digits}f} s"

    def rate_value(key: str, digits: int = 2) -> str:
        return f"{float(throughput[key]):.{digits}f}"

    def line(label: str, text: Any) -> str:
        return f"{label:<26}{text}"

    audit = ppo.get("unchanged_weight_audit", {})
    audit_text = "PASS" if audit.get("ok", True) else "FAIL"
    p10 = f"{inference['p10_real_batch_size']:.2f}"
    p90 = f"{inference['p90_real_batch_size']:.2f}"
    lines = [
        "=" * 64,
        f"STAGE 2.5 PPO | UPDATE {record['update']}",
        "=" * 64,
        "",
        "WORK",
        line("games", record["games_in_update"]),
        line("learner rows", record.get("learner_rows", 0)),
        line("terminal rows", record.get("terminal_rows", 0)),
        line("truncated rows", record.get("truncated_rows", 0)),
        "",
        "STARTUP",
        line("state initialization", f"{record['startup']['state_initialization_seconds']:.4f} s"),
        "",
        "TIMING",
        line("episode/spec", value("episode_spec_construction_seconds")),
        line("runner/rollout", value("runner_rollout_seconds")),
        line("batch construction", value("batch_construction_seconds")),
        line("collection total", value("collection_seconds")),
        line("PPO update", value("ppo_update_seconds")),
        line("checkpoint", value("checkpoint_seconds")),
        line("other/overhead", value("overhead_seconds")),
        line("update total", value("update_seconds")),
        "",
        "THROUGHPUT",
        line("rollout games/s", rate_value("rollout_games_per_second")),
        line("collection games/s", rate_value("collection_games_per_second")),
        line("update games/s", rate_value("update_games_per_second")),
        line("update games/hour", f"{throughput['update_games_per_hour']:.2f}"),
        line("updates/hour", f"{throughput['updates_per_hour']:.2f}"),
        line("learner rows/s", f"{throughput['update_learner_rows_per_second']:.2f}"),
        "",
        "INFERENCE (aggregate timings)",
        line("physical calls", inference["physical_calls"]),
        line("real/logical requests", f"{inference['real_requests']} / {inference['logical_requests']}"),
        line("physical rows", inference["physical_rows"]),
        line("padding rows", f"{inference['padding_rows']} ({inference['padding_fraction'] * 100.0:.2f}%)"),
        line("occupancy", f"{inference['occupancy'] * 100.0:.2f}%"),
        line("mean real batch", f"{inference['mean_real_batch_size']:.2f}"),
        line("min / median batch", f"{inference['min_real_batch_size']:.2f} / {inference['median_real_batch_size']:.2f}"),
        line("p10 / p90 batch", f"{p10} / {p90}"),
        line("max real batch", f"{inference['max_real_batch_size']:.2f}"),
        line("aggregate inference", f"{inference['aggregate_inference_seconds']:.4f} s"),
        line("aggregate queue wait", f"{inference['aggregate_queue_wait_seconds']:.4f} s"),
        "",
        "INFERENCE PHASES",
        line("input validation", f"{inference.get('input_validation_seconds', 0.0):.4f} s"),
        line("host input prepare", f"{inference.get('host_input_prepare_seconds', 0.0):.4f} s"),
        line("context validation", f"{inference.get('context_validation_seconds', 0.0):.4f} s"),
        line("support validation", f"{inference.get('support_validation_seconds', 0.0):.4f} s"),
        line("row/RNG prepare", f"{inference.get('row_rng_prepare_seconds', 0.0):.4f} s"),
        line("policy/device", f"{inference.get('policy_call_seconds', 0.0):.4f} s"),
        line("output conversion", f"{inference.get('output_conversion_seconds', 0.0):.4f} s"),
        line("adapter total", f"{inference.get('adapter_total_seconds', 0.0):.4f} s"),
        "",
        "BANK",
        line("count", bank["count"]),
        line("mean", f"{bank['mean']:.2f}"),
        line("median", f"{bank['median']:.2f}"),
        line("min", f"{bank['min']:.2f}"),
        line("bottom-10% mean", f"{bank['bottom_decile_mean']:.2f}"),
        line("p10 / p25", f"{bank['p10']:.2f} / {bank['p25']:.2f}"),
        line("p75 / p90", f"{bank['p75']:.2f} / {bank['p90']:.2f}"),
        line("max", f"{bank['max']:.2f}"),
        line("zero-bank", f"{bank['zero_bank_fraction'] * 100.0:.2f}%"),
        "",
        "PPO",
        line("loss", f"{ppo.get('loss', 0.0):.6g}"),
        line("policy loss", f"{ppo.get('policy_loss', 0.0):.6g}"),
        line("value loss", f"{ppo.get('value_loss', 0.0):.6g}"),
        line("entropy", f"{ppo.get('entropy_surrogate', 0.0):.6g}"),
        line("KL", f"{ppo.get('kl', 0.0):.6g}"),
        line("clip fraction", f"{ppo.get('clip_fraction', 0.0):.6g}"),
        line("gradient norm", f"{ppo.get('gradient_norm', 0.0):.6g}"),
        line("epochs", ppo.get("epochs", 0)),
        line("weight audit", audit_text),
        "",
        f"checkpoint: {record['checkpoint']}",
        "=" * 64,
    ]
    return "\n".join(lines)


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
    parser.add_argument(
        "--opening", default="standard_mixed",
        help="built-in opening identity for the Stage 2.5 runner")
    parser.add_argument(
        "--scratch-hold-prior-tau", type=float, default=None,
        help="scratch-only output-bias hold prior temperature")
    parser.add_argument("--model-size", choices=("tiny", "small", "large"), default="tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=("fast", "official"), default="fast")
    parser.add_argument(
        "--training-composition",
        choices=(CANDIDATE_VS_FROZEN, CURRENT_VS_CURRENT_ECONOMIC),
        default=CANDIDATE_VS_FROZEN,
    )
    parser.add_argument(
        "--reward-mode", choices=(TERMINAL_WLT, TERMINAL_OWN_BANK),
        default=TERMINAL_WLT,
    )
    parser.add_argument("--bank-reward-baseline", type=float, default=3000.0)
    parser.add_argument("--bank-reward-scale", type=float, default=50000.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--batch-backend", action="store_true")
    parser.add_argument("--inference-batch-wait-ms", type=float, default=20.0)
    parser.add_argument("--rollout-size", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=144)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument(
        "--stage25-inference-validation", choices=("strict", "fast", "none"),
        default="strict",
        help="parent Stage 2.5 diagnostic validation mode (default: strict)")
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
    parser.add_argument(
        "--json-stdout", action="store_true",
        help="also emit each complete machine-readable update record")
    parser.add_argument(
        "--jax-compilation-cache-dir", type=Path,
        help="optional JAX persistent compilation-cache directory")
    return parser


def _reward_config(args: argparse.Namespace) -> RewardConfig:
    config = RewardConfig(
        mode=args.reward_mode,
        bank_baseline=args.bank_reward_baseline,
        bank_scale=args.bank_reward_scale,
    )
    if (args.training_composition == CURRENT_VS_CURRENT_ECONOMIC
            and config.mode != TERMINAL_OWN_BANK):
        raise ValueError(
            "current_vs_current_economic requires --reward-mode "
            "terminal_own_bank")
    return config


def _training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_composition": args.training_composition,
        "reward": _reward_config(args).to_json_dict(),
    }


def _validate_rollout_controls(args: argparse.Namespace) -> None:
    if args.envs_per_worker < 1:
        raise ValueError("envs-per-worker must be positive")
    if args.batch_backend and args.engine != "fast":
        raise ValueError("--batch-backend requires --engine fast")
    if (not math.isfinite(args.inference_batch_wait_ms)
            or args.inference_batch_wait_ms < 0):
        raise ValueError(
            "inference-batch-wait-ms must be finite and >= 0")
    if (args.scratch_hold_prior_tau is not None
            and not args.scratch):
        raise ValueError("--scratch-hold-prior-tau requires --scratch")
    if (args.scratch_hold_prior_tau is not None
            and (not math.isfinite(args.scratch_hold_prior_tau)
                 or args.scratch_hold_prior_tau <= 0.0)):
        raise ValueError("scratch-hold-prior-tau must be finite and positive")


def _runner_config(
        args: argparse.Namespace, *, seed: int,
        reward_config: RewardConfig) -> RunnerConfig:
    _validate_rollout_controls(args)
    return RunnerConfig(
        backend_name=args.engine,
        backend_configuration={"seed": seed, "numThreads": 1},
        num_envs=args.envs_per_worker,
        batch_backend=args.batch_backend,
        inference_batch_wait_seconds=args.inference_batch_wait_ms / 1000.0,
        max_turns=args.max_turns, low_telemetry=True, stage25_enabled=True,
        stage25_mode="stochastic",
        stage25_fixed_inference_batch_size=args.physical_batch_size,
        reward_config=reward_config, opening=args.opening)


def _checkpoint_training_contract(meta: Mapping[str, Any]) -> dict[str, Any]:
    stored = meta.get("training_contract")
    if isinstance(stored, dict):
        return dict(stored)
    # Checkpoints written before this CLI exposed composition/reward options
    # were always candidate-v-frozen with the RunnerConfig W/L default.
    cli_args = meta.get("cli_args")
    if isinstance(cli_args, dict):
        return {
            "training_composition": cli_args.get(
                "training_composition", CANDIDATE_VS_FROZEN),
            "reward": {
                "mode": cli_args.get("reward_mode", TERMINAL_WLT),
                "bank_baseline": float(
                    cli_args.get("bank_reward_baseline", 3000.0)),
                "bank_scale": float(
                    cli_args.get("bank_reward_scale", 50000.0)),
            },
        }
    return {}


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
        from rl_manager.stage25_policy import init_stage25_params
        params = init_stage25_params(
            config.model, seed=args.seed,
            scratch_hold_prior_tau=args.scratch_hold_prior_tau)
        return init_stage25_ppo_state(
            config, seed=args.seed, params=params), {}
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
    expected_contract = _training_contract(args)
    stored_contract = _checkpoint_training_contract(meta)
    if stored_contract != expected_contract:
        raise ValueError(
            "PPO checkpoint training/reward contract does not match the "
            f"requested contract: {stored_contract!r} != {expected_contract!r}")
    if meta.get("executor") != runtime_executor:
        raise ValueError(
            "PPO checkpoint executor provenance does not match the configured "
            f"factory: {meta.get('executor')!r} != {runtime_executor!r}")
    learner_identity = _identity_from_meta(meta, "behavior_identity")
    learner_adapter = Stage25InferenceAdapter(
        params=params, config=config.model, name=learner_identity.name,
        version=learner_identity.version, seed=args.seed, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
    if learner_adapter.identity != learner_identity:
        raise ValueError(
            "PPO checkpoint behavior identity does not match loaded parameters "
            "and curriculum")
    opponent_identity = _identity_from_meta(meta, "opponent_identity")
    opponent_adapter = Stage25InferenceAdapter(
        params=opponent_params, config=config.model, name=opponent_identity.name,
        version=opponent_identity.version, seed=args.seed, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
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
) -> tuple[
    Stage25TrajectoryBuffer,
    Stage25InferenceAdapter,
    Stage25PPOBatch,
    dict[str, Any],
]:
    from rl_manager.executor_factory import make_stage25_executor_factory
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_ppo import build_stage25_ppo_batch
    collection_started = time.perf_counter()
    reward_config = _reward_config(args)
    episode_spec_started = time.perf_counter()
    learner = Stage25InferenceAdapter(
        params=state.params, config=config.model, name="stage25_learner",
        version="ppo-native-v1", seed=seed, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
    opponent = Stage25InferenceAdapter(
        params=(state.params if state.opponent_params is None else state.opponent_params),
        config=config.model,
        name=("stage25_opponent" if state.opponent_identity is None
              else state.opponent_identity.name),
        version=("frozen-v1" if state.opponent_identity is None
                 else state.opponent_identity.version),
        seed=seed + 1, mode="stochastic",
        validation_mode=args.stage25_inference_validation)
    if state.opponent_identity is not None and opponent.identity != state.opponent_identity:
        raise ValueError("restored opponent identity does not match its parameters")
    capacity = max(1, args.rollout_size * 2 * 26)
    trajectory = Stage25TrajectoryBuffer(capacity)
    runner_config = _runner_config(
        args, seed=seed, reward_config=reward_config)
    runner = ParallelSelfPlayRunner(
        runner_config, num_workers=args.workers, master_seed=seed,
        executor_factory=make_stage25_executor_factory(),
        stage25_trajectory_buffer=trajectory)
    specs = tuple(build_episode_spec(
        index, seed + index, args.training_composition, learner, opponent)
                  for index in range(args.rollout_size))
    episode_spec_seconds = time.perf_counter() - episode_spec_started
    runner_started = time.perf_counter()
    results = runner.run(specs)
    runner_rollout_seconds = time.perf_counter() - runner_started
    batch_started = time.perf_counter()
    batch = build_stage25_ppo_batch(
        trajectory, learner_identity=learner.identity, gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        normalize_advantages=config.normalize_advantages)
    batch_construction_seconds = time.perf_counter() - batch_started
    collection_seconds = time.perf_counter() - collection_started
    trajectory_summary = trajectory.diagnostic_summary()
    stats = {
        "rollout_rows": len(trajectory), "learner_rows": len(batch.classes),
        **trajectory_summary,
        "final_banks": [float(bank) for result in results for bank in result.final_banks],
        "inference_metrics": runner.inference_metrics,
        "behavior_identity": learner.identity.to_json_dict(),
        "opponent_identity": opponent.identity.to_json_dict(),
        "opening_provenance": runner.provenance.get("opening"),
        "training_composition": args.training_composition,
        "reward": reward_config.to_json_dict(),
        "executor_provenance": _executor_factory_provenance(
            runner.provenance["executor_factory"]),
        "timing": {
            "episode_spec_construction_seconds": episode_spec_seconds,
            "runner_rollout_seconds": runner_rollout_seconds,
            "batch_construction_seconds": batch_construction_seconds,
            "collection_seconds": collection_seconds,
        },
    }
    trajectory.validate_executor_provenance(stats["executor_provenance"])
    return trajectory, learner, batch, stats


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from rl_manager.stage25_checkpoint import save_stage25_ppo_checkpoint
    from rl_manager.executor_factory import make_stage25_executor_factory
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_ppo import ppo_update
    startup_started = time.perf_counter()
    _configure_jax_compilation_cache(getattr(
        args, "jax_compilation_cache_dir", None))
    _validate_rollout_controls(args)
    if args.updates < 1 or args.rollout_size < 1 or args.workers < 1:
        raise ValueError("updates, rollout-size, and workers must be positive")
    config = _config(args)
    if args.workers == 1 and config.physical_batch_size != 1:
        raise ValueError(
            "Packet 5A single-process collection cannot honor a physical "
            "batch larger than one; use --workers 2+ or set "
            "--physical-batch-size 1")
    _reward_config(args)
    state, source_meta = _new_state(args, config)
    startup_seconds = time.perf_counter() - startup_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    all_metrics = []
    for _ in range(args.updates):
        update_started = time.perf_counter()
        _, learner, ppo_batch, rollout_stats = _collection(
            state, config, seed=state.rollout_seed, args=args,
            previous_params=None)
        state = replace(state, behavior_identity=learner.identity)
        if state.opponent_params is None:
            from rl_manager.stage25_inference import Stage25InferenceAdapter
            initial_opponent = Stage25InferenceAdapter(
                params=state.params, config=config.model,
                name="stage25_opponent", version="frozen-v1",
                seed=state.rollout_seed + 1, mode="stochastic",
                validation_mode=args.stage25_inference_validation)
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
        ppo_started = time.perf_counter()
        state, update_stats = ppo_update(state, ppo_batch, config)
        ppo_update_seconds = time.perf_counter() - ppo_started
        from rl_manager.stage25_inference import Stage25InferenceAdapter
        next_opponent = Stage25InferenceAdapter(
            params=previous_learner_params, config=config.model,
            name="stage25_opponent", version="frozen-v1",
            seed=state.rollout_seed + 1, mode="stochastic",
            validation_mode=args.stage25_inference_validation)
        state = replace(state, opponent_params=previous_learner_params,
                        opponent_identity=next_opponent.identity)
        state = replace(state, rollout_seed=state.rollout_seed + args.rollout_size)
        metadata = {
            "run": {
                "cli": "rl_manager.stage25_ppo_cli",
                "source": str(args.init) if args.init else None,
                "opening": rollout_stats.get("opening_provenance"),
            },
            "training_contract": _training_contract(args),
            "resume_from": (
                None if not args.resume else {
                    "path": str(args.resume),
                    "payload_kind": source_meta.get("payload_kind"),
                    "update_counter": source_meta.get("update_counter"),
                }),
        }
        checkpoint = args.output_dir / "latest.npz"
        checkpoint_started = time.perf_counter()
        save_stage25_ppo_checkpoint(
            checkpoint, state.params, state.optimizer_state, state.rng,
            config.model, seed=args.seed, update_counter=state.update_counter,
            rollout_seed=state.rollout_seed, rollout_progression=state.rollout_progression,
            ppo_config=config.to_dict(), optimizer_config=config.to_dict(),
            curriculum=config.model.curriculum, behavior_identity=state.behavior_identity,
            provenance={"run": metadata},
            physical_contract=_physical_contract(config),
            executor=executor_provenance,
            metadata={
                "cli_args": vars(args),
                "training_contract": _training_contract(args),
            },
            opponent_params=state.opponent_params,
            opponent_identity=state.opponent_identity,
            source_identity=(source_meta.get("source_identity") or None),
            source_history_version=(
                (source_meta.get("source_e_identity") or {}).get("history_version")),
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        update_seconds = time.perf_counter() - update_started
        collection_timing = dict(rollout_stats.get("timing", {}))
        for timing_name in (
                "episode_spec_construction_seconds", "runner_rollout_seconds",
                "batch_construction_seconds", "collection_seconds"):
            collection_timing.setdefault(timing_name, 0.0)
        collection_seconds = float(collection_timing.get("collection_seconds", 0.0))
        runner_rollout_seconds = float(collection_timing.get(
            "runner_rollout_seconds", 0.0))
        timing = {
            **collection_timing,
            "ppo_batch_construction_seconds": 0.0,
            "ppo_update_seconds": ppo_update_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "update_seconds": update_seconds,
            "accounted_phase_seconds": (
                collection_seconds + ppo_update_seconds + checkpoint_seconds),
            "overhead_seconds": max(
                0.0, update_seconds - collection_seconds - ppo_update_seconds
                - checkpoint_seconds),
        }
        update_stats = dict(update_stats)
        ppo_timing = update_stats.get("timing")
        if not isinstance(ppo_timing, dict):
            ppo_timing = {}
        update_stats["timing"] = {**ppo_timing, "wall_seconds": ppo_update_seconds}
        inference_metrics = rollout_stats.get("inference_metrics", {})
        record = {
            "update": state.update_counter,
            **rollout_stats,
            "games_in_update": args.rollout_size,
            "update_metrics": update_stats,
            "checkpoint": str(checkpoint),
            "startup": {"state_initialization_seconds": startup_seconds},
            "timing": timing,
            "throughput": {
                "rollout_games_per_second": _rate(
                    args.rollout_size, runner_rollout_seconds),
                "rollout_games_per_hour": _rate(
                    args.rollout_size * 3600.0, runner_rollout_seconds),
                "rollout_learner_rows_per_second": _rate(
                    rollout_stats.get("learner_rows", 0), runner_rollout_seconds),
                "collection_games_per_second": _rate(
                    args.rollout_size, collection_seconds),
                "update_games_per_second": _rate(
                    args.rollout_size, update_seconds),
                "update_games_per_hour": _rate(
                    args.rollout_size * 3600.0, update_seconds),
                "update_learner_rows_per_second": _rate(
                    rollout_stats.get("learner_rows", 0), update_seconds),
                "updates_per_hour": _rate(3600.0, update_seconds),
            },
            "inference_summary": _inference_summary(inference_metrics),
            "bank_summary": _bank_statistics(rollout_stats.get("final_banks", [])),
        }
        _append_jsonl(metrics_path, record)
        if getattr(args, "json_stdout", False):
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        else:
            print(_format_report(record), flush=True)
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
