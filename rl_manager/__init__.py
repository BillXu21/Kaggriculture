"""RL manager public API with accelerator-safe lazy imports.

CPU rollout workers import framework-neutral ``rl_manager`` submodules.  The
package initializer must therefore never import JAX-owning policy or PPO code.
Public symbols remain available through PEP 562 lazy attribute loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "ACTION_TENSOR_SHAPES": ("rl_manager.decode", "ACTION_TENSOR_SHAPES"),
    "ARTIFACT_METADATA_SCHEMA_VERSION": (
        "rl_manager.runner", "ARTIFACT_METADATA_SCHEMA_VERSION"),
    "BatchedPlanPolicy": ("rl_manager.types", "BatchedPlanPolicy"),
    "CANDIDATE_VS_FROZEN": ("rl_manager.types", "CANDIDATE_VS_FROZEN"),
    "DEBUG_TRACE_SCHEMA_VERSION": (
        "rl_manager.debug_trace", "DEBUG_TRACE_SCHEMA_VERSION"),
    "DIAGNOSTICS_SCHEMA_VERSION": (
        "rl_manager.diagnostics", "DIAGNOSTICS_SCHEMA_VERSION"),
    "aggregate_economic_diagnostics": (
        "rl_manager.diagnostics", "aggregate_economic_diagnostics"),
    "build_economic_diagnostics": (
        "rl_manager.diagnostics", "build_economic_diagnostics"),
    "CURRENT_VS_CURRENT_ECONOMIC": (
        "rl_manager.types", "CURRENT_VS_CURRENT_ECONOMIC"),
    "DebugTraceError": ("rl_manager.debug_trace", "DebugTraceError"),
    "E_VS_E": ("rl_manager.types", "E_VS_E"),
    "E_VS_PASS": ("rl_manager.types", "E_VS_PASS"),
    "EXECUTOR_FACTORY_VERSION": (
        "rl_manager.executor_factory", "EXECUTOR_FACTORY_VERSION"),
    "EpisodeResult": ("rl_manager.runner", "EpisodeResult"),
    "EpisodeSpec": ("rl_manager.runner", "EpisodeSpec"),
    "PromotionConfig": ("rl_manager.evaluation", "PromotionConfig"),
    "PromotionDecision": ("rl_manager.evaluation", "PromotionDecision"),
    "FROZEN_VS_CANDIDATE": (
        "rl_manager.types", "FROZEN_VS_CANDIDATE"),
    "farm_utilization_snapshot": (
        "rl_manager.land", "farm_utilization_snapshot"),
    "observed_land_purchase_events": (
        "rl_manager.land", "observed_land_purchase_events"),
    "GAME_TURNS": ("rl_manager.runner", "GAME_TURNS"),
    "JaxEPlanPolicy": ("rl_manager.policy", "JaxEPlanPolicy"),
    "LOGPROB_GROUPS": ("rl_manager.decode", "LOGPROB_GROUPS"),
    "MANAGER_START_DAY": ("rl_manager.runner", "MANAGER_START_DAY"),
    "PPOBatch": ("rl_manager.ppo", "PPOBatch"),
    "PPOBatchedPolicy": ("rl_manager.ppo_adapter", "PPOBatchedPolicy"),
    "PPOConfig": ("rl_manager.ppo_policy", "PPOConfig"),
    "PPOPolicy": ("rl_manager.ppo_policy", "PPOPolicy"),
    "PPOTrainState": ("rl_manager.ppo", "PPOTrainState"),
    "PPO_SNAPSHOT_FORMAT": (
        "rl_manager.ppo_checkpoint", "PPO_SNAPSHOT_FORMAT"),
    "PromotionRatchet": ("rl_manager.ratchet", "PromotionRatchet"),
    "BestCheckpointRetention": (
        "rl_manager.ppo_retention", "BestCheckpointRetention"),
    "PassPlanPolicy": ("rl_manager.policy", "PassPlanPolicy"),
    "ParallelRolloutError": (
        "rl_manager.parallel", "ParallelRolloutError"),
    "ParallelSelfPlayRunner": (
        "rl_manager.parallel", "ParallelSelfPlayRunner"),
    "PolicyIdentity": ("rl_manager.types", "PolicyIdentity"),
    "PolicyOutputs": ("rl_manager.types", "PolicyOutputs"),
    "QueuedPlanProvider": ("rl_manager.provider", "QueuedPlanProvider"),
    "RL_PPO_CHECKPOINT_FORMAT": (
        "rl_manager.ppo_checkpoint", "RL_PPO_CHECKPOINT_FORMAT"),
    "RunnerConfig": ("rl_manager.runner", "RunnerConfig"),
    "RewardConfig": ("rl_manager.reward", "RewardConfig"),
    "REWARD_MODES": ("rl_manager.reward", "REWARD_MODES"),
    "TERMINAL_OWN_BANK": ("rl_manager.reward", "TERMINAL_OWN_BANK"),
    "TERMINAL_WLT": ("rl_manager.reward", "TERMINAL_WLT"),
    "SeedStream": ("rl_manager.seeds", "SeedStream"),
    "SelfPlayRunner": ("rl_manager.runner", "SelfPlayRunner"),
    "TOTAL_MANAGER_DAYS": ("rl_manager.runner", "TOTAL_MANAGER_DAYS"),
    "TRAJECTORY_SCHEMA_VERSION": (
        "rl_manager.trajectory", "TRAJECTORY_SCHEMA_VERSION"),
    "TraceRecorder": ("rl_manager.debug_trace", "TraceRecorder"),
    "TrajectoryBuffer": ("rl_manager.trajectory", "TrajectoryBuffer"),
    "advantage_stats": ("rl_manager.gae", "advantage_stats"),
    "build_artifact_metadata": (
        "rl_manager.runner", "build_artifact_metadata"),
    "build_episode_spec": ("rl_manager.runner", "build_episode_spec"),
    "build_integration_diagnostics": (
        "rl_manager.diagnostics", "build_integration_diagnostics"),
    "build_ppo_batch": ("rl_manager.ppo", "build_ppo_batch"),
    "build_trace": ("rl_manager.debug_trace", "build_trace"),
    "canonical_json_bytes": (
        "rl_manager.debug_trace", "canonical_json_bytes"),
    "compute_gae": ("rl_manager.gae", "compute_gae"),
    "decode_outputs_to_action_tensors": (
        "rl_manager.decode", "decode_outputs_to_action_tensors"),
    "decode_outputs_to_plans": (
        "rl_manager.decode", "decode_outputs_to_plans"),
    "evaluate_promotion": (
        "rl_manager.evaluation", "evaluate_promotion"),
    "init_train_state": ("rl_manager.ppo", "init_train_state"),
    "load_ppo_checkpoint": (
        "rl_manager.ppo_checkpoint", "load_ppo_checkpoint"),
    "load_trace": ("rl_manager.debug_trace", "load_trace"),
    "load_trajectory": ("rl_manager.trajectory", "load_trajectory"),
    "make_default_executor_factory": (
        "rl_manager.executor_factory", "make_default_executor_factory"),
    "params_fingerprint": ("rl_manager.policy", "params_fingerprint"),
    "plans_from_action_tensors": (
        "rl_manager.decode", "plans_from_action_tensors"),
    "ppo_batched_policy_from_state": (
        "rl_manager.ppo_adapter", "ppo_batched_policy_from_state"),
    "ppo_snapshot_from_state": (
        "rl_manager.ppo_adapter", "ppo_snapshot_from_state"),
    "ppo_update": ("rl_manager.ppo", "ppo_update"),
    "prng_key_from_id": ("rl_manager.ppo_adapter", "prng_key_from_id"),
    "save_ppo_checkpoint": (
        "rl_manager.ppo_checkpoint", "save_ppo_checkpoint"),
    "save_ppo_snapshot": (
        "rl_manager.ppo_checkpoint", "save_ppo_snapshot"),
    "load_ppo_snapshot": (
        "rl_manager.ppo_checkpoint", "load_ppo_snapshot"),
    "save_trace": ("rl_manager.debug_trace", "save_trace"),
    "seat_policies": ("rl_manager.types", "seat_policies"),
    "select_ppo_subset": ("rl_manager.ppo_adapter", "select_ppo_subset"),
    "validate_trace": ("rl_manager.debug_trace", "validate_trace"),
    "summarize_evaluation": (
        "rl_manager.evaluation", "summarize_evaluation"),
    "write_diagnostics": ("rl_manager.diagnostics", "write_diagnostics"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") \
            from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
