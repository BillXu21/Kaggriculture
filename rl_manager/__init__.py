"""rl_manager: narrow Stage-A rollout/self-play/trajectory infrastructure (issue #9).

This package owns ONLY the RL harness seams around the frozen components:

- framework-neutral batched plan-policy protocol (`BatchedPlanPolicy`) with
  PPO-ready logprob/value slots;
- `JaxEPlanPolicy`: one batched `bc_manager_jax.forward(..., model_variant="E")`
  call per policy group/day (never one call per environment) with exact
  own-only E observation contract and deterministic issue-#8 decoding;
- lockstep N-env self-play runner over `oracle.backend` engines with the
  committed `standard_mixed` d0-d3 opening and the exact d4h0 manager handoff;
- exact `bc_manager.live.encode_live_inputs` / `EconomicHistory` semantics via
  runner-owned daily-start `(day, cash)` state and `economic_prev_start`;
- compact preallocated trajectory buffer with strict-schema NPZ serialization
  plus an optional JSON sidecar for rich diagnostics;
- explicit seed streams, episode ownership, and full provenance
  (policy/opponent/executor-factory/opening/backend/engine);
- official-vs-fast comparison seam with first-divergence reports.

Stage B (PPO V0 core) lives in `rl_manager.ppo_policy` / `rl_manager.gae` /
`rl_manager.ppo` / `rl_manager.ppo_checkpoint`: a PPO policy over the
mutable E trunk + small value head with an immutable frozen-E snapshot,
frozen sell quantities, GAE, and a strict RL-native checkpoint format. The
executor, opening book, oracle, fast env, bc_manager, and bc_manager_jax
are consumed through their public interfaces only — never modified.
"""

from rl_manager.decode import (
    ACTION_TENSOR_SHAPES,
    LOGPROB_GROUPS,
    decode_outputs_to_action_tensors,
    decode_outputs_to_plans,
    plans_from_action_tensors,
)
from rl_manager.executor_factory import (
    EXECUTOR_FACTORY_VERSION,
    make_default_executor_factory,
)
from rl_manager.gae import advantage_stats, compute_gae
from rl_manager.ppo import (
    PPOBatch,
    PPOTrainState,
    build_ppo_batch,
    init_train_state,
    ppo_update,
)
from rl_manager.ppo_checkpoint import (
    RL_PPO_CHECKPOINT_FORMAT,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from rl_manager.ppo_policy import PPOConfig, PPOPolicy
from rl_manager.policy import JaxEPlanPolicy, params_fingerprint
from rl_manager.provider import QueuedPlanProvider
from rl_manager.runner import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    GAME_TURNS,
    MANAGER_START_DAY,
    TOTAL_MANAGER_DAYS,
    EpisodeResult,
    EpisodeSpec,
    RunnerConfig,
    SelfPlayRunner,
    build_artifact_metadata,
    build_episode_spec,
)
from rl_manager.seeds import SeedStream
from rl_manager.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryBuffer,
    load_trajectory,
)
from rl_manager.types import (
    CANDIDATE_VS_FROZEN,
    E_VS_E,
    FROZEN_VS_CANDIDATE,
    BatchedPlanPolicy,
    PolicyIdentity,
    PolicyOutputs,
    seat_policies,
)

__all__ = [
    "ACTION_TENSOR_SHAPES",
    "ARTIFACT_METADATA_SCHEMA_VERSION",
    "CANDIDATE_VS_FROZEN",
    "E_VS_E",
    "EXECUTOR_FACTORY_VERSION",
    "FROZEN_VS_CANDIDATE",
    "GAME_TURNS",
    "LOGPROB_GROUPS",
    "MANAGER_START_DAY",
    "RL_PPO_CHECKPOINT_FORMAT",
    "TOTAL_MANAGER_DAYS",
    "TRAJECTORY_SCHEMA_VERSION",
    "BatchedPlanPolicy",
    "EpisodeResult",
    "EpisodeSpec",
    "JaxEPlanPolicy",
    "PPOBatch",
    "PPOConfig",
    "PPOPolicy",
    "PPOTrainState",
    "PolicyIdentity",
    "PolicyOutputs",
    "QueuedPlanProvider",
    "RunnerConfig",
    "SeedStream",
    "SelfPlayRunner",
    "TrajectoryBuffer",
    "advantage_stats",
    "build_artifact_metadata",
    "build_episode_spec",
    "build_ppo_batch",
    "compute_gae",
    "decode_outputs_to_action_tensors",
    "decode_outputs_to_plans",
    "init_train_state",
    "load_ppo_checkpoint",
    "load_trajectory",
    "make_default_executor_factory",
    "params_fingerprint",
    "plans_from_action_tensors",
    "ppo_update",
    "save_ppo_checkpoint",
    "seat_policies",
]
