"""Stage-A `BatchedPlanPolicy` adapter over the B1 PPO policy (issue #9 B2).

This module is pure integration glue: it adapts `rl_manager.ppo_policy.
PPOPolicy` to the framework-neutral `rl_manager.types.BatchedPlanPolicy`
protocol consumed by `SelfPlayRunner`, plus the small helpers the tiny
end-to-end smoke needs (state -> eval-policy reconstruction and a
deterministic subset selection over an already-GAE-normalized batch).

Contract implemented here:

- consumes contiguous own-only E input arrays plus an explicit string
  `prng_id`; row-aware calls hash the immutable policy identity into one
  root JAX key and fold in each stable row id (sampling attaches to row
  identity, never batch composition or position);
- stochastic mode samples through the B1 vmap; deterministic mode is the
  exact argmax/logit>0 decode and reproduces the frozen JAX-E decode before
  any policy drift;
- returns exact action tensors, the six logprob groups + raw-summed total,
  the scalar value, and the immutable identity (name/version/fingerprint);
- sell quantities ALWAYS come from the immutable frozen-E snapshot inside
  `PPOPolicy.act`; this adapter adds no second forward path;
- one batched call per request batch (the runner groups by identity/day),
  never one call per environment.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    normalize_e_history_version,
)
from rl_manager.ppo import PPOBatch, PPOTrainState
from rl_manager.ppo_policy import (
    CurriculumMaskConfig,
    PPOConfig,
    PPOPolicy,
    TargetedExplorationConfig,
    curriculum_behavior_fingerprint,
)
from rl_manager.policy import params_fingerprint
from rl_manager.types import PolicyIdentity, PolicyOutputs


def prng_key_from_id(prng_id: str) -> jax.Array:
    """Deterministic root PRNG key from an explicit string identifier.

    sha256-based so the key is stable across processes/platforms for the
    same `prng_id` (the runner encodes episode/day/policy identity there).
    """
    if not isinstance(prng_id, str) or not prng_id:
        raise ValueError(
            f"prng_id must be a non-empty string identifier, got {prng_id!r}")
    digest = hashlib.sha256(prng_id.encode("utf-8")).digest()
    # PRNGKey seeds are uint32; take the leading 32 bits of the digest.
    return jax.random.PRNGKey(int.from_bytes(digest[:4], "big"))


class PPOBatchedPolicy:
    """`BatchedPlanPolicy` view of one `PPOPolicy` (stochastic or greedy)."""

    def __init__(
        self,
        policy: PPOPolicy,
        *,
        name: str = "ppo_candidate",
        version: str = "ppo-v0",
        deterministic: bool = False,
        e_history_version: str = E_HISTORY_CORRECTED_V1,
        curriculum: CurriculumMaskConfig | None = None,
        exploration: TargetedExplorationConfig | None = None,
    ) -> None:
        self._policy = policy
        self._deterministic = bool(deterministic)
        self.curriculum = curriculum or policy.curriculum
        if self.curriculum != policy.curriculum:
            raise ValueError(
                "adapter curriculum must match the bound PPO policy")
        requested_exploration = exploration or policy.exploration
        if requested_exploration != policy.exploration:
            raise ValueError(
                "adapter exploration must match the bound PPO policy")
        self.exploration = (TargetedExplorationConfig()
                            if self._deterministic else requested_exploration)
        self.e_history_version = normalize_e_history_version(e_history_version)
        self.identity = PolicyIdentity(
            name=name,
            version=version,
            fingerprint=curriculum_behavior_fingerprint(
                policy.params, self.curriculum, self.exploration),
            e_history_version=self.e_history_version,
        )
        # Batching proof instrumentation (tests assert on these).
        self.call_count = 0
        self.batch_size_history: list[int] = []

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    def refresh_identity(self) -> None:
        """Re-derive the fingerprint after the underlying params changed."""
        self.identity = PolicyIdentity(
            name=self.identity.name,
            version=self.identity.version,
            fingerprint=curriculum_behavior_fingerprint(
                self._policy.params, self.curriculum, self.exploration),
            e_history_version=self.e_history_version,
        )

    def plan_batch(
        self,
        inputs: Mapping[str, np.ndarray],
        prng_id: str,
    ) -> PolicyOutputs:
        # The seed identifier is part of the protocol even in deterministic
        # mode; validating it keeps both modes on one explicit contract.
        rng = prng_key_from_id(prng_id)
        result = self._policy.act(
            inputs, deterministic=self._deterministic, rng=rng)
        self.call_count += 1
        self.batch_size_history.append(int(result["batch_size"]))
        return PolicyOutputs(
            action_tensors=result["action_tensors"],
            logprob_groups=result["logprob_groups"],
            logprob_total=result["logprob_total"],
            value=result["value"],
            batch_size=int(result["batch_size"]),
        )

    def plan_batch_with_row_ids(
        self,
        inputs: Mapping[str, np.ndarray],
        row_ids: Sequence[str],
        prng_id: str,
    ) -> PolicyOutputs:
        """Sample a central batch with stable seeds for each logical row."""
        if not isinstance(prng_id, str) or not prng_id:
            raise ValueError("prng_id must be a non-empty string identifier")
        batch_size = int(np.asarray(next(iter(inputs.values()))).shape[0])
        if len(row_ids) != batch_size or any(
                not isinstance(row_id, str) or not row_id for row_id in row_ids):
            raise ValueError("row_ids must contain one non-empty string per row")
        seeds = np.asarray([
            int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest()[:4],
                           "big")
            for row_id in row_ids], dtype=np.uint32)
        result = self._policy.act(
            inputs, deterministic=self._deterministic,
            # `prng_id` is still validated as part of the policy protocol, but
            # row-aware sampling must not depend on the scheduler's grouping.
            # Keep the historical unrestricted stream independent of the
            # provenance fingerprint; the support restriction itself still
            # changes the resulting distribution.
            rng=prng_key_from_id(
                f"policy={self.identity.name}@{self.identity.version}:"
                f"{params_fingerprint(self._policy.params)}"),
            decision_seeds=seeds)
        self.call_count += 1
        self.batch_size_history.append(int(result["batch_size"]))
        return PolicyOutputs(
            action_tensors=result["action_tensors"],
            logprob_groups=result["logprob_groups"],
            logprob_total=result["logprob_total"],
            value=result["value"],
            batch_size=int(result["batch_size"]),
        )


def ppo_batched_policy_from_state(
    state: PPOTrainState,
    config,
    *,
    ppo_config: PPOConfig | None = None,
    name: str = "ppo_candidate",
    version: str = "ppo-v0",
    deterministic: bool = False,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    curriculum: CurriculumMaskConfig | None = None,
    exploration: TargetedExplorationConfig | None = None,
) -> PPOBatchedPolicy:
    """Reconstruct an eval/train-capable adapter from a loaded train state.

    `PPOPolicy.__init__` derives params/value-head/rng from
    `(frozen_params, seed)`; a checkpoint resume must instead reuse the
    EXACT stored mutable params, frozen snapshot, and PRNG stream, so they
    are re-assigned onto a freshly constructed instance (no B1 edit).
    """
    policy = PPOPolicy(
        state.frozen_params, config, seed=0,
        ppo_config=ppo_config or PPOConfig(), curriculum=curriculum,
        exploration=exploration)
    policy.params = state.params
    policy.frozen_params = state.frozen_params
    policy.rng = state.rng
    return PPOBatchedPolicy(
        policy, name=name, version=version, deterministic=deterministic,
        e_history_version=e_history_version, curriculum=curriculum,
        exploration=exploration)


def ppo_snapshot_from_state(
    state: PPOTrainState,
    config,
    *,
    ppo_config: PPOConfig | None = None,
    name: str = "ppo_snapshot",
    version: str = "ratchet-v1",
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    curriculum: CurriculumMaskConfig | None = None,
) -> PPOBatchedPolicy:
    """Build a detached deterministic policy from one exact train state.

    The copied trees ensure later live-learner updates cannot change the
    opponent that was evaluated and promoted.
    """
    frozen_params = jax.tree_util.tree_map(
        lambda leaf: jnp.array(leaf), state.frozen_params)
    params = jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), state.params)
    policy = PPOPolicy(
        frozen_params, config, seed=0, ppo_config=ppo_config or PPOConfig(),
        curriculum=curriculum)
    policy.params = params
    policy.frozen_params = frozen_params
    policy.rng = jnp.array(state.rng)
    return PPOBatchedPolicy(
        policy, name=name, version=version, deterministic=True,
        e_history_version=e_history_version, curriculum=curriculum)


def select_ppo_subset(batch: PPOBatch, size: int) -> PPOBatch:
    """Deterministic evenly-spaced subset of an already-normalized batch.

    GAE/advantage normalization happened over the FULL selected rollout
    before this slice (`build_ppo_batch`); subsetting afterwards therefore
    keeps the local-smoke semantics required by issue #9 B2 (tiny 2-8 row
    minibatch while GAE covered the complete candidate trajectory).
    Indices are evenly spaced and strictly increasing; no resampling of
    actions or logprobs happens anywhere.
    """
    n = batch.size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"subset size must be a positive int, got {size!r}")
    if size > n:
        raise ValueError(
            f"subset size {size} exceeds batch size {n}")
    indices = np.unique(np.round(np.linspace(0, n - 1, size)).astype(np.int64))
    if indices.size != size:
        raise ValueError(
            f"evenly-spaced selection produced {indices.size} unique rows "
            f"for size {size}; batch too small/degenerate")

    def take(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(array)[indices])

    return PPOBatch(
        inputs={key: take(value) for key, value in batch.inputs.items()},
        action_tensors={name: take(value) for name, value
                        in batch.action_tensors.items()},
        old_logprob=take(batch.old_logprob),
        advantages=take(batch.advantages),
        returns=take(batch.returns),
        values=take(batch.values),
        learner_fingerprint=batch.learner_fingerprint,
        learner_fingerprints=(None if batch.learner_fingerprints is None else
                               tuple(batch.learner_fingerprints[index]
                                     for index in indices)),
    )


def recompute_stored_action_logprobs(
        policy: PPOPolicy,
        inputs: Mapping[str, np.ndarray],
        action_tensors: Mapping[str, np.ndarray],
        *,
        physical_batch_size: int | None = None,
        ) -> np.ndarray:
    """Recompute stored-action `logprob_total` in original row order.

    Slices the logical batch into contiguous physical chunks evaluated with
    `PPOPolicy.evaluate_actions()` — actions are never resampled. When
    `physical_batch_size` is set (fixed-B rollout inference), each chunk is
    evaluated at exactly that physical shape; a short final chunk is padded
    with the shared rollout helper (repeat chunk row 0) and padded outputs
    are discarded. Without a fixed size the whole logical batch is evaluated
    once, preserving existing behavior. Out-of-support stored actions still
    fail loudly inside `evaluate_actions()`.
    """
    from rl_manager.parallel import pad_batch_to_physical

    real_inputs = {name: np.asarray(array) for name, array in inputs.items()}
    real_actions = {name: np.asarray(array)
                    for name, array in action_tensors.items()}
    if not real_inputs or not real_actions:
        raise ValueError("logprob audit requires inputs and stored actions")
    rows = int(next(iter(real_inputs.values())).shape[0])
    if rows < 1:
        raise ValueError("logprob audit requires at least one stored row")
    for name, array in real_actions.items():
        if int(array.shape[0]) != rows:
            raise ValueError(
                f"stored action {name!r} has {int(array.shape[0])} rows, "
                f"expected {rows}")
    if physical_batch_size is None:
        return np.asarray(
            policy.evaluate_actions(inputs, action_tensors)["logprob_total"])
    size = int(physical_batch_size) if not isinstance(
        physical_batch_size, bool) else -1
    if size < 1:
        raise ValueError(
            "physical batch size must be a positive int, "
            f"got {physical_batch_size!r}")
    chunks = []
    for start in range(0, rows, size):
        chunk_inputs = {name: array[start:start + size]
                        for name, array in real_inputs.items()}
        chunk_actions = {name: array[start:start + size]
                         for name, array in real_actions.items()}
        real_count = int(next(iter(chunk_inputs.values())).shape[0])
        padded_inputs, _ = pad_batch_to_physical(chunk_inputs, size)
        padded_actions, _ = pad_batch_to_physical(chunk_actions, size)
        recomputed = np.asarray(policy.evaluate_actions(
            padded_inputs, padded_actions)["logprob_total"])
        chunks.append(recomputed[:real_count])
    return np.concatenate(chunks, axis=0)


__all__ = [
    "PPOBatchedPolicy",
    "ppo_batched_policy_from_state",
    "ppo_snapshot_from_state",
    "prng_key_from_id",
    "recompute_stored_action_logprobs",
    "select_ppo_subset",
]
