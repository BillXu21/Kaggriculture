"""Stage-A `BatchedPlanPolicy` adapter over the B1 PPO policy (issue #9 B2).

This module is pure integration glue: it adapts `rl_manager.ppo_policy.
PPOPolicy` to the framework-neutral `rl_manager.types.BatchedPlanPolicy`
protocol consumed by `SelfPlayRunner`, plus the small helpers the tiny
end-to-end smoke needs (state -> eval-policy reconstruction and a
deterministic subset selection over an already-GAE-normalized batch).

Contract implemented here:

- consumes contiguous own-only E input arrays plus an explicit string
  `prng_id`; the id is hashed into one root JAX key whose per-row decision
  seeds are `fold_in(root, row_index)` inside `PPOPolicy.act` (sampling
  attaches to row identity, never batch position);
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
from typing import Mapping

import jax
import numpy as np

from rl_manager.ppo import PPOBatch, PPOTrainState
from rl_manager.ppo_policy import PPOConfig, PPOPolicy
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
    ) -> None:
        self._policy = policy
        self._deterministic = bool(deterministic)
        self.identity = PolicyIdentity(
            name=name,
            version=version,
            fingerprint=params_fingerprint(policy.params),
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
            fingerprint=params_fingerprint(self._policy.params),
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


def ppo_batched_policy_from_state(
    state: PPOTrainState,
    config,
    *,
    ppo_config: PPOConfig | None = None,
    name: str = "ppo_candidate",
    version: str = "ppo-v0",
    deterministic: bool = False,
) -> PPOBatchedPolicy:
    """Reconstruct an eval/train-capable adapter from a loaded train state.

    `PPOPolicy.__init__` derives params/value-head/rng from
    `(frozen_params, seed)`; a checkpoint resume must instead reuse the
    EXACT stored mutable params, frozen snapshot, and PRNG stream, so they
    are re-assigned onto a freshly constructed instance (no B1 edit).
    """
    policy = PPOPolicy(
        state.frozen_params, config, seed=0,
        ppo_config=ppo_config or PPOConfig())
    policy.params = state.params
    policy.frozen_params = state.frozen_params
    policy.rng = state.rng
    return PPOBatchedPolicy(
        policy, name=name, version=version, deterministic=deterministic)


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
    )


__all__ = [
    "PPOBatchedPolicy",
    "ppo_batched_policy_from_state",
    "prng_key_from_id",
    "select_ppo_subset",
]
