"""Conventional inspectable PPO V0 update over Stage-A trajectories (#9 B5).

Host code loops over epochs and minibatches; JAX never sees a Python
example loop. The loss/update pair is jitted once per (ManagerConfig,
PPOConfig) and operates on batch-leading contiguous arrays, so it is
sharding-compatible by construction.

Loss per minibatch:

    ratio        = exp(new_logprob - old_logprob)
    pi_loss      = -mean(min(ratio*A, clip(ratio)*A))
    value_loss   = mean((value - returns)^2)
    entropy      = mean(raw-summed entropy over all 17 cats + 54 Bernoullis)
    kl_to_frozen = mean analytic KL(current || frozen E), coef default 0
    loss         = pi_loss + value_coef*value_loss - entropy_coef*entropy
                   + kl_to_frozen_coef*kl_to_frozen

Advantages are normalized ONCE over the full valid update batch before any
epoch/minibatch split (`build_ppo_batch` does this via `rl_manager.gae`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bc_manager_jax.model import (
    ManagerConfig,
    _forward_eval,
    _forward_eval_with_representation,
    _prepare_inputs,
)

from rl_manager.gae import advantage_stats, compute_gae, valid_trainable_rows
from rl_manager.ppo_policy import (
    PPOConfig,
    action_index_tensors,
    combined_params_template,
    distribution_logits,
    frozen_leaf_mask,
    group_logprob_and_entropy,
    kl_to_frozen,
    make_ppo_optimizer,
    value_from_representation,
)


@dataclasses.dataclass(frozen=True)
class PPOBatch:
    """Contiguous host arrays for one PPO update (Stage-A schema actions)."""

    inputs: dict[str, np.ndarray]
    action_tensors: dict[str, np.ndarray]
    old_logprob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.old_logprob.shape[0])


def build_ppo_batch(
    arrays: Mapping[str, np.ndarray],
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: np.ndarray | None = None,
    normalize_advantages: bool = True,
    epsilon: float = 1e-8,
) -> PPOBatch:
    """Convert finalized/loaded trajectory arrays into a PPO batch.

    Only candidate/trainable valid rows enter the batch; every output array
    is made contiguous. GAE runs on the filtered rows grouped by
    episode+seat exactly as recorded.
    """
    rows = valid_trainable_rows(arrays)
    if rows.size == 0:
        raise ValueError("no candidate/trainable valid trajectory rows")
    inputs = {key[len("input_"):]: np.ascontiguousarray(arrays[key][rows])
              for key in arrays if key.startswith("input_")}
    action_tensors = {
        name: np.ascontiguousarray(arrays[f"action_{name}"][rows])
        for name in ("crop", "animal", "land", "fertilizer", "care",
                     "sell_presence", "sell_quantity")}
    gae = compute_gae(
        rewards=np.asarray(arrays["reward"])[rows],
        values=np.asarray(arrays["value"])[rows],
        terminated=np.asarray(arrays["terminated"])[rows],
        truncated=np.asarray(arrays["truncated"])[rows],
        episode_index=np.asarray(arrays["episode_index"])[rows],
        seat=np.asarray(arrays["seat"])[rows],
        day=np.asarray(arrays["day"])[rows],
        gamma=gamma, gae_lambda=gae_lambda,
        bootstrap_values=(None if bootstrap_values is None
                          else np.asarray(bootstrap_values)[rows]),
        normalize=normalize_advantages, epsilon=epsilon)
    return PPOBatch(
        inputs=inputs,
        action_tensors=action_tensors,
        old_logprob=np.ascontiguousarray(
            np.asarray(arrays["logprob_total"])[rows], dtype=np.float32),
        advantages=gae["advantages"],
        returns=gae["returns"],
        values=np.ascontiguousarray(
            np.asarray(arrays["value"])[rows], dtype=np.float32))


# ----------------------------------------------------------- train state


@dataclasses.dataclass
class PPOTrainState:
    """Mutable PPO training state (RL checkpoint payload, issue #9 req. 5)."""

    params: dict            # {"base": mutable E copy, "value": head}
    opt_state: Any          # optax state over the combined tree
    frozen_params: dict     # immutable frozen-E snapshot
    rng: jax.Array          # explicit PRNG stream for shuffles etc.
    step: int
    rollout_seed: int | None = None


def init_train_state(frozen_params: Mapping, config: ManagerConfig, *,
                     seed: int, ppo_config: PPOConfig,
                     model_variant: str = "E") -> PPOTrainState:
    """Mutable base copy + small value head + fresh optimizer state."""
    from rl_manager.ppo_policy import PPOPolicy  # local: avoids cycle cost

    policy = PPOPolicy(frozen_params, config, seed=seed,
                       model_variant=model_variant, ppo_config=ppo_config)
    mask = frozen_leaf_mask(policy.params)
    opt_state = make_ppo_optimizer(ppo_config, mask).init(policy.params)
    return PPOTrainState(params=policy.params, opt_state=opt_state,
                         frozen_params=policy.frozen_params, rng=policy.rng,
                         step=0)


# ------------------------------------------------------- compiled update


def clipped_surrogate_terms(log_ratio: jax.Array, advantages: jax.Array,
                            clip_eps: float) -> dict[str, jax.Array]:
    """Pure ratio/clipped-surrogate pieces (exposed for exact tests)."""
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = jnp.minimum(ratio * advantages, clipped * advantages)
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
    return {"ratio": ratio, "pi_loss": -jnp.mean(surrogate),
            "approx_kl": approx_kl, "clip_fraction": clip_fraction}


def explained_variance(values: jax.Array, returns: jax.Array) -> jax.Array:
    """1 - Var(residual)/Var(returns); epsilon-stable denominator."""
    return 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)


@functools.lru_cache(maxsize=None)
def _compiled_update(config: ManagerConfig, ppo_config: PPOConfig,
                     model_variant: str = "E"):
    """JIT'd single-minibatch update; cached per architecture/config."""
    template = combined_params_template(config, model_variant)
    optimizer = make_ppo_optimizer(ppo_config, frozen_leaf_mask(template))

    def loss_fn(params, frozen_params, prepared_inputs, indices,
                old_logprob, advantages, returns):
        mut_outputs, representation = _forward_eval_with_representation(
            params["base"], prepared_inputs, config, model_variant)
        logits = distribution_logits(mut_outputs)
        stats = group_logprob_and_entropy(logits, indices)
        value = value_from_representation(representation, params["value"])
        terms = clipped_surrogate_terms(stats["logprob_total"] - old_logprob,
                                        advantages, ppo_config.clip_eps)
        value_loss = jnp.mean(jnp.square(value - returns))
        entropy = jnp.mean(stats["entropy_total"])
        frozen_logits = distribution_logits(_forward_eval(
            frozen_params, prepared_inputs, config, model_variant))
        kl_frozen = jnp.mean(kl_to_frozen(logits, frozen_logits))
        loss = (terms["pi_loss"] + ppo_config.value_coef * value_loss
                - ppo_config.entropy_coef * entropy
                + ppo_config.kl_to_frozen_coef * kl_frozen)
        metrics = {
            "loss": loss,
            "pi_loss": terms["pi_loss"],
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": terms["approx_kl"],
            "clip_fraction": terms["clip_fraction"],
            "kl_to_frozen": kl_frozen,
            "explained_variance": explained_variance(value, returns),
        }
        metrics.update({f"entropy_{name}": jnp.mean(array) for name, array
                        in stats["entropy_groups"].items()})
        return loss, metrics

    def core(params, opt_state, frozen_params, prepared_inputs, indices,
             old_logprob, advantages, returns):
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, frozen_params, prepared_inputs,
                                   indices, old_logprob, advantages, returns)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["loss"] = loss
        return new_params, new_opt_state, metrics

    return jax.jit(core)


# ------------------------------------------------------------ host loop


_METRIC_KEYS = ("loss", "pi_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "kl_to_frozen", "explained_variance",
                "entropy_crop", "entropy_animal", "entropy_land",
                "entropy_fertilizer", "entropy_care", "entropy_sell_presence")


def ppo_update(state: PPOTrainState, batch: PPOBatch, config: ManagerConfig,
               ppo_config: PPOConfig, *,
               model_variant: str = "E") -> tuple[PPOTrainState, dict]:
    """Run `epochs` x (N / minibatch_size) jitted minibatch updates.

    Shuffle uses an explicit deterministic PRNG stream carried in the train
    state; N must be divisible by minibatch_size (fail loud rather than
    silently dropping rows). Returns the advanced state plus size-weighted
    metric means and full-batch advantage stats.
    """
    n = batch.size
    mb = ppo_config.minibatch_size
    if n % mb != 0:
        raise ValueError(
            f"batch size {n} must be divisible by minibatch_size {mb} "
            f"(fail loud instead of silently dropping rows)")
    variant = model_variant
    core = _compiled_update(config, ppo_config, variant)

    perm_key, next_rng = jax.random.split(state.rng)
    perm = np.asarray(jax.random.permutation(perm_key, n))
    dev_inputs = _prepare_inputs({k: jnp.asarray(v)
                                  for k, v in batch.inputs.items()})
    dev_indices = action_index_tensors(batch.action_tensors, n)
    dev_old = jnp.asarray(batch.old_logprob)
    dev_adv = jnp.asarray(batch.advantages)
    dev_ret = jnp.asarray(batch.returns)

    params, opt_state = state.params, state.opt_state
    sums = {key: 0.0 for key in _METRIC_KEYS}
    total_rows = 0
    for _epoch in range(ppo_config.epochs):
        for start in range(0, n, mb):
            sel = perm[start:start + mb]
            gather = lambda array: array[jnp.asarray(sel)]  # noqa: E731
            params, opt_state, metrics = core(
                params, opt_state, state.frozen_params,
                {k: gather(v) for k, v in dev_inputs.items()},
                {k: gather(v) for k, v in dev_indices.items()},
                gather(dev_old), gather(dev_adv), gather(dev_ret))
            weight = len(sel)
            total_rows += weight
            for key in _METRIC_KEYS:
                sums[key] += float(metrics[key]) * weight
    metrics_out = {key: sums[key] / total_rows for key in _METRIC_KEYS}
    metrics_out.update(advantage_stats(batch.advantages))
    new_state = PPOTrainState(params=params, opt_state=opt_state,
                              frozen_params=state.frozen_params,
                              rng=next_rng, step=state.step
                              + ppo_config.epochs * (n // mb),
                              rollout_seed=state.rollout_seed)
    return new_state, metrics_out


__all__ = [
    "PPOBatch",
    "PPOTrainState",
    "build_ppo_batch",
    "clipped_surrogate_terms",
    "explained_variance",
    "init_train_state",
    "ppo_update",
]
