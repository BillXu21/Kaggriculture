"""Conventional inspectable PPO V0 update over Stage-A trajectories (#9 B5).

Host code loops over epochs and minibatches; JAX never sees a Python
example loop. The loss/update pair is jitted once per (ManagerConfig,
PPOConfig) and operates on batch-leading contiguous arrays, so it is
sharding-compatible by construction.

Loss per minibatch:

    ratio        = exp(new_logprob - old_logprob)
    pi_loss      = -mean(min(ratio*A, clip(ratio)*A))
    value_loss   = mean((value - returns)^2)
    entropy      = mean behavior-policy entropy over all 17 cats + 54 Bernoullis
    kl_to_frozen = mean analytic KL(current curriculum policy || frozen E),
                    coef default 0
    loss         = pi_loss + value_coef*value_loss - entropy_coef*entropy
                   + kl_to_frozen_coef*kl_to_frozen

Advantages are normalized ONCE over the full valid update batch before any
epoch/minibatch split (`build_ppo_batch` does this via `rl_manager.gae`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping, Sequence

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
    CurriculumMaskConfig,
    PPOConfig,
    TargetedExplorationConfig,
    action_index_tensors,
    apply_curriculum_mask,
    apply_targeted_exploration,
    combined_params_template,
    curriculum_behavior_fingerprint,
    distribution_logits,
    frozen_leaf_mask,
    group_logprob_and_entropy,
    kl_to_frozen,
    make_ppo_optimizer,
    value_from_representation,
)
from rl_manager.policy import params_fingerprint


@dataclasses.dataclass(frozen=True)
class PPOBatch:
    """Contiguous host arrays for one PPO update (Stage-A schema actions)."""

    inputs: dict[str, np.ndarray]
    action_tensors: dict[str, np.ndarray]
    old_logprob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray
    learner_fingerprint: str | None = None
    learner_fingerprints: tuple[str, ...] | None = None

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
    sidecar_records: Sequence[Any] | None = None,
    learner_fingerprints: Sequence[str] | None = None,
) -> PPOBatch:
    """Convert finalized/loaded trajectory arrays into a PPO batch.

    Only candidate/trainable valid rows enter the batch; every output array
    is made contiguous. GAE runs on the filtered rows grouped by
    episode+seat exactly as recorded.
    """
    rows = valid_trainable_rows(arrays)
    if rows.size == 0:
        raise ValueError("no candidate/trainable valid trajectory rows")
    fingerprints: list[str] = []
    if sidecar_records is not None:
        if len(sidecar_records) < int(rows.max()) + 1:
            raise ValueError("trajectory sidecar is shorter than PPO arrays")
        fingerprints = [str(record["policy_fingerprint"])
                        if isinstance(record, Mapping) else
                        str(record.policy_fingerprint)
                        for row in rows
                        for record in (sidecar_records[int(row)],)]
    elif learner_fingerprints is not None:
        values = list(learner_fingerprints)
        if len(values) != len(arrays["valid"]):
            raise ValueError("learner_fingerprints must align with trajectory rows")
        fingerprints = [str(values[int(row)]) for row in rows]
    unique_fingerprints = set(fingerprints)
    if len(unique_fingerprints) > 1:
        raise ValueError(
            "one PPO batch contains trainable rows sampled from multiple "
            f"learner fingerprints: {sorted(unique_fingerprints)}")
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
            np.asarray(arrays["value"])[rows], dtype=np.float32),
        learner_fingerprint=(next(iter(unique_fingerprints))
                             if unique_fingerprints else None),
        learner_fingerprints=(tuple(fingerprints) if fingerprints else None))


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
                     model_variant: str = "E",
                     initial_base_params: Mapping | None = None,
                     curriculum: CurriculumMaskConfig | None = None,
                     exploration: TargetedExplorationConfig | None = None
                     ) -> PPOTrainState:
    """Mutable base copy + small value head + fresh optimizer state."""
    from rl_manager.ppo_policy import PPOPolicy  # local: avoids cycle cost

    policy = PPOPolicy(frozen_params, config, seed=seed,
                        model_variant=model_variant, ppo_config=ppo_config,
                        initial_base_params=initial_base_params,
                        curriculum=curriculum, exploration=exploration)
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
                     model_variant: str = "E",
                     curriculum: CurriculumMaskConfig = CurriculumMaskConfig(),
                     exploration: TargetedExplorationConfig =
                     TargetedExplorationConfig()):
    """JIT'd single-minibatch update; cached per architecture/config."""
    template = combined_params_template(config, model_variant)
    optimizer = make_ppo_optimizer(ppo_config, frozen_leaf_mask(template))

    def loss_fn(params, frozen_params, prepared_inputs, indices,
                old_logprob, advantages, returns):
        mut_outputs, representation = _forward_eval_with_representation(
            params["base"], prepared_inputs, config, model_variant)
        logits = apply_curriculum_mask(
            distribution_logits(mut_outputs), curriculum)
        behavior_logits = apply_targeted_exploration(
            logits, exploration, curriculum)
        stats = group_logprob_and_entropy(behavior_logits, indices)
        value = value_from_representation(representation, params["value"])
        terms = clipped_surrogate_terms(stats["logprob_total"] - old_logprob,
                                        advantages, ppo_config.clip_eps)
        value_loss = jnp.mean(jnp.square(value - returns))
        entropy = jnp.mean(stats["entropy_total"])
        frozen_logits = apply_curriculum_mask(
            distribution_logits(_forward_eval(
                frozen_params, prepared_inputs, config, model_variant)),
            curriculum)
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


@functools.lru_cache(maxsize=None)
def _compiled_epoch_metrics(config: ManagerConfig, ppo_config: PPOConfig,
                            model_variant: str = "E",
                            curriculum: CurriculumMaskConfig = CurriculumMaskConfig(),
                            exploration: TargetedExplorationConfig =
                            TargetedExplorationConfig()):
    """JIT'd full-batch diagnostics evaluated after each PPO epoch."""
    def evaluate(params, frozen_params, prepared_inputs, indices,
                 old_logprob, advantages, returns):
        mut_outputs, representation = _forward_eval_with_representation(
            params["base"], prepared_inputs, config, model_variant)
        logits = apply_curriculum_mask(
            distribution_logits(mut_outputs), curriculum)
        behavior_logits = apply_targeted_exploration(
            logits, exploration, curriculum)
        stats = group_logprob_and_entropy(behavior_logits, indices)
        value = value_from_representation(representation, params["value"])
        terms = clipped_surrogate_terms(stats["logprob_total"] - old_logprob,
                                        advantages, ppo_config.clip_eps)
        frozen_logits = apply_curriculum_mask(
            distribution_logits(_forward_eval(
                frozen_params, prepared_inputs, config, model_variant)),
            curriculum)
        metrics = {
            "loss": terms["pi_loss"],
            "pi_loss": terms["pi_loss"],
            "value_loss": jnp.mean(jnp.square(value - returns)),
            "entropy": jnp.mean(stats["entropy_total"]),
            "approx_kl": terms["approx_kl"],
            "clip_fraction": terms["clip_fraction"],
            "kl_to_frozen": jnp.mean(kl_to_frozen(logits, frozen_logits)),
            "explained_variance": explained_variance(value, returns),
        }
        metrics.update({f"entropy_{name}": jnp.mean(array) for name, array
                        in stats["entropy_groups"].items()})
        return metrics

    return jax.jit(evaluate)


# ------------------------------------------------------------ host loop


_METRIC_KEYS = ("loss", "pi_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "kl_to_frozen", "explained_variance",
                "entropy_crop", "entropy_animal", "entropy_land",
                "entropy_fertilizer", "entropy_care", "entropy_sell_presence")


def ppo_update(state: PPOTrainState, batch: PPOBatch, config: ManagerConfig,
               ppo_config: PPOConfig, *,
               model_variant: str = "E",
               curriculum: CurriculumMaskConfig | None = None,
               exploration: TargetedExplorationConfig | None = None
               ) -> tuple[PPOTrainState, dict]:
    """Run `epochs` x (N / minibatch_size) jitted minibatch updates.

    Shuffle uses an explicit deterministic PRNG stream carried in the train
    state; N must be divisible by minibatch_size (fail loud rather than
    silently dropping rows). Returns the advanced state plus size-weighted
    metric means and full-batch advantage stats. Epoch diagnostics are
    evaluated after each epoch; target-KL stopping and pathological rejection
    are reported in the returned metadata. A rejected update returns ``state``
    itself, preserving its parameters, optimizer state, RNG, and step.
    """
    active_curriculum = curriculum or CurriculumMaskConfig()
    active_exploration = exploration or TargetedExplorationConfig()
    n = batch.size
    mb = ppo_config.minibatch_size
    if batch.learner_fingerprints is not None:
        fingerprints = set(batch.learner_fingerprints)
        if len(batch.learner_fingerprints) != n or len(fingerprints) > 1:
            raise ValueError(
                "one PPO batch contains trainable rows sampled from multiple "
                f"learner fingerprints: {sorted(fingerprints)}")
    if batch.learner_fingerprint is not None:
        expected_fingerprint = curriculum_behavior_fingerprint(
            state.params, active_curriculum, active_exploration)
        legacy_fingerprint = params_fingerprint(state.params)
        accepted_fingerprints = {expected_fingerprint}
        if active_curriculum.unrestricted and active_exploration.inactive:
            accepted_fingerprints.add(legacy_fingerprint)
        if (len(str(batch.learner_fingerprint)) == 64 and
                batch.learner_fingerprint not in accepted_fingerprints):
            raise ValueError(
                "PPO batch learner fingerprint does not match the active "
                f"policy curriculum/parameters: stored="
                f"{batch.learner_fingerprint!r}, expected="
                f"{expected_fingerprint!r}")
    if n % mb != 0:
        raise ValueError(
            f"batch size {n} must be divisible by minibatch_size {mb} "
            f"(fail loud instead of silently dropping rows)")
    variant = model_variant
    core = _compiled_update(config, ppo_config, variant, active_curriculum,
                            active_exploration)

    perm_key, next_rng = jax.random.split(state.rng)
    perm = np.asarray(jax.random.permutation(perm_key, n))
    dev_inputs = _prepare_inputs({k: jnp.asarray(v)
                                  for k, v in batch.inputs.items()})
    dev_indices = action_index_tensors(
        batch.action_tensors, n, curriculum=active_curriculum)
    dev_old = jnp.asarray(batch.old_logprob)
    dev_adv = jnp.asarray(batch.advantages)
    dev_ret = jnp.asarray(batch.returns)

    params, opt_state = state.params, state.opt_state
    sums = {key: 0.0 for key in _METRIC_KEYS}
    total_rows = 0
    epoch_metrics: list[dict[str, float]] = []
    epochs_ran = 0
    minibatches_ran = 0
    stop_reason = "completed"
    rejection_reason: str | None = None
    epoch_core = _compiled_epoch_metrics(
        config, ppo_config, variant, active_curriculum, active_exploration)

    def finite_tree(tree) -> bool:
        return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in
                   jax.tree_util.tree_leaves(tree))

    for epoch in range(ppo_config.epochs):
        rejected_metric: str | None = None
        for start in range(0, n, mb):
            sel = perm[start:start + mb]
            gather = lambda array: array[jnp.asarray(sel)]  # noqa: E731
            params, opt_state, metrics = core(
                params, opt_state, state.frozen_params,
                {k: gather(v) for k, v in dev_inputs.items()},
                {k: gather(v) for k, v in dev_indices.items()},
                gather(dev_old), gather(dev_adv), gather(dev_ret))
            minibatches_ran += 1
            rejected_metric = (next(
                (key for key, value in metrics.items()
                 if not np.all(np.isfinite(np.asarray(value)))), None)
                if ppo_config.reject_update_kl is not None else None)
            if rejected_metric is not None:
                stop_reason = "rejected_nonfinite"
                rejection_reason = f"nonfinite minibatch metric: {rejected_metric}"
                break
            weight = len(sel)
            total_rows += weight
            for key in _METRIC_KEYS:
                sums[key] += float(metrics[key]) * weight
            if (ppo_config.reject_update_kl is not None and
                    (not finite_tree(params) or not finite_tree(opt_state))):
                stop_reason = "rejected_nonfinite"
                rejection_reason = "nonfinite parameter or optimizer state"
                break
        if rejection_reason is not None:
            break

        full_metrics = epoch_core(
            params, state.frozen_params, dev_inputs, dev_indices,
            dev_old, dev_adv, dev_ret)
        epoch_metric = {key: float(value) for key, value in full_metrics.items()}
        rejected_metric = (next(
            (key for key, value in epoch_metric.items()
             if not np.isfinite(value)), None)
            if ppo_config.reject_update_kl is not None else None)
        if (rejected_metric is not None or
                (ppo_config.reject_update_kl is not None and
                 (not finite_tree(params) or not finite_tree(opt_state)))):
            stop_reason = "rejected_nonfinite"
            rejection_reason = (f"nonfinite epoch metric: {rejected_metric}"
                                if rejected_metric is not None
                                else "nonfinite parameter or optimizer state")
            break
        epoch_metric["epoch"] = float(epoch + 1)
        epoch_metric["minibatches"] = float(n // mb)
        epoch_metrics.append(epoch_metric)
        epochs_ran += 1

        if (ppo_config.reject_update_kl is not None and
                epoch_metric["approx_kl"] > ppo_config.reject_update_kl):
            stop_reason = "rejected_kl"
            rejection_reason = (
                f"approx_kl {epoch_metric['approx_kl']:.6g} exceeds "
                f"reject_update_kl {ppo_config.reject_update_kl:.6g}")
            break
        if (ppo_config.target_kl is not None and
                epoch_metric["approx_kl"] > ppo_config.target_kl):
            stop_reason = "target_kl"
            break

    if rejection_reason is not None:
        metrics_out = {key: sums[key] / total_rows for key in _METRIC_KEYS} \
            if total_rows else {}
        metrics_out.update({"epoch_metrics": epoch_metrics,
                            "epochs_ran": epochs_ran,
                            "minibatches_ran": minibatches_ran,
                            "rows_ran": total_rows,
                            "stop_reason": stop_reason,
                            "rejection_reason": rejection_reason,
                            "accepted": False})
        return state, metrics_out

    metrics_out = {key: sums[key] / total_rows for key in _METRIC_KEYS}
    metrics_out.update(advantage_stats(batch.advantages))
    metrics_out.update({"epoch_metrics": epoch_metrics,
                        "epochs_ran": epochs_ran,
                        "minibatches_ran": minibatches_ran,
                        "rows_ran": total_rows,
                        "stop_reason": stop_reason,
                        "rejection_reason": None,
                        "accepted": True})
    new_state = PPOTrainState(params=params, opt_state=opt_state,
                              frozen_params=state.frozen_params,
                              rng=next_rng, step=state.step
                              + minibatches_ran,
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
