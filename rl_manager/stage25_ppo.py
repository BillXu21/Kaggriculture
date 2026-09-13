"""Native JAX PPO primitives for strict Stage 2.5 trajectories.

This module is intentionally separate from the legacy grouped-action PPO.  A
Stage 2.5 update uses one joint autoregressive likelihood for the nine stored
classes and manager-boundary rewards discounted once per recorded row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl_manager.stage25_policy import (
    ACTION_CLASS_COUNTS,
    Stage25ModelConfig,
    _host_contexts,
    _host_inputs,
    _stage25_jit,
    init_stage25_params,
)
from rl_manager.stage25_types import Stage25BehaviorIdentity
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer


PPO_SCHEMA_VERSION = "stage25_ppo_v1"
RESUME_BOUNDARY = "after_completed_rollout_update_before_next_collection"


def compute_stage25_gae(
        *, episode_id: Any, seat: Any, day: Any, rewards: Any, values: Any,
        terminated: Any, truncated: Any, bootstrap_values: Any,
        bootstrap_patched: Any, gamma: float, gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute manager-boundary GAE independently for each episode and seat.

    A genuine terminal has zero bootstrap.  A truncation uses its explicitly
    recorded bootstrap for the one-step delta, then stops the GAE recursion.
    Rows are processed in chronological day order without assuming that rows
    for different seats or episodes are contiguous in storage.
    """
    arrays = {
        "episode_id": np.asarray(episode_id),
        "seat": np.asarray(seat),
        "day": np.asarray(day),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "bootstrap_values": np.asarray(bootstrap_values, dtype=np.float32),
        "bootstrap_patched": np.asarray(bootstrap_patched, dtype=bool),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if not shapes or any(shape != (len(arrays["episode_id"]),) for shape in shapes.values()):
        raise ValueError(f"GAE fields must all be rank-one and row-aligned; got {shapes}")
    if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gamma and gae_lambda must be finite values in [0, 1]")
    if not np.isfinite(float(gamma)) or not np.isfinite(float(gae_lambda)):
        raise ValueError("gamma and gae_lambda must be finite values in [0, 1]")
    if np.any(arrays["terminated"] & arrays["truncated"]):
        raise ValueError("a GAE row cannot be both terminated and truncated")
    if np.any(arrays["truncated"] & ~arrays["bootstrap_patched"]):
        raise ValueError("every truncated GAE row requires a recorded bootstrap")
    if np.any(arrays["terminated"] & arrays["bootstrap_patched"]):
        raise ValueError("terminal GAE rows cannot carry a bootstrap")
    for name in ("rewards", "values", "bootstrap_values"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} contains NaN or Inf")

    advantages = np.zeros(len(arrays["episode_id"]), dtype=np.float32)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(arrays["episode_id"], arrays["seat"], strict=True)):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)

    for key, indices in groups.items():
        ordered = sorted(indices, key=lambda index: int(arrays["day"][index]))
        ordered_days = [int(arrays["day"][index]) for index in ordered]
        if len(set(ordered_days)) != len(ordered_days):
            raise ValueError(f"duplicate manager day in episode/seat {key}")
        if any(next_day != current_day + 1
               for current_day, next_day in zip(ordered_days, ordered_days[1:])):
            raise ValueError(
                f"episode/seat {key} contains an unsupported partial-day truncation")
        if not (arrays["terminated"][ordered[-1]] or arrays["truncated"][ordered[-1]]):
            raise ValueError(f"episode/seat {key} is missing a terminal or truncation boundary")
        running = np.float32(0.0)
        for position in range(len(ordered) - 1, -1, -1):
            index = ordered[position]
            terminal = bool(arrays["terminated"][index])
            truncation = bool(arrays["truncated"][index])
            if (terminal or truncation) and position != len(ordered) - 1:
                raise ValueError(f"episode/seat {key} contains rows after an end boundary")
            if terminal:
                next_value = np.float32(0.0)
                continues = np.float32(0.0)
            elif truncation:
                next_value = arrays["bootstrap_values"][index]
                continues = np.float32(0.0)
            else:
                next_value = arrays["values"][ordered[position + 1]]
                continues = np.float32(1.0)
            delta = (arrays["rewards"][index]
                     + np.float32(gamma) * next_value
                     - arrays["values"][index])
            running = delta + np.float32(gamma * gae_lambda) * continues * running
            advantages[index] = running

    returns = advantages + arrays["values"]
    return advantages, returns.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Stage25PPOConfig:
    """Native PPO settings; all discounts are per manager decision row."""

    model: Stage25ModelConfig = Stage25ModelConfig.tiny()
    physical_batch_size: int = 16
    minibatch_size: int = 32
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    normalize_advantages: bool = True
    audit_atol: float = 1e-5
    audit_rtol: float = 1e-5

    def __post_init__(self) -> None:
        integer_fields = ("physical_batch_size", "minibatch_size", "epochs")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        positive = ("learning_rate", "gradient_clip", "audit_atol", "audit_rtol")
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        bounded = ("gamma", "gae_lambda", "clip_epsilon")
        for name in bounded:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (name != "clip_epsilon" and value > 1.0):
                raise ValueError(f"{name} must be finite and in its supported range")
        for name in ("entropy_coefficient", "value_coefficient", "weight_decay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), **{
            name: getattr(self, name) for name in (
                "physical_batch_size", "minibatch_size", "epochs",
                "learning_rate", "gamma", "gae_lambda", "clip_epsilon",
                "entropy_coefficient", "value_coefficient", "gradient_clip",
                "weight_decay", "normalize_advantages", "audit_atol", "audit_rtol")}}


@dataclass(frozen=True)
class Stage25PPOBatch:
    """Validated learner-only logical PPO rows.

    ``old_component_logprobs`` is retained for diagnostics.  The optimization
    ratio uses only ``old_joint_logprobs`` and the policy's new joint value.
    """

    inputs: Mapping[str, np.ndarray]
    classes: np.ndarray
    old_component_logprobs: np.ndarray
    old_joint_logprobs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_id: np.ndarray
    seat: np.ndarray
    day: np.ndarray
    physical_contexts: Sequence[Any] | None = None
    row_ids: np.ndarray | None = None
    behavior_identity: Stage25BehaviorIdentity | None = None

    def __post_init__(self) -> None:
        classes = np.asarray(self.classes)
        if classes.ndim != 2 or classes.shape[1] != len(ACTION_CLASS_COUNTS) or not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("classes must be an integer array with shape [N, 9]")
        classes = classes.astype(np.int32, copy=True)
        counts = np.asarray(ACTION_CLASS_COUNTS, dtype=np.int32)
        if np.any(classes < 0) or np.any(classes >= counts[None, :]):
            raise ValueError("classes contain an out-of-vocabulary action")
        n = classes.shape[0]
        fields = {
            "old_component_logprobs": (self.old_component_logprobs, (n, 9)),
            "old_joint_logprobs": (self.old_joint_logprobs, (n,)),
            "old_values": (self.old_values, (n,)),
            "advantages": (self.advantages, (n,)),
            "returns": (self.returns, (n,)),
            "episode_id": (self.episode_id, (n,)),
            "seat": (self.seat, (n,)),
            "day": (self.day, (n,)),
        }
        normalized: dict[str, np.ndarray] = {}
        for name, (value, shape) in fields.items():
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name in ("old_component_logprobs", "old_joint_logprobs", "old_values", "advantages", "returns"):
                array = array.astype(np.float32, copy=True)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains NaN or Inf")
            else:
                array = array.astype(np.int64, copy=True)
            normalized[name] = array
        if not np.allclose(normalized["old_joint_logprobs"], np.sum(normalized["old_component_logprobs"], axis=1, dtype=np.float32), atol=1e-5, rtol=1e-5):
            raise ValueError("old_joint_logprobs must equal the raw component-logprob sum")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("inputs must be a nonempty mapping")
        for name, value in self.inputs.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != n:
                raise ValueError(f"input {name!r} must have leading row dimension {n}")
            if array.dtype.hasobject:
                raise ValueError(f"input {name!r} has object dtype")
        if self.physical_contexts is not None and len(self.physical_contexts) != n:
            raise ValueError("physical_contexts must contain one item per logical row")
        if self.row_ids is not None:
            row_ids = np.asarray(self.row_ids, dtype=np.int64)
            if row_ids.shape != (n,):
                raise ValueError("row_ids must have shape [N]")
            object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "classes", classes)
        for name, array in normalized.items():
            object.__setattr__(self, name, array)

    def take(self, indices: Any) -> "Stage25PPOBatch":
        indices = np.asarray(indices, dtype=np.int64)
        inputs = {name: np.asarray(value)[indices].copy() for name, value in self.inputs.items()}
        contexts = None if self.physical_contexts is None else tuple(self.physical_contexts[int(i)] for i in indices)
        ids = None if self.row_ids is None else self.row_ids[indices]
        return Stage25PPOBatch(
            inputs, self.classes[indices], self.old_component_logprobs[indices],
            self.old_joint_logprobs[indices], self.old_values[indices],
            self.advantages[indices], self.returns[indices], self.episode_id[indices],
            self.seat[indices], self.day[indices], contexts, ids,
            self.behavior_identity)


def _identity_matches(actual: Stage25BehaviorIdentity, expected: Stage25BehaviorIdentity) -> bool:
    return actual.to_json_dict() == expected.to_json_dict()


def build_stage25_ppo_batch(
        trajectory: Stage25TrajectoryBuffer, *, learner_identity: Stage25BehaviorIdentity,
        gamma: float = 0.99, gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
) -> Stage25PPOBatch:
    """Select only explicit learner rows and compute isolated GAE."""
    if not isinstance(trajectory, Stage25TrajectoryBuffer):
        raise TypeError("trajectory must be a Stage25TrajectoryBuffer")
    if not isinstance(learner_identity, Stage25BehaviorIdentity):
        raise TypeError("learner_identity must be Stage25BehaviorIdentity")
    rows = list(trajectory.iter_rows())
    selected = []
    for row in rows:
        if not row.trainable:
            continue
        if not _identity_matches(row.learner_identity, learner_identity):
            raise ValueError("trainable trajectory row has an incompatible learner identity")
        selected.append(row)
    if not selected:
        raise ValueError("trajectory contains no trainable learner rows")
    inputs = {name: np.stack([row.inputs[name] for row in selected]) for name in selected[0].inputs}
    components = np.stack([row.component_logprobs for row in selected]).astype(np.float32)
    old_joint = np.asarray([row.joint_logprob for row in selected], dtype=np.float32)
    old_values = np.asarray([row.value for row in selected], dtype=np.float32)
    episode_id = np.asarray([row.episode_id for row in selected], dtype=np.int64)
    seat = np.asarray([row.seat for row in selected], dtype=np.int64)
    day = np.asarray([row.day for row in selected], dtype=np.int64)
    rewards = np.asarray([row.reward for row in selected], dtype=np.float32)
    terminated = np.asarray([row.terminated for row in selected], dtype=bool)
    truncated = np.asarray([row.truncated for row in selected], dtype=bool)
    bootstrap = np.asarray([row.bootstrap_value for row in selected], dtype=np.float32)
    bootstrap_patched = np.asarray([row.bootstrap_patched for row in selected], dtype=bool)
    advantages, returns = compute_stage25_gae(
        episode_id=episode_id, seat=seat, day=day, rewards=rewards,
        values=old_values, terminated=terminated, truncated=truncated,
        bootstrap_values=bootstrap, bootstrap_patched=bootstrap_patched,
        gamma=gamma, gae_lambda=gae_lambda)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / np.sqrt(advantages.var() + 1e-8)
    return Stage25PPOBatch(
        inputs=inputs, classes=np.stack([row.classes for row in selected]),
        old_component_logprobs=components, old_joint_logprobs=old_joint,
        old_values=old_values, advantages=advantages.astype(np.float32),
        returns=returns, episode_id=episode_id, seat=seat, day=day,
        physical_contexts=None, row_ids=np.arange(len(selected), dtype=np.int64),
        behavior_identity=learner_identity)


build_ppo_batch = build_stage25_ppo_batch


def joint_clipped_surrogate(
        old_joint_logprobs: Any, new_joint_logprobs: Any,
        advantages: Any, clip_epsilon: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the ordinary PPO surrogate using one joint autoregressive ratio."""
    old = jnp.asarray(old_joint_logprobs, dtype=jnp.float32)
    new = jnp.asarray(new_joint_logprobs, dtype=jnp.float32)
    adv = jnp.asarray(advantages, dtype=jnp.float32)
    ratio = jnp.exp(new - old)
    clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    terms = jnp.minimum(ratio * adv, clipped * adv)
    return -jnp.mean(terms), {"ratio": ratio, "clipped_ratio": clipped, "terms": terms}


def _pad_inputs(inputs: Mapping[str, np.ndarray], total: int) -> dict[str, np.ndarray]:
    result = {}
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.shape[0] == total:
            result[name] = array
        else:
            result[name] = np.concatenate(
                (array, np.repeat(array[-1:], total - array.shape[0], axis=0)), axis=0)
    return result


def _default_context_values(batch: int) -> tuple[jax.Array, ...]:
    return tuple(jnp.zeros((batch,) if index in (0, 3, 4) else (batch, 3) if index in (2, 5) else (batch, 4,), dtype=jnp.int32) for index in range(6))


def _physical_views(batch: Stage25PPOBatch, model: Stage25ModelConfig, physical_batch_size: int):
    n = len(batch.classes)
    chunks = (n + physical_batch_size - 1) // physical_batch_size
    total = chunks * physical_batch_size
    padded_inputs = _pad_inputs(batch.inputs, total)
    prepared, capacity, size = _host_inputs(padded_inputs, model)
    del size
    explicit = batch.physical_contexts is not None
    contexts = None
    if batch.physical_contexts is not None:
        contexts = _host_contexts(tuple(batch.physical_contexts) + (batch.physical_contexts[-1],) * (total - n), total)
    if contexts is None:
        contexts = _default_context_values(total)
    row_ids = np.asarray(batch.row_ids if batch.row_ids is not None else np.arange(n), dtype=np.int32)
    row_ids = np.pad(row_ids, (0, total - n), mode="edge")
    classes = np.pad(batch.classes, ((0, total - n), (0, 0)), mode="edge")
    views = (
        jax.tree_util.tree_map(lambda value: value.reshape((chunks, physical_batch_size) + value.shape[1:]), prepared),
        capacity.reshape((chunks, physical_batch_size) + capacity.shape[1:]),
        jnp.asarray(classes, dtype=jnp.int32).reshape((chunks, physical_batch_size, 9)),
        tuple(value.reshape((chunks, physical_batch_size) + value.shape[1:]) for value in contexts),
        jnp.asarray(row_ids).reshape((chunks, physical_batch_size)),
        chunks, total, n)
    return views + (explicit,)


def _map_policy_eval(params: Mapping[str, Any], batch: Stage25PPOBatch, config: Stage25PPOConfig):
    prepared, capacity, classes, contexts, row_ids, chunks, total, n, explicit = _physical_views(batch, config.model, config.physical_batch_size)
    keys = jnp.zeros((chunks, config.physical_batch_size, 2), dtype=jnp.uint32)
    def run(one):
        inputs, ledger, actions, context, ids, key = one
        return _stage25_jit(params, inputs, ledger, key, actions, context, ids, config.model, "eval", explicit)
    mapped = jax.lax.map(run, (prepared, capacity, classes, contexts, row_ids, keys))
    def flatten(value):
        if isinstance(value, Mapping):
            return {name: flatten(child) for name, child in value.items()}
        array = jnp.asarray(value)
        return array.reshape((total,) + array.shape[2:])[:n]
    return flatten(mapped)


def _objective_from_output(
        output: Mapping[str, Any], batch: Stage25PPOBatch,
        config: Stage25PPOConfig) -> tuple[jax.Array, dict[str, jax.Array], Mapping[str, Any]]:
    new_joint = jnp.asarray(output["joint_logprob"], dtype=jnp.float32)
    old_joint = jnp.asarray(batch.old_joint_logprobs, dtype=jnp.float32)
    advantages = jnp.asarray(batch.advantages, dtype=jnp.float32)
    policy_loss, ratio_metrics = joint_clipped_surrogate(
        old_joint, new_joint, advantages, config.clip_epsilon)
    new_value = jnp.asarray(output["value"], dtype=jnp.float32)
    old_values = jnp.asarray(batch.old_values, dtype=jnp.float32)
    returns = jnp.asarray(batch.returns, dtype=jnp.float32)
    value_loss = 0.5 * jnp.mean(jnp.square(new_value - returns))
    entropy_surrogate = jnp.mean(jnp.asarray(output["prefix_entropy_surrogate"], dtype=jnp.float32))
    total_loss = (policy_loss + config.value_coefficient * value_loss
                  - config.entropy_coefficient * entropy_surrogate)
    metrics = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_surrogate": entropy_surrogate,
        "kl": jnp.mean(old_joint - new_joint),
        "clip_fraction": jnp.mean((jnp.abs(ratio_metrics["ratio"] - 1.0) > config.clip_epsilon).astype(jnp.float32)),
        "value_prediction": jnp.mean(new_value),
        "old_value_prediction": jnp.mean(old_values),
        "ratio_mean": jnp.mean(ratio_metrics["ratio"]),
        "component_logprobs": jnp.asarray(output["component_logprobs"]),
        "new_joint_logprobs": new_joint,
        "new_values": new_value,
        "valid": jnp.asarray(output["valid"]),
    }
    return total_loss, {**metrics, **ratio_metrics}, output


def evaluate_stage25_ppo(
        params: Mapping[str, Any], batch: Stage25PPOBatch,
        config: Stage25PPOConfig) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Evaluate one logical batch through fixed physical shapes."""
    output = _map_policy_eval(params, batch, config)
    loss, metrics, _ = _objective_from_output(output, batch, config)
    del loss
    return output, {name: np.asarray(value) for name, value in metrics.items()}


def audit_stage25_ppo_rollout(
        params: Mapping[str, Any], batch: Stage25PPOBatch,
        config: Stage25PPOConfig) -> dict[str, Any]:
    """Fail before updating if unchanged weights do not reproduce rollout data."""
    output, metrics = evaluate_stage25_ppo(params, batch, config)
    classes = np.asarray(output["classes"])
    if not np.array_equal(classes, batch.classes):
        raise ValueError("PPO unchanged-weight audit changed stored classes")
    if not np.all(np.asarray(output["valid"], dtype=bool)):
        raise ValueError("PPO unchanged-weight audit produced invalid diagnostic rows")
    errors = {
        "component_logprobs": float(np.max(np.abs(np.asarray(output["component_logprobs"]) - batch.old_component_logprobs))),
        "joint_logprob": float(np.max(np.abs(np.asarray(output["joint_logprob"]) - batch.old_joint_logprobs))),
        "value": float(np.max(np.abs(np.asarray(output["value"]) - batch.old_values))),
    }
    for name, error in errors.items():
        expected = {"component_logprobs": batch.old_component_logprobs,
                    "joint_logprob": batch.old_joint_logprobs,
                    "value": batch.old_values}[name]
        actual = {"component_logprobs": output["component_logprobs"],
                  "joint_logprob": output["joint_logprob"],
                  "value": output["value"]}[name]
        if not np.allclose(np.asarray(actual), expected, atol=config.audit_atol, rtol=config.audit_rtol, equal_nan=False):
            raise ValueError(f"PPO unchanged-weight audit mismatch in {name}: max error {error}")
    if not np.isfinite(np.asarray(metrics["kl"])).all():
        raise ValueError("PPO unchanged-weight audit produced nonfinite KL")
    return {"ok": True, "max_abs_error": errors,
            "initial_ratio_mean": float(np.asarray(metrics["ratio_mean"])),
            "initial_kl": float(np.asarray(metrics["kl"]))}


def make_stage25_ppo_optimizer(params: Mapping[str, Any], config: Stage25PPOConfig):
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(config.learning_rate, b1=0.9, b2=0.999,
                    eps=1e-8, weight_decay=config.weight_decay),
    )


def _normal_key(value: Any) -> jax.Array:
    key = jnp.asarray(value, dtype=jnp.uint32)
    if key.shape != (2,):
        raise ValueError("rng must have shape [2]")
    return key


@dataclass(frozen=True)
class Stage25PPOTrainState:
    params: Mapping[str, Any]
    optimizer_state: Any
    rng: jax.Array
    update_counter: int = 0
    rollout_seed: int = 0
    rollout_progression: Mapping[str, Any] = None
    behavior_identity: Stage25BehaviorIdentity | None = None

    def __post_init__(self) -> None:
        if isinstance(self.update_counter, bool) or self.update_counter < 0:
            raise ValueError("update_counter must be a nonnegative integer")
        _normal_key(self.rng)


def init_stage25_ppo_state(
        config: Stage25PPOConfig, *, seed: int = 0,
        params: Mapping[str, Any] | None = None,
        behavior_identity: Stage25BehaviorIdentity | None = None,
) -> Stage25PPOTrainState:
    """Initialize PPO with a fresh optimizer, optionally from native BC/E params."""
    if params is None:
        params = init_stage25_params(config.model, seed=seed)
    optimizer = make_stage25_ppo_optimizer(params, config)
    return Stage25PPOTrainState(
        params=params, optimizer_state=optimizer.init(params),
        rng=jax.random.PRNGKey(seed), rollout_seed=seed,
        rollout_progression={"completed_rollouts": 0},
        behavior_identity=behavior_identity)


def _validate_update_output(output: Mapping[str, Any], batch: Stage25PPOBatch) -> None:
    if not np.all(np.asarray(output["valid"], dtype=bool)):
        raise ValueError("PPO objective produced invalid policy diagnostics")
    for name in ("component_logprobs", "joint_logprob", "value", "prefix_entropy_surrogate"):
        if not np.all(np.isfinite(np.asarray(output[name]))):
            raise ValueError(f"PPO objective produced nonfinite {name}")
    if not np.array_equal(np.asarray(output["classes"]), batch.classes):
        raise ValueError("PPO objective did not preserve teacher-forced classes")


def ppo_update(
        state: Stage25PPOTrainState, batch: Stage25PPOBatch,
        config: Stage25PPOConfig) -> tuple[Stage25PPOTrainState, dict[str, Any]]:
    """Run all PPO epochs on one frozen rollout, atomically on validation failure."""
    if batch.behavior_identity is not None and state.behavior_identity is not None and not _identity_matches(batch.behavior_identity, state.behavior_identity):
        raise ValueError("PPO batch behavior identity does not match the frozen state identity")
    audit = audit_stage25_ppo_rollout(state.params, batch, config)
    optimizer = make_stage25_ppo_optimizer(state.params, config)
    params = state.params
    opt_state = state.optimizer_state
    key = _normal_key(state.rng)
    reports: list[dict[str, float]] = []
    n = len(batch.classes)
    for epoch in range(config.epochs):
        epoch_key = jax.random.fold_in(key, epoch)
        order = np.asarray(jax.random.permutation(epoch_key, n), dtype=np.int64)
        for start in range(0, n, config.minibatch_size):
            minibatch = batch.take(order[start:start + config.minibatch_size])
            def objective(tree):
                output = _map_policy_eval(tree, minibatch, config)
                objective_loss, objective_metrics, objective_output = _objective_from_output(output, minibatch, config)
                return objective_loss, (objective_metrics, objective_output)
            (loss, (metrics, output)), grads = jax.value_and_grad(objective, has_aux=True)(params)
            _validate_update_output(output, minibatch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            reports.append({
                "loss": float(np.asarray(loss)),
                "policy_loss": float(np.asarray(metrics["policy_loss"])),
                "value_loss": float(np.asarray(metrics["value_loss"])),
                "entropy_surrogate": float(np.asarray(metrics["entropy_surrogate"])),
                "kl": float(np.asarray(metrics["kl"])),
                "clip_fraction": float(np.asarray(metrics["clip_fraction"])),
                "gradient_norm": float(np.asarray(grad_norm)),
            })
    next_identity = state.behavior_identity
    if next_identity is not None:
        from rl_manager.stage25_inference import parameter_fingerprint
        next_identity = replace(next_identity, parameter_fingerprint=parameter_fingerprint(params))
    next_state = replace(
        state, params=params, optimizer_state=opt_state,
        rng=jax.random.fold_in(key, state.update_counter + 1),
        update_counter=state.update_counter + 1,
        rollout_progression={"completed_rollouts": state.update_counter + 1},
        behavior_identity=next_identity)
    summary = {name: float(np.mean([report[name] for report in reports]))
               for name in reports[0]} if reports else {}
    summary.update({"rollout_rows": n, "epochs": config.epochs,
                    "unchanged_weight_audit": audit})
    return next_state, summary


update_stage25_ppo = ppo_update


class Stage25PPOTrainer:
    """Small stateful facade used by callers that own a training loop."""

    def __init__(self, state: Stage25PPOTrainState, config: Stage25PPOConfig) -> None:
        self.state = state
        self.config = config

    def update(self, batch: Stage25PPOBatch) -> dict[str, Any]:
        self.state, metrics = ppo_update(self.state, batch, self.config)
        return metrics


Stage25PPOState = Stage25PPOTrainState
PPOConfig = Stage25PPOConfig
PPOBatch = Stage25PPOBatch
init_train_state = init_stage25_ppo_state
