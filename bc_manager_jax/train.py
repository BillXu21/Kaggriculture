"""AdamW + global-clip BC train step for the JAX daily-manager model.

Existing-semantics training (mirrors `bc_manager.training.TrainingConfig`
defaults): AdamW over ALL parameters with lr 3e-4, weight decay 1e-2
(decoupled, no exclusions), betas 0.9/0.999, eps 1e-8, and global L2
gradient clipping at 1.0 applied BEFORE the AdamW update via
`optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(...))`.
No scheduler. Dropout uses an explicit PRNG key; the step returns the
advanced key so consecutive steps see fresh dropout masks.

Data parallelism: the SAME compiled step runs on one device or N devices.
Batches are sharded along axis 0 with `NamedSharding(mesh, ('data', ...))`;
params/opt state are replicated (`PartitionSpec()`). All loss reductions in
`bc_manager_jax.loss` are full-array `jnp.mean`s over the global batch, so
GSPMD inserts the correct cross-replica reductions automatically — this is
verified numerically by the forced multi-CPU subprocess test
(`tests/test_bc_manager_jax_train.py`). Global batches must be divisible by
the device count, which keeps every per-device shard equal and makes the
mean-of-shards exactly the global mean.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bc_manager.loss import GROUP_NAMES
from bc_manager_jax.loss import (
    loss_from_validated,
    validate_target_shapes,
)
from bc_manager_jax.model import (
    ManagerConfig,
    _Dropout,
    _forward_core,
    _prepare_inputs,
    resolve_model_variant,
    validate_inputs,
)


@dataclass(frozen=True)
class TrainConfig:
    """Fixed training configuration matching `bc_manager.training` defaults."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.gradient_clip <= 0.0:
            raise ValueError(
                f"gradient_clip must be positive, got {self.gradient_clip}")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("betas must be in [0, 1)")


def make_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """clip_by_global_norm BEFORE adamw — exact existing semantics."""
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(learning_rate=config.lr, b1=config.beta1,
                    b2=config.beta2, eps=config.eps,
                    weight_decay=config.weight_decay),
    )


def init_opt_state(params: Mapping, config: TrainConfig):
    return make_optimizer(config).init(params)


def _prepare_targets(validated: Mapping[str, np.ndarray]) -> dict[str, jax.Array]:
    prepared = {}
    for key, value in validated.items():
        dtype = jnp.float32 if key in ("presence", "quantity") else jnp.int32
        prepared[key] = jnp.asarray(value, dtype=dtype)
    return prepared


@functools.lru_cache(maxsize=None)
def _compiled_train_step(model_config: ManagerConfig,
                         train_config: TrainConfig,
                         model_variant: str = "V0"):
    optimizer = make_optimizer(train_config)

    def core(params, opt_state, drop_rng, inputs, targets):
        def loss_fn(p):
            outputs = _forward_core(
                p, inputs, model_config,
                _Dropout(model_config.dropout, drop_rng), model_variant)
            return loss_from_validated(outputs, targets)

        (total, groups), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        next_rng = jax.random.split(drop_rng)[1]
        return new_params, new_opt_state, next_rng, total, groups

    return jax.jit(core)


def train_step(params, opt_state, rng: jax.Array,
               inputs: Mapping[str, object], targets: Mapping[str, object],
               model_config: ManagerConfig,
               train_config: TrainConfig = TrainConfig(),
               model_variant: str = "V0"):
    """One BC optimization step; returns
    (new_params, new_opt_state, next_rng, total_loss, group_losses).

    Inputs/targets are validated eagerly outside jit. `rng` must be a
    PRNGKey; it is split so dropout masks never repeat across steps.
    Variant E requires `economic_context` in `inputs`; V0 rejects it.
    """
    variant = resolve_model_variant(model_variant)
    validate_inputs(inputs, model_config, variant)
    batch = int(np.asarray(inputs["board_kind"]).shape[0])
    validated = validate_target_shapes(targets, batch,
                                       model_config.count_classes)
    prepared_inputs = _prepare_inputs(inputs)
    prepared_targets = _prepare_targets(validated)
    core = _compiled_train_step(model_config, train_config, variant)
    new_params, new_opt_state, next_rng, total, groups = core(
        params, opt_state, rng, prepared_inputs, prepared_targets)
    return new_params, new_opt_state, next_rng, total, groups


def loss_and_groups(params, inputs: Mapping[str, object],
                    targets: Mapping[str, object],
                    model_config: ManagerConfig,
                    rng: jax.Array | None = None,
                    model_variant: str = "V0"):
    """Un-jitted helper computing (total, groups) for tests/diagnostics.

    With `rng=None` and zero dropout this equals the eval forward loss.
    """
    variant = resolve_model_variant(model_variant)
    validate_inputs(inputs, model_config, variant)
    batch = int(np.asarray(inputs["board_kind"]).shape[0])
    validated = validate_target_shapes(targets, batch,
                                       model_config.count_classes)
    outputs = _forward_core(
        params, _prepare_inputs(inputs), model_config,
        _Dropout(model_config.dropout, rng), variant)
    return loss_from_validated(outputs, validated)
