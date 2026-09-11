"""bc_manager_jax: pure-JAX mirror of the daily-manager BC Transformer.

Stage-1 surface: eval-exact forward, exact manager loss, strict
PyTorch->JAX checkpoint conversion, and native save/load.
Stage-2 additions: AdamW+global-clip train step (`train`), replicated
data-parallel helpers (`sharding`), and the throughput benchmark CLI
(`benchmark`, runnable as `python -m bc_manager_jax.benchmark`).
Issue #8 additions: model variants V0 and E (the promoted economic-context
variant); J/JE are deliberately unsupported.
Issue #9 addition: `manager_representation` / `forward_with_representation`
expose the final manager token for RL value heads without duplicating the
Transformer; forward outputs/numerics are unchanged.

The model surface is imported eagerly, but checkpoint/loss/training helpers
are resolved lazily. This keeps native model imports Torch-free while
preserving the established package-level API.
"""

from importlib import import_module

from bc_manager_jax.model import (
    SUPPORTED_MODEL_VARIANTS,
    ManagerConfig,
    empty_params,
    forward,
    forward_with_representation,
    init_params,
    init_train_params,
    manager_representation,
    predict_counts,
    predict_land,
    predict_sells,
    resolve_model_variant,
    large_manager_config,
    tiny_manager_config,
    validate_inputs,
)

_LAZY_EXPORTS = {
    "NATIVE_CHECKPOINT_FORMAT": ("bc_manager_jax.checkpoint", "NATIVE_CHECKPOINT_FORMAT"),
    "TORCH_CHECKPOINT_FORMAT": ("bc_manager_jax.checkpoint", "TORCH_CHECKPOINT_FORMAT"),
    "convert_torch_state_dict": ("bc_manager_jax.checkpoint", "convert_torch_state_dict"),
    "expected_torch_state_shapes": ("bc_manager_jax.checkpoint", "expected_torch_state_shapes"),
    "load_native": ("bc_manager_jax.checkpoint", "load_native"),
    "load_torch_checkpoint": ("bc_manager_jax.checkpoint", "load_torch_checkpoint"),
    "save_native": ("bc_manager_jax.checkpoint", "save_native"),
    "GROUP_NAMES": ("bc_manager_jax.loss", "GROUP_NAMES"),
    "loss_from_validated": ("bc_manager_jax.loss", "loss_from_validated"),
    "manager_loss": ("bc_manager_jax.loss", "manager_loss"),
    "validate_target_shapes": ("bc_manager_jax.loss", "validate_target_shapes"),
    "create_data_mesh": ("bc_manager_jax.sharding", "create_data_mesh"),
    "shard_batch": ("bc_manager_jax.sharding", "shard_batch"),
    "TrainConfig": ("bc_manager_jax.train", "TrainConfig"),
    "init_opt_state": ("bc_manager_jax.train", "init_opt_state"),
    "make_optimizer": ("bc_manager_jax.train", "make_optimizer"),
}


def __getattr__(name: str):
    """Load non-model helpers only when their package API is requested."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "NATIVE_CHECKPOINT_FORMAT",
    "TORCH_CHECKPOINT_FORMAT",
    "GROUP_NAMES",
    "SUPPORTED_MODEL_VARIANTS",
    "ManagerConfig",
    "TrainConfig",
    "convert_torch_state_dict",
    "create_data_mesh",
    "empty_params",
    "expected_torch_state_shapes",
    "forward",
    "forward_with_representation",
    "large_manager_config",
    "init_opt_state",
    "init_params",
    "init_train_params",
    "load_native",
    "load_torch_checkpoint",
    "loss_from_validated",
    "make_optimizer",
    "manager_loss",
    "manager_representation",
    "predict_counts",
    "predict_land",
    "predict_sells",
    "resolve_model_variant",
    "save_native",
    "shard_batch",
    "tiny_manager_config",
    "validate_inputs",
    "validate_target_shapes",
]
