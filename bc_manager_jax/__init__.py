"""bc_manager_jax: pure-JAX mirror of the daily-manager BC Transformer.

Stage-1 surface: eval-exact forward, exact manager loss, strict
PyTorch->JAX checkpoint conversion, and native save/load. No train step,
sharding, or serving here (stage 2+).
"""

from bc_manager_jax.checkpoint import (
    NATIVE_CHECKPOINT_FORMAT,
    TORCH_CHECKPOINT_FORMAT,
    convert_torch_state_dict,
    expected_torch_state_shapes,
    load_native,
    load_torch_checkpoint,
    save_native,
)
from bc_manager_jax.loss import GROUP_NAMES, manager_loss
from bc_manager_jax.model import (
    ManagerConfig,
    empty_params,
    forward,
    init_params,
    predict_counts,
    predict_land,
    predict_sells,
    tiny_manager_config,
    validate_inputs,
)

__all__ = [
    "NATIVE_CHECKPOINT_FORMAT",
    "TORCH_CHECKPOINT_FORMAT",
    "GROUP_NAMES",
    "ManagerConfig",
    "convert_torch_state_dict",
    "empty_params",
    "expected_torch_state_shapes",
    "forward",
    "init_params",
    "load_native",
    "load_torch_checkpoint",
    "manager_loss",
    "predict_counts",
    "predict_land",
    "predict_sells",
    "save_native",
    "tiny_manager_config",
    "validate_inputs",
]
