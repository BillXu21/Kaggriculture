"""bc_manager_jax: pure-JAX mirror of the daily-manager BC Transformer.

Stage-1 surface: eval-exact forward, exact manager loss, strict
PyTorch->JAX checkpoint conversion, and native save/load.
Stage-2 additions: AdamW+global-clip train step (`train`), replicated
data-parallel helpers (`sharding`), and the throughput benchmark CLI
(`benchmark`, runnable as `python -m bc_manager_jax.benchmark`).
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
from bc_manager_jax.loss import (
    GROUP_NAMES,
    loss_from_validated,
    manager_loss,
    validate_target_shapes,
)
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
from bc_manager_jax.sharding import create_data_mesh, shard_batch
from bc_manager_jax.train import TrainConfig, init_opt_state, make_optimizer

__all__ = [
    "NATIVE_CHECKPOINT_FORMAT",
    "TORCH_CHECKPOINT_FORMAT",
    "GROUP_NAMES",
    "ManagerConfig",
    "TrainConfig",
    "convert_torch_state_dict",
    "create_data_mesh",
    "empty_params",
    "expected_torch_state_shapes",
    "forward",
    "init_opt_state",
    "init_params",
    "load_native",
    "load_torch_checkpoint",
    "loss_from_validated",
    "make_optimizer",
    "manager_loss",
    "predict_counts",
    "predict_land",
    "predict_sells",
    "save_native",
    "shard_batch",
    "tiny_manager_config",
    "validate_inputs",
    "validate_target_shapes",
]
