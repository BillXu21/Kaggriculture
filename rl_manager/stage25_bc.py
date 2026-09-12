"""Minimal native JAX/Optax teacher-forced BC for Stage 2.5.

Representation, recurrence, and physical support remain authoritative in
``rl_manager.stage25_policy``.  This module owns only fixed-shape batching,
masked likelihood reductions, the optimizer mask, and native training state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
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
    evaluate_actions,
    init_stage25_params,
)


ACTION_COUNT = len(ACTION_CLASS_COUNTS)
BC_CHECKPOINT_FORMAT = "stage25_bc_training_state_v1"


@dataclass(frozen=True)
class Stage25BCConfig:
    """Optimization settings for the small native BC runner."""

    model: Stage25ModelConfig = Stage25ModelConfig.tiny()
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    gradient_clip: float = 1.0
    dropout: float | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0 or not np.isfinite(self.lr):
            raise ValueError("lr must be finite and positive")
        if self.weight_decay < 0.0 or not np.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("betas must lie in [0, 1)")
        if self.eps <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("eps and gradient_clip must be positive")
        if self.dropout is not None and not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    @property
    def train_dropout(self) -> float:
        return self.model.dropout if self.dropout is None else self.dropout


@dataclass(frozen=True)
class Stage25BCBatch:
    """A fixed-shape batch and its real-row mask."""

    inputs: Mapping[str, Any]
    actions: np.ndarray
    real_row_mask: np.ndarray
    physical_contexts: Sequence[Any] | None = None
    row_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        actions = _validate_actions(self.actions, check_range=False)
        mask = np.asarray(self.real_row_mask, dtype=bool)
        if mask.shape != (actions.shape[0],):
            raise ValueError("real_row_mask must have shape [B]")
        if not mask.any():
            raise ValueError("a batch must contain at least one real row")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "real_row_mask", mask)
        if self.row_ids is not None:
            ids = np.asarray(self.row_ids, dtype=np.int64)
            if ids.shape != mask.shape:
                raise ValueError("row_ids must have shape [B]")
            object.__setattr__(self, "row_ids", ids)
        if self.physical_contexts is not None and len(self.physical_contexts) != len(mask):
            raise ValueError("physical_contexts must contain one item per row")


def _validate_actions(actions: Any, *, check_range: bool = True) -> np.ndarray:
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != ACTION_COUNT:
        raise ValueError(f"teacher-forced actions must have shape [N, {ACTION_COUNT}]")
    if not np.issubdtype(array.dtype, np.integer):
        if (not np.issubdtype(array.dtype, np.floating)
                or not np.all(np.isfinite(array))
                or not np.all(array == np.floor(array))):
            raise ValueError("teacher-forced actions must contain integer class indices")
    array = array.astype(np.int32, copy=False)
    counts = np.asarray(ACTION_CLASS_COUNTS, dtype=np.int32)
    if check_range and (np.any(array < 0) or np.any(array >= counts[None, :])):
        raise ValueError("teacher-forced actions contain a class outside its vocabulary")
    return np.array(array, dtype=np.int32, copy=True)


def _row_count(inputs: Mapping[str, Any]) -> int:
    if not isinstance(inputs, Mapping) or "board_kind" not in inputs:
        raise ValueError("inputs must be a mapping containing board_kind")
    value = np.asarray(inputs["board_kind"])
    if value.ndim < 1:
        raise ValueError("inputs arrays must have a leading row dimension")
    return int(value.shape[0])


def _take_rows(value: Any, indices: np.ndarray, n: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _take_rows(child, indices, n) for key, child in value.items()}
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != n:
        raise ValueError("all input arrays must share the dataset row count")
    return np.array(array[indices], copy=True)


def _normalize_row_ids(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.arange(n, dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (n,):
        raise ValueError("row_ids must match dataset row count")
    try:
        return raw.astype(np.int64)
    except (TypeError, ValueError):
        # Adapter row identities are stable strings.  Training only needs a
        # numeric key for JAX's row-index seam, so preserve order deterministically.
        return np.arange(n, dtype=np.int64)


def make_fixed_batch(
        inputs: Mapping[str, Any], actions: Any, batch_size: int, *,
        start: int = 0, stop: int | None = None,
        physical_contexts: Sequence[Any] | None = None,
        row_ids: Any = None,
) -> Stage25BCBatch:
    """Make one fixed-size batch, repeating the final example if needed."""
    n = _row_count(inputs)
    labels = _validate_actions(actions, check_range=False)
    if labels.shape[0] != n:
        raise ValueError("inputs and actions must contain the same row count")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    begin = int(start)
    end = n if stop is None else int(stop)
    if not 0 <= begin < end <= n or end - begin > batch_size:
        raise ValueError("batch range is outside the fixed batch size")
    source = np.arange(begin, end, dtype=np.int64)
    real = source.size
    if real < batch_size:
        source = np.pad(source, (0, batch_size - real), mode="edge")
    mask = np.zeros((batch_size,), dtype=bool)
    mask[:real] = True
    contexts = None
    if physical_contexts is not None:
        if len(physical_contexts) != n:
            raise ValueError("physical_contexts must match dataset row count")
        contexts = tuple(physical_contexts[index] for index in source)
    ids = _normalize_row_ids(row_ids, n)
    return Stage25BCBatch(_take_rows(inputs, source, n), labels[source], mask,
                          contexts, ids[source])


def iter_fixed_batches(
        inputs: Mapping[str, Any], actions: Any, batch_size: int, *,
        seed: int = 0, epoch: int = 0, shuffle: bool = True,
        start_batch: int = 0,
        physical_contexts: Sequence[Any] | None = None,
        row_ids: Any = None,
) -> Iterable[Stage25BCBatch]:
    """Yield deterministic fixed-shape batches for one epoch."""
    n = _row_count(inputs)
    labels = _validate_actions(actions, check_range=False)
    if labels.shape[0] != n or n == 0:
        raise ValueError("inputs/actions must be nonempty and row-aligned")
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
        rng.shuffle(order)
    shuffled_inputs = _take_rows(inputs, order, n)
    shuffled_labels = labels[order]
    shuffled_contexts = (None if physical_contexts is None
                         else tuple(physical_contexts[index] for index in order))
    shuffled_ids = _normalize_row_ids(row_ids, n)[order]
    if start_batch < 0:
        raise ValueError("start_batch must be nonnegative")
    for batch_index, begin in enumerate(range(0, n, batch_size)):
        if batch_index < start_batch:
            continue
        yield make_fixed_batch(shuffled_inputs, shuffled_labels, batch_size,
                               start=begin, stop=min(begin + batch_size, n),
                               physical_contexts=shuffled_contexts,
                               row_ids=shuffled_ids)


def _model_config(config: Stage25BCConfig | Stage25ModelConfig) -> Stage25ModelConfig:
    return config.model if isinstance(config, Stage25BCConfig) else config


def _settings(config: Stage25BCConfig | Stage25ModelConfig) -> Stage25BCConfig:
    return config if isinstance(config, Stage25BCConfig) else Stage25BCConfig(model=config, batch_size=1)


def _normal_key(rng: Any) -> jax.Array:
    if rng is None:
        return jax.random.PRNGKey(0)
    if np.isscalar(rng):
        return jax.random.PRNGKey(int(rng))
    key = jnp.asarray(rng, dtype=jnp.uint32)
    if key.shape != (2,):
        raise ValueError("rng must be a JAX PRNGKey with shape [2]")
    return key


def _default_context_values(batch: int) -> tuple[jax.Array, ...]:
    return tuple(jnp.zeros((batch,) if index in (0, 3, 4)
                           else (batch, 3) if index in (2, 5)
                           else (batch, 4,), dtype=jnp.int32)
                 for index in range(6))


def _core_inputs(batch: Stage25BCBatch, model: Stage25ModelConfig):
    prepared, capacity, size = _host_inputs(batch.inputs, model)
    contexts = _host_contexts(batch.physical_contexts, size)
    explicit = contexts is not None
    if contexts is None:
        contexts = _default_context_values(size)
    row_ids = (jnp.zeros((size,), dtype=jnp.int32) if batch.row_ids is None
               else jnp.asarray(batch.row_ids, dtype=jnp.int32))
    return prepared, capacity, contexts, row_ids, explicit


def _validated_output(params: Mapping[str, Any], batch: Stage25BCBatch,
                     model: Stage25ModelConfig) -> Mapping[str, Any]:
    # Keep this host call before optimizer work: invalid chains fail loudly.
    return evaluate_actions(
        params, batch.inputs, model, actions=batch.actions,
        physical_contexts=batch.physical_contexts, row_ids=batch.row_ids,
        reject_invalid=True)


def _training_output(params: Mapping[str, Any], batch: Stage25BCBatch,
                     model: Stage25ModelConfig, rng: Any) -> Mapping[str, Any]:
    """Run the shared policy core in its explicit training/dropout mode."""
    prepared, capacity, contexts, row_ids, explicit = _core_inputs(batch, model)
    key = _normal_key(rng)
    keys = jnp.broadcast_to(key, (batch.actions.shape[0], 2))
    return _stage25_jit(
        params, prepared, capacity, keys, jnp.asarray(batch.actions, jnp.int32),
        contexts, row_ids, model, "train", explicit)


def _metrics_from_output(output: Mapping[str, Any], actions: jax.Array,
                         real_mask: jax.Array, *, rng: jax.Array,
                         dropout: float, training: bool) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = jnp.asarray(output["logits"], dtype=jnp.float32)
    support = jnp.asarray(output["masks"], dtype=bool)
    safe_logits = jnp.where(support, logits, -1.0e30)
    log_norm = jax.nn.logsumexp(safe_logits, axis=-1)
    selected = jnp.take_along_axis(safe_logits,
                                   actions[..., None], axis=-1)[..., 0]
    component_nll = log_norm - selected
    predictions = jnp.argmax(safe_logits, axis=-1)
    row_mask = real_mask.astype(jnp.float32)
    denominator = jnp.maximum(jnp.sum(row_mask), 1.0)
    joint_rows = jnp.sum(component_nll, axis=1)
    loss = jnp.sum(joint_rows * row_mask) / denominator
    per_step_nll = jnp.sum(component_nll * row_mask[:, None], axis=0) / denominator
    per_step_accuracy = jnp.sum(
        (predictions == actions).astype(jnp.float32) * row_mask[:, None],
        axis=0) / denominator
    metrics = {
        "loss": loss,
        "joint_nll": loss,
        "nll": loss,
        "per_step_nll": per_step_nll,
        "per_step_accuracy": per_step_accuracy,
        "step_nll": per_step_nll,
        "step_accuracy": per_step_accuracy,
        "valid_rows": jnp.sum(row_mask),
        "validity": jnp.asarray(output["valid"]),
    }
    return loss, metrics


def loss_and_metrics(
        params: Mapping[str, Any], batch: Stage25BCBatch,
        config: Stage25BCConfig | Stage25ModelConfig, *,
        training: bool = False, rng: Any = None,
) -> dict[str, Any]:
    """Return masked joint NLL and per-step diagnostics for a fixed batch."""
    settings = _settings(config)
    model = _model_config(config)
    if training and settings.train_dropout and rng is None:
        raise ValueError("training with nonzero dropout requires an explicit rng")
    output = _validated_output(params, batch, model)
    if training and settings.train_dropout:
        output = _training_output(params, batch, model, rng)
    loss, metrics = _metrics_from_output(
        output, jnp.asarray(batch.actions, jnp.int32),
        jnp.asarray(batch.real_row_mask), rng=_normal_key(rng),
        dropout=settings.train_dropout, training=training)
    del loss
    return {name: np.asarray(value) if name != "loss" else value
            for name, value in metrics.items()}


def _trainable_mask(params: Mapping[str, Any]) -> Mapping[str, Any]:
    def include(path: tuple[Any, ...], _: Any) -> bool:
        return not any(getattr(part, "key", None) == "value_head"
                       for part in path)
    return jax.tree_util.tree_map_with_path(include, params)


def make_optimizer(params: Mapping[str, Any], config: Stage25BCConfig | Stage25ModelConfig):
    settings = _settings(config)
    return optax.chain(
        optax.clip_by_global_norm(settings.gradient_clip),
        optax.masked(optax.adamw(
            learning_rate=settings.lr, b1=settings.beta1, b2=settings.beta2,
            eps=settings.eps, weight_decay=settings.weight_decay),
            _trainable_mask(params)),
    )


def init_opt_state(params: Mapping[str, Any],
                   config: Stage25BCConfig | Stage25ModelConfig):
    return make_optimizer(params, config).init(params)


def train_step(
        params: Mapping[str, Any], opt_state: Any, rng: Any,
        batch: Stage25BCBatch,
        config: Stage25BCConfig | Stage25ModelConfig,
) -> tuple[dict[str, Any], Any, jax.Array, dict[str, Any]]:
    """Perform one masked native BC update."""
    settings = _settings(config)
    model = _model_config(config)
    if settings.train_dropout and rng is None:
        raise ValueError("training with nonzero dropout requires an explicit rng")
    key = _normal_key(rng)
    _validated_output(params, batch, model)
    prepared, capacity, contexts, row_ids, explicit = _core_inputs(batch, model)
    labels = jnp.asarray(batch.actions, dtype=jnp.int32)
    real_mask = jnp.asarray(batch.real_row_mask)
    optimizer = make_optimizer(params, settings)

    def objective(tree):
        train_keys = jnp.broadcast_to(key, (labels.shape[0], 2))
        output = _stage25_jit(
            tree, prepared, capacity, train_keys, labels,
            contexts, row_ids, model,
            "train" if settings.train_dropout else "eval", explicit)
        return _metrics_from_output(
            output, labels, real_mask, rng=key,
            dropout=settings.train_dropout, training=True)

    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    next_rng = jax.random.split(key)[1]
    reported = {name: np.asarray(value) for name, value in metrics.items()}
    reported["loss"] = float(np.asarray(loss))
    reported["step_loss"] = reported["loss"]
    return next_params, next_opt_state, next_rng, reported


def _path_part(part: Any) -> str:
    key = getattr(part, "key", None)
    if key is not None:
        return str(key)
    index = getattr(part, "idx", None)
    if index is not None:
        return str(index)
    return str(getattr(part, "name", part))


def _flatten(tree: Any) -> tuple[dict[str, np.ndarray], Any]:
    pairs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    result = {}
    for path, value in pairs:
        name = "/".join(_path_part(part) for part in path)
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"cannot serialize object leaf {name!r}")
        result[name] = array
    return result, treedef


def _unflatten(items: Mapping[str, np.ndarray], template: Any,
                *, prefix: str) -> Any:
    pairs, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = []
    expected = set()
    for path, value in pairs:
        name = "/".join(_path_part(part) for part in path)
        key = f"{prefix}:{name}"
        expected.add(key)
        if key not in items:
            raise ValueError(f"checkpoint missing {key!r}")
        array = np.asarray(items[key])
        expected_array = np.asarray(value)
        if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
            raise ValueError(f"checkpoint {key!r} shape/dtype mismatch")
        leaves.append(jnp.asarray(array))
    actual = {key for key in items if key.startswith(prefix + ":")}
    if actual != expected:
        raise ValueError(f"checkpoint {prefix} tree mismatch")
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in fields(value) if field.name != "manager_config"}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _config_json(config: Stage25BCConfig | Stage25ModelConfig) -> dict[str, Any]:
    settings = _settings(config)
    return {"model": _jsonable(settings.model),
            "batch_size": settings.batch_size, "lr": settings.lr,
            "weight_decay": settings.weight_decay, "beta1": settings.beta1,
            "beta2": settings.beta2, "eps": settings.eps,
            "gradient_clip": settings.gradient_clip, "dropout": settings.dropout}


def save_checkpoint(
        path: str | Path, params: Mapping[str, Any], opt_state: Any,
        rng: Any, *, config: Stage25BCConfig | Stage25ModelConfig,
        step: int = 0, epoch: int = 0, seed: int = 0,
        shuffle_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write params, Optax state, explicit RNG, and resume metadata to NPZ."""
    try:
        from rl_manager.stage25_checkpoint import save_stage25_bc_checkpoint
    except ImportError:  # pragma: no cover - compatibility with the base tree
        save_stage25_bc_checkpoint = None
    if save_stage25_bc_checkpoint is not None:
        order = dict(shuffle_state or {})
        order.setdefault("seed", int(seed))
        save_stage25_bc_checkpoint(
            path, params, opt_state, _normal_key(rng), _model_config(config),
            seed=int(seed), step=int(step), epoch=int(epoch),
            optimizer_config=_settings(config),
            data_order_position=order, metadata=metadata)
        return
    param_items, _ = _flatten(params)
    opt_items, _ = _flatten(opt_state)
    items = {f"param:{name}": value for name, value in param_items.items()}
    items.update({f"opt:{name}": value for name, value in opt_items.items()})
    items["rng"] = np.asarray(_normal_key(rng), dtype=np.uint32)
    meta = {"format": BC_CHECKPOINT_FORMAT, "config": _config_json(config),
            "step": int(step), "epoch": int(epoch),
            "shuffle_state": dict(shuffle_state or {}),
            "metadata": dict(metadata or {})}
    items["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **items)
    destination.unlink(missing_ok=True)
    temporary.rename(destination)


def load_checkpoint(
        path: str | Path, *, config: Stage25BCConfig | Stage25ModelConfig,
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
) -> tuple[dict[str, Any], Any, jax.Array, dict[str, Any]]:
    """Load and strictly validate a native BC training-state archive."""
    settings = _settings(config)
    model = _model_config(config)
    try:
        from rl_manager.stage25_checkpoint import load_stage25_bc_checkpoint
    except ImportError:  # pragma: no cover - compatibility with the base tree
        load_stage25_bc_checkpoint = None
    if load_stage25_bc_checkpoint is not None:
        template = init_stage25_params(model, seed=0) if params is None else params
        optimizer_template = init_opt_state(template, settings)
        return load_stage25_bc_checkpoint(
            path, config=model, optimizer_state_template=optimizer_template,
            optimizer_config=settings, expected_data_order_seed=seed)
    if params is None:
        params = init_stage25_params(model, seed=0)
    with np.load(Path(path), allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    if "__meta__" not in items:
        raise ValueError("checkpoint missing metadata")
    meta = json.loads(items.pop("__meta__").tobytes().decode("utf-8"))
    if meta.get("format") != BC_CHECKPOINT_FORMAT:
        raise ValueError(f"unrecognized BC checkpoint format {meta.get('format')!r}")
    loaded_params = _unflatten(items, params, prefix="param")
    template_state = init_opt_state(params, settings)
    loaded_opt = _unflatten(items, template_state, prefix="opt")
    if "rng" not in items or np.asarray(items["rng"]).shape != (2,):
        raise ValueError("checkpoint RNG is missing or malformed")
    return loaded_params, loaded_opt, jnp.asarray(items["rng"], jnp.uint32), meta


def import_encoder_checkpoint(path: str | Path, model: Stage25ModelConfig,
                              *, seed: int = 0,
                              return_metadata: bool = False) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Import an existing native/Torch E checkpoint without Torch at startup."""
    try:
        from rl_manager.stage25_checkpoint import import_historical_encoder
    except ImportError:  # pragma: no cover - compatibility with the base tree
        import_historical_encoder = None
    if import_historical_encoder is not None:
        encoder, import_meta = import_historical_encoder(path, model)
        params = init_stage25_params(model, seed=seed, encoder_params=encoder)
        return (params, import_meta) if return_metadata else params
    source = Path(path)
    if source.suffix.lower() == ".npz":
        from bc_manager_jax.checkpoint import load_native
        encoder, import_meta = load_native(source, model.manager_config, model_variant="E")
    else:
        from bc_manager_jax.checkpoint import load_torch_checkpoint
        encoder, import_meta = load_torch_checkpoint(source, model.manager_config,
                                                     model_variant="E")
    params = init_stage25_params(model, seed=seed, encoder_params=encoder)
    return (params, import_meta) if return_metadata else params


def load_array_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load a pickle-free NPZ dataset with ``actions`` or ``labels``."""
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    label_key = "actions" if "actions" in arrays else "labels" if "labels" in arrays else None
    if label_key is None:
        raise ValueError("dataset NPZ must contain actions or labels")
    actions = arrays.pop(label_key)
    inputs = {key[6:] if key.startswith("input_") else key: value
              for key, value in arrays.items()}
    return inputs, _validate_actions(actions)


__all__ = [
    "ACTION_COUNT", "ACTION_CLASS_COUNTS", "BC_CHECKPOINT_FORMAT",
    "Stage25BCBatch", "Stage25BCConfig", "init_opt_state", "iter_fixed_batches",
    "load_array_dataset", "load_checkpoint", "make_fixed_batch", "make_optimizer",
    "loss_and_metrics", "save_checkpoint", "train_step", "import_encoder_checkpoint",
]
