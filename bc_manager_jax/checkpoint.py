"""Strict PyTorch -> JAX checkpoint conversion and native JAX save/load.

PyTorch side reads `bc_manager_checkpoint_v1` payloads produced by
`bc_manager.training.save_checkpoint` (path or in-memory payload), on CPU,
with `torch.load(..., weights_only=True)` so no arbitrary pickle execution
is required for the tensor+primitive payloads saved by this repo.

Mapping contract (see `bc_manager_jax.model` module docstring):

- every `nn.Linear` weight [out, in] is transposed to a kernel [in, out];
- packed self-attention `in_proj_weight`/`in_proj_bias` stay PACKED as
  `qkv_kernel`/`qkv_bias` with chunk order q|k|v;
- embeddings, LayerNorm parameters, and head weights/bias map unchanged;
- `manager_token [1, 1, d]` is stored squeezed as `(d,)`.

Strictness: the converter enumerates EVERY expected state-dict key and
shape from the config and rejects missing keys, unexpected keys, shape
mismatches, non-float32 tensors, and incompatible model configs (including
own-only vs include_opponent_board) loudly.

Variant handling (issue #8): the torch payload's top-level `model_variant`
(absent -> "V0") selects the expected shapes; exactly V0 and E are
supported, J/JE are rejected with an explicit unsupported-variant message.
The variant is persisted OUTSIDE `model_config` in the native NPZ metadata
and is checked strictly against any requested variant — never inferred from
weight shapes.

Native format `bc_manager_jax_checkpoint_v1`: one .npz archive with
flattened parameter arrays plus a JSON metadata record. No pickle.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.economics import normalize_model_variant
from bc_manager_jax.model import (
    ECONOMIC_DIM,
    ManagerConfig,
    empty_params,
    resolve_model_variant,
)

TORCH_CHECKPOINT_FORMAT = "bc_manager_checkpoint_v1"
NATIVE_CHECKPOINT_FORMAT = "bc_manager_jax_checkpoint_v1"


# ------------------------------------------------- expected torch state


def expected_torch_state_shapes(config: ManagerConfig,
                                model_variant: str = "V0") -> dict[str, tuple[int, ...]]:
    """Every expected `DailyManagerTransformer.state_dict()` key and shape."""
    variant = resolve_model_variant(model_variant)
    d = config.d_model
    c = config.count_classes
    feature_dim = 5 * d + 11 + 2 + 8 + 4  # embeddings + numeric + NaN ind + bool + mask
    self_resource_dim = 35 + (ECONOMIC_DIM if variant == "E" else 0)
    shapes: dict[str, tuple[int, ...]] = {
        "manager_token": (1, 1, d),
        "role_embedding.weight": (2, d),
        "tile_encoder.kind_embedding.weight": (8, d),
        "tile_encoder.crop_embedding.weight": (7, d),
        "tile_encoder.animal_embedding.weight": (5, d),
        "tile_encoder.row_embedding.weight": (10, d),
        "tile_encoder.col_embedding.weight": (10, d),
        "tile_encoder.project.0.weight": (d, feature_dim),
        "tile_encoder.project.0.bias": (d,),
        "tile_encoder.project.3.weight": (d, d),
        "tile_encoder.project.3.bias": (d,),
        "global_encoders.day_embedding.weight": (30, d),
        "global_encoders.self_resource.0.weight": (d, self_resource_dim),
        "global_encoders.self_resource.0.bias": (d,),
        "global_encoders.self_resource.3.weight": (d, d),
        "global_encoders.self_resource.3.bias": (d,),
        "global_encoders.market.0.weight": (d, 18),
        "global_encoders.market.0.bias": (d,),
        "global_encoders.market.3.weight": (d, d),
        "global_encoders.market.3.bias": (d,),
        "global_encoders.town.0.weight": (d, 9),
        "global_encoders.town.0.bias": (d,),
        "global_encoders.town.3.weight": (d, d),
        "global_encoders.town.3.bias": (d,),
        "global_encoders.labor.0.weight": (d, 3),
        "global_encoders.labor.0.bias": (d,),
        "global_encoders.labor.3.weight": (d, d),
        "global_encoders.labor.3.bias": (d,),
        "global_encoders.day_scalar.weight": (d, 2),
        "global_encoders.day_scalar.bias": (d,),
        "encoder_norm.weight": (d,),
        "encoder_norm.bias": (d,),
        "crop_head.weight": (5 * c, d),
        "crop_head.bias": (5 * c,),
        "animal_head.weight": (3 * c, d),
        "animal_head.bias": (3 * c,),
        "land_head.weight": (4, d),
        "land_head.bias": (4,),
        "fertilizer_head.weight": (5 * c, d),
        "fertilizer_head.bias": (5 * c,),
        "care_head.weight": (3 * c, d),
        "care_head.bias": (3 * c,),
        "sell_presence_head.weight": (54, d),
        "sell_presence_head.bias": (54,),
        "sell_quantity_head.weight": (54, d),
        "sell_quantity_head.bias": (54,),
    }
    for i in range(config.num_layers):
        prefix = f"encoder.layers.{i}"
        shapes.update({
            f"{prefix}.self_attn.in_proj_weight": (3 * d, d),
            f"{prefix}.self_attn.in_proj_bias": (3 * d,),
            f"{prefix}.self_attn.out_proj.weight": (d, d),
            f"{prefix}.self_attn.out_proj.bias": (d,),
            f"{prefix}.linear1.weight": (config.ffn_dim, d),
            f"{prefix}.linear1.bias": (config.ffn_dim,),
            f"{prefix}.linear2.weight": (d, config.ffn_dim),
            f"{prefix}.linear2.bias": (d,),
            f"{prefix}.norm1.weight": (d,),
            f"{prefix}.norm1.bias": (d,),
            f"{prefix}.norm2.weight": (d,),
            f"{prefix}.norm2.bias": (d,),
        })
    return shapes


# ---------------------------------------------------------- conversion


def _check_exact(actual: set[str], expected: set[str], what: str) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"state dict mismatch ({what}): "
            f"missing keys {missing}; unexpected keys {unexpected}")


def convert_torch_state_dict(state_dict: Mapping[str, Any],
                             config: ManagerConfig,
                             model_variant: str = "V0") -> dict:
    """Convert a torch state dict into the JAX params pytree, strictly.

    `model_variant` must be V0 or E; J/JE fail loudly as unsupported. The
    variant is NEVER inferred from weight shapes — the caller (or
    `load_torch_checkpoint`, from the payload metadata) supplies it and the
    expected shapes are checked exactly.
    """
    variant = resolve_model_variant(model_variant)
    expected = expected_torch_state_shapes(config, variant)
    _check_exact(set(state_dict.keys()), set(expected.keys()), "keys")

    def tensor(key: str) -> np.ndarray:
        value = np.asarray(state_dict[key])
        if value.shape != expected[key]:
            raise ValueError(
                f"state dict shape mismatch for {key!r}: got {value.shape}, "
                f"expected {expected[key]}")
        if value.dtype != np.float32:
            raise ValueError(
                f"state dict dtype for {key!r} must be float32, got "
                f"{value.dtype}")
        # ALWAYS copy: np.asarray can alias torch CPU storage, and later
        # in-place torch updates (e.g. optimizer.step) would otherwise
        # silently mutate the converted JAX parameters.
        return np.array(value, dtype=np.float32, copy=True)

    def linear(prefix: str) -> dict[str, np.ndarray]:
        # torch nn.Linear computes x @ W.T; store kernel [in, out] = W.T.
        return {"kernel": tensor(f"{prefix}.weight").T.copy(),
                "bias": tensor(f"{prefix}.bias")}

    params: dict[str, Any] = {
        "manager_token": tensor("manager_token").reshape(-1).copy(),
        "role_embedding": tensor("role_embedding.weight"),
        "tile_encoder": {
            "kind_embedding": tensor("tile_encoder.kind_embedding.weight"),
            "crop_embedding": tensor("tile_encoder.crop_embedding.weight"),
            "animal_embedding":
                tensor("tile_encoder.animal_embedding.weight"),
            "row_embedding": tensor("tile_encoder.row_embedding.weight"),
            "col_embedding": tensor("tile_encoder.col_embedding.weight"),
            "project": {"0": linear("tile_encoder.project.0"),
                        "3": linear("tile_encoder.project.3")},
        },
        "global_encoders": {
            "day_embedding":
                tensor("global_encoders.day_embedding.weight"),
            "self_resource": {
                "0": linear("global_encoders.self_resource.0"),
                "3": linear("global_encoders.self_resource.3")},
            "market": {"0": linear("global_encoders.market.0"),
                       "3": linear("global_encoders.market.3")},
            "town": {"0": linear("global_encoders.town.0"),
                     "3": linear("global_encoders.town.3")},
            "labor": {"0": linear("global_encoders.labor.0"),
                      "3": linear("global_encoders.labor.3")},
            "day_scalar": linear("global_encoders.day_scalar"),
        },
        "encoder": {"layers": []},
        "encoder_norm": {
            "weight": tensor("encoder_norm.weight"),
            "bias": tensor("encoder_norm.bias"),
        },
        "heads": {
            "crop": linear("crop_head"),
            "animal": linear("animal_head"),
            "land": linear("land_head"),
            "fertilizer": linear("fertilizer_head"),
            "care": linear("care_head"),
            "sell_presence": linear("sell_presence_head"),
            "sell_quantity": linear("sell_quantity_head"),
        },
    }
    for i in range(config.num_layers):
        prefix = f"encoder.layers.{i}"
        params["encoder"]["layers"].append({
            # Packed QKV stays packed; chunk order q|k|v matches torch.
            "qkv_kernel": tensor(f"{prefix}.self_attn.in_proj_weight").T.copy(),
            "qkv_bias": tensor(f"{prefix}.self_attn.in_proj_bias"),
            "out_kernel": tensor(f"{prefix}.self_attn.out_proj.weight").T.copy(),
            "out_bias": tensor(f"{prefix}.self_attn.out_proj.bias"),
            "linear1": linear(f"{prefix}.linear1"),
            "linear2": linear(f"{prefix}.linear2"),
            "norm1_weight": tensor(f"{prefix}.norm1.weight"),
            "norm1_bias": tensor(f"{prefix}.norm1.bias"),
            "norm2_weight": tensor(f"{prefix}.norm2.weight"),
            "norm2_bias": tensor(f"{prefix}.norm2.bias"),
        })

    spec = empty_params(config, variant)
    expected_tree, actual_tree = (
        jax.tree_util.tree_structure(spec),
        jax.tree_util.tree_structure(params))
    if expected_tree != actual_tree:
        raise ValueError(
            "converted parameter tree structure does not match the config "
            "spec; this is an internal converter bug")
    return params


def _require_matching_config(payload_config: Mapping[str, Any],
                             config: ManagerConfig | None) -> ManagerConfig:
    fields = {f.name for f in dataclasses.fields(ManagerConfig)}
    payload_keys = set(payload_config.keys())
    if payload_keys != fields:
        raise ValueError(
            f"checkpoint model_config keys {sorted(payload_keys)} do not "
            f"match ManagerConfig fields {sorted(fields)}; refusing to "
            f"guess defaults")
    restored = ManagerConfig(**{
        name: payload_config[name] for name in fields})
    if config is not None and restored != config:
        raise ValueError(
            f"checkpoint model_config {restored} is incompatible with the "
            f"requested config {config}; refusing conversion across "
            f"architectures (e.g. own-only vs include_opponent_board)")
    return restored


def _payload_model_variant(payload: Mapping[str, Any]) -> str:
    """Top-level torch `model_variant`; absent -> "V0" (old checkpoints)."""
    raw = payload.get("model_variant", "V0")
    try:
        return normalize_model_variant(raw)
    except ValueError as error:
        raise ValueError(
            f"checkpoint carries an invalid top-level model_variant "
            f"{raw!r}: {error}") from error


def load_torch_checkpoint(
    source: str | Path | Mapping[str, Any],
    config: ManagerConfig | None = None,
    *,
    model_variant: str | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Load a `bc_manager_checkpoint_v1` payload from a path or dict and
    convert its state dict strictly. Returns (params, metadata).

    The payload's top-level `model_variant` (absent -> "V0") selects the
    conversion contract; exactly V0 and E are supported and J/JE are
    rejected loudly. When `model_variant` is given it must match the stored
    variant exactly — the variant is never inferred from weight shapes.
    """
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        import torch
        try:
            payload = torch.load(str(source), map_location="cpu",
                                 weights_only=True)
        except Exception as error:  # noqa: BLE001 - loud re-wrap below
            raise ValueError(
                f"{source}: failed to load bc_manager_checkpoint_v1 payload "
                f"safely with torch.load(weights_only=True): {error}") \
                from error
    if not isinstance(payload, Mapping) or \
            payload.get("format") != TORCH_CHECKPOINT_FORMAT:
        found = payload.get("format") if isinstance(payload, Mapping) \
            else type(payload).__name__
        raise ValueError(
            f"unrecognized checkpoint format {found!r}; expected "
            f"{TORCH_CHECKPOINT_FORMAT!r}")

    resolved = _require_matching_config(payload["model_config"], config)
    stored_variant = _payload_model_variant(payload)
    if model_variant is not None and \
            resolve_model_variant(model_variant) != stored_variant:
        raise ValueError(
            f"checkpoint stores model_variant {stored_variant!r} but the "
            f"requested variant is "
            f"{resolve_model_variant(model_variant)!r}; refusing to guess")
    params = convert_torch_state_dict(payload["model_state_dict"], resolved,
                                      stored_variant)
    metadata = {
        "format": payload["format"],
        "model_config": dict(payload["model_config"]),
        "model_variant": stored_variant,
        "epoch": payload.get("epoch"),
        "kind": payload.get("kind"),
        "training_config": payload.get("training_config"),
        "validation_metrics": payload.get("validation_metrics"),
    }
    return params, metadata


# --------------------------------------------------------- native format


def save_native(path: str | Path, params: Mapping, config: ManagerConfig,
                metadata: Mapping[str, Any] | None = None,
                model_variant: str = "V0") -> None:
    """Save params/config/metadata as a small pickle-free .npz archive.

    The resolved variant is stored as a top-level `model_variant` record in
    the JSON metadata — OUTSIDE `model_config`, mirroring the torch
    checkpoint layout.
    """
    variant = resolve_model_variant(model_variant)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for tokens, array in jax.tree_util.tree_flatten_with_path(params)[0]:
        parts = [str(getattr(entry, "key", None)
                     if getattr(entry, "key", None) is not None
                     else entry.idx) for entry in tokens]
        key = "param:" + "/".join(parts)
        flat[key] = np.asarray(array, dtype=np.float32)
    meta = {
        "format": NATIVE_CHECKPOINT_FORMAT,
        "model_config": dataclasses.asdict(config),
        "model_variant": variant,
        "metadata": dict(metadata) if metadata else {},
    }
    flat["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.name + ".tmp")
    # Write through a file object so numpy cannot append its own extension.
    with open(tmp, "wb") as handle:
        np.savez(handle, **flat)
    path.unlink(missing_ok=True)
    tmp.rename(path)


def load_native(path: str | Path,
                config: ManagerConfig | None = None,
                *,
                model_variant: str | None = None) -> tuple[dict, dict]:
    """Load a native .npz checkpoint; fail loudly on corruption or any
    config/structure incompatibility.

    Old native files written before variant metadata existed load as V0.
    When `model_variant` is given it must match the stored variant exactly.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            items = {key: archive[key] for key in archive.files}
    except Exception as error:  # noqa: BLE001 - loud re-wrap below
        raise ValueError(
            f"{path}: corrupt or unreadable native checkpoint: {error}") \
            from error
    if "__meta__" not in items:
        raise ValueError(f"{path}: native checkpoint missing __meta__ record")
    meta = json.loads(items["__meta__"].tobytes().decode("utf-8"))
    if meta.get("format") != NATIVE_CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: unrecognized native checkpoint format "
            f"{meta.get('format')!r}; expected {NATIVE_CHECKPOINT_FORMAT!r}")
    resolved = _require_matching_config(meta["model_config"], config)
    stored_variant = _payload_model_variant(meta)
    if model_variant is not None and \
            resolve_model_variant(model_variant) != stored_variant:
        raise ValueError(
            f"{path}: native checkpoint stores model_variant "
            f"{stored_variant!r} but the requested variant is "
            f"{resolve_model_variant(model_variant)!r}")
    meta["model_variant"] = stored_variant

    spec = empty_params(resolved, stored_variant)
    treedef = jax.tree_util.tree_structure(spec)
    leaves_meta = jax.tree_util.tree_flatten_with_path(spec)[0]
    params_leaves: list[jnp.ndarray] = []
    for tokens, template in leaves_meta:
        parts = [str(getattr(entry, "key", None)
                     if getattr(entry, "key", None) is not None
                     else entry.idx) for entry in tokens]
        key = "param:" + "/".join(parts)
        if key not in items:
            raise ValueError(f"{path}: native checkpoint missing param {key!r}")
        array = items[key]
        if array.shape != template.shape or array.dtype != np.float32:
            raise ValueError(
                f"{path}: param {key!r} has shape/dtype "
                f"({array.shape}, {array.dtype}); expected "
                f"({template.shape}, float32)")
        params_leaves.append(jnp.asarray(array))
    return jax.tree_util.tree_unflatten(treedef, params_leaves), meta
