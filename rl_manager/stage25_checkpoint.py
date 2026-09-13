"""Strict, pickle-free native checkpoints for Stage 2.5.

The native format is one ``.npz`` archive. ``__meta__`` is a UTF-8 JSON
record stored as a uint8 array; every other entry is a numeric array. The
parameter tree is rebuilt from ``init_stage25_params`` and every path, shape,
and dtype is checked against that template before arrays are returned to JAX.

Historical BC-E conversion is isolated in ``import_historical_encoder``.
Only an explicit ``.pt``/``.pth`` path may enter the Torch converter. Native
``.npz`` archives stay Torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_LEGACY,
    normalize_e_history_version,
)
from bc_manager_jax.model import ManagerConfig
from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
    ACTION_SCHEMA_VERSION,
)
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


STAGE25_CHECKPOINT_VERSION = "stage25_native_checkpoint_v1"
INFERENCE_PAYLOAD_KIND = "stage25_inference_params_v1"
BC_TRAINING_PAYLOAD_KIND = "stage25_bc_training_state_v1"
PPO_TRAINING_PAYLOAD_KIND = "stage25_ppo_training_state_v1"
ARCHITECTURE_VERSION = "stage25_policy_v1"
OBSERVATION_SCHEMA_VERSION = "stage25_corrected_e_own_only_v1"
PERSISTENT_LEDGER_VERSION = "stage25_crop_capacity_ledger_v1"
PHYSICAL_SUPPORT_VERSION = ACTION_SCHEMA_VERSION
BC_TARGET_VERSION = "stage25_outcome_proxy_v1"
RESUME_BOUNDARY = "after_completed_update_before_next_batch"
PPO_RESUME_BOUNDARY = "after_completed_rollout_update_before_next_rollout"

OBSERVATION_VOCABULARY = (
    "board_kind", "board_crop", "board_animal", "board_numeric",
    "board_bool", "board_mask", "scalars", "shed_counts", "seed_counts",
    "carried_counts", "unlocked", "market_inventory", "market_prices",
    "shop_counts", "day", "days_remaining", "economic_context", "crop_capacity",
)


class Stage25CheckpointError(ValueError):
    """Raised when a Stage 2.5 checkpoint violates its persisted contract."""


_REQUIRED_META = frozenset({
    "format", "payload_kind", "architecture_version", "action_schema_version",
    "observation_schema_version", "persistent_ledger_version",
    "physical_support_version", "action_vocabulary", "action_class_counts",
    "observation_vocabulary", "config", "model_config", "init_params",
    "dtype_precision", "precision", "curriculum", "bc_target",
    "e_history_version", "e_identity", "source_identity", "provenance",
    "executor", "leaf_manifest",
})

# Metadata fields written by ``_metadata`` itself (required plus optional).
# Generic ``metadata`` must never collide with these; callers must use the
# explicit arguments so history/source identity can never be silently dropped.
_OWNED_META = _REQUIRED_META | frozenset({
    "source_e_identity", "step", "epoch", "optimizer_config",
    "optimizer_leaf_count", "optimizer_leaf_manifest", "optimizer_tree",
    "data_order_position", "resume_boundary", "source_history_version",
    "ppo_config",
    "update_counter", "rollout_seed", "rollout_progression",
    "behavior_identity", "physical_contract",
})

_PPO_REQUIRED_META = frozenset({
    "ppo_config", "update_counter", "rollout_seed", "rollout_progression",
    "behavior_identity", "physical_contract", "optimizer_config",
    "optimizer_leaf_count", "optimizer_leaf_manifest", "optimizer_tree",
    "resume_boundary",
})


def _flatten_arrays(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a parameter/state pytree without accepting arbitrary objects."""
    if isinstance(tree, Mapping):
        result: dict[str, np.ndarray] = {}
        for key in sorted(tree):
            if not isinstance(key, str) or not key or "/" in key:
                raise Stage25CheckpointError(
                    f"checkpoint mapping key must be nonempty slash-free string: {key!r}")
            child = f"{prefix}/{key}" if prefix else key
            result.update(_flatten_arrays(tree[key], child))
        return result
    if isinstance(tree, (tuple, list)):
        result = {}
        for index, value in enumerate(tree):
            child = f"{prefix}/{index}" if prefix else str(index)
            result.update(_flatten_arrays(value, child))
        return result
    try:
        array = np.asarray(tree)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise Stage25CheckpointError(
            f"checkpoint leaf {prefix or '<root>'!r} is not an array") from exc
    if array.dtype.hasobject:
        raise Stage25CheckpointError(
            f"checkpoint leaf {prefix or '<root>'!r} has forbidden object dtype")
    return {prefix: array}


def validate_array_tree(tree: Any, template: Any, *, what: str) -> dict[str, np.ndarray]:
    """Return flattened arrays after exact path, shape, and dtype validation."""
    actual = _flatten_arrays(tree)
    expected = _flatten_arrays(template)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    if missing or unexpected:
        raise Stage25CheckpointError(
            f"{what} tree mismatch: missing={missing}, unexpected={unexpected}")
    for path, expected_array in expected.items():
        actual_array = actual[path]
        if actual_array.shape != expected_array.shape:
            raise Stage25CheckpointError(
                f"{what} leaf {path!r} shape {actual_array.shape} != {expected_array.shape}")
        if actual_array.dtype != expected_array.dtype:
            raise Stage25CheckpointError(
                f"{what} leaf {path!r} dtype {actual_array.dtype} != {expected_array.dtype}")
    return actual


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage25CheckpointError("metadata mapping keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise Stage25CheckpointError("metadata cannot contain non-finite floats")
        return value
    raise Stage25CheckpointError(
        f"metadata value {type(value).__name__!r} is not JSON-serializable")


def _config_json(config: Stage25ModelConfig) -> dict[str, Any]:
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("config must be Stage25ModelConfig")
    return _jsonable(dataclasses.asdict(config))


def _config_from_json(payload: Mapping[str, Any]) -> Stage25ModelConfig:
    if not isinstance(payload, Mapping):
        raise Stage25CheckpointError("checkpoint config must be an object")
    allowed = {field.name for field in dataclasses.fields(Stage25ModelConfig)}
    if set(payload) != allowed:
        raise Stage25CheckpointError(
            f"checkpoint config keys {sorted(payload)} != {sorted(allowed)}")
    values = dict(payload)
    manager = values.get("manager_config")
    manager_fields = {field.name for field in dataclasses.fields(ManagerConfig)}
    if not isinstance(manager, Mapping) or set(manager) != manager_fields:
        raise Stage25CheckpointError("checkpoint manager_config is incompatible")
    values["manager_config"] = ManagerConfig(**dict(manager))
    curriculum = values.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise Stage25CheckpointError("checkpoint curriculum is invalid")
    values["curriculum"] = Stage25CurriculumConfig(**dict(curriculum))
    try:
        return Stage25ModelConfig(**values)
    except Exception as exc:
        raise Stage25CheckpointError(f"checkpoint config is incompatible: {exc}") from exc


def _leaf_manifest(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for key, array in sorted(flat.items())
    }


def _tree_signature(tree: Any) -> str:
    """Stable container/leaf signature used to reject incompatible resumes."""
    try:
        return repr(jax.tree_util.tree_structure(tree))
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise Stage25CheckpointError("state is not a valid JAX pytree") from exc


def _metadata(*, payload_kind: str, config: Stage25ModelConfig, seed: int,
              flat: Mapping[str, np.ndarray], metadata: Mapping[str, Any] | None,
              source_identity: Mapping[str, Any] | None,
              provenance: Mapping[str, Any] | None,
              executor: Mapping[str, Any] | None,
              e_history_version: str, source_history_version: str | None = None,
              step: int | None = None,
              epoch: int | None = None, optimizer_config: Any = None,
              data_order_position: Any = None) -> dict[str, Any]:
    history = normalize_e_history_version(e_history_version)
    config_payload = _config_json(config)
    required: dict[str, Any] = {
        "format": STAGE25_CHECKPOINT_VERSION,
        "payload_kind": payload_kind,
        "architecture_version": ARCHITECTURE_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "persistent_ledger_version": PERSISTENT_LEDGER_VERSION,
        "physical_support_version": PHYSICAL_SUPPORT_VERSION,
        "action_vocabulary": list(ACTION_ORDER),
        "action_class_counts": list(ACTION_CLASS_COUNTS),
        "observation_vocabulary": list(OBSERVATION_VOCABULARY),
        "config": config_payload,
        "model_config": config_payload,
        "init_params": {"seed": int(seed), "initializer": "init_stage25_params"},
        "dtype_precision": {"params": "float32", "precision": "float32"},
        "precision": "float32",
        "curriculum": _jsonable(dataclasses.asdict(config.curriculum)),
        "bc_target": BC_TARGET_VERSION,
        "e_history_version": history,
        "e_identity": {
            "variant": "E", "history_version": history,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        },
        "source_identity": _jsonable(source_identity or {}),
        "provenance": _jsonable(provenance or {}),
        "executor": _jsonable(executor or {}),
        "leaf_manifest": _leaf_manifest(flat),
    }
    if source_history_version is not None:
        required["source_e_identity"] = {
            "variant": "E",
            "history_version": normalize_e_history_version(source_history_version),
            "transfer": "encoder_only",
        }
    if step is not None:
        required["step"] = int(step)
    if epoch is not None:
        required["epoch"] = int(epoch)
    if optimizer_config is not None:
        required["optimizer_config"] = _jsonable(optimizer_config)
    if data_order_position is not None:
        required["data_order_position"] = _jsonable(data_order_position)
    if payload_kind == BC_TRAINING_PAYLOAD_KIND:
        required["resume_boundary"] = RESUME_BOUNDARY
    if metadata:
        extras = _jsonable(metadata)
        if not isinstance(extras, Mapping):
            raise Stage25CheckpointError("metadata must be a mapping")
        collisions = sorted(set(extras) & _OWNED_META)
        if collisions:
            raise Stage25CheckpointError(
                f"metadata contains reserved checkpoint keys {collisions}; pass "
                f"them through the explicit operating/source-history, "
                f"source_identity, provenance, or executor arguments instead")
        required.update(extras)
    return required


def _write_archive(path: str | Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["__meta__"] = np.frombuffer(
        json.dumps(_jsonable(meta), sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "xb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


def _read_archive(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            items = {key: archive[key] for key in archive.files}
    except Exception as exc:  # noqa: BLE001
        raise Stage25CheckpointError(
            f"{path}: corrupt or unreadable native Stage 2.5 checkpoint: {exc}") from exc
    meta_array = items.pop("__meta__", None)
    if meta_array is None:
        raise Stage25CheckpointError(f"{path}: checkpoint missing __meta__ record")
    if meta_array.dtype != np.uint8 or meta_array.ndim != 1:
        raise Stage25CheckpointError(f"{path}: __meta__ must be a uint8 vector")
    try:
        meta = json.loads(meta_array.tobytes().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise Stage25CheckpointError(f"{path}: invalid JSON metadata") from exc
    if not isinstance(meta, dict):
        raise Stage25CheckpointError(f"{path}: metadata must be a JSON object")
    for key, array in items.items():
        if array.dtype.hasobject:
            raise Stage25CheckpointError(f"{path}: leaf {key!r} has object dtype")
    return items, meta


def _validate_meta(meta: Mapping[str, Any], path: Path, expected_kind: str) -> None:
    if meta.get("format") != STAGE25_CHECKPOINT_VERSION:
        raise Stage25CheckpointError(
            f"{path}: version-incompatible format {meta.get('format')!r}")
    if meta.get("payload_kind") != expected_kind:
        raise Stage25CheckpointError(
            f"{path}: payload kind {meta.get('payload_kind')!r} != {expected_kind!r}")
    checks = {
        "architecture_version": ARCHITECTURE_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "persistent_ledger_version": PERSISTENT_LEDGER_VERSION,
        "physical_support_version": PHYSICAL_SUPPORT_VERSION,
        "action_vocabulary": list(ACTION_ORDER),
        "action_class_counts": list(ACTION_CLASS_COUNTS),
        "observation_vocabulary": list(OBSERVATION_VOCABULARY),
        "bc_target": BC_TARGET_VERSION,
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise Stage25CheckpointError(
                f"{path}: incompatible {key}: {meta.get(key)!r} != {expected!r}")
    for key in _REQUIRED_META:
        if key not in meta:
            raise Stage25CheckpointError(f"{path}: metadata missing required field {key!r}")
    if meta.get("config") != meta.get("model_config"):
        raise Stage25CheckpointError(f"{path}: config/model_config metadata disagree")
    config = meta.get("config")
    if isinstance(config, Mapping) and meta.get("curriculum") != config.get("curriculum"):
        raise Stage25CheckpointError(f"{path}: curriculum metadata disagrees with config")
    if meta.get("precision") != "float32" or not isinstance(meta.get("dtype_precision"), Mapping):
        raise Stage25CheckpointError(f"{path}: unsupported dtype/precision metadata")
    try:
        history = normalize_e_history_version(meta["e_history_version"])
    except Exception as exc:
        raise Stage25CheckpointError(f"{path}: incompatible E history version") from exc
    identity = meta.get("e_identity")
    if not isinstance(identity, Mapping) or identity.get("variant") != "E" \
            or identity.get("history_version") != history:
        raise Stage25CheckpointError(f"{path}: corrected E identity is inconsistent")
    source_identity_meta = meta.get("source_e_identity")
    if source_identity_meta is not None:
        if (not isinstance(source_identity_meta, Mapping)
                or source_identity_meta.get("variant") != "E"
                or source_identity_meta.get("transfer") != "encoder_only"):
            raise Stage25CheckpointError(
                f"{path}: source E identity is inconsistent")
        try:
            normalize_e_history_version(source_identity_meta.get("history_version"))
        except Exception as exc:
            raise Stage25CheckpointError(
                f"{path}: invalid source E history version") from exc
    if meta.get("resume_boundary") not in (None, RESUME_BOUNDARY,
                                            PPO_RESUME_BOUNDARY):
        raise Stage25CheckpointError(f"{path}: unsupported resume boundary")
    if expected_kind == BC_TRAINING_PAYLOAD_KIND:
        for key in ("step", "epoch", "data_order_position", "optimizer_leaf_count",
                    "optimizer_leaf_manifest", "optimizer_tree", "optimizer_config",
                    "resume_boundary"):
            if key not in meta:
                raise Stage25CheckpointError(f"{path}: BC metadata missing required field {key!r}")
        for key in ("step", "epoch", "optimizer_leaf_count"):
            value = meta.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Stage25CheckpointError(f"{path}: BC metadata field {key!r} is invalid")
        position = meta.get("data_order_position")
        if not isinstance(position, Mapping):
            raise Stage25CheckpointError(f"{path}: data_order_position must be an object")
        for key in ("epoch", "batch", "seed"):
            value = position.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Stage25CheckpointError(
                    f"{path}: data_order_position field {key!r} is invalid")
    if expected_kind == PPO_TRAINING_PAYLOAD_KIND:
        for key in _PPO_REQUIRED_META:
            if key not in meta:
                raise Stage25CheckpointError(
                    f"{path}: PPO metadata missing required field {key!r}")
        if meta.get("resume_boundary") != PPO_RESUME_BOUNDARY:
            raise Stage25CheckpointError(
                f"{path}: unsupported PPO resume boundary "
                f"{meta.get('resume_boundary')!r}")
        update_counter = meta.get("update_counter")
        if (isinstance(update_counter, bool) or not isinstance(update_counter, int)
                or update_counter < 0):
            raise Stage25CheckpointError(
                f"{path}: PPO update_counter must be a nonnegative integer")
        rollout_seed = meta.get("rollout_seed")
        if rollout_seed is not None and (
                isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int)
                or rollout_seed < 0):
            raise Stage25CheckpointError(
                f"{path}: PPO rollout_seed must be a nonnegative integer or null")
        if not isinstance(meta.get("rollout_progression"), (Mapping, list, tuple, str,
                                                              int, float, bool)) \
                and meta.get("rollout_progression") is not None:
            raise Stage25CheckpointError(
                f"{path}: PPO rollout_progression is not JSON metadata")
        for key in ("behavior_identity", "physical_contract"):
            if not isinstance(meta.get(key), Mapping):
                raise Stage25CheckpointError(
                    f"{path}: PPO {key} must be an object")
        if not isinstance(meta.get("ppo_config"), Mapping):
            raise Stage25CheckpointError(f"{path}: PPO ppo_config must be an object")
        if not isinstance(meta.get("optimizer_config"), Mapping):
            raise Stage25CheckpointError(
                f"{path}: PPO optimizer_config must be an object")
        optimizer_count = meta.get("optimizer_leaf_count")
        if (isinstance(optimizer_count, bool) or not isinstance(optimizer_count, int)
                or optimizer_count < 0):
            raise Stage25CheckpointError(
                f"{path}: PPO optimizer_leaf_count must be a nonnegative integer")


def _check_history(meta: Mapping[str, Any], *, expected: str | None,
                   allow_legacy_e: bool, path: Path) -> None:
    history = normalize_e_history_version(meta["e_history_version"])
    if history == E_HISTORY_LEGACY and not allow_legacy_e:
        raise Stage25CheckpointError(
            f"{path}: legacy E history requires explicit allow_legacy_e=True")
    if expected is not None and history != normalize_e_history_version(expected) \
            and not (allow_legacy_e and history == E_HISTORY_LEGACY):
        raise Stage25CheckpointError(
            f"{path}: e_history_version {history!r} does not match requested "
            f"{normalize_e_history_version(expected)!r}")


def _rebuild(template: Any, flat: Mapping[str, np.ndarray], prefix: str = "") -> Any:
    if isinstance(template, Mapping):
        return {key: _rebuild(template[key], flat, f"{prefix}/{key}" if prefix else key)
                for key in sorted(template)}
    if isinstance(template, tuple):
        return tuple(_rebuild(value, flat, f"{prefix}/{index}" if prefix else str(index))
                     for index, value in enumerate(template))
    if isinstance(template, list):
        return [_rebuild(value, flat, f"{prefix}/{index}" if prefix else str(index))
                for index, value in enumerate(template)]
    return jnp.asarray(flat[prefix])


def _validate_flat_against_manifest(flat: Mapping[str, np.ndarray], meta: Mapping[str, Any],
                                    path: Path) -> None:
    manifest = meta.get("leaf_manifest")
    if not isinstance(manifest, Mapping):
        raise Stage25CheckpointError(f"{path}: invalid leaf_manifest")
    if set(flat) != set(manifest):
        raise Stage25CheckpointError(
            f"{path}: leaf mismatch: missing={sorted(set(manifest) - set(flat))}, "
            f"extra={sorted(set(flat) - set(manifest))}")
    for key, array in flat.items():
        record = manifest[key]
        if not isinstance(record, Mapping) or record.get("shape") != list(array.shape) \
                or record.get("dtype") != str(array.dtype):
            raise Stage25CheckpointError(f"{path}: corrupt leaf manifest for {key!r}")


def _load_params(path: str | Path, expected_kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any], Stage25ModelConfig]:
    path = Path(path)
    flat, meta = _read_archive(path)
    _validate_meta(meta, path, expected_kind)
    config = _config_from_json(meta["config"])
    _validate_flat_against_manifest(flat, meta, path)
    params = init_stage25_params(config, seed=int(meta["init_params"]["seed"]))
    param_items = {key[len("param:"):]: value for key, value in flat.items()
                   if key.startswith("param:")}
    non_param = {key: value for key, value in flat.items() if not key.startswith("param:")}
    if non_param and expected_kind == INFERENCE_PAYLOAD_KIND:
        raise Stage25CheckpointError(
            f"{path}: unexpected non-parameter leaves {sorted(non_param)}")
    expected_template = _flatten_arrays(params)
    missing = sorted(set(expected_template) - set(param_items))
    extra = sorted(set(param_items) - set(expected_template))
    if missing or extra:
        raise Stage25CheckpointError(
            f"params tree mismatch: missing={missing}, unexpected={extra}")
    for key, expected_array in expected_template.items():
        actual_array = param_items[key]
        if actual_array.shape != expected_array.shape:
            raise Stage25CheckpointError(
                f"params leaf {key!r} shape {actual_array.shape} != {expected_array.shape}")
        if actual_array.dtype != expected_array.dtype:
            raise Stage25CheckpointError(
                f"params leaf {key!r} dtype {actual_array.dtype} != {expected_array.dtype}")
    return flat, meta, config


def _source_identity(source: Any, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        result: dict[str, Any] = {"name": source_path.name}
        if source_path.exists():
            result["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if meta and meta.get("format"):
            result["format"] = meta["format"]
        return result
    return {"format": meta.get("format"), "name": "in_memory"} if meta else {"name": "in_memory"}


def save_stage25_inference_checkpoint(
    path: str | Path, params: Mapping[str, Any], config: Stage25ModelConfig, *,
    seed: int = 0, metadata: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    executor: Mapping[str, Any] | None = None,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    source_history_version: str | None = None,
) -> Path:
    """Save inference parameters with an atomic replace."""
    template = init_stage25_params(config, seed=int(seed))
    flat = {f"param:{key}": value for key, value in
            validate_array_tree(params, template, what="params").items()}
    meta = _metadata(payload_kind=INFERENCE_PAYLOAD_KIND, config=config,
                     seed=int(seed), flat=flat, metadata=metadata,
                     source_identity=source_identity, provenance=provenance,
                     executor=executor, e_history_version=e_history_version,
                     source_history_version=source_history_version)
    return _write_archive(path, flat, meta)


def load_stage25_inference_checkpoint(
    path: str | Path, *, config: Stage25ModelConfig | None = None,
    seed: int | None = None,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
    allow_legacy_e: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load inference parameters, rejecting every incompatible leaf."""
    flat, meta, stored_config = _load_params(path, INFERENCE_PAYLOAD_KIND)
    path = Path(path)
    _check_history(meta, expected=expected_e_history_version,
                   allow_legacy_e=allow_legacy_e, path=path)
    if config is not None and config != stored_config:
        raise Stage25CheckpointError("checkpoint config is incompatible with requested config")
    stored_seed = int(meta["init_params"]["seed"])
    if seed is not None and int(seed) != stored_seed:
        raise Stage25CheckpointError(f"checkpoint seed {stored_seed} != requested seed {int(seed)}")
    params = init_stage25_params(stored_config, seed=stored_seed)
    return _rebuild(params, {key[len("param:"):]: value for key, value in flat.items()
                             if key.startswith("param:")}), dict(meta)


def save_stage25_bc_checkpoint(
    path: str | Path, params: Mapping[str, Any], optimizer_state: Any, rng: Any,
    config: Stage25ModelConfig, *, seed: int = 0, step: int, epoch: int,
    optimizer_config: Any = None, data_order_position: Any = 0,
    metadata: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    executor: Mapping[str, Any] | None = None,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    source_history_version: str | None = None,
) -> Path:
    """Save resumable BC state at the post-update/pre-next-batch boundary."""
    if optimizer_config is None:
        raise Stage25CheckpointError(
            "optimizer_config is required for a resumable BC checkpoint")
    if data_order_position == 0:
        data_order_position = {"epoch": 0, "batch": 0, "seed": int(seed)}
    if not isinstance(data_order_position, Mapping):
        raise Stage25CheckpointError("data_order_position must be an object")
    else:
        data_order_position = dict(data_order_position)
        data_order_position.setdefault("epoch", int(epoch))
        data_order_position.setdefault("batch", 0)
        data_order_position.setdefault("seed", int(seed))
    if isinstance(step, bool) or int(step) < 0 or isinstance(epoch, bool) or int(epoch) < 0:
        raise Stage25CheckpointError("step and epoch must be nonnegative integers")
    template = init_stage25_params(config, seed=int(seed))
    param_flat = validate_array_tree(params, template, what="params")
    arrays = {f"param:{key}": value for key, value in param_flat.items()}
    try:
        opt_leaves = jax.tree_util.tree_leaves(optimizer_state)
    except Exception as exc:  # pragma: no cover
        raise Stage25CheckpointError("optimizer_state is not a valid JAX pytree") from exc
    opt_arrays = []
    for index, leaf in enumerate(opt_leaves):
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise Stage25CheckpointError(f"optimizer leaf {index} has forbidden object dtype")
        opt_arrays.append(array)
        arrays[f"opt:{index:05d}"] = array
    rng_array = np.asarray(rng)
    if rng_array.dtype.hasobject or rng_array.shape != (2,) or rng_array.dtype != np.uint32:
        raise Stage25CheckpointError(f"rng must be uint32 [2], got {rng_array.shape}/{rng_array.dtype}")
    arrays["rng"] = np.array(rng_array, copy=True)
    meta = _metadata(payload_kind=BC_TRAINING_PAYLOAD_KIND, config=config,
                     seed=int(seed), flat=arrays, metadata=metadata,
                     source_identity=source_identity, provenance=provenance,
                     executor=executor, e_history_version=e_history_version,
                     source_history_version=source_history_version,
                     step=int(step), epoch=int(epoch), optimizer_config=optimizer_config,
                     data_order_position=data_order_position)
    meta["optimizer_leaf_count"] = len(opt_arrays)
    meta["optimizer_tree"] = _tree_signature(optimizer_state)
    meta["optimizer_leaf_manifest"] = _leaf_manifest(
        {f"opt:{index:05d}": value for index, value in enumerate(opt_arrays)})
    return _write_archive(path, arrays, meta)


def load_stage25_bc_checkpoint(
    path: str | Path, *, config: Stage25ModelConfig | None = None,
    seed: int | None = None, optimizer_state_template: Any | None = None,
    optimizer_config: Any | None = None,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
    allow_legacy_e: bool = False,
    expected_data_order_seed: int | None = None,
) -> tuple[dict[str, Any], Any, jax.Array, dict[str, Any]]:
    """Load ``(params, optimizer_state, rng, metadata)``.

    The optimizer tree is reconstructed from the caller's exact template, or
    from the persisted ``bc_manager_jax.train.TrainConfig``. Requiring one is
    intentional: optimizer container structure must never be guessed.
    """
    flat, meta, stored_config = _load_params(path, BC_TRAINING_PAYLOAD_KIND)
    path = Path(path)
    _check_history(meta, expected=expected_e_history_version,
                   allow_legacy_e=allow_legacy_e, path=path)
    if config is not None and config != stored_config:
        raise Stage25CheckpointError("checkpoint config is incompatible with requested config")
    stored_seed = int(meta["init_params"]["seed"])
    if seed is not None and int(seed) != stored_seed:
        raise Stage25CheckpointError("checkpoint seed does not match requested seed")
    if expected_data_order_seed is not None:
        position = meta.get("data_order_position")
        if (not isinstance(position, Mapping)
                or int(position.get("seed", -1)) != int(expected_data_order_seed)):
            raise Stage25CheckpointError("checkpoint data-order seed does not match requested seed")
    if optimizer_config is not None:
        if _jsonable(optimizer_config) != meta.get("optimizer_config"):
            raise Stage25CheckpointError(
                "checkpoint optimizer configuration does not match requested configuration")
    params_template = init_stage25_params(stored_config, seed=stored_seed)
    params = _rebuild(
        params_template,
        {key[len("param:"):]: value for key, value in flat.items()
         if key.startswith("param:")})
    rng = flat.get("rng")
    if rng is None or rng.shape != (2,) or rng.dtype != np.uint32:
        raise Stage25CheckpointError(f"{path}: missing or invalid rng leaf")
    opt_items = {key: value for key, value in flat.items() if key.startswith("opt:")}
    expected_opt_count = int(meta.get("optimizer_leaf_count", -1))
    expected_opt_keys = {f"opt:{index:05d}" for index in range(expected_opt_count)}
    if set(opt_items) != expected_opt_keys:
        raise Stage25CheckpointError(f"{path}: optimizer leaf set is incomplete or has extras")
    _validate_flat_against_manifest(opt_items,
                                    {"leaf_manifest": meta.get("optimizer_leaf_manifest")}, path)
    if optimizer_state_template is None and optimizer_config is None:
        optimizer_config = meta.get("optimizer_config")
    if optimizer_state_template is None and optimizer_config is not None:
        try:
            from bc_manager_jax.train import TrainConfig
            from rl_manager.stage25_bc import Stage25BCConfig, init_opt_state as init_bc_opt_state
            if isinstance(optimizer_config, Stage25BCConfig):
                optimizer_state_template = init_bc_opt_state(params, optimizer_config)
            elif isinstance(optimizer_config, TrainConfig):
                from bc_manager_jax.train import init_opt_state
                optimizer_state_template = init_opt_state(params, optimizer_config)
            elif isinstance(optimizer_config, Mapping) and "model" in optimizer_config:
                # The native BC wrapper persists Stage25BCConfig, not the
                # legacy manager TrainConfig. Rebuild that exact optimizer
                # contract without routing normal startup through Torch.
                values = dict(optimizer_config)
                values["model"] = _config_from_json(values["model"])
                bc_config = Stage25BCConfig(**values)
                optimizer_state_template = init_bc_opt_state(params, bc_config)
            else:
                from bc_manager_jax.train import init_opt_state
                optimizer_state_template = init_opt_state(
                    params, TrainConfig(**dict(optimizer_config)))
        except Exception as exc:
            raise Stage25CheckpointError(
                f"{path}: optimizer_config cannot rebuild optimizer state: {exc}") from exc
    if optimizer_state_template is None:
        raise Stage25CheckpointError(
            "optimizer_state_template or optimizer_config is required to resume BC")
    expected_leaves = jax.tree_util.tree_leaves(optimizer_state_template)
    if _tree_signature(optimizer_state_template) != meta.get("optimizer_tree"):
        raise Stage25CheckpointError("checkpoint optimizer pytree structure is incompatible")
    if len(expected_leaves) != expected_opt_count:
        raise Stage25CheckpointError(
            f"{path}: optimizer leaf count {expected_opt_count} != template {len(expected_leaves)}")
    for index, expected in enumerate(expected_leaves):
        actual = opt_items[f"opt:{index:05d}"]
        expected_array = np.asarray(expected)
        if actual.shape != expected_array.shape or actual.dtype != expected_array.dtype:
            raise Stage25CheckpointError(
                f"{path}: optimizer leaf {index} shape/dtype {actual.shape}/{actual.dtype} != "
                f"{expected_array.shape}/{expected_array.dtype}")
    optimizer_state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(optimizer_state_template),
        [jnp.asarray(opt_items[f"opt:{index:05d}"]) for index in range(expected_opt_count)])
    return params, optimizer_state, jnp.asarray(rng), dict(meta)


def _metadata_object(value: Any, *, what: str) -> dict[str, Any]:
    """Normalize an identity/contract object without accepting opaque leaves."""
    to_json = getattr(value, "to_json_dict", None)
    if callable(to_json):
        value = to_json()
    normalized = _jsonable(value)
    if not isinstance(normalized, Mapping):
        raise Stage25CheckpointError(f"{what} must be a metadata object")
    return dict(normalized)


def _curriculum_metadata(
    config: Stage25ModelConfig,
    curriculum: Stage25CurriculumConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if curriculum is None:
        active = config.curriculum
    elif isinstance(curriculum, Stage25CurriculumConfig):
        active = curriculum
    elif isinstance(curriculum, Mapping):
        try:
            active = Stage25CurriculumConfig(**dict(curriculum))
        except Exception as exc:
            raise Stage25CheckpointError(
                f"PPO curriculum is incompatible: {exc}") from exc
    else:
        raise Stage25CheckpointError("PPO curriculum must be a config or object")
    if active != config.curriculum:
        raise Stage25CheckpointError(
            "PPO curriculum must match the Stage25ModelConfig curriculum")
    return _jsonable(dataclasses.asdict(active))


def _validate_ppo_arrays(flat: Mapping[str, np.ndarray], path: Path) -> None:
    allowed = {key for key in flat if key == "rng" or key.startswith("param:")
               or key.startswith("opt:")}
    if allowed != set(flat):
        raise Stage25CheckpointError(
            f"{path}: PPO archive contains unexpected leaves "
            f"{sorted(set(flat) - allowed)}")


def save_stage25_ppo_checkpoint(
    path: str | Path, params: Mapping[str, Any], optimizer_state: Any, rng: Any,
    config: Stage25ModelConfig, *, seed: int = 0, update_counter: int = 0,
    rollout_seed: int | None = None, rollout_progression: Any = None,
    ppo_config: Any = None, optimizer_config: Any = None,
    curriculum: Stage25CurriculumConfig | Mapping[str, Any] | None = None,
    behavior_identity: Any = None, provenance: Mapping[str, Any] | None = None,
    physical_contract: Mapping[str, Any] | None = None,
    executor: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    source_history_version: str | None = None,
) -> Path:
    """Save native PPO state at a completed-rollout/update boundary.

    The boundary deliberately excludes in-progress episodes and minibatches;
    the persisted rollout progression identifies the next complete rollout to
    collect, while the optimizer tree is validated from its exact template.
    """
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("config must be Stage25ModelConfig")
    if isinstance(update_counter, bool) or not isinstance(update_counter, (int, np.integer)) \
            or int(update_counter) < 0:
        raise Stage25CheckpointError(
            "update_counter must be a nonnegative integer")
    if rollout_seed is not None and (
            isinstance(rollout_seed, bool)
            or not isinstance(rollout_seed, (int, np.integer))
            or int(rollout_seed) < 0):
        raise Stage25CheckpointError(
            "rollout_seed must be a nonnegative integer or null")
    template = init_stage25_params(config, seed=int(seed))
    param_flat = validate_array_tree(params, template, what="params")
    arrays = {f"param:{key}": value for key, value in param_flat.items()}
    try:
        opt_leaves = jax.tree_util.tree_leaves(optimizer_state)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise Stage25CheckpointError(
            "optimizer_state is not a valid JAX pytree") from exc
    opt_arrays: list[np.ndarray] = []
    for index, leaf in enumerate(opt_leaves):
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise Stage25CheckpointError(
                f"optimizer leaf {index} has forbidden object dtype")
        opt_arrays.append(array)
        arrays[f"opt:{index:05d}"] = array
    rng_array = np.asarray(rng)
    if rng_array.dtype.hasobject or rng_array.shape != (2,) \
            or rng_array.dtype != np.uint32:
        raise Stage25CheckpointError(
            f"rng must be uint32 [2], got {rng_array.shape}/{rng_array.dtype}")
    arrays["rng"] = np.array(rng_array, copy=True)

    ppo_payload = {} if ppo_config is None else _metadata_object(
        ppo_config, what="PPO configuration")
    optimizer_payload = (ppo_payload if optimizer_config is None else
                         _metadata_object(optimizer_config,
                                          what="optimizer configuration"))
    physical_payload = _metadata_object(
        physical_contract if physical_contract is not None else {
            "version": PHYSICAL_SUPPORT_VERSION,
            "action_vocabulary": list(ACTION_ORDER),
            "action_class_counts": list(ACTION_CLASS_COUNTS),
        }, what="physical_contract")
    identity_payload = _metadata_object(
        behavior_identity if behavior_identity is not None else {},
        what="behavior_identity")
    progression_payload = ({} if rollout_progression is None else
                           _jsonable(rollout_progression))
    meta = _metadata(
        payload_kind=PPO_TRAINING_PAYLOAD_KIND, config=config, seed=int(seed),
        flat=arrays, metadata=metadata, source_identity=source_identity,
        provenance=provenance, executor=executor,
        e_history_version=e_history_version,
        source_history_version=source_history_version,
        optimizer_config=optimizer_payload)
    meta.update({
        "ppo_config": ppo_payload,
        "update_counter": int(update_counter),
        "rollout_seed": (None if rollout_seed is None else int(rollout_seed)),
        "rollout_progression": progression_payload,
        "behavior_identity": identity_payload,
        "physical_contract": physical_payload,
        "optimizer_leaf_count": len(opt_arrays),
        "optimizer_tree": _tree_signature(optimizer_state),
        "optimizer_leaf_manifest": _leaf_manifest({
            f"opt:{index:05d}": value
            for index, value in enumerate(opt_arrays)}),
        "resume_boundary": PPO_RESUME_BOUNDARY,
    })
    return _write_archive(path, arrays, meta)


def load_stage25_ppo_checkpoint(
    path: str | Path, *, config: Stage25ModelConfig | None = None,
    seed: int | None = None, optimizer_state_template: Any | None = None,
    ppo_config: Any | None = None, optimizer_config: Any | None = None,
    curriculum: Stage25CurriculumConfig | Mapping[str, Any] | None = None,
    expected_behavior_identity: Any | None = None,
    expected_physical_contract: Mapping[str, Any] | None = None,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
    allow_legacy_e: bool = False,
) -> tuple[dict[str, Any], Any, jax.Array, dict[str, Any]]:
    """Load ``(params, optimizer_state, rng, metadata)`` for native PPO.

    An optimizer template is required because arbitrary optimizer pytrees
    cannot be safely reconstructed from JSON.  This keeps resume exact and
    makes a mid-minibatch or mid-episode claim impossible at this boundary.
    """
    flat, meta, stored_config = _load_params(path, PPO_TRAINING_PAYLOAD_KIND)
    path = Path(path)
    _validate_ppo_arrays(flat, path)
    _check_history(meta, expected=expected_e_history_version,
                   allow_legacy_e=allow_legacy_e, path=path)
    if config is not None and config != stored_config:
        raise Stage25CheckpointError(
            "checkpoint config is incompatible with requested config")
    stored_seed = int(meta["init_params"]["seed"])
    if seed is not None and int(seed) != stored_seed:
        raise Stage25CheckpointError("checkpoint seed does not match requested seed")
    if curriculum is not None and _curriculum_metadata(stored_config, curriculum) \
            != meta["curriculum"]:
        raise Stage25CheckpointError(
            "checkpoint curriculum is incompatible with requested curriculum")
    if ppo_config is not None and _metadata_object(
            ppo_config, what="PPO configuration") != meta["ppo_config"]:
        raise Stage25CheckpointError(
            "checkpoint PPO configuration does not match requested configuration")
    if optimizer_config is not None and _metadata_object(
            optimizer_config, what="optimizer configuration") != meta["optimizer_config"]:
        raise Stage25CheckpointError(
            "checkpoint optimizer configuration does not match requested configuration")
    if expected_behavior_identity is not None and _metadata_object(
            expected_behavior_identity, what="behavior_identity") != meta["behavior_identity"]:
        raise Stage25CheckpointError(
            "checkpoint behavior identity does not match requested identity")
    if expected_physical_contract is not None and _metadata_object(
            expected_physical_contract, what="physical_contract") != meta["physical_contract"]:
        raise Stage25CheckpointError(
            "checkpoint physical contract does not match requested contract")

    params_template = init_stage25_params(stored_config, seed=stored_seed)
    params = _rebuild(
        params_template,
        {key[len("param:"):]: value for key, value in flat.items()
         if key.startswith("param:")})
    rng = flat.get("rng")
    if rng is None or rng.shape != (2,) or rng.dtype != np.uint32:
        raise Stage25CheckpointError(
            f"{path}: missing or invalid rng leaf")
    if optimizer_state_template is None:
        raise Stage25CheckpointError(
            "optimizer_state_template is required to resume PPO")
    try:
        expected_leaves = jax.tree_util.tree_leaves(optimizer_state_template)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise Stage25CheckpointError(
            "optimizer_state_template is not a valid JAX pytree") from exc
    if _tree_signature(optimizer_state_template) != meta["optimizer_tree"]:
        raise Stage25CheckpointError(
            "checkpoint optimizer pytree structure is incompatible")
    opt_items = {key: value for key, value in flat.items()
                 if key.startswith("opt:")}
    expected_opt_count = meta["optimizer_leaf_count"]
    expected_opt_keys = {f"opt:{index:05d}" for index in range(expected_opt_count)}
    if set(opt_items) != expected_opt_keys:
        raise Stage25CheckpointError(
            f"{path}: optimizer leaf set is incomplete or has extras")
    _validate_flat_against_manifest(
        opt_items, {"leaf_manifest": meta["optimizer_leaf_manifest"]}, path)
    if len(expected_leaves) != expected_opt_count:
        raise Stage25CheckpointError(
            f"{path}: optimizer leaf count {expected_opt_count} != "
            f"template {len(expected_leaves)}")
    for index, expected in enumerate(expected_leaves):
        actual = opt_items[f"opt:{index:05d}"]
        expected_array = np.asarray(expected)
        if actual.shape != expected_array.shape or actual.dtype != expected_array.dtype:
            raise Stage25CheckpointError(
                f"{path}: optimizer leaf {index} shape/dtype "
                f"{actual.shape}/{actual.dtype} != "
                f"{expected_array.shape}/{expected_array.dtype}")
    optimizer_state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(optimizer_state_template),
        [jnp.asarray(opt_items[f"opt:{index:05d}"])
         for index in range(expected_opt_count)])
    return params, optimizer_state, jnp.asarray(rng), dict(meta)


def initialize_stage25_ppo_from_checkpoint(
    source: str | Path, config: Stage25ModelConfig | None = None, *,
    seed: int | None = None,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
    allow_legacy_e: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load params for a fresh PPO run from native BC or inference state.

    Optimizer state is intentionally not transferred: the caller creates a
    fresh PPO optimizer for the returned parameter tree.
    """
    source_path = Path(source)
    flat, meta = _read_archive(source_path)
    kind = meta.get("payload_kind")
    if kind == INFERENCE_PAYLOAD_KIND:
        params, loaded_meta = load_stage25_inference_checkpoint(
            source_path, config=config, seed=seed,
            expected_e_history_version=expected_e_history_version,
            allow_legacy_e=allow_legacy_e)
        return params, loaded_meta
    if kind != BC_TRAINING_PAYLOAD_KIND:
        raise Stage25CheckpointError(
            f"{source_path}: PPO initialization requires native BC or inference "
            f"checkpoint, got {kind!r}")
    flat, loaded_meta, stored_config = _load_params(
        source_path, BC_TRAINING_PAYLOAD_KIND)
    _check_history(loaded_meta, expected=expected_e_history_version,
                   allow_legacy_e=allow_legacy_e, path=source_path)
    if config is not None and config != stored_config:
        raise Stage25CheckpointError(
            "source checkpoint config is incompatible with requested config")
    stored_seed = int(loaded_meta["init_params"]["seed"])
    if seed is not None and int(seed) != stored_seed:
        raise Stage25CheckpointError("source checkpoint seed does not match requested seed")
    params_template = init_stage25_params(stored_config, seed=stored_seed)
    params = _rebuild(
        params_template,
        {key[len("param:"):]: value for key, value in flat.items()
         if key.startswith("param:")})
    return params, dict(loaded_meta)


init_stage25_ppo_from_checkpoint = initialize_stage25_ppo_from_checkpoint


def _extract_encoder(params: Mapping[str, Any]) -> dict[str, Any]:
    names = ("manager_token", "role_embedding", "tile_encoder", "global_encoders",
             "encoder", "encoder_norm")
    source = params if "manager_token" in params else params.get("encoder", params)
    if not isinstance(source, Mapping):
        raise Stage25CheckpointError("historical checkpoint has no encoder mapping")
    missing = [name for name in names if name not in source]
    if missing:
        raise Stage25CheckpointError(f"historical E encoder missing {missing}")
    return {name: source[name] for name in names}


def import_historical_encoder(
    source: str | Path | Mapping[str, Any], config: Stage25ModelConfig, *,
    allow_legacy_e: bool = False,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import only the corrected-E encoder from a historical checkpoint.

    A path ending in ``.pt``/``.pth`` is the sole Torch conversion seam.
    Existing native JAX archives are loaded without importing Torch. Source
    dimensions/dtypes are checked strictly; there is no resize or tiling.
    """
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("config must be Stage25ModelConfig")
    source_meta: dict[str, Any] = {}
    if isinstance(source, Mapping):
        from bc_manager_jax.checkpoint import load_torch_checkpoint
        params, source_meta = load_torch_checkpoint(
            source, config.manager_config, model_variant="E",
            expected_e_history_version=(None if allow_legacy_e else expected_e_history_version))
    else:
        source_path = Path(source)
        suffix = source_path.suffix.lower()
        if suffix in (".pt", ".pth"):
            from bc_manager_jax.checkpoint import load_torch_checkpoint
            params, source_meta = load_torch_checkpoint(
                source_path, config.manager_config, model_variant="E",
                expected_e_history_version=(None if allow_legacy_e else expected_e_history_version))
        elif suffix == ".npz":
            _, native_meta = _read_archive(source_path)
            if native_meta.get("format") == STAGE25_CHECKPOINT_VERSION:
                if native_meta.get("payload_kind") == INFERENCE_PAYLOAD_KIND:
                    params, source_meta = load_stage25_inference_checkpoint(
                        source_path, config=config,
                        expected_e_history_version=expected_e_history_version,
                        allow_legacy_e=allow_legacy_e)
                else:
                    # Encoder import does not need optimizer reconstruction.
                    flat, source_meta, stored = _load_params(source_path, BC_TRAINING_PAYLOAD_KIND)
                    if stored != config:
                        raise Stage25CheckpointError("historical Stage 2.5 config is incompatible")
                    params = _rebuild(
                        init_stage25_params(stored, int(source_meta["init_params"]["seed"])),
                        {key[len("param:"):]: value for key, value in flat.items()
                         if key.startswith("param:")})
                    _check_history(source_meta, expected=expected_e_history_version,
                                   allow_legacy_e=allow_legacy_e, path=source_path)
            else:
                from bc_manager_jax.checkpoint import load_native
                params, source_meta = load_native(
                    source_path, config.manager_config, model_variant="E",
                    expected_e_history_version=(None if allow_legacy_e else expected_e_history_version))
        else:
            raise Stage25CheckpointError(
                f"{source_path}: native import requires .npz; Torch is reserved for explicit .pt paths")
    history = source_meta.get("e_history_version", E_HISTORY_LEGACY)
    try:
        history = normalize_e_history_version(history)
    except Exception as exc:
        raise Stage25CheckpointError("historical source has invalid E history identity") from exc
    if history == E_HISTORY_LEGACY and not allow_legacy_e:
        raise Stage25CheckpointError("legacy E import requires explicit allow_legacy_e=True")
    if expected_e_history_version is not None \
            and history != normalize_e_history_version(expected_e_history_version) \
            and not (allow_legacy_e and history == E_HISTORY_LEGACY):
        raise Stage25CheckpointError("historical E history version is incompatible")
    encoder = _extract_encoder(params)
    return encoder, {
        "source": _jsonable(source_meta),
        "source_identity": _source_identity(source, source_meta),
        "e_history_version": history,
        "e_identity": {"variant": "E", "history_version": history},
        "imported": "encoder_only",
        "discarded": ["heads", "optimizer_state", "decoder_state", "moments"],
    }


save_inference_checkpoint = save_stage25_inference_checkpoint
load_inference_checkpoint = load_stage25_inference_checkpoint
save_bc_checkpoint = save_stage25_bc_checkpoint
load_bc_checkpoint = load_stage25_bc_checkpoint
save_ppo_checkpoint = save_stage25_ppo_checkpoint
load_ppo_checkpoint = load_stage25_ppo_checkpoint


__all__ = [
    "ACTION_ORDER", "ACTION_CLASS_COUNTS", "ARCHITECTURE_VERSION",
    "BC_TARGET_VERSION", "BC_TRAINING_PAYLOAD_KIND", "INFERENCE_PAYLOAD_KIND",
    "PPO_RESUME_BOUNDARY", "PPO_TRAINING_PAYLOAD_KIND",
    "OBSERVATION_SCHEMA_VERSION", "OBSERVATION_VOCABULARY",
    "PERSISTENT_LEDGER_VERSION", "PHYSICAL_SUPPORT_VERSION", "RESUME_BOUNDARY",
    "STAGE25_CHECKPOINT_VERSION", "Stage25CheckpointError", "validate_array_tree",
    "save_stage25_inference_checkpoint", "load_stage25_inference_checkpoint",
    "save_stage25_bc_checkpoint", "load_stage25_bc_checkpoint",
    "save_stage25_ppo_checkpoint", "load_stage25_ppo_checkpoint",
    "initialize_stage25_ppo_from_checkpoint", "init_stage25_ppo_from_checkpoint",
    "save_inference_checkpoint", "load_inference_checkpoint", "save_bc_checkpoint",
    "load_bc_checkpoint", "save_ppo_checkpoint", "load_ppo_checkpoint",
    "import_historical_encoder",
]
