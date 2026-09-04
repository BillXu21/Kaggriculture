"""RL PPO checkpoint format `rl_manager_ppo_checkpoint_v2` (issue #9 req. 5).

A NEW rl_manager-native format — the issue-#8 BC native format
(`bc_manager_jax_checkpoint_v2`) is never altered or overloaded. One
pickle-free .npz archive (allow_pickle=False) with flattened parameter /
optimizer-state arrays plus a strict JSON metadata record:

- `param:<tree path>`  mutable combined params (base trunk + value head);
- `frozenparam:<path>` immutable frozen-E snapshot;
- `opt:<i>`            optimizer state leaves in template-flatten order;
- `rng`                explicit PRNG stream state, uint32 [2];
- `__meta__`           JSON: format, ManagerConfig, PPOConfig, variant=E,
                       E history version, step, rollout seed, provenance,
                       leaf counts.

Load reconstructs EVERYTHING from the strict expected template tree and
verifies each stored array's path/shape/dtype loudly; a resumed state runs
the next update bit-identically to the original.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_LEGACY,
    normalize_e_history_version,
)
from bc_manager_jax.model import (
    ManagerConfig,
    empty_params,
    resolve_model_variant,
)

from rl_manager.ppo_policy import (
    CurriculumMaskConfig,
    PPOConfig,
    combined_params_template,
    frozen_leaf_mask,
    make_ppo_optimizer,
)

RL_PPO_CHECKPOINT_FORMAT = "rl_manager_ppo_checkpoint_v2"
PPO_SNAPSHOT_FORMAT = "rl_manager_ppo_snapshot_v2"
_LEGACY_RL_PPO_CHECKPOINT_FORMAT = "rl_manager_ppo_checkpoint_v1"
_LEGACY_PPO_SNAPSHOT_FORMAT = "rl_manager_ppo_snapshot_v1"


def _leaf_path(tokens) -> str:
    parts = [str(getattr(entry, "key", None)
                 if getattr(entry, "key", None) is not None else entry.idx)
             for entry in tokens]
    return "/".join(parts)


def save_ppo_checkpoint(
    path: str | Path,
    state,
    config: ManagerConfig,
    ppo_config: PPOConfig,
    *,
    model_variant: str = "E",
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    provenance: Mapping[str, Any] | None = None,
    curriculum: CurriculumMaskConfig | None = None,
) -> Path:
    """Save train state + configs + provenance atomically, pickle-free."""
    variant = resolve_model_variant(model_variant)
    if variant != "E":
        raise ValueError(
            f"RL PPO checkpoints store variant E only, got {variant!r}")
    history_version = normalize_e_history_version(e_history_version)
    active_curriculum = curriculum or CurriculumMaskConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat: dict[str, np.ndarray] = {}
    for tokens, leaf in jax.tree_util.tree_flatten_with_path(state.params)[0]:
        flat["param:" + _leaf_path(tokens)] = np.asarray(leaf,
                                                         dtype=np.float32)
    for tokens, leaf in jax.tree_util.tree_flatten_with_path(
            state.frozen_params)[0]:
        flat["frozenparam:" + _leaf_path(tokens)] = np.asarray(
            leaf, dtype=np.float32)
    opt_leaves = jax.tree_util.tree_leaves(state.opt_state)
    for index, leaf in enumerate(opt_leaves):
        flat[f"opt:{index:05d}"] = np.asarray(leaf)
    rng = np.asarray(state.rng)
    if rng.shape != (2,) or rng.dtype != np.uint32:
        raise ValueError(f"state.rng must be a uint32 [2] PRNG key, got "
                         f"{rng.shape}/{rng.dtype}")
    flat["rng"] = rng

    checkpoint_provenance = dict(provenance) if provenance else {}
    checkpoint_provenance["curriculum"] = active_curriculum.to_json_dict()
    meta = {
        "format": RL_PPO_CHECKPOINT_FORMAT,
        "model_config": dataclasses.asdict(config),
        "ppo_config": dataclasses.asdict(ppo_config),
        "model_variant": variant,
        "e_history_version": history_version,
        "curriculum": active_curriculum.to_json_dict(),
        "step": int(state.step),
        "rollout_seed": (None if state.rollout_seed is None
                         else int(state.rollout_seed)),
        "provenance": checkpoint_provenance,
        "n_param_leaves": len(jax.tree_util.tree_leaves(state.params)),
        "n_frozen_leaves": len(jax.tree_util.tree_leaves(state.frozen_params)),
        "n_opt_leaves": len(opt_leaves),
    }
    flat["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)

    tmp = path.with_name(path.name + ".tmp")
    # Write through a file object so numpy cannot append its own extension.
    with open(tmp, "wb") as handle:
        np.savez(handle, **flat)
    path.unlink(missing_ok=True)
    tmp.rename(path)
    return path


def _require_matching_config(stored: Mapping, requested, name: str) -> None:
    if requested is None:
        return
    requested_dict = (dataclasses.asdict(requested)
                      if dataclasses.is_dataclass(requested)
                      else dict(requested))
    if requested_dict != dict(stored):
        raise ValueError(
            f"checkpoint {name} {dict(stored)!r} != requested "
            f"{requested_dict!r}; refusing to load an incompatible checkpoint")


def load_ppo_checkpoint(
    path: str | Path,
    *,
    config: ManagerConfig | None = None,
    ppo_config: PPOConfig | None = None,
    model_variant: str = "E",
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
):
    """Strictly reconstruct `(PPOTrainState, metadata)` from a checkpoint."""
    from rl_manager.ppo import PPOTrainState  # local import: avoid cycle

    variant = resolve_model_variant(model_variant)
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            items = {key: archive[key] for key in archive.files}
    except Exception as error:  # noqa: BLE001 - loud re-wrap below
        raise ValueError(
            f"{path}: corrupt or unreadable RL PPO checkpoint: {error}") \
            from error
    if "__meta__" not in items:
        raise ValueError(f"{path}: RL PPO checkpoint missing __meta__ record")
    meta = json.loads(items["__meta__"].tobytes().decode("utf-8"))
    if meta.get("format") not in (RL_PPO_CHECKPOINT_FORMAT,
                                   _LEGACY_RL_PPO_CHECKPOINT_FORMAT):
        raise ValueError(
            f"{path}: unrecognized RL PPO checkpoint format "
            f"{meta.get('format')!r}; expected one of "
            f"{RL_PPO_CHECKPOINT_FORMAT!r}/{_LEGACY_RL_PPO_CHECKPOINT_FORMAT!r}")
    stored_variant = meta.get("model_variant")
    if stored_variant != variant:
        raise ValueError(
            f"{path}: checkpoint stores model_variant {stored_variant!r} but "
            f"the requested variant is {variant!r}")
    stored_history = normalize_e_history_version(
        meta.get("e_history_version", E_HISTORY_LEGACY))
    stored_curriculum = CurriculumMaskConfig.from_json_dict(
        meta.get("curriculum") or meta.get("provenance", {}).get("curriculum"))
    meta["curriculum"] = stored_curriculum.to_json_dict()
    meta["provenance"] = dict(meta.get("provenance") or {})
    meta["provenance"].setdefault(
        "curriculum", stored_curriculum.to_json_dict())
    if expected_e_history_version is not None and stored_history != \
            normalize_e_history_version(expected_e_history_version):
        raise ValueError(
            f"{path}: checkpoint e_history_version {stored_history!r} does not "
            f"match requested "
            f"{normalize_e_history_version(expected_e_history_version)!r}")

    stored_model_config = ManagerConfig(**meta["model_config"])
    _require_matching_config(meta["model_config"], config, "model_config")
    stored_ppo_config = PPOConfig(**meta["ppo_config"])
    _require_matching_config(meta["ppo_config"], ppo_config, "ppo_config")

    def rebuild(prefix: str, template: Mapping) -> dict:
        flat_leaves = []
        for tokens, expected in jax.tree_util.tree_flatten_with_path(template)[0]:
            key = prefix + _leaf_path(tokens)
            if key not in items:
                raise ValueError(f"{path}: checkpoint missing array {key!r}")
            array = items[key]
            if array.shape != expected.shape or array.dtype != np.float32:
                raise ValueError(
                    f"{path}: array {key!r} has shape/dtype "
                    f"({array.shape}, {array.dtype}); expected "
                    f"({expected.shape}, float32)")
            flat_leaves.append(jnp.asarray(array))
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(template), flat_leaves)

    template = combined_params_template(stored_model_config, variant)
    params = rebuild("param:", template)
    frozen_params = rebuild("frozenparam:",
                            empty_params(stored_model_config, variant))

    mask = frozen_leaf_mask(template)
    opt_template = make_ppo_optimizer(stored_ppo_config, mask).init(params)
    opt_leaves_meta = jax.tree_util.tree_flatten(opt_template)[0]
    if meta.get("n_opt_leaves") != len(opt_leaves_meta):
        raise ValueError(
            f"{path}: optimizer leaf count {meta.get('n_opt_leaves')} != "
            f"expected {len(opt_leaves_meta)} for this param tree/config")
    opt_flat = []
    for index, expected in enumerate(opt_leaves_meta):
        key = f"opt:{index:05d}"
        if key not in items:
            raise ValueError(f"{path}: checkpoint missing array {key!r}")
        array = items[key]
        if array.shape != np.asarray(expected).shape \
                or array.dtype != np.asarray(expected).dtype:
            raise ValueError(
                f"{path}: optimizer array {key!r} has shape/dtype "
                f"({array.shape}, {array.dtype}); expected "
                f"{np.asarray(expected).shape}/"
                f"{np.asarray(expected).dtype}")
        opt_flat.append(jnp.asarray(array))
    opt_state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(opt_template), opt_flat)

    rng = items["rng"]
    if rng.shape != (2,) or rng.dtype != np.uint32:
        raise ValueError(f"{path}: rng array must be uint32 [2], got "
                         f"{rng.shape}/{rng.dtype}")

    state = PPOTrainState(params=params, opt_state=opt_state,
                          frozen_params=frozen_params,
                          rng=jnp.asarray(rng), step=int(meta["step"]),
                          rollout_seed=meta.get("rollout_seed"))
    return state, meta


def save_ppo_snapshot(
    path: str | Path,
    state,
    config: ManagerConfig,
    ppo_config: PPOConfig,
    *,
    snapshot_identity: Mapping[str, Any],
    e_history_version: str = E_HISTORY_CORRECTED_V1,
    provenance: Mapping[str, Any] | None = None,
    curriculum: CurriculumMaskConfig | None = None,
) -> Path:
    """Persist only the detached policy/frozen-E trees for a ratchet snapshot."""
    variant = resolve_model_variant("E")
    history_version = normalize_e_history_version(e_history_version)
    active_curriculum = curriculum or CurriculumMaskConfig()
    identity_history = snapshot_identity.get("e_history_version")
    if identity_history not in (None, history_version):
        raise ValueError("snapshot identity e_history_version does not match payload")
    identity_curriculum = snapshot_identity.get("curriculum")
    if identity_curriculum is not None and \
            identity_curriculum != active_curriculum.to_json_dict():
        raise ValueError("snapshot identity curriculum does not match payload")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for tokens, leaf in jax.tree_util.tree_flatten_with_path(state.params)[0]:
        flat["param:" + _leaf_path(tokens)] = np.asarray(leaf, dtype=np.float32)
    for tokens, leaf in jax.tree_util.tree_flatten_with_path(
            state.frozen_params)[0]:
        flat["frozenparam:" + _leaf_path(tokens)] = np.asarray(
            leaf, dtype=np.float32)
    snapshot_provenance = dict(provenance) if provenance else {}
    snapshot_provenance["curriculum"] = active_curriculum.to_json_dict()
    meta = {
        "format": PPO_SNAPSHOT_FORMAT,
        "model_config": dataclasses.asdict(config),
        "ppo_config": dataclasses.asdict(ppo_config),
        "model_variant": variant,
        "e_history_version": history_version,
        "curriculum": active_curriculum.to_json_dict(),
        "step": int(state.step),
        "snapshot_identity": dict(snapshot_identity),
        "provenance": snapshot_provenance,
    }
    flat["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **flat)
    path.unlink(missing_ok=True)
    tmp.rename(path)
    return path


def load_ppo_snapshot(
    path: str | Path,
    *,
    config: ManagerConfig | None = None,
    ppo_config: PPOConfig | None = None,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
):
    """Load a saved snapshot as a deterministic normal runner policy."""
    from rl_manager.ppo import PPOTrainState  # local import: avoid cycle
    from rl_manager.ppo_adapter import ppo_snapshot_from_state

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            items = {key: archive[key] for key in archive.files}
    except Exception as error:  # noqa: BLE001 - loud re-wrap below
        raise ValueError(
            f"{path}: corrupt or unreadable PPO snapshot: {error}") from error
    if "__meta__" not in items:
        raise ValueError(f"{path}: PPO snapshot missing __meta__ record")
    meta = json.loads(items["__meta__"].tobytes().decode("utf-8"))
    if meta.get("format") not in (PPO_SNAPSHOT_FORMAT,
                                   _LEGACY_PPO_SNAPSHOT_FORMAT):
        raise ValueError(
            f"{path}: unrecognized PPO snapshot format {meta.get('format')!r}")
    if meta.get("model_variant") != "E":
        raise ValueError(f"{path}: PPO snapshots require model variant E")
    stored_history = normalize_e_history_version(
        meta.get("e_history_version", E_HISTORY_LEGACY))
    curriculum_metadata_present = "curriculum" in meta
    stored_curriculum = CurriculumMaskConfig.from_json_dict(
        meta.get("curriculum") or meta.get("provenance", {}).get("curriculum"))
    meta["curriculum"] = stored_curriculum.to_json_dict()
    meta["provenance"] = dict(meta.get("provenance") or {})
    meta["provenance"].setdefault(
        "curriculum", stored_curriculum.to_json_dict())
    if expected_e_history_version is not None and stored_history != \
            normalize_e_history_version(expected_e_history_version):
        raise ValueError(
            f"{path}: snapshot e_history_version {stored_history!r} does not "
            f"match requested "
            f"{normalize_e_history_version(expected_e_history_version)!r}")
    stored_config = ManagerConfig(**meta["model_config"])
    _require_matching_config(meta["model_config"], config, "model_config")
    stored_ppo = PPOConfig(**meta["ppo_config"])
    _require_matching_config(meta["ppo_config"], ppo_config, "ppo_config")

    def rebuild(prefix: str, template: Mapping) -> dict:
        leaves = []
        for tokens, expected in jax.tree_util.tree_flatten_with_path(template)[0]:
            key = prefix + _leaf_path(tokens)
            if key not in items:
                raise ValueError(f"{path}: snapshot missing array {key!r}")
            array = items[key]
            if array.shape != expected.shape or array.dtype != np.float32:
                raise ValueError(
                    f"{path}: array {key!r} has shape/dtype "
                    f"({array.shape}, {array.dtype}); expected "
                    f"({expected.shape}, float32)")
            leaves.append(jnp.asarray(array))
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(template), leaves)

    params = rebuild("param:", combined_params_template(stored_config, "E"))
    frozen = rebuild("frozenparam:", empty_params(stored_config, "E"))
    state = PPOTrainState(
        params=params, opt_state=None, frozen_params=frozen,
        rng=jnp.zeros((2,), dtype=jnp.uint32), step=int(meta["step"]))
    identity = meta.get("snapshot_identity", {})
    policy = ppo_snapshot_from_state(
        state, stored_config, ppo_config=stored_ppo,
        name=str(identity.get("name", "ppo_snapshot")),
        version=str(identity.get("version", "ratchet-v1")),
        e_history_version=stored_history, curriculum=stored_curriculum)
    if identity.get("e_history_version") not in (None, stored_history):
        raise ValueError(f"{path}: snapshot identity history version mismatch")
    from rl_manager.policy import params_fingerprint
    expected_fingerprint = (params_fingerprint(state.params)
                            if not curriculum_metadata_present else
                            policy.identity.fingerprint)
    if identity.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"{path}: snapshot fingerprint does not match arrays")
    return policy, meta


__all__ = ["PPO_SNAPSHOT_FORMAT", "RL_PPO_CHECKPOINT_FORMAT",
           "load_ppo_checkpoint", "load_ppo_snapshot", "save_ppo_checkpoint",
           "save_ppo_snapshot"]
