"""Focused contract tests for the native Stage 2.5 checkpoint boundary."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_CORRECTED_V1, E_HISTORY_LEGACY
from bc_manager_jax.checkpoint import save_native
from bc_manager_jax.model import init_train_params
from bc_manager_jax.train import TrainConfig, init_opt_state
from rl_manager import stage25_checkpoint as checkpoint
from rl_manager.stage25_checkpoint import (
    BC_TRAINING_PAYLOAD_KIND,
    INFERENCE_PAYLOAD_KIND,
    RESUME_BOUNDARY,
    Stage25CheckpointError,
    import_historical_encoder,
    load_stage25_bc_checkpoint,
    load_stage25_inference_checkpoint,
    save_stage25_bc_checkpoint,
    save_stage25_inference_checkpoint,
)
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


def _config() -> Stage25ModelConfig:
    return Stage25ModelConfig.tiny()


def _same_tree(left, right) -> bool:
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree_util.tree_leaves(left),
                                jax.tree_util.tree_leaves(right)))


def test_inference_round_trip_persists_native_contract(tmp_path: Path) -> None:
    config = _config()
    params = init_stage25_params(config, seed=17)
    path = tmp_path / "inference.npz"

    save_stage25_inference_checkpoint(
        path, params, config, seed=17,
        source_identity={"checkpoint": "historical-e"},
        provenance={"run": "focused-test"},
        executor={"profile": "native-test"},
    )
    loaded, meta = load_stage25_inference_checkpoint(path, config=config)

    assert _same_tree(params, loaded)
    assert meta["payload_kind"] == INFERENCE_PAYLOAD_KIND
    assert meta["e_identity"]["history_version"] == E_HISTORY_CORRECTED_V1
    assert meta["source_identity"]["checkpoint"] == "historical-e"
    assert meta["executor"]["profile"] == "native-test"
    with np.load(path, allow_pickle=False) as archive:
        assert "__meta__" in archive.files
        assert all(archive[name].dtype != object for name in archive.files)
        json.loads(archive["__meta__"].tobytes().decode("utf-8"))


def test_bc_round_trip_restores_optimizer_rng_and_resume_boundary(tmp_path: Path) -> None:
    config = _config()
    params = init_stage25_params(config, seed=19)
    optimizer_config = TrainConfig()
    opt_state = init_opt_state(params, optimizer_config)
    path = tmp_path / "bc.npz"

    save_stage25_bc_checkpoint(
        path, params, opt_state, np.asarray([3, 5], dtype=np.uint32), config,
        seed=19, step=8, epoch=2, optimizer_config=optimizer_config,
        data_order_position={"epoch": 2, "batch": 4},
    )
    loaded, loaded_opt, rng, meta = load_stage25_bc_checkpoint(
        path, config=config)

    assert _same_tree(params, loaded)
    assert _same_tree(opt_state, loaded_opt)
    assert np.array_equal(np.asarray(rng), np.asarray([3, 5], dtype=np.uint32))
    assert meta["payload_kind"] == BC_TRAINING_PAYLOAD_KIND
    assert meta["step"] == 8 and meta["epoch"] == 2
    assert meta["resume_boundary"] == RESUME_BOUNDARY
    assert meta["data_order_position"] == {"epoch": 2, "batch": 4, "seed": 19}


def test_missing_extra_corrupt_and_version_incompatible_leaves_reject(tmp_path: Path) -> None:
    config = _config()
    params = init_stage25_params(config, seed=23)
    original = tmp_path / "original.npz"
    save_stage25_inference_checkpoint(original, params, config, seed=23)
    with np.load(original, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}

    missing = tmp_path / "missing.npz"
    missing_items = dict(items)
    missing_items.pop(next(key for key in missing_items if key.startswith("param:")))
    with open(missing, "wb") as handle:
        np.savez(handle, **missing_items)
    with pytest.raises(Stage25CheckpointError, match="leaf mismatch|tree mismatch"):
        load_stage25_inference_checkpoint(missing, config=config)

    extra = tmp_path / "extra.npz"
    extra_items = dict(items)
    extra_items["param:unexpected"] = np.zeros((), dtype=np.float32)
    with open(extra, "wb") as handle:
        np.savez(handle, **extra_items)
    with pytest.raises(Stage25CheckpointError, match="leaf mismatch"):
        load_stage25_inference_checkpoint(extra, config=config)

    bad_version = tmp_path / "version.npz"
    bad_items = dict(items)
    meta = json.loads(bad_items["__meta__"].tobytes().decode("utf-8"))
    meta["format"] = "stage25_native_checkpoint_v0"
    bad_items["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with open(bad_version, "wb") as handle:
        np.savez(handle, **bad_items)
    with pytest.raises(Stage25CheckpointError, match="version-incompatible"):
        load_stage25_inference_checkpoint(bad_version, config=config)


def test_atomic_replace_failure_preserves_last_good_checkpoint(tmp_path: Path,
                                                               monkeypatch) -> None:
    config = _config()
    params = init_stage25_params(config, seed=29)
    path = tmp_path / "atomic.npz"
    save_stage25_inference_checkpoint(path, params, config, seed=29)
    before = path.read_bytes()

    def fail_replace(_tmp, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_stage25_inference_checkpoint(path, params, config, seed=29)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_encoder_only_native_import_is_torch_free_and_discards_heads(tmp_path: Path,
                                                                     monkeypatch) -> None:
    config = _config()
    historical = init_train_params(config.manager_config, seed=31, model_variant="E")
    path = tmp_path / "historical-native.npz"
    save_native(path, historical, config.manager_config, model_variant="E",
                e_history_version=E_HISTORY_CORRECTED_V1)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("native Stage 2.5 import attempted Torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    encoder, meta = import_historical_encoder(path, config)
    seeded = init_stage25_params(config, seed=99, encoder_params=encoder)

    assert set(encoder) == {
        "manager_token", "role_embedding", "tile_encoder", "global_encoders",
        "encoder", "encoder_norm",
    }
    assert meta["imported"] == "encoder_only"
    assert "moments" in meta["discarded"]
    for name in encoder:
        assert _same_tree(encoder[name], seeded["encoder"][name])
    assert "heads" not in seeded


def test_legacy_e_native_import_requires_explicit_opt_in(tmp_path: Path) -> None:
    config = _config()
    historical = init_train_params(config.manager_config, seed=37, model_variant="E")
    path = tmp_path / "legacy-native.npz"
    save_native(path, historical, config.manager_config, model_variant="E",
                e_history_version=E_HISTORY_LEGACY)

    with pytest.raises(ValueError, match="e_history_version"):
        import_historical_encoder(path, config)
    encoder, meta = import_historical_encoder(path, config, allow_legacy_e=True)
    assert meta["e_history_version"] == E_HISTORY_LEGACY
    assert encoder
