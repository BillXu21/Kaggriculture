"""Focused contract tests for native Stage 2.5 PPO persistence."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_LEGACY
from bc_manager_jax.train import TrainConfig
from rl_manager import stage25_checkpoint as checkpoint
from rl_manager.stage25_bc import init_opt_state
from rl_manager.stage25_checkpoint import (
    PPO_RESUME_BOUNDARY,
    PPO_TRAINING_PAYLOAD_KIND,
    Stage25CheckpointError,
    initialize_stage25_ppo_from_checkpoint,
    load_stage25_ppo_checkpoint,
    save_stage25_bc_checkpoint,
    save_stage25_inference_checkpoint,
    save_stage25_ppo_checkpoint,
)
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


def _config() -> Stage25ModelConfig:
    return Stage25ModelConfig.tiny()


def _same_tree(left, right) -> bool:
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree_util.tree_leaves(left),
                                jax.tree_util.tree_leaves(right)))


def _ppo_state(config: Stage25ModelConfig):
    params = init_stage25_params(config, seed=7)
    optimizer_config = TrainConfig()
    return params, init_opt_state(params, optimizer_config)


def test_ppo_round_trip_is_distinct_and_preserves_resume_contract(
        tmp_path: Path) -> None:
    config = _config()
    params, optimizer_state = _ppo_state(config)
    path = tmp_path / "ppo.npz"
    ppo_config = {"gamma": 0.99, "gae_lambda": 0.95, "epochs": 2}
    behavior = {
        "name": "stage25-ppo", "version": "candidate-1",
        "parameter_fingerprint": "p" * 64,
    }
    physical = {
        "version": "stage25_physical_v1", "executor": "executor-v07",
    }
    save_stage25_ppo_checkpoint(
        path, params, optimizer_state, np.asarray([11, 13], dtype=np.uint32),
        config, seed=7, update_counter=12, rollout_seed=101,
        rollout_progression={"episode": 4, "next_day": 9},
        ppo_config=ppo_config, behavior_identity=behavior,
        physical_contract=physical, provenance={"run": "focused-test"},
        executor={"profile": "native-test"},
        source_identity={"checkpoint": "bc-seed-7"},
        source_history_version=E_HISTORY_LEGACY,
    )

    loaded, loaded_opt, rng, meta = load_stage25_ppo_checkpoint(
        path, config=config, seed=7, optimizer_state_template=optimizer_state,
        ppo_config=ppo_config, expected_behavior_identity=behavior,
        expected_physical_contract=physical)

    assert _same_tree(params, loaded)
    assert _same_tree(optimizer_state, loaded_opt)
    assert np.array_equal(np.asarray(rng), np.asarray([11, 13], dtype=np.uint32))
    assert meta["payload_kind"] == PPO_TRAINING_PAYLOAD_KIND
    assert meta["resume_boundary"] == PPO_RESUME_BOUNDARY
    assert meta["update_counter"] == 12
    assert meta["rollout_seed"] == 101
    assert meta["rollout_progression"] == {"episode": 4, "next_day": 9}
    assert meta["behavior_identity"] == behavior
    assert meta["provenance"] == {"run": "focused-test"}
    assert meta["source_identity"] == {"checkpoint": "bc-seed-7"}
    assert meta["source_e_identity"]["history_version"] == E_HISTORY_LEGACY
    with np.load(path, allow_pickle=False) as archive:
        assert "__meta__" in archive.files
        assert all(archive[name].dtype != object for name in archive.files)
        assert json.loads(archive["__meta__"].tobytes().decode("utf-8"))["payload_kind"] \
            == PPO_TRAINING_PAYLOAD_KIND


@pytest.mark.xfail(
    strict=True,
    reason=(
        "save_stage25_ppo_checkpoint accepts a curriculum argument but never "
        "uses it; it silently persists config.curriculum instead of rejecting "
        "a mismatched explicit curriculum, so the load-time curriculum check "
        "can only ever see the model-config curriculum"
    ),
)
def test_ppo_checkpoint_rejects_explicit_curriculum_mismatch(tmp_path: Path) -> None:
    from rl_manager.stage25_config import Stage25CurriculumConfig

    config = _config()  # disabled curriculum
    params, optimizer_state = _ppo_state(config)
    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    with pytest.raises(Stage25CheckpointError, match="curriculum"):
        save_stage25_ppo_checkpoint(
            tmp_path / "mismatch.npz", params, optimizer_state,
            np.asarray([2, 3], dtype=np.uint32), config, seed=7,
            curriculum=enabled, ppo_config={"x": 1})


def test_fresh_ppo_initialization_accepts_native_inference_and_bc(
        tmp_path: Path) -> None:
    config = _config()
    params, optimizer_state = _ppo_state(config)
    inference = tmp_path / "inference.npz"
    bc = tmp_path / "bc.npz"
    save_stage25_inference_checkpoint(inference, params, config, seed=7)
    save_stage25_bc_checkpoint(
        bc, params, optimizer_state, np.asarray([1, 2], dtype=np.uint32), config,
        seed=7, step=3, epoch=1, optimizer_config=TrainConfig())

    from_inference, inference_meta = initialize_stage25_ppo_from_checkpoint(
        inference, config=config, seed=7)
    from_bc, bc_meta = initialize_stage25_ppo_from_checkpoint(
        bc, config=config, seed=7)
    assert _same_tree(params, from_inference)
    assert _same_tree(params, from_bc)
    assert inference_meta["payload_kind"] != PPO_TRAINING_PAYLOAD_KIND
    assert bc_meta["payload_kind"] != PPO_TRAINING_PAYLOAD_KIND


def test_ppo_rejects_incompatible_config_missing_extra_and_bad_rng(
        tmp_path: Path) -> None:
    config = _config()
    params, optimizer_state = _ppo_state(config)
    original = tmp_path / "original.npz"
    save_stage25_ppo_checkpoint(
        original, params, optimizer_state, np.asarray([2, 3], dtype=np.uint32),
        config, seed=7, ppo_config={"gamma": 0.99})
    with pytest.raises(Stage25CheckpointError, match="config is incompatible"):
        load_stage25_ppo_checkpoint(
            original, config=Stage25ModelConfig.tiny(output_init_scale=0.03),
            optimizer_state_template=optimizer_state)

    with np.load(original, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    items.pop(next(key for key in items if key.startswith("param:")))
    missing = tmp_path / "missing.npz"
    with open(missing, "wb") as handle:
        np.savez(handle, **items)
    with pytest.raises(Stage25CheckpointError, match="leaf mismatch|tree mismatch"):
        load_stage25_ppo_checkpoint(missing, optimizer_state_template=optimizer_state)

    with np.load(original, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    items["unexpected"] = np.zeros((), dtype=np.float32)
    extra = tmp_path / "extra.npz"
    with open(extra, "wb") as handle:
        np.savez(handle, **items)
    with pytest.raises(Stage25CheckpointError, match="unexpected leaves|leaf mismatch"):
        load_stage25_ppo_checkpoint(extra, optimizer_state_template=optimizer_state)

    with pytest.raises(Stage25CheckpointError, match="forbidden object"):
        save_stage25_ppo_checkpoint(
            tmp_path / "object.npz", params, (np.asarray(["opaque"], dtype=object),),
            np.asarray([2, 3], dtype=np.uint32), config)
    with pytest.raises(Stage25CheckpointError, match="rng must be uint32"):
        save_stage25_ppo_checkpoint(
            tmp_path / "bad-rng.npz", params, optimizer_state,
            np.asarray([2, 3], dtype=np.int32), config)


def test_atomic_ppo_replace_failure_preserves_previous_archive(
        tmp_path: Path, monkeypatch) -> None:
    config = _config()
    params, optimizer_state = _ppo_state(config)
    path = tmp_path / "atomic-ppo.npz"
    save_stage25_ppo_checkpoint(path, params, optimizer_state,
                                np.asarray([2, 3], dtype=np.uint32), config)
    before = path.read_bytes()

    def fail_replace(_temporary, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_stage25_ppo_checkpoint(path, params, optimizer_state,
                                    np.asarray([2, 3], dtype=np.uint32), config)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
