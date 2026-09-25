"""Focused contract tests for native Stage 2.5 PPO persistence."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_LEGACY
from bc_manager_jax.train import TrainConfig
from rl_manager.stage25_ppo import Stage25PPOConfig, make_stage25_ppo_optimizer
from rl_manager import stage25_checkpoint as checkpoint
from rl_manager.stage25_checkpoint import (
    PPO_RESUME_BOUNDARY,
    PPO_TRAINING_PAYLOAD_KIND,
    Stage25CheckpointError,
    initialize_stage25_ppo_from_checkpoint,
    migrate_stage25_bc_checkpoint_for_ppo,
    load_stage25_bc_checkpoint,
    load_stage25_ppo_checkpoint,
    save_stage25_bc_checkpoint,
    save_stage25_inference_checkpoint,
    save_stage25_ppo_checkpoint,
)
from rl_manager.stage25_policy import (
    Stage25ModelConfig,
    evaluate_actions,
    greedy_act,
    init_stage25_params,
)
from rl_manager.stage25_mechanics import PhysicalContext
from rl_manager.stage25_ppo import make_stage25_ppo_optimizer
from rl_manager.stage25_bc import (
    Stage25BCBatch,
    Stage25BCConfig,
    init_opt_state,
    train_step,
)


def _config() -> Stage25ModelConfig:
    return Stage25ModelConfig.tiny()


def _same_tree(left, right) -> bool:
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree_util.tree_leaves(left),
                                jax.tree_util.tree_leaves(right)))


def _write_legacy_physical_bc(path: Path, architecture: str) -> tuple[dict, dict]:
    config = _config()
    params = init_stage25_params(config, seed=29)
    # The migration source used the old tree; optimizer contents are present
    # only to make this a realistic BC training archive and are discarded.
    save_stage25_bc_checkpoint(
        path, params, (np.zeros((1,), dtype=np.float32),),
        np.asarray([1, 2], dtype=np.uint32), config, seed=29, step=7, epoch=2,
        optimizer_config={"name": "discarded-test-optimizer"},
        source_identity={"checkpoint": "physical-baseline-bc"},
        provenance={"dataset": "physical-morning-baseline"},
    )
    with np.load(path, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    meta = json.loads(items.pop("__meta__").tobytes().decode("utf-8"))
    missing = (["replaceable_conditioning",
                "available_crop_slots_conditioning"]
               if architecture == "stage25_policy_v1" else
               ["available_crop_slots_conditioning"])
    for name in missing:
        items.pop(f"param:{name}")
    old_observation = (
        "stage25_corrected_e_own_only_v1"
        if architecture == "stage25_policy_v1" else
        "stage25_corrected_e_own_only_replaceable_today_v2")
    vocab = checkpoint.OBSERVATION_VOCABULARY[
        :-2 if architecture == "stage25_policy_v1" else -1]
    meta.update({
        "architecture_version": architecture,
        "observation_schema_version": old_observation,
        "observation_vocabulary": list(vocab),
        "crop_baseline_semantics": "physical_morning_board_counts",
        "bc_target": checkpoint.BC_TARGET_VERSION,
    })
    meta["e_identity"]["observation_schema_version"] = old_observation
    meta["leaf_manifest"] = checkpoint._leaf_manifest(items)
    items["__meta__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode(),
        dtype=np.uint8)
    with open(path, "wb") as handle:
        np.savez(handle, **items)
    params_flat = checkpoint._flatten_arrays(params)
    for name in missing:
        params_flat.pop(name)
    return params_flat, meta


def _migration_inputs() -> dict[str, np.ndarray]:
    rows = 1
    return {
        "board_kind": np.zeros((rows, 100), dtype=np.int16),
        "board_crop": np.zeros((rows, 100), dtype=np.int8),
        "board_animal": np.zeros((rows, 100), dtype=np.int8),
        "board_numeric": np.zeros((rows, 100, 11), dtype=np.float32),
        "board_bool": np.zeros((rows, 100, 8), dtype=np.bool_),
        "board_mask": np.zeros((rows, 100, 4), dtype=np.uint8),
        "scalars": np.zeros((rows, 4), dtype=np.float32),
        "shed_counts": np.zeros((rows, 12), dtype=np.int32),
        "seed_counts": np.zeros((rows, 5), dtype=np.int32),
        "carried_counts": np.zeros((rows, 12), dtype=np.int32),
        "unlocked": np.asarray([[1, 0, 0, 0]], dtype=np.uint8),
        "market_inventory": np.zeros((rows, 9), dtype=np.int32),
        "market_prices": np.zeros((rows, 9), dtype=np.float32),
        "shop_counts": np.zeros((rows, 9), dtype=np.int32),
        "day": np.asarray([4], dtype=np.int16),
        "days_remaining": np.asarray([25], dtype=np.int16),
        "economic_context": np.zeros((rows, 14), dtype=np.float32),
        "crop_capacity": np.asarray([[2, 0, 0, 0, 0]], dtype=np.int16),
        "replaceable_today": np.asarray([[3, 0, 0, 0, 0]], dtype=np.int16),
        "available_crop_slots": np.asarray([61], dtype=np.int16),
    }


def _ppo_state(config: Stage25ModelConfig):
    params = init_stage25_params(config, seed=7)
    ppo_config = Stage25PPOConfig(model=config, physical_batch_size=1,
                                  minibatch_size=1, epochs=1)
    return params, make_stage25_ppo_optimizer(params, ppo_config).init(params)


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

    loaded, loaded_opt, rng, meta, loaded_opponent = load_stage25_ppo_checkpoint(
        path, config=config, seed=7, optimizer_state_template=optimizer_state,
        ppo_config=ppo_config, expected_behavior_identity=behavior,
        expected_physical_contract=physical, return_opponent=True)

    assert _same_tree(params, loaded)
    assert _same_tree(optimizer_state, loaded_opt)
    assert _same_tree(params, loaded_opponent)
    optimizer = make_stage25_ppo_optimizer(
        params, Stage25PPOConfig(model=config, physical_batch_size=1,
                                 minibatch_size=1, epochs=1))
    gradients = jax.tree_util.tree_map(np.ones_like, params)
    update_a, next_opt_a = optimizer.update(gradients, optimizer_state, params)
    update_b, next_opt_b = optimizer.update(gradients, loaded_opt, loaded)
    assert _same_tree(update_a, update_b)
    assert _same_tree(next_opt_a, next_opt_b)
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


def test_ppo_checkpoint_rejects_explicit_curriculum_mismatch(tmp_path: Path) -> None:
    from rl_manager.stage25_config import Stage25CurriculumConfig

    config = _config()  # disabled curriculum
    params, optimizer_state = _ppo_state(config)
    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    existing = tmp_path / "existing.npz"
    save_stage25_ppo_checkpoint(
        existing, params, optimizer_state, np.asarray([2, 3], dtype=np.uint32),
        config, seed=7, ppo_config={"x": 1})
    before = existing.read_bytes()
    with pytest.raises(Stage25CheckpointError, match="curriculum"):
        save_stage25_ppo_checkpoint(
            existing, params, optimizer_state,
            np.asarray([2, 3], dtype=np.uint32), config, seed=7,
            curriculum=enabled, ppo_config={"x": 1})
    assert existing.read_bytes() == before


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


@pytest.mark.parametrize(
    "architecture,zero_leaves,preserved",
    [
        ("stage25_policy_v1",
         ["available_crop_slots_conditioning", "replaceable_conditioning"],
         []),
        ("stage25_policy_v2_replaceable_today",
         ["available_crop_slots_conditioning"], ["replaceable_conditioning"]),
    ],
)
def test_explicit_physical_bc_migration_is_weights_only_and_preserves_policy(
        tmp_path: Path, architecture: str, zero_leaves: list[str],
        preserved: list[str]) -> None:
    config = _config()
    path = tmp_path / f"{architecture}.npz"
    source_params, _ = _write_legacy_physical_bc(path, architecture)

    migrated, metadata = migrate_stage25_bc_checkpoint_for_ppo(
        path, config=config)
    migrated_flat = checkpoint._flatten_arrays(migrated)
    for name, value in source_params.items():
        assert np.array_equal(value, migrated_flat[name]), name
    for name in zero_leaves:
        assert np.array_equal(migrated_flat[name],
                              np.zeros_like(migrated_flat[name]))
    assert metadata["transfer"] == "weights_only_architecture_migration"
    assert metadata["resumable"] is False
    assert metadata["optimizer_state_discarded"] is True
    assert metadata["source_metadata"]["provenance"] == {
        "dataset": "physical-morning-baseline"}
    assert metadata["architecture_migration"][
        "zero_initialized_conditioning_leaves"] == zero_leaves
    assert metadata["architecture_migration"][
        "preserved_conditioning_leaves"] == preserved

    # v1 never observed either value; v2 already observed replaceable_today.
    # The new observations are deliberately nonzero in `inputs`. With the
    # missing conditioning leaves zero, all old policy outputs remain exact.
    inputs = _migration_inputs()
    reference_inputs = dict(inputs)
    reference_inputs["available_crop_slots"] = np.asarray([0], dtype=np.int16)
    if "replaceable_conditioning" in zero_leaves:
        reference_inputs["replaceable_today"] = np.zeros((1, 5), dtype=np.int16)
    physical = (PhysicalContext(1, (25, 50, 75, 100), (0, 0, 0)),)
    old_output = greedy_act(
        migrated, reference_inputs, config, physical_contexts=physical)
    new_output = greedy_act(
        migrated, inputs, config, physical_contexts=physical)
    assert jax.tree_util.tree_structure(old_output) == \
        jax.tree_util.tree_structure(new_output)
    for old, new in zip(jax.tree_util.tree_leaves(old_output),
                        jax.tree_util.tree_leaves(new_output)):
        assert np.array_equal(np.asarray(old), np.asarray(new))
    old_eval = evaluate_actions(
        migrated, reference_inputs, config,
        classes=np.asarray(old_output["classes"], dtype=np.int16),
        physical_contexts=physical)
    new_eval = evaluate_actions(
        migrated, inputs, config,
        classes=np.asarray(old_output["classes"], dtype=np.int16),
        physical_contexts=physical)
    for old, new in zip(jax.tree_util.tree_leaves(old_eval),
                        jax.tree_util.tree_leaves(new_eval)):
        assert np.array_equal(np.asarray(old), np.asarray(new))

    ppo_config = Stage25PPOConfig(model=config, physical_batch_size=1,
                                  minibatch_size=1, epochs=1)
    optimizer = make_stage25_ppo_optimizer(migrated, ppo_config)
    optimizer_state = optimizer.init(migrated)
    assert jax.tree_util.tree_leaves(optimizer_state)

    if architecture == "stage25_policy_v1":
        batch = Stage25BCBatch(
            inputs=inputs,
            actions=np.asarray([[0, 0, 0, 0, 100, 100, 100, 100, 100]],
                               dtype=np.int16),
            real_row_mask=np.asarray([True]), physical_contexts=physical)
        bc_config = Stage25BCConfig(model=config, dropout=0.0)
        bc_optimizer = init_opt_state(migrated, bc_config)
        updated, _, _, metrics = train_step(
            migrated, bc_optimizer, np.asarray([4, 9], dtype=np.uint32),
            batch, bc_config)
        assert np.isfinite(metrics["loss"])
        assert not np.array_equal(
            np.asarray(updated["available_crop_slots_conditioning"]),
            np.asarray(migrated["available_crop_slots_conditioning"]))

    # Strict normal resume/load still rejects the older architecture.
    with pytest.raises(Stage25CheckpointError, match="architecture_version"):
        load_stage25_bc_checkpoint(path, config=config)


@pytest.mark.parametrize("metadata_change,error", [
    ({"crop_baseline_semantics": None}, "explicit physical morning"),
    ({"bc_target": "stage25_outcome_proxy_v1"}, "BC target semantics"),
])
def test_explicit_migration_rejects_unproven_or_synthetic_baseline(
        tmp_path: Path, metadata_change: dict[str, str | None], error: str):
    path = tmp_path / "unproven-or-synthetic-baseline.npz"
    _write_legacy_physical_bc(path, "stage25_policy_v2_replaceable_today")
    with np.load(path, allow_pickle=False) as archive:
        items = {key: archive[key] for key in archive.files}
    meta = json.loads(items.pop("__meta__").tobytes().decode("utf-8"))
    for key, value in metadata_change.items():
        if value is None:
            meta.pop(key)
        else:
            meta[key] = value
    items["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    with open(path, "wb") as handle:
        np.savez(handle, **items)
    with pytest.raises(Stage25CheckpointError, match=error):
        migrate_stage25_bc_checkpoint_for_ppo(path, config=_config())


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
