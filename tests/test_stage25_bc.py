"""Focused native Stage 2.5 teacher-forced BC tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_LEGACY
from rl_manager.stage25_bc import (
    Stage25BCBatch,
    Stage25BCConfig,
    init_opt_state,
    load_checkpoint,
    loss_and_metrics,
    make_fixed_batch,
    save_checkpoint,
    train_step,
)
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params


ROOT = Path(__file__).resolve().parents[1]


def _inputs(rows: int = 2):
    return {
        "board_kind": np.zeros((rows, 100), np.int16),
        "board_crop": np.zeros((rows, 100), np.int8),
        "board_animal": np.zeros((rows, 100), np.int8),
        "board_numeric": np.zeros((rows, 100, 11), np.float32),
        "board_bool": np.zeros((rows, 100, 8), bool),
        "board_mask": np.zeros((rows, 100, 4), np.uint8),
        "scalars": np.zeros((rows, 4), np.float32),
        "shed_counts": np.zeros((rows, 12), np.int32),
        "seed_counts": np.zeros((rows, 5), np.int32),
        "carried_counts": np.zeros((rows, 12), np.int32),
        "unlocked": np.tile([[1, 0, 0, 0]], (rows, 1)).astype(np.uint8),
        "market_inventory": np.zeros((rows, 9), np.int32),
        "market_prices": np.zeros((rows, 9), np.float32),
        "shop_counts": np.zeros((rows, 9), np.int32),
        "day": np.zeros((rows,), np.int16),
        "days_remaining": np.full((rows,), 29, np.int16),
        "economic_context": np.zeros((rows, 14), np.float32),
        "crop_capacity": np.zeros((rows, 5), np.int16),
    }


def _batch(rows: int = 2, batch_size: int = 2):
    actions = np.tile([[0, 0, 0, 0, 100, 100, 100, 100, 100]], (rows, 1))
    return make_fixed_batch(_inputs(rows), actions, batch_size)


def _flat_equal(left, right):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))


def test_invalid_labels_fail_before_update():
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2)
    params = init_stage25_params(config.model, seed=2)
    bad = np.tile([[4, 0, 0, 0, 100, 100, 100, 100, 100]], (2, 1))
    batch = Stage25BCBatch(_inputs(), bad, np.ones(2, bool))
    state = init_opt_state(params, config)
    with pytest.raises(ValueError, match="outside its vocabulary"):
        train_step(params, state, jax.random.PRNGKey(1), batch, config)


def test_padding_does_not_change_loss_or_metrics():
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=4)
    params = init_stage25_params(config.model, seed=3)
    a = loss_and_metrics(params, _batch(1, 4), config)
    b = loss_and_metrics(params, _batch(4, 4), config)
    assert np.isclose(a["loss"], b["loss"])
    assert np.allclose(a["per_step_nll"], b["per_step_nll"])
    assert np.allclose(a["per_step_accuracy"], b["per_step_accuracy"])


def test_tiny_update_changes_policy_but_not_value():
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-2, weight_decay=1e-1)
    params = init_stage25_params(config.model, seed=4)
    batch = _batch()
    state = init_opt_state(params, config)
    updated, _, _, _ = train_step(params, state, jax.random.PRNGKey(5), batch, config)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree_util.tree_leaves(params["output_projections"]),
                   jax.tree_util.tree_leaves(updated["output_projections"])))
    assert _flat_equal(params["value_head"], updated["value_head"])


def test_training_dropout_uses_shared_path_and_requires_rng():
    model = Stage25ModelConfig.tiny(dropout=0.1)
    config = Stage25BCConfig(model=model, batch_size=2, weight_decay=0.0)
    params = init_stage25_params(model, seed=14)
    batch = _batch()
    state = init_opt_state(params, config)
    train_step(params, state, jax.random.PRNGKey(15), batch, config)
    with pytest.raises(ValueError, match="explicit rng"):
        train_step(params, state, None, batch, config)


def test_resume_reproduces_next_update_exactly(tmp_path):
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=1e-3, weight_decay=0.0)
    params = init_stage25_params(config.model, seed=16)
    state = init_opt_state(params, config)
    batch = _batch()
    params, state, rng, _ = train_step(
        params, state, jax.random.PRNGKey(17), batch, config)
    direct = train_step(params, state, rng, batch, config)
    path = tmp_path / "resume.npz"
    save_checkpoint(path, params, state, rng, config=config, step=1, epoch=0,
                    shuffle_state={"epoch": 0, "batch": 1})
    loaded = load_checkpoint(path, config=config)
    resumed = train_step(loaded[0], loaded[1], loaded[2], batch, config)
    assert _flat_equal(direct[0], resumed[0])
    assert _flat_equal(direct[1], resumed[1])
    assert np.array_equal(np.asarray(direct[2]), np.asarray(resumed[2]))


def test_loss_smoke_and_fitting_reduces_loss():
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2,
                             lr=3e-3, weight_decay=0.0)
    params = init_stage25_params(config.model, seed=6)
    batch = _batch()
    initial = float(loss_and_metrics(params, batch, config)["loss"])
    state = init_opt_state(params, config)
    rng = jax.random.PRNGKey(7)
    for _ in range(3):
        params, state, rng, _ = train_step(params, state, rng, batch, config)
    final = float(loss_and_metrics(params, batch, config)["loss"])
    assert np.isfinite(initial) and np.isfinite(final)
    assert final < initial


@pytest.mark.xfail(
    strict=True,
    reason=(
        "stage25_bc.save_checkpoint silently drops e_history_version/"
        "source_identity/provenance metadata that collide with reserved "
        "checkpoint keys, so a legacy-history E transfer is recorded as "
        "corrected-history parity with empty provenance."
    ),
)
def test_bc_wrapper_preserves_import_history_and_source_identity(tmp_path):
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2)
    params = init_stage25_params(config.model, seed=41)
    state = init_opt_state(params, config)
    path = tmp_path / "history.npz"
    save_checkpoint(path, params, state, jax.random.PRNGKey(41), config=config,
                    step=1, epoch=0, metadata={
                        "e_history_version": E_HISTORY_LEGACY,
                        "source_identity": {"checkpoint": "historical-e"},
                        "provenance": {"run": "review"},
                    })
    meta = load_checkpoint(path, config=config)[3]
    assert meta["e_history_version"] == E_HISTORY_LEGACY
    assert meta["source_identity"] == {"checkpoint": "historical-e"}
    assert meta["provenance"] == {"run": "review"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the high-level BC wrapper exposes no explicit legacy-history opt-in, "
        "so an explicitly legacy E BC checkpoint cannot be resumed."
    ),
)
def test_bc_wrapper_resumes_explicit_legacy_history_checkpoint(tmp_path):
    from rl_manager.stage25_checkpoint import save_stage25_bc_checkpoint

    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2)
    params = init_stage25_params(config.model, seed=43)
    state = init_opt_state(params, config)
    path = tmp_path / "legacy.npz"
    save_stage25_bc_checkpoint(path, params, state,
                               np.asarray([7, 9], dtype=np.uint32), config.model,
                               seed=43, step=2, epoch=0, optimizer_config=config,
                               e_history_version=E_HISTORY_LEGACY)
    loaded = load_checkpoint(path, config=config)
    assert _flat_equal(params, loaded[0])


def test_native_save_load_and_torch_blocked_startup(tmp_path):
    config = Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2)
    params = init_stage25_params(config.model, seed=8)
    state = init_opt_state(params, config)
    path = tmp_path / "bc.npz"
    save_checkpoint(path, params, state, jax.random.PRNGKey(9), config=config,
                    step=4, epoch=2)
    loaded, loaded_state, rng, meta = load_checkpoint(path, config=config)
    assert _flat_equal(params, loaded)
    assert _flat_equal(state, loaded_state)
    assert np.array_equal(np.asarray(rng), np.array([0, 9], np.uint32))
    assert meta["step"] == 4
    script = f"""
import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Torch imported by native BC startup')
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
from rl_manager.stage25_bc import Stage25BCConfig, load_checkpoint
from rl_manager.stage25_policy import Stage25ModelConfig
load_checkpoint(r'{path}', config=Stage25BCConfig(model=Stage25ModelConfig.tiny(), batch_size=2))
print('native-ok')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
