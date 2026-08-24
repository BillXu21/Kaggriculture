"""Stage-2 focused tests: JAX BC train step and replicated data parallelism.

Covers: single-device step semantics (params update, finite metrics,
deterministic key advancement), exact clip->AdamW order against a manual
reference implementation, one-step agreement with the existing PyTorch
BC training semantics (dropout=0), and a forced multi-CPU subprocess
proving 1-vs-4 device numerical equivalence and correct batch sharding.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bc_manager_jax.benchmark import synthetic_batch
from bc_manager_jax.loss import loss_from_validated, validate_target_shapes
from bc_manager_jax.model import (
    ManagerConfig,
    _Dropout,
    _forward_core,
    _prepare_inputs,
    init_params,
    tiny_manager_config,
)
from bc_manager_jax.train import (
    TrainConfig,
    init_opt_state,
    make_optimizer,
    train_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- single device


def _fresh_state(seed=0, dropout=0.1, batch=8):
    config = tiny_manager_config(dropout=dropout)
    params = init_params(config, seed=seed)
    opt_state = init_opt_state(params, TrainConfig())
    inputs, targets = synthetic_batch(config, batch, seed=seed + 1)
    return config, params, opt_state, inputs, targets


def test_train_step_updates_params_and_losses_are_finite():
    config, params, opt_state, inputs, targets = _fresh_state()
    rng = jax.random.PRNGKey(0)
    new_params, new_opt, next_rng, total, groups = train_step(
        params, opt_state, rng, inputs, targets, config)
    assert set(groups) == {
        "crop", "animal", "land", "fertilizer", "care",
        "sell_presence", "sell_quantity"}
    assert np.isfinite(float(total))
    assert all(np.isfinite(float(v)) for v in groups.values())
    leaves_before = jax.tree_util.tree_leaves(params)
    leaves_after = jax.tree_util.tree_leaves(new_params)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(leaves_before, leaves_after))


def test_step_is_deterministic_and_key_advances():
    config, params, opt_state, inputs, targets = _fresh_state()
    rng = jax.random.PRNGKey(42)
    a = train_step(params, opt_state, rng, inputs, targets, config)
    b = train_step(params, opt_state, rng, inputs, targets, config)
    assert float(a[3]) == float(b[3])
    assert all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(a[0]),
                               jax.tree_util.tree_leaves(b[0])))
    # Advancing with the returned key yields fresh dropout masks: the same
    # params/opt state stepped with rng vs next_rng differ deterministically.
    c = train_step(params, opt_state, a[2], inputs, targets, config)
    assert float(c[3]) != float(a[3])
    c_again = train_step(params, opt_state, a[2], inputs, targets, config)
    assert float(c[3]) == float(c_again[3])


def test_clip_then_adamw_matches_manual_reference_exactly():
    """Semantic check: global-norm clip is applied BEFORE AdamW."""
    config, params, _, inputs, targets = _fresh_state(dropout=0.0)
    train_config = TrainConfig()

    validated = validate_target_shapes(
        targets, inputs["board_kind"].shape[0], config.count_classes)
    prepared_inputs = _prepare_inputs(inputs)

    def loss_fn(p):
        outputs = _forward_core(p, prepared_inputs, config,
                                _Dropout(0.0, None))
        return loss_from_validated(outputs, validated)[0]

    _, grads = jax.value_and_grad(loss_fn)(params)
    flat_grads = [np.asarray(g) for g in jax.tree_util.tree_leaves(grads)]
    global_norm = float(np.sqrt(sum(float(np.sum(g * g))
                                    for g in flat_grads)))
    scale = min(1.0, train_config.gradient_clip / global_norm)

    # Manual AdamW (decoupled weight decay, bias correction, t=1).
    lr, wd, b1, b2, eps = (train_config.lr, train_config.weight_decay,
                           train_config.beta1, train_config.beta2,
                           train_config.eps)
    leaves_p, treedef = jax.tree_util.tree_flatten(params)
    leaves_g = jax.tree_util.tree_leaves(grads)
    updated = []
    for p_leaf, g_leaf in zip(leaves_p, leaves_g):
        p = np.asarray(p_leaf, dtype=np.float64)
        g = np.asarray(g_leaf, dtype=np.float64) * scale
        m = (1 - b1) * g
        v = (1 - b2) * g * g
        m_hat = m / (1 - b1)
        v_hat = v / (1 - b2)
        updated.append(
            (p - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * p))
            .astype(np.float32))
    reference = jax.tree_util.tree_unflatten(treedef, updated)

    opt_state = init_opt_state(params, train_config)
    stepped, _, _, _, _ = train_step(params, opt_state,
                                     jax.random.PRNGKey(0), inputs,
                                     targets, config, train_config)
    for got, want in zip(jax.tree_util.tree_leaves(stepped),
                         jax.tree_util.tree_leaves(reference)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=1e-5, atol=1e-7)


def test_converted_params_do_not_alias_torch_storage():
    """Regression: conversion must copy; in-place torch updates (optimizer
    steps) must never silently mutate converted JAX parameters."""
    import torch

    from bc_manager.model import DailyManagerTransformer, \
        tiny_manager_config as torch_tiny_config
    from bc_manager_jax.checkpoint import convert_torch_state_dict

    torch.manual_seed(3)
    torch_model = DailyManagerTransformer(torch_tiny_config(dropout=0.0))
    config = tiny_manager_config(dropout=0.0)
    params = convert_torch_state_dict(torch_model.state_dict(), config)
    snapshot = [np.array(np.asarray(leaf), copy=True)
                for leaf in jax.tree_util.tree_leaves(params)]
    with torch.no_grad():
        for p in torch_model.parameters():
            p.add_(1.0)
    for leaf, was in zip(jax.tree_util.tree_leaves(params), snapshot):
        np.testing.assert_array_equal(np.asarray(leaf), was)


def test_one_step_matches_pytorch_adamw_clip_semantics():
    """Proportional cross-framework check (dropout=0).

    Losses must agree tightly before the step. Parameter updates are only
    bounded-agreement: Adam's first-step normalization divides by |g|, so
    float32 gradient noise (~1e-6 relative from stage-1 parity) can flip
    near-zero coordinates, giving per-coordinate differences up to ~2*lr.
    """
    import torch

    from bc_manager.loss import manager_loss as torch_manager_loss
    from bc_manager.model import DailyManagerTransformer, tiny_manager_config \
        as torch_tiny_config
    from bc_manager_jax.checkpoint import convert_torch_state_dict

    torch.manual_seed(0)
    torch_model = DailyManagerTransformer(
        torch_tiny_config(dropout=0.0))
    torch_model.eval()
    config = tiny_manager_config(dropout=0.0)
    params = convert_torch_state_dict(torch_model.state_dict(), config)

    inputs_np, targets_np = synthetic_batch(config, 8, seed=1)
    torch_inputs = {k: torch.from_numpy(np.ascontiguousarray(v))
                    for k, v in inputs_np.items()}
    torch_targets = {k: torch.from_numpy(np.ascontiguousarray(v))
                     for k, v in targets_np.items()}

    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=3e-4,
                                  weight_decay=1e-2)
    with torch.no_grad():
        pass
    outputs = torch_model(torch_inputs)
    torch_loss, _ = torch_manager_loss(outputs, torch_targets)
    optimizer.zero_grad()
    torch_loss.backward()
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
    optimizer.step()

    opt_state = init_opt_state(params, TrainConfig())
    new_params, _, _, jax_loss, _ = train_step(
        params, opt_state, jax.random.PRNGKey(0), inputs_np, targets_np,
        config)

    assert abs(float(torch_loss.detach()) - float(jax_loss)) < 1e-4
    diffs = [
        np.abs(np.asarray(a) - np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(new_params),
                        jax.tree_util.tree_leaves(convert_torch_state_dict(
                            torch_model.state_dict(), config)))
    ]
    flat = np.concatenate([d.reshape(-1) for d in diffs])
    print(f"one-step param diff vs torch: max={flat.max():.3e} "
          f"mean={flat.mean():.3e}")
    assert flat.max() <= 2 * 3e-4 + 1e-6  # bounded by Adam's step size
    assert flat.mean() <= 1e-4


# ------------------------------------------------------ multi device


MULTI_DEVICE_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import numpy as np, jax
assert jax.device_count() == 4, jax.device_count()
from bc_manager_jax.benchmark import synthetic_batch
from bc_manager_jax.model import init_params, tiny_manager_config
from bc_manager_jax.sharding import (create_data_mesh, replicate_tree,
                                     shard_batch, describe_sharding)
from bc_manager_jax.train import TrainConfig, init_opt_state, train_step

config = tiny_manager_config(dropout=0.1)
params = init_params(config, seed=0)
inputs_np, targets_np = synthetic_batch(config, 8, seed=1)
rng = jax.random.PRNGKey(0)

# Single device reference.
opt_state_1 = init_opt_state(params, TrainConfig())
p1, _, _, total1, groups1 = train_step(params, opt_state_1, rng,
                                       inputs_np, targets_np, config)

# Four-device replicated data parallel.
mesh = create_data_mesh(4)
sharded_inputs = shard_batch(inputs_np, mesh)
sharded_targets = shard_batch(targets_np, mesh)
in_spec = str(next(iter(sharded_inputs.values())).sharding.spec)
p4, _, _, total4, groups4 = train_step(
    replicate_tree(params, mesh),
    replicate_tree(init_opt_state(params, TrainConfig()), mesh),
    rng, sharded_inputs, sharded_targets, config)

param_close = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                  for a, b in zip(jax.tree_util.tree_leaves(p1),
                                  jax.tree_util.tree_leaves(p4)))
print("RESULT:" + json.dumps({
    "total1": float(total1), "total4": float(total4),
    "total_abs_diff": abs(float(total1) - float(total4)),
    "group_max_abs_diff": max(abs(float(groups1[k]) - float(groups4[k]))
                              for k in groups1),
    "param_max_abs_diff": param_close,
    "batch_sharding_spec": in_spec,
    "param_sharding_spec": describe_sharding(
        {k: v for k, v in list(replicate_tree(params, mesh).items())[:1]}),
}))
"""


def test_four_device_replicated_step_matches_single_device(tmp_path):
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["JAX_PLATFORMS"] = "cpu"
    script = tmp_path / "multi_device_check.py"
    script.write_text(MULTI_DEVICE_SCRIPT, encoding="utf-8")
    completed = subprocess.run([sys.executable, str(script)], env=env,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result_line = next(line for line in completed.stdout.splitlines()
                       if line.startswith("RESULT:"))
    result = json.loads(result_line[len("RESULT:"):])
    print(json.dumps(result, indent=2))
    # Global-batch losses agree between 1 and 4 devices.
    assert result["total_abs_diff"] < 1e-4
    assert result["group_max_abs_diff"] < 1e-4
    # Updated params agree numerically.
    assert result["param_max_abs_diff"] < 1e-5
    # Batch arrays are sharded along the 'data' axis; params replicated.
    # (JAX abbreviates PartitionSpec as P(...) in reprs.)
    assert "'data'" in result["batch_sharding_spec"]
    assert "P()" in json.dumps(result["param_sharding_spec"])


def test_mesh_and_batch_validation_errors():
    from bc_manager_jax import sharding

    with pytest.raises(ValueError, match="divisible"):
        sharding.check_global_batch(10, 4)
    with pytest.raises(ValueError, match="device_count"):
        sharding.create_data_mesh(len(jax.devices()) + 1)


def test_non_divisible_global_batch_rejected_in_benchmark_case():
    from bc_manager_jax.benchmark import run_case

    args = type("Args", (), {})()
    args.dtype = "f32"
    args.warmup = 0
    args.iterations = 1
    args.seed = 0
    row = run_case("own", tiny_manager_config(), None, "random init",
                   device_count=1, global_batch=3, args=args)
    assert row["status"] == "ok"  # divisible by 1
    row = run_case("own", tiny_manager_config(), None, "random init",
                   device_count=2, global_batch=3, args=args)
    assert row["status"] == "skipped"
    assert "not divisible" in row["reason"]


# --------------------------------------------------- issue #8: E variant


def test_e_variant_single_device_jit_forward_and_train_step():
    """Tiny JIT E forward + one train step on one device; finite, updates."""
    config = tiny_manager_config(dropout=0.1)
    params = init_params(config, seed=70, model_variant="E")
    opt_state = init_opt_state(params, TrainConfig())
    inputs_np, targets_np = synthetic_batch(config, 4, seed=71,
                                            model_variant="E")
    econ = np.asarray(inputs_np["economic_context"])
    assert econ.shape == (4, 14) and np.isfinite(econ).all()

    outputs = _forward_core(params, _prepare_inputs(inputs_np), config,
                            _Dropout(0.0, None), "E")
    assert all(np.isfinite(np.asarray(leaf)).all()
               for leaf in jax.tree_util.tree_leaves(outputs))

    rng = jax.random.PRNGKey(0)
    new_params, new_opt, next_rng, total, groups = train_step(
        params, opt_state, rng, inputs_np, targets_np, config,
        model_variant="E")
    assert set(groups) == {"crop", "animal", "land", "fertilizer",
                           "care", "sell_presence", "sell_quantity"}
    assert np.isfinite(float(total))
    assert all(np.isfinite(float(v)) for v in groups.values())
    leaves_before = jax.tree_util.tree_leaves(params)
    leaves_after = jax.tree_util.tree_leaves(new_params)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(leaves_before, leaves_after))


E_MULTI_DEVICE_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
expected_devices = int(os.environ["EXPECTED_DEVICES"])
compare_single = os.environ["COMPARE_SINGLE"] == "1"
variant = os.environ.get("MODEL_VARIANT", "E")
import numpy as np, jax
assert jax.device_count() == expected_devices, jax.device_count()
from bc_manager_jax.benchmark import synthetic_batch
from bc_manager_jax.model import init_params, tiny_manager_config
from bc_manager_jax.sharding import (create_data_mesh, replicate_tree,
                                     shard_batch)
from bc_manager_jax.train import TrainConfig, init_opt_state, train_step

config = tiny_manager_config(dropout=0.1)
params = init_params(config, seed=0, model_variant=variant)
inputs_np, targets_np = synthetic_batch(config, 8, seed=1,
                                        model_variant=variant)
rng = jax.random.PRNGKey(0)

result = {"devices": expected_devices, "variant": variant}
if compare_single:
    opt_state_1 = init_opt_state(params, TrainConfig())
    p1, _, _, total1, groups1 = train_step(params, opt_state_1, rng,
                                           inputs_np, targets_np, config,
                                           model_variant=variant)

mesh = create_data_mesh(expected_devices)
sharded_inputs = shard_batch(inputs_np, mesh)
sharded_targets = shard_batch(targets_np, mesh)
in_spec = str(next(iter(sharded_inputs.values())).sharding.spec)
p_n, _, _, total_n, groups_n = train_step(
    replicate_tree(params, mesh),
    replicate_tree(init_opt_state(params, TrainConfig()), mesh),
    rng, sharded_inputs, sharded_targets, config, model_variant=variant)

if compare_single:
    param_close = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                      for a, b in zip(jax.tree_util.tree_leaves(p1),
                                      jax.tree_util.tree_leaves(p_n)))
    result.update({
        "total_abs_diff": abs(float(total1) - float(total_n)),
        "group_max_abs_diff": max(abs(float(groups1[k]) - float(groups_n[k]))
                                  for k in groups1),
        "param_max_abs_diff": param_close,
    })
result["total_finite"] = bool(np.isfinite(float(total_n)))
result["batch_sharding_spec"] = in_spec
print("RESULT:" + json.dumps(result))
"""


def _run_e_multi_device_script(tmp_path, expected_devices: int,
                               compare_single: bool) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = \
        f"--xla_force_host_platform_device_count={expected_devices}"
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["JAX_PLATFORMS"] = "cpu"
    env["EXPECTED_DEVICES"] = str(expected_devices)
    env["COMPARE_SINGLE"] = "1" if compare_single else "0"
    env["MODEL_VARIANT"] = "E"
    script = tmp_path / f"e_multi_device_{expected_devices}.py"
    script.write_text(E_MULTI_DEVICE_SCRIPT, encoding="utf-8")
    completed = subprocess.run([sys.executable, str(script)], env=env,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result_line = next(line for line in completed.stdout.splitlines()
                       if line.startswith("RESULT:"))
    result = json.loads(result_line[len("RESULT:"):])
    print(json.dumps(result, indent=2))
    return result


def test_four_device_e_replicated_step_matches_single_device(tmp_path):
    """Routine N=4 logical-CPU NamedSharding train-path check for E."""
    result = _run_e_multi_device_script(tmp_path, 4, compare_single=True)
    assert result["total_finite"]
    assert result["total_abs_diff"] < 1e-4
    assert result["group_max_abs_diff"] < 1e-4
    assert result["param_max_abs_diff"] < 1e-5
    assert "'data'" in result["batch_sharding_spec"]


def test_eight_device_e_logical_smoke(tmp_path):
    """One bounded N=8 logical-CPU smoke (tiny batch, single step).

    Logical multi-device validation ONLY — not a throughput/scaling claim.
    """
    result = _run_e_multi_device_script(tmp_path, 8, compare_single=False)
    assert result["total_finite"]
    assert "'data'" in result["batch_sharding_spec"]
