"""Regression checks for import-time accelerator work in Stage 2.5 modules."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_stage25_accelerator_imports_do_not_construct_device_arrays():
    script = r"""
import importlib
import jax
import jax.numpy as jnp

def forbidden(*args, **kwargs):
    raise AssertionError("accelerator operation executed during module import")

for name in (
    "array", "asarray", "arange", "empty", "eye", "full", "identity",
    "ones", "ones_like", "zeros", "zeros_like",
):
    setattr(jnp, name, forbidden)
for name in ("default_backend", "device_put", "devices"):
    setattr(jax, name, forbidden)
for name in ("PRNGKey", "key", "split", "fold_in", "uniform", "normal",
             "categorical", "randint", "choice", "bernoulli"):
    if hasattr(jax.random, name):
        setattr(jax.random, name, forbidden)

for module in (
    "rl_manager.stage25_policy",
    "rl_manager.stage25_inference",
    "rl_manager.stage25_ppo",
    "rl_manager.stage25_checkpoint",
    "bc_manager_jax.model",
):
    importlib.import_module(module)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
