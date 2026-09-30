"""Regression tests for the zero-cost Stage 2.5 parameter template.

`stage25_params_template` must be structurally identical to
`init_stage25_params` (same leaf names, shapes and dtypes) because the
checkpoint loader uses that tree only to validate leaves and to rebuild stored
arrays into it.  `load_stage25_inference_checkpoint` must therefore return
byte-identical parameters either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

from rl_manager.stage25_checkpoint import (  # noqa: E402
    _flatten_arrays,
    load_stage25_inference_checkpoint,
)
from rl_manager.stage25_policy import (  # noqa: E402
    Stage25ModelConfig,
    init_stage25_params,
    stage25_params_template,
)

CONFIGS = (
    Stage25ModelConfig.tiny(),
    Stage25ModelConfig.small(),
    Stage25ModelConfig.large(),
)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"d{c.d_model}")
def test_template_matches_init_structure(config):
    template = _flatten_arrays(stage25_params_template(config))
    reference = _flatten_arrays(init_stage25_params(config, seed=0))
    assert set(template) == set(reference)
    for key, leaf in template.items():
        assert leaf.shape == reference[key].shape, key
        assert leaf.dtype == reference[key].dtype, key


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"d{c.d_model}")
def test_template_rejects_non_config(config):
    with pytest.raises(TypeError):
        stage25_params_template(object())  # type: ignore[arg-type]


def _tree_signature(tree) -> dict[str, tuple[tuple[int, ...], str, bytes]]:
    flat = _flatten_arrays(tree)
    return {
        key: (tuple(np.asarray(leaf).shape), str(np.asarray(leaf).dtype),
              np.asarray(leaf).tobytes())
        for key, leaf in flat.items()
    }


def test_inference_load_is_bit_identical_to_random_template(tmp_path):
    """The shipped checkpoint must load identically with either template."""

    from rl_manager.stage25_checkpoint import (
        INFERENCE_PAYLOAD_KIND,
        _load_params,
        _rebuild,
    )

    checkpoint = Path(
        r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\stage25_bc_7m_best_inference.npz")
    if not checkpoint.is_file():
        pytest.skip("canonical checkpoint is unavailable")

    fast_flat, meta, stored_config = _load_params(
        checkpoint, INFERENCE_PAYLOAD_KIND, zero_template=True)
    zero_template = stage25_params_template(stored_config)
    rebuilt = _rebuild(
        zero_template,
        {key[len("param:"):]: value for key, value in fast_flat.items()
         if key.startswith("param:")},
    )

    reference_template = init_stage25_params(
        stored_config, seed=int(meta["init_params"]["seed"]))
    reference = _rebuild(
        reference_template,
        {key[len("param:"):]: value for key, value in fast_flat.items()
         if key.startswith("param:")},
    )

    assert _tree_signature(rebuilt) == _tree_signature(reference)

    # And the public entry point agrees with the reference construction.
    loaded, _ = load_stage25_inference_checkpoint(checkpoint)
    assert _tree_signature(loaded) == _tree_signature(reference)


def test_template_is_immutable_like_a_real_parameter_tree():
    """Template leaves are plain arrays; nothing may alias across calls."""
    import jax.numpy as jnp

    config = Stage25ModelConfig.tiny()
    first = stage25_params_template(config)
    second = stage25_params_template(config)
    with pytest.raises(TypeError):
        jax.tree_util.tree_leaves(first)[0][...] = 1.0
    # Two independent calls must not share any leaf object.
    first_ids = {id(leaf) for leaf in jax.tree_util.tree_leaves(first)}
    second_ids = {id(leaf) for leaf in jax.tree_util.tree_leaves(second)}
    assert not (first_ids & second_ids)
    assert all(jnp.all(leaf == 0) for leaf in jax.tree_util.tree_leaves(second))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
