"""Tiny-CPU acceptance tests for the Stage 2.5 JAX policy contract.

These tests exercise the public functional seam only. Physical expected values
come from the framework-free Packet 1 helpers.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
    PhysicalContext,
    animal_target_support_mask,
    crop_delta_support_mask,
    physical_context_from_board,
    physical_crop_capacity,
    land_target_support_mask,
)
from rl_manager.stage25_policy import (
    Stage25ModelConfig,
    evaluate_actions,
    greedy_act,
    init_stage25_params,
    parameter_spec,
    stage25_parameter_count,
    stochastic_act,
)


ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_KEYS = {
    "classes", "component_logprobs", "joint_logprob",
    "conditional_entropies", "prefix_entropy_surrogate", "value",
}


def _config(**overrides):
    return Stage25ModelConfig.tiny(**overrides)


def _encoded(batch_size: int = 1, *, economics: float = 0.0,
             capacity: np.ndarray | None = None,
             goals: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Small corrected-E-shaped arrays; no Parquet or Torch dependency."""
    b = int(batch_size)
    inputs = {
        "board_kind": np.zeros((b, 100), dtype=np.int16),
        "board_crop": np.zeros((b, 100), dtype=np.int8),
        "board_animal": np.zeros((b, 100), dtype=np.int8),
        "board_numeric": np.zeros((b, 100, 11), dtype=np.float32),
        "board_bool": np.zeros((b, 100, 8), dtype=bool),
        "board_mask": np.zeros((b, 100, 4), dtype=np.uint8),
        "scalars": np.full((b, 4), economics, dtype=np.float32),
        "shed_counts": np.zeros((b, 12), dtype=np.int32),
        "seed_counts": np.zeros((b, 5), dtype=np.int32),
        "carried_counts": np.zeros((b, 12), dtype=np.int32),
        "unlocked": np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8), (b, 1)),
        "market_inventory": np.zeros((b, 9), dtype=np.int32),
        "market_prices": np.full((b, 9), economics, dtype=np.float32),
        "shop_counts": np.zeros((b, 9), dtype=np.int32),
        "day": np.zeros((b,), dtype=np.int16),
        "days_remaining": np.full((b,), 29, dtype=np.int16),
        "economic_context": np.full((b, 14), economics, dtype=np.float32),
        "crop_capacity": (np.full((b, 5), 20, dtype=np.int16)
                          if capacity is None else np.asarray(capacity)),
        "crop_goals": (np.zeros((b, 5), dtype=np.int16)
                       if goals is None else np.asarray(goals)),
    }
    return inputs


def _encoded_board(board: list[list[object]], unlocked: tuple[str, ...], **kwargs):
    """Encode only the physical fields consumed by the JAX seam."""
    from bc_manager.constants import TILE_KIND_IDS

    result = _encoded(1, **kwargs)
    for index, tile in enumerate(sum(board, [])):
        if tile is None:
            continue
        if isinstance(tile, dict):
            kind = tile.get("kind", "UNKNOWN")
            result["board_kind"][0, index] = TILE_KIND_IDS.get(kind, TILE_KIND_IDS["UNKNOWN"])
            result["board_mask"][0, index, 0] = 1
            crop = tile.get("crop")
            if crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
                result["board_crop"][0, index] = (
                    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"
                ).index(crop) + 1
            animal = tile.get("animal")
            if animal in ("GOOSE", "COW", "SHEEP"):
                result["board_animal"][0, index] = ("GOOSE", "COW", "SHEEP").index(animal) + 1
                result["board_mask"][0, index, 2] = 1
        elif tile in TILE_KIND_IDS:
            result["board_kind"][0, index] = TILE_KIND_IDS[tile]
            result["board_mask"][0, index, 0] = 1
    result["unlocked"][0] = np.array([
        int(q in unlocked) for q in ("NW", "NE", "SW", "SE")], dtype=np.uint8)
    return result


def _board(*, future_land: bool = False, unknown: bool = False,
           housing: bool = False) -> list[list[object]]:
    board = [["LOCKED"] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            board[y][x] = None
    board[0][0] = "WEED"
    board[0][1] = {"kind": "PLANT", "crop": "WHEAT"}
    if unknown:
        board[0][2] = {"mystery": True}
    if housing:
        board[1][0] = {"kind": "COOP"}
        board[1][1] = {"kind": "PASTURE"}
        board[1][2] = {"kind": "PASTURE", "animal": "COW"}
    if future_land:
        for y in range(5):
            for x in range(5, 10):
                board[y][x] = None
    return board


def _contexts() -> tuple[PhysicalContext, ...]:
    return (
        physical_context_from_board(
            _board(unknown=True, housing=True), ("NW",),
            unplaced_animals=(1, 1, 0)),
        physical_context_from_board(
            _board(future_land=True), ("NW", "NE",),
            unplaced_animals=(0, 0, 0)),
        PhysicalContext(1, (0, 0, 0, 0), (0, 0, 0), 0, 0, (0, 0, 0)),
    )


def _invoke(fn, *, params, inputs, config, key=None, contexts=None,
            capacity=None, classes=None, row_ids=None):
    """Pass contract values by documented names while accepting kw/pos style."""
    sig = inspect.signature(fn)
    values = {
        "params": params, "config": config, "inputs": inputs,
        "observations": inputs, "encoded": inputs,
        "physical_context": contexts, "physical_contexts": contexts,
        "contexts": contexts, "crop_capacity": capacity,
        "capacities": capacity, "rng": key, "key": key, "rng_keys": key,
        "row_ids": row_ids, "classes": classes, "actions": classes,
        "action_classes": classes,
    }
    kwargs = {}
    positional = []
    for name, parameter in sig.parameters.items():
        if name in values and values[name] is not None:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(values[name])
            elif parameter.kind is not inspect.Parameter.VAR_KEYWORD:
                kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise AssertionError(f"unrecognized required Stage 2.5 API argument: {name}")
    return fn(*positional, **kwargs)


def _as_output(result):
    assert isinstance(result, dict)
    assert _OUTPUT_KEYS <= set(result), sorted(set(_OUTPUT_KEYS) - set(result))
    return result


def _valid(result):
    # The required public spelling is checked separately; this alias keeps the
    # semantic checks useful while the implementation worker finishes that
    # output rename.
    return np.asarray(result.get("valid", result.get("validity")))


def test_exact_action_schema_and_parameter_tree_accounting():
    config = _config()
    assert ACTION_ORDER == (
        "land", "goose", "cow", "sheep", "wheat", "carrot", "tomato",
        "strawberry", "melon")
    assert ACTION_CLASS_COUNTS == (4, 101, 101, 101, 201, 201, 201, 201, 201)
    spec = parameter_spec(config)
    params = init_stage25_params(config, seed=7)
    assert jax.tree_util.tree_structure(spec) == jax.tree_util.tree_structure(params)
    for expected, actual in zip(jax.tree_util.tree_leaves(spec),
                                jax.tree_util.tree_leaves(params)):
        assert tuple(expected.shape) == tuple(actual.shape)
    assert stage25_parameter_count(config) == sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(spec))
    decoder = sum(ACTION_CLASS_COUNTS) * (2 * config.d_model + 1)
    assert decoder == 1312 * (2 * config.d_model + 1)


def test_fresh_encoder_preserves_functional_layernorm_and_rejects_fractional_classes():
    config = _config()
    params = init_stage25_params(config, seed=8)
    for layer in params["encoder"]["encoder"]["layers"]:
        assert np.all(np.asarray(layer["norm1_weight"]) == 1.0)
        assert np.all(np.asarray(layer["norm2_weight"]) == 1.0)
    assert np.all(np.asarray(params["encoder"]["encoder_norm"]["weight"]) == 1.0)
    with pytest.raises(ValueError, match="integer class"):
        evaluate_actions(
            params, _encoded(1), config,
            classes=np.full((1, 9), 1.5, dtype=np.float32),
        )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_jax_support_matches_packet1_python_helpers(index):
    config = _config()
    params = init_stage25_params(config, seed=3)
    contexts = _contexts()
    capacities = np.array([[9] * 5, [25] * 5, [0] * 5], dtype=np.int16)
    goals = np.array([[0] * 5, [0] * 5, [60, 60, 60, 60, 60]], dtype=np.int16)
    # Keep the three rows physically distinct: occupied/empty housing, future
    # unlocked land, and an all-unknown zero-capacity footprint.
    physical = [
        _encoded_board(_board(unknown=True, housing=True), ("NW",),
                       capacity=capacities[:1], goals=goals[:1]),
        _encoded_board(_board(future_land=True), ("NW", "NE"),
                       capacity=capacities[1:2], goals=goals[1:2]),
        _encoded_board([[{"mystery": True}] * 5 + ["LOCKED"] * 5
                        for _ in range(5)] + [["LOCKED"] * 10 for _ in range(5)],
                       ("NW",), capacity=capacities[2:3],
                       goals=goals[2:3]),
    ]
    inputs = {key: np.concatenate([row[key] for row in physical], axis=0)
              for key in physical[0]}
    result = _as_output(_invoke(
        greedy_act, params=params, inputs=inputs, config=config))
    counts = np.asarray(result["diagnostics"]["support_counts"])
    masks = np.asarray(result["masks"])
    context = contexts[index]
    expected_land = np.asarray(land_target_support_mask(context.observed_land))
    assert counts[index, 0] == expected_land.sum()
    assert np.array_equal(masks[index, 0, :4], expected_land)
    land_target = int(np.asarray(result["classes"])[index, 0]) + 1
    prefix = []
    for step in range(3):
        expected = animal_target_support_mask(context, land_target, step, prefix)
        assert counts[index, step + 1] == sum(expected)
        assert np.array_equal(masks[index, step + 1, :101], expected)
        prefix.append(int(np.asarray(result["classes"])[index, step + 1]))
    goals = np.asarray(inputs["crop_goals"])[index].astype(np.int32)
    total_capacity = int(capacities[index].sum())
    for step in range(5):
        expected = crop_delta_support_mask(
            int(goals[step]), total_capacity - int(goals[:step].sum()))
        assert counts[index, step + 4] == sum(expected)
        assert np.array_equal(masks[index, step + 4, :201], expected)
        cls = int(np.asarray(result["classes"])[index, step + 4])
        goals[step] += cls - 100


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Packet 2 crop support uses sum(crop_capacity) as the residual budget "
        "and ignores the Packet 1A physical C reachable from the supplied "
        "physical_contexts. A caller that reproduces the scalar Packet 1A C "
        "across the five capacity slots may decode a crop-goal sum up to 5*C."
    ),
)
def test_jax_crop_goal_sum_is_bounded_by_packet1_physical_capacity():
    config = _config()
    params = init_stage25_params(config, seed=101)
    # Two footprints with different physical crop capacity: B=25 vs B=0.
    full = [[None] * 5 + ["LOCKED"] * 5 for _ in range(5)] + [
        ["LOCKED"] * 10 for _ in range(5)]
    blocked = [["LOCKED"] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            blocked[y][x] = {"kind": "COOP"}
    ctx_full = physical_context_from_board(full, ("NW",))
    ctx_blocked = physical_context_from_board(blocked, ("NW",))
    capacity_full = physical_crop_capacity(ctx_full, 1, (0, 0, 0))
    capacity_blocked = physical_crop_capacity(ctx_blocked, 1, (0, 0, 0))
    assert capacity_full == 25 and capacity_blocked == 0
    # Reproduce the scalar Packet 1A C in each of the five supplied slots.
    inputs = _encoded(1, capacity=np.full((1, 5), capacity_full, dtype=np.int16),
                      goals=np.zeros((1, 5), dtype=np.int16))
    # Wheat goal 40 exceeds the physical C=25 but not sum(crop_capacity)=125.
    classes = np.array([[0, 0, 0, 0, 140, 100, 100, 100, 100]], dtype=np.int16)
    out = evaluate_actions(
        params, inputs, config, classes=classes, reject_invalid=False,
        physical_contexts=(ctx_full,))
    assert not bool(np.asarray(out["valid"])[0])


def test_encoded_observation_physical_seam_is_economic_support_invariant():
    config = _config()
    params = init_stage25_params(config, seed=11)
    physical = _contexts()[0]
    encoded = _encoded_board(_board(unknown=True, housing=True), ("NW",),
                             economics=0.0, capacity=np.full((1, 5), 10, dtype=np.int16))
    economic_only = _encoded_board(_board(unknown=True, housing=True), ("NW",),
                                   economics=999.0, capacity=np.full((1, 5), 10, dtype=np.int16))
    a = _as_output(_invoke(greedy_act, params=params, inputs=encoded, config=config,
                           contexts=None, row_ids=np.array([42])))
    b = _as_output(_invoke(greedy_act, params=params, inputs=economic_only, config=config,
                           contexts=(physical,), row_ids=np.array([42])))
    assert np.array_equal(np.asarray(a["diagnostics"]["support_counts"]),
                          np.asarray(b["diagnostics"]["support_counts"]))


def test_enabled_curriculum_is_applied_after_physical_support():
    params = init_stage25_params(_config(), seed=12)
    context = (_contexts()[1],)
    inputs = _encoded(1, capacity=np.full((1, 5), 20, dtype=np.int16))
    disabled = _as_output(_invoke(
        greedy_act, params=params, inputs=inputs, config=_config(),
        contexts=context))
    enabled_config = _config(curriculum=Stage25CurriculumConfig(
        enabled=True, max_land_expansion_per_decision=0,
        max_animal_additions_per_species_per_decision=0,
        max_positive_crop_delta=3))
    enabled = _as_output(_invoke(
        greedy_act, params=params, inputs=inputs, config=enabled_config,
        contexts=context))
    assert disabled["diagnostics"]["support_counts"][0, 0] == 3
    assert enabled["diagnostics"]["support_counts"][0, 0] == 1
    assert np.all(np.asarray(enabled["diagnostics"]["support_counts"]) <=
                  np.asarray(disabled["diagnostics"]["support_counts"]))


def test_sampled_actions_valid_and_decoded_totals_fit_capacity():
    config = _config()
    params = init_stage25_params(config, seed=13)
    capacity = np.array([[20] * 5, [30] * 5, [0] * 5], dtype=np.int16)
    inputs = _encoded(3, capacity=capacity,
                      goals=np.array([[0] * 5, [0] * 5, [60] * 5], dtype=np.int16))
    result = _as_output(_invoke(
        stochastic_act, params=params, inputs=inputs,
        config=config, key=jax.random.split(jax.random.PRNGKey(5), 3)))
    assert _valid(result).all()
    classes = np.asarray(result["classes"], dtype=np.int16)
    assert classes.shape == (3, 9)
    assert np.all((classes >= 0) &
                  (classes < np.asarray(ACTION_CLASS_COUNTS, dtype=np.int16)))
    goals = np.asarray(inputs["crop_goals"])
    decoded = goals + classes[:, 4:] - 100
    assert decoded.shape == (3, 5)
    assert np.all(decoded >= 0)
    assert np.all(decoded.sum(axis=1) <= capacity.sum(axis=1))


def test_public_output_uses_required_valid_field():
    result = _as_output(_invoke(
        greedy_act, params=init_stage25_params(_config(), seed=2),
        inputs=_encoded(1), config=_config()))
    assert "valid" in result


def test_sample_then_evaluate_has_exact_class_logprob_agreement():
    config = _config()
    params = init_stage25_params(config, seed=17)
    inputs = _encoded(2, capacity=np.full((2, 5), 20, dtype=np.int16))
    sampled = _as_output(_invoke(
        stochastic_act, params=params, inputs=inputs, config=config,
        key=jax.random.split(jax.random.PRNGKey(23), 2)))
    evaluated = _as_output(_invoke(
        evaluate_actions, params=params, inputs=inputs, config=config,
        classes=sampled["classes"]))
    assert np.array_equal(np.asarray(sampled["classes"]),
                          np.asarray(evaluated["classes"]))
    assert np.allclose(np.asarray(sampled["component_logprobs"]),
                       np.asarray(evaluated["component_logprobs"]),
                       rtol=0.0, atol=1e-6)
    assert np.allclose(np.asarray(sampled["joint_logprob"]),
                       np.asarray(evaluated["joint_logprob"]),
                       rtol=0.0, atol=1e-6)


def test_changed_action_only_changes_later_logits_and_value_is_action_independent():
    config = _config()
    params = init_stage25_params(config, seed=19)
    inputs = _encoded(1)
    classes = np.array([[1, 1, 1, 1, 100, 100, 100, 100, 100]], dtype=np.int16)
    base = _as_output(_invoke(evaluate_actions, params=params, inputs=inputs,
                              config=config,
                              classes=classes, row_ids=np.array([303])))
    changed_classes = np.array(classes, copy=True)
    changed_classes[0, 2] = 2
    changed = _as_output(_invoke(evaluate_actions, params=params, inputs=inputs,
                                 config=config,
                                 classes=changed_classes, row_ids=np.array([303])))
    logits_a = np.asarray(base["logits"])
    logits_b = np.asarray(changed["logits"])
    assert np.array_equal(logits_a[0, 0], logits_b[0, 0])
    assert np.array_equal(logits_a[0, 1], logits_b[0, 1])
    assert np.array_equal(logits_a[0, 2], logits_b[0, 2])
    assert np.any(logits_a[0, 3:] != logits_b[0, 3:])
    assert np.array_equal(np.asarray(base["value"]), np.asarray(changed["value"]))


def test_greedy_is_masked_argmax_and_same_seed_is_row_stable_under_reorder_padding():
    config = _config()
    params = init_stage25_params(config, seed=29)
    ids = np.array([77, 88], dtype=np.int64)
    caps = np.full((2, 5), 20, dtype=np.int16)
    seed = jax.random.PRNGKey(31)
    row_keys = jnp.stack([jax.random.fold_in(seed, int(row_id)) for row_id in ids])
    base = _as_output(_invoke(
        stochastic_act, params=params, inputs=_encoded(2),
        config=config, key=row_keys))
    reorder_ids = np.array([999, 88, 77], dtype=np.int64)
    reorder_keys = jnp.stack([jax.random.fold_in(seed, int(row_id))
                              for row_id in reorder_ids])
    reordered = _as_output(_invoke(
        stochastic_act, params=params, inputs=_encoded(3),
        config=config, key=reorder_keys))
    assert np.array_equal(np.asarray(base["classes"])[0],
                          np.asarray(reordered["classes"])[2])
    assert np.array_equal(np.asarray(base["classes"])[1],
                          np.asarray(reordered["classes"])[1])
    greedy = _as_output(_invoke(
        greedy_act, params=params, inputs=_encoded(2, capacity=caps), config=config))
    assert "masks" in greedy and "logits" in greedy
    masks = np.asarray(greedy["masks"])
    logits = np.asarray(greedy["logits"])
    classes = np.asarray(greedy["classes"])
    for step, count in enumerate(ACTION_CLASS_COUNTS):
        valid = masks[:, step, :count].astype(bool)
        masked = np.where(valid, logits[:, step, :count], -np.inf)
        assert np.array_equal(classes[:, step], masked.argmax(axis=-1))


def test_teacher_forced_nll_has_finite_gradients_through_all_trainable_blocks():
    config = _config()
    params = init_stage25_params(config, seed=37)
    inputs = _encoded(1)
    classes = jnp.asarray([[1, 1, 1, 1, 100, 100, 100, 100, 100]], dtype=jnp.int16)

    def loss(tree):
        out = _invoke(
            evaluate_actions, params=tree, inputs=inputs, config=config,
            classes=classes)
        return -jnp.mean(jnp.asarray(out["joint_logprob"]))

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)
    assert bool(jnp.any(jax.tree_util.tree_leaves(grads["encoder"])[0] != 0))
    assert bool(jnp.any(grads["capacity_conditioning"] != 0))
    assert bool(jnp.any(grads["recurrent_decoder"]["Wz"] != 0))
    assert bool(jnp.any(jax.tree_util.tree_leaves(grads["output_projections"])[0] != 0))


def test_native_import_and_tiny_forward_do_not_require_torch():
    script = """
import builtins, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Torch imported by native Stage 2.5 policy')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import jax, numpy as np
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params, greedy_act
c = Stage25ModelConfig.tiny()
p = init_stage25_params(c, seed=1)
assert p
out = greedy_act(p, {
    'board_kind': np.zeros((1, 100), dtype=np.int16),
    'board_crop': np.zeros((1, 100), dtype=np.int8),
    'board_animal': np.zeros((1, 100), dtype=np.int8),
    'board_numeric': np.zeros((1, 100, 11), dtype=np.float32),
    'board_bool': np.zeros((1, 100, 8), dtype=bool),
    'board_mask': np.zeros((1, 100, 4), dtype=np.uint8),
    'scalars': np.zeros((1, 4), dtype=np.float32),
    'shed_counts': np.zeros((1, 12), dtype=np.int32),
    'seed_counts': np.zeros((1, 5), dtype=np.int32),
    'carried_counts': np.zeros((1, 12), dtype=np.int32),
    'unlocked': np.array([[1, 0, 0, 0]], dtype=np.uint8),
    'market_inventory': np.zeros((1, 9), dtype=np.int32),
    'market_prices': np.zeros((1, 9), dtype=np.float32),
    'shop_counts': np.zeros((1, 9), dtype=np.int32),
    'day': np.zeros((1,), dtype=np.int16),
    'days_remaining': np.full((1,), 29, dtype=np.int16),
    'economic_context': np.zeros((1, 14), dtype=np.float32),
    'crop_capacity': np.full((1, 5), 20, dtype=np.int16),
    'crop_goals': np.zeros((1, 5), dtype=np.int16),
}, c)
assert out['classes'].shape == (1, 9)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
