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
             ledger: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Small corrected-E-shaped arrays; no Parquet or Torch dependency.

    ``ledger`` is the persistent goal ledger ``K`` of shape ``[B, 5]``.  The
    physical capacity ``C`` is not an input; it is derived from the encoded
    board / ``physical_contexts``.
    """
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
        "crop_capacity": (np.zeros((b, 5), dtype=np.int16)
                          if ledger is None else np.asarray(ledger)),
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


def test_jax_support_matches_packet1_python_helpers():
    config = _config()
    params = init_stage25_params(config, seed=3)
    contexts = _contexts()
    # Persistent goal ledgers K, one per row.  Row 2's K exceeds its physical
    # capacity, which the decoder must contract rather than reject.
    ledger = np.array([[9] * 5, [0] * 5, [60] * 5], dtype=np.int16)
    # Keep the three rows physically distinct: occupied/empty housing, future
    # unlocked land, and an all-unknown zero-capacity footprint.
    physical = [
        _encoded_board(_board(unknown=True, housing=True), ("NW",),
                       ledger=ledger[:1]),
        _encoded_board(_board(future_land=True), ("NW", "NE"),
                       ledger=ledger[1:2]),
        _encoded_board([[{"mystery": True}] * 5 + ["LOCKED"] * 5
                        for _ in range(5)] + [["LOCKED"] * 10 for _ in range(5)],
                       ("NW",), ledger=ledger[2:3]),
    ]
    inputs = {key: np.concatenate([row[key] for row in physical], axis=0)
              for key in physical[0]}
    result = _as_output(_invoke(
        greedy_act, params=params, inputs=inputs, config=config,
        contexts=contexts))
    counts = np.asarray(result["diagnostics"]["support_counts"])
    masks = np.asarray(result["masks"])
    classes = np.asarray(result["classes"])
    for index, context in enumerate(contexts):
        expected_land = np.asarray(
            land_target_support_mask(context.observed_land))
        assert counts[index, 0] == expected_land.sum()
        assert np.array_equal(masks[index, 0, :4], expected_land)
        land_target = int(classes[index, 0]) + 1
        prefix = []
        for step in range(3):
            expected = animal_target_support_mask(
                context, land_target, step, prefix)
            assert counts[index, step + 1] == sum(expected)
            assert np.array_equal(masks[index, step + 1, :101], expected)
            prefix.append(int(classes[index, step + 1]))
        animals = tuple(prefix)
        # Packet 1A physical capacity from the decoded land+animal prefix.
        capacity = physical_crop_capacity(context, land_target, animals)
        goals = ledger[index].astype(np.int32).copy()
        for step in range(5):
            expected = crop_delta_support_mask(
                int(goals[step]), capacity - int(goals[:step].sum()))
            assert counts[index, step + 4] == sum(expected)
            assert np.array_equal(masks[index, step + 4, :201], expected)
            goals[step] += int(classes[index, step + 4]) - 100
        assert int(goals.sum()) <= capacity


def _plain_board(blocked: bool = False, future_land: bool = False):
    board = [["LOCKED"] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            board[y][x] = {"kind": "COOP"} if blocked else None
    if future_land:
        for y in range(5):
            for x in range(5, 10):
                board[y][x] = None
    return board


def _hold_classes() -> np.ndarray:
    return np.array([[0, 0, 0, 0, 100, 100, 100, 100, 100]], dtype=np.int16)


def _crop_masks_match_packet1(masks, classes, ledger, context):
    land_target = int(classes[0, 0]) + 1
    animals = tuple(int(x) for x in classes[0, 1:4])
    capacity = physical_crop_capacity(context, land_target, animals)
    goals = ledger.astype(np.int32).copy()
    for step in range(5):
        expected = np.asarray(crop_delta_support_mask(
            int(goals[step]), capacity - int(goals[:step].sum())))
        assert np.array_equal(np.asarray(masks)[0, step + 4, :201], expected)
        goals[step] += int(classes[0, step + 4]) - 100
    assert int(goals.sum()) <= capacity
    return capacity


def test_physical_capacity_is_derived_from_context_not_ledger():
    config = _config()
    params = init_stage25_params(config, seed=101)
    ctx_full = physical_context_from_board(_plain_board(), ("NW",))
    ctx_blocked = physical_context_from_board(_plain_board(blocked=True), ("NW",))
    assert physical_crop_capacity(ctx_full, 1, (0, 0, 0)) == 25
    assert physical_crop_capacity(ctx_blocked, 1, (0, 0, 0)) == 0
    ledger = np.full((1, 5), 21, dtype=np.int16)
    inputs = _encoded(1, ledger=ledger)
    full = _as_output(greedy_act(params, inputs, config,
                                 physical_contexts=(ctx_full,)))
    blocked = _as_output(greedy_act(params, inputs, config,
                                    physical_contexts=(ctx_blocked,)))
    # Same K, different physical C -> different autoregressive crop support.
    assert not np.array_equal(np.asarray(full["masks"])[:, 4:],
                              np.asarray(blocked["masks"])[:, 4:])
    assert int(full["diagnostics"]["total_capacity"][0]) == 25
    assert int(blocked["diagnostics"]["total_capacity"][0]) == 0
    # K=[21]*5 never grants 105 physical cells; decoded goals fit C.
    assert int(np.asarray(full["decoded_goals"]).sum()) <= 25
    assert int(np.asarray(blocked["decoded_goals"]).sum()) == 0
    assert bool(np.asarray(blocked["valid"])[0])
    # Changing K alone does not change the derived C.
    for value in (0, 21, 60):
        out = greedy_act(params, _encoded(
            1, ledger=np.full((1, 5), value, dtype=np.int16)), config,
            physical_contexts=(ctx_full,))
        assert int(out["diagnostics"]["total_capacity"][0]) == 25


def test_land_expansion_changes_physical_capacity():
    config = _config()
    params = init_stage25_params(config, seed=102)
    # NE is a future locked quadrant: expanding land makes its cells usable.
    context = physical_context_from_board(_plain_board(), ("NW",))
    assert physical_crop_capacity(context, 1, (0, 0, 0)) == 25
    assert physical_crop_capacity(context, 2, (0, 0, 0)) == 50
    inputs = _encoded(1, ledger=np.zeros((1, 5), dtype=np.int16))
    for land_class, expected in ((0, 25), (1, 50)):
        classes = _hold_classes().copy()
        classes[0, 0] = land_class
        out = evaluate_actions(
            params, inputs, config, classes=classes,
            physical_contexts=(context,))
        assert int(out["diagnostics"]["total_capacity"][0]) == expected
        _crop_masks_match_packet1(out["masks"], classes, np.zeros(5), context)


def test_animal_housing_reduces_and_reuse_preserves_capacity():
    config = _config()
    params = init_stage25_params(config, seed=103)
    ctx_bare = physical_context_from_board(_plain_board(), ("NW",))
    inputs = _encoded(1, ledger=np.zeros((1, 5), dtype=np.int16))
    # New housing reduces C: goose 1 (1 coop) + cow 1 + sheep 1 (2 pasture).
    target = _hold_classes().copy()
    target[0, 1:4] = (1, 1, 1)
    assert physical_crop_capacity(ctx_bare, 1, (1, 1, 1)) == 22
    bare = evaluate_actions(params, inputs, config, classes=target,
                            physical_contexts=(ctx_bare,))
    assert int(bare["diagnostics"]["total_capacity"][0]) == 22
    # One reusable empty coop: B drops by one structure cell, but the goose
    # reuses it, so C stays 25 - 1 = 24 instead of paying twice.
    reuse = _plain_board()
    reuse[0][0] = {"kind": "COOP"}
    ctx_coop = physical_context_from_board(reuse, ("NW",))
    assert ctx_coop.reusable_empty_coops == 1
    goose = _hold_classes().copy()
    goose[0, 1] = 1
    assert physical_crop_capacity(ctx_coop, 1, (1, 0, 0)) == 24
    assert physical_crop_capacity(ctx_bare, 1, (1, 0, 0)) == 24
    reused = evaluate_actions(params, inputs, config, classes=goose,
                              physical_contexts=(ctx_coop,))
    assert int(reused["diagnostics"]["total_capacity"][0]) == 24
    _crop_masks_match_packet1(reused["masks"], goose, np.zeros(5), ctx_coop)


def test_hold_preserves_persistent_goal_despite_lower_occupancy():
    config = _config()
    params = init_stage25_params(config, seed=104)
    # Observed occupancy is one wheat; the persistent goal is five.
    board = _board()
    context = physical_context_from_board(board, ("NW",))
    ledger = np.array([[5, 0, 0, 0, 0]], dtype=np.int16)
    inputs = _encoded_board(board, ("NW",), ledger=ledger)
    out = evaluate_actions(params, inputs, config, classes=_hold_classes(),
                           physical_contexts=(context,))
    assert bool(np.asarray(out["valid"])[0])
    assert np.array_equal(np.asarray(out["decoded_goals"])[0], ledger[0])


def test_zero_capacity_forces_contraction_and_remains_valid():
    config = _config()
    params = init_stage25_params(config, seed=105)
    context = physical_context_from_board(_plain_board(blocked=True), ("NW",))
    ledger = np.full((1, 5), 21, dtype=np.int16)
    out = _as_output(greedy_act(
        params, _encoded(1, ledger=ledger), config,
        physical_contexts=(context,)))
    assert bool(np.asarray(out["valid"])[0])
    assert np.array_equal(np.asarray(out["decoded_goals"]),
                          np.zeros((1, 5), dtype=np.int32))


def test_ledger_input_is_required_and_rectangular():
    config = _config()
    params = init_stage25_params(config, seed=106)
    omitted = _encoded(1)
    del omitted["crop_capacity"]
    with pytest.raises(ValueError, match="persistent goal ledger"):
        greedy_act(params, omitted, config)
    for bad in (np.zeros((1,), dtype=np.int16),
                np.zeros((1, 3), dtype=np.int16),
                np.full((1, 5), 200, dtype=np.int16),
                np.full((1, 5), -1, dtype=np.int16),
                np.full((1, 5), 1.5, dtype=np.float32)):
        with pytest.raises(ValueError):
            greedy_act(params, _encoded(1), config, crop_capacity=bad)


def test_sampled_and_teacher_forced_crop_masks_match_packet1():
    config = _config()
    params = init_stage25_params(config, seed=107)
    rng = np.random.default_rng(20260911)
    quads = ("NW", "NE", "SW", "SE")
    kinds = (None, "WEED", {"kind": "PLANT", "crop": "WHEAT"},
             {"kind": "COOP"}, {"kind": "PASTURE"},
             {"kind": "PASTURE", "animal": "COW"}, "LOCKED",
             {"mystery": True})
    for trial in range(8):
        unlocked = tuple(quads[: int(rng.integers(1, 5))])
        board = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(10):
            for x in range(10):
                quad = ("NW" if y < 5 and x < 5 else "NE" if y < 5
                        else "SW" if x < 5 else "SE")
                if quad in unlocked:
                    board[y][x] = kinds[int(rng.integers(0, len(kinds)))]
        # A feasible animal choice always exists, so C is nonnegative.
        context = physical_context_from_board(board, unlocked)
        ledger = rng.integers(0, 80, size=(1, 5)).astype(np.int16)
        inputs = _encoded_board(board, unlocked, ledger=ledger)
        greedy = _as_output(greedy_act(
            params, inputs, config, physical_contexts=(context,),
            row_ids=np.array([trial])))
        _crop_masks_match_packet1(
            greedy["masks"], np.asarray(greedy["classes"]), ledger[0], context)
        sampled = _as_output(stochastic_act(
            params, inputs, config, rng_keys=jnp.array([[trial + 1, 7]]),
            physical_contexts=(context,), row_ids=np.array([trial])))
        assert bool(np.asarray(sampled["valid"])[0])
        _crop_masks_match_packet1(
            sampled["masks"], np.asarray(sampled["classes"]), ledger[0],
            context)
        forced = evaluate_actions(
            params, inputs, config, classes=sampled["classes"],
            physical_contexts=(context,))
        assert np.array_equal(np.asarray(forced["masks"]),
                              np.asarray(sampled["masks"]))
        # Exact conditional-likelihood parity with the correct physical C.
        assert np.allclose(np.asarray(sampled["component_logprobs"]),
                           np.asarray(forced["component_logprobs"]),
                           rtol=0.0, atol=1e-6)
        assert np.allclose(np.asarray(sampled["joint_logprob"]),
                           np.asarray(forced["joint_logprob"]),
                           rtol=0.0, atol=1e-6)


def test_encoded_observation_physical_seam_is_economic_support_invariant():
    config = _config()
    params = init_stage25_params(config, seed=11)
    physical = _contexts()[0]
    encoded = _encoded_board(_board(unknown=True, housing=True), ("NW",),
                             economics=0.0, ledger=np.full((1, 5), 10, dtype=np.int16))
    economic_only = _encoded_board(_board(unknown=True, housing=True), ("NW",),
                                   economics=999.0, ledger=np.full((1, 5), 10, dtype=np.int16))
    a = _as_output(_invoke(greedy_act, params=params, inputs=encoded, config=config,
                           contexts=None, row_ids=np.array([42])))
    b = _as_output(_invoke(greedy_act, params=params, inputs=economic_only, config=config,
                           contexts=(physical,), row_ids=np.array([42])))
    assert np.array_equal(np.asarray(a["diagnostics"]["support_counts"]),
                          np.asarray(b["diagnostics"]["support_counts"]))


def test_enabled_curriculum_is_applied_after_physical_support():
    params = init_stage25_params(_config(), seed=12)
    context = (_contexts()[1],)
    inputs = _encoded(1, ledger=np.full((1, 5), 20, dtype=np.int16))
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


def test_sampled_actions_valid_and_decoded_totals_fit_physical_capacity():
    config = _config()
    params = init_stage25_params(config, seed=13)
    ledger = np.array([[0] * 5, [21] * 5, [60] * 5], dtype=np.int16)
    contexts = _contexts()
    inputs = _encoded(3, ledger=ledger)
    result = _as_output(_invoke(
        stochastic_act, params=params, inputs=inputs,
        config=config, key=jax.random.split(jax.random.PRNGKey(5), 3),
        contexts=contexts))
    assert _valid(result).all()
    classes = np.asarray(result["classes"], dtype=np.int16)
    assert classes.shape == (3, 9)
    assert np.all((classes >= 0) &
                  (classes < np.asarray(ACTION_CLASS_COUNTS, dtype=np.int16)))
    decoded = ledger.astype(np.int32) + classes[:, 4:] - 100
    assert decoded.shape == (3, 5)
    assert np.all(decoded >= 0)
    for index, context in enumerate(contexts):
        land_target = int(classes[index, 0]) + 1
        animals = tuple(int(x) for x in classes[index, 1:4])
        capacity = physical_crop_capacity(context, land_target, animals)
        assert int(decoded[index].sum()) <= capacity


def test_public_output_uses_required_valid_field():
    result = _as_output(_invoke(
        greedy_act, params=init_stage25_params(_config(), seed=2),
        inputs=_encoded(1), config=_config()))
    assert "valid" in result


def test_evaluate_actions_rejects_invalid_and_preserves_supplied_classes():
    config = _config()
    params = init_stage25_params(config, seed=61)
    inputs = _encoded(1, ledger=np.zeros((1, 5), dtype=np.int16))
    out_of_vocab = np.array([[9, 1, 1, 1, 100, 100, 100, 100, 100]],
                            dtype=np.int16)
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate_actions(params, inputs, config, classes=out_of_vocab)
    # In-vocabulary but physically unsupported: goal would go below zero.
    unsupported = np.array([[0, 0, 0, 0, 0, 100, 100, 100, 100]],
                           dtype=np.int16)
    with pytest.raises(ValueError, match="unsupported or invalid"):
        evaluate_actions(params, inputs, config, classes=unsupported)
    # Diagnostics mode preserves the supplied sequence and marks validity; it
    # never substitutes a repaired class 0 in the authoritative classes.
    out = evaluate_actions(params, inputs, config, classes=out_of_vocab,
                           reject_invalid=False)
    assert int(np.asarray(out["classes"])[0, 0]) == 9
    assert not bool(np.asarray(out["valid"])[0])
    assert float(np.asarray(out["component_logprobs"])[0, 0]) == 0.0
    assert np.all(np.asarray(out["component_logprobs"])[0, 1:] == 0.0)
    assert np.all(np.asarray(out["conditional_entropies"])[0, 1:] == 0.0)
    unsup = evaluate_actions(params, inputs, config, classes=unsupported,
                             reject_invalid=False)
    assert int(np.asarray(unsup["classes"])[0, 4]) == 0
    assert not bool(np.asarray(unsup["valid"])[0])
    assert float(np.asarray(unsup["component_logprobs"])[0, 4]) == 0.0
    assert float(np.asarray(unsup["component_logprobs"])[0, 5]) == 0.0


def test_host_rejects_empty_or_noncanonical_unlocked():
    config = _config()
    params = init_stage25_params(config, seed=62)
    base = _encoded(1, ledger=np.zeros((1, 5), dtype=np.int16))
    empty = dict(base)
    empty["unlocked"] = np.zeros((1, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="NW quadrant"):
        greedy_act(params, empty, config)
    noncanonical = dict(base)
    noncanonical["unlocked"] = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="canonical land prefix"):
        greedy_act(params, noncanonical, config)


def test_init_stage25_params_imports_encoder_leaves_and_drops_legacy_heads():
    from bc_manager_jax.model import init_train_params

    config = _config()
    full_e = init_train_params(config.manager_config, seed=123,
                               model_variant="E")
    seeded = init_stage25_params(config, seed=999, encoder_params=full_e)
    for name in ("manager_token", "role_embedding", "tile_encoder",
                 "global_encoders", "encoder", "encoder_norm"):
        source = jax.tree_util.tree_leaves(full_e[name])
        imported = jax.tree_util.tree_leaves(seeded["encoder"][name])
        assert len(source) == len(imported)
        for left, right in zip(source, imported):
            assert np.array_equal(np.asarray(left), np.asarray(right))
    assert "heads" not in seeded
    assert (jax.tree_util.tree_structure(parameter_spec(config))
            == jax.tree_util.tree_structure(seeded))


def test_sample_then_evaluate_has_exact_class_logprob_agreement():
    config = _config()
    params = init_stage25_params(config, seed=17)
    inputs = _encoded(2, ledger=np.full((2, 5), 20, dtype=np.int16))
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
        greedy_act, params=params, inputs=_encoded(2, ledger=caps), config=config))
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
    # A nonzero persistent ledger keeps capacity conditioning differentiable.
    inputs = _encoded_board(_board(), ("NW",),
                            ledger=np.full((1, 5), 4, dtype=np.int16))
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

    def entropy_loss(tree):
        out = _invoke(
            evaluate_actions, params=tree, inputs=inputs, config=config,
            classes=classes)
        return jnp.mean(jnp.asarray(out["prefix_entropy_surrogate"]))

    entropy_grads = jax.grad(entropy_loss)(params)
    entropy_leaves = jax.tree_util.tree_leaves(entropy_grads)
    assert entropy_leaves and all(
        bool(jnp.isfinite(leaf).all()) for leaf in entropy_leaves)
    assert bool(jnp.any(entropy_grads["recurrent_decoder"]["Wz"] != 0))


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
    'crop_capacity': np.zeros((1, 5), dtype=np.int16),
}, c)
assert out['classes'].shape == (1, 9)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
