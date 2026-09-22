"""Focused tests for the parent-owned native Stage 2.5 inference seam."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_manager.stage25_checkpoint import save_stage25_inference_checkpoint
from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_inference import (
    Stage25InferenceAdapter, _normalise_row_ids, _normalise_row_tokens,
    _root_key, _row_rng_keys,
)
from rl_manager.stage25_mechanics import (
    PhysicalContext, animal_target_support_mask, crop_delta_support_mask,
    land_target_support_mask, physical_crop_capacity,
)
from rl_manager.stage25_policy import (
    _validate_stage25_inputs,
    greedy_act, init_stage25_params, stochastic_act, tiny_stage25_config,
)
from rl_manager.stage25_types import Stage25PolicyOutputs, stage25_row_token


def _inputs(batch: int = 1) -> dict[str, np.ndarray]:
    return {
        "board_kind": np.zeros((batch, 100), dtype=np.int16),
        "board_crop": np.zeros((batch, 100), dtype=np.int8),
        "board_animal": np.zeros((batch, 100), dtype=np.int8),
        "board_numeric": np.zeros((batch, 100, 11), dtype=np.float32),
        "board_bool": np.zeros((batch, 100, 8), dtype=bool),
        "board_mask": np.zeros((batch, 100, 4), dtype=np.uint8),
        "scalars": np.zeros((batch, 4), dtype=np.float32),
        "shed_counts": np.zeros((batch, 12), dtype=np.int32),
        "seed_counts": np.zeros((batch, 5), dtype=np.int32),
        "carried_counts": np.zeros((batch, 12), dtype=np.int32),
        "unlocked": np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8), (batch, 1)),
        "market_inventory": np.zeros((batch, 9), dtype=np.int32),
        "market_prices": np.zeros((batch, 9), dtype=np.float32),
        "shop_counts": np.zeros((batch, 9), dtype=np.int32),
        "day": np.zeros((batch,), dtype=np.int16),
        "days_remaining": np.full((batch,), 29, dtype=np.int16),
        "economic_context": np.zeros((batch, 14), dtype=np.float32),
        "crop_capacity": np.zeros((batch, 5), dtype=np.int16),
    }


def _context() -> PhysicalContext:
    # An all-empty NW quadrant has 25 compatible cells for every hypothetical
    # target; this is exactly the context derived from _inputs().
    return PhysicalContext(1, (25, 25, 25, 25), (0, 0, 0), 0, 0, (0, 0, 0))


def _support(context: PhysicalContext) -> dict[str, object]:
    capacity = max(physical_crop_capacity(
        context, context.observed_land, context.placed_animals), 0)
    return {
        "land": list(land_target_support_mask(context.observed_land)),
        "animals": [list(animal_target_support_mask(
            context, context.observed_land, species,
            context.placed_animals[:species])) for species in range(3)],
        "crops": [list(crop_delta_support_mask(0, capacity)) for _ in range(5)],
    }


def _adapter(tmp_path: Path | None = None, *, curriculum=None):
    config = tiny_stage25_config(curriculum=curriculum or Stage25CurriculumConfig())
    params = init_stage25_params(config, seed=7)
    if tmp_path is None:
        return Stage25InferenceAdapter(params=params, config=config)
    checkpoint = tmp_path / "stage25.npz"
    save_stage25_inference_checkpoint(checkpoint, params, config, seed=7)
    return checkpoint, config


def test_outputs_have_contract_dtypes_identity_and_one_batch_call():
    adapter = _adapter()
    result = adapter.plan_batch_with_row_ids(
        _inputs(2), ("row-a", "row-b"), "episode=1/day=0",
        physical_contexts=(_context(), _context()))

    assert isinstance(result, Stage25PolicyOutputs)
    assert result.classes.dtype == np.int16
    assert result.component_logprobs.dtype == np.float32
    assert result.joint_logprob.dtype == np.float32
    assert result.value.dtype == np.float32
    assert result.decoded_goals.dtype == np.int16
    assert result.valid.dtype == np.bool_
    assert result.policy_identity == adapter.identity
    assert adapter.call_count == 1
    assert adapter.batch_size_history == [2]

    deterministic = Stage25InferenceAdapter(
        params=adapter.params, config=adapter.config, mode="deterministic")
    greedy = deterministic.plan_batch_with_row_ids(
        _inputs(2), ("row-a", "row-b"), "episode=1/day=0",
        physical_contexts=(_context(), _context()))
    assert deterministic.deterministic is True
    assert deterministic.call_count == 1
    assert greedy.valid.all()


def test_inference_phase_metrics_are_finite_and_reconcile():
    adapter = _adapter()
    adapter.infer_batch(
        _inputs(2), physical_contexts=(_context(), _context()),
        supports=(_support(_context()), _support(_context())),
        row_ids=("timed-a", "timed-b"), prng_id="timed")

    phases = adapter.inference_phase_seconds
    assert set(phases) == {
        "input_validation_seconds", "host_input_prepare_seconds",
        "context_validation_seconds", "support_validation_seconds",
        "row_rng_prepare_seconds", "policy_call_seconds",
        "output_conversion_seconds", "adapter_total_seconds",
    }
    assert all(np.isfinite(value) and value >= 0.0
               for value in phases.values())
    measured = sum(value for name, value in phases.items()
                   if name != "adapter_total_seconds")
    assert measured <= phases["adapter_total_seconds"] * 1.5 + 1.0e-6
    assert phases["policy_call_seconds"] > 0.0


def test_fast_and_none_preserve_row_stable_policy_outputs():
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=19)
    contexts = (_context(), _context())
    inputs = _inputs(2)
    supports = (_support(contexts[0]), _support(contexts[1]))
    outputs = {}
    for mode in ("strict", "fast", "none"):
        adapter = Stage25InferenceAdapter(
            params=params, config=config, validation_mode=mode,
            seed=31)
        outputs[mode] = adapter.infer_batch(
            inputs, physical_contexts=contexts, supports=supports,
            row_ids=("same-a", "same-b"), prng_id="same")
    for name in ("classes", "component_logprobs", "joint_logprob",
                 "value", "decoded_goals", "valid"):
        np.testing.assert_array_equal(
            getattr(outputs["fast"], name), getattr(outputs["strict"], name))
        np.testing.assert_array_equal(
            getattr(outputs["none"], name), getattr(outputs["strict"], name))


def test_validation_modes_define_diagnostic_support_boundary():
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=23)
    context = _context()
    malformed = {"land": (), "animals": (), "crops": ()}
    with pytest.raises(ValueError, match="support"):
        Stage25InferenceAdapter(
            params=params, config=config, validation_mode="strict").infer_batch(
                _inputs(), physical_contexts=(context,), supports=(malformed,),
                row_ids=("strict",), prng_id="mode")
    # The payload is diagnostic transport only.  The native policy still gets
    # the immutable context and constructs its authoritative masks.
    for mode in ("none",):
        output = Stage25InferenceAdapter(
            params=params, config=config, validation_mode=mode).infer_batch(
                _inputs(), physical_contexts=(context,), supports=(malformed,),
                row_ids=(mode,), prng_id="mode")
        assert output.valid.all()


def test_fast_skips_diagnostic_support_value_cross_check():
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=37)
    context = _context()
    support = _support(context)
    animals = [list(mask) for mask in support["animals"]]
    invalid = animals[0].index(False)
    animals[0][invalid] = True
    malformed = {**support, "animals": animals}
    with pytest.raises(ValueError, match="physical support"):
        Stage25InferenceAdapter(
            params=params, config=config, validation_mode="strict").infer_batch(
                _inputs(), physical_contexts=(context,), supports=(malformed,),
                row_ids=("strict-value",), prng_id="mode")
    output = Stage25InferenceAdapter(
        params=params, config=config, validation_mode="fast").infer_batch(
            _inputs(), physical_contexts=(context,), supports=(malformed,),
            row_ids=("strict-value",), prng_id="mode")
    assert output.valid.all()


def test_validation_mode_rejects_unknown_values():
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=29)
    with pytest.raises(ValueError, match="validation_mode"):
        Stage25InferenceAdapter(
            params=params, config=config, validation_mode="disabled")


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    (
        ("missing required field", lambda values: values.pop("scalars"),
         "missing required"),
        ("unknown field", lambda values: values.update(extra=np.zeros((1,))),
         "unknown"),
        ("wrong batch length", lambda values: values.update(
            day=np.zeros((2,), dtype=np.int16)), "rows"),
        ("wrong board_kind shape", lambda values: values.update(
            board_kind=np.zeros((1, 99), dtype=np.int16)), "shape"),
        ("wrong board_mask shape", lambda values: values.update(
            board_mask=np.zeros((1, 100, 3), dtype=np.uint8)), "shape"),
        ("wrong economic_context shape", lambda values: values.update(
            economic_context=np.zeros((1, 13), dtype=np.float32)), "shape"),
        ("non-finite economic_context", lambda values: values.update(
            economic_context=np.full((1, 14), np.nan, dtype=np.float32)),
            "non-finite"),
        ("wrong crop ledger shape", lambda values: values.update(
            crop_capacity=np.zeros((1, 4), dtype=np.int16)), "shape"),
        ("crop ledger below zero", lambda values: values.update(
            crop_capacity=np.full((1, 5), -1, dtype=np.int16)), r"\[0, 100\]"),
        ("crop ledger above 100", lambda values: values.update(
            crop_capacity=np.full((1, 5), 101, dtype=np.int16)), r"\[0, 100\]"),
        ("nonintegral crop ledger", lambda values: values.update(
            crop_capacity=np.full((1, 5), 1.5, dtype=np.float32)), "integers"),
        ("wrong unlocked shape", lambda values: values.update(
            unlocked=np.zeros((1, 3), dtype=np.uint8)), "shape"),
        ("non-prefix unlocked", lambda values: values.update(
            unlocked=np.array([[1, 0, 1, 0]], dtype=np.uint8)), "prefix"),
        ("zero unlocked land", lambda values: values.update(
            unlocked=np.zeros((1, 4), dtype=np.uint8)), "NW quadrant"),
        ("object dtype", lambda values: values.update(
            board_kind=np.zeros((1, 100), dtype=object)), "dtype"),
        ("forbidden non-finite float", lambda values: values.update(
            scalars=np.full((1, 4), np.inf, dtype=np.float32)), "non-finite"),
    ),
)
def test_fast_rollout_validation_rejects_p2_malformed_inputs(
        name, mutate, message):
    del name
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=211)
    values = {key: value.copy() for key, value in _inputs().items()}
    mutate(values)
    adapter = Stage25InferenceAdapter(
        params=params, config=config, validation_mode="fast")
    with pytest.raises(ValueError, match=message):
        adapter.infer_batch(values, row_ids=("invalid",), prng_id="validation")


def test_board_numeric_nullable_nan_and_canonical_arrays_preserve_contract():
    config = tiny_stage25_config()
    values = _inputs()
    values["board_numeric"][0, 0, 0] = np.nan
    validated = _validate_stage25_inputs(values, config)
    assert np.isnan(validated.inputs["board_numeric"][0, 0, 0])
    assert validated.inputs["board_numeric"].dtype == np.float32
    assert validated.inputs["shed_counts"].dtype == np.int32
    assert np.shares_memory(validated.inputs["board_numeric"],
                            values["board_numeric"])
    assert np.shares_memory(validated.inputs["shed_counts"],
                            values["shed_counts"])
    assert not np.shares_memory(validated.inputs["board_kind"],
                                values["board_kind"])


def test_row_ids_make_sampling_stable_under_reorder_and_padding():
    adapter = _adapter()
    contexts = (_context(), _context())
    original_inputs = _inputs(2)
    first = adapter.plan_batch_with_row_ids(
        original_inputs, ("a", "b"), "same-day", physical_contexts=contexts)

    reordered_inputs = {key: value[[1, 0]] for key, value in original_inputs.items()}
    reordered = adapter.plan_batch_with_row_ids(
        reordered_inputs, ("b", "a"), "same-day",
        physical_contexts=contexts[::-1])
    np.testing.assert_array_equal(reordered.classes, first.classes[[1, 0]])
    np.testing.assert_array_equal(reordered.component_logprobs,
                                  first.component_logprobs[[1, 0]])
    np.testing.assert_array_equal(reordered.value, first.value[[1, 0]])

    padded_inputs = {key: np.concatenate((value, value[:1]), axis=0)
                     for key, value in original_inputs.items()}
    padded = adapter.plan_batch_with_row_ids(
        padded_inputs, ("a", "b", "padding"), "same-day",
        physical_contexts=(_context(), _context(), _context()))
    np.testing.assert_array_equal(padded.classes[:2], first.classes)
    np.testing.assert_array_equal(padded.joint_logprob[:2], first.joint_logprob)


def test_optimized_row_keys_are_exactly_the_legacy_fold_sequence():
    adapter = _adapter()
    ids = _normalise_row_ids(("row-a", 17, "row-c"), 3)
    root = _root_key("same-day", adapter.identity, adapter.seed)
    legacy = np.asarray(jnp.stack([
        jax.random.fold_in(root, int(row_id)) for row_id in ids
    ]), dtype=np.uint32)
    optimized = np.asarray(_row_rng_keys(root, ids), dtype=np.uint32)
    np.testing.assert_array_equal(optimized, legacy)


def test_precomputed_tokens_match_the_canonical_request_hash():
    row_ids = (
        "episode=1/seat=0/day=4/behavior=a@v:111111111111",
        "episode=1/seat=1/day=12/behavior=b@v:222222222222",
        "episode=9/seat=0/day=20/behavior=c@v:333333333333",
        17,
    )
    expected = np.asarray([stage25_row_token(row_id) for row_id in row_ids],
                          dtype=np.int32)
    np.testing.assert_array_equal(
        _normalise_row_tokens(expected.tolist(), len(row_ids)), expected)


def test_precomputed_root_path_matches_legacy_policy_outputs_exactly():
    adapter = _adapter()
    inputs = _inputs(3)
    contexts = (_context(),) * 3
    row_ids = ("precomputed-a", "precomputed-b", "precomputed-c")
    tokens = _normalise_row_ids(row_ids, 3)
    output = adapter.infer_batch(
        inputs, physical_contexts=contexts, row_ids=row_ids,
        row_tokens=tokens, prng_id="same-day")
    root = _root_key("same-day", adapter.identity, adapter.seed)
    legacy = stochastic_act(
        adapter.params, inputs, adapter.config,
        rng_keys=np.asarray(jnp.stack([
            jax.random.fold_in(root, int(row_id)) for row_id in tokens
        ]), dtype=np.uint32), physical_contexts=contexts, row_ids=tokens,
        reject_invalid=False)
    for name in ("classes", "component_logprobs", "joint_logprob", "value",
                 "decoded_goals", "valid"):
        np.testing.assert_array_equal(getattr(output, name),
                                      np.asarray(legacy[name]))
    assert list(adapter._root_key_cache) == ["same-day"]


def test_optimized_adapter_matches_legacy_policy_outputs_exactly():
    adapter = _adapter()
    inputs = _inputs(3)
    contexts = (_context(),) * 3
    row_ids = ("a", "b", "c")
    ids = _normalise_row_ids(row_ids, 3)
    output = adapter.plan_batch_with_row_ids(
        inputs, row_ids, "same-day", physical_contexts=contexts)
    root = _root_key("same-day", adapter.identity, adapter.seed)
    legacy = stochastic_act(
        adapter.params, inputs, adapter.config,
        rng_keys=np.asarray(jnp.stack([
            jax.random.fold_in(root, int(row_id)) for row_id in ids
        ]), dtype=np.uint32), physical_contexts=contexts, row_ids=ids,
        reject_invalid=False)
    for name in ("classes", "component_logprobs", "joint_logprob", "value",
                 "decoded_goals", "valid"):
        np.testing.assert_array_equal(getattr(output, name), np.asarray(legacy[name]))


def test_deterministic_adapter_classes_and_scores_match_policy_exactly():
    stochastic = _adapter()
    adapter = Stage25InferenceAdapter(
        params=stochastic.params, config=stochastic.config, mode="deterministic")
    inputs = _inputs(2)
    contexts = (_context(),) * 2
    ids = _normalise_row_ids(("a", "b"), 2)
    output = adapter.plan_batch_with_row_ids(
        inputs, ("a", "b"), "ignored", physical_contexts=contexts)
    legacy = greedy_act(
        adapter.params, inputs, adapter.config, physical_contexts=contexts,
        row_ids=ids)
    for name in ("classes", "component_logprobs", "joint_logprob", "value",
                 "decoded_goals", "valid"):
        legacy_name = "validity" if name == "valid" else name
        np.testing.assert_array_equal(getattr(output, name), np.asarray(legacy[legacy_name]))


def test_neighboring_rows_do_not_change_row_stable_stochastic_outputs():
    adapter = _adapter()
    base = adapter.plan_batch_with_row_ids(
        _inputs(1), ("target",), "same-day", physical_contexts=(_context(),))
    expanded = adapter.plan_batch_with_row_ids(
        _inputs(3), ("neighbor-left", "target", "neighbor-right"),
        "same-day", physical_contexts=(_context(),) * 3)
    for name in ("classes", "component_logprobs", "joint_logprob", "value",
                 "decoded_goals", "valid"):
        if name == "value":
            np.testing.assert_allclose(
                getattr(expanded, name)[1], getattr(base, name)[0],
                rtol=0.0, atol=1.0e-6)
        else:
            np.testing.assert_array_equal(
                getattr(expanded, name)[1], getattr(base, name)[0])


def test_infer_batch_rejects_physical_support_that_exceeds_context():
    adapter = _adapter()
    context = _context()
    total_capacity = physical_crop_capacity(
        context, context.observed_land, context.placed_animals)
    valid_support = {
        "land": list(land_target_support_mask(context.observed_land)),
        "animals": [list(animal_target_support_mask(
            context, context.observed_land, species,
            context.placed_animals[:species])) for species in range(3)],
        "crops": [list(crop_delta_support_mask(0, total_capacity))
                   for _ in range(5)],
    }
    invalid_animals = list(valid_support["animals"][0])
    invalid_animals[invalid_animals.index(False)] = True
    invalid_support = {
        **valid_support,
        "animals": [
            invalid_animals,
            valid_support["animals"][1],
            valid_support["animals"][2],
        ],
    }
    with pytest.raises(ValueError, match="physical support"):
        adapter.infer_batch(
            _inputs(), physical_contexts=(context,),
            supports=(invalid_support,), row_ids=("row",), prng_id="day")


def test_row_action_is_stable_across_runner_and_parent_prng_namespaces():
    adapter = _adapter()
    row = "episode=0/seat=0/day=4/behavior=" + adapter.identity.identity_id()
    local = adapter.plan_batch_with_row_ids(
        _inputs(1), (row,), "stage25/policy=" + adapter.identity.identity_id(),
        physical_contexts=(_context(),))
    parent = adapter.plan_batch_with_row_ids(
        _inputs(1), (row,), "stage25/behavior=" + adapter.identity.identity_id(),
        physical_contexts=(_context(),))
    np.testing.assert_array_equal(local.classes, parent.classes)


def test_checkpoint_binds_curriculum_and_rejects_provider_mismatch(tmp_path):
    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    checkpoint, config = _adapter(tmp_path, curriculum=enabled)
    loaded = Stage25InferenceAdapter(checkpoint, config=config,
                                     provider_curriculum=enabled)
    assert loaded.curriculum == enabled
    with pytest.raises(ValueError, match="curriculum"):
        Stage25InferenceAdapter(checkpoint, provider_curriculum=Stage25CurriculumConfig())


def test_physical_context_shape_and_bad_batch_boundary_fail_loudly():
    adapter = _adapter()
    with pytest.raises(ValueError, match="physical_contexts"):
        adapter.plan_batch_with_row_ids(
            _inputs(), ("row",), "day", physical_contexts=())
    bad = _inputs()
    bad["day"] = np.zeros((), dtype=np.int16)
    with pytest.raises(ValueError, match="batched"):
        adapter.plan_batch(bad, "day")


def test_physical_context_must_match_canonical_observation():
    adapter = _adapter()
    mismatched = PhysicalContext(2, (25, 25, 25, 25), (0, 0, 0), 0, 0, (0, 0, 0))
    with pytest.raises(ValueError, match="observed_land"):
        adapter.plan_batch_with_row_ids(
            _inputs(), ("row",), "day", physical_contexts=(mismatched,))


def test_teacher_force_and_unchanged_weight_audit_do_not_overwrite_rollout():
    adapter = _adapter()
    inputs = _inputs(2)
    rollout = adapter.plan_batch_with_row_ids(
        inputs, ("a", "b"), "audit-day",
        physical_contexts=(_context(), _context()))
    before = {name: np.array(getattr(rollout, name), copy=True)
              for name in ("classes", "component_logprobs", "joint_logprob",
                           "value", "decoded_goals", "valid")}
    forced = adapter.teacher_force(
        inputs, rollout.classes, physical_contexts=(_context(), _context()),
        row_ids=("a", "b"))
    assert forced.policy_identity == adapter.identity
    report = adapter.audit_unchanged_weights(
        inputs, rollout, physical_contexts=(_context(), _context()),
        row_ids=("a", "b"))
    assert report["ok"] is True
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(rollout, name), value)
