"""Focused tests for the parent-owned native Stage 2.5 inference seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_manager.stage25_checkpoint import save_stage25_inference_checkpoint
from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_mechanics import PhysicalContext
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_types import Stage25PolicyOutputs


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
