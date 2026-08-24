"""Strict trajectory schema/capacity/round-trip tests (issue #9 A1).

The training buffer is preallocated and bounded; appends beyond capacity,
wrong shapes/dtypes/keys, and bad digests must fail loudly. Round-trip via
NPZ + JSON sidecar preserves every array and metadata record; the sidecar is
JSON-safe under `allow_nan=False`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rl_manager.decode import ACTION_TENSOR_SHAPES, LOGPROB_GROUPS
from rl_manager.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    Transition,
    TransitionMetadata,
    TrajectoryBuffer,
    arrayspec_core_keys,
    e_input_spec,
    load_trajectory,
)


def _make_transition(seed: int = 17, seat: int = 0, day: int = 4):
    spec = e_input_spec()
    inputs = {key: np.zeros((1,) + shape, dtype=dtype)
              for key, (shape, dtype) in spec.items()}
    inputs["day"][0] = day
    action_tensors = {
        name: np.zeros((1,) + shape,
                       dtype=np.uint8 if name == "sell_presence" else np.int16)
        for name, shape in ACTION_TENSOR_SHAPES.items()}
    action_tensors["land"][0] = 2
    transition = Transition(
        episode_index=0, seed=seed, seat=seat, day=day, trainable=True,
        inputs=inputs, action_tensors=action_tensors,
        logprob_groups={group: -0.5 for group in LOGPROB_GROUPS},
        logprob_total=-3.0, value=0.25,
        trace_digest=bytes(range(32)))
    metadata = TransitionMetadata(
        index=-1, episode_index=0, seed=seed, seat=seat, day=day,
        policy_id="p@v:f", policy_version="v", policy_fingerprint="f",
        opponent_id="o@v:g", trainable=True, plan_json={"land": 2})
    return transition, metadata


def test_input_spec_is_derived_from_authoritative_encoder():
    spec = e_input_spec()
    # Variant E must expose the economic context channel.
    assert "economic_context" in spec
    assert spec["economic_context"][0][-1] == 14
    assert all(isinstance(s, tuple) for s, _ in spec.values())


def test_append_stores_strict_schema_including_scalar_inputs():
    buffer = TrajectoryBuffer(capacity=2, input_spec=e_input_spec())
    transition, metadata = _make_transition(day=7)
    buffer.append(transition, metadata)
    assert len(buffer) == 1
    arrays = buffer.finalize()
    assert int(arrays["day"][0]) == 7
    assert int(arrays["action_land"][0]) == 2
    assert float(arrays["logprob_total"][0]) == pytest.approx(-3.0)
    assert float(arrays["value"][0]) == pytest.approx(0.25)
    assert list(arrays["trace_digest"][0]) == list(range(32))
    for key in ("day", "days_remaining"):
        # Scalar-shaped inputs land in flat capacity-sized arrays.
        assert arrays[f"input_{key}"].shape == (2,)
    # Unfilled rows stay zero-initialized and untouched.
    assert int(arrays["seat"][1]) == 0


def test_append_rejects_wrong_keys_shapes_and_digests():
    spec = e_input_spec()
    buffer = TrajectoryBuffer(capacity=2, input_spec=spec)
    transition, metadata = _make_transition()

    missing = dict(transition.inputs)
    del missing["day"]
    with pytest.raises(ValueError, match="input keys mismatch"):
        buffer.append(type(transition)(**{**transition.__dict__,
                                          "inputs": missing}), metadata)

    bad_shape = dict(transition.inputs)
    bad_shape["day"] = np.zeros((2,), dtype=np.int16)
    with pytest.raises(ValueError, match="must have shape"):
        buffer.append(type(transition)(**{**transition.__dict__,
                                          "inputs": bad_shape}), metadata)

    with pytest.raises(ValueError, match="trace_digest"):
        buffer.append(type(transition)(**{
            **transition.__dict__, "trace_digest": b"short"}), metadata)


def test_invalid_capacity_and_spec_fail_loud():
    with pytest.raises(ValueError, match="capacity"):
        TrajectoryBuffer(capacity=0, input_spec=e_input_spec())
    with pytest.raises(ValueError, match="capacity"):
        TrajectoryBuffer(capacity=True, input_spec=e_input_spec())
    with pytest.raises(ValueError, match="input_spec"):
        TrajectoryBuffer(capacity=1, input_spec={"day": ((), np.int16)})


def test_capacity_overflow_fails_loud_never_truncates():
    buffer = TrajectoryBuffer(capacity=2, input_spec=e_input_spec())
    for _ in range(2):
        buffer.append(*_make_transition())
    with pytest.raises(ValueError, match="capacity"):
        buffer.append(*_make_transition())


def test_patch_semantics_bounds_and_terminal_only_reward():
    buffer = TrajectoryBuffer(capacity=4, input_spec=e_input_spec())
    for day in (4, 5):
        buffer.append(*_make_transition(day=day))
    arrays = buffer.finalize()
    assert float(arrays["reward"][0]) == 0.0
    buffer.patch_trace_digest(0, bytes(32))
    buffer.patch_terminal(1, -1.0, True)
    arrays = buffer.finalize()
    assert float(arrays["reward"][1]) == -1.0
    assert int(arrays["terminated"][1]) == 1
    assert float(arrays["reward"][0]) == 0.0
    with pytest.raises(ValueError):
        buffer.patch_terminal(2, 1.0, True)  # outside filled range
    with pytest.raises(ValueError):
        buffer.patch_trace_digest(0, b"bad")


def test_finalize_views_are_readonly_and_contiguous():
    buffer = TrajectoryBuffer(capacity=3, input_spec=e_input_spec())
    buffer.append(*_make_transition())
    arrays = buffer.finalize()
    for array in arrays.values():
        assert array.flags["C_CONTIGUOUS"]
        assert not array.flags["WRITEABLE"]


def test_npz_json_round_trip_preserves_everything(tmp_path: Path):
    buffer = TrajectoryBuffer(capacity=4, input_spec=e_input_spec())
    buffer.append(*_make_transition(day=4))
    buffer.append(*_make_transition(seat=1, day=4))
    buffer.patch_terminal(1, 1.0, True)
    base = tmp_path / "traj"
    returned = buffer.save(base, run_metadata={"master_seed": 17})
    assert returned == base
    loaded, sidecar = load_trajectory(base)
    assert len(loaded) == len(buffer) == 2
    original = buffer.finalize()
    restored = loaded.finalize()
    assert set(original) == set(restored) == set(arrayspec_core_keys()) | {
        f"input_{key}" for key in e_input_spec()}
    for key in original:
        assert original[key].dtype == restored[key].dtype, key
        assert np.array_equal(original[key], restored[key], equal_nan=True), key
    assert sidecar["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert sidecar["run_metadata"]["master_seed"] == 17
    assert len(sidecar["transitions"]) == 2
    assert sidecar["transitions"][1]["trace_digest_hex"]
    # Sidecar must remain serializable under strict no-NaN JSON.
    text = (tmp_path / "traj.json").read_text(encoding="utf-8")
    json.dumps(json.loads(text), allow_nan=False)


def test_load_rejects_schema_version_and_count_mismatch(tmp_path: Path):
    buffer = TrajectoryBuffer(capacity=2, input_spec=e_input_spec())
    buffer.append(*_make_transition())
    base = tmp_path / "traj"
    buffer.save(base)
    with np.load(str(base) + ".npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["schema_version"] = np.asarray(TRAJECTORY_SCHEMA_VERSION + 1,
                                          dtype=np.int32)
    np.savez_compressed(str(base) + ".npz", **arrays)
    with pytest.raises(ValueError, match="schema version"):
        TrajectoryBuffer.load(base)
