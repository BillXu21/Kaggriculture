"""Focused tests for the framework-free Stage 2.5 trajectory boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from rl_manager.stage25_trajectory import (
    STAGE25_TRAJECTORY_SCHEMA_VERSION,
    Stage25TrajectoryBuffer,
    Stage25TrajectoryRow,
    audit_teacher_force,
    load_stage25_trajectory,
    stage25_input_spec,
)
from rl_manager.stage25_types import Stage25BehaviorIdentity
from rl_manager.stage25_types import Stage25PolicyOutputs


def _identity(name: str) -> Stage25BehaviorIdentity:
    return Stage25BehaviorIdentity(
        name=name, version="v1", parameter_fingerprint=f"params-{name}",
        observation_schema_version="stage25_corrected_e_own_only_v1",
        policy_schema_version="stage25_policy_v1", e_history_version="E_CORRECTED_V1",
        curriculum_version="stage25_curriculum_v1",
        curriculum_fingerprint=f"curriculum-{name}",
    )


def _inputs(day: int = 4) -> dict[str, np.ndarray]:
    result = {}
    for name, (shape, dtype) in stage25_input_spec().items():
        result[name] = np.zeros(shape, dtype=dtype)
    result["day"] = np.asarray(day, dtype=np.int16)
    result["days_remaining"] = np.asarray(25 - day, dtype=np.int16)
    result["economic_context"] = np.arange(14, dtype=np.float32)
    result["crop_capacity"] = np.asarray([1, 2, 3, 4, 5], dtype=np.int16)
    result["unlocked"][0] = 1
    return result


def _row(episode: int = 1, seat: int = 0, day: int = 4, row_id: str | None = None) -> Stage25TrajectoryRow:
    classes = np.asarray([0, 0, 0, 0, 100, 100, 100, 100, 100], dtype=np.int16)
    components = np.asarray([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9], dtype=np.float32)
    return Stage25TrajectoryRow(
        episode_id=episode, seat=seat, day=day, row_id=row_id,
        inputs=_inputs(day), classes=classes, component_logprobs=components,
        joint_logprob=np.asarray(components.sum(), dtype=np.float32),
        value=np.asarray(0.25, dtype=np.float32), learner_identity=_identity("learner"),
        opponent_identity=_identity("opponent"),
        provenance={"executor": {"name": "test-executor", "version": "v1"}},
    )


def test_round_trip_preserves_inputs_outputs_identity_and_provenance(tmp_path: Path):
    buffer = Stage25TrajectoryBuffer(3)
    buffer.append(_row(day=4, row_id="r0"))
    buffer.append(_row(episode=2, seat=1, day=5, row_id="r1"))
    buffer.patch_terminal(1, np.asarray(1.5, dtype=np.float32))
    base = tmp_path / "stage25"
    buffer.save(base, run_metadata={"master_seed": 17})

    loaded, sidecar = load_stage25_trajectory(base)
    assert len(loaded) == 2
    assert sidecar["schema_version"] == STAGE25_TRAJECTORY_SCHEMA_VERSION
    assert sidecar["run_metadata"] == {"master_seed": 17}
    np.testing.assert_array_equal(loaded.finalize()["input_crop_capacity"], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(loaded.finalize()["classes"], [_row().classes, _row().classes])
    assert float(loaded.finalize()["reward"][1]) == pytest.approx(1.5)
    assert loaded.rows[0].learner_identity.identity_id() == _identity("learner").identity_id()
    assert loaded.rows[1].provenance["executor"]["name"] == "test-executor"


def test_append_can_consume_the_shared_one_row_policy_output_contract():
    source = _row()
    outputs = Stage25PolicyOutputs(
        classes=source.classes[None, :],
        component_logprobs=source.component_logprobs[None, :],
        joint_logprob=np.asarray([source.joint_logprob], dtype=np.float32),
        value=np.asarray([source.value], dtype=np.float32),
        decoded_goals=np.zeros((1, 5), dtype=np.int16),
        valid=np.ones(1, dtype=np.bool_), policy_identity=source.learner_identity,
        batch_size=1)
    buffer = Stage25TrajectoryBuffer(1)
    assert buffer.append(
        inputs=source.inputs, outputs=outputs, episode_id=1, seat=0, day=4,
        opponent_identity=source.opponent_identity, provenance=source.provenance) == 0


def test_episode_seat_day_identity_is_separate_and_opening_rows_are_rejected():
    buffer = Stage25TrajectoryBuffer(4)
    buffer.append(_row(episode=10, seat=0, day=4))
    buffer.append(_row(episode=10, seat=1, day=4))
    buffer.append(_row(episode=11, seat=0, day=4))
    with pytest.raises(ValueError, match="duplicate episode/seat/day"):
        buffer.append(_row(episode=10, seat=0, day=4))
    with pytest.raises(ValueError, match="day must be"):
        buffer.append(_row(episode=99, seat=0, day=3))


def test_terminal_reward_is_applied_once_and_only_to_final_row():
    buffer = Stage25TrajectoryBuffer(3)
    buffer.append(_row(day=4))
    buffer.close_outgoing(
        episode_index=1, seat=0, next_day=5, next_inputs=_inputs(5),
        next_crop_capacity=np.asarray([1, 2, 3, 4, 5], dtype=np.int16))
    buffer.append(_row(day=5))
    with pytest.raises(ValueError, match="final manager row"):
        buffer.patch_terminal(0, np.asarray(2.0, dtype=np.float32))
    buffer.patch_terminal(1, np.asarray(2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="terminal/truncation"):
        buffer.patch_terminal(1, np.asarray(2.0, dtype=np.float32))
    assert buffer.finalize()["reward"].tolist() == [0.0, 2.0]


def test_outgoing_transition_must_close_before_next_manager_row():
    buffer = Stage25TrajectoryBuffer(2)
    buffer.append(_row(day=4))
    with pytest.raises(ValueError, match="outgoing manager transition"):
        buffer.append(_row(day=5))
    buffer.close_outgoing(
        episode_index=1, seat=0, next_day=5, next_inputs=_inputs(5),
        next_crop_capacity=np.asarray([1, 2, 3, 4, 5], dtype=np.int16))
    assert buffer.append(_row(day=5)) == 1


def test_truncation_requires_and_preserves_bootstrap_value(tmp_path: Path):
    buffer = Stage25TrajectoryBuffer(2)
    buffer.append(_row())
    buffer.patch_truncated(0)
    with pytest.raises(ValueError, match="requires a patched bootstrap"):
        buffer.save(tmp_path / "stage25-truncation-without-bootstrap")
    buffer.patch_bootstrap(0, np.asarray(0.75, dtype=np.float32))
    assert float(buffer.finalize()["bootstrap_value"][0]) == pytest.approx(0.75)


def test_invalid_shape_dtype_and_zero_likelihood_validity_are_checked():
    buffer = Stage25TrajectoryBuffer(2)
    bad_inputs = _inputs()
    bad_inputs["crop_capacity"] = np.zeros((2, 5), dtype=np.int16)
    with pytest.raises(ValueError, match="input 'crop_capacity'"):
        buffer.append(Stage25TrajectoryRow(**{**_row().__dict__, "inputs": bad_inputs}))
    with pytest.raises(ValueError, match="classes must"):
        buffer.append(Stage25TrajectoryRow(**{**_row().__dict__, "classes": np.zeros(9, dtype=np.int32)}))
    zero = _row()
    zero = Stage25TrajectoryRow(**{
        **zero.__dict__, "component_logprobs": np.zeros(9, dtype=np.float32),
        "joint_logprob": np.asarray(0.0, dtype=np.float32),
    })
    assert buffer.append(zero) == 0
    invalid = Stage25TrajectoryRow(**{
        **_row(episode=2).__dict__, "valid": False})
    with pytest.raises(ValueError, match="invalid Stage 2.5"):
        buffer.append(invalid)


def test_npz_and_sidecar_are_pickle_free(tmp_path: Path):
    buffer = Stage25TrajectoryBuffer(1)
    buffer.append(_row())
    base = tmp_path / "safe"
    buffer.save(base)
    with np.load(str(base) + ".npz", allow_pickle=False) as archive:
        assert all(array.dtype != object for array in archive.values())
    with zipfile.ZipFile(str(base) + ".npz") as archive:
        assert not any(name.endswith(".pkl") for name in archive.namelist())
    json.loads((tmp_path / "safe.json").read_text(encoding="utf-8"))
    def teacher_force(inputs, classes):
        del inputs
        row = buffer.rows[0]
        return {"classes": classes, "component_logprobs": row.component_logprobs[None, :],
                "joint_logprob": np.asarray([row.joint_logprob], dtype=np.float32),
                "value": np.asarray([row.value], dtype=np.float32)}

    audit = audit_teacher_force(buffer, teacher_force)
    assert audit.passed and audit.rows_checked == 1


def test_trajectory_module_does_not_import_accelerator_frameworks():
    script = """
import sys
import rl_manager.stage25_trajectory
assert not any(name == 'jax' or name.startswith('jax.') or
               name == 'torch' or name.startswith('torch.')
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
