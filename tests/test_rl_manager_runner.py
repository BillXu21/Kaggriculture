"""Self-play runner Stage-A acceptance tests (issue #9 A3).

Covers, with exactly ONE complete fast-engine game (run twice for the
deterministic rerun-equality requirement, numThreads=1):

- deterministic E-vs-E full game: 52 transitions (d4..d29 x both seats),
  no manager decisions d0-d3, first manager decision at d4h0, DONE/DONE,
  terminal-only rewards, per-day joint-action digests sealed on every
  manager row, exact rerun equality (`equal_nan=True` for the NaN sentinel
  in board_numeric), NPZ+JSON round-trip, and full provenance;
- composition/orientation/batching proofs on short TRUNCATED fast-engine
  chunks (never additional complete games): N=2 lockstep envs grouped by
  policy identity produce ONE policy call per (identity, day) with a
  contiguous batch spanning envs -- never one call per environment --
  plus trainable-flag/opponent-provenance checks for both seat
  orientations via a constant-plan fake policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.decode import ACTION_TENSOR_SHAPES
from rl_manager.executor_factory import EXECUTOR_FACTORY_VERSION
from rl_manager.policy import JaxEPlanPolicy, params_fingerprint
from rl_manager.provenance import sha256_hex
from rl_manager.runner import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    GAME_TURNS,
    MANAGER_START_DAY,
    RunnerConfig,
    SelfPlayRunner,
    build_episode_spec,
)
from rl_manager.seeds import SeedStream
from rl_manager.trajectory import TrajectoryBuffer, e_input_spec, load_trajectory
from rl_manager.types import (
    CANDIDATE_VS_FROZEN,
    E_VS_E,
    FROZEN_VS_CANDIDATE,
    PolicyIdentity,
    PolicyOutputs,
)

MASTER_SEED = 17
NUM_MANAGER_DAYS = 26  # d4..d29 per seat
TOTAL_TRANSITIONS = 2 * NUM_MANAGER_DAYS
_UNSEALED_DIGEST = hashlib.sha256(b"unsealed").digest()


@pytest.fixture(scope="module")
def tiny_e():
    config = tiny_manager_config()
    params = init_params(config, seed=11, model_variant="E")
    return params, config


class _ConstantPlanPolicy:
    """Fake batched policy: fixed valid plan tensors + call instrumentation.

    Records `(identity_id, day, batch_size)` per `plan_batch` call so tests
    can prove grouping happens per (policy identity, day) rather than per
    environment. The constant plan (all-zero targets, land count 1, no
    sells) is decodable and accepted by the unmodified executor.
    """

    def __init__(self, name: str) -> None:
        self.identity = PolicyIdentity(
            name=name, version="fake-v1", fingerprint=f"fake-{name}")
        self.calls: list[tuple[str, int, int]] = []

    def plan_batch(self, inputs, prng_id):
        if not isinstance(prng_id, str) or not prng_id:
            raise ValueError("prng_id must be a non-empty string identifier")
        batch_size = int(np.asarray(inputs["board_kind"]).shape[0])
        day = int(np.asarray(inputs["day"]).ravel()[0])
        self.calls.append((self.identity.identity_id(), day, batch_size))
        action_tensors = {
            name: np.zeros((batch_size,) + shape, dtype=np.int16)
            for name, shape in ACTION_TENSOR_SHAPES.items()}
        action_tensors["land"] = np.ones((batch_size,), dtype=np.int16)
        zeros = np.zeros(batch_size, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=action_tensors,
            logprob_groups={group: zeros.copy() for group in (
                "crop", "animal", "land", "fertilizer", "care",
                "sell_presence")},
            logprob_total=zeros.copy(),
            value=zeros.copy(),
            batch_size=batch_size,
        )


def _runner_config(**overrides) -> RunnerConfig:
    kwargs: dict = {
        "backend_name": "fast",
        "backend_configuration": {"seed": 0, "numThreads": 1},
        "max_turns": GAME_TURNS,
        "num_envs": 1,
    }
    kwargs.update(overrides)
    return RunnerConfig(**kwargs)


# --------------------------------------------------------------------------
# Complete fast game (the single authorized smoke), run exactly twice.
# --------------------------------------------------------------------------

_GAME_CACHE: dict[str, object] = {}


def _tiny_e_params():
    config = tiny_manager_config()
    params = init_params(config, seed=11, model_variant="E")
    return params, config


def full_game_pair():
    """Run the E-vs-E fast game twice (record_rollout=True); cache results.

    Self-contained (builds its own tiny JAX E) so
    `test_rl_manager_parity.py` can reuse the exact same deterministic pair
    without re-running any complete game.
    """
    if "pair" in _GAME_CACHE:
        return _GAME_CACHE["pair"]
    params, config = _tiny_e_params()
    policy = JaxEPlanPolicy(params, config, name="jax_e")
    seed = SeedStream(MASTER_SEED).episode_seed(0)
    buffers, results = [], []
    for _rep in range(2):
        buffer = TrajectoryBuffer(capacity=64, input_spec=e_input_spec())
        runner = SelfPlayRunner(
            _runner_config(record_rollout=True),
            trajectory_buffer=buffer, master_seed=MASTER_SEED)
        spec = build_episode_spec(
            0, seed, E_VS_E, policy, policy)
        results.append(runner.run([spec])[0])
        buffers.append(buffer)
    _GAME_CACHE["pair"] = {
        "results": results, "buffers": buffers, "params": params,
        "config": config, "policy": policy, "runner": runner,
        "provenance": dict(runner.provenance),
    }
    return _GAME_CACHE["pair"]


def test_full_game_deterministic_52_transitions_and_schema(tiny_e):
    game = full_game_pair()
    result, buffer = game["results"][0], game["buffers"][0]

    assert result.statuses == ["DONE", "DONE"]
    assert result.terminated is True
    assert result.transitions == TOTAL_TRANSITIONS
    assert len(buffer) == TOTAL_TRANSITIONS

    # finalize() exposes the full preallocated capacity; filled rows are
    # the first `len(buffer)`.
    arrays = {key: value[:len(buffer)]
              for key, value in buffer.finalize().items()}
    seats = arrays["seat"].tolist()
    days = arrays["day"].tolist()
    # Exactly d4..d29 for each seat; no manager decision before d4h0.
    assert sorted(days) == sorted(list(range(4, 30)) * 2)
    for seat in (0, 1):
        assert sorted(d for s, d in zip(seats, days) if s == seat) \
            == list(range(4, 30))
    assert int(arrays["day"].min()) == MANAGER_START_DAY

    # Terminal-only rewards: only final-day rows may carry +/-1 (both zero
    # on a tie), matching the final bank margin sign.
    reward_rows = [(i, float(r)) for i, r in enumerate(arrays["reward"])
                   if float(r) != 0.0]
    final_indices = [i for i, d in enumerate(days) if d == 29]
    assert all(days[i] == 29 for i, _ in reward_rows)
    assert sum(r for _, r in reward_rows) == 0.0
    if result.winner_seat >= 0:
        assert sorted(i for i, _ in reward_rows) == sorted(final_indices)
        assert arrays["terminated"][final_indices[0]] == 1
        margin_sign = np.sign(result.margin)
        rewarded = {seats[i]: r for i, r in reward_rows}
        assert np.sign(rewarded[0]) == margin_sign
        assert np.sign(rewarded[1]) == -margin_sign
    else:
        assert not reward_rows  # exact tie: +0/-0 per issue #9 B4

    # Every manager row's primitive-trace digest was sealed after its day's
    # turns completed (never left with the placeholder).
    for i in range(TOTAL_TRANSITIONS):
        row_digest = bytes(arrays["trace_digest"][i].tolist())
        assert row_digest != _UNSEALED_DIGEST
        assert row_digest.hex() \
            == buffer.sidecar_records[i].trace_digest_hex

    # Provenance: policy/opponent identities, fingerprints, ownership.
    fingerprint = params_fingerprint(game["params"])
    for record in buffer.sidecar_records:
        assert record.policy_fingerprint == fingerprint
        assert record.policy_version == "stage-a-v1"
        assert record.opponent_id == record.policy_id  # E-vs-E baseline
        assert record.trainable is False  # no trainable seats in E-vs-E
    assert game["provenance"]["master_seed"] == MASTER_SEED
    assert game["provenance"]["manager_start_day"] == MANAGER_START_DAY
    assert game["provenance"]["backend"]["configuration"]["numThreads"] == 1
    assert game["provenance"]["opening"]["name"] == "standard_mixed"
    assert game["provenance"]["opening"]["digest"]

    # Timing split is recorded for every requested phase.
    for phase in ("manager_inference", "agent_actions", "env_step",
                  "orchestration"):
        assert phase in result.timing_seconds


def test_full_game_rerun_equality_equal_nan(tmp_path: Path, tiny_e):
    game = full_game_pair()
    result_a, result_b = game["results"]
    buffer_a, buffer_b = game["buffers"]

    for field in ("seed", "composition", "final_banks", "margin",
                  "winner_seat", "rewards", "statuses", "transitions",
                  "terminated", "trace_digest"):
        assert getattr(result_a, field) == getattr(result_b, field), field

    arrays_a = buffer_a.finalize()
    arrays_b = buffer_b.finalize()
    assert set(arrays_a) == set(arrays_b)
    for key in arrays_a:
        # board_numeric carries the NaN sentinel: comparisons MUST use
        # equal_nan=True; everything must still match exactly otherwise.
        assert arrays_a[key].dtype == arrays_b[key].dtype, key
        assert np.array_equal(arrays_a[key], arrays_b[key],
                              equal_nan=True), key

    # NPZ + JSON sidecar round-trip preserves the episode exactly, via the
    # normal runner artifact path (no caller-side run_metadata assembly).
    base = tmp_path / "full_game_traj"
    assert game["runner"].save_trajectory_artifact(
        base, buffer_a, result_a) == base
    loaded, sidecar = load_trajectory(base)
    assert sidecar["run_metadata"]["master_seed"] == MASTER_SEED
    assert len(loaded) == TOTAL_TRANSITIONS
    loaded_arrays = loaded.finalize()
    for key in arrays_a:
        assert np.array_equal(loaded_arrays[key][:TOTAL_TRANSITIONS],
                              arrays_a[key][:TOTAL_TRANSITIONS],
                              equal_nan=True), key


def test_full_game_opening_handoff_and_episode_digest(tiny_e):
    """The committed opening alone drives d0-d3 (diagnostics recorded for
    both seats) and the episode-level digest reproduces across the
    deterministic pair."""
    game = full_game_pair()
    result = game["results"][0]
    for seat in range(2):
        diagnostics = result.opening_diagnostics[seat]
        assert isinstance(diagnostics, dict)
    assert result.trace_digest == game["results"][1].trace_digest


def test_artifact_save_load_records_mandatory_provenance_automatically(
        tmp_path: Path, tiny_e):
    """Issue #9 A1 correction: `save_trajectory_artifact` merges episode
    outcome + runner provenance + policy/opponent identities into the JSON
    sidecar automatically (reuses the cached complete tiny-E game; no caller
    assembly anywhere in this test)."""
    game = full_game_pair()
    result, buffer, runner = (
        game["results"][0], game["buffers"][0], game["runner"])
    base = tmp_path / "artifact_traj"
    assert runner.save_trajectory_artifact(base, buffer, result) == base
    loaded, sidecar = load_trajectory(base)
    meta = sidecar["run_metadata"]

    # Schema/version + run-level provenance.
    assert meta["artifact_schema_version"] == ARTIFACT_METADATA_SCHEMA_VERSION
    assert meta["master_seed"] == MASTER_SEED
    assert meta["manager_start_day"] == MANAGER_START_DAY

    # Exact per-episode outcome values.
    episode = meta["episode"]
    assert episode["episode_index"] == result.episode_index == 0
    assert episode["seed"] == result.seed
    assert episode["composition"] == E_VS_E
    assert episode["final_banks"] == result.final_banks
    assert episode["margin"] == result.margin
    assert episode["winner_seat"] == result.winner_seat
    assert episode["rewards"] == result.rewards
    assert episode["statuses"] == ["DONE", "DONE"]
    assert episode["terminated"] is True
    assert episode["transitions"] == TOTAL_TRANSITIONS
    assert episode["trace_digest"] == result.trace_digest
    assert episode["rollout_recorded"] is True  # trace ref, not a giant copy
    assert set(episode["timing_seconds"]) == {
        "manager_inference", "agent_actions", "env_step", "orchestration"}

    # Opening identity/digest + backend/engine provenance.
    assert meta["opening"]["name"] == "standard_mixed"
    assert meta["opening"]["digest"]
    assert meta["backend"]["backend"] == "fast"
    assert meta["backend"]["configuration"]["numThreads"] == 1

    # Executor factory version/hash/identifier.
    factory = meta["executor_factory"]
    assert factory["name"] == "executor_v0"
    assert factory["version"] == EXECUTOR_FACTORY_VERSION
    assert factory["identifier"] == f"executor_v0@{EXECUTOR_FACTORY_VERSION}"
    assert factory["version_sha256"] == sha256_hex(factory["identifier"])

    # Per-seat policy/opponent identities/versions/fingerprints.
    fingerprint = params_fingerprint(game["params"])
    identity = game["policy"].identity
    assert [record["seat"] for record in meta["policies"]] == [0, 1]
    for record in meta["policies"]:
        assert record["policy"]["identity_id"] == identity.identity_id()
        assert record["policy"]["name"] == identity.name
        assert record["policy"]["version"] == identity.version
        assert record["policy"]["fingerprint"] == fingerprint
        assert record["opponent"]["identity_id"] == identity.identity_id()
        assert record["opponent"]["fingerprint"] == fingerprint
        assert record["trainable"] is False  # E-vs-E baseline

    # Round-trip preserves rows exactly; sidecar is strictly JSON-safe with
    # deterministic key ordering.
    assert len(loaded) == TOTAL_TRANSITIONS
    text = (tmp_path / "artifact_traj.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    json.dumps(parsed, allow_nan=False, sort_keys=True)
    assert list(parsed["run_metadata"]) == sorted(parsed["run_metadata"])


# --------------------------------------------------------------------------
# Composition / orientation / batching proofs on truncated fast chunks.
# --------------------------------------------------------------------------

_TRUNCATED_TURNS = 130  # reaches the d4+d5 manager boundaries, never DONE


def _truncated_specs(composition, candidate, frozen, episode_indices):
    seeds = [SeedStream(MASTER_SEED).episode_seed(i + 100)
             for i in episode_indices]
    return [build_episode_spec(i, seed, composition, candidate, frozen)
            for i, seed in zip(episode_indices, seeds)]


def _run_truncated(composition, candidate, frozen, buffer=None):
    config = _runner_config(num_envs=2, max_turns=_TRUNCATED_TURNS)
    runner = SelfPlayRunner(config, trajectory_buffer=buffer,
                            master_seed=MASTER_SEED)
    specs = _truncated_specs(composition, candidate, frozen, (0, 1))
    results = runner.run(specs)
    assert all(not r.terminated for r in results)  # truncated, not done
    return runner, results


def test_batching_e_vs_e_one_call_per_day_across_two_envs():
    policy = _ConstantPlanPolicy("shared_e")
    _run_truncated(E_VS_E, policy, policy)
    by_day: dict[int, list[int]] = {}
    for _identity, day, batch_size in policy.calls:
        by_day.setdefault(day, []).append(batch_size)
    # Both lockstep envs x both seats share ONE contiguous batch per day.
    assert sorted(by_day) == [4, 5]
    assert all(sizes == [4] for sizes in by_day.values())
    assert len(policy.calls) == 2  # two days, NOT 2 envs x 2 seats x 2 days


def test_batching_candidate_vs_frozen_two_calls_per_day():
    candidate = _ConstantPlanPolicy("candidate")
    frozen = _ConstantPlanPolicy("frozen")
    buffer = TrajectoryBuffer(capacity=16, input_spec=e_input_spec())
    _run_truncated(CANDIDATE_VS_FROZEN, candidate, frozen, buffer=buffer)
    by_day: dict[int, list[tuple[str, int]]] = {}
    for identity, day, batch_size in candidate.calls + frozen.calls:
        by_day.setdefault(day, []).append((identity, batch_size))
    assert sorted(by_day) == [4, 5]
    for day in (4, 5):
        calls = by_day[day]
        assert sorted(calls) == [
            (candidate.identity.identity_id(), 2),
            (frozen.identity.identity_id(), 2)]
    # Distinct identities never share a batch.
    assert len(candidate.calls) == 2 and len(frozen.calls) == 2

    # Trainable ownership + opponent provenance for this orientation.
    arrays = buffer.finalize()
    for record, seat in zip(buffer.sidecar_records, arrays["seat"]):
        assert record.trainable is (int(seat) == 0)
        expected_policy = candidate if int(seat) == 0 else frozen
        expected_opponent = frozen if int(seat) == 0 else candidate
        assert record.policy_id == expected_policy.identity.identity_id()
        assert record.opponent_id == expected_opponent.identity.identity_id()
    # Truncated episodes: terminal reward stays absent, truncation flagged
    # on the final row of each of the 2 envs x 2 seats.
    assert float(np.abs(arrays["reward"]).sum()) == 0.0
    assert int(arrays["truncated"].sum()) == 4


def test_orientation_frozen_vs_candidate_trainable_seat_1():
    candidate = _ConstantPlanPolicy("candidate")
    frozen = _ConstantPlanPolicy("frozen")
    buffer = TrajectoryBuffer(capacity=16, input_spec=e_input_spec())
    _run_truncated(FROZEN_VS_CANDIDATE, candidate, frozen, buffer=buffer)
    arrays = buffer.finalize()
    for record, seat in zip(buffer.sidecar_records, arrays["seat"]):
        assert record.trainable is (int(seat) == 1)
        expected_policy = frozen if int(seat) == 0 else candidate
        expected_opponent = candidate if int(seat) == 0 else frozen
        assert record.policy_id == expected_policy.identity.identity_id()
        assert record.opponent_id == expected_opponent.identity.identity_id()


def test_build_episode_spec_rejects_unknown_composition_and_evse_aliasing(
        tiny_e):
    params, config = tiny_e
    policy = JaxEPlanPolicy(params, config)
    other = JaxEPlanPolicy(params, config, name="other")  # different identity
    with pytest.raises(ValueError, match="unknown composition"):
        build_episode_spec(0, 1, "nope", policy, policy)
    with pytest.raises(ValueError, match="identical policies"):
        build_episode_spec(0, 1, E_VS_E, policy, other)
    spec = build_episode_spec(0, 1, FROZEN_VS_CANDIDATE, policy, other)
    assert spec.trainable_seats == (1,)
    assert spec.policies[0] is other and spec.policies[1] is policy


def test_truncated_results_carry_distinct_explicit_seeds():
    candidate = _ConstantPlanPolicy("candidate")
    frozen = _ConstantPlanPolicy("frozen")
    _runner, results = _run_truncated(CANDIDATE_VS_FROZEN, candidate, frozen)
    seeds = [r.seed for r in results]
    assert len(set(seeds)) == len(seeds)  # explicit distinct seed stream
    stream = SeedStream(MASTER_SEED)
    assert seeds == [stream.episode_seed(i + 100) for i in (0, 1)]
