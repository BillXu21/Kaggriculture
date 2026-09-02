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

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.decode import ACTION_TENSOR_SHAPES
from rl_manager.debug_trace import canonical_json_bytes, validate_trace
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
from tests.test_rl_manager_debug_trace import _state as _canonical_trace_state

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

    # The manager handoff starts at d4: opening-day history is deliberately
    # not consumed by the first manager decision. d5 uses the realized d4
    # daily-start bank, independently for each seat.
    for seat in (0, 1):
        rows = [i for i, (s, d) in enumerate(zip(seats, days))
                if s == seat]
        d4 = next(i for i in rows if days[i] == 4)
        d5 = next(i for i in rows if days[i] == 5)
        assert arrays["input_economic_context"][d4, 13] == 0.0
        assert arrays["input_economic_context"][d5, 13] == 1.0

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


def test_runner_accepts_explicit_seat_openings_and_records_both_provenances():
    config = _runner_config(
        openings=("fourth_quadrant_s0", "fourth_quadrant_s1"))
    assert config.opening_for_seat(0) == "fourth_quadrant_s0"
    assert config.opening_for_seat(1) == "fourth_quadrant_s1"
    runner = SelfPlayRunner(config, executor_factory=_TraceExecutorFactory())
    opening = runner.provenance["opening"]
    assert opening["mode"] == "per_seat"
    assert [record["seat"] for record in opening["by_seat"]] == [0, 1]
    assert [record["name"] for record in opening["by_seat"]] == [
        "fourth_quadrant_s0", "fourth_quadrant_s1"]
    assert [record["source_provenance"]["source_seat"]
            for record in opening["by_seat"]] == [0, 1]
    assert opening["by_seat"][0]["digest"] != opening["by_seat"][1]["digest"]


def test_runner_single_opening_keeps_legacy_resolution_and_provenance():
    config = _runner_config(opening="pasture_heavy")
    assert config.opening_for_seat(0) == config.opening_for_seat(1) \
        == "pasture_heavy"
    runner = SelfPlayRunner(config, executor_factory=_TraceExecutorFactory())
    assert runner.provenance["opening"]["name"] == "pasture_heavy"


def test_runner_d4_previous_execution_uses_realized_d3_hires(monkeypatch):
    """The FQ trace reaches d4 with observed labor, not HIRE intent counts."""
    from oracle.backend import make_backend as make_real_backend

    class CapturingPolicy(_ConstantPlanPolicy):
        def __init__(self):
            super().__init__("capture-d3-hires")
            self.d4_scalars = None

        def plan_batch(self, inputs, prng_id):
            if int(inputs["day"][0]) == 4:
                self.d4_scalars = inputs["scalars"].copy()
            return super().plan_batch(inputs, prng_id)

    class RecordingBackend:
        name = "fast"

        def __init__(self, configuration):
            self._backend = make_real_backend("fast", configuration)

        def reset(self):
            return self._backend.reset()

        def canonical_state(self):
            return self._backend.canonical_state()

        def step(self, actions):
            return self._backend.step(actions)

        @property
        def rewards(self):
            return self._backend.rewards

        @property
        def statuses(self):
            return self._backend.statuses

    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: RecordingBackend(configuration),
    )
    policy = CapturingPolicy()
    runner = SelfPlayRunner(
        _runner_config(
            openings=("fourth_quadrant_s0", "standard_mixed"),
            max_turns=97,
        ),
        master_seed=17,
    )
    result = runner.run([build_episode_spec(0, 17, E_VS_E, policy, policy)])[0]
    assert all(d["divergence"]["occurred"] is False
               for d in result.opening_diagnostics)
    assert policy.d4_scalars is not None
    # Fast-engine d3 observations realize 6 and 5 hires for this fixed pair;
    # the corresponding costs are 20 and 12, respectively.
    assert policy.d4_scalars[:, 2:4].tolist() == [[6.0, 20.0], [5.0, 12.0]]


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


def test_native_batch_runner_matches_scalar_runner_on_truncated_chunk():
    scalar_policy = _ConstantPlanPolicy("scalar")
    scalar_runner, scalar_results = _run_truncated(
        E_VS_E, scalar_policy, scalar_policy)

    batch_policy = _ConstantPlanPolicy("batch")
    config = _runner_config(
        num_envs=2, max_turns=_TRUNCATED_TURNS, batch_backend=True,
        low_telemetry=True, read_only_agent_observations=True)
    batch_runner = SelfPlayRunner(config, master_seed=MASTER_SEED)
    batch_results = batch_runner.run(
        _truncated_specs(E_VS_E, batch_policy, batch_policy, (0, 1)))

    assert len(batch_results) == len(scalar_results) == 2
    for scalar, batched in zip(scalar_results, batch_results):
        assert batched.final_banks == scalar.final_banks
        assert batched.margin == scalar.margin
        assert batched.rewards == scalar.rewards
        assert batched.statuses == scalar.statuses
        assert batched.transitions == scalar.transitions
        assert batched.trace_digest == scalar.trace_digest
    assert batch_runner.timing_totals["env_step"] > 0.0
    assert batch_policy.calls == [
        (batch_policy.identity.identity_id(), 4, 4),
        (batch_policy.identity.identity_id(), 5, 4),
    ]


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


def test_runner_handoff_uses_manager_day_boundary_for_e_history(monkeypatch):
    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: _TraceBackend(configuration),
    )
    policy = _ConstantPlanPolicy("history-boundary")
    buffer = TrajectoryBuffer(capacity=4, input_spec=e_input_spec())
    runner = SelfPlayRunner(
        _runner_config(max_turns=121), trajectory_buffer=buffer,
        executor_factory=_TraceExecutorFactory(), master_seed=MASTER_SEED)
    spec = build_episode_spec(0, MASTER_SEED, E_VS_E, policy, policy)

    runner.run([spec])
    arrays = buffer.finalize()
    for seat in (0, 1):
        rows = [i for i, (row_seat, day) in enumerate(
            zip(arrays["seat"], arrays["day"]))
                if row_seat == seat]
        d4 = next(i for i in rows if arrays["day"][i] == 4)
        d5 = next(i for i in rows if arrays["day"][i] == 5)
        assert arrays["input_economic_context"][d4, 13] == 0.0
        assert arrays["input_economic_context"][d5, 13] == 1.0


class _TraceBackend:
    """Small backend seam using the real canonical fixture, not an env loop."""

    name = "fake-trace"

    def __init__(self, configuration):
        self._step = 0
        self._rewards = [0.0, 0.0]

    def reset(self):
        self._step = 0
        return self._observations()

    def step(self, actions):
        del actions
        self._step += 1
        return self._observations(), list(self._rewards), self.statuses

    def canonical_state(self):
        state = _canonical_trace_state(
            self._step, day=self._step // 24, hour=self._step % 24)
        for farm in state["farms"]:
            tile = farm["tiles"][0][0]
            farm["tiles"] = [[None] * 10 for _ in range(10)]
            farm["tiles"][0][0] = tile
        return state

    @property
    def rewards(self):
        return list(self._rewards)

    @property
    def statuses(self):
        return ["ACTIVE", "ACTIVE"]

    def _observations(self):
        state = self.canonical_state()
        return [
            {
                "day": state["day"],
                "hour": state["hour"],
                "step": state["step"],
                "player": seat,
                "farms": copy.deepcopy(state["farms"]),
                "private": copy.deepcopy(state["privates"][seat]),
                "market": copy.deepcopy(state["market"]),
                "town": copy.deepcopy(state["town"]),
            }
            for seat in range(2)
        ]


class _TraceExecutor:
    def __init__(self, seat):
        self.seat = seat
        self.debug_trace_turn = None

    def __call__(self, obs):
        day, hour = int(obs["day"]), int(obs["hour"])
        if day >= 4:
            self.debug_trace_turn = {
                "schema_version": 1,
                "day": day,
                "hour": hour,
                "seat": self.seat,
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}


class _TraceExecutorFactory:
    name = "trace-test-executor"
    version = "trace-test-v1"

    def create(self, *, backend_name, seat, configuration, provider):
        del backend_name, configuration, provider
        return _TraceExecutor(seat)


class _MutatingTraceExecutor(_TraceExecutor):
    def __call__(self, obs):
        if int(obs["day"]) >= 4:
            obs["farms"][self.seat]["money"] = 1
        return super().__call__(obs)


class _MutatingTraceExecutorFactory(_TraceExecutorFactory):
    def create(self, *, backend_name, seat, configuration, provider):
        del backend_name, configuration, provider
        return _MutatingTraceExecutor(seat)


def _debug_trace_run(*, max_turns: int, enabled: bool, monkeypatch):
    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: _TraceBackend(configuration),
    )
    policy = _ConstantPlanPolicy("trace")
    runner = SelfPlayRunner(
        _runner_config(
            max_turns=max_turns,
            record_debug_trace=enabled,
            debug_trace_seat=1,
        ),
        executor_factory=_TraceExecutorFactory(),
        master_seed=MASTER_SEED,
    )
    spec = build_episode_spec(0, MASTER_SEED, E_VS_E, policy, policy)
    return runner.run([spec])[0]


def test_debug_trace_opt_in_records_reset_decisions_and_terminal_state(monkeypatch):
    result = _debug_trace_run(max_turns=2, enabled=True, monkeypatch=monkeypatch)
    trace = result.debug_trace
    assert trace is not None
    validate_trace(trace)
    assert trace["metadata"]["seed"] == MASTER_SEED
    assert trace["metadata"]["seat"] == 1
    assert trace["metadata"]["view"] == "joint"
    assert [turn["step"] for turn in trace["turns"]] == [0, 1, 2]
    assert [turn["canonical_state"]["step"] for turn in trace["turns"]] == [0, 1, 2]
    assert all("joint_actions" in turn for turn in trace["turns"][:2])
    assert "joint_actions" not in trace["turns"][-1]
    assert trace["turns"][0]["joint_actions"]["0"]["farmer"]


def test_debug_trace_executor_sidecars_attach_to_same_turn_and_seat(monkeypatch):
    result = _debug_trace_run(max_turns=97, enabled=True, monkeypatch=monkeypatch)
    trace = result.debug_trace
    assert trace is not None
    handoff = next(turn for turn in trace["turns"]
                   if turn["step"] == 96)
    assert set(handoff["executor_debug"]) == {"0", "1"}
    assert all(snapshot["day"] == 4 and snapshot["hour"] == 0
               for snapshot in handoff["executor_debug"].values())
    assert all("executor_debug" not in turn
                for turn in trace["turns"] if turn["step"] < 96)


def test_read_only_agent_observation_view_rejects_mutation(monkeypatch):
    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: _TraceBackend(configuration),
    )
    policy = _ConstantPlanPolicy("readonly")
    runner = SelfPlayRunner(
        _runner_config(
            max_turns=97, read_only_agent_observations=True),
        executor_factory=_MutatingTraceExecutorFactory(),
        master_seed=MASTER_SEED,
    )
    spec = build_episode_spec(0, MASTER_SEED, E_VS_E, policy, policy)
    with pytest.raises(TypeError, match="read-only"):
        runner.run([spec])


def test_debug_trace_is_deterministic_and_behavior_matches_disabled_capture(monkeypatch):
    first = _debug_trace_run(max_turns=2, enabled=True, monkeypatch=monkeypatch)
    second = _debug_trace_run(max_turns=2, enabled=True, monkeypatch=monkeypatch)
    disabled = _debug_trace_run(max_turns=2, enabled=False, monkeypatch=monkeypatch)
    assert first.debug_trace is not None and second.debug_trace is not None
    assert canonical_json_bytes(first.debug_trace) \
        == canonical_json_bytes(second.debug_trace)
    for field in ("final_banks", "margin", "winner_seat", "rewards",
                  "statuses", "transitions", "terminated", "trace_digest"):
        assert getattr(first, field) == getattr(disabled, field), field
    assert disabled.debug_trace is None


def test_debug_trace_invalid_selector_configuration():
    with pytest.raises(ValueError, match="debug_trace_seat"):
        SelfPlayRunner(_runner_config(debug_trace_seat=2))
    with pytest.raises(ValueError, match="debug_trace_view"):
        SelfPlayRunner(_runner_config(debug_trace_view=""))
