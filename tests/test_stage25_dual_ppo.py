from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rl_manager.runner import build_episode_spec
from rl_manager.stage25_checkpoint import (
    DUAL_PPO_TRAINING_PAYLOAD_KIND,
    PPO_TRAINING_PAYLOAD_KIND,
    Stage25CheckpointError,
    _read_archive,
    load_stage25_dual_ppo_checkpoint,
    save_stage25_dual_ppo_checkpoint,
    save_stage25_ppo_checkpoint,
)
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_policy import init_stage25_params, stochastic_act
from rl_manager.stage25_ppo import (
    Stage25PPOBatch,
    Stage25PPOConfig,
    audit_stage25_ppo_rollout,
    build_stage25_dual_ppo_batches,
    init_stage25_dual_ppo_state,
    ppo_update_dual,
)
from rl_manager.stage25_trajectory import (
    INPUT_SPEC,
    Stage25TrajectoryBuffer,
    Stage25TrajectoryRow,
)
from rl_manager.types import DUAL_POLICY_SELF_PLAY


def _fixture(config: Stage25PPOConfig):
    params = init_stage25_params(config.model, seed=13)
    policy_a = Stage25InferenceAdapter(
        params=params, config=config.model, name="stage25_dual_a",
        version="ppo-native-v1", seed=31, mode="stochastic")
    policy_b = Stage25InferenceAdapter(
        params=params, config=config.model, name="stage25_dual_b",
        version="ppo-native-v1", seed=37, mode="stochastic")
    state = init_stage25_dual_ppo_state(
        config, seed=101, params=params,
        behavior_identity_a=policy_a.identity,
        behavior_identity_b=policy_b.identity)
    return params, state


def _inputs(day: int = 4, rows: int = 1):
    result = {
        name: np.zeros((rows,) + shape, dtype=dtype)
        for name, (shape, dtype) in INPUT_SPEC.items()
    }
    result["unlocked"][:, 0] = 1
    result["days_remaining"][:] = 25 - day
    result["day"][:] = day
    result["opponent_summary"][:, 0] = 0.5
    return result


def _trajectory(state, games: int = 6):
    trajectory = Stage25TrajectoryBuffer(games * 2)
    ids = (state.policy_a.behavior_identity,
           state.policy_b.behavior_identity)
    for episode in range(games):
        seat_policies = ids if episode % 2 == 0 else ids[::-1]
        for seat in (0, 1):
            learner = seat_policies[seat]
            opponent = seat_policies[1 - seat]
            row = Stage25TrajectoryRow(
                episode_id=episode, seed=100 + episode, seat=seat, day=4,
                inputs={name: value[0] for name, value in _inputs().items()},
                classes=np.zeros((9,), dtype=np.int16),
                component_logprobs=np.zeros((9,), dtype=np.float32),
                joint_logprob=np.asarray(0.0, dtype=np.float32),
                value=np.asarray(0.0, dtype=np.float32),
                learner_identity=learner, opponent_identity=opponent,
                provenance={"executor": {"name": "test", "version": "v1"}},
                terminated=True, reward_patched=True,
                row_id=f"episode={episode}/seat={seat}/day=4")
            trajectory.append(row)
    return trajectory


def _ppo_batch(params, identity, config, *, source_prefix: str) -> Stage25PPOBatch:
    inputs = _inputs(rows=2)
    inputs["scalars"][1, :] = 1.0
    sampled = stochastic_act(
        params, inputs, config.model,
        jax.random.split(jax.random.PRNGKey(23), 2),
        row_ids=np.arange(2), reject_invalid=True)
    values = np.asarray(sampled["value"], dtype=np.float32)
    return Stage25PPOBatch(
        inputs=inputs, classes=np.asarray(sampled["classes"]),
        old_component_logprobs=np.asarray(sampled["component_logprobs"]),
        old_joint_logprobs=np.asarray(sampled["joint_logprob"]),
        old_values=values, advantages=np.asarray([0.5, 1.5], np.float32),
        returns=values + np.asarray([0.75, 1.25], np.float32),
        episode_id=np.asarray([1, 2]), seat=np.asarray([0, 1]),
        day=np.asarray([4, 4]), row_ids=np.arange(2),
        source_row_ids=(f"{source_prefix}-0", f"{source_prefix}-1"),
        behavior_identity=identity)


def _same_tree(left, right):
    a, b = jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    return len(a) == len(b) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def _physical_contract(config):
    from rl_manager.stage25_ppo_cli import _physical_contract as shared

    return shared(config)


def test_dual_composition_trainable_seats_and_even_rollout_seat_balance():
    config = Stage25PPOConfig()
    _, state = _fixture(config)
    a = Stage25InferenceAdapter(
        params=state.policy_a.params, config=config.model,
        name=state.policy_a.behavior_identity.name,
        version=state.policy_a.behavior_identity.version, seed=31)
    b = Stage25InferenceAdapter(
        params=state.policy_b.params, config=config.model,
        name=state.policy_b.behavior_identity.name,
        version=state.policy_b.behavior_identity.version, seed=37)
    specs = [build_episode_spec(i, 500 + i, DUAL_POLICY_SELF_PLAY, a, b)
             for i in range(512)]
    assert all(spec.trainable_seats == (0, 1) for spec in specs)
    assert all(spec.policies == (a, b) for spec in specs[::2])
    assert all(spec.policies == (b, a) for spec in specs[1::2])
    assert sum(spec.policies[0] is a for spec in specs) == 256
    assert sum(spec.policies[1] is a for spec in specs) == 256


def test_dual_seat_assignment_is_deterministic_and_odd_size_balanced():
    config = Stage25PPOConfig()
    _, state = _fixture(config)
    a = Stage25InferenceAdapter(
        params=state.policy_a.params, config=config.model,
        name=state.policy_a.behavior_identity.name,
        version=state.policy_a.behavior_identity.version, seed=31)
    b = Stage25InferenceAdapter(
        params=state.policy_b.params, config=config.model,
        name=state.policy_b.behavior_identity.name,
        version=state.policy_b.behavior_identity.version, seed=37)
    first = [build_episode_spec(i, i, DUAL_POLICY_SELF_PLAY, a, b)
             for i in range(7)]
    second = [build_episode_spec(i, i, DUAL_POLICY_SELF_PLAY, a, b)
              for i in range(7)]
    assert [s.policies for s in first] == [s.policies for s in second]
    assert sum(s.policies[0] is a for s in first) == 4
    assert sum(s.policies[0] is b for s in first) == 3


def test_dual_identity_partition_is_complete_disjoint_and_seat_independent():
    _, state = _fixture(Stage25PPOConfig())
    trajectory = _trajectory(state, games=9)
    batch_a, batch_b = build_stage25_dual_ppo_batches(
        trajectory, learner_identity_a=state.policy_a.behavior_identity,
        learner_identity_b=state.policy_b.behavior_identity)
    ids_a, ids_b = set(batch_a.source_row_ids), set(batch_b.source_row_ids)
    assert ids_a.isdisjoint(ids_b)
    assert ids_a | ids_b == {row.row_id for row in trajectory.rows}
    assert all(row.learner_identity == state.policy_a.behavior_identity
               for row in trajectory.rows if row.row_id in ids_a)
    assert all(row.learner_identity == state.policy_b.behavior_identity
               for row in trajectory.rows if row.row_id in ids_b)
    assert set(batch_a.seat) == {0, 1}
    assert set(batch_b.seat) == {0, 1}


def test_dual_init_has_distinct_identity_params_optimizer_and_rng():
    params, state = _fixture(Stage25PPOConfig())
    assert state.policy_a.behavior_identity != state.policy_b.behavior_identity
    assert state.policy_a.behavior_identity.parameter_fingerprint == \
        state.policy_b.behavior_identity.parameter_fingerprint
    assert state.policy_a.params is not state.policy_b.params
    assert _same_tree(state.policy_a.params, state.policy_b.params)
    assert state.policy_a.optimizer_state is not state.policy_b.optimizer_state
    assert not np.array_equal(state.policy_a.rng, state.policy_b.rng)
    assert state.policy_a.update_counter == state.policy_b.update_counter == 0
    assert _same_tree(params, state.policy_a.params)


def test_dual_ppo_audits_both_frozen_parameters_then_updates_independently(monkeypatch):
    import rl_manager.stage25_ppo as ppo

    config = Stage25PPOConfig(
        physical_batch_size=1, minibatch_size=1, epochs=1, learning_rate=1e-3)
    params, state = _fixture(config)
    batch_a = _ppo_batch(
        state.policy_a.params, state.policy_a.behavior_identity, config,
        source_prefix="A")
    batch_b = _ppo_batch(
        state.policy_b.params, state.policy_b.behavior_identity, config,
        source_prefix="B")
    original_a, original_b = state.policy_a, state.policy_b
    calls = []
    real_audit = audit_stage25_ppo_rollout

    def track_audit(policy_params, batch, audit_config):
        calls.append((policy_params is original_a.params,
                      policy_params is original_b.params,
                      batch.behavior_identity.name))
        return real_audit(policy_params, batch, audit_config)

    monkeypatch.setattr(ppo, "audit_stage25_ppo_rollout", track_audit)
    next_state, metrics = ppo_update_dual(
        state, batch_a, batch_b, config, rollout_size=2)
    assert calls[:2] == [
        (True, False, original_a.behavior_identity.name),
        (False, True, original_b.behavior_identity.name),
    ]
    assert next_state.generation == 1
    assert next_state.policy_a.update_counter == next_state.policy_b.update_counter == 1
    assert metrics["rows_A"] == metrics["rows_B"] == 2
    assert not _same_tree(next_state.policy_a.params, params)
    assert not _same_tree(next_state.policy_b.params, params)
    assert not np.array_equal(next_state.policy_a.rng, next_state.policy_b.rng)
    assert state.policy_a is original_a and state.policy_b is original_b
    assert state.generation == 0


def test_failed_second_policy_update_cannot_advance_generation(monkeypatch):
    import rl_manager.stage25_ppo as ppo

    config = Stage25PPOConfig()
    _, state = _fixture(config)
    batch_a = _ppo_batch(
        state.policy_a.params, state.policy_a.behavior_identity, config,
        source_prefix="A")
    batch_b = _ppo_batch(
        state.policy_b.params, state.policy_b.behavior_identity, config,
        source_prefix="B")
    monkeypatch.setattr(ppo, "audit_stage25_ppo_rollout",
                        lambda *_args: {"passed": True})
    calls = 0

    def fail_b(policy, batch, _config, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("synthetic B failure")
        return replace(policy, update_counter=policy.update_counter + 1), {
            "rollout_rows": len(batch.classes), "timing": {}}

    monkeypatch.setattr(ppo, "_ppo_update_after_audit", fail_b)
    with pytest.raises(FloatingPointError, match="synthetic B failure"):
        ppo_update_dual(state, batch_a, batch_b, config, rollout_size=2)
    assert state.generation == 0
    assert state.policy_a.update_counter == state.policy_b.update_counter == 0


def test_dual_checkpoint_round_trip_and_contract_rejection(tmp_path: Path):
    config = Stage25PPOConfig(
        physical_batch_size=1, minibatch_size=1, epochs=1,
        learning_rate=1e-3)
    _, initial_state = _fixture(config)
    state, _ = ppo_update_dual(
        initial_state,
        _ppo_batch(initial_state.policy_a.params,
                   initial_state.policy_a.behavior_identity, config,
                   source_prefix="A"),
        _ppo_batch(initial_state.policy_b.params,
                   initial_state.policy_b.behavior_identity, config,
                   source_prefix="B"),
        config, rollout_size=3)
    path = tmp_path / "dual.npz"
    contract = {"training_composition": DUAL_POLICY_SELF_PLAY,
                "reward": {"mode": "terminal_wlt"}}
    physical = _physical_contract(config)
    executor = {"name": "test", "version": "v1"}
    save_stage25_dual_ppo_checkpoint(
        path, state, config, seed=101, training_contract=contract,
        physical_contract=physical, executor=executor,
        source_identity={"name": "bc.npz", "sha256": "abc"},
        metadata={"source_checkpoint": "bc.npz"})
    loaded, meta = load_stage25_dual_ppo_checkpoint(
        path, ppo_config=config,
        optimizer_state_template_a=state.policy_a.optimizer_state,
        optimizer_state_template_b=state.policy_b.optimizer_state,
        expected_training_contract=contract,
        expected_physical_contract=physical, expected_executor=executor)
    assert meta["payload_kind"] == DUAL_PPO_TRAINING_PAYLOAD_KIND
    assert loaded.generation == state.generation == 1
    assert loaded.rollout_seed == state.rollout_seed
    assert loaded.rollout_progression == state.rollout_progression
    for before, after in ((state.policy_a, loaded.policy_a),
                          (state.policy_b, loaded.policy_b)):
        assert _same_tree(before.params, after.params)
        assert _same_tree(before.optimizer_state, after.optimizer_state)
        np.testing.assert_array_equal(before.rng, after.rng)
        assert before.update_counter == after.update_counter
        assert before.rollout_seed == after.rollout_seed
        assert before.behavior_identity == after.behavior_identity
    with pytest.raises(Stage25CheckpointError, match="training/reward contract"):
        load_stage25_dual_ppo_checkpoint(
            path, ppo_config=config,
            optimizer_state_template_a=state.policy_a.optimizer_state,
            optimizer_state_template_b=state.policy_b.optimizer_state,
            expected_training_contract={"different": True})


def test_single_policy_checkpoint_is_not_a_dual_checkpoint(tmp_path: Path):
    config = Stage25PPOConfig()
    params, state = _fixture(config)
    path = tmp_path / "single.npz"
    save_stage25_ppo_checkpoint(
        path, state.policy_a.params, state.policy_a.optimizer_state,
        state.policy_a.rng, config.model, seed=101,
        update_counter=0, rollout_seed=0,
        ppo_config=config.to_dict(), optimizer_config=config.to_dict(),
        behavior_identity=state.policy_a.behavior_identity,
        physical_contract=_physical_contract(config))
    _arrays, metadata = _read_archive(path)
    assert metadata["payload_kind"] == PPO_TRAINING_PAYLOAD_KIND
    with pytest.raises(Stage25CheckpointError, match="payload kind"):
        load_stage25_dual_ppo_checkpoint(
            path, ppo_config=config,
            optimizer_state_template_a=state.policy_a.optimizer_state,
            optimizer_state_template_b=state.policy_b.optimizer_state)


def test_head_to_head_metrics_follow_policy_identity_through_seat_swap():
    from rl_manager.stage25_dual_ppo_cli import _head_to_head

    config = Stage25PPOConfig()
    _, state = _fixture(config)
    identity_a = state.policy_a.behavior_identity
    identity_b = state.policy_b.behavior_identity
    results = [
        SimpleNamespace(
            policy_identities=(
                {"seat": 0, "policy": identity_a.to_json_dict()},
                {"seat": 1, "policy": identity_b.to_json_dict()}),
            final_banks=[100.0, 50.0], winner_seat=0),
        SimpleNamespace(
            policy_identities=(
                {"seat": 0, "policy": identity_b.to_json_dict()},
                {"seat": 1, "policy": identity_a.to_json_dict()}),
            final_banks=[20.0, 90.0], winner_seat=0),
        SimpleNamespace(
            policy_identities=(
                {"seat": 0, "policy": identity_a.to_json_dict()},
                {"seat": 1, "policy": identity_b.to_json_dict()}),
            final_banks=[40.0, 40.0], winner_seat=-1),
    ]
    metrics = _head_to_head(results, identity_a, identity_b)
    assert metrics["A_wins"] == 1
    assert metrics["B_wins"] == 1
    assert metrics["ties"] == 1
    assert metrics["mean_bank_A"] == pytest.approx((100 + 90 + 40) / 3)
    assert metrics["mean_bank_B"] == pytest.approx((50 + 20 + 40) / 3)


def test_terminal_rewards_follow_policy_identity_and_own_bank_across_seats():
    from rl_manager.reward import RewardConfig, terminal_rewards

    config = Stage25PPOConfig()
    _, state = _fixture(config)
    identity_a = state.policy_a.behavior_identity
    identity_b = state.policy_b.behavior_identity
    a_results = []
    for a_seat in (0, 1):
        seat_ids = ((identity_a, identity_b) if a_seat == 0
                    else (identity_b, identity_a))
        banks = [0.0, 0.0]
        banks[a_seat] = 12000.0
        banks[1 - a_seat] = 4000.0
        rewards = terminal_rewards(banks, RewardConfig())
        a_results.append(rewards[a_seat])
        assert rewards[1 - a_seat] == -1.0
        assert seat_ids[a_seat] == identity_a
    assert a_results == [1.0, 1.0]

    bank_config = RewardConfig(
        mode="terminal_own_bank", bank_baseline=3000.0,
        bank_scale=50000.0)
    a_rewards = []
    for a_seat in (0, 1):
        banks = [0.0, 0.0]
        banks[a_seat], banks[1 - a_seat] = 1000.0, 9000.0
        rewards = terminal_rewards(banks, bank_config)
        a_rewards.append(rewards[a_seat])
    assert a_rewards[0] == a_rewards[1]
    assert a_rewards[0] == pytest.approx(np.tanh(-2000.0 / 50000.0))


def test_behavior_shaping_diagnostics_follow_identity_through_seat_swap():
    from rl_manager.reward import BehaviorShapingConfig, BehaviorShapingFeature
    from rl_manager.stage25_dual_ppo_cli import _shaping_by_policy

    _, state = _fixture(Stage25PPOConfig())
    a_id, b_id = (state.policy_a.behavior_identity,
                  state.policy_b.behavior_identity)
    def detail(reward):
        return {"total": reward, "features": {"goose": {
            "reward": reward, "initial_count": 0, "final_count": 1,
            "target_reached": False}}}

    result = SimpleNamespace(
        policy_identities=(
            {"seat": 0, "policy": b_id.to_json_dict()},
            {"seat": 1, "policy": a_id.to_json_dict()}),
        behavior_shaping={0: detail(0.02), 1: detail(0.08)})
    shaping = BehaviorShapingConfig(goose=BehaviorShapingFeature(5, 0.1))
    metrics = _shaping_by_policy([result], shaping, a_id, b_id)
    assert metrics["A"]["total"] == pytest.approx(0.08)
    assert metrics["B"]["total"] == pytest.approx(0.02)
    assert metrics["A"]["features"]["goose"][
        "mean_episode_contribution"] == pytest.approx(0.08)


def test_central_inference_owner_routes_equal_weight_identities_separately():
    from queue import Queue

    from rl_manager.parallel import ParallelSelfPlayRunner
    from rl_manager.parallel_protocol import Stage25InferenceRequest, Stage25RequestIdentity
    from rl_manager.runner import RunnerConfig
    from rl_manager.stage25_provider import Stage25PlanProvider
    from rl_manager.stage25_types import (
        Stage25PolicyOutputs, stage25_row_token)
    from test_stage25_provider import _obs

    config = Stage25PPOConfig()
    _, state = _fixture(config)
    identities = (state.policy_a.behavior_identity,
                  state.policy_b.behavior_identity)

    class RoutedPolicy:
        supports_precomputed_row_tokens = True
        seed = 19

        def __init__(self, identity):
            self.identity = identity
            self.behavior_identity = identity
            self.calls = []

        def infer_batch(self, *, inputs, crop_capacity, physical_contexts,
                        supports, row_ids, prng_id, row_tokens=None):
            del inputs, physical_contexts, supports, prng_id, row_tokens
            self.calls.append(tuple(row_ids))
            batch = len(row_ids)
            return Stage25PolicyOutputs(
                classes=np.zeros((batch, 9), dtype=np.int16),
                component_logprobs=np.zeros((batch, 9), dtype=np.float32),
                joint_logprob=np.zeros(batch, dtype=np.float32),
                value=np.full(batch, self.seed, dtype=np.float32),
                decoded_goals=np.zeros((batch, 5), dtype=np.int16),
                valid=np.ones(batch, dtype=np.bool_),
                policy_identity=self.identity, batch_size=batch)

    def request(identity, episode, seat, worker):
        provider = Stage25PlanProvider(
            episode, seat, 4, behavior_identity=identity)
        prepared = provider.prepare_inference_context(
            _obs(day=4), behavior_identity=identity)
        row_identity = Stage25RequestIdentity(episode, seat, 4, identity)
        return Stage25InferenceRequest(
            identity=row_identity, worker_id=worker, prng_id="test",
            row_token=stage25_row_token(row_identity.request_id),
            inputs={name: value for name, value in prepared.inputs.items()
                    if name != "crop_capacity"},
            crop_capacity=np.asarray([prepared.crop_capacity], dtype=np.int16),
            physical_context=prepared.physical_context, support=prepared.support,
            queued_at=0.0)

    policies = (RoutedPolicy(identities[0]), RoutedPolicy(identities[1]))
    runner = ParallelSelfPlayRunner(
        RunnerConfig(stage25_enabled=True,
                     stage25_fixed_inference_batch_size=2), num_workers=2)
    responses = [Queue(), Queue()]
    requests_a = [request(identities[0], 2, 0, 0),
                  request(identities[0], 0, 0, 1)]
    requests_b = [request(identities[1], 3, 1, 1),
                  request(identities[1], 1, 1, 0)]
    runner._dispatch_stage25(
        identities[0], requests_a, 0.0,
        {identities[0]: policies[0], identities[1]: policies[1]}, responses)
    runner._dispatch_stage25(
        identities[1], requests_b, 0.0,
        {identities[0]: policies[0], identities[1]: policies[1]}, responses)
    assert len(policies[0].calls) == len(policies[1].calls) == 1
    assert all("stage25_dual_a" in row_id for row_id in policies[0].calls[0])
    assert all("stage25_dual_b" in row_id for row_id in policies[1].calls[0])
    assert policies[0].calls[0] == tuple(sorted(policies[0].calls[0]))
    assert policies[1].calls[0] == tuple(sorted(policies[1].calls[0]))
    returned = [responses[index].get_nowait()
                for index in (0, 1) for _ in range(responses[index].qsize())]
    assert len(returned) == 4
    assert {response.outputs.policy_identity for response in returned} == set(identities)


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_two_identity_spawned_rollout_preserves_ownership_and_row_outputs():
    from rl_manager.parallel import ParallelSelfPlayRunner
    from rl_manager.runner import RunnerConfig

    config = Stage25PPOConfig()
    _, state = _fixture(config)
    policy_a = Stage25InferenceAdapter(
        params=state.policy_a.params, config=config.model,
        name=state.policy_a.behavior_identity.name,
        version=state.policy_a.behavior_identity.version,
        seed=state.policy_a.rollout_seed, mode="stochastic")
    policy_b = Stage25InferenceAdapter(
        params=state.policy_b.params, config=config.model,
        name=state.policy_b.behavior_identity.name,
        version=state.policy_b.behavior_identity.version,
        seed=state.policy_b.rollout_seed, mode="stochastic")
    specs = [build_episode_spec(
        episode, 900 + episode, DUAL_POLICY_SELF_PLAY, policy_a, policy_b)
        for episode in range(4)]

    def collect(workers):
        trajectory = Stage25TrajectoryBuffer(64)
        runner = ParallelSelfPlayRunner(
            RunnerConfig(
                stage25_enabled=True, stage25_mode="stochastic",
                max_turns=144, openings=("none", "none"),
                low_telemetry=True, stage25_fixed_inference_batch_size=2),
            num_workers=workers, inference_batch_wait_seconds=0.01,
            stage25_trajectory_buffer=trajectory)
        runner.run(specs)
        return trajectory.rows, runner

    local_rows, _ = collect(1)
    parallel_rows, parallel = collect(2)
    assert policy_a.call_count > 0 and policy_b.call_count > 0
    def by_key(rows):
        return {(row.episode_id, row.seat, row.day): row for row in rows}

    local, spawned = by_key(local_rows), by_key(parallel_rows)
    assert local.keys() == spawned.keys()
    for key, row in local.items():
        expected = state.policy_a.behavior_identity if (
            key[0] % 2 == key[1]) else state.policy_b.behavior_identity
        assert row.learner_identity == expected
        assert spawned[key].learner_identity == expected
        assert row.row_id == spawned[key].row_id
        np.testing.assert_array_equal(row.classes, spawned[key].classes)
        np.testing.assert_allclose(row.component_logprobs,
                                   spawned[key].component_logprobs,
                                   atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(row.value, spawned[key].value,
                                   atol=1e-6, rtol=1e-6)
    assert parallel.inference_metrics["real_requests"] > 0


def _dual_cli_args(tmp_path: Path):
    from rl_manager.stage25_dual_ppo_cli import _parser

    return _parser().parse_args([
        "--init", str(tmp_path / "bc.npz"),
        "--output-dir", str(tmp_path / "run"),
        "--workers", "1", "--physical-batch-size", "1",
        "--rollout-size", "2", "--updates", "1",
    ])


def test_dual_cli_parser_accepts_linear_own_bank_reward(tmp_path: Path):
    from rl_manager.stage25_dual_ppo_cli import (
        _parser,
        _training_contract,
    )

    args = _parser().parse_args([
        "--init", str(tmp_path / "bc.npz"),
        "--output-dir", str(tmp_path / "run"),
        "--workers", "1", "--physical-batch-size", "1",
        "--rollout-size", "2", "--updates", "1",
        "--reward-mode", "terminal_own_bank_linear",
        "--bank-reward-baseline", "3000",
        "--bank-reward-scale", "100000",
    ])

    assert _training_contract(args)["reward"] == {
        "mode": "terminal_own_bank_linear",
        "bank_baseline": 3000.0,
        "bank_scale": 100000.0,
        "behavior_shaping": {},
    }


def _patch_dual_cli_environment(monkeypatch, tmp_path, *, fail_update=False):
    import rl_manager.runner as runner
    import rl_manager.stage25_ppo_cli as shared_cli
    import rl_manager.stage25_dual_ppo_cli as dual_cli

    args = _dual_cli_args(tmp_path)
    config = shared_cli._config(args)
    _, state = _fixture(config)
    executor = {"name": "test", "version": "v1"}
    monkeypatch.setattr(shared_cli, "_resolve_executor_factory", lambda _name: object())
    monkeypatch.setattr(runner, "_executor_factory_provenance",
                        lambda _factory: executor)
    monkeypatch.setattr(dual_cli, "_new_or_resume_state", lambda *_: (
        state, {"init_params": {"seed": args.seed},
                "source_identity": {"name": "bc.npz", "sha256": "abc"}}))
    a_id, b_id = (state.policy_a.behavior_identity,
                  state.policy_b.behavior_identity)
    results = [SimpleNamespace(
        policy_identities=(
            {"seat": 0, "policy": b_id.to_json_dict()},
            {"seat": 1, "policy": a_id.to_json_dict()}),
        final_banks=[40.0, 90.0], winner_seat=0, behavior_shaping=None)]
    trajectory = SimpleNamespace(diagnostic_summary=lambda: {"terminal_rows": 4})
    batch_a = SimpleNamespace(classes=np.zeros((3, 9), dtype=np.int16))
    batch_b = SimpleNamespace(classes=np.zeros((4, 9), dtype=np.int16))
    monkeypatch.setattr(dual_cli, "_collection", lambda *_a, **_kw: {
        "executor_provenance": executor,
        "opening_provenance": {"opening": "test"},
        "inference_metrics": {}, "rollout_profile": None,
        "timing": {"rollout_seconds": 1.0,
                   "batch_construction_seconds": 0.1,
                   "collection_seconds": 1.1},
        "results": results, "trajectory": trajectory,
        "batch_a": batch_a, "batch_b": batch_b,
    })
    save_calls = []
    monkeypatch.setattr(
        "rl_manager.stage25_checkpoint.save_stage25_dual_ppo_checkpoint",
        lambda *pos, **kw: save_calls.append((pos, kw)))
    if fail_update:
        def fail(*_args, **_kwargs):
            raise FloatingPointError("Policy B update failed")

        monkeypatch.setattr(dual_cli, "ppo_update_dual", fail, raising=False)
        # run imports this primitive from stage25_ppo at call time
        monkeypatch.setattr("rl_manager.stage25_ppo.ppo_update_dual", fail)
    else:
        def update(old_state, _batch_a, _batch_b, _config, *, rollout_size):
            generation = old_state.generation + 1
            progression = {
                "completed_rollouts": generation,
                "next_episode_index": rollout_size,
            }
            next_state = type(old_state)(
                policy_a=replace(old_state.policy_a, update_counter=generation),
                policy_b=replace(old_state.policy_b, update_counter=generation),
                generation=generation,
                rollout_seed=old_state.rollout_seed + rollout_size,
                rollout_progression=progression)
            return next_state, {
                "ppo_A": {"loss": 1.0}, "ppo_B": {"loss": 2.0},
                "audit_A": {"passed": True}, "audit_B": {"passed": True},
                "timing": {"ppo_A_seconds": 0.2, "ppo_B_seconds": 0.3},
            }

        monkeypatch.setattr("rl_manager.stage25_ppo.ppo_update_dual", update)
    return args, state, save_calls


def test_dual_cli_emits_identity_based_metrics_and_checkpoint(monkeypatch, tmp_path):
    import json
    import rl_manager.stage25_dual_ppo_cli as dual_cli

    args, _state, save_calls = _patch_dual_cli_environment(monkeypatch, tmp_path)
    records = dual_cli.run(args)
    record = records[0]
    assert record["generation"] == 1
    assert record["rows_A"] == 3 and record["rows_B"] == 4
    assert record["head_to_head"]["B_wins"] == 1
    assert record["head_to_head"]["mean_bank_A"] == 90.0
    assert record["head_to_head"]["mean_bank_B"] == 40.0
    assert record["initialization"]["A_B_initial_params_equal"] is True
    assert record["initialization"]["A_behavior_identity"] != \
        record["initialization"]["B_behavior_identity"]
    assert len(save_calls) == 1
    output = json.loads((args.output_dir / "metrics.jsonl").read_text())
    assert output["generation"] == 1


def test_dual_cli_does_not_checkpoint_a_failed_second_policy_update(
        monkeypatch, tmp_path):
    import rl_manager.stage25_dual_ppo_cli as dual_cli

    args, _state, save_calls = _patch_dual_cli_environment(
        monkeypatch, tmp_path, fail_update=True)
    with pytest.raises(FloatingPointError, match="Policy B update failed"):
        dual_cli.run(args)
    assert save_calls == []
    assert not (args.output_dir / "latest.npz").exists()
