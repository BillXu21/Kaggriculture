"""Focused checks for the Stage 2.5 PPO CLI metadata boundary."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rl_manager import stage25_checkpoint as checkpoint
from rl_manager import stage25_ppo as ppo
from rl_manager import stage25_ppo_cli as cli
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.runner import _executor_factory_provenance
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_trajectory import INPUT_SPEC, Stage25TrajectoryRow
from rl_manager.types import CANDIDATE_VS_FROZEN, CURRENT_VS_CURRENT_ECONOMIC


def test_ppo_cli_records_the_real_executor_identity(monkeypatch, tmp_path) -> None:
    config = tiny_stage25_config()
    learner = Stage25InferenceAdapter(
        params=init_stage25_params(config, seed=0), config=config,
        name="stage25_learner", version="ppo-native-v1", mode="stochastic")

    captured: dict = {}

    def fake_collection(state, config, *, seed, args, previous_params=None):
        del state, config, seed, args, previous_params
        return None, learner, {}

    def fake_build_batch(trajectory, *, learner_identity, **kwargs):
        del trajectory, learner_identity, kwargs
        return None

    def fake_update(state, batch, config):
        del batch, config
        return state, {}

    def fake_save(path, *args, **kwargs):
        del args
        captured.update(kwargs)
        return path

    monkeypatch.setattr(cli, "_collection", fake_collection)
    monkeypatch.setattr(ppo, "build_stage25_ppo_batch", fake_build_batch)
    monkeypatch.setattr(ppo, "ppo_update", fake_update)
    monkeypatch.setattr(checkpoint, "save_stage25_ppo_checkpoint", fake_save)

    args = cli._parser().parse_args([
        "--scratch", "--model-size", "tiny", "--engine", "fast",
        "--workers", "1", "--rollout-size", "1", "--physical-batch-size", "1",
        "--minibatch-size", "1", "--epochs", "1", "--updates", "1", "--seed", "17",
        "--output-dir", str(tmp_path)])
    cli.run(args)

    assert captured["executor"] == _executor_factory_provenance(
        make_stage25_executor_factory())
    assert captured["metadata"]["training_contract"] == {
        "training_composition": CANDIDATE_VS_FROZEN,
        "reward": {
            "mode": "terminal_wlt",
            "bank_baseline": 3000.0,
            "bank_scale": 50000.0,
        },
    }


def test_current_current_cli_contract_requires_own_bank_reward() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--training-composition", CURRENT_VS_CURRENT_ECONOMIC,
        "--output-dir", "out"])
    with pytest.raises(ValueError, match="requires --reward-mode"):
        cli._reward_config(args)


def test_collection_uses_both_seats_and_does_not_mutate_snapshot(monkeypatch):
    config = cli._config(cli._parser().parse_args([
        "--scratch", "--model-size", "tiny", "--physical-batch-size", "1",
        "--output-dir", "out"]))
    params = init_stage25_params(config.model, seed=23)
    state = ppo.init_stage25_ppo_state(config, seed=23, params=params)
    before = [np.array(value, copy=True)
              for value in jax.tree_util.tree_leaves(state.params)]
    before_optimizer = [np.array(value, copy=True)
                        for value in jax.tree_util.tree_leaves(
                            state.optimizer_state)]
    captured_specs = []

    class FakeRunner:
        def __init__(self, runner_config, *, stage25_trajectory_buffer,
                     executor_factory, **kwargs):
            del kwargs
            self.buffer = stage25_trajectory_buffer
            self.provenance = {"executor_factory": executor_factory}
            self.inference_metrics = {}

        def run(self, specs):
            captured_specs.extend(specs)
            executor = _executor_factory_provenance(
                self.provenance["executor_factory"])
            for spec in specs:
                for seat, policy in enumerate(spec.policies):
                    for day in (4, 5):
                        inputs = {
                            name: np.zeros(shape, dtype=dtype)
                            for name, (shape, dtype) in INPUT_SPEC.items()
                        }
                        self.buffer.append(Stage25TrajectoryRow(
                            episode_id=spec.episode_index, seat=seat, day=day,
                            inputs=inputs, classes=np.zeros(9, dtype=np.int16),
                            component_logprobs=np.zeros(9, dtype=np.float32),
                            joint_logprob=np.asarray(0.0, dtype=np.float32),
                            value=np.asarray(0.0, dtype=np.float32),
                            learner_identity=spec.policies[0].identity,
                            opponent_identity=policy.identity,
                            provenance={"executor": executor},
                            trainable=seat in spec.trainable_seats,
                            predecessor_closed=day == 5))
                    self.buffer.patch_terminal(
                        len(self.buffer) - 1, float(seat + 1), True)
            return [SimpleNamespace(final_banks=[3100.0, 3200.0])
                    for _ in specs]

    monkeypatch.setattr(cli, "ParallelSelfPlayRunner", FakeRunner)
    args = cli._parser().parse_args([
        "--scratch", "--model-size", "tiny", "--physical-batch-size", "1",
        "--training-composition", CURRENT_VS_CURRENT_ECONOMIC,
        "--reward-mode", "terminal_own_bank", "--rollout-size", "1",
        "--output-dir", "out"])
    trajectory, learner, stats = cli._collection(
        state, config, seed=23, args=args)

    spec = captured_specs[0]
    assert spec.policies == (learner, learner)
    assert spec.policies[0].identity == spec.policies[1].identity
    assert spec.trainable_seats == (0, 1)
    assert len(trajectory) == 4
    assert stats["learner_rows"] == 4
    assert stats["training_composition"] == CURRENT_VS_CURRENT_ECONOMIC
    assert stats["reward"]["mode"] == "terminal_own_bank"
    assert [float(row.reward) for row in trajectory.rows] == [0.0, 1.0,
                                                               0.0, 2.0]
    assert all(np.array_equal(before_value, after_value)
               for before_value, after_value in zip(
                   before, jax.tree_util.tree_leaves(state.params)))
    assert all(np.array_equal(before_value, after_value)
               for before_value, after_value in zip(
                   before_optimizer,
                   jax.tree_util.tree_leaves(state.optimizer_state)))


def test_candidate_collection_only_batches_learner_seat(monkeypatch):
    config = cli._config(cli._parser().parse_args([
        "--scratch", "--model-size", "tiny", "--physical-batch-size", "1",
        "--output-dir", "out"]))
    state = ppo.init_stage25_ppo_state(
        config, seed=23, params=init_stage25_params(config.model, seed=23))

    class FakeRunner:
        def __init__(self, runner_config, *, stage25_trajectory_buffer,
                     executor_factory, **kwargs):
            del runner_config, kwargs
            self.buffer = stage25_trajectory_buffer
            self.provenance = {"executor_factory": executor_factory}
            self.inference_metrics = {}

        def run(self, specs):
            executor = _executor_factory_provenance(
                self.provenance["executor_factory"])
            for spec in specs:
                for seat, policy in enumerate(spec.policies):
                    inputs = {
                        name: np.zeros(shape, dtype=dtype)
                        for name, (shape, dtype) in INPUT_SPEC.items()
                    }
                    index = self.buffer.append(Stage25TrajectoryRow(
                        episode_id=spec.episode_index, seat=seat, day=4,
                        inputs=inputs, classes=np.zeros(9, dtype=np.int16),
                        component_logprobs=np.zeros(9, dtype=np.float32),
                        joint_logprob=np.asarray(0.0, dtype=np.float32),
                        value=np.asarray(0.0, dtype=np.float32),
                        learner_identity=spec.policies[0].identity,
                        opponent_identity=policy.identity,
                        provenance={"executor": executor},
                        trainable=seat in spec.trainable_seats))
                    self.buffer.patch_terminal(index, float(seat + 1), True)
            return [SimpleNamespace(final_banks=[3100.0, 3200.0])]

    monkeypatch.setattr(cli, "ParallelSelfPlayRunner", FakeRunner)
    args = cli._parser().parse_args([
        "--scratch", "--model-size", "tiny", "--physical-batch-size", "1",
        "--rollout-size", "1", "--output-dir", "out"])
    trajectory, _, stats = cli._collection(state, config, seed=23, args=args)
    assert len(trajectory) == 2
    assert stats["learner_rows"] == 1
    assert [row.trainable for row in trajectory.rows] == [True, False]
