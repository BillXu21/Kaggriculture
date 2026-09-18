"""Focused checks for the Stage 2.5 PPO CLI metadata boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rl_manager import stage25_checkpoint as checkpoint
from rl_manager import stage25_ppo as ppo
from rl_manager import stage25_ppo_cli as cli
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.reward import RewardConfig
from rl_manager.runner import _executor_factory_provenance
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_trajectory import INPUT_SPEC, Stage25TrajectoryRow
from rl_manager.types import CANDIDATE_VS_FROZEN, CURRENT_VS_CURRENT_ECONOMIC


def test_bank_statistics_use_deterministic_bottom_decile() -> None:
    stats = cli._bank_statistics([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    assert stats["count"] == 10
    assert stats["mean"] == pytest.approx(45.0)
    assert stats["median"] == pytest.approx(45.0)
    assert stats["bottom_decile_mean"] == pytest.approx(0.0)
    assert stats["p10"] == pytest.approx(9.0)
    assert stats["p25"] == pytest.approx(22.5)
    assert stats["p75"] == pytest.approx(67.5)
    assert stats["p90"] == pytest.approx(81.0)
    assert stats["zero_bank_fraction"] == pytest.approx(0.1)


def test_inference_summary_reports_distribution_and_aggregate_labels() -> None:
    summary = cli._inference_summary({
        "physical_inference_calls": 4,
        "real_requests": 24,
        "logical_requests": 24,
        "real_batch_sizes": [2, 4, 8, 10],
        "physical_rows": 32,
        "padding_rows": 8,
        "occupancy": 0.75,
        "inference_seconds": 1.25,
        "queue_wait_seconds": 2.5,
    })

    assert summary["physical_calls"] == 4
    assert summary["physical_rows"] == 32
    assert summary["padding_fraction"] == pytest.approx(0.25)
    assert summary["occupancy"] == pytest.approx(0.75)
    assert summary["mean_real_batch_size"] == pytest.approx(6.0)
    assert summary["min_real_batch_size"] == pytest.approx(2.0)
    assert summary["median_real_batch_size"] == pytest.approx(6.0)
    assert summary["p10_real_batch_size"] == pytest.approx(2.6)
    assert summary["p90_real_batch_size"] == pytest.approx(9.4)
    assert summary["max_real_batch_size"] == pytest.approx(10.0)
    assert summary["aggregate_inference_seconds"] == pytest.approx(1.25)
    assert summary["aggregate_queue_wait_seconds"] == pytest.approx(2.5)


def test_throughput_rate_is_safe_and_derived_from_wall_seconds() -> None:
    assert cli._rate(1024, 40.0) == pytest.approx(25.6)
    assert cli._rate(1024 * 3600.0, 40.0) == pytest.approx(92160.0)
    assert cli._rate(1024, 0.0) == 0.0


@pytest.mark.parametrize("json_stdout", [False, True])
def test_ppo_cli_records_the_real_executor_identity(
        monkeypatch, tmp_path, capsys, json_stdout) -> None:
    config = tiny_stage25_config()
    learner = Stage25InferenceAdapter(
        params=init_stage25_params(config, seed=0), config=config,
        name="stage25_learner", version="ppo-native-v1", mode="stochastic")

    captured: dict = {}
    sentinel_batch = object()

    def fake_collection(state, config, *, seed, args, previous_params=None):
        del state, config, seed, args, previous_params
        return None, learner, sentinel_batch, {
            "final_banks": [0.0, 100.0],
            "learner_rows": 3,
            "inference_metrics": {
                "batch_sizes": [2, 4, 8],
                "real_batch_sizes": [2, 4, 8],
                "physical_batch_sizes": [4, 4, 8],
                "physical_rows": 16,
                "padding_rows": 2,
                "physical_inference_calls": 3,
                "real_requests": 14,
                "logical_requests": 14,
                "inference_seconds": 0.5,
                "queue_wait_seconds": 0.25,
            },
        }

    def fake_build_batch(trajectory, *, learner_identity, **kwargs):
        del trajectory, learner_identity, kwargs
        raise AssertionError("run() rebuilt the PPO batch after collection")

    def fake_update(state, batch, config):
        assert batch is sentinel_batch
        del config
        return state, {}

    def fake_save(path, *args, **kwargs):
        del args
        captured.update(kwargs)
        return path

    monkeypatch.setattr(cli, "_collection", fake_collection)
    monkeypatch.setattr(ppo, "build_stage25_ppo_batch", fake_build_batch)
    monkeypatch.setattr(ppo, "ppo_update", fake_update)
    monkeypatch.setattr(checkpoint, "save_stage25_ppo_checkpoint", fake_save)

    argv = [
        "--scratch", "--model-size", "tiny", "--engine", "fast",
        "--workers", "1", "--rollout-size", "1", "--physical-batch-size", "1",
        "--minibatch-size", "1", "--epochs", "1", "--updates", "1", "--seed", "17",
        "--output-dir", str(tmp_path)]
    if json_stdout:
        argv.append("--json-stdout")
    args = cli._parser().parse_args(argv)
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
    assert captured["metadata"]["cli_args"]["opening"] == "standard_mixed"
    assert captured["metadata"]["cli_args"]["scratch_hold_prior_tau"] is None
    stdout = capsys.readouterr().out
    if json_stdout:
        assert json.loads(stdout)["inference_metrics"]["batch_sizes"] == [2, 4, 8]
    else:
        assert "batch_sizes" not in stdout
    records = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    machine_record = json.loads(records[0])
    assert machine_record["inference_metrics"]["batch_sizes"] == [2, 4, 8]
    assert machine_record["final_banks"] == [0.0, 100.0]
    assert machine_record["timing"]["ppo_batch_construction_seconds"] == 0.0
    assert machine_record["checkpoint"] == str(tmp_path / "latest.npz")
    assert all(np.isfinite(value) and value >= 0.0
               for value in machine_record["timing"].values())
    assert machine_record["throughput"]["update_games_per_second"] > 0.0


def test_current_current_cli_contract_requires_own_bank_reward() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--training-composition", CURRENT_VS_CURRENT_ECONOMIC,
        "--output-dir", "out"])
    with pytest.raises(ValueError, match="requires --reward-mode"):
        cli._reward_config(args)


def test_rollout_control_defaults_preserve_runner_defaults() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--output-dir", "out"])
    runner_config = cli._runner_config(
        args, seed=17, reward_config=RewardConfig())

    assert args.workers == 1
    assert args.envs_per_worker == 1
    assert args.batch_backend is False
    assert args.stage25_inference_validation == "strict"
    assert args.inference_batch_wait_ms == 20.0
    assert runner_config.num_envs == 1
    assert runner_config.batch_backend is False
    assert runner_config.inference_batch_wait_seconds == 0.02
    assert runner_config.stage25_fixed_inference_batch_size == 2
    assert runner_config.backend_configuration["numThreads"] == 1


def test_rollout_controls_reach_runner_config() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--engine", "fast", "--envs-per-worker", "2",
        "--batch-backend", "--inference-batch-wait-ms", "7.5",
        "--physical-batch-size", "16", "--output-dir", "out"])
    runner_config = cli._runner_config(
        args, seed=23, reward_config=RewardConfig())

    assert runner_config.num_envs == 2
    assert runner_config.batch_backend is True
    assert runner_config.inference_batch_wait_seconds == pytest.approx(0.0075)
    assert runner_config.stage25_fixed_inference_batch_size == 16


def test_opening_reaches_runner_config() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--opening", "tetsuya_s1", "--output-dir", "out"])
    runner_config = cli._runner_config(
        args, seed=23, reward_config=RewardConfig())
    assert runner_config.opening == "tetsuya_s1"


def test_hold_prior_requires_scratch() -> None:
    args = cli._parser().parse_args([
        "--init", "model.npz", "--scratch-hold-prior-tau", "1.0",
        "--output-dir", "out"])
    with pytest.raises(ValueError, match="requires --scratch"):
        cli._runner_config(args, seed=23, reward_config=RewardConfig())


def test_batch_backend_rejects_official_engine() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--engine", "official", "--batch-backend",
        "--output-dir", "out"])
    with pytest.raises(ValueError, match="requires --engine fast"):
        cli._runner_config(args, seed=0, reward_config=RewardConfig())


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
    trajectory, learner, _, stats = cli._collection(
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
    trajectory, _, _, stats = cli._collection(state, config, seed=23, args=args)
    assert len(trajectory) == 2
    assert stats["learner_rows"] == 1
    assert [row.trainable for row in trajectory.rows] == [True, False]
