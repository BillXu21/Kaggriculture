"""Focused checks for the Stage 2.5 PPO CLI metadata boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import jax
import numpy as np
import pytest

from rl_manager import stage25_checkpoint as checkpoint
from rl_manager import stage25_ppo as ppo
from rl_manager import stage25_ppo_cli as cli
from rl_manager.executor_factory import (
    make_default_executor_factory,
    make_stage25_executor_factory,
)
from rl_manager.reward import RewardConfig
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


def test_rollout_control_defaults_preserve_runner_defaults() -> None:
    args = cli._parser().parse_args([
        "--scratch", "--output-dir", "out"])
    assert args.checkpoint_every == 1
    assert args.executor == "strip"
    runner_config = cli._runner_config(
        args, seed=17, reward_config=RewardConfig())

    assert args.workers == 1
    assert args.envs_per_worker == 1
    assert args.batch_backend is False
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


def test_executor_modes_have_distinct_canonical_provenance() -> None:
    legacy_args = cli._parser().parse_args([
        "--scratch", "--executor", "legacy", "--output-dir", "out"])
    strip_args = cli._parser().parse_args([
        "--scratch", "--executor", "strip", "--output-dir", "out"])
    legacy = _executor_factory_provenance(
        cli._resolve_executor_factory(legacy_args.executor))
    strip = _executor_factory_provenance(
        cli._resolve_executor_factory(strip_args.executor))

    assert legacy != strip
    assert legacy["effective_profile"]["agent_config"]["foreman"][
        "shed_access_tiles"] == [[4, 4], [5, 4], [4, 5], [5, 5]]
    for provenance in (legacy, strip):
        assert json.loads(json.dumps(provenance, allow_nan=False)) == provenance


def test_executor_provenance_normalizes_scalar_subclasses() -> None:
    class StringSubclass(str):
        pass

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    profile = MappingProxyType({
        StringSubclass("values"): MappingProxyType({
            StringSubclass("text"): StringSubclass("ok"),
            StringSubclass("integer"): IntSubclass(4),
            StringSubclass("float"): FloatSubclass(2.5),
        }),
    })
    provenance = _executor_factory_provenance(SimpleNamespace(
        name="subclasses", version="v1", effective_profile=profile))
    values = provenance["effective_profile"]["values"]

    assert values == {"text": "ok", "integer": 4, "float": 2.5}
    assert all(type(key) is str for key in values)
    assert type(values["text"]) is str
    assert type(values["integer"]) is int
    assert type(values["float"]) is float
    assert json.loads(json.dumps(provenance, allow_nan=False)) == provenance


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), {1: "bad"}, object()])
def test_executor_provenance_rejects_non_json_values(bad) -> None:
    factory = SimpleNamespace(
        name="invalid", version="v1", effective_profile={"bad": bad})
    with pytest.raises((TypeError, ValueError), match="executor provenance"):
        _executor_factory_provenance(factory)


def _write_resume_checkpoint(tmp_path, executor: str):
    args = cli._parser().parse_args([
        "--resume", str(tmp_path / f"{executor}-source.npz"),
        "--executor", executor, "--model-size", "tiny",
        "--physical-batch-size", "1", "--minibatch-size", "1",
        "--epochs", "1", "--seed", "7", "--output-dir", str(tmp_path)])
    config = cli._config(args)
    fresh = ppo.init_stage25_ppo_state(config, seed=7)
    frozen = init_stage25_params(config.model, seed=8)
    learner = Stage25InferenceAdapter(
        params=fresh.params, config=config.model, name="stage25_learner",
        version="ppo-native-v1", seed=7, mode="stochastic")
    opponent = Stage25InferenceAdapter(
        params=frozen, config=config.model, name="stage25_opponent",
        version="frozen-v1", seed=8, mode="stochastic")
    provenance = _executor_factory_provenance(
        cli._resolve_executor_factory(executor))
    checkpoint.save_stage25_ppo_checkpoint(
        args.resume, fresh.params, fresh.optimizer_state, fresh.rng,
        config.model, seed=7, update_counter=70, rollout_seed=281,
        rollout_progression={"completed_rollouts": 70},
        ppo_config=config.to_dict(), optimizer_config=config.to_dict(),
        curriculum=config.model.curriculum,
        behavior_identity=learner.identity, opponent_params=frozen,
        opponent_identity=opponent.identity,
        physical_contract=cli._physical_contract(config),
        executor=provenance,
        metadata={"training_contract": cli._training_contract(args)})
    return args, config, fresh, frozen, learner, opponent, provenance


def _same_tree(left, right) -> bool:
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree_util.tree_leaves(left),
                                jax.tree_util.tree_leaves(right)))


def test_legacy_checkpoint_round_trip_canonicalizes_tuple_provenance(
        tmp_path) -> None:
    args, config, fresh, frozen, learner, opponent, provenance = (
        _write_resume_checkpoint(tmp_path, "legacy"))
    with np.load(args.resume, allow_pickle=False) as archive:
        stored = json.loads(
            archive["__meta__"].tobytes().decode("utf-8"))["executor"]

    assert stored == provenance
    assert stored["effective_profile"]["agent_config"]["foreman"][
        "shed_access_tiles"] == [[4, 4], [5, 4], [4, 5], [5, 5]]
    state, meta = cli._new_state(args, config)
    assert _same_tree(state.params, fresh.params)
    assert _same_tree(state.optimizer_state, fresh.optimizer_state)
    assert _same_tree(state.rng, fresh.rng)
    assert _same_tree(state.opponent_params, frozen)
    assert state.rollout_seed == 281
    assert state.rollout_progression == {"completed_rollouts": 70}
    assert state.update_counter == 70
    assert state.behavior_identity == learner.identity
    assert state.opponent_identity == opponent.identity
    assert meta["executor"] == provenance


def test_resume_rejects_changed_executor_setting(tmp_path, monkeypatch) -> None:
    args, config, *_ = _write_resume_checkpoint(tmp_path, "legacy")
    original = cli._resolve_executor_factory("legacy")
    changed = make_default_executor_factory(
        replace(original.agent_config, tasks_per_worker=11))
    monkeypatch.setattr(cli, "_resolve_executor_factory", lambda _: changed)

    with pytest.raises(ValueError, match="executor provenance does not match"):
        cli._new_state(args, config)


@pytest.mark.parametrize(("saved_as", "resume_as"), [
    ("legacy", "strip"), ("strip", "legacy")])
def test_checkpoint_cross_resume_between_executor_modes_is_rejected(
        tmp_path, saved_as, resume_as) -> None:
    saved_args, _, *_ = _write_resume_checkpoint(tmp_path, saved_as)
    resume_args = cli._parser().parse_args([
        "--resume", str(saved_args.resume), "--executor", resume_as,
        "--model-size", "tiny", "--physical-batch-size", "1",
        "--minibatch-size", "1", "--epochs", "1", "--seed", "7",
        "--output-dir", str(tmp_path)])
    config = cli._config(resume_args)

    with pytest.raises(ValueError, match="executor provenance does not match"):
        cli._new_state(resume_args, config)


@pytest.mark.parametrize(
    ("start", "updates", "cadence", "expected_saves"), [
        (0, 3, 1, [1, 2, 3]),
        (0, 25, 10, [10, 20, 25]),
        (70, 25, 10, [80, 90, 95]),
        (0, 3, 10, [3]),
    ])
def test_checkpoint_cadence_uses_absolute_counter_and_final_boundary(
        monkeypatch, tmp_path, start, updates, cadence, expected_saves) -> None:
    source = (["--resume", str(tmp_path / "source.npz")] if start
              else ["--scratch"])
    args = cli._parser().parse_args([
        *source, "--model-size", "tiny", "--physical-batch-size", "1",
        "--minibatch-size", "1", "--updates", str(updates),
        "--checkpoint-every", str(cadence), "--output-dir", str(tmp_path)])
    config = cli._config(args)
    base_state = ppo.init_stage25_ppo_state(config, seed=7)
    learner = Stage25InferenceAdapter(
        params=base_state.params, config=config.model,
        name="stage25_learner", version="ppo-native-v1", mode="stochastic")
    saves = []

    def fake_state(*_args):
        return replace(base_state, update_counter=start,
                       rollout_seed=100 + start), {}

    def fake_collection(*_args, **_kwargs):
        return None, learner, {}

    def fake_batch(*_args, **_kwargs):
        return None

    def fake_update(state, _batch, _config):
        next_update = state.update_counter + 1
        return replace(
            state, update_counter=next_update,
            rollout_progression={"completed_rollouts": next_update}), {}

    def fake_save(path, *_args, **kwargs):
        saves.append(kwargs["update_counter"])
        return path

    monkeypatch.setattr(cli, "_new_state", fake_state)
    monkeypatch.setattr(cli, "_collection", fake_collection)
    monkeypatch.setattr(ppo, "build_stage25_ppo_batch", fake_batch)
    monkeypatch.setattr(ppo, "ppo_update", fake_update)
    monkeypatch.setattr(checkpoint, "save_stage25_ppo_checkpoint", fake_save)

    records = cli.run(args)

    assert saves == expected_saves
    assert [row["update"] for row in records] == list(
        range(start + 1, start + updates + 1))
    assert [row["update"] for row in records if row["checkpoint_saved"]] == (
        expected_saves)
    assert all(row["timing"]["checkpoint_seconds"] == 0.0
               for row in records if not row["checkpoint_saved"])
    assert records[-1]["checkpoint"] == str(tmp_path / "latest.npz")
    if start:
        assert records[0]["checkpoint"] == str(args.resume)
    elif cadence > 1:
        assert records[0]["checkpoint"] is None


@pytest.mark.parametrize("value", [0, -1])
def test_checkpoint_cadence_rejects_nonpositive_values(tmp_path, value) -> None:
    args = cli._parser().parse_args([
        "--scratch", "--output-dir", str(tmp_path),
        "--checkpoint-every", str(value)])
    with pytest.raises(ValueError, match="checkpoint-every must be positive"):
        cli.run(args)


def _patch_failure_run(monkeypatch, tmp_path, *, fail_rollout: bool):
    args = cli._parser().parse_args([
        "--scratch", "--physical-batch-size", "1", "--minibatch-size", "1",
        "--checkpoint-every", "10", "--output-dir", str(tmp_path)])
    config = cli._config(args)
    state = ppo.init_stage25_ppo_state(config, seed=7)
    learner = Stage25InferenceAdapter(
        params=state.params, config=config.model, name="stage25_learner",
        version="ppo-native-v1", mode="stochastic")
    monkeypatch.setattr(cli, "_new_state", lambda *_args: (state, {}))
    if fail_rollout:
        def fail_collection(*_args, **_kwargs):
            raise RuntimeError("rollout failed")

        monkeypatch.setattr(cli, "_collection", fail_collection)
    else:
        monkeypatch.setattr(
            cli, "_collection", lambda *_args, **_kwargs: (None, learner, {}))

        def fail_update(*_args):
            raise RuntimeError("PPO update failed")

        monkeypatch.setattr(ppo, "build_stage25_ppo_batch", lambda *_a, **_k: None)
        monkeypatch.setattr(ppo, "ppo_update", fail_update)

    def fail_save(*_args, **_kwargs):
        raise AssertionError("failed rollout or update must not be checkpointed")

    monkeypatch.setattr(checkpoint, "save_stage25_ppo_checkpoint", fail_save)
    return args


def test_failed_rollout_never_saves_final_checkpoint(monkeypatch, tmp_path) -> None:
    args = _patch_failure_run(monkeypatch, tmp_path, fail_rollout=True)
    with pytest.raises(RuntimeError, match="rollout failed"):
        cli.run(args)


def test_failed_ppo_update_never_saves_final_checkpoint(monkeypatch, tmp_path) -> None:
    args = _patch_failure_run(monkeypatch, tmp_path, fail_rollout=False)
    with pytest.raises(RuntimeError, match="PPO update failed"):
        cli.run(args)
