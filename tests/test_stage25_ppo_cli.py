"""Focused checks for the Stage 2.5 PPO CLI metadata boundary."""

from __future__ import annotations

import pytest

from rl_manager import stage25_checkpoint as checkpoint
from rl_manager import stage25_ppo as ppo
from rl_manager import stage25_ppo_cli as cli
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.runner import _executor_factory_provenance
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config


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
