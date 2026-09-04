"""Focused Phase-A current-v-current economic self-play tests."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.cli import build_parser, plan_training
from rl_manager.land import farm_utilization_snapshot, observed_land_purchase_events
from rl_manager.ppo import build_ppo_batch, init_train_state, ppo_update
from rl_manager.ppo_policy import PPOConfig
from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.reward import (RewardConfig, TERMINAL_OWN_BANK, TERMINAL_WLT,
                               terminal_rewards)
from rl_manager.runner import GAME_TURNS, RunnerConfig, SelfPlayRunner, build_episode_spec
from rl_manager.trajectory import TrajectoryBuffer, e_input_spec
from rl_manager.types import CURRENT_VS_CURRENT_ECONOMIC, PolicyIdentity, PolicyOutputs
from rl_manager.decode import ACTION_TENSOR_SHAPES


class _ConstantPlanPolicy:
    def __init__(self, fingerprint: str = "live-fingerprint") -> None:
        self.identity = PolicyIdentity("live", "v0", fingerprint)

    def plan_batch(self, inputs, prng_id):
        del prng_id
        batch = int(np.asarray(inputs["day"]).shape[0])
        actions = {name: np.zeros((batch,) + shape, dtype=np.int16)
                   for name, shape in ACTION_TENSOR_SHAPES.items()}
        actions["land"] = np.ones(batch, dtype=np.int16)
        zeros = np.zeros(batch, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=actions,
            logprob_groups={name: zeros.copy() for name in (
                "crop", "animal", "land", "fertilizer", "care",
                "sell_presence")},
            logprob_total=zeros.copy(), value=zeros.copy(), batch_size=batch)


def _config(**overrides) -> RunnerConfig:
    values = {
        "backend_name": "fast",
        "backend_configuration": {"seed": 0, "numThreads": 1},
        "max_turns": GAME_TURNS,
    }
    values.update(overrides)
    return RunnerConfig(**values)


def _farm(*, unlocked=("NW",), money=3000.0):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {"money": money, "tiles": tiles,
            "unlocked_quadrants": list(unlocked)}


def test_cli_current_current_plan_records_reward_and_52_row_budget(tmp_path: Path):
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"placeholder")
    args = build_parser().parse_args([
        "train", "--e-checkpoint", str(checkpoint),
        "--executor-factory", "executor_v0@stage-a-v1", "--master-seed", "17",
        "--training-composition", CURRENT_VS_CURRENT_ECONOMIC,
        "--reward-mode", TERMINAL_OWN_BANK, "--episodes-per-update", "384",
        "--minibatch-size", "256", "--output-dir", "out",
        "--checkpoint", "out/final.npz"])
    plan = plan_training(args)
    assert plan["training_composition"] == CURRENT_VS_CURRENT_ECONOMIC
    assert plan["reward"] == {
        "mode": TERMINAL_OWN_BANK, "bank_baseline": 3000.0,
        "bank_scale": 50000.0}
    assert plan["episodes_per_update"] * 52 == 19968


def test_current_current_resolves_same_live_policy_and_both_trainable():
    live = _ConstantPlanPolicy()
    frozen = _ConstantPlanPolicy("bc-fingerprint")
    spec = build_episode_spec(0, 17, CURRENT_VS_CURRENT_ECONOMIC, live, frozen)
    assert spec.policies == (live, live)
    assert spec.trainable_seats == (0, 1)


def test_bank_rewards_are_independent_and_wlt_is_rejected_for_current_current():
    config = RewardConfig(mode=TERMINAL_OWN_BANK)
    rewards = terminal_rewards([3000.0, 53000.0], config)
    assert rewards == [0.0, np.tanh(1.0)]
    assert terminal_rewards([53000.0, 3000.0], config) == [np.tanh(1.0), 0.0]
    assert terminal_rewards([53000.0, 3000.0], RewardConfig(mode=TERMINAL_WLT)) == [
        1.0, -1.0]
    live = _ConstantPlanPolicy()
    spec = build_episode_spec(0, 17, CURRENT_VS_CURRENT_ECONOMIC, live, live)
    with pytest.raises(ValueError, match="requires terminal_own_bank"):
        SelfPlayRunner(_config()).run([spec])


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_complete_current_current_game_has_52_rows_and_deterministic_ids(
        tmp_path: Path):
    live = _ConstantPlanPolicy()
    rows = []
    for _ in range(2):
        buffer = TrajectoryBuffer(52, e_input_spec())
        runner = SelfPlayRunner(
            _config(reward_config=RewardConfig(mode=TERMINAL_OWN_BANK)),
            trajectory_buffer=buffer)
        spec = build_episode_spec(0, 17, CURRENT_VS_CURRENT_ECONOMIC, live, live)
        result = runner.run([spec])[0]
        assert result.transitions == 52
        assert len(buffer) == 52
        arrays = buffer.finalize()
        rows.append([(int(ep), int(seat), int(day)) for ep, seat, day in zip(
            arrays["episode_index"], arrays["seat"], arrays["day"])])
        assert [int(arrays["seat"].tolist().count(seat)) for seat in (0, 1)] == [
            26, 26]
        assert len(set(rows[-1])) == 52
        assert result.policy_identities[0]["trainable"]
        assert result.policy_identities[1]["trainable"]
        assert len(result.utilization_snapshots) == 54
        final_rows = {int(arrays["seat"][index]): float(arrays["reward"][index])
                      for index in range(len(buffer)) if int(arrays["day"][index]) == 29}
        assert np.allclose(
            [final_rows[seat] for seat in (0, 1)],
            terminal_rewards(result.final_banks,
                             RewardConfig(mode=TERMINAL_OWN_BANK)))
        batch = build_ppo_batch(
            arrays, gamma=0.99, gae_lambda=0.95,
            sidecar_records=buffer.sidecar_records)
        assert batch.size == 52
        assert batch.learner_fingerprint == live.identity.fingerprint
    assert rows[0] == rows[1]
    runner.save_trajectory_artifact(tmp_path / "current", buffer, result)
    metadata = json.loads(
        (tmp_path / "current.json").read_text(encoding="utf-8"))["run_metadata"]
    assert metadata["reward_config"] == {
        "mode": TERMINAL_OWN_BANK, "bank_baseline": 3000.0,
        "bank_scale": 50000.0}
    assert len(metadata["episode"]["utilization_snapshots"]) == 54


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_current_current_bank_only_batch_updates_without_nan():
    live = _ConstantPlanPolicy()
    buffer = TrajectoryBuffer(52, e_input_spec())
    runner = SelfPlayRunner(
        _config(reward_config=RewardConfig(mode=TERMINAL_OWN_BANK)),
        trajectory_buffer=buffer)
    spec = build_episode_spec(0, 17, CURRENT_VS_CURRENT_ECONOMIC, live, live)
    runner.run([spec])
    config = tiny_manager_config()
    ppo_config = PPOConfig(minibatch_size=52, epochs=1)
    state = init_train_state(init_params(config, seed=3, model_variant="E"),
                             config, seed=4, ppo_config=ppo_config)
    batch = build_ppo_batch(
        buffer.finalize(), gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        sidecar_records=buffer.sidecar_records)
    new_state, metrics = ppo_update(state, batch, config, ppo_config)
    assert new_state.step == 1
    assert metrics["accepted"] is True
    assert all(np.isfinite(float(value)) for key, value in metrics.items()
               if key not in ("epoch_metrics", "rejection_reason", "stop_reason"))


def test_land_event_requires_observed_unlock_and_uses_causal_time():
    before = _farm()
    submitted_only = _farm()
    assert not observed_land_purchase_events(
        before, submitted_only, episode=1, seat=0, day=7, hour=0)
    after = _farm(unlocked=("NW", "NE"))
    events = observed_land_purchase_events(
        before, after, episode=1, seat=0, day=7, hour=13)
    assert events == [{
        "episode": 1, "seat": 0, "quadrant": "NE",
        "submitted_day": 7, "submitted_hour": 13,
        "causal_day": 7, "causal_hour": 13,
    }]


def test_utilization_counts_exclude_locked_and_empty_structures():
    farm = _farm(unlocked=("NW", "NE"))
    farm["tiles"][0][0] = {"kind": "PLANT"}
    farm["tiles"][0][1] = {"kind": "COOP"}
    farm["tiles"][0][2] = {"kind": "PASTURE", "animal": "COW"}
    farm["tiles"][0][3] = "LOCKED"
    snapshot = farm_utilization_snapshot(farm, day=4, episode=2, seat=1)
    assert snapshot["unlocked_squares"] == 99
    assert snapshot["crop_squares"] == 1
    assert snapshot["animal_squares"] == 1
    assert snapshot["productive_squares"] == 2
    assert snapshot["productive_occupancy"] == 2 / 99


def test_ppo_builder_rejects_mixed_learner_fingerprints():
    live = _ConstantPlanPolicy()
    buffer = TrajectoryBuffer(4, e_input_spec())
    runner = SelfPlayRunner(
        _config(max_turns=130, reward_config=RewardConfig(mode=TERMINAL_OWN_BANK)),
        trajectory_buffer=buffer)
    spec = build_episode_spec(0, 17, CURRENT_VS_CURRENT_ECONOMIC, live, live)
    runner.run([spec])
    buffer.sidecar_records[1].policy_fingerprint = "other"
    with pytest.raises(ValueError, match="multiple learner fingerprints"):
        build_ppo_batch(buffer.finalize(), gamma=0.99, gae_lambda=0.95,
                        sidecar_records=buffer.sidecar_records)


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_parallel_current_current_merge_has_each_row_once():
    live = _ConstantPlanPolicy()
    buffer = TrajectoryBuffer(16, e_input_spec())
    config = _config(max_turns=130, low_telemetry=True,
                     read_only_agent_observations=True,
                     reward_config=RewardConfig(mode=TERMINAL_OWN_BANK))
    specs = [build_episode_spec(index, 17 + index,
                                CURRENT_VS_CURRENT_ECONOMIC, live, live)
             for index in range(2)]
    results = ParallelSelfPlayRunner(
        config, num_workers=2, trajectory_buffer=buffer).run(specs)
    assert [result.episode_index for result in results] == [0, 1]
    arrays = buffer.finalize()
    keys = [(int(episode), int(seat), int(day)) for episode, seat, day in zip(
        arrays["episode_index"][:len(buffer)], arrays["seat"][:len(buffer)],
        arrays["day"][:len(buffer)])]
    assert len(keys) == 8
    assert len(set(keys)) == 8
    assert sorted(keys) == sorted(
        (episode, seat, day) for episode in (0, 1)
        for seat in (0, 1) for day in (4, 5))
    assert all(len(result.manager_crop_rows) == 4 for result in results)
    assert all(
        row["achieved_final_crops"] is not None
        for result in results for row in result.manager_crop_rows)
