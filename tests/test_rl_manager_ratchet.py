"""Focused promotion-ratchet tests; no engine games are launched."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np

from bc_manager.live import encode_live_inputs
from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.evaluation import evaluate_promotion, summarize_evaluation
from rl_manager.ppo import PPOBatch, init_train_state, ppo_update
from rl_manager.ppo_adapter import (
    ppo_batched_policy_from_state,
    ppo_snapshot_from_state,
)
from rl_manager.ppo_checkpoint import (
    PPO_SNAPSHOT_FORMAT,
    load_ppo_snapshot,
    save_ppo_snapshot,
)
from rl_manager.ppo_policy import PPOConfig
from rl_manager.policy import JaxEPlanPolicy
from rl_manager.ratchet import PromotionRatchet
from rl_manager.runner import build_episode_spec
from rl_manager.types import CANDIDATE_VS_FROZEN


def _encoded(day: int = 4):
    farm = {"farmer": [0, 0], "hands": [], "hires_today": 0,
            "money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"]}
    obs = {"day": day, "hour": 0, "step": day * 24, "player": 0,
           "farms": [farm, dict(farm)], "market": {"inventory": {},
           "prices": {}}, "town": {"unlocked_shops": []},
           "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}
    return encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=day * 24,
        economic_prev_start=(day - 1, 3000.0))


def _state_and_batch():
    config = tiny_manager_config()
    ppo_config = PPOConfig(minibatch_size=2, epochs=1)
    state = init_train_state(
        init_params(config, seed=3, model_variant="E"), config,
        seed=17, ppo_config=ppo_config)
    policy = ppo_batched_policy_from_state(
        state, config, ppo_config=ppo_config)._policy
    rows = [_encoded(4), _encoded(5)]
    inputs = {key: np.concatenate([row[key] for row in rows], axis=0)
              for key in rows[0]}
    sampled = policy.act(inputs, rng=jax.random.PRNGKey(9))
    return config, ppo_config, state, PPOBatch(
        inputs=inputs, action_tensors=sampled["action_tensors"],
        old_logprob=sampled["logprob_total"],
        advantages=np.asarray([1.0, -1.0], dtype=np.float32),
        returns=np.asarray([1.0, -1.0], dtype=np.float32),
        values=np.asarray(sampled["value"], dtype=np.float32))


def _result(seed=3000, statuses=("DONE", "DONE")):
    return SimpleNamespace(
        seed=seed, composition=CANDIDATE_VS_FROZEN,
        final_banks=[200.0, 100.0], statuses=list(statuses), terminated=True,
        trace_digest="x" * 64, opening_diagnostics=[],
        executor_diagnostics=[])


def test_hold_keeps_opponent_and_fatal_anomaly_blocks_promotion():
    original = JaxEPlanPolicy(
        init_params(tiny_manager_config(), seed=1, model_variant="E"),
        tiny_manager_config(), name="bc_e")
    ratchet = PromotionRatchet(original)
    replacement = JaxEPlanPolicy(
        init_params(tiny_manager_config(), seed=2, model_variant="E"),
        tiny_manager_config(), name="candidate")
    assert not ratchet.apply(False, replacement)
    assert ratchet.current_opponent is original
    summary = summarize_evaluation([_result(statuses=("ACTIVE", "DONE"))],
                                   expected_seeds=[3000])
    decision = evaluate_promotion(summary)
    assert not decision.passed
    assert any("fatal_anomalies" in reason for reason in decision.failed_reasons)
    assert not ratchet.apply(decision.passed, replacement)
    assert ratchet.current_opponent is original


def test_pass_replaces_opponent_and_second_pass_replaces_first():
    config, ppo_config, state, _batch = _state_and_batch()
    original = JaxEPlanPolicy(state.frozen_params, config, name="bc_e")
    first = ppo_snapshot_from_state(
        state, config, ppo_config=ppo_config, name="promotion_001")
    second = ppo_snapshot_from_state(
        state, config, ppo_config=ppo_config, name="promotion_002")
    ratchet = PromotionRatchet(original)
    assert ratchet.apply(True, first)
    assert ratchet.current_opponent is first
    assert ratchet.apply(True, second)
    assert ratchet.current_opponent is second
    assert ratchet.promotions == 2
    spec = build_episode_spec(0, 3000, CANDIDATE_VS_FROZEN,
                              first, ratchet.current_opponent)
    assert spec.policies[1] is second
    assert ratchet.original_opponent is original


def test_promoted_snapshot_is_detached_and_promotion_keeps_ppo_state():
    config, ppo_config, state, batch = _state_and_batch()
    updated, _metrics = ppo_update(state, batch, config, ppo_config)
    snapshot = ppo_snapshot_from_state(
        updated, config, ppo_config=ppo_config, name="promotion_001")
    inputs = batch.inputs
    before = snapshot.plan_batch(inputs, "snapshot").action_tensors
    step = updated.step
    opt_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(
        updated.opt_state)]
    live_params = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf).copy(), updated.params)
    live_params["base"]["heads"]["land"]["bias"] = (
        live_params["base"]["heads"]["land"]["bias"] + 100.0)
    later = updated
    later.params = live_params
    after = snapshot.plan_batch(inputs, "snapshot").action_tensors
    assert all(np.array_equal(before[name], after[name]) for name in before)
    ratchet = PromotionRatchet(JaxEPlanPolicy(
        updated.frozen_params, config, name="bc_e"))
    ratchet.apply(True, snapshot)
    assert updated.step == step
    assert all(np.array_equal(want, got) for want, got in zip(
        opt_leaves, [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(
            updated.opt_state)]))


def test_promoted_snapshot_roundtrips_as_normal_frozen_policy(tmp_path):
    config, ppo_config, state, batch = _state_and_batch()
    snapshot = ppo_snapshot_from_state(
        state, config, ppo_config=ppo_config, name="promotion_001")
    path = save_ppo_snapshot(
        tmp_path / "promotion_001.npz", state, config, ppo_config,
        snapshot_identity=snapshot.identity.to_json_dict())
    loaded, meta = load_ppo_snapshot(path, config=config,
                                     ppo_config=ppo_config)
    assert meta["format"] == PPO_SNAPSHOT_FORMAT
    before = snapshot.plan_batch(batch.inputs, "snapshot").action_tensors
    after = loaded.plan_batch(batch.inputs, "snapshot").action_tensors
    assert all(np.array_equal(before[name], after[name]) for name in before)
