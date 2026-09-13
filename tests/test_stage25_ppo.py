from __future__ import annotations

import jax
import numpy as np

from rl_manager.stage25_ppo import (
    Stage25PPOBatch,
    Stage25PPOConfig,
    compute_stage25_gae,
    init_stage25_ppo_state,
    joint_clipped_surrogate,
    ppo_update,
)
from rl_manager.stage25_policy import init_stage25_params, stochastic_act
from rl_manager.stage25_trajectory import INPUT_SPEC


def test_gae_isolates_seats_and_respects_terminal_and_truncation():
    advantages, returns = compute_stage25_gae(
        episode_id=np.array([1, 1, 1, 1]),
        seat=np.array([0, 0, 1, 1]),
        day=np.array([4, 5, 4, 5]),
        rewards=np.array([0, 2, 0, 3], np.float32),
        values=np.array([1, 1, 10, 10], np.float32),
        terminated=np.array([0, 1, 0, 0]),
        truncated=np.array([0, 0, 0, 1]),
        bootstrap_values=np.array([0, 0, 0, 4], np.float32),
        bootstrap_patched=np.array([0, 0, 0, 1]),
        gamma=0.9, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, [0.755, 1.0, -3.907, -3.4], atol=1e-5)
    np.testing.assert_allclose(returns, [1.755, 2.0, 6.093, 6.6], atol=1e-5)


def test_joint_ratio_is_not_a_headwise_ratio_and_clips_the_joint_term():
    loss, metrics = joint_clipped_surrogate(
        np.array([-1.0, -1.0], np.float32),
        np.array([-1.0 + np.log(2.0), -1.0 - np.log(2.0)], np.float32),
        np.array([1.0, -1.0], np.float32), 0.2)
    np.testing.assert_allclose(metrics["ratio"], [2.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(loss, -0.2, atol=1e-6)


def test_ppo_config_records_fixed_physical_shape_and_discount_convention():
    config = Stage25PPOConfig(physical_batch_size=2, minibatch_size=7, epochs=3)
    assert config.physical_batch_size == 2
    assert config.to_dict()["gamma"] == 0.99
    assert config.to_dict()["model"]["d_model"] == 16


def test_policy_and_value_heads_both_update_on_real_objective():
    config = Stage25PPOConfig(
        physical_batch_size=2, minibatch_size=2, epochs=1,
        learning_rate=1e-3, entropy_coefficient=0.0)
    inputs = {
        name: np.zeros((2,) + shape, dtype=dtype)
        for name, (shape, dtype) in INPUT_SPEC.items()
    }
    inputs["unlocked"][:, 0] = 1
    inputs["days_remaining"][:] = 29
    inputs["scalars"][1, :] = 1.0
    params = init_stage25_params(config.model, seed=23)
    sampled = stochastic_act(
        params, inputs, config.model, jax.random.split(jax.random.PRNGKey(5), 2),
        row_ids=np.arange(2), reject_invalid=True)
    old_values = np.asarray(sampled["value"], dtype=np.float32)
    batch = Stage25PPOBatch(
        inputs=inputs, classes=np.asarray(sampled["classes"]),
        old_component_logprobs=np.asarray(sampled["component_logprobs"]),
        old_joint_logprobs=np.asarray(sampled["joint_logprob"]),
        old_values=old_values, advantages=np.ones(2, dtype=np.float32),
        returns=old_values + 1.0, episode_id=np.array([1, 1]),
        seat=np.array([0, 0]), day=np.array([4, 5]),
        row_ids=np.arange(2),
    )
    state = init_stage25_ppo_state(config, seed=23, params=params)
    updated, metrics = ppo_update(state, batch, config)
    assert np.isfinite(metrics["gradient_norm"])
    policy_delta = max(
        float(np.max(np.abs(np.asarray(after) - np.asarray(before))))
        for before, after in zip(
            jax.tree_util.tree_leaves(params["output_projections"]),
            jax.tree_util.tree_leaves(updated.params["output_projections"])))
    value_delta = max(
        float(np.max(np.abs(np.asarray(after) - np.asarray(before))))
        for before, after in zip(
            jax.tree_util.tree_leaves(params["value_head"]),
            jax.tree_util.tree_leaves(updated.params["value_head"])))
    assert policy_delta > 0.0
    assert value_delta > 0.0
