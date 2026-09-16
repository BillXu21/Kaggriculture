from __future__ import annotations

import jax
import numpy as np
import optax
import pytest

from rl_manager.stage25_ppo import (
    Stage25PPOBatch,
    Stage25PPOConfig,
    _compiled_stage25_ppo_step,
    _map_policy_eval,
    _objective_from_output,
    _padded_training_views,
    compute_stage25_gae,
    init_stage25_ppo_state,
    joint_clipped_surrogate,
    make_stage25_ppo_optimizer,
    ppo_update,
)
from rl_manager.stage25_policy import init_stage25_params, stochastic_act
from rl_manager.stage25_trajectory import INPUT_SPEC


def _ppo_fixture(config: Stage25PPOConfig, rows: int = 2):
    inputs = {
        name: np.zeros((rows,) + shape, dtype=dtype)
        for name, (shape, dtype) in INPUT_SPEC.items()
    }
    inputs["unlocked"][:, 0] = 1
    inputs["days_remaining"][:] = 29
    inputs["scalars"][:, 1] = np.arange(rows, dtype=np.float32)
    params = init_stage25_params(config.model, seed=23)
    keys = jax.random.split(jax.random.PRNGKey(5), rows)
    sampled = stochastic_act(
        params, inputs, config.model, keys, row_ids=np.arange(rows),
        reject_invalid=True)
    old_values = np.asarray(sampled["value"], dtype=np.float32)
    batch = Stage25PPOBatch(
        inputs=inputs, classes=np.asarray(sampled["classes"]),
        old_component_logprobs=np.asarray(sampled["component_logprobs"]),
        old_joint_logprobs=np.asarray(sampled["joint_logprob"]),
        old_values=old_values, advantages=np.linspace(
            0.5, 1.5, rows, dtype=np.float32),
        returns=old_values + np.linspace(
            0.75, 1.25, rows, dtype=np.float32),
        episode_id=np.ones(rows, dtype=np.int64),
        seat=np.zeros(rows, dtype=np.int64),
        day=np.arange(4, 4 + rows, dtype=np.int64),
        row_ids=np.arange(rows),
    )
    return params, batch


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


def test_compiled_step_matches_precompiled_objective_and_optimizer():
    config = Stage25PPOConfig(
        physical_batch_size=2, minibatch_size=2, epochs=1,
        learning_rate=1e-3, entropy_coefficient=0.03, weight_decay=0.01)
    params, batch = _ppo_fixture(config)
    optimizer = make_stage25_ppo_optimizer(params, config)
    optimizer_state = optimizer.init(params)

    old_output = _map_policy_eval(params, batch, config)
    old_loss, old_metrics, _ = _objective_from_output(old_output, batch, config)

    def old_objective(tree):
        output = _map_policy_eval(tree, batch, config)
        loss, metrics, _ = _objective_from_output(output, batch, config)
        return loss, metrics

    (reference_loss, reference_metrics), reference_grads = jax.value_and_grad(
        old_objective, has_aux=True)(params)
    reference_updates, reference_opt_state = optimizer.update(
        reference_grads, optimizer_state, params)
    reference_params = optax.apply_updates(params, reference_updates)

    views = _padded_training_views(batch, config)
    (compiled_params, compiled_opt_state, compiled_grads, compiled_loss,
     compiled_metrics, compiled_output) = _compiled_stage25_ppo_step(
         params, optimizer_state, *views[:-1], config, views[-1])

    for name in ("component_logprobs", "joint_logprob", "value",
                 "prefix_entropy_surrogate"):
        np.testing.assert_allclose(
            np.asarray(compiled_output[name])[:2], np.asarray(old_output[
                "prefix_entropy_surrogate" if name == "prefix_entropy_surrogate"
                else name]), atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(np.asarray(compiled_output["classes"])[:2],
                                  np.asarray(old_output["classes"]))
    np.testing.assert_array_equal(np.asarray(compiled_output["valid"])[:2],
                                  np.asarray(old_output["valid"]))
    for name in ("loss", "policy_loss", "value_loss", "entropy_surrogate",
                 "kl", "clip_fraction", "ratio_mean"):
        np.testing.assert_allclose(
            np.asarray(compiled_metrics[name]),
            np.asarray(old_metrics[name]), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(compiled_loss), np.asarray(reference_loss),
                               atol=2e-6, rtol=2e-6)
    for left, right in zip(jax.tree_util.tree_leaves(compiled_grads),
                           jax.tree_util.tree_leaves(reference_grads)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   atol=3e-6, rtol=3e-6)
    for left, right in zip(jax.tree_util.tree_leaves(compiled_params),
                           jax.tree_util.tree_leaves(reference_params)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   atol=3e-6, rtol=3e-6)
    for left, right in zip(jax.tree_util.tree_leaves(compiled_opt_state),
                           jax.tree_util.tree_leaves(reference_opt_state)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   atol=3e-6, rtol=3e-6)


def test_compiled_short_minibatch_excludes_physical_padding():
    padded_config = Stage25PPOConfig(
        physical_batch_size=2, minibatch_size=2, epochs=1, learning_rate=1e-3)
    compact_config = Stage25PPOConfig(
        physical_batch_size=2, minibatch_size=1, epochs=1, learning_rate=1e-3)
    params, padded_batch = _ppo_fixture(padded_config, rows=1)
    _, compact_batch = _ppo_fixture(compact_config, rows=1)
    padded_state = init_stage25_ppo_state(
        padded_config, seed=23, params=params)
    compact_state = init_stage25_ppo_state(
        compact_config, seed=23, params=params)
    padded_next, padded_metrics = ppo_update(
        padded_state, padded_batch, padded_config)
    compact_next, compact_metrics = ppo_update(
        compact_state, compact_batch, compact_config)
    for left, right in zip(jax.tree_util.tree_leaves(padded_next.params),
                           jax.tree_util.tree_leaves(compact_next.params)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   atol=3e-6, rtol=3e-6)
    for left, right in zip(jax.tree_util.tree_leaves(padded_next.optimizer_state),
                           jax.tree_util.tree_leaves(compact_next.optimizer_state)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(padded_metrics["loss"], compact_metrics["loss"],
                               atol=3e-6, rtol=3e-6)


def test_invalid_physical_row_is_rejected_before_state_commit():
    config = Stage25PPOConfig(
        physical_batch_size=2, minibatch_size=2, epochs=1, learning_rate=1e-3)
    params, batch = _ppo_fixture(config)
    invalid_classes = batch.classes.copy()
    invalid_classes[0, -1] = 101
    invalid_batch = Stage25PPOBatch(
        inputs=batch.inputs, classes=invalid_classes,
        old_component_logprobs=batch.old_component_logprobs,
        old_joint_logprobs=batch.old_joint_logprobs,
        old_values=batch.old_values, advantages=batch.advantages,
        returns=batch.returns, episode_id=batch.episode_id, seat=batch.seat,
        day=batch.day, row_ids=batch.row_ids)
    state = init_stage25_ppo_state(config, seed=23, params=params)
    with pytest.raises(ValueError, match="unchanged-weight audit|invalid"):
        ppo_update(state, invalid_batch, config)
    for left, right in zip(jax.tree_util.tree_leaves(state.params),
                           jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    for left, right in zip(jax.tree_util.tree_leaves(state.optimizer_state),
                           jax.tree_util.tree_leaves(
                               make_stage25_ppo_optimizer(params, config).init(params))):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(np.asarray(state.rng), np.asarray(jax.random.PRNGKey(23)))
    assert state.update_counter == 0
