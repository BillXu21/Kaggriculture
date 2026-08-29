"""PPO update tests: manual loss cases, tiny updates, sell-freeze proof."""

import jax
import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.gae import compute_gae
from rl_manager.ppo import (
    PPOBatch,
    build_ppo_batch,
    clipped_surrogate_terms,
    explained_variance,
    init_train_state,
    ppo_update,
)
from rl_manager.ppo_policy import PPOConfig, PPOPolicy


def _encoded(day: int = 4, money: float = 3000.0):
    import bc_manager.live as live

    farm = {"farmer": [0, 0], "hands": [], "hires_today": 0,
            "money": money, "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"]}
    obs = {"day": day, "hour": 0, "step": day * 24, "player": 0,
           "farms": [farm, dict(farm)],
           "market": {"inventory": {}, "prices": {}},
           "town": {"unlocked_shops": []},
           "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}
    return live.encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=day * 24,
        economic_prev_start=(day - 1, money))


@pytest.fixture(scope="module")
def tiny_e():
    config = tiny_manager_config()
    return init_params(config, seed=3, model_variant="E"), config


def test_clipped_surrogate_manual_cases():
    # ratio == 1 -> no clipping, pi_loss == -mean(adv), approx_kl == 0.
    log_ratio = jax.numpy.zeros(4)
    advantages = jax.numpy.asarray([1.0, -1.0, 2.0, -2.0])
    terms = clipped_surrogate_terms(log_ratio, advantages, clip_eps=0.2)
    assert np.allclose(terms["pi_loss"], -float(np.mean(advantages)),
                       atol=1e-6)
    assert np.allclose(terms["approx_kl"], 0.0, atol=1e-7)
    assert np.allclose(terms["clip_fraction"], 0.0, atol=1e-7)
    # Extreme positive ratio with positive advantage clips at 1.2.
    log_ratio = jax.numpy.asarray(np.full(4, np.log(3.0)))
    terms = clipped_surrogate_terms(log_ratio, jax.numpy.ones(4),
                                    clip_eps=0.2)
    assert np.allclose(terms["pi_loss"], -1.2, atol=1e-6)
    assert np.allclose(terms["clip_fraction"], 1.0, atol=1e-7)
    # Negative advantage uses the UNclipped (smaller) surrogate branch:
    # min(3*A, 1.2*A) = 3*A for A<0 -> pi_loss = -3*mean(A).
    terms = clipped_surrogate_terms(log_ratio, -jax.numpy.ones(4),
                                    clip_eps=0.2)
    assert np.allclose(terms["pi_loss"], 3.0, atol=1e-6)


def test_explained_variance_perfect_and_finite_on_constant_returns():
    returns = jax.numpy.asarray([1.0, 2.0, 3.0])
    assert np.allclose(explained_variance(returns, returns), 1.0, atol=1e-6)
    constant_returns = jax.numpy.asarray([2.0, 2.0, 2.0])
    value = jax.numpy.asarray([5.0, 5.0, 5.0])
    # Var(returns)=0 -> epsilon-stable denominator keeps the metric finite.
    assert np.isfinite(float(explained_variance(value, constant_returns)))


def _make_batch(policy, batch_inputs, *, rewards=(1.0, -1.0)):
    sampled = policy.act(batch_inputs, rng=jax.random.PRNGKey(123))
    gae = compute_gae(
        rewards=np.asarray(rewards, dtype=np.float64),
        values=np.asarray(sampled["value"], dtype=np.float64),
        terminated=np.array([1, 1]), truncated=np.zeros(2),
        episode_index=np.arange(2), seat=np.zeros(2, dtype=int),
        day=np.array([4, 4]), gamma=0.99, gae_lambda=0.95,
        normalize=True)
    return PPOBatch(inputs=batch_inputs,
                    action_tensors=sampled["action_tensors"],
                    old_logprob=sampled["logprob_total"],
                    advantages=gae["advantages"], returns=gae["returns"],
                    values=np.asarray(sampled["value"], dtype=np.float32))


def _two_row_inputs():
    rows = [_encoded(day=d) for d in (4, 5)]
    return {k: np.concatenate([r[k] for r in rows], axis=0) for k in rows[0]}


def test_tiny_update_finite_changes_expected_params(tiny_e):
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    ppo_config = PPOConfig(minibatch_size=2, epochs=2, lr=1e-3)
    state = init_train_state(params, config, seed=42, ppo_config=ppo_config)
    new_state, metrics = ppo_update(state, batch, config, ppo_config)

    assert all(np.isfinite(metrics[key]) for key in (
        "loss", "pi_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "kl_to_frozen", "explained_variance"))

    def leaves(tree):
        return [np.asarray(a) for a in jax.tree_util.tree_leaves(tree)]

    # Frozen snapshot untouched; mutable trunk and value head changed;
    # sell-quantity head bit-identical despite nonzero AdamW decay.
    for got, want in zip(leaves(new_state.frozen_params), leaves(params)):
        assert np.array_equal(got, want)
    sq_old = leaves(state.params["base"]["heads"]["sell_quantity"])
    sq_new = leaves(new_state.params["base"]["heads"]["sell_quantity"])
    for got, want in zip(sq_new, sq_old):
        assert np.array_equal(got, want)
    trunk_changed = any(not np.array_equal(a, b) for a, b in zip(
        leaves(new_state.params["base"]["encoder"]),
        leaves(state.params["base"]["encoder"])))
    value_changed = any(not np.array_equal(a, b) for a, b in zip(
        leaves(new_state.params["value"]), leaves(state.params["value"])))
    assert trunk_changed and value_changed
    assert new_state.step == ppo_config.epochs * 1


def test_target_kl_stops_after_first_epoch_and_reports_counts(tiny_e):
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    ppo_config = PPOConfig(minibatch_size=2, epochs=4, lr=1e-3,
                           target_kl=1e-12)
    state = init_train_state(params, config, seed=42, ppo_config=ppo_config)
    new_state, metrics = ppo_update(state, batch, config, ppo_config)

    assert metrics["accepted"] is True
    assert metrics["stop_reason"] == "target_kl"
    assert metrics["epochs_ran"] == 1
    assert metrics["minibatches_ran"] == 1
    assert len(metrics["epoch_metrics"]) == 1
    assert metrics["epoch_metrics"][0]["approx_kl"] > ppo_config.target_kl
    assert new_state.step == 1


def test_pathological_kl_rejection_returns_exact_previous_state(tiny_e):
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    ppo_config = PPOConfig(minibatch_size=2, epochs=2, lr=1e-3,
                           reject_update_kl=1e-12)
    state = init_train_state(params, config, seed=42, ppo_config=ppo_config)
    new_state, metrics = ppo_update(state, batch, config, ppo_config)

    assert metrics["accepted"] is False
    assert metrics["stop_reason"] == "rejected_kl"
    assert metrics["rejection_reason"]
    assert new_state is state
    assert new_state.step == state.step
    assert np.array_equal(np.asarray(new_state.rng), np.asarray(state.rng))
    for got, want in zip(jax.tree_util.tree_leaves(new_state.params),
                         jax.tree_util.tree_leaves(state.params)):
        assert np.array_equal(got, want)


def test_sell_freeze_with_aggressive_weight_decay(tiny_e):
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    ppo_config = PPOConfig(minibatch_size=2, epochs=3, lr=0.1,
                           weight_decay=0.5)
    state = init_train_state(params, config, seed=1, ppo_config=ppo_config)
    new_state, _ = ppo_update(state, batch, config, ppo_config)

    def leaves(tree):
        return [np.asarray(a) for a in jax.tree_util.tree_leaves(tree)]

    for got, want in zip(
            leaves(new_state.params["base"]["heads"]["sell_quantity"]),
            leaves(state.params["base"]["heads"]["sell_quantity"])):
        assert np.array_equal(got, want)  # exact bit equality
    # Sanity: the aggressive optimizer DID move other parameters.
    assert any(not np.array_equal(a, b) for a, b in zip(
        leaves(new_state.params["base"]["encoder"]),
        leaves(state.params["base"]["encoder"])))


def test_metrics_exact_for_identity_ratio_batch(tiny_e):
    """old_logprob == current logprob -> approx_kl ~0, clip fraction 0."""
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    ppo_config = PPOConfig(minibatch_size=2, epochs=1, entropy_coef=0.0,
                           kl_to_frozen_coef=0.0)
    state = init_train_state(params, config, seed=7, ppo_config=ppo_config)
    _, metrics = ppo_update(state, batch, config, ppo_config)
    assert abs(metrics["approx_kl"]) < 1e-5
    assert metrics["clip_fraction"] < 1e-6
    # KL to frozen is exactly 0 before any drift (params still equal).
    assert abs(metrics["kl_to_frozen"]) < 1e-6
    # Advantage normalization over the full batch happened once.
    assert abs(metrics["adv_mean"]) < 1e-6


def test_minibatch_divisibility_enforced(tiny_e):
    params, config = tiny_e
    batch_inputs = _two_row_inputs()
    policy = PPOPolicy(params, config, seed=17)
    batch = _make_batch(policy, batch_inputs)
    state = init_train_state(params, config, seed=7,
                             ppo_config=PPOConfig(minibatch_size=2))
    with pytest.raises(ValueError, match="divisible"):
        ppo_update(state, batch, config, PPOConfig(minibatch_size=8))


def test_trajectory_to_ppo_batch_uses_candidate_rows_only(tiny_e):
    from rl_manager.decode import LOGPROB_GROUPS
    from rl_manager.trajectory import (
        Transition,
        TransitionMetadata,
        TrajectoryBuffer,
        e_input_spec,
    )
    from rl_manager.types import PolicyIdentity

    _, config = tiny_e
    params = init_params(config, seed=11, model_variant="E")
    policy = PPOPolicy(params, config, seed=21)
    batch_inputs = _two_row_inputs()
    sampled = policy.act(batch_inputs, rng=jax.random.PRNGKey(5))
    buffer = TrajectoryBuffer(capacity=4, input_spec=e_input_spec())
    identity = PolicyIdentity("p", "v", "f" * 64)

    def append(row, episode_index, seat, trainable):
        single = {k: v[row:row + 1] for k, v in batch_inputs.items()}
        action = {name: t[row:row + 1]
                  for name, t in sampled["action_tensors"].items()}
        groups = {g: float(sampled["logprob_groups"][g][row])
                  for g in LOGPROB_GROUPS}
        buffer.append(
            Transition(episode_index=episode_index, seed=episode_index,
                       seat=seat, day=4 + row, trainable=trainable,
                       inputs=single, action_tensors=action,
                       logprob_groups=groups,
                       logprob_total=float(sampled["logprob_total"][row]),
                       value=float(sampled["value"][row]),
                       trace_digest=bytes(32)),
            TransitionMetadata(index=0, episode_index=episode_index,
                               seed=episode_index, seat=seat, day=4 + row,
                               policy_id=identity.name,
                               policy_version=identity.version,
                               policy_fingerprint=identity.fingerprint,
                               opponent_id="o", trainable=trainable,
                               plan_json={}))

    append(0, episode_index=0, seat=0, trainable=True)
    append(1, episode_index=1, seat=0, trainable=True)
    append(0, episode_index=9, seat=1, trainable=False)  # opponent row

    arrays = {name: array.copy() for name, array in buffer.finalize().items()}
    arrays["reward"][0], arrays["terminated"][0] = 1.0, 1
    arrays["reward"][1], arrays["terminated"][1] = -1.0, 1
    batch = build_ppo_batch(arrays, gamma=0.99, gae_lambda=0.95)
    assert batch.size == 2  # only candidate/trainable valid rows survive
    assert batch.inputs["board_kind"].flags["C_CONTIGUOUS"]
    assert batch.action_tensors["crop"].flags["C_CONTIGUOUS"]
    assert abs(float(np.mean(batch.advantages))) < 1e-6
