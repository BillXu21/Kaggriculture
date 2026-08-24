"""PPO V0 policy tests (issue #9 B1/B2/B3; packet decisions 3/4/6/7/9)."""

import jax
import numpy as np
import pytest

from bc_manager.model import NUM_PRODUCTS, SELL_BIN_COUNT
from bc_manager_jax.model import (
    forward as jax_forward,
    forward_with_representation,
    init_params,
    manager_representation,
    tiny_manager_config,
)
from rl_manager.ppo_policy import (
    PPO_GROUPS,
    PPOPolicy,
    bernoulli_entropy,
    bernoulli_logprobs,
    categorical_entropy,
    categorical_logprobs,
)
from rl_manager.policy import JaxEPlanPolicy


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
    params = init_params(config, seed=3, model_variant="E")
    return params, config


@pytest.fixture(scope="module")
def batch(tiny_e):
    rows = [_encoded(day=day, money=1000.0 + 100.0 * day)
            for day in (4, 5, 6, 7)]
    return {key: np.concatenate([row[key] for row in rows], axis=0)
            for key in rows[0]}


def test_representation_seam_exact_forward_parity(tiny_e, batch):
    """forward_with_representation outputs equal `forward` exactly."""
    params, config = tiny_e
    outputs_a = jax_forward(params, batch, config, model_variant="E")
    outputs_b, representation = forward_with_representation(
        params, batch, config, model_variant="E")
    assert set(outputs_b) == set(outputs_a)
    for name in outputs_a:
        assert np.array_equal(np.asarray(outputs_a[name]),
                              np.asarray(outputs_b[name])), name
    assert representation.shape == (4, config.d_model)
    assert bool(np.all(np.isfinite(np.asarray(representation))))
    direct = manager_representation(params, batch, config, model_variant="E")
    assert np.array_equal(np.asarray(direct), np.asarray(representation))


def test_deterministic_policy_equals_frozen_jax_e_decode(tiny_e, batch):
    params, config = tiny_e
    policy = PPOPolicy(params, config, seed=101)
    reference = JaxEPlanPolicy(params, config).plan_batch(batch, "prng/ref")
    result = policy.act(batch, deterministic=True)
    for name, expected in reference.action_tensors.items():
        assert np.array_equal(result["action_tensors"][name],
                              np.asarray(expected)), name


def test_value_head_init_does_not_alter_base_tree(tiny_e, batch):
    params, config = tiny_e
    policy = PPOPolicy(params, config, seed=5)
    base_leaves = jax.tree_util.tree_leaves(policy.params["base"])
    frozen_leaves = jax.tree_util.tree_leaves(policy.frozen_params)
    original_leaves = jax.tree_util.tree_leaves(params)
    for got, want in zip(base_leaves, original_leaves):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    for got, want in zip(frozen_leaves, original_leaves):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    result = policy.act(batch, deterministic=True)
    assert result["value"].shape == (4,)
    assert bool(np.all(np.isfinite(result["value"])))


def test_stochastic_shapes_ranges_and_seed_reproducibility(tiny_e, batch):
    _, config = tiny_e
    policy = PPOPolicy(init_params(config, seed=9, model_variant="E"),
                       config, seed=2)
    same_a = policy.act(batch, rng=jax.random.PRNGKey(77))
    same_b = policy.act(batch, rng=jax.random.PRNGKey(77))
    other = policy.act(batch, rng=jax.random.PRNGKey(78))
    tensors = same_a["action_tensors"]
    assert tensors["crop"].shape == (4, 5)
    assert tensors["animal"].shape == (4, 3)
    assert tensors["land"].shape == (4,)
    assert tensors["fertilizer"].shape == (4, 5)
    assert tensors["care"].shape == (4, 3)
    assert tensors["sell_presence"].shape == (4, NUM_PRODUCTS, SELL_BIN_COUNT)
    assert tensors["sell_quantity"].shape == (4, NUM_PRODUCTS, SELL_BIN_COUNT)
    for name in ("crop", "animal", "fertilizer", "care"):
        assert int(tensors[name].min()) >= 0 and int(tensors[name].max()) <= 100
    assert int(tensors["land"].min()) >= 1 and int(tensors["land"].max()) <= 4
    assert set(np.unique(tensors["sell_presence"])) <= {0, 1}
    for name in tensors:
        assert np.array_equal(same_a["action_tensors"][name],
                              same_b["action_tensors"][name]), name
    assert any(not np.array_equal(same_a["action_tensors"][name],
                                  other["action_tensors"][name])
               for name in tensors)


def test_stochastic_requires_rng(tiny_e, batch):
    _, config = tiny_e
    policy = PPOPolicy(tiny_e[0], config, seed=1)
    with pytest.raises(ValueError, match="rng"):
        policy.act(batch, deterministic=False)


def test_logprob_recomputation_exact_from_stored_actions(tiny_e, batch):
    _, config = tiny_e
    policy = PPOPolicy(init_params(config, seed=21, model_variant="E"),
                       config, seed=3)
    sampled = policy.act(batch, rng=jax.random.PRNGKey(4))
    recomputed = policy.evaluate_actions(batch, sampled["action_tensors"])
    assert set(sampled["logprob_groups"]) == set(PPO_GROUPS)
    for group in PPO_GROUPS:
        assert np.array_equal(sampled["logprob_groups"][group],
                              recomputed["logprob_groups"][group]), group
    assert np.array_equal(sampled["logprob_total"],
                          recomputed["logprob_total"])
    total_sum = sum(recomputed["logprob_groups"][g] for g in PPO_GROUPS)
    assert np.allclose(recomputed["logprob_total"], total_sum, rtol=0.0,
                       atol=1e-6)


def test_categorical_and_bernoulli_hand_checks():
    # One row, one 3-way categorical: [B=1, K=1, classes=3].
    logits = np.array([[[0.3, -1.2, 2.0]]], dtype=np.float32)
    logp = np.log(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True))
    got = categorical_logprobs(
        jax.numpy.asarray(logits), jax.numpy.asarray([[1]]))
    assert np.allclose(got, logp[0, 0, 1], atol=1e-6)
    p = np.exp(logp)[0, 0]
    manual_entropy = -float(np.sum(p * logp[0, 0]))
    assert np.allclose(categorical_entropy(jax.numpy.asarray(logits)),
                       manual_entropy, atol=1e-6)

    b_logits = jax.numpy.asarray(np.array([[0.7, -2.0]], dtype=np.float32))
    bits = jax.numpy.asarray(np.array([[1, 0]], dtype=np.float32))
    lsig = lambda x: -np.logaddexp(0.0, -x)  # noqa: E731
    bl = np.asarray(b_logits)
    manual = lsig(bl[0, 0]) + lsig(-bl[0, 1])
    assert np.allclose(bernoulli_logprobs(b_logits, bits), manual, atol=1e-6)
    q = 1.0 / (1.0 + np.exp(-bl[0]))
    manual_b_entropy = -(q * lsig(bl[0]) + (1 - q) * lsig(-bl[0])).sum()
    assert np.allclose(bernoulli_entropy(b_logits), manual_b_entropy,
                       atol=1e-6)


def test_entropy_groups_and_raw_sum(tiny_e, batch):
    _, config = tiny_e
    policy = PPOPolicy(init_params(config, seed=33, model_variant="E"),
                       config, seed=8)
    result = policy.act(batch, rng=jax.random.PRNGKey(9))
    assert set(result["entropy_groups"]) == set(PPO_GROUPS)
    entropy_sum = sum(result["entropy_groups"][g] for g in PPO_GROUPS)
    assert np.allclose(result["entropy_total"], entropy_sum, rtol=0.0,
                       atol=1e-6)
    # Entropy of a uniform 101-way categorical is log(101); with random
    # small-init logits each group entropy is positive and finite.
    for group in ("crop", "fertilizer"):
        assert bool(np.all(result["entropy_groups"][group] > 0.0))


def test_sell_quantities_always_from_frozen_snapshot(tiny_e, batch):
    params, config = tiny_e
    policy = PPOPolicy(params, config, seed=12)
    frozen_outputs = jax_forward(params, batch, config, model_variant="E")
    quantity = np.floor(
        np.expm1(np.clip(np.asarray(frozen_outputs["sell_quantity_log1p"]),
                         0.0, None)) + 0.5)
    for rng_seed in (0, 1, 2):
        result = policy.act(batch, rng=jax.random.PRNGKey(rng_seed))
        presence = result["action_tensors"]["sell_presence"].astype(bool)
        expected = np.where(presence, quantity, 0)
        assert np.array_equal(result["action_tensors"]["sell_quantity"],
                              expected.astype(np.int16))


def test_own_only_contract_rejects_leaked_keys(tiny_e, batch):
    _, config = tiny_e
    policy = PPOPolicy(tiny_e[0], config, seed=1)
    leaked = dict(batch)
    leaked["opp_board_kind"] = np.zeros((4, 100), dtype=np.int16)
    with pytest.raises(ValueError, match="own-only E"):
        policy.act(leaked, deterministic=True)
