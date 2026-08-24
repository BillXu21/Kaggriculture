"""RL PPO checkpoint tests: roundtrip + deterministic resume (req. 5)."""

import jax
import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.gae import compute_gae
from rl_manager.ppo import PPOBatch, init_train_state, ppo_update
from rl_manager.ppo_checkpoint import (
    RL_PPO_CHECKPOINT_FORMAT,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
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


def _leaves(tree):
    return [np.asarray(a) for a in jax.tree_util.tree_leaves(tree)]


@pytest.fixture(scope="module")
def env():
    config = tiny_manager_config()
    params = init_params(config, seed=3, model_variant="E")
    rows = [_encoded(day=d) for d in (4, 5)]
    batch_inputs = {k: np.concatenate([r[k] for r in rows], axis=0)
                    for k in rows[0]}
    policy = PPOPolicy(params, config, seed=17)
    sampled = policy.act(batch_inputs, rng=jax.random.PRNGKey(123))
    gae = compute_gae(
        rewards=np.array([1.0, -1.0]), values=sampled["value"],
        terminated=np.array([1, 1]), truncated=np.zeros(2),
        episode_index=np.arange(2), seat=np.zeros(2, dtype=int),
        day=np.array([4, 4]), gamma=0.99, gae_lambda=0.95, normalize=True)
    batch = PPOBatch(inputs=batch_inputs,
                     action_tensors=sampled["action_tensors"],
                     old_logprob=sampled["logprob_total"],
                     advantages=gae["advantages"], returns=gae["returns"],
                     values=np.asarray(sampled["value"], dtype=np.float32))
    ppo_config = PPOConfig(minibatch_size=2, epochs=2)
    state = init_train_state(params, config, seed=42, ppo_config=ppo_config)
    return config, ppo_config, batch, state


def test_roundtrip_full_state_and_provenance(env, tmp_path):
    config, ppo_config, _batch, state = env
    state.rollout_seed = 2026
    provenance = {"bc_checkpoint": "artifacts/local/bc-v1-E/best.pt",
                  "created_by": "test"}
    path = save_ppo_checkpoint(tmp_path / "ck.npz", state, config,
                               ppo_config,
                               provenance=provenance)
    loaded, meta = load_ppo_checkpoint(path)
    assert meta["format"] == RL_PPO_CHECKPOINT_FORMAT
    assert meta["model_variant"] == "E"
    assert meta["step"] == state.step
    assert meta["rollout_seed"] == 2026
    assert meta["provenance"] == provenance
    for got, want in zip(_leaves(loaded.params), _leaves(state.params)):
        assert np.array_equal(got, want)
    for got, want in zip(_leaves(loaded.frozen_params),
                         _leaves(state.frozen_params)):
        assert np.array_equal(got, want)
    for got, want in zip(_leaves(loaded.opt_state),
                         _leaves(state.opt_state)):
        assert np.array_equal(got, want)
    assert np.array_equal(np.asarray(loaded.rng), np.asarray(state.rng))
    assert loaded.step == state.step


def test_resume_next_update_bit_identical(env, tmp_path):
    config, ppo_config, batch, state = env
    after_a, _ = ppo_update(state, batch, config, ppo_config)
    path = save_ppo_checkpoint(tmp_path / "mid.npz", after_a, config,
                               ppo_config)
    loaded, _meta = load_ppo_checkpoint(path)
    original_b, metrics_a = ppo_update(after_a, batch, config, ppo_config)
    resumed_b, metrics_b = ppo_update(loaded, batch, config, ppo_config)
    for got, want in zip(_leaves(resumed_b.params),
                         _leaves(original_b.params)):
        assert np.array_equal(got, want)  # bitwise resume determinism
    assert metrics_a == metrics_b


def test_load_rejects_wrong_format_and_corruption(env, tmp_path):
    config, ppo_config, _batch, _state = env
    bad = tmp_path / "bad.npz"
    np.savez(bad, __meta__=np.frombuffer(
        b'{"format": "something_else"}', dtype=np.uint8))
    with pytest.raises(ValueError, match="unrecognized"):
        load_ppo_checkpoint(bad)

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"not an npz at all")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        load_ppo_checkpoint(empty)


def test_load_rejects_incompatible_configs(env, tmp_path):
    config, ppo_config, _batch, state = env
    path = save_ppo_checkpoint(tmp_path / "ck.npz", state, config,
                               ppo_config)
    other_model = tiny_manager_config(d_model=8)
    with pytest.raises(ValueError, match="model_config"):
        load_ppo_checkpoint(path, config=other_model)
    other_ppo = PPOConfig(minibatch_size=2, lr=9.9)
    with pytest.raises(ValueError, match="ppo_config"):
        load_ppo_checkpoint(path, ppo_config=other_ppo)


def test_save_rejects_non_e_variant(env, tmp_path):
    config, ppo_config, _batch, state = env
    with pytest.raises(ValueError, match="variant E"):
        save_ppo_checkpoint(tmp_path / "v0.npz", state, config, ppo_config,
                            model_variant="V0")
