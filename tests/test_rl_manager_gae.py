"""GAE(λ) tests: hand-computed terminal/tie/interleaved/truncated cases."""

import numpy as np
import pytest

from rl_manager.gae import advantage_stats, compute_gae


def test_terminal_episode_hand_computed():
    rewards = np.array([0.0, 0.0, 1.0])
    values = np.array([0.1, 0.2, 0.3])
    terminated = np.array([0, 0, 1])
    truncated = np.zeros(3)
    episode = np.zeros(3, dtype=int)
    seat = np.zeros(3, dtype=int)
    day = np.array([4, 5, 6])
    out = compute_gae(rewards, values, terminated, truncated, episode, seat,
                      day, gamma=0.99, gae_lambda=0.95, normalize=False)
    adv2 = 1.0 - 0.3
    adv1 = (0.99 * 0.3 - 0.2) + 0.99 * 0.95 * adv2
    adv0 = (0.99 * 0.2 - 0.1) + 0.99 * 0.95 * adv1
    assert np.allclose(out["advantages"], [adv0, adv1, adv2], atol=1e-6)
    assert np.allclose(out["returns"], out["advantages"] + values, atol=1e-6)


def test_tie_reward_zero_hand_computed():
    rewards = np.zeros(2)
    values = np.array([0.5, 0.25])
    terminated = np.array([0, 1])
    truncated = np.zeros(2)
    out = compute_gae(rewards, values, terminated, truncated,
                      np.zeros(2, dtype=int), np.zeros(2, dtype=int),
                      np.arange(2), gamma=0.5, gae_lambda=0.5,
                      normalize=False)
    # delta1 = 0 + 0 - .25 = -.25; delta0 = 0 + .5*.25 - .5 = -.375
    adv1 = -0.25
    adv0 = (0.5 * 0.25 - 0.5) + 0.5 * 0.5 * adv1
    assert np.allclose(out["advantages"], [adv0, adv1], atol=1e-7)


def test_interleaved_seats_grouped_correctly():
    # Rows interleave two episodes: ep0/seat0 at even positions, ep1/seat1
    # at odd positions; each episode ends with its terminal reward.
    rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0])
    values = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.6])
    terminated = np.array([0, 0, 0, 0, 1, 1])
    truncated = np.zeros(6)
    episode = np.array([0, 1, 0, 1, 0, 1])
    seat = np.array([0, 1, 0, 1, 0, 1])
    day = np.array([4, 4, 5, 5, 6, 6])
    out = compute_gae(rewards, values, terminated, truncated, episode, seat,
                      day, gamma=0.99, gae_lambda=0.95, normalize=False)
    solo = compute_gae(rewards[::2], values[::2], terminated[::2],
                       truncated[::2], np.zeros(3, dtype=int),
                       np.zeros(3, dtype=int), day[::2],
                       gamma=0.99, gae_lambda=0.95, normalize=False)
    assert np.allclose(out["advantages"][::2], solo["advantages"], atol=1e-7)
    assert np.allclose(out["returns"], out["advantages"] + values)


def test_truncated_bootstrap_value_used():
    rewards = np.array([0.0, 0.0])
    values = np.array([0.2, 0.4])
    terminated = np.zeros(2)
    truncated = np.array([0, 1])
    bootstrap = np.array([0.0, 2.0])
    out = compute_gae(rewards, values, terminated, truncated,
                      np.zeros(2, dtype=int), np.zeros(2, dtype=int),
                      np.arange(2), gamma=0.5, gae_lambda=0.9,
                      bootstrap_values=bootstrap, normalize=False)
    # last: delta = 0 + .5*2.0 - .4 = .6; first: 0 + .5*.4 - .2 + .45*.6
    assert np.allclose(out["advantages"], [0.5 * 0.4 - 0.2 + 0.45 * 0.6, 0.6],
                       atol=1e-7)


def test_truncated_without_bootstrap_fails_loud():
    with pytest.raises(ValueError, match="bootstrap"):
        compute_gae(np.zeros(2), np.zeros(2), np.zeros(2), np.array([0, 1]),
                    np.zeros(2, dtype=int), np.zeros(2, dtype=int),
                    np.arange(2), gamma=0.9, gae_lambda=0.95)


def test_duplicate_gap_and_incomplete_fail_loud():
    base = dict(gamma=0.9, gae_lambda=0.95)
    with pytest.raises(ValueError, match="duplicate"):
        compute_gae(np.zeros(3), np.zeros(3), np.array([0, 0, 1]),
                    np.zeros(3), np.zeros(3, dtype=int),
                    np.zeros(3, dtype=int), np.array([4, 4, 5]), **base)
    with pytest.raises(ValueError, match="gap"):
        compute_gae(np.zeros(3), np.zeros(3), np.array([0, 0, 1]),
                    np.zeros(3), np.zeros(3, dtype=int),
                    np.zeros(3, dtype=int), np.array([4, 6, 7]), **base)
    with pytest.raises(ValueError, match="incomplete"):
        compute_gae(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),
                    np.zeros(2, dtype=int), np.zeros(2, dtype=int),
                    np.arange(2), **base)
    with pytest.raises(ValueError, match="mid-group"):
        compute_gae(np.zeros(3), np.zeros(3), np.array([0, 1, 0]),
                    np.zeros(3), np.zeros(3, dtype=int),
                    np.zeros(3, dtype=int), np.arange(3), **base)


def test_normalization_zero_mean_unit_variance():
    n = 8
    rng = np.random.default_rng(0)
    advantages_seed = rng.normal(size=n) * 3.0 + 5.0
    rewards = advantages_seed
    values = np.zeros(n)
    terminated = np.zeros(n)
    terminated[-1] = 1
    truncated = np.zeros(n)
    out = compute_gae(rewards, values, terminated, truncated,
                      np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                      np.arange(n), gamma=1.0, gae_lambda=1.0,
                      normalize=True)
    assert abs(float(np.mean(out["advantages"]))) < 1e-6
    assert abs(float(np.std(out["advantages"])) - 1.0) < 1e-3


def test_advantage_stats():
    stats = advantage_stats(np.array([1.0, 3.0]))
    assert stats == {"adv_mean": 2.0, "adv_std": 1.0, "adv_min": 1.0,
                     "adv_max": 3.0}
