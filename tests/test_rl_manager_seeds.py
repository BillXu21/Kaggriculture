"""Deterministic seed-stream tests (issue #9 Stage A).

Episode seeds must be pure functions of (master_seed, episode_index) so
parallel workers stay reproducible regardless of scheduling/interleaving.
"""

import pytest

from rl_manager.seeds import SeedStream


def test_episode_seeds_deterministic_per_master_seed():
    assert [SeedStream(7).episode_seed(i) for i in range(8)] == \
        [SeedStream(7).episode_seed(i) for i in range(8)]


def test_episode_seeds_differ_across_master_seeds():
    assert SeedStream(1).episode_seed(0) != SeedStream(2).episode_seed(0)


def test_schedule_independence_prefix_and_random_order():
    """Drawing fewer seeds first, or in any order, never changes a value."""
    full = [SeedStream(42).episode_seed(i) for i in range(16)]
    short = [SeedStream(42).episode_seed(i) for i in range(3)]
    assert short == full[:3]
    order = [9, 0, 15, 4, 4]
    assert [SeedStream(42).episode_seed(i) for i in order] == \
        [full[i] for i in order]


def test_episode_ownership_distinct_indices_distinct_seeds():
    seeds = {SeedStream(123).episode_seed(i) for i in range(64)}
    assert len(seeds) == 64


def test_policy_and_environment_streams_are_separate_namespaces():
    stream = SeedStream(5)
    assert stream.policy_seed("candidate") != stream.episode_seed(0)
    assert stream.environment_seed(0) != stream.episode_seed(0)


def test_invalid_master_seed_and_index_fail_loud():
    with pytest.raises(ValueError):
        SeedStream(-1)
    with pytest.raises(ValueError):
        SeedStream(True)  # type: ignore[arg-type]
    stream = SeedStream(1)
    with pytest.raises(ValueError):
        stream.episode_seed(-1)
