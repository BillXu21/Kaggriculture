"""Offline (no official engine) tests for the deterministic legal-ish action
generator and fast-engine replay repeatability.

These always run: they need only the compiled fast extension, never
``kaggle_environments``.
"""

from __future__ import annotations

from oracle.action_generator import LegalishActionGenerator
from oracle.canonical import canonical_state_fast

from fast_env import FastKaggricultureEnv


def _drive(seed: int, turns: int):
    """Drive one fast episode with the seeded generator; return the trace."""
    env = FastKaggricultureEnv({"seed": seed})
    generator = LegalishActionGenerator(seed)
    observations = env.reset()
    trace = []
    canonical_states = []
    rewards_history = []
    statuses_history = []
    for turn in range(turns):
        pair = generator.next_pair(turn, observations)
        trace.append(pair)
        observations, rewards, statuses = env.step(pair)
        canonical_states.append(canonical_state_fast(observations, rewards, statuses))
        rewards_history.append(list(rewards))
        statuses_history.append(list(statuses))
        if statuses[0] != "ACTIVE":
            break
    return generator, trace, canonical_states, rewards_history, statuses_history


def test_generator_trace_is_deterministic_for_fixed_seed() -> None:
    _, trace_a, states_a, rewards_a, statuses_a = _drive(42, 120)
    _, trace_b, states_b, rewards_b, statuses_b = _drive(42, 120)
    assert trace_a == trace_b
    assert states_a == states_b
    assert rewards_a == rewards_b
    assert statuses_a == statuses_b


def test_different_seeds_produce_different_traces() -> None:
    _, trace_a, *_ = _drive(7, 60)
    _, trace_b, *_ = _drive(999, 60)
    assert trace_a != trace_b


def test_fast_reset_and_replay_reproduces_identical_states() -> None:
    # Build one trace, then replay it verbatim on TWO fresh fast engines;
    # per-turn canonical states, rewards, and statuses must be identical.
    _, trace, states, rewards, statuses = _drive(17, 150)

    for _ in range(2):
        env = FastKaggricultureEnv({"seed": 17})
        observations = env.reset()
        for index, pair in enumerate(trace):
            observations, step_rewards, step_statuses = env.step(pair)
            assert canonical_state_fast(observations, step_rewards, step_statuses) == states[index]
            assert list(step_rewards) == rewards[index]
            assert list(step_statuses) == statuses[index]


def test_generator_tracks_action_family_coverage() -> None:
    generator, *_ = _drive(0, 90)
    coverage = generator.coverage
    assert coverage, "coverage histogram must be recorded"
    # Market and unit families must both be attempted within 90 turns.
    assert any(family.startswith("market.") for family in coverage)
    assert any(family.startswith("unit.") for family in coverage)
