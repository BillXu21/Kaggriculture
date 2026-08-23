"""Independent stateful official-vs-fast A/B tests.

These tests are deliberately separate from the primary same-action corpus:
each backend owns fresh executor instances and computes its own action from its
own observation before either backend is stepped.
"""

from __future__ import annotations

import copy

import pytest

from oracle import (
    ClosedLoopDivergenceError,
    make_backend,
    make_deterministic_executor_factory,
    run_closed_loop,
)
from oracle.provenance import ProvenanceError, verify_official_provenance


try:
    verify_official_provenance()
    OFFICIAL_AVAILABLE = True
    _SKIP_REASON = ""
except ProvenanceError as error:
    OFFICIAL_AVAILABLE = False
    _SKIP_REASON = str(error)

pytestmark = pytest.mark.skipif(not OFFICIAL_AVAILABLE, reason=_SKIP_REASON)


def _pass_action(obs):
    player = int(obs.get("player", 0))
    hands = len(obs["farms"][player].get("hands") or [])
    return {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}


def _agent_factory(backend_name, seat, configuration):
    del backend_name, seat, configuration

    class Agent:
        def __call__(self, obs):
            return _pass_action(obs)

    return Agent()


def test_closed_loop_builds_four_fresh_agents_and_preserves_terminal_accounting():
    identities = []

    def factory(backend_name, seat, configuration):
        agent = _agent_factory(backend_name, seat, configuration)
        identities.append((backend_name, seat, id(agent)))
        return agent

    result = run_closed_loop(
        {"seed": 7, "episodeSteps": 2},
        max_steps=1,
        agent_factories={"official": factory, "fast": factory},
    )
    assert len(identities) == 4
    assert len({identity for _, _, identity in identities}) == 4
    assert result.steps_executed == 1
    assert result.final_step == result.terminal_step == 1
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.agent_calls == 4
    assert result.observation_comparisons == 4


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_three_full_stateful_executor_episodes_have_closed_loop_parity(seed):
    result = run_closed_loop({"seed": seed}, max_steps=719)
    assert result.steps_executed == 719
    assert result.final_step == result.terminal_step == 719
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.official_rewards == result.fast_rewards
    assert result.observation_comparisons == 1440
    assert result.agent_calls == 719 * 4
    assert result.action_families["farmer.PLANT"] > 0
    assert result.action_families["farmer.WATER"] > 0
    assert result.action_families["market.BUY_SEED"] > 0
    assert result.wall_time_seconds > 0


def test_deliberate_action_drift_is_reported_before_either_step():
    def factory(backend_name, seat, configuration):
        del configuration

        class Agent:
            def __call__(self, obs):
                action = _pass_action(obs)
                if backend_name == "fast" and seat == 0:
                    action["farmer"] = ["NORTH"]
                return action

        return Agent()

    with pytest.raises(ClosedLoopDivergenceError) as excinfo:
        run_closed_loop(
            {"seed": 7}, max_steps=10,
            agent_factories={"official": factory, "fast": factory},
        )
    report = excinfo.value.report
    assert report.phase == "action"
    assert report.turn_index == 0
    assert report.seed == 7
    assert report.step == 0 and report.day == 0 and report.hour == 0
    assert report.seat == 0
    assert report.field_path == "action[0].farmer[0]"
    assert report.official_action["farmer"] == ["PASS"]
    assert report.fast_action["farmer"] == ["NORTH"]


def test_deliberate_reset_observation_drift_is_reported_with_context():
    class DriftBackend:
        def __init__(self, inner):
            self.inner = inner
            self.name = inner.name

        def reset(self):
            observations = copy.deepcopy(self.inner.reset())
            observations[0]["remainingOverageTime"] -= 1
            return observations

        def step(self, actions):
            return self.inner.step(actions)

        def canonical_state(self):
            return self.inner.canonical_state()

        @property
        def rewards(self):
            return self.inner.rewards

        @property
        def statuses(self):
            return self.inner.statuses

        def validate_status_history(self):
            return self.inner.validate_status_history()

    def fast_factory(configuration):
        return DriftBackend(make_backend("fast", configuration))

    with pytest.raises(ClosedLoopDivergenceError) as excinfo:
        run_closed_loop(
            {"seed": 7},
            backend_factories={
                "official": lambda configuration: make_backend(
                    "official", configuration
                ),
                "fast": fast_factory,
            },
            agent_factories={
                "official": _agent_factory,
                "fast": _agent_factory,
            },
        )
    report = excinfo.value.report
    assert report.phase == "reset_observation"
    assert report.turn_index == -1
    assert report.seed == 7
    assert report.seat == 0
    assert report.field_path == "observation[0].remainingOverageTime"
    assert report.official_action is None and report.fast_action is None


def test_default_factory_is_the_existing_deterministic_executor_path():
    factory = make_deterministic_executor_factory()
    official = factory("official", 0, {"seed": 0})
    fast = factory("fast", 0, {"seed": 0})
    assert official is not fast
    assert callable(official) and callable(fast)
