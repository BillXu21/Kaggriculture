"""Regression tests for Stage 2.5 per-day diagnostics retention.

The adapter used to deep-copy ``result.diagnostics`` on *every* primitive turn
but only ever retained the last snapshot per manager day, so a 719-turn game
copied 1,142 diagnostics to keep 24. It now stores the live reference and
detaches it once, when the day rolls over.

The invariant that makes that safe: ``StripExecutorController._diagnostics``
returns a fresh top-level dict whose *values* alias mutable controller state,
so a retained entry must not be observed after that state is reused. These
tests pin both halves of that contract:

* a completed day is detached from later controller mutation, and
* the retained mapping is identical to per-turn deep-copying.
"""

from __future__ import annotations

import copy
from typing import Any

from executor_v0.strip_executor import StripExecutorConfig
from rl_manager.executor_factory import Stage25StripExecutorAgent


class _Result:
    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics

    def action_dict(self) -> dict[str, Any]:
        return {"actions": []}


class _MutableController:
    """Stand-in whose diagnostics alias state reused on the next turn.

    This mirrors ``_diagnostics()`` returning ``dict(self._daily)``: the outer
    dict is new each turn, the inner objects are shared and mutated in place.
    """

    def __init__(self) -> None:
        self.shared: dict[str, Any] = {"route": "initial"}
        self.turns = 0

    def act(self, obs, plan):
        del obs, plan
        self.turns += 1
        self.shared["route"] = f"turn-{self.turns}"
        self.shared["turns_seen"] = self.turns
        return _Result({"day_state": self.shared, "turn": self.turns})


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def daily_plan(self, obs, seat, previous_execution=None):
        del obs, seat, previous_execution
        self.calls += 1
        return object()


def _agent() -> tuple[Stage25StripExecutorAgent, _MutableController]:
    agent = Stage25StripExecutorAgent(
        provider=_Provider(),
        seat=0,
        strip_config=StripExecutorConfig(aggressive_sell_all=True),
        profile={},
    )
    controller = _MutableController()
    agent.controller = controller  # type: ignore[assignment]
    return agent, controller


def test_completed_day_is_detached_from_later_mutation() -> None:
    agent, controller = _agent()
    for _turn in range(3):
        agent({"day": 4})
    # day 4 is finished: freeze it by starting day 5.
    agent({"day": 5})
    agent({"day": 5})
    agent({"day": 5})

    day4 = agent.diagnostics_json()["days"]["4"]
    # The controller has since mutated the shared object many more times.
    assert day4["day_state"]["turns_seen"] == 3
    assert day4["day_state"]["route"] == "turn-3"
    assert day4["turn"] == 3
    # and it must not track the live controller any more
    assert controller.turns == 6
    assert day4["day_state"] is not controller.shared


def test_retention_matches_per_turn_deepcopy_reference() -> None:
    """The retained mapping equals the original per-turn-deepcopy behaviour."""
    schedule = [4, 4, 4, 5, 5, 6, 6, 6, 6]

    agent, _ = _agent()
    for day in schedule:
        agent({"day": day})
    observed = agent.diagnostics_json()["days"]

    # Reference implementation: copy on every turn.
    reference: dict[str, Any] = {}
    controller = _MutableController()
    for day in schedule:
        controller.act({"day": day}, None)
        reference[str(day)] = copy.deepcopy(
            {"day_state": controller.shared, "turn": controller.turns})

    assert sorted(observed) == sorted(reference)
    assert observed == reference


def test_final_day_is_detached_by_diagnostics_json() -> None:
    """The last day needs no explicit freeze; the output copy detaches it."""
    agent, controller = _agent()
    agent({"day": 9})
    agent({"day": 9})
    document = agent.diagnostics_json()
    day9 = document["days"]["9"]
    assert day9["turn"] == 2
    assert day9["day_state"]["turns_seen"] == 2
    controller.turns = 99
    controller.shared["route"] = "mutated-after-game"
    # ``document`` was produced by a deep copy, so it is immune.
    assert document["days"]["9"]["day_state"]["route"] == "turn-2"


def test_low_telemetry_retains_nothing() -> None:
    agent = Stage25StripExecutorAgent(
        provider=_Provider(),
        seat=0,
        strip_config=StripExecutorConfig(aggressive_sell_all=True),
        profile={},
        low_telemetry=True,
    )
    agent.controller = _MutableController()  # type: ignore[assignment]
    for day in (4, 4, 5, 5):
        agent({"day": day})
    document = agent.diagnostics_json()
    assert document["days"] == {}
    assert document["telemetry_mode"] == "reduced"
    assert document["diagnostics_reduced"] is True


def test_day_plan_is_requested_once_per_day() -> None:
    agent, _ = _agent()
    for day in (4, 4, 4, 5, 5):
        agent({"day": day})
    assert agent.provider.calls == 2  # type: ignore[attr-defined]
