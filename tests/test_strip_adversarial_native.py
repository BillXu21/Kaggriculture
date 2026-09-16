"""Packet I bounded native-engine legality smoke for the strip executor.

Runs a *fixed manager plan* (no JAX policy) through the real fast engine for a
short prefix of a game, so every emitted primitive/market action is judged by
the actual engine rather than by a second-hand legality model.

This is intentionally not a benchmark and not a full game: it stops well
before the 720-step horizon and asserts only mechanical acceptance,
determinism, and the forbidden-auto-sale invariant.  The whole module skips
cleanly when the native extension is unavailable.
"""

from __future__ import annotations

import pytest

from executor_v0.plan import DailyPlan
from executor_v0.strip_executor import StripExecutorConfig
from rl_manager.executor_factory import Stage25StripExecutorAgent

TURNS = 48
FORBIDDEN_SELL = {"WHEAT", "FERTILIZER"}


def _plan() -> DailyPlan:
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    animals = ("GOOSE", "COW", "SHEEP")
    products = (*crops, "EGG", "MILK", "WOOL", "FERTILIZER")
    return DailyPlan.create(
        crop_targets={crop: (2 if crop == "WHEAT" else 0) for crop in crops},
        animal_targets={animal: 0 for animal in animals},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in crops},
        care_by_animal={animal: 0 for animal in animals},
        sell_quantities={
            product: {anchor: 0 for anchor in (0, 4, 8, 12, 16, 20)}
            for product in products
        },
    )


def _pass_action(obs) -> dict:
    farms = obs.get("farms") or ()
    hands = farms[1].get("hands") if len(farms) > 1 else ()
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (hands or ())],
        "market": [],
    }


class _Provider:
    def __init__(self, plan: DailyPlan) -> None:
        self._plan = plan

    def daily_plan(self, obs, seat, previous_execution=None) -> DailyPlan:
        del obs, seat, previous_execution
        return self._plan


def _require_native():
    pytest.importorskip("fast_env")
    try:
        from fast_env import FastKaggricultureEnv  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"native fast engine unavailable: {exc}")


def _run_prefix(*, seed: int, turns: int):
    from oracle.backend import canonical_observations, make_backend

    backend = make_backend("fast", {"seed": seed, "numThreads": 1})
    observations = backend.reset()
    obs = canonical_observations(
        backend, observations, canonical_state=backend.canonical_state()
    )
    agent = Stage25StripExecutorAgent(
        provider=_Provider(_plan()),
        seat=0,
        strip_config=StripExecutorConfig(aggressive_sell_all=True),
        profile={},
    )

    actions_log: list[dict] = []
    statuses_log: list[str] = []
    for _turn in range(turns):
        action = agent(obs[0])
        actions_log.append(action)
        observations, _rewards, statuses = backend.step(
            [action, _pass_action(obs[1])]
        )
        statuses_log.append(str(statuses[0]))
        obs = canonical_observations(
            backend, observations, canonical_state=backend.canonical_state()
        )
    return actions_log, statuses_log, agent.diagnostics_json()


def test_strip_fixed_plan_actions_are_accepted_by_fast_engine():
    _require_native()
    actions, statuses, diagnostics = _run_prefix(seed=7, turns=TURNS)

    assert set(statuses) <= {"ACTIVE", "INACTIVE", "DONE"}, statuses
    assert "ERROR" not in statuses and "INVALID" not in statuses

    markets = [market for action in actions for market in action["market"]]
    assert all(order[0] != "SELL" or order[1] not in FORBIDDEN_SELL
               for order in markets), markets

    assert diagnostics["days"], "strip executor recorded no day of work"
    executed = sum(
        route["interaction_turns"] for day in diagnostics["days"].values()
        for route in day.get("route_diagnostics", [])
    )
    assert executed > 0, "strip executor completed no interaction in the prefix"


def test_strip_fixed_plan_prefix_is_deterministic():
    _require_native()
    first, first_statuses, _ = _run_prefix(seed=11, turns=TURNS)
    second, second_statuses, _ = _run_prefix(seed=11, turns=TURNS)
    assert first == second
    assert first_statuses == second_statuses
