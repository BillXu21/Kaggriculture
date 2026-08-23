"""Offline oracle tests: canonicalization, deep diff, corruption interceptor.

These run WITHOUT the official engine installed; they exercise the tiny
comparator/interceptor seams directly so first-divergence behavior is proven
independently of a real official runtime.
"""

from __future__ import annotations

import copy

import pytest

from oracle.canonical import MISSING, canonical_state_fast, deep_diff
from oracle.official_backend import OfficialAnomalyError, status_anomalies
from oracle.replay import DivergenceError, run_same_action_replay


def pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def fast_observation(money: float = 3000.0) -> dict[str, object]:
    shed = {name: 0 for name in (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
    )}
    seeds = {name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    return {
        "player": 0,
        "farms": [
            {
                "money": money,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": money,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {"shed": dict(shed), "seeds": dict(seeds), "inventories": [dict(shed)]},
        "market": {
            "inventory": {name: 10000 for name in shed},
            "prices": {name: 1 for name in shed},
        },
        "town": {"unlocked_shops": ["BAKERY", "BAKERY"]},
        "day": 0,
        "hour": 0,
        "step": 0,
        "remainingOverageTime": 60,
    }


class StubBackend:
    """Minimal deterministic backend advancing through canned canonical states."""

    name = "stub"

    def __init__(self, states: list[dict]) -> None:
        self._states = copy.deepcopy(states)
        self._step = 0
        self.rewards = [0.0, 0.0]
        self.statuses = ["ACTIVE", "ACTIVE"]
        self.last_actions: tuple | None = None

    def reset(self) -> list[dict]:
        self._step = 0
        return []

    def observations(self) -> list[dict]:
        return []

    def step(self, actions) -> tuple:
        self.last_actions = (copy.deepcopy(actions[0]), copy.deepcopy(actions[1]))
        self._step = min(self._step + 1, len(self._states) - 1)
        return [], list(self.rewards), list(self.statuses)

    def canonical_state(self) -> dict:
        return self._states[self._step]

    def validate_status_history(self) -> None:
        anomalies = status_anomalies([self.statuses])
        if anomalies:
            raise OfficialAnomalyError(str(anomalies))


def base_canonical(step: int, money: float = 3000.0) -> dict:
    return {
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [
            {"money": money, "marker": f"m{step}"},
            {"money": money, "marker": f"m{step}"},
        ],
        "privates": [{"shed": {}}, {"shed": {}}],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "rewards": [0.0, 0.0],
        "statuses": ["ACTIVE", "ACTIVE"],
    }


def test_deep_diff_reports_exact_field_paths_and_order() -> None:
    left = {"a": {"b": 1, "list": [1, 2, 3]}, "town": {"shops": ["BAKERY"]}}
    right = {"a": {"b": 2, "list": [1, 5]}, "town": {"shops": ["BAKERY", "BAKERY"]}}
    diffs = deep_diff(left, right)
    paths = [diff.path for diff in diffs]
    assert paths == [
        "state.a.b",
        "state.a.list (len 3 vs 2)",
        "state.a.list[1]",
        "state.town.shops (len 1 vs 2)",
    ]
    assert diffs[0].official_value == 1 and diffs[0].fast_value == 2


def test_deep_diff_flags_missing_keys_and_bool_int_confusion() -> None:
    diffs = deep_diff({"x": True}, {"x": 1})
    assert len(diffs) == 1 and diffs[0].path == "state.x"
    diffs = deep_diff({"x": 1}, {})
    assert diffs[0].fast_value is MISSING


def test_canonical_fast_preserves_shop_multiplicity_and_zero_fills() -> None:
    obs = fast_observation()
    obs["town"]["unlocked_shops"] = ["BAKERY", "BAKERY", "PET_CAFE"]
    obs["private"]["shed"]["WHEAT"] = 3
    obs["private"]["inventories"] = [{"EGG": 2}]
    canonical = canonical_state_fast([obs, obs], [0.0, 0.0], ["ACTIVE", "ACTIVE"])
    assert canonical["town"]["unlocked_shops"] == ["BAKERY", "BAKERY", "PET_CAFE"]
    assert canonical["privates"][0]["shed"]["WHEAT"] == 3
    assert canonical["privates"][0]["inventories"][0]["EGG"] == 2
    assert canonical["privates"][0]["inventories"][0]["MILK"] == 0


def test_status_history_anomaly_invalidates_despite_terminal_done() -> None:
    history = [
        ["ACTIVE", "ACTIVE"],
        ["ERROR", "ACTIVE"],
        ["DONE", "DONE"],  # terminal DONE masks the earlier ERROR seat 0
    ]
    assert status_anomalies(history) == [(1, 0, "ERROR")]
    clean = [["ACTIVE", "ACTIVE"], ["DONE", "DONE"]]
    assert status_anomalies(clean) == []


def test_corruption_interceptor_stops_at_exact_turn_and_path() -> None:
    # Four-step deterministic canned "runtime"; the interceptor corrupts
    # exactly one fast canonical field at turn 2.
    states = [base_canonical(step) for step in range(4)]
    official = StubBackend(states)
    fast = StubBackend(states)

    def corrupt(canonical: dict, turn: int) -> None:
        if turn == 2:
            canonical["farms"][0]["money"] += 1.0

    original_canonical = fast.canonical_state

    def mutated_canonical() -> dict:
        state = original_canonical()
        corrupt(state, max(fast._step - 1, 0))
        return state

    fast.canonical_state = mutated_canonical  # type: ignore[method-assign]

    trace = [[pass_action(), pass_action()] for _ in range(3)]
    with pytest.raises(DivergenceError) as excinfo:
        run_same_action_replay(
            {"seed": 11}, trace, official_backend=official, fast_backend=fast
        )
    report = excinfo.value.report
    assert report.turn_index == 2
    assert report.step == 3 and report.day == 0 and report.hour == 3
    assert report.field_path == "state.farms[0].money"
    assert report.official_value == 3000.0 and report.fast_value == 3001.0
    assert report.seed == 11
    assert report.p0_action == pass_action() and report.p1_action == pass_action()
    payload = report.to_dict()
    assert payload["field_path"] == "state.farms[0].money"


def test_clean_stub_replay_runs_to_completion() -> None:
    states = [base_canonical(step) for step in range(4)]
    official = StubBackend(states)
    fast = StubBackend(states)
    trace = [[pass_action(), pass_action()] for _ in range(3)]
    result = run_same_action_replay(
        {"seed": 5}, trace, official_backend=official, fast_backend=fast
    )
    assert result.turns_executed == 3
