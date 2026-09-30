"""Regression tests for the canonical controller observation copy.

`_controller_observation` used to deep-copy the whole raw observation and then
throw the copied ``farms`` away, replacing it with the canonical farms.  The
optimized path must produce an observation that is equal to the original while
still fully isolating the controller from engine-owned state.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agent_match import _controller_observation  # noqa: E402


class _Canonical:
    observation_mode = "canonical"


class _Raw:
    observation_mode = "raw"


class _Other:
    observation_mode = "weird"


def _raw() -> dict:
    return {
        "day": 6, "hour": 4, "player": 0,
        "farms": [{"farmer": [0, 0], "hands": [], "money": 10.0}],
        "market": {"prices": {"WHEAT": 3}, "inventory": {}},
        "nested": {"a": [1, 2, {"b": 3}]},
    }


def _canonical() -> list[dict]:
    return [
        {"farmer": [1, 1], "hands": [[2, 2]], "money": 20.0, "tiles": ["WEED"]},
        {"farmer": [3, 3], "hands": [], "money": 30.0, "tiles": []},
    ]


def test_canonical_observation_matches_the_original_construction():
    raw = _raw()
    canonical = _canonical()
    expected = copy.deepcopy(dict(raw))
    expected["farms"] = copy.deepcopy(list(canonical))
    expected.setdefault("step", int(expected["day"]) * 24 + int(expected.get("hour", 0)))

    actual = _controller_observation(raw, _Canonical(), canonical)
    assert actual == expected


def test_canonical_observation_does_not_alias_canonical_farms():
    canonical = _canonical()
    observation = _controller_observation(_raw(), _Canonical(), canonical)
    observation["farms"][0]["money"] = 999.0
    observation["farms"][0]["hands"].append([9, 9])
    assert canonical[0]["money"] == 20.0
    assert canonical[0]["hands"] == [[2, 2]]


def test_canonical_observation_does_not_alias_raw_observation():
    raw = _raw()
    observation = _controller_observation(raw, _Canonical(), _canonical())
    observation["nested"]["a"][2]["b"] = 99
    assert raw["nested"]["a"][2]["b"] == 3


def test_raw_mode_is_unchanged_and_isolated():
    raw = _raw()
    observation = _controller_observation(raw, _Raw(), None)
    assert observation == raw
    observation["nested"]["a"][2]["b"] = 99
    assert raw["nested"]["a"][2]["b"] == 3


def test_step_is_derived_when_absent():
    raw = _raw()
    observation = _controller_observation(raw, _Canonical(), _canonical())
    assert observation["step"] == 6 * 24 + 4


def test_step_is_not_overwritten_when_present():
    raw = dict(_raw(), step=99)
    observation = _controller_observation(raw, _Canonical(), _canonical())
    assert observation["step"] == 99


def test_canonical_requires_canonical_farms():
    with pytest.raises(RuntimeError):
        _controller_observation(_raw(), _Canonical(), None)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        _controller_observation(_raw(), _Other(), _canonical())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
