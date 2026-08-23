"""Focused tests for the opening_book trace package (issue #4, stage 1).

Runs entirely offline: committed traces plus a small synthetic replay fixture.
No gitignored sample data and no Kaggle access are required.
"""

from __future__ import annotations

import copy
import json

import pytest

from opening_book import DEFAULT_IDENTITY, EXPECTED_TURNS, TraceError
from opening_book.extract import extract_opening_trace, write_trace
from opening_book.trace import (
    action_for,
    built_in_identities,
    compute_content_digest,
    load_built_in_trace,
    validate_trace,
)


# ---------------------------------------------------------------------------
# Synthetic replay fixture (no raw sample data dependency)
# ---------------------------------------------------------------------------

def make_synthetic_replay() -> dict:
    """Minimal 1.32.7-shaped replay with 97 steps covering days 0-3 + d4h0."""
    steps = []
    for turn in range(97):  # observations for turns 0..96 (d0h0 .. d4h0)
        obs = {"day": turn // 24, "hour": turn % 24}
        seat_entries = []
        for seat in range(2):
            if turn == 0:
                action = None
            else:
                acted_turn = turn - 1
                action = {
                    "farmer": ["PASS"],
                    "hands": [],
                    "market": [["BUY_SEED", "WHEAT", 1 + seat + acted_turn]],
                }
            seat_entries.append({"observation": obs, "action": action})
        steps.append(seat_entries)
    return {
        "module_version": "1.32.7",
        "info": {
            "EpisodeId": 123,
            "seed": 42,
            "TeamNames": ["Alpha", "Beta"],
        },
        "rewards": [100, 200],
        "steps": steps,
    }


@pytest.fixture()
def synthetic_replay_path(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(make_synthetic_replay()), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Built-in identities
# ---------------------------------------------------------------------------

def test_built_in_identities_and_default():
    assert built_in_identities() == ("standard_mixed", "pasture_heavy")
    assert DEFAULT_IDENTITY == "standard_mixed"
    for identity in built_in_identities():
        doc = load_built_in_trace(identity)
        assert doc["identity"] == identity
        validate_trace(doc)


def test_load_rejects_unknown_identity():
    with pytest.raises(TraceError, match="unknown opening identity"):
        load_built_in_trace("wheat_monoculture")


def test_builtin_traces_are_96_ordered_unique_turns():
    for identity in built_in_identities():
        doc = load_built_in_trace(identity)
        turns = doc["turns"]
        assert len(turns) == EXPECTED_TURNS == 96
        keys = [(t["day"], t["hour"]) for t in turns]
        assert keys == [(d, h) for d in range(4) for h in range(24)]
        assert len(set(keys)) == 96
        # handoff turn d4h0 must not exist anywhere in the document
        assert all(day <= 3 for day, _ in keys)
        assert doc["module_version"] == "1.32.7"


def test_builtin_indexing_d0h0_d3h23_and_d4h0_rejected():
    doc = load_built_in_trace()
    first = action_for(doc, 0, 0)
    last = action_for(doc, 3, 23)
    for action in (first, last):
        assert set(action) == {"farmer", "hands", "market"}
    with pytest.raises(TraceError, match="outside the opening horizon"):
        action_for(doc, 4, 0)
    with pytest.raises(TraceError):
        action_for(doc, 3, 24)
    with pytest.raises(TraceError):
        action_for(doc, -1, 0)


def test_action_for_returns_defensive_copy():
    doc = load_built_in_trace()
    a1 = action_for(doc, 0, 0)
    a1["market"].append(["HACK"])
    a1["farmer"][0] = "HACKED"
    a2 = action_for(doc, 0, 0)
    assert a2 != a1
    assert a2["farmer"][0] != "HACKED"
    # mutating the returned action must not corrupt the loaded document either
    assert action_for(load_built_in_trace(), 0, 0) == a2


def test_builtin_provenance_records_source_seat_and_digest():
    seen_seats = set()
    for identity in built_in_identities():
        doc = load_built_in_trace(identity)
        prov = doc["provenance"]
        assert prov["source_seat"] in (0, 1)
        seen_seats.add(prov["source_seat"])
        assert isinstance(prov["source_episode"], int)
        assert isinstance(prov["source_seed"], int)
        assert prov["source_player"]
        assert len(prov["source_replay_sha256"]) == 64
        assert doc["content_digest"] == compute_content_digest(doc["turns"])
    # provenance must be capable of recording either source seat; both
    # committed traces carry valid seat metadata (both sources are seat 0).
    assert seen_seats <= {0, 1}


# ---------------------------------------------------------------------------
# Validation failures (mutated copies of a committed trace; offline)
# ---------------------------------------------------------------------------

@pytest.fixture()
def standard_doc():
    return load_built_in_trace("standard_mixed")


def _assert_invalid(doc, match):
    with pytest.raises(TraceError, check=lambda e: match in str(e)):
        validate_trace(doc)


def test_reject_duplicate_turn(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["turns"][5]["hour"] = 4  # duplicate of turn index 4
    _assert_invalid(doc, "contiguous")


def test_reject_missing_turn(standard_doc):
    doc = copy.deepcopy(standard_doc)
    del doc["turns"][10]
    doc["turns"].append(dict(doc["turns"][-1]))  # keep length at 96
    _assert_invalid(doc, "expected")


def test_reject_wrong_turn_count(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["turns"] = doc["turns"][:95]
    _assert_invalid(doc, "exactly 96")


def test_reject_unordered_turns(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["turns"][0], doc["turns"][1] = doc["turns"][1], doc["turns"][0]
    _assert_invalid(doc, "contiguous")


def test_reject_bad_module_version(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["module_version"] = "1.32.6"
    _assert_invalid(doc, "module_version")


def test_reject_unknown_identity(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["identity"] = "aggressive"
    _assert_invalid(doc, "identity")


def test_reject_wrong_horizon(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["horizon"]["last_day"] = 4
    _assert_invalid(doc, "horizon")


def test_reject_market_over_cap(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["turns"][0]["action"]["market"] = [["HIRE"]] * 11
    _assert_invalid(doc, "max is 10")


def test_reject_bad_action_shape(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["turns"][3]["action"] = {"farmer": ["PASS"], "hands": []}
    _assert_invalid(doc, "keys")

    doc = copy.deepcopy(standard_doc)
    doc["turns"][3]["action"]["hands"] = ["WEST"]
    _assert_invalid(doc, "hands[0]")

    doc = copy.deepcopy(standard_doc)
    doc["turns"][3]["action"]["farmer"] = []
    _assert_invalid(doc, "'farmer'")


def test_reject_digest_mismatch(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["content_digest"] = "0" * 64
    _assert_invalid(doc, "content_digest mismatch")


def test_reject_provenance_mismatch(standard_doc):
    doc = copy.deepcopy(standard_doc)
    doc["provenance"]["source_seat"] = 2
    _assert_invalid(doc, "source_seat")

    doc = copy.deepcopy(standard_doc)
    doc["provenance"]["source_replay_sha256"] = "xyz"
    _assert_invalid(doc, "source_replay_sha256")

    doc = copy.deepcopy(standard_doc)
    doc["provenance"]["source_episode"] = 0
    _assert_invalid(doc, "source_episode")


# ---------------------------------------------------------------------------
# Deterministic extractor over the synthetic fixture
# ---------------------------------------------------------------------------

def test_extractor_deterministic_and_valid(synthetic_replay_path, tmp_path):
    out1 = tmp_path / "t1.json"
    out2 = tmp_path / "t2.json"
    doc1 = extract_opening_trace(synthetic_replay_path, 0, "standard_mixed")
    write_trace(doc1, str(out1))
    doc2 = extract_opening_trace(synthetic_replay_path, 0, "standard_mixed")
    write_trace(doc2, str(out2))
    assert out1.read_bytes() == out2.read_bytes()
    validate_trace(doc1)
    assert doc1["provenance"] == {
        "source_episode": 123,
        "source_seat": 0,
        "source_seed": 42,
        "source_player": "Alpha",
        "source_replay_sha256": doc1["provenance"]["source_replay_sha256"],
    }
    assert len(doc1["provenance"]["source_replay_sha256"]) == 64
    # exact literal playback: turn t carries the action submitted at turn t
    assert doc1["turns"][0]["action"]["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert doc1["turns"][95]["action"]["market"] == [["BUY_SEED", "WHEAT", 96]]
    assert doc1["turns"][95]["day"] == 3 and doc1["turns"][95]["hour"] == 23


def test_extractor_rejects_bad_version_and_seat(tmp_path):
    replay = make_synthetic_replay()
    replay["module_version"] = "1.33.0"
    path = tmp_path / "bad_version.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(TraceError, match="module_version"):
        extract_opening_trace(str(path), 0, "standard_mixed")

    good = tmp_path / "good.json"
    good.write_text(json.dumps(make_synthetic_replay()), encoding="utf-8")
    with pytest.raises(TraceError, match="seat"):
        extract_opening_trace(str(good), 2, "standard_mixed")


def test_extractor_rejects_incomplete_horizon(tmp_path):
    replay = make_synthetic_replay()
    replay["steps"] = replay["steps"][:50]  # truncates day 2 onward
    path = tmp_path / "short.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(TraceError, match="no action step"):
        extract_opening_trace(str(path), 0, "standard_mixed")
