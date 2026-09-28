"""Opening-trace contract and fail-closed validation.

Traces are literal, ordered primitive action dicts for one source seat.  The
legacy built-ins cover d0-d3 (96 turns); extended traces may end at any
(day, hour) boundary and hand off on the immediately following primitive turn.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

ENGINE_VERSION = "1.32.7"
TRACE_FORMAT_VERSION = 1
FIRST_DAY = 0
LAST_DAY = 3  # legacy extractor/default horizon only
TURNS_PER_DAY = 24
EXPECTED_TURNS = (LAST_DAY - FIRST_DAY + 1) * TURNS_PER_DAY
MAX_MARKET_ORDERS = 10

DEFAULT_IDENTITY = "standard_mixed"
IDENTITIES = (
    "standard_mixed",
    "pasture_heavy",
    "carrot_start",
    "fourth_quadrant_s0",
    "fourth_quadrant_s1",
    "tetsuya_s1",
    "dsm_d0_d6h3",
)
VALID_SEATS = (0, 1)
_ACTION_KEYS = frozenset({"farmer", "hands", "market"})


class TraceError(ValueError):
    pass


def _fail(msg: str) -> None:
    raise TraceError(msg)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def compute_content_digest(turns: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(turns)).hexdigest()


def validate_action(action: Any, *, label: str) -> None:
    if not isinstance(action, dict):
        _fail(f"{label}: action must be a dict, got {type(action).__name__}")
    keys = frozenset(action)
    if keys != _ACTION_KEYS:
        _fail(f"{label}: action keys must be exactly {sorted(_ACTION_KEYS)}, got {sorted(keys)}")
    farmer = action["farmer"]
    if not isinstance(farmer, list) or not farmer or not isinstance(farmer[0], str):
        _fail(f"{label}: 'farmer' must be a non-empty list starting with an op string")
    hands = action["hands"]
    if not isinstance(hands, list):
        _fail(f"{label}: 'hands' must be a list")
    for i, op in enumerate(hands):
        if not isinstance(op, list) or not op or not isinstance(op[0], str):
            _fail(f"{label}: hands[{i}] must be a non-empty list starting with an op string")
    market = action["market"]
    if not isinstance(market, list):
        _fail(f"{label}: 'market' must be a list")
    if len(market) > MAX_MARKET_ORDERS:
        _fail(f"{label}: market has {len(market)} orders, max is {MAX_MARKET_ORDERS}")
    for i, order in enumerate(market):
        if not isinstance(order, list) or not order or not isinstance(order[0], str):
            _fail(f"{label}: market[{i}] must be a non-empty list starting with an order-op string")


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict):
        _fail("provenance must be a dict")
    episode = provenance.get("source_episode")
    if not isinstance(episode, int) or isinstance(episode, bool) or episode <= 0:
        _fail(f"provenance.source_episode must be a positive int, got {episode!r}")
    seat = provenance.get("source_seat")
    if seat not in VALID_SEATS:
        _fail(f"provenance.source_seat must be one of {list(VALID_SEATS)}, got {seat!r}")
    seed = provenance.get("source_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail(f"provenance.source_seed must be an int, got {seed!r}")
    player = provenance.get("source_player")
    if not isinstance(player, str) or not player:
        _fail(f"provenance.source_player must be a non-empty string, got {player!r}")
    digest = provenance.get("source_replay_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        _fail("provenance.source_replay_sha256 must be a 64-char lowercase hex digest")


def trace_last_phase(doc: dict[str, Any]) -> tuple[int, int]:
    horizon = doc.get("horizon") or {}
    last_day = horizon.get("last_day")
    last_hour = horizon.get("last_hour", TURNS_PER_DAY - 1)
    if not isinstance(last_day, int) or isinstance(last_day, bool):
        _fail(f"horizon.last_day must be an int, got {last_day!r}")
    if not isinstance(last_hour, int) or isinstance(last_hour, bool) or not 0 <= last_hour < TURNS_PER_DAY:
        _fail(f"horizon.last_hour must be in [0, {TURNS_PER_DAY - 1}], got {last_hour!r}")
    return last_day, last_hour


def trace_handoff_phase(doc: dict[str, Any]) -> tuple[int, int]:
    day, hour = trace_last_phase(doc)
    hour += 1
    if hour >= TURNS_PER_DAY:
        day, hour = day + 1, 0
    return day, hour


def _expected_turn_phases(last_day: int, last_hour: int) -> list[tuple[int, int]]:
    if last_day < FIRST_DAY:
        _fail(f"horizon.last_day must be >= {FIRST_DAY}, got {last_day}")
    phases: list[tuple[int, int]] = []
    for day in range(FIRST_DAY, last_day + 1):
        final_hour = last_hour if day == last_day else TURNS_PER_DAY - 1
        phases.extend((day, hour) for hour in range(final_hour + 1))
    return phases


def validate_trace(doc: Any) -> None:
    if not isinstance(doc, dict):
        _fail(f"trace document must be a dict, got {type(doc).__name__}")
    if doc.get("format_version") != TRACE_FORMAT_VERSION:
        _fail(f"format_version must be {TRACE_FORMAT_VERSION}, got {doc.get('format_version')!r}")
    identity = doc.get("identity")
    if identity not in IDENTITIES:
        _fail(f"identity must be one of {list(IDENTITIES)}, got {identity!r}")
    if doc.get("module_version") != ENGINE_VERSION:
        _fail(f"module_version must be {ENGINE_VERSION!r}, got {doc.get('module_version')!r}")
    horizon = doc.get("horizon")
    if not isinstance(horizon, dict):
        _fail("horizon must be a dict")
    allowed = {"first_day", "last_day", "turns_per_day", "last_hour", "handoff_day", "handoff_hour", "turn_count"}
    if any(key not in allowed for key in horizon):
        _fail(f"horizon has unexpected fields: {sorted(set(horizon) - allowed)}")
    if horizon.get("first_day") != FIRST_DAY:
        _fail(f"horizon.first_day must be {FIRST_DAY}")
    if horizon.get("turns_per_day") != TURNS_PER_DAY:
        _fail(f"horizon.turns_per_day must be {TURNS_PER_DAY}")
    last_day, last_hour = trace_last_phase(doc)
    expected_phases = _expected_turn_phases(last_day, last_hour)
    handoff_day, handoff_hour = trace_handoff_phase(doc)
    if "handoff_day" in horizon and horizon["handoff_day"] != handoff_day:
        _fail(f"horizon.handoff_day must be {handoff_day}")
    if "handoff_hour" in horizon and horizon["handoff_hour"] != handoff_hour:
        _fail(f"horizon.handoff_hour must be {handoff_hour}")
    if "turn_count" in horizon and horizon["turn_count"] != len(expected_phases):
        _fail(f"horizon.turn_count must be {len(expected_phases)}")
    _validate_provenance(doc.get("provenance"))

    turns = doc.get("turns")
    if not isinstance(turns, list):
        _fail(f"turns must be a list, got {type(turns).__name__}")
    if len(turns) != len(expected_phases):
        _fail(f"trace must contain exactly {len(expected_phases)} turns, got {len(turns)}")
    for pos, (turn, expected) in enumerate(zip(turns, expected_phases)):
        if not isinstance(turn, dict):
            _fail(f"turn[{pos}] must be a dict, got {type(turn).__name__}")
        day, hour = turn.get("day"), turn.get("hour")
        if (day, hour) != expected:
            _fail(f"turn[{pos}]: expected (day,hour)={expected}, got ({day},{hour})")
        validate_action(turn.get("action"), label=f"turn (day={day}, hour={hour})")
    digest = doc.get("content_digest")
    actual = compute_content_digest(turns)
    if digest != actual:
        _fail(f"content_digest mismatch: recorded {digest!r}, computed {actual!r}")


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def built_in_identities() -> tuple[str, ...]:
    return IDENTITIES


def load_built_in_trace(identity: str = DEFAULT_IDENTITY) -> dict[str, Any]:
    if identity not in IDENTITIES:
        _fail(f"unknown opening identity {identity!r}; known: {list(IDENTITIES)}")
    path = os.path.join(_DATA_DIR, f"{identity}.json")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceError(f"{path}: invalid JSON: {exc}") from exc
    validate_trace(doc)
    if doc.get("identity") != identity:
        _fail(f"{path}: file identity {doc.get('identity')!r} does not match {identity!r}")
    return copy.deepcopy(doc)


def action_for(trace_doc: dict[str, Any], day: int, hour: int) -> dict[str, Any]:
    if not isinstance(day, int) or not isinstance(hour, int):
        raise TraceError(f"(day, hour) must be ints, got day={day!r} hour={hour!r}")
    last_day, last_hour = trace_last_phase(trace_doc)
    if day < FIRST_DAY or day > last_day or not 0 <= hour < TURNS_PER_DAY or (day == last_day and hour > last_hour):
        hd, hh = trace_handoff_phase(trace_doc)
        raise TraceError(f"(day={day}, hour={hour}) is outside opening horizon; handoff is ({hd},{hh})")
    index = day * TURNS_PER_DAY + hour
    turn = trace_doc["turns"][index]
    if (turn.get("day"), turn.get("hour")) != (day, hour):
        raise TraceError(f"trace turn at index {index} is not (day={day}, hour={hour})")
    return copy.deepcopy(turn["action"])
