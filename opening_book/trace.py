"""Opening-trace contract and fail-closed validation (issue #4, stage 1).

A trace is the literal, ordered sequence of submitted primitive action dicts
(raw/executor shape ``{"farmer": ..., "hands": [...], "market": [...]}``) for
one source seat over days 0-3 inclusive (4 x 24 = 96 turns). The handoff turn
day 4 hour 0 is outside the trace by construction.

Traces are literal source-seat playback data, not interpreted workload
bundles. All validation fails closed with :class:`TraceError`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

ENGINE_VERSION = "1.32.7"
TRACE_FORMAT_VERSION = 1

# Opening horizon: days 0-3 inclusive; handoff at day 4 hour 0 (outside trace).
FIRST_DAY = 0
LAST_DAY = 3
TURNS_PER_DAY = 24
EXPECTED_TURNS = (LAST_DAY - FIRST_DAY + 1) * TURNS_PER_DAY  # 96

# MECHANICS.md "Shared Market": up to 10 market order slots per player per turn.
MAX_MARKET_ORDERS = 10

DEFAULT_IDENTITY = "standard_mixed"
IDENTITIES = ("standard_mixed", "pasture_heavy")

VALID_SEATS = (0, 1)

_ACTION_KEYS = frozenset({"farmer", "hands", "market"})


class TraceError(ValueError):
    """Raised for any malformed or untrusted opening trace."""


def _fail(msg: str) -> None:
    raise TraceError(msg)


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic compact serialization used for digests and files."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def compute_content_digest(turns: list[dict[str, Any]]) -> str:
    """SHA-256 hex digest over the canonical serialization of the turns."""
    return hashlib.sha256(canonical_json_bytes(turns)).hexdigest()


def validate_action(action: Any, *, label: str) -> None:
    """Validate one submitted primitive action dict (raw/executor shape)."""
    if not isinstance(action, dict):
        _fail(f"{label}: action must be a dict, got {type(action).__name__}")
    keys = frozenset(action)
    if keys != _ACTION_KEYS:
        _fail(
            f"{label}: action keys must be exactly {sorted(_ACTION_KEYS)}, "
            f"got {sorted(keys)}"
        )
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
        _fail(
            f"{label}: market has {len(market)} orders, "
            f"max is {MAX_MARKET_ORDERS}"
        )
    for i, order in enumerate(market):
        if not isinstance(order, list) or not order or not isinstance(order[0], str):
            _fail(
                f"{label}: market[{i}] must be a non-empty list starting "
                f"with an order-op string"
            )


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
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        _fail(
            "provenance.source_replay_sha256 must be a 64-char lowercase hex "
            f"digest, got {digest!r}"
        )


def validate_trace(doc: Any) -> None:
    """Fail-closed validation of a full trace document.

    Checks format/version/horizon metadata, provenance shape, exactly 96 turns
    strictly ordered and contiguous over (day 0..3, hour 0..23), per-turn
    action shape and market cap, and the content digest.
    """
    if not isinstance(doc, dict):
        _fail(f"trace document must be a dict, got {type(doc).__name__}")
    if doc.get("format_version") != TRACE_FORMAT_VERSION:
        _fail(
            f"format_version must be {TRACE_FORMAT_VERSION}, "
            f"got {doc.get('format_version')!r}"
        )
    identity = doc.get("identity")
    if identity not in IDENTITIES:
        _fail(f"identity must be one of {list(IDENTITIES)}, got {identity!r}")
    if doc.get("module_version") != ENGINE_VERSION:
        _fail(f"module_version must be {ENGINE_VERSION!r}, got {doc.get('module_version')!r}")
    horizon = doc.get("horizon")
    expected_horizon = {
        "first_day": FIRST_DAY,
        "last_day": LAST_DAY,
        "turns_per_day": TURNS_PER_DAY,
    }
    if horizon != expected_horizon:
        _fail(f"horizon must be {expected_horizon}, got {horizon!r}")
    _validate_provenance(doc.get("provenance"))

    turns = doc.get("turns")
    if not isinstance(turns, list):
        _fail(f"turns must be a list, got {type(turns).__name__}")
    if len(turns) != EXPECTED_TURNS:
        _fail(f"trace must contain exactly {EXPECTED_TURNS} turns, got {len(turns)}")

    seen: set[tuple[int, int]] = set()
    expected_index = 0
    for pos, turn in enumerate(turns):
        if not isinstance(turn, dict):
            _fail(f"turn[{pos}] must be a dict, got {type(turn).__name__}")
        day = turn.get("day")
        hour = turn.get("hour")
        if not isinstance(day, int) or not isinstance(hour, int):
            _fail(
                f"turn[{pos}]: day/hour must be ints, got day={day!r} hour={hour!r}"
            )
        expected_day = FIRST_DAY + expected_index // TURNS_PER_DAY
        expected_hour = expected_index % TURNS_PER_DAY
        if (day, hour) != (expected_day, expected_hour):
            _fail(
                f"turn[{pos}]: expected (day,hour)=({expected_day},{expected_hour}), "
                f"got ({day},{hour}); trace must be contiguous and ordered"
            )
        if (day, hour) in seen:
            _fail(f"turn[{pos}]: duplicate (day,hour)=({day},{hour})")
        seen.add((day, hour))
        validate_action(turn.get("action"), label=f"turn (day={day}, hour={hour})")
        expected_index += 1

    digest = doc.get("content_digest")
    actual = compute_content_digest(turns)
    if digest != actual:
        _fail(f"content_digest mismatch: recorded {digest!r}, computed {actual!r}")


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def built_in_identities() -> tuple[str, ...]:
    return IDENTITIES


def load_built_in_trace(identity: str = DEFAULT_IDENTITY) -> dict[str, Any]:
    """Load and fully validate a committed identity trace.

    Returns a fresh deep copy on every call so callers cannot mutate the
    loaded artifact.
    """
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
    """Return a defensive deep copy of the action for one (day, hour) turn."""
    if not isinstance(day, int) or not isinstance(hour, int):
        raise TraceError(f"(day, hour) must be ints, got day={day!r} hour={hour!r}")
    if not (FIRST_DAY <= day <= LAST_DAY) or not (0 <= hour < TURNS_PER_DAY):
        raise TraceError(
            f"(day={day}, hour={hour}) is outside the opening horizon "
            f"days {FIRST_DAY}-{LAST_DAY}, hours 0-{TURNS_PER_DAY - 1}; "
            "the day 4 hour 0 handoff turn is not part of the trace"
        )
    index = (day - FIRST_DAY) * TURNS_PER_DAY + hour
    turn = trace_doc["turns"][index]
    if (turn.get("day"), turn.get("hour")) != (day, hour):
        # Validated traces are contiguous, so this is unreachable; kept as a
        # cheap guard for callers passing unvalidated documents.
        raise TraceError(f"trace turn at index {index} is not (day={day}, hour={hour})")
    return copy.deepcopy(turn["action"])
