"""Deterministic canonical debug-trace artifact contract (issue #11).

The trace stores full canonical state snapshots observed by a future runner;
it never owns or reruns an environment.  Canonical state is supplied by the
caller through the public ``canonical_state()`` seam, so raw fast-engine tile
fields are deliberately not translated here.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEBUG_TRACE_SCHEMA_VERSION = 1

_ENVELOPE_KEYS = frozenset({"schema_version", "metadata", "turns"})
_TURN_KEYS = frozenset({
    "step",
    "day",
    "hour",
    "current_seat",
    "canonical_state",
    "joint_actions",
    "executor_debug",
})
_REQUIRED_TURN_KEYS = frozenset({"step", "day", "hour", "canonical_state"})
_STATE_KEYS = frozenset({
    "step",
    "day",
    "hour",
    "farms",
    "privates",
    "market",
    "town",
    "rewards",
    "statuses",
})
_FARM_KEYS = frozenset({
    "money",
    "tiles",
    "farmer",
    "hands",
    "unlocked_quadrants",
    "hires_today",
})
_PRIVATE_KEYS = frozenset({"shed", "seeds", "inventories"})
_PLANT_KEYS = frozenset({
    "kind",
    "crop",
    "planted_day",
    "max_lifespan_step",
    "yield_units",
    "watered_today",
    "consecutive_unwatered",
    "fertilized_until_day",
})
_LIVESTOCK_KEYS = frozenset({
    "kind",
    "animal",
    "placed_day",
    "yield_units",
    "consecutive_unfed",
    "fed_today",
    "cared_today",
    "fertilizer_available",
    "pending_care_bonus",
})
_METADATA_TYPED_FIELDS = {
    "seed": int,
    "seat": int,
    "view": str,
    "backend": str,
    "engine": str,
}
_METADATA_KEYS = frozenset({*(_METADATA_TYPED_FIELDS), "provenance"})
_NONDETERMINISTIC_METADATA_KEYS = frozenset({
    "created_at",
    "generated_at",
    "generation_timestamp",
    "time",
    "timestamp",
    "wall_clock",
    "wallclock",
})


class DebugTraceError(ValueError):
    """Raised when a debug trace is malformed or not JSON-safe."""


def _fail(message: str) -> None:
    raise DebugTraceError(message)


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a dict, got {type(value).__name__}")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        _fail(f"{label} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be >= {minimum}, got {value!r}")


def _validate_json_safe(value: Any, label: str = "document", active: set[int] | None = None) -> None:
    """Reject values that JSON cannot represent without coercion."""
    if active is None:
        active = set()
    if value is None or type(value) is bool or type(value) is str:
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{label} contains non-finite float {value!r}")
        return
    if isinstance(value, dict):
        marker = id(value)
        if marker in active:
            _fail(f"{label} contains a cyclic value")
        active.add(marker)
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{label} has a non-string JSON object key {key!r}")
            _validate_json_safe(item, f"{label}.{key}", active)
        active.remove(marker)
        return
    if isinstance(value, list):
        marker = id(value)
        if marker in active:
            _fail(f"{label} contains a cyclic value")
        active.add(marker)
        for index, item in enumerate(value):
            _validate_json_safe(item, f"{label}[{index}]", active)
        active.remove(marker)
        return
    _fail(f"{label} contains non-JSON value of type {type(value).__name__}")


def _require_fields(value: dict[str, Any], required: frozenset[str], label: str) -> None:
    missing = required - value.keys()
    if missing:
        _fail(f"{label} missing required fields {sorted(missing)}")


def _validate_tile(tile: Any, label: str) -> None:
    if tile is None or tile == "LOCKED":
        return
    tile_dict = _require_dict(tile, label)
    if "age" in tile_dict:
        _fail(f"{label} uses raw fast tile alias 'age'; supply planted_day/placed_day")
    kind = tile_dict.get("kind")
    if kind == "PLANT":
        _require_fields(tile_dict, _PLANT_KEYS, label)
        _require_int(tile_dict["planted_day"], f"{label}.planted_day")
    elif kind in ("COOP", "PASTURE") and "animal" in tile_dict:
        _require_fields(tile_dict, _LIVESTOCK_KEYS, label)
        _require_int(tile_dict["placed_day"], f"{label}.placed_day")


def _validate_canonical_state(state: Any, turn: dict[str, Any]) -> None:
    state_dict = _require_dict(state, "turn.canonical_state")
    _require_fields(state_dict, _STATE_KEYS, "turn.canonical_state")
    for field in ("step", "day", "hour"):
        _require_int(state_dict[field], f"turn.canonical_state.{field}", minimum=0)
        if state_dict[field] != turn[field]:
            _fail(
                f"turn.canonical_state.{field} {state_dict[field]!r} does not "
                f"match turn.{field} {turn[field]!r}"
            )

    farms = state_dict["farms"]
    if not isinstance(farms, list) or len(farms) < 2:
        _fail("turn.canonical_state.farms must contain both seats")
    for farm_index, farm in enumerate(farms):
        farm_dict = _require_dict(farm, f"turn.canonical_state.farms[{farm_index}]")
        _require_fields(farm_dict, _FARM_KEYS, f"turn.canonical_state.farms[{farm_index}]")
        tiles = farm_dict["tiles"]
        if not isinstance(tiles, list):
            _fail(f"turn.canonical_state.farms[{farm_index}].tiles must be a list")
        for row_index, row in enumerate(tiles):
            if not isinstance(row, list):
                _fail(
                    f"turn.canonical_state.farms[{farm_index}].tiles[{row_index}] "
                    "must be a list"
                )
            for column_index, tile in enumerate(row):
                _validate_tile(
                    tile,
                    f"turn.canonical_state.farms[{farm_index}].tiles[{row_index}]"
                    f"[{column_index}]")

    privates = state_dict["privates"]
    if not isinstance(privates, list) or len(privates) != len(farms):
        _fail("turn.canonical_state.privates must align with farms")
    for seat, private in enumerate(privates):
        private_dict = _require_dict(private, f"turn.canonical_state.privates[{seat}]")
        _require_fields(private_dict, _PRIVATE_KEYS, f"turn.canonical_state.privates[{seat}]")

    for field in ("market", "town"):
        _require_dict(state_dict[field], f"turn.canonical_state.{field}")
    for field in ("rewards", "statuses"):
        values = state_dict[field]
        if not isinstance(values, list) or len(values) != len(farms):
            _fail(f"turn.canonical_state.{field} must align with farms")


def _validate_metadata(metadata: Any) -> None:
    metadata_dict = _require_dict(metadata, "metadata")
    unknown = metadata_dict.keys() - _METADATA_KEYS
    if unknown:
        _fail(f"metadata has unsupported fields {sorted(unknown)}")
    for field, expected_type in _METADATA_TYPED_FIELDS.items():
        if field not in metadata_dict:
            continue
        value = metadata_dict[field]
        if type(value) is not expected_type:
            _fail(f"metadata.{field} must be a {expected_type.__name__}")
        if field in ("seed", "seat") and value < 0:
            _fail(f"metadata.{field} must be >= 0")
        if field in ("view", "backend", "engine") and not value:
            _fail(f"metadata.{field} must be a non-empty string")
    if "provenance" in metadata_dict:
        provenance = _require_dict(metadata_dict["provenance"], "metadata.provenance")

        def reject_wall_clock_fields(value: Any, label: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.casefold() in _NONDETERMINISTIC_METADATA_KEYS:
                        _fail(f"{label}.{key} is nondeterministic wall-clock metadata")
                    reject_wall_clock_fields(item, f"{label}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    reject_wall_clock_fields(item, f"{label}[{index}]")

        reject_wall_clock_fields(provenance, "metadata.provenance")


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        source: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        source = metadata
    else:
        _fail(f"metadata must be a mapping, got {type(metadata).__name__}")
    copied = copy.deepcopy(dict(source))
    _validate_json_safe(copied, "metadata")
    _validate_metadata(copied)
    return copied


def _validate_turn(turn: Any, previous_step: int | None = None) -> None:
    turn_dict = _require_dict(turn, "turn")
    missing = _REQUIRED_TURN_KEYS - turn_dict.keys()
    if missing:
        _fail(f"turn missing required fields {sorted(missing)}")
    unknown = turn_dict.keys() - _TURN_KEYS
    if unknown:
        _fail(f"turn has unsupported fields {sorted(unknown)}")
    for field in ("step", "day", "hour"):
        _require_int(turn_dict[field], f"turn.{field}", minimum=0)
    if turn_dict["hour"] > 23:
        _fail(f"turn.hour must be in 0..23, got {turn_dict['hour']!r}")
    if previous_step is not None and turn_dict["step"] <= previous_step:
        _fail(
            f"turn.step must be strictly increasing; got {turn_dict['step']} "
            f"after {previous_step}"
        )
    if "current_seat" in turn_dict:
        _require_int(turn_dict["current_seat"], "turn.current_seat", minimum=0)
    if "joint_actions" in turn_dict:
        actions = _require_dict(turn_dict["joint_actions"], "turn.joint_actions")
        if any(type(seat) is not str for seat in actions):
            _fail("turn.joint_actions keys must be string seat ids")
    if "executor_debug" in turn_dict:
        debug = _require_dict(turn_dict["executor_debug"], "turn.executor_debug")
        for seat, snapshot in debug.items():
            if type(seat) is not str:
                _fail("turn.executor_debug keys must be string seat ids")
            if not isinstance(snapshot, dict):
                _fail(f"turn.executor_debug[{seat!r}] must be a dict")
    _validate_canonical_state(turn_dict["canonical_state"], turn_dict)


def validate_trace(document: Any) -> None:
    """Fail-closed validation of a complete debug-trace envelope."""
    document_dict = _require_dict(document, "trace envelope")
    if document_dict.keys() != _ENVELOPE_KEYS:
        missing = _ENVELOPE_KEYS - document_dict.keys()
        unknown = document_dict.keys() - _ENVELOPE_KEYS
        if missing:
            _fail(f"trace envelope missing required fields {sorted(missing)}")
        _fail(f"trace envelope has unsupported fields {sorted(unknown)}")
    _require_int(document_dict["schema_version"], "schema_version", minimum=1)
    if document_dict["schema_version"] != DEBUG_TRACE_SCHEMA_VERSION:
        _fail(
            f"schema_version must be {DEBUG_TRACE_SCHEMA_VERSION}, got "
            f"{document_dict['schema_version']!r}"
        )
    _validate_json_safe(document_dict)
    _validate_metadata(document_dict["metadata"])
    turns = document_dict["turns"]
    if not isinstance(turns, list):
        _fail(f"turns must be a list, got {type(turns).__name__}")
    previous_step: int | None = None
    for index, turn in enumerate(turns):
        try:
            _validate_turn(turn, previous_step)
        except DebugTraceError as exc:
            _fail(f"turn[{index}]: {exc}")
        previous_step = turn["step"]


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    """Return the stable compact UTF-8 representation of a valid trace."""
    validate_trace(document)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DebugTraceError(f"trace is not strictly JSON-serializable: {exc}") from exc


class TraceRecorder:
    """Collect defensive copies of observed canonical snapshots in order."""

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        self._metadata = _copy_metadata(metadata)
        self._turns: list[dict[str, Any]] = []

    def append_turn(
        self,
        *,
        step: int,
        day: int,
        hour: int,
        canonical_state: dict[str, Any],
        current_seat: int | None = None,
        joint_actions: dict[str, Any] | None = None,
        executor_debug: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Record one observed turn; optional fields are omitted when absent."""
        turn: dict[str, Any] = {
            "step": step,
            "day": day,
            "hour": hour,
            "canonical_state": canonical_state,
        }
        if current_seat is not None:
            turn["current_seat"] = current_seat
        if joint_actions is not None:
            turn["joint_actions"] = joint_actions
        if executor_debug is not None:
            turn["executor_debug"] = executor_debug
        _validate_json_safe(turn, "turn")
        _validate_turn(turn, self._turns[-1]["step"] if self._turns else None)
        self._turns.append(copy.deepcopy(turn))

    def build(self) -> dict[str, Any]:
        """Return a validated defensive copy of the current envelope."""
        document = {
            "schema_version": DEBUG_TRACE_SCHEMA_VERSION,
            "metadata": copy.deepcopy(self._metadata),
            "turns": copy.deepcopy(self._turns),
        }
        validate_trace(document)
        return document


def build_trace(
    metadata: Mapping[str, Any] | None = None,
    turns: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a trace from turn mappings using the same strict recorder API."""
    recorder = TraceRecorder(metadata)
    for index, source_turn in enumerate(turns):
        if not isinstance(source_turn, Mapping):
            _fail(f"turn[{index}] must be a dict")
        source_turn = dict(source_turn)
        unknown = source_turn.keys() - _TURN_KEYS
        if unknown:
            _fail(f"turn[{index}] has unsupported fields {sorted(unknown)}")
        missing = _REQUIRED_TURN_KEYS - source_turn.keys()
        if missing:
            _fail(f"turn[{index}] missing required fields {sorted(missing)}")
        recorder.append_turn(
            step=source_turn["step"],
            day=source_turn["day"],
            hour=source_turn["hour"],
            canonical_state=source_turn["canonical_state"],
            current_seat=source_turn.get("current_seat"),
            joint_actions=source_turn.get("joint_actions"),
            executor_debug=source_turn.get("executor_debug"),
        )
    return recorder.build()


def save_trace(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Validate and write one deterministic UTF-8 trace file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(document))
    return destination


def load_trace(path: str | Path) -> dict[str, Any]:
    """Load, validate, and return a fresh trace document."""
    source = Path(path)
    try:
        document = json.loads(source.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DebugTraceError(f"{source}: invalid trace JSON: {exc}") from exc
    validate_trace(document)
    return copy.deepcopy(document)


__all__ = [
    "DEBUG_TRACE_SCHEMA_VERSION",
    "DebugTraceError",
    "TraceRecorder",
    "build_trace",
    "canonical_json_bytes",
    "load_trace",
    "save_trace",
    "validate_trace",
]
