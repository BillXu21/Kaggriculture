"""Versioned, fixed ``DailyPlan`` tape for deterministic executor A/B runs.

The tape is a strategy-free boundary: it stores only canonical plans keyed by
absolute game day.  It never stores observations, engine tiles, model objects,
or executor state.  A tape must carry explicit provenance, including the
checkpoint identity; callers may not silently substitute a missing checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

from executor_v0.plan import SELL_BIN_ANCHORS, DailyPlan
from replay_daily.constants import PRODUCTS

__all__ = [
    "FIXED_PLAN_TAPE_SCHEMA_VERSION",
    "FixedPlanTape",
    "FixedPlanTapeProvider",
    "PlanTapeError",
]


FIXED_PLAN_TAPE_SCHEMA_VERSION = 1
_REQUIRED_PROVENANCE = (
    "manager",
    "checkpoint",
    "model_variant",
    "seed",
    "seat",
    "opening_identity",
    "source_repo_sha",
    "backend",
    "engine",
    "recording_window",
)
_PLAN_FIELDS = frozenset({
    "crop_targets",
    "animal_targets",
    "land_count",
    "fertilizer_by_crop",
    "care_by_animal",
    "sell_quantities",
})

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"]
    | dict[str, "JSONValue"]
)


class PlanTapeError(ValueError):
    """Raised when a fixed-plan tape or provider request is invalid."""


def _fail(message: str) -> None:
    raise PlanTapeError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_json_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            _fail(f"{label} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{label} keys must be strings, got {key!r}")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    _fail(f"{label} is not JSON-safe: {type(value).__name__}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string, got {value!r}")
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _recording_window(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        _fail("provenance.recording_window must be a mapping")
    expected = {"start_day", "end_day"}
    if set(value) != expected:
        _fail(
            "provenance.recording_window must contain exactly "
            "start_day and end_day"
        )
    start = value["start_day"]
    end = value["end_day"]
    if not _is_int(start) or not _is_int(end):
        _fail("provenance.recording_window days must be integers")
    if start > end:
        _fail(
            "provenance.recording_window start_day must be <= end_day, "
            f"got {start}>{end}"
        )
    return {"start_day": start, "end_day": end}


def _validate_named_provenance(value: Any, *, label: str) -> dict[str, JSONValue]:
    """Validate the narrow name/version shape used for backend and engine."""
    if not isinstance(value, Mapping):
        _fail(f"provenance.{label} must be a mapping")
    if "name" not in value or "version" not in value:
        _fail(f"provenance.{label} requires name and version")
    result = copy.deepcopy(dict(value))
    _validate_json_value(result, label=f"provenance.{label}")
    _nonempty_string(result["name"], f"provenance.{label}.name")
    _nonempty_string(result["version"], f"provenance.{label}.version")
    return result


def _validate_provenance(value: Any) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        _fail("provenance must be a mapping")
    missing = sorted(set(_REQUIRED_PROVENANCE) - set(value))
    if missing:
        _fail(f"provenance is missing required fields: {missing}")

    result = copy.deepcopy(dict(value))
    _validate_json_value(result, label="provenance")
    for field in ("manager", "checkpoint", "model_variant",
                  "opening_identity", "source_repo_sha"):
        result[field] = _nonempty_string(result[field], f"provenance.{field}")

    for field in ("seed", "seat"):
        if not _is_int(result[field]):
            _fail(f"provenance.{field} must be an integer, got {result[field]!r}")
    if result["seat"] not in (0, 1):
        _fail(f"provenance.seat must be 0 or 1, got {result['seat']}")
    result["backend"] = _validate_named_provenance(
        result["backend"], label="backend"
    )
    result["engine"] = _validate_named_provenance(
        result["engine"], label="engine"
    )
    result["recording_window"] = _recording_window(result["recording_window"])
    return result


def _validate_plan_json(value: Any, *, label: str) -> DailyPlan:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    if set(value) != _PLAN_FIELDS:
        _fail(
            f"{label} fields must be exactly {sorted(_PLAN_FIELDS)}, "
            f"got {sorted(value)}"
        )
    try:
        sell_json = value["sell_quantities"]
        if not isinstance(sell_json, Mapping):
            _fail(f"{label}.sell_quantities must be a mapping")
        expected_anchors = {str(anchor) for anchor in SELL_BIN_ANCHORS}
        if set(sell_json) != expected_anchors:
            _fail(
                f"{label}.sell_quantities must use canonical anchors "
                f"{sorted(expected_anchors)}"
            )
        # DailyPlan's canonical JSON view is {"anchor": {"product": qty}};
        # reconstruct the inverse mapping without introducing another plan type.
        sell_quantities: dict[str, dict[int, Any]] = {}
        for anchor, products in sell_json.items():
            anchor_int = int(anchor)
            if not isinstance(products, Mapping):
                _fail(
                    f"{label}.sell_quantities[{anchor!r}] must be a mapping"
                )
            if set(products) != set(PRODUCTS):
                _fail(
                    f"{label}.sell_quantities[{anchor!r}] products must be "
                    f"exactly {list(PRODUCTS)}"
                )
            for product, quantity in products.items():
                sell_quantities.setdefault(product, {})[anchor_int] = quantity
        return DailyPlan.create(
            crop_targets=value["crop_targets"],
            animal_targets=value["animal_targets"],
            land_count=value["land_count"],
            fertilizer_by_crop=value["fertilizer_by_crop"],
            care_by_animal=value["care_by_animal"],
            sell_quantities=sell_quantities,
        )
    except PlanTapeError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise PlanTapeError(f"{label} is malformed: {error}") from error


@dataclass(frozen=True)
class _PlanEntry:
    day: int
    plan: DailyPlan


class FixedPlanTape:
    """Immutable absolute-day mapping of canonical plans plus provenance."""

    def __init__(
        self,
        *,
        provenance: Mapping[str, Any],
        entries: tuple[_PlanEntry, ...],
        artifact_sha256: str | None = None,
    ) -> None:
        self._provenance = _validate_provenance(provenance)
        self._entries = entries
        self._by_day = {entry.day: entry.plan for entry in entries}
        expected = self._compute_sha256()
        if artifact_sha256 is not None and artifact_sha256 != expected:
            _fail(
                "artifact_sha256 mismatch: "
                f"recorded {artifact_sha256!r}, computed {expected!r}"
            )
        self._artifact_sha256 = expected

    @classmethod
    def create(
        cls,
        *,
        plans: Mapping[int, DailyPlan] | Iterable[tuple[int, DailyPlan]],
        provenance: Mapping[str, Any],
    ) -> "FixedPlanTape":
        """Validate and canonically order ``(absolute_day, DailyPlan)`` pairs."""
        if isinstance(plans, Mapping):
            pairs = plans.items()
        else:
            try:
                pairs = iter(plans)
            except TypeError as error:
                raise PlanTapeError("plans must be a mapping or iterable") from error
        entries: list[_PlanEntry] = []
        seen: set[int] = set()
        for index, pair in enumerate(pairs):
            try:
                day, plan = pair
            except (TypeError, ValueError) as error:
                _fail(f"plans[{index}] must be a (day, DailyPlan) pair")
                raise AssertionError from error  # pragma: no cover
            if not _is_int(day):
                _fail(f"plans[{index}].day must be an integer, got {day!r}")
            if day in seen:
                _fail(f"duplicate plan day {day}")
            if not isinstance(plan, DailyPlan):
                _fail(
                    f"plans[{index}] must contain a DailyPlan, got "
                    f"{type(plan).__name__}"
                )
            seen.add(day)
            entries.append(_PlanEntry(day=day, plan=plan))
        entries.sort(key=lambda entry: entry.day)
        if not isinstance(provenance, Mapping):
            _fail("provenance must be a mapping")
        window = _recording_window(provenance.get("recording_window"))
        expected_days = set(range(window["start_day"], window["end_day"] + 1))
        actual_days = {entry.day for entry in entries}
        missing = sorted(expected_days - actual_days)
        extra = sorted(actual_days - expected_days)
        if missing or extra:
            _fail(
                "plans must exactly cover every day in "
                f"provenance.recording_window; missing={missing}, extra={extra}"
            )
        return cls(provenance=provenance, entries=tuple(entries))

    @property
    def provenance(self) -> dict[str, JSONValue]:
        """A defensive JSON-safe copy of the explicit recording provenance."""
        return copy.deepcopy(self._provenance)

    @property
    def plans(self) -> tuple[tuple[int, DailyPlan], ...]:
        """Canonical ascending ``(absolute_day, DailyPlan)`` pairs."""
        return tuple((entry.day, entry.plan) for entry in self._entries)

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIXED_PLAN_TAPE_SCHEMA_VERSION,
            "provenance": copy.deepcopy(self._provenance),
            "plans": [
                {"day": entry.day, "plan": entry.plan.to_json_dict()}
                for entry in self._entries
            ],
        }

    def _compute_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._payload_dict())).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        """Return the versioned artifact including its content fingerprint."""
        document = self._payload_dict()
        document["artifact_sha256"] = self._artifact_sha256 if hasattr(
            self, "_artifact_sha256"
        ) else self._compute_sha256()
        return document

    def to_json(self) -> str:
        """Return deterministic compact JSON suitable for artifact comparison."""
        return _canonical_json_bytes(self.to_json_dict()).decode("utf-8")

    @classmethod
    def from_json_dict(cls, document: Any) -> "FixedPlanTape":
        """Parse and strictly validate one tape artifact document."""
        if not isinstance(document, Mapping):
            _fail("tape document must be a mapping")
        expected_keys = {"schema_version", "provenance", "plans", "artifact_sha256"}
        if set(document) != expected_keys:
            _fail(
                f"tape document fields must be exactly {sorted(expected_keys)}, "
                f"got {sorted(document)}"
            )
        if document["schema_version"] != FIXED_PLAN_TAPE_SCHEMA_VERSION:
            _fail(
                "unsupported fixed-plan tape schema_version: "
                f"{document['schema_version']!r}"
            )
        digest = document["artifact_sha256"]
        if not isinstance(digest, str):
            _fail("artifact_sha256 must be a string")
        raw_plans = document["plans"]
        if not isinstance(raw_plans, list):
            _fail("plans must be a list")

        pairs: list[tuple[int, DailyPlan]] = []
        seen: set[int] = set()
        for index, raw_entry in enumerate(raw_plans):
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"day", "plan"}:
                _fail(f"plans[{index}] must contain exactly day and plan")
            day = raw_entry["day"]
            if not _is_int(day):
                _fail(f"plans[{index}].day must be an integer, got {day!r}")
            if day in seen:
                _fail(f"duplicate plan day {day}")
            seen.add(day)
            pairs.append((day, _validate_plan_json(raw_entry["plan"],
                                                   label=f"plans[{index}].plan")))
        tape = cls.create(plans=pairs, provenance=document["provenance"])
        if tape.artifact_sha256 != digest:
            _fail(
                "artifact_sha256 mismatch: "
                f"recorded {digest!r}, computed {tape.artifact_sha256!r}"
            )
        return tape

    @classmethod
    def from_json(cls, text: str) -> "FixedPlanTape":
        try:
            document = json.loads(text)
        except (TypeError, json.JSONDecodeError) as error:
            raise PlanTapeError(f"invalid fixed-plan tape JSON: {error}") from error
        return cls.from_json_dict(document)

    def save(self, path: str | Path) -> Path:
        """Write deterministic UTF-8 JSON and return the written path."""
        output = Path(path)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FixedPlanTape":
        input_path = Path(path)
        try:
            text = input_path.read_text(encoding="utf-8")
        except OSError as error:
            raise PlanTapeError(f"cannot read fixed-plan tape {input_path}: {error}") from error
        return cls.from_json(text)


class FixedPlanTapeProvider:
    """Strict, stateless ``PlanProvider`` backed by one immutable tape."""

    def __init__(self, tape: FixedPlanTape) -> None:
        if not isinstance(tape, FixedPlanTape):
            raise PlanTapeError(
                f"FixedPlanTapeProvider requires a FixedPlanTape, got "
                f"{type(tape).__name__}"
            )
        self.tape = tape

    def daily_plan(
        self,
        obs: Mapping,
        seat: int,
        previous_execution: Mapping[str, int] | None = None,
    ) -> DailyPlan:
        if not _is_int(seat):
            _fail(f"seat must be an integer, got {seat!r}")
        if seat not in (0, 1):
            _fail(f"seat must be 0 or 1, got {seat}")
        encoded_seat = self.tape.provenance["seat"]
        if seat != encoded_seat:
            _fail(
                f"seat mismatch: tape records seat {encoded_seat}, "
                f"request received seat {seat}"
            )
        try:
            raw_day = obs["day"]
        except (KeyError, TypeError) as error:
            raise PlanTapeError("observation is missing required day") from error
        if isinstance(raw_day, bool):
            _fail(f"observation day must be integer-like, got {raw_day!r}")
        try:
            day = int(raw_day)
        except (TypeError, ValueError) as error:
            raise PlanTapeError(
                f"observation day must be integer-like, got {raw_day!r}"
            ) from error
        if isinstance(raw_day, float) and raw_day != day:
            _fail(f"observation day must be integral, got {raw_day!r}")

        window = self.tape.provenance["recording_window"]
        if not window["start_day"] <= day <= window["end_day"]:
            _fail(
                f"observation day {day} is outside recording window "
                f"{window['start_day']}..{window['end_day']}"
            )
        try:
            return self.tape._by_day[day]
        except KeyError as error:
            _fail(f"no fixed DailyPlan recorded for absolute game day {day}")
            raise AssertionError from error  # pragma: no cover
