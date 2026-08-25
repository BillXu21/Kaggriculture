"""Strict deterministic comparison of two fixed-plan runner artifacts.

The runner deliberately emits one arm at a time.  This module is the only
comparison stage: it validates the runner artifact digest and schema, proves
that all frozen inputs are equal, and then reports transparent per-day and
window deltas without producing a strategic score.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


__all__ = [
    "FIXED_PLAN_COMPARE_SCHEMA_VERSION",
    "ArtifactValidationError",
    "FixedInputMismatchError",
    "FixedPlanCompareError",
    "MultiDayFixedPlanComparison",
    "compare_multiday_fixed_plan",
]


FIXED_PLAN_COMPARE_SCHEMA_VERSION = 1
_RUNNER_SCHEMA_VERSION = 1


class FixedPlanCompareError(ValueError):
    """Base error for invalid or unsafe fixed-plan comparisons."""


class ArtifactValidationError(FixedPlanCompareError):
    """Raised when a source artifact is malformed or has a bad digest."""


class FixedInputMismatchError(FixedPlanCompareError):
    """Raised when two valid artifacts do not share frozen inputs."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"artifact contains a non-canonical JSON value: {error}"
        ) from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ArtifactValidationError(
        f"artifact value is not JSON-safe: {type(value).__name__}"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(_json_safe(value))).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same(left: Any, right: Any) -> bool:
    return _canonical_json_bytes(_json_safe(left)) == _canonical_json_bytes(
        _json_safe(right)
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(f"{label} must be an integer")
    return value


def _load_artifact(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            raw = path.read_text(encoding="utf-8")
            document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactValidationError(f"cannot read {label} artifact {path}: {error}") from error
    elif isinstance(value, Mapping):
        document = copy.deepcopy(dict(value))
    else:
        raise ArtifactValidationError(
            f"{label} artifact must be a path or mapping, got {type(value).__name__}"
        )
    if not isinstance(document, Mapping):
        raise ArtifactValidationError(f"{label} artifact root must be a mapping")
    result = dict(document)
    _validate_artifact(result, label)
    return result


def _validate_artifact(document: Mapping[str, Any], label: str) -> None:
    required = {
        "schema_version", "artifact_sha256", "executor_provenance",
        "tape_fingerprint", "tape", "replay", "seat", "window", "backend",
        "strategy_inputs", "days", "totals", "executor_diagnostics",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ArtifactValidationError(
            f"{label} artifact missing required fields: {', '.join(missing)}"
        )
    if document["schema_version"] != _RUNNER_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"{label} artifact schema_version must be {_RUNNER_SCHEMA_VERSION}, "
            f"got {document['schema_version']!r}"
        )
    digest = _string(document["artifact_sha256"], f"{label}.artifact_sha256")
    unsigned = {key: value for key, value in document.items() if key != "artifact_sha256"}
    expected = hashlib.sha256(_canonical_json_bytes(_json_safe(unsigned))).hexdigest()
    if digest != expected:
        raise ArtifactValidationError(
            f"{label} artifact_sha256 mismatch: declared {digest}, expected {expected}"
        )
    _string(document["executor_provenance"], f"{label}.executor_provenance")
    _string(document["tape_fingerprint"], f"{label}.tape_fingerprint")

    tape = _mapping(document["tape"], f"{label}.tape")
    _string(tape.get("fingerprint"), f"{label}.tape.fingerprint")
    if tape["fingerprint"] != document["tape_fingerprint"]:
        raise ArtifactValidationError(
            f"{label} tape fingerprint disagrees between top-level and tape object"
        )

    replay = _mapping(document["replay"], f"{label}.replay")
    for key in ("sha256", "configuration_sha256"):
        _string(replay.get(key), f"{label}.replay.{key}")
    _integer(replay.get("episode_id"), f"{label}.replay.episode_id")
    _integer(replay.get("seed"), f"{label}.replay.seed")
    _mapping(replay.get("configuration"), f"{label}.replay.configuration")

    seat = _integer(document["seat"], f"{label}.seat")
    if seat not in (0, 1):
        raise ArtifactValidationError(f"{label}.seat must be 0 or 1")

    window = _mapping(document["window"], f"{label}.window")
    for key in ("start_day", "end_day", "length", "turns_per_day", "turns"):
        _integer(window.get(key), f"{label}.window.{key}")
    if window["end_day"] - window["start_day"] + 1 != window["length"]:
        raise ArtifactValidationError(f"{label}.window day bounds disagree with length")
    if window["length"] < 1 or window["turns"] < 0:
        raise ArtifactValidationError(f"{label}.window has invalid length or turn count")
    if window["turns"] != window["length"] * window["turns_per_day"]:
        raise ArtifactValidationError(
            f"{label}.window.turns must equal length * turns_per_day"
        )

    backend = _mapping(document["backend"], f"{label}.backend")
    _string(backend.get("name"), f"{label}.backend.name")
    _mapping(backend.get("engine"), f"{label}.backend.engine")

    strategy = _mapping(document["strategy_inputs"], f"{label}.strategy_inputs")
    _string(strategy.get("opponent_trace_sha256"), f"{label}.strategy_inputs.opponent_trace_sha256")
    if strategy.get("fixed_plan_provider") is not True:
        raise ArtifactValidationError(
            f"{label}.strategy_inputs.fixed_plan_provider must be true"
        )
    if strategy.get("live_manager_invocations") != 0:
        raise ArtifactValidationError(
            f"{label}.strategy_inputs.live_manager_invocations must be zero"
        )

    days = document["days"]
    if not isinstance(days, Sequence) or isinstance(days, (str, bytes)):
        raise ArtifactValidationError(f"{label}.days must be a sequence")
    if len(days) != window["length"]:
        raise ArtifactValidationError(
            f"{label}.days length {len(days)} does not match window.length {window['length']}"
        )
    for offset, day in enumerate(days):
        item = _mapping(day, f"{label}.days[{offset}]")
        expected_day = window["start_day"] + offset
        if item.get("day") != expected_day:
            raise ArtifactValidationError(
                f"{label}.days[{offset}].day must be {expected_day}, got {item.get('day')!r}"
            )

    totals = _mapping(document["totals"], f"{label}.totals")
    if totals.get("turns") != window["turns"]:
        raise ArtifactValidationError(
            f"{label}.totals.turns must equal window.turns"
        )
    _mapping(document["executor_diagnostics"], f"{label}.executor_diagnostics")


def _identity(document: Mapping[str, Any]) -> dict[str, Any]:
    replay = document["replay"]
    window = document["window"]
    backend = document["backend"]
    strategy = document["strategy_inputs"]
    return {
        "tape_fingerprint": document["tape_fingerprint"],
        "replay_sha256": replay["sha256"],
        "replay_episode_id": replay["episode_id"],
        "seed": replay["seed"],
        "configuration_sha256": replay["configuration_sha256"],
        "configuration": replay["configuration"],
        "seat": document["seat"],
        "window": {
            key: window[key]
            for key in ("start_day", "end_day", "length", "turns_per_day", "turns")
        },
        "backend": backend["name"],
        "engine": backend["engine"],
        "backend_provenance": backend,
        "opponent_trace_sha256": strategy["opponent_trace_sha256"],
        "turn_count": {"window": window["turns"], "totals": document["totals"]["turns"]},
    }


def _check_frozen_inputs(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    left = _identity(baseline)
    right = _identity(candidate)
    mismatches = [
        key for key in left if not _same(left[key], right[key])
    ]
    if mismatches:
        raise FixedInputMismatchError(
            "fixed-input mismatch; refusing comparison: " + ", ".join(mismatches)
        )


def _value_delta(baseline: Any, candidate: Any) -> Any:
    if _is_number(baseline) and _is_number(candidate):
        return candidate - baseline
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        return {
            str(key): _value_delta(baseline.get(key), candidate.get(key))
            for key in sorted(set(baseline) | set(candidate), key=str)
        }
    return None


def _record(
    baseline: Any,
    candidate: Any,
    path: str,
    warnings: list[str],
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        warnings.append(f"nullable or unavailable metric: {path}")
    result: dict[str, Any] = {
        "baseline": _json_safe(baseline),
        "candidate": _json_safe(candidate),
        "delta": _json_safe(_value_delta(baseline, candidate)),
    }
    if isinstance(baseline, Sequence) and not isinstance(baseline, (str, bytes)) \
            and isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        result["count_delta"] = len(candidate) - len(baseline)
    return result


_CATEGORY_FIELDS: dict[str, tuple[str, ...]] = {
    "bank_wealth": (
        "cash_start", "cash_end", "cash_delta", "wealth_start", "wealth_end", "wealth_delta",
    ),
    "animal_loss_escape": (
        "animal_loss_evidence", "animal_escape_evidence",
        "animal_count_decreases_observed", "animal_count_decreases_without_stable_evidence",
    ),
    "weeds_crops": ("weeds_created", "crops_destroyed_observed", "weeds_start", "weeds_end"),
    "target_projection": ("plan_targets", "projection_changes"),
    "end_of_day_debt": ("eod_work_debt",),
    "pending_task_turns": ("pending_task_turns",),
    "action_turns": ("action_turn_counts",),
    "water_reason_split": ("water_interactions",),
    "harvested_units": ("harvested_units",),
    "feed_shortage_starvation": ("survival",),
    "hires_cost": ("hires", "previous_labor"),
    "unaffordable_orders": ("unaffordable_orders",),
}


def _day_value(day: Mapping[str, Any], field: str) -> Any:
    if field == "previous_labor":
        diagnostics = day.get("diagnostics")
        return diagnostics.get(field) if isinstance(diagnostics, Mapping) else None
    return day.get(field)


def _extract_day_metrics(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    day: int,
    warnings: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for category, fields in _CATEGORY_FIELDS.items():
        category_result: dict[str, Any] = {}
        for field in fields:
            category_result[field] = _record(
                _day_value(baseline, field),
                _day_value(candidate, field),
                f"day {day}.{field}",
                warnings,
            )
        metrics[category] = category_result
    return metrics


def _aggregate_value(values: list[Any], field: str) -> Any:
    available = [value for value in values if value is not None]
    if not available:
        return None
    if all(_is_number(value) for value in available) and len(available) == len(values):
        if field.endswith("_start"):
            return values[0]
        if field.endswith("_end"):
            return values[-1]
        return sum(available)
    if all(isinstance(value, Mapping) for value in available) and len(available) == len(values):
        keys = sorted({key for value in available for key in value}, key=str)
        return {
            str(key): _aggregate_value([value.get(key) for value in values], str(key))
            for key in keys
        }
    if all(isinstance(value, Sequence) and not isinstance(value, (str, bytes))
           for value in available) and len(available) == len(values):
        result: list[Any] = []
        for value in values:
            result.extend(value)
        return result
    if len(available) == len(values) and all(_same(value, available[0]) for value in available[1:]):
        return available[0]
    return None


def _window_metrics(
    baseline_days: Sequence[Mapping[str, Any]],
    candidate_days: Sequence[Mapping[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, fields in _CATEGORY_FIELDS.items():
        category_result: dict[str, Any] = {}
        for field in fields:
            baseline_value = _aggregate_value(
                [_day_value(day, field) for day in baseline_days], field
            )
            candidate_value = _aggregate_value(
                [_day_value(day, field) for day in candidate_days], field
            )
            category_result[field] = _record(
                baseline_value,
                candidate_value,
                f"window.{field}",
                warnings,
            )
        result[category] = category_result
    return result


def _count(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, Mapping):
        return len(value)
    if _is_number(value):
        return int(value)
    return None


def _sum_day_field(document: Mapping[str, Any], field: str) -> int | float | None:
    values = [day.get(field) for day in document["days"]]
    if not values or any(not _is_number(value) for value in values):
        return None
    return sum(values)


def _sum_nested(document: Mapping[str, Any], field: str, nested: str) -> int | float | None:
    values = [
        (day.get(field) or {}).get(nested)
        if isinstance(day.get(field), Mapping) else None
        for day in document["days"]
    ]
    if not values or any(not _is_number(value) for value in values):
        return None
    return sum(values)


def _divergence_present(document: Mapping[str, Any]) -> bool:
    def visit(node: Any, parent_key: str = "") -> bool:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lowered = str(key).lower()
                if "fallback" in lowered:
                    if isinstance(value, bool) and value:
                        return True
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
                        return True
                    if isinstance(value, Mapping) and value:
                        return True
                    if isinstance(value, str) and value:
                        return True
                if "opening" in lowered and any(token in lowered for token in ("diverg", "mismatch", "error")):
                    if value is True or (isinstance(value, (str, Sequence, Mapping)) and value):
                        return True
                if visit(value, lowered):
                    return True
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            return any(visit(item, parent_key) for item in node)
        return False

    return visit(document)


def _safety_flags(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_escape = sum(_count(day.get("animal_escape_evidence")) or 0 for day in baseline["days"])
    candidate_escape = sum(_count(day.get("animal_escape_evidence")) or 0 for day in candidate["days"])
    baseline_loss = sum(_count(day.get("animal_loss_evidence")) or 0 for day in baseline["days"])
    candidate_loss = sum(_count(day.get("animal_loss_evidence")) or 0 for day in candidate["days"])
    baseline_survival_debt = sum(
        _count((day.get("eod_work_debt") or {}).get("survival")) or 0
        if isinstance(day.get("eod_work_debt"), Mapping) else 0
        for day in baseline["days"]
    )
    candidate_survival_debt = sum(
        _count((day.get("eod_work_debt") or {}).get("survival")) or 0
        if isinstance(day.get("eod_work_debt"), Mapping) else 0
        for day in candidate["days"]
    )
    baseline_weeds = _sum_day_field(baseline, "weeds_created")
    candidate_weeds = _sum_day_field(candidate, "weeds_created")
    baseline_destroyed = _sum_day_field(baseline, "crops_destroyed_observed")
    candidate_destroyed = _sum_day_field(candidate, "crops_destroyed_observed")
    baseline_cash = baseline["totals"].get("cash_delta")
    candidate_cash = candidate["totals"].get("cash_delta")
    baseline_wealth = baseline["totals"].get("wealth_delta")
    candidate_wealth = candidate["totals"].get("wealth_delta")
    efficiency_changed = not _same(
        _window_metrics(baseline["days"], baseline["days"], [])["action_turns"],
        _window_metrics(candidate["days"], candidate["days"], [])["action_turns"],
    )
    return [
        {
            "code": "fallback_or_opening_divergence",
            "priority": 1,
            "active": _divergence_present(baseline) or _divergence_present(candidate),
            "fails_promotion": True,
            "details": {
                "baseline_present": _divergence_present(baseline),
                "candidate_present": _divergence_present(candidate),
            },
        },
        {
            "code": "avoidable_animal_escape_or_loss",
            "priority": 2,
            "active": candidate_escape > baseline_escape or candidate_loss > baseline_loss,
            "fails_promotion": True,
            "details": {
                "baseline_escape": baseline_escape, "candidate_escape": candidate_escape,
                "baseline_loss": baseline_loss, "candidate_loss": candidate_loss,
            },
        },
        {
            "code": "survival_debt_increase",
            "priority": 3,
            "active": candidate_survival_debt > baseline_survival_debt,
            "fails_promotion": True,
            "details": {
                "baseline": baseline_survival_debt, "candidate": candidate_survival_debt,
            },
        },
        {
            "code": "maintained_crop_weed_increase",
            "priority": 4,
            "active": (
                candidate_weeds is not None and baseline_weeds is not None
                and candidate_weeds > baseline_weeds
            ) or (
                candidate_destroyed is not None and baseline_destroyed is not None
                and candidate_destroyed > baseline_destroyed
            ),
            "fails_promotion": True,
            "details": {
                "baseline_weeds_created": baseline_weeds,
                "candidate_weeds_created": candidate_weeds,
                "baseline_crops_destroyed": baseline_destroyed,
                "candidate_crops_destroyed": candidate_destroyed,
            },
        },
        {
            "code": "fixed_input_mismatch",
            "priority": 5,
            "active": False,
            "fails_promotion": True,
            "details": {"comparison_reached": True},
        },
        {
            "code": "descriptive_bank_wealth_delta",
            "priority": 6,
            "active": (
                _is_number(candidate_cash) and _is_number(baseline_cash)
                and candidate_cash != baseline_cash
            ) or (
                _is_number(candidate_wealth) and _is_number(baseline_wealth)
                and candidate_wealth != baseline_wealth
            ),
            "fails_promotion": False,
            "details": {
                "baseline_cash_delta": baseline_cash, "candidate_cash_delta": candidate_cash,
                "baseline_wealth_delta": baseline_wealth, "candidate_wealth_delta": candidate_wealth,
            },
        },
        {
            "code": "descriptive_efficiency_delta",
            "priority": 6,
            "active": efficiency_changed,
            "fails_promotion": False,
            "details": {"strategic_score": None, "reason": "descriptive_only"},
        },
    ]


def _warnings(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metric_warnings: list[str],
    same_executor_allowed: bool,
) -> list[str]:
    result = list(dict.fromkeys(metric_warnings))
    for label, document in (("baseline", baseline), ("candidate", candidate)):
        diagnostics = document["executor_diagnostics"]
        illegal = diagnostics.get("illegal_actions")
        if isinstance(illegal, Mapping) and illegal.get("available") is False:
            result.append(f"{label}: illegal-action metric is unavailable: {illegal.get('reason')}")
        for offset, day in enumerate(document["days"]):
            harvested = day.get("harvested_units")
            if isinstance(harvested, Mapping) and (
                harvested.get("unmeasured_actions", 0) or harvested.get("unavailable_reason")
            ):
                result.append(
                    f"{label} day {day.get('day', offset)}: harvested units include unmeasured/proxy actions"
                )
    if same_executor_allowed:
        result.append("same executor provenance was allowed only by the explicit test-only flag")
    result.append("manager debt is reported descriptively and never independently fails promotion")
    return list(dict.fromkeys(result))


def _source_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    identity = _identity(document)
    return {
        "artifact_sha256": document["artifact_sha256"],
        "label": document.get("label"),
        "executor_provenance": document["executor_provenance"],
        "frozen_identity": identity,
    }


@dataclass(frozen=True)
class MultiDayFixedPlanComparison:
    """Deterministic versioned comparison artifact."""

    document: dict[str, Any]

    @property
    def artifact_sha256(self) -> str:
        return str(self.document["artifact_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.document)

    def to_json(self) -> str:
        return _canonical_json_bytes(self.document).decode("utf-8")

    def save(self, path: str | os.PathLike[str]) -> Path:
        output = Path(path)
        try:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(self.to_json() + "\n")
        except FileExistsError as error:
            raise FixedPlanCompareError(f"refusing to overwrite existing output: {output}") from error
        return output


def compare_multiday_fixed_plan(
    baseline_artifact: Any,
    candidate_artifact: Any,
    *,
    output_path: str | os.PathLike[str] | None = None,
    label: str,
    allow_same_executor_for_test: bool = False,
) -> MultiDayFixedPlanComparison:
    """Compare two validated runner artifacts under identical frozen inputs."""
    if not isinstance(label, str) or not label:
        raise FixedPlanCompareError("label must be a non-empty string")
    baseline = _load_artifact(baseline_artifact, "baseline")
    candidate = _load_artifact(candidate_artifact, "candidate")
    _check_frozen_inputs(baseline, candidate)
    same_executor = baseline["executor_provenance"] == candidate["executor_provenance"]
    if same_executor and not allow_same_executor_for_test:
        raise FixedPlanCompareError(
            "executor provenance is identical; use distinct revisions or the explicit "
            "allow_same_executor_for_test test-only flag"
        )

    metric_warnings: list[str] = []
    baseline_days = baseline["days"]
    candidate_days = candidate["days"]
    days: list[dict[str, Any]] = []
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        day = int(baseline_day["day"])
        days.append({
            "day": day,
            "metrics": _extract_day_metrics(
                baseline_day, candidate_day, day, metric_warnings
            ),
        })
    window = {
        "window": _window_metrics(baseline_days, candidate_days, metric_warnings),
        "turns": _record(
            baseline["window"]["turns"], candidate["window"]["turns"],
            "window.turns", metric_warnings,
        ),
        "days": _record(
            baseline["window"]["length"], candidate["window"]["length"],
            "window.days", metric_warnings,
        ),
    }
    safety_flags = _safety_flags(baseline, candidate)
    active_failures = [flag for flag in safety_flags if flag["active"] and flag["fails_promotion"]]
    warnings = _warnings(baseline, candidate, metric_warnings, same_executor)
    document: dict[str, Any] = {
        "schema_version": FIXED_PLAN_COMPARE_SCHEMA_VERSION,
        "label": label,
        "sources": {
            "baseline": _source_summary(baseline),
            "candidate": _source_summary(candidate),
        },
        "frozen_inputs": {
            "identical": True,
            "identity": _identity(baseline),
        },
        "executor_provenance": {
            "baseline": baseline["executor_provenance"],
            "candidate": candidate["executor_provenance"],
            "distinct": not same_executor,
            "same_executor_allowed_for_test": bool(allow_same_executor_for_test),
        },
        "days": days,
        "window": window,
        "safety": {
            "promotion_safe": not active_failures,
            "primary_failure": active_failures[0]["code"] if active_failures else None,
            "flags": safety_flags,
            "manager_debt_is_not_a_failure": True,
        },
        "warnings": warnings,
    }
    document["artifact_sha256"] = hashlib.sha256(
        _canonical_json_bytes(document)
    ).hexdigest()
    result = MultiDayFixedPlanComparison(_json_safe(document))
    if output_path is not None:
        result.save(output_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--allow-same-executor-for-test", action="store_true",
        help="test-only escape hatch for synthetic same-provenance comparisons",
    )
    args = parser.parse_args(argv)
    result = compare_multiday_fixed_plan(
        args.baseline,
        args.candidate,
        output_path=args.output,
        label=args.label,
        allow_same_executor_for_test=args.allow_same_executor_for_test,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
