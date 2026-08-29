"""Deterministic evaluation summaries and explainable promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


ORIENTATIONS = ("candidate_vs_frozen", "frozen_vs_candidate")


@dataclass(frozen=True)
class PromotionConfig:
    """Thresholds for snapshot promotion; economic floors are opt-in."""

    min_w_minus_l: int = 6
    min_mean_margin: float = 0.0
    min_median_margin: float = 0.0
    candidate_mean_bank_floor: float | None = None
    candidate_median_bank_floor: float | None = None
    fail_on_opening_diagnostics: bool = False
    fail_on_executor_diagnostics: bool = False


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    conditions: tuple[dict[str, Any], ...]
    failed_reasons: tuple[str, ...]
    policy_identity: Any = None
    opponent_identity: Any = None
    eval_seed_set: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conditions"] = list(self.conditions)
        payload["failed_reasons"] = list(self.failed_reasons)
        return payload


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear percentile using the inclusive [0, n-1] rank convention."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 12)


def _distribution(values: Sequence[float], *, fractions: bool) -> dict[str, Any]:
    result = {
        "mean": _mean(values),
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    if fractions:
        result.update({
            "frac_below_1k": (sum(value < 1_000 for value in values)
                               / len(values)) if values else None,
            "frac_below_10k": (sum(value < 10_000 for value in values)
                                / len(values)) if values else None,
        })
    return result


def _runtime_failures(record: Any, path: str = "") -> list[dict[str, Any]]:
    """Find explicit runtime errors without treating guard delegation as fatal."""
    failures: list[dict[str, Any]] = []
    if isinstance(record, Mapping):
        for key, value in record.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in {"fallback_errors", "runtime_errors"} and value:
                failures.append({"path": child_path, "detail": value})
            elif key in {"runtime_error", "exception"} and value:
                failures.append({"path": child_path, "detail": value})
            else:
                failures.extend(_runtime_failures(value, child_path))
    elif isinstance(record, (list, tuple)):
        for index, value in enumerate(record):
            failures.extend(_runtime_failures(value, f"{path}[{index}]"))
    return failures


def _compact_executor_diagnostic(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = {"seat": record.get("seat")}
    for key in ("fallback_errors", "runtime_errors", "runtime_error",
                "exception", "illegal_actions", "provider_diagnostics"):
        if key in record:
            compact[key] = record[key]
    return compact


def summarize_evaluation(
    results: Sequence[Any],
    *,
    expected_seeds: Sequence[int] | None = None,
    expected_orientations: Sequence[str] = ORIENTATIONS,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate candidate-perspective outcomes and evaluation health."""
    wins = losses = ties = 0
    margins: list[float] = []
    candidate_banks: list[float] = []
    opponent_banks: list[float] = []
    per_orientation: dict[str, dict[str, int]] = {}
    per_seat: dict[str, dict[str, int]] = {}
    fatal_anomalies: list[dict[str, Any]] = []
    opening_diagnostics: list[dict[str, Any]] = []
    executor_diagnostics: list[dict[str, Any]] = []
    seed_margins: list[tuple[int, str, float]] = []
    seen: dict[tuple[int, str], int] = {}

    for result in results:
        seed = int(result.seed)
        composition = str(result.composition)
        key = (seed, composition)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "duplicate_result", "detail": seen[key],
            })
        if composition not in ORIENTATIONS:
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "invalid_orientation", "detail": composition,
            })
            continue

        statuses = [str(status) for status in result.statuses]
        if statuses != ["DONE", "DONE"]:
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "statuses", "detail": statuses,
            })
        if statuses == ["DONE", "DONE"] \
                and not bool(getattr(result, "terminated", True)):
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "incomplete_trajectory", "detail": "not terminated",
            })
        trace_digest = getattr(result, "trace_digest", None)
        if not trace_digest:
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "invalid_trace", "detail": "empty trace digest",
            })

        for record in getattr(result, "opening_diagnostics", None) or []:
            opening_diagnostics.append({
                "seed": seed, "orientation": composition, "detail": record,
            })
            for failure in _runtime_failures(record):
                fatal_anomalies.append({
                    "seed": seed, "orientation": composition,
                    "kind": "runtime_fallback", "source": "opening",
                    "detail": failure,
                })
        for record in getattr(result, "executor_diagnostics", None) or []:
            compact = _compact_executor_diagnostic(record)
            executor_diagnostics.append({
                "seed": seed, "orientation": composition, "detail": compact,
            })
            for failure in _runtime_failures(record):
                fatal_anomalies.append({
                    "seed": seed, "orientation": composition,
                    "kind": "runtime_fallback", "source": "executor",
                    "detail": failure,
                })

        try:
            banks = [float(bank) for bank in result.final_banks]
            if len(banks) != 2 or not all(math.isfinite(bank) for bank in banks):
                raise ValueError("final_banks must contain two finite values")
        except (TypeError, ValueError) as exc:
            fatal_anomalies.append({
                "seed": seed, "orientation": composition,
                "kind": "invalid_result", "detail": str(exc),
            })
            continue

        candidate_seat = 0 if composition == "candidate_vs_frozen" else 1
        margin = banks[candidate_seat] - banks[1 - candidate_seat]
        margins.append(margin)
        candidate_banks.append(banks[candidate_seat])
        opponent_banks.append(banks[1 - candidate_seat])
        outcome = "W" if margin > 0 else "L" if margin < 0 else "T"
        wins += outcome == "W"
        losses += outcome == "L"
        ties += outcome == "T"
        for bucket in (
            per_orientation.setdefault(
                composition, {"games": 0, "W": 0, "L": 0, "T": 0}),
            per_seat.setdefault(
                str(candidate_seat), {"games": 0, "W": 0, "L": 0, "T": 0}),
        ):
            bucket["games"] += 1
            bucket[outcome] += 1
        seed_margins.append((seed, composition, margin))

    if expected_seeds is not None:
        expected = {(int(seed), str(orientation))
                    for seed in expected_seeds
                    for orientation in expected_orientations}
        for seed, orientation in sorted(expected - set(seen)):
            fatal_anomalies.append({
                "seed": seed, "orientation": orientation,
                "kind": "missing_result", "detail": "expected result absent",
            })
        for seed, orientation in sorted(set(seen) - expected):
            fatal_anomalies.append({
                "seed": seed, "orientation": orientation,
                "kind": "unexpected_result", "detail": "result not in panel",
            })

    candidate = _distribution(candidate_banks, fractions=True)
    opponent = _distribution(opponent_banks, fractions=True)
    margin_summary = _distribution(margins, fractions=False)
    worst = sorted(seed_margins, key=lambda entry: (entry[2], entry[0], entry[1]))[:5]
    health = {
        "fatal_anomalies": fatal_anomalies,
        "opening_diagnostics": opening_diagnostics,
        "executor_diagnostics": executor_diagnostics,
        "warnings": [],
    }
    return {
        "evaluation_schema_version": 2,
        "games": len(results),
        "completed_games": len(margins),
        "wlt": {"W": wins, "L": losses, "T": ties},
        "win_rate": (wins / len(margins)) if margins else None,
        "paired_margins": margins,
        "margins": margin_summary,
        "median_margin": margin_summary["median"],
        "mean_margin": margin_summary["mean"],
        "banks": {
            "candidate": candidate,
            "opponent": opponent,
            "candidate_median": candidate["median"],
            "candidate_mean": candidate["mean"],
            "candidate_p10": candidate["p10"],
            "candidate_min": candidate["min"],
            "candidate_max": candidate["max"],
            "candidate_frac_below_1k": candidate["frac_below_1k"],
            "candidate_frac_below_10k": candidate["frac_below_10k"],
            "opponent_median": opponent["median"],
            "opponent_mean": opponent["mean"],
            "opponent_p10": opponent["p10"],
            "opponent_min": opponent["min"],
            "opponent_max": opponent["max"],
            "opponent_frac_below_1k": opponent["frac_below_1k"],
            "opponent_frac_below_10k": opponent["frac_below_10k"],
        },
        "margin_p10": margin_summary["p10"],
        "margin_min": margin_summary["min"],
        "margin_max": margin_summary["max"],
        "per_orientation": per_orientation,
        "seat_splits": per_seat,
        **health,
        "anomalies": fatal_anomalies,
        "health": health,
        "provenance": dict(provenance or {}),
        "worst_seeds": [
            {"seed": seed, "orientation": orientation, "margin": margin}
            for seed, orientation, margin in worst
        ],
    }


def evaluate_promotion(
    summary: Mapping[str, Any],
    config: PromotionConfig = PromotionConfig(),
) -> PromotionDecision:
    """Apply all configured gates and return every exact rejection reason."""
    wlt = summary.get("wlt", {})
    w_minus_l = int(wlt.get("W", 0)) - int(wlt.get("L", 0))
    mean_margin = summary.get("margins", {}).get(
        "mean", summary.get("mean_margin"))
    median_margin = summary.get("margins", {}).get(
        "median", summary.get("median_margin"))
    fatal_count = len(summary.get("fatal_anomalies", summary.get("anomalies", [])))
    conditions: list[dict[str, Any]] = []
    reasons: list[str] = []

    def add(name: str, observed: Any, operator: str, threshold: Any,
            passed: bool, reason: str) -> None:
        conditions.append({
            "name": name, "passed": passed, "observed": observed,
            "operator": operator, "threshold": threshold,
        })
        if not passed:
            reasons.append(reason)

    add("w_minus_l", w_minus_l, ">=", config.min_w_minus_l,
        w_minus_l >= config.min_w_minus_l,
        f"w_minus_l {w_minus_l} < {config.min_w_minus_l}")
    add("mean_margin", mean_margin, ">", config.min_mean_margin,
        mean_margin is not None and mean_margin > config.min_mean_margin,
        ("mean_margin unavailable" if mean_margin is None else
         f"mean_margin {mean_margin} <= {config.min_mean_margin}"))
    add("median_margin", median_margin, ">=", config.min_median_margin,
        median_margin is not None and median_margin >= config.min_median_margin,
        ("median_margin unavailable" if median_margin is None else
         f"median_margin {median_margin} < {config.min_median_margin}"))
    add("fatal_anomalies", fatal_count, "==", 0, fatal_count == 0,
        f"fatal_anomalies={fatal_count}")

    candidate = summary.get("banks", {}).get("candidate", {})
    for name, metric, floor in (
        ("candidate_mean_bank", "mean", config.candidate_mean_bank_floor),
        ("candidate_median_bank", "median", config.candidate_median_bank_floor),
    ):
        if floor is not None:
            observed = candidate.get(metric)
            add(name, observed, ">=", floor,
                observed is not None and observed >= floor,
                (f"{name} unavailable" if observed is None else
                 f"{name} {observed} < {floor}"))
    for name, enabled in (
        ("opening_diagnostics", config.fail_on_opening_diagnostics),
        ("executor_diagnostics", config.fail_on_executor_diagnostics),
    ):
        if enabled:
            count = len(summary.get(name, []))
            add(name, count, "==", 0, count == 0, f"{name}={count}")

    provenance = summary.get("provenance", {})
    return PromotionDecision(
        passed=not reasons,
        conditions=tuple(conditions),
        failed_reasons=tuple(reasons),
        policy_identity=provenance.get("candidate_identity"),
        opponent_identity=provenance.get("opponent_identity"),
        eval_seed_set=provenance.get("seed_set"),
    )


def format_promotion_result(
    summary: Mapping[str, Any], decision: PromotionDecision,
) -> str:
    """Compact long-run log line with no unexplained HOLD result."""
    wlt = summary["wlt"]
    provenance = summary.get("provenance", {})
    gate = "PASS" if decision.passed else "HOLD"
    return (
        f"EVAL {provenance.get('seed_set', 'unknown')} "
        f"{wlt['W']}-{wlt['L']}-{wlt['T']} "
        f"margin_mean={summary['mean_margin']} "
        f"bank_mean={summary['banks']['candidate_mean']} "
        f"gate={gate} reasons={list(decision.failed_reasons)!r}"
    )
