"""Audit CARE/FERTILIZE intent and completion evidence from a panel artifact.

This module is deliberately a postprocessor.  It does not execute the agent or
alter executor decisions.  The panel's observed completion fields are retained
as state observations; they are not promoted to accepted-action counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SPECIES = ("GOOSE", "COW", "SHEEP")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
FAMILIES = {
    "CARE": ("care_by_animal", "care", "care_completed_observed", SPECIES),
    "FERTILIZE": (
        "fertilizer_by_crop",
        "fertilizer",
        "fertilizer_completed_observed",
        CROPS,
    ),
}
MATERIAL_MIN_UNITS = 1


class AuditError(ValueError):
    """Raised when the input is not the requested panel or is not auditable."""


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{name} must be an integer, got {value!r}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{name} must be an object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assignment_counts(record: Mapping[str, Any]) -> dict[str, int]:
    """Count interaction assignments, never movement/pickup claims."""
    counts: dict[str, int] = defaultdict(int)
    for trace in record.get("turn_trace") or []:
        if not isinstance(trace, Mapping):
            continue
        for assignment in trace.get("assignments") or []:
            if not isinstance(assignment, Mapping):
                continue
            family = assignment.get("op_family")
            key = assignment.get("task_key")
            if family not in FAMILIES or not isinstance(key, str):
                continue
            prefix = f"{family}:"
            if key.startswith(prefix):
                entity = key[len(prefix):].split(":", 1)[0]
                counts[entity] += 1
    return dict(counts)


def _related_pending(record: Mapping[str, Any], family: str, entity: str) -> list[str]:
    prefix = f"{family}:{entity}:"
    return sorted(
        key for key in (record.get("pending_task_turns") or {})
        if isinstance(key, str) and key.startswith(prefix)
    )


def _related_unaffordable(record: Mapping[str, Any], family: str) -> list[str]:
    """Return explicit failed-buy evidence, never infer it from a shortfall."""
    result: list[str] = []
    for item in record.get("unaffordable_market_orders") or []:
        if isinstance(item, Mapping) and isinstance(item.get("task"), str):
            task = item["task"]
            if family == "FERTILIZE" and task == "BUY_PRODUCT:FERTILIZER":
                result.append(task)
    for trace in record.get("turn_trace") or []:
        if not isinstance(trace, Mapping):
            continue
        market = trace.get("market") or {}
        if not isinstance(market, Mapping):
            continue
        for category in ("survival", "expansion"):
            values = market.get(category) or {}
            if not isinstance(values, Mapping):
                continue
            for task in values.get("unaffordable_keys") or []:
                if family == "FERTILIZE" and task == "BUY_PRODUCT:FERTILIZER":
                    result.append(task)
    return sorted(set(result))


def _classification(
    *,
    family: str,
    entity: str,
    requested: int,
    feasible: int,
    observed: int,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    shortfall = requested - observed
    if requested <= 0 or shortfall <= 0:
        return None
    projection_shortfall = requested - feasible
    if projection_shortfall > 0:
        return {
            "classification": "manager requested infeasible work",
            "evidence": {
                "projection_requested": requested,
                "projection_feasible": feasible,
                "projection_shortfall": projection_shortfall,
            },
        }
    failed_buy = _related_unaffordable(record, family)
    if failed_buy:
        return {
            "classification": "missing inventory/failed buy",
            "evidence": {"unaffordable_market_tasks": failed_buy},
        }
    pending = _related_pending(record, family, entity)
    return {
        "classification": "unresolved",
        "evidence": {
            "pending_task_keys": pending,
            "missing_signals": [
                "per-action engine acceptance/failure",
                "daily post-action board state before day reset",
                "complete unassigned reasons for this task family",
            ],
            "illegal_actions": "engine validity unavailable in panel schema",
        },
    }


def _validate_panel(
    document: Mapping[str, Any],
    *,
    expected_repo_sha: str | None,
    expected_checkpoint_sha256: str | None,
    expected_seeds: Sequence[int] | None,
    expected_seats: Sequence[int] | None,
    expected_backend: str,
    expected_opening: str,
    expected_day_range: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if document.get("schema_version") != 2:
        raise AuditError(f"expected panel schema_version 2, got {document.get('schema_version')!r}")
    if document.get("artifact_type") != "executor_v07_full_game_evaluator":
        raise AuditError(f"unexpected artifact_type: {document.get('artifact_type')!r}")
    provenance = _mapping(document.get("source_provenance"), "source_provenance")
    checkpoint = _mapping(document.get("checkpoint"), "checkpoint")
    request = _mapping(document.get("request"), "request")
    if expected_repo_sha is not None and provenance.get("repo_sha") != expected_repo_sha:
        raise AuditError("panel source revision does not match expected revision")
    if expected_checkpoint_sha256 is not None and checkpoint.get("sha256") != expected_checkpoint_sha256:
        raise AuditError("panel checkpoint hash does not match expected checkpoint")
    if request.get("backend") != expected_backend:
        raise AuditError("panel backend does not match expected backend")
    if request.get("opening") != expected_opening:
        raise AuditError("panel opening does not match expected opening")
    opponent = _mapping(document.get("opponent"), "opponent")
    if opponent.get("identity") != "PASS":
        raise AuditError("panel opponent is not PASS")
    seeds = list(request.get("seeds") or [])
    seats = list(request.get("seats") or [])
    if expected_seeds is not None and seeds != list(expected_seeds):
        raise AuditError(f"panel seeds do not match expected selection: {seeds!r}")
    if expected_seats is not None and seats != list(expected_seats):
        raise AuditError(f"panel seats do not match expected selection: {seats!r}")
    games = document.get("games")
    if not isinstance(games, list):
        raise AuditError("panel games must be a list")
    expected_ids = [(seed, seat) for seed in seeds for seat in seats]
    actual_ids = [(g.get("seed"), g.get("seat")) for g in games if isinstance(g, Mapping)]
    if actual_ids != expected_ids:
        raise AuditError(f"panel game Cartesian product mismatch: {actual_ids!r}")
    anomalies: list[dict[str, Any]] = []
    for game in games:
        if not isinstance(game, Mapping):
            raise AuditError("panel game must be an object")
        if game.get("status") != "complete" or game.get("transitions") != 719:
            anomalies.append({"seed": game.get("seed"), "seat": game.get("seat"), "kind": "game_status", "value": {"status": game.get("status"), "transitions": game.get("transitions")}})
        final = _mapping(game.get("final"), "game.final")
        if final.get("statuses") != ["DONE", "DONE"]:
            anomalies.append({"seed": game.get("seed"), "seat": game.get("seat"), "kind": "engine_status", "value": final.get("statuses")})
        diagnostics = _mapping(game.get("executor_diagnostics"), "executor_diagnostics")
        if diagnostics.get("fallback_errors"):
            anomalies.append({"seed": game.get("seed"), "seat": game.get("seat"), "kind": "fallback_errors", "value": diagnostics["fallback_errors"]})
        illegal = diagnostics.get("illegal_actions") or {}
        if illegal.get("available") is not True:
            anomalies.append({"seed": game.get("seed"), "seat": game.get("seat"), "kind": "illegal_action_signal_unavailable", "value": illegal.get("reason")})
        days = _mapping(diagnostics.get("days"), "executor_diagnostics.days")
        if expected_day_range is not None:
            expected_days = list(range(expected_day_range[0], expected_day_range[1] + 1))
            actual_days = sorted(int(day) for day in days)
            if actual_days != expected_days:
                raise AuditError(f"day coverage mismatch for {game.get('seed')}/{game.get('seat')}: {actual_days!r}")
        for day, record in days.items():
            if not isinstance(record, Mapping):
                raise AuditError(f"day record {day} is not an object")
            if record.get("errors"):
                anomalies.append({"seed": game.get("seed"), "seat": game.get("seat"), "day": day, "kind": "day_errors", "value": record["errors"]})
    return anomalies


def build_audit(
    document: Mapping[str, Any],
    *,
    artifact_sha256: str | None = None,
    expected_repo_sha: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_seats: Sequence[int] | None = None,
    expected_backend: str = "fast",
    expected_opening: str = "standard_mixed",
    expected_day_range: tuple[int, int] | None = None,
    material_min_units: int = MATERIAL_MIN_UNITS,
) -> dict[str, Any]:
    if material_min_units < 1:
        raise AuditError("material_min_units must be >= 1")
    anomalies = _validate_panel(
        document,
        expected_repo_sha=expected_repo_sha,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_seeds=expected_seeds,
        expected_seats=expected_seats,
        expected_backend=expected_backend,
        expected_opening=expected_opening,
        expected_day_range=expected_day_range,
    )
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, dict[str, int]]] = {family: {} for family in FAMILIES}
    game_summaries: list[dict[str, Any]] = []
    runtime_signals = {
        "day_records": 0,
        "fallback_errors": 0,
        "day_errors": 0,
        "unresolved_generator_entries": 0,
        "unaffordable_market_orders": 0,
        "prior_debt_suppression_days": 0,
        "current_suppression_days": 0,
    }
    for game in document["games"]:
        diagnostics = _mapping(game["executor_diagnostics"], "executor_diagnostics")
        runtime_signals["fallback_errors"] += len(diagnostics.get("fallback_errors") or [])
        game_totals = {
            family: {
                "requested": 0,
                "feasible_projected": 0,
                "eligible": 0,
                "submitted_assigned": 0,
                "observed_completed_state_count": 0,
                "raw_shortfall_requested_minus_observed": 0,
                "requested_positive_rows": 0,
                "zero_completion_rows": 0,
                "material_shortfall_rows": 0,
            }
            for family in FAMILIES
        }
        zero_count = 0
        material_count = 0
        days = _mapping(diagnostics["days"], "executor_diagnostics.days")
        runtime_signals["day_records"] += len(days)
        for day_key, record_value in sorted(days.items(), key=lambda item: int(item[0])):
            record = _mapping(record_value, f"day {day_key}")
            runtime_signals["day_errors"] += len(record.get("errors") or [])
            runtime_signals["unresolved_generator_entries"] += len(record.get("unresolved_generator") or [])
            runtime_signals["unaffordable_market_orders"] += len(record.get("unaffordable_market_orders") or [])
            if record.get("next_day_expansion_suppressed"):
                runtime_signals["prior_debt_suppression_days"] += 1
            if (record.get("survival") or {}).get("expansion_suppressed_current"):
                runtime_signals["current_suppression_days"] += 1
            requested_plans = _mapping(record.get("requested"), "requested")
            feasible_plans = _mapping(record.get("feasible"), "feasible")
            projection = _mapping(record.get("projection_changes"), "projection_changes")
            assignments = _assignment_counts(record)
            for family, (plan_key, projection_key, observed_key, entities) in FAMILIES.items():
                requested_map = _mapping(requested_plans.get(plan_key), f"requested.{plan_key}")
                feasible_map = _mapping(feasible_plans.get(plan_key), f"feasible.{plan_key}")
                projection_map = _mapping(projection.get(projection_key), f"projection_changes.{projection_key}")
                observed_map = _mapping(record.get(observed_key), observed_key)
                for entity in entities:
                    requested = _int(requested_map.get(entity, 0), f"{family}.{entity}.requested")
                    feasible = _int(feasible_map.get(entity, 0), f"{family}.{entity}.feasible")
                    detail_value = projection_map.get(entity, {})
                    detail = _mapping(detail_value, f"{family}.{entity}.projection")
                    eligible = _int(detail.get("eligible", 0), f"{family}.{entity}.eligible")
                    observed = _int(observed_map.get(entity, 0), f"{family}.{entity}.observed")
                    assigned = assignments.get(entity, 0)
                    shortfall = requested - observed
                    classification = _classification(
                        family=family,
                        entity=entity,
                        requested=requested,
                        feasible=feasible,
                        observed=observed,
                        record=record,
                    )
                    row = {
                        "seed": game["seed"],
                        "seat": game["seat"],
                        "day": int(day_key),
                        "family": family,
                        "entity": entity,
                        "requested": requested,
                        "eligible": eligible,
                        "feasible_projected": feasible,
                        "submitted_assigned": assigned,
                        "observed_completed": observed,
                        "raw_shortfall_requested_minus_observed": shortfall,
                        "zero_completion": requested > 0 and observed == 0,
                        "material_shortfall": shortfall >= material_min_units,
                        "classification": classification,
                        "observation_comparison": "state count, not accepted action count",
                    }
                    rows.append(row)
                    bucket = totals[family].setdefault(entity, {"requested": 0, "eligible": 0, "feasible_projected": 0, "submitted_assigned": 0, "observed_completed_state_count": 0, "raw_shortfall_requested_minus_observed": 0, "requested_positive_rows": 0, "zero_completion_rows": 0, "material_shortfall_rows": 0})
                    for key, value in (("requested", requested), ("eligible", eligible), ("feasible_projected", feasible), ("submitted_assigned", assigned), ("observed_completed_state_count", observed), ("raw_shortfall_requested_minus_observed", shortfall)):
                        bucket[key] += value
                        game_totals[family][key] += value
                    if requested > 0:
                        bucket["requested_positive_rows"] += 1
                        game_totals[family]["requested_positive_rows"] += 1
                    if row["zero_completion"]:
                        bucket["zero_completion_rows"] += 1
                        game_totals[family]["zero_completion_rows"] += 1
                        zero_count += 1
                    if row["material_shortfall"]:
                        bucket["material_shortfall_rows"] += 1
                        game_totals[family]["material_shortfall_rows"] += 1
                        material_count += 1
        game_summaries.append({"seed": game["seed"], "seat": game["seat"], "status": game["status"], "transitions": game["transitions"], "final_bank": game.get("final", {}).get("bank"), "care": game_totals["CARE"], "fertilize": game_totals["FERTILIZE"], "zero_completion_rows": zero_count, "material_shortfall_rows": material_count})

    zero_rows = [row for row in rows if row["zero_completion"]]
    material_rows = [row for row in rows if row["material_shortfall"]]
    classification_counts: dict[str, int] = defaultdict(int)
    for row in material_rows:
        classification_counts[(row["classification"] or {}).get("classification", "unclassified")] += 1
    result: dict[str, Any] = {
        "audit_schema_version": 1,
        "audit_type": "executor_v07_issue14a_care_fertilize",
        "source_panel": {
            "artifact_sha256": artifact_sha256,
            "artifact_internal_sha256": document.get("artifact_sha256"),
            "repo_sha": document.get("source_provenance", {}).get("repo_sha"),
            "checkpoint": document.get("checkpoint"),
            "request": document.get("request"),
        },
        "methodology": {
            "row_count": len(rows),
            "material_definition": f"raw requested - observed_completed >= {material_min_units}; signed raw values are retained",
            "requested": "requested plan field for the day",
            "eligible": "projection_changes eligibility count",
            "feasible_projected": "feasible plan field after project_plan",
            "submitted_assigned": "turn_trace assignments whose interaction opcode is CARE or FERTILIZE, grouped by entity; movement/pickup is not completion",
            "observed_completed": "diagnostic board-state count: cared_today for CARE, active fertilizer state for FERTILIZE",
            "shortfall_caveat": "observed_completed is not a daily accepted-action ledger; CARE is overwritten at the next-day boundary and fertilizer state persists",
        },
        "validation": {
            "panel_game_count": len(document["games"]),
            "cartesian_24_cases": len(document["games"]) == 24,
            "anomalies": anomalies,
            "proven_executor_shortfalls": [],
        },
        "runtime_signals": runtime_signals,
        "totals_by_entity": totals,
        "per_game": game_summaries,
        "per_day": rows,
        "zero_completion_rows": zero_rows,
        "material_shortfall_rows": material_rows,
        "classification_counts": dict(sorted(classification_counts.items())),
    }
    unsigned = json.loads(json.dumps(result, sort_keys=True, separators=(",", ":")))
    result["report_sha256"] = _sha256_bytes(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())
    return result


def _csv_ints(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise AuditError(f"invalid integer list: {value!r}") from exc
    if not result:
        raise AuditError("integer list must not be empty")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--seats", required=True)
    parser.add_argument("--backend", default="fast")
    parser.add_argument("--opening", default="standard_mixed")
    parser.add_argument("--day-start", type=int, default=4)
    parser.add_argument("--day-end", type=int, default=29)
    parser.add_argument("--material-min-units", type=int, default=MATERIAL_MIN_UNITS)
    args = parser.parse_args(argv)
    try:
        artifact_path = Path(args.artifact)
        payload = artifact_path.read_bytes()
        document = json.loads(payload)
        result = build_audit(
            document,
            artifact_sha256=_sha256_bytes(payload),
            expected_repo_sha=args.expected_repo_sha,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_seeds=_csv_ints(args.seeds),
            expected_seats=_csv_ints(args.seats),
            expected_backend=args.backend,
            expected_opening=args.opening,
            expected_day_range=(args.day_start, args.day_end),
            material_min_units=args.material_min_units,
        )
        output = Path(args.output)
        output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except (AuditError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
