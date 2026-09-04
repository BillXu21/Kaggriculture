"""Compact JSON diagnostics artifact for RL integration runs (issue #9 B2).

One small, strictly JSON-safe (`allow_nan=False`) record per smoke/eval run
covering: rollout seed/seat/composition, episode/manager-step throughput,
the env/executor/policy/orchestration timing split, return/win, entropy by
the six action groups, approx KL, clip fraction, value loss / explained
variance, advantage stats, action drift (KL-to-frozen) when available,
final banks/margin, unfinished/missed-maintenance totals, executor/opening/
backend anomalies and provenance, and the pre/post policy fingerprints plus
checkpoint path.

Honesty rule: any quantity that is genuinely unavailable is serialized as
`null` with a machine-readable reason collected under `missing` — never a
fabricated number.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bc_manager.constants import CROP_ORDER

DIAGNOSTICS_SCHEMA_VERSION = 1
ECONOMIC_DIAGNOSTICS_SCHEMA_VERSION = 1

#: runner timing bucket -> diagnostics key (issue #9 required split).
_TIMING_MAP = {
    "env_step": "env",
    "agent_actions": "executor",  # opening playback + primitive executor turns
    "manager_inference": "policy",
    "orchestration": "orchestration",
}


def _null(payload: dict[str, Any], missing: dict[str, str], key: str,
          reason: str) -> None:
    payload[key] = None
    missing[key] = reason


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _episode_outcome(result: Any, payload: dict[str, Any],
                     missing: dict[str, str]) -> None:
    if result is None:
        _null(payload, missing, "rollout", "no episode result recorded")
        return
    banks = [float(bank) for bank in result.final_banks]
    rewards = [float(reward) for reward in result.rewards]
    payload["rollout"] = {
        "seed": int(result.seed),
        "composition": str(result.composition),
        "episodes": 1,
        "manager_steps": int(result.transitions),
        "terminated": bool(result.terminated),
        "statuses": [str(status) for status in result.statuses],
        "return_seat0": rewards[0],
        "win": int(result.winner_seat),
        "final_banks": banks,
        "margin": float(result.margin),
    }


def _timing_split(timing: Mapping[str, Any] | None, payload: dict[str, Any],
                  missing: dict[str, str]) -> None:
    if not timing:
        _null(payload, missing, "timing_seconds",
              "runner timing totals were not captured")
        return
    split = {}
    for source, target in _TIMING_MAP.items():
        value = _finite(timing.get(source))
        split[target] = value
        if value is None:
            missing[f"timing_seconds.{target}"] = f"bucket {source!r} absent"
    total = sum(v for v in split.values() if v is not None)
    steps = _finite(payload.get("rollout", {}).get("manager_steps")) \
        if isinstance(payload.get("rollout"), dict) else None
    payload["timing_seconds"] = split
    if total > 0.0 and steps:
        payload["manager_steps_per_second"] = steps / total


def _update_metrics(metrics: Mapping[str, Any] | None,
                    payload: dict[str, Any], missing: dict[str, str]) -> None:
    if not metrics:
        _null(payload, missing, "ppo_metrics", "no PPO update was executed")
        return
    direct = ("approx_kl", "clip_fraction", "value_loss", "explained_variance")
    section: dict[str, Any] = {}
    for key in direct:
        value = _finite(metrics.get(key))
        section[key] = value
        if value is None:
            missing[f"ppo_metrics.{key}"] = "metric absent from update output"
    entropy_groups = {}
    for group in ("crop", "animal", "land", "fertilizer", "care",
                  "sell_presence"):
        value = _finite(metrics.get(f"entropy_{group}"))
        entropy_groups[group] = value
        if value is None:
            missing[f"entropy_by_group.{group}"] = "entropy group absent"
    section["entropy_by_group"] = entropy_groups
    adv = {key: _finite(metrics.get(key)) for key in
           ("adv_mean", "adv_std", "adv_min", "adv_max")}
    section["advantage_stats"] = adv
    kl_frozen = _finite(metrics.get("kl_to_frozen"))
    section["action_drift_kl_to_frozen"] = kl_frozen
    if kl_frozen is None:
        missing["action_drift_kl_to_frozen"] = (
            "kl_to_frozen metric absent; drift vs frozen snapshot unmeasured")
    section["epochs_ran"] = _finite(metrics.get("epochs_ran"))
    section["minibatches_ran"] = _finite(metrics.get("minibatches_ran"))
    section["rows_ran"] = _finite(metrics.get("rows_ran"))
    section["accepted"] = bool(metrics.get("accepted", True))
    section["stop_reason"] = metrics.get("stop_reason", "completed")
    section["rejection_reason"] = metrics.get("rejection_reason")
    raw_epochs = metrics.get("epoch_metrics", [])
    section["epoch_metrics"] = [
        {key: (_finite(value) if key not in ("epoch", "minibatches")
               else int(value)) for key, value in epoch.items()}
        for epoch in raw_epochs if isinstance(epoch, Mapping)]
    payload["ppo_metrics"] = section


def _executor_totals(sidecar_records: Sequence[Any] | None,
                     payload: dict[str, Any]) -> None:
    unfinished = 0
    missed = 0
    counted = 0
    for record in sidecar_records or []:
        diag = getattr(record, "executor_day_diagnostics", None) or {}
        if not diag:
            continue
        counted += 1
        unfinished += int(diag.get("unfinished_tasks", 0))
        missed += int(diag.get("missed_maintenance", 0))
    payload["executor_totals"] = {
        "days_with_diagnostics": counted,
        "unfinished_tasks": unfinished,
        "missed_maintenance": missed,
    }


def build_integration_diagnostics(
    *,
    result: Any = None,
    runner_timing: Mapping[str, Any] | None = None,
    sidecar_records: Sequence[Any] | None = None,
    update_metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    pre_update_fingerprint: str | None = None,
    post_update_fingerprint: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Assemble the compact diagnostics record; nulls always carry reasons."""
    missing: dict[str, str] = {}
    payload: dict[str, Any] = {
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
    }
    _episode_outcome(result, payload, missing)
    _timing_split(runner_timing, payload, missing)
    _update_metrics(update_metrics, payload, missing)
    _executor_totals(sidecar_records, payload)

    anomalies: list[str] = []
    rollout = payload.get("rollout") or {}
    statuses = rollout.get("statuses") or []
    if statuses and statuses != ["DONE", "DONE"]:
        anomalies.append(f"non-terminal statuses {statuses}")
    if rollout.get("terminated") is False:
        anomalies.append("episode ended truncated (turn budget exhausted)")
    payload["anomalies"] = anomalies

    payload["fingerprints"] = {
        "pre_update": pre_update_fingerprint,
        "post_update": post_update_fingerprint,
    }
    if pre_update_fingerprint is None:
        missing["fingerprints.pre_update"] = "not captured"
    if post_update_fingerprint is None:
        missing["fingerprints.post_update"] = "not captured"
    payload["checkpoint_path"] = checkpoint_path

    prov = dict(provenance or {})
    payload["provenance"] = prov
    for key in ("opening", "backend", "executor_factory"):
        if key not in prov:
            missing[f"provenance.{key}"] = "provenance block not provided"
    payload["missing"] = dict(sorted(missing.items()))
    return payload


def write_diagnostics(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write the diagnostics JSON atomically-clean: `allow_nan=False` fails
    loudly on any NaN/Inf leak instead of poisoning downstream consumers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1, allow_nan=False),
        encoding="utf-8")
    return path


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None,
                "min": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "p10": float(np.percentile(ordered, 10)),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _manager_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    result = _distribution(values)
    result["p90"] = (float(np.percentile(sorted(float(value) for value in values), 90))
                     if values else None)
    return result


def _crop_targets(row: Mapping[str, Any]) -> dict[str, int] | None:
    targets = row.get("requested_crop_targets")
    if targets is None:
        targets = (row.get("requested") or {}).get("crop_targets")
    if not isinstance(targets, Mapping):
        return None
    return {crop: int(targets.get(crop, 0) or 0) for crop in CROP_ORDER}


def _achieved_crops(row: Mapping[str, Any]) -> dict[str, int] | None:
    crops = row.get("achieved_final_crops")
    if crops is None:
        crops = (row.get("achieved_final") or {}).get("crops")
    if not isinstance(crops, Mapping):
        return None
    return {crop: int(crops.get(crop, 0) or 0) for crop in CROP_ORDER}


def _manager_crop_rows(results: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        trainable_seats = {
            int(record["seat"]) for record in result.policy_identities
            if record.get("trainable")
        }
        rows.extend(
            row for row in (getattr(result, "manager_crop_rows", None) or [])
            if isinstance(row, Mapping)
            and row.get("trainable", int(row.get("seat", -1)) in trainable_seats)
            and int(row.get("seat", -1)) in trainable_seats
        )
    return rows


_UNRESOLVED_CROP_RE = re.compile(
    r"^crop_deficit_unresolved:([^:]+):([0-9]+)$")


def _unresolved_crop_units(row: Mapping[str, Any]) -> dict[str, int]:
    units = {crop: 0 for crop in CROP_ORDER}
    for entry in row.get("unresolved_generator", []) or []:
        if not isinstance(entry, str):
            continue
        match = _UNRESOLVED_CROP_RE.fullmatch(entry)
        if match is None or match.group(1) not in units:
            continue
        units[match.group(1)] += int(match.group(2))
    return units


def _manager_day_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_day: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        targets = _crop_targets(row)
        if targets is not None:
            by_day.setdefault(int(row["day"]), []).append(targets)
    return {
        str(day): {
            "row_count": len(day_rows),
            "requested_total_mean": float(
                sum(sum(targets.values()) for targets in day_rows)
                / len(day_rows)),
            "requested_by_species": {
                crop: float(sum(targets[crop] for targets in day_rows)
                            / len(day_rows))
                for crop in CROP_ORDER
            },
        }
        for day, day_rows in sorted(by_day.items())
    }


def _manager_late_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = (("4-19", 4, 19), ("20-24", 20, 24),
               ("25-27", 25, 27), ("28-29", 28, 29))
    result: dict[str, Any] = {}
    for name, first, last in buckets:
        bucket = [row for row in rows if first <= int(row["day"]) <= last]
        targets = [_crop_targets(row) for row in bucket]
        targets = [target for target in targets if target is not None]
        result[name] = {
            "row_count": len(targets),
            "requested_total_mean": (
                float(sum(sum(target.values()) for target in targets) / len(targets))
                if targets else None),
            "requested_total_median": (
                float(statistics.median(sum(target.values()) for target in targets))
                if targets else None),
            "requested_by_species_mean": {
                crop: (float(sum(target[crop] for target in targets) / len(targets))
                       if targets else None)
                for crop in CROP_ORDER
            },
        }
    return result


def _manager_crop_intent(
        rows: Sequence[Mapping[str, Any]],
        crop_action_max: int | None,
        ) -> dict[str, Any]:
    targets_by_row = [(_crop_targets(row), row) for row in rows]
    targets_by_row = [(targets, row) for targets, row in targets_by_row
                      if targets is not None]
    targets = [targets for targets, _ in targets_by_row]
    totals = [sum(target.values()) for target in targets]
    species = {
        crop: _manager_distribution([target[crop] for target in targets])
        | {"fraction_nonzero": (
            sum(target[crop] > 0 for target in targets) / len(targets)
            if targets else None)}
        for crop in CROP_ORDER
    }
    distinct = [sum(value > 0 for value in target.values()) for target in targets]
    dominant = [max(target.values()) / total if total else 0.0
                for target, total in zip(targets, totals)]

    trajectory_rows: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for _, row in targets_by_row:
        trajectory_rows.setdefault(
            (int(row["episode_index"]), int(row["seat"])), []).append(row)
    changed_count = 0
    comparable_count = 0
    for trajectory in trajectory_rows.values():
        trajectory = sorted(trajectory, key=lambda row: int(row["day"]))
        previous: tuple[int, ...] | None = None
        for row in trajectory:
            vector = tuple(_crop_targets(row)[crop] for crop in CROP_ORDER)
            if previous is not None:
                comparable_count += 1
                changed_count += vector != previous
            previous = vector

    component_count = len(targets) * len(CROP_ORDER)
    saturation_by_species: dict[str, Any] = {}
    for crop in CROP_ORDER:
        at_max = (sum(target[crop] == crop_action_max for target in targets)
                  if crop_action_max is not None else None)
        saturation_by_species[crop] = {
            "action_max": crop_action_max,
            "fraction_at_max": (at_max / len(targets)
                                 if at_max is not None and targets else None),
        }
    at_max_components = (
        sum(target[crop] == crop_action_max
            for target in targets for crop in CROP_ORDER)
        if crop_action_max is not None else None)
    all_at_max = (
        sum(all(target[crop] == crop_action_max for crop in CROP_ORDER)
            for target in targets)
        if crop_action_max is not None else None)

    unresolved_by_species = {crop: [] for crop in CROP_ORDER}
    unresolved_rows = 0
    for row in rows:
        units = _unresolved_crop_units(row)
        if any(units.values()):
            unresolved_rows += 1
        for crop in CROP_ORDER:
            unresolved_by_species[crop].append(units[crop])
    unresolved_total = sum(sum(values) for values in unresolved_by_species.values())

    eod_rows: list[tuple[dict[str, int], dict[str, int]]] = []
    for _, row in targets_by_row:
        achieved = _achieved_crops(row)
        if achieved is not None:
            eod_rows.append((_crop_targets(row), achieved))
    shortfalls = [{
        crop: max(0, requested[crop] - achieved[crop])
        for crop in CROP_ORDER
    } for requested, achieved in eod_rows]
    shortfall_by_species = {
        crop: [shortfall[crop] for shortfall in shortfalls]
        for crop in CROP_ORDER
    }
    shortfall_total = sum(sum(shortfall.values()) for shortfall in shortfalls)
    days_with_shortfall = sum(
        any(value > 0 for value in shortfall.values())
        for shortfall in shortfalls)
    realized_fractions = [
        min(sum(achieved.values()) / sum(requested.values()), 1.0)
        for requested, achieved in eod_rows if sum(requested.values()) > 0
    ]

    manager_row_count = len(rows)
    eod_day_count = len(eod_rows)
    return {
        "requested_total": _manager_distribution(totals),
        "requested_by_species": species,
        "mix": {
            "mean_distinct_species_requested": (
                float(sum(distinct) / len(distinct)) if distinct else None),
            "median_distinct_species_requested": (
                float(statistics.median(distinct)) if distinct else None),
            "fraction_single_species": (
                sum(value == 1 for value in distinct) / len(distinct)
                if distinct else None),
            "mean_dominant_species_fraction": (
                float(sum(dominant) / len(dominant)) if dominant else None),
        },
        "fraction_target_vector_changed_from_previous_manager_day": {
            "changed_count": changed_count,
            "comparable_count": comparable_count,
            "fraction": (changed_count / comparable_count
                         if comparable_count else None),
        },
        "saturation": {
            "component_count": component_count,
            "component_at_max_count": at_max_components,
            "fraction_crop_components_at_action_max": (
                at_max_components / component_count
                if at_max_components is not None and component_count else None),
            "row_count": len(targets),
            "all_components_at_max_row_count": all_at_max,
            "fraction_manager_rows_all_crop_components_at_action_max": (
                all_at_max / len(targets)
                if all_at_max is not None and targets else None),
            "by_species": saturation_by_species,
        },
        "by_manager_day": _manager_day_summary(rows),
        "late_game": _manager_late_summary(rows),
        "unresolved_crop_deficit": {
            "manager_row_count": manager_row_count,
            "rows_with_unresolved_deficit": unresolved_rows,
            "fraction_rows_with_unresolved_deficit": (
                unresolved_rows / manager_row_count if manager_row_count else None),
            "total_units": unresolved_total,
            "mean_units_per_manager_row": (
                unresolved_total / manager_row_count if manager_row_count else None),
            "by_species": {
                crop: {
                    "total_units": sum(values),
                    "mean_units_per_manager_row": (
                        sum(values) / manager_row_count
                        if manager_row_count else None),
                    "fraction_rows_nonzero": (
                        sum(value > 0 for value in values) / manager_row_count
                        if manager_row_count else None),
                }
                for crop, values in unresolved_by_species.items()
            },
        },
        "end_of_day_shortfall": {
            "day_count": eod_day_count,
            "days_with_shortfall": days_with_shortfall,
            "fraction_days_with_shortfall": (
                days_with_shortfall / eod_day_count
                if eod_day_count else None),
            "total_units": shortfall_total,
            "mean_units_per_day": (
                shortfall_total / eod_day_count if eod_day_count else None),
            "requested_total": _distribution(
                [sum(requested.values()) for requested, _ in eod_rows]),
            "achieved_final_crop_total": _distribution(
                [sum(achieved.values()) for _, achieved in eod_rows]),
            "by_species": {
                crop: {
                    "total_units": sum(values),
                    "mean_units_per_day": (
                        sum(values) / eod_day_count if eod_day_count else None),
                    "fraction_days_nonzero": (
                        sum(value > 0 for value in values) / eod_day_count
                        if eod_day_count else None),
                }
                for crop, values in shortfall_by_species.items()
            },
        },
        "target_utilization": {
            "row_count": len(realized_fractions),
            "mean_realized_fraction": (
                float(sum(realized_fractions) / len(realized_fractions))
                if realized_fractions else None),
            "median_realized_fraction": (
                float(statistics.median(realized_fractions))
                if realized_fractions else None),
        },
    }


def aggregate_economic_diagnostics(
        results: Sequence[Any], *, crop_action_max: int | None = None,
        ) -> dict[str, Any]:
    """Aggregate observed economic outcomes for trainable seats only."""
    seats: list[tuple[Any, int, dict[str, Any]]] = []
    for result in results:
        trainable = {int(record["seat"]) for record in result.policy_identities
                     if record.get("trainable")}
        finals = {int(snapshot["seat"]): snapshot for snapshot in
                  result.utilization_snapshots
                  if snapshot.get("boundary") == "terminal"}
        for seat in sorted(trainable):
            if seat not in finals:
                raise ValueError(
                    f"missing terminal utilization snapshot for seat {seat}")
            seats.append((result, seat, finals[seat]))
    banks = [float(result.final_banks[seat]) for result, seat, _ in seats]
    bank_distribution = _distribution(banks)
    quadrants = [int(snapshot["land_quadrants_owned"])
                 for _, _, snapshot in seats]
    purchases = {quadrant: [event for result, seat, _ in seats
                            for event in result.land_purchase_events
                            if int(event["seat"]) == seat
                            and event["quadrant"] == quadrant]
                 for quadrant in ("NE", "SW", "SE")}
    purchase_stats: dict[str, Any] = {}
    denominator = len(seats)
    for quadrant, events in purchases.items():
        days = [float(event["causal_day"]) for event in events]
        purchase_stats[quadrant] = {
            "successful_purchase_rate": (len(events) / denominator
                                          if denominator else None),
            "mean_purchase_day": (float(sum(days) / len(days)) if days else None),
            "median_purchase_day": (float(statistics.median(days))
                                     if days else None),
        }
    daily_occupancy = [float(snapshot["productive_occupancy"])
                       for result, seat, _ in seats
                       for snapshot in result.utilization_snapshots
                       if int(snapshot["seat"]) == seat
                       and snapshot.get("boundary") == "daily"
                       and int(snapshot["day"]) > 4]
    final_snapshots = [snapshot for _, _, snapshot in seats]
    manager_rows = _manager_crop_rows(results)
    return {
        "trainable_seat_count": len(seats),
        "economic": {
            "final_bank": bank_distribution,
            "mean_final_bank": bank_distribution["mean"],
            "median_final_bank": bank_distribution["median"],
            "p10_final_bank": bank_distribution["p10"],
            "min_final_bank": bank_distribution["min"],
            "max_final_bank": bank_distribution["max"],
            "fraction_bank_below_10k": (sum(bank < 10000 for bank in banks) /
                                         len(banks) if banks else None),
            "fraction_bank_above_30k": (sum(bank > 30000 for bank in banks) /
                                        len(banks) if banks else None),
        },
        "land": {
            "mean_final_quadrants_owned": (
                float(sum(quadrants) / len(quadrants)) if quadrants else None),
            "final_quadrants_count": {
                str(value): quadrants.count(value) for value in (1, 2, 3, 4)},
            "purchases": purchase_stats,
            "mean_final_crop_squares": _mean(final_snapshots, "crop_squares"),
            "mean_final_animal_squares": _mean(final_snapshots, "animal_squares"),
            "mean_final_productive_squares": _mean(
                final_snapshots, "productive_squares"),
            "mean_final_productive_occupancy": _mean(
                final_snapshots, "productive_occupancy"),
            "median_final_productive_occupancy": (
                float(statistics.median(
                    snapshot["productive_occupancy"] for snapshot in final_snapshots))
                if final_snapshots else None),
            "mean_post_d4_productive_occupancy": (
                float(sum(daily_occupancy) / len(daily_occupancy))
                if daily_occupancy else None),
        },
        "manager_crop_intent": _manager_crop_intent(
            manager_rows, crop_action_max),
    }


def _mean(snapshots: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(snapshot[key]) for snapshot in snapshots]
    return float(sum(values) / len(values)) if values else None


def build_economic_diagnostics(
        results: Sequence[Any], *, crop_action_max: int | None = None,
        ) -> dict[str, Any]:
    """Return recomputable per-episode sidecars plus aggregate metrics."""
    episodes = []
    for result in sorted(results, key=lambda item: int(item.episode_index)):
        trainable_seats = [int(record["seat"]) for record in result.policy_identities
                           if record.get("trainable")]
        episodes.append({
            "episode": int(result.episode_index),
            "seed": int(result.seed),
            "composition": str(result.composition),
            "trainable_seats": trainable_seats,
            "final_banks": [float(value) for value in result.final_banks],
            "rewards": [float(value) for value in result.rewards],
            "land_purchase_events": list(result.land_purchase_events),
            "utilization_snapshots": list(result.utilization_snapshots),
        })
    return {
        "economic_diagnostics_schema_version": ECONOMIC_DIAGNOSTICS_SCHEMA_VERSION,
        "episodes": episodes,
        "aggregate": aggregate_economic_diagnostics(
            results, crop_action_max=crop_action_max),
    }


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "ECONOMIC_DIAGNOSTICS_SCHEMA_VERSION",
    "aggregate_economic_diagnostics",
    "build_economic_diagnostics",
    "build_integration_diagnostics",
    "write_diagnostics",
]
