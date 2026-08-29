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
from pathlib import Path
from typing import Any, Mapping, Sequence

DIAGNOSTICS_SCHEMA_VERSION = 1

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


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "build_integration_diagnostics",
    "write_diagnostics",
]
