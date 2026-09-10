"""Four-process sharded wrapper for :mod:`tools.evaluate_stage25_upkeep`.

The evaluator owns model loading, stochastic decoding, engine execution, and
capture production.  This module only plans identity-preserving seed/seat
groups, starts independent evaluator interpreters, and validates/merges their
JSONL results.  In particular, it must stay free of JAX imports: child
processes are launched with ``python -m`` rather than forked after imports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from tools.evaluate_stage25_upkeep import (
    COMPARISON_REFERENCES,
    VARIANTS,
    episode_id_for,
    parse_game_filter,
)


EVALUATOR_MODULE = "tools.evaluate_stage25_upkeep"
SCHEMA_VERSION = 1
TELEMETRY_FIELDS: dict[str, tuple[str, ...]] = {
    "candidate_bank": ("candidate_bank", "bank"),
    "opponent_bank": ("opponent_bank",),
    "margin": ("margin",),
    "completed_useful_work": (
        "completed_useful_work", "completed_work", "work_completed",
        "completed_tasks",
    ),
    "missed_maintenance": ("missed_maintenance", "missed_maintenance_total"),
    "duplicate_claims": (
        "duplicate_claims", "duplicate_claim", "coassigned_turns",
    ),
    "target_abandonment": (
        "target_abandonment", "target_abandoned", "target_abandonments",
        "ended_unobserved",
    ),
    "movement_between_interactions": (
        "movement_between_interactions", "movement_between_actions",
        "cand_movement", "movement",
    ),
    "hiring_expense": (
        "hiring_expense", "hiring_cost", "hire_cost", "hiring_cost_total",
    ),
    # Backward-compatible surface retained for existing consumers.
    "completed_work": ("completed_work", "work_completed", "completed_tasks"),
    "travel_abandonment": (
        "travel_abandonment", "travel_abandoned", "travel_abandonments",
    ),
    "hiring_cost": ("hiring_cost", "hire_cost", "hiring_cost_total"),
    "scheduler_runtime": (
        "scheduler_runtime", "scheduler_runtime_ms", "runtime_ms",
    ),
}


GamePair = tuple[int, int]  # (seed_index, seat)
Identity = tuple[str, int, int, int]  # variant, seed_index, seat, episode_id


@dataclass(frozen=True)
class Shard:
    """One deterministic child assignment."""

    index: int
    games: tuple[GamePair, ...]


def plan_game_pairs(
    seeds: Sequence[int], game_filters: Sequence[str] | None = None,
) -> tuple[list[int], list[GamePair]]:
    """Return the full seed list and selected pairs in stable input order."""
    ordered = [int(seed) for seed in seeds]
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("--seeds must be a non-empty ordered list of unique integers")
    selected = parse_game_filter(
        list(game_filters) if game_filters is not None else None, ordered)
    pairs = [
        (index, seat)
        for index in range(len(ordered))
        for seat in (0, 1)
        if selected is None or (index, seat) in selected
    ]
    return ordered, pairs


def shard_game_pairs(game_pairs: Sequence[GamePair], processes: int = 4) -> list[Shard]:
    """Distribute complete seed groups round-robin across children.

    Both seats of a seed remain in one shard, which keeps the natural paired
    bootstrap unit intact.  The function also works for a partial filter where
    only one seat of a seed was requested.
    """
    if processes < 1:
        raise ValueError("processes must be positive")
    by_seed: dict[int, list[GamePair]] = defaultdict(list)
    seed_order: list[int] = []
    for pair in game_pairs:
        index, seat = int(pair[0]), int(pair[1])
        normalized = (index, seat)
        if seat not in (0, 1) or normalized in by_seed.get(index, []):
            raise ValueError(f"invalid or duplicate game pair: {pair!r}")
        if index not in by_seed:
            seed_order.append(index)
        by_seed[index].append(normalized)
    # A filtered run may contain fewer seed groups than the requested worker
    # count.  Do not launch an empty child: the existing evaluator interprets
    # an empty ``--game-filter`` as an unfiltered full-panel request.
    actual_processes = min(processes, len(seed_order))
    assignments: list[list[GamePair]] = [[] for _ in range(actual_processes)]
    for group_number, seed_index in enumerate(seed_order):
        assignments[group_number % processes].extend(
            sorted(by_seed[seed_index], key=lambda pair: pair[1]))
    return [Shard(index, tuple(games)) for index, games in enumerate(assignments)]


def expected_identities(
    seeds: Sequence[int], master_seed: int, variants: Sequence[str],
    game_pairs: Sequence[GamePair],
) -> set[Identity]:
    """Compute the exact row identities expected from a complete merge."""
    n = len(seeds)
    return {
        (str(variant), int(index), int(seat),
         episode_id_for(master_seed, n, int(index), int(seat)))
        for variant in variants
        for index, seat in game_pairs
    }


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def engine_provenance(backend: str, repo_root: Path) -> dict[str, Any]:
    """Collect engine identity without importing either engine or JAX."""
    module_name = "fast_env._kaggriculture_env" if backend == "fast" else "kaggle_environments"
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    module_path = Path(spec.origin) if spec and spec.origin else None
    try:
        package_version = importlib.metadata.version("kaggle-environments")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "backend": backend,
        "module": module_name,
        "module_path": str(module_path) if module_path else None,
        "module_sha256": _sha256(module_path),
        "package_version": package_version,
        "runner_sha256": _sha256(repo_root / "rl_manager" / "runner.py"),
    }


def _validate_config(
    seeds: Sequence[int], variants: Sequence[str], master_seed: int,
    processes: int,
) -> None:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty ordered list of unique integers")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty list of unique names")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if master_seed < 0:
        raise ValueError("master seed must be nonnegative")
    if processes < 1:
        raise ValueError("processes must be positive")


def _config_manifest(
    *, repo_root: Path, wrapper_path: Path, evaluator_path: Path,
    checkpoint: Path | None, e_checkpoint: Path | None, seeds: Sequence[int],
    master_seed: int, variants: Sequence[str], backend: str,
    e_history_version: str, processes: int, game_filters: Sequence[str] | None,
    game_pairs: Sequence[GamePair], preflight_only: bool, capture_dir: Path | None,
    underfoot_first: bool, deadline_safe_planting: bool,
    deadline_safe_hiring: bool, persistent_worker_queues: bool,
    queue_ownership_repair: bool, schedule_informed_hiring: bool,
    batch_reserved_supplies: bool, underfoot_queue_insertion: bool,
    schedule_hiring_economic_repair: bool,
    starvation_workload_visibility_repair: bool,
) -> dict[str, Any]:
    diff = _git(repo_root, "diff", "HEAD")
    return {
        "schema_version": SCHEMA_VERSION,
        "wrapper": "run_stage25_upkeep_sharded",
        "evaluator_module": EVALUATOR_MODULE,
        "config": {
            "checkpoint": str(checkpoint) if checkpoint else None,
            "e_checkpoint": str(e_checkpoint) if e_checkpoint else None,
            "seeds": list(seeds),
            "master_seed": master_seed,
            "variants": list(variants),
            "backend": backend,
            "e_history_version": e_history_version,
            "processes": processes,
            "game_filters": list(game_filters) if game_filters is not None else None,
            "capture_dir": str(capture_dir) if capture_dir else None,
            "preflight_only": preflight_only,
            "underfoot_first": underfoot_first,
            "deadline_safe_planting": deadline_safe_planting,
            "deadline_safe_hiring": deadline_safe_hiring,
            "persistent_worker_queues": persistent_worker_queues,
            "queue_ownership_repair": queue_ownership_repair,
            "batch_reserved_supplies": batch_reserved_supplies,
            "underfoot_queue_insertion": underfoot_queue_insertion,
            "schedule_informed_hiring": schedule_informed_hiring,
            "schedule_hiring_economic_repair": schedule_hiring_economic_repair,
            "starvation_workload_visibility_repair": (
                starvation_workload_visibility_repair),
        },
        "source": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
            "dirty": bool(_git(repo_root, "status", "--porcelain")),
            "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
            "evaluator_sha256": _sha256(evaluator_path),
            "wrapper_sha256": _sha256(wrapper_path),
        },
        "patch": {
            "git_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
            "wrapper_sha256": _sha256(wrapper_path),
        },
        "checkpoints": {
            "ppo_sha256": _sha256(checkpoint),
            "bc_e_sha256": _sha256(e_checkpoint),
        },
        "engine": engine_provenance(backend, repo_root),
        # A game is one seed/seat identity.  The evaluator emits one row per
        # selected variant for that identity, so retain both counts explicitly.
        "planned_games": len(game_pairs),
        "planned_evaluator_rows": len(game_pairs) * len(variants),
        "planned_seed_seat_pairs": len(game_pairs),
        "ordered_seeds": list(seeds),
        "status": "preflight",
    }


def _identity_for_row(row: Mapping[str, Any], seeds: Sequence[int], master_seed: int) -> Identity:
    try:
        variant = str(row["variant"])
        seed = int(row["seed"])
        seat = int(row["seat"])
        episode = int(row["episode_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed game row: {row!r}") from exc
    if variant not in VARIANTS or seed not in seeds or seat not in (0, 1):
        raise ValueError(f"unexpected game identity: {row!r}")
    index = list(seeds).index(seed)
    expected_episode = episode_id_for(master_seed, len(seeds), index, seat)
    if episode != expected_episode:
        raise ValueError(
            f"episode identity mismatch for {variant}/{seed}:{seat}: "
            f"expected {expected_episode}, got {episode}")
    statuses = row.get("statuses")
    if list(statuses or []) != ["DONE", "DONE"]:
        raise ValueError(f"non-DONE game row for {variant}/{seed}:{seat}: {statuses!r}")
    return variant, index, seat, episode


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing shard output: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"game row is not an object in {path}:{line_number}")
        rows.append(value)
    return rows


def merge_rows(
    shard_rows: Mapping[int, Sequence[Mapping[str, Any]]], *, seeds: Sequence[int],
    master_seed: int, variants: Sequence[str], game_pairs: Sequence[GamePair],
    shard_plans: Sequence[Shard] | None = None,
) -> list[dict[str, Any]]:
    """Validate shard ownership and return canonical rows."""
    expected = expected_identities(seeds, master_seed, variants, game_pairs)
    expected_by_shard = {
        shard.index: expected_identities(seeds, master_seed, variants, shard.games)
        for shard in (shard_plans or [])
    }
    found: dict[Identity, dict[str, Any]] = {}
    for shard_index, rows in shard_rows.items():
        for row in rows:
            identity = _identity_for_row(row, seeds, master_seed)
            if identity not in expected:
                raise ValueError(f"unexpected game identity in shard {shard_index}: {identity}")
            if expected_by_shard and identity not in expected_by_shard.get(shard_index, set()):
                raise ValueError(f"game identity assigned to wrong shard {shard_index}: {identity}")
            if identity in found:
                raise ValueError(f"duplicate game identity: {identity}")
            found[identity] = dict(row)
    missing = expected - set(found)
    if missing:
        raise ValueError(f"omitted game identities: {sorted(missing)}")
    return [
        found[identity]
        for identity in sorted(found, key=lambda item: (item[0], item[1], item[2], item[3]))
    ]


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    middle = (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
    return {
        "mean": sum(values) / len(values), "median": middle,
        "min": min(values), "max": max(values),
    }


def _summary(rows: Sequence[Mapping[str, Any]], variant: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    margins = [float(row["margin"]) for row in rows]
    banks = [float(row["bank"]) for row in rows]
    opponent = [float(row["opponent_bank"]) for row in rows]
    wins = sum(value > 0 for value in margins)
    losses = sum(value < 0 for value in margins)
    return {
        "evaluation_schema_version": 2,
        "variant": variant,
        "games": len(rows), "completed_games": len(rows),
        "wlt": {"W": wins, "L": losses, "T": len(rows) - wins - losses},
        "win_rate": wins / len(rows) if rows else None,
        "paired_margins": margins,
        "margins": _numeric_summary(margins), "banks": {
            "candidate": _numeric_summary(banks),
            "opponent": _numeric_summary(opponent),
        },
        "mean_margin": sum(margins) / len(margins) if margins else None,
        "provenance": dict(manifest),
    }


def _comparison(rows: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["variant"])][(int(row["seed"]), int(row["seat"]))] = row
    result: list[dict[str, Any]] = []
    for variant in variants:
        own = list(grouped[variant].values())
        reference = COMPARISON_REFERENCES[variant]
        ref = grouped.get(reference, {}) if reference else {}
        bank_deltas = [float(row["bank"]) - float(ref[(int(row["seed"]), int(row["seat"]))]["bank"])
                       for row in own if (int(row["seed"]), int(row["seat"])) in ref]
        margin_deltas = [float(row["margin"]) - float(ref[(int(row["seed"]), int(row["seat"]))]["margin"])
                         for row in own if (int(row["seed"]), int(row["seat"])) in ref]
        result.append({
            "variant": variant, "games": len(own),
            "mean_bank": sum(float(row["bank"]) for row in own) / len(own) if own else None,
            "mean_opponent_bank": sum(float(row["opponent_bank"]) for row in own) / len(own) if own else None,
            "mean_margin": sum(float(row["margin"]) for row in own) / len(own) if own else None,
            "comparison_reference": reference,
            "mean_paired_bank_delta": sum(bank_deltas) / len(bank_deltas) if bank_deltas else None,
            "paired_bank_deltas": bank_deltas,
            "mean_paired_margin_delta": sum(margin_deltas) / len(margin_deltas) if margin_deltas else None,
            "paired_margin_deltas": margin_deltas,
        })
    return result


def _flatten_matches(value: Any, key_names: set[str], path: str = "") -> list[tuple[str, Any]]:
    matches: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in key_names:
                matches.append((child_path, child))
            matches.extend(_flatten_matches(child, key_names, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            matches.extend(_flatten_matches(child, key_names, f"{path}[{index}]"))
    return matches


def _telemetry_numbers(value: Any) -> list[float]:
    """Turn exposed scalar/count containers into numeric observations."""
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, Mapping):
        numeric = [float(item) for item in value.values()
                   if isinstance(item, (int, float)) and not isinstance(item, bool)
                   and math.isfinite(float(item))]
        return numeric if numeric else [float(len(value))]
    if isinstance(value, (list, tuple, set)):
        return [float(len(value))]
    return []


def aggregate_telemetry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Traverse raw rows and report numeric totals plus unavailable fields.

    Diagnostic payloads differ between fast/official and capture modes.  This
    intentionally reports absent values instead of inventing zeroes.
    """
    output: dict[str, Any] = {"rows": len(rows), "fields": {}}
    for field, aliases in TELEMETRY_FIELDS.items():
        aliases_set = {alias.lower() for alias in aliases}
        count = 0
        numeric_sum = 0.0
        numeric_count = 0
        missing_rows = 0
        locations: list[str] = []
        for row in rows:
            matches = _flatten_matches(row, aliases_set)
            if not matches:
                missing_rows += 1
                continue
            count += len(matches)
            locations.extend(path for path, _ in matches[:8])
            for _, value in matches:
                for number in _telemetry_numbers(value):
                    numeric_sum += number
                    numeric_count += 1
        output["fields"][field] = {
            "available": count > 0,
            "count": count,
            "numeric_count": numeric_count,
            "sum": numeric_sum if numeric_count else None,
            "mean": numeric_sum / numeric_count if numeric_count else None,
            "missing_rows": missing_rows,
            "locations": sorted(set(locations)),
        }
    return output


def _bootstrap_groups(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], variants: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {
        variant: defaultdict(list) for variant in variants}
    for row in rows:
        variant = str(row["variant"])
        index = list(seeds).index(int(row["seed"]))
        grouped[variant][index].append({
            "seat": int(row["seat"]), "episode_id": int(row["episode_id"]),
            "margin": float(row["margin"]),
        })
    return {
        variant: [
            {"seed_index": index, "seed": int(seeds[index]),
             "seats": sorted(seat_rows, key=lambda item: item["seat"])}
            for index, seat_rows in sorted(grouped[variant].items())
        ]
        for variant in variants
    }


def _child_command(
    *, checkpoint: Path, e_checkpoint: Path, output_dir: Path, capture_dir: Path | None,
    seeds: Sequence[int], master_seed: int, variants: Sequence[str], backend: str,
    e_history_version: str, game_pairs: Sequence[GamePair],
    underfoot_first: bool = False, deadline_safe_planting: bool = False,
    deadline_safe_hiring: bool = False, persistent_worker_queues: bool = False,
    queue_ownership_repair: bool = False,
    batch_reserved_supplies: bool = False,
    underfoot_queue_insertion: bool = False,
    schedule_informed_hiring: bool = False,
    schedule_hiring_economic_repair: bool = False,
    starvation_workload_visibility_repair: bool = False,
) -> list[str]:
    filters = [f"{seeds[index]}:{seat}" for index, seat in game_pairs]
    command = [
        sys.executable, "-m", EVALUATOR_MODULE,
        "--checkpoint", str(checkpoint), "--e-checkpoint", str(e_checkpoint),
        "--output-dir", str(output_dir), "--seeds", *(str(seed) for seed in seeds),
        "--master-seed", str(master_seed), "--variants", *variants,
        "--backend", backend, "--e-history-version", e_history_version,
        "--game-filter", *filters,
    ]
    if capture_dir is not None:
        command.extend(("--capture-dir", str(capture_dir)))
    for enabled, flag in (
        (underfoot_first, "--underfoot-first"),
        (deadline_safe_planting, "--deadline-safe-planting"),
        (deadline_safe_hiring, "--deadline-safe-hiring"),
        (persistent_worker_queues, "--persistent-worker-queues"),
        (queue_ownership_repair and persistent_worker_queues,
         "--queue-ownership-repair"),
        (batch_reserved_supplies and queue_ownership_repair and persistent_worker_queues,
         "--batch-reserved-supplies"),
        (underfoot_queue_insertion and queue_ownership_repair and persistent_worker_queues,
         "--underfoot-queue-insertion"),
        (schedule_informed_hiring, "--schedule-informed-hiring"),
        (schedule_hiring_economic_repair,
         "--schedule-hiring-economic-repair"),
        (starvation_workload_visibility_repair,
         "--starvation-workload-visibility-repair"),
    ):
        if enabled:
            command.append(flag)
    return command


def run_sharded(
    *, checkpoint: Path | None, e_checkpoint: Path | None, seeds: Sequence[int],
    master_seed: int = 25, backend: str = "official", e_history_version: str = "E_LEGACY",
    variants: Sequence[str] = tuple(VARIANTS), output_dir: Path,
    processes: int = 4, game_filters: Sequence[str] | None = None,
    capture_dir: Path | None = None, preflight_only: bool = False,
    underfoot_first: bool = False, deadline_safe_planting: bool = False,
    deadline_safe_hiring: bool = False, persistent_worker_queues: bool = False,
    queue_ownership_repair: bool = False,
    batch_reserved_supplies: bool = False,
    underfoot_queue_insertion: bool = False,
    schedule_informed_hiring: bool = False,
    schedule_hiring_economic_repair: bool = False,
    starvation_workload_visibility_repair: bool = False,
    resume: bool = False,
    popen_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run children and merge outputs; raises while retaining failed artifacts."""
    seeds, game_pairs = plan_game_pairs(seeds, game_filters)
    variants = list(variants)
    queue_ownership_repair = bool(
        queue_ownership_repair and persistent_worker_queues)
    batch_reserved_supplies = bool(
        batch_reserved_supplies and queue_ownership_repair)
    underfoot_queue_insertion = bool(
        underfoot_queue_insertion and queue_ownership_repair)
    starvation_workload_visibility_repair = bool(
        starvation_workload_visibility_repair)
    schedule_hiring_economic_repair = bool(schedule_hiring_economic_repair)
    _validate_config(seeds, variants, master_seed, processes)
    if backend not in ("fast", "official"):
        raise ValueError("backend must be fast or official")
    if e_history_version not in ("E_LEGACY", "E_CORRECTED_V1"):
        raise ValueError("unsupported e-history-version")
    if not game_pairs:
        raise ValueError("game filter selected no games")
    shards = shard_game_pairs(game_pairs, processes)
    checkpoint = checkpoint.resolve() if checkpoint else None
    e_checkpoint = e_checkpoint.resolve() if e_checkpoint else None
    if not preflight_only:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        if e_checkpoint is None or not e_checkpoint.is_file():
            raise FileNotFoundError(f"missing e-checkpoint: {e_checkpoint}")
        try:
            official_available = importlib.util.find_spec("kaggle_environments") is not None
        except (ImportError, AttributeError, ValueError):
            official_available = False
        if backend == "official" and not official_available:
            raise RuntimeError("official backend provenance check failed: kaggle_environments is unavailable")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"cannot resume without manifest: {manifest_path}")
        prior_manifest = json.loads(manifest_path.read_text())
        prior_config = prior_manifest.get("config") or {}
        expected_config = {
            "seeds": list(seeds), "master_seed": master_seed,
            "variants": variants, "backend": backend,
            "e_history_version": e_history_version,
            "processes": processes, "game_filters": (
                list(game_filters) if game_filters is not None else None),
            "underfoot_first": underfoot_first,
            "deadline_safe_planting": deadline_safe_planting,
            "deadline_safe_hiring": deadline_safe_hiring,
            "persistent_worker_queues": persistent_worker_queues,
            "queue_ownership_repair": queue_ownership_repair,
            "batch_reserved_supplies": batch_reserved_supplies,
            "underfoot_queue_insertion": underfoot_queue_insertion,
            "schedule_informed_hiring": schedule_informed_hiring,
            "schedule_hiring_economic_repair": schedule_hiring_economic_repair,
            "starvation_workload_visibility_repair": (
                starvation_workload_visibility_repair),
        }
        mismatches = {
            key: (prior_config.get(key), value)
            for key, value in expected_config.items()
            if prior_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        prior_checkpoints = prior_manifest.get("checkpoints") or {}
        current_checkpoints = {
            "ppo_sha256": _sha256(checkpoint),
            "bc_e_sha256": _sha256(e_checkpoint),
        }
        if prior_checkpoints != current_checkpoints:
            raise ValueError(
                "resume checkpoint hash mismatch: "
                f"{prior_checkpoints} != {current_checkpoints}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if capture_dir is not None:
        capture_dir = capture_dir.resolve()
        if capture_dir.exists():
            if not resume:
                raise FileExistsError(f"capture directory already exists: {capture_dir}")
        else:
            capture_dir.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[1]
    wrapper_path = Path(__file__).resolve()
    evaluator_path = repo_root / "tools" / "evaluate_stage25_upkeep.py"
    manifest = _config_manifest(
        repo_root=repo_root, wrapper_path=wrapper_path, evaluator_path=evaluator_path,
        checkpoint=checkpoint, e_checkpoint=e_checkpoint, seeds=seeds,
        master_seed=master_seed, variants=variants, backend=backend,
        e_history_version=e_history_version, processes=processes,
        game_filters=game_filters, game_pairs=game_pairs,
        preflight_only=preflight_only, capture_dir=capture_dir,
        underfoot_first=underfoot_first,
        deadline_safe_planting=deadline_safe_planting,
        deadline_safe_hiring=deadline_safe_hiring,
        persistent_worker_queues=persistent_worker_queues,
        queue_ownership_repair=queue_ownership_repair,
        batch_reserved_supplies=batch_reserved_supplies,
        underfoot_queue_insertion=underfoot_queue_insertion,
        schedule_informed_hiring=schedule_informed_hiring,
        schedule_hiring_economic_repair=(
            schedule_hiring_economic_repair),
        starvation_workload_visibility_repair=(
            starvation_workload_visibility_repair),
    )
    manifest["shards"] = [
        {"index": shard.index, "games": [
            {"seed_index": index, "seed": seeds[index], "seat": seat,
             "episode_id": episode_id_for(master_seed, len(seeds), index, seat)}
            for index, seat in shard.games
        ]}
        for shard in shards
    ]
    manifest["coverage"] = {
        "expected_identities": len(expected_identities(seeds, master_seed, variants, game_pairs)),
        "assigned_identities": sum(len(item["games"]) for item in manifest["shards"]) * len(variants),
        "complete": True,
    }
    _json_write(output_dir / "manifest.json", manifest)
    _json_write(output_dir / "preflight.json", {
        "schema_version": SCHEMA_VERSION, "ordered_seeds": seeds,
        "master_seed": master_seed, "variants": variants,
        "expected_episode_ids": sorted(
            {episode for _, _, _, episode in expected_identities(seeds, master_seed, variants, game_pairs)}),
        "shards": manifest["shards"], "coverage": manifest["coverage"],
        "checkpoint_loading": not preflight_only,
    })
    if preflight_only:
        manifest["status"] = "preflight_only"
        _json_write(output_dir / "manifest.json", manifest)
        return manifest

    shard_root = output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    processes_by_shard: dict[int, Any] = {}
    logs: dict[int, tuple[Any, Any]] = {}
    reused_shards: set[int] = set()
    factory = popen_factory or subprocess.Popen
    try:
        for shard in shards:
            child_output = shard_root / f"shard_{shard.index}"
            if resume and child_output.is_dir():
                try:
                    existing_rows = _read_rows(child_output / "games.jsonl")
                    merge_rows(
                        {shard.index: existing_rows}, seeds=seeds,
                        master_seed=master_seed, variants=variants,
                        game_pairs=shard.games, shard_plans=[shard])
                except (FileNotFoundError, ValueError, json.JSONDecodeError):
                    # A partial child cannot be appended to: the evaluator
                    # intentionally opens games.jsonl in append mode.  Remove
                    # only this controlled shard directory before retrying.
                    shutil.rmtree(child_output)
                else:
                    reused_shards.add(shard.index)
                    continue
            elif resume and child_output.exists():
                child_output.unlink()
            child_capture = capture_dir / f"shard_{shard.index}" if capture_dir else None
            command = _child_command(
                checkpoint=checkpoint, e_checkpoint=e_checkpoint, output_dir=child_output,
                capture_dir=child_capture, seeds=seeds, master_seed=master_seed,
                variants=variants, backend=backend, e_history_version=e_history_version,
                game_pairs=shard.games,
                underfoot_first=underfoot_first,
                deadline_safe_planting=deadline_safe_planting,
                deadline_safe_hiring=deadline_safe_hiring,
                persistent_worker_queues=persistent_worker_queues,
                queue_ownership_repair=queue_ownership_repair,
                batch_reserved_supplies=batch_reserved_supplies,
                underfoot_queue_insertion=underfoot_queue_insertion,
                schedule_informed_hiring=schedule_informed_hiring,
                schedule_hiring_economic_repair=(
                    schedule_hiring_economic_repair),
                starvation_workload_visibility_repair=(
                    starvation_workload_visibility_repair),
            )
            shard_manifest = {
                "schema_version": SCHEMA_VERSION, "shard_index": shard.index,
                "ordered_seeds": seeds, "variants": variants,
                "master_seed": master_seed, "command": command,
                "games": manifest["shards"][shard.index]["games"],
                "config": manifest["config"],
                "source": manifest["source"],
                "patch": manifest["patch"],
                "checkpoints": manifest["checkpoints"],
                "engine": manifest["engine"],
                "status": "launched",
            }
            _json_write(shard_root / f"shard_{shard.index}.manifest.json", shard_manifest)
            stdout = (shard_root / f"shard_{shard.index}.stdout.log").open("w")
            stderr = (shard_root / f"shard_{shard.index}.stderr.log").open("w")
            logs[shard.index] = (stdout, stderr)
            processes_by_shard[shard.index] = factory(
                command, cwd=repo_root, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, text=True,
            )
            print(
                f"launched shard {shard.index}: {len(shard.games)} seed/seat pairs",
                flush=True)
    except Exception:
        for stdout, stderr in logs.values():
            stdout.close()
            stderr.close()
        manifest["status"] = "launch_failed"
        _json_write(output_dir / "manifest.json", manifest)
        raise

    return_codes: dict[int, int] = {}
    for index, process in processes_by_shard.items():
        return_codes[index] = int(process.wait())
        stdout, stderr = logs[index]
        stdout.close()
        stderr.close()
        print(f"completed shard {index}: returncode={return_codes[index]}", flush=True)
    failed = {index: code for index, code in return_codes.items() if code != 0}
    if failed:
        manifest["status"] = "child_failed"
        manifest["child_returncodes"] = return_codes
        _json_write(output_dir / "manifest.json", manifest)
        raise RuntimeError(
            f"stage25 shard child failure(s) {failed}; partial outputs preserved under {output_dir}")

    # The real evaluator writes its own manifest.  Add the wrapper's complete
    # provenance/assignment record to it; the fallback also makes fake or
    # early-exit child runners leave a manifest at the conventional path.
    for shard in shards:
        sibling_path = shard_root / f"shard_{shard.index}.manifest.json"
        shard_manifest = json.loads(sibling_path.read_text())
        shard_manifest["status"] = "completed"
        child_output = shard_root / f"shard_{shard.index}"
        child_manifest_path = child_output / "manifest.json"
        if child_manifest_path.is_file():
            child_manifest = json.loads(child_manifest_path.read_text())
        else:
            child_manifest = dict(shard_manifest)
        child_manifest["shard_wrapper_manifest"] = shard_manifest
        _json_write(child_manifest_path, child_manifest)
        _json_write(sibling_path, shard_manifest)

    shard_rows = {
        shard.index: _read_rows(shard_root / f"shard_{shard.index}" / "games.jsonl")
        for shard in shards
    }
    rows = merge_rows(
        shard_rows, seeds=seeds, master_seed=master_seed, variants=variants,
        game_pairs=game_pairs, shard_plans=shards,
    )
    with (output_dir / "games.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    telemetry = aggregate_telemetry(rows)
    _json_write(output_dir / "telemetry.json", telemetry)
    bootstrap_groups = _bootstrap_groups(rows, seeds, variants)
    _json_write(output_dir / "bootstrap_groups.json", bootstrap_groups)
    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        variant_summaries[variant] = _summary(variant_rows, variant, manifest)
        variant_summaries[variant]["telemetry"] = aggregate_telemetry(variant_rows)
        _json_write(output_dir / f"{variant}.json", variant_summaries[variant])
    comparison = _comparison(rows, variants)
    _json_write(output_dir / "comparison.json", comparison)
    _json_write(output_dir / "summary.json", {
        "schema_version": SCHEMA_VERSION, "ordered_seeds": seeds,
        "variants": variant_summaries, "comparison": comparison,
        "telemetry": telemetry, "bootstrap_groups": bootstrap_groups,
    })
    manifest["status"] = "complete"
    manifest["child_returncodes"] = return_codes
    manifest["reused_shards"] = sorted(reused_shards)
    manifest["merged_games"] = len(rows)
    manifest["telemetry_fields"] = telemetry["fields"]
    _json_write(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--e-checkpoint", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--master-seed", type=int, default=25)
    parser.add_argument("--backend", choices=("fast", "official"), default="official")
    parser.add_argument("--e-history-version", choices=("E_LEGACY", "E_CORRECTED_V1"), default="E_LEGACY")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--game-filters", "--game-filter", dest="game_filters",
                        nargs="*", default=None, metavar="SEED:SEAT")
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--underfoot-first", action="store_true")
    parser.add_argument("--deadline-safe-planting", action="store_true")
    parser.add_argument("--deadline-safe-hiring", action="store_true")
    parser.add_argument("--persistent-worker-queues", action="store_true")
    parser.add_argument("--queue-ownership-repair", action="store_true",
                        help="candidate-only repair; effective only with persistent worker queues")
    parser.add_argument("--batch-reserved-supplies", action="store_true",
                        help="candidate-only repair; effective with queue ownership repair")
    parser.add_argument("--underfoot-queue-insertion", action="store_true",
                        help="candidate-only repair; effective with queue ownership repair")
    parser.add_argument("--schedule-informed-hiring", action="store_true")
    parser.add_argument("--schedule-hiring-economic-repair", action="store_true",
                        help="candidate-only repair; meaningful with schedule-informed hiring")
    parser.add_argument("--starvation-workload-visibility-repair", action="store_true",
                        help="candidate-only repair")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.preflight_only and (args.checkpoint is None or args.e_checkpoint is None):
        _parser().error("--checkpoint and --e-checkpoint are required unless --preflight-only is used")
    try:
        run_sharded(
            checkpoint=args.checkpoint, e_checkpoint=args.e_checkpoint,
            seeds=args.seeds, master_seed=args.master_seed, backend=args.backend,
            e_history_version=args.e_history_version, variants=args.variants,
            output_dir=args.output_dir, processes=args.processes,
            game_filters=args.game_filters, capture_dir=args.capture_dir,
            preflight_only=args.preflight_only,
            underfoot_first=args.underfoot_first,
            deadline_safe_planting=args.deadline_safe_planting,
            deadline_safe_hiring=args.deadline_safe_hiring,
            persistent_worker_queues=args.persistent_worker_queues,
            queue_ownership_repair=args.queue_ownership_repair,
            batch_reserved_supplies=args.batch_reserved_supplies,
            underfoot_queue_insertion=args.underfoot_queue_insertion,
            schedule_informed_hiring=args.schedule_informed_hiring,
            schedule_hiring_economic_repair=(
                args.schedule_hiring_economic_repair),
            starvation_workload_visibility_repair=(
                args.starvation_workload_visibility_repair),
            resume=args.resume,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
