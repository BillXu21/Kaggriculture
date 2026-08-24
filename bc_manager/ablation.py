"""Four-variant closed-loop panel for issue #6 (V0/J/E/JE).

Runs the bounded ablation panel through the existing official-engine
harness: `standard_mixed` days 0-3 literal opening -> checkpoint BC manager
-> unchanged ExecutorAgent, paired across the exact diagnostic seeds
7/17/42/123/2026 and both seats with a PASS opponent. One fresh downstream
provider/executor per variant x seed x seat; identical configuration
everywhere; no baseline-from-day0 arm (the pairing is across variants on
identical seed/seat under the mandated opening).

Strict evidence rules:

- every checkpoint must exist, carry format ``bc_manager_checkpoint_v1``,
  store a top-level ``model_variant`` matching its mapping slot, and include
  teacher-forced ``validation_metrics`` (prerequisite evidence, never the
  winner criterion);
- any game that fails opening acceptance, diverges, or shows a status
  anomaly fails the whole panel loudly (status "failed"/"partial"; a partial
  artifact is written only with that explicit status);
- model selection ranks variants by CLOSED-LOOP FINAL BANK median, then
  mean. Teacher-forced metrics and coherence diagnostics are reported as
  prerequisites/diagnostics only. The five seeds are a diagnostic panel,
  never a tuning leaderboard.

No real training happens here; without real checkpoints the panel can only
``--validate-only`` preflight or fail loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from bc_manager.economics import MODEL_VARIANTS, normalize_model_variant
from bc_manager.training import checkpoint_model_variant, load_checkpoint
from executor_v0.smoke import detect_engine

__all__ = [
    "PANEL_VARIANTS",
    "DEFAULT_SEEDS",
    "DEFAULT_SEATS",
    "SUPPORTED_OPENINGS",
    "SEED17_COLLAPSE_BANK_THRESHOLD",
    "load_panel_checkpoint",
    "run_panel",
    "aggregate_variant_games",
    "rank_variants",
    "main",
]

PANEL_VARIANTS = MODEL_VARIANTS  # exactly V0/J/E/JE
DEFAULT_SEEDS = (7, 17, 42, 123, 2026)
DEFAULT_SEATS = (0, 1)
SUPPORTED_OPENINGS = ("standard_mixed",)
SUPPORTED_OPPONENTS = ("pass",)

# Issue #6 reports seed 17 collapsing to single-digit/zero cash but pins no
# numeric threshold, so the collapse flag uses this transparent heuristic
# cutoff AND always reports the raw final bank next to it.
SEED17_COLLAPSE_BANK_THRESHOLD = 100.0

EXIT_OK = 0
EXIT_PANEL_FAILURE = 1
EXIT_USAGE = 2
EXIT_ENGINE_UNAVAILABLE = 3


# ------------------------------------------------------------- checkpoints


def load_panel_checkpoint(path: str | Path, expected_variant: str) -> dict:
    """Strictly validate one panel checkpoint; return its evidence summary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{expected_variant} checkpoint not found: {path}; supply real "
            f"trained checkpoints — smoke weights are never substituted")
    payload = load_checkpoint(path)  # validates format bc_manager_checkpoint_v1
    variant = checkpoint_model_variant(payload)
    if variant != normalize_model_variant(expected_variant):
        raise ValueError(
            f"{path}: checkpoint stores model_variant {variant!r} but the "
            f"panel mapping expects {expected_variant!r}")
    metrics = payload.get("validation_metrics") or {}
    if "total" not in metrics:
        raise ValueError(
            f"{path}: checkpoint lacks teacher-forced validation_metrics."
            f"total; run/attach the real validation before paneling")
    return {
        "path": str(path),
        "variant": variant,
        "epoch": payload.get("epoch"),
        "validation_total": float(metrics["total"]),
    }


def parse_checkpoint_mapping(pairs) -> dict[str, str]:
    """Parse VARIANT=PATH strings into a complete {variant: path} mapping."""
    mapping: dict[str, str] = {}
    for item in pairs:
        variant, sep, path = str(item).partition("=")
        if not sep or not path:
            raise ValueError(
                f"checkpoint mapping {item!r} must be VARIANT=PATH")
        variant = normalize_model_variant(variant)
        if variant in mapping:
            raise ValueError(f"duplicate checkpoint for variant {variant}")
        mapping[variant] = path
    missing = [v for v in PANEL_VARIANTS if v not in mapping]
    if missing:
        raise ValueError(
            f"checkpoint mapping incomplete; missing variants: {missing}")
    unknown = sorted(set(mapping) - set(PANEL_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants in mapping: {unknown}")
    return mapping


# ------------------------------------------------------------ game records


def summarize_game(variant: str, record: Mapping[str, Any]) -> dict:
    """Compact per-game panel record from one run_opening_game result."""
    seat = int(record["seat"])
    banks = record.get("final_banks") or []
    bank_self = float(banks[seat]) if len(banks) > seat else None
    opening_diag = record["opening_diagnostics"]
    downstream = record.get("downstream_diagnostics") or {}
    provider = downstream.get("provider_diagnostics") or {}

    unfinished_total = missed_total = 0
    for day_record in (downstream.get("days") or {}).values():
        unfinished_total += len(day_record.get("unfinished_tasks") or [])
        missed_total += len(day_record.get("missed_maintenance") or [])

    return {
        "variant": variant,
        "seed": int(record["seed"]),
        "seat": seat,
        "passed": bool(record["passed"]),
        "failure_reasons": list(record["failure_reasons"]),
        "final_bank_self": bank_self,
        "final_banks": banks,
        "opening_diverged": bool(opening_diag["divergence"]["occurred"]),
        "opening_fallback_active": bool(opening_diag["fallback_active"]),
        "status_anomaly_count": len(record.get("status_anomalies") or []),
        "fallback_error_count": len(downstream.get("fallback_errors") or []),
        "unfinished_tasks_total": unfinished_total,
        "missed_maintenance_total": missed_total,
        "coherence_aggregate": provider.get("aggregate") or {"days": 0},
    }


def aggregate_variant_games(games, *, v0_seed2026_bank: float | None = None) \
        -> dict:
    """Aggregate one variant's games; banks drive selection, nothing else."""
    banks = [g["final_bank_self"] for g in games
             if g["final_bank_self"] is not None]
    summary: dict[str, Any] = {
        "games": len(games),
        "banks": banks,
        "bank_mean": float(sum(banks) / len(banks)) if banks else None,
        "bank_median": float(np.median(banks)) if banks else None,
        "divergence_count": sum(1 for g in games if g["opening_diverged"]),
        "fallback_error_count": sum(g["fallback_error_count"] for g in games),
        "status_anomaly_count": sum(g["status_anomaly_count"]
                                    for g in games),
        "unfinished_tasks_total": sum(g["unfinished_tasks_total"]
                                      for g in games),
        "missed_maintenance_total": sum(g["missed_maintenance_total"]
                                        for g in games),
    }
    coherence_days = [g["coherence_aggregate"] for g in games
                      if g["coherence_aggregate"].get("days")]
    summary["closed_loop_coherence_days"] = sum(
        c["days"] for c in coherence_days)
    summary["coherence_over_1x_rate"] = (
        float(sum(c["over_1x_rate"] * c["days"] for c in coherence_days)
              / summary["closed_loop_coherence_days"])
        if coherence_days else None)
    summary["coherence_over_2x_rate"] = (
        float(sum(c["over_2x_rate"] * c["days"] for c in coherence_days)
              / summary["closed_loop_coherence_days"])
        if coherence_days else None)

    def _seed_game(seed: int) -> dict | None:
        matches = [g for g in games if g["seed"] == seed]
        return matches[0] if matches else None

    seed17 = _seed_game(17)
    if seed17 is not None and seed17["final_bank_self"] is not None:
        summary["seed17_final_bank"] = seed17["final_bank_self"]
        summary["seed17_collapse"] = bool(
            seed17["final_bank_self"] < SEED17_COLLAPSE_BANK_THRESHOLD)
        summary["seed17_collapse_threshold"] = SEED17_COLLAPSE_BANK_THRESHOLD
    seed2026 = _seed_game(2026)
    if seed2026 is not None and seed2026["final_bank_self"] is not None:
        summary["seed2026_final_bank"] = seed2026["final_bank_self"]
        if v0_seed2026_bank is not None and v0_seed2026_bank > 0:
            summary["seed2026_upside_retention_vs_v0"] = float(
                seed2026["final_bank_self"] / v0_seed2026_bank)
    return summary


def rank_variants(variant_summaries: Mapping[str, Mapping[str, Any]]) \
        -> list[dict]:
    """Selection rule: closed-loop final-bank median desc, then mean desc.

    Teacher-forced validation totals and coherence aggregates are carried
    alongside as prerequisites/diagnostics and are NOT ranking inputs.
    """
    ranked = sorted(
        variant_summaries.items(),
        key=lambda item: (
            -(item[1]["bank_median"]
              if item[1]["bank_median"] is not None else float("-inf")),
            -(item[1]["bank_mean"]
              if item[1]["bank_mean"] is not None else float("-inf")),
        ))
    return [{
        "rank": i + 1,
        "variant": variant,
        "bank_median": summary["bank_median"],
        "bank_mean": summary["bank_mean"],
        "selection_criterion": "closed_loop_final_bank_median_then_mean",
    } for i, (variant, summary) in enumerate(ranked)]


# ------------------------------------------------------------------ runner


def run_panel(engine_make: Callable[..., Any], *, checkpoints: Mapping[str, str],
              opening: str = "standard_mixed",
              seeds: tuple[int, ...] = DEFAULT_SEEDS,
              seats: tuple[int, ...] = DEFAULT_SEATS,
              opponent: str = "pass", device: str = "cpu") -> dict:
    """Run the full variant x seed x seat matrix; fail loudly on any gap."""
    from opening_book.eval import make_checkpoint_downstream_factory, \
        run_opening_game

    if opening not in SUPPORTED_OPENINGS:
        raise ValueError(
            f"unsupported panel opening {opening!r}; supported: "
            f"{list(SUPPORTED_OPENINGS)}")
    if opponent not in SUPPORTED_OPPONENTS:
        raise ValueError(
            f"unsupported panel opponent {opponent!r}; supported: "
            f"{list(SUPPORTED_OPPONENTS)}")
    mapping = parse_checkpoint_mapping(
        [f"{v}={checkpoints[v]}" for v in checkpoints])
    summaries = {v: load_panel_checkpoint(mapping[v], v)
                 for v in PANEL_VARIANTS}

    games: list[dict] = []
    errors: list[dict] = []
    for variant in PANEL_VARIANTS:
        for seed in seeds:
            for seat in seats:
                factory = make_checkpoint_downstream_factory(
                    mapping[variant], device, seat)
                try:
                    record = run_opening_game(
                        engine_make, opening=opening, seat=seat, seed=seed,
                        opponent_kind=opponent, downstream_factory=factory,
                        mode=f"panel_{variant}")
                except Exception as exc:  # noqa: BLE001 - recorded, then loud
                    errors.append({
                        "variant": variant, "seed": seed, "seat": seat,
                        "error_type": type(exc).__name__, "message": str(exc),
                    })
                    continue
                games.append(summarize_game(variant, record))

    expected_games = len(PANEL_VARIANTS) * len(seeds) * len(seats)
    failed_games = [g for g in games if not g["passed"]]
    status = "complete"
    if errors or failed_games or len(games) != expected_games:
        status = "failed" if errors or len(games) < expected_games \
            else "partial"

    v0_seed2026 = next((g["final_bank_self"] for g in games
                        if g["variant"] == "V0" and g["seed"] == 2026), None)
    by_variant = {
        variant: aggregate_variant_games(
            [g for g in games if g["variant"] == variant],
            v0_seed2026_bank=v0_seed2026)
        for variant in PANEL_VARIANTS
    }
    for variant, summary in zip(PANEL_VARIANTS,
                                [by_variant[v] for v in PANEL_VARIANTS]):
        summary["teacher_forced_validation_total"] = \
            summaries[variant]["validation_total"]

    return {
        "status": status,
        "selection_rule": ("closed-loop final bank median, then mean; "
                           "teacher-forced metrics and coherence are "
                           "prerequisites/diagnostics, never the criterion"),
        "panel": {
            "opening": opening,
            "seeds": list(seeds),
            "seats": list(seats),
            "opponent": opponent,
            "device": device,
            "expected_games": expected_games,
            "completed_games": len(games),
        },
        "checkpoints": summaries,
        "errors": errors,
        "failed_games": [
            {"variant": g["variant"], "seed": g["seed"], "seat": g["seat"],
             "failure_reasons": g["failure_reasons"]}
            for g in failed_games
        ],
        "games": games,
        "by_variant": by_variant,
        "ranking": rank_variants(by_variant),
    }


# ---------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_manager.ablation",
        description="Four-variant (V0/J/E/JE) closed-loop panel: "
                    "standard_mixed days 0-3 -> checkpoint BC manager -> "
                    "unchanged ExecutorAgent, seeds 7/17/42/123/2026, both "
                    "seats, pass opponent, official engine. Selection ranks "
                    "closed-loop final-bank median then mean; teacher-forced "
                    "metrics/coherence are prerequisites/diagnostics only.")
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="VARIANT=PATH", dest="checkpoints",
                        help="checkpoint per variant, e.g. "
                             "--checkpoint V0=/kaggle/working/v0/best.pt; "
                             "repeat for all of V0,J,E,JE (real trained "
                             "checkpoints with validation_metrics; smoke "
                             "weights are rejected)")
    parser.add_argument("--opening", choices=SUPPORTED_OPENINGS,
                        default="standard_mixed")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(DEFAULT_SEEDS),
                        help="exact diagnostic seeds (default: 7 17 42 123 "
                             "2026)")
    parser.add_argument("--seats", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="seats to pair (default: 0 1)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="JSON artifact output path (default stdout)")
    parser.add_argument("--validate-only", action="store_true",
                        help="preflight only: validate checkpoints/config/"
                             "matrix completeness without importing or "
                             "running the engine")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        mapping = parse_checkpoint_mapping(args.checkpoints)
        summaries = {v: load_panel_checkpoint(mapping[v], v)
                     for v in PANEL_VARIANTS}
        seeds = tuple(int(s) for s in args.seeds)
        seats = tuple(int(s) for s in args.seats)
        if not seeds or not seats:
            raise ValueError("at least one seed and one seat are required")
        if sorted(seats) != sorted(set(seats)) or \
                any(s not in (0, 1) for s in seats):
            raise ValueError("seats must be 0 and/or 1")
    except (ValueError, FileNotFoundError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.validate_only:
        report = {
            "status": "validated",
            "mode": "validate_only",
            "panel": {"opening": args.opening, "seeds": seeds,
                      "seats": seats, "expected_games":
                          len(PANEL_VARIANTS) * len(seeds) * len(seats)},
            "checkpoints": summaries,
        }
        _emit(report, args.out)
        return EXIT_OK

    engine = detect_engine()
    if not engine["available"]:
        print(f"engine unavailable: {engine['reason']}", file=sys.stderr)
        return EXIT_ENGINE_UNAVAILABLE
    from oracle.provenance import ProvenanceError, verify_official_provenance
    try:
        provenance = verify_official_provenance()
    except ProvenanceError as exc:
        print(f"provenance mismatch: {exc}", file=sys.stderr)
        return EXIT_ENGINE_UNAVAILABLE

    from opening_book.eval import _official_runner
    report = run_panel(_official_runner(), checkpoints=mapping,
                       opening=args.opening, seeds=seeds, seats=seats,
                       device=args.device)
    report["provenance"] = provenance
    report["engine_version"] = engine["version"]
    _emit(report, args.out)
    return EXIT_OK if report["status"] == "complete" else EXIT_PANEL_FAILURE


def _emit(report: dict, out: str | None) -> None:
    payload = json.dumps(report, sort_keys=True, allow_nan=False)
    if out:
        Path(out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    raise SystemExit(main())
