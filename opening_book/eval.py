"""Official 1.32.7 evaluation CLI for the opening book (issue #4, stage 3).

Runs bounded opening-only and paired BC-handoff games through the pinned
official ``kaggle_environments==1.32.7`` engine behind the repository
provenance guard (``oracle.provenance.verify_official_provenance``). No
training, tuning, heuristic repair, or fast-engine fallback ever happens
here; the official engine is the only authority.

Opening-only acceptance per game:
- exactly 96 scripted turns, no divergence/fallback, clean d4h0 handoff;
- zero official pre-terminal status anomalies;
- handoff farm summary matches the structural milestones derived from the
  source replay (exact crops/animals/land) and money/shed-WHEAT fall inside
  justified source/opponent-aware tolerances (research documents same-script
  day-3 cash variance 10..120 across 17 dominant openings and 211..213 for
  pasture-heavy; ranges below add margin for live-market perturbation).

Paired BC mode compares, under identical seed/seat/opponent:
1. baseline: existing BC + executor (``executor_v0.agent.make_agent``) from d0;
2. opener: opening wrapper days 0-3 -> a FRESH identical BC + executor at d4.
A real checkpoint path is required; absence is reported, never substituted
with smoke weights. Nothing is retrained.

Exit codes: 0 pass, 1 validation failure, 2 usage error,
3 engine unavailable / provenance mismatch.

Exact Kaggle command/cell pattern for the paired comparison with the real
checkpoint (run from the repository root attached as a dataset or copied
into /kaggle/working)::

    !pip install kaggle-environments==1.32.7
    !python -m opening_book.eval --mode paired \\
        --opening standard_mixed --seat 0 --seed 1146601720 \\
        --opponent pass --downstream checkpoint --device cpu \\
        --checkpoint /kaggle/working/bc-v0-score2950/best.pt \\
        --out /kaggle/working/paired_s1146601720.jsonl

Example local opening-only runs::

    python -m opening_book.eval --opening standard_mixed --seat 0 \\
        --seeds 1146601720 1979016230 --opponent pass --out r.jsonl
    python -m opening_book.eval --opening pasture_heavy --seat 1 \\
        --seeds 95055022 --opponent mirror
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

from oracle.provenance import (
    ProvenanceError,
    verify_official_provenance,
)
from executor_v0.smoke import detect_engine

from .agent import OpeningAgent, make_opening_agent
from .trace import IDENTITIES, load_built_in_trace

__all__ = [
    "ENVELOPES",
    "evaluate_handoff_envelope",
    "main",
    "run_opening_game",
]

ENGINE_ENV_ID = "kaggriculture"

EXIT_OK = 0
EXIT_VALIDATION_FAILURE = 1
EXIT_USAGE = 2
EXIT_ENGINE_UNAVAILABLE = 3

REAL_CHECKPOINT_HINT = "/kaggle/working/bc-v0-score2950/best.pt"

# Handoff envelopes derived from the verified source replays (stage 1
# provenance) plus documented same-script variance. Structural milestones are
# exact; money/shed are tolerant ranges with recorded justification.
ENVELOPES: dict[str, dict[str, Any]] = {
    "standard_mixed": {
        "milestones": {
            "crops": {"WHEAT": 7, "MELON": 12},
            "animals": {"COW": 3, "SHEEP": 2},
            "land_count": 1,
        },
        "money_range": [0, 250],
        "shed_wheat_range": [0, 12],
        "money_justification": (
            "source episode 95515912 ends d3 at money 29; research section "
            "4.1 observes day-3 end 10..120 across 17 same-script openings "
            "under differing opponents/markets; range adds perturbation margin"
        ),
        "shed_justification": (
            "source replay holds 0 shed WHEAT at d4h0; JIT feed restock "
            "timing shifts the residual stock by a few units at most"
        ),
        "source_reference": {
            "episode": 95515912, "seat": 0, "seed": 1146601720,
            "money": 29, "shed_wheat": 0,
        },
    },
    "pasture_heavy": {
        "milestones": {
            "crops": {"WHEAT": 6, "MELON": 4, "STRAWBERRY": 3},
            "animals": {"COW": 1, "SHEEP": 4},
            "land_count": 1,
        },
        "money_range": [120, 320],
        "shed_wheat_range": [0, 16],
        "money_justification": (
            "source episodes 95055022/95481731 end d3 at money 211..213 "
            "(byte-identical script); range covers live-market perturbation"
        ),
        "shed_justification": (
            "source replay holds 8 shed WHEAT at d4h0; feed-first buying "
            "keeps the stock within a few units of the source"
        ),
        "source_reference": {
            "episode": 95055022, "seat": 0, "seed": 1979016230,
            "money": 213, "shed_wheat": 8,
        },
    },
}


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def evaluate_handoff_envelope(identity: str,
                              summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one handoff farm summary against the identity envelope."""
    envelope = ENVELOPES[identity]
    checks: list[dict[str, Any]] = []
    milestones = envelope["milestones"]

    crops = summary.get("crops") or {}
    animals = summary.get("animals") or {}
    checks.append(_check(
        "crops", crops == milestones["crops"],
        f"expected {milestones['crops']}, got {crops}"))
    checks.append(_check(
        "animals", animals == milestones["animals"],
        f"expected {milestones['animals']}, got {animals}"))
    land = summary.get("land_count")
    checks.append(_check(
        "land_count", land == milestones["land_count"],
        f"expected {milestones['land_count']}, got {land!r}"))

    money = summary.get("money")
    lo, hi = envelope["money_range"]
    checks.append(_check(
        "money", isinstance(money, (int, float)) and lo <= money <= hi,
        f"expected within [{lo}, {hi}] ({envelope['money_justification']}), "
        f"got {money!r}"))
    shed = summary.get("shed_wheat")
    slo, shi = envelope["shed_wheat_range"]
    checks.append(_check(
        "shed_wheat",
        shed is None or (isinstance(shed, (int, float)) and slo <= shed <= shi),
        f"expected within [{slo}, {shi}] or unknown "
        f"({envelope['shed_justification']}), got {shed!r}"))

    failed = [c for c in checks if not c["ok"]]
    return {"identity": identity, "ok": not failed,
            "checks": checks, "failed_reasons": [c["detail"] for c in failed]}


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

def pass_action(obs: Any) -> dict[str, Any]:
    """Legal-shaped all-PASS action sized to the acting seat's hands."""
    try:
        seat = int(obs["player"])
        hands = len(obs["farms"][seat].get("hands") or [])
    except Exception:  # noqa: BLE001 - PASS responder must never raise
        hands = 0
    return {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}


def make_pass_downstream() -> Callable[[Any], dict[str, Any]]:
    return pass_action


def make_checkpoint_downstream_factory(
        checkpoint: str, device: str, seat: int) -> Callable[[], Any]:
    """Fresh ``executor_v0.make_agent`` per game; never shares state."""
    from executor_v0.agent import make_agent

    def factory() -> Any:
        return make_agent(checkpoint=checkpoint, device=device, seat=seat)

    return factory


def adapt_one_arg(agent: Callable[[Any], Any]) -> Callable[[Any, Any], Any]:
    """Adapt a one-argument stateful agent to the official (obs, config)."""

    def callable_fn(obs, config):  # noqa: ARG001 - config unused by contract
        return agent(obs)

    return callable_fn


def build_opponents(opponent_kind: str, opening: str,
                    seat: int) -> tuple[list[Any], OpeningAgent | None]:
    """Build the official two-slot agent list inputs for the non-tested seat.

    Returns (agents_list_with_placeholder, mirror_instance_or_None); the
    caller inserts the tested wrapper at ``seat``.
    """
    if opponent_kind == "pass":
        def opponent(obs, config):  # noqa: ARG001
            return pass_action(obs)
        return [opponent, opponent], None
    if opponent_kind == "mirror":
        mirror = OpeningAgent(load_built_in_trace(opening),
                              make_pass_downstream(), 1 - seat)
        return [adapt_one_arg(mirror), adapt_one_arg(mirror)], mirror
    raise ValueError(f"unknown opponent kind {opponent_kind!r}")


# ---------------------------------------------------------------------------
# Official game execution
# ---------------------------------------------------------------------------

def _official_runner() -> Callable[..., Any]:
    """Import and return the official ``kaggle_environments.make``."""
    import kaggle_environments

    return kaggle_environments.make


def _status_anomalies(steps: Any, seats: list[int]) -> list[dict[str, Any]]:
    """Pre-terminal status anomalies for the given seats (smoke.py policy)."""
    anomalies: list[dict[str, Any]] = []
    for step_index, step_state in enumerate(steps[:-1]):
        for seat in seats:
            status = str(step_state[seat].status)
            if status not in ("ACTIVE", "INACTIVE"):
                anomalies.append({"step": step_index, "seat": seat,
                                  "status": status})
                if len(anomalies) >= 10:
                    return anomalies
    return anomalies


def _final_banks(steps: Any) -> list[float] | None:
    try:
        farms = steps[-1][0].observation.farms
        return [float(farm["money"]) for farm in farms]
    except Exception:  # noqa: BLE001 - banks are best-effort
        return None


def run_opening_game(engine_make: Callable[..., Any], *, opening: str,
                     seat: int, seed: int, opponent_kind: str,
                     downstream_factory: Callable[[], Any],
                     mode: str = "opening") -> dict[str, Any]:
    """Run one official game with the wrapper in ``seat``; return a record."""
    wrapper = make_opening_agent(opening=opening,
                                 downstream=downstream_factory(), seat=seat)
    opponents, mirror = build_opponents(opponent_kind, opening, seat)
    agents: list[Any] = [None, None]
    agents[seat] = adapt_one_arg(wrapper)
    other = 1 - seat
    agents[other] = opponents[other]

    env = engine_make(ENGINE_ENV_ID, configuration={"seed": seed})
    env.reset()
    steps = env.run(agents)

    monitored_seats = [seat] + ([other] if mirror is not None else [])
    anomalies = _status_anomalies(steps, monitored_seats)
    diag = wrapper.diagnostics_json()
    mirror_diag = mirror.diagnostics_json() if mirror is not None else None

    handoff = diag["handoff"]
    envelope = evaluate_handoff_envelope(opening, handoff["farm_summary"])

    acceptance = [
        _check("turns_replayed_96", diag["turns_replayed"] == 96,
               f"replayed {diag['turns_replayed']}"),
        _check("no_divergence", not diag["divergence"]["occurred"],
               diag["divergence"]["reason"] or "clean"),
        _check("clean_d4h0_handoff",
               handoff["turn"] == [4, 0] and handoff["clean_d4h0_handoff"],
               f"handoff turn {handoff['turn']}"),
        _check("no_status_anomalies", not anomalies,
               f"{len(anomalies)} anomalies"),
        _check("envelope", envelope["ok"],
               "; ".join(envelope["failed_reasons"]) or "within envelope"),
    ]
    if mirror_diag is not None:
        acceptance.append(_check(
            "mirror_clean",
            not mirror_diag["divergence"]["occurred"]
            and mirror_diag["turns_replayed"] == 96,
            f"mirror replayed {mirror_diag['turns_replayed']}, diverged="
            f"{mirror_diag['divergence']['occurred']}"))

    failed = [c for c in acceptance if not c["ok"]]
    return {
        "mode": mode,
        "engine_env_id": ENGINE_ENV_ID,
        "opening": opening,
        "source_provenance": diag["source_provenance"],
        "seed": seed,
        "seat": seat,
        "opponent": opponent_kind,
        "opening_diagnostics": diag,
        "mirror_diagnostics": mirror_diag,
        "status_anomalies": anomalies,
        "envelope": envelope,
        "acceptance": acceptance,
        "passed": not failed,
        "failure_reasons": [c["detail"] for c in failed],
        "final_rewards": [state.reward for state in steps[-1]],
        "final_banks": _final_banks(steps),
        "steps": len(steps),
    }


def run_paired_comparison(engine_make: Callable[..., Any], *, opening: str,
                          seat: int, seed: int, opponent_kind: str,
                          downstream_factory: Callable[[], Any]) -> dict[str, Any]:
    """Baseline (BC from d0) vs opener->same fresh BC at d4, paired settings."""
    baseline = _run_downstream_only_game(
        engine_make, seat=seat, seed=seed,
        opponent_kind=opponent_kind, downstream_factory=downstream_factory)
    opener = run_opening_game(
        engine_make, opening=opening, seat=seat, seed=seed,
        opponent_kind=opponent_kind,
        downstream_factory=downstream_factory, mode="paired_opener_to_bc")
    return {
        "mode": "paired",
        "opening": opening,
        "seed": seed,
        "seat": seat,
        "opponent": opponent_kind,
        "baseline": baseline,
        "opener": opener,
        "passed": bool(baseline["passed"]) and bool(opener["passed"]),
    }


def _run_downstream_only_game(engine_make: Callable[..., Any], *, seat: int,
                              seed: int, opponent_kind: str,
                              downstream_factory: Callable[[], Any]) -> dict[str, Any]:
    """One full game of the raw downstream agent from day 0 (baseline arm)."""
    if opponent_kind != "pass":
        raise ValueError("paired baseline supports opponent=pass only for "
                         "deterministic pairing")
    agents: list[Any] = [None, None]
    agents[seat] = adapt_one_arg(downstream_factory())

    def pass_opponent(obs, config):  # noqa: ARG001
        return pass_action(obs)

    agents[1 - seat] = pass_opponent
    env = engine_make(ENGINE_ENV_ID, configuration={"seed": seed})
    env.reset()
    steps = env.run(agents)
    anomalies = _status_anomalies(steps, [seat])
    return {
        "mode": "paired_baseline_bc_only",
        "seed": seed,
        "seat": seat,
        "opponent": opponent_kind,
        "status_anomalies": anomalies,
        "passed": not anomalies,
        "failure_reasons": [f"status anomaly {a}" for a in anomalies],
        "final_rewards": [state.reward for state in steps[-1]],
        "final_banks": _final_banks(steps),
        "steps": len(steps),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m opening_book.eval",
        description="Official 1.32.7 evaluation for the opening book. See "
                    "the module docstring for the exact Kaggle paired-BC "
                    "command pattern.",
    )
    parser.add_argument("--opening", choices=sorted(IDENTITIES),
                        default="standard_mixed")
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=None, dest="seeds",
                        action="append",
                        help="explicit seed; repeatable")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        dest="seeds",
                        help="one or more explicit seeds")
    parser.add_argument("--opponent", choices=("pass", "mirror"),
                        default="pass")
    parser.add_argument("--downstream", choices=("pass", "checkpoint"),
                        default="pass")
    parser.add_argument("--checkpoint", default=None,
                        help=f"path to the real best.pt (e.g. {REAL_CHECKPOINT_HINT})")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("opening", "paired"),
                        default="opening",
                        help="opening=wrapper validation; paired=baseline BC "
                             "vs opener->BC comparison (requires --checkpoint)")
    parser.add_argument("--out", default=None,
                        help="JSONL output path (appended across "
                             "invocations); defaults to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    seeds = args.seeds or []
    if not seeds:
        print("usage error: at least one --seed/--seeds value is required",
              file=sys.stderr)
        return EXIT_USAGE
    if args.downstream == "checkpoint":
        if not args.checkpoint:
            print("usage error: --downstream checkpoint requires --checkpoint "
                  f"PATH (real checkpoint: {REAL_CHECKPOINT_HINT}; do not "
                  "substitute smoke weights)", file=sys.stderr)
            return EXIT_USAGE
        if not os.path.isfile(args.checkpoint):
            print(f"usage error: checkpoint not found: {args.checkpoint}; "
                  f"the real BC checkpoint is expected at {REAL_CHECKPOINT_HINT}",
                  file=sys.stderr)
            return EXIT_USAGE
    if args.mode == "paired":
        if args.downstream != "checkpoint":
            print("usage error: --mode paired requires --downstream checkpoint",
                  file=sys.stderr)
            return EXIT_USAGE
        if args.opponent != "pass":
            print("usage error: --mode paired supports --opponent pass only",
                  file=sys.stderr)
            return EXIT_USAGE

    engine = detect_engine()
    if not engine["available"]:
        print(f"engine unavailable: {engine['reason']}", file=sys.stderr)
        return EXIT_ENGINE_UNAVAILABLE
    try:
        provenance = verify_official_provenance()
    except ProvenanceError as exc:
        print(f"provenance mismatch: {exc}", file=sys.stderr)
        return EXIT_ENGINE_UNAVAILABLE

    if args.downstream == "checkpoint":
        downstream_factory: Callable[[], Any] = \
            make_checkpoint_downstream_factory(args.checkpoint, args.device,
                                               args.seat)
    else:
        downstream_factory = make_pass_downstream

    out_fh = open(args.out, "a", encoding="utf-8") if args.out else None
    overall_ok = True
    try:
        runner = _official_runner()
        for seed in seeds:
            if args.mode == "paired":
                record = run_paired_comparison(
                    runner, opening=args.opening, seat=args.seat, seed=seed,
                    opponent_kind=args.opponent,
                    downstream_factory=downstream_factory)
            else:
                record = run_opening_game(
                    runner, opening=args.opening, seat=args.seat, seed=seed,
                    opponent_kind=args.opponent,
                    downstream_factory=downstream_factory)
            record["provenance"] = provenance
            record["engine_version"] = engine["version"]
            record["command"] = {
                "argv": list(argv if argv is not None else sys.argv[1:]),
                "checkpoint_path": args.checkpoint,
                "device": args.device,
            }
            payload = json.dumps(record, sort_keys=True)
            if out_fh:
                out_fh.write(payload + "\n")
            else:
                print(payload)
            overall_ok = overall_ok and record["passed"]
            status = "PASS" if record["passed"] else "FAIL"
            print(f"[{status}] opening={args.opening} seed={seed} "
                  f"seat={args.seat} opponent={args.opponent} "
                  f"mode={args.mode}", file=sys.stderr)
    finally:
        if out_fh:
            out_fh.close()
    return EXIT_OK if overall_ok else EXIT_VALIDATION_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
