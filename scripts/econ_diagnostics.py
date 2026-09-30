"""Economic waste report from one real full game.

MEASURE ONLY. Runs a single fixed-seed full game, captures the retained
per-day executor diagnostics for both seats, and rolls them up into the
quantities that decide whether the executor is realizing the manager's plan
mechanically: interactions completed vs. planned, routes finished vs. started,
idle workers, pass/movement-only turn tails, unmet supply demand, and market
cash behaviour.

No behaviour is changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import parallel_full_game_validation as harness  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\stage25_bc_7m_best_inference.npz")


def walk_keys(value: Any, prefix: str = "", depth: int = 0,
              out: dict[str, str] | None = None) -> dict[str, str]:
    if out is None:
        out = {}
    if depth > 2:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out[path] = type(item).__name__
            if isinstance(item, (dict, list)) and depth < 2:
                walk_keys(item, path, depth + 1, out)
    elif isinstance(value, list) and value:
        walk_keys(value[0], f"{prefix}[0]", depth + 1, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=41003)
    parser.add_argument("--opening", default=harness.OPENING_NAME)
    parser.add_argument("--row-claim", action="store_true")
    parser.add_argument("--show-keys", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import rl_manager.executor_factory as factory_module

    agents: dict[int, Any] = {}
    original_init = factory_module.Stage25StripExecutorAgent.__init__

    def patched_init(self, **kwargs):
        original_init(self, **kwargs)
        agents.setdefault(self.seat, self)

    factory_module.Stage25StripExecutorAgent.__init__ = patched_init
    checkpoint = harness._resolve_checkpoint_path(args.checkpoint)
    checkpoint_sha = harness._sha256_file(checkpoint)
    try:
        record = harness._run_full_game_task(
            (0, args.seed, 0), str(checkpoint), checkpoint_sha,
            enable_row_claim_board=args.row_claim,
            opponent=harness.OPPONENT_SYMETRIC,
            opening_name=args.opening,
        )
    finally:
        factory_module.Stage25StripExecutorAgent.__init__ = original_init

    print(f"seed={args.seed} banks={record['final_banks']} "
          f"mean={statistics.fmean(record['final_banks']):.1f} "
          f"digest={str(record.get('trace_digest'))[:16]}")

    report: dict[str, Any] = {
        "seed": args.seed, "final_banks": [int(v) for v in record["final_banks"]],
        "seats": {},
    }
    for seat, agent in sorted(agents.items()):
        document = agent.diagnostics_json()
        days = document.get("days", {})
        if args.show_keys and days:
            first = next(iter(days.values()))
            print(f"\n=== seat {seat} day-document keys ===")
            for path, kind in sorted(walk_keys(first).items()):
                print(f"  {kind:<6} {path}")
        report["seats"][str(seat)] = {
            "day_count": len(days),
            "effective_profile": document.get("effective_profile"),
            "days": days,
        }

    # economic rollup
    for seat, payload in report["seats"].items():
        days = payload["days"]
        if not days:
            continue
        print(f"\n=== seat {seat} economic rollup "
              f"({len(days)} manager days) ===")
        totals: Counter[str] = Counter()
        per_day: list[tuple[int, int, int, int, int, int]] = []
        for key in sorted(days, key=lambda k: int(k)):
            day = days[key]
            routes = day.get("route_diagnostics") or []
            done = int(day.get("completed_routes", 0) or 0)
            unfinished = int(day.get("unfinished_routes", 0) or 0)
            idle = len(day.get("idle_workers") or ())
            interactions = int(day.get("actual_interactions_completed", 0) or 0)
            pass_tail = int(day.get("actual_pass_tail_turns", 0) or 0)
            movement = int(
                day.get("actual_final_movement_only_turns", 0) or 0)
            per_day.append((int(key), len(routes), done, unfinished, idle,
                            interactions))
            totals["routes_seen"] += len(routes)
            totals["routes_done"] += done
            totals["routes_unfinished"] += unfinished
            totals["idle_worker_slots"] += idle
            totals["interactions"] += interactions
            totals["pass_tail_turns"] += pass_tail
            totals["movement_only_turns"] += movement
            for route in routes:
                totals["route_interaction_turns"] += int(
                    route.get("interaction_turns", 0) or 0)
                totals["route_movement_turns"] += int(
                    route.get("movement_turns", 0) or 0)
                totals["route_pass_turns"] += int(
                    route.get("pass_turns", 0) or 0)
                phase = str(route.get("phase", "?"))
                totals[f"phase_{phase}"] += 1
                totals[f"phase_{phase}_interaction_turns"] += int(
                    route.get("interaction_turns", 0) or 0)
        print(f"  routes seen {totals['routes_seen']}, done "
              f"{totals['routes_done']}, unfinished "
              f"{totals['routes_unfinished']} "
              f"({100 * totals['routes_unfinished'] / max(1, totals['routes_seen']):.1f}%)")
        print(f"  interactions completed {totals['interactions']:,}")
        print(f"  route turn split: interaction "
              f"{totals['route_interaction_turns']:,}, movement "
              f"{totals['route_movement_turns']:,}, pass "
              f"{totals['route_pass_turns']:,}")
        if totals["route_interaction_turns"]:
            print(f"  movement+pass overhead: "
                  f"{100 * (totals['route_movement_turns'] + totals['route_pass_turns']) / totals['route_interaction_turns']:.1f}% "
                  f"of interaction turns")
        print(f"  idle worker slots {totals['idle_worker_slots']:,}")
        print("  route phases: " + ", ".join(
            f"{k[6:]}={v}" for k, v in sorted(totals.items())
            if k.startswith("phase_") and not k.endswith("_interaction_turns")))
        print(f"  pass-tail turns {totals['pass_tail_turns']:,}, "
              f"final movement-only {totals['movement_only_turns']:,}")
        worst = sorted(per_day, key=lambda r: -r[3])[:6]
        print("  days with most unfinished routes: " + ", ".join(
            f"d{d[0]}({d[1]}r/{d[3]}unf/{d[4]}idle/{d[5]}int)"
            for d in worst))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
