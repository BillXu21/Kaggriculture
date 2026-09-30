"""Profile the strip controller on one real full game, in process.

MEASURE ONLY. Runs a single fixed-seed full game through the canonical
harness task (in process, so runtime instrumentation is visible) under
``cProfile``, then attributes CPU to executor/manager modules.

This answers "what is the current bottleneck at this commit" without needing
several games. It is a profiling tool: it never changes decisions.

Example::

    python scripts/profile_strip_controller.py --seed 41003 \
      --checkpoint C:/path/stage25_bc_7m_best_inference.npz \
      --output artifacts/overnight/profile_41003.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import parallel_full_game_validation as harness  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\stage25_bc_7m_best_inference.npz")
DEFAULT_SEED = 41003

# Modules whose cost is the strip rollout/executor path under study.
FOCUS_PREFIXES = ("executor_v0/", "rl_manager/", "fast_env/")


def module_of(function: str) -> str:
    """Module path for a pstats function label, e.g. 'executor_v0/strip_routes.py'."""
    if "(" in function:
        head = function.split("(", 1)[0]
    else:
        head = function
    tail = head.replace("\\", "/")
    for marker in ("/site-packages/", "/Lib/", "/executor_v0/", "/rl_manager/",
                   "/fast_env/"):
        index = tail.rfind(marker)
        if index >= 0:
            if marker in ("/site-packages/", "/Lib/"):
                return tail[index + 1:]
            return tail[index + 1:]
    if tail.startswith("~"):
        return "<c-builtin>"
    if tail.startswith("<"):
        return "<generated>"
    return tail.rsplit("/", 1)[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1,
                        help="unused; single in-process game")
    parser.add_argument("--row-claim", action="store_true")
    parser.add_argument("--opening", default=harness.OPENING_NAME)
    parser.add_argument("--opponent", default=harness.OPPONENT_SYMETRIC)
    parser.add_argument("--profile-limit", type=int, default=45)
    parser.add_argument("--no-cprofile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    checkpoint = harness._resolve_checkpoint_path(args.checkpoint)
    checkpoint_sha = harness._sha256_file(checkpoint)
    task = (0, args.seed, 0)

    profiler = cProfile.Profile() if not args.no_cprofile else None
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    record = harness._run_full_game_task(
        task, str(checkpoint), checkpoint_sha,
        enable_row_claim_board=args.row_claim,
        opponent=args.opponent,
        opening_name=args.opening,
    )
    if profiler is not None:
        profiler.disable()
    wall = time.perf_counter() - started

    banks = [int(v) for v in record["final_banks"]]
    print(f"seed={args.seed} opening={args.opening} "
          f"row_claim={args.row_claim}")
    print(f"  banks        : {banks}")
    print(f"  mean bank    : {sum(banks) / len(banks):.1f}")
    print(f"  digest       : {record.get('trace_digest')}")
    print(f"  turns        : {record.get('turns')}  "
          f"statuses={record.get('statuses')}")
    print(f"  mgr days     : "
          f"{record.get('validation', {}).get('stage25_manager_days')}")
    print(f"  wall (profiled) : {wall:.1f}s")

    summary: dict[str, object] = {
        "seed": args.seed,
        "opening": args.opening,
        "row_claim": bool(args.row_claim),
        "checkpoint_sha256": checkpoint_sha,
        "final_banks": banks,
        "mean_bank": sum(banks) / len(banks),
        "trace_digest": record.get("trace_digest"),
        "turns": record.get("turns"),
        "statuses": record.get("statuses"),
        "manager_days": record.get("validation", {}).get(
            "stage25_manager_days"),
        "profiled_wall_seconds": wall,
        "cprofile_enabled": profiler is not None,
    }

    if profiler is not None:
        stats = pstats.Stats(profiler)
        by_module: dict[str, float] = defaultdict(float)
        by_module_calls: dict[str, int] = defaultdict(int)
        for key, value in stats.stats.items():
            module = module_of(key[0])
            by_module[module] += value[2]
            by_module_calls[module] += value[0]
        total_profiled = sum(by_module.values()) or 1.0

        rows = []
        for key, value in stats.stats.items():
            rows.append({
                "function": key[0],
                "line": key[1],
                "calls": value[0],
                "primitive_calls": value[1],
                "tottime": round(value[2], 4),
                "cumtime": round(value[3], 4),
            })
        rows.sort(key=lambda r: -r["tottime"])
        top_total = rows[:args.profile_limit]
        rows.sort(key=lambda r: -r["cumtime"])
        top_cum = rows[:args.profile_limit]
        # keep the full table so offline analysis never needs another game
        all_rows = sorted(rows, key=lambda r: -r["tottime"])

        focus = {m: v for m, v in by_module.items()
                 if m.startswith(FOCUS_PREFIXES)}
        focus_total = sum(focus.values()) or 1.0
        summary["total_profiled_tottime_seconds"] = round(total_profiled, 3)
        summary["focus_share_percent"] = {
            m: round(100.0 * v / total_profiled, 2) for m, v in
            sorted(focus.items(), key=lambda kv: -kv[1])[:25]
        }
        summary["focus_calls"] = {
            m: by_module_calls[m] for m in
            sorted(focus, key=lambda m: -by_module[m])[:25]
        }
        summary["top_tottime"] = top_total
        summary["top_cumtime"] = top_cum
        summary["all_rows"] = all_rows
        summary["focus_internal_share_percent"] = {
            m: round(100.0 * v / focus_total, 2) for m, v in
            sorted(focus.items(), key=lambda kv: -kv[1])[:25]
        }

        print("\n=== module share of profiled tottime (focus only) ===")
        for module, value in sorted(focus.items(),
                                    key=lambda kv: -kv[1])[:18]:
            print(f"  {100.0 * value / total_profiled:6.2f}%  "
                  f"{by_module_calls[module]:>9,} calls  {module}")
        print("\n=== top functions by tottime ===")
        for row in top_total[:28]:
            print(f"  {row['tottime']:8.2f}s {row['calls']:>9,}  "
                  f"{row['function']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
