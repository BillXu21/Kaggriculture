"""Attribute every copy.deepcopy call on a real game to its call site.

MEASURE ONLY. Runs one fixed-seed full game in process with ``copy.deepcopy``
wrapped so each call records its caller file:line and the size of the object
being copied. Produces a call-site histogram of copy time, which is what
decides whether a defensive copy can be removed with exact parity.

Example::

    python scripts/attribute_deepcopy.py --seed 41003 \
      --output artifacts/overnight/deepcopy_41003.json
"""

from __future__ import annotations

import argparse
import copy as stdlib_copy
import json
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


def object_size(value: object) -> int:
    """Shallow-ish size: containers count, leaf scalars count as small."""
    try:
        if isinstance(value, dict):
            return 64 + sum(object_size(k) + object_size(v)
                            for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return 56 + sum(object_size(v) for v in value)
    except (RecursionError, TypeError):
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=41003)
    parser.add_argument("--opening", default=harness.OPENING_NAME)
    parser.add_argument("--row-claim", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sites: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "seconds": 0.0, "size": 0})
    sizes: dict[str, list[int]] = defaultdict(list)
    frames = sys._getframe
    original = stdlib_copy.deepcopy

    def instrumented(value, memo=None, _nil=None):
        # capture the first frame outside copy.py / this wrapper
        f = frames(1)
        site = "unknown"
        while f is not None:
            name = f.f_code.co_filename
            if "copy.py" not in name and "attribute_deepcopy" not in name:
                short = name.replace("\\", "/")
                for marker in ("/executor_v0/", "/rl_manager/", "/fast_env/",
                               "/evaluation/", "/opening_book/"):
                    if marker in short:
                        short = short[short.rfind(marker) + 1:]
                        break
                else:
                    short = short.rsplit("/", 1)[-1]
                site = f"{short}:{f.f_lineno}"
                break
            f = f.f_back
        start = time.process_time()
        try:
            return original(value, memo, _nil)
        finally:
            elapsed = time.process_time() - start
            entry = sites[site]
            entry["calls"] += 1
            entry["seconds"] += elapsed
            if len(sizes[site]) < 400:
                sizes[site].append(object_size(value))

    checkpoint = harness._resolve_checkpoint_path(args.checkpoint)
    checkpoint_sha = harness._sha256_file(checkpoint)

    stdlib_copy.deepcopy = instrumented
    import copy as _c
    _c.deepcopy = instrumented
    started = time.perf_counter()
    try:
        record = harness._run_full_game_task(
            (0, args.seed, 0), str(checkpoint), checkpoint_sha,
            enable_row_claim_board=args.row_claim,
            opponent=harness.OPPONENT_SYMETRIC,
            opening_name=args.opening,
        )
    finally:
        stdlib_copy.deepcopy = original
        _c.deepcopy = original
    wall = time.perf_counter() - started

    print(f"seed={args.seed} banks={record['final_banks']} "
          f"digest={str(record.get('trace_digest'))[:16]} "
          f"wall={wall:.1f}s")
    total_calls = sum(v["calls"] for v in sites.values())
    total_seconds = sum(v["seconds"] for v in sites.values())
    print(f"\ntotal deepcopy: {total_calls:,} calls, "
          f"{total_seconds:.2f}s CPU ({time.process_time() - started:.1f}s "
          f"process CPU)")

    print("\n=== deepcopy call sites by CPU seconds ===")
    for site, entry in sorted(sites.items(),
                              key=lambda kv: -kv[1]["seconds"])[:25]:
        observed = sizes[site]
        med = sorted(observed)[len(observed) // 2] if observed else 0
        print(f"  {entry['seconds']:7.3f}s {entry['calls']:>9,} calls "
              f"medsize~{med:>7,}  {site}")

    summary = {
        "seed": args.seed,
        "opening": args.args if False else args.opening,
        "final_banks": [int(v) for v in record["final_banks"]],
        "trace_digest": record.get("trace_digest"),
        "wall_seconds": wall,
        "total_deepcopy_calls": total_calls,
        "total_deepcopy_cpu_seconds": total_seconds,
        "sites": {
            site: {
                "calls": entry["calls"],
                "seconds": entry["seconds"],
                "median_object_size": (
                    sorted(sizes[site])[len(sizes[site]) // 2]
                    if sizes[site] else 0),
            }
            for site, entry in sites.items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
