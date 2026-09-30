"""Low-overhead phase timing for one real full game.

MEASURE ONLY. cProfile distorts this workload badly: the chain/trie path makes
tens of millions of Python calls, so per-call profiling overhead dominated the
result and inflated ``copy.py`` by ~4x. This instead wraps a small number of
named call sites with ``time.process_time()`` accumulators, which adds
negligible overhead and gives trustworthy CPU attribution.

Example::

    python scripts/phase_timing.py --seed 41003 \
      --output artifacts/overnight/phases_41003.json
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import parallel_full_game_validation as harness  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\stage25_bc_7m_best_inference.npz")

COUNTERS: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "seconds": 0.0, "max": 0.0})


def wrap(owner: Any, name: str, label: str) -> None:
    """Wrap ``owner.name`` with a process-time accumulator labelled ``label``."""
    original = getattr(owner, name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        start = time.process_time()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.process_time() - start
            entry = COUNTERS[label]
            entry["calls"] += 1
            entry["seconds"] += elapsed
            if elapsed > entry["max"]:
                entry["max"] = elapsed

    setattr(owner, name, timed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=41003)
    parser.add_argument("--opening", default=harness.OPENING_NAME)
    parser.add_argument("--row-claim", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import evaluation.agent_match as agent_match
    import executor_v0.strip_cost as strip_cost
    import executor_v0.strip_executor as strip_executor
    import executor_v0.strip_hiring as strip_hiring
    import executor_v0.strip_market as strip_market
    import executor_v0.strip_prefix_trie as strip_prefix_trie
    import executor_v0.strip_routes as strip_routes
    import executor_v0.strip_supply as strip_supply
    import executor_v0.strip_work as strip_work
    import rl_manager.executor_factory as executor_factory

    # Patch the names as bound in ``strip_executor``: it imports these helpers
    # directly, so wrapping the defining module would not be observed.
    sites: list[tuple[Any, str, str]] = [
        (strip_executor.StripExecutorController, "act", "executor.act"),
        (executor_factory.Stage25StripExecutorAgent, "__call__",
         "agent.__call__"),
        (strip_executor, "build_strip_work_plan", "work.build_plan"),
        (strip_executor, "build_route_supply_plans", "supply.build_plans"),
        (strip_executor, "build_market_turn_plan", "market.build_plan"),
        (strip_executor, "plan_strip_hiring", "hiring.plan_strip_hiring"),
        (strip_executor, "assign_horizontal_routes_frontier",
         "routes.assign_frontier"),
        (strip_executor, "generate_horizontal_route_candidates",
         "routes.gen_candidates"),
        (strip_routes, "_pack_large_route_set_frontier",
         "routes.pack_large_frontier"),
        (strip_routes, "_pack_small_route_sets_frontier",
         "routes.pack_small_frontier"),
        (strip_routes, "_compute_chain_plan_for_context",
         "routes.chain_plan_compute"),
        (strip_prefix_trie.RouteCostTrie, "evaluate", "trie.evaluate"),
        (strip_prefix_trie.RouteCostTrie, "_extend", "trie.extend"),
        (strip_prefix_trie.RouteCostTrie, "_root", "trie.root"),
        (strip_cost, "simulate_route_cost", "cost.simulate_route_cost"),
        (agent_match, "_controller_observation", "agent.observation_copy"),
        (agent_match, "_call_controller", "agent.call_controller"),
        (executor_factory.Stage25StripExecutorAgent, "diagnostics_json",
         "executor.diagnostics_json"),
    ]
    installed: list[tuple[Any, str, Any]] = []
    for owner, name, label in sites:
        if hasattr(owner, name):
            wrap(owner, name, label)
            installed.append((owner, name, getattr(owner, name)))
        else:
            print(f"  (skip {label}: {name} not present)")

    checkpoint = harness._resolve_checkpoint_path(args.checkpoint)
    checkpoint_sha = harness._sha256_file(checkpoint)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        record = harness._run_full_game_task(
            (0, args.seed, 0), str(checkpoint), checkpoint_sha,
            enable_row_claim_board=args.row_claim,
            opponent=harness.OPPONENT_SYMETRIC,
            opening_name=args.opening,
        )
    finally:
        for owner, name, timed in installed:
            setattr(owner, name, timed.__wrapped__)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start

    print(f"seed={args.seed} banks={record['final_banks']} "
          f"digest={str(record.get('trace_digest'))[:16]}")
    print(f"wall={wall:.1f}s  process CPU={cpu:.1f}s")

    total = sum(v["seconds"] for v in COUNTERS.values())
    print(f"\n=== phase CPU (nested phases overlap; not additive) ===")
    print(f"  {'phase':<34}{'CPU s':>9}{'calls':>10}{'us/call':>10}{'max ms':>9}")
    for label, entry in sorted(COUNTERS.items(),
                               key=lambda kv: -kv[1]["seconds"]):
        per = 1e6 * entry["seconds"] / max(1, entry["calls"])
        print(f"  {label:<34}{entry['seconds']:9.2f}{entry['calls']:>10.0f}"
              f"{per:10.1f}{1000 * entry['max']:9.1f}")

    # exclusive attribution for the top-level split
    act = COUNTERS["executor.act"]["seconds"]
    print(f"\n  executor.act total CPU: {act:.2f}s "
          f"({100 * act / cpu:.1f}% of process CPU)")
    for label in ("routes.pack_large_frontier", "work.build_plan",
                  "market.build_plan", "supply.build_plans",
                  "trie.evaluate"):
        if label in COUNTERS:
            share = 100 * COUNTERS[label]["seconds"] / max(1e-9, act)
            print(f"    {label:<34}{COUNTERS[label]['seconds']:8.2f}s "
                  f"({share:5.1f}% of act)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "seed": args.seed,
            "opening": args.opening,
            "row_claim": bool(args.row_claim),
            "final_banks": [int(v) for v in record["final_banks"]],
            "trace_digest": record.get("trace_digest"),
            "wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "phases": dict(COUNTERS),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
