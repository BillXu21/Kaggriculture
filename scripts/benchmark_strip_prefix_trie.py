"""Offline replay only: ablations, exact chain parity, and trie microbenchmark.

No controller, engine, game, or checkpoint is imported or executed.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import pickle
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASE = "1fd8eac5a8843418b1e531a8bafd52448d9a1de8"


class _PlanFixture:
    """Compatibility name for the existing captured pickle, never captures."""


class FixtureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_PlanFixture":
            return _PlanFixture
        return super().find_class(module, name)


def load_source(name, source):
    module = ModuleType(name)
    module.__package__ = "executor_v0"
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def base_source(path):
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def definition(source, name):
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == name)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(source.splitlines()[start - 1 : node.end_lineno]) + "\n"


def original_prefix_module(base, previous):
    """Ablate state-key reuse with original segment and simulator helpers."""
    module = load_source("_ablation_prefix_original", base)
    exec("from dataclasses import field, fields\n", module.__dict__)
    exec(
        "def _sum_sorted_pairs(sequences):\n"
        "    totals = defaultdict(int)\n"
        "    for pairs in sequences:\n"
        "        for item, quantity in pairs:\n"
        "            totals[item] += quantity\n"
        "    return _pairs(totals)\n",
        module.__dict__,
    )
    for name in (
        "RouteCostPrefixContext",
        "RouteCostPrefixState",
        "RouteCostPrefixMemo",
        "prepare_route_cost_prefix",
        "_extend_route_cost_prefix_uncached",
        "_extend_unconstrained_route_cost_prefix",
        "_extend_constrained_route_cost_prefix",
        "extend_route_cost_prefix",
        "finish_route_cost_prefix",
        "simulate_route_cost_with_prefix_memo",
    ):
        source = definition(previous, name)
        if name == "_extend_constrained_route_cost_prefix":
            start = source.index("            progress_mode =")
            end = source.index("            if not can_progress:", start)
            source = (
                source[:start]
                + (
                    "            can_progress = _work_can_progress(\n"
                    "                work, feasible_ids, available_inventory, available_global\n"
                    "            )\n"
                )
                + source[end:]
            )
        exec(source, module.__dict__)
    return module


def normalize(fixtures, module):
    refs = {}
    for fx in fixtures:
        for pair in fx.oriented_cost_segments:
            for segment in pair:
                if id(segment) in refs:
                    continue
                work = tuple(
                    tuple(
                        module.RouteCostWork(
                            **{
                                f: getattr(w, f)
                                for f in module.RouteCostWork.__dataclass_fields__
                                if module.RouteCostWork.__dataclass_fields__[f].init
                            }
                        )
                        for w in tile
                    )
                    for tile in segment.work_by_tile
                )
                refs[id(segment)] = module.RouteCostSegment(
                    segment.segment_id, segment.traversal, work, segment.physical_row_id
                )
    return refs


def plan_key(plan):
    return (
        plan.movement_turns,
        plan.completion_turns,
        plan.useful_interactions,
        plan.useful_segments,
        plan.unfinished_interactions,
        tuple((c.route_id, s.traversal, s.entry_distance) for c, s in plan.assigned),
    )


def paths(fx):
    indices = tuple(i for i in range(len(fx.candidates)) if fx.mask & (1 << i))
    orders = [indices]
    if len({fx.candidates[i].row_key.global_row for i in indices}) > 1:
        orders.append(tuple(reversed(indices)))
    for order in orders:
        for bits in range(1 << len(order)):
            yield tuple((i, (bits >> off) & 1) for off, i in enumerate(order))


def cost_key(result):
    return (
        result.total_turns,
        result.setup_travel_turns,
        result.pickup_travel_turns,
        result.inter_segment_travel_turns,
        result.effective_interactions_completed_before_deadline,
        result.segments_completed_before_deadline,
        result.effective_interactions_missed,
    )


def replay(
    fixtures,
    routes,
    cost,
    *,
    prefix=False,
    trie=False,
    profile=False,
    validate_cost=None,
):
    if trie:
        from executor_v0.strip_prefix_trie import RouteCostTrie

        factory = RouteCostTrie
        if validate_cost is not None:

            class CheckedTrie(RouteCostTrie):
                checked_paths = 0

                def evaluate(self, mask, path):
                    expected = validate_cost.simulate_route_cost(
                        self.start,
                        tuple(self.segments[i][s] for i, s, _ in path),
                        remaining_action_slots=self.budget,
                        carried_inventory=self.carried,
                        shed_stock=self.shed,
                        global_resources=self.global_resources,
                        include_segment_results=False,
                    )
                    actual = super().evaluate(mask, path)
                    assert cost_key(actual) == cost_key(expected), (
                        mask,
                        path,
                        cost_key(actual),
                        cost_key(expected),
                    )
                    self.checked_paths += 1
                    return actual

                def statistics(self):
                    return dict(super().statistics(), exact_paths=self.checked_paths)

            factory = CheckedTrie
    keys = []
    group = object()
    reuse = None
    tries = {}
    counters = defaultdict(float)
    started = time.process_time()
    for fx in fixtures:
        if fx.group != group:
            group = fx.group
            if reuse is not None and trie:
                for item in tries.values():
                    for k, v in item.statistics().items():
                        counters[k] += v
                tries = {}
            reuse = cost.RouteCostPrefixMemo() if prefix else None
        if trie:
            fixed_key = (
                fx.worker_position,
                fx.remaining_action_slots,
                tuple(sorted(fx.worker_inventory.items())),
                None if fx.shed_stock is None else tuple(sorted(fx.shed_stock.items())),
                None
                if fx.global_resources is None
                else tuple(sorted(fx.global_resources.items())),
            )
            reuse = tries.get(fixed_key)
            if reuse is None:
                reuse = factory(
                    fx.oriented_cost_segments,
                    fx.worker_position,
                    remaining_action_slots=fx.remaining_action_slots,
                    carried_inventory=fx.worker_inventory,
                    shed_stock=fx.shed_stock,
                    global_resources=fx.global_resources,
                    profile=profile,
                )
                tries[fixed_key] = reuse
        if prefix:
            routes.simulate_route_cost = lambda *a, **kw: (
                cost.simulate_route_cost_with_prefix_memo(
                    *a,
                    **{k: v for k, v in kw.items() if k != "include_segment_results"},
                    memo=reuse,
                )
            )
        else:
            routes.simulate_route_cost = cost.simulate_route_cost
        plan = routes._compute_chain_plan_for_mask(
            fx.candidates,
            fx.worker_position,
            fx.mask,
            fx.remaining_action_slots,
            fx.worker_inventory,
            fx.shed_stock,
            fx.global_resources,
            fx.oriented_cost_segments,
            **({"_prefix_trie": reuse} if trie else {}),
        )
        keys.append(plan_key(plan))
    elapsed = time.process_time() - started
    if reuse is not None and trie:
        for item in tries.values():
            for k, v in item.statistics().items():
                counters[k] += v
    return elapsed, keys, dict(counters)


def production_replay(fixtures, routes, cost, *, trie=False, profile=False):
    """Replay the base's plan LRUs as well, counting only actual cache misses."""
    routes.simulate_route_cost = cost.simulate_route_cost
    routes._cached_chain_plan_for_mask.cache_clear()
    routes._cached_chain_plan_for_context.cache_clear()
    if trie:
        routes._cached_small_chain_plan_for_context.cache_clear()
    original_factory = getattr(routes, "RouteCostTrie", None)
    if profile:

        class ProfiledTrie(original_factory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, profile=True)

        routes.RouteCostTrie = ProfiledTrie
    keys = []
    counters = defaultdict(float)
    group = object()
    context = None

    def release():
        if trie and context is not None:
            for item in context.prefix_tries.values():
                for k, v in item.statistics().items():
                    counters[k] += v
            context.prefix_tries.clear()

    started = time.process_time()
    for fx in fixtures:
        if fx.group != group:
            release()
            group = fx.group
            context = routes._RoutePlanContext.create(fx.candidates)
        small = len(fx.candidates) <= 8
        cache = (
            routes._cached_small_chain_plan_for_context
            if trie and small
            else routes._cached_chain_plan_for_mask
            if small
            else routes._cached_chain_plan_for_context
        )
        before = cache.cache_info().misses
        args = (
            fx.worker_position,
            fx.mask,
            fx.remaining_action_slots,
            fx.worker_inventory,
            fx.shed_stock,
            fx.global_resources,
        )
        if small and not trie:
            plan = routes._chain_plan_for_mask(fx.candidates, *args)
        else:
            plan = routes._chain_plan_for_context(
                context, *args, **({"_small_route_set": small} if trie else {})
            )
        keys.append(plan_key(plan))
        if cache.cache_info().misses != before:
            counters["plan_cache_misses"] += 1
            if fx.mask and not trie:
                indices = [i for i in range(len(fx.candidates)) if fx.mask & (1 << i)]
                orders = (
                    2
                    if len({fx.candidates[i].row_key.global_row for i in indices}) > 1
                    else 1
                )
                path_count = orders * (1 << len(indices))
                counters["evaluated_paths"] += path_count
                counters["segment_visits_before"] += path_count * len(indices)
        else:
            counters["plan_cache_hits"] += 1
    release()
    if profile:
        routes.RouteCostTrie = original_factory
    return time.process_time() - started, keys, dict(counters)


def route_benchmark(fixtures, module, count):
    calls = []
    for fx in fixtures:
        if not fx.mask:
            continue
        for path in paths(fx):
            if len(path) < 2:
                continue
            calls.append((fx, tuple(fx.oriented_cost_segments[i][s] for i, s in path)))
            if len(calls) == count:
                break
        if len(calls) == count:
            break
    samples = []
    for _ in range(3):
        start = time.process_time()
        for fx, segments in calls:
            module.simulate_route_cost(
                fx.worker_position,
                segments,
                carried_inventory=fx.worker_inventory,
                remaining_action_slots=10**9
                if fx.remaining_action_slots is None
                else fx.remaining_action_slots,
                shed_stock=fx.shed_stock,
                global_resources=fx.global_resources,
                include_segment_results=False,
            )
        samples.append(time.process_time() - start)
    return {
        "calls": len(calls),
        "cpu_samples": samples,
        "us_per_call": statistics.median(samples) * 1e6 / len(calls),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--previous-cost", type=Path)
    parser.add_argument(
        "--mode", choices=("ablation", "trie", "validate", "production"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--route-calls", type=int, default=4000)
    args = parser.parse_args()
    with args.fixtures.open("rb") as stream:
        payload = FixtureUnpickler(stream).load()
    fixtures = payload["fixtures"]
    if args.limit:
        fixtures = fixtures[: args.limit]
    cost_source = base_source("executor_v0/strip_cost.py")
    routes_source = base_source("executor_v0/strip_routes.py")
    report = {
        "schema_version": 1,
        "base_ref": BASE,
        "fixtures": len(fixtures),
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "cpu_clock": "time.process_time",
        "games_run": 0,
    }
    report["phase_clock"] = (
        "time.perf_counter (high-resolution elapsed timings in a separate instrumented replay)"
    )
    report["python"] = sys.version
    base = load_source("_clean_base_cost", cost_source)
    base_routes = load_source("_clean_base_routes", routes_source)
    if args.mode == "ablation":
        previous = args.previous_cost.read_text(encoding="utf-8")
        report["previous_cost_sha256"] = hashlib.sha256(previous.encode()).hexdigest()
        precompute = load_source("_ablation_precompute", previous)
        prefix = original_prefix_module(cost_source, previous)
        variants = [
            ("base", base, False),
            ("precompute_only", precompute, False),
            ("prefix_original", prefix, True),
            ("previous_combined", precompute, True),
        ]
        expected = None
        for name, module, memo in variants:
            refs = normalize(fixtures, module)
            original = [fx.oriented_cost_segments for fx in fixtures]
            for fx in fixtures:
                fx.oriented_cost_segments = tuple(
                    tuple(refs[id(s)] for s in pair)
                    for pair in fx.oriented_cost_segments
                )
            elapsed, keys, _ = replay(fixtures, base_routes, module, prefix=memo)
            if expected is None:
                expected = keys
            assert keys == expected, name
            row = {
                "chain_cpu": elapsed,
                "exact_plans": len(keys),
                "route_cost": route_benchmark(fixtures, module, args.route_calls),
            }
            report[name] = row
            print(name, json.dumps(row), flush=True)
            for fx, segments in zip(fixtures, original):
                fx.oriented_cost_segments = segments
    else:
        import executor_v0.strip_cost as cost
        import executor_v0.strip_routes as routes

        refs = normalize(fixtures, cost)
        for fx in fixtures:
            fx.oriented_cost_segments = tuple(
                tuple(refs[id(s)] for s in pair) for pair in fx.oriented_cost_segments
            )
        if args.mode == "validate":
            elapsed, _, counters = replay(
                fixtures, routes, cost, trie=True, validate_cost=base
            )
            report.update(validation_cpu=elapsed, counters=counters)
            print(json.dumps(report, indent=2), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            return
        runner = production_replay if args.mode == "production" else replay
        baseline, expected, before = runner(fixtures, base_routes, base)
        label = (
            "base with plan LRUs" if args.mode == "production" else "base exhaustive"
        )
        print(f"{label} CPU: {baseline:.6f}s", flush=True)
        elapsed, actual, counters = runner(fixtures, routes, cost, trie=True)
        assert actual == expected, "selected chain-plan mismatch"
        print(f"trie CPU: {elapsed:.6f}s ({baseline / elapsed:.3f}x)", flush=True)
        profiled, profiled_keys, profile = runner(
            fixtures, routes, cost, trie=True, profile=True
        )
        assert profiled_keys == expected
        report.update(
            base_chain_cpu=baseline,
            trie_cpu=elapsed,
            speedup=baseline / elapsed,
            exact_plans=len(expected),
            counters=counters,
            profiled_cpu=profiled,
            profile=profile,
        )
        report["base_counters"] = before
        report["base_route_cost"] = route_benchmark(fixtures, base, args.route_calls)
        report["candidate_route_cost"] = route_benchmark(
            fixtures, cost, args.route_calls
        )
        print(json.dumps(report, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
