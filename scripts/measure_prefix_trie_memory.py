"""Measure true structural-trie memory on the captured cache-aware workload.

MEASURE ONLY. Replays the already-captured seed-41003 chain workload (no games,
no engine, no new observations) and measures the live structural trie:

* peak live trie nodes and peak total trie bytes
* approximate bytes/node
* process RSS delta (GetProcessMemoryInfo, Windows)
* simultaneously live contexts / tries
* whether ``context.prefix_tries.clear()`` releases the trees at the intended
  packing boundaries

Memory accounting note: ``_Node`` and ``_State`` are ``slots=True`` dataclasses
with no ``__dict__``, so generic ``__dict__`` traversal undercounts them. The
walker below accounts explicitly for every node, its ``children`` dict and key
tuples, its ``_State``, and the state's ``position``/``inventory``/
``global_resources``/``feasible_ids`` containers. Referenced leaf objects
(``int``, ``str``, ``RouteCostSegment``) are shared between nodes and are not
counted again; the RSS delta below is the independent cross-check.

No production code is modified.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import gc
import pickle
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import executor_v0.strip_routes as routes  # noqa: E402

_PROC = ctypes.windll.kernel32.GetCurrentProcess()
_RSS_FN: Any = None


def _load_benchmark() -> Any:
    """Import the benchmark module that defined the capture's fixture class."""
    import importlib.util

    path = REPO / "scripts" / "benchmark_strip_prefix_trie.py"
    spec = importlib.util.spec_from_file_location("_trie_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trie_bench"] = module
    spec.loader.exec_module(module)
    return module


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_bytes() -> int:
    """Current working set of this process, in bytes.

    GetProcessMemoryInfo needs explicit argtypes: without them ctypes marshals
    the struct pointer as a 32-bit int and the call silently fails, returning a
    zeroed struct. K32GetProcessMemoryInfo is the kernel32 fallback.
    """
    global _RSS_FN
    if _RSS_FN is None:
        info = _PMC()
        info.cb = ctypes.sizeof(_PMC)
        for fn in (ctypes.windll.psapi.GetProcessMemoryInfo,
                   ctypes.windll.kernel32.K32GetProcessMemoryInfo):
            fn.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
            fn.restype = wt.BOOL
            if fn(_PROC, ctypes.byref(info), ctypes.sizeof(_PMC)):
                _RSS_FN = fn
                break
        else:  # pragma: no cover - diagnostics only
            raise OSError("GetProcessMemoryInfo unavailable")
    info = _PMC()
    info.cb = ctypes.sizeof(_PMC)
    if not _RSS_FN(_PROC, ctypes.byref(info), ctypes.sizeof(_PMC)):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(info.WorkingSetSize)


def state_bytes(state: Any) -> int:
    total = sys.getsizeof(state) + sys.getsizeof(state.position)
    total += sys.getsizeof(state.inventory)
    if state.global_resources is not None:
        total += sys.getsizeof(state.global_resources)
    total += sys.getsizeof(state.feasible_ids)
    return total


def walk_trie(trie: Any) -> tuple[int, int]:
    """Return (live nodes, approximate bytes) owned by one trie."""
    seen: set[int] = set()
    total = 0
    for name in ("_mask_roots", "_roots"):
        bucket = getattr(trie, name, None)
        if not bucket:
            continue
        total += sys.getsizeof(bucket)
        nodes = (bucket.values() if isinstance(bucket, dict) else
                 (item[2] for item in bucket))
        for root in nodes:
            stack = [root]
            while stack:
                node = stack.pop()
                if node is None or id(node) in seen:
                    continue
                seen.add(id(node))
                total += sys.getsizeof(node)
                children = node.children
                total += sys.getsizeof(children)
                for key in children:
                    total += sys.getsizeof(key)
                total += state_bytes(node.state)
                if node.result is not None:
                    total += sys.getsizeof(node.result)
                stack.extend(children.values())
    # per-trie fixed overhead (small; excludes the shared segment table)
    for name in ("carried", "shed", "global_resources"):
        value = getattr(trie, name, None)
        if isinstance(value, (dict, set, list, tuple)):
            total += sys.getsizeof(value)
    return len(seen), total


def walk_context(context: Any) -> tuple[int, int, int]:
    nodes = 0
    total = 0
    for trie in context.prefix_tries.values():
        n, b = walk_trie(trie)
        nodes += n
        total += b
    return nodes, total, len(context.prefix_tries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures", type=Path, default=REPO / "artifacts" / "prefix-trie"
        / "fixtures.pkl")
    args = ap.parse_args()

    with args.fixtures.open("rb") as stream:
        # the capture was pickled from the benchmark's __main__; re-register the
        # fixture class there so unpickling resolves it without changing data
        sys.modules["__main__"]._PlanFixture = _load_benchmark()._PlanFixture
        payload = pickle.load(stream)
    fixtures = payload["fixtures"]
    groups = 1 + sum(
        1 for a, b in zip(fixtures, fixtures[1:]) if a.group is not b.group
        and a.group != b.group)
    print(f"  workload : {len(fixtures):,} captured chain plans, "
          f"{groups:,} packing groups (contexts)")
    print(f"  provenance: base={payload.get('base_ref')} seed={payload.get('seed')}"
          f" turns={payload.get('turns')}")

    gc.collect()
    rss0 = rss_bytes()
    routes._cached_chain_plan_for_mask.cache_clear()
    routes._cached_chain_plan_for_context.cache_clear()
    routes._cached_small_chain_plan_for_context.cache_clear()

    peak_created = 0
    peak_group_nodes = peak_group_bytes = peak_group_tries = 0
    created_in_group = 0
    seen_created: dict[int, int] = {}
    released_ok = 0
    not_released = 0
    rss_peak = rss0
    live_contexts = 0
    peak_contexts = 0
    prev_context = None
    context = None
    group = object()
    misses = 0

    started = time.process_time()
    for index, fx in enumerate(fixtures):
        if fx.group != group:
            if prev_context is not None:
                nodes, nbytes, ntries = walk_context(prev_context)
                if nodes > peak_group_nodes:
                    peak_group_nodes, peak_group_bytes = nodes, nbytes
                    peak_group_tries = ntries
                peak_created = max(peak_created, created_in_group)
                created_in_group = 0
                seen_created.clear()
                # the intended boundary must release every simulator node
                prev_context.prefix_tries.clear()
                gc.collect()
                left, _, _ = walk_context(prev_context)
                if left == 0:
                    released_ok += 1
                else:
                    not_released += 1
                live_contexts -= 1
            group = fx.group
            context = routes._RoutePlanContext.create(fx.candidates)
            prev_context = context
            live_contexts += 1
            peak_contexts = max(peak_contexts, live_contexts)
        small = len(fx.candidates) <= 8
        cache = (routes._cached_small_chain_plan_for_context if small
                 else routes._cached_chain_plan_for_context)
        before = cache.cache_info().misses
        routes._chain_plan_for_context(
            context, fx.worker_position, fx.mask, fx.remaining_action_slots,
            fx.worker_inventory, fx.shed_stock, fx.global_resources,
            _small_route_set=small,
        )
        if cache.cache_info().misses != before:
            misses += 1
        for trie in context.prefix_tries.values():
            total_created = getattr(trie, "nodes_created", 0)
            seen_created[id(trie)] = total_created
        created_in_group = sum(seen_created.values())
        if index % 128 == 0:
            rss_peak = max(rss_peak, rss_bytes())
    cpu = time.process_time() - started
    if prev_context is not None:
        nodes, nbytes, ntries = walk_context(prev_context)
        if nodes > peak_group_nodes:
            peak_group_nodes, peak_group_bytes, peak_group_tries = (
                nodes, nbytes, ntries)
        peak_created = max(peak_created, created_in_group)
        prev_context.prefix_tries.clear()
        gc.collect()
        left, _, _ = walk_context(prev_context)
        if left == 0:
            released_ok += 1
        else:
            not_released += 1

    print(f"  replay CPU (probe-instrumented; NOT a perf number): {cpu:.1f}s")
    print(f"  plan-cache misses: {misses:,}  (fidelity check vs 13,692)")
    print("\n=== peak live trie (worst single packing call) ===")
    print(f"  peak live trie nodes            : {peak_group_nodes:,}")
    print(f"  peak live tries in that context : {peak_group_tries}")
    print(f"  nodes created that group        : {peak_created:,}")
    print(f"  approx bytes/node               : "
          f"{peak_group_bytes / max(1, peak_group_nodes):,.0f}")
    print(f"  PEAK TOTAL TRIE BYTES           : {peak_group_bytes / 1e6:.1f} MB "
          f"({peak_group_bytes:,} B)")
    print("  (peak reported; no mean over groups)")
    print("\n=== process RSS (independent cross-check) ===")
    print(f"  RSS before replay               : {rss0 / 1e6:.1f} MB")
    print(f"  RSS peak during replay          : {rss_peak / 1e6:.1f} MB")
    print(f"  RSS delta (peak - before)       : "
          f"{(rss_peak - rss0) / 1e6:+.1f} MB")
    print("  includes captured fixture set   : ~12.6 MB pickle + 59,787 fixture "
          "objects (unavoidable, also present in base)")
    print("\n=== lifetime / release ===")
    print(f"  packing groups (contexts)       : {groups:,}")
    print(f"  peak simultaneously live ctxs   : {peak_contexts}")
    print(f"  peak simultaneously live tries : {peak_group_tries} (within one ctx)")
    print(f"  prefix_tries.clear() empties    : {released_ok} ok / "
          f"{not_released} not-empty")
    print("\n=== 4-worker context ===")
    print(f"  4 workers peaking simultaneously: "
          f"{peak_group_bytes * 4 / 1e6:.0f} MB trie total, "
          f"{(rss_peak - rss0) * 4 / 1e6:.0f} MB RSS delta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
