# Strip prefix trie: offline implementation and microbenchmark

Implemented on `perf/strip-prefix-trie`, directly from
`1fd8eac5a8843418b1e531a8bafd52448d9a1de8`, in the isolated
`strip-prefix-trie` worktree. The previous combined candidate was only an
ablation input. No games, panels, captures, checkpoints, or training ran.

The structural trie passes the requested >1.5x gate: **2.88x** on exhaustive
captured-chain evaluation and **2.18–2.20x** with the original plan LRUs enabled.
It does not reach the approximately 26k theoretical node estimate.

## Workload and clocks

The existing seed-41003 pickle contains 59,787 chain-plan calls from 272 captured
assignment groups, including 1,691 fixed worker/resource contexts. Its SHA-256 is
`101d15e620d4c309ccc43b04f7eeeedc8567ef43a1e0d83eef35b0cb89dfbfba`.
Only this recorded data was replayed; no new observations were generated.

CPU totals use `time.process_time`, on Windows with Python 3.13.1. Route-cost
figures are median CPU per call over three batches of the first 4,000 captured
multi-segment chains. Fixture loading and normalization are outside the timers.
Exhaustive replay includes all calls and deliberately bypasses the plan LRUs;
production replay preserves the base's separate 4,096-entry small/large caches.
Both timed paths construct comparable final-plan identities.

Phase timings use high-resolution `time.perf_counter` in a **separate
instrumented replay**. They are elapsed phase timings, not claimed CPU samples
for individual short edges. Windows process CPU accounting is too coarse for
those short spans. Phase totals exclude the timer and outer enumeration overhead.

## Step 1: ablation

| Design | Exhaustive chain CPU | Route-cost CPU, us/call |
| --- | ---: | ---: |
| Clean base | 41.6875 s | 125.00 |
| Precompute only | 40.6250 s | 128.91 |
| State-key prefix reuse, original route-cost helpers | 55.2813 s | 117.19 |
| Previous combined design | 52.5781 s | 121.09 |

All four produced exactly the same 59,787 selected chain plans. The prefix-only
ablation loads the clean base evaluator, segment classes, summary LRU and
resource-progress helper, then adds the previous state-key prefix evaluator
with its precomputed progress-mode dispatch removed. The previous design's
short-chain fallback is preserved. The route-cost microbenchmark exercises the
standalone simulator; prefix reuse is measured in the chain workload.

This replay uses already frozen captured segments, so eager segment-construction
cost is excluded. The small precompute-only timing difference is not evidence
of a useful production gain. The standalone base and prefix-only simulator
source is identical; their latency differences reflect measurement variation.

The previous combined source is snapshotted under
`artifacts/prefix-trie/previous_combined_cost.py`; normalized text SHA-256:
`6ea64adf68b993264d6c89be638b37cc1a9adee2e3241e3e69e455b89b66789f`.
Raw measurements: `artifacts/prefix-trie/ablation.json`.

## Step 2: structural trie

`executor_v0/strip_prefix_trie.py` adds an assignment-local evaluator. Each child
is keyed by `(candidate index, orientation)` and owns the state after extending
its parent's state through exactly one existing frozen segment. No state,
inventory, dependency set, or resource ledger is hashed per edge. State is
unhashable and stores mutable node-owned ledgers, position, elapsed time and
cumulative chain objectives. Sibling nodes copy the ledgers before extension.

The base plans pickup from the entire chain before its first segment. Each
subset therefore prepares its initial inventory and pickup cost once. Distinct
initial resource/pickup contexts have separate roots, compared without ledger
hashing. Pickup item order has no effect on the existing chain objectives:
the base makes the entire planned inventory available before segment evaluation
and charges one action per picked item. Work and continuation resource checks
then execute in the original tile/work order through the unchanged
`_work_can_progress` helper.

The trie materializes the exact fields consumed by chain selection. The original
order/orientation enumeration, objective comparisons, final path tie-break and
selected `RouteSegment` construction remain intact. No alternative is pruned.
Different paths or fixed worker/resource contexts are not merged merely because
their simulator states happen to be equal.

`strip_routes.py` wires the evaluator into small-set and large-set packing.
Both plan-cache capacities are preserved. Packing clears trie references before
returning, so the global plan LRUs retain completed plans rather than simulator
trees. The original `executor_v0/strip_cost.py`, including `simulate_route_cost`
and `RouteCostSegment.__post_init__`, is unchanged. Its normalized source SHA-256
is `7e72a6aed05860bbe548f2a094fd4df910108551bdf5472faad1c6c4a83d1b40`.

## Timing and work reduction

| Replay | Base CPU | Trie CPU | Speedup | Segment visits before / after | Trie nodes, including roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exhaustive, all captured calls | 42.6719 s | 14.8125 s | **2.881x** | 816,576 / 332,556 | 341,412 |
| Original plan LRUs, first run | 22.8281 s | 10.4688 s | **2.181x** | 384,912 / 167,426 | 169,464 |
| Original plan LRUs, final profiled packet | 23.1563 s | 10.5313 s | **2.199x** | 384,912 / 167,426 | 169,464 |

The exhaustive count includes calls that the production plan LRUs would skip;
the cache-aware count is the directly comparable approximately 392k-visit
workload in the task. Both cache-aware implementations had exactly **13,692
misses and 46,095 hits**, and evaluated the same 128,828 paths on misses.
There were 8,856 roots in exhaustive replay and 2,038 in cache-aware replay.
Segment visits after reuse count one simulator extension per non-root node.

| Instrumented phase, elapsed | Exhaustive replay | Cache-aware replay |
| --- | ---: | ---: |
| State extension, excluding copying | 10.5688 s | 5.2110 s |
| State and ledger copying | 1.0902 s | 0.5636 s |
| Trie lookup and child creation | 1.0099 s | 0.5048 s |
| Result construction | 0.8870 s | 0.4682 s |
| Subset/root preparation | 0.6964 s | 0.2028 s |

The instrumented replays consumed 16.1875 s and 10.7031 s of CPU respectively;
speedups above use uninstrumented replay. Copying is not dominant: it is about
8.1% of the measured cache-aware phase time. Extension remains the largest cost.

Standalone original route-cost CPU varied between runs: exhaustive replay's
4k-call medians were 132.81 us/base and 140.63 us/candidate; the first cache-aware
run measured 156.25 us for both; the final run measured 140.63 and 156.25 us.
These execute unchanged simulator source, so no standalone evaluator speedup
is claimed. The benefit is in shared chain evaluation.

Raw data: `artifacts/prefix-trie/benchmark.json`, `production.json`, and
`production-profile.json`.

## Exactness and checks

- Every one of the **315,696** captured enumerated paths matched the clean base
  on total/completion turns, setup travel, pickup travel, inter-segment travel,
  useful interactions, useful segments and unfinished interactions.
- All **59,787** selected plans matched on route order, traversal/orientation,
  entry distance and every chain-result objective field. All ablations and both
  trie replay modes matched the clean-base selected plans.
- **228 focused tests passed**, including original cost/route/executor/hiring
  tests and new dependency, resource exhaustion, continuation, deadline,
  whole-chain pickup, sibling isolation, root separation, unhashable-state and
  cache-lifetime checks. Hiring code was not changed.
- Ruff checks and `git diff --check` passed. Formatting preserved the Python ASTs.

Path parity evidence: `artifacts/prefix-trie/path-parity.json`.
No full-game validation was performed or authorized by this packet.

## Reproduction

From this worktree, using the preserved local copies of the original inputs:

```powershell
python scripts/benchmark_strip_prefix_trie.py --mode ablation --fixtures artifacts/prefix-trie/fixtures.pkl --previous-cost artifacts/prefix-trie/previous_combined_cost.py --output artifacts/prefix-trie/ablation.json
python scripts/benchmark_strip_prefix_trie.py --mode trie --fixtures artifacts/prefix-trie/fixtures.pkl --output artifacts/prefix-trie/benchmark.json
python scripts/benchmark_strip_prefix_trie.py --mode production --fixtures artifacts/prefix-trie/fixtures.pkl --output artifacts/prefix-trie/production-profile.json
python scripts/benchmark_strip_prefix_trie.py --mode validate --fixtures artifacts/prefix-trie/fixtures.pkl --output artifacts/prefix-trie/path-parity.json
python -m pytest tests/test_strip_prefix_trie.py tests/test_strip_cost.py tests/test_strip_routes.py tests/test_strip_executor.py tests/test_strip_hiring.py -q
```

The offline benchmark has no game-running mode and imports no game runner.
All artifacts are local generated files under the repository's ignored
`artifacts/` directory; the implementation, tests, harness and this report are
reviewable worktree changes.

## Memory, and full-game validation (later packets)

Peak memory was previously unmeasured. It is small because the trie is scoped to
a single packing call, not to the process: `_pack_small_route_sets_frontier` and
`_pack_large_route_set_frontier` each build one `_RoutePlanContext`, evaluate all
worker prefixes into its `prefix_tries`, and call `context.prefix_tries.clear()`
before returning. Nodes are never removed individually, so live nodes rise
monotonically within a packing call and reset at its boundary.

`scripts/measure_prefix_trie_memory.py` replays the captured cache-aware
workload and probes at each of the 272 packing boundaries:

| Measurement | Value |
| --- | ---: |
| peak live trie nodes (worst single packing call) | 10,681 |
| peak live tries in that context | 5 |
| approximate bytes/node | 1,771 |
| **peak total trie bytes** | **18.9 MB** |
| process RSS before / peak | 135.9 MB / 145.1 MB |
| **process RSS delta** | **+10.1 MB** |
| peak simultaneously live contexts | 1 |
| `prefix_tries.clear()` boundaries verified empty | 272 ok / 0 not-empty |
| 4 workers peaking simultaneously | 76 MB trie, ~40 MB RSS delta |

The walker accounts explicitly for `_Node`, its `children` dict and key tuples,
its `_State`, and the state's `position`/`inventory`/`global_resources`/
`feasible_ids` containers, because these are `slots=True` dataclasses with no
`__dict__` and generic traversal undercounts them. The smaller RSS delta is
expected: freed per-group tries are recycled through the same allocator arenas.
The replay reproduced **13,692** plan-cache misses, matching `production.json`
exactly, so it measured the same workload.

Full-game validation was then authorized and run: 4 parallel workers, seeds
41001-41004, `standard_mixed_d6h3`, 7M checkpoint, row-claim OFF, symmetric
opponent. Panel wall **101.8 s -> 84.1 s (1.21x)**, with all four final bank
pairs, all four action digests, 719 turns, 24 manager days and DONE/DONE
identical to the established baseline (mean bank/seat 71,353.1, unchanged).
Measured on a machine with ~3.8 of 20 cores busy from unrelated load, so the
panel wall carries a small optimistic bias. No further games were run.
