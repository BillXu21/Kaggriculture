# Strip executor overnight optimization run

Branch: `overnight/strip-speed-money`
Base: `a01e75c25b11d5442dbec0fa3244bbe72b3c7cd7` (`perf/strip-prefix-trie`)
Worktree: `.worktrees/strip-speed-money-overnight`

Mission: improve rollout/executor speed **and** final economic performance
("money"), maintaining a Pareto frontier rather than one score.

Canonical validated baseline (do not regress):

| seed | seat 0 | seat 1 |
|---|---:|---:|
| 41001 | 72587 | 70810 |
| 41002 | 74061 | 75159 |
| 41003 | 79167 | 73221 |
| 41004 | 64229 | 61591 |

mean bank/seat `71,353.1`; 4-worker panel wall 84.1 s; row-claim OFF;
`standard_mixed_d6h3`; 719 turns; 24 manager days; all four action digests
bit-exact.

## Current best speed-only candidate

- branch/SHA: `overnight/strip-speed-money` @ `9c5f85e`
- **panel wall: 40.8 s** (4 workers, seeds 41001-41004)
- exact-parity status: **PASS** -- 4/4 final bank pairs, 4/4 action digests
  bit-identical to the validated baseline
- bank: mean 71,353.1, min 61,591 (unchanged)

### Speed measurement honesty note

The inherited "84.1 s" panel figure was measured with ~3.8 of 20 cores busy
from unrelated load. Re-measuring the **unmodified base `a01e75c`** on a quiet
machine gives a **48.3 s** panel wall, so the old number was inflated by
roughly 74% by contention. The controlled, back-to-back comparison is:

| | panel wall | mean/game | sum of games | turns/s | mean bank |
|---|---:|---:|---:|---:|---:|
| base `a01e75c` | 48.3 s | 39.4 s | 157.5 s | 59.54 | 71,353.1 |
| candidate `9c5f85e` | **40.8 s** | **32.6 s** | **130.5 s** | **70.48** | 71,353.1 |
| delta | **-7.5 s (1.18x)** | -17.3% | -17.1% | +18.4% | 0.0 |

Per-game wall: 41001 -18.1%, 41002 -15.9%, 41003 -17.8%, 41004 -16.6% --
consistent across all four seeds, which is what makes the number credible. The
base was measured on a *quieter* machine (0.64 of 20 cores) than the candidate
(1.7 of 20), so 1.18x is, if anything, slightly understated.

**The honest claim is 1.18x panel wall / ~17% per-game, not 2.06x.** Anyone
comparing against the historical 84.1 s would overstate the gain by ~1.7x.

## Current best economic candidate

- branch/SHA: `overnight/strip-speed-money` @ `9c5f85e`
- mean bank: 71,353.1
- per-seed: 72587/70810, 74061/75159, 79167/73221, 64229/61591
- speed: 40.8 s panel wall

No behaviour-changing candidate survived: both money experiments regressed
(see Dead ends). The best economic candidate is therefore the same commit as
the best speed candidate, which is the correct outcome given the evidence that
the executor is already execution-saturated.

## Pareto candidates

| commit | panel wall | mean bank | min bank | parity | notes |
|---|---:|---:|---:|---|---|
| `a01e75c` (base) | 48.3 s | 71,353.1 | 61,591 | exact | reference, re-timed quietly |
| `c0fdffe` | - | 71,353.1 | 61,591 | exact | diagnostics freeze; single game 40.9 -> 32.2 s |
| `1717fd9` | - | 71,353.1 | 61,591 | exact | farms copy removed |
| `9ecdcd2` | - | 71,353.1 | 61,591 | exact | traversal tuples shared; 315,696-path parity |
| `b3c0967` | - | 71,353.1 | 61,591 | exact | `_extend` accumulators hoisted |
| **`9c5f85e`** | **40.8 s** | **71,353.1** | **61,591** | **exact** | **recommended** |

## Dead ends

| idea | result | verdict |
|---|---|---|
| `purchases_enabled=True` (executor buys the seeds its routes require) | seed 41003 mean bank 76,194.0 -> **74,516.0 (-2.2%)**; unfinished routes 16.0% -> 20.9% | **rejected** |
| Withhold WHEAT from aggressive selling (protect planned requirement) | seed 41003 banks 79167/73221 and digest `9941ecb4` **bit-identical** | **rejected, hypothesis refuted** |
| Larger chain-plan LRU | already proven zero capacity misses at `a01e75c` | not retried |
| Memo keyed on full simulator state | already proven a regression | not retried |
| Eager `RouteCostSegment` summaries | already proven a regression | not retried |

### Why the money track is bounded (evidence, seed 41003, both seats)

`scripts/econ_diagnostics.py` + `scripts/analyze_econ.py` on the real game:

- **Zero lost interactions.** `feasible_effective_interaction_turns` equals
  `effective_interactions_completed_before_deadline` exactly (1,475 and 1,424),
  `effective_interactions_missed` is **0**, and the feasible-but-not-completed
  gap is **0.0%** on every one of the 24 days. The executor already realises
  100% of the interactions its plan makes feasible.
- **Zero timing waste.** `timing_complete_before_deadline` is true for
  **319/319** and **317/317** canonical route costs. No route runs out of time.
- **Worker capacity is saturated.** `current_workers == target_workers` on all
  24 days, `wanted_hires = submittable_hires = affordable_hires = 0`,
  `stop_reason = COVERED`, `overloaded_rows_detected = 0`, and
  `rows_expected_complete_with_n_workers == ..._with_n_plus_one_workers` every
  day (e.g. 15 == 15). `packed_segment_groups` shows exactly one row per
  worker, 14-15 rows on a 15-row board. A 16th worker would complete **no**
  additional row, so declining to hire is correct.
- **The 16-18% unfinished routes are resource-infeasible, not labour-infeasible.**
  `resource_feasible = False` on 79/319 and 89/317, which exactly equals the
  `route_complete_before_deadline = False` count. `PLANT`-stage routes are
  blocked **15/15**. The `enable_row_helpers=False` path is therefore not the
  constraint: splitting rows across more workers cannot help when the blocker is
  missing seeds.
- **Cash is not the constraint.** `market_blocked` is empty and
  `market_no_progress_failures` is empty in the whole game while cash climbs to
  ~79,000. Only **4 buy orders** are ever submitted (1 `BUY_SEED:TOMATO`,
  3 `BUY_SEED:WHEAT`) against route requirements of STRAWBERRY 41, WHEAT 41,
  TOMATO 8, MELON 6. Seed demand is only raised for *spatially represented*
  `PLANT` work (`strip_market.py`: `item.kind != "PLANT" or item.tile is None`),
  so days whose routes require `STRAWBERRY=9` raise no buy demand at all.
- Turning purchases on to capture that demand **loses** money, because final
  bank is the score and the seeds do not pay back before the horizon.

**Conclusion:** with the manager's plan held fixed, the executor is already
mechanically saturated on labour, time and execution. The residual 16-18% is
bounded by the manager's crop/resource strategy, which this run is explicitly
forbidden to redesign or retrain. Further money would have to come from the
manager, not the executor.

---

## Baseline profile of `a01e75c` (seed 41003, in-process cProfile)

Profiled wall 91.6 s under cProfile (~40.9 s unprofiled). Total profiled
tottime 89.0 s. Attribution by owner:

| owner | tottime | share |
|---|---:|---:|
| stdlib `copy` (+ its C builtins) | ~30 s | ~34% |
| jax (`compiler.py`, 156 calls) | 8.0 s | 9.0% |
| `executor_v0/strip_prefix_trie.py` | 7.7 s | 8.6% |
| `executor_v0/strip_routes.py` | 2.2 s | 2.5% |
| `executor_v0/strip_cost.py` | 3.6 s | 4.0% |
| `executor_v0/strip_work.py` | 0.9 s | 1.0% |

Cumulative entry points: `strip_executor.act` **42.97 s** over 1,142 calls
(= 2 seats x 571 delegated turns), of which `plan_strip_hiring` is 30.41 s
over 134 calls and the large-board frontier packing under `strip_routes` is
25.26 s.

### Exact `copy.deepcopy` attribution (seed 41003)

`scripts/attribute_deepcopy.py`, one real game, 4,621 calls / 6.02 s CPU:

| call site | calls | CPU s | median object size |
|---|---:|---:|---:|
| `rl_manager/executor_factory.py:107` | 1,142 | **3.984** | 70,396 |
| `evaluation/agent_match.py:156` | 1,438 | 1.141 | 7,361 |
| `evaluation/agent_match.py:161` | 1,438 | 0.672 | 6,088 |
| `rl_manager/executor_factory.py:116` | 2 | 0.172 | 2,164,787 |

So **~6 s of a ~41 s game is pure defensive copying**, and the single largest
item is a per-turn deepcopy of executor diagnostics that retains only 48
snapshots (24 days x 2 seats) out of 1,142.
