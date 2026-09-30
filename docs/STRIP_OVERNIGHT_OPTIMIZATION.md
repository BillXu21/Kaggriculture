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

- branch/SHA: `overnight/strip-speed-money` @ `a01e75c` (start of run)
- panel wall: 84.1 s (inherited, measured under ~3.8/20 cores of foreign load)
- exact-parity status: PASS (4/4 banks, 4/4 digests)
- bank: mean 71,353.1

## Current best economic candidate

- branch/SHA: `overnight/strip-speed-money` @ `a01e75c`
- mean bank: 71,353.1
- per-seed: 72587/70810, 74061/75159, 79167/73221, 64229/61591
- speed: 84.1 s panel wall

## Pareto candidates

| commit | panel wall | mean bank | min bank | parity | notes |
|---|---:|---:|---:|---|---|
| `a01e75c` | 84.1 s | 71,353.1 | 61,591 | exact | inherited baseline |

## Dead ends

(none yet)

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
