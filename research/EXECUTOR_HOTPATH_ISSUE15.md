# Issue #15 Executor Hot-Path Profile

Date: 2026-08-27

## Scope

This worktree starts at `e63e8337ba9e30a6f394d69da23da538ed7ad6c2` on
`executor-v07-fixed-plan`. The benchmark uses the externally supplied BC-E
checkpoint at `artifacts/local/bc-v1-E/best.pt` from the user's main checkout;
the checkpoint is not copied or modified.

The local machine is Windows 11, Python 3.13.1, and the repository fast native
extension. Kaggle TPU-host numbers were not reproduced locally.

## Profile Method

The dedicated helpers are:

- `scripts/profile_selfplay_agent.py`: fixed-seed complete games, cProfile and
  runtime wrappers;
- `scripts/benchmark_selfplay_agent.py`: warmup-discarded wall-clock runs for
  N=1, 2, and 8.

The profile uses strict default executor construction, so a fallback aborts the
run instead of being hidden. Fixed seeds begin with `17` for N=1 and `17, 42`
for N=2. Each complete game is reset plus 719 accepted primitive turns.

Before optimization, no-cProfile N=1/N=2 profile results were:

| metric | N=1 | N=2 |
|---|---:|---:|
| steady seconds | 3.922 | 8.120 |
| `agent_actions` seconds | 2.378 | 4.473 |
| `agent_actions` share | 60.6% | 55.1% |
| deep-copy seconds | 1.049 | 2.107 |
| canonical-board calls | 5,036 | 10,072 |
| canonical-board seconds | 0.194 | 0.349 |
| task-generation seconds | 0.708 | 1.205 |
| foreman seconds | 0.245 | 0.426 |
| turn-snapshot diagnostic seconds | 0.382 | 0.735 |
| new-day setup seconds | 0.011 | 0.024 |

The cProfile run independently showed `copy.deepcopy` as the largest single
function family. On ordinary executor turns, the old path built a canonical
board in task generation, twice while deriving feed state, and once more for
achieved diagnostics; day setup added another boundary scan. The opening
wrapper's non-delegation work was small and was not changed.

After canonical snapshot reuse, the no-cProfile N=1/N=2 profile results were:

| metric | N=1 | N=2 |
|---|---:|---:|
| steady seconds | 3.661 | 7.861 |
| `agent_actions` seconds | 2.117 | 4.112 |
| canonical-board calls | 1,246 | 2,492 |
| canonical-board milliseconds/turn | 0.069 | 0.068 |
| task-generation milliseconds/turn | 0.781 | 0.765 |
| foreman milliseconds/turn | 0.350 | 0.311 |
| turn-snapshot milliseconds/turn | 0.527 | 0.503 |

The profile wrappers add measurement overhead, so wall-clock comparisons below
use the separate uninstrumented benchmark helper.

## Changes

1. `ExecutorAgent._act` canonicalizes the current own board exactly once and
   passes that snapshot to day setup, task generation, feed derivation, sell
   selection, optional cleanup, and achieved diagnostics.
2. Task-generation and optional-cleanup APIs accept the already canonical board
   while retaining their old standalone behavior when it is absent.
3. `AgentConfig.record_turn_snapshot` defaults to `True`, preserving existing
   diagnostics. `RunnerConfig.low_telemetry=True` selects a strict executor
   configuration with turn snapshots disabled for RL training rollouts.
4. `RunnerConfig.read_only_agent_observations=True` supplies recursively
   read-only dict/list-shaped views to agents, eliminating per-call deep copies
   without allowing accidental nested mutation. It is opt-in and defaults to
   the old deep-copy behavior.

No strategy, lifecycle, task priority, market, opening, action-space, engine,
or worker-topology behavior was changed.

## End-to-End Benchmark

Command shape for each row was one warmup followed by two measured repeats,
using seeds `17,42,2026,7,123,1013,1022,1003` and `numThreads=1`. Values are
medians of measured repeats on the local host.

| mode | N | steady seconds | games/s | primitive turns/s | agent_actions s | agent share |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 3.681 | 0.272 | 195.3 | 2.218 | 60.3% |
| optimized default | 1 | 4.357 | 0.229 | 165.0 | 2.489 | 57.1% |
| low telemetry + read-only | 1 | 2.315 | 0.432 | 310.6 | 0.825 | 35.6% |
| base | 2 | 7.656 | 0.261 | 187.8 | 4.632 | 60.5% |
| optimized default | 2 | 6.892 | 0.290 | 208.7 | 4.002 | 58.1% |
| low telemetry + read-only | 2 | 4.933 | 0.405 | 291.5 | 1.720 | 34.9% |
| base | 8 | 28.630 | 0.279 | 200.9 | 17.615 | 61.5% |
| optimized default | 8 | 20.605 | 0.388 | 279.2 | 12.367 | 60.0% |
| low telemetry + read-only | 8 | 19.118 | 0.418 | 300.9 | 6.876 | 36.0% |

The default mode improved N=2 by 11.2% and N=8 by 38.9% games/s in this
two-repeat run. N=1 was noisy and 18.4% slower in the default comparison, so
that result is not claimed as a default N=1 gain. The explicit training mode
improved games/s versus base by 59.0%, 55.1%, and 49.8% at N=1, N=2, and N=8;
its `agent_actions` reduction was 62.8%, 62.9%, and 61.0% respectively.

## Behavior Parity

- Base and optimized default seed-17 complete games both finished
  `DONE/DONE`, with final banks `[28505.0, 25662.0]`.
- Both produced the same 719-joint-action SHA-256 fingerprint:
  `897ba48bed992da5461a826a621077bc1a5af76c719a92718303d77882f83ad8`.
- Base and optimized default seed-17 action trace digests both equal
  `c988de9d1a1355c0d87e017f0eed893984d129a596a72b02eb0762203dea596c`.
- The two-seed N=2 banks and trace digests also match exactly:
  `[28505.0,25662.0]` / `c988de9d...3dea596c` and
  `[12487.0,11736.0]` / `11eb1e...ef454d`.
- The training mode has the same banks, statuses, and action digests as the
  default optimized mode for N=1, N=2, and N=8 benchmark cases.
- Focused executor/runner tests: `51 passed, 1 skipped`; the skip is the
  existing sample/official-environment gate.
- Ruff and bytecode compilation passed for all changed Python files.

## Rejected or Deferred Candidates

- Opening-agent rewrite: rejected because measured wrapper guard/replay work
  was negligible relative to executor, copies, and diagnostics.
- Planner/layout fusion and broad sorting/allocation rewrites: deferred because
  they have larger ordering/parity risk and the measured canonical/copy/snapshot
  changes already provide the dominant safe gain.
- Removing deep copies from the default runner: rejected for default behavior;
  injected/custom agents do not have a repository-enforced no-mutation
  contract. The safe read-only view is explicit instead.
- Disabling diagnostics by default: rejected; normal/default executor
  diagnostics remain unchanged. `low_telemetry` is explicit for training.

## Integration

No dependency on issue #16's native batched/backend work or issue #17's
process/TPU-owner topology was introduced. Both can use the explicit
`low_telemetry` and `read_only_agent_observations` runner knobs when composing
future rollout workers, but neither issue's owned files were modified.
