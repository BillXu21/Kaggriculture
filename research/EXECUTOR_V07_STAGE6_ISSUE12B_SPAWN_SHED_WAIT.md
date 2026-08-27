# Executor V0.7 Stage 6 Issue #12B: Spawn/Shed Waiting Classification

- Date checked: 2026-08-27
- Status: **non-reproduction; no executor patch**
- Source artifact: `artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert.json`
- Artifact file SHA-256: `bd9dd3ef58f7b87f33aa2fcad980aa826a6a1ccfcf94012bb491927853027546`
- Artifact internal content SHA-256: `8862450c5036b5791c1a7d843220fe274ffb73b90022ed7856e892611f3db159`
- Source revision: `7204103fc2ca6e351cc2a95506c961977d199bb2`
- Shed-reservation revision: `cf282a1665579b663dfd323ee5636407976a048a` (ancestor of the source revision)
- Confidence: `CONFIRMED_EXPERIMENT` for observed trace facts; classification is bounded to the available fields

## Detection and counts

The 24-game isolated Stage 4 turn-trace artifact was scanned for consecutive
per-worker `PASS` assignments of at least eight turns, retaining worker, board
position, day/hour span, and neighboring feed/market fields.

- 69 total long `PASS` runs were found.
- **Classification 1 — shed-zone wait:** 24 runs, exactly one per game; every
  run is worker index 1 (`hand_0`) at `[4,5]` from `d4h1` through `d4h10`.
- **Classification 2 — non-shed wait:** 45 runs; these occur during hiring ramp
  or late-game labor surplus and are not spawn/shed failures.
- All 24 games completed with 719 transitions and `DONE/DONE` status.

## Representative reconstruction: seed 7, seat 0

The `d4h0` trace submits an affordable WHEAT survival buy. During the fill
latency, the early FEED workload is present, but worker 1 at `[4,5]` records
`PASS` for `d4h1`–`d4h10` while other workers clear the starvation/feed work.
The worker resumes at `d4h11` with work toward `COLLECT_FERTILIZER`; starvation
is then false and the feed shortage is zero. There is no animal-count decrease
in this interval, and the trace has no unaffordable order, fallback/day/status
error, or repeated pickup overdraw.

This is legitimate waiting caused by market fill latency and feasible-task
availability, not Kaggle AFK and not a bookkeeping defect. The `cf282a1`
`shed_budget` reservation is present in `executor_v0/foreman.py` and prevents
concurrent workers from over-picking the same shed quantity.

## Observability and conclusion

The artifact does not contain per-turn `assignment.reason` or complete private
shed/carried-inventory snapshots. Therefore the trace cannot prove every
engine-side acceptance or reconstruct the full private inventory transition;
the classification uses the recorded assignments plus inferred market/feed
signals. No critical work or animal loss was observed, so no instrumentation or
executor behavior change is justified tonight.

This finding is separate from the resolved Kaggle AFK packaging omission:
that issue concerned submission archive/runtime packaging, whereas this is a
local fast-backend executor trace. It does not reopen the packaging fix.

Optional one-game future handoff (not run as part of this note):

```text
python -m tools.run_executor_v07_panel --checkpoint "<BC-E best.pt>" --seeds 7 --seats 0 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --output "artifacts/executor-v07-stage6-issue12b-seed7-seat0.json" --label "executor-v07-stage6-issue12b-seed7-seat0"
```
