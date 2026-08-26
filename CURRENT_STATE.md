# Kaggriculture Current State

Last updated: 2026-08-25

## Active Direction

The executor-debugging phase is effectively complete. The project now returns to the learning-first path:

**one final V0.7 generalization panel -> merge/freeze the two-layer stack -> real BC-E closed-loop baseline -> PPO smoke -> small self-play development run.**

Current architecture remains:

`standard_mixed opening d0-d3 -> learned daily manager -> deterministic executor`

A learned tactical/middle controller and primitive-action RL remain deferred until real self-play shows the daily manager + deterministic executor boundary is insufficient.

Diagnostic rule:

> Did the executor fail to execute a feasible strategy, or did the manager choose an unsustainable strategy? Only the first normally justifies an executor patch.

## Git / Branch State

- Main docs tip before the latest compaction commits: `903163c704cbda852e11104f33bf8e4f06bb3f06`.
- Behavioral merge base for the active executor work: `3726e373c65b8221c4062138174898f6cf756119`.
- Final executor branch: `executor-v07-fixed-plan`.
- Final branch HEAD: `b1b9b306b48a6ae3fcb2464109088e7ecea91b7c`.
- Important V0.7 commits: `02984a0` panel outcome ledger, `a7c826d` shed-room survival-buy fix/frozen behavior, `b1b9b30` final evidence/docs.
- Branch is pushed to `origin/executor-v07-fixed-plan`, clean, not merged.
- It is currently 15 commits ahead and 4 behind main, with merge base `3726e373...`; reconcile the two documentation histories deliberately when merging.

## Real BC-E Checkpoint

Local ignored artifact:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

- variant: E-own;
- epoch: 27;
- validation total: `2.910865758929336`;
- SHA-256: `F4B029D3E463ABA1DBD0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`.

Never commit it.

## Executor V0.7 — Frozen Behavior

V0.7 behavior is frozen at `a7c826d` pending the final 24-game ON panel and merge.

### R4 rejected

The narrow watering reservation R4 was rejected/reverted after real BC-E fixed-plan regression. Expert-intent improvements did not survive the real policy distribution. No R5/R6/R7 search.

### Accepted mechanical fix

Survival WHEAT purchases now respect remaining shed capacity. This fixed a real seed-17 failure while preserving survival-before-hire ordering.

Seed-17 PASS banks improved from V0.6-era `8,062 / 8,489` to V0.7 `17,005 / 14,961`.

### Prior-debt suppression remains ON as architectural debt

Real BC-E PASS panel on seeds `17,42,2026`, both seats:

| mode | mean bank | median | minimum | starvation loss units |
| --- | ---: | ---: | ---: | ---: |
| prior-debt ON | 34,100 | 24,966.5 | 14,961 | 0 |
| prior-debt OFF | 98.8 | 31.5 | 0 | 38 |

OFF is a catastrophic manager-policy collapse, not a subtle executor difference. Keep the veto ON for the first RL experiments as an explicit safety/curriculum shield, but do not redefine it as a permanent mechanical invariant. Log how often it fires; a better learned policy should eventually stop requesting vetoed expansion and make the shield removable.

Do not run another broad OFF panel.

### Residual seed-17 day-22 loss

The branch classifies the remaining six-animal/seat loss as overflow/manager-policy debt: carried cows reach day end while the shed is full and are discarded. It is not starvation. No extra heuristic was added. If revisited, inspect only whether an animal was mechanically purchased/left impossible to place or store in the remaining window; otherwise leave it to manager/RL capacity learning.

### Validation

Final V0.7 branch report:

- `686 passed, 104 skipped, 0 failed`;
- four real BC-E traces, 720 turns each, schema-valid;
- trace-on/off action/result parity passed;
- Node viewer probes passed;
- CLI smoke passed;
- worktree clean;
- no large trace JSON committed.

Detailed branch note: `research/EXECUTOR_V07_FINAL.md`.

## Viewer

Issue #11's passive canonical debug-trace instrumentation was integrated into the V0.7 branch. It is useful as a causal sidecar for manager/task/assignment/feed/water/debt diagnosis.

Manual use showed the stock Kaggle viewer is much better for seeing overall farm patterns. Do not spend current project time redesigning the custom frontend. Later, prefer augmenting or visually following the Kaggle viewer rather than building a second polished game renderer.

A browser helper-scope bug (`number` / `json`) was discovered during manual use and sent for a permanent local fix; verify the integrated branch contains the fix before relying on an older viewer tip.

## Manager Contract Audit

The feared crop-contraction interface blocker is mostly absent:

- crop labels are absolute end-of-day composition;
- count heads can predict lower values;
- RL decode writes absolute `crop_targets`;
- executor projection preserves crop targets;
- crop reconciliation can release crops above target.

So crop contraction is representable in principle. Deliberate digging/price-crash crop abandonment is advanced behavior and is not required now; the replay corpus may contain too few demonstrations to expect BC to learn it reliably.

Animals intentionally remain non-destructive: projection uses `max(current, requested)` because there is no normal strategic remove/kill action. A lower animal target means stop adding/replacing, never deliberately starve existing animals.

Do not redesign the manager action space without measured closed-loop evidence.

## JAX / TPU Status

TPU work is paused.

- Synthetic/random PyTorch <-> JAX parity was green for 30 tests.
- The newly activated real-checkpoint parity test failed, but source inspection found a test bug: the JAX side loads trained checkpoint parameters while the PyTorch reference model is freshly initialized and never loads `model_state_dict`. Fix that test before interpreting the failure as JAX drift.
- Kaggle TPU-focused subprocess tests repeatedly failed backend initialization. Restarting/avoiding obvious early JAX use did not resolve it.
- No real TPU throughput measurement exists; do not quote one.

This is not currently the project blocker. Resume TPU debugging only when it blocks the actual RL path.

## Immediate Next Experiment

Run one final **24-game V0.7 prior-debt-ON PASS generalization panel** using the real BC-E checkpoint, both seats, seeds:

`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`

Use `tools.run_executor_v07_panel` from `executor-v07-fixed-plan`, official backend if practical.

Compare against the old V0.6 24-game reference (approximately mean `30,110`, median `24,358`, min `8,062`, `<1k=0`, `<10k=2`). Only ask whether the shed-room fix preserves known-collapse recoveries, introduces new catastrophes, and keeps starvation near zero.

Do not turn this into a new heuristic loop.

## After the Panel

If the final ON panel is credible:

1. reconcile/merge `executor-v07-fixed-plan` into main;
2. treat issue #7 / Executor V0.7 as mechanically frozen;
3. run the real BC-E closed-loop/self-play baseline through the RL rollout infrastructure;
4. run the existing PPO plumbing smoke initialized from BC-E;
5. run a small candidate-vs-frozen-BC-E development experiment with the debt shield ON;
6. log expansion-suppression frequency as a learning diagnostic;
7. only then revisit reward shaping, decision frequency, tactical layers, or executor changes if closed-loop evidence isolates the need.

## Explicitly Deferred

- deliberate crop digging / advanced price-crash abandonment;
- more watering/task-assignment heuristics;
- corner/serpentine/Top-1 geometry imitation;
- new debt/cash thresholds;
- tactical/middle-layer RL;
- raw primitive-action movement RL;
- broad executor sweeps;
- viewer frontend redesign;
- TPU debugging that does not block the real RL path.

## Full Session Compaction

See `research/PROJECT_CHECKPOINT_2026-08-25_V07_FREEZE.md` for the durable long-session handoff and exact evidence/interpretation boundaries.
