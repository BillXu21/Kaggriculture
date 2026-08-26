# Project Checkpoint — 2026-08-25 V0.7 Freeze / RL Handoff

This is the compact handoff record for the long 2026-08-25 session. It supersedes the earlier same-day interim checkpoint where facts conflict.

## Core direction

Keep the current two-layer architecture through the first real RL experiments:

`standard_mixed opening d0-d3 -> learned daily manager -> deterministic executor`

The executor is a strategy compiler, not a second strategic policy. Exact mechanics, legality, routing, prerequisites, bookkeeping, minimum-safe maintenance, and mechanically preventable asset loss belong in deterministic code. Expansion pace, crop/animal mix, liquidity, contraction/abandonment, and market strategy belong to the learned manager/RL.

Default diagnosis: **did the executor fail to execute a feasible strategy, or did the manager choose an unsustainable strategy?** Only the first normally justifies another executor patch.

Do not add a learned tactical/middle layer or primitive-action RL before the frozen two-layer stack has been exercised in real self-play. Do not resume broad executor heuristic/layout search without new closed-loop evidence.

## Git state

- Main before this checkpoint: `903163c704cbda852e11104f33bf8e4f06bb3f06` (docs-only commits on top of behavioral base `3726e373c65b8221c4062138174898f6cf756119`).
- Final V0.7 branch: `executor-v07-fixed-plan`.
- Final branch HEAD: `b1b9b306b48a6ae3fcb2464109088e7ecea91b7c`.
- Important branch commits:
  - `02984a0` — panel market-outcome ledger;
  - `a7c826d` — shed-room survival WHEAT buy fix / frozen V0.7 behavior;
  - `b1b9b30` — final V0.7 docs/evidence.
- Branch is pushed to `origin/executor-v07-fixed-plan`, clean, not merged.
- Compared with current main it is 15 commits ahead and 4 behind, merge base `3726e373...`; reconcile docs deliberately rather than blindly merging both histories.

## Real BC-E checkpoint

Local ignored artifact:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

Metadata/provenance recorded by the V0.7 branch:

- variant E-own;
- epoch 27;
- validation total `2.910865758929336`;
- SHA-256 `F4B029D3E463ABA1DBD0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`.

Never commit the checkpoint.

## Executor V0.7 final result

V0.7 is mechanically frozen at `a7c826d` pending one final generalization panel / merge decision.

### R4

R4 (reserve exact weed-boundary WATER for an on-tile worker) was **rejected and reverted** after real BC-E fixed-plan regression. Its expert-intent gains did not generalize to the actual BC-E policy distribution. Recorded regression included wealth `-14,302`, cash `-9,515`, weeds `+11`, crops destroyed `+14`, survival debt `69 -> 84`, starvation `22 -> 44`, and harvest `124 -> 98`.

No R5/R6/R7 search. This closes the watering-reservation heuristic thread.

### Accepted mechanical fix

A real seed-17 defect was found in survival WHEAT purchasing: purchases could ignore remaining shed capacity. WHEAT survival buys now respect available shed room while retaining survival-before-hire ordering.

Seed-17 PASS banks improved from the V0.6-era `8,062 / 8,489` to V0.7 `17,005 / 14,961`.

### Prior-debt suppression ablation

The broad previous-day EOD-debt expansion veto remains enabled by default as **explicit architectural debt**, because removing it currently destroys BC-E closed-loop stability.

Real BC-E PASS panel, seeds `17,42,2026`, both seats:

| Mode | Mean bank | Median | Minimum | Starvation loss units |
| --- | ---: | ---: | ---: | ---: |
| prior-debt suppression ON | 34,100 | 24,966.5 | 14,961 | 0 |
| prior-debt suppression OFF | 98.8 | 31.5 | 0 | 38 |

Per-game ON banks: `17,005, 14,961, 23,346, 26,587, 56,742, 65,959`.

Interpretation: this is no longer a subtle executor tuning question. BC-E currently over-expands badly enough that the veto behaves as a safety/curriculum shield. Keep it for the first RL experiments, but do not redefine it as a permanent mechanic. Log how often it fires. A successful RL policy should eventually request fewer vetoed expansions and make the shield removable.

Do **not** run another broad OFF panel merely to reconfirm collapse.

### Residual seed-17 animal loss

The branch documents the remaining day-22 six-animal/seat loss as manager-policy debt: cows are carried into day end while the shed is full and are discarded on refresh; this is overflow, not starvation. No new executor heuristic was added. If revisited visually, ask only whether the executor knowingly purchased/left an animal impossible to place/store in the remaining mechanical window; otherwise treat it as manager capacity/overcommitment debt.

### Validation

Final branch report:

- Python suite: `686 passed, 104 skipped, 0 failed`;
- four real BC-E debug traces validate schema and have 720 turns each;
- trace-on/off action/result parity passed;
- Node viewer probes passed;
- CLI smoke passed;
- worktree clean;
- no large trace JSON committed.

Authoritative detailed note on the branch: `research/EXECUTOR_V07_FINAL.md`.

## Viewer status

Issue #11 instrumentation was integrated into the V0.7 branch rather than merged independently. The canonical debug-trace/CLI/snapshot path is passive and useful for causal executor diagnosis.

The custom visual presentation is not a strategic priority. In use, the default Kaggle viewer is materially easier for holistic farm-pattern inspection because it has the polished game imagery/layout. Treat the custom viewer primarily as a diagnostic sidecar for manager/task/assignment/feed/water/debt metadata. A future improvement can augment or visually mimic the existing Kaggle viewer rather than building a second full visual frontend.

A browser-render helper-scope bug (`number` / `json` unavailable outside `ViewerCore`) was discovered during manual use and was sent for a permanent local fix; do not assume an older standalone viewer tip includes that fix unless checked.

## Manager contract audit — current conclusion

The feared crop-contraction interface bug is mostly **not present**:

- BC crop labels are absolute end-of-day crop composition targets;
- count heads can output lower counts;
- RL decode writes those absolute values into `DailyPlan.crop_targets`;
- executor projection passes crop targets through;
- crop reconciliation can release crops above target.

Therefore crop contraction is representable in principle. Deliberate digging/price-crash crop abandonment is an advanced behavior and is not a current prerequisite; the replay corpus may contain too few examples to expect BC to learn it reliably yet.

Animals are intentionally different: projection uses `max(current, requested)` because there is no normal strategic remove/kill action. A lower animal target should mean stop adding/replacing, never deliberately starve an existing animal.

Before PPO redesign, measure actual BC-E target behavior rather than changing the action space speculatively.

## JAX / TPU experiment status

The JAX/TPU exploration is **paused**, not failed as an architectural direction.

Useful findings:

1. Synthetic/random PyTorch <-> JAX parity suite was green for 30 tests before the real checkpoint test.
2. The newly activated real-checkpoint parity test failed with ~9.78 max crop-logit difference, but source inspection found the test itself compares trained JAX checkpoint parameters against a freshly initialized PyTorch model because it never loads `payload["model_state_dict"]` into the PyTorch reference. Fix the test before interpreting that failure as model-port drift.
3. Kaggle TPU focused subprocess tests repeatedly failed to initialize the TPU backend. Restarting and trying to avoid early notebook JAX initialization did not resolve it. Do not spend more session time on this until the executor/BC baseline work is complete.

No real TPU throughput numbers were obtained. Do not quote any TPU performance claim.

## Immediate next experiment

Run one final **24-game V0.7 ON generalization panel** against fixed PASS using the real BC-E checkpoint, then stop executor evaluation unless it exposes a clear catastrophic mechanical regression.

Seeds, both seats:

`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`

Use `tools.run_executor_v07_panel` on `executor-v07-fixed-plan`, official backend if practical, prior-debt suppression ON. Compare with the old V0.6 24-game reference:

- mean bank about `30,110`;
- median about `24,358`;
- minimum `8,062`;
- `<1k = 0`;
- `<10k = 2`;
- old residual animal drops concentrated on seed 17.

Questions only:

- does the shed-room fix preserve the known-collapse recoveries?
- are there new `<1k` / `<10k` catastrophes?
- does starvation remain near zero beyond the three development seeds?
- what is the final bank distribution relative to V0.6?

Do not run another OFF panel or use this as a new heuristic-search loop.

## After the 24-game panel

If the ON panel is credible:

1. merge/reconcile `executor-v07-fixed-plan` into main, preserving the newer main continuity docs and the branch's V0.7 implementation/evidence;
2. consider issue #7 mechanically complete/frozen;
3. run the real BC-E closed-loop/self-play baseline through the exact rollout infrastructure intended for RL;
4. run the existing PPO plumbing smoke initialized from BC-E;
5. start a small candidate-vs-frozen-BC-E development run with prior-debt suppression ON;
6. log expansion-suppression frequency as a policy-learning diagnostic;
7. only revisit reward shaping, manager frequency, tactical layers, or executor behavior when closed-loop evidence isolates the need.

## Explicitly deferred

- deliberate crop digging/advanced price-crash abandonment learning;
- corner/serpentine/Top-1 geometry imitation;
- more watering/task-assignment heuristics;
- new debt-veto thresholds;
- learned tactical/middle layer;
- primitive-action movement RL;
- broad executor sweeps;
- viewer frontend redesign;
- TPU debugging until it blocks the actual RL path.
