# Kaggriculture Current State

Last updated: 2026-08-26

## Active Direction

The final 24-game real-BC-E / V0.7 PASS generalization panel **did not validate the lower tail**. The middle of the distribution remains healthy, but three games collapsed below 1k bank.

Current phase:

**visual replay forensics -> targeted log/mechanical-capacity diagnosis -> accept/fix/freeze executor -> throughput profiling/scaling -> small RL initialization experiments.**

Current architecture remains:

`standard_mixed opening d0-d3 -> learned daily manager -> deterministic executor`

A learned tactical/middle controller and primitive-action RL remain deferred. Scratch or partially reset daily-manager training is now a credible future experiment, but it is not yet a committed architecture change.

Diagnostic rule:

> Did the executor fail to execute a feasible strategy, did the manager choose an unsustainable strategy, or did a marginal strategy become catastrophic because of executor inefficiency?

Only an obvious, reproducible failure to execute a feasible strategy normally justifies another executor patch.

## Final 24-Game V0.7 Panel

Real BC-E checkpoint, PASS opponent, prior-debt suppression ON, both seats, seeds:

`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`

Aggregate:

- games: `24`
- mean bank: `28,587.4`
- median bank: `25,321.5`
- minimum: `2`
- maximum: `65,959`
- `<1k`: `3`
- `<10k`: `3`

Old V0.6 approximate reference:

- mean about `30,110`
- median about `24,358`
- minimum `8,062`
- `<1k = 0`
- `<10k = 2`

Interpretation: V0.7 has a slightly better median but worse mean and dramatically worse catastrophic tail. Do **not** call generalization validated. The panel's printed `losses` field was `None`, so do not infer starvation/loss-unit totals from this run without separate trace evidence.

Per-game banks:

| seed | seat 0 | seat 1 |
| ---: | ---: | ---: |
| 7 | 18,319 | **2** |
| 17 | 17,005 | 14,961 |
| 42 | 23,346 | 26,587 |
| 123 | 25,077 | 25,077 |
| 2026 | 56,742 | **65,959** |
| 1013 | 58,122 | 60,983 |
| 1022 | 23,531 | 22,712 |
| 1003 | 25,819 | 25,566 |
| 1026 | 31,840 | 23,341 |
| 1011 | 29,001 | 35,309 |
| 1024 | 26,673 | 49,994 |
| 1019 | **76** | **55** |

The pattern suggests at least one discrete collapse mechanism rather than uniformly poor executor efficiency: seed 7 is highly seat-sensitive, while seed 1019 collapses in both seats.

## Immediate Visual Forensics

Generate/inspect replay traces for a small distribution-spanning set:

1. `seed 7 / seat 1` — asymmetric catastrophe, bank `2`;
2. `seed 1019 / seat 0` — catastrophe, bank `76`;
3. `seed 1019 / seat 1` — catastrophe, bank `55`;
4. `seed 7 / seat 0` — same-seed healthy control, bank `18,319`;
5. `seed 2026 / seat 1` — excellent trajectory, bank `65,959`;
6. optional `seed 1013 / seat 1` (`60,983`) or a median trajectory for another control.

For each replay, identify only the **first visibly bad day** and one or two days before it. Look for:

- aggressive expansion / animal or crop target jumps;
- farm geometry or workload complexity;
- worker travel, duplication, task stealing, or idle time;
- missed must-water / harvest / feed deadlines;
- shed congestion, overflow, or unplaceable animals;
- market ordering / cash collapse;
- the transition from recoverable overload to irreversible death spiral.

Use the stock Kaggle viewer for holistic pattern recognition and the custom passive trace only when causal manager/task/assignment/debt data is needed.

## Targeted Log / Feasibility Diagnosis

After the visual pass identifies the first bad window, inspect that bounded day range in logs rather than analyzing all 720 turns blindly.

Desired classification:

1. **manager failure** — requested plan is provably unsustainable;
2. **executor failure** — plan is mechanically feasible but scheduling/routing/market execution misses it;
3. **interaction failure** — plan is near the feasibility frontier and executor inefficiency pushes it over the edge.

Useful diagnostics around the first bad day:

- requested vs projected/feasible plan;
- land/crop/animal target changes;
- workers/hires and available worker-actions;
- conservative lower bound on mandatory work;
- EOD work debt by task class;
- expansion-suppression events;
- must-water / harvest / feed misses;
- shed occupancy and overflow;
- skipped buys and reasons;
- animals bought vs remaining place/store capacity;
- cash start/end and irreversible loss events.

A particularly useful causal test remains replaying the **same captured BC-E DailyPlan tape** through V0.6 and V0.7. If V0.6 survives a plan tape that V0.7 kills, prioritize finding an executor regression. If both fail similarly, treat it primarily as manager/executor compatibility.

## Executor V0.7 Status

Behavioral candidate remains frozen at `a7c826d` on branch `executor-v07-fixed-plan`, but the executor is **not yet accepted as generalization-safe** because of the new catastrophic tails.

Do not restart R5/R6/R7 or broad heuristic search. Allow only one bounded obvious-mechanical-defect pass driven by visual/log evidence.

Known retained rule:

- prior-day work-debt expansion suppression remains ON as explicit architectural debt because the earlier OFF ablation catastrophically failed.

Known accepted mechanical fix:

- survival WHEAT purchases respect remaining shed capacity.

The earlier durable-decision ID collision has been fixed on `executor-v07-fixed-plan`: the branch's R4/prior-debt decision is now **D-037**, preserving main's existing D-031..D-036. Merge still requires deliberate reconciliation of the diverged documentation histories.

## Real BC-E Checkpoint

Local ignored artifact:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

- variant: E-own;
- epoch: 27;
- validation total: `2.910865758929336`;
- SHA-256: `F4B029D3E463ABA1DBD0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`.

Never commit it.

## RL Direction Under Consideration

Current hypothesis: BC-E may know strong elite-player strategy but request plans that assume more precise execution than the current heuristic can reliably provide. A closed-loop-trained manager may need to learn around the executor's actual production frontier.

Candidate initialization experiments after executor acceptance/freeze:

1. **full BC-E init** — maximum inherited game knowledge and maximum inherited elite habits;
2. **BC-E trunk + reset action heads** — retain state representation while relearning strategy around this executor;
3. **scratch daily manager** — cleanest executor-aware policy, likely needing a stronger curriculum.

A longer deterministic opening is a promising curriculum lever. Keep the optimized opening for the early game and consider extending it beyond d3 so scratch/near-scratch learning begins from a productive farm with a shorter effective decision horizon.

### Reward idea — planned experiment, not frozen contract

Prefer one terminal episode reward rather than hard-to-define dense economic shaping.

Initial candidate ingredients:

- continuous final-cash term, normalized by a fixed constant;
- explicit bankruptcy penalty using the engine's real bankruptcy/loss condition;
- smaller terminal competitive outcome term early in training.

Early training should prioritize **make money + survive** before pure win optimization. If policy quality becomes high, reduce cash shaping and move toward outcome-dominated or outcome-only training.

Do not add a plan-infeasibility penalty by default. If needed later, it must be based on a conservative mechanically provable workload lower bound, not merely executor EOD debt, so executor bugs are not mislabeled as manager mistakes.

## Throughput / 96-Core Readiness

Before serious self-play, profile and scale the rollout stack. Measure separately:

1. engine stepping;
2. executor `_act`;
3. canonical-state conversion / encoding;
4. IPC/orchestration/trajectory bookkeeping;
5. batched JAX policy inference.

Benchmark persistent environment concurrency approximately at:

`1 -> 4 -> 12 -> 24 -> 48 -> 96`

Prefer many independent single-threaded games unless measurement proves otherwise. Batch all day-boundary policy requests by immutable policy identity so many CPU environments feed large accelerator inference batches.

Rust executor ports are conditional on profiling. If Python executor time is substantial, consider one parity-tested Rust executor core capable of reproducing frozen V0.6 and V0.7 behavior. Do not port first and measure later.

## JAX / TPU Status

TPU work remains paused.

- Synthetic/random PyTorch <-> JAX parity was green.
- Real-checkpoint parity test still has the known test bug where trained JAX parameters are compared against a fresh random PyTorch model.
- Kaggle TPU subprocesses repeatedly failed TPU backend initialization.
- No real TPU throughput measurement exists.

CPU rollout scaling is now a more immediate performance question than isolated TPU debugging.

## Explicitly Deferred

- R5/R6/R7 and broad executor heuristic search;
- deliberate crop digging / advanced price-crash abandonment;
- corner/serpentine/Top-1 geometry imitation;
- new arbitrary debt/cash thresholds;
- learned tactical/middle layer;
- primitive-action movement RL;
- custom viewer frontend redesign;
- sophisticated league/PFSP work before a basic candidate-vs-frozen policy loop works;
- Rust port unless profiling shows executor cost is material;
- TPU debugging that does not block the real training path.

## Full Session Compaction

Prior frozen-state handoff: `research/PROJECT_CHECKPOINT_2026-08-25_V07_FREEZE.md`.

New post-panel planning details should be kept in `research/POST_V07_GENERALIZATION_PLAN_2026-08-26.md`.