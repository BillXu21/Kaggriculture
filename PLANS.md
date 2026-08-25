# Kaggriculture Plans

Last updated: 2026-08-25

## Strategic Objective

Build a competitive Kaggriculture agent where **reinforcement learning owns meaningful farm and market strategy** and deterministic code owns exact mechanics that are poor uses of model capacity: legality, pathfinding, worker execution, prerequisites, bookkeeping, and prevention of mechanically avoidable asset loss.

The project has enough infrastructure to stop broad executor invention. The immediate goal is to freeze a credible two-layer stack, inspect it visually, and then return to learning.

Current intended stack:

`standard_mixed d0-d3 -> learned daily manager -> deterministic executor`

A learned tactical/middle controller remains a plausible future architecture, but it is deferred until the two-layer stack has been tested in closed-loop self-play. Raw primitive-action RL is also deferred.

## Current Phase — Finish and Freeze Executor V0.7

Issue #7 is now a **bounded finish/validation packet**, not an open-ended optimization project.

Real BC-E checkpoint:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

This file is local/ignored and must never be committed.

### Required V0.7 decisions

1. **R4 watering reservation**
   - Review the narrow rule that reserves an exact weed-boundary WATER task for a worker already standing on that tile.
   - Decide exactly one: promote unchanged, make one narrow correctness repair, or reject.
   - Do not continue into R5/R6/R7 heuristic search.

2. **Real BC-E multi-day validation**
   - Use the fixed-plan 3/5/7-day harness with DailyPlan tapes recorded from the real BC-E manager where practical.
   - Keep earlier expert-intent results, but label them separately; they are executor evidence, not BC-E evidence.

3. **Seed-17 diagnosis**
   - Reproduce both seats with the actual `standard_mixed -> BC-E -> executor` stack.
   - Determine whether the remaining animal losses are mechanical (resource exists but scheduling/delivery fails) or strategic (BC-E creates an unsustainable farm/cash state).
   - Only the mechanical case should normally produce an executor patch.

4. **Prior-debt veto ablation**
   - Compare the current broad previous-day EOD-debt expansion veto against **no previous-day debt veto**, retaining only current hard survival protection.
   - Use the real BC-E checkpoint on a small deterministic panel.
   - Do not replace it with another arbitrary debt/cash threshold.
   - If removing the veto exposes a bad BC-E strategy, that is useful training signal rather than automatic justification for a hidden executor governor.

5. **Freeze V0.7**
   - Focused tests and deterministic parity must pass.
   - No opening divergence/fallback errors.
   - No mechanically avoidable animal escape when feed is physically/financially available.
   - Minimum-safe watering for maintained crops.
   - No hidden economic strategy or manager-target rewriting.

### Multi-day executor promotion ladder

Executor changes now use:

1. exact mechanics/unit tests;
2. one-day fixtures for rapid diagnosis;
3. 3-day fixed-plan A/B;
4. 5-day fixed-plan A/B;
5. 7-day fixed-plan A/B for promotion candidates;
6. bounded real BC-E full-game panel;
7. broad panels only when the candidate is already clearly strong.

One-day results remain useful but are no longer sufficient for promotion.

## Next Phase — Integrate the Debug Viewer

Issue #11 branch `issue-11-replay-debug-viewer` at validated tip `0f72bcd28ef20703718a8a16503b6776c4d4b046` is **READY TO MERGE LATER**.

Its instrumentation was independently validated as behaviorally passive: base-vs-branch primitive actions and trace-enabled-vs-disabled behavior matched under deterministic official-engine checks.

After V0.7 behavior is frozen:

1. integrate issue #11 carefully around the new executor `_act` logic;
2. preserve passive post-decision snapshot semantics;
3. rerun trace-on/off exact action parity using real BC-E;
4. generate a tiny local trace set;
5. do not commit giant trace JSON artifacts.

### Initial visual inspection set

Start with only a few real BC-E trajectories:

- seed 17 seat 0;
- seed 17 seat 1;
- seed 42 seat 0;
- one healthy/high-bank trajectory from the bounded panel.

Inspect for obvious mechanical failure only:

- feed available but not delivered;
- workers crossing or stealing exact-deadline work unnecessarily;
- repeated pickup/drop churn;
- must-water tasks serviced too late despite local capacity;
- dependency/build/place mistakes;
- clearly wasteful routing/assignment;
- inconsistent inventory/task state.

Limit this stage to **one or two small correction passes**. Do not use the viewer as an excuse to hand-code elite strategy.

## Manager Contract Audit Before Serious PPO

Before serious self-play training, audit whether the current action/projection contract can express the strategic behaviors RL is expected to learn.

Highest-priority question: **can desired crop and animal inventories decrease?**

Elite replay inspection showed strategically meaningful contraction/abandonment (for example, allowing a crashed-price crop to die rather than continuing to spend labor maintaining it). If current projection semantics clamp requested crop/animal targets upward to current inventory, the learned manager cannot express contraction at all.

Required distinction:

- manager chooses whether an asset is strategically worth maintaining/replacing;
- executor decides how to execute that intent mechanically.

Do not implement product-price abandonment heuristics inside the executor.

The audit should check every manager output for the same class of problem: whether it represents a true desired strategic state or an irreversible/additive instruction that prevents RL from exploring useful behavior.

## First Self-Play / RL Sequence

Once V0.7 and the manager contract are credible:

### 1. Closed-loop BC-E baseline

Run the frozen real BC-E manager through the frozen executor in the same rollout infrastructure intended for RL.

Measure complete-game distributions, not teacher-forced imitation quality:

- W/L/T and bank/margin;
- catastrophic collapse rate;
- animal/crop losses;
- farm size and labor trajectory;
- cash/liquidity trajectory;
- manager target changes;
- executor survival/maintenance failures;
- opponent/market regime where relevant.

### 2. PPO plumbing smoke with the real checkpoint

Use the existing `rl_manager` PPO path:

- initialize from the promoted BC-E policy;
- run a tiny complete rollout;
- GAE/logprob recomputation/update/checkpoint roundtrip;
- verify no divergence or frozen-snapshot corruption.

This is still a plumbing gate, not a policy-quality claim.

### 3. Small development run

Only after the smoke is clean:

- candidate vs frozen BC-E;
- small fixed seed set, both seats;
- terminal objective remains primary;
- inspect full trajectories and strategy changes;
- do not immediately add dense shaping when the first run is imperfect.

### 4. Diagnose before redesign

If learning fails, determine why:

- manager action space cannot express the needed adaptation;
- observation lacks relevant strategic state;
- terminal-only credit assignment is too weak;
- executor still contains a mechanical bottleneck;
- daily decision frequency is too coarse;
- a tactical middle layer is genuinely needed.

Only then consider reward shaping, richer action frequency, or a learned tactical policy.

## Executor / Manager Boundary

### Deterministic executor owns

- exact action legality and mechanics;
- worker routing/assignment/loading;
- exact prerequisite chains;
- exact cash/order sequencing;
- minimum-safe watering for crops the strategy maintains;
- feeding existing animals and preventing mechanically avoidable escape;
- mechanically implied seed/feed/fertilizer/animal purchases;
- passive diagnostics and reproducible execution.

### Learned manager/RL owns

- crop and animal composition;
- expansion/contraction;
- land and labor capacity at the strategic level;
- deciding whether an asset should continue to be maintained;
- liquidity, cash reserve, recovery, and deleveraging;
- market/product strategy;
- opponent- and shop-dependent adaptation.

The default debugging question is:

> Did the executor fail to execute a feasible strategy, or did the manager choose an unsustainable strategy?

Only the first normally justifies an executor change.

## Deferred Until Evidence Supports Them

- learned tactical/middle model;
- raw primitive movement RL;
- corner/serpentine/Top-1 layout imitation sweeps;
- strategic crop-abandonment heuristics in deterministic code;
- broad executor hyperparameter searches;
- neural pathfinding;
- arbitrary maintenance reward bonuses;
- naive spot mark-to-market reward shaping;
- large architecture sweeps before the manager contract is audited;
- sophisticated league/PFSP machinery before fixed-mixture self-play works;
- large 64+64 executor panels for minor changes.

## Infrastructure Already Available

The following are no longer blockers:

- pinned official 1.32.7 differential oracle and full-episode parity evidence;
- high-throughput fast engine;
- promoted BC-E checkpoint and JAX E port;
- `standard_mixed` opening;
- executor V0.6 survival diagnostics;
- 3/5/7-day fixed-plan executor harness on the active V0.7 branch;
- Stage-A self-play rollout/trajectory infrastructure;
- PPO core/adapter/checkpoint/diagnostic plumbing;
- validated custom debug replay viewer on issue #11.

The current bottleneck is therefore **policy/executor boundary quality**, not missing infrastructure.

## Session Workflow

At the end of substantial work:

1. keep `CURRENT_STATE.md` compact and authoritative;
2. add completed milestones/failures to `HISTORY.md`;
3. revise this roadmap when priorities move;
4. add durable architecture/evaluation choices to `DECISIONS.md`;
5. update `MECHANICS.md` only when source/behavioral mechanics change;
6. update RL design notes when the manager/action/reward contract changes;
7. record exact commands/provenance before expensive runs.
