# Kaggriculture Plans

Last updated: 2026-08-25

## Strategic Objective

Build a competitive Kaggriculture agent where **reinforcement learning owns meaningful farm/market strategy** and deterministic code owns exact mechanics: legality, pathfinding, worker execution, prerequisites, bookkeeping, minimum-safe maintenance, and prevention of mechanically avoidable asset loss.

Near-term architecture stays:

`standard_mixed d0-d3 -> learned daily manager -> deterministic executor`

Do not add a learned tactical/middle controller or primitive-action RL before real self-play demonstrates the need.

## Current Phase — Close Executor V0.7

The executor heuristic-search phase is over. Branch `executor-v07-fixed-plan` is pushed and clean at `b1b9b306b48a6ae3fcb2464109088e7ecea91b7c`; frozen behavior is `a7c826d`.

Final decisions:

- R4 watering reservation: **rejected/reverted** after real BC-E regression.
- Survival WHEAT shed-room clamp: **accepted**.
- Broad prior-day work-debt expansion suppression: **temporarily retained ON as explicit architectural debt**, because real BC-E collapses catastrophically with it OFF.
- Passive issue #11 debug tracing/viewer: integrated into the branch.
- No R5/R6/R7 or alternative debt-threshold search.

Real BC-E PASS panel, seeds `17,42,2026`, both seats:

| mode | mean bank | median | min | starvation loss units |
| --- | ---: | ---: | ---: | ---: |
| debt shield ON | 34,100 | 24,966.5 | 14,961 | 0 |
| debt shield OFF | 98.8 | 31.5 | 0 | 38 |

Interpretation: the current BC policy depends heavily on the shield to avoid over-expansion. Keep it ON for the first RL experiments and log suppression frequency. The long-term goal is for RL to make it unnecessary, not to hand-design a more strategic executor governor.

## Immediate Gate — One Final 24-Game ON Panel

Run the real BC-E checkpoint with V0.7, fixed PASS opponent, both seats, debt shield ON, seeds:

`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`

Use `tools.run_executor_v07_panel` from `executor-v07-fixed-plan`, official backend if practical.

Only answer:

1. are there new catastrophic `<1k` or `<10k` cases?
2. does the shed-room fix preserve the known-collapse recoveries?
3. does starvation stay near zero outside the three development seeds?
4. how does the bank distribution compare with the old V0.6 24-game reference (mean about `30,110`, median about `24,358`, min `8,062`, `<1k=0`, `<10k=2`)?

Do not run another OFF panel. Do not respond to a mediocre but non-catastrophic result with another heuristic search.

## Merge / Freeze Gate

If the final ON panel is credible:

1. reconcile `executor-v07-fixed-plan` with main; branch and main have diverged documentation histories, so preserve both the V0.7 implementation/evidence and the newer main continuity docs intentionally;
2. consider Executor V0.7 mechanically frozen and issue #7 complete enough for RL;
3. keep generated traces/checkpoints/artifacts local/ignored.

## Viewer Policy

Use the custom canonical debug trace as a **diagnostic sidecar**, not as the main visual experience. The stock Kaggle viewer is easier for holistic farm-pattern inspection and already has polished imagery.

Only use custom traces when manager/task/assignment/feed/water/debt metadata is needed to isolate a mechanical bug. Defer frontend redesign; a future version should augment or visually follow the existing Kaggle viewer rather than rebuilding the game presentation from scratch.

## Manager Contract — No Immediate Redesign

Current audit result:

- crop contraction is representable end-to-end because crop targets are absolute counts and reconciliation can release excess;
- animal targets are intentionally non-destructive (`max(current, requested)`), so lower desired count means stop adding/replacing rather than killing/starving animals;
- deliberate crop digging / price-crash abandonment is advanced behavior and does not need to be learned now, especially if demonstrations are sparse.

Measure what BC-E and PPO actually output before redesigning the manager action space.

## First Real RL Sequence

### 1. Real closed-loop BC-E baseline

Run the frozen BC-E checkpoint through the exact rollout infrastructure intended for RL with frozen V0.7.

Record complete-game distributions and strategy diagnostics:

- banks/margins/W-L-T;
- collapse rate;
- animal/crop losses;
- farm-size/labor/cash trajectories;
- manager target trajectories;
- expansion-suppression frequency;
- executor survival/maintenance failures.

This becomes the pre-PPO baseline. Teacher-forced validation is not the quality gate.

### 2. PPO plumbing smoke from BC-E

Use the existing `rl_manager` PPO path with the real checkpoint:

- complete rollout;
- stored-action logprob recomputation;
- GAE;
- one small update;
- checkpoint/resume roundtrip;
- deterministic pre/post eval sanity;
- frozen BC-E snapshot must remain immutable.

This is still plumbing validation, not a policy-quality claim.

### 3. Small development run vs frozen BC-E

Keep prior-debt suppression ON initially. Train/evaluate on a small fixed set before any broad run.

Important diagnostic: **suppression rate should fall as policy quality improves.** If PPO learns capacity management, it should stop asking for expansion that the executor vetoes. Bank improvement without declining dependence on the shield is weaker evidence.

### 4. Diagnose before redesign

If learning stalls, classify the bottleneck before adding complexity:

- action space cannot express needed strategy;
- observation misses strategic state;
- daily decision frequency is too coarse;
- terminal credit is insufficient;
- executor still has a reproducible mechanical defect;
- tactical task choice truly requires a learned middle layer.

Only then consider denser reward shaping, more frequent manager actions, or a tactical model.

## JAX / TPU

TPU throughput work is paused, not promoted to a blocker.

Known state:

- synthetic/random PyTorch <-> JAX parity tests are green;
- the newly activated real-checkpoint parity test has a test bug: the JAX side loads trained weights while the PyTorch reference is freshly initialized and never loads `model_state_dict`;
- Kaggle TPU subprocesses repeatedly failed TPU backend initialization;
- no real TPU throughput result exists.

Fix the real-checkpoint parity test and TPU ownership/runtime issue later, when the actual RL path needs TPU execution. Do not spend current time debugging the notebook environment in isolation.

## Executor / Manager Boundary

Deterministic executor owns exact mechanics, legal execution, routing/assignment, prerequisites, cash/order sequencing, feeding existing animals, minimum-safe watering for maintained crops, mechanically implied resource acquisition, and passive diagnostics.

Learned manager/RL owns expansion pace, crop/animal composition, labor/farm capacity at strategic level, liquidity/recovery, whether an asset remains worth maintaining/replacing, market/product strategy, and opponent/shop adaptation.

Default question:

> Did the executor fail to execute a feasible strategy, or did the manager choose an unsustainable strategy?

Only the first normally justifies executor changes.

## Explicitly Deferred

- deliberate crop digging / sophisticated crop abandonment;
- watering/task-assignment heuristic variants;
- corner/serpentine/Top-1 geometry imitation;
- new debt/cash veto thresholds;
- learned tactical/middle policy;
- raw primitive-action RL;
- broad executor parameter sweeps;
- custom viewer redesign;
- TPU debugging that is not blocking the real training path;
- sophisticated league/PFSP work before fixed candidate-vs-frozen-E self-play works.

## Continuity

Authoritative long-session compaction: `research/PROJECT_CHECKPOINT_2026-08-25_V07_FREEZE.md`.

At the end of the next substantial run, update `CURRENT_STATE.md`, add a completed milestone to `HISTORY.md`, and record any genuinely durable new architecture/evaluation decision in `DECISIONS.md`.
