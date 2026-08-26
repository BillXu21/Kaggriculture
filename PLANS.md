# Kaggriculture Plans

Last updated: 2026-08-26

## Strategic Objective

Build a competitive Kaggriculture agent where **reinforcement learning owns meaningful farm/market strategy** and deterministic code owns exact mechanics: legality, pathfinding, worker execution, prerequisites, bookkeeping, minimum-safe maintenance, and prevention of mechanically avoidable asset loss.

Working architecture remains:

`optimized deterministic opening -> learned daily manager -> deterministic executor`

The opening is currently `standard_mixed d0-d3`. Extending it later is a valid RL curriculum experiment. Learned tactical/middle control and primitive-action RL remain deferred.

## Current Phase — Explain the Catastrophic Tail

The final 24-game real-BC-E / V0.7 PASS panel produced mean `28,587.4`, median `25,321.5`, min `2`, max `65,959`, `<1k = 3`, `<10k = 3`.

The old V0.6 reference was approximately mean `30,110`, median `24,358`, min `8,062`, `<1k = 0`, `<10k = 2`.

The middle of the V0.7 distribution is acceptable, but the catastrophic tail is not. Do **not** call V0.7 generalization validated, restart broad heuristic search, or launch large self-play yet.

The immediate question is whether the failures are:

1. BC-E requesting an unsustainable strategy;
2. the executor failing to execute a mechanically feasible strategy; or
3. a policy-executor interaction where a marginal BC plan becomes catastrophic because of heuristic inefficiency.

## Immediate Step 1 — Visual Replay Forensics

Inspect:

- seed 7 / seat 1 — bank 2;
- seed 1019 / seat 0 — bank 76;
- seed 1019 / seat 1 — bank 55;
- seed 7 / seat 0 — bank 18,319 same-seed control;
- seed 2026 / seat 1 — bank 65,959 strong success;
- optionally seed 1013 / seat 1 or a median trajectory.

For each, identify the **first visibly bad day** and 1–2 days of lead-up. Use the stock Kaggle viewer for global pattern recognition and the passive custom trace only for causal details.

Look for aggressive expansion, workload/geometry jumps, excessive travel or task stealing, missed must-water/harvest/feed deadlines, shed congestion/overflow, impossible animal handling, market/cash sequencing failure, or a discrete event that starts the death spiral.

## Immediate Step 2 — Targeted Log / Capacity Diagnosis

After the visual pass, inspect only the bounded failure window. Record:

- BC requested plan;
- requested vs feasible plan;
- workers and available worker-actions;
- conservative lower bound on mandatory work;
- EOD debt by class;
- critical deadline misses;
- expansion-suppression events;
- cash start/end;
- shed occupancy/overflow;
- skipped buys and reasons;
- animal purchases vs placement/storage capacity;
- classification: manager, executor, or interaction.

When practical, replay the exact same captured BC-E DailyPlan tape through V0.6 and V0.7. V0.6-survives/V0.7-dies is strong executor-regression evidence; both-die similarly is stronger manager/executor-compatibility evidence.

Allow only one bounded obvious-mechanical-defect pass. Do not encode strategic governors such as "too many cows", delayed expansion, crop preference, or new arbitrary cash/debt vetoes into the executor.

## Step 3 — Throughput / 96-Core Readiness

Before serious RL, profile wall time separately for:

1. engine stepping;
2. executor `_act`;
3. canonical-state conversion / encoding;
4. IPC/orchestration/trajectory work;
5. batched JAX inference.

Benchmark persistent concurrency roughly at `1 -> 4 -> 12 -> 24 -> 48 -> 96`.

Prefer many independent single-threaded games feeding grouped day-boundary observations into large policy batches unless measurement proves otherwise. Never do per-environment accelerator calls.

Rust executor work is conditional on profiling. If Python executor time is material, prefer one parity-tested Rust core that can reproduce frozen V0.6 and V0.7 behavior instead of two separate rewrites.

## Step 4 — Small RL Initialization Experiments

Working hypothesis: BC-E may imitate elite strategies that operate close to the execution frontier. Closed-loop RL may need to learn strategy around the executor's actual capabilities.

Compare small equal-compute candidates before scaling:

1. **full BC-E init** — maximum inherited knowledge, but maximum inherited elite habits;
2. **BC-E trunk + reset action heads** — preserve state representation while relearning strategy around this executor;
3. **scratch daily manager** — clean executor-aware policy, but likely worse sample efficiency.

The reset-head middle ground is a high-priority experiment.

## Opening Curriculum

Keep an optimized deterministic opening for initial RL. Consider extending the opening beyond d3 so scratch/near-scratch learning begins from a productive farm with fewer than the current 26 learned daily decisions.

This can reduce trivial early bankruptcy, shorten credit assignment, and preserve sample efficiency without inheriting BC-E's late-game strategic habits.

## Reward Plan — Terminal and Simple First

Prefer one terminal episode reward rather than dense daily economic shaping.

Initial candidate:

`normalized final cash + explicit bankruptcy penalty + smaller terminal outcome term`

Early training should prioritize **make money + survive**. Later, as policies become strong, anneal toward outcome-dominated or outcome-only reward because final cash becomes less aligned with top-level competitive play.

Do not add dense hand-built economic-value rewards by default. Do not add plan-infeasibility punishment initially; if later needed, it must use a conservative mechanically provable workload lower bound rather than executor EOD debt.

## First RL Sequence After Executor Acceptance

1. establish real BC-E baseline through the exact RL rollout infrastructure;
2. run the real-checkpoint PPO plumbing smoke;
3. run tiny equal-compute initialization comparisons;
4. keep prior-day debt suppression ON initially and log its firing rate;
5. compare bank distribution, collapse rate, survival, requested expansion, suppression rate, entropy/KL/clip/value diagnostics;
6. scale only after rollout throughput is measured.

A stronger learned manager should reduce dependence on the prior-debt expansion shield rather than merely improve bank while constantly triggering it.

## Executor / Manager Boundary

Executor owns exact mechanics and feasible execution. Manager/RL owns strategic expansion, composition, liquidity, capacity, and market choices.

Default diagnostic:

> Did the executor fail to execute a feasible strategy, did the manager choose an unsustainable strategy, or did a near-feasible strategy cross the failure boundary because of executor inefficiency?

Only the first normally justifies executor changes.

## Merge / Documentation Gate

Before merging `executor-v07-fixed-plan`:

- preserve both diverged documentation histories;
- resolve the duplicate durable-decision ID for the branch's R4/prior-debt decision without overwriting main D-031..D-036;
- record the forensic conclusion and any accepted mechanical patch;
- keep large traces/checkpoints local/ignored.

## Explicitly Deferred

- R5/R6/R7 or broad executor heuristic search;
- deliberate crop-digging / elite geometry imitation;
- new arbitrary strategic veto thresholds;
- learned tactical/middle policy;
- primitive-action RL;
- custom viewer redesign;
- sophisticated league/PFSP work before basic self-play works;
- Rust implementation without profile evidence;
- TPU debugging that does not block the selected training path.

## Continuity

Detailed post-panel plan: `research/POST_V07_GENERALIZATION_PLAN_2026-08-26.md`.

Prior compaction: `research/PROJECT_CHECKPOINT_2026-08-25_V07_FREEZE.md`.