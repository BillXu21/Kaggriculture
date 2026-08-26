# Post-V0.7 Generalization Plan — 2026-08-26

Status: active research plan, not a frozen architecture decision.

## Why this note exists

The final 24-game real-BC-E / V0.7 PASS panel did not validate the lower tail. Aggregate results:

- games: 24
- mean bank: 28,587.4
- median bank: 25,321.5
- min: 2
- max: 65,959
- <1k: 3
- <10k: 3

Old V0.6 approximate reference:

- mean about 30,110
- median about 24,358
- min 8,062
- <1k = 0
- <10k = 2

The middle of the V0.7 distribution is fine, but the catastrophic tail is materially worse. The key research question is whether those crashes come from:

1. BC-E requesting unsustainable plans;
2. the deterministic executor failing a mechanically feasible plan; or
3. a policy-executor mismatch where BC-E operates near the execution frontier and small heuristic inefficiencies push a marginal plan into a death spiral.

Current prior is that (3) is plausible, but evidence must come first.

## Phase A — Visual replay forensics

Inspect a deliberately distribution-spanning set:

- seed 7 / seat 1 — bank 2, asymmetric catastrophe;
- seed 1019 / seat 0 — bank 76;
- seed 1019 / seat 1 — bank 55;
- seed 7 / seat 0 — bank 18,319, same-seed control;
- seed 2026 / seat 1 — bank 65,959, strong success;
- optional seed 1013 / seat 1 — bank 60,983, or a median trajectory.

For each replay, identify the **first visibly bad day** and one or two days of lead-up. Do not spend time interpreting late-game wreckage after the farm is already irrecoverable.

Look for:

- sudden land/crop/animal expansion;
- workload or geometry becoming much more complex;
- excessive travel, task stealing/duplication, or idle workers during critical work;
- must-water, harvest, or feed misses;
- shed congestion, overflow, or unplaceable animals;
- market ordering / cash sequencing failures;
- a discrete event that changes a recoverable farm into an irreversible collapse.

Use the stock Kaggle viewer for holistic pattern recognition. Use the custom passive trace only for causal manager/task/assignment/debt data.

## Phase B — Targeted log and capacity diagnosis

After the viewer identifies the first bad window, inspect only that bounded range in logs.

Required outputs per failure:

- first bad day and first irreversible event;
- BC requested DailyPlan immediately before it;
- requested vs projected/feasible plan;
- workers/hires and available worker-actions;
- conservative lower bound on mandatory work;
- end-of-day debt by task class;
- missed must-water / harvest / feed deadlines;
- expansion-suppression events;
- cash start/end;
- shed occupancy/overflow;
- skipped buys and reasons;
- animal purchases vs remaining place/store capacity;
- classification: manager, executor, or interaction.

A useful causal A/B is to replay the **same captured BC-E DailyPlan tape** through V0.6 and V0.7:

- V0.6 survives, V0.7 dies -> prioritize an executor regression;
- both die similarly -> primarily manager/executor compatibility;
- both are marginal but fail differently -> policy-executor interaction / execution-frontier problem.

### Mechanical-fix bar

Patch only obvious, reproducible defects such as:

- feasible critical work missed because of routing/scheduling defects;
- impossible animal purchase/storage behavior;
- wrong shed/cash/order arithmetic;
- task duplication/stealing/idle behavior that directly causes deadlines to fail;
- exact mechanical-priority mistakes.

Do not encode strategic governors such as "too many cows", "delay land expansion", "grow less X", or new arbitrary cash/debt vetoes into the executor.

Allow one bounded correction pass, then freeze/accept the executor and move to learning.

## Phase C — Throughput / 96-core readiness

Before serious RL, measure where rollout time actually goes:

1. engine stepping;
2. executor `_act`;
3. canonical-state conversion / feature encoding;
4. orchestration, IPC, and trajectory bookkeeping;
5. batched JAX policy inference.

Benchmark persistent concurrency roughly at:

`1 -> 4 -> 12 -> 24 -> 48 -> 96`

Preferred topology unless measurement disproves it:

**many independent single-threaded environment workers -> grouped day-boundary requests -> large policy batches -> plans returned to persistent workers.**

Do not perform per-environment accelerator calls.

### Rust executor option

Port only if profiling shows executor time is material. Prefer one Rust executor core with configuration/rule selection capable of reproducing frozen V0.6 and V0.7, rather than maintaining two separate rewrites.

Required before production use:

- primitive-action parity on recorded states/traces;
- same terminal outcomes on fixed deterministic games;
- measured throughput benefit large enough to justify maintenance cost.

## Phase D — RL initialization experiments

Working hypothesis: BC-E may contain strong elite-player state understanding but also highly confident strategic habits that assume more precise execution than the current heuristic can provide. Closed-loop RL may need to learn strategy around the executor's actual production frontier.

Compare small equal-compute runs before scaling:

### A. Full BC-E initialization

Pros: maximum inherited game knowledge and likely fastest route to competent play.

Risk: PPO may spend many samples unlearning elite habits that are fragile under this executor.

### B. BC-E trunk + reset action heads

Pros: preserve learned state representation while forcing strategy outputs to be relearned around the executor. This is the highest-priority middle-ground experiment.

### C. Scratch daily manager

Pros: cleanest executor-aware policy with no inherited elite assumptions.

Risk: worst sample efficiency unless curriculum is strong.

## Opening curriculum

Keep an optimized deterministic opening in every initial RL experiment. Consider extending it beyond the current d0-d3 handoff so the learned manager starts from a productive farm and a shorter effective horizon.

Benefits:

- fewer than the current 26 learned daily decisions;
- less early random bankruptcy;
- more samples spent on meaningful economic decisions;
- avoids inheriting late-game BC habits while retaining a competent bootstrap.

If scratch/near-scratch succeeds late-game, progressively hand earlier days to the learned policy only if useful.

## Reward plan — terminal and simple first

Prefer one terminal episode reward over dense daily economic shaping.

Initial candidate ingredients:

- normalized final cash term;
- explicit bankruptcy penalty using the engine's actual bankruptcy/loss condition;
- smaller terminal competitive outcome term early in training.

Principles:

- final cash is simple and hard to game relative to hand-built daily asset valuations;
- bankruptcy needs a separate negative signal so two repeatedly bankrupting agents cannot define "good play" only by relative wins;
- early training should prioritize make money + survive;
- as play becomes strong, reduce cash shaping and move toward outcome-dominated or outcome-only reward because final cash becomes less aligned with high-level competitive play.

Do not add a dense "economic value" reward by default; valuing crops, animals, land, inventory, future production, and price impact robustly is difficult and invites reward exploitation.

Do not add an infeasible-plan penalty initially. If later needed, it must be based on a conservative mechanically provable lower bound on required work vs available work, not executor EOD debt, so an executor bug does not become a manager penalty.

## First RL sequence after executor acceptance

1. establish real BC-E baseline through the exact RL rollout infrastructure;
2. run the existing real-checkpoint PPO plumbing smoke;
3. run tiny equal-compute initialization comparisons;
4. keep prior-day debt suppression ON initially and log its firing rate;
5. evaluate bank distribution, collapse rate, survival, requested expansion, suppression rate, entropy/KL/clip/value diagnostics;
6. scale only after throughput behavior is measured.

A stronger learned manager should reduce its dependence on the prior-debt expansion shield rather than merely improve final bank while constantly triggering it.

## Scope limits

Still deferred:

- R5/R6/R7 and broad executor heuristic search;
- learned tactical/middle layer;
- primitive-action movement RL;
- new arbitrary debt/cash strategic vetoes;
- sophisticated league/PFSP work before basic self-play works;
- Rust port without profile evidence;
- TPU debugging that does not block the actual selected training path.