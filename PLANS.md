# Kaggriculture Plans

Last updated: 2026-08-27

## Strategic Objective

Build a competitive Kaggriculture agent in which **reinforcement learning owns meaningful farm and market decisions**, while deterministic infrastructure handles exact mechanics that are poor uses of learning capacity: legality, pathfinding, task execution, state bookkeeping, and prevention of meaningless invalid-action cascades.

The immediate phase is now **closed-loop BC/executor validation plus implementation of reusable RL infrastructure**. The host has stated that 1.32.7 should be the last balance change except game-breaking bugs, so the project can prepare to freeze the engine contract while still checking for bug-fix releases.

Detailed RL design lives in [`research/RL_DESIGN.md`](research/RL_DESIGN.md).

A temporary high local coding-agent-usage window is being used for large, mechanically testable infrastructure packets that are likely to be needed regardless of later strategy choices. The bounded sprint plan is in [`research/CODING_AGENT_SPRINT.md`](research/CODING_AGENT_SPRINT.md). This does **not** supersede the staged learning gates: infrastructure may be prepared early, but PPO/self-play complexity is not activated before the simpler stationary problems work.

Issue #7 / Executor V0.7 is now frozen at `a7c826d`; the exact current decision, bounded PASS panel, R4 rejection, and viewer #11 closure are in [`research/EXECUTOR_V07_FINAL.md`](research/EXECUTOR_V07_FINAL.md). The real BC-E validation input was externally supplied read-only from `C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt` (variant E, epoch 27) and remains uncommitted; neither issue #7 executor selection nor checkpoint availability remains a blocker.

Issue #13 Stage 1 and the Stage 5 post-Stage-4 archive compatibility check (2026-08-27) are accepted: the exact deterministic current archive at source revision `11ecead2d5efe8bf87fc0da533c739e344d7eaa6` was freshly extracted and raw-loaded with `get_last_callable` under official `kaggle-environments==1.32.7`; repository-root source fallback was absent, strict mode was enabled, the full 720-entry status history had zero anomalies, and candidate seat 1 versus PASS reproduced bank `47,290` with pinned fingerprint `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc`. The prior pre-behavior identity remains recorded as bank `54,439` with fingerprint `516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40` and archive SHA `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf`; no Stage 3 retention, Stage 6/7, PPO, or strategy work is authorized by this compatibility update.
- Stage 7 issue #12C is documented as a no-patch defer: optional spare watering remains default-off, and its unbounded-distance concern requires the fixed 24-game false-vs-true A/B before any enabling or clamp decision. See `research/EXECUTOR_V07_STAGE7_ISSUE12C_OPTIONAL_WATER_AUDIT.md`; do not run that panel as part of documentation work.
- Stage 6/7 issue #12 idle cleanup is now a measured PASS-only layer, fully documented in `research/EXECUTOR_V07_IDLE_CLEANUP_PASS_ONLY.md`. The normal foreman runs first; optional work is recomputed per turn and can replace only literal normal PASS actions, with no persistence or normal accounting contamination. Keep `optional_spare_watering` (WATER-only) distinct from `optional_idle_cleanup` (WEED+WATER). The official A/B/C panel is mean-positive and majority-winning for both arms but has 8 losses each and materially worse tails; WATER-only median delta is negative while WEED+WATER median delta is positive. Leave cleanup OFF by default and require independent review before any promotion; the next step is not PPO.

Issue #9 Stage B2 remains implemented and locally validated as plumbing, but is not the current mutation target until the issue-#13 archive invariant is accepted.

Issue #15 executor hot-path profiling/optimization is complete on isolated
branch `throughput/15-agent-hotpath`; see
`research/EXECUTOR_HOTPATH_ISSUE15.md`. Canonical snapshot reuse is default and
the copy/diagnostic reductions are explicit training knobs. Further executor
rewrites remain deferred until target-host integration evidence.

Issue #16 batched fast self-play is implemented as an additive native owner and
opt-in runner path. Keep scalar mode as the correctness reference; future
worker scheduling may consume `oracle.batched_backend.BatchedEngineBackend`
without coupling to PPO or executor strategy.

Issue #17 parallel rollout workers are integrated with the #15 executor knobs
and #16 batched backend. Keep one JAX/libtpu-owning parent, spawn CPU workers,
and use the scalar/default path as the correctness reference for any future
target-host measurement.

The issue #17 batching extension is now implemented additively: retain
`policy_day`/variable-size defaults, and benchmark opt-in mixed-day
`policy` scope with fixed physical sizes through the RL manager CLI. Validate
deterministic signatures and occupancy before making any default change.

## Phase 0 — Freeze 1.32.7, Rules, RNG, and Provenance

### Current engine events

**1.32.6 / PR #1394**

- town-center demand became flat 1× once/day;
- shops are sampled with replacement;
- duplicate shop instances consume independently;
- maximum shop instances = 8.

**1.32.7 / PR #1399**

- adds a nonlinear `hinge` scarcity curve;
- carrot, tomato, and egg use `hinge` on the scarcity side;
- carrot scarcity target also rises from 0.20 to 1.00;
- the intent is to make carrots, tomatoes, and goose/egg production conditionally valuable under favorable randomized shop demand.

### Goals

- lock exact 1.32.7 source/package/spec hashes;
- distinguish source-confirmed behavior from live-server rollout assumptions;
- identify all stochastic mechanics and reveal times;
- verify 1.32.7 price curves with behavioral tests;
- archive important public notebooks/submissions with provenance;
- avoid training against stale pre-1.32.7 economics.

### Deliverables

- versioned engine manifest;
- mechanics ledger;
- exact market curve table and regression tests;
- randomness/reveal-time map;
- public-baseline catalog with engine compatibility notes;
- competition/rules snapshot.

### Exit Criteria

- fresh local environment reproduces the 1.32.7 source contract;
- exact source/spec hashes are recorded;
- hinge regression values pass;
- all strategically relevant RNG is identified;
- unresolved mechanics are explicit.

## Phase 1 — Evaluation, Public Baselines, and Trace Collection

### Goals

- reproduce/archive several strong public deterministic agents;
- build deterministic local tournament harness;
- evaluate fixed seeds in both seats;
- save replays and machine-readable metrics;
- establish a frozen opponent pool;
- collect full trajectories for BC/action-abstraction analysis;
- quantify the new 1.32.7 scarcity regimes before model training.

### Required Metrics

- W/L/T and paired win rate;
- final bank and margin;
- seat-specific results;
- runtime/timeout status;
- invalid/ineffective actions;
- shed overflow, crop loss, animal escape, stranded value;
- production and sales by product;
- market price/inventory trajectories;
- shop composition by day;
- scarcity-knee crossings and max price for carrot/tomato/egg;
- exact seed, agent hashes, and engine identity.

### Initial 1.32.7 Studies

1. **Scarcity distribution** — reproduce host-reported no-production opportunity frequencies (~50% tomato, 26% carrot, 22% egg).
2. **Crossing timing** — when does each product first cross its hinge knee?
3. **Pivot frontier** — how late can a crop/animal pivot still pay back?
4. **Opponent suppression** — how much does reactive production eliminate the opportunity?
5. **Public baseline staleness** — which strong routes fail to exploit or actively mishandle 1.32.7 regimes?
6. **Shop-regime variance** — how much outcome variance is explained by shop multiset/hinge events?

### Exit Criteria

- deterministic replay works;
- paired evaluation works;
- strong current-engine traces exist;
- 1.32.7 opportunity frequencies/timing are measured;
- we know whether adaptive pivoting is large enough to justify explicit model capacity.

## Phase 2 — RL Environment and Action Space

### Core design

Use **hierarchical intent-level RL**, not raw primitive navigation.

### Worker task families

- plant;
- water;
- harvest;
- fertilize;
- dig;
- build coop/pasture;
- place/feed/care animals;
- collect fertilizer;
- pickup/drop;
- purposeful movement;
- continue/wait.

Use generated mechanically feasible candidates and pointer/entity scoring.

### Market head

Dedicated autoregressive market actions:

- type;
- product;
- quantity;
- stop token;
- preserve order sequencing.

Compare exact quantity, buckets, parameterized integer distributions, and generated quantity candidates. Mechanically useful curve landmarks such as scarcity knees may be included as candidate quantities, but the generator must not decide whether exploiting them is good.

### Mechanical masks

Mask impossible actions. Do **not** mask merely bad-looking strategy or encode static product priorities.

### Decision frequency

Compare:

1. turn-level persistent tasks;
2. event-driven semi-MDP;
3. hybrid global/day-level + event-driven worker + frequent market decisions.

Current favorite: hybrid.

1.32.7 economic events such as shop unlocks and hinge crossings are natural replanning triggers.

### Exit Criteria

- exact action grammar documented;
- task conflicts/resolution documented;
- masks tested;
- no strategic policy hidden in candidate generation;
- acceptable throughput.

## Phase 3 — Observation and Model Design

### Required observation groups

- time/turns remaining;
- money, land, labor;
- own private inventory/seeds/carried items;
- both farm maps/entities;
- worker entities;
- crop/animal lifecycle features;
- shop **count vector/multiset**;
- market price/inventory/history;
- product curve parameters (`base`, `I0`, `T`, shape/targets);
- normalized scarcity and distance to each knee;
- local/marginal price sensitivity where useful;
- opponent visible production pipeline;
- optional recent history/recurrent state.

### Model hypothesis

Start entity-oriented:

- farm/tile/entity encoders;
- product/market entities;
- shared transformer trunk;
- optional recurrent memory;
- worker-task pointer head;
- market autoregressive head;
- optional global strategy head;
- value head.

### Exit Criteria

- schema versioned;
- normalization documented;
- actor-visible vs training-only information separated;
- inference cost/model-size estimate recorded.

## Phase 4 — Reward Design

### Competitive objective

Leading terminal reward:

- win `+1`;
- tie `0`;
- loss `-1`.

### Dense credit assignment

Investigate potential-based shaping rather than arbitrary maintenance bonuses.

Candidate potential inputs:

- bank;
- realistic liquidation value;
- time-realizable crop/animal output;
- asset productive lifetime;
- failure/overflow risk;
- opponent-equivalent value;
- learned/model-based continuation value.

### 1.32.7 guardrail

Do **not** use naive `inventory × current spot price` as potential value. Hinge prices can spike, but selling meaningful quantity changes the price and a production pivot may arrive too late. Shaping must account for price impact and remaining time or it may reward fake mark-to-market wealth.

### Auxiliary objectives

- future bank;
- future market price/inventory;
- time/probability to hinge crossing;
- near-term production;
- opponent sales/hidden inventory;
- pivot profitability;
- win probability.

### Exit Criteria

- exact reward equation;
- terminal/shaping invariants tested;
- no maintenance-loop or hinge-mark-to-market exploit;
- reward magnitude/variance measured.

## Phase 5 — Behavior Cloning Bootstrap

### Plan

1. collect fresh demonstrations under 1.32.7;
2. store observations, primitive actions, intent labels, and engine metadata;
3. clone precision-sensitive logistics strongly;
4. consider weaker/segmented imitation weights for high-level product/economic choices so stale public strategy does not dominate;
5. test state-conditioned generalization across unseen shops/scarcity regimes;
6. fine-tune with RL so the policy can depart from scripts.

### Exit Criteria

- cloned policy runs viable farms;
- mechanical/logistics performance is useful;
- model does not simply memorize turn-indexed product choices;
- unseen 1.32.7 regimes do not catastrophically break it.

## Phase 6 — PPO Robustness and Competitive Training

PPO remains the first algorithm candidate, conditional on throughput.

### Stage A — regime adaptation

Train against frozen public agents across varied 1.32.7 seeds, including ordinary and hinge-scarcity regimes.

Rare regimes may be oversampled early for learning, followed by training/evaluation on the natural distribution.

### Stage B — frozen competitive pool

Track performance by:

- opponent family;
- shop regime;
- hinge opportunity type;
- seat;
- catastrophic failure rate.

### Stage C — population/self-play

Mix champion, historical checkpoints, public baselines, and strategy-diverse policies.

### Stage D — targeted exploiters

Potential exploiters:

- aggressive scarcity-chaser;
- scarcity-suppressor/flooder;
- product-collision agent;
- unusual timing/liquidation policy.

Goal: teach game-theoretic adaptation rather than blind spike chasing.

## Phase 7 — Opponent Modeling and Memory

Goals:

- infer hidden opponent inventory and likely sales;
- forecast visible production supply;
- predict whether the opponent will suppress a scarcity opportunity;
- detect strategy families;
- adapt before market collisions.

Compare explicit history features, recurrent actor state, auxiliary hidden-inventory prediction, and dedicated opponent encoders.

## Phase 8 — Optional Search / Planning Hybrids

Later, combine learned policy/value with short-horizon exact simulation for:

- market order sequencing;
- production pivots after unusual shop draws;
- scarcity exploitation/suppression;
- daily resource allocation;
- terminal liquidation.

Do not start here before a functioning RL baseline.

## Phase 9 — Submission Hardening

- exact Kaggle runtime smoke;
- no-network verification;
- deterministic packaging;
- dependency minimization;
- timeout/memory tests;
- safe fallback behavior;
- provenance/hash recording;
- submission/leaderboard tracking.

## Immediate Planning Agenda

0. **Run the BC V1 ablation on Kaggle (issue #6 next gate).** Implementation
   is complete and locally validated (commits `2f48564`..`fc95752`; see
   `research/BC_V1_ABLATION_RUN.md`). Execute the exact runbook: train
   V0/J/E/JE in one identical matrix over the five-day corpus, strict
   `--validate-only` preflight, then the 40-game paired closed-loop panel
   (seeds 7/17/42/123/2026 × both seats, bank median-then-mean ranking).
   Record real results in `HISTORY.md`/`CURRENT_STATE.md` before any
   follow-up; teacher-forced/coherence metrics alone never promote a
   variant (D-026); no winner exists yet.
1. ~~finish the real engine-executed BC-manager + executor smoke and require non-empty per-day diagnostics/compliance~~ Done for the repo-local checkpoint plumbing path; continue to inspect compliance before score claims;
2. implement the common official-engine evaluation/match runner from `research/CODING_AGENT_SPRINT.md`;
3. implement the scalar exact fast engine and differential oracle against pinned 1.32.7 (differential oracle live; Stage-2b mechanic-cluster parity slices 1-4 done at zero divergence; full-episode legal-ish corpus DONE 2026-08-23: 8 complete 720-step seeds, zero divergence, D-022; independent stateful closed-loop A/B DONE for three fixed-plan seeds plus one repo-local checkpoint episode; throughput/benchmarks DONE 2026-08-23, D-025);
4. implement population/league infrastructure with deterministic paired scheduling and explicit promotion gates;
5. implement algorithm-neutral rollout storage, returns/GAE, and deterministic minibatching;
6. prove self-play orchestration with fake/frozen policies before attaching a learner;
7. only then add the PPO-specific value/log-prob/loss interface after the executor compliance gate;
8. freeze 1.32.7 source/spec/hash and keep upstream bug-fix watch;
9. map remaining RNG/reveal timing and maintain engine differential tests;
10. benchmark official and fast simulator throughput — DONE 2026-08-23 (`docs/benchmarks/ISSUE2_THROUGHPUT.md`, D-025): scalar dict API 4.7x vs official, native floor 341x, default-pool scaling ~2.9x at N>=512; observation-writer cost (84% of large-batch step time) is the single deferred optimization candidate for a distinct correction stage;
11. build frozen competent-opponent panels and strategy-family cross-play;
12. recheck upstream immediately before substantive training.

## Deferred Until Evidence Supports Them

- raw primitive-movement PPO from scratch;
- pure SAC over the full action space;
- MCTS over all 720 primitive turns;
- neural pathfinding;
- hard-coded fixed product ranking;
- large architecture sweeps before interface/reward contracts are stable;
- expensive training before engine lock and throughput measurements;
- sophisticated PFSP/league matchmaking before fixed-mixture self-play works;
- distributed rollout infrastructure beyond the local single-owner `spawn` topology before local throughput is profiled;
- approximate/vectorized custom simulation before scalar differential parity.

## Rejected or Unsupported Assumptions

- high single-player bank automatically means competitive strength;
- public leaderboard strategy remains optimal after 1.32.7;
- carrots/tomatoes/eggs are globally good because they can spike;
- carrots/tomatoes/eggs are globally bad because old public routes ignored them;
- discussion prose overrides engine source;
- RL must control every movement primitive to be real RL;
- naive mark-to-market value is safe under nonlinear price impact;
- host no-production scarcity percentages describe strategic self-play frequencies.

## Session Workflow

At the end of substantial work:

1. update `CURRENT_STATE.md`;
2. append to `HISTORY.md`;
3. revise this roadmap if priorities changed;
4. add durable choices to `DECISIONS.md`;
5. update `MECHANICS.md` for source/behavior changes;
6. update `research/RL_DESIGN.md` when the RL interface/reward hypothesis changes.
