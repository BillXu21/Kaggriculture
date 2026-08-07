# Kaggriculture Plans

Last updated: 2026-08-06

## Strategic Objective

Build a robust Kaggriculture agent that outperforms copied deterministic public schedules by combining exact execution, closed-loop recovery, opponent-aware planning, and market strategy.

The project should only add learning where it beats a strong deterministic or optimization-based baseline under a frozen competitive evaluation suite.

## Phase 0 — Mechanics, Rules, and Provenance

### Goals

- identify the exact live engine version and source;
- distinguish source-confirmed behavior from documentation claims and discussion speculation;
- archive important public notebooks, submissions, and reports;
- record hashes and provenance for everything used;
- track engine changes and likely behavioral impact.

### Deliverables

- versioned engine manifest;
- mechanics ledger with confidence labels;
- public-baseline catalog;
- competition/rules snapshot;
- engine-drift regression checklist.

### Exit Criteria

- a fresh environment can reproduce the same core mechanics tests;
- every public baseline used in evaluation has an immutable hash and origin;
- unresolved mechanics are explicitly listed rather than silently assumed.

## Phase 1 — Evaluation and Baseline Reproduction

### Goals

- reproduce at least one strong public deterministic agent;
- build a deterministic local tournament harness;
- evaluate fixed seeds with both seat assignments;
- save replays and machine-readable metrics;
- establish a frozen opponent pool.

### Required Metrics

- wins, losses, ties, and win rate;
- final bank for both players;
- score margin;
- seat-specific performance;
- runtime and timeout status;
- invalid or ineffective actions where detectable;
- shed overflow;
- crop loss and animal escape;
- stranded terminal value;
- production and sales by product;
- market prices and inventory over time when relevant.

### Exit Criteria

- repeated runs with the same inputs are identical;
- seat-swapped paired results are available;
- baseline reproduction is close enough to its expected behavior to use as a reference.

## Phase 2 — Deterministic Executor and Closed-Loop Repair

### Goals

- represent a coherent farm route as structured intent rather than an opaque action list;
- validate expected state before executing each action;
- repair deviations without corrupting the rest of the route;
- preserve profitable schedules under weeds, partial market fills, cash differences, and engine changes.

### Candidate Components

- state normalizer;
- route schedule;
- precondition checks;
- worker assignment and path repair;
- crop and animal lifecycle tracker;
- shed-capacity guard;
- cash-flow guard;
- end-of-game liquidation controller.

### Exit Criteria

- repair controller beats blind replay under perturbation tests;
- no meaningful regression against the copied route in clean deterministic episodes;
- failures are logged with explicit reason codes.

## Phase 3 — Public Strategy Catalog and Expert Selection

### Goals

- identify major public strategy families;
- describe their land, labor, crop, livestock, and market profiles;
- implement or archive representative experts;
- choose among coherent experts using public episode state.

### Candidate Experts

- recurring-crop-heavy;
- livestock-heavy;
- mixed industrial public baseline;
- low-capital conservative route;
- collision-resistant product mix;
- terminal-liquidation-focused route;
- recovery route for early cash or planting failures.

### Exit Criteria

- expert selection improves frozen-pool win rate over the single best static expert;
- selection remains deterministic and explainable before considering learned selectors.

## Phase 4 — Opponent Modeling

### Goals

- infer strategy family from visible farm development;
- estimate future product supply and likely sale windows;
- predict expansion, livestock, and labor commitments;
- condition route selection and market policy on opponent behavior.

### Initial Approach

Start with deterministic features and templates:

- unlocked quadrants by day;
- worker count and hiring pattern;
- crop counts and maturity schedule;
- animal and structure counts;
- visible bank trajectory;
- characteristic movement or layout patterns;
- historical market changes consistent with likely sales.

Only use a learned classifier if a rule-based fingerprint is insufficient.

## Phase 5 — Market Strategy

### Goals

- model price impact and order sequencing;
- optimize hold-versus-sell decisions;
- exploit town-demand timing;
- avoid predictable market collisions;
- test whether deliberate interference can improve match win rate rather than merely reduce both scores.

### Candidate Methods

- exact short-horizon simulation;
- dynamic programming over inventory and cash;
- scenario search over opponent sale schedules;
- robust optimization against a strategy distribution;
- game-theoretic mixture selection.

### Guardrail

Market actions should be judged by paired win rate and score margin against competitive opponents, not by isolated single-player profit.

## Phase 6 — Macro Planning and Optimization

### Goals

Optimize high-level decisions while retaining deterministic low-level execution.

Candidate decision variables:

- expansion timing;
- daily worker count;
- crop allocation;
- structure placement;
- animal mix;
- fertilizer allocation;
- inventory targets;
- sale windows;
- expert transitions.

Candidate methods:

- beam search;
- mixed-integer or constraint programming;
- evolutionary search;
- Bayesian optimization;
- offline policy selection;
- self-play over macro-actions.

## Phase 7 — Learning, Only If Justified

Potential learning targets:

- opponent strategy classification;
- value estimation for macro plans;
- expert selection;
- market timing;
- compact residual policy over a deterministic controller.

Primitive-action end-to-end reinforcement learning is deferred unless evidence shows that structured planning has reached a clear ceiling.

## Phase 8 — Submission Hardening

- exact Kaggle runtime smoke;
- no-network verification;
- deterministic packaging;
- dependency minimization;
- timeout and memory tests;
- fallback behavior for malformed or unexpected state;
- submission provenance and hash recording;
- active-submission and leaderboard tracking.

## Deferred Ideas

- full primitive-action PPO or other deep RL;
- MCTS over all 720 turns;
- neural pathfinding;
- large language model policy inside the episode;
- expensive world-model training before competitive baseline parity;
- broad architecture abstractions before the first harness works.

## Rejected or Currently Unsupported Assumptions

- High single-player bank automatically means high competitive strength.
- The current public leaderboard meta will remain stable.
- Discussion descriptions are authoritative over source behavior.
- A copied action list is robust enough for the final competition.
- More model complexity is useful without a stronger evaluation protocol.

## Session Workflow

At the end of every substantial session:

1. update `CURRENT_STATE.md` with only active, current facts;
2. append results and failures to `HISTORY.md`;
3. revise this roadmap if priorities changed;
4. add durable choices to `DECISIONS.md`;
5. update `MECHANICS.md` when source or experiments change mechanical understanding.
