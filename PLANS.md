# Kaggriculture Plans

Last updated: 2026-08-07

## Strategic Objective

Build a competitive Kaggriculture agent in which **reinforcement learning owns meaningful farm and market decisions**, while deterministic infrastructure handles mechanics that are known exactly and are poor uses of learning capacity: legality, pathfinding, task execution, state bookkeeping, and safety against meaningless invalid actions.

The immediate project phase is design, not implementation. Use the next week while Pokémon work finishes to settle the RL interface, reward, observation, curriculum, and evaluation contracts and to watch for additional engine changes.

Detailed RL design lives in [`research/RL_DESIGN.md`](research/RL_DESIGN.md).

## Phase 0 — Engine Stability, Rules, and Provenance

### Goals

- track current engine changes and package versions;
- distinguish source-confirmed behavior from server rollout assumptions;
- identify all stochastic mechanics and reveal times;
- archive important public notebooks, submissions, and reports;
- record hashes and provenance for everything used;
- avoid building large training infrastructure against a moving contract.

### Current engine event

Package 1.32.6 / upstream PR #1394 materially changes economics:

- town-center demand is flat 1× and once/day;
- former day-10/day-20 demand ramps are removed;
- shops are sampled with replacement;
- duplicate shops consume independently;
- total shop instances remain capped at 8.

### Deliverables

- versioned engine manifest;
- mechanics ledger with confidence labels;
- public-baseline catalog;
- competition/rules snapshot;
- engine-drift regression checklist;
- randomness map: source, draw timing, observation timing, and strategic effect.

### Exit Criteria

- fresh local environment reproduces the current source contract;
- exact engine/spec hashes are recorded;
- all stochastic mechanics relevant to training are identified;
- unresolved mechanics are explicit rather than silently assumed.

## Phase 1 — Evaluation, Public Baselines, and Trace Collection

### Goals

- reproduce/archive several strong public deterministic agents;
- build a deterministic local tournament harness;
- evaluate fixed seeds with both seat assignments;
- save replays and machine-readable metrics;
- establish a frozen opponent pool;
- collect full trajectories for behavior cloning and action-abstraction analysis.

### Required Metrics

- wins, losses, ties, and win rate;
- final bank for both players;
- score margin;
- seat-specific performance;
- runtime and timeout status;
- invalid/ineffective actions where detectable;
- shed overflow;
- crop loss and animal escape;
- stranded terminal value;
- production and sales by product;
- market prices/inventory over time;
- shop composition by day/final episode;
- seed and exact agent/engine identities.

### Initial research experiments

1. **Shop-regime variance:** same frozen agent over many seeds; stratify outcomes by shop multiset.
2. **Opponent market sensitivity:** vary opponent product mix/sale timing with our agent fixed.
3. **Public trace diversity:** measure whether public action lists remain mostly turn-indexed or respond materially to state.

### Exit Criteria

- repeated runs with identical inputs are identical;
- seat-swapped paired results are available;
- strong public trajectories are available under current engine behavior;
- stochastic shop effects are quantified enough to design training distributions.

## Phase 2 — RL Environment and Action-Space Design

This phase is intentionally before model training.

### Core design target

Use **hierarchical intent-level RL** rather than raw primitive navigation.

The policy should decide strategy and task intent; deterministic code compiles intent into primitive legal actions.

### Worker-task action design

Candidate task families:

- plant crop at target tile;
- water;
- harvest;
- fertilize;
- dig;
- build structure;
- place animal;
- feed/care;
- collect fertilizer;
- pickup/drop;
- purposeful movement/positioning;
- continue/wait.

Use generated feasible candidates and a pointer/entity-scoring head rather than a massive flat categorical space.

### Market action design

Design a dedicated autoregressive market head:

- order type;
- product/resource;
- quantity;
- end/stop token;
- up to engine market-order limit.

Compare exact quantity, quantity buckets, and generated quantity candidates.

### Mechanical masks

Mask impossible actions aggressively, including invalid targets, unavailable inventory, impossible quantities, and capacity/affordability failures.

Do not mask strategically poor but mechanically valid actions.

### Decision-frequency candidates

Compare:

1. turn-level policy with persistent tasks;
2. event-driven semi-MDP decisions;
3. hybrid global/day-level + event-driven worker + market decisions.

Current leading hypothesis: hybrid.

### Exit Criteria

- exact action schema documented;
- conflict resolution between multiple workers documented;
- action-mask contract tested;
- action abstraction retains meaningful policy control;
- no large amount of farm strategy is secretly encoded in candidate generation/executor;
- interface has acceptable simulator throughput.

## Phase 3 — Observation and Model Design

### Observation groups

- day/turn/time remaining;
- own/opponent money and land/labor state;
- own private shed/seeds/carried inventory;
- both farm maps as entities/tiles;
- worker entities;
- crop/animal lifecycle features;
- market price/inventory and recent deltas;
- town-shop **count vector/multiset** rather than binary flags;
- known demand/tick timing;
- opponent visible production pipeline;
- optional short history/recurrent state for hidden inventory inference.

### Model hypothesis

Start from an entity-oriented model rather than a flat MLP:

- entity/tile encoders;
- shared transformer trunk;
- optional recurrent/global memory;
- worker-task pointer head;
- market autoregressive head;
- optional daily strategy head;
- value head.

A privileged centralized critic is a later comparison, not a V0 requirement.

### Exit Criteria

- tensor/entity schema versioned;
- normalization documented;
- actor-visible versus training-only information explicitly separated;
- model capacity estimate and inference cost recorded.

## Phase 4 — Reward Design

### Competitive objective

Final policy should optimize head-to-head outcome.

Leading terminal reward:

- win `+1`;
- tie `0`;
- loss `-1`.

Bank margin remains a diagnostic and possible curriculum signal but should not replace the final objective unnoticed.

### Dense credit assignment

Investigate **potential-based reward shaping** rather than arbitrary event bonuses.

Candidate potential features:

- bank;
- liquidation value of inventory;
- near-term crop value;
- collectible animal output;
- time-adjusted value of current assets;
- risk of crop death/animal escape/overflow;
- opponent-equivalent value or estimated relative position.

Potential shaping must be tested on fixed trajectories to ensure it telescopes correctly and does not change terminal ranking.

### Auxiliary objectives

Prefer auxiliary prediction tasks over reward hacks:

- future bank;
- near-term harvest/output;
- future market price/inventory;
- opponent sales/hidden inventory inference;
- failure risk;
- win probability.

### Discounting

Compare `gamma=1.0` and very-near-one values. The environment's true objective is terminal, so conventional short-horizon discounting is not automatically appropriate.

### Exit Criteria

- reward equation written exactly;
- terminal conditions tested;
- shaping invariants checked on replayed trajectories;
- reward magnitude/variance measured;
- no maintenance-loop exploit found in reward sanity tests.

## Phase 5 — Behavior-Cloning Bootstrap

### Rationale

Strong public deterministic agents provide demonstrations of precision-sensitive logistics. Use them to avoid forcing random RL exploration to rediscover basic farming competence.

### Plan

1. run multiple public experts across varied 1.32.6 seeds/opponents;
2. store actor observation, engine action, and relevant state metadata each decision point;
3. map primitive traces to intent/task labels where possible;
4. train task and market heads by behavior cloning;
5. measure BC accuracy by action family and episode phase;
6. perturb states and test whether the model conditions on state rather than merely memorizing turn index.

### Guardrail

Behavior cloning is initialization, not the final policy. RL must be allowed to depart from public strategies.

### Exit Criteria

- cloned policy completes viable farms without catastrophic logistics failure;
- clean-route performance is in useful range of source experts;
- generalization measured across unseen seeds/shop compositions/opponents.

## Phase 6 — PPO Robustness and Competitive Training

PPO is the default first RL algorithm, subject to throughput/interface measurements.

### Stage A — robustness fine-tuning

Train against fixed public agents across random seeds/shop regimes and injected perturbations.

### Stage B — frozen competitive pool

Train/evaluate against a versioned mixture of strong public and internal agents.

### Stage C — population self-play

Introduce:

- current champion;
- historical checkpoints;
- diverse strategy families;
- strong public baselines;
- later targeted exploiters.

Do not copy Pokémon opponent percentages without new evidence.

### Stage D — exploiters and specialization

Train policies that exploit common economic patterns if population diversity benefits.

### Promotion metrics

At minimum:

- paired seat-swapped win rate;
- uncertainty/confidence intervals;
- score margin distribution;
- performance by shop regime;
- performance by opponent family;
- catastrophic failure rate;
- runtime.

## Phase 7 — Opponent Modeling and Economic Memory

### Goals

- infer opponent hidden inventory/sale timing from public farm and market changes;
- forecast future supply;
- detect public strategy families;
- adapt product mix and sale timing before collisions happen.

### Compare

- explicit handcrafted history features;
- recurrent actor state;
- auxiliary opponent-inventory prediction;
- dedicated opponent encoder.

The objective is not to hand-code the counter-strategy; it is to provide the policy enough information to learn one.

## Phase 8 — Optional Search / Planning Hybrids

Because the exact simulator is available, later experiments may combine learned value/policy with short-horizon search over macro actions.

Candidate uses:

- market order sequencing;
- daily resource allocation;
- production pivots after unusual shop draws;
- terminal liquidation.

Do not start here before the RL baseline exists.

## Phase 9 — Submission Hardening

- exact Kaggle runtime smoke;
- no-network verification;
- deterministic packaging;
- dependency minimization;
- timeout/memory tests;
- safe handling of malformed/unexpected state;
- submission provenance/hash recording;
- active-submission and leaderboard tracking.

## Planning Agenda While Pokémon Finishes

The next week should produce design artifacts, not expensive training:

1. engine/version/RNG map;
2. exact action grammar;
3. worker-task candidate design;
4. market order quantity design;
5. decision-frequency comparison plan;
6. observation schema;
7. reward/potential equations;
8. BC dataset schema;
9. PPO rollout/self-play design;
10. throughput budget;
11. evaluation gates;
12. final upstream engine recheck before implementation.

## Deferred Until Evidence Supports Them

- raw `NORTH/SOUTH/EAST/WEST` end-to-end PPO from scratch;
- pure SAC over the full action space;
- MCTS over all 720 primitive turns;
- neural pathfinding;
- a mostly hard-coded winning farm where ML only adjusts a few market numbers;
- large architecture sweeps before action/reward contracts are stable;
- expensive training before engine stability and simulator throughput are known.

## Rejected or Currently Unsupported Assumptions

- High single-player bank automatically means high competitive strength.
- The public leaderboard meta will remain stable after 1.32.6.
- Discussion descriptions are authoritative over source behavior.
- A copied action list is robust enough for the final competition.
- RL must either control every primitive movement or is not "real RL."
- Dense handcrafted maintenance rewards are harmless.
- More model complexity is useful without a stronger evaluation protocol.

## Session Workflow

At the end of every substantial session:

1. update `CURRENT_STATE.md` with only active/current facts;
2. append results/failures to `HISTORY.md`;
3. revise this roadmap if priorities changed;
4. add durable choices to `DECISIONS.md`;
5. update `MECHANICS.md` when source/experiments change mechanical understanding;
6. update `research/RL_DESIGN.md` when the RL interface/reward hypothesis changes.
