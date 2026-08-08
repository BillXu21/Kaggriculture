# Kaggriculture Historical Record

This file is append-only except for correcting factual errors. New entries are added in reverse chronological order.

## 2026-08-07 — 1.32.6 Town Rebalance and RL-Centered Planning

### Upstream engine change

Reviewed and source-confirmed merged upstream PR `Kaggle/kaggle-environments#1394` (`Kaggriculture town rebalance`).

The change:

- changes default `townCenterSellInterval` from 12 to 24 turns;
- therefore reduces default town-center consumption from twice/day to once/day;
- removes the old town-center demand schedule that increased to 2× after day 10 and 4× after day 20;
- samples town shops with replacement from the full shop table;
- allows duplicate shop names;
- makes each duplicate shop instance consume independently;
- caps total unlocked shop instances at 8 as before in effective maximum count.

Confirmed upstream package source snapshot `bded87b0d7879078c726a93a4884d044f79c4eed` identifies `kaggle-environments` as version `1.32.6`.

The live leaderboard rollout was announced but has not yet been independently locked to an observed server build in this repository.

### Strategic interpretation

The rebalance increases the value of adaptive economic behavior:

- weaker town-center demand means player-generated oversupply should persist longer;
- product gluts and opponent sale timing matter more;
- shop replacement sampling creates materially different per-episode demand regimes;
- duplicated shops can strongly favor particular product categories;
- fixed deterministic public schedules should become less universally optimal across seeds.

The town-shop observation must be treated as a multiset/count vector rather than a binary set.

### RL direction clarified

The project direction changed from "deterministic route first, learning later if useful" to an explicitly **RL-centered hybrid**.

The intended division of responsibility is now:

- learned policy owns production, resource allocation, task assignment, adaptation, and market strategy;
- deterministic infrastructure owns pathfinding, mechanical legality, task execution/persistence, and bookkeeping;
- candidate generation may remove impossible actions but should not encode strategic preferences by hiding mechanically valid actions.

Raw primitive movement PPO from scratch remains deferred. This is now understood as an action-abstraction decision, not a rejection of RL.

### Public RL discussion considered

A competitor reported poor results from standard PPO/SAC attempts because of:

- large observation/action spaces;
- long crop reward delays;
- catastrophic cascades from small logistical mistakes;
- difficulty learning exact watering/feed/seed timing through random exploration.

This was treated as evidence for imitation bootstrap and hierarchical action abstraction rather than evidence against RL.

### New RL design

Added `research/RL_DESIGN.md` covering:

- hierarchical worker-task intent actions;
- deterministic execution of selected intents;
- dedicated autoregressive market head;
- action masking rules;
- turn-level vs event-driven vs hybrid decision frequency;
- entity-based observation/model design;
- recurrent opponent-state inference;
- W/L/T terminal objective;
- potential-based reward shaping;
- auxiliary prediction losses;
- public-agent behavior cloning;
- PPO robustness training and population self-play;
- pre-training experiments for shop variance, market sensitivity, action abstraction, reward sanity, and memory.

### Reward planning

Current leading reward direction:

- final competitive objective aligned with win/tie/loss;
- avoid arbitrary positive maintenance rewards such as watering/harvesting bonuses;
- investigate potential-based shaping using liquidation/future economic value;
- use auxiliary prediction tasks for representation learning instead of silently changing the objective;
- compare `gamma=1.0` with values extremely close to one because the true objective is terminal.

### Demonstration/bootstrap plan

Strong deterministic public agents will be used as training data in addition to opponents:

1. archive exact agent/version provenance;
2. run over varied seeds/shop regimes/opponents;
3. collect state/action trajectories;
4. map primitive actions into intent-level labels;
5. behavior-clone initial competence;
6. fine-tune with PPO/self-play so the model can depart from fixed public scripts.

### Next-week planning agenda

While Pokémon work finishes and Kaggriculture has time to stabilize:

1. map exact 1.32.6 actions and RNG;
2. design worker-task candidate generation;
3. design market quantity/order representation;
4. decide policy decision frequency;
5. version the observation schema;
6. formalize potential functions/reward invariants;
7. design BC trajectory format;
8. design PPO/self-play curriculum;
9. estimate simulator/vectorization throughput;
10. define evaluation/promotion gates;
11. recheck upstream engine changes before implementation/training.

### Files updated

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `HISTORY.md`
- new `research/RL_DESIGN.md`

## 2026-08-06 — Repository Initialization

### Repository

- Created private GitHub repository: `BillXu21/Kaggriculture`.
- Initialized the default branch with project documentation.
- Established continuity files to reduce context loss between chats and agents.

### Strategic Assessment

- Current game structure appears highly deterministic.
- Physical farms are separate, with limited direct interaction.
- The shared market is the primary adversarial coupling mechanism.
- Strong public leaderboard entries are currently dominated by copies or variants of a few deterministic public notebooks.
- The project will remain in planning and mechanics-tracking mode while the engine and rules continue changing.

### Initial Architecture Hypothesis

The initial architecture hypothesis was:

1. deterministic production-route executor;
2. state-based validation and repair;
3. phase-level replanning;
4. opponent-aware market and production policy;
5. coherent expert selection;
6. optional optimization or learning at the macro level.

This was superseded/clarified on 2026-08-07: the project now intends RL to own meaningful strategic decisions, with deterministic code limited primarily to mechanics/execution.

### Research Findings Carried Into the Repository

The following findings were established before repository initialization and should be reverified against the exact live engine before implementation:

- Two players each manage a separate 10×10 farm.
- Matches span thirty days with twenty-four turns per day.
- Banked money determines final reward.
- Unsold inventory has no terminal value.
- Crop and animal schedules are largely deterministic.
- The market is shared and uses inventory-dependent pricing.
- Some daily events are driven by the episode seed.
- Public state exposes enough opponent farm information to support strategy fingerprinting and supply forecasting.
- Strong public strategies use mixed industrial production rather than simple single-crop loops.

### Public Baseline Direction

The first competitive reference should be a strong public deterministic route, preserved with:

- source URL or notebook identity;
- download date;
- immutable file hash;
- engine version assumptions;
- any local modifications;
- known performance evidence.

The project will not rely on redistribution-license concerns as a reason to avoid downloading publicly available Kaggle notebook artifacts, but provenance and third-party boundaries should still be tracked accurately.

### Compute and Workflow Lessons Imported From Pokémon TCG Work

- Chat-context loss can cause stale configuration reuse and wasted compute.
- Every expensive run must be specified in a durable file before execution.
- Current state must remain concise and authoritative.
- Full history must preserve failed experiments, commands, hashes, and output paths.
- Evaluation should not depend on a single seat, weak opponents, or unversioned artifacts.

### Files Established

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `HISTORY.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `AGENTS.md`
- `research/README.md`
- `.gitignore`

### Next Actions At Initialization

1. Establish the exact current engine identity.
2. Archive important public notebooks and agents.
3. Catalog major strategy families.
4. Define the initial fixed-seed, seat-swapped evaluation protocol.
5. Delay competitive implementation until those contracts are recorded.
