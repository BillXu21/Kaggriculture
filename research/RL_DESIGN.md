# Kaggriculture RL Design

Last updated: 2026-08-16
Status: planning document; no training implementation has started
Current engine target: `kaggle-environments >= 1.32.7`

## Goal

Make reinforcement learning the core adaptive decision-maker without forcing the model to relearn deterministic mechanics such as shortest-path movement or basic action legality.

The intended split is:

- **engine/mechanics code** handles legality, pathfinding, task execution, and state bookkeeping;
- **the learned policy** decides what the farm should do, how resources should be allocated, when plans should change, and how to react to the opponent, market, and randomized town demand;
- **evaluation** is ultimately based on head-to-head win rate, not isolated farm profit.

The model should own meaningful production and economic decisions. Deterministic code must not quietly become the real strategy.

## Why RL Makes Sense After 1.32.7

The environment still has precision-sensitive deterministic logistics, but the economic decision problem is increasingly state-dependent.

Important properties:

- physical mechanics are largely deterministic;
- the opponent cannot directly interfere with farm layout;
- the shared market couples players economically;
- opponent farm state is public but opponent shed/inventory is hidden;
- weeds add small stochastic disturbances;
- since 1.32.6, shops are sampled with replacement, so episodes realize different demand regimes;
- lower town-center demand makes player-generated gluts more persistent;
- since 1.32.7, carrot, tomato, and egg can enter sharp scarcity-price regimes when randomized shop demand is high and production is low.

This creates a useful learning problem: identify an emerging opportunity, estimate whether a pivot can pay back before the season ends, anticipate whether the opponent will exploit/suppress it, and allocate labor/land/capital accordingly.

A fixed product ranking is now explicitly wrong.

## 1.32.7 Scarcity Mechanic Relevant to RL

PR #1399 adds a `hinge` scarcity curve for carrot, tomato, and egg.

For scarcity distance `x = I0 - inventory` and product calibration quantity `T`:

`u = x / T`

`hinge = u + 8 * max(0, u - 1)^2`

Thus the curve behaves roughly linearly until scarcity reaches `T`, then accelerates quadratically.

Current knees:

| Product | T | Knee inventory `I0-T` | Base price |
|---|---:|---:|---:|
| Carrot | 450 | 9,550 | 35 |
| Tomato | 200 | 9,800 | 60 |
| Egg | 332 | 9,668 | 50 |

Host-reported probability of substantial scarcity under **no production**:

- tomato: ~50%;
- carrot: ~26%;
- egg: ~22%.

These percentages should be reproduced empirically and are not policy priors to hard-code.

### Why this matters

The value of producing one of these products is approximately a function of:

- current shop multiset;
- current market inventory and price;
- distance to the scarcity knee;
- expected future town draw/consumption;
- own production lead time and capacity;
- opponent visible production pipeline;
- inferred opponent inventory and likely sale timing;
- remaining season length;
- opportunity cost of abandoning current production.

This is exactly the kind of conditional value estimation that a learned policy/value function should be able to improve on versus a single deterministic route.

## Core Hypothesis: Hierarchical Intent RL

Do not make the first model choose raw `NORTH`, `SOUTH`, `EAST`, and `WEST` sequences for every worker.

Instead, let the model choose **intent-level actions** and let a deterministic executor compile intents into legal primitive actions.

Example:

- model: `water strawberry at tile 63`;
- executor: chooses the path, moves the assigned worker, performs `WATER`, and marks task completion/failure.

The design should behave as a semi-Markov hierarchy: tasks may persist for several primitive turns, while the policy is queried when workers become idle, tasks fail/complete, major economic events occur, or global replanning is due.

## Proposed Action Space

### A. Worker Task Head

For each available worker, score feasible task candidates.

Task families:

- `WAIT` / continue current task;
- `PLANT(crop, tile)`;
- `WATER(tile)`;
- `HARVEST(tile)`;
- `FERTILIZE(tile)`;
- `DIG(tile)`;
- `BUILD_COOP(tile)`;
- `BUILD_PASTURE(tile)`;
- `PLACE(animal, tile)`;
- `FEED(tile)`;
- `CARE(tile)`;
- `COLLECT_FERTILIZER(tile)`;
- `PICKUP(item, amount)`;
- `DROP(item, amount)`;
- purposeful movement/positioning when movement itself is strategic.

Use entity/pointer scoring rather than a huge flat categorical space.

The candidate generator may encode **mechanical feasibility**, not strategic preference. It may reject harvesting an empty tile; it should not reject planting carrots because a heuristic thinks carrots are normally weak.

### B. Global/Farm Strategy Head

At lower-frequency boundaries, optionally choose/update farm-level targets:

- expansion timing;
- target hired hands;
- crop allocation;
- structure/animal mix;
- fertilizer allocation;
- cash/inventory reserves;
- production pivots after shop/market changes;
- end-game rotation/liquidation posture.

1.32.7 makes a global head more plausible because production pivots may require coordinated changes across many workers rather than isolated local task choices.

### C. Market Head

Use a dedicated autoregressive market policy because order sequencing and price impact matter.

Each emitted order contains:

- order type;
- product/resource;
- quantity;
- stop/end token.

Quantity representation remains open:

1. exact bounded integer;
2. discrete/log buckets plus `ALL`/`MAX_SAFE`;
3. parameterized bounded distribution;
4. generated candidate quantities tied to inventory/cash/capacity and market-curve landmarks.

For 1.32.7, useful **mechanically derived quantity landmarks** may include quantities that move inventory to a scarcity knee or other price-curve breakpoints. These can be offered as candidates, but candidate generation must not decide whether exploiting the breakpoint is strategically good.

### D. Action Masking

Mask mechanical impossibilities aggressively:

- invalid interaction target;
- unavailable resource;
- impossible structure action;
- impossible quantity;
- capacity violations;
- actions the engine contract cannot execute.

Do **not** mask merely unprofitable actions. In particular, do not encode static product rankings into masks or candidate generation.

## Decision Frequency

Compare three interfaces before locking V0:

### Turn-level

Policy evaluated every primitive turn, with persistent `CONTINUE_TASK` behavior.

### Event-driven semi-MDP

Policy queried on task completion/failure, idle workers, day boundaries, shop unlocks, major market changes, or other replanning events.

### Hybrid

Low-frequency global strategy + event-driven worker tasks + frequent market decisions.

Current leading hypothesis: **hybrid**.

1.32.7 adds natural economic replanning triggers:

- shop unlock changes demand composition;
- a hinge product approaches/crosses `I0-T`;
- opponent begins producing a hinge product;
- price/inventory velocity indicates an opportunity is appearing/disappearing;
- remaining time crosses the latest profitable pivot point for a crop/animal.

These triggers should determine *when to ask the policy*, not *what decision to make*.

## Observation Design

### Global/time

- day and turn within day;
- turns remaining;
- own/opponent money;
- land unlocked;
- hired-hand counts;
- shop-instance counts;
- next known town/shop timing;
- engine capacities/limits.

### Own private economy

- shed inventory by item;
- seeds;
- worker-carried inventory;
- free shed capacity;
- cash.

### Farm entities

For each relevant tile/object:

- location;
- unlocked state;
- crop/weed/structure/animal type;
- age/growth state;
- water/fertilizer state;
- yield/output state;
- feed/care state;
- occupancy/distances where mechanically useful.

Encode both farms with player/visibility flags.

### Market product entities

For every product include at least:

- current inventory;
- current price;
- base price;
- `I0`;
- `T`;
- scarcity/glut shape identifiers or embeddings;
- below/above target parameters;
- normalized scarcity `(I0-inventory)/T`;
- signed distance to scarcity knee `(I0-T)-inventory`;
- indicator/continuous feature for being beyond the hinge knee;
- local price sensitivity / marginal price change if useful;
- recent inventory and price deltas;
- known consumption implied by current shops;
- time until town consumption ticks.

Do **not** provide a handcrafted label such as `GOOD_PRODUCT`. Expose mechanics/state and let the model value it.

### Opponent information

- visible farm entities;
- money trajectory;
- land/labor development;
- crop maturity pipeline;
- livestock/output pipeline;
- evidence from market changes consistent with opponent sales;
- optional inferred hidden inventory through recurrent state/auxiliary prediction.

## Model Architecture Candidates

Preferred starting family: **entity transformer + recurrent/global state**, not a flat MLP.

Possible structure:

1. farm/tile/entity encoders;
2. worker encoders;
3. product/market entities containing curve and scarcity features;
4. shop-count/global tokens;
5. shared transformer trunk;
6. optional GRU/LSTM memory;
7. worker-task pointer head;
8. market autoregressive head;
9. optional global strategy head;
10. value head.

A centralized privileged critic is worth testing later, especially if hidden opponent inventory makes actor-value estimation noisy.

## Reward Design

### True Objective

Final training should align with competitive outcome.

Candidate terminal reward:

- win: `+1`;
- tie: `0`;
- loss: `-1`.

Bank margin remains an important diagnostic/curriculum signal but should not silently replace W/L.

### Potential-Based Shaping

Investigate:

`r'_t = r_t + beta * (gamma * Phi(s_{t+1}) - Phi(s_t))`

Potential candidates may include:

- bank;
- liquidation value of inventory;
- time-realizable crop/animal output;
- current assets weighted by remaining productive lifetime;
- shed overflow / crop death / animal escape risks;
- opponent-equivalent economic value;
- learned or model-based continuation value.

**1.32.7 warning:** naive mark-to-market inventory/crop value can massively overvalue a temporary hinge spike. If the policy itself produces/sells the resource, price moves. `Phi` must therefore avoid treating `quantity × current_spot_price` as realizable value for large quantities without price impact and time-to-production/sale considerations.

That makes an exact/approximate liquidation simulator or learned continuation value particularly attractive.

### Auxiliary Learning

Useful targets:

- next-day bank;
- near-term harvest/output;
- future market inventory/price;
- probability/time-to-crossing of each hinge knee;
- future realized shop demand;
- opponent sales/hidden inventory;
- crop/animal failure risk;
- production-pivot profitability;
- eventual win probability.

Auxiliary targets should help representation learning without redefining reward.

### Discounting

Compare `gamma = 1.0` and very-near-one values. The actual objective is terminal.

## Imitation Bootstrap

Strong deterministic public agents remain useful demonstrations.

Plan:

1. archive exact agents and engine provenance;
2. roll them out across many **1.32.7** seeds/opponents;
3. record full state/action trajectories;
4. map primitive actions to intent/task labels;
5. behavior-clone initial logistics competence;
6. fine-tune with RL so the model can depart from public routes.

Important caveats after 1.32.7:

- pre-1.32.7 public traces encode outdated product economics;
- BC should emphasize mechanical competence rather than treating demonstrated product mix as optimal labels forever;
- collect fresh 1.32.7 traces and deliberately include unusual shop regimes;
- state perturbation/counterfactual training may be useful so the model sees opportunities not present in the public expert's fixed route.

Potential approach: down-weight or separate global economic decisions during BC while strongly cloning precision-sensitive worker logistics.

## Candidate RL Algorithm

PPO remains the default first algorithm because:

- actions are mostly discrete/structured;
- masks matter;
- recurrent PPO is well understood;
- BC initialization is straightforward;
- self-play/frozen-opponent curricula fit naturally;
- project experience already exists with PPO infrastructure.

Do not lock PPO until simulator throughput and action interface are measured.

## Training Curriculum

### Stage 0 — Offline imitation

Clone public/internal demonstrations for viable mechanics and basic economy.

### Stage 1 — Regime adaptation / robustness

Train against fixed public baselines across varied 1.32.7 seeds. Ensure the distribution includes:

- ordinary shop compositions;
- tomato-scarcity episodes;
- carrot-scarcity episodes;
- egg-scarcity episodes;
- overlapping scarcity opportunities;
- episodes where the opponent supplies the scarce product and collapses the opportunity.

Do not artificially balance these regimes in the final training distribution without accounting for their real occurrence probabilities; targeted oversampling may be useful early for learning, followed by correction/fine-tuning on the natural distribution.

### Stage 2 — Frozen competitive pool

Train/evaluate against versioned strong public/internal agents. Track performance by economic regime, not only pooled average.

### Stage 3 — Population/self-play

Maintain champion, historical checkpoints, public baselines, and strategy-diverse agents to avoid overfitting one response to scarcity events.

### Stage 4 — Targeted exploiters

If useful, train opponents that aggressively exploit or suppress hinge resources, forcing the main policy to learn game-theoretic responses rather than blindly chase spikes.

## Experiments To Design Before Training

### E1 — 1.32.7 scarcity distribution

With no player production of each target resource, measure over many seeds:

- whether scarcity crosses `T`;
- first crossing day/turn;
- minimum inventory;
- maximum price;
- final shop multiset;
- conditional distribution by shop counts.

Compare with host-reported ~50% tomato / 26% carrot / 22% egg figures.

### E2 — Production-pivot frontier

For each target product, estimate the latest profitable commitment time under different observed scarcity states.

Examples:

- carrot planting is quick but one-shot;
- tomato has longer setup but recurring yield;
- eggs require coop/goose/feed logistics.

Measure profit and, more importantly, head-to-head win impact after accounting for price impact from the production itself.

### E3 — Opponent suppression / competition

Hold shop regime fixed and vary opponent response:

- ignores scarce product;
- reacts immediately;
- reacts late;
- already has production;
- deliberately floods the market.

This tests whether the policy needs explicit opponent modeling to exploit scarcity safely.

### E4 — Shop-regime variance

Run frozen deterministic policies across many seeds and quantify score/win variance attributable to shop multiset and hinge events.

### E5 — Action abstraction

Compare primitive/task/event-driven/hybrid interfaces on action size, effective horizon, BC accuracy, robustness, and throughput.

### E6 — Reward shaping sanity

Replay fixed trajectories and verify potential shaping telescopes, preserves terminal ranking, and does not create fake reward from temporary hinge prices or self-induced mark-to-market bubbles.

### E7 — Memory requirement

Compare feed-forward explicit history features vs recurrent memory for opponent hidden inventory/sales and scarcity-response prediction.

### E8 — Throughput

Benchmark complete 720-turn games/sec and decision points/sec before selecting model size or rollout configuration.

## Planning Agenda Before Implementation

1. lock exact 1.32.7 engine source/spec/hash;
2. enumerate all action schemas;
3. map RNG sources and reveal timing;
4. design intent/task candidate generation;
5. choose market quantity representation;
6. version observation schema including market-curve/knee features;
7. formalize reward/potential candidates;
8. define BC dataset and what parts of demonstrations receive strong vs weak imitation weight;
9. define PPO/self-play curriculum and promotion gates;
10. run the 1.32.7 scarcity/pivot studies before large training;
11. benchmark simulator/vectorization throughput;
12. recheck upstream for bug fixes immediately before implementation.

## Open Design Questions

- Separate daily/global strategy head or task-only V0?
- Simultaneous or autoregressive worker assignment?
- Conflict handling for multiple workers targeting one entity?
- Best market quantity representation?
- How frequently should market policy run?
- Should market curve parameters be raw features, product embeddings, or both?
- Does recurrence materially improve hidden-inventory/opponent-response inference?
- Is a privileged centralized critic worth the training-only plumbing?
- What potential avoids mark-to-market exploitation under nonlinear price impact?
- How much BC should apply to strategic crop/product choices versus mechanical execution?
- Should early RL oversample rare scarcity regimes and then correct to natural frequency?
- Can the simulator be vectorized fast enough for meaningful PPO/self-play?

## Non-Goals For Now

- raw primitive movement PPO from random initialization;
- launching expensive training before engine lock/throughput measurements;
- hand-coding a fixed product priority table from 1.32.7;
- using current spot price as naive realizable inventory value;
- large architecture sweeps before action/reward contracts stabilize;
- assuming host-reported scarcity percentages remain true under strategic player production;
- assuming 1.32.7 cannot receive bug-fix changes.
