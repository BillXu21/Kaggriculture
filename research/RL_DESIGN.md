# Kaggriculture RL Design

Last updated: 2026-08-07
Status: planning document; no training implementation has started

## Goal

Make reinforcement learning the core adaptive decision-maker without forcing the model to relearn deterministic mechanics such as shortest-path movement or basic action legality.

The intended split is:

- **engine/mechanics code** handles legality, pathfinding, task execution, and state bookkeeping;
- **the learned policy** decides what the farm should do, how resources should be allocated, when plans should change, and how to react to the opponent and market;
- **evaluation** is ultimately based on head-to-head win rate, not isolated farm profit.

This is deliberately different from a mostly hard-coded strategy with a small learned market module. The model should own meaningful production and economic decisions.

## Why RL Still Makes Sense

Kaggriculture has a difficult primitive action space and delayed economic rewards, but those are reasons to design the RL interface carefully rather than abandon learning.

Important properties:

- physical mechanics are largely deterministic and precision-sensitive;
- the opponent cannot directly interfere with the farm layout;
- the shared market couples both players economically;
- public opponent state exposes substantial information about future supply;
- opponent private inventory is hidden;
- weeds introduce small stochastic disturbances;
- from engine version 1.32.6 onward, town shops are sampled with replacement, creating materially different demand regimes between episodes;
- reduced town-center demand makes the shared market less resistant to player sell pressure and therefore increases the importance of opponent-aware economics.

The key design problem is therefore **where to place the RL boundary**.

## Core Hypothesis: Hierarchical Intent RL

Do not make the first model choose raw `NORTH`, `SOUTH`, `EAST`, and `WEST` sequences for every worker.

Instead, let the model choose **intent-level actions** and let a deterministic executor compile those intents into legal primitive actions.

Example:

- model: `water strawberry at tile 63`;
- executor: assigns the selected worker, computes the path, moves until adjacent/on-target as required, performs `WATER`, then marks the task complete.

This preserves strategic control while removing a large amount of uninteresting navigation credit assignment.

The design should be a semi-Markov hierarchy: tasks can persist for several primitive turns, while the policy is queried when a worker becomes idle, a task becomes invalid, a major event occurs, or a global replanning decision is due.

## Proposed Action Space

This is a design target, not yet a locked contract.

### A. Worker Task Head

For each available worker, choose from a generated set of currently meaningful task candidates.

Candidate task families:

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
- movement/positioning target when movement itself is strategically useful.

The candidate generator should encode **mechanical feasibility**, not strategic preference. For example, excluding `HARVEST` on an empty tile is acceptable; deciding which mature crop is worth harvesting is the model's job.

Candidate tasks can be scored with an entity/pointer head instead of allocating a huge fixed categorical output over every possible action × crop × tile combination.

### B. Farm Strategy Head

At lower-frequency boundaries—probably start of day and selected major events—the policy may also choose or update farm-level targets such as:

- whether/when to unlock land;
- target number of hired hands;
- crop allocation targets;
- desired structure/animal mix;
- fertilizer allocation priorities;
- desired inventory reserves;
- risk/cash buffer;
- whether to pivot production because of opponent development or town-shop composition.

These outputs should guide candidate generation and task priorities, not become a fixed scripted route.

Whether this head is necessary in V0 or emerges from the task-level policy is an open experiment.

### C. Market Head

Market actions deserve a dedicated head because they have different structure and are the main direct interaction channel.

The market policy should choose an autoregressive sequence of up to the engine limit of market orders. Each order conceptually contains:

- order type;
- product/resource;
- quantity;
- stop/end token.

Candidate order types include seed purchases, product purchases, animal purchases, selling, hiring, and land purchases as supported by the live action contract.

Quantity representation is unresolved. Candidate options:

1. exact integer quantity where the feasible range is small;
2. logarithmic/discrete buckets plus `ALL` / `MAX_SAFE`;
3. a parameterized distribution over a bounded integer;
4. generated candidate quantities based on current inventory, cash, and capacity.

Because unit order execution and price impact matter, the order sequence should remain visible to the policy rather than collapsing market behavior into one aggregate target.

### D. Action Masking

Mechanical impossibilities should be masked aggressively:

- unaffordable purchase;
- locked/invalid interaction target;
- unavailable seed/product/animal;
- shed-capacity violation;
- impossible structure action;
- task requiring an absent object;
- quantities outside engine bounds.

Do **not** mask merely bad strategy. The policy must be able to learn that distinction.

## Decision Frequency

Three candidates should be compared before locking the environment wrapper:

### Turn-level

Policy evaluates every primitive turn but can emit `CONTINUE_TASK` for workers already executing an intent.

Pros: maximum responsiveness.
Cons: longer credit assignment and more inference.

### Event-driven semi-MDP

Policy is called only when a task completes/fails, a worker becomes idle, day boundaries occur, town/shop state changes, or market/farm conditions cross a replanning trigger.

Pros: much shorter effective horizon and more meaningful decisions.
Cons: implementation is more involved and simultaneous worker decisions must remain coherent.

### Hybrid

A low-frequency global strategy decision plus event-driven worker task decisions and turn-level market decisions.

This is the current leading hypothesis.

## Observation Design

The observation should retain raw information while adding mechanically derived features that reduce needless learning burden.

### Global/time

- day and turn within day;
- turns remaining;
- own/opponent money;
- unlocked land;
- hired-hand count;
- shop-instance counts;
- shop unlock count and next unlock timing;
- engine-relevant capacities and limits.

### Own private economy

- shed inventory by item;
- seed inventory;
- each worker's carried inventory;
- free shed capacity;
- immediately spendable cash.

### Farm entities

For every tile/object:

- location;
- unlocked state;
- crop/weed/structure/animal type;
- age/growth state;
- water/fertilizer state;
- yield/output state;
- feed/care state;
- worker occupancy or distance features where useful.

Encode both own and opponent farms, with a player/visibility flag.

### Market

- current inventory by product;
- current prices;
- deviation from initial inventory;
- recent price/inventory deltas;
- known town demand implied by current shop multiset;
- time until town consumption ticks.

### Opponent information

- all visible opponent farm entities;
- money trajectory;
- land/labor development;
- crop maturity pipeline;
- livestock production pipeline;
- recent market changes attributable or partially attributable to opponent behavior.

A recurrent policy or compact recent-history encoder may be valuable because opponent shed and carried inventory are hidden.

## Model Architecture Candidates

Preferred starting family: **entity transformer + recurrent/global state**, not a flat MLP.

Possible structure:

1. tile/entity encoders for both farms;
2. worker/entity encoders;
3. market/product entities;
4. shop-count and global scalar tokens;
5. shared transformer trunk;
6. optional GRU/LSTM memory over turns/decision points;
7. worker-task pointer head;
8. market autoregressive head;
9. optional daily strategy head;
10. value head.

A centralized critic with privileged simulator state is worth testing later for variance reduction, but V0 can use only actor-visible observations to keep the training stack simpler.

## Reward Design

### True Objective

Competition strength is primarily head-to-head winning, so the final training objective should align with win/loss/tie rather than only maximizing raw bank.

Candidate terminal reward:

- win: `+1`;
- tie: `0`;
- loss: `-1`.

Bank margin remains an important metric and may be useful in curricula, but should not silently replace the competitive objective.

### Potential-Based Dense Shaping

Avoid direct handcrafted rewards such as `+0.1 for watering` or `+1 for harvesting`; these can teach the model to optimize proxy events instead of winning.

Instead, investigate potential-based shaping:

`r'_t = r_t + beta * (gamma * Phi(s_{t+1}) - Phi(s_t))`

where `Phi` estimates liquidation/future economic value and is defined consistently at terminal state.

Candidate components of `Phi`:

- banked money;
- liquidation value of shed inventory at current market conditions;
- near-term harvest value of existing crops;
- collectible animal output;
- value of seeds/animals/structures only to the extent they can contribute before season end;
- liabilities/risks such as imminent animal escape, crop death, shed overflow, or production that cannot be sold before termination;
- opponent-equivalent value or estimated win margin.

With a mathematically consistent potential, the dense signal can improve credit assignment without changing the underlying optimal policy objective.

### Auxiliary Learning Instead of Reward Hacking

Useful auxiliary targets may include:

- next-day bank;
- future market prices/inventory;
- harvest/output over the next N turns;
- opponent product sales;
- opponent hidden inventory estimate;
- crop/animal failure risk;
- probability of eventual win.

These can improve representations without changing the reward function.

### Discounting

Because the actual objective is terminal, compare `gamma = 1.0` against values extremely close to one. A small conventional gamma can incorrectly prefer earlier money even when only final bank/win matters.

GAE lambda can still be below one for variance control.

## Imitation Bootstrap

The existence of strong deterministic public agents is an advantage for RL training.

Plan:

1. archive several strong public agents under exact engine provenance;
2. run them across many 1.32.6 seeds and opponent combinations;
3. record full state/action trajectories;
4. map primitive actions back into our intent/task representation where possible;
5. behavior-clone the initial policy;
6. fine-tune with RL so the policy can depart from the public scripts.

This gives the model competence in precision-sensitive logistics before asking policy gradient updates to discover farming from bankruptcy-level random exploration.

Important caveat: a pure time-indexed deterministic trace can be memorized. Training must include varying shop draws, weeds, opponents, and perturbations so the model learns state-conditioned behavior rather than only turn numbers.

## Candidate RL Algorithm

PPO is the default first algorithm because:

- actions are mostly discrete/structured;
- action masks are important;
- recurrent PPO is well understood;
- behavior-cloned initialization is straightforward;
- self-play and frozen-opponent curricula fit naturally;
- we already have experience operating PPO training infrastructure.

SAC is not the natural first choice for this largely discrete/autoregressive action space.

Do not lock PPO permanently until environment throughput and action-interface experiments are measured.

## Training Curriculum

### Stage 0 — Offline imitation

Behavior clone strong deterministic/public traces.

### Stage 1 — Robustness fine-tuning

Train against fixed public baselines across many seeds and perturbations. Optimize recovery from weeds, random shop compositions, route drift, and market deviations.

### Stage 2 — Competitive opponent pool

Train against a frozen mixture of strong public and internal agents. Track paired win rate and not just mean bank.

### Stage 3 — Population/self-play

Maintain historical checkpoints and strategy diversity so the policy does not overfit one market behavior.

Potential opponent mixture:

- current champion;
- strong public agents;
- historical checkpoints;
- deliberately different economic strategies;
- later exploiters targeted at market weaknesses.

Exact percentages should be empirical, not copied from Pokémon.

### Stage 4 — Exploiters / strategic specialization

If useful, train policies that exploit specific common strategies and add them to the population or distill their behavior into a more general policy.

## 1.32.6 Balance Change Implications

The August 2026 town rebalance changes the learning problem materially.

### Reduced town-center demand

The town center now removes only one of each non-fertilizer product once per day and no longer ramps to 2×/4× later.

Expected consequences:

- player oversupply persists longer;
- market crashes from coordinated or competing production become more severe;
- timing of sales becomes more important;
- opponent production forecasting becomes more valuable;
- a high-output deterministic route can be punished if both players flood the same product.

### Shops sampled with replacement

Every shop unlock is now sampled from the full shop table with replacement, with at most eight instances. Duplicate shops each consume independently.

Expected consequences:

- episode-to-episode product demand has materially higher variance;
- crop/animal portfolio adaptation has more value;
- the agent should encode the shop multiset as counts, not binary unlocked/not-unlocked flags;
- fixed public schedules become less universally optimal;
- newly revealed shops create natural replanning points;
- future shop draws remain uncertain even after earlier shops are known.

This change strengthens the case for state-conditioned RL over a single deterministic action tape.

## Experiments To Design Before Training

No large training run should start until these are specified.

### E1 — Shop-regime variance

Run the same frozen deterministic policy over many seeds and measure outcome/production/price variance as a function of final and partial shop composition.

Questions:

- how much final bank variance is explained by shop multiset?
- which duplicated shops matter most?
- how early can an adaptive pivot pay back its switching cost?

### E2 — Opponent market sensitivity

Hold our policy fixed and vary opponent production/sale policies.

Measure:

- product price trajectories;
- value of selling earlier/later;
- value of changing product mix;
- whether deliberate market pressure can flip match outcomes.

### E3 — Action abstraction comparison

Compare primitive, task-level, event-driven, and hybrid interfaces on:

- action-space size;
- fraction of masked actions;
- effective horizon;
- simulator throughput;
- behavior-cloning accuracy;
- robustness to perturbations.

### E4 — Reward-shaping sanity

Before RL, replay fixed trajectories through candidate reward functions and verify:

- terminal ranking matches true W/L objective;
- potential shaping telescopes as intended;
- no reward is obtained from pointless repetitive maintenance;
- bankruptcy/failure trajectories produce useful negative temporal signal without changing the final objective.

### E5 — Memory requirement

Compare a Markov feed-forward policy with explicit history features and recurrent memory for opponent-inventory/sale inference.

## Planning Agenda For The Next Week

While Pokémon work finishes and the engine has time to stabilize:

1. lock the exact 1.32.6 engine and enumerate all action schemas;
2. quantify all sources of randomness and when they are revealed;
3. design the intent/task candidate generator on paper;
4. decide the market quantity/action representation;
5. specify observation tensors/entities and normalization;
6. formalize potential-based reward candidates;
7. define the BC dataset schema from public-agent rollouts;
8. define the PPO/self-play curriculum and evaluation gates;
9. estimate simulator/vectorization throughput requirements;
10. review upstream engine changes again before implementation begins.

## Open Design Questions

- Should the first policy have a separate daily strategic head or only task-level decisions?
- Should task assignment be simultaneous for all workers or autoregressive by worker?
- How should conflicts be handled when multiple workers select the same target?
- What is the best market quantity representation?
- Should market decisions occur every primitive turn or only when economically relevant?
- How much action history is needed to infer opponent hidden inventory?
- Is a centralized privileged critic worth the additional training-only state plumbing?
- What potential function gives the lowest-variance learning signal without encoding too much strategy by hand?
- How much behavior cloning is useful before it anchors the policy too strongly to public scripts?
- Can the simulator be vectorized fast enough for PPO without rewriting core engine behavior?

## Non-Goals For Now

- implementing the model;
- launching expensive RL training;
- locking a large neural architecture;
- hand-authoring a complete winning farm strategy;
- using arbitrary dense event rewards without objective-preservation analysis;
- assuming engine 1.32.6 will be the final competition engine.
