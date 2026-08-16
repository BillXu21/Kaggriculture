# Kaggriculture V0 Action and Observation Interface

Last updated: 2026-08-16
Status: design proposal; intended first implementation target after engine lock
Engine target: `kaggle-environments >= 1.32.7` with exact source/hash to be frozen before implementation

## Goal

Define the simplest policy interface that:

1. can behavior-clone public expert trajectories without lossy intent reconstruction;
2. preserves the full strategically relevant Kaggriculture action space;
3. exposes enough state for market/shop adaptation and opponent inference;
4. supports PPO refinement with exact policy log-probabilities and action masks;
5. avoids weeks of architecture/action-abstraction work before proving learning;
6. leaves a clean path to hierarchical/task-level control later if direct primitive control proves brittle.

The V0 design is intentionally **BC-first and primitive-action preserving**.

## Key Revision From Earlier Planning

Earlier planning favored an intent-level semi-MDP in which the policy chose tasks such as `WATER(tile)` and a deterministic executor handled movement.

For V0, do **not** start there.

Strong public agents already produce exact engine actions on every turn. Behavior cloning therefore gives direct labels for primitive movement and interaction. Introducing a task language before testing primitive BC would require reconstructing latent intent from traces, could encode strategy into the wrapper, and creates another large source of bugs/assumptions.

V0 hypothesis:

> Direct factorized primitive-action BC is simple enough to learn because expert demonstrations remove the random-exploration/navigation problem. PPO then refines a competent policy rather than learning farming from scratch.

Hierarchical intents remain a V1 experiment if closed-loop primitive BC is too fragile under state drift or if PPO cannot make useful long-horizon changes.

---

# 1. Exact Engine Action Contract

At each turn the agent returns:

```python
{
    "farmer": [op, ...args],
    "hands": [[op, ...args], ...],
    "market": [[op, ...args], ...],
}
```

There is one action for the main farmer, one for each currently active hand, and an ordered market queue of at most 10 orders.

## Worker opcodes

Current documented worker operations:

- `NORTH`
- `SOUTH`
- `EAST`
- `WEST`
- `PASS`
- `PICKUP <item> [n]`
- `DROP`
- `PLANT <crop>`
- `WATER`
- `HARVEST`
- `FERTILIZE`
- `BUILD_COOP`
- `BUILD_PASTURE`
- `DIG`
- `PLACE <item> [n]`
- `FEED`
- `COLLECT_FERTILIZER`
- `CARE`

Note: the current README documents `DROP`; the compact JSON action description omits it. The implementation lock must verify the source parser and regression-test all opcodes rather than relying only on prose.

## Market opcodes

- `BUY_SEED <crop> <n>`
- `BUY_PRODUCT <item> <n>`
- `BUY_ANIMAL <animal> <n>`
- `SELL <item> <n>`
- `HIRE`
- `BUY_LAND`

Orders are ordered and market sequencing matters.

---

# 2. V0 Policy Action Representation

## 2.1 One policy decision per engine turn

Keep the environment as the real 720-turn MDP initially.

Do not add task persistence, macro skipping, or event-driven time abstraction before primitive BC has been measured.

Benefits:

- exact alignment with demonstrations;
- exact reward/discount semantics;
- no ambiguity over task duration;
- easy replay comparison;
- easier PPO log-prob accounting;
- fewer wrapper bugs.

## 2.2 Autoregressive worker decoding

Decode workers in a stable order:

1. main farmer;
2. hands in engine observation/action-list order.

Use a shared worker policy/head so behavior generalizes across varying hand counts.

For each worker, factor the action:

### Opcode head

Categorical over the worker opcodes.

### Conditional item/crop head

Only active when required:

- `PLANT` -> crop in `{WHEAT, CARROT, TOMATO, STRAWBERRY, MELON}`;
- `PICKUP` -> item present/available under the mechanical contract;
- `PLACE` -> valid carried item under the mechanical contract.

### Conditional quantity head

Only active for `PICKUP` and `PLACE` when quantity is meaningful.

V0 proposal: categorical quantity `1..512`, dynamically masked to the mechanically possible range. In most worker cases the effective maximum is much smaller because shed/carry inventories are bounded. A single shared quantity representation is simpler than several special cases.

The dataset should record the original explicit/default quantity so normalization is deterministic.

## 2.3 Why autoregressive workers instead of independent parallel heads

Worker actions have joint mechanical dependencies.

The clearest example is atomic planting: if multiple workers request more seeds of a crop than are owned, all simultaneous plant requests for that crop can fail.

Autoregressive decoding allows the mechanical mask to track already-selected same-turn requests and prevent impossible joint actions without post-hoc action repair.

Post-hoc repair is undesirable because PPO's sampled action/log-prob would no longer match the engine action actually executed.

Worker representations can still be computed in parallel; only the final action selections need to be sequentially conditioned.

## 2.4 Ordered market decoder

After worker actions, decode the market queue autoregressively for up to 10 slots.

Each slot has:

1. `STOP` or market opcode;
2. conditional product/crop/animal argument if required;
3. conditional quantity if required.

Teacher forcing makes this straightforward during BC.

During PPO, the joint market log-probability is the sum of the conditional token log-probabilities.

### Quantity

Use the same categorical `1..512` V0 quantity vocabulary, dynamically masked.

Rationale:

- `SELL` is effectively bounded by owned shed inventory;
- `BUY_PRODUCT`/`BUY_ANIMAL` are constrained by shed capacity;
- seed quantities can exceed 100, so a 100-only vocabulary is unnecessarily restrictive;
- 512 logits are cheap;
- if archived public traces contain quantities >512, expand the vocabulary before training rather than silently clipping.

Do not use coarse percentage buckets until there is evidence exact quantities are unnecessary. Market price impact makes exact quantity strategically meaningful.

## 2.5 Mechanical action masking

Mask **impossible** actions, not merely bad ones.

Examples of acceptable mechanical masks:

- movement that leaves the board;
- plant interaction on a mechanically invalid tile;
- `PLANT` crop with no remaining same-turn seed budget;
- `WATER` where no waterable crop is present;
- `HARVEST` where nothing is harvestable;
- `FEED`/`CARE`/fertilizer collection where the required object is absent;
- structure placement/build actions that cannot legally execute;
- shed interaction when not in a valid shed-adjacent position;
- quantity greater than available inventory/capacity/cash when this is deterministically known;
- `BUY_LAND` when all land is unlocked;
- queue positions after `STOP`.

Examples that should remain available:

- planting a crop with terrible current economics;
- selling into a bad market;
- buying land at a strategically foolish time;
- caring for an animal when another action would be more profitable;
- choosing a legal but inefficient movement path.

The wrapper must not become the strategy.

## 2.6 Seat/player conditioning

Canonicalize observations as `self` and `opponent`, but retain the engine player/seat ID as a scalar feature.

Reason: seat should not force the network to learn duplicate representations, but any subtle order/tie effects should remain representable.

---

# 3. V0 Observation Representation

Use the full actor-visible observation plus mechanically derived features. Do not provide hidden opponent state to the actor.

The representation should make important mechanics easy to infer without directly encoding a strategy.

## 3.1 Time/global features

Include:

- day `0..29`;
- hour `0..23`;
- absolute step `0..719` derived from day/hour;
- turns remaining;
- day fraction / season fraction;
- seat/player ID;
- turns until day end;
- turns until next town-center tick;
- turns until next shop-consumption tick;
- days/turns until next shop unlock;
- number of shop instances already unlocked.

Normalize raw time scalars to roughly `[0,1]` while retaining discrete embeddings where useful.

## 3.2 Farm spatial tensors

Represent **both farms** as fixed 10x10 spatial tensors, canonicalized as self/opponent.

Per tile include raw/categorical information such as:

### Base tile type

- locked;
- empty;
- weed;
- plant;
- coop;
- pasture.

### Plant features

- crop identity;
- planted age / planted day;
- watered today;
- consecutive unwatered;
- yield units;
- fertilizer remaining;
- ongoing vs one-shot crop;
- mechanically derived turns/days to next production/maturity;
- mechanically derived time until decay/death risk.

### Animal/structure features

- structure type;
- animal identity or empty structure;
- placed age/day;
- yield units;
- fed today;
- consecutive unfed;
- cared today;
- fertilizer available;
- pending care bonus;
- mechanically derived time to next scheduled production;
- mechanically derived escape risk/time.

### Worker occupancy

Encode main-farmer/hand occupancy or counts on each tile. Opponent worker inventories remain hidden.

Keep self and opponent farm channels structurally identical where public information permits.

## 3.3 Worker tokens

Create one token/vector for each own current worker.

Features:

- farmer vs hand;
- worker index/order;
- x/y position;
- current tile embedding gathered from the self farm map;
- distance to each shed-adjacent center tile;
- carried inventory by item from `private.inventories`;
- total carried items;
- optionally previous action/opcode.

Opponent workers can be represented through the opponent spatial occupancy channels and/or separate public worker tokens without private inventory.

The same worker encoder/head should be shared across the farmer and hands, with role/index features distinguishing them.

## 3.4 Self private economy

Include:

- shed inventory by item;
- total shed occupancy;
- free shed capacity;
- seed inventory by crop;
- own worker carried inventories;
- current money;
- hires today;
- next deterministic hire cost;
- unlocked quadrants.

Counts should use sensible bounded normalization or `log1p` where ranges can be large.

## 3.5 Opponent public economy

Include:

- opponent money;
- unlocked quadrants;
- hires today;
- public farmer/hand positions;
- full visible opponent farm tile state.

Do not fabricate opponent shed/seeds/carried inventories.

## 3.6 Market product tokens

Treat each market product as a small structured entity/token.

Dynamic features:

- current inventory;
- current price;
- inventory delta from `I0`;
- price relative to base;
- recent inventory deltas;
- recent price deltas;
- current known town demand rate from unlocked shops;
- turns until next relevant town consumption;
- signed normalized scarcity/glut distance.

Static/mechanical features:

- product identity;
- base price;
- `I0`;
- `T`;
- scarcity-side curve family/target;
- glut-side curve family/target.

For 1.32.7 specifically include a mechanically derived signed distance to the scarcity knee:

`scarcity_ratio = (I0 - inventory) / T`

so `1.0` means the product is exactly at its calibration knee. This is not a strategic heuristic; it is a normalized coordinate in the official price function.

For carrot/tomato/egg, the network can then easily distinguish pre-hinge and post-hinge regimes.

## 3.7 Town/shop representation

Because shops are sampled with replacement, represent shops as counts, not booleans.

Include:

- count of each shop type;
- total unlocked shop instances;
- mechanically implied consumption per product per shop tick/day;
- time to next shop unlock;
- remaining possible shop unlock count.

Do not provide future random shop draws or episode seed.

## 3.8 Short history / opponent-flow features

The environment is partially observed because opponent shed/seeds/carried inventory are hidden.

V0 should include inexpensive history features before committing to a recurrent architecture:

- market inventory delta over 1, 4, and 24 turns;
- market price delta over 1, 4, and 24 turns;
- own/opponent money delta over 1, 4, and 24 turns;
- prior own market orders/actions;
- prior shop-count changes;
- mechanically known town consumption over the interval.

A useful derived feature is residual market flow after subtracting known town demand and our own known contribution where execution is inferable. Treat this as an estimate, not hidden truth, because simultaneous order behavior and price-floor rules can create ambiguity.

Compare these explicit history features against a GRU/LSTM only after the feed-forward BC baseline is measured.

---

# 4. Normalization Principles

Do not blindly z-score categorical mechanics.

Suggested conventions:

- bounded day/hour/step -> divide by known maximum;
- money and potentially large seed counts -> `log1p` plus optional raw clipped ratio;
- shed/item quantities -> divide by shed capacity where applicable;
- market inventory -> express both raw normalized inventory and `(inventory-I0)/T`;
- prices -> `price/base` plus optionally `log1p(price)`;
- durations -> divide by season/day horizon as appropriate;
- booleans -> 0/1;
- identities -> embeddings/one-hot, not numeric ordinals.

Keep the feature builder deterministic and versioned.

---

# 5. V0 Model Boundary — Avoid Architecture Rabbit Holes

This document defines the **interface**, not a fancy architecture.

The first model only needs enough structure to test learnability.

A reasonable minimal baseline is:

- small shared CNN for each 10x10 farm;
- MLP encoders for global/product/worker features;
- shared worker action head with autoregressive action conditioning;
- autoregressive market decoder;
- value head;
- optional small fusion attention layer only if needed.

Do not start with a large entity transformer, world model, planning module, or league architecture before BC closed-loop performance is known.

Architecture complexity is earned by a measured failure mode.

---

# 6. Behavior-Cloning Dataset Contract

Every training decision record should contain at least:

- exact engine/package/source hash;
- expert agent identity/hash;
- opponent identity/hash;
- seed;
- seat;
- absolute step/day/hour;
- raw actor observation;
- versioned transformed observation/features;
- exact raw expert action dict;
- factorized worker/market action labels;
- action masks used by our interface;
- post-step observation/reward/status;
- whether any submitted expert action was ineffective/no-op if detectable.

Preserve the raw observation/action so feature/action encodings can be redesigned without rerunning all expert episodes.

## BC losses/metrics

Track more than aggregate cross-entropy:

- worker opcode accuracy;
- conditional argument accuracy;
- quantity accuracy/MAE;
- exact full worker-action match;
- exact market queue match;
- per-game-phase accuracy;
- per-action-family accuracy;
- legal-action rate;
- action entropy;
- closed-loop final bank and win rate versus the source expert/opponents;
- first divergence step from expert trajectory;
- performance after divergence.

Teacher-forced accuracy is necessary but not sufficient. Closed-loop rollout quality is the real BC test.

---

# 7. Required Learning Ladder Before Self-Play

## L0 — Dataset/encoding sanity

Overfit a tiny set of episodes.

Expected result: near-perfect action prediction. If this fails, do not touch PPO.

## L1 — Held-out BC

Train on many expert trajectories and evaluate held-out seeds/opponents under teacher forcing.

## L2 — Closed-loop BC

Run the cloned policy in the real simulator against frozen opponents.

Measure how rapidly small action errors compound and whether it can recover from off-expert states.

## L3 — Fixed-opponent PPO refinement

Starting from BC weights, optimize against one frozen strong opponent over randomized seeds/shop regimes.

Require clear held-out improvement before increasing opponent complexity.

## L4 — Fixed opponent mixture

Train against a frozen versioned panel and test held-out opponents/regimes.

## L5 — Slowly changing population / self-play

Only after L3/L4 show that PPO can improve a stationary BC policy.

This is the direct process correction imported from the Pokémon experience: do not build the moving-target system before proving the learner underneath it moves in the right direction.

---

# 8. Experiments That Decide Whether V0 Is Enough

## A0 — Public quantity distribution

Before locking quantity vocabulary, measure all worker and market quantities in archived public traces.

If >512 is materially used, expand the categorical range or introduce a better parameterization.

## A1 — Primitive BC navigation robustness

Measure closed-loop BC after first divergence.

If the policy gets permanently lost because movement has no persistent goal, compare:

1. better history/state features;
2. recurrent policy;
3. DAgger-like correction where a suitable expert can label perturbed states;
4. hierarchical task/goal actions.

Do not jump directly to #4 without evidence.

## A2 — Parallel vs autoregressive worker actions

Autoregressive is safer for joint legality but costs inference/training throughput. Compare it against parallel shared heads only if sequential decoding becomes a meaningful bottleneck.

Any parallel version must handle joint-action constraints without silently changing sampled actions after the fact.

## A3 — History versus recurrence

Start with explicit multi-timescale market/money deltas. Add recurrent memory only if opponent hidden-state inference or navigation recovery materially benefits.

## A4 — Feature ablations

Once BC works, ablate groups such as:

- derived crop lifecycle features;
- scarcity-ratio/market-curve features;
- shop demand summaries;
- history deltas;
- opponent farm state.

This should tell us which feature engineering actually matters rather than relying on intuition.

---

# 9. Current Recommendation

Implement V0 as:

> **full actor-visible state + mechanically derived lifecycle/economic features -> direct factorized primitive worker actions + ordered autoregressive market actions**

with behavior cloning first.

Do not implement hierarchical intent actions in the first packet.

Do not begin self-play until a BC policy can complete strong closed-loop games and PPO can improve it against a fixed opponent/panel.
