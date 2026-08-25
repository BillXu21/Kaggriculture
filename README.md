# Kaggriculture

Research, planning, evaluation, and agent development for Kaggle's **Kaggriculture** simulation competition.

## Current Direction

Kaggriculture combines highly deterministic farm mechanics with a shared adversarial market and seed-driven episode variation.

Recent balance changes materially increased the value of adaptive economics:

- **1.32.6** reduced town-center demand and changed town shops to sampling with replacement;
- **1.32.7** added sharp scarcity-side `hinge` curves for carrot, tomato, and egg, making those products situationally valuable under favorable randomized shop demand.

The project is explicitly **RL-centered**, but not raw primitive-action RL from scratch.

Working architecture:

> learned intent/task policy + deterministic mechanical executor + learned market policy + self-play

The policy should own meaningful production and economic decisions. Deterministic code should mainly handle mechanically exact busywork such as pathfinding, legal-action masking, persistent task execution, and state bookkeeping.

## Project Status

The repository remains in planning/research mode. The current implementation target is `kaggle-environments >= 1.32.7`; the host has stated that this should be the last balance change except game-breaking bugs, but the exact engine must still be locked and regression-tested before expensive training.

Current planning focus:

1. exact 1.32.7 mechanics, hashes, and RNG mapping;
2. scarcity-regime/pivot measurements;
3. intent-level worker action space;
4. market order/quantity action space;
5. observation/entity schema including market-curve/knee features;
6. potential-based reward shaping resistant to nonlinear price-impact exploits;
7. behavior-cloning bootstrap from strong current-engine traces;
8. PPO/self-play curriculum and evaluation gates;
9. simulator/vectorization throughput requirements.

Read these files first:

- [`CURRENT_STATE.md`](CURRENT_STATE.md): concise active state and immediate priorities.
- [`PLANS.md`](PLANS.md): complete phased roadmap.
- [`research/RL_DESIGN.md`](research/RL_DESIGN.md): current RL action/reward/model design.
- [`HISTORY.md`](HISTORY.md): chronological project record.
- [`DECISIONS.md`](DECISIONS.md): durable decisions and rationale.
- [`MECHANICS.md`](MECHANICS.md): versioned mechanics ledger and uncertainty tracking.
- [`AGENTS.md`](AGENTS.md): operating instructions for future chats and coding agents.
- [`viewer/README.md`](viewer/README.md): exact debug-trace generation, validation, and viewer workflow.

## Repository Layout

```text
.
├── README.md
├── AGENTS.md
├── CURRENT_STATE.md
├── PLANS.md
├── HISTORY.md
├── DECISIONS.md
├── MECHANICS.md
├── research/
│   ├── README.md
│   ├── PUBLIC_BASELINES.md
│   └── RL_DESIGN.md
├── src/
│   └── kaggriculture/
├── scripts/
├── tests/
├── baselines/
├── vendor/
└── artifacts/
```

Implementation directories will be populated when the first bounded coding packet begins. Generated runs, replays, and large artifacts should not be committed unless intentionally small reference fixtures.

## RL Boundary

### Learned decisions

- production mix and crop rotation;
- land/labor investment;
- crop/animal allocation;
- worker task assignment;
- pivots after shop reveals or scarcity changes;
- opponent-aware supply response;
- market buying/selling/order timing;
- strategic memory and later self-play adaptation.

### Deterministic mechanics

- shortest-path movement;
- basic action legality;
- target feasibility;
- task persistence;
- exact simulator bookkeeping;
- action masks for impossible actions.

A mechanically valid but strategically bad action should generally remain available. The wrapper must not quietly become the strategy.

## Reward Direction

The eventual competitive objective should align with head-to-head winning rather than only maximizing absolute bank.

Current leading design:

- terminal win/tie/loss reward;
- potential-based dense shaping using realizable continuation/liquidation value;
- auxiliary prediction tasks for market, production, scarcity-regime evolution, opponent behavior, and win probability;
- no arbitrary maintenance bonuses;
- no naive `quantity × current spot price` valuation under nonlinear hinge curves.

## Public Agents as Training Data

Strong deterministic public agents are useful for benchmarking and behavior-cloning bootstrap.

The goal is not permanent imitation. Prefer fresh 1.32.7 traces, vary seeds/shop compositions/opponents, and allow RL to depart from public product choices that may be stale under the new economics.

## 1.32.7 Economic Change

Merged upstream PR #1399 introduces a scarcity `hinge` curve:

`u = (I0 - inventory) / T`

`hinge = u + 8 * max(0, u - 1)^2`

for carrot, tomato, and egg on the scarcity side.

Current knees:

- carrot: `T=450`, knee inventory 9550;
- tomato: `T=200`, knee inventory 9800;
- egg: `T=332`, knee inventory 9668.

The host reports that with no production, randomized shop demand creates significant price increases in roughly 50% of tomato games, 26% of carrot games, and 22% of egg games. These figures should be reproduced empirically and should not be hard-coded as strategic priors.

The main implication is that product value is explicitly **conditional on episode state**. A strong policy should detect emerging demand/scarcity, estimate time-to-profit, account for its own market impact, and anticipate opponent response.

## Working Principles

- Prefer exact simulator evidence over discussion claims.
- Track engine version, commit, and file hashes for serious runs.
- Evaluate with fixed seeds and both seats.
- Compare against a frozen, versioned competitive pool.
- Record failed experiments and compute incidents.
- Update continuity documents before switching chats or launching expensive runs.
- Keep RL strategically meaningful while removing deterministic mechanical busywork.
- Never encode a static product ranking into action masks/candidate generation.

## Evaluation Standard

A serious evaluation should record at least:

- wins/losses/ties;
- final bank and margin;
- seat-specific results;
- fixed seed list;
- exact agent/engine hashes;
- runtime and invalid/no-op counts;
- shed overflow, stranded inventory, crop loss, animal escape;
- production/sales by product;
- market price/inventory trajectories;
- shop composition by day;
- carrot/tomato/egg scarcity-knee crossings and max prices;
- opponent family/version.

## License

No project license has been selected yet. Third-party artifacts should retain provenance and remain distinguishable from original project code.
