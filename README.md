# Kaggriculture

Research, planning, evaluation, and agent development for Kaggle's **Kaggriculture** simulation competition.

## Current Direction

Kaggriculture combines highly deterministic farm mechanics with a shared adversarial market and seed-driven episode variation. Engine version 1.32.6 materially increased economic variance by reducing town-center demand and sampling town shops with replacement.

The project is now explicitly **RL-centered**, but not raw primitive-action RL from scratch.

The working architecture is:

> learned intent/task policy + deterministic mechanical executor + learned market policy + self-play

The policy should own meaningful production and economic decisions. Deterministic code should mainly handle things that are mechanically known and poor uses of learning capacity, such as pathfinding, legal-action masking, persistent task execution, and state bookkeeping.

The first model should therefore learn *what to do* rather than waste most of its capacity rediscovering how to move one tile north repeatedly.

## Project Status

The repository is in its planning and research phase. Major training is intentionally deferred while Pokémon work finishes and while the Kaggriculture engine/rules have time to stabilize.

Current planning focus:

1. exact 1.32.6 mechanics and RNG mapping;
2. intent-level worker action space;
3. market order/quantity action space;
4. observation/entity schema;
5. potential-based reward shaping;
6. behavior-cloning bootstrap from strong public deterministic traces;
7. PPO/self-play curriculum and evaluation gates;
8. simulator/vectorization throughput requirements.

Read these files first:

- [`CURRENT_STATE.md`](CURRENT_STATE.md): concise active state and immediate priorities.
- [`PLANS.md`](PLANS.md): complete phased roadmap.
- [`research/RL_DESIGN.md`](research/RL_DESIGN.md): current RL action/reward/model design.
- [`HISTORY.md`](HISTORY.md): chronological project record.
- [`DECISIONS.md`](DECISIONS.md): durable decisions and rationale.
- [`MECHANICS.md`](MECHANICS.md): versioned mechanics ledger and uncertainty tracking.
- [`AGENTS.md`](AGENTS.md): operating instructions for future chats and coding agents.

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

Empty implementation directories will be added when the first bounded implementation packet begins. Generated runs, replays, and large artifacts should not be committed unless they are intentionally small reference fixtures.

## RL Boundary

The leading design is a hierarchical/semi-Markov policy.

### Learned decisions

- production mix;
- land/labor investment;
- crop/animal allocation;
- worker task assignment;
- pivots after shop reveals;
- opponent-aware supply response;
- market buying/selling/order timing;
- later, strategic memory and self-play adaptation.

### Deterministic mechanics

- shortest-path movement;
- basic action legality;
- target feasibility;
- task persistence;
- exact simulator bookkeeping;
- action masks for impossible actions.

A mechanically valid but strategically bad action should generally remain available to the model. The wrapper must not quietly become the strategy.

## Reward Direction

The eventual competitive objective should align with head-to-head winning rather than only maximizing absolute bank.

Current leading design:

- terminal win/tie/loss reward;
- potential-based dense shaping using liquidation/future economic value;
- auxiliary prediction tasks for market, production, opponent behavior, and win probability;
- no arbitrary bonuses for watering, harvesting, feeding, or other maintenance events unless objective preservation is demonstrated.

## Public Agents as Training Data

Strong deterministic public agents are useful for more than benchmarking. Their trajectories can bootstrap precision-sensitive logistics through behavior cloning before PPO/self-play fine-tuning.

The goal is not to clone them permanently. Training should vary seeds, shop compositions, opponents, and perturbations so the policy learns state-conditioned behavior and can depart from fixed public scripts.

## Working Principles

- Prefer exact simulator evidence over discussion claims.
- Track engine version, commit, and file hashes for every serious run.
- Evaluate competitive agents with fixed seeds and both seat assignments.
- Compare against a frozen, versioned opponent pool rather than weak built-ins.
- Record failed experiments and compute incidents, not only successful results.
- Update continuity documents before switching chats or starting expensive runs.
- Avoid overengineering while mechanics and rules remain unsettled.
- Keep RL strategically meaningful while removing deterministic mechanical busywork.

## Evaluation Standard

A serious evaluation should record at least:

- wins, losses, and ties;
- final bank for both players;
- seat-0 and seat-1 results;
- fixed seed list;
- exact agent and engine hashes;
- runtime and invalid/no-op action counts;
- shed overflow and stranded terminal inventory;
- crop loss and animal escape;
- production and sales by product;
- market-price/inventory trajectories;
- shop composition by episode/day;
- opponent family and strategy version.

## 1.32.6 Economic Change

The current source-confirmed town rebalance:

- town center now buys once/day rather than twice/day;
- late-game 2×/4× town-center demand multipliers are removed;
- shops are sampled with replacement;
- duplicate shop instances are allowed and consume independently;
- total shop instances remain capped at eight.

This weakens automatic market cleanup and makes seed-dependent shop demand more important, strengthening the case for adaptive state-conditioned policies.

## License

No project license has been selected yet. Third-party artifacts should retain provenance and be kept distinguishable from original project code.
