# Kaggriculture

Research, planning, evaluation, and agent development for Kaggle's **Kaggriculture** simulation competition.

## Current Direction

Kaggriculture currently appears to be a highly deterministic farm-production and logistics game with limited direct player interaction. The main coupling between players is the shared market: production, purchases, sales, and town demand alter public inventory and prices.

The initial strategy is therefore **not** primitive-action reinforcement learning. The project will first establish:

1. exact engine and rules tracking;
2. reproducible public-baseline evaluation;
3. deterministic route execution;
4. closed-loop state repair;
5. opponent-aware market and production decisions;
6. only then, optimization or learning where it adds measurable value.

The likely long-term architecture is:

> deterministic executor + state-based repair + opponent-aware macro planning

## Project Status

The repository is in its planning and research phase. Engine behavior and competition rules may still change, so major implementation work is intentionally deferred until the contracts stabilize.

Read these files first:

- [`CURRENT_STATE.md`](CURRENT_STATE.md): concise active state, best-known configuration, current experiments, and immediate priorities.
- [`PLANS.md`](PLANS.md): complete roadmap and deferred ideas.
- [`HISTORY.md`](HISTORY.md): chronological append-only project record.
- [`DECISIONS.md`](DECISIONS.md): durable decisions and their rationale.
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
│   └── README.md
├── src/
│   └── kaggriculture/
├── scripts/
├── tests/
├── baselines/
├── vendor/
└── artifacts/
```

Empty implementation directories will be added when the first bounded implementation packet begins. Generated runs, replays, and large artifacts should not be committed unless they are intentionally small reference fixtures.

## Working Principles

- Prefer exact simulator evidence over discussion claims.
- Track engine version, commit, and file hashes for every serious run.
- Evaluate every competitive agent with fixed seeds and both seat assignments.
- Compare against a frozen, versioned opponent pool rather than weak built-in agents.
- Record failed experiments and compute incidents, not only successful results.
- Update continuity documents before switching chats or starting expensive runs.
- Avoid overengineering while mechanics and rules remain unsettled.

## Evaluation Standard

A serious evaluation should record at least:

- wins, losses, and ties;
- final bank for both players;
- seat-0 and seat-1 results;
- fixed seed list;
- exact agent and engine hashes;
- runtime and invalid/no-op action counts;
- shed overflow and stranded terminal inventory;
- crop loss, animal escape, and repair events;
- production and sales by product;
- market-price trajectories where relevant.

## Competition Notes

The physical farms are separate. Meaningful interaction currently appears concentrated in the shared product market and in adapting to the opponent's visible public farm state. Strong public agents are presently dominated by deterministic or nearly deterministic action schedules, so the first useful improvements are likely robustness, repair, route selection, market timing, and opponent modeling rather than unrestricted turn-by-turn learning.

## License

No project license has been selected yet. Third-party code and competition data must retain their own provenance and should be tracked separately from original project code.
