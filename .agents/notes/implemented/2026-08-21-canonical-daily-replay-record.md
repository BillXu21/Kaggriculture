# Agent Note: Canonical Daily Replay Record

Status: implemented
Date: 2026-08-21

## Problem

The first behavior-cloning and RL experiments will use a once-per-day farm-manager abstraction rather than 720 primitive actions. Raw Kaggriculture replays are large and expensive to repeatedly parse, while the first neural representation and action encoding are still expected to change during experimentation.

The project therefore needs a durable intermediate replay format that captures the mechanically relevant daily state, realized management decisions, and enough provenance/detail to support later reinterpretation without reparsing or reacquiring the original raw episodes.

## Decision

Preprocess each replay into a canonical record for every `(episode, seat, day)`.

A record is defined by:

- **start state:** the first observation for that seat with the current `day` at `hour == 0`, before that day's actions;
- **daily event ledger:** mechanically faithful aggregates of that seat's actions during the day, while preserving exact primitive sale hours;
- **end state:** the first observation at the next day's `hour == 0`, or the terminal state for the final day.

Do not infer day boundaries from `step % 24` when the explicit `day` and `hour` fields are available.

Canonicalize seat ordering as `self` and `opponent` rather than player 0/player 1 so both seats can share one representation.

### Metadata and provenance

Retain at least:

- episode ID and source daily partition/date;
- seat and player/opponent names;
- seed;
- `module_version`;
- manifest `avg_score`, `min_score`, and derived `max_score` when available;
- both terminal bank rewards;
- source/raw replay provenance sufficient to trace any derived example back to the original episode.

### Start-of-day state

Retain the mechanically relevant daily state rather than a model-specific tensor.

For the acting player:

- money;
- shed inventory;
- seed inventory;
- unlocked quadrants/land state;
- full 10x10 board;
- previous-day executor feedback when derivable, initially at least workers hired and total hire cost.

For the public opponent state:

- money;
- unlocked quadrants/land state;
- full public 10x10 board.

Opponent state is retained in the canonical dataset even if the first PPO/BC experiment masks it.

For the shared environment:

- day/season progress;
- market inventory and prices for every product;
- town shop multiset/counts, preserving duplicate shop instances.

### Tile lifecycle representation

The canonical board should preserve raw tile identity/state and mechanically derived lifecycle timing.

For crops, retain or derive enough information to expose:

- crop kind;
- age/raw growth counter where available;
- days until next harvest/output;
- whether harvestable now;
- fertilizer duration/status;
- water/dry/weed-related state required by the current engine.

For animals, retain or derive enough information to expose:

- animal kind;
- raw production/cooldown state where available;
- days until next product;
- feed/starvation state;
- care/bonus state;
- fertilizer availability/state when represented by the engine.

The purpose of derived `days_until_*` fields is mechanical clarity, not heuristic profitability scoring.

### Canonical daily management labels

The first BC adapter may derive V0 targets from the canonical record as follows:

- **crop target:** desired/end-of-day crop composition by crop type, based primarily on next-day board state;
- **animal target:** desired/end-of-day animal counts by species;
- **land target:** resulting unlocked-land/quadrant state or expansion decision;
- **fertilizer allocation:** number of fertilizer applications during the day by crop type;
- **selling:** quantity sold for every product in six intraday windows anchored at hours `0, 4, 8, 12, 16, 20`.

For selling, preserve exact primitive sale hours in the event ledger even if V0 trains on six bins. The initial six bins are `[0-3], [4-7], [8-11], [12-15], [16-19], [20-23]` and are named by their anchor hour. This avoids throwing away information needed by a future 24-turn or reactive selling policy.

Seed purchases, animal purchases, worker hires, routing, watering, routine harvesting, building/placement, and other executor mechanics are not V0 BC targets when they are implied by the high-level daily plan, but their events may be retained in the ledger because storage after daily aggregation is cheap.

### Event-ledger retention

At minimum, retain compact aggregates sufficient to reconstruct or audit:

- plants by crop type;
- digs/removals and prior tile type where available;
- fertilizer applications;
- harvests;
- animal purchases/placements;
- seed purchases;
- product purchases including wheat;
- land purchases;
- worker hires and derived hire cost;
- sales by product and exact hour.

This intermediate format should be broad enough that future BC/action adapters can be regenerated cheaply without reopening the raw 720-turn JSON.

## Rationale

The daily-manager problem contains roughly 30 learned decisions per episode, so post-preprocessing dataset size and RAM throughput are unlikely to be the primary bottleneck. The expensive/awkward step is parsing large raw replays. Preserving a richer canonical daily table is therefore preferable to prematurely collapsing data into the first network's tensors.

Lifecycle timing is more useful than raw object identity alone because the manager must reason about production pipelines and remaining season horizon. At the same time, derived features should remain mechanical rather than encode a hard-coded strategy.

End-of-day target composition fits the deterministic-executor architecture: the manager specifies economic intent and the executor handles seed purchases, placement, paths, workers, and routine maintenance needed to realize that intent.

Six selling windows retain the first important intraday strategic detail while keeping the farm-management cadence daily.

## Alternatives considered and future extensions

### Direct action-delta targets instead of end-state targets

A future model may predict `+N/-N` crop or animal changes rather than absolute end-of-day targets. This can be derived cheaply from start/end state and the event ledger, so the canonical dataset should not lock either choice.

### Explicit per-tile crop replacement decisions

V0 treats crop composition mostly at the crop-count level and lets the executor preserve sensible existing plants. If replacement age, location, or fertilizer timing becomes strategically important, promote replacement to learned control using the retained tile lifecycle state plus DIG/PLANT ledger information.

### Age-bucketed or tile-specific fertilizer allocation

V0 uses fertilizer applications by crop type. A later policy may allocate by crop age, expected harvest cycle, quadrant, or explicit tile. The canonical lifecycle state and exact fertilizer events should support this without raw replay reprocessing.

### Learned wheat/feed economics

V0 initially allows heuristic wheat/feed procurement. Wheat is the leading subsystem to promote into RL because grow/buy/hold/sell/feed choices depend strongly on market economics. Retain all wheat buys, crops, shed state, feed-relevant animal state, and sales so this can be added later.

### Finer or reactive selling

V0 uses six windows. Later experiments may use all 24 turns, a separate high-frequency selling policy, conditional price/market triggers, or multiple manager calls per day. Exact sale hours and full market/town trajectories should be retained when practical so these alternatives remain available.

### Learned harvesting

V0 harvests mechanically. Strategic harvesting may matter when an isolated harvest requires expensive extra labor, shed capacity is constrained, or timing materially affects market interaction. Harvest events and lifecycle state should be retained so learned harvest intent can be introduced if experiments justify it.

### Richer worker/workload feedback

V0 starts with simple previous-day worker count and hire cost. Later variants may add requested/completed workload, primitive-action counts, unfulfilled tasks, or minimum mechanical workload estimates if the manager otherwise cannot learn the cost of ambitious plans.

### Opponent-aware manager

The canonical record retains public opponent board/economy even if V0 masks it. Later experiments can unmask opponent features, add inferred holdings/uncertainty, or train deliberately adversarial market behavior without changing the replay extraction contract.

## Consequences

- Implement the replay parser as a stable canonicalization layer, not as the BC model's tensorizer.
- Validate a small set of daily records manually before scaling to the five daily Kaggle partitions.
- Preserve exact source metadata and enough event detail to audit labels.
- Allow later training code to choose score thresholds, opponent masking, target parameterizations, and selling cadence without repeating the expensive raw replay parse.
- Do not add strategic heuristic scores to the canonical state merely because they are convenient; keep the intermediate format mechanically descriptive.
