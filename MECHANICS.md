# Kaggriculture Mechanics Ledger

Last updated: 2026-08-06

## Purpose

This file separates verified engine behavior from documentation, discussion claims, and stale assumptions.

Confidence labels:

- `CONFIRMED_SOURCE`: observed in a specific recorded engine source snapshot.
- `CONFIRMED_EXPERIMENT`: reproduced with a controlled behavioral test.
- `DISCUSSION_CLAIM`: reported publicly but not independently verified.
- `OUTDATED`: known to describe an older engine or contract.
- `UNKNOWN`: unresolved or insufficiently specified.

## Engine Identity

- Live Kaggle server version: `UNKNOWN`
- Local package version: not installed
- Vendored source commit: not established
- Engine file SHA-256: not established
- Specification file SHA-256: not established
- Last source research snapshot: 2026-08-04
- Status: **do not treat this ledger as an engine lock yet**

The first implementation task must record exact source and hashes before relying on constants below.

## Match Contract

| Mechanic | Current understanding | Confidence |
|---|---|---|
| Players | 2 | `CONFIRMED_SOURCE` in pre-repo research snapshot |
| Farm size | Separate 10×10 farm per player | `CONFIRMED_SOURCE` |
| Farm quadrants | Four 5×5 quadrants | `CONFIRMED_SOURCE` |
| Match length | 30 days × 24 turns = 720 turns | `CONFIRMED_SOURCE` |
| Starting money | 3,000 | `CONFIRMED_SOURCE` |
| Starting labor | One farmer | `CONFIRMED_SOURCE` |
| Starting land | Northwest quadrant | `CONFIRMED_SOURCE` |
| Additional land costs | NE 1,000; SW 2,000; SE 4,000 | `CONFIRMED_SOURCE` |
| Worker actions | One action per worker each turn | `CONFIRMED_SOURCE` |
| Market orders | Up to 10 per turn | `CONFIRMED_SOURCE` |
| Default action timeout | Approximately 1 second | `CONFIRMED_SOURCE`; reverify server settings |
| Final reward | Final banked money | `CONFIRMED_SOURCE` |
| Terminal inventory value | Zero unless sold before termination | `CONFIRMED_SOURCE` |

## Observation and Hidden Information

### Publicly visible

- both players' banked money;
- farm tiles and visible contents;
- worker positions;
- unlocked quadrants;
- hire count;
- shared market inventory and prices;
- town shops;
- current day and hour.

Confidence: `CONFIRMED_SOURCE` in pre-repo research snapshot.

### Private to each player

- own shed contents;
- own seed inventory;
- inventories carried by own workers.

Opponent shed, seeds, and carried inventories are hidden.

Confidence: `CONFIRMED_SOURCE`.

## Worker Actions

Observed action names in the researched source snapshot:

- movement: `NORTH`, `SOUTH`, `EAST`, `WEST`;
- idle: `PASS`;
- logistics: `PICKUP`, `DROP`, `PLACE`;
- crops: `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `DIG`;
- structures: `BUILD_COOP`, `BUILD_PASTURE`;
- animals: `FEED`, `CARE`;
- fertilizer: `COLLECT_FERTILIZER`.

Invalid or illegal actions generally become silent no-ops rather than terminating the episode.

Confidence: `CONFIRMED_SOURCE`; exact argument schemas remain to be locked.

## Market Actions

Observed market action names:

- `BUY_SEED`;
- `BUY_PRODUCT`;
- `BUY_ANIMAL`;
- `SELL`;
- `HIRE`;
- `BUY_LAND`.

Confidence: `CONFIRMED_SOURCE`.

## Crop Snapshot

These values were read from the official environment source during pre-repository research and require hash-based revalidation.

| Crop | Seed cost | First/mature yield timing | Recurrence | Maximum yield |
|---|---:|---|---|---:|
| Wheat | 10 | first day 2; max day 4 | none | 6 |
| Carrot | 20 | first day 2; max day 3 | none | 4 |
| Tomato | 50 | first day 8 | every 1 day | 4 |
| Strawberry | 100 | first day 10 | every 2 days | 4 |
| Melon | 80 | first day 10; max day 12 | none | 6 |

Confidence: `CONFIRMED_SOURCE` in the 2026-08-04 snapshot.

Additional crop behavior:

- planting day counts as unwatered;
- two consecutive unwatered daily refreshes turn a plant into weed;
- one-shot crop yield depends on watering during its later growth window;
- fertilizer doubles the relevant yield increment and remains active for the current day plus the next two days;
- mature non-recurring crops decay after their lifespan;
- simultaneous planting requests can be atomic by crop: if requested quantity exceeds owned seeds, the entire same-turn planting group may fail.

Confidence: `CONFIRMED_SOURCE`; atomic planting should be one of the first behavioral regression tests.

## Animal Snapshot

| Animal | Cost | Structure | First yield | Recurrence | Maximum held output | Product |
|---|---:|---|---|---|---:|---|
| Goose | 300 | Coop | day 4 | daily | 4 | Egg |
| Cow | 400 | Pasture | day 8 | every 2 days | 6 | Milk |
| Sheep | 500 | Pasture | day 6 | every 3 days | 6 | Wool |

Additional behavior:

- animals consume wheat feed;
- two consecutive unfed daily refreshes cause escape while leaving the structure;
- care plus feed creates a pending production bonus;
- animals generate collectible fertilizer daily.

Confidence: `CONFIRMED_SOURCE` in the 2026-08-04 snapshot. Exact care-bonus amount must be reverified because prose and source may have differed.

## Labor and Logistics

- hired-hand daily prices follow the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...;
- hired hands disappear at the end of the day;
- workers reset near the central shed at day end;
- carried inventory automatically drops to the shed at day end;
- default shed capacity is 100;
- overflow is discarded;
- seeds are stored separately from shed product capacity;
- movement onto locked tiles is allowed, while tile actions there generally no-op.

Confidence: `CONFIRMED_SOURCE` in pre-repo research.

## Shared Market

Products observed in the market:

- wheat;
- carrot;
- tomato;
- strawberry;
- melon;
- egg;
- milk;
- wool;
- fertilizer.

Researched constants:

- initial target inventory: 10,000;
- price floor: 1;
- base prices: wheat 25, carrot 35, tomato 60, strawberry 120, melon 250, egg 50, milk 160, wool 200, fertilizer 100.

Behavioral notes:

- price follows a product-specific non-linear scarcity/glut curve;
- orders execute one unit at a time in lockstep;
- both players see the same pre-commit unit price during simultaneous execution;
- order sequence and quantity affect results;
- sales at the price floor may not add market supply;
- town shops periodically unlock and consume products;
- town-center demand periodically consumes products and increases later in the match.

Confidence: `CONFIRMED_SOURCE` in the 2026-08-04 snapshot. Exact formulas and town schedules must be copied from the locked source rather than paraphrased.

## Known Recent Engine Drift

### 2026-08-04 shed-capacity enforcement

Pre-repo research found an upstream engine change enforcing shed capacity for market `BUY_PRODUCT` and `BUY_ANIMAL` behavior.

Status: `CONFIRMED_SOURCE` for that upstream commit, but the exact live Kaggle server rollout time is not established.

Required regression test:

- fill shed to capacity;
- attempt product and animal purchases;
- confirm money, inventory, market supply, and partial-fill behavior.

## Randomness and Determinism

Current interpretation:

- most crop, animal, movement, labor, and market mechanics are deterministic given engine version and both action streams;
- some daily events, including weeds and town-shop behavior, appear episode-seed-driven;
- the opponent policy is the dominant strategic uncertainty;
- seeded events should be repeatable under a fixed engine and action sequence.

Confidence: mixed `CONFIRMED_SOURCE` and `UNKNOWN`; the exact random draw schedule must be mapped experimentally.

## Required First Regression Tests

1. 720-turn episode termination and terminal reward.
2. Starting money, worker, land, and positions.
3. Land-purchase costs and unlock order.
4. Market-order count limit.
5. Simultaneous market lockstep and price updates.
6. Atomic planting failure when seeds are insufficient.
7. Watering, weed conversion, maturity, recurrence, and decay.
8. Fertilizer duration and yield effect.
9. Animal feeding, care bonus, output cap, fertilizer, and escape.
10. Day-end hand disappearance, worker reset, and carried-inventory drop.
11. Shed overflow and purchase-capacity behavior.
12. Terminal unsold inventory receiving no reward.
13. Repeatability under identical seed and action streams.
14. Seat-swapped equivalence for symmetric agents.

## Unresolved Questions

- Exact live server engine/package version.
- Exact episode-seed construction and random draw schedule.
- Whether any server configuration differs from repository defaults.
- Exact care bonus in the live engine.
- Exact town shop unlock and consumption schedule.
- Full consequences of price-floor sales.
- Whether additional engine changes landed after the 2026-08-04 research snapshot.
