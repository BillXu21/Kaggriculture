# Kaggriculture Mechanics Ledger

Last updated: 2026-08-07

## Purpose

This file separates verified engine behavior from documentation, discussion claims, and stale assumptions.

Confidence labels:

- `CONFIRMED_SOURCE`: observed in a specific recorded engine source snapshot.
- `CONFIRMED_EXPERIMENT`: reproduced with a controlled behavioral test.
- `DISCUSSION_CLAIM`: reported publicly but not independently verified.
- `OUTDATED`: known to describe an older engine or contract.
- `UNKNOWN`: unresolved or insufficiently specified.

## Engine Identity

- Latest confirmed upstream package version: `1.32.6`
- Upstream 1.32.6 source snapshot found at commit: `bded87b0d7879078c726a93a4884d044f79c4eed`
- Town-rebalance PR: `Kaggle/kaggle-environments#1394`
- Town-rebalance PR merge commit: `1fa13d78387eb3661b1e621a4f5df150e6c3b646`
- Local package version: not installed/locked in this repository yet
- Vendored source commit: not established
- Engine file SHA-256: not established
- Specification file SHA-256: not established
- Live Kaggle leaderboard server version: 1.32.6 rollout announced, not independently server-verified
- Last source research snapshot: 2026-08-07
- Status: **source version is known, but do not treat this repository as a complete engine lock until source/spec hashes and local behavioral tests are recorded**

## Match Contract

| Mechanic | Current understanding | Confidence |
|---|---|---|
| Players | 2 | `CONFIRMED_SOURCE` |
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
- town shops, including duplicate shop names under 1.32.6;
- current day and hour.

Confidence: `CONFIRMED_SOURCE`.

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

| Crop | Seed cost | First/mature yield timing | Recurrence | Maximum yield |
|---|---:|---|---|---:|
| Wheat | 10 | first day 2; max day 4 | none | 6 |
| Carrot | 20 | day 2; max day 3 | none | 4 |
| Tomato | 50 | first day 8 | every 1 day | 4 |
| Strawberry | 100 | first day 10 | every 2 days | 4 |
| Melon | 80 | first day 10; max day 12 | none | 6 |

Confidence: `CONFIRMED_SOURCE` in current/recent official source; still rehash exact file before implementation.

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

Confidence: `CONFIRMED_SOURCE`. Exact care-bonus amount must still be reverified because older prose and source may have differed.

## Labor and Logistics

- hired-hand daily prices follow the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...;
- hired hands disappear at the end of the day;
- workers reset near the central shed at day end;
- carried inventory automatically drops to the shed at day end;
- default shed capacity is 100;
- overflow is discarded;
- seeds are stored separately from shed product capacity;
- movement onto locked tiles is allowed, while tile actions there generally no-op.

Confidence: `CONFIRMED_SOURCE`.

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
- lower town demand in 1.32.6 means player sell pressure should persist more strongly in inventory/prices.

Confidence: `CONFIRMED_SOURCE`; exact formulas should be copied from the locked source rather than paraphrased when implementation starts.

## Town Demand — 1.32.6 Current Contract

### Town center

`CONFIRMED_SOURCE` from merged PR #1394:

- `townCenterSellInterval` default changed from 12 to 24 turns;
- with 24 turns/day, town center consumes once per day;
- each tick removes one of every non-fertilizer product;
- demand is flat for the whole season;
- the old `TOWN_CENTER_DEMAND_SCHEDULE` with 2× after day 10 and 4× after day 20 was removed.

The previous ledger statement that town-center demand increases later in the match is now `OUTDATED`.

### Town shops

`CONFIRMED_SOURCE` from merged PR #1394:

- shop unlock interval remains unchanged (default every 3 days);
- each unlock samples uniformly from the full shop table **with replacement**;
- duplicate shop names can therefore appear in `town.unlocked_shops`;
- every duplicate instance consumes independently;
- unlocking stops at 8 total shop instances (`MAX_SHOP_INSTANCES = 8`);
- shop consumption cadence remains unchanged (default every 4 turns);
- single-product shops continue to consume at their existing 2× per-tick rule.

Consequences for modeling:

- town state must be encoded as a multiset/count vector, not a binary set;
- future shop composition is stochastic even after early unlocks are observed;
- duplicated demand can make some products much more attractive in particular episodes;
- weaker town-center cleanup increases the strategic effect of player-generated gluts.

## Known Recent Engine Drift

### 2026-08-04 shed-capacity enforcement

Upstream engine change enforced shed capacity for market `BUY_PRODUCT` and `BUY_ANIMAL` behavior.

Status: `CONFIRMED_SOURCE`; exact server rollout timing should still be tied to a package/server snapshot when tests begin.

Required regression test:

- fill shed to capacity;
- attempt product and animal purchases;
- confirm money, inventory, market supply, and partial-fill behavior.

### 2026-08-07 / package 1.32.6 town rebalance

Merged upstream PR #1394:

- town center 2 ticks/day → 1 tick/day;
- town center late-game multipliers removed;
- shops sampled with replacement;
- duplicate shop instances consume independently;
- maximum total shop instances remains 8.

Package source snapshot `bded87b0d7879078c726a93a4884d044f79c4eed` identifies itself as `1.32.6`.

Required regression tests:

- verify exactly one town-center tick/day at defaults;
- verify flat demand on days 0–29;
- force/observe duplicate shops across seeds;
- verify duplicated instances multiply demand independently;
- verify no more than 8 shop instances unlock;
- verify deterministic replay for identical seed and action streams.

## Randomness and Determinism

Current interpretation:

- most crop, animal, movement, labor, and market mechanics are deterministic given engine version and both action streams;
- weeds are seed-driven stochastic events;
- town-shop draws are seed-driven and, from 1.32.6 onward, sampled with replacement;
- the current shop multiset is public, but future shop draws remain unknown;
- duplicated shop draws increase episode-to-episode economic variance;
- the opponent policy remains the main strategic uncertainty beyond engine RNG;
- seeded events should be repeatable under a fixed engine and action sequence.

Confidence: mostly `CONFIRMED_SOURCE`; exact RNG draw ordering should still be mapped experimentally because changes in draw order can affect seed-level reproducibility across versions.

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
15. Town center consumes exactly once/day at flat 1× under 1.32.6 defaults.
16. Duplicate shop sampling with replacement and independent duplicate demand.
17. Eight-instance town-shop cap.

## Unresolved Questions

- Exact live leaderboard server version at any given time during rollout.
- Exact episode-seed construction and RNG draw schedule.
- Whether any server configuration differs from repository defaults.
- Exact care bonus in the live engine.
- Full consequences of price-floor sales.
- Whether additional engine changes land after 1.32.6.
- Magnitude of win/bank variance introduced by shop replacement sampling.
