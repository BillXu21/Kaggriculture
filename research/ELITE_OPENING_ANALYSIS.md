# Elite Opening Analysis — Replayable Opening Book Design (Issue #3)

Date: 2026-08-23
Status: research/design only — no runtime policy changes.
Engine contract: `kaggle-environments==1.32.7`, upstream commit
`28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`, starting money 3000, 30 days x 24 turns.

## 1. Method, data, and reproducibility

**Cohort selection rule (reproducible).** From the canonical schema-v3 sample
`data/canonical/2026-08-20-sample.parquet` (15 episodes x 30 days x 2 seats,
schema versions exactly `{3}`), read the `metadata` column **only** (Arrow
projection; no corpus-wide `to_pylist()`), group by episode, and rank by
descending `min(final_bank_seat0, final_bank_seat1)`. The top 10 local episodes
were selected and combined with remote episode `95531759`, fetched fully
in memory via the configured Kaggle API (`KaggleApi.build_kaggle_client()` ->
`competition_api_client.get_episode_replay`) and **never written to disk**.
This resolves the earlier availability ambiguity: episode `95515912` IS present
locally (Parquet row + raw replay `data/samples/2026-08-20/95515912.json`) and
is the strongest local episode.

**Analysis tooling.** `research/analyze_elite_openings.py` (committed with this
note) performs selection, per-replay canonical extraction via
`replay_daily.extractor.extract_replay`, day/hour summaries, market-signature
clustering, exact hour traces, and internal assertions. Raw ~30 MB replays are
loaded one at a time and released; the full corpus is never materialized.

```
python research\analyze_elite_openings.py --trace 0:0:1 --trace 1:0:1
# -> exit 0, "ALL ASSERTIONS PASSED", 11 episodes analyzed
```

**Evidence sources cited below**

- `data/canonical/2026-08-20-sample.parquet` (canonical daily records, schema v3)
- `data/samples/2026-08-20/<episode>.json` raw 720-turn replays
- in-memory replay of episode 95531759 (module_version 1.32.7 verified)
- `replay_daily/extractor.py` (action alignment: `steps[i][seat].action`
  transforms observation `i-1` into `i`; events attributed to `(day, hour)` of
  observation `i-1`)
- `replay_daily/constants.py` (engine CROPS/ANIMALS/LAND_PRICES tables, Fibonacci
  hire costs — exact copies of upstream 1.32.7 constants)
- `MECHANICS.md` sections "Shared Market" and "Fast-Engine Differential Parity"
  (as committed at HEAD `519830c`)
- `DECISIONS.md` D-017/D-018/D-020

Observed facts are labeled **[OBS]**; design recommendations are labeled
**[REC]**. All hour-level numbers come from the ordered event ledger and hourly
observations of the listed replays.

## 2. Selected replay table

Selection: top 10 local by descending `min(bank_seat0, bank_seat1)`, plus
remote 95531759. Opening horizon inspected: days 0–3 for every seat, plus land/
harvest milestones through day 12 from canonical records.

| Episode | Seed | Players (seat0 / seat1) | Final banks | Source |
|---|---|---|---|---|
| 95515912 | 1146601720 | Eddy Despradel / Victor @ Tufa Labs | 136216 / 137522 | local raw |
| 95055022 | 1979016230 | ReCurSiON / Izzoudine Mohamed KANTA | 110665 / 106921 | local raw |
| 94735084 | 1542433895 | Michael Timbs / lucaskna | 103305 / 106327 | local raw |
| 94956733 | 1462732054 | Eddy Despradel / Laplacenvmv | 89821 / 91551 | local raw |
| 95481731 | 1463130186 | ReCurSiON / Izzoudine Mohamed KANTA | 97195 / 89465 | local raw |
| 95415654 | 1450934916 | peikopon / Xiaowenhao404 | 89458 / 87773 | local raw |
| 95552453 | 1686596305 | Xiaowenhao404 / peikopon | 87246 / 89269 | local raw |
| 95004769 | 1210496324 | lucaskna / James Holland | 85755 / 85864 | local raw |
| 95294795 | 1209746757 | Laplacenvmv / lucaskna | 77290 / 78182 | local raw |
| 95214822 | 1366807384 | Suneeth reddy / James Holland | 73497 / 70106 | local raw |
| 95531759 | 1533196586 | SaiKushal185 / Efe Can Celiksoy | 159781 / 161160 | remote, in-memory |

All 11 replays report `module_version == 1.32.7`; all selected local metadata
(seed, banks) matches the raw replay bodies (asserted by the script).
Both seats are strong in every selected episode by construction of the
selection score.

## 3. Cross-replay similarity (days 0–3)

Market-signature clustering over all 22 seat-openings (ordered market orders
with hours; worker movement excluded):

| Day | Dominant cluster support | Dominant signature (ordered, with hours) |
|---|---|---|
| 0 | **17 / 22** | `HIRE x5 @h0; BUY_ANIMAL COW 2 @h0; BUY_ANIMAL SHEEP 2 @h0; BUY_SEED WHEAT 7 @h0; BUY_SEED MELON 12 @h0; BUY_PRODUCT WHEAT 6 @h0; SELL WHEAT 3 @h1; BUY_PRODUCT WHEAT 2 @h6; BUY_PRODUCT WHEAT 1 @h12` — **hour-level identical across all 17** |
| 1 | 16 / 22 | `SELL FERTILIZER 3..4 @h18; BUY_PRODUCT WHEAT 5 @h18` (15/17 sell exactly 3) |
| 2 | 20 / 22 | `HIRE x4 @h0; BUY_PRODUCT WHEAT 2 @h2; BUY_PRODUCT WHEAT 2 @h8` |
| 3 | 15–18 / 22 | `SELL FERTILIZER 4 @h0; BUY_ANIMAL COW 1; HIRE x5` (14/17 order COW before hires; 3/17 hire first, then COW + `BUY_SEED CARROT 1` @h1) |

The two non-dominant seat-openings are ReCurSiON seat 0 in episodes 95055022
and 95481731 (Section 5), plus one reordered micro-variant (95515912 seat 1:
same quantities, MELON seeds bought before WHEAT seeds, resale before the
WHEAT seed buy).

**Determinism finding [OBS].** These elite agents are deterministic scripted
bots:

- ReCurSiON seat 0 plays a byte-identical opening (same orders, same money path
  3000 -> 36 -> 35 -> 195 -> 211/213) in two different episodes with different
  seeds and opponents.
- The dominant day-0 market sequence is hour-identical across 17 seat-openings
  played by different players on different seeds; even worker movement-op counts
  match across seeds (e.g. seat-0 `WEST 22 / NORTH 22` in both 94735084 and
  95531759), indicating seed-independent farm layouts.
- Money paths differ only slightly under the identical script
  (day-0 end 17..23; day-1 end ~175..271; day-2 end 48..142; day-3 end 10..120)
  because product purchase costs depend on live market inventory, which is
  perturbed by town-center consumption and the opponent.

**First material divergence [OBS].** Within the dominant family the first
divergence is policy micro-timing, not environment-forced: day 1 hour 18
(`SELL FERTILIZER 3` vs `4`, i.e. collecting/selling one more fertilizer unit),
and day 3 (`BUY_ANIMAL COW` before vs after the five hires). Shop/town unlock
occurs at day 3 hour 0 in every episode (one seed-dependent shop:
PIZZA_SHOP, BAKERY, SMOOTHIE_SHOP, ICE_CREAM_SHOP, YARN_STORE, BRUNCH_SPOT,
PET_CAFE) and **no elite script changes behavior at or after unlock within the
opening window**. Opponent-driven market effects appear only as ±$1..$150 cash
noise around identical orders. Event-level limits: canonical daily records do
not store per-unit realized prices, so per-order price effects were derived from
hourly market observations in the raw replays.

## 4. Archetype A — "standard mixed farm" (dominant, support 17/22)

Defining emphasis: 7 WHEAT + 12 MELON crops, 2 COW + 2 SHEEP on day 0; heavy
day-0 labor burst; fertilizer collected and sold from day 1 to finance just-in-time
(JIT) wheat feed stock; cow #3 on day 3; no land purchase before day 6.

### 4.1 Day/hour summary (representative: episode 95531759, both seats identical)

| Day | Money start->end | Hires (submitted/realized/cost) | Market orders (hour) | Production | Notes |
|---|---|---|---|---|---|
| 0 | 3000 -> 23 (min 22) | 5/5/12 | h0: HIRE x5, COW 2, SHEEP 2, WHEAT seed 7, MELON seed 12, BUY_PRODUCT WHEAT 6; h1: SELL WHEAT 3 (+84); h6: BUY WHEAT 2 (-55); h12: BUY WHEAT 1 (-28) | plant WHEAT 7 + MELON 12; BUILD_PASTURE x4; PLACE x4; WATER 19; FEED 2; CARE SHEEP 2 | hands act from h1 (hired during h0 processing); min cash $22 at h0 end |
| 1 | 23 -> 176 | 0/0/0 | h18: SELL FERTILIZER 3; h18: BUY_PRODUCT WHEAT 5 | COLLECT_FERTILIZER x3; FEED 4; CARE COW 2 + SHEEP 2 | hands reset to 0 overnight [OBS]; light farmer-only day until fertilizer sale finances feed restock; net +153 |
| 2 | 176 -> 49 | 4/4/7 | h0: HIRE x4; h2: BUY WHEAT 2; h8: BUY WHEAT 2 | WATER 19; FEED 4; COLLECT_FERTILIZER x4; BUILD_PASTURE 1 | re-hire each morning as needed |
| 3 | 49 -> 30 | 5/5/12 | h0: SELL FERTILIZER 4; h0: BUY_ANIMAL COW 1; h0-h1: HIRE x5 | WATER 14; FEED 4; PLACE 1 | shop unlocks d3h0 — script unchanged |

Milestones beyond the window [OBS, canonical records]: first WHEAT harvest
d4 h7 (~6 yield/unit at `max_yield_day=4`, not at `first_yield_day=2`);
first WOOL d6 h2; land NE purchased d6 h4 ($1000, wool-financed); first MILK
d8 h3; first MELON d10 h6 (at `first_yield_day=10`); land SW d11 h0 ($2000).

### 4.2 Exact hour trace (d0 h1, episode 95531759 seat 1) [OBS]

```json
{"seat":1,"day":0,"hour":1,"money_before":22.0,"money_after":106.0,
 "market_inventory_wheat":9989,"market_price_wheat":28,
 "action":{"farmer":["PICKUP","SHEEP",1],
           "hands":[["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"]],
           "market":[["SELL","WHEAT",3]]}}
```

Day-0 h0 context (same replay, seat 0): money 3000 -> 22 across the h0 order
batch; market WHEAT inventory 10000 -> 9989 and price 25 -> 28 by the h1
observation (6 units bought by the player; town center consumed the rest).

### 4.3 Primitive-action skeleton (playback form) [REC]

Market orders are literal and hour-indexed. Worker ops are expressed as the
observed per-day workload bundles (tile-exact placement is executor-owned
mechanics per D-011/D-018):

```
d0 h0: market = [HIRE x5, BUY_ANIMAL COW 2, BUY_ANIMAL SHEEP 2,
                 BUY_SEED WHEAT 7, BUY_SEED MELON 12, BUY_PRODUCT WHEAT 6]
d0 h1: market = [SELL WHEAT 3]; workers: pickup animals, build 4 pastures,
       place 4 animals, plant 7 WHEAT + 12 MELON, water all 19, feed, CARE sheep
d0 h6: market = [BUY_PRODUCT WHEAT 2]
d0 h12: market = [BUY_PRODUCT WHEAT 1]
d1 h18: market = [SELL FERTILIZER 3, BUY_PRODUCT WHEAT 5]
        workers: collect fertilizer x3, feed 4, CARE cows+sheep (no hires needed)
d2 h0: market = [HIRE x4]; h2: [BUY WHEAT 2]; h8: [BUY WHEAT 2]
        workers: water 19, feed 4, collect fertilizer x4, build pasture 1
d3 h0: market = [SELL FERTILIZER 4, BUY_ANIMAL COW 1, HIRE x5]
        workers: place cow, water 14, feed 4
handoff at d4 h0
```

Expected handoff state (day 3 end): money 10..120; crops exactly 7 WHEAT +
12 MELON planted and watered; animals 3 COW + 2 SHEEP placed and fed;
shed holds ~5-6 WHEAT feed stock; 1 quadrant unlocked; no harvest pending
before d4 h7.

Confidence: high (17/22 seat-openings, hour-identical market play, two
independent episodes confirming determinism).

## 5. Archetype B — "pasture-heavy" (ReCurSiON, support 2/22)

Preserved as a genuinely distinct champion identity [OBS]: ReCurSiON seat 0 in
episodes 95055022 and 95481731 plays an identical, materially different script.

Defining emphasis: fewer crops (5 WHEAT + 5 MELON), more animals earlier
(1 COW + 4 SHEEP, 5 pastures), feed-first wheat buying (8 units before any
hire), near-idle day 1, larger fertilizer sales (5/day), strawberry
diversification on day 3.

| Day | Money start->end | Hires | Market orders (hour) | Production |
|---|---|---|---|---|
| 0 | 3000 -> 36 (min 7) | 4/4/7 | h0: BUY_PRODUCT WHEAT 8 **first**, then HIRE x4, COW 1, SHEEP 4, MELON seed 5, WHEAT seed 5; h13: SELL WHEAT 1, MELON seed 1 | plant 5+5; BUILD_PASTURE x5; PLACE x5; FEED 5; WATER 10; CARE COW 1/SHEEP 4 |
| 1 | 36 -> 35 | 1/1/1 | h0: HIRE 1 | COLLECT_FERTILIZER x5 only — deliberate near-idle day |
| 2 | 35 -> 195 | 2/2/2 | h0: SELL FERTILIZER 5, HIRE x2, BUY WHEAT 6; then 1 unit each @h3,h5,h9,h14,h18 | FEED 5; WATER 10; COLLECT_FERTILIZER x5 |
| 3 | 195 -> 211..213 | 3/3/4 | h0: SELL FERTILIZER 5, HIRE x3, WHEAT seed 1, STRAWBERRY seed 3; then wheat singles @h3..h16 | plant STRAWBERRY 3 + WHEAT 1; FEED 5 |

Milestones: first WHEAT harvest d4 h8; first WOOL d6 h0; land NE d6 h16;
land SW d10 h0. Confidence: medium-high (2 seat-openings, both top-10 banks,
byte-identical across seeds).

## 6. The episode-95531759 day-0 WHEAT trade, resolved [OBS]

Ordered trace (both seats, identical; raw replay, hourly observations):

1. d0 h0: `BUY_PRODUCT WHEAT 6` inside the h0 batch. Total h0 spend
   3000 -> 22 (=2978: hires 12 + COW 800 + SHEEP 1000 + seeds 1030 + wheat
   ~136 effective). Market WHEAT inventory 10000, price 25 pre-trade; 9989/28
   at the h1 observation (player bought 6; town center consumed ~5 during
   processing).
2. d0 h1: `SELL WHEAT 3` -> money 22 -> 106 (**+84**, exactly 3 x 28).
3. d0 h6: `BUY_PRODUCT WHEAT 2` -> 106 -> 51 (-55 = 27 + 28, per-unit lockstep
   pricing with post-buy inventory increments).
4. d0 h12: `BUY_PRODUCT WHEAT 1` -> 51 -> 23 (-28).
5. Day-end shed holds ~6 WHEAT used as animal feed stock; FEED ops occur d0
   (x2) and daily thereafter; d1 h18 `BUY_PRODUCT WHEAT 5` restocks feed after
   the fertilizer sale finances it.

Economics: total wheat-related outflow 219 for 9 units bought; inflow 84 for 3
resold; net -135 for 6 retained feed units (~22.5/unit, at or slightly below
the 25 base price). The resale is **logistics recovery of an intentionally
oversized feed purchase, not profitable arbitrage**: `MECHANICS.md`
("Fast-Engine Differential Parity", committed at `519830c`) states a same-turn
BUY/SELL round trip nets exactly zero because buy quotes use post-buy inventory
pricing — the player's own purchase moves the price against itself. Recurrence:
the `BUY WHEAT 6 @h0 / SELL WHEAT 3 @h1` pair appears in **all 17 dominant
cluster seat-openings** (and the pasture variant buys 8 and resells 1), so it
is a consistent elite habit — but no evidence suggests it is a material profit
source. **[REC]** V0 may reproduce it literally (it is harmless and universal)
but must not treat it as a required profit step; skipping the resale and buying
only the feed need is an acceptable simplification.

No impossible day-0 harvest interpretation exists: zero HARVEST events occur on
day 0 in any selected seat-opening (asserted); the first WHEAT harvest is d4,
consistent with harvesting at `max_yield_day=4` for maximum yield rather than
`first_yield_day=2`.

## 7. Handoff trigger analysis [REC]

| Candidate | Evidence | Assessment |
|---|---|---|
| Fixed end-of-day-3 (resume d4 h0) | Scripts are deterministic through d3; state at d3 end is uniform (Section 4.3); nothing happens d4 h0–h6 | **Recommended.** Simplest, fully covered by traces, resumes before the first harvest |
| Shop-unlock event (d3 h0) | Unlock occurs in every episode; **no elite script reacts** within the window | Earlier but buys nothing; would force the manager to finish d3 watering for no gain |
| First-harvest-ready (d4 h7) | First WHEAT harvest d4 h7 (dominant) / d4 h8 (pasture) | Event timing is crop-state-dependent; opener would idle 7 hours; rejected |
| State-distribution match | Elite d3-end states are tight ranges | Redundant on top of fixed-day + guards; defer |

**Recommendation:** the opening book owns days 0–3 (hours 0..71) and hands off
at day 4 hour 0, before the d4 h7 first wheat harvest. The general manager +
executor therefore owns all harvesting, the d6 wool-financed land purchase, and
everything after.

## 8. Minimal V0 opening-book contract [REC]

Deterministic primitive playback of one archetype skeleton (Sections 4.3/5),
indexed by `(day, hour)`, bypassing the BC manager during the window, with only
these guards:

1. **Phase sync:** assert observation `(day, hour)` equals the script cursor;
   on mismatch, advance-or-fail loudly (no silent reindexing).
2. **Cash/market guard:** before each market batch, require
   `money >= estimated batch cost + tolerance` (observed same-script variance
   is small but nonzero: day-0 end 17..23). `BUY_PRODUCT`/`BUY_ANIMAL` execute
   per-unit lockstep; accept partial fills only for the speculative WHEAT
   oversize, never for seeds/animals/hires.
3. **Preconditions:** seeds pool sufficient before PLANT; pasture structure
   built before PLACE; hands present before assigning hand ops (hands reset
   every day — morning hires are mandatory when the script says so); shed
   capacity before PICKUP.
4. **Validity/idempotency:** each scripted order submitted at most once
   (cursor-monotonic); illegal/rejected order => stop-and-handoff, not retry.
5. **Divergence detection & fallback:** checkpoint money/crop-count/animal-count
   at each day boundary against Section 4.3 ranges; outside range (e.g.
   opponent-driven market shock, rejected order) => immediately hand off to the
   general manager/executor with the current state (its normal midgame input),
   or all-PASS safe mode if that fails.

Deliberately excluded: heuristic planning, price-reactive selling, opponent
modeling, routing optimization (see Section 10).

## 9. Champion-panel recommendation [REC]

Preserve exactly two opening identities (genuinely distinct, both evidenced):

- **Identity A "standard-mixed"** (dominant cluster, 17/22): the Section 4
  skeleton. Default identity.
- **Identity B "pasture-heavy"** (ReCurSiON, 2/22): the Section 5 skeleton.
  Strategic variance: earlier/larger animal economy, strawberry branch,
  different risk profile (min cash $7).

The day-1/day-3 micro-timing variants (fertilizer 3 vs 4, COW-before-hires) are
cosmetic duplicates of Identity A and should not become separate champions.

## 10. Deferred micro-optimizations / non-goals for V0

- **Wheat buy/resell trick** (Section 6): reproduce literally if convenient;
  never required as a profit step. No repeat/economic evidence of material gain.
- **Opponent-aware market timing:** no elite opening branch is
  opponent-triggered within days 0–3; effects observed are ±$ noise.
- **Globally optimal routing / search:** elite movement is simple repeated
  shuttle patterns; copying tile-exact paths is optional, not strategic.
- **Shop-unlock adaptation:** no evidence of elite use inside the window.
- **General executor optimization:** out of scope here (issue #1 artifact).

## 11. Limitations

- Official-engine replay validation of extracted traces was NOT run
  (`kaggle_environments` is not installed in this environment; oracle Stage-2a
  covers short traces only). Claims about mechanics cite committed source
  provenance (`replay_daily/constants.py`, `MECHANICS.md` at `519830c`) and
  observed replay behavior; no official-engine execution claim is made.
- Cohort is drawn from one local partition sample (2026-08-20) plus one remote
  episode; the selection rule is reproducible but the cohort is a sample, not
  the full five-day corpus.
- Canonical daily records do not retain per-unit realized prices; wheat-trade
  economics were reconstructed from hourly raw-replay observations.
- During this analysis, unrelated concurrent work was committed at HEAD
  `519830c` (`fast_env/api.py`, Rust core, oracle tests). Those files were not
  used as evidence; the committed `MECHANICS.md` additions strengthen the
  market-mechanics citation.
