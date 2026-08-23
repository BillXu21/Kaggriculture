# Kaggriculture Current State

Last updated: 2026-08-23

## Snapshot

- Phase: **Stage-2b slices 1-4 done: worker/ordering/hiring/market cluster, crop/seed/tile lifecycle cluster, animal/structure/fertilizer lifecycle cluster, and town/world/day-RNG/reset/terminal cluster each at zero first divergence vs the real official 1.32.7 engine (27 + 16 + 12 + 10 focused tests); the >16-hired-hands deferral is CLOSED by the MAX_HANDS=240 exact-layout revision (5 real-official hands parity tests green; see below); next gates are the remaining Stage-2b clusters and random/legal-ish full 720-turn traces, then closed-loop A/B; plus issue #2 throughput gates (GIL release, configurable Rayon thread count, batched/multi-core/memory benchmarks — fused executor/day-step explicitly deferred) and a real `best.pt` game through local `kaggle_environments` 1.32.7 (temp venv documented in `oracle/README.md`); issue #4 opening book implemented and officially validated (15/16 strict matrix passes, see below)**.
- Engine/corpus: `kaggle-environments 1.32.7`, canonical replay schema **v3**.
- Training direction: **BC -> closed-loop executor validation -> PPO/RL refinement**.
- Primary goal: build a refinement/self-play pipeline that measurably improves a competent learned starting policy.

## Opening Book (Issue #4)

`opening_book/` provides literal replay-derived elite openings plus the
runtime wrapper and official evaluator:

- Two committed 96-turn identities (days 0-3, d0h0..d3h23), extracted
  deterministically from verified raw replays with full provenance/digests:
  `standard_mixed` (episode 95515912 seat 0, dominant cluster) and
  `pasture_heavy` (episode 95055022 seat 0, ReCurSiON). Handoff at day 4
  hour 0 delegates unchanged to an injected downstream agent.
- Runtime wrapper (`opening_book/agent.py`, `make_opening_agent`) replays
  literal actions under minimal one-way guards (phase cursor, hand
  cardinality, action shape/market cap); any guard failure permanently
  delegates and records deterministic JSON diagnostics.
- Official evaluator (`python -m opening_book.eval`) runs opening-only and
  paired BC-handoff games behind the pinned provenance guard; module
  docstring carries the exact Kaggle command for the real-checkpoint paired
  comparison.
- Validation: 53 focused tests; official 1.32.7 matrix (2 seeds x 2 seats x
  PASS/mirror x both identities) = standard_mixed 8/8, pasture_heavy 7/8
  strict envelope passes. The one failure is environment variance, not code:
  seed 1146601720 seat1 vs PASS spawned WEEDs on the d3h11 strawberry target;
  all 96 turns replayed, zero divergence/anomalies, clean handoff. Strict
  envelope kept by decision; no heuristic weed repair.
- Limitation: real `/kaggle/working/bc-v0-score2950/best.pt` absent locally,
  so paired BC evaluation has not run; no end-to-end BC gain is claimed.

## Differential Oracle (Stage 2a + 2b slice 1)

`oracle/` provides a same-action replay harness: the exact same action pair is
submitted to the pinned official 1.32.7 engine and the fast Rust engine each
turn BEFORE an immediate canonical full-state compare; the run stops at the
first divergent field with seed/step/day/hour/path/values/actions context
(usage + temp-official setup: `oracle/README.md`, decision D-020).

Validated so far: initial-state parity, short legal traces (BUY_SEED/PLANT/
WATER), both-seat privacy comparison, deliberate-corruption first-divergence,
terminal rewards/statuses at `episodeSteps=3`, provenance tamper rejection,
a 28-turn pass-only day-boundary smoke, and — Stage-2b slice 1 — the worker-
inventory / same-turn-ordering / hiring / market cluster at zero divergence
(`tests/test_oracle_mechanics.py`, 27 scenarios; three exact fast-engine
divergences found and fixed: money-decode f32 noise, MAX_QUANTITY=100 order
clamps, ValueError-on-silent-noop wire translation; see `MECHANICS.md`),
and — Stage-2b slice 2 — the crop/seed/tile lifecycle cluster at zero
divergence (`tests/test_oracle_crops.py`, 16 tests / 15 scenarios, 2,136 turn
pairs, 74 day boundaries; one exact fast-engine divergence found and fixed:
`_decay_plants` gated the yield decrement on `> 0` and converted at `== 0`
where the official engine decrements unconditionally and converts at `<= 0`,
letting a zero-yield ongoing crop survive forever; see `MECHANICS.md`),
and — Stage-2b slice 4 — the town/world/day-RNG/reset/terminal cluster at
zero divergence (`tests/test_oracle_town_world.py`, 10 tests / 10 scenarios,
~1,100 turn pairs including one 648-turn PASS-only season segment; shop
unlock timing/duplicate multiplicity/8-instance cap, town + town-center
consumption incl. step-0 fire and negative stock, shared per-day RNG stream
with weed/shop draw ordering, day-boundary reset ordering, terminal
rewards/statuses and no-post-terminal; no engine changes required; see
`MECHANICS.md`).
Former deferral CLOSED (2026-08-23): the fast engine now uses the exact
default-contract hand capacity `MAX_HANDS = maxMarketOrdersPerTurn(10) *
turnsPerDay(24) = 240` — one hand per atomic HIRE order, market queue
truncated to 10 orders/turn, hands cleared at every day reset. Breaking wire
layout: `OBS_SIZE` 5630→8766, `ACTION_SLOTS` 27→251 (market rows moved from
slot 17 to slot 241), `MASK_SIZE` 3562→34026; derivation, offsets, buffer
deltas, and the locked HIRE-mask gate are recorded in `MECHANICS.md`
(MAX_HANDS=240 section) and decision D-021. Evidence:
`tests/test_fast_env.py` (15 tests incl. 23-hand scalar API, hand actions to
all hands, day-reset/rehire Fibonacci restart, mask formula both sides) and
`tests/test_oracle_hands.py` (5 real-official same-action replays: exactly-16
boundary, 17th–23rd crossing, 23-hand hires + subsequent hand actions,
day-end reset from 23 hands + rehire parity, per-turn mask ==
official-reachable gate) — all green. **Remaining mechanic/full-episode parity
is still open Stage-2b work; no full-parity or training-safety claim is made.**

## Learned-Control Contract

The policy is a **once-per-day farm manager**. V0 manager owns:

- crop composition;
- animal target counts;
- land expansion;
- fertilizer allocation by crop type;
- CARE allocation by animal type;
- six-bin daily selling intent.

The deterministic executor owns mechanics:

- exact tile placement and minimum-change crop reconciliation;
- animal structures/placement;
- worker assignment/routing/loading;
- hiring needed for the requested workload;
- watering, feed, harvest, collection, and other routine maintenance;
- seed/feed/fertilizer/animal purchases mechanically implied by the plan;
- exact fertilizer/CARE targets;
- primitive sell execution.

The executor is a strategy compiler under D-011, not a hidden economic policy.

## Canonical Corpus

Five elite post-patch 1.32.7 partitions, 2026-08-17 through 2026-08-21:

- 3,486 episodes;
- 6,972 seat trajectories;
- 209,160 `(episode, seat, day)` rows;
- schema versions exactly `{3}`.

Important coordinate contract: simulator worker positions are `[x,y]`; canonical board/event coordinates are `[y,x]`; tile lookup is `tiles[y][x]` after unpacking `x,y`.

Private Kaggle dataset mount used for BC:

`/kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327`

See `research/FIVE_DAY_V3_CORPUS.md`.

## First BC V0 Result

Reference model: D-019 default tile Transformer, 1,071,040 parameters.

Reference run:

- code at run start: `692bca50e8ba0b687e48fd970e67bbe17014f03f`;
- train Aug17-20, validation Aug21;
- `min_score >= 2950`;
- 25,500 train rows, 5,700 validation rows;
- CUDA + AMP, batch 256, AdamW 3e-4;
- 30 epochs, ~237 s;
- best epoch 29;
- best validation total **2.8889**.

Held-out Aug21 model vs train-only day baseline:

| metric | model | day baseline |
| --- | ---: | ---: |
| crop exact accuracy | **0.7128** | 0.4752 |
| crop MAE | **1.2731** | 3.6217 |
| animal exact accuracy | **0.8267** | 0.4540 |
| animal MAE | **0.2681** | 1.6936 |
| fertilizer nonzero recall | **0.7522** | 0.4557 |
| CARE whole-vector exact | **0.5998** | 0.1754 |
| land accuracy | **0.9912** | 0.9089 |
| sell presence accuracy | **0.9394** | 0.8923 |

Rare state-conditioned branches also work materially better than the calendar baseline: tomato nonzero recall 83.8%, goose 96.8%, goose CARE 95.5%, wheat fertilizer 60.4%.

Selling remains the clearest teacher-forced weakness: true positive rate 11.21%, predicted 9.38%, positive recall 64.84%.

Conclusion: **D-019 passes its intended representation diagnostic.** Do not spend the next cycle on model scaling/tuning before closed-loop evidence.

Detailed run/eval: `research/FIRST_BC_V0_EVAL.md`.

## BC V0 Simplification Backlog

The BC design did already record many deferrals in D-019 and its implementation note, but they were spread across several files. `research/FIRST_BC_V0_EVAL.md` now contains one explicit backlog of V0 shortcuts and revisit triggers, including:

- stateless once-per-day policy;
- opponent-public board disabled;
- no opponent-private inference;
- absolute counts rather than deltas/per-tile targets;
- type-level fertilizer/CARE rather than tile-specific control;
- factorized heads with no joint feasibility model;
- six-bin selling and fixed 0.5 presence threshold;
- equal loss weights and no sparse reweighting;
- one ~1.07M architecture with no sweep/scheduler;
- five-day `>=2950` elite corpus with no dedup/family weighting;
- one held-out date rather than a broad rolling/generalization suite;
- no DAgger/on-policy correction, value head, PPO, or uncertainty-aware execution.

Known weak points such as rare fertilizer recall and conservative selling remain explicitly listed so they are not forgotten after the first closed-loop success.

## Deterministic Executor V0

**Implemented (issue #1, commits `11e85fa`..`ed1685a`).** The complete
closed loop lives in `executor_v0/` (usage: `executor_v0/README.md`):

live schema-v3 observation encoding (exact BC adapter parity) ->
once-per-day manager (`CheckpointPlanProvider` or injected fake) ->
immutable `DailyPlan` -> mechanical feasibility projection ->
deterministic animal layout / minimum-change crop reconciliation ->
per-turn explicit task generation -> greedy foreman dispatch ->
hour-0 crude hiring, exact-shortage purchases, BUY_LAND, six-bin sells ->
legal-shaped `{"farmer", "hands", "market"}` action dict plus JSON
diagnostics (requested/feasible/achieved/submitted/observed) and a
deterministic all-PASS safe-mode fallback.

Key mechanic: `PLANT <crop>` consumes the global own `private.seeds[crop]`
pool atomically at the engine; seeds are never picked up or carried, and the
foreman reserves global seeds per crop within each turn.

Validation so far: 249 tests pass; live encoder parity (synthetic + real);
determinism coverage; a 719/720-turn replay-observation plumbing smoke
(shape/state robustness only — counterfactual actions were never executed by
the engine). A real 1.32.7 game has NOT been run because
`kaggle_environments` is not installed here; the smoke harness skip path
(exit 3) is verified.

The current first-draft algorithm is recorded in `research/EXECUTOR_V0_PLAN.md`.

Core design:

1. Run the BC manager at hour 0 and preserve `requested -> feasible -> achieved` separately.
2. Do **not** reserve near-shed tiles for future animals. Use productive crop space until animal targets actually require conversion.
3. When making room for livestock, prefer legal empty tiles or the **least-invested nearby crop**: young/unfertilized plants are cheap to sacrifice; mature/fertilized/near-harvest crops are expensive.
4. Keep established livestock infrastructure sticky; otherwise reconcile crop targets with minimum destruction/relayout.
5. Generate explicit maintenance/transition/logistics tasks and revalidate them from the live observation every primitive turn.
6. Use broad urgency/deadline tiers rather than a large hand-tuned strategic priority table.
7. Use soft worker specialization from carried inventory. Prefer workers needing few item types; planned bundles should normally stay around <=2 types.
8. Greedily load and dispatch workers. Never hold a ready worker at the shed merely to synchronize loading.
9. Hour 0: farmer may take one useful bulk pickup; market stages supplies and hires hands. New hands act from hour 1 and begin greedy loading/dispatch.
10. Recompute a simple deterministic greedy worker-task assignment every turn using Manhattan distance, shed detours, and inventory affinity.
11. Purchase only shortages mechanically implied by the manager plan/maintenance; do not add a separate economic strategy.
12. Execute six-bin sells literally as inventory becomes available within the requested bin.

## Executor V0 Simplification Backlog

`research/EXECUTOR_V0_PLAN.md` explicitly marks these as temporary shortcuts rather than settled optimal choices:

- no prediction/reservation for future animal onset;
- crude crop-sacrifice value rather than exact marginal farm value;
- no global layout/facility-location optimization;
- greedy worker matching rather than Hungarian/min-cost-flow/search/VRP;
- one-step Manhattan routing and no multi-turn route search;
- soft inventory specialization rather than explicit optimized worker roles;
- greedy shed staging rather than inventory-buffer optimization;
- crude workload-based hiring;
- basic wheat/feed procurement;
- literal six-bin selling with no within-bin price reaction;
- no opponent-aware executor economics;
- no confidence-aware plan projection;
- no search-based foreman yet.

Search/assignment/layout sophistication is a **later performance option**, not part of the first closed-loop build.

Revisit these when traces show low intent compliance, missed maintenance despite sufficient labor, excessive movement/wait/reloading, shed stalls, destructive crop conversions, worker assignment thrashing, or high compliance but losses traceable to executor-owned simplifications.

## Closed-Loop Success Criteria

The first executor is successful when it can:

- encode live observations with the same semantics as BC training;
- make all 30 manager calls;
- finish real 1.32.7 games without illegal-action cascades/deadlock;
- track requested/feasible/achieved crop, animal, land, fertilizer, CARE, and selling;
- report missed maintenance, pending tasks, movement/pickup/idle actions, and emergency purchases;
- run a small fixed-seed, seat-swapped panel against one frozen competent opponent.

If compliance is poor, improve execution before blaming BC. If compliance is high and economic trajectories are poor, revisit the manager/action abstraction before adding PPO complexity.

## Near-Term Sequence

1. ~~Turn `research/EXECUTOR_V0_PLAN.md` into a bounded implementation contract and build the smallest complete executor.~~ Done (issue #1).
2. Get the current `best.pt` through full local 1.32.7 games (`python -m executor_v0.smoke --manager checkpoint --checkpoint best.pt`).
3. Inspect compliance before optimizing score.
4. Run a small paired fixed-seed/seat-swapped frozen-opponent panel.
5. Use evidence to choose the next bottleneck: executor search/layout, BC refinement/data, opponent inputs, or PPO.
6. Only after the stationary closed-loop problem is reliable, expand to broader opponent panels and changing-population/self-play.

## Do Not Forget

Before substantial work read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, relevant implemented notes, `research/FIRST_BC_V0_EVAL.md`, and `research/EXECUTOR_V0_PLAN.md`.

Before expensive runs record exact code/configuration, engine identity, data/version/filter, seeds/opponents, outputs, stop conditions, and recovery plan.
