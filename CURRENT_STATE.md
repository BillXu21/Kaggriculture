# Kaggriculture Current State

Last updated: 2026-08-22

## Snapshot

- Phase: **first BC manager is trained and passes held-out/day-baseline diagnostics; next gate is the smallest complete deterministic executor and real closed-loop games**.
- Engine/corpus: `kaggle-environments 1.32.7`, canonical replay schema **v3**.
- Training direction: **BC -> closed-loop executor validation -> PPO/RL refinement**.
- Primary goal: build a refinement/self-play pipeline that measurably improves a competent learned starting policy.

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

1. Turn `research/EXECUTOR_V0_PLAN.md` into a bounded implementation contract and build the smallest complete executor.
2. Get the current `best.pt` through full local 1.32.7 games.
3. Inspect compliance before optimizing score.
4. Run a small paired fixed-seed/seat-swapped frozen-opponent panel.
5. Use evidence to choose the next bottleneck: executor search/layout, BC refinement/data, opponent inputs, or PPO.
6. Only after the stationary closed-loop problem is reliable, expand to broader opponent panels and changing-population/self-play.

## Do Not Forget

Before substantial work read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, relevant implemented notes, `research/FIRST_BC_V0_EVAL.md`, and `research/EXECUTOR_V0_PLAN.md`.

Before expensive runs record exact code/configuration, engine identity, data/version/filter, seeds/opponents, outputs, stop conditions, and recovery plan.
