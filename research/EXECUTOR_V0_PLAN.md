# Deterministic Executor V0 Plan

Date: 2026-08-22
Status: first implementation draft; intentionally simple

## Implementation status (2026-08-23)

Sections 1–11 are **implemented** in `executor_v0/` (commits `11e85fa`..
`ed1685a`, issue #1) exactly as simple first drafts — nothing here is
optimized. Usage: `executor_v0/README.md`. Validation: 249 tests; live
encoder parity; determinism coverage; a 719/720-turn replay-observation
plumbing smoke (shape/state robustness only — no engine executed the
counterfactual actions). A real `kaggle_environments` 1.32.7 game is still
pending because the package is absent locally.

Mechanics locked during implementation (verified against canonical replay
data and the repo's established 1.32.7 facts):

- `PLANT <crop>` consumes the **global own `private.seeds[crop]` pool
  atomically at the engine**; seeds are never picked up or carried, crop
  items in inventories are products, not seeds. The foreman reserves global
  seeds per crop within each turn; BUY_SEED shortage = task demand minus
  global seeds.
- Shed pickup/drop uses the four center tiles `[y,x]
  {(4,4),(4,5),(5,4),(5,5)}` as **overwhelmingly observed valid locations**
  in elite replays — evidence-backed configuration, not a source-locked
  universal rule.
- Market queue order is SELL -> HIRE -> BUY under the 10-order cap;
  lower-priority candidates that do not fit are deferred and recomputed next
  turn (never counted as submitted).

Additional backlog discovered during implementation (in addition to the
simplifications below): conservative LOCKED-tile avoidance though the engine
may permit stepping there; worker interaction/current-tile legality pending
engine-smoke confirmation; `tasks_per_worker=10` and pickup batch 5 are
provisional constants; no emergency-feed safety buy or affordability
sophistication; the farmer-anchored layout can thrash before the first build
commits (revisit trigger below).

## Purpose

Build the smallest deterministic executor that can turn one daily BC-manager
plan into legal, coherent Kaggriculture 1.32.7 primitive actions. The first
closed-loop goal is diagnostic, not leaderboard optimization: complete real
games, measure how faithfully the executor realizes manager intent, and make
manager failures distinguishable from logistics failures.

The executor is a strategy compiler under D-011. It may enforce mechanical
feasibility and choose efficient low-level execution, but it should not quietly
replace the learned crop/animal/market strategy with its own economic policy.

## Input contract

At `hour == 0`, run the BC manager once and cache a daily plan containing:

- crop target counts by crop type;
- animal target counts by animal type;
- unlocked-land target count;
- fertilizer applications by crop type;
- CARE applications by animal type;
- sell intent for 9 products across six four-hour bins.

The executor acts on the real state throughout the day. It must log the raw
manager request separately from any mechanical feasibility projection and from
the final achieved state.

## 1. Mechanical feasibility projection

Use a small transparent projection layer before task generation.

- Land cannot be unbought: `feasible_land = max(current_land, requested_land)`.
- Do not remove existing animals merely because the requested target decreases;
  V0 only buys/places animals needed to increase toward the target.
- Crop counts remain genuine mutable targets because crops can be dug/replaced.
- Fertilizer and CARE requests may be clipped only by actual eligible targets or
  unavailable resources/mechanics.
- Selling is clipped by what can actually be sold as inventory becomes
  available within the requested bin.

Always retain `manager_requested`, `executor_feasible`, and `achieved` values in
logs. Frequent projection is evidence that the manager/action abstraction may
need revision and must not be hidden by the executor.

## 2. Layout and plant-to-animal transition

### No reserved animal zone

Do **not** leave the nearest shed tiles empty from the start in anticipation of
future animals. V0 uses productive crop space until an animal target actually
requires conversion.

### Sticky livestock footprint

Once a tile becomes livestock infrastructure, keep it stable unless a later
mechanical reason forces a change. Prefer new livestock slots nearer the shed,
but do not destroy valuable crops solely to minimize one or two steps of travel.

### Crop sacrifice rule

When a new animal slot needs a currently cropped tile, prefer the **lowest
sunk-value nearby crop**, not a crop that is close to harvest.

Desired ordering:

1. legal empty tile, if reasonably near the shed;
2. just-planted / very young, unfertilized crop;
3. progressively more invested crops;
4. mature, fertilized, or near-harvest crops are expensive to destroy.

A first implementation may use a deliberately crude deterministic score such
as:

```text
animal_slot_cost =
    shed_distance
  + age_or_growth_penalty
  + fertilizer_sunk_cost_penalty
  + near_harvest_penalty
  + stored_yield_or_other_investment_penalty
```

Exact coefficients are provisional. The important invariant is that sunk crop
investment dominates small travel-distance differences. Among similarly cheap
crops, prefer the tile nearer the shed.

### Crop reconciliation

Preserve existing crops that already contribute toward the requested target.
When a crop is over target, mark only the excess as replaceable, ranking the
cheapest-to-sacrifice plants first. Fill crop deficits using legal empty tiles
before digging another useful crop when practical. Minimize unnecessary
relayout between days.

## 3. Task graph

Represent mechanical work as explicit tasks rather than one large imperative
agent procedure. A task should carry at least:

```text
kind
location
required item/resource, if any
priority/deadline
source manager intent or maintenance reason
dependencies
```

Examples:

- WATER crop;
- HARVEST crop;
- FEED animal;
- CARE animal;
- FERTILIZE crop;
- COLLECT_FERTILIZER;
- DIG crop;
- PLANT crop;
- BUILD_COOP / BUILD_PASTURE;
- PLACE animal;
- required PICKUP / logistics;
- sell-bin market intent.

Structural work should use simple dependencies, e.g. `DIG -> BUILD -> PLACE`
or `DIG -> PLANT`.

Regenerate/revalidate pending work from the actual observation every primitive
turn instead of assuming yesterday's or the previous turn's planned state was
achieved.

## 4. Task urgency

Use broad priority/deadline classes rather than a large hand-tuned strategic
priority table.

1. maintenance that risks losing current production if missed;
2. ready productive work and requested CARE/fertilizer;
3. manager-directed farm transitions such as crop replacement and animal
   construction/placement;
4. optional cleanup/logistics.

Within a class, use actual lifecycle deadlines where available, then execution
cost. Exact ordering should remain easy to inspect and should be revised only
when real-game traces show a problem.

## 5. Worker inventory and soft specialization

Engine mechanics allow a worker to carry multiple item types and quantities;
the practical cost is pickup turns and routing, not a worker carry-capacity
limit. A `PICKUP` consumes that worker's action for the turn and handles one item
type.

V0 therefore uses **soft specialization**, not permanent professions:

- a worker already carrying wheat becomes cheap for FEED tasks;
- a worker carrying fertilizer becomes cheap for FERTILIZE tasks;
- an empty worker is naturally attractive for WATER/HARVEST/DIG/PLANT work;
- penalize assignments that require introducing another carried item type;
- prefer planned bundles with at most roughly two distinct carried item types
  unless there is a clear mechanical reason to exceed that.

Do not hard-code identities such as "hand 2 is always the fertilizer worker."
Carried inventory should create temporary specialization automatically.

## 6. Start-of-day loading, shed staging, and hiring

Relevant engine ordering:

- current workers act first;
- the market queue is processed afterward;
- therefore worker pickups can free shed capacity and same-turn market orders
  can refill that capacity;
- newly bought goods cannot be picked up until the next primitive turn because
  market processing occurs after worker actions;
- newly hired hands also cannot act on the hiring turn, but can act on the next
  turn;
- hired hands are day-local and must be hired again on later days.

Use a simple **greedy load-and-dispatch** routine:

### Hour 0

- Let the farmer take one useful bulk pickup from existing shed stock if that
  materially improves its first route.
- Stage purchases needed by the incoming hands using the newly available shed
  capacity where possible.
- Hire the day's desired hands immediately through the market queue.
- Do not make the farmer wait for other workers if it can already do useful
  work.

### Hour 1 onward

- New hands greedily take the most useful available bulk pickup.
- Dispatch a worker as soon as it has enough inventory for a useful route.
- A worker may spend another pickup turn on a second item type when clearly
  worthwhile, but ready workers never wait for synchronized loading.
- Continue using worker pickups to free shed space and same-turn market orders
  to stage later supplies when helpful.

The first version should favor getting workers productive quickly over finding
perfect inventory bundles.

## 7. Hiring heuristic

Hiring is a daily labor purchase, not a persistent workforce. V0 should estimate
the day's work coarsely and hire near the start of the day.

A provisional workload estimate may count:

```text
required interactions
+ rough travel lower bound
+ expected pickup actions
```

Then choose enough hands to make the work plausibly finish before day end,
subject to cash/legal constraints and a conservative safety cap. Any numeric
"useful actions per worker" constants are provisional and must be calibrated
from closed-loop traces rather than treated as game theory.

## 8. Greedy worker-task assignment

Recompute assignments every primitive turn.

First rule: if a worker is already standing on a useful executable task, do it.
Otherwise use a deterministic greedy cost such as:

```text
assignment_cost =
    Manhattan travel distance
  + shed detour if a required item is missing
  + penalty for each new carried item type required
  - bonus when the worker already carries the needed item
```

Process workers in deterministic order and assign the cheapest available
feasible task. Use stable tie-breaking so identical states produce identical
actions. Move one Manhattan step toward the assigned task, then recompute next
turn.

No global worker-route optimization is required for V0.

## 9. Mechanical purchasing

Purchase shortages implied by active manager/executor tasks rather than adding
an independent economic policy.

- Buy seeds needed by the crop reconciliation plan.
- Buy animals required to increase toward animal targets.
- Buy feed/fertilizer/product inputs required for mechanical completion.
- A small emergency/safety purchase that prevents mechanical animal failure is
  acceptable, but it must be logged because frequent use suggests the manager
  needs a learned wheat/feed decision later.
- Do not speculate or stockpile because a product looks profitable; product
  economics remain manager/RL territory.

## 10. Selling

For each four-hour bin, initialize the model-requested remaining sell quantity
per product. Sell mechanically as inventory becomes available within the bin,
up to the remaining requested amount. Record any unfilled quantity at bin/day
end.

Do not add an executor price-timing strategy inside the bin in V0. More reactive
selling is a deliberately deferred learned/optimized subsystem.

## 11. Compliance and closed-loop diagnostics

Per day, log at minimum:

- manager-requested, feasible, and achieved crop counts;
- manager-requested, feasible, and achieved animal counts;
- land target and hit/miss;
- fertilizer requested/completed;
- CARE requested/completed;
- sell requested/submitted/unfilled;
- missed water/feed/maintenance;
- tasks left pending at day boundary;
- worker movement actions;
- productive interaction actions;
- pickup/drop actions;
- idle/pass actions;
- emergency purchases;
- illegal/ineffective actions;
- final bank and paired W/L/T at game level.

If manager-intent compliance is poor, improve execution before blaming BC. If
compliance is high and economic trajectories are poor, revisit the manager or
abstraction.

## Deliberate V0 simplifications and future-upgrade backlog

These are **not** claims of optimality. They are explicit shortcuts for the
first closed-loop test and should remain visible so later work does not forget
them.

- **No anticipation of future animal onset.** We do not predict when animals
  will start or reserve nearby tiles in advance. Later: learn/estimate onset and
  compare proactive crop placement against reactive sacrifice.
- **Crude crop-sacrifice value.** Current ranking uses lifecycle/fertilizer
  investment proxies, not exact expected remaining crop value, market value,
  replant cost, or labor opportunity cost. Later: derive a better marginal
  replacement-value estimate if conversion losses matter.
- **No global layout optimization.** Near-shed preference plus sticky existing
  structures replaces facility-location or whole-farm optimization. Later:
  measure elite layout heatmaps and executor travel before adding complexity.
- **Greedy worker assignment.** No Hungarian matching, beam search, min-cost
  flow, VRP, or multi-turn route search. Later: upgrade only if movement,
  collisions, idle time, or completion failures are material.
- **One-step Manhattan navigation.** No long-horizon path or task-bundle search.
  Later: search short horizons if simple reassignment thrashes.
- **Soft specialization only.** Inventory affinity substitutes for explicit
  worker roles or joint bundle planning. Later: optimize worker bundles/routes
  if repeated pickups dominate.
- **Naive shed staging.** Load-and-dispatch is greedy and does not globally
  optimize the 100-unit shed buffer across the day. Later: add staged inventory
  planning if shed capacity causes waits or lost purchases.
- **Crude hiring estimate.** No economic optimization of labor count. Later:
  fit workload/completion models from real executor logs or include hiring in a
  learned manager if it proves strategically important.
- **Basic wheat/feed procurement.** Feed shortages are handled mechanically;
  wheat/feed economics are not learned in V0. Later: move wheat/feed ownership
  toward the manager when the first closed-loop system is stable.
- **Literal six-bin selling.** No within-bin market reaction, price forecasting,
  or order optimization. Later: 24-turn seller or reactive learned seller is
  already preserved by the canonical event data.
- **No opponent-aware executor logic.** Opponent information must not secretly
  enter low-level task choices as economic strategy. Later opponent awareness
  belongs primarily in the learned manager.
- **No uncertainty-aware plan projection.** Argmax BC outputs are treated as
  requests regardless of confidence. Later: calibration/uncertainty may be
  useful for conservative execution or PPO initialization.
- **No search-based foreman yet.** Search remains a plausible later performance
  optimization after the greedy executor establishes a working baseline.

### Revisit triggers

Prioritize the above backlog when closed-loop traces show one or more of:

- manager-intent compliance materially below roughly 90%;
- missed maintenance despite sufficient theoretical labor;
- large fractions of turns spent moving, waiting, or reloading;
- frequent shed-capacity stalls;
- repeated destruction of valuable crops during animal conversion;
- worker assignment thrashing;
- high compliance but substantial bank loss traceable to an executor-owned
  simplification rather than the manager.

Do not optimize these preemptively merely because a more sophisticated
algorithm exists.

## First implementation stop condition

The first bounded implementation is successful when it can:

1. load the trained BC checkpoint and encode live observations consistently;
2. run all 30 manager calls in real 1.32.7 games;
3. finish games without illegal-action cascades or deadlock;
4. emit the compliance diagnostics above;
5. run a small fixed-seed, seat-swapped panel against one frozen competent
   opponent.

Only after this should the project decide whether the next marginal work belongs
in executor search/layout, BC refinement, opponent inputs, or PPO.
