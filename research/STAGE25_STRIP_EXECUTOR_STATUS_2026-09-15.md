# Stage 2.5 Strip Executor Status — 2026-09-15

Branch: `codex/stage25-upkeep-ablation`

Current executor baseline for this note: `8c9dd94291ee4ea92cdf6e88828e27aa09022f5f` (`Add executor-owned instasell to strip market`).

This note is the current handoff for the experimental strip executor. Older upkeep-ablation and executor-v0.7 reports remain historical evidence; they do not describe the current strip contract.

## Current direction

The executor is intentionally becoming a mechanical realization layer rather than an economic safety layer.

The strategic manager owns meaningful economic choices: crop/animal/land targets and, in the Stage 2.5 design, the persistent crop-capacity ledger and other strategic actions. The executor should realize those requests efficiently even when the request is economically poor. It should prevent physical impossibility, illegal-action cascades, and obvious execution waste; it should not quietly protect the manager from overextension or bankruptcy with increasingly elaborate strategic guardrails.

This means an old policy can perform very badly behind the new executor without proving that routing itself is broken. In particular, P-final was trained behind an older executor with capital/debt/hiring protections, so closed-loop score parity is not expected when those protections are removed.

## Strip executor architecture

The current strip path is separate from the legacy persistent-scheduler executor.

### Packet 1 — work projection

- Build a mechanical daily work forecast from the manager plan and observed farm state.
- Manager owns crop/animal/land intent.
- Executor owns watering, harvest, care, fertilizer, feed execution, structure/placement realization, and exact legality/mechanics.
- Care and fertilizer are heuristic/mechanical rather than manager heads in the intended Stage 2.5 contract.

### Packet 2 — route ownership

- Current geometry is fixed horizontal five-tile rows.
- `StripRoute` is intentionally generic so route generation can change later without replacing the execution contract.
- One worker owns one route; no helpers or dynamic splitting after freeze.
- Existing workers are assigned deterministically to candidate rows.
- Entry selection uses the nearer endpoint with deterministic tie breaking; movement is vertical-first then horizontal.
- A worker sweeps its route once, performing supported local work before moving past a tile.

The row geometry is provisional. Replay evidence later showed strong vertical-first / compact-territory behavior from a newer expert, but there is not yet enough controlled evidence to justify replacing rows. Keep rows until a measured routing bottleneck appears.

### Packet 3 — route supply preparation

- Derive route-specific supply demand from represented work.
- Existing carried inventory satisfies its owner's route first.
- Shed reservations are deterministic and non-overlapping.
- Pickup is batched per item at shed access tiles and observation-confirmed.
- No general mid-route refill system is implemented.
- Fertilizer is stock-only; the strip market does not buy fertilizer.

### Packet 4 — market bootstrap

Daily order is currently:

`plan -> preliminary work -> market procurement -> observed reconciliation -> hiring -> finalize routes/supplies -> execute`

The market planner can submit SELL, BUY_PRODUCT WHEAT, BUY_SEED, BUY_ANIMAL, and next BUY_LAND intents under the shared market-order cap. Buys are confirmed from the next observation rather than assumed from submitted commands.

A known limitation is that bootstrap happens before route execution. Same-day worker production is therefore not generally available to finance initial procurement unless it is already in the shed. This mattered badly in the FQ day-slice attempt and is one reason that slice is not a useful first routing benchmark.

### Packet 5 — coverage-driven hiring

- Hiring is derived from candidate route coverage rather than an ROI gate.
- Existing workers cover the first routes; useful later routes can require a prefix of new hires.
- Pure fertilizer-only work does not drive hiring; ordinary required work can.
- A prospective hire must be able to reach at least one driving action before end of day.
- Hires are observation-confirmed.
- No helpers, route splitting, or post-freeze hiring is added.

Packet 5B corrected first-use ETA to include movement, route sweep travel, pickup turns, preceding interactions, and the first driving interaction. This prevents optimistic late-day hires that cannot actually reach useful work.

### Packet 5C — executor-owned insta-sell bridge

Commit `8c9dd94291ee4ea92cdf6e88828e27aa09022f5f` adds opt-in `StripExecutorConfig.aggressive_sell_all` and wires it through both bootstrap and post-finalization market planning.

In aggressive mode, observed shed products can be sold without relying on manager sell bins. SELL orders are planned before BUY orders and share one simulated market ledger, so same-turn sale proceeds can make a purchase affordable in that same planner call. Route reservations and represented feed demand are protected.

This is a temporary liquidity bridge. A learned selling policy may replace it later if time permits, but the project should not block current Stage 2.5 work on that future model.

## Immediate sell-policy correction still pending

The first P-final smoke with Packet 5C confirmed that selling is now happening, but it also exposed an undesirable temporary behavior: the bridge sold WHEAT and FERTILIZER.

For the current heuristic contract, insta-sell should liquidate finished outputs only:

- CARROT
- TOMATO
- STRAWBERRY
- MELON
- EGG
- MILK
- WOOL

It should not automatically sell:

- WHEAT — executor-side FEED input;
- FERTILIZER — executor-side upkeep input and not purchased by strip procurement.

This should remain a tiny bounded correction, not a new strategic selling system.

## P-final closed-loop smoke

A small diagnostic compared the old P-final policy under its historical executor against the same policy under the strip executor on seeds `309507` and `615013`, both seat orientations.

Historical executor arm banks:

- 82,166
- 96,614
- 102,072
- 81,990

Mean: 90,710.5.

Pre-insta-sell strip arm banks:

- 0
- 0
- 0
- 0

This is a diagnostic failure, not a promotion comparison. P-final was trained under an executor that performed substantial economic protection/repair, while the strip executor is designed to follow the manager plan much more literally.

The useful per-day signal was upstream of route traversal: cash collapsed, hiring collapsed, and routes became unassigned. When a route did receive a worker in the displayed diagnostics, `completed_routes == assigned_routes` on the observed days. Example for seed 309507 seat 0:

- day 4: cash before hiring 251, 5 workers, 5 assigned, 5 completed;
- day 5: cash before hiring 47, 4 workers, 4 assigned, 4 completed;
- day 6: cash before hiring 1, 4 workers, 4 of 5 assigned, 4 completed;
- day 7: cash 0, 2 workers, 2 of 5 assigned, 2 completed;
- later days: often one worker and one completed route.

This does not prove the row design is optimal, but it is evidence that the immediate catastrophic failure was not simply workers failing to traverse assigned rows.

## Packet 5C smoke

One rerun with `aggressive_sell_all=True` still ended at bank 0, but sales were now present:

- day 4: sold 4 FERTILIZER;
- day 5: sold 7 WHEAT;
- day 11: sold 15 MELON.

The sale bridge therefore works, but adding it alone does not make P-final compatible with the new executor economics.

The very small amount of sellable finished output also leaves an executor logistics question open: the strip executor supports HARVEST interactions, but it does not yet implement a general worker-inventory-to-shed delivery/deposit route. This may matter for same-day and later liquidity, especially for animal products and harvests collected into worker inventory. Do not treat it as the proven cause of the zero-bank P-final result until measured directly.

## FQ replay/day-slice lesson

The earlier FQ day-6 slice was too demanding for a first routing benchmark. FQ used same-day production, deposit, sales, and land purchase to finance expansion from a very small morning bank. Strip bootstrap currently asks procurement and hiring to settle before normal route execution, so it cannot reproduce that financing loop cleanly.

Use FQ later as a stress test after ordinary closed-loop behavior is stable. Do not redesign row routing from the FQ failure alone.

## Executor philosophy to preserve

Do not reintroduce broad economic safeguards merely to make old P-final scores look normal.

Allowed executor responsibilities include:

- exact legality;
- deterministic pathing/routing;
- supply reservation and pickup;
- mechanical care/fertilizer/watering/harvest rules;
- executing manager-requested expansion and animal/crop targets;
- simple bounded checks required to make an action physically realizable.

Strategic consequences should remain visible to the learner. If a manager asks for an overextended farm and the mechanically correct realization bankrupts it, that is valid training feedback rather than something the executor must hide.

## Next training experiment

BC/outcome-proxy preparation for the new Stage 2.5 architecture is not ready yet, while TPU access is time-sensitive. A short scratch Stage 2.5 self-play run is therefore reasonable as an experiment before BC initialization exists.

Treat this as a plumbing / learnability / TPU-utilization experiment, not a competitive training baseline.

Proposed constraints:

- use the native Stage 2.5 JAX policy/action/checkpoint stack;
- initialize the Stage 2.5 policy from scratch rather than fabricating a BC warm start;
- use the fast backend for training throughput;
- keep the current standard opening / manager-start contract unless the runbook explicitly changes it;
- use the strip executor with the temporary executor-owned sell bridge after the WHEAT/FERTILIZER exclusion correction;
- do not interpret initial score magnitude as a promotion result;
- persist exact branch/commit, action contract, executor identity, engine provenance, seeds, batch size, optimizer/PPO config, and checkpoint metadata;
- start with a bounded compile + short training smoke before spending a long TPU allocation;
- stop on NaN/nonfinite updates, repeated executor/runtime errors, invalid checkpoint/resume identity, or a clearly degenerate action distribution that does not recover during the smoke.

If the scratch run is healthy, it can continue as an experiment while the BC pipeline is prepared separately. BC remains the likely path for a stronger initialization; the scratch run should not silently become the new canonical baseline just because it was started first.

## Near-term sequence

1. Restrict insta-sell so WHEAT and FERTILIZER are retained.
2. Run one or two ordinary full-game strip smokes to confirm sales, routing completion, and no runtime failures. Do not require P-final economic parity.
3. Launch the bounded scratch Stage 2.5 TPU/self-play experiment if the new architecture stack is ready.
4. In parallel/later, finish outcome-proxy replay processing and BC setup for a stronger Stage 2.5 initialization.
5. Return to executor refinement only when controlled diagnostics identify a mechanical bottleneck. The most likely open seams are product delivery/deposit, bootstrap timing/liquidity, and eventually route geometry; none should be redesigned speculatively.

## Stage 2.5 executor mechanical repair — 2026-09-20

The Stage 2.5 executor repair packet supersedes the temporary WHEAT/FERTILIZER
sell exclusion and the one-row-per-worker assumption:

- aggressive sell mode now considers every canonical `PRODUCTS` entry;
  outstanding observation-confirmed route reservations still protect committed
  shed stock;
- wheat with observed `yield_units >= 3`, or with the existing expiry/horizon
  exception, is harvested on that observation and does not receive competing
  routine or fertilizer-linked WATER;
- horizontal candidates are packed as ordered row segments per worker, with
  deterministic nearest-endpoint traversal, segment-level diagnostics, and a
  bounded no-new-supply helping handoff when a worker exhausts its chain;
- hiring evaluates the same packed capacity model and only requests a worker
  when the additional capacity completes additional useful work before the
  day boundary.

Focused executor/work/market/supply/hiring/integration validation passed 145
tests. Official 1.32.7 deterministic parity for seed 41001 also passed 719
turns in both seats with matching terminal status and reward. The requested
epoch8-vs-epoch4 BC artifacts were not present in this checkout; the available
repo-local BC checkpoint is `E_LEGACY` and is rejected by the current
`E_CORRECTED_V1` runner, so no BC replay claim is made here.
