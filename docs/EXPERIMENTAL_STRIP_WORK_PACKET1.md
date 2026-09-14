# Experimental strip work model — Packet 1 (Packet 1B corrected)

Packet 1 adds a read-only forecast for a future strip executor.  It turns one
daily `executor_v0.plan.DailyPlan` and the acting farm's current observation
into complete spatial work chains, including downstream work that is currently
blocked by a prerequisite.

## Manager / executor boundary

```text
Manager owns:
- crop targets
- animal targets
- land target
- strategic sell intent

Executor owns:
- watering
- harvest mechanics
- feeding
- care
- wheat/strawberry fertilizer timing under configured permissions
- local interaction chains
```

In particular the strip builder ignores the legacy `DailyPlan.care_by_animal`
and `DailyPlan.fertilizer_by_crop` count fields (the old executor may still
use them; they are not removed).  CARE is generated for every mechanically
care-worthwhile animal and fertilizer for every mechanically eligible plant
when the corresponding strip permission is ON.  For now:

```text
wheat fertilizer permission = ON
strawberry fertilizer permission = ON
```

by experimental config default (`StripWorkConfig`).

## Contract

`executor_v0.strip_work.build_strip_work_plan(obs, plan, config=...)` returns a
`StripWorkPlan`.  The result contains immutable `WorkItem` interactions,
ordered `WorkChain` links, a `SupplySnapshot`, per-five-tile `RowSummary`
records, and aggregate `WorkDiagnostics`.  It accepts the requested plan (not
the executor's clipped feasible projection), is deterministic and JSON-safe,
does not mutate either input, and has no worker, scheduler, hiring, or route
state.

Board and task coordinates are canonical `(y, x)` and index `tiles[y][x]`.
Engine worker positions remain `[x, y]`; the model does not consume them.
`row_key_for_tile` exposes quadrant, local row, global row, and the five-column
segment for Packet 2.

Chains retain complete commitments such as `PLANT → WATER`,
`HARVEST → PLANT → WATER`, `FEED → CARE`, `FERTILIZE → WATER`, and
`DIG → BUILD → PLACE`.  Seeds are global requirements and never become pickup
work.  Sell intents expose a delivery prerequisite when requested product is
carried but not in the shed; no delivery action is executed by this packet.

## Status semantics

`READY` means the primitive work item is mechanically executable from the
current represented state, subject only to worker/location assignment which
Packet 1 intentionally does not model.  A work item with an unresolved
prerequisite is **not** READY; it is `BLOCKED` with
`block_reason = DEPENDENCY_BLOCKED` and the prerequisite listed in
`depends_on`.  A dependency merely being represented as another `READY` item
does **not** satisfy it — Packet 1 never simulates execution, so every
`depends_on` edge names future sequential work.

Example: a new crop forecasts `PLANT (READY) → WATER (DEPENDENCY_BLOCKED)`;
a replacement forecasts `HARVEST (READY) → PLANT (DEPENDENCY_BLOCKED) →
WATER (DEPENDENCY_BLOCKED)`.  The chain object remains a valid foreseeable
commitment (its `interaction_turns` still counts every member for workload
planning) even though only its first primitive is `READY`.  `BLOCKED` chains
are future work, not broken chains.

Other block reasons keep their narrower meanings: `MISSING_SUPPLY` (shed/
carried inventory shortage such as feed or fertilizer),
`MISSING_GLOBAL_RESOURCE` (seed shortage, unaffordable purchase),
`MISSING_PURCHASE` (an animal PLACE whose affordable purchase is still
pending), `LOCKED_LAND` (work whose coordinates cannot exist until a land
purchase executes), `NO_SPATIAL_SLOT`, or `OTHER_MECHANICAL_BLOCK`.  A
`BUY_LAND` item itself is `READY` when affordable (sequential land prices are
charged in order and each later purchase depends on the previous one) and
`MISSING_GLOBAL_RESOURCE` when not; downstream plot-less work stays
`LOCKED_LAND` on those ids without fabricated coordinates.  Labor is not a
block reason here: estimated non-travel interactions are exposed for later
assignment.

Scarce fertilizer is never dropped: all mechanically useful applications are
represented in stable order, the affordable prefix is `READY`, and the excess
is `MISSING_SUPPLY`, so later packets can see total demand versus supply.

Authoritative crop/animal lifecycle rules come from the existing constants,
canonical lifecycle, and upkeep helpers.  Watering ages, initial WHEAT and
STRAWBERRY fertilizer permissions, and the preferred fertilizer age windows
are experimental executor defaults in `StripWorkConfig`, not claims about
immutable engine law.  The model does not add fertilizer policy for other
crops.

## Explicit boundary

This packet does not change gameplay and does not establish that strip routing
is superior.  It does not assign workers, route or pathfind, persist progress,
hire or transfer helpers, split routes, reserve or batch supplies, execute
pickup or delivery actions, change market sequencing, add expansion policy,
change manager observations, alter PPO/BC/self-play, or replace the current
executor.  The existing `executor_v0` remains the production/conservative
baseline and is not routed through this module.

## Packet 2 interface

Packet 2 can consume `StripWorkPlan.row_summaries` and
`diagnostics.row_workload`, group non-travel work by `RowKey`, and add
deterministic five-tile sweep geometry and route travel.  It must treat
`travel_turns=0` in Packet 1 as a placeholder, respect `depends_on` ordering
even between `READY` and `BLOCKED` items (a `READY` prerequisite does not
make its dependent executable), preserve blocked chains for future
scheduling, and leave strategic targets unchanged.
