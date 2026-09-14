# Experimental strip work model — Packet 1

Packet 1 adds a read-only forecast for a future strip executor.  It turns one
daily `executor_v0.plan.DailyPlan` and the acting farm's current observation
into complete spatial work chains, including downstream work that is currently
blocked by a prerequisite.  The manager still owns strategic crop, animal,
land, care, fertilizer, and sell requests; this model only makes their
mechanical consequences visible.

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

Work status and block reasons are intentionally separate.  `READY` means the
interaction's known mechanics and supplies permit it now; `BLOCKED` retains a
reason such as `DEPENDENCY_BLOCKED`, `MISSING_SUPPLY`,
`MISSING_GLOBAL_RESOURCE`, `MISSING_PURCHASE`, `LOCKED_LAND`,
`NO_SPATIAL_SLOT`, or `OTHER_MECHANICAL_BLOCK`.  Labor is not a block reason
here: estimated non-travel interactions are exposed for later assignment.

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
`travel_turns=0` in Packet 1 as a placeholder, preserve blocked chains for
future scheduling, and leave strategic targets unchanged.
