# Experimental strip work Packet 2

Packet 2 is an opt-in executor experiment.  The production `ExecutorAgent`
does not import or invoke it.

## Contract

`StripRoute` is a generic route record: it owns a unique tile set and an
arbitrary deterministic traversal of that set, with a stable worker owner,
entry tile, cursor, phase, and route diagnostics.  This generic representation
is an extension point for future route generators; Packet 2 itself generates
only five-tile horizontal quadrant rows from Packet 1 `RowKey` values.

At the day boundary, candidates are sorted by the existing `RowKey` order and
workers are sorted farmer-first (`FARMER`, `HAND:0`, `HAND:1`, ...).  Candidate
`i` is assigned to worker `i` until one list ends.  Excess routes are explicitly
unassigned and excess workers remain idle.  The nearer row endpoint is chosen
by Manhattan distance from the worker's observed position; ties choose the
left endpoint.  The resulting traversal never reverses.

Travel is one legal cardinal move at a time, vertically toward the target
`y` first and horizontally toward its `x` second.  Engine positions are
`[x, y]` while route/work coordinates are `(y, x)`.  A movement stores a
pending next-tile index; the cursor advances only after the next observation
confirms arrival at that tile.

Ownership is fixed for the day.  The controller rebuilds the current Packet 1
work forecast once per turn, but never rematches ordinary travel, workload, or
completion.  A worker interacts only with its current owned tile.  It executes
one supported `READY` local item, remains there, and waits for the next real
observation so a newly unlocked continuation can run before moving.  Local
ties use the fixed maintenance-first order `FEED`, `FERTILIZE`, `WATER`,
`CARE`, `COLLECT_FERTILIZER`, removal/build, `PLACE`, then `PLANT`.

Packet 2 does not hire, pick up supplies, return to the shed, transfer work,
revisit passed tiles, optimize assignment, or search paths.  Inventory-scoped
work is issued only when the assigned worker's actual private inventory has
the required item.  Seeds remain global for `PLANT`; shed stock is not
teleported into a worker.  Work that appears after a tile was passed is kept
in diagnostics and is not repaired.

Diagnostics include daily route counts, unassigned routes, idle workers,
tile-less unresolved work, route ownership/traversal/entry, cursor and phase,
completion and post-completion PASS turns, action/movement/interaction counts,
blocked local work, unavailable carried supplies, and late passed-tile work.
