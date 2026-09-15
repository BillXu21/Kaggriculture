# Experimental strip work Packet 3 — route supply preparation

Packet 3 extends the opt-in Packet 2 controller with inbound, route-local
resource preparation. Packet 2 still owns horizontal five-tile route
generation, worker-to-route assignment, endpoint choice, vertical-first
movement, exclusive ownership, monotonic sweep, local chains, and completion.
Packet 3 does not use the old scheduler and does not change those decisions.

## Daily contract

The day is planned in this order:

`Packet 1 WorkItems → Packet 2 assigned routes → Packet 3 demand and reservations`

Only assigned routes receive supply plans. Demand is collected once from the
route's spatial `WorkItem.required_supplies`. All requirements with
`scope="inventory"` are included, including dependency-blocked foreseeable
work. `scope="global_seed"` is excluded: seeds remain in the global seed pool
and are never picked up by workers. Work demand remains complete even when
inventory or shed stock is insufficient.

For each item, the owner's observed carried inventory satisfies demand first.
The remaining demand is reserved from the observed shared shed in assigned
route order. Reservations are day-stable and satisfy, for every item,
`sum(route reservations) <= initial observed shed stock`; no unassigned route
receives a reservation. A route's demand reconciles as carried + reserved shed
stock + known unfulfilled stock (the authoritative engine has no capacity
shortfall).

Within a route, item order is the first use along its Packet 2 traversal,
then Packet 2's maintenance-first local action order, then stable item/id
ordering. Each item is emitted as one quantity pickup batch where stock is
available. A route visits the nearest canonical shed-access tile only when it
has a positive reservation. If all demand is already carried, or none can be
reserved, it proceeds directly to its unchanged Packet 2 entry endpoint.

## Pinned mechanics findings

The pinned official `kaggle_environments==1.32.7` interpreter and the matching
fast-engine source establish:

- canonical shed-access positions are `[x,y]` `(4,4)`, `(5,4)`, `(4,5)`, and
  `(5,5)`, represented internally as `(y,x)` `(4,4)`, `(4,5)`, `(5,4)`, and
  `(5,5)`;
- `PICKUP item n` is legal at any of those four positions, accepts an
  unbounded positive quantity, and transfers `min(n, observed shed stock)`;
- worker inventory is a sparse item→quantity mapping covering products and
  animals, while seeds are a separate private global mapping;
- workers have no finite carrying capacity. Packet 3 therefore does not invent
  capacity limits, allocation caps, or drop/transfer behavior;
- partial pickup is mechanically possible when requested quantity exceeds
  visible stock. Packet 3 confirms actual inventory gain, may make one bounded
  follow-up attempt, and records any remainder as failed/unfulfilled. A command
  itself is never treated as proof.

The source audit is `kaggle_environments/envs/kaggriculture/kaggriculture.py`
(`_is_shed_adjacent`, `_apply_unit_action`) and
`rust/kaggriculture_env/src/lib.rs` (`is_shed_adjacent_at`,
`move_from_shed`, `apply_unit_action`).

## Execution boundary

`PREPARE_SUPPLIES` precedes normal `TRAVEL_TO_ENTRY`. Pickup state stores the
requested batch, inventory before the command, confirmed acquired quantity,
attempt count, and failed remainder. Once all batches are fulfilled or
conclusively unavailable, the worker travels to the exact Packet 2 entry
endpoint and performs the unchanged monotonic sweep.

There is no market buying, cash reservation, hiring, product delivery,
selling, dropping, mid-route refill, helper transfer, route splitting, or
opportunistic work on another route. A worker's prepared resources can only
satisfy supply requirements encountered on its own assigned tiles. Shortage,
reservation, pickup, failure, and remaining carried quantities remain in route
and day diagnostics. Reservations and pickup state reset at the next day.
