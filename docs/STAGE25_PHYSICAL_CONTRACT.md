# Stage 2.5 Packet 1A: physical mechanics contract

This document is authoritative for the Stage 2.5 land, animal-housing, crop
capacity, and board-relative crop-delta helpers. Later policy/training packets
must reproduce these rules exactly; they should not infer them from executor
code. Compatibility-named crop-ledger helpers apply deltas to a supplied
boundary baseline; the live provider re-derives that baseline from the current
physical board on every daily decision.
The public framework-free implementation is
`rl_manager.stage25_mechanics`.

## Action schema

There is one manager decision per observed day. The nine autoregressive heads
are, in order:

`land, goose, cow, sheep, wheat, carrot, tomato, strawberry, melon`

Their class counts are `(4, 101, 101, 101, 201, 201, 201, 201, 201)`.

- Land class `j` decodes to absolute target `j + 1` unlocked quadrants.
- Animal class `j` decodes to absolute target `j` placed animals.
- Crop class `j` decodes to signed delta `j - 100`.

Land and animal targets may not be below their observed values. Animal targets
are daily acquisition goals: a lower goal on a later day cancels unfinished
acquisition but cannot remove an already placed animal. Compatible animals
already in shed/carried inventory satisfy acquisition demand and must not be
purchased again; they are represented by `PhysicalContext.unplaced_animals`.

For crops, class `100` is the only HOLD meaning and always decodes to zero.
The fixed vocabulary is permanently `-100..+100`; there is no permanent
`+25` cap and no separate CLEAR class. A sampled class is decoded or rejected
by the support contract. It is never clipped or repaired after sampling.

CARE, fertilizer, and selling are not Stage 2.5 policy heads.

## Source-ground physical context

The pinned Kaggriculture 1.32.7 engine has a 10x10 board divided into four
5x5 quadrants. `NW` is initially unlocked; land expands in the prefix order
`NW -> NE -> SW -> SE`. Unlock prices are economic only and never enter a
support mask. Unlocking changes target `LOCKED` cells to `None`; it creates
no structures, crops, or weeds, so newly unlocked cells are immediately
usable.

The engine requires the current tile to be exactly `None` for
`BUILD_COOP`, `BUILD_PASTURE`, or `PLANT`. A weed or plant therefore
needs a legal `DIG` first; the physical support counts both as recoverable
capacity. Occupied structures and empty structures are sticky and are not
crop/build cells. Unknown tile shapes and locked cells outside the requested
footprint are unusable.

Animal compatibility is fixed:

| animal | structure | capacity |
| --- | --- | --- |
| GOOSE | COOP | one animal per structure tile |
| COW | PASTURE | one animal per structure tile |
| SHEEP | PASTURE | one animal per structure tile |

An empty compatible structure can be reused. Empty pastures are one shared
reusable pool for cows and sheep (there is no separate cow/sheep capacity
pool); each pasture tile still holds only one animal. An animal escape leaves
the bare structure in place. Harvesting leaves both animal and structure in
place.

`PhysicalContext` contains:

- `observed_land`;
- `crop_build_cells_by_land[land_target - 1]`, the recoverable crop/build
  footprint `B` for each hypothetical absolute land target;
- observed placed counts for goose/cow/sheep;
- reusable empty coop and pasture counts;
- unplaced compatible animals in shed/carried inventory.

The pure `physical_context_from_board` helper derives this context from the
canonical board and unlocked prefix. It does not read money, prices, feed,
labor, affordability, profitability, or any other economic state.
`unplaced_animal_counts` is the pure shed-plus-carried inventory counter used
to populate the final context field.

For an absolute animal target vector `T`, let `P` be observed placed counts
and let `E_coop`/`E_pasture` be reusable empty compatible structures:

```text
d_goose = T_goose - P_goose
d_pasture = (T_cow - P_cow) + (T_sheep - P_sheep)
new_housing_cells =
    max(0, d_goose - E_coop)
  + max(0, d_pasture - E_pasture)
C = B - new_housing_cells
```

The helper does not clamp negative `C` to zero. An animal prefix is
physically feasible iff every target is at least observed placement and
`C >= 0`.

Sequential animal masks account for observed placement, prior sampled targets,
compatible reusable structures, and the requested land footprint. They do not
reserve housing for future unsampled species. This is intentional: the goose
head is evaluated first, then cow, then sheep; cows and sheep share the
pasture reuse pool when their heads are reached.

## Crop residual support

After land and all three animal targets determine `C`, crops are decoded in
the fixed order above. At the current decision boundary, let `B_i` be the
physical count of crop `i`; for earlier newly decoded goals, define:

```text
R_i = C - sum(decoded_goal_j for earlier j)
```

The valid signed delta interval is:

```text
max(-100, -B_i) <= delta_i <= min(100, R_i) - B_i
```

Equivalently, `-B_i <= delta_i <= min(100, R_i) - B_i` intersected with the
fixed `-100..+100` vocabulary. A future species' old goal is not reserved;
later heads may contract it. Therefore HOLD can become unavailable when an
earlier choice consumes residual capacity, while full contraction
`delta_i = -B_i` remains possible whenever the residual is nonnegative.
The final decoded goal sum is never greater than `C`. Fixed-order allocation
asymmetry is intentional; there is no random order, solver, allocator, or
economic safeguard.

## Board-relative crop goals

At every manager boundary, derive the five integer crop baselines from the
physical board currently observed. Decode the sampled signed deltas exactly
once into that boundary's requested end-of-day goals in `0..100`.

Goals are requests, not guaranteed execution results; incomplete execution is
allowed. Harvests, crop loss, unfinished planting, worker backlog, and
temporary vacancies therefore affect the next physical baseline. For example,
with `30` observed wheat, delta `0` requests an end-of-day wheat count of `30`,
not a carried-forward strategic goal from an earlier day. No unrealized crop
intention is carried forward automatically.

Future resume/checkpoint wiring is outside Packet 1A, but must preserve the
physical pre-decision baseline for the accepted boundary when current-day
diagnostics require it; it must not use requested goals as the next boundary's
baseline. Later integrations should store sampled class indices as `int16
[B,9]` and the model-facing pre-decision crop capacity as `int16 [B,5]`, while
keeping class indices, signed deltas, physical baselines, and decoded goals
distinct.

## Authoritative implementation boundary

Later JAX/PPO code must reproduce the constants, decode mappings, support
intervals, prefix ordering, no-clipping behavior, and board-relative goal
semantics above. It must not import executor internals to rediscover physical
rules, and economic state must not affect the permanent physical masks.

Source evidence: `replay_daily/constants.py` for the verified engine tables
and land order; `executor_v0/layout.py` and `executor_v0/tasks.py` for
sticky structures, matching reuse, and recoverable layout claims; and the
pinned official 1.32.7 source/test evidence recorded by
`oracle/provenance.py`, `tests/test_oracle_animals.py`, and
`tests/test_oracle_crops.py`.
