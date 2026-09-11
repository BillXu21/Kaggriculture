# Stage 2.5 Packet 1B: outcome, preprocessing, and curriculum contract

This document is authoritative for Stage 2.5 outcome proxies, preprocessing,
optional curriculum masks, crop-logit bias, and crop-capacity shortfall. It
consumes the physical rules in
[`STAGE25_PHYSICAL_CONTRACT.md`](STAGE25_PHYSICAL_CONTRACT.md); it does not
replace or reinterpret them. Packet 1B is framework-free and disabled
features must be behaviorally identical to the Packet 1A baseline.

## Outcome proxies

Outcome proxies describe observed outcomes. They are not fabricated
transactions, economic labels, or maintenance-aware reconstructions.

### Crops: synthetic persistent `K`

For each contiguous `(episode, seat)` manager history, construct a synthetic
persistent crop-goal vector `K` using Packet 1A:

1. At the first manager boundary after opening or an explicit reset,
   initialize `K` from the five observed planted-crop counts.
2. At each later adjacent boundary, decode the five sampled crop classes to
   signed deltas and apply each delta exactly once. The post-action `K` is the
   next boundary's pre-action `K`.
3. Adjacency requires the same episode and seat, ordered manager boundaries,
   the expected consecutive day/boundary identity, and no duplicate or
   missing row. An opening, explicit environment reset, episode/seat change,
   schema/version change, gap, duplicate, or out-of-order row starts a new
   sequence; it never bridges to the previous `K`.
4. A reset row needs a valid observed planted-count baseline. If the baseline,
   adjacency, or delta transition is invalid, exclude the affected proxy
   rather than guessing, clipping, or repairing it.

The full unfiltered history is built before any date or minimum-score filter,
so filters cannot manufacture adjacency. Harvests, losses, unfinished work,
and temporary occupancy do not rewrite `K`; maintenance-aware reconstruction
from those events is deferred.

### Land and animals: observed end outcomes

- The land outcome is the absolute observed unlocked-land count at the end of
  the defined episode/window, not an expansion delta and not a target inferred
  from a purchase.
- Animal outcomes are the species-specific placed counts observed at that end.
  Do not reconstruct them as purchases minus losses: loss/escape/death/culling
  components are excluded from the outcome label. Do not fabricate a sale or
  sale quantity from a count difference; a sale is present only when an
  observed sale event supplies it.
- A missing or physically invalid end observation is an invalid outcome, not a
  default zero or a current-state substitution. End observations are checked
  against Packet 1A support and target domains.

### Required invalid-count report

The lightweight builder emits the following stable integer fields, including
zero values. Future preprocessing must preserve these names and meanings:

```text
crop_delta_outside_vocabulary_components
crop_physical_incompatibility_components
land_invalidity_rows
animal_acquisition_invalidity_components
history_reset_gap_rows
animal_loss_ambiguity_components
animal_loss_ambiguity_rows
target_invalidity_components, target_invalidity_rows
incomplete_ar_chain_rows
excluded_rows, invalid_rows, selection_excluded_rows, component_excluded_rows
```

The `*_components` fields count affected model components and may exceed row
counts. `animal_loss_ambiguity_rows` counts a row once when any animal
decreases; its component field counts each affected species. `excluded_rows`
is the aggregate number of returned rows excluded by selection or by having
no valid component. Date and minimum-score filtering are reported separately
and happen only after history construction and invalid accounting.

`incomplete_ar_chain_rows` counts selected rows that retain at least one valid
diagnostic component but do not form a complete nine-action autoregressive
chain. Such rows are retained separately as partial diagnostics and are never
emitted as complete trainable examples. It is distinct from `excluded_rows`,
which covers rows with no usable component at all.

## Physical validation and exclusions

Every land, animal, and crop action/proxy is validated through the public
Packet 1A helpers and constants. In particular, land uses absolute-target
support; animal prefixes use the sequential Packet 1A housing/reuse rules;
crop deltas use the persistent-goal and residual-capacity support. The
physical context contains no economic state.

Validity is autoregressive. A component is usable only when its own class is
populated, that class is physically supported under the exact preceding
observed class prefix, and every earlier action in the nine-step order
(`land, goose, cow, sheep, wheat, carrot, tomato, strawberry, melon`) is
itself valid. Once a step is invalid, no later step may be treated as
trainable, because doing so would require reconstructing a prefix that was
never observed. A missing or invalid earlier action is never replaced with
current observed counts, HOLD, zero, a clipped action, or any other repair.

A complete nine-action teacher-forcing example therefore requires a complete
valid observed prefix/action sequence. Rows with an incomplete chain are
excluded from the complete trainable output, counted in
`incomplete_ar_chain_rows`, and, when they retain partial observed outcomes,
kept only as diagnostics. Partial diagnostics do not imply trainability.
There is no latent-intent reconstruction, marginalization over missing
prefixes, weak-label learning, missing-action imputation, or probabilistic
repair of the observed sequence.

An unsupported class, invalid transition, impossible end observation, or
missing required component is excluded and counted. It is never clipped,
repaired, replaced with HOLD, replaced with a current value, or made valid by
post-sampling correction. No Packet 1B rule may reserve future crop capacity,
split the shared cow/sheep pasture pool, or change the Packet 1A no-contraction
and full-contraction guarantees.

## Optional curriculum caps

The optional configuration is versioned as `stage25_curriculum_v1` and is
`enabled = false` by default. Its only caps are:

- `max_positive_crop_delta`: relative to the current persistent crop goal,
  allow `delta <= cap` for positive crop deltas; every physically supported
  `delta <= 0` remains available.
- `max_land_expansion_per_decision`: allow an absolute land target no greater
  than `observed_land + cap`.
- `max_animal_additions_per_species_per_decision`: allow an absolute species
  target no greater than `current_placed_species + cap`.

Each curriculum mask is the intersection of its cap with the corresponding
Packet 1A physical mask. A valid intersection must be nonempty; an empty
intersection is a configuration/data error and must fail loudly. Caps never
block necessary contraction (including full crop-goal contraction), and are
not implemented by clipping or repair. The curriculum has no economics,
progression, affordability, purchase, reward, or controller semantics.

For an ablation, rollout and update schedules, seeds, sample order, optimizer
settings, update count, and checkpoint load/save protocol are frozen. The
disabled control must preserve behavior and checkpoint identity. Sampling and
evaluation must use the same effective support for the same state; evaluation
must not silently remove or add curriculum support. The config version and
enabled/cap values are part of run/checkpoint provenance.

## Crop bias and initialization

An optional crop-logit bias is applied symmetrically to the signed delta
vocabulary:

```text
b(delta) = -|delta| / tau,    tau > 0
```

It changes logits only; hard Packet 1A and curriculum masks still decide
support. HOLD (`delta = 0`) has zero bias, and equal-magnitude expansion and
contraction receive equal bias. This initialization should favor small
absolute deltas without putting nearly all initial probability mass on HOLD;
it is an initialization choice, not a permanent penalty or support rule.

Land and animal heads are absolute-target heads. Any initialization prior or
logit bias for them must be state-relative to the current observed land or
current placed count, not a fixed global target prior. Otherwise an apparently
neutral initialization changes with the starting state and is not a neutral
ablation. Crop initialization is relative by its signed-delta definition.

## Crop-capacity shortfall

The shortfall configuration is versioned as `stage25_crop_shortfall_v1` with
defaults:

```text
crop_shortfall_tolerance = 5
crop_shortfall_coef = 0.0
```

`crop_shortfall_tolerance` is a nonnegative integer and
`crop_shortfall_coef` is a finite nonnegative real. The pure helper accepts
only an already-measured aggregate integer `U_crop` and computes:

```text
penalty = -crop_shortfall_coef * max(0, U_crop - crop_shortfall_tolerance)
```

With the default tolerance, shortfalls 0 through 5 are free, 6 contributes
one penalty unit before coefficient scaling, and 15 contributes ten. `U_crop`
is one shared allowance across all five crops, not five independent
allowances. The helper does not infer it from occupancy, raw goal-minus-
occupancy, task queues, movement counts, or economic state. Invalid types,
negative values, non-finite coefficients, and negative tolerances are rejected;
a zero coefficient returns exactly zero.

Future live measurement is deferred to Packet 1C. At each manager boundary,
evaluate outgoing commitments exactly once before accepting/applying the next
strategic decision and attribute the result to that outgoing manager
transition. There are no primitive-turn penalties or duplicate assessments;
terminal closure assesses the final outgoing decision exactly once. Measure
outstanding requested NEW crop capacity that is not yet established, exclude
routine vacancies from harvesting established maintained capacity, and allow
persistently outstanding eligible commitments to incur cost at later
boundaries until completed or cancelled. Cancellation cannot erase a
shortfall already assessed. Animal and land penalties use separate scales and
remain disabled/unimplemented here.

## Packet 1C seams

Packet 1C may wire these contracts into rollouts, trajectories, checkpoints,
training/evaluation adapters, and artifact metadata. It must consume the
Packet 1A action/support API and the Packet 1B versioned proxy, curriculum,
bias, shortfall, and invalid-count schemas without redefining them. It must
preserve sampled class indices separately from signed deltas, absolute
targets, synthetic `K`, physical context, masks, and end outcomes, and must
prove disabled-path behavior/checkpoint identity and sample/evaluation
support parity.

## Non-goals

Packet 1B does not edit Packet 1A, inspect or change executor/PPO internals,
train or tune a policy, add JAX/Torch/PyArrow/NumPy dependencies, change
engine physics, add economics/progression/affordability, invent sale or loss
events, redesign rewards, perform maintenance-aware goal reconstruction, or
define resume/checkpoint wiring beyond the seam and provenance requirements
above. It does not enable curriculum caps or crop bias by default and does
not use invalid-row exclusion, clipping, repair, or filtering as a substitute
for physical support.
