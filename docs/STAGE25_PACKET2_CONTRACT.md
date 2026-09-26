# Stage 2.5 Packet 2 policy contract

Status: authoritative for the Packet 2 functional JAX policy seam. Packet 1
mechanics and vocabularies remain authoritative.

## Public API

- `Stage25ModelConfig` is immutable. Its `d_model`, `num_layers`,
  `num_heads`, `ffn_dim`, and `dropout` fields select the corrected-E
  encoder/trunk. `curriculum` is the versioned optional-cap configuration;
  `crop_prior_tau > 0`, `output_init_scale > 0`, and
  `state_relative_scale_init > 0` control initialization. The
  `tiny/small/large` convenience constructors are public.
  They are respectively `(D,L,H,F)=(16,1,1,32)`, `(128,4,4,384)`, and
  `(256,7,8,1024)`; PPO-facing dropout defaults to zero.
- `parameter_spec(config)` returns a zero-valued pytree with the exact
  parameter-tree structure and shapes.
- `init_stage25_params(config, seed=0, encoder_params=None)` returns that tree
  with deterministic JAX initialization. An optional E encoder tree seeds only
  the encoder; legacy heads are discarded.
- `stage25_parameter_count(config_or_params)` returns the count of all leaves
  in the complete native tree and checks the nine decoder pairs.
- `stochastic_act(params, inputs, config, rng=None, rng_keys=None, *,
  reject_invalid=True, physical_contexts=None, crop_capacity=None,
  row_ids=None)` performs one compiled batched encode/decode call.
- `greedy_act(params, inputs, config, *, physical_contexts=None,
  crop_capacity=None, row_ids=None)` performs the same call with masked
  argmax.
- `evaluate_actions(params, inputs, config, actions=None, classes=None, *,
  reject_invalid=True, physical_contexts=None, crop_capacity=None,
  row_ids=None)` teacher-forces stored classes and never resamples or repairs.
  Invalid/out-of-vocabulary classes fail loudly by default. The diagnostics
  mode (`reject_invalid=False`) preserves the exact supplied sequence, returns
  explicit validity, and marks the affected and downstream likelihoods and
  entropies invalid instead of substituting a repaired class.

`inputs` contain the own-only corrected-E arrays plus the pre-decision physical
crop baseline `crop_capacity` (`B`, integer `[B, 5]`, entries in `[0, 100]`),
`replaceable_today` forecast (integer-compatible
`[B, 5]`, canonical order `[WHEAT, CARROT, TOMATO, STRAWBERRY, MELON]`,
entries in `[0, 100]`), and `available_crop_slots` (integer `[B]`, entries in
`[0, 100]`). Both features are materialized at the morning boundary by the
canonical replay/lifecycle and physical-mechanics helpers. The forecast is produced by the shared
`replay_daily.lifecycle.replaceable_today` function in both the canonical
offline adapter and live provider. One-shot crops must be harvest-releasable
by h21; recurring crops must have their final useful production available for
HARVEST -> DIG retirement with the final harvest no later than h20. TOMATO's
final-retirement age is mechanics-derived from the engine constants, using the
same rule as STRAWBERRY; it is not inferred from a Tetsuya removal sample.
The policy normalizes both new fields by 100 and applies learned side
conditioning before the value head and decoder; neither field changes the
corrected-E encoder dimensions or adds an action head. `available_crop_slots`
is current morning free space and is distinct from autoregressive residual
capacity after the sampled/teacher-forced land and animal prefix.

The lifecycle features change the native policy architecture and observation
contract: checkpoint metadata uses
`stage25_policy_v3_crop_lifecycle_capacity` and
`stage25_corrected_e_own_only_crop_lifecycle_capacity_v3`; trajectories use
`stage25_trajectory_v3_crop_lifecycle_capacity`. Older checkpoints and
trajectories fail explicit version checks rather than silently dropping these
inputs. The one-way physical-baseline BC migration is documented in
`STAGE25_CROP_LIFECYCLE_CONTRACT.md`.
The baseline is required and unambiguous: a scalar `[B]` baseline or an
omitted baseline is rejected, there is no separate caller-supplied
`crop_goals`, and the provider derives the baseline from current physical
occupancy at every daily boundary. If the baseline is not embedded in
`inputs`, a caller may pass `crop_capacity` through the explicit keyword seam;
`physical_contexts` may likewise be passed explicitly.

`B` conditions the encoder/decoder and supplies each crop head's delta base
(`goal_i = B_i + class_i - 100`). It does not define available space and need
not fit the footprint: a physical baseline with `sum(B) > C` is valid and the
autoregressive masks force contraction. The physical capacity
`C = B(requested_land) - required_new_housing_cells` is always derived by the
policy from the decoded Packet 1A context (observed placed animals, reusable
empty coops, shared empty pastures, requested land footprint) after the land
and animal actions; it is never supplied by the caller, never taken from
`sum(B)`, never reserved for future crops, and never clamped to zero. A
negative `C` yields empty crop support.

`physical_contexts` are Packet 1A `PhysicalContext` values, one per row;
without them, the physical subset is decoded from the encoded board,
unlocked-prefix, and inventory arrays. `row_ids` are immutable decision
identities. For stochastic calls, `rng_keys` has shape `[B, 2]` and is folded
with each row id and decoder step, so reorder/padding cannot change a row's
sample.

Every action output contains `classes[int32[B,9]]` (callers may store it as
int16), `component_logprobs[float32[B,9]]`, `joint_logprob[float32[B]]`
(the raw component sum), `conditional_entropies[float32[B,9]]`,
`prefix_entropy_surrogate[float32[B]]`, `value[float32[B]]`, and `valid[bool[B]]`.
`validity` is retained as a compatibility alias. Diagnostic `logits` and
`masks` are padded to `[B,9,201]`; `decoded_goals` is `[B,5]`. The nine steps
are exactly:

`land, goose, cow, sheep, wheat, carrot, tomato, strawberry, melon`

with class counts `(4, 101, 101, 101, 201, 201, 201, 201, 201)`. Land/animal
classes are absolute targets; crop class `j` is signed delta `j - 100`.
Support is exact Packet 1A physical support intersected with curriculum
support. Sampling and teacher forcing use the same mask. There is no clipping,
projection, repair, permanent `+25` cap, or economic alteration of physical
support. Value depends on encoded state and pre-decision capacity, never on
the chosen action prefix.

## Parameter accounting

The decoder has one step-specific projection, bias, and action embedding per
head. For hidden width `D`:

`N_decoder = (4 + 3*101 + 5*201) * (2D + 1) = 1312 * (2D + 1)`.

The non-decoder tree is the corrected-E own-board encoder/trunk, a `5 x D`
capacity-conditioning matrix, a second `5 x D` replaceable-today conditioning
matrix, the two `D x D` recurrent matrices, recurrent
bias/step embeddings/state-relative scale, and a `D -> 1` value head. For
`L=num_layers` and FFN width `F`:

`N_non_decoder = 12D^2 + 213D + 5 + L*(4D^2 + 2DF + 9D + F)`.

The corrected-E encoder/trunk is
`10D^2 + 192D + L*(4D^2 + 2DF + 9D + F)`; the remaining
`2D^2 + 21D + 5` is the two conditioning matrices, recurrent decoder state,
nine-step embeddings/scales, and value head. Counts for the `tiny`, `small`,
and `large` convenience constructors (`(D,L,H,F)` of `(16,1,1,32)`,
`(128,4,4,384)`, and `(256,7,8,1024)`):

| config | D | decoder | non-decoder | total |
|---|---:|---:|---:|---:|
| tiny | 16 | 43,296 | 8,709 | 52,005 |
| small | 128 | 337,184 | 885,381 | 1,222,565 |
| large | 256 | 673,056 | 6,369,285 | 7,042,341 |

`parameter_spec` and `stage25_parameter_count` are executable authority; a
transposed-kernel storage convention does not change counts.

## Initialization and executable summaries

Defaults are `crop_prior_tau=25.0`, `output_init_scale=0.02`, and
`state_relative_scale_init=0.08`. Crop bias is exactly
`-|delta| / crop_prior_tau` for delta `-100..100`. At zero crop features:

`P(delta=0)=0.0203628339`, `P(delta=1)=0.0195643958`,
`P(delta=25)=0.0074910680`, and `P(delta=100)=0.0003729583`.
Aggregated over the full vocabulary, `P(|delta|<=5)=0.2012540637` and
`P(delta>0)=0.4898185830`; these are the analytic crop-prior probabilities
before the small random recurrent/output contributions. The state-relative
land/animal scale starts at `0.08`, so expansion odds decay modestly with the
distance from the observed absolute target while all physically supported
current targets remain available.

Land and animal heads use a state-relative modest-acquisition prior rather
than a static low-class bias. This prior is visible at initialization and
does not collapse the support.

Executable tiny summary:

```python
import jax
import jax.numpy as jnp
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params, greedy_act
from tests.test_stage25_policy import _encoded, _contexts

c = Stage25ModelConfig.tiny()
p = init_stage25_params(c, seed=7)
out = greedy_act(
    p, _encoded(1, ledger=jnp.full((1, 5), 20, dtype=jnp.int16)), c,
    physical_contexts=(_contexts()[0],), row_ids=jnp.array([7]),
)
print(out["classes"], out["joint_logprob"], out["value"])
print(sum(x.size for x in jax.tree_util.tree_leaves(p)))
```

## Compiled core and non-goals

The act/evaluate seams use one jitted JAX core for encoding,
nine-step autoregressive generation/evaluation, physical masks, likelihood,
entropy, and value; Python does not perform per-head inference. Native import
and tiny forward must work when importing `torch` raises. Torch is allowed only
in an explicit historical-checkpoint conversion seam.

This packet adds no BC training, checkpoint conversion/resume migration, PPO
logic, executor DIG-policy change, or TPU support. It launches no training or
evaluation runs and does not add CARE, fertilizer, selling, economic support
masking, post-hoc repair, or a second physical-context implementation.
