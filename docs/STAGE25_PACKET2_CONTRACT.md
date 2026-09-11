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

`inputs` contain the own-only corrected-E arrays plus pre-decision
`crop_capacity` and optional `crop_goals`. A caller may instead pass
`crop_capacity` and `physical_contexts` through the explicit keyword seams.
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
capacity-conditioning matrix, the two `D x D` recurrent matrices, recurrent
bias/step embeddings/state-relative scale, and a `D -> 1` value head. For
`L=num_layers` and FFN width `F`:

`N_non_decoder = 12D^2 + 208D + 5 + L*(4D^2 + 2DF + 9D + F)`.

The corrected-E encoder/trunk is
`10D^2 + 192D + L*(4D^2 + 2DF + 9D + F)`; the remaining
`2D^2 + 16D + 5` is the capacity conditioning, recurrent decoder state,
nine-step embeddings/scales, and value head. The normal `L=4, F=384, H=4`
counts are:

| D | decoder | non-decoder | total |
|---:|---:|---:|---:|
| 16 | 43,296 | 8,629 | 51,925 |
| 128 | 337,184 | 884,741 | 1,221,925 |
| 256 | 673,056 | 6,368,005 | 7,041,061 |

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
    p, _encoded(1), c, physical_contexts=(_contexts()[0],),
    crop_capacity=jnp.full((1, 5), 20, dtype=jnp.int16),
    row_ids=jnp.array([7]),
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

Packet 2 has no BC training, native checkpoint conversion/resume, PPO,
trajectory, provider, executor, or TPU support. It launches no training or
evaluation runs and does not add CARE, fertilizer, selling, economic support
masking, post-hoc repair, or a second physical-context implementation.
