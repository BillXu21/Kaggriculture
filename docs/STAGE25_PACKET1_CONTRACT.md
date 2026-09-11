# Stage 2.5 Packet 1 — Authoritative Contract

Status: **authoritative for Stage 2.5 Packet 1**. This document consolidates
Packets 1A, 1B, and 1C. Where older Stage 2.5 proposal text conflicts with this
document, this document governs. Packet 1A remains authoritative for mechanics
and physical support; Packet 1B remains authoritative for BC data, curriculum,
and crop-shortfall semantics.

## Fixed action schema

The once-daily manager acts in this immutable order:

`land, goose, cow, sheep, wheat, carrot, tomato, strawberry, melon`

The corresponding class counts are:

`(4, 101, 101, 101, 201, 201, 201, 201, 201)`

Land and animal actions are absolute targets. Crop class `j` means delta
`j - 100`, so class `100` is HOLD and the fixed crop vocabulary is
`[-100, +100]`. The persistent crop-capacity ledger transitions exactly once as
`K' = K + delta`; sampled classes are never clipped, projected, or repaired.
There is no permanent `+25` cap.

## Observation contract

The manager makes one decision at each daily boundary. The policy observes
only the acting seat's corrected-E inputs; opponent-private state and opponent
actions are never inputs. Persistent `crop_capacity[5]` is an additional
strategic-state input. It is stored pre-decision, before the nine classes are
sampled, and is updated only by the sampled crop deltas after that decision.

For Packet 2/5 support evaluation, the stored own-only observation must retain
the authoritative physical fields needed by Packet 1A: canonical 10x10 board
kind/crop/animal state, unlocked-quadrant prefix, shed counts, and carried
inventory counts. The current corrected-E trajectory-facing arrays are:
`board_kind`, `board_crop`, `board_animal`, `board_numeric`, `board_bool`,
`board_mask`, `scalars`, `shed_counts`, `seed_counts`, `carried_counts`,
`unlocked`, `market_inventory`, `market_prices`, `shop_counts`, `day`,
`days_remaining`, and `economic_context`. Packet 2 must make the physical
subset losslessly and deterministically decodable back to Packet 1A's board,
unlocked prefix, and inventory representation. Therefore physical context is
not redundantly stored when this round-trip guarantee is met; `K` alone is
never sufficient. A round-trip test is an acceptance gate before rollout.

Economic channels may remain in corrected-E for the encoder, but money, prices,
feed, labor, affordability, and profitability must not alter permanent
physical support.

## Executor seam and selling

Stage 2.5 v1 has no CARE, fertilizer, or selling policy action. At the executor
seam it relies on a versioned deterministic aggressive liquidation/upkeep
procedure. Learned market timing is explicitly future work and may be a
separate small model. No Stage 2.5 PPO likelihood, trajectory action, or model
head may contain legacy CARE/fertilizer/sell fields.

## Packet 2 JAX model contract

Stage 2.5's future model reuses the current corrected-E own-board encoder/trunk
where compatible and adds `crop_capacity[5]`. Its decoder is one shared
recurrent/autoregressive core with step-specific output projections and action
embeddings. The order is fixed to the nine steps above: one land step, three
animal steps, and five crop steps. The semantic shapes at hidden width `D` are:

| step family | output projection | bias | action embedding |
| --- | --- | --- | --- |
| land | `[D, 4]` | `[4]` | `[4, D]` |
| each animal | `[D, 101]` | `[101]` | `[101, D]` |
| each crop | `[D, 201]` | `[201]` | `[201, D]` |

Projections are step-specific; sharing one projection across same-sized heads
is not equivalent. For this contract's output/embedding parameter accounting,
the specified class aggregate is the normative constant `4 + 8*101 = 812`, not
the earlier incorrect 1312 total. This accounting constant is intentionally
pinned separately from the per-step class-count tuple above and must not be
re-derived from that tuple. Output weights, biases, and action embeddings together contain
`812 * (2D + 1)` parameters: `208,684` at `D=128` and `416,556` at `D=256`.
These are semantic shapes; a repository storage convention may transpose a
kernel.

One compiled JAX call must perform encoding, autoregressive action generation,
Packet 1A physical support masking, conditional logprob evaluation, and value
output. There must be no Python-level per-head inference loop. The value is
action-independent unless Packet 2 deliberately versions this contract.

## PPO likelihood and entropy

For stored state `s`, persistent ledger `K`, physical context, and action prefix
`a`, the joint likelihood is the raw sum of conditional logprobs:

```text
log pi(a | s,K) = sum_i log pi_i(a_i | s,K,a_<i)
```

Sampling and stored-action evaluation use the same conditional decoder
equations and the same effective support mask. Evaluation is teacher-forced by
the stored prefix. Trajectories store sampled class indices, not projected or
repaired actions. PPO uses the raw joint summed logprob for its ratio; it does
not average per-step logprobs.

Conditional categorical entropy at each step is exact for that conditional
distribution. Summing those entropies along one stored behavior prefix is not
the updated-policy full joint entropy. Stage 2.5 v1 names this metric
`prefix_entropy_surrogate`: the sum of conditional entropies evaluated along
the stored behavior prefix. It must not be labeled exact joint entropy. PPO
entropy implementation is outside Packet 1.

## Initialization

Crop output biases use a trainable small-change prior, for example
`bias(delta) = -|delta| / tau` with `tau > 0`. Kernel/output initialization
must be small enough for this prior to be visible, while the initialization
must not collapse nearly all probability onto HOLD.

Land and animal outputs are absolute targets. A static low-class bias is not
generally equivalent to favoring modest additions. Packet 2 must intentionally
implement a state-relative modest-acquisition initialization for these heads;
this is a requirement, not a Packet 1 implementation.

## Curriculum behavior identity

Packet 1B's curriculum is disabled by default and versioned as
`stage25_curriculum_v1`. Its effective support is always:

```text
Packet 1A physical support ∩ curriculum support
```

Sampling and stored-action evaluation must intersect the identical supports.
The exact curriculum enabled flag and cap settings contribute to behavior
identity/fingerprint. Checkpoint metadata pins the curriculum version/settings.
The curriculum is fixed for a complete rollout/update cycle and can change
only between cycles; silent mid-batch or mid-update changes are invalid. The
disabled path must preserve physical support and checkpoint identity.

## Future trajectory and provider contract

Packet 2/5 must migrate the trajectory schema; Packet 1 does not perform that
migration. The minimum stored behavior state is:

- pre-decision `crop_capacity[5]`, `int16`;
- sampled class indices `classes[9]`, `int16`, in the fixed action order;
- the complete own-only corrected-E observation inputs listed above;
- episode, seat, day, and an immutable row identity/decision seed;
- `stage25_physical_v1`, policy/action schema version, and
  `stage25_curriculum_v1` plus exact settings;
- executor profile identity, when deterministic behavior depends on it;
- old joint logprob, value, reward, termination/truncation, and diagnostics as
  currently appropriate.

The class indices are the authoritative stochastic actions. Signed deltas and
decoded goals may be stored as diagnostics only. Because the authoritative
physical observation round-trips to the Packet 1A representation, no duplicate
`PhysicalContext` is required; if that round-trip cannot be guaranteed for a
future encoder, the migration must instead store the exact Packet 1A physical
context fields explicitly before training proceeds.

The current `TrajectoryBuffer` stores legacy seven action tensors and six
logprob groups, while `InferenceRequest` carries encoded inputs plus
episode/seat/day/policy/prng metadata and `QueuedPlanProvider` keys plans by
day. These are inspection findings, not the Stage 2.5 schema. Packet 2/5 must
replace/extend those seams so the nine class sequence, row identity, physical
context, pre-decision ledger, and behavior fingerprint travel together.

## Row randomness, batching, and numerical acceptance

Randomness is stable per immutable row identity. Changing batch composition,
row order, worker assignment, or padding must not change the sampled action for
that row. Fixed physical inference batches remain supported; padding rows are
discarded before trajectory insertion. Support masks must be identical during
rollout sampling and teacher-forced evaluation.

An unchanged-policy audit currently observes joint-logprob discrepancies of
about `0.0252` for small BC and `0.0337` for large BC under dynamic physical
rollout batches; that failed audit printed `physical_batch_size=None`. A fixed
physical-batch test is pending. This packet does not weaken numerical
tolerances. Packet 2/5 acceptance requires unchanged-weight stored actions to
reproduce rollout logprob through the actual update evaluation path before
serious TPU training.

## Shortfall live seam

Packet 1B defines `stage25_crop_shortfall_v1`, with default tolerance `5` and
coefficient `0.0`. Eligible shortfall is outstanding requested NEW crop
capacity not yet established. It excludes maintenance vacancies caused by
harvest, raw goal-minus-occupancy, task-queue length, and movement backlog.

Future live integration calls the pure helper at each manager boundary in this
order:

1. assess outgoing strategic commitments;
2. compute already-measured eligible crop shortfall;
3. apply and log the optional shortfall penalty to the outgoing manager
   transition;
4. only then accept, sample, and apply the next plan.

There are no primitive-turn charges and no duplicate boundary evaluation.
Terminal closure assesses the final outgoing decision exactly once. Cancellation
cannot erase shortfall already assessed at the outgoing boundary. Animal and
land penalties remain unimplemented, and the coefficient remains zero by
default. Future live state must provide the outgoing commitment's requested
new-capacity evidence, what capacity has already been established, and a
boundary/commitment identity sufficient to deduplicate assessment. Packet 1
does not build a tracker or wire a provider/executor.

## Native JAX checkpoint direction

Stage 2.5 is JAX-native. Packet 2/3 native checkpoint metadata must pin at least:

`architecture_version`, `action_schema_version`, `observation_schema_version`,
`persistent_ledger_version`, `physical_support_version`, curriculum
version/settings, E-history version, ordered vocabularies, model dimensions,
dtype/precision, executor profile identity where behavior depends on it,
training stage/step, optimizer state/RNG for resumable snapshots, and the
historical source checkpoint identity when importing one.

Normal native Stage 2.5 startup must not require Torch. Torch belongs only in
an explicit historical-checkpoint conversion/import seam. Packet 1 does not
implement the new neural checkpoint format; the current strict parameter-tree
checkpoint remains a Packet 2/3 integration boundary.

## Cross-packet authority and audit

The focused Packet 1A and 1B modules/tests remain the source for their approved
semantics: `rl_manager.stage25_mechanics`, `stage25_config`, and
`stage25_data`, with their corresponding focused tests. This document is the
single authoritative Stage 2.5 Packet 1 consolidation. Older proposal and
research notes are historical context only; they must not be used as current
authority for action vocabulary, physical masking, curriculum identity, or
shortfall behavior.

The Packet 1C audit checks framework-free imports, agreement on action order and
class counts, version references, disabled-curriculum identity, HOLD class
100, exclusion of legacy CARE/fertilizer/sell PPO fields, and absence of old
absolute-crop-output, permanent `+25` cap, or unmasked physically impossible
joint-plan authority. It does not implement a neural model, autoregressive JAX
decoder, BC/PPO training, trajectory migration, live provider/executor wiring,
live shortfall tracking, a curriculum controller, TPU benchmarks, or training
runs.
