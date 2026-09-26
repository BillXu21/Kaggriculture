# Stage 2.5 crop lifecycle observations

At each pre-decision morning boundary, `crop_capacity[5]` remains the
physical crop-count baseline `B_t` in canonical order `WHEAT, CARROT, TOMATO,
STRAWBERRY, MELON`. Crop classes still encode signed deltas, so the requested
goals are `G_t = B_t + Δ_t`; the next day's baseline is re-read from the
actual board, never from requested goals or a persistent synthetic ledger.

The manager also receives two lifecycle observations:

- `replaceable_today[5]` counts crops safe to remove/replace today. Offline
  replay and live provider both call `replay_daily.lifecycle.replaceable_today`
  on the same canonical morning board.
- `available_crop_slots` is the scalar current-state free crop/build capacity:
  physical capacity under currently unlocked land and currently placed
  animals/structures, minus the current physical crop count. It uses the
  mechanics' existing compatibility rules, including reclaimable weeds and
  crops, and excluding sticky structures. It is morning-state capacity, not
  the later autoregressive residual after choosing land, animals, and crop
  goals.

Both new fields are persisted as `int16` (`replaceable_today: [5]`,
`available_crop_slots: []`) and bounded to `[0,100]`. They do not enter the
corrected-E encoder input. Instead, normalized (`/100`) side-conditioning is
added to the common manager representation before the value head and the
unchanged nine-head autoregressive decoder:

- `replaceable_conditioning: [5, d_model]`
- `available_crop_slots_conditioning: [d_model]`

The architecture is `stage25_policy_v3_crop_lifecycle_capacity`, the
observation schema is
`stage25_corrected_e_own_only_crop_lifecycle_capacity_v3`, and trajectories
use `stage25_trajectory_v3_crop_lifecycle_capacity`. The action schema and
class counts are unchanged. Native checkpoint resume remains strict; old
schemas and architectures are not implicitly upgraded.

An explicit one-way BC-to-PPO initialization seam accepts only supported
physical-morning-baseline BC checkpoints. It copies compatible parameter
leaves exactly, keeps any existing lifecycle conditioner, zero-initializes
only conditioners absent from that source, preserves source provenance, and
discards BC optimizer/RNG state. The transfer is recorded as
`weights_only_architecture_migration` and is never resumable. Unmarked or
synthetic-prior-baseline sources are rejected. The zero initialization keeps
old policy outputs unchanged initially while allowing the new conditioners to
learn during subsequent BC/PPO training.
