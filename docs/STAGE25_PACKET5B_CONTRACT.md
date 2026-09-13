# Stage 2.5 Packet 5B contract: native PPO

Packet 5B adds a small native JAX rollout/update loop on top of the strict
Packet 5A trajectory.  It is bounded to complete rollout/update boundaries;
mid-episode and mid-minibatch resume are not supported.

## Batch and objective

`build_stage25_ppo_batch` explicitly selects rows whose `trainable` flag is
set and whose immutable learner behavior identity matches the requested
learner.  Opponent rows are retained by Packet 5A but never become learner
rows merely because they have the same nine-class action shape.  GAE is
computed in chronological manager-day order separately for every
`(episode_id, seat)`.  A terminal has zero bootstrap; a truncation uses its
recorded value bootstrap and stops the recursion at that boundary.  Missing
bootstrap data, invalid classes, nonfinite fields, duplicate days, or an
unclosed episode/seat are rejected before an update.

The reward is the existing Packet 5A manager-boundary reward, with no new
shaping.  `gamma` and `gae_lambda` are per manager decision row.  Advantages
are normalized once over real learner rows by default; padding is excluded
from this normalization and every loss/diagnostic reduction.

The PPO ratio is exactly `exp(new_joint_logprob - old_joint_logprob)`, and the
ordinary clipped surrogate is applied to that joint ratio.  Component
log-probabilities are retained only for audit/diagnostics.  The entropy term
is the policy's named `prefix_entropy_surrogate`; it is not exact updated
joint-policy entropy.  Value prediction is action-independent and its value
head is trainable in PPO (it was frozen only for BC).  Rollout and likelihood
evaluation use the policy's eval mode with dropout disabled.

The configured physical batch size is used for every objective evaluation,
including gradient evaluation and unchanged-weight audit.  A logical
minibatch is mapped over fixed physical chunks; repeated physical padding is
not included in real-row reductions.  Before the first epoch, the objective
reproduces stored classes, component/joint likelihoods, and values.  The
expected initial audit is ratio approximately `1` and KL approximately `0`.

## Identity and checkpoints

The learner and frozen opponent are explicit `Stage25BehaviorIdentity`
snapshots.  Parameters and curriculum remain frozen for collection and all
epochs consuming its rollout.  Local and spawned-worker inference share the
same seed-scoped RNG namespace and immutable request-id row token; worker
count, reorder, and physical padding do not change a real row.  After an
update, the learner parameter fingerprint changes and the next collection
creates a fresh inference adapter; stale adapters/caches are not reused.

Native PPO checkpoints are distinct from inference and BC payloads:
`stage25_ppo_training_state_v1`.  They are pickle-free NPZ archives with a
strict JSON metadata record and atomic same-directory `fsync`/`os.replace`
writes.  They contain the parameter and optimizer trees, RNG, update counter,
rollout seed/progression, PPO/model configuration, curriculum, learner and
frozen opponent parameter/identity snapshots, physical contracts, executor
identity, operating/source provenance, and the explicit completed-rollout/
update resume boundary.  The physical contract separates the action
vocabulary/support schema from the configured inference batch size; exact v1
resume rejects a batch-size override.  Loading requires the exact PPO
optimizer-state template and rejects incompatible paths, shapes, dtypes,
versions, configuration, identity, executor, and physical-contract fields.
Native BC or inference checkpoints may initialize PPO parameters, but always
receive a fresh PPO optimizer.

The checkpoint's corrected operating history (`e_history_version`/`e_identity`)
is distinct from imported source history (`source_e_identity`) and original
source provenance (`source_identity`).  Re-saving a resume carries those
fields forward and stores only a compact immediate `resume_from` reference;
previous checkpoint metadata is never recursively embedded.  Resume validates
both loaded learner and opponent identities against their parameter trees and
the configured runtime executor before collecting a new rollout.

## Commands

From the repository root:

```powershell
$env:PYTHONPATH=(Get-Location).Path

# 1. Bounded real fast-engine spawned-worker smoke: two updates, save/resume,
#    then one further update.  This uses two workers and physical batch 2.
python -m rl_manager.stage25_ppo_cli --scratch --model-size tiny `
  --engine fast --workers 2 --rollout-size 6 --max-turns 144 `
  --physical-batch-size 2 --minibatch-size 4 --epochs 1 `
  --updates 2 --seed 17 --output-dir artifacts/stage25-packet5b-smoke
python -m rl_manager.stage25_ppo_cli --resume artifacts/stage25-packet5b-smoke/latest.npz `
  --model-size tiny --engine fast --workers 2 --rollout-size 6 --max-turns 144 `
  --physical-batch-size 2 --minibatch-size 4 --epochs 1 `
  --updates 1 --seed 17 --output-dir artifacts/stage25-packet5b-smoke-resumed

# 2. Initial real training run from explicit scratch mode.
python -m rl_manager.stage25_ppo_cli --scratch --model-size tiny `
  --engine fast --workers 2 --rollout-size 2 --physical-batch-size 2 `
  --minibatch-size 4 --epochs 1 --updates 1 --seed 0 `
  --output-dir artifacts/stage25-ppo-run

# 3. Resume that completed update and run the next rollout/update.
python -m rl_manager.stage25_ppo_cli --resume artifacts/stage25-ppo-run/latest.npz `
  --model-size tiny --engine fast --workers 2 --rollout-size 2 `
  --physical-batch-size 2 --minibatch-size 4 --epochs 1 --updates 1 `
  --seed 0 --output-dir artifacts/stage25-ppo-run-resumed
```

The CLI prints one flushed JSON record per update with rollout row counts,
terminal/truncation counts, reward and bank summaries, inference metrics,
policy/value losses, entropy surrogate, KL, clip fraction, gradient norm, and
unchanged-weight audit errors.  TPU numerical validation, executor tuning,
league redesign, shortfall shaping, and Packet 6 optimization remain deferred.
