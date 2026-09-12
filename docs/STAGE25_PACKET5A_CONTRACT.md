# Stage 2.5 Packet 5A contract

Packet 5A provides the rollout and persistence boundary for native Stage 2.5
policies. It does not implement PPO optimization, league or pool selection,
live shortfall accounting, throughput tuning, or large evaluations.

## Topology and identity

`ParallelSelfPlayRunner` is the inference owner. It keeps the native JAX
parameter tree and compiled nine-head call in the parent process. Spawned CPU
workers own the environment, executor, and `Stage25PlanProvider` lifecycle;
their policy object is an external `RemotePlanPolicy` and does not import JAX
or Torch. Legacy V0/E routing remains explicit and operational.

Each manager boundary creates one immutable request identity containing
episode, seat, day, request ID, behavior identity, pre-decision crop capacity,
physical context, and support payload. A response must return the same
identity and the nine sampled integer classes, nine component log probabilities,
raw-summed joint log probability, action-independent value, and actual
behavior identity. A mismatch is rejected before provider state or trajectory
state changes.

Behavior identity fingerprints parameters, observation/policy/history schema
versions, effective curriculum, and physical-support version. Checkpoint and
provider curriculum are bound before rollout. Executor factory name/version is
recorded with every persisted row.

## Fixed batching and randomness

Stage 2.5 uses the existing physical input preparation and padding machinery.
The launch default is physical batch size 16 and the selected size is included
in runner provenance and inference metrics. Rows are grouped only by exact
behavior identity, sorted by immutable request ID, and padded with distinct
`padding/...` identities. Padding is discarded before worker delivery and can
never create an environment decision or trajectory row.

Sampling folds a stable row token into a request-level PRNG key. Consequently,
reordering rows, changing neighboring rows, and changing physical padding do
not change a real row's stochastic action.

## Trajectory schema

`Stage25TrajectoryBuffer` persists schema `stage25_trajectory_v1` using strict
NPZ arrays plus a JSON sidecar; it is pickle-free and checks version, shape,
dtype, row count, and identity metadata on load. Each row contains:

* exact pre-decision predictive inputs and int16 pre-decision `crop_capacity`;
* original sampled int16 classes `[9]`, component logprobs `[9]`, raw-summed
  joint logprob, and old action-independent value;
* reward, episode/seat/day, immutable row/request identity, seed, trainable
  flag, termination/truncation, and bootstrap-patch state/value;
* policy, opponent, curriculum, operating-history, physical-support/context,
  and executor provenance.

Decoded goals are diagnostics only. Legacy CARE, fertilizer, selling action
tensors, and any frozen quantity model are intentionally absent. Actions,
ledger K, identities, and numeric values survive a save/load round trip.

The provider is the sole K ledger owner. The outgoing manager transition is
recorded before the next decision is accepted, and K is applied once by the
provider. No manager row is created before the first opening/manager boundary.

## Terminal and truncation semantics

At genuine game termination the final outgoing decision receives the terminal
reward exactly once and no bootstrap value. At a smoke/horizon truncation the
final outgoing decision is marked truncated and receives the next-state value
through the existing GAE convention. The value-only path builds a non-
deliverable context and never samples or accepts another plan. Shortfall
accounting and any new reward shaping remain deferred; the selected existing
reward definition is unchanged.

## Stored-action audit

The parent adapter can teacher-force stored classes through the unchanged
native policy using the recorded curriculum and physical context. The audit
compares all component log probabilities, joint log probability, value, and
optionally decoded goals/support without mutating rollout arrays or replacing
stored values. Invalid rows and diagnostic zero placeholders are rejected.

This is the unchanged-weight rollout audit only. The PPO update-path audit is a
Packet 5B seam.

## Validation evidence

From the repository root, the focused commands are:

```powershell
$env:PYTHONPATH=(Get-Location).Path
pytest -q tests/test_stage25_trajectory.py tests/test_stage25_inference.py tests/test_stage25_packet5a_parallel.py
pytest -q tests/test_stage25_provider.py tests/test_stage25_packet4_boundary.py tests/test_rl_manager_parallel.py tests/test_rl_manager_runner.py tests/test_rl_manager_trajectory.py tests/test_rl_manager_gae.py
```

The focused Packet 5A suite covers worker import isolation, one request/K
transition per boundary, identity rejection, fixed padding, row-stable RNG,
trajectory round-trip, seat/episode separation, terminal closure, truncation
bootstrap without another plan, no pre-opening rows, and checkpoint/provider
curriculum agreement. The fast-engine spawned-worker smoke exercises both
seats and multiple manager boundaries with stochastic parent inference and
reports placement coverage; a bounded true-terminal run separately checks
final-transition closure. Any full-game or competitive evaluation remains
outside this packet.

Latest results: the Packet 5A command completed with `23 passed` and the
legacy/Packet 4 regression command completed with `67 passed` (each emitted
only the repository's existing pytest-cache permission warning). A native
spawned smoke run recorded 8 decision requests, 4 value-only bootstrap
requests, 12 logical rows, 6 physical calls, logical batch sizes
`[4, 2, 2, 1, 2, 1]`, physical batch size 16 for every call, 96 physical
rows, 84 padding rows, occupancy `0.125`, 8 trajectory rows, and 20 nonzero
animal-placement classes out of 24. Batch composition can vary with worker
arrival timing; row identity and action RNG do not.
