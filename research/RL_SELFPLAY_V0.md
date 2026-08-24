# RL Self-Play V0 — Stage A Rollout Infrastructure (Issue #9)

Date: 2026-08-24. Status: Stage A implemented and locally validated; Stage B
(PPO updates) deliberately NOT started. This document records the exact
architecture, schema, semantics, seams, and local evidence for the new
`rl_manager` package. It is plumbing documentation only — **no policy-quality
claim is made anywhere** (the local games use a tiny random-init E model and
end in a bankruptcy tie).

## Scope

Stage A owns ONLY the RL harness seams around the frozen components:

- `rl_manager/` (new, 11 modules): batched plan-policy protocol, JAX E
  wrapper, deterministic decode, lockstep self-play runner, trajectory
  buffer + strict NPZ/JSON serialization, seed streams, provenance,
  executor factory seam, official-vs-fast parity seam.
- `tests/test_rl_manager_{decode,economics,parity,policy,runner,seeds,
  trajectory}.py` (7 files): 47 passed + 1 skipped (official engine
  dependency absent in this interpreter).

Consumed through public interfaces only, never modified: `executor_v0`,
`opening_book`, `oracle`, `fast_env`, `rust`, `bc_manager`,
`bc_manager_jax`. No private helper is imported (the farm canonicalization
goes through the public `EngineBackend.canonical_state()` seam).

## Module Map

| Module | Role |
|---|---|
| `types.py` | Framework-neutral `BatchedPlanPolicy` protocol (one `plan_batch(inputs, prng_id)` per contiguous request batch), `PolicyIdentity` (name/version/fingerprint; equality on all three fields), `PolicyOutputs` (action tensors + PPO-ready logprob-group/logprob-total/value slots), composition resolver `seat_policies` (`e_vs_e` requires identical identities at both seats). No JAX/torch import. |
| `policy.py` | `JaxEPlanPolicy`: exactly ONE `bc_manager_jax.forward(..., model_variant="E")` call per contiguous batch. Enforces the own-only E contract AT THE WRAPPER SEAM (rejects any leaked opponent-public array or metadata key loudly, stricter than the issue-#8 `validate_inputs`). Deterministic decode; logprob/value slots are diagnostic zeros in Stage A. `params_fingerprint`: sha256 over sorted leaf paths + dtype/shape/raw bytes of the param pytree — snapshot identity without any committed artifact. |
| `decode.py` | Pure-NumPy mirror of the authoritative torch decoder (`executor_v0.manager.decode_daily_plan`): argmax counts, land argmax + 1, sigmoid presence > 0.5, round-half-up expm1 quantity forced to 0 when absent. Parity with torch is enforced by test (`tests/test_rl_manager_decode.py`). Defines `ACTION_TENSOR_SHAPES` (crop/animal/land/fertilizer/care/sell_presence/sell_quantity) and the six stochastic `LOGPROB_GROUPS` (sell_quantity stays a frozen regression output). |
| `runner.py` | Lockstep N-env self-play over `oracle.backend` engines. Groups every day-boundary manager request across all envs/seats by policy identity and answers each `(identity, day)` group with ONE batched policy call. Owns per-episode/per-seat opening agent, executor agent, queued-plan provider, and economic daily-start state — never shared across seats or games. Terminal-only rewards (+1/0/-1 at the final manager transition). |
| `trajectory.py` | Preallocated compact trajectory buffer (append fails loudly on overflow) with strict-schema NPZ serialization + optional JSON sidecar for rich diagnostics. Round-trip validates schema version, array shapes/dtypes, count bounds, and sidecar/npz consistency. |
| `provider.py` | `QueuedPlanProvider`: the runner queues each decoded `DailyPlan` before the seat's first turn of that day; the unmodified executor consumes it via the standard `PlanProvider` protocol. Missing or double-consumed plans fail loudly. |
| `executor_factory.py` | Executor factory/version seam: builds fresh unmodified `executor_v0.make_agent(strict=True)` instances per (backend, seat). Issue #7 can swap V0 -> selected V0.5 by shipping a different factory with zero RL-semantics change. The executor stays entirely outside the RL gradient. |
| `seeds.py` | `SeedStream`: episode/policy/environment seeds are pure functions of `(master_seed, purpose_tag, index)` via `np.random.SeedSequence` entropy composition — independent of worker scheduling or interleaving order. Derived identifiers persist in sidecars. |
| `provenance.py` | Opening identity (name + sha256 of canonical trace + source replay provenance), backend provenance (name/configuration/engine module path), canonical JSON digests. Every stored trajectory ties to the exact stack that produced it. |
| `parity.py` | Official-vs-fast comparison seam: identical composition/seed stack on two backends, compared in order (opening handoff -> per-(seat,day) manager input digests -> decoded plans -> every primitive joint action -> final banks/statuses); first divergence carries seed/backend/seat/day/hour/turn/path/both values/both actions. `official_backend_available()` gates honestly. |

## Trajectory Schema (exact)

Schema version 1 (`TRAJECTORY_SCHEMA_VERSION = 1`). One row per manager
decision/day/seat. All arrays preallocated to an explicit capacity; only
filled rows serialize; `count`/`capacity` are explicit NPZ fields.

Scalars per row:

- `episode_index` int32, `seed` int64, `seat` int8, `day` int16;
- `trainable` uint8, `terminated` uint8, `truncated` uint8, `valid` uint8;
- `reward` float32 (terminal-only patching below), `logprob_total` float32,
  `value` float32;
- `logprob_<group>` float32 for each of the six `LOGPROB_GROUPS`
  (crop, animal, land, fertilizer, care, sell_presence).

Actions per row (`ACTION_TENSOR_SHAPES`, exact encoded components):

- `action_crop`/`action_animal`/`action_land`/`action_fertilizer`/
  `action_care` int16; `action_sell_presence` uint8;
  `action_sell_quantity` int16 `[NUM_PRODUCTS, SELL_BIN_COUNT]`.

Model-facing inputs per row: `input_<key>` arrays pinned EXACTLY to the
canonical own-only E spec derived by calling `bc_manager.live.
encode_live_inputs` itself (`e_input_spec()`) — never re-hardcoded, so the
buffer cannot drift from the encoder. Construction rejects any spec whose
keys/shapes/dtypes differ from the canonical E encoder.

Provenance per row: `trace_digest` 32 raw sha256 bytes (sealed after the
day's primitive turns complete; never left as the placeholder).

JSON sidecar (aligned 1:1 with buffer rows, never model input):
`schema_version`, `npz_schema_version`, `count`, `capacity`,
`run_metadata`, and per-transition JSON-safe records (index, episode_index,
seed, seat, day, policy_id/policy_version/policy_fingerprint, opponent_id,
trainable, plan_json, compact executor_day_diagnostics, trace_digest_hex).
Serialization is `json.dumps(sort_keys=True, allow_nan=False)` (strict
JSON safety enforced at write time); NPZ loads with `allow_pickle=False`.

Automatic artifact provenance (issue #9 A1 correction): the normal save
path is `SelfPlayRunner.save_trajectory_artifact(path, buffer, result)`,
which builds `run_metadata` via `build_artifact_metadata` — callers never
assemble it by hand. The block carries `artifact_schema_version = 1` plus:
per-episode outcome (episode_index, seed, composition, final_banks, margin,
winner_seat, rewards, statuses, transitions, terminated, episode
trace_digest, rollout_recorded trace reference, timing_seconds), opening
name + digest + source provenance, backend/engine name/configuration/module,
executor factory name/version/identifier/version_sha256, per-seat
policy/opponent identities (name/version/fingerprint/identity_id +
trainable), master seed, and manager start day. The low-level
`TrajectoryBuffer.save(run_metadata=...)` API is unchanged for backcompat;
the full primitive trace stays in `EpisodeResult.rollout` and is never
duplicated into the training core.

## Handoff and EconomicHistory Semantics (exact)

- Days 0-3 are driven solely by the committed `standard_mixed` opening
  (literal d0h0..d3h23 playback under one-way guards); diagnostics are
  recorded for both seats. The first manager decision is exactly d4 h0.
- Manager decisions happen once per day d4..d29 per seat = 26 decisions x
  2 seats = 52 transitions per full game (`MANAGER_START_DAY = 4`,
  `TOTAL_MANAGER_DAYS = TOTAL_DAYS - 4`).
- E encoding uses the exact `bc_manager.live.encode_live_inputs` semantics
  through its stateless `economic_prev_start=(day, cash)` path. The runner
  owns the previous daily-start `(day, cash)` observation per seat, reset
  per episode/seat, never inferred from submitted orders.
- Prior-day realized labor comes from the observed `hires_today`
  progression (`workers_hired` + authoritative `total_hire_cost`), never
  from HIRE intents.
- Farm observations are canonicalized through the PUBLIC backend seam
  `EngineBackend.canonical_state()` so fast-engine tile aliases never reach
  the executor or the live encoder. No private oracle helper is imported.

## Rewards

Terminal-only: +1 win / 0 tie / -1 loss assigned to the FINAL manager
transition of each seat (`patch_terminal`), matching the final bank margin
sign; both zero on an exact tie. Raw bank margins stay diagnostics in
`EpisodeResult`. Turn-budget exhaustion (no DONE) marks `truncated` on the
final row instead of `terminated`.

## Provenance and Reproducibility

Every run records: master seed; per-episode derived seed; composition;
per-row policy/opponent identities (name@version:fingerprint12);
parameter fingerprint (sha256 over the full param pytree); opening name +
digest + source-replay provenance; backend name/configuration/engine
module path; executor factory version; manager start day. Episode-level
trace digest = sha256 over the sorted sealed per-day joint-action digests.
All of this is persisted automatically into the trajectory sidecar's
`run_metadata` by `SelfPlayRunner.save_trajectory_artifact` /
`build_artifact_metadata` (`artifact_schema_version = 1`) — no caller-side
assembly. Determinism is test-enforced: two independent full-game runs
produce byte-equal buffers (`equal_nan=True` for the NaN sentinel in
`board_numeric`) and equal episode digests.

## Backend / Executor Factory Seams

- Backends come from `oracle.backend.make_backend(name, configuration)`
  (`fast` default; `numThreads=1` locally). The parity seam runs the same
  stack on a second backend when the official dependency exists.
- Executors come from the injectable `RlExecutorFactory`; the default
  builds strict unmodified `executor_v0` agents and is version-stamped
  (`executor_v0.make_agent(strict=True)@stage-a-v1`). Issue #7 swaps the
  factory, not the RL code.

## First-Divergence Report

`compare_rollouts(a, b)` requires `record_rollout=True` results with equal
seed/composition and returns the FIRST mismatch as a `DivergenceReport`
with phase (`opening_handoff | manager_inputs | plans | actions | banks`),
seat/day/hour/turn indices, field path, both values, and both actions where
practical — enough context to reproduce the divergence from scratch.

## Local Evidence (tiny random-E plumbing only)

- Suite: `python -m pytest tests/test_rl_manager_decode.py
  tests/test_rl_manager_economics.py tests/test_rl_manager_parity.py
  tests/test_rl_manager_policy.py tests/test_rl_manager_runner.py
  tests/test_rl_manager_seeds.py tests/test_rl_manager_trajectory.py
  --basetemp=<tmp>` → **47 passed, 1 skipped** (skip = official
  `kaggle_environments` absent in this interpreter, Python 3.13.1).
- Focused compatibility seams rerun green: `tests/test_bc_manager_jax_parity.py`,
  `tests/test_opening_book_agent.py`, `tests/test_executor_v0_agent.py`,
  `tests/test_oracle_replay.py`, `tests/test_oracle_import_isolation.py`
  → 72 passed, 8 skipped.
- Two deterministic complete fast games (numThreads=1, tiny random-init E,
  E-vs-E): DONE/DONE, exactly 52 transitions (d4..d29 x both seats),
  byte-equal rerun, episode digest
  `fd910f3d8f5a6a8f864b5d76daad87c477611c75742ab44f959c0d946dd88a10`.
  Final banks [0.0, 0.0] — a bankruptcy tie. This is a PLUMBING result
  proving harness correctness only; it says nothing about policy quality.
  Timing split of one pair (~12.7 s wall): manager_inference ~1.06 s,
  agent_actions ~1.08 s, env_step ~0.80 s, orchestration ~0.005 s.
- N=2 lockstep batching proof (truncated chunks, never extra complete
  games): E-vs-E produces ONE policy call per day with a contiguous
  batch spanning both envs (batch size 4 counting seats); candidate/frozen
  compositions produce TWO calls per day (batch 2 each), grouped strictly
  by policy identity — never one call per environment.
- Official engine unavailable in the current interpreter
  (`import kaggle_environments` fails; do NOT install into this
  environment). Exact reproduction command once the dependency exists:
  `python -m pytest tests/test_rl_manager_parity.py -k official -v`.
- Real BC-E checkpoint absent locally (`artifacts/local/bc-v1-E/best.pt`);
  no real-checkpoint rollout has run.

## Eventual Topology (design only — UNMEASURED)

Many configurable CPU env workers -> contiguous day batches grouped by
policy identity -> one batched JAX forward per (identity, day) ->
compact trajectory buffers. There is no Python loop over examples inside
JAX and no per-environment accelerator call; host loops over envs/backends/
executors stay plain Python. Nothing about multi-process dispatch,
throughput, or scaling is measured in Stage A; measurement waits for real
hardware and real checkpoints.

## Stage B1 (PPO V0 core — implemented 2026-08-24)

`rl_manager.ppo_policy` / `rl_manager.gae` / `rl_manager.ppo` /
`rl_manager.ppo_checkpoint` implement the PPO core over the Stage-A schema:
mutable E trunk + independently initialized small value head on the additive
manager-representation seam; immutable frozen-E snapshot that always supplies
sell quantities (exact issue-#8 round-half-up rule, gated by THIS policy's
presence bits); 17 conditionally independent categoricals + 54 Bernoullis
with the raw-summed joint logprob and six-group entropy; vmapped sampling
keyed by explicit per-row decision seeds (`fold_in(root_key, row_index)`);
GAE(lambda) grouped by (episode, seat) in day order with terminal ±1/0
rewards and full-batch advantage normalization before any minibatch split;
masked-AdamW update (frozen leaves get no gradient step AND no decay) with
conventional plumbing defaults (clip .2 / value .5 / entropy .01 / grad clip
1.0 / gamma .99 / lambda .95, all configurable, none tuned); optional
analytic KL-to-frozen regularizer (default disabled); strict pickle-free RL
checkpoint format `rl_manager_ppo_checkpoint_v1` storing params, frozen
snapshot, optimizer state, explicit PRNG stream, step, rollout seed, and
provenance — a resumed state runs the next update bit-identically.

## Stage B2 (Stage-A integration + tiny live smoke + CLIs — implemented 2026-08-24)

New modules (no B1 semantic edits):

- `rl_manager/ppo_adapter.py`: `PPOBatchedPolicy` implements the Stage-A
  `BatchedPlanPolicy` protocol over a `PPOPolicy`. It consumes contiguous
  own-only E arrays plus an explicit string `prng_id`
  (sha256 -> uint32 root key; per-row seeds attach to row identity), offers
  stochastic (training) and deterministic (eval) modes — deterministic mode
  reproduces the frozen JAX-E decode exactly before any drift — returns exact
  action tensors, six logprob groups + total, value, and an immutable
  name/version/fingerprint identity. Also: `ppo_batched_policy_from_state`
  (checkpoint-resume reconstruction reusing EXACT stored params/frozen/rng)
  and `select_ppo_subset` (deterministic evenly-spaced 2–8 row subset taken
  AFTER full-trajectory GAE/normalization, for local one-step smokes).
- `rl_manager/diagnostics.py`: compact strictly JSON-safe diagnostics record
  (`allow_nan=False`) covering rollout seed/composition/steps, timing split
  env/executor/policy/orchestration, return/win/banks/margin, entropy by the
  six groups, approx KL, clip fraction, value loss, explained variance,
  advantage stats, KL-to-frozen action drift, executor unfinished/
  missed-maintenance totals, anomalies/provenance, pre/post policy
  fingerprints, checkpoint path. Unavailable quantities are serialized as
  `null` with machine-readable reasons under `missing`.
- `rl_manager/cli.py`: guarded train/eval commands (`python -m rl_manager.cli
  train|eval`). Train REQUIRES an existing real BC-E torch checkpoint
  (`--e-checkpoint`, fail loud if missing), an explicit executor factory
  selection, backend, master seed, worker/env/thread knobs (default safe 1;
  >1 worker fails loud as not-yet-implemented), episode/update/minibatch
  settings, and output/checkpoint paths. Eval exposes exactly the fixed seed
  sets smoke(17,42,2026)/dev(200..263)/holdout(5000..5031), always plans BOTH
  seat orientations, prints the planned game count, and refuses dev/holdout
  without `--confirm-expensive`. Evaluation output follows one fixed schema
  (W/L/T, paired margins, median/mean banks, per-orientation split,
  anomalies, worst seeds). Tests cover ONLY parsing/planning/aggregation —
  the commands are never executed by tests.

Tiny live smoke evidence (`tests/test_rl_manager_ppo_integration.py`,
ONE complete fast-engine game total, numThreads=1, tiny random-init E):
stochastic PPO candidate vs frozen `JaxEPlanPolicy`,
composition `candidate_vs_frozen` -> DONE/DONE, 52 transitions (26 trainable
candidate rows), terminal reward patched on the final manager row; GAE over
the complete candidate trajectory normalizes advantages to mean ~0; exact
stored-action logprob recompute (`evaluate_actions` == stored
`logprob_total`, bit-equal) BEFORE the update proves no resampling; ONE PPO
update call (epochs=1, one 4-row evenly-spaced minibatch) leaves all metrics
finite, changes base+value params, keeps the frozen snapshot bit-identical;
checkpoint save/load preserves every leaf including the PRNG stream, loaded
deterministic eval equals pre-save eval on a fixed batch, and resumed-state
next-update is bit-identical; diagnostics artifact written and re-parsed.
Deterministic-mode adapter parity with frozen E is proven separately on a
fixed synthetic batch. Official-engine and real-checkpoint paths are gated
skips recording their rerun conditions.

Validation totals (this packet): rl_manager suite + focused issue-#8 JAX
parity/train = 130 passed + 4 skipped in ~187 s (skips: official engine
dependency absent x2, real BC-E checkpoint absent x2). Complete fast-engine
games executed by the new B2 tests: exactly ONE. No quality claim anywhere:
the smoke uses a tiny random-init E and one gradient step; serious training
remains blocked on issue #7 (executor selection) and the real BC-E
checkpoint.


