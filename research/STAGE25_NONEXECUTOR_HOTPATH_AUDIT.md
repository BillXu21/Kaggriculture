# Stage 2.5 Non-Executor Rollout Hot-Path Audit

Status: measured audit + instrumentation only. No production optimization was
implemented.

- Branch: `perf/stage25-nonexecutor-hotpath-audit`
- Base branch/SHA: `perf/stage25-rollout-timing` @
  `ea5287cdd9ad22ba88876c43d1d78c4061c075d9`
- Authoritative workload: 512 games/update, 36 workers x 4 envs, fast batched
  backend, physical inference B32, 20 ms wait, small Stage 2.5 policy, warmed
  legacy executor.
- Scope: `BatchedFastEnv.step()` and Stage 2.5 provider/manager-row
  preparation. Executor internals and parent dispatch were deliberately left
  alone.

## Instrumentation added (opt-in only)

All new timers/counters are behind `--stage25-rollout-profile`. With the flag
off, no timing calls, profile allocations, or semantic changes are added.

- `rl_manager/rollout_profile.py`: new worker detail fields, count fields, and
  a `derived` reconciliation block plus new `normalized` values.
- `rl_manager/runner.py`: reuse `BatchedFastEnv.last_timing_seconds` for the
  three FastEnv subphases; pass the profile into provider preparation/accept;
  time bootstrap preparation.
- `rl_manager/stage25_provider.py`: subphase timers inside
  `_stage_observation()` / `prepare_inference_context()` and a measured input
  freeze copy (time + bytes), plus the accept-side input copy.
- `rl_manager/stage25_ppo_cli.py`: surface the full provider/environment
  breakdown, reconciliation, and normalization.
- `tests/test_stage25_nonexecutor_hotpath_audit.py`: focused schema,
  reconciliation, ON/OFF parity, serialization, and accelerator-isolation
  tests. `tests/test_stage25_packet5a_parallel.py` updated only so its
  `_stage_observation` spy forwards the new optional `profile` kwarg.

New worker detail seconds: `fast_action_encode_seconds`,
`fast_native_step_seconds`, `fast_observation_decode_seconds`,
`stage25_provider_{prelude,step_resolution,previous_execution_validation,
encode_live_inputs,crop_count,unplaced_animals,canonical_board,
physical_context,input_freeze_copy,support_payload,row_token,
context_object_build,accept_input_copy,bootstrap}_seconds`.

New worker counts: `stage25_provider_input_freeze_bytes`,
`stage25_provider_accept_copy_bytes`.

## Part A/B — FastEnv `step()` decomposition

`BatchedFastEnv.step()` already records `last_timing_seconds`
(`action_encode`, `native_step`, `observation_decode`); these are now plumbed
into the profile and reconcile to `fast_batch_step_seconds` (local end-to-end
unattributed remainder ~0.9%).

Local measurements (Windows, Python 3.13, B4 = one worker's 4 envs):

| subphase | e2e legacy executor | PASS microbench |
| --- | --- | --- |
| action encode | 5.1% | 2.9% |
| native step | 3.4% | 2.8% |
| observation decode | 90.6% | 94.0% |

Per-active-env-turn (e2e): encode ~42 us, native ~28 us, decode ~741 us.

**Answer: `batch.step` is dominated by Python observation decoding, not native
Rust.** Applying the measured fractions to the authoritative 218.28 worker-s:

- observation decode ~198-205 worker-s (~10.3-10.6% of worker wall)
- action encode ~6-11 worker-s (~0.3-0.6%)
- native step ~6-7 worker-s (~0.3-0.4%)

Decode allocation volume (`_decode_observation_pair`, one env = both seat
views): empty board 88 objects / ~10 KB; fully populated board 607 objects /
~74 KB. Every tile is rebuilt as a nested Python dict/list; public state is
shared between seat views but private inventories/hands are rebuilt per seat.

Action encoding is small (`_unit_row` / `_market_row` string lookups over
`farmer` + hands + up to 10 market orders), so it is not a target.

## Part C — Thread topology

Rust source: `rust/kaggriculture_env/src/lib.rs`.

- `numThreads=None` -> `pool=None` -> batch helpers schedule on Rayon's
  **process-global** pool.
- `numThreads=N` -> a private Rayon pool of N threads.
- **Both paths are bypassed below `PARALLEL_MIN_ENVS = 128`.** `advance`,
  `observe_all`, and `masks_all` take a plain serial loop when
  `states.len() < 128`.

With 36 workers x 4 envs, every `RustBatchEnv` owns 4 states, so the native
step is serial and never touches Rayon. `numThreads` is effectively a no-op at
this topology; there is no native oversubscription or contention between
spawned workers.

Local B4 microbenchmark (`numThreads=None` vs `1`, 300 steps x 5):

- None: 0.650 ms/step; encode 2.94%, native 3.20%, decode 94.03%
- 1: 0.669 ms/step; encode 2.96%, native 2.92%, decode 93.81%

Identical within noise, confirming the serial path. If `envs_per_worker` were
ever raised to >=128, `numThreads=None` would make all workers share one global
Rayon pool (potential oversubscription), so an explicit per-worker
`numThreads` should be set before crossing that threshold.

## Part D/E/F — Provider preparation decomposition

Controlled full-`prepare_inference_context()` microbenchmark, strict
validation (support materialized), fractions of `prepare`:

| subphase | sparse board | populated board |
| --- | --- | --- |
| support payload (`_support_payload`) | 70.3% | 53.6% |
| encode live inputs | 15.2% | 34.2% |
| physical context | 6.8% | 5.4% |
| crop count + ledger | 2.6% | 2.4% |
| canonical board | 0.9% | 1.2% |
| input freeze copy | 1.0% | 0.8% |
| row token | 0.8% | 0.8% |
| prelude (validation/key/identity) | 0.7% | 0.6% |
| context object build | 0.5% | 0.4% |
| step resolution / prev-exec / unplaced | <0.3% | <0.3% |

Applying to the authoritative 52.03 worker-s:

- support payload ~28-36 worker-s (~1.5-1.9% of worker wall)
- encode live inputs ~8-18 worker-s (~0.4-0.9%)
- board/context second traversal (canonical board + physical context)
  ~3.4-4.0 worker-s (~0.2%)
- freeze copy ~0.4-0.5 worker-s (~0.02%)

`encode_live_inputs` is the largest single manager-input cost, but the
strict-mode `_support_payload` (land/animal/crop support masks) is the dominant
provider-preparation cost overall. The "second board/context traversal" the
packet asked about is small (~6.6-7.7% of prepare), so a shared canonical-board
seam is not justified by measurement; if pursued, the minimal seam would be an
`encode_live_inputs_with_context(...)` returning the canonical board for reuse
by `physical_context_from_board`.

Freeze/accept copies: the freeze loop copies every input array and marks it
read-only; `_commit` then makes a second full copy for retained `last_inputs`.
Measured 6,314 bytes/manager-row (both copies), scaling with hand count
(observed up to ~12.6 KB/row). This is only ~0.02% of worker wall, so
defensive copying is not worth removing now. A future no-copy transfer would
require proving every input array is freshly owned by the encoder and never
mutated afterwards, plus tests that (a) frozen arrays remain read-only, (b)
`_commit`'s retained copy is not aliased by the encoder or executor, and (c)
profile ON/OFF inputs stay bit-identical.

## Part G — Manager acceptance / complete boundary accounting

The existing accept/trajectory/batch-build/daily-utilization/identity metrics
are now surfaced in the audit output. Authoritative snapshot (worker-s):
provider prepare 52.03, remote response wait 111.53, request build 0.60, queue
put 0.11, response stack 0.49. Local small-run accept 0.035, trajectory 0.020,
manager batch build 0.0006, daily utilization 0.0005, identity 0.0003, accept
input copy 0.0007 (all dominated by jax inference in that run). Inference wait
was intentionally not optimized.

## Part H — Parent side

Recorded for completeness and left alone: concat 0.43 s, padding 0.18,
capacity/context 0.10, adapter 9.30, output slicing 0.20, response queue put
0.19. All <0.5% of worker wall.

## Part I — Amdahl upper bounds (aggregate worker wall)

Denominator ~36 x 53.55 = 1,927.8 worker-s (cross-checks: environment
223.46/1927.8 = 11.6%, matching the packet's ~12%).

| candidate | worker wall share | max theoretical speedup |
| --- | --- | --- |
| environment total | 11.6% | 1.131x (+13.1%) |
| `batch.step` total | 11.3% | 1.128x (+12.8%) |
| observation decode | 10.3-10.6% | 1.115-1.119x (+11.5-11.9%) |
| provider prepare | 2.7% | 1.028x (+2.8%) |
| support payload | 1.5-1.9% | 1.015-1.019x |
| encode live inputs | 0.4-0.9% | 1.004-1.009x |
| board/context traversal | 0.18-0.21% | ~1.002x |
| action encode | 0.3-0.6% | ~1.003-1.006x |
| native step | 0.3-0.4% | ~1.003-1.004x |
| input freeze copy | 0.02-0.03% | ~1.0002x |
| parent adapter | 0.48% | ~1.005x (out of scope) |

## Part J/K — Notes

- The strip executor was not profiled; the stable legacy workload was used.
- Profile OFF adds no inner-loop timing calls; ON/OFF produce logically
  identical provider contexts (test-enforced).
- Native FastEnv import succeeded locally (`_kaggriculture_env.cp313-win_amd64.pyd`);
  no missing-extension failures were observed. The extension binary is
  git-ignored and was copied into the worktree only for local measurement.

## Ranked implementation candidates (realistic payoff)

1. **FastEnv observation decode** — the only large non-executor target
   (~10.3% of worker wall; ~11.5% ceiling). Reduce Python object/alloc volume
   or decode lazily per consumer.
2. **Provider strict support payload** — ~1.5-1.9% ceiling; compute masks
   lazily or from already-derived state.
3. **Provider `encode_live_inputs`** — ~0.4-0.9% ceiling.
4. **Provider board/context second traversal** — ~0.2% ceiling; low priority,
   reuse seam only if combined with (3).
5. FastEnv action encode / native step — ~0.3-0.6% each; low.
6. Provider freeze/accept copies — ~0.02%; not worth it.
7. Parent dispatch — out of scope; negligible.

## Explicitly NOT worth optimizing

Action encode, native step, provider freeze/accept copies, `canonical_board`
alone, `unplaced_animals`, `step_resolution`,
`previous_execution_validation`, `row_token`, `context_object_build`,
`crop_count`, parent dispatch, and native `numThreads` at the current B4
topology.
