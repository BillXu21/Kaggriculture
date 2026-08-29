# Kaggriculture Historical Record

## 2026-08-28 — Issue #21 Multi-Trainer TPU Prototype

Added `rl_manager/multitrainer_benchmark.py`, a standalone one-process JAX
benchmark for independent PPO trainers at N=1/2/4 (optional N=8). Each trainer
gets a separate state/optimizer/RNG and is explicitly assigned to
`jax.devices()[i]`; parameters, optimizer state, inference outputs, and PPO
arrays are placement-checked. The report separates first-call compilation from
synchronized steady state and compares sequential updates with dispatch-all-
then-block timing. N greater than visible devices is skipped explicitly rather
than reusing a device.

Added `rl_manager/multitrainer.py` as a JAX-free identity routing seam. Requests
are grouped by exact `PolicyIdentity`, and trainable trajectory rows are
partitioned only when sidecar identity fields match a registered trainer;
unknown, duplicate, mismatched, or contaminated rows fail loudly. No changes
were made to `parallel.py`, and no rollout worker imports JAX/libtpu.

Focused CPU routing/configuration validation passed **5 tests**. A tiny CPU
benchmark smoke completed and reported one device for the existing unsharded
single-trainer path. TPU throughput, overlap, and scaling remain unmeasured.
The exact Kaggle commands are in `research/RL_MULTI_TRAINER_TPU.md`.

## 2026-08-28 — Issue #17 Central Mixed-Day Inference Batching

Added opt-in central inference batching for `ParallelSelfPlayRunner` without
changing the worker/JAX ownership topology or defaults. `RunnerConfig` now
accepts `inference_batch_scope` (`policy_day` default or `policy`),
`fixed_inference_batch_size` (`None` default or positive B), and
`inference_batch_wait_seconds` (20 ms default). The owner sorts real requests
by episode/seat/day/request ID before deterministic B-sized chunking, duplicates
the first valid real row for short physical batches, assigns deterministic
padding row IDs, and routes/records only real rows. Metrics retain the old
fields and add real/physical counts and histograms, padding, occupancy, queue
wait, and inference timing.

`PPOBatchedPolicy.plan_batch_with_row_ids` now uses a policy-snapshot root plus
stable row-ID seeds, so stochastic real-row action/logprob results do not vary
with neighboring rows, padding, arrival order, or scheduler grouping. Train/eval
CLI flags and the CPU benchmark helper expose scope, fixed size, wait, and
Cartesian sweeps. Focused validation passed **37**, skipped **2**; the full
`test_rl_manager_*.py` suite passed **130**, skipped **3**. A local spawned
fast/mock smoke with 2 workers, policy scope, B=4, wait=2 ms completed 4
truncated episodes with 16 real rows in four physical B=4 calls at occupancy
1.0; direct CPU mixed-day tests coalesced days 8/15/21 into one real batch.
No TPU speedup or real-checkpoint throughput claim is made.

This file is append-only except for correcting factual errors. New entries are added in reverse chronological order.

## 2026-08-27 — Throughput Branch Integration

Created clean worktree `Kaggriculture-throughput-integration` on branch
`throughput/integration` from base
`e63e8337ba9e30a6f394d69da23da538ed7ad6c2`, then merged issue #15, issue #16,
and issue #17 in the requested order with normal merge commits. The only
source conflicts were additive runner/documentation overlaps: both executor
flags and the native batch flag were retained, and the parallel worker hook
was retained. Batch workers now use the shared observation adaptation path;
the spawned default factory preserves low-telemetry configuration; train/eval
CLI plans expose low-telemetry, read-only, and batch-backend options.

Focused validation passed `150 passed, 3 skipped`; normal full validation passed
`757 passed, 104 skipped`. The official-capable full run passed `844` and
skipped `15`, with the pre-existing unbuffered-official plan-sidecar parity
failure plus a separate editable-venv submission-path failure. A bounded
constant-policy complete-game parity run matched all three scalar/batched
configurations at 719 primitive actions, 52 manager rows, `DONE/DONE`, final
banks `[3.0, 3.0]`, and action SHA256
`0139dfc0e76755c7c8227a1b4475900026ca1a40bffd46adad4d5c07acbb9869`.
The two-worker integrated smoke returned 4/4 episodes, 16 manager requests,
two owner batches of 8, and no routing or worker errors. Local CPU benchmark
results and the exact TPU command are recorded in the final integration
report; no TPU result is claimed.

## 2026-08-27 - Issue #15 Executor Hot-Path Profile and Optimization

Created isolated worktree/branch `throughput/15-agent-hotpath` from base
`e63e8337ba9e30a6f394d69da23da538ed7ad6c2` under
`.worktrees/throughput-15-agent-hotpath`. The dedicated profiler measured the
complete BC-E self-play agent path at N=1/N=2 before and after optimization.
Before optimization, ordinary turns repeatedly built canonical boards (5,036
calls over one N=1 profile), runner observation deep copies were the largest
single cProfile family, and per-turn debug snapshots were the next meaningful
diagnostic cost. Opening wrapper work was not material.

Accepted changes reuse one canonical own-board snapshot through task
generation, feed state, sells, optional cleanup, day setup, and achieved
diagnostics. Existing standalone task APIs still canonicalize when no snapshot
is supplied. `AgentConfig.record_turn_snapshot` preserves the old default;
`RunnerConfig.low_telemetry=True` disables that expensive snapshot for explicit
training runs. `RunnerConfig.read_only_agent_observations=True` provides safe
dict/list-shaped read-only views, avoiding per-call deep copies while rejecting
accidental nested mutation; the old deep-copy path remains default.

Local uninstrumented medians (one warmup, two repeats, fixed seeds, fast engine,
`numThreads=1`) were base versus optimized default: N=1 `3.681 -> 4.357` s
(`0.272 -> 0.229` games/s, noisy regression not claimed), N=2
`7.656 -> 6.892` s (`0.261 -> 0.290`), and N=8 `28.630 -> 20.605` s
(`0.279 -> 0.388`). Explicit low-telemetry/read-only training mode measured
N=1/2/8 at `2.315/4.933/19.118` s and `0.432/0.405/0.418` games/s. No TPU
claim is made.

Parity passed: base and optimized seed-17 complete games both ended
`DONE/DONE`, final banks `[28505.0, 25662.0]`, and 719-joint-action SHA256
`897ba48bed992da5461a826a621077bc1a5af76c719a92718303d77882f83ad8`; the
N=2 seed-42 bank/digest also matched. Focused executor/runner tests passed
`51`, with one existing skip; Ruff and compile checks passed. No issue #16/#17
files or interfaces were changed.
## 2026-08-27 — Issue #17: Parallel Rollouts with One JAX Owner

Implemented the local multiprocessing topology on branch
`throughput/17-parallel-rollouts` from `e63e8337`. The parent retains policy
snapshots and central inference; Python `spawn` workers build independent
backend/opening/executor state and exchange only encoded manager-day NumPy
rows through bounded queues. Requests are routed by stable
`episode/seat/day/policy` IDs, centrally sorted and batched by policy identity
and day, and worker trajectory shards/results are normalized by episode/seat/day.

The lazy `rl_manager` package initializer and worker startup guard prevent
`jax`, `jaxlib`, `torch_xla`, `bc_manager_jax`, and `optax` from loading in CPU
workers. Worker exceptions, abnormal exits, missing/duplicate episodes, and
duplicate trajectory rows fail loudly and all children are joined. The default
executor factory is reconstructed inside workers; custom factories must be
spawn-pickleable. PPO's optional row-aware policy seam derives stable sampling
seeds from logical row IDs.

Tests: the focused CLI/process suite passed `20 passed, 1 skipped` before the
native extension was copied into the isolated worktree; with the freshly built
local extension, the spawned parallel smoke passed `4 passed`, including two
workers with two environments per worker, central batching, deterministic
result normalization, trajectory-row completeness, and worker import
isolation. TPU throughput was not measured locally. The exact Kaggle command,
benchmark script, and scaling fields are documented in
`research/RL_MANAGER_PARALLEL_ROLLOUTS.md`.

## 2026-08-27 — Stage 6/7 Issue #12: PASS-Only Idle Cleanup A/B/C Evidence

Recorded the official 24-game A/B/C panel for the true PASS-only cleanup layer
at `1a0ac65` on branch `executor-v07-fixed-plan`. The earlier optional result
was not promotion evidence: optional tasks had been placed in the normal
foreman pool, so ordinary assignment selection (including underfoot selection)
could change instead of replacing only a literal normal PASS; that flawed arm
was `60,948.5` mean vs OFF `63,592.3` (`-2,643.8`, `8W/16L`).

The corrected flow runs the normal foreman first, recomputes optional
candidates from the current observation, and applies them only to workers left
at literal PASS. Claims are discarded after the turn and excluded from normal
task, hiring, debt, and market accounting. `optional_spare_watering` selects
WATER-only; `optional_idle_cleanup` selects weed-first DIG plus optional WATER;
diagnostics record `cleanup_mode`.

Official panel results (same BC-E/checkpoint, official 1.32.7 backend,
`standard_mixed`, PASS, seeds
`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`, both seats, prior-debt
ON, aggressive selling ON): OFF mean/median `63,592.3/65,509.5`; WATER-only
`67,712.6/65,242.0`, delta `+4,120.3/-267.5`; WEED+WATER
`66,886.3/66,662.5`, delta `+3,294.0/+1,153.0`. Both optional arms were
`16W/0T/8L`; worst deltas were `-15,820` and `-16,669`, best deltas
`+30,696` and `+27,821`. Full per-game banks, paired deltas, artifacts,
telemetry, and two first-divergence timeline examples are in
`research/EXECUTOR_V07_IDLE_CLEANUP_PASS_ONLY.md`.

Telemetry confirms mechanical isolation: OFF/B/C baseline PASS actions
`4035/3748/3558`, replacements `0/3409/3298`, replacement rates
`0/90.96%/92.69%`, remaining PASS `4035/339/260`, cleanup movements
`0/2973/2930`, optional WATER interactions `0/436/210`, weed DIG interactions
`0/0/158`, and normal non-PASS actions changed `0/0/0`. Unaffordable orders
were `0/174/168`; animal-loss events were `0/1/0`; all 72 games completed
719 transitions without fallback/day/status errors.

Interpretation: mechanical safety passed, but strategic value is mixed and
seed-dependent. WATER-only has a slightly negative median delta and
WEED+WATER a positive median, yet neither satisfies the strict strong-evidence
promotion rule because of the eight losses and materially worse tails. Cleanup
remains OFF by default; independent review is the next step, not PPO.

## 2026-08-27 — Stage 7 Issue #12C Opportunistic Watering Audit

The bounded opportunistic-watering implementation passed a requirement-by-
requirement mechanical audit at repository HEAD `c676fcf53d4064667fb454bfc20b7d5d59b7f6d6`.
Weed-boundary WATER remains `MAINTENANCE`, yield-positive WATER remains
`PRODUCTIVE`, and optional WATER is `OPTIONAL`; candidate generation excludes
watered, urgent, malformed, locked, and non-plant cases. `AgentConfig` defaults
`optional_spare_watering=False`, optional dispatch is explicitly gated, all normal
priority classes remain ahead of optional work, starvation filtering remains in
place, Stage 4 dependencies remain keyed as `WATER:`, and optional work is absent
from hiring/debt/pending accounting while remaining traceable.

Existing focused tests cover default-off behavior, enabled spare-worker dispatch,
priority ordering, nearest/distinct-worker selection, malformed/yield exclusions,
and debt isolation. The only identified gap is no maximum-distance clamp: a lone
far optional target can consume walking turns when explicitly enabled. This is a
strategic threshold choice rather than a proven mechanical defect. No source/test/
tool/config behavior change was made and no 24-game A/B was run.

Defer the exact same fixed 24-game panel with optional watering OFF versus ON
(seeds `7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`, both seats, same
BC-E checkpoint/source/config, PASS, `standard_mixed`, fast backend, prior-debt
suppression ON, aggressive selling ON, turn trace ON). Compare banks, movement,
missed maintenance, debt, and errors/status anomalies; consider a distance clamp
only after that evidence. Full note: `research/EXECUTOR_V07_STAGE7_ISSUE12C_OPTIONAL_WATER_AUDIT.md`.

## 2026-08-27 — Stage 6 Issue #12B Spawn/Shed Wait Non-Reproduction

The isolated Stage 4 24-game turn-trace artifact was scanned for consecutive
per-worker `PASS` runs of at least eight turns. It contains 69 such runs: 24
shed-zone runs, exactly one per game, all worker index 1 at `[4,5]` on
`d4h1`–`d4h10`; the other 45 are hiring-ramp or end-game labor-surplus waits.
The representative seed-7/seat-0 reconstruction shows an affordable d4h0
WHEAT buy followed by fill latency while other workers clear feed/starvation
work; the worker resumes at d4h11. There was no animal loss, critical work loss,
unaffordable order, fallback/day/status error, or repeated pickup overdraw.

The behavior is legitimate market-fill/task-availability waiting, not Kaggle
AFK and not a bookkeeping defect. The `cf282a1` shed-budget reservation is
present and prevents concurrent over-pickup. The artifact lacks per-turn
`assignment.reason` and complete private shed/carried-inventory snapshots, so
this remains a bounded classification. It is distinct from the resolved Kaggle
AFK packaging omission. No instrumentation or executor behavior change was
justified; detailed evidence is in
`research/EXECUTOR_V07_STAGE6_ISSUE12B_SPAWN_SHED_WAIT.md`.

## 2026-08-27 — Stage 5 Accepted / Current Archive Compatibility Identity

The fixed Stage 5 panel is accepted by root judgment with the retained Stage 4
life-cycle sequencing (`b9c88ff`). The isolated 24-game panel remains fully
recorded in `research/EXECUTOR_V07_STAGE5_POST_STAGE4_REGRESSION.md`: mean
`63,592.3` vs actual pre-behavior `60,778.1` (`+2,814.2`), median `65,509.5`
vs `60,956` (`+4,553.5`), six negative per-game deltas, no `<1k`/`<10k`
cases, and no errors, status anomalies, unaffordable orders, or animal losses.
Independent paid review classified the negative cases as labor/cash tradeoffs,
not feed/starvation failures; no concrete distinguishing feed exception exists,
and a cash forecast would violate the stated constraints.

Stage 3 fertilizer retention remains rejected and cleanly reverted at
`7204103`: the combined Stage 3+4 experiment drove all 24 banks to zero, so
aggressive mode again sells FERTILIZER. The combined rejection evidence and all
negative cases remain preserved; no Stage 3 retention or strategy change was
introduced by this compatibility update.

The exact archive was intentionally rebuilt from current source revision
`11ecead2d5efe8bf87fc0da533c739e344d7eaa6` using the authorized read-only BC-E
checkpoint (`best.pt`, SHA-256
`f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`):

| revision | archive SHA-256 | official seed-7/seat-1 bank | action fingerprint |
|---|---|---:|---|
| pre-behavior `8f716bec` | `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf` | `54,439.0` | `516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40` |
| current `11ecead2` | `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2` | `47,290.0` | `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc` |

The current archive remains the ignored
`artifacts/local/submissions/bc-e-v07.tar.gz` with 50 members. The final exact
verifier used repository-local `.venv` `kaggle-environments==1.32.7`, raw
`get_last_callable`, strict mode, fresh extraction, repo-root import isolation,
and a full status scan. It passed with 720 status-history entries, zero
anomalies, 719 candidate actions, no repository-root source origins, and
official 1.32.7 provenance.

## 2026-08-27 — Issue #13 Stage 1 Acceptance Follow-up

Completed the previously partial acceptance with the repository-local
`.venv`. The exact installation command was
`& '.venv\\Scripts\\python.exe' -m pip install --disable-pip-version-check
'kaggle-environments==1.32.7'`; it succeeded and the version check reported
`1.32.7` from `.venv\\Lib\\site-packages\\kaggle_environments\\__init__.py`.
The existing repository runtime command
`& '.venv\\Scripts\\python.exe' -m pip install --disable-pip-version-check
-r requirements.txt` was also needed for the declared `torch`/`pyarrow`
runtime dependencies; no dependency manifests changed.

Final exact command:
`& '.venv\\Scripts\\python.exe' tools\\verify_submission.py
artifacts\\local\\submissions\\bc-e-v07.tar.gz`

Result: **PASS**. The exact ignored archive
`artifacts/local/submissions/bc-e-v07.tar.gz` retained SHA-256
`4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf` and 50
members. The raw extracted `main.py` was loaded through
`kaggle_environments.agent.get_last_callable`; official provenance matched
package `1.32.7`, upstream commit
`28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`, and the pinned interpreter file
hashes. Strict mode was enabled with `KAGGRICULTURE_SUBMISSION_STRICT=1`.
Candidate seat 1 versus PASS, seed 7, produced bank `54439.0`, 720 complete
status-history entries, zero anomalies, 719 candidate actions, and action
fingerprint
`516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40`.
Repository-root source paths were absent after child-path sanitization and
all six local runtime packages loaded from the extracted archive.

The first real run found and fixed the bounded verifier issue allowing the
repository-local editable path only as `.venv` third-party site-packages; the
final run proved the path assertion and fingerprint pin. Focused
`tests/test_submission_tools.py` remained **4 passed**, including loud
omitted-`fast_env` failure, and Ruff remained clean. This follow-up is the
small tracked acceptance commit after `1d0bffd72b160a23c0122b03791900212133da9f`;
no executor strategy or Stage 2+ behavior changed.

## 2026-08-27 — Issue #13 Stage 1: Reproducible BC-E Archive Builder and Verifier

Implemented the packaging/runtime invariant before any executor behavior
change. `tools/build_submission.py` uses the tracked
`tools/submission_main.py`, verifies the authorized BC-E checkpoint, discovers
and enforces the six local runtime packages (`executor_v0`, `bc_manager`,
`opening_book`, `oracle`, `replay_daily`, `fast_env`), stages the checkpoint
only as archive-root `best.pt`, includes opening-book data, and writes a
deterministic gzip/tar archive with normalized metadata and no caches or native
extensions. `tools/verify_submission.py` safely extracts to a fresh directory,
sanitizes `PYTHONPATH`, checks repository-root absence from `sys.path` and
runtime package origins, raw-loads extracted `main.py` through
`kaggle_environments.agent.get_last_callable`, enables the bounded strict
diagnostic switch, scans the complete status history, and checks the pinned
seed-7/seat-1 reference bank.

- Authorized checkpoint SHA-256 verified from the read-only input:
  `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`.
  The prior durable spelling had 65 characters and one extra `D`, so it was
  corrected here and in the current V0.7 final note to the actual 64-character
  SHA-256 digest.
- Build command: `python tools/build_submission.py --checkpoint
  "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt"
  --output "artifacts/local/submissions/bc-e-v07.tar.gz"`.
- Output (ignored): `artifacts/local/submissions/bc-e-v07.tar.gz`, SHA-256
  `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf`, 50
  members; root members include `main.py`, `best.pt`, and
  `submission_manifest.json`.
- Focused tests: `tests/test_submission_tools.py` — **4 passed**; `ruff`
  passed on all changed source/tests.
- Exact verifier command: `python tools/verify_submission.py
  "artifacts/local/submissions/bc-e-v07.tar.gz"`.
  Result is **PARTIAL**, with the exact child error
  `ModuleNotFoundError: No module named 'kaggle_environments'` after archive
  extraction; no bank or trajectory fingerprint is claimed. The omitted-
  `fast_env` regression fails earlier and loudly with `ModuleNotFoundError`.
- The pre-behavior-change reference remains candidate seat 1 versus PASS,
  final bank **54,439**; a deterministic action fingerprint is intentionally
  not pinned until the official 1.32.7 dependency is available in the isolated
  verifier process.

## 2026-08-25 — Executor V0.7 Frozen: R4 Rejected, Shed-Room Fix Accepted, Viewer #11 Closed

Closed issue #7 / Executor V0.7 at `a7c826d` without changing code or tests in
this documentation pass. R4 was rejected and reverted after the real BC-E
fixed-plan 7d regression: wealth **-14,302**, cash **-9,515**, weeds **+11**,
crops destroyed **+14**, survival debt **69 -> 84**, starvation **22 -> 44**,
and harvest **124 -> 98**. Expert-intent-only improvements are not acceptance
evidence.

Accepted changes are the panel outcome ledger (`02984a0`) and the survival
WHEAT shed-room clamp (`a7c826d`). The clamp uses official/default
`shed_capacity=100` and preserves survival-before-hire order. The final
no-R4 fixed-plan panel is bounded to PASS, seeds 17/42/2026, both seats, and
six rows:

| seed | seat | prior-debt ON bank | prior-debt OFF bank |
| ---: | ---: | ---: | ---: |
| 17 | 0 | 17,005 | 265 |
| 17 | 1 | 14,961 | 265 |
| 42 | 0 | 23,346 | 30 |
| 42 | 1 | 26,587 | 33 |
| 2026 | 0 | 56,742 | 0 |
| 2026 | 1 | 65,959 | 0 |

ON aggregates are mean **34,100**, median **24,966.5**, minimum **14,961**;
OFF aggregates are mean **98.8**, median **31.5**, minimum **0**. Loss units
are ON starvation **0** / overflow **12**, versus OFF starvation **38** /
overflow **12**. The default prior-debt setting remains ON as explicit
heuristic architectural debt, bounded only to this panel; it is not a
generalization or acceptance claim.

The seed-17 day-20 starvation-purchase defect was fixed: banks changed from
`8062/8489` to `17005/14961`. The residual day-22 six-animal/seat condition
is true manager-policy debt: carried cows are discarded at day end into a full
shed, not a starvation or executor defect; no further heuristic was added.
Viewer #11 is closed and passive: schema/runner/snapshot/viewer/debug-trace
CLI support is committed, four ignored real BC-E E_VS_E traces were generated
and parsed, and ON/OFF action/result parity is exact for seeds 17 and 42.
E_VS_E banks intentionally differ from the PASS panel, and no traces were
committed. The real BC-E validation input was externally supplied read-only
from
`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`, variant E epoch 27, SHA-256
`F4B029D3E463ABA1DB0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`.
It is intentionally absent from this isolated worktree because local
artifacts/checkpoints are ignored and remains uncommitted.

## 2026-08-24 — Issue #9 Stage B2: Stage-A Integration, Tiny Live PPO Smoke, CLIs, Diagnostics

Completed the issue #9 infrastructure locally on top of B1 (`06cc25c`): the
PPO policy now conforms to the Stage-A batched interface and a tiny
end-to-end rollout -> trajectory -> GAE -> one update -> checkpoint/eval
artifact chain runs green.

- `rl_manager/ppo_adapter.py` (new): `PPOBatchedPolicy` implements the
  Stage-A `BatchedPlanPolicy` protocol over the B1 `PPOPolicy` — contiguous
  own-only E arrays + explicit string `prng_id` (sha256 -> uint32 root key,
  per-row decision seeds via fold_in), stochastic training mode and
  deterministic eval mode (exact frozen-E decode before drift), exact action
  tensors / six logprob groups + total / value / immutable identity; sell
  quantities always from the immutable frozen snapshot. Plus
  `ppo_batched_policy_from_state` (checkpoint resume reusing EXACT stored
  params/frozen/rng) and `select_ppo_subset` (deterministic evenly-spaced
  2–8 row subset AFTER full-trajectory GAE/normalization).
- `rl_manager/diagnostics.py` (new): compact strictly JSON-safe diagnostics
  record (`allow_nan=False`) — rollout seed/composition/steps, timing split
  env/executor/policy/orchestration, return/win/banks/margin, six-group
  entropy, approx KL, clip fraction, value loss, explained variance,
  advantage stats, KL-to-frozen drift, executor unfinished/
  missed-maintenance totals, anomalies/provenance, pre/post fingerprints,
  checkpoint path; missing values are null + machine-readable reason.
- `rl_manager/cli.py` (new): guarded train/eval commands. Train requires an
  existing real BC-E checkpoint (fail loud if missing) and an explicit
  executor factory; worker/env/thread knobs default to safe 1 (>1 worker
  fails loud as not-yet-implemented). Eval exposes fixed seed sets
  smoke(17,42,2026)/dev(200..263)/holdout(5000..5031), always both seats,
  prints planned game count, requires `--confirm-expensive` for dev/holdout;
  fixed output schema W/L/T, paired margins, median/mean banks,
  per-orientation split, anomalies, worst seeds. Tests cover ONLY
  parser/planning/aggregation; commands never executed by tests.
- `rl_manager/__init__.py`: exports the new adapter/diagnostics symbols.
- Tests: `tests/test_rl_manager_ppo_integration.py` (10 tests: adapter
  fields/single-batched-call/prng determinism/own-only rejection,
  deterministic init parity with frozen E, ONE complete fast game ->
  52 transitions -> full-trajectory GAE -> exact stored-action logprob
  recompute -> ONE 4-row-minibatch PPO update with finite metrics, changed
  trainable params, bit-identical frozen snapshot, checkpoint roundtrip +
  bit-identical resume + pre/post eval equality, JSON-safe diagnostics
  artifact, honest-nulls check, official-engine and real-checkpoint gated
  skips) and `tests/test_rl_manager_cli.py` (11 tests: safe defaults,
  fail-loud validation, seed-set planning counts, confirmation gate, fixed
  aggregation schema on synthetic records).
- Validation: rl_manager suite + focused issue-#8 JAX parity/train =
  130 passed + 4 skipped in ~187 s (skips: official engine dependency
  absent x2, real BC-E checkpoint absent x2). Complete fast-engine games
  executed by new B2 tests: exactly ONE (numThreads=1). Ruff clean on all
  changed/new files.
- Non-claims: tiny random-init E, one gradient step — plumbing correctness
  only, no policy-quality claim. Serious training remains blocked on issue
  #7 (executor selection) and the absent real BC-E checkpoint.

## 2026-08-24 — Issue #9 Stage A Correction: Automatic Artifact Provenance (A1)

Small correction commit on top of the Stage A harness: trajectory artifacts
now record full provenance automatically instead of requiring callers to
hand-assemble `run_metadata`.

- `rl_manager/runner.py`: new `build_artifact_metadata` +
  `SelfPlayRunner.save_trajectory_artifact(path, buffer, result)` — merges
  per-episode outcome (final banks/margin/winner/rewards/statuses/
  transitions/terminated/episode trace digest/rollout trace ref/timing),
  opening name+digest, backend/engine provenance, executor factory
  name/version/identifier/version_sha256, per-seat policy/opponent
  identities (recorded at finalize time via the new `EpisodeResult.
  policy_identities`), master seed, composition, and manager start day into
  the sidecar `run_metadata` under `artifact_schema_version = 1`. The full
  primitive trace is never duplicated into the training core.
- `rl_manager/trajectory.py`: sidecar JSON now written with
  `allow_nan=False` (strict JSON safety at write time);
  `TrajectoryBuffer.save(run_metadata=...)` API unchanged for backcompat.
- Tests: `tests/test_rl_manager_runner.py` proves a complete cached tiny-E
  rollout save/load carries all mandatory provenance automatically with
  exact episode values; rl_manager suite green (48 passed, 1 skipped —
  official-engine skip unchanged). See `research/RL_SELFPLAY_V0.md`
  "Trajectory Schema" / "Provenance and Reproducibility".

## 2026-08-24 — Issue #9 Stage A Implemented: RL Self-Play Rollout/Trajectory Harness (`rl_manager`)

Added the new `rl_manager` package (11 modules) and 7 test files on `main`
as pure RL harness infrastructure around the frozen components. No PPO,
no executor/opening-book/oracle/fast-env/Rust/BC changes; frozen components
are consumed through public interfaces only (farm canonicalization goes
through the public `EngineBackend.canonical_state()` seam, never a private
helper).

- **Batched policy seam:** framework-neutral `BatchedPlanPolicy` protocol
  with immutable `PolicyIdentity` (name/version/fingerprint; equality on
  all three fields) and PPO-ready logprob-group/logprob-total/value slots.
  `JaxEPlanPolicy` wraps `bc_manager_jax.forward(..., model_variant="E")`
  — exactly ONE call per contiguous request batch, own-only E contract
  enforced loudly at the wrapper seam, deterministic issue-#8 decode in
  pure NumPy (`decode.py`, torch parity enforced by test), parameter
  fingerprint = sha256 over sorted leaf paths + raw bytes.
- **Runner:** lockstep N-env self-play over `oracle.backend` engines;
  every day-boundary manager request across all envs/seats is grouped by
  policy identity and answered with ONE batched policy call per
  (identity, day). Per-episode/per-seat opening agent (`standard_mixed`
  literal d0-d3 playback, clean d4h0 handoff), unmodified executor agent
  via an injectable factory (`executor_v0.make_agent(strict=True)
  @stage-a-v1`; issue #7 swaps the factory), queued-plan provider (missing/
  double consumption fail loudly), and runner-owned daily-start `(day,
  cash)` state feeding the exact stateless `economic_prev_start` E path;
  realized labor from observed `hires_today`, never HIRE intents.
- **Trajectory schema v1:** preallocated compact buffer (append fails loud
  on overflow); one row per manager decision/day/seat d4..d29; scalars +
  six logprob groups + seven action tensors + 32-byte sealed per-day
  joint-action trace digest + model-facing inputs pinned EXACTLY to the
  canonical E encoder spec derived by calling `encode_live_inputs` itself.
  Strict NPZ serialization (schema/count validation, `allow_pickle=False`)
  plus a JSON-safe sidecar (policy/opponent identities, plan JSON, compact
  executor diagnostics). Terminal-only rewards (+1/0/-1 on the final
  manager row; both zero on tie).
- **Seeds/provenance:** episode/policy/environment seeds are pure
  functions of `(master_seed, tag, index)` via `SeedSequence`; runs record
  opening digest + source-replay provenance, backend configuration/engine
  module, executor factory version, master seed.
- **Parity seam:** official-vs-fast comparison of the identical stack —
  opening handoff -> manager input digests -> plans -> primitive actions ->
  banks/statuses — with first-divergence reports carrying seed/backend/
  seat/day/hour/turn/path/values/actions. Honest gating when the official
  dependency is absent.
- **Local evidence:** 47 passed + 1 skipped across the 7 rl_manager test
  files (skip = `kaggle_environments` absent in this interpreter, Python
  3.13.1; rerun command recorded in the skip message). Focused seams green:
  JAX parity/opening handoff/executor agent/oracle replay+isolation =
  72 passed + 8 skipped. Two deterministic complete fast games
  (numThreads=1, tiny random-init E): DONE/DONE, exactly 52 transitions
  each, byte-equal rerun (`equal_nan=True` for the board NaN sentinel),
  episode digest `fd910f3d8f5a6a8f864b5d76daad87c477611c75742ab44f959c0d946dd88a10`,
  final banks [0.0, 0.0] bankruptcy tie — PLUMBING only, no quality claim.
  N=2 batching proof: E-vs-E one policy call/day (contiguous batch 4),
  candidate/frozen two calls/day (batch 2 each).
- **Not done / not claimed:** no PPO/optimizer/value loss; no real-
  checkpoint rollout (`artifacts/local/bc-v1-E/best.pt` absent); no
  official-engine parity run locally; eventual many-CPU-worker topology is
  design only and UNMEASURED. Serious training blocked on issue #7
  executor selection. Details: `research/RL_SELFPLAY_V0.md`.

## 2026-08-24 — Issue #8 Implemented: Promoted BC-E Manager Ported to JAX (V0+E Only)

Extended `bc_manager_jax` with model variants **V0** and **E** on `main`.
E is the four-way closed-loop ablation winner (median bank 25,873 vs V0
9,251.5), so it is the only new architecture/input ported; **J/JE are
deliberately unsupported** in JAX and fail loudly
(`'J' is not supported by bc_manager_jax`). No executor, opening-book,
fast-env, or Rust code changed.

- **Variant seam outside the frozen config:** the seven-field serialized
  `ManagerConfig` is unchanged; the variant mirrors the torch checkpoint's
  top-level `model_variant` (absent -> V0). Native NPZ stores
  `model_variant` top-level in its JSON metadata; pre-variant native files
  load as V0. Expected variants are checked strictly — never inferred from
  weight shapes.
- **Exact E contract:** `economic_context` float32 `[B, 14]`, finite,
  concatenated after the six self-resource feature blocks and before the
  SAME two-layer MLP (first-layer input 35 -> 49). Trunk/tokens/heads/
  loss/decode byte-identical to V0; V0 rejects `economic_context` as an
  unknown input. Parameters: default V0 1,071,040 / E 1,072,832; tiny V0
  37,008 / E 37,232.
- **Authoritative features only:** tests build every economic row via
  `bc_manager.economics` (`economic_context`, `EconomicHistory`) covering
  day-0 invalid, adjacent join, gap invalidation, reset/backwards day,
  all-land saturation, zero/negative cash, and exact channel order; no
  formula is re-derived anywhere.
- **Local CPU evidence (tiny):** deterministic PyTorch E -> JAX E strict
  conversion parity across all seven output groups: worst max abs
  6.855e-07, worst mean abs 1.101e-07 (gates: 2e-6 / 5e-7). Decoded
  counts/land exact; total+per-group loss parity within 9.5e-7. Single-
  device JIT E forward + train step finite. N=4 logical-CPU NamedSharding
  subprocess: 1-vs-4 total diff 1.9e-6, group diff 2.4e-7, param diff
  3.7e-9, batch spec `P('data', None)`; one bounded N=8 logical smoke
  finite. N=4/N=8 are forced host-CPU logical validation ONLY — no
  throughput/scaling claim.
- **Benchmark:** rows record `model_variant` additively; random mode gains
  `--variant {V0,E}`; checkpoint mode uses the stored variant; synthetic E
  batches come from the authoritative `economic_context`. The exact
  eventual Kaggle 8-device command targets `/kaggle/working/bc-v1-E/best.pt`
  and is explicitly UNMEASURED (`research/JAX_TPU_V5_RUN.md`).
- **Not done / not claimed:** the real BC-E checkpoint is absent locally
  (`artifacts/local/bc-v1-E/best.pt`), so no real-checkpoint conversion or
  parity has run; a bounded skip-if-absent test records the exact rerun
  command. No TPU measurement exists.
- **Tests:** `tests/test_bc_manager_jax_parity.py` (31 incl. 1 skip),
  `tests/test_bc_manager_jax_train.py` (11),
  `tests/test_bc_manager_jax_benchmark.py` (6) — 47 passed + 1 skip; all
  pre-existing issue-#5 V0 tests unchanged and green; PyTorch BC
  regressions 114 passed.

## 2026-08-23 — Issue #6 Implemented: BC V1 Four-Variant Ablation (V0/J/E/JE) With Fixed Closed-Loop Panel Gate

Implemented the complete BC V1 ablation in four bounded commits on `main`
(`2f48564` Stage 0 audit -> `192d0dc` E foundation -> `3d7fae1` J/JE decoder
-> `fc95752` live integration + panel CLI), then recorded the durable state
and exact Kaggle runbook (`research/BC_V1_ABLATION_RUN.md`, decision D-026).
No executor behavior, opening playback, or engine code changed; diagnostics
exposure is purely additive.

- **Variants** over the unchanged D-019 trunk/data layer: V0 baseline
  1,071,040 params; J joint plan decoder 1,204,288 (+133,248); E realized-
  economic context 1,072,832 (+1,792); JE both 1,206,080. V0/J checkpoints
  keep the exact pre-V1 encoder path (regression-tested). E/JE feed a live
  `EconomicHistory` tracker that exactly mirrors batch derivation including
  day gaps and partition boundaries.
- **Feature discipline** (Stage 0 audit,
  `research/BC_V1_ECONOMIC_CONTEXT.md`): submitted market intents are never
  treated as realized fills — no gross revenue/spend/fill inference; only
  observed money snapshots and the `hires_today` counter are used. Coherence
  diagnostics are JSON-safe (explicit zero-cash/over-threshold flags, never
  Infinity) and diagnostic-only: never clipped into plans, never fed back.
- **Panel gate** (`python -m bc_manager.ablation`, D-026): strict checkpoint
  gates (format v1, stored `model_variant` matching the mapping, teacher-
  forced `validation_metrics.total`; smoke weights rejected), official
  1.32.7 provenance guard, `standard_mixed` opening days 0–3 -> tested BC ->
  unchanged executor, seeds 7/17/42/123/2026 × seats {0,1} = 40 games,
  ranking by closed-loop final-bank median then mean; seed-17 collapse flag
  and seed-2026 retention reported beside raw banks. Teacher-forced totals
  and coherence are prerequisites/diagnostics only and never promote a
  variant. `--validate-only` preflights without importing the engine.
- **Local evidence:** stage sweeps grew to 275 passed across BC V1 +
  bc_manager + executor_v0 + opening_book suites at `fc95752`; independent
  Ox audit 62 new + 163 compat = 225 passed (PASS_WITH_FINDINGS, no code
  blockers); one official opening-only seed-7/seat-0 smoke under pinned
  1.32.7 replayed all 96 turns with clean handoff and zero divergence/
  fallback/status anomalies — plumbing only, no BC weights attached.
- **Not done / not claimed:** real five-day corpus Parquet and trained
  checkpoints are absent locally, so there are NO teacher-forced variant
  results, NO closed-loop panel results, and NO winner. The Kaggle runbook
  (fresh clone, exact deps incl. `kaggle-environments==1.32.7`, corpus
  locate-or-fail, four identical-matrix training commands, preflight,
  primary panel command, artifact pass criteria) is copy-paste ready in
  `research/BC_V1_ABLATION_RUN.md`. PPO/recurrence/value heads, plan
  affordability clipping, JAX V1, and executor changes remain out of scope.

## 2026-08-23 — Issue #2 A/B Benchmarks: Official 1.32.7 vs Fast Engine vs diffmap Reference

Ran the reproducible same-machine benchmark suite required by issue #2
(decision D-025). New artifacts: `scripts/benchmark_engine_throughput.py`
(worker/all/report subcommands, deterministic scripted traces, warmup-aware
median/min/max, loud rejection of NaN/impossible rates),
`docs/benchmarks/issue2_results.json` (raw machine-readable summary), and
`docs/benchmarks/ISSUE2_THROUGHPUT.md` (generated report). Focused tests:
`tests/test_benchmark_script.py` (7 passed, 1 official smoke skipped without
the pinned venv).

- Reference build: upstream `diffmap/kaggicultureRL` @ `ef8bb3a` cloned under
  the authorized temp root, wheel built release-mode with the temp-local GNU
  toolchain into an isolated venv (`CARGO_TARGET_DIR` outside the clone; the
  upstream checkout stayed byte-clean). Reference is provenance-pinned 1.32.6
  with old shapes (OBS_SIZE 5630, ACTION_SLOTS 27) — performance reference only.
- Scalar full episodes (720 steps = reset + exactly 719 accepted step calls,
  both engines): official 1.32.7 ~1.30 s/episode (~553 turns/s); fast dict API
  ~0.279 s (~2,580 turns/s) = **4.7x speedup**; fast native floor ~188k
  turns/s (341x vs official); reference native ~499k turns/s.
- Batch steady-state (`step_into`, preallocated buffers, transitions counted
  as N*steps): ours scales from ~58k t/s (N=512, T=1) to ~167k t/s (default
  pool) = 2.87x at N=512 and 2.89x at N=1024; best cell N=128/T=4 ~204k t/s.
  Below N=128 the serial loop makes thread counts irrelevant, as designed.
- Profile finding: observation writing is **84%** of our steady step_into cost
  at N=512 (224k obs t/s vs 1.16M transition-only t/s). The unmodified
  reference core is ~2.7x faster per env-transition at N=1, consistent with
  its old 16-hand layout vs our exact-contract MAX_HANDS=240 writer.
- Decision recorded: no engine change in this stage; observation-writer cost
  is the single bounded optimization candidate, deferred to a distinct stage.
- Memory: theoretical obs buffer 70,128 B/env + action tensor 12,048 B/env;
  measured RSS deltas ~65-107 KB/env (allocator/pool overhead included,
  explicitly not GameState sizes).
- Environment: i7-12700H (14C/20T), Windows 11 build 26200, Python 3.13.1,
  numpy 2.5.2, kaggle-environments 1.32.7 pinned wheel, repo @ 63c8113.
  No TPU claim; laptop absolute numbers do not transfer.

## 2026-08-23 — Issue #2 Throughput Seam: GIL Release + Optional Instance-Local Rayon Pool

Implemented the throughput seam for the Rust batch backend (decision D-024).
`reset`, `step`, `step_transition`, `observe_into`, `action_masks_into`, and
`step_into` now validate inputs and extract exclusive raw buffer slices under
the GIL, then run the whole native transition/observation pass with
`py.allow_threads`. `RustBatchEnv` gained an optional instance-local Rayon
pool (`num_threads=N`; `None` keeps the global-pool default; `0` invalid),
exposed as `FastKaggricultureEnv(configuration={"numThreads": N})` and a
`num_threads()` accessor. Parallel fan-out still engages only at >= 128
environments; small-batch behavior is unchanged.

- Evidence (`tests/test_batch_throughput_seam.py`, 5 passed, system Python
  3.13): GIL spinner test observed ~84M Python-thread counter ticks during a
  ~4.7 s 512-env native call; caller-owned buffers stayed byte-identical to a
  quiet reference under concurrent Python-thread pressure; trajectories were
  byte-identical across worker counts 1/2/4/default over 130 envs x 30 steps
  including the day boundary.
- Prior-session heavy evidence retained: cargo fmt/check clean, 16 Rust unit
  tests green, fresh release build, full pytest 364 passed / 88 skipped,
  official 96-turn and full seed-0 719-step parity both passing.
- Docs updated: `fast_env/README.md` (API + semantics + safety boundary),
  `CURRENT_STATE.md`, `DECISIONS.md` (D-024).
- Not claimed: any throughput/speedup/scaling result — benchmarks are not run
  yet. Fused executor/day-step batching and distributed rollout remain
  deferred. Target measurement topology: Kaggle TPU v5 host (~96 vCPUs),
  process-level parallelism x in-process batch environments.


## 2026-08-23 — Independent Stateful Closed-Loop Agent A/B: Zero Divergence

Added the secondary policy-interface gate after the same-action parity corpus.
`oracle/closed_loop.py::run_closed_loop` creates independent official/fast
backends and four fresh stateful agent instances, compares reset and every
next presented observation, computes actions independently from each backend's
observation, compares actions before either step, then compares canonical next
state/rewards/statuses immediately. The official full status history is still
checked for hidden ERROR/INVALID/TIMEOUT anomalies.

- `make_deterministic_executor_factory` uses the existing stateful
  `executor_v0.ExecutorAgent` and `FixedPlanProvider` with a nontrivial fixed
  crop/animal/fertilizer/sell plan. Its narrow fast view adapter converts only
  age/placed-day wire aliases and sparse private maps required by the existing
  executor; no agent state or actions are shared.
- Fixed-plan seeds `0`, `7`, and `42` each completed reset + 719 accepted
  primitive steps, reached canonical step 719 with `DONE/DONE`, and matched
  official/fast rewards exactly. The union covered 30 submitted farmer,
  hand, and market action families.
- Repo-local `data/temp/bc-train-smoke/ckpt/best.pt` was available and passed
  one real checkpoint/executor A/B episode on seed 0: 719 accepted steps,
  `DONE/DONE`, equal `[0.0, 0.0]` rewards. This is plumbing evidence only,
  not a competitive result.
- Deliberate reset-observation drift and pre-step action drift tests report
  seed/step/day/hour/seat/path/official/fast/actions and stop before stepping.
- Machine-readable report: `research/closed_loop_ab_report.json`; runner:
  `python scripts/run_closed_loop_ab.py` in the pinned official venv.
- Focused validation: `tests/test_oracle_closed_loop.py` — **7 passed** in the
  official venv; the report run covered four full episodes in 52.65 seconds.
  Import isolation and the existing same-action corpus remain separate gates.

Not claimed: universal parity, competitive BC quality, executor redesign/tuning,
PPO/self-play, throughput, or benchmark results.

## 2026-08-23 — Full-Episode Same-Action Parity Corpus: Zero First Divergence on 8 Complete 720-Step Episodes

Built the deterministic legal-ish corpus stage (decision D-022) and drove the
differential oracle through complete default-configuration episodes. **Zero
first divergence across all 8 fixed seeds** — no simulator/interface mismatch
surfaced at full-episode scale; the earlier mechanic-cluster fixes held.

- New `oracle/action_generator.py::LegalishActionGenerator`: fixed-RNG,
  state-aware reflex policy; reads only the pre-transition fast observation
  pair, emits ONE action pair per turn to BOTH engines (same-action gate
  untouched). Deliberately covers the official silent-noop surface
  (malformed market entries, unknown ops, bad quantities, >10-order
  truncation bursts, extra hand slots, unaffordable orders).
- New `scripts/run_parity_corpus.py`: runs complete episodes per seed,
  counts day-boundary transitions via a fast-backend wrapper, writes the
  JSON report (`research/parity_corpus_report.json`, schema_version 1), and
  records a `(generator_seed, turn_index)` repro for any divergence.
- Corpus result (seeds 0, 1, 2, 7, 17, 42, 123, 999): every episode ran the
  full reset + 719 accepted primitive steps, terminal DONE/DONE at canonical
  step 719 (day 29 hour 23) with official rewards == fast rewards for both
  seats (e.g. seed 0 `[2.0, 0.0]`, seed 17 `[144.0, 0.0]`); exactly 29 day
  transitions each; total 5,752 turn pairs, ~24.5 s wall. Coverage union:
  33 action families, 28,508 attempted family instances (movement, pickup/
  place/drop, plant/water/fertilize/harvest/dig, build coop/pasture,
  feed/care/collect-fertilizer by farmer AND hands, BUY_SEED/BUY_PRODUCT/
  BUY_ANIMAL/SELL/HIRE/BUY_LAND, malformed/no-op/truncation).
- Primitive-turn accounting locked in docs/tests: "720-step episode" = one
  reset observation + exactly 719 accepted `step` calls; DONE at canonical
  step 719 = day 29 hour 23.
- Repeatability locked: same generator seed reproduces an identical trace;
  fast reset+replay on fresh engines reproduces identical canonical states,
  rewards, and statuses per turn (`tests/test_action_generator.py`).
- New tests: `tests/test_action_generator.py` (4 offline, always run) and
  `tests/test_oracle_corpus.py` (4 official-gated: short legal-ish episode,
  two full default episodes with terminal accounting, divergence
  attributability under a corrupted fast state). Full repository suite with
  the venv interpreter: **427 passed, 1 skipped** (by-design system-python
  isolation skip). One flaky failure was observed once in the concurrent
  UNTRACKED `tests/test_bc_manager_jax_parity.py` workstream when run inside
  the full suite; it passes in isolation and passed in the final full-suite
  rerun — left untouched as concurrent work, not owned here.
- No Rust/native changes were needed; no maturin rebuild required.
- Scope of claim: bounded — parity proven for the states these 8 episodes
  reach (33 families, both seats, 30 days, terminal lifecycle). Not a
  universal mathematical proof; closed-loop A/B remains the next gate.

## 2026-08-23 — MAX_HANDS=240 Exact-Layout Revision Closes the >16-Hired-Hands Deferral

Replaced the fixed 16-slot fast-engine hand layout with the exact
default-contract capacity `MAX_HANDS = maxMarketOrdersPerTurn(10) *
turnsPerDay(24) = 240`, derived in `scripts/generate_fast_protocol.py` from
the pinned schema defaults. Breaking wire layout: `OBS_SIZE` 5630→8766,
`ACTION_SLOTS` 27→251 (market action rows moved from slot 17 to slot 241),
`MASK_SIZE` 3562→34026; only the two MAX_HANDS-scaled observation blocks
moved (full offset list and per-env buffer deltas in `MECHANICS.md`, decision
D-021). All Rust loops parameterized; the extension now exports `MAX_HANDS`,
`ACTION_SLOTS`, `MASK_SIZE`; stale preallocated buffers fail loudly.

Test corrections made while finishing the prior worker's uncommitted work:

- `tests/test_fast_env.py::test_hire_mask_matches_official_reachable_semantics`
  asserted an always-open HIRE mask at 23 hands. Corrected to the official
  affordability gate: 23 hires cost fib(24)-1 = 75,024 of 100,000, leaving
  24,976 < fib(23) = 46,368 for the next hire, so the mask must be 0; the
  reset state proves the open side (mask 1). The maximal-pressure sub-case
  also ran 24 hiring steps, which crossed the day-0→day-1 reset (hands clear,
  mask legitimately reopens); corrected to 23 steps so the closed-mask state
  is actually observed at day 0 hour 23.
- The same file's 23-hand scenario asserted 23 distinct spawn positions;
  hands spawn on the fixed 4-tile farmhouse access set
  (`[[4,4],[5,4],[4,5],[5,5]]`, least-occupied wins), so the assertion was
  replaced with the correct spawn-domain check.

Validation (all green): `tests/test_fast_env.py` 15 passed (incl. 23-hand
scalar API, hand actions reaching all hands, day-end reset from 23 hands +
rehire Fibonacci restart, memory/shape sanity); `tests/test_oracle_hands.py`
5 passed against the real pinned official 1.32.7 engine in the documented
temp venv (exactly-16 boundary with hand actions, 16→17 crossing turn,
23-hand hires + two turns of subsequent hand actions, day-end reset from 23
hands + next-day rehire parity, and per-turn fast HIRE mask == official-
reachable gate); full repository suite via the venv interpreter: **401
passed, 1 skipped** (the by-design system-python isolation skip), including
all concurrent opening_book tests; `cargo fmt --check` clean; `cargo test
--release` 15 passed; `scripts/generate_fast_protocol.py --check` OK (run
with the pinned 1.32.7 venv interpreter). No maturin rebuild was needed:
no behavior or generated bytes changed after the prior worker's build.

Not claimed: full-episode parity, training safety — full 720-turn episodes
through the oracle remain open Stage-2b work. `bc_manager` MAX_HANDS=8 is a
separate head-slot constant, unchanged.

## 2026-08-23 — Issue #4 Implemented: Elite Opening Book with Official 1.32.7 Validation

Implemented the V0 opening book in three bounded commits on `main` and
validated it through the pinned official engine. No BC, executor, or engine
code changed.

- **Stage 1 (`1cbda01`) — trace/provenance artifact.** New `opening_book/`
  package: deterministic single-replay extractor
  (`python -m opening_book.extract`), fail-closed trace validation/loading,
  and two committed compact identities of exactly 96 literal submitted
  primitive actions (days 0-3, d0h0..d3h23; d4h0 excluded) with full
  provenance (episode/seat/seed/player/source-replay SHA256/content digest):
  `standard_mixed` from episode 95515912 seat 0 (dominant cluster, market
  signature verified hour-identical to the research note) and `pasture_heavy`
  from episode 95055022 seat 0 (ReCurSiON; byte-identical opening re-verified
  across episodes 95055022/95481731). Trace regeneration is byte-identical.
- **Stage 2 (`ccad132`) — runtime wrapper.** `make_opening_agent(opening,
  downstream, seat)` replays the literal trace for days 0-3 under minimal
  one-way guards (phase cursor equality, observed hand cardinality vs trace
  hands, Stage 1 action-shape/market-cap reuse), then delegates unchanged to
  the injected downstream agent starting exactly at day 4 hour 0. Any guard
  failure records one divergence reason/turn plus a best-effort farm summary
  and permanently delegates; the script never rejoins. Deterministic JSON
  diagnostics expose identity/source provenance/turns replayed/handoff.
- **Stage 3 (`2705bf3`) — official evaluator.** `python -m opening_book.eval`
  runs opening-only and paired BC-handoff games behind
  `oracle.provenance.verify_official_provenance()` (pinned wheel
  `2a1bb862...`, upstream `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`),
  with strict handoff envelopes derived from the source replays and a
  documented paired mode requiring the real checkpoint. The module docstring
  carries the exact Kaggle command/cell for the paired comparison with
  `/kaggle/working/bc-v0-score2950/best.pt`.
- **Validation.** 53 focused offline tests (20 trace + 18 wrapper + 15
  evaluator) pass under default Python. Official matrix in the repo-local
  ignored venv (2 seeds x 2 seats x PASS/mirror x both identities = 16 full
  720-turn games): standard_mixed **8/8** and pasture_heavy **7/8** strict
  envelope passes; every passing cell also showed exactly 96 scripted turns,
  zero divergence/fallback, clean d4h0 handoff, and zero official status
  anomalies.
- **Known failure (kept strict by decision).** pasture_heavy seed 1146601720
  seat 1 vs PASS ended with STRAWBERRY 2 instead of 3: the official engine's
  weed spawn placed WEEDs on tiles `[y=2][x=1]`/`[y=4][x=1]` by d3 h10
  (empty in the source episode), so the literal d3h11 `PLANT STRAWBERRY`
  was silently ignored. Worker positions match the source replay every hour;
  all 96 turns replayed with no divergence/anomalies and a clean handoff.
  Same seed passes under the mirror opponent. This is environment variance,
  not wrapper behavior; per decision the envelope stays exact and no
  heuristic weed repair was added.
- **BC limitation.** The real `best.pt` is absent locally; paired BC
  evaluation has not run and the tiny smoke checkpoint was not substituted.
  No end-to-end BC gain is claimed. The evaluator fails clearly (exit 2) and
  documents the exact Kaggle command for later execution.

## 2026-08-23 — Stage 2b slice 4: Town/World Updates, Day RNG, Reset, and Terminal Differential Parity

Probed the town/world cluster (shop unlock/consumption, global day-end RNG,
daily reset ordering, terminal lifecycle) against the real pinned official
1.32.7 engine and committed the bounded result on `main` (no push). The
cluster was already at zero divergence: no engine changes were required.

- **Probes:** new `tests/test_oracle_town_world.py` — 10 focused tests over
  10 real-official same-action scenarios (~1,100 turn pairs including one
  648-turn PASS-only season segment; full canonical compare every turn)
  covering shop unlock timing at end-of-day `(day + 1) % interval == 0`
  (canonical steps 72/144/216/...), draw-with-replacement multiplicity with
  three duplicate FARMERS_MARKET instances and the 8-instance cap held
  through day 27, per-turn market-inventory trajectory recomputed from the
  official shop tables (every-4-step shop drain, single-product x2
  multiplier, every-24-step town-center product set firing at step 0 on an
  empty town), unconditional consumption driving WHEAT stock below −20,000
  with prices tracking the scarcity branch and never breaching
  `PRICE_FLOOR`, the shared per-day `Random((seed * 1_000_003) ^ day)`
  stream across both farms (weed draws row-major farm 0 then farm 1, then
  the shop choice; planted tiles shift stream position;
  `weedSpawnChance` 0 / default / 0.5 all verified), same-seed
  bit-identical repeatability across fresh backends vs different-seed
  divergence wherever a draw occurs, day-boundary reset ordering (hands
  removed, carried inventories dropped to shed, `hires_today` reset,
  farmer returned to spawn, watering counters advanced — all before any
  same-boundary shop unlock), and terminal lifecycle (DONE + reward =
  final farm money exactly at step `episodeSteps - 1`; official wrapper
  refuses post-terminal steps with `FailedPrecondition`; fast engine
  transitions nothing further).
- **Divergences found:** none — zero divergences across all scenarios.
- **Validation:** focused venv run of `tests/test_oracle_town_world.py`:
  10 passed. Prior implementation evidence: fresh maturin rebuild, exact
  RNG/draw ordering checks, and the full venv suite at 390 passed /
  1 justified skip.
- **Not claimed:** whole-engine/full-episode parity, >16 simultaneous hands,
  closed-loop A/B, benchmarks, training safety. New issue #2 throughput
  gates (GIL release, configurable Rayon thread count, batched/multi-core/
  memory benchmarks; fused executor/day-step explicitly deferred) are
  recorded as future work only — not part of this parity slice.

## 2026-08-23 — Stage 2b slice 3: Animal/Structure/Fertilizer Lifecycle Differential Parity

Probed the animal/structure/fertilizer lifecycle cluster against the real
pinned official 1.32.7 engine and committed the bounded result on `main`
(no push). The cluster was already at zero first divergence: no engine
changes were required, and the canonical compare was not weakened.

- **Probes:** new `tests/test_oracle_animals.py` — 12 focused tests over 11
  real-official same-action scenarios (1,320 turn pairs, 48 day boundaries;
  every turn compared canonically) covering the `ANIMALS` constants table (GOOSE/COOP/EGG,
  COW+SHEEP/PASTURE/MILK/WOOL with differing first-yield/interval/max-held
  timings), free BUILD_COOP/BUILD_PASTURE on empty owned tiles only (blocked
  by plant/structure/LOCKED), BUY_ANIMAL per-unit partial fills on
  insufficient funds and shed capacity plus malformed-order skips, PLACE onto
  matching unoccupied structures vs fall-through shed paths and mismatched
  no-ops, FEED wheat consumption/once-per-day/no-wheat no-ops, CARE timing,
  COLLECT_FERTILIZER placement-day unavailability/once-per-day/daily
  regeneration, production timing and quantities including pending-care-bonus
  accrual, consumption on fed production days, LOSS on unfed production days,
  and max_held caps, exact escape timing at the second consecutive unfed
  refresh with the bare structure remaining, DIG semantics (no-op on placed
  animals, clears empty structures), HARVEST full drains, hired-hand chore
  with its own inventory (hands never survive a day boundary), and day-end
  inventory-drop insertion-order priority with animals carried under a tight
  shedCapacity.
- **Divergences found:** none — every scenario's first divergence count is
  zero; all probe iterations were semantic-assert corrections in the test
  file itself (both engines agreed at every compared field).
- **Validation:** fresh maturin develop --release rebuild into the temp
  oracle venv; `cargo fmt --check` clean; 16 Rust tests pass; focused oracle
  set in the venv (`test_oracle_animals.py test_oracle_crops.py
  test_oracle_mechanics.py test_oracle_replay.py test_oracle_offline.py
  test_oracle_import_isolation.py test_fast_env.py`): 78 passed / 1 justified
  skip; full repository suite under system Python: 266 passed / 62 skipped /
  0 failures.
- **Not claimed:** town/shop consumption-unlock parity, global RNG/day-end
  sweeps beyond unavoidable weed observation, broad random/full episodes,
  >16 hands, closed-loop A/B, benchmarks, training safety.

## 2026-08-23 — Stage 2b slice 2: Crop/Seed/Tile Lifecycle Differential Parity

Drove the crop/seed/tile lifecycle cluster to zero first divergence against
the real pinned official 1.32.7 engine and committed the bounded result on
`main` (no push).

- **Probes:** new `tests/test_oracle_crops.py` — 16 focused tests over 15
  real-official scenarios (2,136 turn pairs, 74 day boundaries) covering the
  `CROPS` constants table, atomic PLANT seed consumption and fresh-tile fields
  (`consecutive_unwatered=1`, ongoing vs single-harvest lifespans), silent
  no-op invalid plants, group PLANT demand blocking including phantom hand
  entries, WATER once-per-day with the single-harvest bonus window, FERTILIZE
  active-day+2 expiry and yield caps, HARVEST pre-first-yield no-op and full
  drain, carrot replant field resets, unwatered decay to WEED plus DIG
  recovery, mature melon two-step unit decay to WEED, ongoing tomato/
  strawberry interval accrual with gap days and end-of-life decay, DIG never
  clearing a placed animal, and wrong-tile guards.
- **Divergence found and fixed:** official `_decay_plants` decrements
  `yield_units` unconditionally when the decay schedule fires and converts the
  tile to WEED at `<= 0`; the fast engine gated on `yield > 0` and converted
  at `== 0`, so an ongoing crop harvested down to exactly zero yield at its
  production completion stayed alive as a zero-yield PLANT forever instead of
  becoming a WEED at `max_lifespan_step`
  (`rust/kaggriculture_env/src/lib.rs`, locked by
  `test_ongoing_tomato_daily_interval_harvest_survival_and_zero_yield_decay`).
- **Validation:** `cargo fmt --check` clean; focused
  `pytest tests/test_oracle_crops.py` in the temp oracle venv: 16 passed;
  prior-worker evidence on this exact tree: fresh maturin rebuild, focused
  oracle set 66 passed / 1 skipped, full suite 266 passed / 50 skipped.
- **Not claimed:** broad whole-engine parity and training safety remain open;
  the >16 hired-hands observation deferral is preserved unchanged.

## 2026-08-23 — Stage 2b slice 1: Worker/Ordering/Hiring/Market Differential Parity

Drove the first targeted mechanics cluster to zero first divergence against
the real pinned official 1.32.7 engine and committed the bounded result on
`main` (no push).

- **Probes:** new `tests/test_oracle_mechanics.py` — 27 focused scenarios
  (~60 primitive turns total) covering worker inventory (unbounded PICKUP
  subject to shed stock, item→quantity semantics, seeds never carried, day-end
  drop with overflow discarded), same-turn ordering (workers before market:
  pickup frees shed room before same-turn buy, same-turn buys not pickable,
  deposit-then-sell), hiring (Fibonacci prices incl. `farmHandCostMult=3`,
  hour-0 hire, no same-turn act, next-turn availability, unaffordable stop,
  daily reset), and market (10-slot truncation, both-player per-slot lockstep
  from the same pre-commit inventory, atomic HIRE/BUY_LAND, mid-order aborts
  on funds/shed capacity, unbounded quantities, zero-net round-trip). Each
  scenario replays identical action pairs through both engines with full
  canonical compare every turn; key mechanics additionally pinned with exact
  semantic assertions.
- **Divergences found and fixed (each with a named regression):**
  1. `fast_env/api.py` money decode: raw f32 normalize(10000) round-trip
     produced spurious divergences on any money change (2993.0 vs
     2992.999755859375); decode now rounds to the exact integer money.
  2. `rust/.../lib.rs`: transition quantities were clamped to
     `MAX_QUANTITY=100`; official order/PICKUP/PLACE quantities are unbounded
     (first evidence: BUY_SEED WHEAT 150 → official 150 seeds/1500 money vs
     fast 100/2000). Clamps replaced with resource bounds, BUY_SEED cost
     computed in i64, official `_process_market` 100k per-slot lockstep escape
     mirrored, and the PLACE shed path gained its official `n <= 0` no-op.
  3. `fast_env/api.py`: malformed actions raised ValueError where the
     official interpreter is a silent no-op (11th market order, unknown unit
     op, seed-name PICKUP, unknown PLANT crop, non-dict action, missing
     farmer, non-integer quantity); translation now emits no-op rows and
     truncates hands/market lists like the official interpreter.
- **Validation:** fresh GNU-toolchain maturin rebuild into the temp oracle
  venv; `cargo fmt --check` clean; 15 Rust tests pass; focused oracle set in
  the venv: 50 passed, 1 justified skip; full repository suite (system
  Python): 266 passed, 34 skipped (official-engine live tests skip without
  the venv), 0 failures — run with `--basetemp` under `Temp\opencode` because
  the shared `pytest-of-liuyi` temp root is access-denied in this worktree
  (pre-existing environmental issue, unrelated to these changes).
- **Deferred with evidence:** >16 simultaneous hired hands (official has no
  cap; fast core fixes 16 hand slots + fixed observation block; reaching 17
  needs ≥4180 money of hires inside one day). Recorded in `MECHANICS.md`.
  No full-parity or training-safety claim is made.

## 2026-08-23 — Stage 2a: Official Differential Oracle Implemented and Validated

Completed, validated, documented, and committed the Stage-2a official
differential-oracle infrastructure (decision D-020). Single bounded commit on
`main`; no push.

- **New modules (`oracle/`):** `provenance.py` (exact-pin guard:
  `kaggle-environments==1.32.7`, wheel SHA256 `2a1bb862...c4c8f`, interpreter
  files byte-matching upstream commit `28b6d8af...ab8c`); `backend.py`
  (official/fast engine seam, lazy official imports); `official_backend.py`
  (raw same-pair submission, full status-history anomaly detection so terminal
  DONE cannot mask ERROR/INVALID/TIMEOUT); `canonical.py` (one canonical
  schema: step/day/hour, both farms with full board + crop/animal lifecycle,
  per-seat private shed/seeds/inventories, market inventory/prices/params,
  town shops with duplicate multiplicity, rewards, statuses; field-path deep
  diff); `replay.py` (same-action turn-by-turn replay, first-divergence report
  with seed/step/day/hour/path/values/actions, deliberate-corruption mutator
  seam). Usage + temp-official setup: `oracle/README.md`.
- **Fast API fixes (`fast_env/api.py`, regression-tested):** wire unit
  operation ids now translate to the Rust core's internal op codes via
  `UNIT_OP_CODES` (previously e.g. PLANT wire 8 landed in the build-structure
  arm internal 8); observation decoding inverts the FIXED
  `generated_protocol::SEASON_STEPS = 720` instead of the configurable
  `episodeSteps`.
- **Validation:** fresh GNU-toolchain rebuild of the extension into the temp
  oracle venv (`Temp\opencode\kagg_oracle_venv`, rustup/cargo homes under
  `Temp\opencode`), then focused run against the REAL official engine:
  23 passed, 1 justified skip (in-process Kaggle-import assertion is vacuous
  when the oracle legitimately imported it; fresh-process isolation tests own
  that guarantee) — covering initial-state parity, short legal traces,
  both-seat privacy, deliberate corruption at exact turn/path, terminal
  rewards at `episodeSteps=3`, provenance tamper rejection, and fresh-process
  import isolation. New regressions added to `tests/test_fast_env.py` for
  wire-op translation (+ behavioral PLANT-arm proof) and fixed-720 decode with
  `episodeSteps=5`. Day-boundary smoke: 28 pass-only turns across day 0 -> 1,
  zero divergence, ~0.9 s. Full repository suite (system Python, no official):
  266 passed, 7 skipped (official-dependent oracle tests skip via provenance
  guard); an unrelated pre-existing `PermissionError` on the stale
  `Temp\pytest-of-liuyi` pytest root required `--basetemp` under the temp root.
- **Explicitly NOT done (Stage 2b):** broad mechanic probes, random/legal-ish
  corpus, multiple full 720-turn episodes, closed-loop A/B, benchmark report.
  No full-parity or training-safety claim is made.

## 2026-08-23 — Issue #1 Implemented: Deterministic Closed-Loop BC Executor V0

Implemented the complete issue #1 packet (`Codex packet: minimal closed-loop
BC executor V0`) in staged local commits on `main`, all validated before
acceptance. No push occurred until the docs commit of this date.

- **Commit chain:** `11e85fa` live observation encoder with exact BC adapter
  parity; `b998eb4` immutable `DailyPlan`, manager wrapper (injection seam,
  checkpoint/fake providers, once-per-day caching), mechanical feasibility
  projection; `cef81fa` deterministic animal layout + minimum-change crop
  reconciliation; `6c5b7fe`+`a8709d0` explicit per-turn task records and
  generation (COLLECT_FERTILIZER gated on canonical raw
  `fertilizer_available`); `01dbf0a` greedy foreman dispatch; `fb01ab6`
  stateful agent integration (hiring, purchases, land, six-bin sells,
  diagnostics, safe-mode fallback, smoke harness); `ed1685a` correctness
  repairs (global seed legality/reservation, honest market-cap bookkeeping,
  continuously current achieved diagnostics, non-tautological privacy test).
- **Closed loop per turn:** live schema-v3 obs -> once-per-day manager call
  with previous-day realized-labor feedback (observed `hires_today`
  progression priced by the exact Fibonacci hire cost) -> requested/feasible
  plan projection -> layout/reconciliation -> task regeneration ->
  foreman dispatch -> bounded deterministic market queue (SELL in the active
  four-hour bin clipped to actual shed inventory via `clip_sell`; hour-0-only
  crude workload hiring; exact-shortage BUY_SEED/BUY_PRODUCT/BUY_ANIMAL and
  single BUY_LAND) -> legal-shaped action dict + JSON diagnostics +
  deterministic all-PASS fallback on any runtime failure.
- **Seed mechanic (1.32.7):** `PLANT <crop>` consumes the global own
  `private.seeds[crop]` pool atomically at the engine; seeds are never
  picked up or carried. The foreman reserves global seeds per crop within
  each turn (deterministic across workers); blocked plants stay unassigned
  with an honest `no_global_seeds` reason; shortages surface as BUY_SEED.
- **Validation:** full suite grew 102 -> 249 tests, all passing (authorized
  pytest basetemp required on this worktree). Live encoder parity verified on
  synthetic and real replay observations. Determinism coverage at every
  layer. A 719/720-turn replay-observation plumbing smoke exercised the full
  agent with zero illegal shapes and zero fallback errors; counterfactual
  actions were NOT executed by an engine, so this proves shape/state
  robustness only, not game legality or quality. A real
  `kaggle_environments` 1.32.7 game was NOT run because the package is not
  installed in this worktree; the smoke harness detection/skip path
  (`SKIP:` message, exit code 3) is verified and nothing was installed or
  vendored.
- **Deliberate V0 simplifications preserved** (see
  `research/EXECUTOR_V0_PLAN.md`): no reserved animal zone/future onset
  prediction; centralized lifecycle-proxy sacrifice score; sticky layout, no
  facility optimization; greedy assignment without search/VRP; one-step
  Manhattan movement without sidestep/pathfinder; soft inventory
  specialization; crude hour-0 hiring (`tasks_per_worker=10` provisional);
  mechanical feed/resource procurement without emergency-feed safety buys;
  literal six-bin selling with no price timing; market cap can defer
  lower-priority hires/buys to next-turn recomputation; no opponent strategy,
  learned executor, or PPO.
- **Newly recorded assumptions/backlog:** shed access uses the four center
  tiles as overwhelmingly observed valid locations from elite replays, not a
  source-locked universal rule; movement conservatively avoids LOCKED tiles
  even though the engine may permit stepping there; worker interaction
  requiring the current tile needs engine-smoke confirmation; pickup batch 5
  is provisional; the farmer-anchored layout can thrash before the first
  build commits (documented revisit, intentionally not repaired in this
  packet).

## 2026-08-22 — Canonical Schema v3: Official Worker `[x,y]` Tile-Lookup Correction

Correctness fix under D-018 (commit `e67f1b7`; no architecture change; D-019
untouched).

- **Root cause:** official 1.32.7 worker positions are `[x, y]` with board
  lookup `tiles[y][x]`, but `_events_from_action` unpacked them transposed
  (`y, x = pos[0], pos[1]`), so every tile-dependent worker event read the
  transposed tile. Affected: CARE animal attribution, FERTILIZE crop
  attribution, HARVEST item attribution, and DIG replaced-tile labels. PLANT
  is action-provided and was never affected.
- **Fix:** `x, y = int(pos[0]), int(pos[1])`; bounds checked in x/y;
  `tile = tiles[y][x]`; emitted ledger tile coordinates remain canonical
  `[y, x]`. Honest unknown behavior preserved: CARE stays `animal: null`
  unless the actual pre-action tile establishes GOOSE/COW/SHEEP; a decoy at
  the transposed tile can never be attributed.
- **Schema:** `SCHEMA_VERSION` bumped 2 → 3 (single authoritative constant).
  Parquet storage and the BC adapter expect v3 via the imported constant;
  v1/v2/mixed logical records and Parquet files fail loudly in both
  `replay_daily.storage` and the BC loader. No migration: regenerate from raw.
- **Asymmetric regression tests:** worker at `[x=2, y=5]`, actual tile only at
  `board[5][2]`, deliberately different decoy at `board[2][5]`, across CARE
  (species + honest unknown + hand actor + both seats), FERTILIZE, HARVEST,
  DIG, and PLANT coordinate semantics. Real-sample smoke expectation updated
  to the official convention. Full suite: 102 tests pass.
- **Local sample audit:** regenerated
  `data/canonical/2026-08-20-sample.parquet` at schema v3 (900 records,
  805,435 bytes, SHA-256
  `932617FF02EF7B5DF74C5AF2E766F3EC3423B3FAC24513992E218C0629F4054E`);
  exact read-back parity; every raw pre-action worker op reconciled against
  the emitted ledgers with zero mismatches. Corrected counts: CARE COW 5,614 /
  SHEEP 4,091 / GOOSE 20 / unknown 0 (9,725 entries); FERTILIZE 2,020 entries
  (known 2,011 = STRAWBERRY 1,960, WHEAT 27, TOMATO 24; unknown 9); DIG 889
  submissions; HARVEST 11,948 submissions with 11,905 item-bearing ledger
  entries (43 no-item submissions intentionally omitted under the existing
  item-only HARVEST ledger semantics). Total tile-dependent submissions:
  **24,582**.
- **Erratum:** the implementation commit report/message stated 14,582
  reconciled worker ops; the correct total is 24,582 (CARE 9,725 +
  HARVEST 11,948 + DIG 889 + FERTILIZE 2,020). The reconciliation itself was
  complete; only the reported aggregate was wrong. Recorded here without
  amending history.
- The v2 event labels were semantically wrong (transposed lookups), so no
  migration path exists or is desired; all processed corpora must be
  regenerated from raw replays at v3 before BC training.

## 2026-08-22 — First BC Manager: Adapter/Baseline, Tile Transformer, Training CLI

Implemented the complete first behavior-cloning stack over the canonical
schema-v2 records; D-019 published. No full training run (five-day v2 corpus
pending).

- **Compact BC data layer** (66fbaea): bc_manager/adapter.py reads
  schema-v2 Parquet directly with PyArrow (dotted-path projection, no logical
  reconstruction), verifies schema_version == 2, selects rows only by the
  date allowlist + equal min_score cutoff, and converts once into compact
  NumPy arrays (own/opponent-public boards, resource/market/town/labor/day
  features, count/CARE/sell targets). bc_manager/baseline.py fits a per-day
  empirical baseline on train rows only. Date-held-out splits; never random.
- **Tile Transformer + loss** (6b9db3b): stateless day/hour0 manager —
  shared tile encoder (kind/crop/animal embeddings, scaled lifecycle
  numerics with NaN indicators, bool/presence channels, row/col embeddings),
  MANAGER + 5 global tokens (106 sequence length), standard norm-first
  TransformerEncoder, structured heads for crop/animal/land/fertilizer/CARE
  counts plus sell presence and log1p quantity. Seven fixed-weight group
  losses; metadata keys rejected loudly; opponent PUBLIC board optional/off
  by default. Default config 1,071,040 trainable parameters; tiny CPU
  config for tests.
- **Training CLI** (86b8433): python -m bc_manager.cli — in-RAM tensor
  dataset, AdamW + gradient clipping + optional CUDA AMP, sparse diagnostics
  (exact/MAE/nonzero recall incl. per-animal GOOSE visibility) beside the
  train-only day baseline, early stopping, atomic best/last checkpoints that
  serialize model config and reload to equivalent eval outputs.
- **Validation:** 92 tests pass (65 data-layer, 16 model/loss, 11 training)
  including real forward/backward on the local 900-row v2 sample, genuine
  tiny-batch overfit (~770x loss reduction), checkpoint equivalence, and a
  synthetic two-date end-to-end CLI smoke. Full five-day training was not
  run; old v1 data fails loudly everywhere.
- **Decision publication:** D-019 added to DECISIONS.md; implemented note
  .agents/notes/implemented/2026-08-22-use-configurable-tile-transformer-for-initial-bc-manager.md;
  usage/handoff commands in bc_manager/README.md.

## 2026-08-22 — Canonical Schema v2: CARE-by-animal Correction

Logical extension/correction under D-018 (no redesign; D-017/D-018 unchanged).

- `events.care` now records every submitted CARE intent with its pre-action
  tile `[y, x]`, the animal identity established by that board tile (or
  `null`), and the exact primitive hour. CARE previously fell into the generic
  `worker_ops_other` aggregate with no species attribution.
- CARE takes no arguments: it targets the worker's own pre-action tile under
  the verified alignment rule (`steps[i].action` transforms `obs[i-1]`).
  Unknown/non-animal CARE stays `animal: null` and never increments a species.
- New derived target `targets.care_by_animal` exactly mirrors the known-animal
  daily counts for GOOSE/COW/SHEEP.
- `SCHEMA_VERSION` bumped 1 → 2 in `replay_daily/constants.py`. Parquet Arrow
  schema, conformance guards, normalization, reconstruction, and round-trip
  equality extended for CARE. Writers reject logical records with a foreign
  `schema_version`; `read_parquet`/JSONL readers fail loudly on v1 or
  mixed-version processed data. No migration machinery: regenerate from raw.
- Real-sample validation: all 15 local replays re-extracted to v2 Parquet
  (900 records, exact read-back parity). 6,642 known CARE events (COW 3,724,
  SHEEP 2,918) plus 3,083 unknown intents preserved as `animal: null`; GOOSE
  does not occur in the local elite sample and is covered synthetically.
  Regenerated artifact: `data/canonical/2026-08-20-sample.parquet`, 806,735
  bytes, SHA-256 `F7176542FE34B72DCEFCF70799DEDC34F17D8DB2DBF372680BD0FEC597023441`.
- 57 focused tests pass (13 new covering CARE attribution, hours, seats,
  privacy, merge/target mirror, round-trip with null animal, no double count,
  and v1/mixed rejection). Evidence in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`.

## 2026-08-21 — Parquet Production Storage for Canonical Records

Adopted Zstandard-compressed Parquet as the production canonical physical
format under D-018; the logical `(episode, seat, day)` schema is unchanged.

- `replay_daily/storage.py` maps one logical record to one nested Arrow row;
  fail-loud conformance guards reject unknown canonical keys instead of
  silently dropping them, and bare string tile sentinels round-trip exactly.
- CLI `extract` defaults to `--format parquet`; JSONL remains an explicit
  debug/inspection output and is never written automatically.
- Full-sample validation: all 900 records from the 15-replay sample compare
  with exact Python equality across fresh extraction, the previously validated
  JSONL, and the new `data/canonical/2026-08-20-sample.parquet` (795,154 bytes,
  98.8% smaller than JSONL).
- Benchmark (single process): extraction ~89 MB/s of raw replay (~0.3 h per
  100 GiB projected), so no parallel preprocessing stage was justified; raw
  Arrow reads are faster than JSONL parsing. No NPZ evaluation needed.
- The old ignored JSONL sample was deleted after parity confirmation; raw
  replays remain the source of truth. Evidence in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`. PyArrow dependency declared
  in `requirements.txt` (`pyarrow>=14`). 44 tests pass.

## 2026-08-21 — Canonical Daily Sample Validated

Completed the local 1.32.7 canonical replay foundation and generated the ignored
15-replay sample at `data/canonical/2026-08-20-sample.jsonl`.

- 900 records cover both seats and all 30 days for each replay;
- corpus-wide boundary, privacy, lifecycle, hire, land, fertilizer, shop, and
  six-window SELL checks passed;
- the output is deterministic across a second CLI run;
- representative findings are recorded in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`.

Training and the deterministic executor remain unstarted; no broader dataset
processing was performed.

## 2026-08-16 — 1.32.7 Situational Resource Rebalance

### Upstream engine change

Reviewed and source-confirmed merged upstream PR `Kaggle/kaggle-environments#1399` (`Make underused resources situational`).

Upstream `pyproject.toml` now declares `kaggle-environments` version `1.32.7`.

PR metadata:

- head commit: `1fbd3b7571653434329d288dee9e068f54ff01c0`;
- merge commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.

The host stated this should be the last balance change except game-breaking bugs. Live leaderboard rollout was announced but has not been independently server-locked in this repository.

### New hinge price curve

PR #1399 adds a new scarcity-side market function:

`u = x / T`

`hinge = u + 8 * max(0, u - 1)^2`

where `x = I0 - market_inventory` when a product is scarce.

Behavior:

- below `T`, the curve is linear in normalized scarcity;
- above `T`, a quadratic term creates a steep price spike;
- `hinge(T) = 1`, preserving the meaning of the market target parameter.

### Products changed

- carrot: scarcity curve `log`/0.20 → `hinge`/1.00, `T=450`, knee inventory 9550;
- tomato: `linear`/0.40 → `hinge`/0.40, `T=200`, knee inventory 9800;
- egg: `linear`/0.40 → `hinge`/0.40, `T=332`, knee inventory 9668;
- glut-side curves are unchanged.

Tomato and egg are therefore unchanged through their old linear knee and diverge only in deeper scarcity. Carrot changes more broadly because its scarcity target also increases from 0.20 to 1.00.

Explicit source test values include:

- carrot: 9550 → $70, 9400 → $113, 9100 → $385;
- tomato: 9800 → $84, 9700 → $144, 9500 → $552;
- egg: 9668 → $70, 9502 → $120, 9170 → $460.

### Random-shop interaction

Relevant shop demand noted by the PR:

- carrot: pet cafes and farmers markets; pet cafe is single-product and consumes double;
- tomato: pizza shops and farmers markets;
- egg: bakeries and brunch spots.

Because shops have been sampled with replacement since 1.32.6, duplicate demand can push these products through their scarcity knees.

Host-reported substantial-price-increase frequencies assuming **no production**:

- tomato: ~50% of games;
- carrot: ~26%;
- egg: ~22%.

These are recorded as host-reported statistics, not engine constants, and should be reproduced empirically under the locked local engine.

### Strategic interpretation

The change strengthens the RL-centered design rather than weakening it.

The new decision problem is conditional:

- detect that an episode is developing an unusual demand regime;
- estimate whether a crop/animal pivot can produce before the opportunity disappears;
- account for the fact that our own production/sales reduce scarcity;
- anticipate whether the opponent is already producing or will react;
- decide whether the expected competitive gain exceeds the opportunity cost of changing the farm plan.

This is especially relevant to end-game crop rotation and situational goose/egg production.

### Reward-design consequence

Added a durable guardrail against naive mark-to-market shaping.

Under the hinge curve, `quantity × current spot price` can hugely overstate realizable value because selling quantity moves the price back toward the knee. Future reward/potential design should use marginal-price-aware liquidation, time-to-sale constraints, exact/approximate simulation, or validated learned continuation value.

### RL observation consequence

Planned product entities should expose:

- current inventory/price;
- base, `I0`, `T`;
- curve shape and target parameters;
- normalized scarcity;
- signed distance to the knee;
- recent inventory/price velocity;
- shop multiplicity/known demand;
- own/opponent production pipeline;
- time remaining.

No static `GOOD_PRODUCT` feature or fixed product-priority table should be encoded.

### New pre-training studies

1. reproduce the host-reported no-production scarcity frequencies;
2. measure first knee-crossing time and maximum prices by shop composition;
3. estimate latest profitable pivot time for carrot, tomato, and goose/egg production;
4. measure how opponent production suppresses the opportunity;
5. test public deterministic baselines for 1.32.7 staleness;
6. verify reward potentials do not exploit temporary hinge spot prices.

### Files updated

- `CURRENT_STATE.md`
- `MECHANICS.md`
- `PLANS.md`
- `DECISIONS.md`
- `HISTORY.md`
- `research/RL_DESIGN.md`

## 2026-08-07 — 1.32.6 Town Rebalance and RL-Centered Planning

### Upstream engine change

Reviewed and source-confirmed merged upstream PR `Kaggle/kaggle-environments#1394` (`Kaggriculture town rebalance`).

The change:

- changes default `townCenterSellInterval` from 12 to 24 turns;
- therefore reduces default town-center consumption from twice/day to once/day;
- removes the old town-center demand schedule that increased to 2× after day 10 and 4× after day 20;
- samples town shops with replacement from the full shop table;
- allows duplicate shop names;
- makes each duplicate shop instance consume independently;
- caps total unlocked shop instances at 8 as before in effective maximum count.

Confirmed upstream package source snapshot `bded87b0d7879078c726a93a4884d044f79c4eed` identifies `kaggle-environments` as version `1.32.6`.

The live leaderboard rollout was announced but has not yet been independently locked to an observed server build in this repository.

### Strategic interpretation

The rebalance increases the value of adaptive economic behavior:

- weaker town-center demand means player-generated oversupply should persist longer;
- product gluts and opponent sale timing matter more;
- shop replacement sampling creates materially different per-episode demand regimes;
- duplicated shops can strongly favor particular product categories;
- fixed deterministic public schedules should become less universally optimal across seeds.

The town-shop observation must be treated as a multiset/count vector rather than a binary set.

### RL direction clarified

The project direction changed from "deterministic route first, learning later if useful" to an explicitly **RL-centered hybrid**.

The intended division of responsibility is now:

- learned policy owns production, resource allocation, task assignment, adaptation, and market strategy;
- deterministic infrastructure owns pathfinding, mechanical legality, task execution/persistence, and bookkeeping;
- candidate generation may remove impossible actions but should not encode strategic preferences by hiding mechanically valid actions.

Raw primitive movement PPO from scratch remains deferred. This is now understood as an action-abstraction decision, not a rejection of RL.

### Public RL discussion considered

A competitor reported poor results from standard PPO/SAC attempts because of:

- large observation/action spaces;
- long crop reward delays;
- catastrophic cascades from small logistical mistakes;
- difficulty learning exact watering/feed/seed timing through random exploration.

This was treated as evidence for imitation bootstrap and hierarchical action abstraction rather than evidence against RL.

### New RL design

Added `research/RL_DESIGN.md` covering:

- hierarchical worker-task intent actions;
- deterministic execution of selected intents;
- dedicated autoregressive market head;
- action masking rules;
- turn-level vs event-driven vs hybrid decision frequency;
- entity-based observation/model design;
- recurrent opponent-state inference;
- W/L/T terminal objective;
- potential-based reward shaping;
- auxiliary prediction losses;
- public-agent behavior cloning;
- PPO robustness training and population self-play;
- pre-training experiments for shop variance, market sensitivity, action abstraction, reward sanity, and memory.

### Reward planning

Current leading reward direction:

- final competitive objective aligned with win/tie/loss;
- avoid arbitrary positive maintenance rewards such as watering/harvesting bonuses;
- investigate potential-based shaping using liquidation/future economic value;
- use auxiliary prediction tasks for representation learning instead of silently changing the objective;
- compare `gamma=1.0` with values extremely close to one because the true objective is terminal.

### Demonstration/bootstrap plan

Strong deterministic public agents will be used as training data in addition to opponents:

1. archive exact agent/version provenance;
2. run over varied seeds/shop regimes/opponents;
3. collect state/action trajectories;
4. map primitive actions into intent-level labels;
5. behavior-clone initial competence;
6. fine-tune with PPO/self-play so the model can depart from fixed public scripts.

### Next-week planning agenda

While Pokémon work finishes and Kaggriculture has time to stabilize:

1. map exact 1.32.6 actions and RNG;
2. design worker-task candidate generation;
3. design market quantity/order representation;
4. decide policy decision frequency;
5. version the observation schema;
6. formalize potential functions/reward invariants;
7. design BC trajectory format;
8. design PPO/self-play curriculum;
9. estimate simulator/vectorization throughput;
10. define evaluation/promotion gates;
11. recheck upstream engine changes before implementation/training.

### Files updated

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `HISTORY.md`
- new `research/RL_DESIGN.md`

## 2026-08-06 — Repository Initialization

### Repository

- Created private GitHub repository: `BillXu21/Kaggriculture`.
- Initialized the default branch with project documentation.
- Established continuity files to reduce context loss between chats and agents.

### Strategic Assessment

- Current game structure appears highly deterministic.
- Physical farms are separate, with limited direct interaction.
- The shared market is the primary adversarial coupling mechanism.
- Strong public leaderboard entries are currently dominated by copies or variants of a few deterministic public notebooks.
- The project will remain in planning and mechanics-tracking mode while the engine and rules continue changing.

### Initial Architecture Hypothesis

The initial architecture hypothesis was:

1. deterministic production-route executor;
2. state-based validation and repair;
3. phase-level replanning;
4. opponent-aware market and production policy;
5. coherent expert selection;
6. optional optimization or learning at the macro level.

This was superseded/clarified on 2026-08-07: the project now intends RL to own meaningful strategic decisions, with deterministic code limited primarily to mechanics/execution.

### Research Findings Carried Into the Repository

The following findings were established before repository initialization and should be reverified against the exact live engine before implementation:

- Two players each manage a separate 10×10 farm.
- Matches span thirty days with twenty-four turns per day.
- Banked money determines final reward.
- Unsold inventory has no terminal value.
- Crop and animal schedules are largely deterministic.
- The market is shared and uses inventory-dependent pricing.
- Some daily events are driven by the episode seed.
- Public state exposes enough opponent farm information to support strategy fingerprinting and supply forecasting.
- Strong public strategies use mixed industrial production rather than simple single-crop loops.

### Public Baseline Direction

The first competitive reference should be a strong public deterministic route, preserved with:

- source URL or notebook identity;
- download date;
- immutable file hash;
- engine version assumptions;
- any local modifications;
- known performance evidence.

The project will not rely on redistribution-license concerns as a reason to avoid downloading publicly available Kaggle notebook artifacts, but provenance and third-party boundaries should still be tracked accurately.

### Compute and Workflow Lessons Imported From Pokémon TCG Work

- Chat-context loss can cause stale configuration reuse and wasted compute.
- Every expensive run must be specified in a durable file before execution.
- Current state must remain concise and authoritative.
- Full history must preserve failed experiments, commands, hashes, and output paths.
- Evaluation should not depend on a single seat, weak opponents, or unversioned artifacts.

### Files Established

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `HISTORY.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `AGENTS.md`
- `research/README.md`
- `.gitignore`

### Next Actions At Initialization

1. Establish the exact current engine identity.
2. Archive important public notebooks and agents.
3. Catalog major strategy families.
4. Define the initial fixed-seed, seat-swapped evaluation protocol.
5. Delay competitive implementation until those contracts are recorded.

## 2026-08-24 - Issue #7 executor V0.5 overnight pass (worktree branch executor-v05-overnight)

- Base `32fef4ac295e9addaf01cd339eee60a0ad14eaca`; final `885adadda351a39e2797058fe3c4c8cb5f506bac`; no merge to main.
- Validated full-game fast-engine reconstruction of four official replay JSONs (719 turns each, both seats) - exact state parity modulo cosmetic field names; fixed `derive_animal` KeyError on fast-engine age-only animal tiles.
- Built replay manifest (19 episodes; 98178196 classified failure_specimen, rewards [42,42]) and canonical expert daily-plan extraction; built isolated one-day slice harness with boundary verification (38/38 verified on expanded set).
- Accepted fixes: movement legality (locked quadrants walkable), mechanics-derived water urgency classes, hub-anchored coordinated layouts with weed reclamation, build prerequisite gating, plan-implied CARE/FERTILIZE eligibility, sequential same-turn SELL->HIRE accounting, any-hour workload hiring (3-hire cap removed), exact-cost buy gates, accumulated diagnostics.
- Rejected experiments: persistent task ownership hint (wealth 894->864 regardless of bonus size); travel-aware hiring over all tasks (+5.5%->+1.5%); both reverted with evidence.
- Expanded paired result: mean one-day wealth delta 1879 -> 1981 (+5.4%), candidate better on 24/38 slices, day-end weeds 53 -> 22, harvestable leftovers 1 -> 0. Worst regression traced to genuinely saturated single days (crew capacity < interaction+travel demand).
- Tests: 461 passed / 108 skipped locally (full suite incl. jax train/parity/benchmark). Artifacts: research/EXECUTOR_V05_OVERNIGHT.md + three paired day-slice JSONs.

## 2026-08-27 - Issue #16 Native Batched Fast-Environment Path

- Base commit: `e63e8337ba9e30a6f394d69da23da538ed7ad6c2`; isolated branch:
  `throughput/16-batched-fastenv`.
- Profiled the scalar path at N=1/N=2. Per environment-turn costs in us were
  action encoding 8.19/8.81, native transition 0.33/0.14, native observation
  writing 2.88/2.79, observation decode 220.20/223.49, canonicalization
  31.66/33.07, runner farm copies 114.66/116.26, and total wrapper step
  225.58/236.65. Native transition mechanics are not the bottleneck.
- Added `fast_env.BatchedFastEnv`, which owns one native Rust batch engine for
  N games, accepts explicit seeds, and reuses action/observation/reward/status
  buffers. Added `oracle.batched_backend.BatchedEngineBackend` and the
  `FastBatchedBackendAdapter` rollout seam.
- Added opt-in `RunnerConfig(batch_backend=True)`. Existing executor/opening/
  manager/PPO interfaces and the scalar runner remain unchanged; only native
  environment ownership and the per-turn environment call are batched.
- Parity evidence: reset plus 40 mixed turns for seeds 7/19, episodeSteps=3
  terminal behavior, direct action-buffer encoding, private-view isolation,
  and a two-game 130-turn runner fixture all passed. Runner final banks,
  statuses, transition counts, and trace digests matched scalar mode.
- Local benchmark: raw native `step_into` reached 232k/257k/267k transitions/s
  at N=1/2/8 with one thread. Full Python batch wrapper reached 7,570/7,752/
  7,636 transitions/s versus scalar 4,210/4,193/2,901. The self-play fixture
  improved from 402 to 627 primitive turns/s and `env_step` 0.303 to 0.089 s.
  Results and caveats: `docs/benchmarks/ISSUE16_BATCHED_FASTENV.md`.
- Focused result: 20 batch/scalar fast tests passed; the combined runner test
  passed with an approved temp root. Two unrelated existing tests encountered
  a Windows pytest temp-directory permission error under the default temp
  root. No executor-v07, multiprocessing, or submission files were changed.
- Full repository validation after commit: `836 passed, 15 skipped, 1 failed`.
  The sole failure is the existing official parity comparison: its official
  runner has no trajectory buffer, leaving `plans={}`, while the cached fast
  rollout includes plan records. This failure does not execute the batch path
  and is not caused by the issue-#16 files. Focused batch/runner validation
  remains green (`36 passed, 1 skipped`).
