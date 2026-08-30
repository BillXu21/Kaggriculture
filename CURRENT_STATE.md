# Kaggriculture Current State

Last updated: 2026-08-30

## Animal Crop-Sacrifice Placement Correction
- On `codex/fix-animal-crop-sacrifice-placement`, based on
  `codex/promotion-ratchet-minimal` at `ce4f0e4`, animal
  `crop_sacrifice` slots now clear their currently observed living crop before
  BUILD. The resulting chain is `DIG:y,x -> BUILD_*:y,x -> PLACE:ANIMAL:y,x`;
  stale/non-PLANT claims are reported unresolved rather than executed.
- Focused regressions cover the old missing-DIG failure, exact keys,
  duplicate-DIG reuse, stale claims, and an owned-COW three-observation
  completion sequence. Focused executor/layout/task validation passes `161`
  with `4` expected official-engine skips; executor smoke passes `28` with `2`
  expected skips. Ruff, compilation, and diff checks pass.

## Promotion Ratchet
- The bounded `codex/promotion-ratchet-minimal` patch adds opt-in PPO
  promotion checks (`--promotion-every`) on fixed seeds `3000..3031`, both seat
  orientations. A passing deterministic candidate is saved under
  `promotions/promotion_NNN.npz` with its evaluation JSON and becomes the next
  immutable rollout opponent; the live PPO train state and original BC-E
  identity are preserved. `--max-promotions` supports a stop condition.
- Saved snapshots can be loaded as normal deterministic runner policies with
  `load_ppo_snapshot`. Mid-ratchet training resume is not implemented; resume
  from a complete PPO checkpoint remains the existing stationary path.
- Local focused validation: 25 promotion/CLI tests, Ruff, and compilation.
  The Kaggle experiment was not run.

## Issue #33 Fast/Official Parity Audit
- On isolated branch `codex/issue-33-fast-official-parity-audit`, anchored at
  `57db920689ef80fb373128d4b8129054816d133f`, the rebuilt current Rust
  extension was audited against pinned official `kaggle_environments==1.32.7`.
- Reset underlying state and policy-visible observations matched for seeds
  `7,17,42,123`. Eight seat-swapped current BC-E vs PASS traces and one active
  BC-E vs BC-E trace each passed 719 accepted post-reset transitions to
  terminal with exact same-action state and independent observation/action
  parity. Existing 8-seed legal-ish corpus rerun: 5,752 action pairs, 33
  families, zero divergence.
- Native identity: `fast_env/_kaggriculture_env.cp313-win_amd64.pyd`, SHA-256
  `c71fc02cd7acbce2c2cc8a950f894311bcc7b2b3385880fe26e7f49f99a64ffa`, Rust
  source commit `63c8113585575fd6c3edf1417795eb553b44ddae`. Official provenance
  passed version `1.32.7`, upstream `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.
- No semantic divergence or fix was identified. Classification is
  **provisionally training-safe for tested paths**, not universal parity and
  not permission to use fast score magnitude for promotion. Detail and local
  machine-readable artifacts: `research/FAST_OFFICIAL_PARITY_AUDIT_ISSUE33.md`
  and ignored `artifacts/local/fast_official_parity/`.

## Issue #28 Run Setup
- Branch `codex/issue-28-immediate-plant-water`, base `feaf6ccfb4ac37380d9644f73d2cd7f2fb35474a`.
- Executor-only confirmed same-worker PLANT -> WATER continuation. Initial focused/broader suites passed `141/193` tests; full suite `808 passed, 104 skipped` with two host-sensitive failures.
- Initial fast-engine A/B (8 games, aggressive sell) was misleadingly positive (`72,762.75` vs `55,885.0`, `+16,877.75`).
- Official 1.32.7 A/B exposed 6 animal escapes in ON (OFF 0) with mean delta `-4,975` (`70,017.75` vs `65,042.75`); weeds/work debt still improved.
- Root cause: aggressive sell-all bypassed `feed["shed_reserve"]` protection for WHEAT, enabling `SELL WHEAT -> BUY WHEAT` churn while starving. Fix protects `min(shed[WHEAT], shed_reserve)` in aggressive mode. Small fast sanity (seeds 7,17 both seats, aggressive+plant-water+spare-water, `--turn-trace`) now passes 4/4 with 0 escapes, 0 fallbacks. Promotion remains blocked pending official retest.

## Issue #25
- On isolated branch `codex/issues-25-preventive-watering` from
  `origin/throughput/integration` (`45f8800`), `PLACE` into an existing
  compatible empty structure now survives prior-debt capital suppression;
  dependent PLACE tasks for suppressed BUILDs do not leak through; and
  underfoot PLACE cannot bypass higher-priority hard WATER. The existing
  PASS-only spare-water path is enabled for the production submission and
  default RL executor factory, with a same-day remaining-distance bound.
  Direct `AgentConfig` remains opt-in/off by default for isolated tests.
- Focused validation passes 176 tests with 3 skips across the executor,
  animal-placement, optional-water, cleanup, panel, and RL-factory paths.
  A bounded fast panel used the frozen BC-E epoch-27 checkpoint at
  `C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`,
  seeds `7,17`, seats `0,1`, PASS opponent, `standard_mixed`, 719
  transitions. OFF versus WATER-only: aggregate final bank `198,872` vs
  `286,766`; final WEED tiles `53` vs `39`; worker PASS `1,178` vs `601`;
  movement `10,581` vs `11,975`; fallback errors `0/0`; animal escapes `0/0`.
  Candidate animal residue/target error and work debt were mixed by seed.
  This is fast-engine plumbing evidence only; official 1.32.7 is absent.

## Issue #22
 - RL correctness hotfixes are implemented on `codex/issue-22-rl-hotfixes` from
   `c14b327`: training rebuilds the PPO rollout adapter from every returned
   state; spawned default executor factories carry their exact `AgentConfig`;
   and host-facing fresh PPO inference uses the compiled BC-E forward seams.
   The narrow CPU parity diagnostic matches action tensors and raw logits at
   B=1 and fixed padded B=24. Targeted pytest execution was limited to the two
   allowed invocations; the first encountered the host's protected default
   pytest temp directory, and the second ran before the final import/parity
   assertion corrections. Final lightweight direct checks are recorded in
   `HISTORY.md`; no TPU claim is made.

## Issue #23
- PPO stability controls are implemented on `codex/issue-23-ppo-stability`: `PPOConfig.target_kl` is disabled by default and stops after a completed epoch when full-batch stored-action approximate KL exceeds the threshold. `reject_update_kl` is opt-in and rejects nonfinite or over-ceiling updates by returning the exact prior `PPOTrainState`. Update metrics record per-epoch KL/clip diagnostics, actual epoch/minibatch/row counts, acceptance, and stop/rejection reason. `BestCheckpointRetention` maintains evaluation-driven named best `.npz` checkpoints and stores the replacing evaluation/source provenance in checkpoint metadata. Focused validation passed `34`; no TPU benchmark was run.

## Issue #18
- Evaluation/promotion hardening is implemented in `rl_manager/evaluation.py`:
  fatal anomalies are separate from informational opening/executor diagnostics;
  fixed panels detect missing/duplicate results; economic bank/margin metrics,
  W/L/T, and candidate seat splits are first-class. `PromotionConfig` defaults
  to `W-L >= 6`, mean margin `> 0`, median margin `>= 0`, and zero fatal
  anomalies. The CLI persists exact PASS/HOLD conditions and reasons. Focused
  validation was bounded to two pytest invocations plus direct checks and
  targeted compilation; no full suite or panel was run.
## Issue #21
- A standalone one-process multi-trainer prototype is implemented in
  `rl_manager/multitrainer_benchmark.py`. It creates independent PPO states,
  optimizer trees, RNGs, inference/PPO arrays, and explicit assignments
  `trainer i -> jax.devices()[i]`; it refuses device reuse and records visible
  devices, actual placements, compile latency, synchronized steady-state
  inference/update timings, and sequential versus async-dispatch timing.
- `rl_manager/multitrainer.py` adds a JAX-free identity router that groups
  requests by exact `PolicyIdentity` and rejects unknown or cross-policy
  trainable trajectory rows. Rollout workers remain CPU-only and
  `parallel.py` is unchanged.
- Local tiny CPU smoke is placement/routing evidence only: the default
  unsharded single trainer used one visible CPU device. No TPU scaling or
  real-checkpoint measurement exists. Exact N=1/N=2/N=4 and optional N=8 Kaggle
  commands are in `research/RL_MULTI_TRAINER_TPU.md`.

## Issue #17
- Parallel rollout topology is implemented on this branch: the parent is the sole policy/JAX/libtpu owner; `spawn` workers own independent engine/opening/executor state and exchange only encoded manager-day NumPy rows through bounded queues. Default owner batching remains canonical by policy identity/day; opt-in `RunnerConfig(inference_batch_scope="policy")` mixes days, and `fixed_inference_batch_size=B` pads valid rows to exactly B while routing only real rows. Results and trajectory shards normalize by episode/seat/day, and worker failures or missing/duplicate rows fail loudly. The lazy `rl_manager` initializer plus worker startup guard prevents accelerator imports in CPU workers. Focused batching/CLI/PPO validation passes `37 passed, 2 skipped`; the full RL-manager suite passes `130 passed, 3 skipped`; TPU throughput remains unmeasured. Runbook: `research/RL_MANAGER_PARALLEL_ROLLOUTS.md`.

## Snapshot
- Throughput integration branch `throughput/integration` merges issue #15
  (`3abbe138`), issue #16 (`c8390ac` + `0a75eb6`), and issue #17
  (`85fd92f`). Focused composition checks pass `150 passed, 3 skipped`; the
  normal full suite passes `757 passed, 104 skipped`. The integrated runner
  preserves scalar/default, low-telemetry/read-only, native batched, and
  spawned-worker paths. Official-capable validation retains the known parity
  comparison failure where an unbuffered official result has no plan sidecar;
  no submission behavior was changed.
- Issue #16 batched-fastenv is implemented on branch `throughput/16-batched-fastenv`: `fast_env.BatchedFastEnv` owns one native `RustBatchEnv` for N simultaneous seeded games, reuses action/observation/reward/status buffers, and `RunnerConfig(batch_backend=True)` routes self-play through one native batch step. Scalar-fast parity passed for 2 seeds x 40 turns plus terminal/encoder/privacy checks; a two-game 130-turn runner fixture matched banks, statuses, transition counts, and trace digests. Local fixture throughput improved 402 -> 627 primitive turns/s and runner `env_step` 0.303 -> 0.089 s; detailed profiling/results are in `docs/benchmarks/ISSUE16_BATCHED_FASTENV.md`. No TPU claim; the integration branch composes this backend with the #15 executor and #17 multiprocessing options.
- Issue #16 validation after commit: focused suite `36 passed, 1 skipped`; full repository suite `836 passed, 15 skipped, 1 failed`. The one failure is the existing official parity test comparing a buffered fast rollout with an official run created without a trajectory buffer, which necessarily has `plans={}`; it does not exercise the new batch path.
- Stage 1 / issue #13 packaging and the Stage 5 post-Stage-4 compatibility identity are accepted by the exact fresh-extract verifier. The current deterministic 50-member archive was rebuilt from source revision `11ecead2d5efe8bf87fc0da533c739e344d7eaa6` at `artifacts/local/submissions/bc-e-v07.tar.gz`, SHA-256 `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2`; official 1.32.7 provenance passed, repository-root source paths were absent, full status history had 0 anomalies across 720 entries, and seed-7 candidate seat 1 versus PASS ended at bank `47,290.0`. The pinned current action-trace fingerprint is `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc`. The preserved pre-behavior reference at `8f716bec` remains bank `54,439.0`, fingerprint `516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40`, archive SHA `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf`; strict verification and the loud omitted-`fast_env` regression passed, while production strict=False remains unchanged.
- Stage 5 accepts the retained Stage 4 lifecycle sequencing (`b9c88ff`) despite six negative per-game deltas: isolated-panel mean `63,592.3` vs `60,778.1` (+`2,814.2`), median `65,509.5` vs `60,956` (+`4,553.5`), no `<1k`/`<10k` cases, and no errors, status anomalies, unaffordable orders, or animal losses. Independent review classified the negative cases as labor/cash tradeoffs rather than feed/starvation failures; no distinguishing feed exception or unconstrained cash forecast is justified. Stage 3 fertilizer retention remains rejected and cleanly reverted at `7204103`; aggressive mode again sells FERTILIZER.
- Stage 6 issue #12B spawn/shed waiting is a non-reproduction: the isolated 24-game trace has 24 shed-zone long-`PASS` runs, exactly one per game at worker 1 / `[4,5]` / `d4h1`–`d4h10`, classified as legitimate WHEAT market-fill latency while other workers clear feed/starvation work. The other 45 long waits are hiring-ramp/end-game labor surplus. No animal or critical work loss, unaffordable order, fallback/day/status error, or repeated pickup overdraw was observed; `cf282a1` shed reservation is present. The artifact lacks per-turn assignment reasons and complete private inventory snapshots. This is separate from the resolved Kaggle AFK packaging omission; no instrumentation or executor patch is justified. Detail: `research/EXECUTOR_V07_STAGE6_ISSUE12B_SPAWN_SHED_WAIT.md`.
- Stage 7 issue #12C and the Stage 6/7 PASS-only idle-cleanup layer are now measured and documented in `research/EXECUTOR_V07_IDLE_CLEANUP_PASS_ONLY.md`: normal foreman dispatch runs first, optional candidates can replace only literal normal `PASS` actions, candidates are recomputed each turn, and cleanup does not persist or enter hiring/debt/market accounting. WATER-only (`optional_spare_watering`) and WEED+WATER (`optional_idle_cleanup`) remain distinct; diagnostics record `cleanup_mode`. In the official 24-game A/B/C panel, WATER-only was mean `67,712.6` vs OFF `63,592.3` (+`4,120.3`) with median delta `-267.5`, and WEED+WATER was mean `66,886.3` (+`3,294.0`) with median delta `+1,153.0`; both were `16W/0T/8L` with materially worse tails of `-15,820` and `-16,669`. Normal non-PASS changes were zero and replacement rates were 90.96%/92.69%, but strategic value remains mixed and seed-dependent. Cleanup stays **OFF by default**; next step is independent review, not PPO or promotion.
- Issue #15 executor hot-path optimization is implemented on isolated branch `throughput/15-agent-hotpath`: one canonical own-board snapshot is reused across task/feed/sell/diagnostic paths by default; explicit `RunnerConfig.low_telemetry=True` disables per-turn executor snapshots, and `read_only_agent_observations=True` safely removes per-call observation deep copies. Fixed-seed base parity is exact for 719 primitive joint actions, banks, statuses, and digests. Local medians versus base were: default N=2 `0.261 -> 0.290` games/s and N=8 `0.279 -> 0.388`; low-telemetry/read-only training mode N=1/2/8 `0.432/0.405/0.418` games/s. Full report: `research/EXECUTOR_HOTPATH_ISSUE15.md`. No Kaggle TPU claim; no issue #16/#17 dependency.
- Executor V0.7 / issue #7 is closed and frozen at `a7c826d`: R4 was rejected after the real BC-E fixed-plan 7d regression, while the accepted panel outcome ledger (`02984a0`) and survival WHEAT shed-room clamp (`a7c826d`) are retained. The clamp uses official/default `shed_capacity=100` and preserves survival-before-hire ordering. The bounded final no-R4 panel retains prior-debt suppression **ON** as explicit heuristic architectural debt; details and hashes are in `research/EXECUTOR_V07_FINAL.md`.
- The real BC-E checkpoint used for validation was externally supplied read-only from `C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt` (variant E, epoch 27, SHA-256 `F4B029D3E463ABA1DB0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`). Its verified 64-character digest is `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`; it is intentionally absent from this isolated worktree because local artifacts/checkpoints are ignored and remains uncommitted. Viewer #11 is closed and passive: schema/runner/snapshot/viewer/debug-trace CLI support is committed, four ignored real BC-E E_VS_E traces were generated and parsed, and ON/OFF action/result parity held for seeds 17 and 42; E_VS_E banks intentionally are not the PASS-panel result.
- Executor V0.5 overnight pass (issue #7, 2026-08-24, branch `executor-v05-overnight`, base `32fef4a` -> final `885adad`): mechanics-derived water urgency (weed-boundary vs yield-window vs defer), hub-anchored coordinated crop/animal layouts with weed reclamation, exact engine movement legality, sequential same-turn SELL->HIRE cash accounting, any-hour workload hiring without the 3-hire cap, exact-cost market gates, plan-implied CARE/FERTILIZE eligibility, accumulated diagnostics. New one-day replay-slice harness (`tools/day_slice.py`) validated by full-game fast-engine reconstruction of official replays; paired expanded set (38 slices): +5.4% mean one-day wealth, better on 24/38, weeds 53->22. Full evidence: `research/EXECUTOR_V05_OVERNIGHT.md`. BC-E checkpoint was absent during this pass; integration to main was left for review.

- Phase: **Stage-2b mechanic slices 1-4, the MAX_HANDS=240 exact layout, the full-episode legal-ish parity corpus, AND the secondary independent closed-loop agent A/B gate are all at zero first divergence vs the real official 1.32.7 engine**; same-action parity remains primary. The closed-loop gate covers three fixed-seed full episodes plus one repo-local checkpoint episode. Issue #2 throughput seam is implemented (decision D-024) and its A/B benchmarks are now MEASURED (`docs/benchmarks/ISSUE2_THROUGHPUT.md` + raw `docs/benchmarks/issue2_results.json`, decision D-025): scalar dict API 4.7x vs official 1.32.7 per 720-step episode (~2,580 vs ~553 turns/s), native batch floor ~188k turns/s, default-pool scaling ~2.9x at N>=512, observation writing = 84% of large-batch step cost (the one bounded optimization candidate, deferred to a distinct stage). Fused executor/day-step and distributed rollout remain explicitly deferred; next gates are independent final review/push, then broader evaluation.
- Throughput seam (issue #2, 2026-08-23): `FastKaggricultureEnv(configuration={"numThreads": N})` / `RustBatchEnv(..., num_threads=N)` builds an instance-local Rayon pool (`None`/default = global pool, `0` invalid); parallel fan-out only above 128 envs; `reset`/`step`/`step_transition`/`observe_into`/`action_masks_into`/`step_into` release the GIL after validation and exclusive buffer-slice extraction. Evidence: `tests/test_batch_throughput_seam.py` (5 tests: GIL spinner progress under a 512-env call, buffer integrity under concurrent Python-thread pressure, byte-identical trajectories for 130 envs x 30 steps across 1/2/4/default threads incl. day boundary). Measured scaling: serial below N=128 regardless of pool; at N=512/1024 the default pool reaches 2.87-2.89x vs 1 thread; best measured cell N=128/T=4 at ~204k transitions/s. API details: `fast_env/README.md`; benchmark reproduction: `scripts/benchmark_engine_throughput.py`.
- Engine/corpus: `kaggle-environments 1.32.7`, canonical replay schema **v3**.
- Training direction: **BC -> closed-loop executor validation -> PPO/RL refinement**.
- Primary goal: build a refinement/self-play pipeline that measurably improves a competent learned starting policy.
- BC V1 ablation (issue #6, 2026-08-23): the four-way variant matrix **V0/J/E/JE is fully implemented and locally validated** (commits `2f48564`..`fc95752`; 275-test combined sweep + independent audit 225 passed + one official opening-only plumbing smoke). Real corpus/checkpoints were absent locally at implementation time; the Kaggle panel subsequently ran and **promoted E** (see next bullet). The exact Kaggle train/panel runbook is `research/BC_V1_ABLATION_RUN.md`; the fixed paired panel (seeds 7/17/42/123/2026 × both seats, bank median-then-mean ranking) is the only promotion gate — teacher-forced/coherence metrics alone never promote.
- BC-E JAX port (issue #8, 2026-08-24): the real four-way closed-loop panel **promoted E** (closed-loop median 25,873.0 vs V0 9,251.5; JE 179.5; J 55.5). `bc_manager_jax` now supports exactly variants **V0+E** with the variant stored outside the frozen `ManagerConfig` (torch top-level `model_variant`; native NPZ metadata; old files load as V0), loud J/JE rejection, strict expected-variant checkpoint checks, and authoritative `bc_manager.economics` feature generation. Local tiny CPU evidence: PyTorch E → JAX E parity worst max 6.855e-07 / mean 1.101e-07 across all seven output groups; exact decode equality; loss groups within 9.5e-7; single-device JIT E step finite; N=4 logical-CPU NamedSharding subprocess (total diff 1.9e-6) plus one bounded N=8 logical smoke — both forced host-CPU validation ONLY, not throughput. The validation input was externally supplied read-only from the pinned source path above and remains uncommitted; the eventual Kaggle 8-device E benchmark command is documented in `research/JAX_TPU_V5_RUN.md` and explicitly UNMEASURED.
- RL self-play Stage A + Stage B1/B2 (issue #9, 2026-08-24): the `rl_manager` package now covers BOTH the rollout/self-play/trajectory harness AND the PPO V0 plumbing. Stage A: batched plan-policy protocol + own-only-E `JaxEPlanPolicy` wrapper (one batched JAX forward per policy identity/day, never per env), lockstep N-env runner over `oracle.backend` engines with committed `standard_mixed` d0-d3 opening and exact d4h0 handoff, runner-owned daily-start `(day, cash)` state feeding the stateless `economic_prev_start` E path, compact preallocated trajectory buffer (strict-schema NPZ + JSON sidecar), explicit seed streams, full provenance, executor factory seam, official-vs-fast parity seam. Stage B1 (`ppo_policy`/`gae`/`ppo`/`ppo_checkpoint`): mutable E trunk + small value head, immutable frozen-E snapshot supplying sell quantities, 17 categoricals + 54 Bernoullis with raw-summed logprob and six-group entropy, vmapped per-row-seed sampling, GAE with full-batch advantage normalization, masked-AdamW update, strict `rl_manager_ppo_checkpoint_v1`. Stage B2 integration: `PPOBatchedPolicy` adapter implementing the Stage-A protocol (stochastic/deterministic; deterministic init reproduces frozen-E decode exactly), checkpoint-resume reconstruction, deterministic post-GAE subset selection, compact `allow_nan=False` diagnostics artifact (null+reason for missing values), and guarded train/eval CLIs (`python -m rl_manager.cli train|eval`: real-checkpoint required + explicit executor factory; fixed smoke/dev/holdout seed sets, both seats, `--confirm-expensive` gate). Evidence: tiny live smoke = ONE complete fast game (stochastic candidate vs frozen E, numThreads=1) -> 52 transitions -> full-trajectory GAE -> exact stored-action logprob recompute -> ONE 4-row-minibatch update (finite metrics, trainable params changed, frozen snapshot bit-identical) -> checkpoint roundtrip with bit-identical resume and pre/post eval equality; combined rl_manager + issue-#8 JAX parity/train sweep = 130 passed + 4 skipped (~187 s; skips are gated official-engine/real-checkpoint paths). Plumbing only — NO policy-quality claim (tiny random-init E); the next gate is the RL executor-factory swap and serious training decision against the frozen V0.7 executor. Details: `research/RL_SELFPLAY_V0.md`.

## Opening Book (Issue #4)

`opening_book/` provides literal replay-derived elite openings plus the
runtime wrapper and official evaluator:

- Two committed 96-turn identities (days 0-3, d0h0..d3h23), extracted
  deterministically from verified raw replays with full provenance/digests:
  `standard_mixed` (episode 95515912 seat 0, dominant cluster) and
  `pasture_heavy` (episode 95055022 seat 0, ReCurSiON). Handoff at day 4
  hour 0 delegates unchanged to an injected downstream agent.
- Runtime wrapper (`opening_book/agent.py`, `make_opening_agent`) replays
  literal actions under minimal one-way guards (phase cursor, hand
  cardinality, action shape/market cap); any guard failure permanently
  delegates and records deterministic JSON diagnostics.
- Official evaluator (`python -m opening_book.eval`) runs opening-only and
  paired BC-handoff games behind the pinned provenance guard; module
  docstring carries the exact Kaggle command for the real-checkpoint paired
  comparison.
- Validation: 53 focused tests; official 1.32.7 matrix (2 seeds x 2 seats x
  PASS/mirror x both identities) = standard_mixed 8/8, pasture_heavy 7/8
  strict envelope passes. The one failure is environment variance, not code:
  seed 1146601720 seat1 vs PASS spawned WEEDs on the d3h11 strawberry target;
  all 96 turns replayed, zero divergence/anomalies, clean handoff. Strict
  envelope kept by decision; no heuristic weed repair.
- Limitation: real `/kaggle/working/bc-v0-score2950/best.pt` absent locally,
  so paired BC evaluation has not run; no end-to-end BC gain is claimed.

## Differential Oracle (Stage 2a + 2b slice 1)

`oracle/` provides a same-action replay harness: the exact same action pair is
submitted to the pinned official 1.32.7 engine and the fast Rust engine each
turn BEFORE an immediate canonical full-state compare; the run stops at the
first divergent field with seed/step/day/hour/path/values/actions context
(usage + temp-official setup: `oracle/README.md`, decision D-020).

Validated so far: initial-state parity, short legal traces (BUY_SEED/PLANT/
WATER), both-seat privacy comparison, deliberate-corruption first-divergence,
terminal rewards/statuses at `episodeSteps=3`, provenance tamper rejection,
a 28-turn pass-only day-boundary smoke, and — Stage-2b slice 1 — the worker-
inventory / same-turn-ordering / hiring / market cluster at zero divergence
(`tests/test_oracle_mechanics.py`, 27 scenarios; three exact fast-engine
divergences found and fixed: money-decode f32 noise, MAX_QUANTITY=100 order
clamps, ValueError-on-silent-noop wire translation; see `MECHANICS.md`),
and — Stage-2b slice 2 — the crop/seed/tile lifecycle cluster at zero
divergence (`tests/test_oracle_crops.py`, 16 tests / 15 scenarios, 2,136 turn
pairs, 74 day boundaries; one exact fast-engine divergence found and fixed:
`_decay_plants` gated the yield decrement on `> 0` and converted at `== 0`
where the official engine decrements unconditionally and converts at `<= 0`,
letting a zero-yield ongoing crop survive forever; see `MECHANICS.md`),
and — Stage-2b slice 4 — the town/world/day-RNG/reset/terminal cluster at
zero divergence (`tests/test_oracle_town_world.py`, 10 tests / 10 scenarios,
~1,100 turn pairs including one 648-turn PASS-only season segment; shop
unlock timing/duplicate multiplicity/8-instance cap, town + town-center
consumption incl. step-0 fire and negative stock, shared per-day RNG stream
with weed/shop draw ordering, day-boundary reset ordering, terminal
rewards/statuses and no-post-terminal; no engine changes required; see
`MECHANICS.md`).
Former deferral CLOSED (2026-08-23): the fast engine now uses the exact
default-contract hand capacity `MAX_HANDS = maxMarketOrdersPerTurn(10) *
turnsPerDay(24) = 240` — one hand per atomic HIRE order, market queue
truncated to 10 orders/turn, hands cleared at every day reset. Breaking wire
layout: `OBS_SIZE` 5630→8766, `ACTION_SLOTS` 27→251 (market rows moved from
slot 17 to slot 241), `MASK_SIZE` 3562→34026; derivation, offsets, buffer
deltas, and the locked HIRE-mask gate are recorded in `MECHANICS.md`
(MAX_HANDS=240 section) and decision D-021. Evidence:
`tests/test_fast_env.py` (15 tests incl. 23-hand scalar API, hand actions to
all hands, day-reset/rehire Fibonacci restart, mask formula both sides) and
`tests/test_oracle_hands.py` (5 real-official same-action replays: exactly-16
boundary, 17th–23rd crossing, 23-hand hires + subsequent hand actions,
day-end reset from 23 hands + rehire parity, per-turn mask ==
official-reachable gate) — all green.

Full-episode legal-ish corpus DONE (2026-08-23, decision D-022): the
deterministic state-aware generator `oracle/action_generator.py` drove
complete default episodes for seeds 0, 1, 2, 7, 17, 42, 123, 999 through
`run_same_action_replay` — **zero first divergence**; every episode ran the
reset observation + exactly 719 accepted primitive steps with terminal
DONE/DONE at canonical step 719 (day 29 hour 23), equal official/fast
rewards for both seats, and 29 day transitions; coverage union 33 action
families / 28,508 attempted instances incl. malformed/no-op/truncation
surfaces; repeatability (identical trace per seed; identical fast
reset+replay canonical states/rewards/statuses) locked in
`tests/test_action_generator.py`; official-gated episode tests in
`tests/test_oracle_corpus.py`. Report: `research/parity_corpus_report.json`;
full-corpus command: `python scripts/run_parity_corpus.py` (oracle venv).
Bounded claim: parity proven for the states these episodes reach — not a
universal mathematical proof; closed-loop A/B is the next gate.

## Secondary Closed-Loop Agent A/B Gate

`oracle/closed_loop.py::run_closed_loop` is the secondary policy-interface
check. It constructs four fresh stateful agents (official/fast × seat 0/1),
compares the reset and every next presented observation, computes each backend's
actions independently, compares actions before stepping, and then compares the
canonical next state/rewards/statuses immediately. The official full status
history remains validated for anomalies.

- Fixed-plan `executor_v0.ExecutorAgent` factory: seeds `0`, `7`, and `42`,
  each with one reset plus 719 accepted steps, terminal `DONE/DONE` at step
  719, equal rewards, and zero closed-loop divergence.
- Action-family union: 30 submitted farmer/hand/market families; report:
  `research/closed_loop_ab_report.json`.
- Real repo-local checkpoint: `data/temp/bc-train-smoke/ckpt/best.pt`, one
  seed-0 full episode, also `DONE/DONE` at step 719 with equal `[0.0, 0.0]`
  rewards. This is a plumbing A/B result, not a competitive score claim.
- Deliberate observation and action drift tests stop before stepping and report
  seed/step/day/hour/seat/path/official/fast/actions. The primary same-action
  corpus remains the engine-correctness gate; this stage does not replace it.
- Wall time for the three fixed-plan episodes plus checkpoint episode was
  52.65 seconds in the authorized official temp venv on this host.

## Learned-Control Contract

The policy is a **once-per-day farm manager**. V0 manager owns:

- crop composition;
- animal target counts;
- land expansion;
- fertilizer allocation by crop type;
- CARE allocation by animal type;
- six-bin daily selling intent.

The deterministic executor owns mechanics:

- exact tile placement and minimum-change crop reconciliation;
- animal structures/placement;
- worker assignment/routing/loading;
- hiring needed for the requested workload;
- watering, feed, harvest, collection, and other routine maintenance;
- seed/feed/fertilizer/animal purchases mechanically implied by the plan;
- exact fertilizer/CARE targets;
- primitive sell execution.

The executor is a strategy compiler under D-011, not a hidden economic policy.

## Canonical Corpus

Five elite post-patch 1.32.7 partitions, 2026-08-17 through 2026-08-21:

- 3,486 episodes;
- 6,972 seat trajectories;
- 209,160 `(episode, seat, day)` rows;
- schema versions exactly `{3}`.

Important coordinate contract: simulator worker positions are `[x,y]`; canonical board/event coordinates are `[y,x]`; tile lookup is `tiles[y][x]` after unpacking `x,y`.

Private Kaggle dataset mount used for BC:

`/kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327`

See `research/FIVE_DAY_V3_CORPUS.md`.

## First BC V0 Result

Reference model: D-019 default tile Transformer, 1,071,040 parameters.

Reference run:

- code at run start: `692bca50e8ba0b687e48fd970e67bbe17014f03f`;
- train Aug17-20, validation Aug21;
- `min_score >= 2950`;
- 25,500 train rows, 5,700 validation rows;
- CUDA + AMP, batch 256, AdamW 3e-4;
- 30 epochs, ~237 s;
- best epoch 29;
- best validation total **2.8889**.

Held-out Aug21 model vs train-only day baseline:

| metric | model | day baseline |
| --- | ---: | ---: |
| crop exact accuracy | **0.7128** | 0.4752 |
| crop MAE | **1.2731** | 3.6217 |
| animal exact accuracy | **0.8267** | 0.4540 |
| animal MAE | **0.2681** | 1.6936 |
| fertilizer nonzero recall | **0.7522** | 0.4557 |
| CARE whole-vector exact | **0.5998** | 0.1754 |
| land accuracy | **0.9912** | 0.9089 |
| sell presence accuracy | **0.9394** | 0.8923 |

Rare state-conditioned branches also work materially better than the calendar baseline: tomato nonzero recall 83.8%, goose 96.8%, goose CARE 95.5%, wheat fertilizer 60.4%.

Selling remains the clearest teacher-forced weakness: true positive rate 11.21%, predicted 9.38%, positive recall 64.84%.

Conclusion: **D-019 passes its intended representation diagnostic.** Do not spend the next cycle on model scaling/tuning before closed-loop evidence.

Detailed run/eval: `research/FIRST_BC_V0_EVAL.md`.

## BC V1 Ablation (Issue #6)

Implemented, locally validated, **not yet run on real data**. Four manager
variants over the unchanged D-019 trunk and data layer:

| variant | change | parameters |
| --- | --- | ---: |
| V0 | baseline; exact pre-V1 behavior preserved | 1,071,040 |
| J | joint plan decoder (one coherent plan object instead of independent heads) | 1,204,288 |
| E | 14-channel realized-economic context via a live `EconomicHistory` tracker exactly mirroring batch derivation (previous-day net-cash delta, hire affordability from the observed `hires_today` counter) | 1,072,832 |
| JE | joint decoder + economic context | 1,206,080 |

Hard feature rule (Stage 0 audit,
`research/BC_V1_ECONOMIC_CONTEXT.md`): submitted market intents are not
realized fills — gross revenue/spend/fill quantities are never inferred;
only observed money snapshots and the `hires_today` counter feed features.
Coherence diagnostics are recorded per live plan but never clipped into or
fed back to any decision.

Local implementation evidence: stage sweeps up to 275 passed (BC V1 +
bc_manager + executor_v0 + opening_book); independent audit 62 new + 163
compat = 225 passed; one official opening-only seed-7/seat-0 smoke under
pinned 1.32.7 (96 turns, clean handoff, zero divergence/fallback — plumbing
only, no BC weights attached). Those results predate the externally supplied
real corpus/checkpoint and panel; the later E promotion is recorded above.

Promotion gate: the fixed paired closed-loop panel — `standard_mixed`
opening days 0–3 -> tested BC -> unchanged executor, seeds
7/17/42/123/2026 × both seats = 40 games, ranked by final-bank median then
mean; teacher-forced totals and coherence are prerequisites/diagnostics
only. Exact copy-paste Kaggle train/preflight/panel commands:
`research/BC_V1_ABLATION_RUN.md`.

## BC V0 Simplification Backlog

The BC design did already record many deferrals in D-019 and its implementation note, but they were spread across several files. `research/FIRST_BC_V0_EVAL.md` now contains one explicit backlog of V0 shortcuts and revisit triggers, including:

- stateless once-per-day policy;
- opponent-public board disabled;
- no opponent-private inference;
- absolute counts rather than deltas/per-tile targets;
- type-level fertilizer/CARE rather than tile-specific control;
- factorized heads with no joint feasibility model;
- six-bin selling and fixed 0.5 presence threshold;
- equal loss weights and no sparse reweighting;
- one ~1.07M architecture with no sweep/scheduler;
- five-day `>=2950` elite corpus with no dedup/family weighting;
- one held-out date rather than a broad rolling/generalization suite;
- no DAgger/on-policy correction, value head, PPO, or uncertainty-aware execution.

Known weak points such as rare fertilizer recall and conservative selling remain explicitly listed so they are not forgotten after the first closed-loop success.

## Deterministic Executor V0

**Implemented (issue #1, commits `11e85fa`..`ed1685a`).** The complete
closed loop lives in `executor_v0/` (usage: `executor_v0/README.md`):

live schema-v3 observation encoding (exact BC adapter parity) ->
once-per-day manager (`CheckpointPlanProvider` or injected fake) ->
immutable `DailyPlan` -> mechanical feasibility projection ->
deterministic animal layout / minimum-change crop reconciliation ->
per-turn explicit task generation -> greedy foreman dispatch ->
hour-0 crude hiring, exact-shortage purchases, BUY_LAND, six-bin sells ->
legal-shaped `{"farmer", "hands", "market"}` action dict plus JSON
diagnostics (requested/feasible/achieved/submitted/observed) and a
deterministic all-PASS safe-mode fallback.

Key mechanic: `PLANT <crop>` consumes the global own `private.seeds[crop]`
pool atomically at the engine; seeds are never picked up or carried, and the
foreman reserves global seeds per crop within each turn.

Validation so far: 249 tests pass; live encoder parity (synthetic + real);
determinism coverage; a 719/720-turn replay-observation plumbing smoke
(shape/state robustness only — counterfactual actions were never executed by
the engine). A real 1.32.7 game has NOT been run because
`kaggle_environments` is not installed here; the smoke harness skip path
(exit 3) is verified.

The current first-draft algorithm is recorded in `research/EXECUTOR_V0_PLAN.md`.

Core design:

1. Run the BC manager at hour 0 and preserve `requested -> feasible -> achieved` separately.
2. Do **not** reserve near-shed tiles for future animals. Use productive crop space until animal targets actually require conversion.
3. When making room for livestock, prefer legal empty tiles or the **least-invested nearby crop**: young/unfertilized plants are cheap to sacrifice; mature/fertilized/near-harvest crops are expensive.
4. Keep established livestock infrastructure sticky; otherwise reconcile crop targets with minimum destruction/relayout.
5. Generate explicit maintenance/transition/logistics tasks and revalidate them from the live observation every primitive turn.
6. Use broad urgency/deadline tiers rather than a large hand-tuned strategic priority table.
7. Use soft worker specialization from carried inventory. Prefer workers needing few item types; planned bundles should normally stay around <=2 types.
8. Greedily load and dispatch workers. Never hold a ready worker at the shed merely to synchronize loading.
9. Hour 0: farmer may take one useful bulk pickup; market stages supplies and hires hands. New hands act from hour 1 and begin greedy loading/dispatch.
10. Recompute a simple deterministic greedy worker-task assignment every turn using Manhattan distance, shed detours, and inventory affinity.
11. Purchase only shortages mechanically implied by the manager plan/maintenance; do not add a separate economic strategy.
12. Execute six-bin sells literally as inventory becomes available within the requested bin.

## Executor V0 Simplification Backlog

`research/EXECUTOR_V0_PLAN.md` explicitly marks these as temporary shortcuts rather than settled optimal choices:

- no prediction/reservation for future animal onset;
- crude crop-sacrifice value rather than exact marginal farm value;
- no global layout/facility-location optimization;
- greedy worker matching rather than Hungarian/min-cost-flow/search/VRP;
- one-step Manhattan routing and no multi-turn route search;
- soft inventory specialization rather than explicit optimized worker roles;
- greedy shed staging rather than inventory-buffer optimization;
- crude workload-based hiring;
- basic wheat/feed procurement;
- literal six-bin selling with no within-bin price reaction;
- no opponent-aware executor economics;
- no confidence-aware plan projection;
- no search-based foreman yet.

Search/assignment/layout sophistication is a **later performance option**, not part of the first closed-loop build.

Revisit these when traces show low intent compliance, missed maintenance despite sufficient labor, excessive movement/wait/reloading, shed stalls, destructive crop conversions, worker assignment thrashing, or high compliance but losses traceable to executor-owned simplifications.

## Closed-Loop Success Criteria

The first executor is successful when it can:

- encode live observations with the same semantics as BC training;
- make all 30 manager calls;
- finish real 1.32.7 games without illegal-action cascades/deadlock;
- track requested/feasible/achieved crop, animal, land, fertilizer, CARE, and selling;
- report missed maintenance, pending tasks, movement/pickup/idle actions, and emergency purchases;
- run a small fixed-seed, seat-swapped panel against one frozen competent opponent.

If compliance is poor, improve execution before blaming BC. If compliance is high and economic trajectories are poor, revisit the manager/action abstraction before adding PPO complexity.

## Near-Term Sequence

1. ~~Turn `research/EXECUTOR_V0_PLAN.md` into a bounded implementation contract and build the smallest complete executor.~~ Done (issue #1).
2. Get the current `best.pt` through full local 1.32.7 games (`python -m executor_v0.smoke --manager checkpoint --checkpoint best.pt`).
3. Inspect compliance before optimizing score.
4. Run a small paired fixed-seed/seat-swapped frozen-opponent panel.
5. Use evidence to choose the next bottleneck: executor search/layout, BC refinement/data, opponent inputs, or PPO.
6. Only after the stationary closed-loop problem is reliable, expand to broader opponent panels and changing-population/self-play.

## Do Not Forget

Before substantial work read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, relevant implemented notes, `research/FIRST_BC_V0_EVAL.md`, and `research/EXECUTOR_V0_PLAN.md`.

Before expensive runs record exact code/configuration, engine identity, data/version/filter, seeds/opponents, outputs, stop conditions, and recovery plan.
