# Stage 1 handoff — 2026-08-29

Stage 1 is complete. The project now has a working deterministic opening -> learned daily manager -> deterministic executor stack, a BC-E baseline that can be deployed competitively, and a PPO lifecycle that has demonstrated meaningful offline improvement from that baseline.

## What Stage 1 established

### Throughput and infrastructure

- Mixed-day inference batching and host-side rollout improvements made the integrated runner practical for repeated PPO experiments.
- A stable production configuration reached roughly 14 games/s on Kaggle TPU-hosted runs with W96, N1, fixed batch 24, 30 ms inference wait, trajectory collection enabled, and deterministic batching.
- Multiple TPU trainers can use separate TPU devices concurrently; measured scaling was useful through N=4. N=8 was not measured and should not be inferred from device visibility.
- Rollout workers remain CPU-only and must not import JAX/libtpu.

### PPO can improve the BC baseline

A fresh three-update PPO smoke with `lr=1e-4`, 2 max epochs, `target_kl=0.08`, KL-to-frozen 0.001, and no hard KL rejection produced a 64-game holdout result of 61-3 against frozen BC-E with candidate mean bank 43,467.8.

A longer stationary-BC run showed that stronger policies can appear and later collapse. The best measured region of an earlier long run was near update 95 at roughly 53.9k mean / 56.2k median, but later updates degraded badly. Checkpoint retention and promotion discipline are therefore mandatory.

Hard KL rollback is not the preferred policy for this project. Use target-KL as an early-stop signal, keep the completed epoch, and lower LR when updates are too aggressive.

### Submission/runtime parity matters as much as model quality

The first strong PPO submission collapsed to a live rating near 66 despite excellent runner results. The root cause was a train/deploy feature-distribution mismatch in the E economic-context path: the legacy runner passed the current `(day, money)` as the supposed previous-day start, producing zero/invalid previous-day cash delta semantics, while normal Torch deployment used true EconomicHistory.

The same u50 checkpoint recovered to 43,860 mean bank over a 16-game official panel after reproducing the legacy runner semantics in deployment.

A second live failure exposed a lazy dependency on `fast_env.market`. The compatibility builder now vendors the pure-Python market helper inside `executor_v0` and rewrites the lazy import so the Kaggle sandbox does not depend on a top-level `fast_env` package.

Existing legacy-trained checkpoints must continue to use `tools/build_runner_compatible_submission.py`. New checkpoints should be retrained only after the runner economic-history semantics are explicitly versioned and corrected.

### Executor fixes

Issue #25 fixed a real mechanical boundary failure: ready `PLACE` work into an existing compatible structure had been suppressed as expansion. The executor now preserves dependency-free ready PLACE work while still suppressing PLACE whose BUILD dependency was removed.

The existing PASS-only preventive watering layer was mechanically sound but disabled in production. It is now enabled in production/RL defaults with a same-day reachability bound. Hard WATER/FEED and other required work retain priority.

## Final Stage 1 official-engine A/B

All comparisons below used `kaggle_environments==1.32.7`. Version 1.29.3 produced misleading low-bank behavior and must not be used for submission gating.

Eight seeds x both seats were compared for each archive pair.

### PPO u50: old executor -> repaired executor

- mean paired bank delta: +867.8
- median paired bank delta: -648.5
- improved: 7/16
- worse: 9/16
- worst delta: -12,579
- best delta: +19,391
- final weed delta: -57
- animal inventory residue delta: -32
- final on-board animal delta: -4

Interpretation: mechanics improved substantially, but the old u50 policy does not cleanly transfer to the repaired executor. Do not treat this archive as an automatic promotion; the changed closed loop needs retraining or attribution tests.

### BC-E instant-sell: old executor -> repaired executor

- mean paired bank delta: +7,116.4
- median paired bank delta: +6,635
- improved: 11/16
- worse: 5/16
- worst delta: -32,278
- best delta: +31,067
- final weed delta: -34
- animal inventory residue delta: -47
- final on-board animal delta: +38

Interpretation: this is a meaningful executor-level improvement and is the preferred Stage 1 submission/control baseline.

## Stage 1 deployment rules

1. Validate the exact archive bytes, not repository code.
2. Use official `kaggle_environments==1.32.7` for submission gates.
3. Force the packaged `ExecutorAgent._buy_order_cost` BUY_PRODUCT branch so the vendored market helper is exercised before upload.
4. Run both seat orientations on a small official seed panel.
5. Require zero invalid/error/timeout statuses.
6. Legacy PPO/BC-E checkpoints trained under the old E runner semantics must use the runner-compatible provider.
7. Preserve all useful checkpoints; do not trust the terminal PPO update.

## Stage 2 starting point

The next phase should focus on learning with the repaired executor rather than adding more ad-hoc strategy to it.

Priority order:

1. Version and fix the E economic-history runner semantics, add rollover parity tests against EconomicHistory, then retrain new checkpoints under the corrected feature distribution.
2. Fix promotion gating so benign opening diagnostics do not veto promotions.
3. Add terminal economic rewards / bank and margin signals (issue #19 direction) instead of relying only on sparse W/L/T.
4. Move from stationary BC opposition toward current/current economic self-play plus current/snapshot competitive games (issue #20 direction).
5. Keep a stable BC-E instant-sell + repaired-executor control for regression testing.
6. Evaluate retained checkpoints frequently and separately track economic-best, competitive-best, latest, and promoted snapshots.

The Stage 2 objective remains the original competition goal: demonstrate a self-play pipeline that repeatedly produces and promotes policies that materially improve over the starting BC baseline, rather than optimizing directly for a medal.
