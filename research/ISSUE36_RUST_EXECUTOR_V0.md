# Issue #36 Rust Executor V0 Experiment

Date: 2026-08-31
Base: `f91ea26a782bc39bfe7ff8ea88c395d10e438afd`
Branch: `codex/issue-36-rust-executor-v0`

## Question

Can reproducing the current deterministic Python executor in Rust materially
improve end-to-end Stage-2 self-play rollout throughput?

## Profiling Baseline

The existing profiler was extended to emit all runner timing buckets. The
measurement used the local externally supplied BC-E checkpoint
`artifacts/local/bc-v1-E/best.pt`, fast engine, scalar `numThreads=1`, fixed
seeds `(17, 42, 2026, 7, 123, 1013, 1022, 1003)`, one warmup, two measured
repeats, and explicit low-telemetry/read-only observations.

| envs | total s | games/s | turns/s | executor/agent s | agent share | inference s | env-step s | orchestration s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.674 | 0.2140 | 153.8 | 1.896 | 40.6% | 0.488 | 2.269 | 0.0066 |
| 2 | 8.220 | 0.2433 | 174.9 | 3.402 | 41.4% | 0.596 | 4.191 | 0.0074 |
| 8 | 34.357 | 0.2329 | 167.4 | 14.327 | 41.7% | 1.901 | 17.884 | 0.0247 |

The agent bucket includes opening/executor calls and observation preparation;
it is not all pure executor code. A perfect zero-cost agent would have a
theoretical total-throughput ceiling of approximately 1.68x, 1.71x, and 1.72x
for N=1, 2, and 8 respectively. This is not a claim that a port can reach
that ceiling.

## Prototype

`rust/kaggriculture_env/src/lib.rs` now exports `RustExecutorV0`, a PyO3
callable that owns the Python semantic oracle and forwards observations to it.
`executor_v0/rust_backend.py` provides the explicit opt-in factory
`executor_v0@rust-v0`; the existing `executor_v0@stage-a-v1` factory remains
the default. The wrapper is deliberately not wired to manager behavior,
checkpoint identity, PPO, reward, or submission code.

This is a parity-safe native-boundary prototype, not a native semantic port.
The Python oracle remains responsible for every current production behavior:

- plan reconciliation/projection and task generation;
- dependencies, layout, worker assignment, routing, and movement;
- feed/starvation safety and required watering;
- same-worker PLANT -> WATER continuation;
- crop-sacrifice DIG -> BUILD -> PLACE and compatible placement;
- prior-debt suppression;
- market ordering/accounting, hiring, and sell reserves;
- optional spare watering and deterministic per-game reset state;
- diagnostics and safe/strict failure behavior.

Intentionally unsupported in native Rust: all of the above semantic logic.
The Rust object owns only the callable boundary and a call counter. This was
the smallest way to answer the throughput question without creating a second
heuristic that could silently diverge.

## Differential Parity

The direct comparator in `executor_v0/differential.py` stops at the first
complete primitive-action difference and records hashes of optional
observation/plan/history context. It never accepts score similarity as parity.

Results:

- isolated comparator fixtures: exact equality and first-difference reporting
  tests pass;
- current executor task/foreman/layout/safety/placement/optional-work focused
  suite: existing Python oracle tests remain the semantic coverage panel;
- BC-E closed-loop fast panel: seeds `17, 42, 2026`, both seats, 719 primitive
  turns per game, 2,157 joint primitive action pairs, zero first divergence;
- final banks were identical for every case: seed 17 `[27153.0, 46923.0]`,
  seed 42 `[29108.0, 22842.0]`, seed 2026 `[30777.0, 24514.0]`;
- terminal statuses were `DONE/DONE` for every case;
- no PPO snapshot was locally available;
- official-backend parity was not run because the pinned
  `kaggle_environments==1.32.7` package is absent on this host.

Because Rust calls the same Python oracle, the parity result establishes the
adapter seam and reset ownership, not independent Rust semantic coverage.

## End-To-End Benchmark

Same checkpoint, engine, seeds, low-telemetry settings, and scalar native
engine were used for both backends. One warmup and two measured repeats were
used; table values are the selected median repeat.

| envs | Python total s | Rust total s | Python games/s | Rust games/s | Rust change |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.674 | 4.755 | 0.2140 | 0.2103 | -1.7% |
| 2 | 8.220 | 8.382 | 0.2433 | 0.2386 | -1.9% |
| 8 | 34.357 | 34.916 | 0.2329 | 0.2291 | -1.6% |

Rust executor/agent time was `1.927/3.432/14.368` seconds at N=1/2/8,
respectively, so the extra native callback boundary did not reduce the
dominant work. Environment-step and inference variance was larger than the
observed difference. These are local Windows/Python 3.13 measurements, not a
TPU result.

Raw machine-readable outputs:

- `artifacts/local/issue36-python-benchmark.json`
- `artifacts/local/issue36-rust-benchmark.json`
- `artifacts/local/issue36-parity.json`

## Kaggle TPU-Host Command

Run each command as a separate foreground measurement with identical real
checkpoints, fixed `smoke` seeds, and output paths. The 96-worker topology
matches the existing approximately 96-CPU rollout plan; Rust differs only in
the explicit factory identifier.

```bash
python -m rl_manager.cli eval --checkpoint /kaggle/input/ppo/ppo.npz --e-checkpoint /kaggle/input/bc-e/best.pt --executor-factory executor_v0@stage-a-v1 --backend fast --num-workers 96 --num-envs 1 --num-threads 1 --low-telemetry --read-only-agent-observations --batch-backend --seed-set smoke --output-json /kaggle/working/issue36-python-96workers.json
python -m rl_manager.cli eval --checkpoint /kaggle/input/ppo/ppo.npz --e-checkpoint /kaggle/input/bc-e/best.pt --executor-factory executor_v0@rust-v0 --backend fast --num-workers 96 --num-envs 1 --num-threads 1 --low-telemetry --read-only-agent-observations --batch-backend --seed-set smoke --output-json /kaggle/working/issue36-rust-96workers.json
```

Record startup/compile time separately from steady-state rollout time and
compare games/s, primitive turns/s, total wall time, `agent_actions`,
`env_step`, and owner inference seconds. No TPU run was made for this issue.

## Recommendation

**Abandon this adapter as a throughput optimization and keep Python default.**
It passes parity because it delegates to the oracle, but it is slightly slower
end-to-end and provides no evidence that an independent Rust semantic port
would improve the target workload. A future native semantic port should only
start with a new profile showing a materially larger executor share or a
compact array boundary that removes Python mapping/conversion overhead.
