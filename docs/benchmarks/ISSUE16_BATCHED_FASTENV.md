# Issue #16: Batched Fast-Environment Benchmark

Local Windows host measurements on branch `throughput/16-batched-fastenv`.
These are plumbing measurements, not TPU performance claims.

## Profile Before Integration

The scalar wrapper was measured for 200 PASS/mixed turns after warmup. Costs
below are microseconds per environment-turn:

| component | N=1 | N=2 |
|---|---:|---:|
| Python action-dict encoding | 8.19 | 8.81 |
| native transition only | 0.33 | 0.14 |
| native observation writing | 2.88 | 2.79 |
| fused native step | 3.22 | 2.88 |
| Python observation decode | 220.20 | 223.49 |
| `canonical_state_fast` | 31.66 | 33.07 |
| runner canonical farm deep copies | 114.66 | 116.26 |
| scalar wrapper step total | 225.58 | 236.65 |

Native transition mechanics are below 1% of the scalar wrapper total.
Observation decoding and redundant runner state copying dominate environment
cost.

## Batched Native Ceiling

Preallocated raw `RustBatchEnv.step_into`, 200 steps, one Rayon thread:

| N | transitions/sec |
|---:|---:|
| 1 | 232,342 |
| 2 | 257,003 |
| 8 | 266,516 |
| 16 | 87,341 |
| 32 | 87,930 |

The N=16/32 cells were noisy on this laptop and are retained rather than
silently replaced. Existing issue #2 measurements remain the broader raw
engine reference, including roughly 555k transitions/sec at N=512 with the
default pool.

## Full Python Batch Wrapper

The same PASS workload through `BatchedFastEnv.step`, including action encoding
and nested observation decode:

| N | batched transitions/sec | scalar-wrapper transitions/sec | speedup |
|---:|---:|---:|---:|
| 1 | 7,570 | 4,210 | 1.80x |
| 2 | 7,752 | 4,193 | 1.85x |
| 8 | 7,636 | 2,901 | 2.63x |
| 16 | 4,559 | 2,486 | 1.83x |
| 32 | 7,725 | 4,117 | 1.88x |

## Self-Play Reference Runner

Two E-vs-E constant-plan games, 130 primitive turns per game, `numThreads=1`:

| path | wall seconds | primitive turns/sec | measured runner env-step |
|---|---:|---:|---:|
| existing scalar runner | 0.647 | 402 | 0.303 s |
| `batch_backend=True` | 0.415 | 627 | 0.089 s |

Both paths ended with `[9.0, 9.0]` banks and the same per-game trace digest.
This is a plumbing fixture, not a policy-quality result.

## Design Boundary

`BatchedFastEnv` owns the native states and reusable buffers. The runner still
owns one executor/opening/provider per seat and sends explicit action dicts into
the batch encoder. No executor strategy, PPO semantics, multiprocessing, IPC,
or worker scheduling is included. Issue #15 can optimize executor action
production independently; issue #17 can consume the adapter from a future
worker without changing its API.
