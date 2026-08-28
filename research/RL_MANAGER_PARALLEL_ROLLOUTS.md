# Issue #17 Parallel Rollout Runbook

Issue #17 uses Python `multiprocessing` with the Windows/Kaggle-safe
`spawn` method. The parent process retains all policy objects and is the sole
JAX/libtpu owner. Workers import only the framework-neutral protocol, runner,
engine, opening, and executor paths.

## Local CPU topology smoke

Build the native extension once, then run a deterministic mock-policy smoke:

```bash
python -m maturin build --release
python tools/benchmark_parallel_rollouts.py --episodes 8 --num-workers 1 --num-envs 1 --num-threads 1 --output-json artifacts/local/issue17-workers1.json
python tools/benchmark_parallel_rollouts.py --episodes 8 --num-workers 2 --num-envs 1 --num-threads 1 --output-json artifacts/local/issue17-workers2.json
python tools/benchmark_parallel_rollouts.py --episodes 8 --num-workers 4 --num-envs 2 --num-threads 1 --output-json artifacts/local/issue17-workers4-envs2.json
```

Compare the fixed seed list and result digests before comparing throughput.
`numThreads=1` is intentional for scalar `RustBatchEnv(1)` workers; increasing
it creates oversubscribed private pools rather than parallelizing one scalar
episode.

## Kaggle TPU validation

Run one foreground command per cell/session after placing the real BC-E and
PPO checkpoints at the explicit paths. The policy/JAX setup must happen in the
main process before the rollout coordinator starts children:

```bash
python -m rl_manager.cli eval --checkpoint /kaggle/input/ppo/ppo.npz --e-checkpoint /kaggle/input/bc-e/best.pt --executor-factory executor_v0@stage-a-v1 --backend fast --num-workers 4 --num-envs 1 --num-threads 1 --low-telemetry --read-only-agent-observations --batch-backend --seed-set smoke --output-json /kaggle/working/issue17-smoke.json
```

For scaling, repeat the same fixed seed set and immutable checkpoints with
`--num-workers 1,2,4,8`; keep `--num-envs 1` first, then test `2` and `4`.
Record startup/compile time separately from the steady-state wall interval.
The JSON output records games/sec, manager request count, owner inference
batch sizes, inference time, and queue wait time. No TPU performance result is
claimed by the local benchmark.

## Ownership checks

The worker entrypoint fails before constructing CPU state if `jax`, `jaxlib`,
`torch_xla`, `bc_manager_jax`, or `optax` is already loaded. The package
initializer is lazy so importing `rl_manager.parallel_worker` does not load
those modules. A worker failure is sent to the owner with traceback and causes
all children to be terminated and joined; missing or duplicate episode
results are also fatal.

## Integration points

- Issue #15 can replace the registered executor factory or its `create()`
  implementation; no worker protocol change is needed.
- Issue #16's native batched backend composes through `RunnerConfig` in each
  CPU worker; the parent remains the only JAX/libtpu owner.
- A future learner remains in the parent owner process, reusing the same
  policy identity and inference dispatch boundary.
