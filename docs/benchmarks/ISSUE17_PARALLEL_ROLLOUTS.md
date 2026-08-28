# Issue #17 Parallel Rollout Measurements

These are local Windows CPU measurements, not TPU results. They use the
freshly built scalar fast engine, deterministic mock manager policy, master
seed `17`, `num_envs=1`, `num_threads=1`, and a 130-primitive-turn truncated
workload. The native extension was built with `python -m maturin build
--release`; the generated extension is ignored and is not part of this branch.

| workers | episodes | games/s | primitive turns/s | manager requests/s | owner batches | batch sizes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4 | 3.8553 | 501.1886 | 15.4212 | 0 | - |
| 2 | 4 | 1.5711 | 204.2411 | 6.2843 | 4 | 4,4,4,4 |
| 4 | 4 | 1.2629 | 164.1721 | 5.0514 | 6 | 2,5,1,2,5,1 |

The short workload is IPC/process-startup bound on this host, so it does not
show useful scaling. The 4-worker 8-episode run reached 3.0271 games/s with
batch sizes `6,2,6,2,6,2,6,2`, but it is not an apples-to-apples timing cell.
These results validate instrumentation and expose the need to measure a long
steady-state Kaggle workload; they are not evidence of TPU throughput.

All cells returned the same four-episode seed sequence:

```text
1251310250, 2930343534, 2781484340, 58495594
```

Reproduce with `tools/benchmark_parallel_rollouts.py`. The fixed real BC-E
TPU command and the required workers/envs/thread sweep are in
`research/RL_MANAGER_PARALLEL_ROLLOUTS.md`.
