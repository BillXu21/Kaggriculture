# Fast Kaggriculture engine

`FastKaggricultureEnv` is the Stage 1 scalar engine seam:

```python
from fast_env import FastKaggricultureEnv

env = FastKaggricultureEnv(configuration={"seed": 7})
observations = env.reset()  # two dict observations, one private view per seat
observations, rewards, statuses = env.step([
    {"farmer": ["PASS"], "hands": [], "market": []},
    {"farmer": ["PASS"], "hands": [], "market": []},
])
```

The action grammar follows the official JSON shape. Observations preserve the
existing `farms`/`market`/`town`/time fields and each seat receives only its
own `private` shed, seeds, and carried inventories. `state_snapshot()` returns
the latest decoded pair for later differential tooling.

Build/install from the repository root:

```text
python -m pip install maturin numpy
python -m maturin develop --release
```

On Windows machines without Visual Studio MSVC `link.exe`, use the
self-contained GNU toolchain instead (verified route; keep the toolchain in a
local directory via `RUSTUP_HOME`/`CARGO_HOME` if a system install is not
wanted):

```text
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
$env:CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu"
python -m maturin develop --release
```

The normal import path loads NumPy and the local PyO3 extension only; it does
not import `kaggle_environments`, OpenSpiel, or the Kaggle registry.

## Batch backend: `numThreads` and GIL release (issue #2 throughput seam)

`FastKaggricultureEnv`/`RustBatchEnv` accept an optional instance-local Rayon
worker count:

```python
env = FastKaggricultureEnv(configuration={"seed": 7, "numThreads": 4})
# Raw backend: RustBatchEnv(num_envs, episode_steps, ..., num_threads=None)
```

Semantics:

- `numThreads: None` (default) keeps the historical behavior — parallel batch
  work schedules on Rayon's process-wide global pool.
- `numThreads: N` with `N >= 1` builds a private per-instance pool, so several
  batch environments in one process never oversubscribe each other. `N = 1`
  is fully deterministic single-thread mode.
- `numThreads: 0` (or any value `< 1`) raises `ValueError` in both the Python
  wrapper and the Rust constructor; it is never accepted as "auto".
- `backend.num_threads()` reports the configured count, `0` meaning the
  global-pool default.

Parallel fan-out only engages at `>= 128` environments (`PARALLEL_MIN_ENVS`);
below that a serial loop runs regardless of pool configuration.

GIL release: `reset`, `step`, `step_transition`, `observe_into`,
`action_masks_into`, and `step_into` validate inputs and extract raw buffer
slices while holding the GIL, then run the whole native transition +
observation pass via `py.allow_threads`. Safety boundary: every released call
operates either on owned Rust state or on an exclusively borrowed
caller-owned NumPy buffer (`PyReadwriteArray` borrow + released GIL means no
Python code can touch the buffer concurrently); PyO3's borrow guard rejects
conflicting access to the same backend loudly instead of racing.

Determinism: trajectories are byte-identical across worker counts 1/2/4 and
the default global pool over 130 envs x 30 steps including the day boundary
(`tests/test_batch_throughput_seam.py`). No scaling/speedup claim is made
until measured benchmarks exist. Target deployment topology: Kaggle TPU v5
host with ~96 CPU cores running process-level parallelism times in-process
batch environments (one pinned pool per process/batch). Fused executor /
day-step batching and distributed rollout remain explicitly deferred.
