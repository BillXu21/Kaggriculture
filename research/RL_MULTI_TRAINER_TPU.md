# Issue #21 — Multi-Trainer TPU Prototype

This prototype answers a narrow topology question: can one Python process own
JAX/libtpu while independent PPO trainers are assigned to different TPU
devices? Rollout workers are not started by this benchmark and remain outside
the prototype.

## Architecture

- `rl_manager/multitrainer_benchmark.py` is a standalone, synthetic manager-day
  benchmark. It creates one independent `PPOTrainState` per trainer, one
  explicit assignment `trainer i -> jax.devices()[i]`, and one JAX process.
- Parameters, optimizer state, RNG, inference inputs/outputs, PPO action/index
  arrays, and PPO scalar arrays are checked and recorded by actual placement.
- Inference uses the existing JAX E forward path. PPO uses the existing
  compiled single-minibatch update primitive used by `rl_manager.ppo`.
- Sequential timing blocks each trainer before dispatching the next. Concurrent
  timing dispatches all trainer calls first and then blocks on all outputs.
- `rl_manager/multitrainer.py` is the narrow JAX-free routing seam. It groups
  request rows by exact `PolicyIdentity` and rejects unknown or contaminated
  trainable trajectory rows.

No league, scheduler, worker TPU import, or `parallel.py` rewrite is included.
The benchmark refuses to reuse a device: a requested N greater than the number
of visible JAX devices is an explicit skipped result.

## Kaggle Setup

Use the issue-21 commit and leave the preinstalled TPU JAX/JAXLIB unchanged:

```bash
cd /kaggle/working
git clone https://github.com/BillXu21/Kaggriculture.git
cd Kaggriculture
git checkout codex/issue-21-multitrainer
# Do not install or upgrade jax/jaxlib on Kaggle TPU.
if python -c "import optax" 2>/dev/null; then
  echo "using preinstalled optax"
else
  python -m pip install -q --no-deps optax
fi
python - <<'PY'
import jax
import optax
print("backend", jax.default_backend())
print("devices", jax.devices())
assert jax.default_backend() != "cpu"
assert jax.device_count() >= 4
PY
```

Run these commands one at a time. Each command is one Python process; do not
launch eight independent TPU Python processes and do not run commands in
parallel. The checkpoint path is the externally supplied promoted BC-E
checkpoint.

```bash
python -m rl_manager.multitrainer_benchmark \
  --checkpoint /kaggle/working/bc-v1-E/best.pt \
  --trainer-counts 1 --batch-size 256 --warmup 3 --iterations 10 \
  --output-json /kaggle/working/issue21_multitrainer_n1.json
```

```bash
python -m rl_manager.multitrainer_benchmark \
  --checkpoint /kaggle/working/bc-v1-E/best.pt \
  --trainer-counts 2 --batch-size 256 --warmup 3 --iterations 10 \
  --output-json /kaggle/working/issue21_multitrainer_n2.json
```

```bash
python -m rl_manager.multitrainer_benchmark \
  --checkpoint /kaggle/working/bc-v1-E/best.pt \
  --trainer-counts 4 --batch-size 256 --warmup 3 --iterations 10 \
  --output-json /kaggle/working/issue21_multitrainer_n4.json
```

Only if N=4 completes without OOM or placement errors, optionally run N=8:

```bash
python -m rl_manager.multitrainer_benchmark \
  --checkpoint /kaggle/working/bc-v1-E/best.pt \
  --trainer-counts 8 --batch-size 256 --warmup 3 --iterations 10 \
  --output-json /kaggle/working/issue21_multitrainer_n8.json
```

To run the N=1/2/4 matrix in one process instead, use
`--trainer-counts 1,2,4`; separate commands make per-N output and memory
failures easier to inspect.

## Interpretation

The JSON is the measurement artifact. Check `metadata.visible_devices`, each
row's `explicit_assignment`, and `placements` before reading timing fields.
The N=1 `default_single_trainer_diagnostic` reports the unsharded existing
single-trainer path, including compile time and host PPO-array placement.

Compare `inference` compile/steady timings and `ppo_update` compile timings,
then compare `sequential` with `concurrent_dispatch`. Concurrent dispatch is
only a JAX async-dispatch experiment; it is not evidence that TPU work
overlapped unless the real TPU output demonstrates it.

There are no TPU results in this repository. Local CPU/mock tests establish
identity routing and placement/configuration behavior only. Do not claim TPU
scaling without the real Kaggle JSON artifacts.
