# Kaggle TPU v5 (8-core) run — exact cell and pass/fail criteria

Issue #5 throughput smoke for `bc_manager_jax`. This document contains the
exact copy-paste Kaggle notebook cell. **No real TPU measurements exist
yet**; every number currently recorded for this package is local tiny-CPU
plumbing evidence and must not be quoted as TPU throughput.

## Issue #8 update (2026-08-24) — promoted BC-E variant ported; primary target is now E

Issue #8 extended `bc_manager_jax` with model variants **V0** and **E**
(the four-way closed-loop ablation winner, median bank 25,873 vs V0
9,251.5). **J/JE are deliberately NOT ported** and fail loudly as
unsupported (`'J' is not supported by bc_manager_jax`). The variant lives
OUTSIDE the frozen seven-field `ManagerConfig`, mirroring the torch
checkpoint's top-level `model_variant`.

Exact E contract:

- input: `economic_context` float32 `[B, 14]`, finite, built ONLY by the
  authoritative `bc_manager.economics` functions (`economic_context`,
  `derive_economic_context`, `EconomicHistory`) — never re-derived;
- architecture delta: the 14 channels are concatenated after the six
  self-resource feature blocks, before the SAME two-layer self-resource
  MLP (first-layer input 35 -> 49). Trunk/tokens/heads/loss/decode are
  byte-identical to V0;
- parameters: default V0 1,071,040 / E 1,072,832 (+1,792 = 14 x 128);
  tiny V0 37,008 / E 37,232 (+224 = 14 x 16);
- API: `forward(params, inputs, config, model_variant="E")`,
  `train_step(..., model_variant="E")`,
  `convert_torch_state_dict(state, config, "E")`,
  `load_torch_checkpoint(path, config, model_variant="E")` (strict
  expected-variant checks; never inferred from weight shapes),
  `save_native/load_native(..., model_variant=...)`; all old calls remain
  valid with the V0 default;
- native NPZ stores `model_variant` top-level in the JSON metadata
  (outside `model_config`); pre-variant native files load as V0.

Local CPU-only evidence (tiny configs, this host):

- deterministic PyTorch E -> JAX E strict conversion forward parity, all
  seven output groups: worst max abs diff 6.855e-07, worst mean abs diff
  1.101e-07 (tolerances max <= 2e-6, mean <= 5e-7);
- decoded counts/land exact (`array_equal`); total + per-group loss parity
  within 9.5e-7;
- economic feature cases proven against the authoritative path: day-0
  invalid, adjacent-day join, gap invalidation, episode reset/backwards
  day, all-land-unlocked saturation (channel 10 = 8.0, flag 0), zero and
  negative cash, exact seed/animal/land channel order;
- single-device JIT E forward + train step finite; N=4 logical-CPU
  NamedSharding subprocess: 1-vs-4 total loss diff 1.9e-6, group diff
  2.4e-7, updated-param diff 3.7e-9, batch spec `P('data', None)`;
  N=8 logical-CPU smoke: one tiny sharded E step, finite, same spec.
  **N=4/N=8 are logical multi-device validation on forced host CPUs only —
  not throughput or scaling evidence of any kind.**

Real checkpoint status: `artifacts/local/bc-v1-E/best.pt` is ABSENT
locally; no real-checkpoint conversion/parity has run. A bounded
skip-if-absent test exists. Exact rerun once the file is copied in place:

```bash
python -m pytest tests/test_bc_manager_jax_parity.py::test_real_e_checkpoint_parity_if_present -q
```

Exact eventual Kaggle TPU command (UNMEASURED — run only when TPU capacity
exists; do not quote any local number as its expected result):

```bash
python -m bc_manager_jax.benchmark \
  --device-counts 8 \
  --dtype f32 \
  --checkpoint /kaggle/working/bc-v1-E/best.pt \
  --batch-sizes 256,512,1024,2048,4096 \
  --warmup 3 \
  --iterations 10 \
  --output-json /kaggle/working/bc_e_jax_benchmark.json \
  --output-csv /kaggle/working/bc_e_jax_benchmark.csv
```

The benchmark records `model_variant` additively per row (backward
compatible); checkpoint mode always uses the stored variant.

## Scope / non-goals reminder

Faithful JAX mirror of the BC manager only: no PPO/value heads/self-play,
no architecture replacement, no fast-env integration, single-host
(single-VM) TPU slice only.

## Kaggle setup

- Accelerator: **TPU VM v5-8** (8 cores, single process).
- Notebook: standard Kaggle Python notebook; the repo clone lives in
  `/kaggle/working` (ephemeral).

## Cell 1 — environment: clone/update repo, install Optax without touching TPU jaxlib

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/Kaggriculture
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch origin
  git -C "$REPO" merge --ff-only origin/main   # never rebase/force in the notebook
else
  git clone https://github.com/BillXu21/Kaggriculture.git "$REPO"
fi
cd "$REPO"
pip install -q -r requirements-jax.txt   # adds Optax only; jax/jaxlib stay preinstalled
python - <<'PY'
import jax, optax
print("jax", jax.__version__, "optax", optax.__version__)
print("backend", jax.default_backend(), "devices", jax.device_count())
print([d.device_kind for d in jax.devices()])
assert jax.default_backend() != "cpu", "TPU runtime expected"
assert jax.device_count() == 8, f"expected exactly 8 TPU devices, got {jax.device_count()}"
PY
```

If `device_count != 8`, STOP: record the value and do not run benchmarks.

## Cell 2 — focused issue-#5 tests on the TPU host

```bash
%%bash
cd /kaggle/working/Kaggriculture
python -m pytest tests/test_bc_manager_jax_parity.py \
                  tests/test_bc_manager_jax_train.py \
                  tests/test_bc_manager_jax_benchmark.py -q \
                  --basetemp=/kaggle/working/pytest-jax
```

Expected: `30 passed`. The benchmark tests force a 1-device mesh and pass on
the TPU host as well; parity tests use CPU torch tensors via the Kaggle
preinstalled torch.

## Cell 3a — checkpoint benchmark (only if `/kaggle/working/bc-v0-score2950/best.pt` exists)

The checkpoint's stored config fixes ONE token regime; conversion is strict
and any incompatibility fails loudly (never silently skipped).

```bash
%%bash
set -euo pipefail
cd /kaggle/working/Kaggriculture
CKPT=/kaggle/working/bc-v0-score2950/best.pt
if [ -f "$CKPT" ]; then
  python -m bc_manager_jax.benchmark \
    --device-counts 8 --dtype f32 \
    --checkpoint "$CKPT" \
    --batch-sizes 256,512,1024,2048,4096 \
    --warmup 3 --iterations 10 \
    --output-json /kaggle/working/bc_jax_benchmark_checkpoint.json \
    --output-csv  /kaggle/working/bc_jax_benchmark_checkpoint.csv
else
  echo "SKIP: $CKPT not present; checkpoint benchmark explicitly skipped."
fi
```

All flags above verified against `python -m bc_manager_jax.benchmark --help`
(`--checkpoint`, `--device-counts`, `--dtype`, `--batch-sizes`, `--warmup`,
`--iterations`, `--output-json`, `--output-csv`). No invented flags.

## Cell 3b — random-init opponent-board (206-token) regime as a separate command

Run this regardless of the checkpoint branch when the checkpoint config is
own-only (and optionally always), so both token regimes are represented:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/Kaggriculture
python -m bc_manager_jax.benchmark \
  --device-counts 8 --model-config default --regimes opponent --dtype f32 \
  --batch-sizes 256,512,1024,2048,4096 \
  --warmup 3 --iterations 10 \
  --output-json /kaggle/working/bc_jax_benchmark_opponent206.json \
  --output-csv  /kaggle/working/bc_jax_benchmark_opponent206.csv
```

Optional companion (own-only random regime):

```bash
python -m bc_manager_jax.benchmark --device-counts 8 \
  --model-config default --regimes own --dtype f32 \
  --batch-sizes 256,512,1024,2048,4096 --warmup 3 --iterations 10 \
  --output-json /kaggle/working/bc_jax_benchmark_own106.json \
  --output-csv  /kaggle/working/bc_jax_benchmark_own106.csv
```

A 1-vs-8 scaling point can be added with `--device-counts 1,8` (rows for
unavailable counts are honestly reported as skipped, never invented).

## Pass/fail criteria

PASS requires ALL of:

1. Cell 1 asserts backend != cpu and exactly 8 devices.
2. Cell 2 reports `30 passed`; zero test or conversion errors.
3. Every successful JSON/CSV row has BOTH metric families positive:
   `inference_compile_seconds`, `inference_examples_per_second_mean/best`,
   `train_compile_seconds`, `train_examples_per_second_mean/best`
   (inference strictly faster than train per example).
4. No hidden skips: every non-ok row must carry an explicit `reason`
   (acceptable: OOM at large batches). Any other skip must be explained.
5. Both token regimes represented across the run (checkpoint regime plus
   the separate 206-token random run).
6. All JSON/CSV outputs persisted under `/kaggle/working`.

FAIL: device assertion failure, any test/conversion error, missing metric
family, unexplained skips, or outputs not persisted.

## Honesty rules

- Do not quote local CPU numbers as TPU results. (Recorded local plumbing
  smoke, pre-correction fields: JAX 0.10.2, 1 CPU device, tiny 37,008-param
  config, batch 256 — own train ≈2576 ex/s, opponent train ≈858 ex/s.)
- Skipped cases keep null metrics plus their reason; nothing is fabricated.
