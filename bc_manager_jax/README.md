# bc_manager_jax — pure-JAX mirror of the daily-manager BC Transformer

Issue #5 implementation (stages `5c2a97b`, `71ef1bf`, correction `e64021f`).
A faithful pure-JAX reimplementation of `bc_manager.model.DailyManagerTransformer`
with an exact mirror of `bc_manager.loss.manager_loss`, strict
PyTorch→JAX checkpoint conversion, an existing-semantics Optax train step,
single-host replicated data parallelism, and an honest throughput benchmark.

## Scope and non-goals

This package is a **faithful mirror only**. Non-goals (deliberately absent):

- no PPO, value heads, self-play, schedulers, or architecture changes;
- no replacement of the PyTorch BC code (`bc_manager/` stays authoritative);
- no fast-env/Rust integration;
- no multi-host (multi-process) training — single-host addressable devices only;
- no Flax — parameters are plain pytrees, config is a frozen dataclass.

Pure JAX + Optax was chosen because the model is a small static computation
with no Flax feature requirements; `requirements-jax.txt` deliberately does
not pin or install jax/jaxlib so Kaggle TPU preinstalled JAX/JAXLIB builds
remain untouched — it adds only a compatible Optax bound.

## Public API (`bc_manager_jax/__init__.py`)

| Symbol | Purpose |
|---|---|
| `ManagerConfig`, `tiny_manager_config` | frozen config mirroring `bc_manager.model.ManagerConfig` |
| `empty_params(config)` | canonical zero pytree / shape spec |
| `init_params(config, seed)` | N(0, 0.02²) init (NOT torch-init-equivalent; use conversion for parity) |
| `forward(params, inputs, config, *, training=False, rng=None)` | eval-exact forward; input-key validation at this non-jitted boundary |
| `validate_inputs(inputs, config)` | rejects missing/unexpected adapter keys loudly (metadata can never leak) |
| `predict_counts`, `predict_land`, `predict_sells` | decode helpers mirroring torch semantics |
| `manager_loss(outputs, targets, weights=None)` | validating loss; `loss_from_validated` is the jit-safe core; `validate_target_shapes` validates without outputs |
| `TrainConfig`, `make_optimizer`, `init_opt_state`, `train_step(...)` | AdamW(lr 3e-4, wd 1e-2, betas .9/.999, eps 1e-8) after global-norm clip 1.0 — existing `bc_manager.training` semantics |
| `create_data_mesh`, `shard_batch`, `replicate_tree` | NamedSharding replicated data-parallel helpers |
| `convert_torch_state_dict`, `load_torch_checkpoint(path_or_payload, config=None)`, `expected_torch_state_shapes` | strict conversion from `bc_manager_checkpoint_v1` |
| `save_native` / `load_native` | pickle-free `.npz` native format `bc_manager_jax_checkpoint_v1` |

## Checkpoint conversion

- Source format: `bc_manager_checkpoint_v1` payloads written by
  `bc_manager.training.save_checkpoint`; accepted as path or in-memory dict,
  loaded on CPU via `torch.load(..., weights_only=True)`.
- Mapping: every `nn.Linear` weight `[out, in]` is transposed to a kernel
  `[in, out]`; packed attention `in_proj_weight/bias` stay PACKED as
  `qkv_kernel/qkv_bias` with chunk order q|k|v; embeddings/LayerNorm map
  unchanged; `manager_token` is stored squeezed `(d,)`.
- Strictness: every expected state-dict key/shape is enumerated from the
  config; missing keys, unexpected keys, shape mismatches, non-float32
  tensors, and incompatible configs (checked against the payload's stored
  `model_config`) fail loudly. Note: own-only vs `include_opponent_board`
  state dicts are structurally identical, so that incompatibility is caught
  at the checkpoint-metadata level.
- Conversion always copies; converted params never alias torch storage.
- Status: strictly tested against generated checkpoints. A real trained
  `best.pt` was NOT available locally and has not been converted locally.

## Numerical parity evidence (float32 CPU)

All seven output groups compared between seeded PyTorch models and the
converted JAX forward, for tiny own-only, tiny opponent-board (random init),
and default d128/L4/H4 own-only:

- tolerance used by tests: max abs ≤ 2e-6, mean abs ≤ 5e-7;
- worst observed output max abs diff: 1.26e-6 (fertilizer logits, opponent mode);
- decoded count/land predictions exactly equal (`array_equal`);
- loss groups agree within 4.8e-7, total within 1.9e-6; zero-positive
  sell_quantity is exactly 0.0 on both sides.

Eval forward is the hard parity gate; dropout sampling need not match torch
numerically (different RNG streams), only placement/statistics.

## Training

`train_step` matches existing BC semantics: AdamW over ALL parameters
(no decay exclusions), lr 3e-4, weight decay 1e-2, betas 0.9/0.999, eps 1e-8,
global L2 gradient clip 1.0 applied BEFORE AdamW via
`optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(...))`, no scheduler.
Dropout uses an explicit PRNG key; the step returns the advanced key so masks
never repeat. Verified against a manual reference implementation and against
a PyTorch backward+clip+AdamW one-step (dropout=0); post-step parameter
differences are bounded by Adam's sign-like first-step normalization
amplifying float32 gradient noise (max ≈ 2·lr).

Single-device and N-device runs share the same compiled code.

## Replicated data parallelism (single host)

One logical global batch sharded along axis 0 via
`NamedSharding(mesh, ('data', ...))`; parameters and optimizer state fully
replicated (`PartitionSpec()`). All loss reductions are full-global-batch
`jnp.mean`s, so GSPMD inserts correct cross-replica reductions; because the
global batch must divide evenly across devices, mean-of-shards equals the
global mean exactly. Multi-host (`process_count != 1`) is rejected.

Evidence (forced 4-CPU subprocess vs 1 device, same global batch):
total loss diff 1.9e-6, group diff 1.8e-7, updated-param diff 1.3e-8;
sharding specs asserted (`P('data', None)` batch, `P()` params).

## Benchmark

`python -m bc_manager_jax.benchmark --help` (also `scripts/benchmark_bc_jax.py`).

Per successful case row it reports BOTH metric families under stable names:
`inference_compile_seconds`, `inference_examples_per_second_mean/best`,
`train_compile_seconds`, `train_examples_per_second_mean/best` — plus JAX
version/backend/device descriptions/count, requested/used device counts,
token count (106 own-only / 206 opponent), parameter count, dtype mode,
global/per-device batch, warmup/iteration counts. Every timed result is
synchronized with `.block_until_ready()` before `perf_counter` stops;
compilation is reported separately from steady state. Batches default to
256..4096 (must divide by every requested device count); `--device-counts`
supports e.g. `1,8`; unavailable counts/OOM are recorded as explicit
`skipped` rows — never invented. f32 is the parity-preserving default;
optional bf16 casts floating leaves and is labeled per row. JSON and CSV
outputs are supported.

**No real TPU measurements exist.** Local numbers below are tiny-CPU
plumbing smoke only (JAX 0.10.2, 1 CPU device, tiny config 37,008 params,
global batch 256, measured before the inference/train reporting-field
correction): own-only train ≈ 2576 ex/s (compile 5.19 s), opponent train
≈ 858 ex/s (compile 5.93 s). They say nothing about TPU throughput.

The exact Kaggle TPU v5 8-core cell and pass/fail criteria live in
[`research/JAX_TPU_V5_RUN.md`](../research/JAX_TPU_V5_RUN.md).

## Tests

- `tests/test_bc_manager_jax_parity.py` (18): shapes, validation, eval/loss
  parity, decode equality, converter strictness, native round-trip.
- `tests/test_bc_manager_jax_train.py` (8): step semantics, manual-reference
  optimizer check, torch one-step agreement, torch-storage aliasing
  regression, forced 4-CPU 1-vs-4 equivalence subprocess.
- `tests/test_bc_manager_jax_benchmark.py` (4): metadata/results smoke,
  JSON+CSV round-trip of both metric families, skipped-row honesty,
  checkpoint mode, CLI help.

All 30 pass locally alongside the existing PyTorch BC suites
(`test_bc_manager*.py`, 35 passed).
