# Stage 2.5 Packet 3 contract: native BC and checkpoints

Status: authoritative for the Packet 3 native BC and checkpoint seams. Packet 1
mechanics/vocabularies and the Packet 2 policy contract remain authoritative for
physical support, action semantics, and likelihood.

Packet 3 is framework-light: it trains the existing shared Stage 2.5 JAX policy
with teacher forcing, stores strict native checkpoints, and stops before PPO,
executor wiring, live penalties, or large training runs.

## Public entry points

- `rl_manager.stage25_adapter`
  - `load_dataset(...)` / `load_stage25_dataset(...)`: one complete split from
    projected canonical Parquet.
  - `load_train_val(...)` / `load_stage25_train_val(...)`: date-held-out
    train/validation splits.
  - `Stage25AdapterConfig`, `DEFAULT_TRAIN_DATES`, `DEFAULT_VAL_DATES`,
    `DEFAULT_MIN_SCORE`.
- `rl_manager.stage25_bc`
  - `Stage25BCConfig`, `Stage25BCBatch`.
  - `make_fixed_batch`, `iter_fixed_batches`, `loss_and_metrics`, `train_step`.
  - `make_optimizer`, `init_opt_state`.
  - `save_checkpoint`, `load_checkpoint`, `import_encoder_checkpoint`,
    `load_array_dataset`.
- `rl_manager.stage25_checkpoint`
  - `save_stage25_inference_checkpoint` / `load_stage25_inference_checkpoint`.
  - `save_stage25_bc_checkpoint` / `load_stage25_bc_checkpoint`.
  - `import_historical_encoder`, `validate_array_tree`, `Stage25CheckpointError`.
  - Short aliases `save_inference_checkpoint`, `load_inference_checkpoint`,
    `save_bc_checkpoint`, `load_bc_checkpoint`.
- `rl_manager.stage25_bc_cli`: `python -m rl_manager.stage25_bc_cli ...`.

All native paths are Torch-free. Torch is imported only from the explicit
historical-checkpoint conversion seam when a `.pt`/`.pth` source is requested.

## Operating history versus source history

Two distinct E-history identities are recorded:

- **Operating history** (`e_history_version` / `e_identity`): the observation
  semantics used by this Stage 2.5 policy and its training data. Default and
  only supported stage for Stage 2.5 training is `E_CORRECTED_V1`.
- **Source history** (`source_e_identity`): the semantics of the historical
  encoder imported for initialization. It is an `{"variant": "E",
  "history_version": ..., "transfer": "encoder_only"}` record and never changes
  the operating history.

Importing an `E_LEGACY` source encoder with explicit `allow_legacy_e=True` does
**not** make corrected-E training a legacy-history run. The checkpoint records
`e_history_version=E_CORRECTED_V1` plus a legacy `source_e_identity` and source
provenance; it must not claim corrected-source parity.

Conversely, an explicitly legacy-**operating** checkpoint can be loaded only via
explicit `expected_e_history_version`/`allow_legacy_e` controls. The BC CLI
supports corrected operating history only and rejects a legacy-operating resume
with a clear error rather than encoding corrected data under a legacy label.

`source_identity`, `provenance`, and `executor` are explicit arguments. Generic
`metadata` must not contain reserved checkpoint keys (`e_history_version`,
`e_identity`, `source_identity`, `provenance`, `executor`, `source_e_identity`,
`source_history_version`, and the BC/optimizer fields); collisions raise rather
than being silently dropped or given an implicit precedence.

## Resume boundary

The supported resume boundary is `after_completed_update_before_next_batch`.
A BC checkpoint stores the post-update parameters, the complete Optax state, the
explicit next RNG key, and a data-order cursor (`epoch`, `batch`, `seed`). Resume
reconstructs the optimizer tree from the exact caller template (or the persisted
optimizer config), validates the tree signature, leaf count, and every
shape/dtype, and reproduces the next update deterministically. Re-saving after a
resume preserves the operating/source histories and identities.

## Complete-row BC eligibility and padding

Only complete nine-action autoregressive examples from
`rl_manager.stage25_data.build_outcome_proxy_labels` enter BC. A complete row
requires a populated, physically and curriculum-supported class at every step
under the exact preceding observed prefix. Partial diagnostic rows are retained
separately and are never trainable. Synthetic `K` is the pre-decision persistent
goal ledger; the physical capacity `C` is derived by the policy, not supplied.

Batches are fixed-shape. A short final batch repeats its last real row
(`edge` padding); `Stage25BCBatch.real_row_mask` marks real rows. Loss, per-step
NLL, and accuracy reduce only over real rows. The loss is the masked mean of the
policy's `joint_logprob`; per-step NLL uses the policy's `component_logprobs`.
BC never recomputes support or a second masked log-softmax.

Invalid teacher-forced actions (out of vocabulary or physically unsupported) are
detected from the single objective forward's exact policy support and raise
before the optimizer or RNG advance, so diagnostic zero placeholders can never
enter an update. Value-head parameters are excluded from the optimizer
(including weight decay) for the whole of BC.

## Checkpoint identity and atomicity

Native checkpoints are single `.npz` archives with a UTF-8 JSON `__meta__`
record. The parameter tree is rebuilt from `init_stage25_params`; every path,
shape, and dtype is validated against that template and the stored leaf
manifest. Architecture/action/observation/ledger/physical/bc-target versions,
the action and observation vocabularies, curriculum, precision, and E identities
are checked. Writes go to a same-directory temporary file, `fsync`, then atomic
`os.replace`, so a failure leaves the previous checkpoint intact.

## Unavailable real-artifact validation

The August 2026 canonical datasets and the historical `E` checkpoints are not
available in this environment; Packet 3 validation uses generated native
fixtures and miniature datasets. Real-artifact parity, TPU throughput, and
large-data runs are deferred and must be re-established when those artifacts are
available.

## Running native BC

```bash
# Tiny synthetic dataset (NPZ with ``actions`` plus ``input_*`` arrays).
python -m rl_manager.stage25_bc_cli \
    --data data.npz --model-size tiny --batch-size 8 --steps 100 \
    --output stage25_bc.npz

# Initialize from an explicit (optionally legacy) source encoder.
python -m rl_manager.stage25_bc_cli \
    --data data.npz --import source-e.npz --allow-legacy-e \
    --output stage25_bc.npz

# Resume and continue.
python -m rl_manager.stage25_bc_cli \
    --data data.npz --resume stage25_bc.npz --steps 100 \
    --output stage25_bc.npz
```

## Non-goals

Packet 3 does not implement PPO, trajectory/provider/executor wiring, live
shortfall or upkeep penalties, reward changes, TPU benchmarking, Torch-dependent
native startup, or a new physical-support implementation.
