# bc_manager — schema-v3 BC data layer, manager model, and training CLI

Behavior-cloning stack over the canonical daily-replay Parquet (schema v3,
D-018): compact Arrow adapter, empirical day baseline, tile Transformer
(D-019), group-balanced loss, and a simple training/evaluation CLI.

- `adapter.py` — schema-v3 Parquet -> compact NumPy arrays; date-held-out
  selection only (never random splits); v1/v2/mixed data fails loudly.
- `baseline.py` — train-split-only per-day empirical baseline.
- `model.py` / `loss.py` — configurable tile Transformer (~1.071M default
  parameters) and seven-group loss; see
  `.agents/notes/implemented/2026-08-22-use-configurable-tile-transformer-for-initial-bc-manager.md`.
- `training.py` / `cli.py` — in-RAM tensor dataset, AdamW epoch loop,
  sparse diagnostics, best/last checkpoints.

## Five-day corpus

The first full schema-v3 corpus was regenerated and verified on Kaggle from the
elite 1.32.7 daily partitions 2026-08-17 through 2026-08-21:

- 3,486 episodes;
- 6,972 seat trajectories;
- 209,160 `(episode, seat, day)` rows;
- schema versions exactly `{3}`.

Default BC selection is `min_score >= 2950`, with train dates
2026-08-17..2026-08-20 and validation date 2026-08-21. That selects 25,500
train rows and 5,700 held-out validation rows.

## Training command

All five Parquets must be passed to the CLI because the held-out Aug-21 file is
still required for validation:

```bash
python -m bc_manager.cli \
    data/canonical/2026-08-17.parquet \
    data/canonical/2026-08-18.parquet \
    data/canonical/2026-08-19.parquet \
    data/canonical/2026-08-20.parquet \
    data/canonical/2026-08-21.parquet \
    --train-dates 2026-08-17,2026-08-18,2026-08-19,2026-08-20 \
    --val-dates 2026-08-21 \
    --min-score 2950 \
    --checkpoint-dir data/temp/bc-checkpoints
```

Defaults: AdamW lr 3e-4, weight decay 1e-2, batch size 256, 30 epochs,
gradient clip 1.0, seed 0, date split and cutoff as above. Device is CUDA
when available else CPU; `--amp auto` enables CUDA AMP only. Checkpoints
`best.pt`/`last.pt` are written under the caller-provided directory. Model
config is serialized inside each checkpoint; reload with
`bc_manager.training.load_model_from_checkpoint`.

## First full reference run

The first untouched default-size run completed on CUDA+AMP over the schema-v3
five-day corpus. Provenance and complete held-out metrics are recorded in
`research/FIRST_BC_V0_EVAL.md`.

Headline result:

- 1,071,040 parameters;
- best epoch 29/30;
- best Aug-21 validation total 2.8889;
- ~237 s total runtime;
- state-aware model materially beat the train-only day baseline on every
  manager group measured.

Examples of the held-out model vs day baseline gap:

- crop MAE: 1.273 vs 3.622;
- animal MAE: 0.268 vs 1.694;
- CARE MAE: 0.330 vs 1.668;
- land accuracy: 0.991 vs 0.909;
- tomato nonzero recall: 83.8% vs 0%;
- goose nonzero recall: 96.8% vs 0%;
- goose CARE nonzero recall: 95.5% vs 0%.

Selling is less complete: held-out sell-presence recall is ~64.8% despite
93.9% presence accuracy, so sparse accuracy alone should not be treated as
success.

## BC-E to JE distillation

Initialize a larger JE model from an own-only E checkpoint by retaining the
original expert hard labels and mixing them with teacher soft targets:

```bash
python -m bc_manager.cli <all-five-parquets> \
    --variant JE --e-history-version E_LEGACY \
    --d-model 256 --layers 6 --heads 8 --ffn 768 --dropout 0.1 \
    --teacher-checkpoint /kaggle/input/datasets/billll/v0-bc-e/best.pt \
    --distill-weight 0.5 --distill-temperature 2.0 \
    --checkpoint-dir data/temp/bc-je-distill
```

The categorical groups use temperature-scaled KL, sell presence uses soft
sigmoid BCE targets, and sell quantity uses teacher-presence-weighted
SmoothL1 in log1p space. `--distill-weight 0` preserves hard-label-only
training; the default CLI behavior remains unchanged when no teacher is
provided. Teacher metadata is stored in a top-level `distillation` checkpoint
block without serializing the teacher model.

The next gate is closed-loop evaluation with a deterministic executor. Do not
interpret this teacher-forced result as proof of competitive game strength.

## Tiny CPU smoke

```bash
python -m bc_manager.cli <schema-v3.parquet> --tiny --epochs 2 \
    --batch-size 2 --train-dates 2026-08-17,2026-08-18 \
    --val-dates 2026-08-21 --min-score 2950 \
    --checkpoint-dir data/temp/bc-smoke
```

## Guarantees

- v1, v2, or mixed-version processed Parquet is rejected
  (`SchemaVersionError`); CARE targets are never fabricated — regenerate from
  raw replays instead.
- Empty train/validation splits fail loudly rather than training.
- Evaluation metadata (names/scores/final banks/partition/source) never
  reaches the model; unknown batch keys are rejected.
- Sell quantities are bounded per primitive event only in the BC adapter before
  six-bin aggregation; raw canonical sale events remain untouched.
