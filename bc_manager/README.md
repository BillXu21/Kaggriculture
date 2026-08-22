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

Default BC selection remains `min_score >= 2950`, with train dates
2026-08-17..2026-08-20 and validation date 2026-08-21.

## Training command (full-data handoff)

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
`best.pt`/`last.pt` are written under the caller-provided directory (use an
ignored path such as `data/temp`). Model config is serialized inside each
checkpoint; reload with `bc_manager.training.load_model_from_checkpoint`.

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
- No full five-day BC training run has been completed yet. The schema-v3 corpus
  is ready; adapter audit and the first date-held-out training run are next.
