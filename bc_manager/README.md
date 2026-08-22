# bc_manager — schema-v2 BC data layer, manager model, and training CLI

Behavior-cloning stack over the canonical daily-replay Parquet (schema v2,
D-018): compact Arrow adapter, empirical day baseline, tile Transformer
(D-019), group-balanced loss, and a simple training/evaluation CLI.

- `adapter.py` — schema-v2 Parquet -> compact NumPy arrays; date-held-out
  selection only (never random splits); v1/mixed data fails loudly.
- `baseline.py` — train-split-only per-day empirical baseline.
- `model.py` / `loss.py` — configurable tile Transformer (~1.071M default
  parameters) and seven-group loss; see
  `.agents/notes/implemented/2026-08-22-use-configurable-tile-transformer-for-initial-bc-manager.md`.
- `training.py` / `cli.py` — in-RAM tensor dataset, AdamW epoch loop,
  sparse diagnostics, best/last checkpoints.

## Training command (full-data handoff)

Once the five-day schema-v2 corpus exists as Parquet:

```bash
python -m bc_manager.cli data/canonical/2026-08-17.parquet \
    data/canonical/2026-08-18.parquet data/canonical/2026-08-19.parquet \
    data/canonical/2026-08-20.parquet \
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
python -m bc_manager.cli <schema-v2.parquet> --tiny --epochs 2 \
    --batch-size 2 --train-dates 2026-08-17,2026-08-18 \
    --val-dates 2026-08-21 --min-score 2950 \
    --checkpoint-dir data/temp/bc-smoke
```

## Guarantees

- v1 or mixed-version processed Parquet is rejected (`SchemaVersionError`);
  CARE targets are never fabricated — regenerate from raw replays instead.
- Empty train/validation splits fail loudly rather than training.
- Evaluation metadata (names/scores/final banks/partition/source) never
  reaches the model; unknown batch keys are rejected.
- No full five-day training has been run yet (corpus pending).
