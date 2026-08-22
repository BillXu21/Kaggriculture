# Five-Day Canonical Schema-v3 Corpus

Date: 2026-08-22

This note records the first complete post-patch five-day canonical replay corpus
used for behavior cloning.

## Source window

Elite Kaggle daily replay partitions:

- 2026-08-17
- 2026-08-18
- 2026-08-19
- 2026-08-20
- 2026-08-21

All processed rows require embedded `module_version == 1.32.7` and canonical
`schema_version == 3`.

## Regeneration result

The corpus was regenerated sequentially on Kaggle after the schema-v3 worker
coordinate correction (`[x,y]` simulator positions -> `tiles[y][x]` board
lookup). Final verification passed across all five files.

| date | episodes | rows |
| --- | ---: | ---: |
| 2026-08-17 | 699 | 41,940 |
| 2026-08-18 | 697 | 41,820 |
| 2026-08-19 | 695 | 41,700 |
| 2026-08-20 | 698 | 41,880 |
| 2026-08-21 | 697 | 41,820 |
| **total** | **3,486** | **209,160** |

Final check:

- files: 5
- rows: 209,160
- schema versions: `{3}`
- wall-clock regeneration time: 1.08 h
- Kaggle output directory: `/kaggle/working/kaggriculture-canonical-v3`

## Tile-dependent label sanity

Full-corpus CARE attribution totals:

- COW: 1,309,686
- SHEEP: 892,397
- GOOSE: 14,996
- unknown: 13,045
- total CARE submissions represented above: 2,230,124
- unknown CARE rate: ~0.585%

Per-day CARE / fertilizer-unknown audit:

| date | cow | sheep | goose | care unknown | fertilizer unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-08-17 | 276,193 | 171,755 | 755 | 3,609 | 617 |
| 2026-08-18 | 270,216 | 178,026 | 1,345 | 3,182 | 578 |
| 2026-08-19 | 253,730 | 184,481 | 3,188 | 1,626 | 303 |
| 2026-08-20 | 259,163 | 178,237 | 4,530 | 3,133 | 518 |
| 2026-08-21 | 250,384 | 179,898 | 5,178 | 1,495 | 650 |
| **total** | **1,309,686** | **892,397** | **14,996** | **13,045** | **2,666** |

The residual unknown tail is small compared with the broken pre-v3 attribution
and is not currently a blocker for the first BC run. Unknown labels remain
honest rather than being fabricated as a species/crop.

## BC handoff

The first model should use the implemented D-019 defaults:

- model: ~1.071M parameter stateless tile Transformer;
- train dates: 2026-08-17 through 2026-08-20;
- validation date: 2026-08-21;
- default score filter: `min_score >= 2950`;
- opponent-public board disabled initially;
- train-only empirical day baseline reported beside model metrics.

Before starting the first full run, audit the actual `bc_manager.adapter`
train/validation arrays so the exact training path is checked rather than only
raw Parquet structure. In particular inspect CARE sparsity, bounded selling,
score ranges, and train/validation row counts.

No full five-day BC training had been run at the time this note was created.
