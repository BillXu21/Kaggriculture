# Agent Note: Configurable Tile Transformer for the Initial BC Manager

Status: implemented
Date: 2026-08-22

## Problem

D-017/D-018 established an elite daily-replay corpus and a canonical
`(episode, seat, day)` schema-v2 record, but no learned manager existed yet.
The project needed a first behavior-cloning model that:

- predicts the realized daily management decisions (crop composition,
  animal counts, land expansion, fertilizer-by-crop, CARE-by-animal, and
  selling) from start-of-day state;
- consumes the compact schema-v2 arrays without leaking evaluation metadata;
- is small enough to train on CPU/laptop for validation yet configurable to a
  serious size later;
- adds no temporal/value/opponent-inference/self-play complexity before the
  simple stationary problem demonstrably learns.

## Decision

Implement a **stateless once-per-day manager** evaluated at `day/hour0`
(after the daily reset). One forward pass per game day; roughly 30 decisions
per episode. No recurrence, no value head, no opponent inference, no
self-play machinery.

**Shared spatial tile encoding.** The own board is exactly 100 tile tokens
(10x10 row-major). A single shared tile encoder embeds the actual adapter
fields: tile kind, crop/animal ids with explicit absent/UNKNOWN ids, numeric
lifecycle channels (planted/placed day, yield, lifespan, fertilized-until,
unwatered/unfed counters, pending care bonus, age, time-to-next-harvest/
product) with stable fixed scaling and explicit NaN indicators for nullable
derived timing, boolean state channels (watered/fed/cared/fertilizer/
harvestable/past-lifespan/starving), presence-mask channels, and row/column
embeddings. The board is never flattened into an MLP.

**Compact global tokens in the same Transformer.** Five tokens summarize:
SELF RESOURCE (own money/shed/seeds/carried inventories/unlocked land),
MARKET (inventory + prices), TOWN (shop counts), LABOR (hires today,
previous hired workers/cost), DAY (learned day embedding plus normalized
day/days-remaining). Scalars are never tokenized individually.

**Configurable small standard PyTorch Transformer.** A plain
`nn.TransformerEncoder` (`batch_first`, `norm_first`, GELU) over
[MANAGER, 100 own tiles, 5 global tokens] = 106 tokens. Default config is
d_model=128, 4 layers, 4 heads, ffn=384 (~1.071M trainable parameters); a
tiny CPU validation config (16/1/1/32, dropout 0) exists for fast tests.

**Opponent-public board optional and off by default.**
`include_opponent_board=True` appends 100 opponent PUBLIC board tokens
through the same shared encoder with a role embedding distinguishing own
from opponent tiles. Opponent private data (shed/seeds/inventories) has no
feature path and unknown input keys are rejected loudly so result metadata
(names/scores/final banks/partition/source) can never leak into features.

**Structured output heads from the MANAGER representation.**

- crop logits `[B,5,count_max+1]` and animal logits `[B,3,count_max+1]`
  (adapter CROP_ORDER / GOOSE-COW-SHEEP);
- land logits `[B,4]` representing unlocked quadrant counts 1..4;
- fertilizer-by-crop logits `[B,5,count_max+1]`;
- CARE-by-animal logits `[B,3,count_max+1]` from schema-v2
  `targets.care_by_animal`;
- sell presence logits `[B,9,6]` and sell quantity regression `[B,9,6]` in
  log1p space.

No tile-specific fertilizer/CARE heads.

**Loss.** Seven fixed-weight group losses: averaged cross-entropy within
each count group's cell grid, land CE, BCEWithLogits over all 54 sell
presence cells, and SmoothL1 on log1p quantity only where true presence > 0
(differentiable zero when none positive). Group means prevent the 54 selling
cells from dominating. Targets are validated and fail loudly rather than
clipped or fabricated.

**Executor boundary.** A deterministic executor later retains exact tile
placement, animal placement, worker routing, hiring, and legal-order details.
The manager outputs economic intent only; there is intentionally no
executor/legal-order mapping in the model.

**Sell adapter semantics.** Per-event quantities are clamped to 0..100
before six-bin aggregation (`floor(hour/4)`); repeated same-bin events may
legitimately accumulate above 100 and are consumed by the regression target
without clipping. Canonical raw sale events remain untouched in the ledger.

**Validation protocol.** Date-held-out splits only (default train
2026-08-17..20, validation 2026-08-21) with a configurable elite cutoff
(default `min_score >= 2950`); never a random seat-day split. The empirical
day baseline fits on train rows only and is reported beside model metrics.
Sparse diagnostics (exact accuracy, MAE, true/pred nonzero rates, nonzero
recall, including per-animal GOOSE/COW/SHEEP breakdowns) make zero collapse
visible.

Implementation: `bc_manager/model.py`, `bc_manager/loss.py`,
`bc_manager/training.py`, `bc_manager/cli.py`; usage in `bc_manager/README.md`.

## Alternatives considered

- **Flattened board MLP:** rejected — discards spatial structure and
  lifecycle locality; the board is the core observation.
- **Temporal/RNN/30-day Transformer across days:** rejected for V0 — the
  manager is deliberately stateless once-per-day; cross-day memory adds
  complexity before the stationary problem is shown to learn.
- **Tile-specific fertilizer/CARE heads:** rejected — canonical targets are
  farm-level totals by crop/animal, not per-tile actions.
- **Raw sell quantity regression without binning/presence:** rejected —
  sells are sparse and heavy-tailed; presence classification plus masked
  log1p regression behaves far better than raw regression over zeros.
- **Random row split:** rejected — leaks near-duplicate seat-days between
  train/validation; date-held-out splits match deployment.
- **Value head / opponent-private inference / self-play now:** rejected —
  BC initialization comes first; PPO/value work and opponent modeling are
  later stages under the established training progression.

## Consequences

- The five-day schema-v2 corpus must be regenerated on Kaggle from raw
  replays; old v1 processed data fails loudly everywhere and cannot be used
  by fabricating CARE targets.
- Full BC training is deferred until the five-day v2 Parquets exist locally
  or on Kaggle; nothing in this decision requires retraining after corpus
  regeneration because the adapter interface is stable.
- Metrics expose sparse zero collapse directly (nonzero recall/rates), so a
  model that predicts all-zero animals or sells is visible immediately.
- The ~1M default model is intentionally modest; scaling (deeper/wider,
  opponent board enabled) is a config change, not a redesign.
- Checkpoints serialize the model config, so reload does not depend on
  reconstructing flags by hand.
