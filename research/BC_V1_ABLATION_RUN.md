# BC V1 Ablation — Durable State and Exact Kaggle Runbook (issue #6)

> Issue #30 correction: for any new E/JE run, use
> `--e-history-version E_CORRECTED_V1`. The raw-data regeneration, corrected
> BC command, explicit legacy behavior, and PPO readiness command are recorded
> in [`ISSUE30_E_HISTORY_RUNBOOK.md`](ISSUE30_E_HISTORY_RUNBOOK.md).

Status: **implementation complete locally; real corpus/checkpoints absent on
this machine; no real training results, no closed-loop panel results, and no
winner exist yet.** The exact Kaggle execution below is the next gate. This
document records what was built, what was deliberately rejected, the local
evidence, and the copy-paste commands for the real run. Current source is
authoritative over this note.

Implementation commit chain (all on `main`, verified ancestors):

- `2f48564` — Stage 0 schema-v3 economic provenance audit
  (`research/BC_V1_ECONOMIC_CONTEXT.md`);
- `192d0dc` — Stage 1 economic context foundation (`EconomicHistory`,
  coherence diagnostics, V0/E variants);
- `3d7fae1` — Stage 2 J/JE joint plan decoder;
- `fc95752` — Stage 3 live E/JE history integration, diagnostic-only
  executor/opening exposure, four-variant closed-loop panel CLI.

---

## 1. The four variants

All four share the D-019 tile-Transformer trunk, the identical data layer,
date-held-out split, loss, and optimizer; they differ only in decoder shape
and input channels:

| variant | change vs V0 | parameters |
| --- | --- | ---: |
| V0 | baseline (current inputs, factorized heads) | 1,071,040 |
| J | joint plan decoder: one autoregressive-style joint head couples crop composition, animal targets, land, fertilizer/CARE, and selling so plans are decoded as one coherent object instead of independent heads | 1,204,288 (+133,248) |
| E | economic context: 14 extra scalar channels derived exclusively from realized schema-v3 fields (previous-day net-cash delta signed-log-clipped to [-8, 8], hire-affordability ratio `log1p(cash / fib(hires_today+1))` clipped likewise, realized hires/hire-cost from the observed `hires_today` counter) fed through a live `EconomicHistory` tracker that exactly mirrors batch derivation including day gaps and partition boundaries | 1,072,832 (+1,792) |
| JE | joint decoder + economic context together | 1,206,080 |

V0 compatibility is regression-tested: V0/J checkpoints load and decode
through the exact pre-V1 encoder call (no economic key, no tracker), and the
existing executor/opening behavior is unchanged (additive diagnostics keys
only).

### Feature choices and rejections (Stage 0 audit)

The controlling hard rule: **submitted intents are not realized fills.**
`events.buys`/`events.sells`/`events.market_events_ordered` record submitted
market orders; nothing in schema v3 proves any individual order cleared.
Therefore gross revenue, gross spend, and any sell/market fill quantity are
**never inferred** and are absent from V1 features. The only economic
quantities treated as realized are observed state snapshots
(`start.self.money` / `end.self.money`) and the observed `hires_today`
counter (via `start.previous_execution`). Full derivation table:
`research/BC_V1_ECONOMIC_CONTEXT.md`.

Coherence diagnostics (plan-vs-cash lower-bound feasibility, JSON-safe ratios
with explicit `zero_cash`/`over_1x`/`over_2x` flags — never Infinity) are
recorded per live plan but are **diagnostic-only**: never clipped into the
plan, never fed back into any decision.

## 2. Local evidence (bounded claims only)

- Stage implementation sweeps grew to **275 passed** across the BC V1 suites
  (economics/joint/ablation), the existing `bc_manager` suites, executor_v0,
  and opening_book tests (combined run at commit `fc95752`).
- Independent Ox audit of the full implementation: **62 new + 163 compat =
  225 passed**, verdict PASS_WITH_FINDINGS with no code blockers.
- One official opening-only smoke (seed 7, seat 0, `standard_mixed`) ran
  under pinned `kaggle-environments==1.32.7`: all 96 turns replayed, clean
  d4h0 handoff, zero divergence/fallback/status anomalies. This exercised
  plumbing only — **no BC weights were attached and no policy quality is
  claimed**.
- Real five-day corpus Parquet and real trained checkpoints are **absent
  locally**; there are no teacher-forced variant results, no closed-loop
  panel results, and **no winner**. Nothing below fabricates them.

## 3. Panel criterion — the only promotion gate

No V1 variant may be promoted based on teacher-forced validation totals or
coherence diagnostics alone. Promotion requires the fixed paired closed-loop
panel:

- opening: committed `standard_mixed` identity, days 0–3 (96 turns), clean
  d4h0 handoff;
- downstream: the tested BC manager behind the **unchanged**
  `executor_v0.ExecutorAgent`;
- opponents: trivial PASS responder (plumbing-grade, per the evaluation
  rules — this is a controlled ablation comparison between variants, not a
  competitive-strength measurement);
- pairing: seeds **7, 17, 42, 123, 2026** × seats **{0, 1}** × 4 variants =
  40 games;
- ranking: closed-loop final-bank **median, then mean**;
- teacher-forced `validation_metrics.total` and coherence aggregates are
  reported as prerequisites/diagnostics only;
- seed-17 collapse flag (bank < 100.0 heuristic threshold, raw bank always
  shown) and seed-2026 retention vs V0 must be reported beside the ranking.

Strict gates enforced by `bc_manager.ablation`: every checkpoint must exist,
carry format `bc_manager_checkpoint_v1`, store a top-level `model_variant`
matching its mapping slot, and contain teacher-forced
`validation_metrics.total` (smoke weights are rejected loudly); the official
engine must pass the pinned 1.32.7 provenance guard; any divergence,
fallback, or status anomaly marks the run failed (exit 1). `--validate-only`
preflights everything without importing the engine.

## 4. Scope exclusions

This stage deliberately did NOT change: executor behavior (task generation,
layout, foreman, market handling — additive diagnostics keys only), opening
playback/provenance semantics, PPO/recurrence/value heads, affordability
clipping of plans (coherence is measured, never enforced), or the JAX V1
port (`bc_manager_jax` remains a separate guide).

---

## 5. Exact Kaggle runbook

Verified against the actual parsers at commit `fc95752`
(`python -m bc_manager.cli --help`, `python -m bc_manager.ablation --help`).
Defaults quoted from `--help`; explicit flags are shown where the packet
requires them to be identical across variants.

### Cell 1 — get the code (fresh clone; avoid fetch-in-place)

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient  # Git auth via Kaggle secrets
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(
    ["git", "clone",
     f"https://{token}@github.com/BillXu21/Kaggriculture.git",
     "/kaggle/working/Kaggriculture"],
    check=True)
subprocess.run(["git", "-C", "/kaggle/working/Kaggriculture", "log",
                "--oneline", "-3"], check=True)
os.chdir("/kaggle/working/Kaggriculture")
```

Record the printed HEAD; it must be `fc95752` or a descendant on `main`.

### Cell 2 — exact dependencies

```bash
cd /kaggle/working/Kaggriculture
python -m pip install -q -r requirements.txt          # pyarrow/numpy/torch
python -m pip install -q kaggle-environments==1.32.7  # provenance-pinned
python -c "import kaggle_environments, torch; \
print(kaggle_environments.__version__, torch.__version__)"
```

The first printed version must be exactly `1.32.7`; the panel refuses to run
behind any other engine (provenance guard).

### Cell 3 — locate the canonical corpus (fail loudly if absent)

The private dataset is mounted at
`/kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327` (some
notebook attachments expose it as `/kaggle/input/<slug>` instead — check
both). Shell wildcards must be expanded explicitly so the CLI never receives
a literal `*`:

```bash
for d in \
  /kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327 \
  /kaggle/input/kaggriculture-canonical-daily-1327; do
  if [ -d "$d" ]; then DATA_DIR="$d"; fi
done
[ -n "$DATA_DIR" ] || { echo "FATAL: corpus dataset not mounted"; exit 1; }
ls -l "$DATA_DIR"
# Explicit glob expansion; aborts on zero matches:
PARQUETS=()
for f in "$DATA_DIR"/*.parquet; do
  [ -e "$f" ] || { echo "FATAL: no .parquet under $DATA_DIR"; exit 1; }
  PARQUETS+=("$f")
done
echo "Found ${#PARQUETS[@]} partitions:"; printf '%s\n' "${PARQUETS[@]}"
```

Expect the five 2026-08-17..2026-08-21 schema-v3 partitions
(`research/FIVE_DAY_V3_CORPUS.md`).

### Cell 4 — train all four variants in one identical matrix

Identical date split, elite cutoff, seed, optimizer, loss schedule, epochs,
and batch for every variant — only `--variant` and `--checkpoint-dir`
change. Defaults confirmed from `--help`: train dates 2026-08-17..20,
validation 2026-08-21, min_score 2950, d-model 128 / layers 4 / heads 4 /
ffn 384 / dropout 0.1, AdamW lr 3e-4 weight-decay 1e-2, batch 256, 30
epochs, gradient clip 1.0, AMP auto (CUDA only).

**V0 reuse policy:** the historical real V0 checkpoint
(`/kaggle/working/bc-v0-score2950/best.pt`, trained at commit `692bca5`)
predates the variant matrix. Its recipe matches the current defaults, but
for strict same-matrix fairness this runbook **trains V0 fresh alongside
J/E/JE with byte-identical flags**. Reusing the old checkpoint is acceptable
only if its stored config equals these defaults AND its
`validation_metrics.total` is comparable; when in doubt, train fresh (the
cost is ~4 minutes on GPU per the reference run).

```bash
cd /kaggle/working/Kaggriculture
for V in V0 J E JE; do
  python -m bc_manager.cli \
    --variant "$V" \
    --train-dates 2026-08-17,2026-08-18,2026-08-19,2026-08-20 \
    --val-dates 2026-08-21 \
    --min-score 2950 \
    --seed 0 \
    --lr 3e-4 --weight-decay 1e-2 --batch-size 256 --epochs 30 \
    --gradient-clip 1.0 \
    --device auto --amp auto \
    --checkpoint-dir "/kaggle/working/bc-v1-$V" \
    "${PARQUETS[@]}" 2>&1 | tee "/kaggle/working/train_$V.log"
done
```

Each directory receives atomic `best.pt` / `last.pt` checkpoints carrying
`model_config`, `model_variant`, and `validation_metrics`.

### Cell 5 — teacher-forced prerequisite inspection + strict preflight

```bash
cd /kaggle/working/Kaggriculture
python - <<'PY'
import torch
for v in ("V0", "J", "E", "JE"):
    p = torch.load(f"/kaggle/working/bc-v1-{v}/best.pt", map_location="cpu",
                   weights_only=False)
    print(v, "variant=", p.get("model_variant"),
          "val_total=", p["validation_metrics"]["total"])
PY
# Strict preflight: rejects smoke weights, variant mismatches, missing
# metrics; imports no engine. Exit 0 required before the panel.
python -m bc_manager.ablation \
  --checkpoint V0=/kaggle/working/bc-v1-V0/best.pt \
  --checkpoint J=/kaggle/working/bc-v1-J/best.pt \
  --checkpoint E=/kaggle/working/bc-v1-E/best.pt \
  --checkpoint JE=/kaggle/working/bc-v1-JE/best.pt \
  --validate-only --out /kaggle/working/bc_v1_ablation_preflight.json
cat /kaggle/working/bc_v1_ablation_preflight.json
```

Expected preflight artifact: `"status": "validated"`,
`"expected_games": 40`, all four checkpoints listed.

### Cell 6 — the real closed-loop panel (primary command)

Fixed design made explicit where the parser supports it (`--opening` only
accepts `standard_mixed`; seeds/seats default to the protocol values and are
written out anyway):

```bash
cd /kaggle/working/Kaggriculture
python -m bc_manager.ablation \
  --checkpoint V0=/kaggle/working/bc-v1-V0/best.pt \
  --checkpoint J=/kaggle/working/bc-v1-J/best.pt \
  --checkpoint E=/kaggle/working/bc-v1-E/best.pt \
  --checkpoint JE=/kaggle/working/bc-v1-JE/best.pt \
  --opening standard_mixed \
  --seeds 7 17 42 123 2026 \
  --seats 0 1 \
  --device cpu \
  --out /kaggle/working/bc_v1_ablation_panel.json
echo "exit=$?"
```

40 games × 720 steps; runtime is dominated by the official interpreter.

### Cell 7 — artifact inspection and pass criteria

```bash
python - <<'PY'
import json
p = json.load(open("/kaggle/working/bc_v1_ablation_panel.json"))
assert p["status"] == "complete", p["status"]
games = p["games"]
assert len(games) == 40, len(games)
from collections import Counter
per = Counter(g["variant"] for g in games)
assert set(per.values()) == {10}, per
for g in games:
    assert g["passed"], g
    assert not g["opening_diagnostics"]["divergence"]["occurred"], g
    assert g["fallback_errors"] == [] if "fallback_errors" in g else True
    assert not g["status_anomalies"], g
print(json.dumps(p["ranking"], indent=2))
print("seed17 collapse flags:",
      {v: p["variants"][v]["seed17"] for v in p["variants"]})
print("seed2026 retention:", {v: p["variants"][v].get("seed2026")
                              for v in p["variants"]})
PY
```

Pass criteria, in order:

1. `status == "complete"` (any divergence/fallback/anomaly ⇒ `failed`,
   exit 1 — investigate before interpreting anything);
2. 40 games, exactly 10 per variant, official provenance accepted silently
   (the guard aborts otherwise);
3. every game passed: 96 scripted turns, zero opening divergence/fallback,
   clean handoff, zero status anomalies;
4. each checkpoint's teacher-forced `validation_metrics.total` present
   (prerequisite evidence, reported beside the ranking);
5. **selection = closed-loop final-bank median, then mean.** Do NOT promote
   based on teacher-forced totals or coherence rates alone; report the
   seed-17 collapse flags and seed-2026 retention next to whatever leads.

### Cell 8 — optional archive/download

```bash
cd /kaggle/working
tar czf bc_v1_ablation_artifacts.tgz \
  bc-v1-V0/best.pt bc-v1-J/best.pt bc-v1-E/best.pt bc-v1-JE/best.pt \
  bc_v1_ablation_preflight.json bc_v1_ablation_panel.json \
  train_V0.log train_J.log train_E.log train_JE.log
ls -l bc_v1_ablation_artifacts.tgz
sha256sum bc_v1_ablation_artifacts.tgz
```

Download via the notebook output panel; record the SHA-256 with the results.

### After the run

Record actual results (or failures) in `HISTORY.md` and update
`CURRENT_STATE.md` before any follow-up experiment, per the compute-safety
contract. Until that run exists, no variant ranking, winner, or promotion
claim may be written anywhere in this repository.
