# Stage 2.5 champion and panel workflow

The champion registry keeps the immutable BC anchor, one current global
champion, and a configurable number of previous champions. Dual PPO training
continues to train A against B. Panel evaluation consumes inference snapshots
and does not create PPO state or trajectory artifacts.

## Initialize the registry

```powershell
python -m rl_manager.stage25_champion init `
  --registry artifacts/local/champion_registry.json `
  --bc-anchor artifacts/local/bc-v1-E/best.npz `
  --history-depth 3
```

The registry begins with the BC snapshot as both `bc_anchor` and
`current_champion`, and an empty history. The anchor record is never changed by
promotion. Registry JSON writes use a temporary file in the destination folder
and `os.replace`; checkpoint files are not removed when their history
references age out.

## Export and evaluate A/B

Export accepts only the native dual PPO checkpoint payload. It writes one
native Stage 2.5 inference checkpoint with the selected policy's exact
parameters, behavior identity, model and observation/action contract,
E-history and curriculum identity, source checkpoint digest, generation, and
A/B label. Optimizer, RNG, and training state are not copied.

```powershell
python -m rl_manager.stage25_champion export `
  --dual-checkpoint runs/dual/latest.npz --policy A `
  --output runs/dual/generation_20_A.npz
python -m rl_manager.stage25_champion export `
  --dual-checkpoint runs/dual/latest.npz --policy B `
  --output runs/dual/generation_20_B.npz
```

The same strict export seam is available to Python callers:

```python
from rl_manager.stage25_checkpoint import export_stage25_dual_policy_snapshot

export_stage25_dual_policy_snapshot(
    "runs/dual/latest.npz",
    "runs/dual/generation_20_A.npz",
    policy="A",
)
```

Then evaluate either exported snapshot:

```powershell
python -m rl_manager.stage25_panel_eval `
  --candidate runs/dual/generation_20_A.npz `
  --registry artifacts/local/champion_registry.json `
  --games-per-opponent 88 --seed 2026 `
  --engine fast --executor strip --opening standard_mixed `
  --workers 88 --envs-per-worker 4 `
  --physical-batch-size 32 --inference-batch-wait-ms 20
```

Workers, environments per worker, executor, engine, opening, inference batch,
and wait interval are configurable. Evaluation defaults to 88 games per
opponent, 2 workers, 1 environment per worker, FastEnv, strip executor,
`standard_mixed`, physical batch 32, and 20 ms wait. Game counts must be even.
Every artifact contains candidate and registry identities, runtime provenance,
assignments, W/L/T and final-bank metrics per unique opponent and in aggregate,
seat breakdown, timestamp, and source revision. Logical panel roles sharing an
identical parameter fingerprint are evaluated once and listed together.

Seed derivation is:

```text
payload = UTF8("stage25-panel-v1|<base-seed>|<candidate-snapshot-id>|<opponent-snapshot-id>|<pair-index>")
pair_seed = little_endian_uint32(SHA256(payload)[0:4]) mod (2^31 - 1)
```

The two seat orientations for each pair share `pair_seed`; candidate seat 0 is
scheduled first, then candidate seat 1. Episode indices follow the stable
registry panel order and pair order. Worker assignment is left to the parallel
runner after this schedule is fixed and does not enter seed derivation.

## Inspect and manually promote

Evaluation never promotes automatically. Review the JSON artifact, then, if a
candidate is selected, run:

```powershell
python -m rl_manager.stage25_champion promote `
  --registry artifacts/local/champion_registry.json `
  --candidate runs/dual/generation_20_A.npz `
  --evaluation panel-evaluations/panel_eval_<id>.json
```

An explicit optional policy JSON can add minimum win fractions for logical
roles such as `current_champion`, `recent_champion_1`, `recent_champion_2`, or
`bc_anchor`:

```json
{
  "schema_version": "stage25_promotion_policy_v1",
  "minimum_win_fraction": {
    "current_champion": null,
    "recent_champion_1": null,
    "recent_champion_2": null,
    "bc_anchor": null
  }
}
```

Null thresholds impose no gate. No threshold is selected by default. Manual
promotion verifies the candidate fingerprint and file identity against the
evaluation, the registry ID/version/state and champion lineage, snapshot
contracts, and the BC anchor file and parameters. It appends a lineage event,
rotates the current champion into bounded history, and atomically replaces the
registry JSON. It never selects A or B on the caller's behalf.
