# Issue20 submission parity incident — 2026-09-01

## Summary

Issue20 current-v-current economic PPO looked healthy in the runner but collapsed in Kaggle submissions. Deterministic fast self-play for u10 was roughly 33k mean bank with a ~19.5k minimum, while live submission replays fell to near-zero/low-thousands.

This was not normal policy variance and was not primarily an executor watering failure. The submitted Torch policy was not semantically equivalent to the JAX PPO policy used for training/evaluation.

## Root cause 1: PPO sell quantity is a full frozen-network branch

The Issue20 PPO policy has a split deployment contract:

- mutable PPO network: crop, animal, land, fertilizer, care, sell presence;
- immutable frozen BC-E network: `sell_quantity_log1p`.

The PPO implementation computes frozen sell quantity by forwarding the input through the **entire frozen network**. The mutable base's sell-quantity head is optimizer-masked and receives no update.

The first Torch exporter/submission incorrectly represented this as one network: mutable PPO trunk + old frozen sell-quantity head. Once PPO changed the trunk, this was no longer equivalent to the frozen BC network. On a real d4 observation, all non-quantity plan fields matched JAX, but Torch sell quantities collapsed toward zero.

Direct check on d4 seed0:

- JAX training-semantics quantity included values such as 5, 10, 16, 11;
- incorrect single-network Torch was mostly 0/1;
- dual Torch (mutable forward + full frozen forward for quantity) matched JAX exactly at the decoded-plan level.

Observed final check:

```text
EXACT PLAN PARITY: True
```

## Root cause 2: opening -> manager labor history handoff

Training owns `previous_execution` across the deterministic opening. The first d4 manager request therefore receives realized d3 labor.

The deployed executor is hidden behind `opening_book` during d0..d3, so its first call at d4 previously began with zero labor history.

Audited standard opening handoff:

```text
training d4 previous_execution:
workers_hired = 5
hire_cost = 12

deployment before fix:
workers_hired = 0
hire_cost = 0
```

This changed the d4 JAX plan. The generated Issue20 submission now seeds `{workers_hired: 5, hire_cost: 12}` on the first clean d4 manager call.

## What was ruled out

### Deterministic argmax collapse

Deterministic evaluation was stronger than stochastic evaluation, not weaker. u10 deterministic self-self was approximately:

```text
mean 33180
median 34289
p10 22917
min 19496
```

So Kaggle banks around 0-2k were not an argmax-policy problem.

### E_LEGACY cash-history mismatch in this incident

Under `E_LEGACY`, previous-day net-cash channels are intentionally zero/invalid. Passing the current `(day, money)` pair reproduces the legacy runner semantics and was not the sell-collapse cause.

### Opening itself

Live replay inspection showed the standard opening executing correctly through d3. The visible failure began at the d4/d5 learned-manager handoff.

## Submission contract after fix

Issue20 PPO submissions are explicitly dual-network:

```text
observation
    |
encode_live_inputs
    |
    +---------------------+
    |                     |
mutable PPO network    frozen BC-E network
    |                     |
all learned heads      sell_quantity_log1p
except quantity            |
    +----------+----------+
               |
        decode_daily_plan
               |
            executor
```

Archive members include:

```text
best.pt       # mutable PPO network export
frozen_e.pt   # immutable BC-E network
main.py
submission_manifest.json
```

The manifest must declare:

```json
{
  "policy_contract": "ppo_frozen_sell_quantity",
  "opening_handoff_previous_execution": {
    "workers_hired": 5,
    "hire_cost": 12
  }
}
```

The builder now requires an explicit `--policy-contract`. Selecting `ppo_frozen_sell_quantity` without `--frozen-checkpoint` fails instead of silently creating a one-network archive.

## Mandatory pre-submit checks

For Issue20-style PPO, do not submit an archive based only on successful serialization or parameter round-trip.

Run the parity audit after Torch export:

```bash
python tools/audit_issue20_ppo_submission.py \
  --ppo-checkpoint /path/to/final.npz \
  --mutable-torch /path/to/ppo.pt \
  --frozen-torch /path/to/frozen_bc_e.pt \
  --archive /path/to/submission.tar.gz
```

The audit fails closed unless:

1. mutable JAX and mutable Torch raw outputs match for every head;
2. frozen JAX and frozen Torch raw outputs match for every head;
3. the combined dual-network outputs match;
4. the archive contains both checkpoints and declares the correct contract;
5. checkpoint hashes in the archive/manifest match the supplied artifacts;
6. the d4 opening labor handoff is present.

Then run at least one local official 1.32.7 replay from a **fresh Python process**. Notebook-local tests can reuse an already-imported repo `executor_v0` module and therefore resolve checkpoint paths against `/kaggle/working/Kaggriculture` instead of the extracted archive. A fresh process avoids that import-cache contamination and better matches Kaggle execution.

Expected smoke behavior for the corrected u10 archive is tens of thousands of final bank, not hundreds/low-thousands.

## Training-state note

The full resumable PPO artifact is the `.npz` checkpoint, for example:

```text
/kaggle/working/issue20-scratch-selfplay-full/ppo_update_000009.npz
```

Torch `.pt` exports are deployment artifacts and do not contain the full optimizer/RNG training state.

## Stage 2.5 follow-up

The current manager uses absolute crop/animal targets. Replay inspection showed a viable but odd local strategy that dug up crops early and went heavily into animals, including over-requesting goose structures while also buying land.

For a later Stage 2.5 experiment, consider action representations with a stable neutral action (`delta=0`) and initially disable destructive crop removal. Do not mix that redesign into the current continuation run; first measure whether more self-play can escape or improve the current strategy under the unchanged action space.
