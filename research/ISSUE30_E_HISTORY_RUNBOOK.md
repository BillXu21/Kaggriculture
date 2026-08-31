# Issue #30 E History Runbook

Status: implemented and locally validated; no BC retraining or PPO training
was started by this change.

## Contract

`E_CORRECTED_V1` is the default E contract. Channels 12 and 13 use the
observed adjacent daily-start bank delta. The first manager decision at day 4
is invalid (`0/0`); day 5 may use the realized day-4 start. Gaps, backwards
days, and episode boundaries are invalid. Submitted market orders are never
used as fill evidence.

`E_LEGACY` reproduces the old runner behavior, where the current daily start
was supplied as the previous start and the history channels are therefore
deterministically `0/0`. It is accepted only through an explicit legacy
loader, CLI flag, or legacy submission builder. Corrected loaders reject a
legacy E checkpoint before model execution.

## Regenerate Canonical Data

Run from a clean checkout with the raw 1.32.7 replay files mounted. Process
each partition from raw JSON; do not migrate old processed Parquet files.

```bash
cd /kaggle/working/Kaggriculture
mkdir -p /kaggle/working/canonical-v3
python -m replay_daily extract \
  --input /kaggle/input/kaggriculture-raw-replays/2026-08-17 \
  --output /kaggle/working/canonical-v3/2026-08-17.parquet \
  --format parquet \
  --source-dataset kaggriculture-raw-replays \
  --partition-date 2026-08-17 \
  --on-version-mismatch fail
```

Repeat the command for each raw partition date. Confirm that every output
reports the expected record count and that `replay_daily` accepts schema v3.

## Corrected BC-E Training

The explicit flag below prevents an operator from accidentally selecting the
legacy feature contract. The training adapter passes `manager_start_day=4`
and writes `e_history_version` into `best.pt` and `last.pt`.

```bash
cd /kaggle/working/Kaggriculture
python -m bc_manager.cli \
  /kaggle/working/canonical-v3/2026-08-17.parquet \
  /kaggle/working/canonical-v3/2026-08-18.parquet \
  /kaggle/working/canonical-v3/2026-08-19.parquet \
  /kaggle/working/canonical-v3/2026-08-20.parquet \
  /kaggle/working/canonical-v3/2026-08-21.parquet \
  --variant E \
  --e-history-version E_CORRECTED_V1 \
  --train-dates 2026-08-17,2026-08-18,2026-08-19,2026-08-20 \
  --val-dates 2026-08-21 \
  --min-score 2950 \
  --device cuda --amp on \
  --checkpoint-dir /kaggle/working/bc-v1-E-corrected
```

Do not use an old unversioned E checkpoint for corrected evaluation. Load it
only with the explicit `E_LEGACY` compatibility path, or build it with
`tools/build_runner_compatible_submission.py`.

## PPO Readiness, No Training

These checks validate the corrected seams without starting a PPO run:

```bash
cd /kaggle/working/Kaggriculture
python -m compileall -q bc_manager bc_manager_jax executor_v0 evaluation rl_manager tools
python -m pytest tests/test_rl_manager_economics.py \
  tests/test_rl_manager_trajectory.py \
  tests/test_rl_manager_ppo_checkpoint.py \
  tests/test_rl_manager_parallel.py -q
python -m rl_manager.cli train --help
python -m rl_manager.cli eval --help
```

The eventual foreground PPO command must select the corrected checkpoint and
history identity explicitly. It is recorded here for readiness only and was
not run:

```bash
python -m rl_manager.cli train \
  --e-checkpoint /kaggle/working/bc-v1-E-corrected/best.pt \
  --e-history-version E_CORRECTED_V1 \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast --master-seed 17 \
  --num-workers 1 --num-envs 1 --num-threads 1 \
  --episodes-per-update 8 --updates 1 \
  --output-dir /kaggle/working/ppo-issue30 \
  --checkpoint /kaggle/working/ppo-issue30/ppo.npz
```
