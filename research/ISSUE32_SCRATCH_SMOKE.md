# Issue #32 Scratch PPO Smoke

This is a disposable manager-level smoke only. It uses the legacy BC-E model
as the frozen stationary opponent and frozen sell-quantity reference. It does
not implement the Issue #32 economic curriculum, opponent inputs, promotion
ratcheting, new rewards, sell learning, self-play, or executor changes.

Run from the repository root in a Kaggle foreground cell after replacing the
input attachment path if needed:

```bash
python -m rl_manager.cli train \
  --e-checkpoint /kaggle/input/bc-e-legacy/best.pt \
  --e-history-version E_LEGACY \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast --master-seed 32032 \
  --init-mode scratch \
  --num-workers 96 --num-envs 1 --num-threads 1 \
  --low-telemetry --read-only-agent-observations --batch-backend \
  --inference-batch-scope policy --fixed-inference-batch-size 32 \
  --inference-batch-wait-ms 2 \
  --episodes-per-update 384 --updates 10 \
  --epochs 2 --minibatch-size 256 --lr 1e-4 \
  --kl-to-frozen-coef 0 --target-kl 0.08 \
  --promotion-every 0 \
  --output-dir /kaggle/working/issue32-scratch-ppo \
  --checkpoint /kaggle/working/issue32-scratch-ppo/ppo_update_000009.npz
```

The plan has 384 * 26 = 9984 expected candidate rows, divisible by the 256
row minibatch. Stop the run if corrected BC data is ready; do not treat this
disposable run as promotion evidence.

Run a small deterministic paired smoke evaluation against the frozen legacy
BC checkpoint after training. The fixed `smoke` set is only `(17, 42, 2026)`;
it does not consume the reserved dev, holdout, or promotion seed blocks:

```bash
python -m rl_manager.cli eval \
  --checkpoint /kaggle/working/issue32-scratch-ppo/ppo_update_000009.npz \
  --e-checkpoint /kaggle/input/bc-e-legacy/best.pt \
  --e-history-version E_LEGACY \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast --num-workers 1 --num-envs 1 --num-threads 1 \
  --low-telemetry --read-only-agent-observations --batch-backend \
  --seed-set smoke \
  --output-json /kaggle/working/issue32-scratch-ppo/eval-smoke.json
```
