# Issue 20: Current-v-Current Economic Smoke

This bounded Phase-A mode freezes one live learner snapshot for collection,
uses it at both seats, collects both seats' 26 manager rows, and performs one
PPO update only after every game in the update is complete. It uses terminal
own-bank rewards and does not use the frozen BC-E as a rollout opponent.

## Kaggle TPU command

```bash
python -m rl_manager.cli train \
  --e-checkpoint /kaggle/input/bc-e-legacy/best.pt \
  --e-history-version E_LEGACY \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast \
  --master-seed 32032 \
  --init-mode scratch \
  --training-composition current_vs_current_economic \
  --reward-mode terminal_own_bank \
  --bank-reward-baseline 3000 \
  --bank-reward-scale 50000 \
  --num-workers 96 \
  --num-envs 1 \
  --num-threads 1 \
  --low-telemetry \
  --read-only-agent-observations \
  --batch-backend \
  --inference-batch-scope policy \
  --fixed-inference-batch-size 32 \
  --inference-batch-wait-ms 2 \
  --episodes-per-update 384 \
  --updates 10 \
  --epochs 2 \
  --minibatch-size 256 \
  --lr 1e-4 \
  --kl-to-frozen-coef 0 \
  --target-kl 0.08 \
  --promotion-every 0 \
  --output-dir /kaggle/working/issue20-scratch-selfplay \
  --checkpoint /kaggle/working/issue20-scratch-selfplay/final.npz
```

Each complete game contributes `26 * 2 = 52` trainable rows. Therefore 384
games produce `384 * 52 = 19,968` rows per update, which is divisible by 256.
The plan and every PPO checkpoint record the composition and explicit reward
configuration. Per-update `economic_update_*.json` files retain aggregate
metrics plus per-episode purchase events and utilization snapshots.

## Separate smoke evaluation

After the scratch run, evaluate the resulting checkpoint against the frozen
legacy BC-E using the existing deterministic smoke seed set. This is separate
from self-play collection:

```bash
python -m rl_manager.cli eval \
  --checkpoint /kaggle/working/issue20-scratch-selfplay/final.npz \
  --e-checkpoint /kaggle/input/bc-e-legacy/best.pt \
  --e-history-version E_LEGACY \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast \
  --seed-set smoke \
  --output-json /kaggle/working/issue20-scratch-selfplay/eval_smoke.json
```
