# TPU / PPO results — 2026-08-29

Base integration SHA: `45f88001b6cc14f802f10668179a68f6fe3c2bf5` (`throughput/integration`).

This note records the TPU throughput, PPO stability, LR sweep, and long-run results gathered after the #18/#21/#22/#23 integration.

## Multi-trainer TPU scaling

The independent-trainer benchmark placed one trainer on each requested TPU device and compared sequential vs concurrent dispatch.

| trainers | inference rows/s | sequential trainers/s | concurrent trainers/s | same-run speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 123,359.7 | 75.47 | 74.54 | 0.99x |
| 2 | 242,215.7 | 78.07 | 123.77 | 1.59x |
| 4 | 475,042.5 | 77.51 | 168.32 | 2.17x |

Explicit device assignment was observed for TPU devices 0..N-1. Sequential throughput staying nearly flat is expected and is useful evidence that the benchmark was not accidentally counting device visibility as scaling. Inference scaled almost linearly through N=4; PPO update throughput scaled materially but sublinearly.

N=8 was not measured in this run and should not be inferred from device visibility.

## PPO safety-policy change

A first three-update smoke used `lr=1e-4`, `target_kl=0.03`, and `reject_update_kl=0.12`. All three saved checkpoints had `step=0`, proving the hard KL reject returned the exact prior train state each time.

Decision: do not hard-reject potentially useful high-KL policies by default. Use target-KL as an epoch early-stop while accepting the completed epoch, then reduce LR manually/adaptively if needed.

Production tuning baseline became:

- 384 games/update
- 96 rollout workers
- `N=1` env/worker
- mixed-day policy batching
- fixed inference batch 24
- 30 ms inference wait
- PPO minibatch 256
- 2 max epochs
- `lr=1e-4`
- KL-to-frozen coefficient `0.001`
- `target_kl=0.08`
- no `reject_update_kl`

## Three-update accepted smoke

With hard rejection removed, a fresh 3-update run produced this 64-game holdout against frozen BC-E:

- W/L/T: 61/3/0
- candidate mean bank: 43,467.8
- candidate median: 45,464.5
- candidate p10: 36,016.2
- candidate min: 1,500
- mean margin: +29,434.8
- no candidate game below 1k
- 2/64 candidate games below 10k

This established that the integrated PPO lifecycle can materially improve the BC-E policy in only a few updates.

## LR sweep

Three-update branches from the same BC-E initialization, all with target-KL 0.08 and no hard reject:

| LR | W-L | mean bank | median | p10 | checkpoint step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5e-5 | 33-31 | 22,271 | 20,129 | 6,258 | 117 |
| 1e-4 | 61-3 | 43,468 | 45,465 | 36,016 | not copied in the experiment log |
| 2e-4 | 38-26 | 19,872 | 18,370 | 2,394 | 117 |

The response was strongly non-monotonic; `1e-4` was the clear local sweet spot.

For the 5e-5 and 2e-4 runs, step 117 after three updates means 39 minibatches/update exactly, so target-KL stopped each update after the first epoch while preserving that epoch.

## 80-update stationary-BC run

A long run at `lr=1e-4` completed all 80 requested updates and retained every numbered checkpoint.

Checkpoint progression:

- update 0: step 39
- update 1: step 78
- update 2: step 117
- final update 79: step 4,797

One epoch is 39 minibatches and two epochs are 78. Therefore 4,797 / 39 = 123 accepted epochs across 80 updates:

- 37 updates stopped after epoch 1
- 43 updates completed both epochs
- 0 updates were hard-rejected

The final 20 step deltas were mostly 78 with one 39, showing target-KL became less restrictive later in training.

The run is valuable primarily because all 80 checkpoints were preserved; checkpoint selection should be based on holdout/economic evaluation rather than assuming the final update is best.

## Previous 4-hour run context

Before the integration fixes, a stationary-BC run reached a best measured evaluation near update 95:

- mean bank 53,874
- median 56,225
- 64-0 on that evaluation panel

It later collapsed badly by update 135. The promotion gate never fired because benign opening diagnostics were counted as fatal anomalies. This remains the strongest evidence that PPO can find materially better economic policies but is unstable without retention/promotion discipline.

## Current conclusions

1. Independent TPU trainers can use multiple TPU devices concurrently; N=4 is already useful.
2. Hard KL rollback was too conservative for this project. Target-KL early-stop + checkpoint retention is preferable.
3. `lr=1e-4` is substantially better than 5e-5 and 2e-4 in the short sweep.
4. PPO can produce a large economic improvement in only a few accepted updates.
5. Long stationary-BC training remains unstable, so every checkpoint should be retained and evaluated rather than trusting the terminal policy.
6. The next algorithmic work should version/fix the E economic-context runner semantics before new training, then move to economic terminal rewards and live/snapshot self-play.
