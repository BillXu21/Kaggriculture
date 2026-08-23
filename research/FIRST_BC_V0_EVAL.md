# First Full BC V0 Run and Held-Out Evaluation

Date: 2026-08-22

This note records the first full behavior-cloning run over the five-day canonical schema-v3 corpus and the held-out Aug-21 evaluation that followed it.

## Provenance

- Code: `692bca50e8ba0b687e48fd970e67bbe17014f03f`
- Canonical schema: v3
- Engine/module version in corpus: 1.32.7
- Kaggle dataset mount: `/kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327`
- Train dates: 2026-08-17 through 2026-08-20
- Validation date: 2026-08-21
- Score filter: `min_score >= 2950`
- Train rows: 25,500
- Validation rows: 5,700
- Model: D-019 default tile Transformer, 1,071,040 trainable parameters
- Opponent-public board: disabled
- Device: CUDA with AMP
- Checkpoint output during the Kaggle run: `/kaggle/working/bc-v0-score2950`

## Training configuration

- `d_model=128`
- 4 Transformer layers
- 4 attention heads
- FFN 384
- dropout 0.1
- batch size 256
- AdamW, lr 3e-4
- weight decay 1e-2
- gradient clip 1.0
- 30 epochs
- early-stopping patience 6
- seed 0
- seven loss groups with weight 1.0 each

No scheduler, opponent board, loss reweighting, larger model, or other tuning was introduced before this reference run.

## Training result

The run completed all 30 epochs in ~237 seconds. Training examples/s stabilized around 3.65k.

- epoch 1: train total 10.1845, validation total 6.7244
- best epoch: 29
- best validation total: 2.8889
- epoch 30 validation total: 2.9348
- epoch 30 train total: 1.1194

The held-out loss continued improving through late training rather than diverging early. The train/validation gap is real but does not look like immediate memorization collapse.

At epoch 29 the important headline diagnostics were approximately:

- crop elementwise exact accuracy: 0.713
- goose nonzero recall: 0.968
- land exact accuracy: 0.991
- sell-presence accuracy: 0.939

## Day-only baseline comparison

The train-only empirical day baseline is the important control because the elite corpus contains repeated strategy families. On held-out Aug-21, the state-aware Transformer beat it materially across every manager group measured.

| group | metric | model | day baseline |
| --- | --- | ---: | ---: |
| crop | exact accuracy | 0.7128 | 0.4752 |
| crop | exact whole-vector match | 0.2930 | 0.0598 |
| crop | MAE | 1.2731 | 3.6217 |
| crop | nonzero recall | 0.9555 | 0.8443 |
| animal | exact accuracy | 0.8267 | 0.4540 |
| animal | exact whole-vector match | 0.6619 | 0.1389 |
| animal | MAE | 0.2681 | 1.6936 |
| animal | nonzero recall | 0.9958 | 0.9201 |
| fertilizer | exact accuracy | 0.9122 | 0.8818 |
| fertilizer | exact whole-vector match | 0.6582 | 0.5323 |
| fertilizer | MAE | 0.3253 | 0.5980 |
| fertilizer | nonzero recall | 0.7522 | 0.4557 |
| CARE | exact accuracy | 0.8019 | 0.4661 |
| CARE | exact whole-vector match | 0.5998 | 0.1754 |
| CARE | MAE | 0.3296 | 1.6676 |
| CARE | nonzero recall | 0.9818 | 0.9136 |
| land | exact accuracy | 0.9912 | 0.9089 |
| land | MAE | 0.0088 | 0.0911 |
| sell | presence accuracy | 0.9394 | 0.8923 |

This is strong evidence that the model is using state, not merely memorizing a calendar/day template.

## Rare / shifted behavior

Aug-21 differs meaningfully from the train dates, especially in goose usage and some situational crops/fertilizer. The model handled several branches that the day baseline missed entirely.

Selected held-out nonzero recalls:

- crop CARROT: model 75.7%, baseline 57.9%
- crop TOMATO: model 83.8%, baseline 0.0%
- animal GOOSE: model 96.8%, baseline 0.0%
- fertilizer WHEAT: model 60.4%, baseline 0.0%
- fertilizer CARROT: model 33.7%, baseline 0.0%
- fertilizer TOMATO: model 14.0%, baseline 0.0%
- fertilizer STRAWBERRY: model 85.6%, baseline 66.6%
- fertilizer MELON: model 58.1%, baseline 0.0%
- CARE GOOSE: model 95.5%, baseline 0.0%

The rare fertilizer heads remain an obvious weak area, especially tomato/carrot, but they are not a blocker for the first closed-loop test.

## Selling diagnostics

Held-out Aug-21 selling remains conservative rather than solved:

- true sell-presence rate: 0.1121
- predicted sell-presence rate: 0.0938
- sell-presence nonzero recall: 0.6484
- sell-presence accuracy: 0.9394
- positive-cell quantity log-MAE: 0.3529

Because sell cells are sparse, presence accuracy alone is flattering; recall matters. The model still materially beats the day-only presence baseline, but closed-loop execution should treat selling as an area to inspect rather than assume solved.

## Interpretation

The first D-019 representation experiment passes its intended diagnostic:

1. the model learns the demonstrations;
2. held-out Aug-21 performance improves throughout training;
3. the state-aware model materially beats the train-only day baseline;
4. rare state-conditioned branches such as tomato and goose do not simply collapse to zero.

Do not respond by immediately scaling the network, adding opponent-board features, reweighting sparse losses, or running broad hyperparameter sweeps. The dominant uncertainty has moved from teacher-forced BC to closed-loop execution.

## Next gate: closed-loop V0

The next meaningful evaluation is an actual 1.32.7 game loop using `best.pt` plus a deterministic executor/foreman.

Initial success criteria are diagnostic rather than leaderboard-oriented:

- model called once per day and outputs decoded consistently;
- executor completes games without illegal-action cascades or deadlock;
- crop and animal states track requested targets reasonably;
- watering/feed/harvest/Care maintenance stays mechanically healthy;
- six-bin sell intents reach primitive market actions;
- paired fixed-seed, seat-swapped games can be run against one frozen competent opponent;
- manager intent compliance is logged separately from final bank / W-L outcome.

Useful executor-compliance metrics:

- crop target MAE at next day boundary;
- animal target MAE;
- land target hit rate;
- fertilizer requested/completed;
- CARE requested/completed;
- sell intent/submitted;
- missed water/feed/maintenance;
- illegal/ineffective actions;
- idle worker turns;
- unfinished tasks at day end;
- final banks and paired W/L/T.

If compliance is poor, improve execution before blaming the learned manager. If compliance is high and economic trajectories are still poor, then revisit manager targets/representation before adding PPO complexity.

Detailed executor draft and its own deferred-optimization list now live in `research/EXECUTOR_V0_PLAN.md`.

## Deliberate BC V0 simplifications and revisit backlog

Earlier D-019 notes already recorded many of these choices under "explicitly deferred" and "alternatives considered," but they were spread across decision/implementation documents. This section is the explicit backlog so later work does not mistake V0 shortcuts for settled optimal choices.

- **Stateless once-per-day policy.** No recurrence, previous hidden state, or explicit multi-day temporal model. Revisit if closed-loop errors depend on history not reconstructable from the current state or if PPO needs temporal credit/state.
- **One manager call per day.** Only selling retains six intra-day bins. Revisit if strategically important crop/animal/resource decisions repeatedly need same-day reaction.
- **Opponent-public board disabled in the first trained model.** The canonical data preserves it and the model supports it, but V0 did not use it. Revisit after the own-state closed loop works, especially for shared-market/opponent-production adaptation.
- **No opponent-private inference.** Hidden shed/seeds/inventory are not estimated. Revisit only after public-state conditioning is proven useful and partial observability is an identified bottleneck.
- **Absolute count targets rather than deltas or per-tile actions.** Crop/animal/land outputs are resulting counts; fertilizer/CARE are type-level daily totals. Revisit if the executor repeatedly faces ambiguous realization choices or if minimum-change deltas train/execute better.
- **No tile-specific fertilizer/CARE control.** Exact target selection is delegated to the executor. Revisit if lifecycle heterogeneity makes type-level counts insufficient.
- **Factorized output heads.** Crop, animal, land, fertilizer, CARE, sell-presence, and sell-quantity heads share the manager representation but do not enforce a joint feasibility distribution. Revisit if inconsistent combinations are common in closed loop.
- **Fixed count vocabulary 0..100.** Convenient for current targets rather than a claim that categorical count prediction is optimal. Revisit if count resolution/scaling becomes a learning bottleneck.
- **Six four-hour sell bins.** Exact event timing is preserved canonically, but V0 does not learn 24-turn reactive selling. Revisit because selling remains the clearest teacher-forced weakness.
- **Per-event sell cap 100 in the BC adapter.** This removes sell-all sentinel scale before bin aggregation while preserving repeated events. Revisit if closed-loop sale intent systematically under/overstates desired liquidation.
- **Fixed 0.5 sell-presence threshold.** No calibration or threshold tuning was performed. Revisit after measuring closed-loop precision/recall and economic cost of missed/extra sales.
- **Seven equal loss-group weights.** No focal loss, class balancing, sparse-target reweighting, or learned uncertainty weighting. Rare fertilizer recall remains weak. Revisit only if those misses matter in games.
- **Small default model chosen without architecture sweep.** The 1.071M Transformer is a deliberately boring reference, not an optimized size. No scheduler, width/depth sweep, or larger model was tested. Revisit after closed-loop behavior establishes that representation capacity is actually limiting.
- **Five-day elite corpus and `min_score >= 2950`.** No cutoff sweep, later-date expansion, trajectory-family reweighting, or deduplication was used for the first run. The corpus contains substantial repeated strategy families. Revisit if later held-out dates/strategies reveal imitation-family overconcentration.
- **Single held-out date protocol.** Aug-21 is a meaningful distribution shift but not a full generalization suite. Revisit with rolling dates, later partitions, held-out strategy families, and eventually closed-loop opponent/seed panels.
- **No DAgger / on-policy correction.** Training is pure teacher-forced BC from replay states. Revisit if closed-loop state drift creates compounding errors that held-out replay metrics cannot predict.
- **No value head or PPO objective.** BC only initializes strategy. Revisit after the deterministic executor and fixed-opponent closed-loop problem are stable.
- **No confidence-aware execution.** Argmax/count and thresholded sell predictions are treated as actions without using model uncertainty. Revisit if low-confidence outputs correlate with destructive executor choices.

### Current known weak points to retain

- crop counts are often close but not exact, especially wheat/strawberry;
- rare tomato/carrot fertilizer recall is weak;
- sell presence recall is ~64.8% and predictions are conservative;
- teacher-forced metrics cannot establish closed-loop competence;
- date-held-out success does not prove strategy-family or opponent generalization.

These are deferred questions, not reasons to delay the first real game loop. The next evidence should determine which backlog items deserve complexity.
