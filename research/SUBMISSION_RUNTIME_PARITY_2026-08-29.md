# Submission runtime parity failure — 2026-08-29

Base integration SHA: `45f88001b6cc14f802f10668179a68f6fe3c2bf5`.

## Symptom

An exported PPO u50 submission that was extremely strong in the self-play/eval runner reached a live competition rating of about 66, effectively losing almost every rated match.

The initial suspicion was checkpoint export, Torch/JAX parity, or fast-vs-official engine drift. The experiments below localized the failure to the economic-context feature path used by the deployed Torch provider.

## What was ruled out

### Archive / checkpoint identity

The submitted archive contained the intended RL-export checkpoint:

- checkpoint kind: `rl_export`
- epoch: 50
- model variant: E
- checkpoint SHA-256: `4137742e37aea144dba62eb9cb53b68fcfa5171346d535f0a9e1fb39ee3a4e06`
- `aggressive_sell_all=False`
- opening: `standard_mixed`

So this was not a simple wrong-checkpoint packaging mistake.

### Fast vs official engine

The same three-update JAX PPO checkpoint evaluated through the normal RL runner produced:

| backend | W-L-T | candidate mean | opponent mean | mean margin |
| --- | ---: | ---: | ---: | ---: |
| fast | 6-0-0 | 43,826.0 | 16,992.5 | +26,833.5 |
| official | 6-0-0 | 43,188.0 | 17,189.5 | +25,998.5 |

The engines are not bit-identical in this small panel, but the difference is small relative to the deployment collapse. Fast-vs-official divergence was not the root cause.

## Dead direct Torch path

Running the exported RL checkpoint and original BC-E through the standalone Torch `CheckpointPlanProvider` on the official environment produced extremely low banks despite no invalid/error statuses:

- RL mean bank: 745.375
- BC mean bank: 4.25
- RL vs BC W/L/T: 8/0/0
- bad games: 0

Both policies being nearly dead was the key clue: this was a shared Torch live-inference semantic mismatch, not a PPO-specific export failure.

## Root cause: runner E economic history bug

`rl_manager.runner._EpisodeState` intends to feed the previous daily-start `(day, money)` through `economic_prev_start`.

However, the legacy runner stores the current daily start and then passes it immediately:

```python
self.daily_start[seat] = (
    day, float(obs["farms"][seat]["money"])
)
...
return encode_live_inputs(
    obs,
    seat,
    dict(self.previous_execution[seat]),
    step=int(obs["step"]),
    economic_prev_start=self.daily_start[seat],
)
```

`bc_manager.live.encode_live_inputs` only treats an explicit prior start as valid when:

```python
prev_day == day - 1
```

Therefore the legacy runner passes `(current_day, current_money)` as the purported previous start. The E cash-delta channel is marked invalid and receives the legacy zero/invalid delta semantics every day.

The normal Torch `CheckpointPlanProvider`, in contrast, owns a real `EconomicHistory` and supplies the true previous-day cash delta. Existing PPO checkpoints were therefore trained/evaluated under one feature distribution and deployed under another.

## Confirmation

A temporary Torch `RunnerParityProvider` was made to intentionally reproduce the legacy runner behavior by passing the current day/money pair as `economic_prev_start`.

The exact same exported u50 checkpoint immediately recovered:

- W/L/T vs BC-E: 6/2/0
- RL mean bank: 41,909.5
- BC mean bank: 21,829.5
- mean margin: +20,080

This is roughly a 56x change in RL bank from only the economic-context semantics.

The final exact fixed tarball was then tested on 16 official games against BC-E:

- W/L/T: 14/2/0
- RL mean bank: 43,860.375
- RL median: 45,635
- BC mean bank: 22,117.8125
- mean margin: +21,742.5625
- bad games: 0

This confirms the submission failure and the compatibility fix.

## Second deployment mismatch discovered

The tracked canonical submission entrypoint used environment-dependent strictness:

```python
value = os.environ.get("KAGGRICULTURE_SUBMISSION_STRICT", "0")
...
strict=_strict_from_environment()
```

Local archive smoke tests had explicitly enabled strict mode, while Kaggle itself did not. Safe mode catches executor exceptions and silently returns all-PASS actions.

The validated fixed submission therefore hardcodes:

```python
AgentConfig(
    strict=True,
    suppress_expansion_from_prior_debt=True,
    aggressive_sell_all=False,
)
```

For the instant-sell BC control, only `aggressive_sell_all=True` changes.

This strictness mismatch was not the measured cause of the 745-bank collapse, but it made the pre-submission verification non-representative and should not be repeated.

## Third deployment mismatch: lazy `fast_env.market` import

The first fixed archives later failed in the real Kaggle sandbox when a game
entered the survival-feed affordability path. The traceback ended at:

```text
from fast_env.market import market_price
ModuleNotFoundError: No module named 'fast_env'
```

Earlier exact-archive game panels had not happened to execute this lazy branch,
so the missing dependency remained latent. A follow-up attempt that added a
separate top-level `fast_env` package still produced the same live sandbox
error, so the deployment builder should not rely on a newly added top-level
package being available.

The compatibility builder now makes this path self-contained inside the
already-packaged executor package:

1. copy pure-Python `fast_env/market.py` to
   `executor_v0/_submission_market.py`;
2. rewrite the packaged executor import from
   `from fast_env.market import market_price` to
   `from executor_v0._submission_market import market_price`.

The market helper is framework-independent and does not require the native fast
engine extension. Pre-submission tests must explicitly force a `BUY_PRODUCT`
`_buy_order_cost` call so the lazy import is executed rather than hoping a
small gameplay seed panel reaches emergency feed buying.

## Compatibility policy

### Existing checkpoints trained at / before the legacy runner semantics

Deploy with runner-compatible zero/invalid E delta semantics. Use `tools/build_runner_compatible_submission.py`.

Do **not** deploy these checkpoints through the normal `EconomicHistory` Torch provider and assume the result is equivalent.

### Future checkpoints

The runner should be corrected so it feeds the actual previous daily start, but that is a semantic/version boundary and should not be slipped into an unrelated compatibility commit.

Recommended sequence:

1. version the runner economic-context semantics;
2. fix daily-start bookkeeping so day D receives `(D-1, money_at_D-1_start)`;
3. add a regression test comparing runner E inputs against `EconomicHistory` inputs across multiple days;
4. retrain/evaluate from BC-E under the corrected feature distribution;
5. make corrected semantics the default for all new checkpoints;
6. retain the legacy compatibility builder only for old artifacts.

## Submission procedure going forward

Before uploading any archive:

1. extract the exact `.tar.gz` into a clean temporary directory;
2. import that extracted `main.py` rather than repository code;
3. force-call the packaged `ExecutorAgent._buy_order_cost` `BUY_PRODUCT` branch;
4. run both seat orientations over a small official seed panel;
5. require zero invalid/error/timeout statuses;
6. compare bank distribution and W/L against a known baseline;
7. only then submit the exact tested archive bytes.

On 2026-08-29 both the fixed RL u50 archive and the runner-compatible BC-E instant-sell control passed their initial exact-archive official gameplay panels, but live submission exposed the additional lazy-import packaging fault described above. Daily submission quota was exhausted before the self-contained executor-vendored fix could be re-submitted.
