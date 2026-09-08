# Stage 2.5 executor scheduling packet

## Handoff

The implementation is split into independently disabled controls:

| Control | Default | Contract |
| --- | --- | --- |
| `deadline_safe_planting` | off | Require Manhattan travel plus PLANT and the linked WATER interaction to fit in the actual remaining actionable horizon. Hour 22 can qualify when underfoot; hour 23 and terminal step 718 do not. |
| `deadline_safe_hiring` | off | Submit HIRE only when the newly hired worker has a future worker-action slot before reset or termination. |
| `persistent_worker_queues` | off | Use a deterministic greedy persistent queue with ownership, resource reservations, local repair, explicit urgent preemption, and day/episode/worker-count resets. |
| `schedule_informed_hiring` | off | Estimate useful scheduled workload using travel, spawn position, pickup/interaction, PLANT→WATER, remaining capacity, affordability, Fibonacci marginal cost, and order limits. |

All four controls are candidate-only in `UpkeepFactory`. With them disabled, the existing executor path and manager/opponent construction are retained. Pathfinding, crop placement, animal placement, care, fertilizer, wheat threshold, capture, and cleanup behavior are unchanged.

The persistent scheduler is intentionally bounded and inspectable: it performs deterministic route insertion and at most one small improvement pass. It does not add RNG calls, policy inference, observation mutation, exhaustive search, or a new pathfinder. Queue diagnostics record ownership, retention/release/repair, preemption, predicted completion/deadline failures, resource failures, and runtime.

## Baseline provenance

The underfoot artifacts were available outside the checkout and were compared before implementation:

* `C:\Users\liuyi\Downloads\Kaggriculture_Underfoot_CPU.ipynb`, SHA-256 `a8df8c30f0d98f2b813f0dd75e224f2a05ed05536d8247a1396b29ad0d07b8fc`.
* `C:\Users\liuyi\Downloads\underfoot_results.zip`, SHA-256 `38042e088f133bd1b683dc83681c0bf03b69240d51f7ce5152d7d29618b6935c`.
* ZIP `underfoot.patch`, SHA-256 `c4aa15644ca5c7eb9e75df45ba67e55e5604af74fec75834f57916a9d30f3e1b`.
* ZIP `payload.json`, SHA-256 `a98f4bacd9d7a8903bce48d2307e9941d19c98b66c98bef48df292263f7c7c47`.

The embedded patch confirms the provisional underfoot contract: dispatch all immediately executable worker tasks first, then distant claims, and restore output order by worker index. Its payload identifies source commit `4fce6af49b97d5437bbc1564d0d7037416a1beb0` with a dirty source tree; therefore the archive is provenance evidence, not an exact clean-commit reproduction.

The pinned local official engine source was inspected at `data/temp/official-1327-venv/Lib/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`, SHA-256 `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`. Mechanics confirmed during the audit: a new plant starts with one dry-day count; worker actions happen before market hiring; a hired hand cannot act until the next permitted worker turn; and the 30-day horizon has 719 actionable worker turns (terminal state at step 719, final actionable step 718).

The recovered 16-seed ordered panel is:

`144368101, 309507, 615013, 918079, 1221109, 1524137, 1827169, 2130193, 2433221, 2736251, 3039283, 3342311, 3645341, 3948373, 4251401, 2112243121`.

## Validation

Focused local validation completed:

* `python -m pytest tests/test_executor_v0_agent.py tests/test_executor_v0_foreman.py tests/test_executor_v0_scheduler.py -q` — 107 passed.
* `python -m pytest tests/test_executor_v0_hiring.py tests/test_stage25_capture.py tests/test_stage25_sharded.py -q` — 21 passed.
* `python -m compileall -q executor_v0 tools` — passed.
* `ruff check` over changed executor, evaluator, sharding, and focused-test files — passed.
* `git diff --check` — passed.

The complete executor/stage25 regression selection (18 test modules) passed with 340 tests; the initial unconfigured local run exposed only the Windows Git `safe.directory` ownership guard, then passed with an explicit command-local safe-directory setting.

The tests cover default-off behavior, underfoot output ordering, queue retention and actual arrival interaction, exclusive ownership, shared inventory/dependencies, failed-action recovery, urgent preemption, resets, planting/hiring boundaries including terminal timing, candidate-only flag propagation, four-process identity sharding, failure preservation, and validated resume.

## Evaluation handoff

Run [the paired-evaluation notebook](../notebooks/stage25_executor_schedule_ablation.ipynb). It checks out `3431dbf065044a66ea469884dc85e68f74e72769`, keeps P-final sampling stochastic and BC-E on `E_LEGACY`, uses the official `kaggle_environments==1.32.7` authority, and launches four independent one-CPU processes. The default comparison is 32 seed/seat games per arm on the verified `combined_wheat3` baseline variant:

1. Control: `underfoot-first + deadline-safe-planting + deadline-safe-hiring`.
2. Treatment: the same controls plus `persistent-worker-queues`.

The schedule-informed hiring arm is selectable but disabled by default. Every arm preserves the full seed list, uses `episode_id = master_seed * (2 * len(full_seeds)) + 2 * seed_index + seat` with master seed 25, emits pair bootstrap groups with both seats together, and writes source/patch/checkpoint/engine hashes. Shard output is resumable only after manifest configuration/checkpoint hash validation and complete identity validation.

The wrapper reports candidate bank, opponent bank, margin, completed work, missed maintenance, travel abandonment, hiring cost, and scheduler runtime where the underlying evaluator exposes them; missing telemetry is reported as unavailable rather than fabricated. Results and manifests are bundled into a ZIP by the notebook.

No Kaggle checkpoint evaluation was run locally: the P-final and BC-E checkpoint files are not present in this checkout. Competitive conclusions and promotion remain deferred to the runnable notebook and official engine evaluation.
