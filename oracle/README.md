# Official-vs-Fast Differential Oracle (Stages 2a-2b)

Reusable same-action oracle that proves the fast Rust engine matches the
official pinned `kaggle-environments` 1.32.7 engine turn-by-turn. The official
engine is the only correctness oracle; broad mechanic/full-episode parity
sweeps are **Stage 2b work and are not yet done** — do not claim full parity
or training safety from this package alone.

## Public surface (`oracle/__init__.py`)

- `make_backend("official" | "fast", configuration)` — narrow engine seam
  (`reset` / `step` / `canonical_state` / `rewards` / `statuses`);
- `run_same_action_replay(configuration, actions, ...)` — submits the exact
  same action pair to both engines each turn BEFORE comparing canonical full
  states; raises `DivergenceError` at the FIRST divergent field with a
  `DivergenceReport` (seed/step/day/hour/field path/both values/both actions);
- `verify_official_provenance()` — refuses to run against any installed
  `kaggle_environments` that is not exactly version 1.32.7 with interpreter
  files byte-matching upstream commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`
  (wheel SHA256 `2a1bb862...c4c8f`, see `oracle/provenance.py`);
- `canonical_state_official` / `canonical_state_fast` / `deep_diff` — one
  canonical schema compared field-path-exactly: step/day/hour; both farms
  (money, full 10x10 board with crop AND animal lifecycle, farmer/hand
  positions, hires_today, unlocked quadrants); both seats' private
  shed/seeds/farmer+hand inventories; market inventory/prices (+ exact
  `params` when present); town unlocked shops WITH duplicate multiplicity;
  rewards; statuses;
- `status_anomalies` / `OfficialAnomalyError` — the FULL per-step status
  history must stay within {ACTIVE, DONE}; a terminal DONE never masks an
  earlier ERROR/INVALID/TIMEOUT.

Import isolation: importing `oracle` or `fast_env` never imports
`kaggle_environments`; the official backend imports it lazily inside
`make_backend("official", ...)`. Verified by fresh-process tests in
`tests/test_oracle_import_isolation.py`.

## Temp official environment (this Windows worktree)

The official wheel is intentionally NOT installed in the system Python. A
disposable venv lives under the temp root:

```text
VENV=C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv
python -m venv %VENV%
%VENV%\Scripts\python -m pip install kaggle-environments==1.32.7 pytest maturin numpy
```

Rebuild/install the current fast extension into it using the documented
temp-local GNU toolchain (no MSVC needed):

```text
$env:VIRTUAL_ENV      = "C:\Users\liuyi\AppData\Local\Temp\opencode\kagg_oracle_venv"
$env:RUSTUP_HOME      = "C:\Users\liuyi\AppData\Local\Temp\opencode\rustup"
$env:CARGO_HOME       = "C:\Users\liuyi\AppData\Local\Temp\opencode\cargo"
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
$env:CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu"
$env:PATH = "$env:RUSTUP_HOME\toolchains\stable-x86_64-pc-windows-gnu\bin;$env:CARGO_HOME\bin;" + $env:PATH
& $env:VIRTUAL_ENV\Scripts\python.exe -m maturin develop --release   # from repo root
```

Run the oracle tests against the REAL official engine (from repo root):

```text
& $env:VIRTUAL_ENV\Scripts\python.exe -m pytest tests/test_fast_env.py tests/test_oracle_offline.py tests/test_oracle_replay.py tests/test_oracle_import_isolation.py -v
```

Without the venv, `tests/test_oracle_replay.py` skips itself via the
provenance guard; offline comparator/corruption tests always run.

Day-boundary smoke (28 pass-only turns across day 0 -> day 1, real official):
see the Stage-2a HISTORY entry; the trace is
`[[PASS, PASS]] * 28` through `run_same_action_replay({"seed": 7}, trace)`.

## Known limitations (bounded, deliberate)

- Fast engine supports `boardSize=10` and `maxMarketOrdersPerTurn=10` only.
- Former >16-hired-hands deferral CLOSED (2026-08-23): the fast core now uses
  the exact default-contract capacity `MAX_HANDS = 10 orders/turn * 24
  turns/day = 240` (breaking wire layout: OBS_SIZE 8766, ACTION_SLOTS 251,
  MASK_SIZE 34026; see `MECHANICS.md` and decision D-021). Scenarios beyond
  240 hands are unreachable under the pinned default configuration.
- Fast observation money is recovered by rounding the f32 normalize(10000)
  encoding; exact for the integer money values the official engine produces
  (verified to well beyond realistic ranges).
- Observation decoding inverts the FIXED season length
  (`generated_protocol::SEASON_STEPS = 720`), matching the Rust writer, NOT
  the configured `episodeSteps`; non-default `episodeSteps` therefore change
  episode length but decode on the fixed scale (regression-tested).
- Canonical comparison covers everything listed above; it does not compare
  internal RNG streams or unreached code paths. Passing traces prove parity
  only for the actions exercised.
- Full-episode legal-ish corpus DONE (2026-08-23, decision D-022): seeds 0,
  1, 2, 7, 17, 42, 123, 999 each ran a complete default 720-step episode
  (reset observation + exactly 719 accepted primitive `step` calls; terminal
  DONE at canonical step 719 = day 29 hour 23) with ZERO first divergence,
  33 action families covered, and locked repeatability. Report:
  `research/parity_corpus_report.json`. Re-run the full gate with the oracle
  venv interpreter from the repo root:

  ```text
  & $env:VIRTUAL_ENV\Scripts\python.exe scripts\run_parity_corpus.py
  ```

Exit code 0 means zero divergence across all requested seeds. This is a
bounded result: parity is proven for the states those episodes reach, not
universally. The independent closed-loop A/B gate is described below.

## Independent closed-loop A/B

`run_closed_loop(configuration, max_steps=719, backend_factories=...,`
`agent_factories=...)` creates independent backend instances and four fresh
stateful agents. At reset and before every transition it compares the
corresponding observations; each backend's agents then compute actions from
their own observations. Actions are compared before either backend is stepped,
and the canonical next state, rewards, and statuses are compared immediately.
The first failure is a `ClosedLoopDivergenceReport` with seed, step, day, hour,
seat, field path, both values, and both actions. Official full status history
validation is preserved.

The default deterministic fixture is
`make_deterministic_executor_factory()`: it uses the existing stateful
`executor_v0.ExecutorAgent` with a fixed nontrivial `DailyPlan`, not a shared
agent or a fabricated checkpoint. `make_checkpoint_executor_factory(path)` is
the corresponding explicit real-checkpoint adapter. The fast fixture converts
only fast wire tile aliases and sparse private maps that the existing executor
does not consume directly; decisions remain independently computed.

Run the bounded report in the pinned official venv:

```text
& $env:VIRTUAL_ENV\Scripts\python.exe scripts/run_closed_loop_ab.py
```

The report records seeds, transitions, action families, terminal outcomes, wall
time, and repo-local checkpoint evidence. Deliberate observation and action
drift tests prove that the runner stops before stepping at the first
policy-interface mismatch. This secondary gate does not replace the primary
same-action engine parity corpus.
