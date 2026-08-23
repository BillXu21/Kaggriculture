# Official-vs-Fast Differential Oracle (Stage 2a)

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
- Observation decoding inverts the FIXED season length
  (`generated_protocol::SEASON_STEPS = 720`), matching the Rust writer, NOT
  the configured `episodeSteps`; non-default `episodeSteps` therefore change
  episode length but decode on the fixed scale (regression-tested).
- Canonical comparison covers everything listed above; it does not compare
  internal RNG streams or unreached code paths. Passing traces prove parity
  only for the actions exercised.
- No broad random/legal-ish corpus, no multiple full 720-turn episodes, no
  closed-loop A/B, no benchmark report: all Stage 2b.
