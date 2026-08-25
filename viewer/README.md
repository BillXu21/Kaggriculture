# Debug-trace viewer workflow

The viewer is a dependency-free reader for the canonical schema-v1 trace
artifacts emitted by the committed `rl_manager` runner. It never starts or
replays an environment: generation owns the run, and the viewer only loads
stored JSON.

## Prerequisites

- Run commands from the repository root with the project Python environment
  active.
- Use the default deterministic tiny E policy for a local smoke, or pass a
  real compatible BC-E Torch checkpoint with `--e-checkpoint PATH`.
- The default `fast` backend requires the repository native extension
  `fast_env._kaggriculture_env`; the `official` backend requires the pinned
  `kaggle-environments==1.32.7` dependency.
- Node.js is only needed for the optional `tests/viewer_probe.js` helper check;
  the browser viewer itself has no third-party dependency.

## Generate traces

The default `--max-turns 719` runs 719 primitive transitions and stores 720
observed turns: the reset snapshot followed by each observed transition.
Output filenames are deterministic:
`artifacts/debug_traces/seed_<SEED>_seat_<SEAT>.json`.

For an official-backend run in the verified project-local environment, invoke
that interpreter explicitly from the repository root:

```powershell
$py = "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\temp\official-1327-venv\Scripts\python.exe"
& $py -m rl_manager.cli debug-trace --backend official --case 17:0
```

The virtualenv is a read-only runtime dependency; do not install packages into
it. The current worktree supplies the imported `rl_manager`, viewer, and other
repository modules.

One case, using the default fast backend and tiny E policy:

```powershell
python -m rl_manager.cli debug-trace --case 17:0
```

The equivalent explicit single-case form is:

```powershell
python -m rl_manager.cli debug-trace --seed 17 --seat 0
```

Run selected cases sequentially in one invocation (the runner writes one
artifact per case, in argument order):

```powershell
python -m rl_manager.cli debug-trace --case 17:0 --case 42:0 --case 1013:0 --case 2026:1
```

Useful options:

```text
--backend fast|official       Select the committed backend (default: fast).
--e-checkpoint PATH            Use a real BC-E Torch checkpoint instead of tiny E.
--policy-seed 11               Seed the default tiny E policy.
--num-threads 1                Set backend engine threads.
--max-turns 719                Keep the full 720-turn trace contract.
--output-dir DIR               Change the ignored artifact directory.
```

For a real checkpoint, add `--e-checkpoint PATH`; the checkpoint must contain
the compatible `metadata.model_config`. The CLI validates the path before
constructing the environment. `--output-dir` should remain under
`artifacts/` for local runs; generated traces are ignored and must not be
committed.

If fast native code is unavailable, try the official backend without changing
the runner:

```powershell
python -m rl_manager.cli debug-trace --backend official --case 17:0
```

## Launch and open a trace

Start the repository-local server from a second PowerShell window:

```powershell
python -m viewer.server --port 8765
```

Open the exact URL for the first generated artifact:

```text
http://127.0.0.1:8765/viewer/?trace=/artifacts/debug_traces/seed_17_seat_0.json
```

The `trace` query value is a repository-served path, not a `file://` URL. The
server exposes only the viewer assets and JSON files under
`/artifacts/debug_traces/`; repository source files and traversal paths are
not served. To check the HTTP responses in PowerShell:

```powershell
(Invoke-WebRequest http://127.0.0.1:8765/viewer/).StatusCode
(Invoke-WebRequest http://127.0.0.1:8765/artifacts/debug_traces/seed_17_seat_0.json).StatusCode
```

Both should print `200`. If the query URL is inconvenient, open
`http://127.0.0.1:8765/viewer/`, click **Load trace**, and choose the JSON
file, or drag it onto the drop zone. The file picker is also the fallback when
the trace URL is mistyped.

The `trace` query value must be an absolute same-origin path under
`/artifacts/debug_traces/` and must end in `.json`. Remote URLs, protocol-
relative URLs, viewer-source paths, relative paths, traversal, and other
non-artifact paths are rejected in the viewer with an error; file picker and
drag/drop loading remain local fallbacks.

## Viewer controls and alignment

- **Back**, **Forward**, the step slider, **Play/Pause**, and the speed menu
  navigate stored `turns[index]` entries; Left/Right arrows and Space are
  keyboard shortcuts when a form control is not focused.
- **Seat 0/Seat 1** changes the farm, private inventory, workers, and
  executor sidecar shown for that seat.
- Toggle **trails**, **assignment lines/targets**, **task markers**,
  **urgency/status colors**, and **labels** independently; the trail-window
  slider controls the number of recent stored turns used for trails.
- The header reports the displayed day/hour, trace step, seat, view, and
  backend. Canonical state and turn alignment are checked by the viewer:
  `turn.step`, `turn.day`, and `turn.hour` must equal the corresponding
  `canonical_state` fields. Step 0 is the reset snapshot; the final default
  index is 719.
- The opening/reset turn can legitimately have no `executor_debug` sidecar,
  because no executor action has been submitted yet. The executor panel says
  **Executor sidecar absent for this opening turn**; advance one turn to inspect
  task, assignment, survival, and action diagnostics.
- Text panels remain authoritative when an optional coordinate is absent.
  Stored fields are evidence; the viewer does not infer causal explanations.

## Validation and diagnostic notes

Validate one generated artifact with the canonical Python loader, then run the
viewer helper against the same file:

```powershell
python -c "from pathlib import Path; from rl_manager.debug_trace import load_trace; p=Path('artifacts/debug_traces/seed_17_seat_0.json'); t=load_trace(p); assert len(t['turns']) == 720; print(p, 'schema', t['schema_version'], 'turns', len(t['turns']), 'bytes', p.stat().st_size)"
node tests/viewer_probe.js artifacts/debug_traces/seed_17_seat_0.json
```

For multiple cases, repeat the two checks per filename. The probe validates the
schema through viewer helpers, builds a 100-cell view model, and checks that
loading does not mutate the trace.

### Current official-runtime evidence

The official runtime was verified at:

```text
C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\temp\official-1327-venv\Scripts\python.exe
Python 3.13.1; kaggle-environments 1.32.7
```

CLI help succeeded, and this sequential one-turn smoke succeeded:

```powershell
& "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\temp\official-1327-venv\Scripts\python.exe" -m rl_manager.cli debug-trace --backend official --case 17:0 --max-turns 1 --output-dir artifacts/debug_traces
```

Observed artifact evidence:

| case | artifact | schema/turns | bytes | helper/open check |
| --- | --- | ---: | ---: | --- |
| `17:0` one-turn smoke | `artifacts/debug_traces/seed_17_seat_0.json` | v1 / 2 | 9053 | Python loader valid; `node tests/viewer_probe.js artifacts/debug_traces/seed_17_seat_0.json` reported `turns=2`, `cells=100`, `sidecar=false`; HTTP trace route 200 |

The required full-run command was then attempted sequentially for `17:0`,
`42:0`, `1013:0`, and `2026:1`:

```powershell
& $py -m rl_manager.cli debug-trace --backend official --case 17:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 42:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 1013:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 2026:1 --output-dir artifacts/debug_traces
```

All four exited with the same runtime failure before a full artifact was
written: `executor_v0` reaches `_buy_order_cost`, which imports
`fast_env.market`, and the current worktree's `fast_env` package then raises
`ModuleNotFoundError: No module named 'fast_env._kaggriculture_env'`. The
official engine itself starts successfully; no full 720-turn validation or
selected-trajectory observation is claimed from these attempts.

The successful one-turn smoke provides these bounded observations only: the
trace has reset `(step=0, day=0, hour=0)` followed by `(step=1, day=0, hour=1)`;
the reset turn carries joint-action keys for both seats but no
`executor_debug` sidecar, and the final observed turn has neither action nor
sidecar. These fields demonstrate the stored alignment contract, not a causal
claim about executor behavior, feed, market, travel, or end-of-day debt.

The server checks for this stage returned HTTP `200` for `/viewer/` and the
JSON artifact, and `404` for `/README.md` and
`/artifacts/debug_traces/../README.md`. Generated artifacts remain ignored and
are not committed.

When recording observations from a real selected run, cite the displayed
turn/day/hour and the stored fields rather than inferring causes. Useful
evidence includes:

- the first turn without a sidecar and the next turn's manager handoff;
- `survival.unfed_count`, `starvation_boundary_count`, wheat shortage, and
  FEED task/assignment records;
- task `kind`, `source`, `priority`, `unassigned.reasons`, and assignment
  targets alongside worker action/trail positions;
- market inventory/prices and submitted market actions; and
- final-turn rewards/statuses and remaining tasks or debt-like shortages.

## Troubleshooting

### `fast_env._kaggriculture_env` is unavailable

The committed CLI fails closed instead of switching to another environment
loop. Build the repository extension locally from the repository root using
the project-supported path:

```powershell
python -m maturin develop --release
```

If Windows has no MSVC `link.exe`, the documented local GNU-toolchain route is:

```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
$env:CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu"
python -m maturin develop --release
```

Alternatively, install the pinned official dependency in the project
environment and use `--backend official`:

```powershell
python -m pip install kaggle-environments==1.32.7
python -m rl_manager.cli debug-trace --backend official --case 17:0
```

Do not install globally or create a second runner. If neither committed
backend is available, preserve the exact traceback and report generation as
blocked; do not claim 720-turn validation or trajectory observations.

### Other common errors

- `--case` must be `SEED:SEAT`; repeat it for multiple cases, and do not mix it
  with `--seed`/`--seat`.
- A missing checkpoint is rejected before an environment is constructed; pass
  a real path or omit `--e-checkpoint` for the deterministic tiny E policy.
- If port 8765 is busy, choose another port and use the matching port in the
  browser URL and HTTP checks.
- A viewer HTTP 200 only proves the server returned the file; the Python
  loader and Node probe are still required to establish schema/helper loading.
