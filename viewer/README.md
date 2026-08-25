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

CLI help succeeded, and the full selected runs below used this exact
PowerShell interpreter:

```powershell
$py = "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\temp\official-1327-venv\Scripts\python.exe"
```

Equivalent commands after activating that environment, or from an ordinary
environment that already provides the pinned official dependency, use
`python` directly:

```powershell
python -m rl_manager.cli debug-trace --backend official --case 17:0 --output-dir artifacts/debug_traces
```

Observed artifact evidence:

The probe builds its model at reset index 0, so `sidecar=false` in the table
means the expected opening-turn absence; later turns contain the sidecars
described below.

| case | artifact | schema/turns | bytes | helper/open check |
| --- | --- | ---: | ---: | --- |
| `17:0` | `artifacts/debug_traces/seed_17_seat_0.json` | v1 / 720 | 11964111 | Python schema/720-turn/monotonic/age-alias checks passed; probe reported `turns=720`, `cells=100`, `sidecar=false`; viewer query and artifact routes 200 |
| `42:0` | `artifacts/debug_traces/seed_42_seat_0.json` | v1 / 720 | 11954068 | Same checks passed; probe reported `turns=720`, `cells=100`, `sidecar=false`; viewer query and artifact routes 200 |
| `1013:0` | `artifacts/debug_traces/seed_1013_seat_0.json` | v1 / 720 | 11962958 | Same checks passed; probe reported `turns=720`, `cells=100`, `sidecar=false`; viewer query and artifact routes 200 |
| `2026:1` | `artifacts/debug_traces/seed_2026_seat_1.json` | v1 / 720 | 11960383 | Same checks passed; probe reported `turns=720`, `cells=100`, `sidecar=false`; viewer query and artifact routes 200 |

The four full runs were executed sequentially with the following commands;
each CLI summary reported `turns=720` and `terminated=True`:

```powershell
& $py -m rl_manager.cli debug-trace --backend official --case 17:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 42:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 1013:0 --output-dir artifacts/debug_traces
& $py -m rl_manager.cli debug-trace --backend official --case 2026:1 --output-dir artifacts/debug_traces
```

The documented viewer launch and open-check were:

```powershell
python -m viewer.server --port 8765
```

```text
http://127.0.0.1:8765/viewer/?trace=/artifacts/debug_traces/seed_17_seat_0.json
```

Concrete observations from the successful traces (all counts below were
present in each selected artifact and combine both executor seat snapshots):

- The first executor sidecar appears at step 96 (day 4, hour 0), for both
  seats; sidecars continue through step 718 (day 29, hour 22), while terminal
  step 719 has no sidecar. This records the opening-to-executor alignment, not
  a causal claim about the opening policy.
- Task records include 5,798 `PLACE`, 1,318 `FERTILIZE`, 1,198 `BUY_ANIMAL`,
  982 `WATER`, 432 each of `COLLECT_FERTILIZER`, `FEED`, and `CARE`, 406
  `HARVEST`, and 192 `BUY_PRODUCT` records. There are 1,528 assignment
  records and 3,302 unassigned task keys; these are stored task/assignment
  facts, not proof that any task caused a later state.
- Survival sidecars reach `unfed_count=5`,
  `starvation_boundary_count=4`, and `shortage=5`. `expansion_suppressed` is
  `true` in 672 of 1,246 survival snapshots. These fields show recorded
  pressure/suppression indicators; they do not establish starvation or causal
  policy failure.
- The traces contain 24 sidecar snapshots with non-empty `market.submitted`
  and 670 with non-empty `market.unaffordable`. Joint actions include stored
  movement (`NORTH=134`, `WEST=130`, `SOUTH=24`, `EAST=22`) and interaction
  actions including `WATER=104`, `FEED=28`, and `CARE=28`. The viewer helper
  extracts a 12-point farmer trail at the final window for each selected case.
- `eod_work_debt` is present in 50 survival snapshots; its largest `all` list
  contains 53 task keys, with maxima of 5 maintenance, 24 manager, 14
  productive, and 11 survival entries. This documents end-of-day debt fields;
  it does not infer why the work remained unresolved.

The server checks for this stage returned HTTP `200` for `/viewer/` and each
JSON artifact route, and `404` for `/README.md` and
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
