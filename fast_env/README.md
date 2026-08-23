# Fast Kaggriculture engine

`FastKaggricultureEnv` is the Stage 1 scalar engine seam:

```python
from fast_env import FastKaggricultureEnv

env = FastKaggricultureEnv(configuration={"seed": 7})
observations = env.reset()  # two dict observations, one private view per seat
observations, rewards, statuses = env.step([
    {"farmer": ["PASS"], "hands": [], "market": []},
    {"farmer": ["PASS"], "hands": [], "market": []},
])
```

The action grammar follows the official JSON shape. Observations preserve the
existing `farms`/`market`/`town`/time fields and each seat receives only its
own `private` shed, seeds, and carried inventories. `state_snapshot()` returns
the latest decoded pair for later differential tooling.

Build/install from the repository root:

```text
python -m pip install maturin numpy
python -m maturin develop --release
```

On Windows machines without Visual Studio MSVC `link.exe`, use the
self-contained GNU toolchain instead (verified route; keep the toolchain in a
local directory via `RUSTUP_HOME`/`CARGO_HOME` if a system install is not
wanted):

```text
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
$env:CARGO_BUILD_TARGET = "x86_64-pc-windows-gnu"
python -m maturin develop --release
```

The normal import path loads NumPy and the local PyO3 extension only; it does
not import `kaggle_environments`, OpenSpiel, or the Kaggle registry.
