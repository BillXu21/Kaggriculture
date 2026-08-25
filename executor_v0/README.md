# executor_v0 — Deterministic Closed-Loop BC Executor (issue #1)

Complete V0 closed loop from live observation to legal primitive actions.
Implemented per `research/EXECUTOR_V0_PLAN.md`; deliberately simple, nothing
optimized. Simplifications/backlog: see that plan's "Deliberate V0
simplifications" and "Implementation status" sections.

## Pipeline (per primitive turn)

1. Live schema-v3 observation encoding with exact BC adapter parity
   (`bc_manager/live.py`).
2. Once per new day: the injected `PlanProvider` (checkpoint-backed or fake)
   produces an immutable `DailyPlan`, fed the previous day's realized labor
   (observed `hires_today` priced by the exact Fibonacci hire cost).
3. Mechanical requested -> feasible projection (`projection.py`).
4. Deterministic animal layout + minimum-change crop reconciliation
   (`layout.py`), then explicit task regeneration every turn (`tasks.py`).
5. Greedy foreman dispatch (`foreman.py`): underfoot execution first,
   priority-dominated greedy assignment, shed routing for carried items,
   one legal Manhattan step, global-seed reservation for PLANT.
6. Market queue under the 10-order cap, deterministic SELL -> survival WHEAT
   -> HIRE -> other buys: six-bin sells clipped to actual shed inventory
   (`clip_sell`), survival feed purchases before hiring, turn-aware any-hour
   workload hiring, exact-shortage BUY_SEED/BUY_PRODUCT/BUY_ANIMAL and single
   BUY_LAND.
7. Output: `{"farmer": [...], "hands": [[...]], "market": [[...]]}` plus JSON
   diagnostics (requested/feasible/achieved/submitted/observed) and a
   deterministic all-PASS fallback on any runtime failure.

Seed mechanic: `PLANT <crop>` consumes the **global own `private.seeds`
pool atomically at the engine**; seeds are never picked up or carried.

V0.7 survival contract: the executor uses the official/default
`shed_capacity=100` clamp when preserving shed room for survival WHEAT, and
the prior-debt expansion suppression heuristic defaults **ON**. This default
is explicit architectural debt bounded to the three-seed/two-seat PASS panel;
it is not a learned economic policy or a generalization claim. Survival work
and WHEAT purchases remain ahead of hiring.

## Usage

Fake-manager plumbing smoke (no model, no engine needed to build the agent):

```bash
python -m executor_v0.smoke --seed 7 --manager fake --opponent pass
```

First real checkpoint smoke (requires `kaggle_environments` 1.32.x installed;
the harness never installs anything and exits 3 with a SKIP message when the
package is absent):

```bash
python -m executor_v0.smoke --seed 7 --manager checkpoint \
    --checkpoint /path/to/best.pt --device cpu --seat 0 --opponent pass
```

Kaggle submission factory (build once at module import, no network access):

```python
from executor_v0 import make_agent

agent = make_agent(checkpoint="/path/to/best.pt", device="cpu", seat=0)
# or inject any PlanProvider:
# agent = make_agent(provider=my_provider, seat=0)

def act(obs):
    return agent(obs)
```

CLI flags (see `python -m executor_v0.smoke --help`): `--seed`,
`--manager {fake,checkpoint}`, `--checkpoint PATH`, `--device`,
`--seat {None,0,1}` (derived from the observation when omitted),
`--opponent {pass}`.

## Validation status (2026-08-25)

- 249 tests passed in the 2026-08-23 sweep.
- Live encoder parity on synthetic and real replay observations.
- Determinism coverage at every layer.
- 719/720-turn replay-observation plumbing smoke: zero illegal shapes, zero
  fallback errors. Counterfactual actions were NOT executed by an engine —
  this proves shape/state robustness only, not game legality or quality.
- V0.7 real BC-E fixed-plan validation, accepted/rejected changes, bounded
  panel, artifact hashes, and caveats are recorded in
  [`research/EXECUTOR_V07_FINAL.md`](../research/EXECUTOR_V07_FINAL.md). The
  validation checkpoint was externally supplied read-only from
  `C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`
  (variant E, epoch 27) and remains uncommitted; it is intentionally absent
  from this isolated worktree.
