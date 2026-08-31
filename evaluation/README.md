# Asymmetric Evaluation Harness

`evaluation.agent_match` runs two independent primitive controllers through
the scalar `fast` or `official` backend. Controller A's result margin and W/L/T
are always reported from A's perspective. A panel runs every seed in both seat
orientations by default and writes a strict JSON result envelope from
`evaluation.cli`.

## Controllers

- `bc@PATH`: BC-E Torch checkpoint, loaded through
  `bc_manager_jax.checkpoint.load_torch_checkpoint(..., model_variant="E")`.
- `bc-legacy@PATH`: explicitly load an unversioned/legacy E checkpoint with
  `E_LEGACY` semantics; corrected `bc@PATH` rejects that mismatch.
- `ppo@PATH`: detached PPO snapshot, loaded through existing
  `load_ppo_snapshot`.
- `ppo-legacy@PATH` or `snapshot-legacy@PATH`: explicitly load a legacy PPO
  snapshot; corrected snapshot kinds reject the mismatch.
- `external@PATH`: local file or extracted bundle, loaded in a child process.
- `pass`: plumbing baseline.

Internal controllers use `standard_mixed` days 0-3, the existing daily manager
policy, and a seat-specific `AgentConfig`/executor factory. For example, only
the B seat can use `--b-aggressive-sell`.

## Commands

Diagnostic P3 versus BC-E with aggressive selling on BC only:

```text
python -m evaluation.cli --a ppo@promotions/promotion_003.npz --b bc@artifacts/local/bc-v1-E/best.pt --a-name P3 --b-name BC-E-aggressive --b-aggressive-sell --seeds 6000..6031 --backend fast --output artifacts/local/issue35-p3-vs-bc-aggressive.json
```

P3 versus an extracted downloaded agent:

```text
python -m evaluation.cli --a ppo@promotions/promotion_003.npz --b external@/path/to/downloaded_agent --b-entrypoint agent.py:agent --b-name downloaded-agent --seeds 7,17 --backend fast --output artifacts/local/issue35-p3-vs-external.json
```

Use `--backend official` for the pinned official backend when its verified
`kaggle_environments==1.32.7` environment is available. `--seat0-only` is
available for a one-orientation smoke; ordinary panels should retain both
orientations. A timeout is opt-in with `--external-timeout SECONDS`.

## Kaggriculture Callable Contract

Pinned `kaggle_environments==1.32.7` source supports ordinary functions with
these positional forms:

```python
def agent(observation):
    return action

def agent(observation, configuration):
    return action
```

The runtime uses `function.__code__.co_argcount` and passes the first values of
`[structified_observation, structified_configuration]`. The child mirrors this
dispatch and supplies dict-compatible attribute-access `Struct` values. The
configuration is the processed environment configuration; the resolved seed
is normally `None`, matching the competition runtime. Callable objects without
`__code__` receive both values under the pinned runtime and should be wrapped
by a function if they need only the observation.

This is confirmed from the pinned `kaggle_environments/agent.py` loader
(`args = [structify(observation), structify(self.configuration)]` followed by
`args[:agent.__code__.co_argcount]`) and Kaggriculture schema/runtime sources:
the observation has public `farms`, `market`, `town`, `day`, and `hour`, plus
seat-private `private` state.

The returned action is the official JSON-shaped object:
`{"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}`.
The harness validates only this outer shape and JSON compatibility; operation
legality remains the backend's responsibility.

## Isolation And Packaging

Each external controller gets its own persistent subprocess for one game. The
bundle is its working directory, relative files/checkpoints work from there,
and JSONL stdin/stdout carries observations, configuration, actions, and
tracebacks. A new child is used for every game, while module/global state is
preserved across turns within that game. This prevents target imports and
JAX/Python module state from entering the evaluator process, but it is **not a
security sandbox**: downloaded code can still access the host filesystem and
network available to the child.

The V0 loader supports an individual `.py` file or an extracted directory with
`file.py:callable` or `module:callable`. Arbitrary ZIP/archive packaging,
dependency installation, container isolation, and Kaggle's full submission
archive loader are intentionally unsupported; extract the bundle first and
provide its entrypoint. Directory digests include all regular files in sorted
relative-path order, excluding transient `__pycache__`, `.pyc`, and `.pyo`
files. File digests are SHA-256 over file bytes.
