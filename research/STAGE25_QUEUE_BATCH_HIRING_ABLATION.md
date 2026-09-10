# Stage 2.5 queue/batching/hiring evaluation handoff

**Execution status: no runs executed.** This note and its notebook are evaluation handoff artifacts only. No executor, test, or sharder implementation was changed.

## Contract

Use a pinned clean source SHA, the fast engine, four evaluator child
processes, and bounded CPU threads (`numThreads=1` plus the existing BLAS/JAX
thread guards). Keep the stochastic P-final candidate and deterministic frozen
BC-E opponent on `E_LEGACY` with the committed `standard_mixed` opening. Keep
the established common flags:

```text
--underfoot-first --deadline-safe-planting --deadline-safe-hiring
--variants combined combined_wheat3
```

Starvation workload visibility is explicitly OFF: do not pass
`--starvation-workload-visibility-repair`, and verify the manifest records
`false`.

The three candidate-only arms are:

| arm | candidate executor configuration |
| --- | --- |
| A | existing persistent worker queues + `--schedule-informed-hiring` + `--schedule-hiring-economic-repair` |
| B | A plus `--queue-ownership-repair`, `--batch-reserved-supplies`, and `--underfoot-queue-insertion` |
| C | B plus `--suppress-expansion-from-prior-debt off` (blanket prior-debt expansion veto disabled) |

All arms use the same frozen checkpoints, source SHA, opening, variants,
engine, process/thread settings, master seed `25`, and seed order. The
existing queue+scheduling fallback remains part of the pinned source and must
not be removed or silently replaced.

## Seeds and pairing

The full competitive panel preserves the established ordered 32-seed list:

```text
144368101, 309507, 615013, 918079, 1221109, 1524137, 1827169,
2130193, 2433221, 2736251, 3039283, 3342311, 3645341, 3948373,
4251401, 2112243121, 1470672056, 995106988, 1303793286, 521973470,
107449192, 768565387, 1370134739, 2090797777, 425789796, 1027359148,
1688475343, 261654734, 863224086, 1524340281, 97519672, 699089024
```

Run every panel seed in both orientations. The notebook uses the existing
identity formula:

```text
episode_id = master_seed * (2 * len(full_seed_list)) + 2 * seed_index + seat
```

The named supplemental capture slice is `[1392524882, 814255690,
229020805, 1962565411]`, both seats, using the full ordered 36-seed list
(the 32 panel seeds followed by these four cases) with `--game-filters`. This
preserves their original identities within the targeted run and avoids
renumbering them through a shortened seed list. The two contrast cases are
reported as contrasts, not blanket failures; the two regression-watch cases
are retained as targeted regressions, not automatic promotion vetoes.

Only these pairwise comparisons are allowed:

1. A vs B;
2. B vs C.

Do not produce or use an A-vs-C conclusion. Do not select a combined leader
until an operator directly confirms A after reviewing A’s full-panel and
targeted evidence.

## Required evidence and reporting

Each arm must retain:

- the exact source SHA/tree/diff and source-file hashes;
- immutable PPO and BC-E checkpoint SHA-256 hashes;
- engine provenance (`fast_env` module/version/hash where available);
- ordered seeds, both seat orientations, variants, composition, and episode IDs;
- complete sharder manifests, shard ownership, commands, logs, and resume
  validation. Resume is valid only when configuration, seed ordering,
  episode identities, and checkpoint hashes match;
- complete targeted `debug_trace.json.gz` and rollout/executor diagnostics;
- a ZIP containing the traces plus `per_call_profile_inputs.jsonl` and its
  schema/coverage manifest. These are profiling inputs only; no cProfile or
  per-call performance result is claimed by this handoff.

The competitive report must show, for each arm and for A–B/B–C deltas:

- candidate bank and opponent bank;
- margin and W/L/T;
- clustered uncertainty, resampling by seed while retaining both seats and
  both variants in each cluster;
- actual hire spending, not merely submitted HIRE orders;
- observed pickup quantities/batch sizes;
- movement, productive work, and missed maintenance;
- final and, where available, day-level crop/animal counts;
- expansion suppression totals with prior-debt/current-suppression causes;
- evaluator wall runtime and fallback errors.

Capture-derived metrics must remain unavailable when the underlying artifact
does not expose them; do not infer zeros from missing telemetry. The audit’s
movement and completion measures are diagnostic proxies and do not by
themselves establish economic benefit.

## Source guard and execution boundary

The notebook intentionally contains a `REPLACE_WITH_40_HEX_SOURCE_SHA` guard
and checkpoint-path guards. Before a real Kaggle run, bind the source SHA to
the pushed clean evaluator commit and mount immutable PPO/BC-E files. The
notebook then clones, fetches, checks out, preflights all three arms, runs the
targeted capture slice before the full panel, validates identities, produces
only A–B and B–C summaries, and bundles the handoff ZIP.

Until those cells are run against the pinned source and mounted artifacts,
there are no score, win-rate, regression, runtime, or promotion claims.
