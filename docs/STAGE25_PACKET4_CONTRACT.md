# Stage 2.5 Packet 4 — manager-to-executor lifecycle contract

This document is authoritative for the Packet 4 integration seam. Packets 1–3
remain authoritative for action classes, physical/curriculum support, corrected
E inputs, and native checkpoint metadata. Packet 4 stops before PPO, trajectory
migration, parallel-owner wiring, shortfall accounting, league logic, routing
optimization, and large evaluations.

## Persistent owner and decision lifecycle

One episode/seat lifecycle owner holds the crop-capacity ledger `K[5]`, the last
accepted decision identity/day, the resolved daily-plan cache, and the corrected
E daily-start history. The externally supplied-class and local native-inference
paths use this same owner. A first manager boundary initializes `K` from the
observed post-opening crop counts; opening length is supplied by configuration.

For a decision key `(episode, seat, day, policy identity)`, the owner encodes
pre-decision `K` and the authoritative observation, validates all nine classes
against the exact physical/curriculum support, and then commits
`K' = K + classes[4:9] - 100` once. Validation is atomic: an invalid,
conflicting-duplicate, out-of-order, or terminal submission does not consume
the decision, advance history/RNG, or change `K`. Repeated reads of an already
resolved day are benign and return the same plan; duplicate action submission
is an error.

Lifecycle export/import is versioned and contains only lifecycle state: `K`,
decision identity/day, cached plan, corrected-E history, and the effective
curriculum. It does not resume the environment, executor queues, or in-flight
engine work.

## Canonical live-observation boundary

Both the Stage 2.5 live encoder and the executor consume observations produced
by the runner/backend canonicalization seam
(`oracle.backend.canonical_observations`, which reuses the public
`EngineBackend.canonical_state` mapping). That boundary replaces each farm view
with the backend-canonical official-form tiles and resolves one absolute
lifecycle step per observation:

- an explicit valid `step` is preserved;
- when `step` is absent (official 1.32.7 omits it for the non-acting seat), the
  pinned engine convention `day*24 + hour` is derived, and `day`/`hour` are
  validated rather than silently defaulted;
- caller-owned observations are never mutated.

Fast-engine `age` plant/animal tiles are folded into the canonical
`planted_day`/`placed_day` field inside
`replay_daily.lifecycle.canonical_tile`'s derived view; a contradictory
`placed_day`/`age` pair raises. The strict canonical storage schema is
unchanged: the alias is removed only from the derived view. The authoritative
`replay_daily.lifecycle.animal_placed_day` helper backs both derived lifecycle
timing and executor care decisions, so a canonical `placed_day` observation and
an equivalent `age` observation decide identically.

## Lowering and executor profile

Accepted Stage 2.5 classes lower through the versioned daily-plan adapter:
land and animal values are absolute targets; crop values are the committed
persistent goals. CARE, fertilizer, and sell fields are zero transport
scaffolding. The explicit Stage 2.5 executor profile enables mechanical care,
mechanical fertilizer, aggressive deterministic liquidation, and records all
effective settings, including expansion-veto flags. Requested and feasible
plans remain separate diagnostics. Execution outcomes never rewrite classes or
`K`.

Existing V0/E providers and profiles retain their behavior. The Stage 2.5
profile must fail loudly when required upkeep/liquidation behavior is absent;
it must not silently change strategic protection settings.

## Native and external inference

Native inference loads the Packet 3 JAX checkpoint and uses the corrected-E live
observation contract, with explicit deterministic mode for smoke tests and
stable decision keys for stochastic mode. External parent responses carry
classes through a framework-neutral path and do not require JAX in the worker.
One policy call covers the complete daily action.

There is exactly one effective curriculum. When a native checkpoint is attached
and the provider was given no explicit curriculum, the provider adopts the
checkpoint curriculum before sampling or mutating state. An explicitly supplied
curriculum must agree with the checkpoint exactly; a mismatch raises before the
native policy is called or any lifecycle field changes. External class providers
keep their explicit curriculum. The effective curriculum is used for sampling,
validation, provenance, and the versioned lifecycle export/import snapshot.

Importing `rl_manager.stage25_provider` stays framework-neutral. The executor
package initializer is lazy (PEP 562), so importing the provider first never
imports Torch/JAX, while the real executor package/API still imports normally
afterward. The external provider path installs no synthetic `executor_v0`
package.

## Packet 5 seams

Before accepting the next boundary decision, Packet 5 may assess outgoing
commitments using the pure `stage25_crop_shortfall_v1` helper. The assessment
belongs after outgoing execution is measured and before the next plan is
accepted. Terminal closure assesses the final outgoing decision once. Packet 4
does not track commitments, estimate shortfall from goal-minus-occupancy, or
apply a live shortfall penalty.

## Bounded evidence

Packet 4 validation uses targeted lifecycle/executor tests and a tiny native
checkpoint through the actual provider/executor path across a manager-day
boundary. The smoke records engine identity and fixed seeds. Its result is
plumbing evidence only, not competitive evidence.

Reproducible local smoke commands:

```text
python -m scripts.stage25_packet4_smoke --engine fast --seed 17 --days 2 --mode deterministic
python -m scripts.stage25_packet4_smoke --engine fast --seed 0 --days 2 --mode stochastic
python -m scripts.stage25_packet4_smoke --engine fast --seed 1 --days 2 --mode stochastic
python -m scripts.stage25_packet4_smoke --engine official --seed 17 --days 2 --mode deterministic
python -m scripts.stage25_packet4_smoke --engine official --seed 17 --days 3 --mode stochastic
```

The stochastic runs are the bounded animal-placement cases: the fast seed-1 run
and the official 3-day run place animals, and the smoke reports
`buy_animal_orders`/`final_seat_animals`/`max_seat_animals` so coverage is
machine-checkable. Every run executes both seats and reports the crossed
boundaries, steps, engine identity, and the effective Stage 2.5 executor
profile.
