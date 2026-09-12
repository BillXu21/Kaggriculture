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
decision identity/day, cached plan, and corrected-E history. It does not resume
the environment, executor queues, or in-flight engine work.

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

Reproducible local smoke command:

```text
python -m scripts.stage25_packet4_smoke --engine fast --seed 17 --days 2 --mode deterministic
```
