# Kaggriculture Historical Record

This file is append-only except for correcting factual errors. New entries should be added in reverse chronological order under dated headings, or consistently at the end if the project later chooses chronological order.

## 2026-08-06 — Repository Initialization

### Repository

- Created private GitHub repository: `BillXu21/Kaggriculture`.
- Initialized the default branch with project documentation.
- Established continuity files to reduce context loss between chats and agents.

### Strategic Assessment

- Current game structure appears highly deterministic.
- Physical farms are separate, with limited direct interaction.
- The shared market is the primary adversarial coupling mechanism.
- Strong public leaderboard entries are currently dominated by copies or variants of a few deterministic public notebooks.
- The project will remain in planning and mechanics-tracking mode while the engine and rules continue changing.

### Initial Architecture Hypothesis

The likely useful agent design is:

1. deterministic production-route executor;
2. state-based validation and repair;
3. phase-level replanning;
4. opponent-aware market and production policy;
5. coherent expert selection;
6. optional optimization or learning at the macro level.

Primitive-action reinforcement learning is not currently prioritized.

### Research Findings Carried Into the Repository

The following findings were established before repository initialization and should be reverified against the exact live engine before implementation:

- Two players each manage a separate 10×10 farm.
- Matches span thirty days with twenty-four turns per day.
- Banked money determines final reward.
- Unsold inventory has no terminal value.
- Crop and animal schedules are largely deterministic.
- The market is shared and uses inventory-dependent pricing.
- Some daily events are driven by the episode seed.
- Public state exposes enough opponent farm information to support strategy fingerprinting and supply forecasting.
- Strong public strategies use mixed industrial production rather than simple single-crop loops.

### Public Baseline Direction

The first competitive reference should be a strong public deterministic route, preserved with:

- source URL or notebook identity;
- download date;
- immutable file hash;
- engine version assumptions;
- any local modifications;
- known performance evidence.

The project will not rely on redistribution-license concerns as a reason to avoid downloading publicly available Kaggle notebook artifacts, but provenance and third-party boundaries should still be tracked accurately.

### Compute and Workflow Lessons Imported From Pokémon TCG Work

- Chat-context loss can cause stale configuration reuse and wasted compute.
- Every expensive run must be specified in a durable file before execution.
- Current state must remain concise and authoritative.
- Full history must preserve failed experiments, commands, hashes, and output paths.
- Evaluation should not depend on a single seat, weak opponents, or unversioned artifacts.

### Files Established

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `HISTORY.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `AGENTS.md`
- `research/README.md`
- `.gitignore`

### Next Actions

1. Establish the exact current engine identity.
2. Archive important public notebooks and agents.
3. Catalog major strategy families.
4. Define the initial fixed-seed, seat-swapped evaluation protocol.
5. Delay competitive implementation until those contracts are recorded.
