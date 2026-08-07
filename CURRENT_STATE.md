# Kaggriculture Current State

Last updated: 2026-08-06

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Default branch: `main`
- Phase: research, mechanics tracking, and project planning
- Competitive agent implementation: not started
- Current best internal agent: none
- Current evaluation suite: not implemented
- Engine lock: not yet established
- Immediate policy: avoid expensive training or large implementation work until the engine and rules settle

## Current Understanding

Kaggriculture currently behaves mostly like a deterministic single-player production and logistics problem with a shared adversarial market.

The farms are physically separate. The main interaction channels are:

- shared product inventory and prices;
- opponent sales and purchases changing later market conditions;
- town demand consuming shared inventory;
- adapting production and sale timing to the opponent's visible farm, money, workers, land, crops, and animals.

The strongest public agents currently appear to rely heavily on fixed or replay-derived action schedules. A durable competitive agent will likely preserve deterministic execution while adding closed-loop repair and opponent-aware macro decisions.

## Working Architecture Hypothesis

1. **Deterministic route executor**
   - land purchases;
   - worker hiring;
   - crop and animal layout;
   - watering, feeding, harvesting, and collection routes;
   - baseline market orders.

2. **Closed-loop repair controller**
   - detect deviations from expected state;
   - recover from weeds, failed purchases, insufficient cash, blocked planting, shed pressure, misplaced workers, crop risk, and animal risk;
   - avoid blindly replaying invalid actions.

3. **Phase-level replanner**
   - replan at day boundaries or major state changes;
   - operate on farm-level decisions rather than unrestricted primitive actions.

4. **Opponent-aware market policy**
   - infer likely future supply from the public opponent farm;
   - alter product mix, holding, liquidation, and sale timing;
   - recognize common public strategy families.

5. **Expert selector**
   - choose among coherent full-farm plans rather than combining incompatible local actions.

## Confirmed High-Level Mechanics

- Two players, each with a separate 10×10 farm.
- Thirty days with twenty-four turns per day.
- One action per worker per turn and a bounded number of market orders.
- Final reward is banked money; unsold inventory has no terminal value.
- Product prices and market inventory are shared.
- Crop growth, animal production, labor resets, and most logistics are deterministic once actions and the episode seed are fixed.
- Some daily events are seed-driven and must be verified against the exact current engine.

Detailed mechanics and confidence labels belong in [`MECHANICS.md`](MECHANICS.md).

## Current Public Meta

- Much of the leaderboard is reportedly composed of copies or variants of a small number of public notebooks.
- Strong public strategies use industrial mixed farms, large recurring production, livestock, fertilizer, fixed routes, and carefully timed market operations.
- Exact copies should not provide a durable advantage against each other; differentiation is expected to come from repair, route selection, opponent inference, market timing, and robustness.

## Current Experiments

None. No compute experiment should begin until the initial simulator identity, evaluation protocol, and baseline provenance are recorded.

## Immediate Priorities

1. Freeze and document the exact current engine source, package version, commit, and file hashes.
2. Download and archive important public notebooks and submissions with provenance.
3. Build a public-baseline catalog describing route families and major strategy differences.
4. Define the first deterministic, seat-swapped evaluation protocol.
5. Decide the first bounded implementation packet only after the above contracts are clear.

## Initial Implementation Packet — Not Yet Started

When implementation begins, the first packet should be limited to:

- local environment smoke test;
- vendored engine/spec manifest without modifying upstream code;
- pass/random/starter wrappers;
- deterministic tournament runner with fixed seeds and seat swapping;
- JSONL results and compact summary;
- basic engine-contract regression tests;
- single-file submission export validation.

It should explicitly exclude competitive policy work and learning.

## Known Risks

- Engine and documentation drift.
- Public notebooks embedding assumptions tied to older engine versions.
- Chat-context loss causing stale configurations or duplicated compute.
- Evaluating only one seat or weak built-in agents.
- Mistaking high deterministic score for robust competitive strength.
- Terminal inventory left unsold.
- Overengineering before the source contracts stabilize.

## Do Not Forget

- Read `CURRENT_STATE.md`, `PLANS.md`, `DECISIONS.md`, `MECHANICS.md`, and the latest section of `HISTORY.md` before substantial work.
- Before any expensive run, record the exact command, engine identity, agent hash, seeds, opponent pool, expected outputs, and stop conditions.
- After a substantial session, update continuity documents before switching chats.
- Do not treat discussion claims as confirmed mechanics without source or behavioral verification.
