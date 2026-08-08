# Kaggriculture Current State

Last updated: 2026-08-07

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Default branch: `main`
- Phase: engine stabilization + RL interface/reward planning
- Competitive agent implementation: not started
- Current best internal agent: none
- Current evaluation suite: not implemented
- Latest confirmed upstream package line: `kaggle-environments 1.32.6`
- Live leaderboard rollout of 1.32.6: announced/rolling out; not independently server-verified yet
- Engine lock: not yet established locally or vendored
- Immediate policy: do not launch large training while Pokémon work finishes and Kaggriculture engine/rules continue settling

## Major 1.32.6 Balance Change

Upstream PR #1394 (`Kaggriculture town rebalance`) is merged.

Confirmed source changes:

1. **Town-center demand reduced**
   - `townCenterSellInterval`: 12 → 24 turns;
   - at the default 24 turns/day, town center consumes once/day instead of twice/day;
   - town-center demand is permanently 1×;
   - the former 2× after day 10 / 4× after day 20 schedule is removed.

2. **Town shops sampled with replacement**
   - each unlock samples from the full shop table rather than only unused shop types;
   - duplicate shops are allowed;
   - each duplicate shop instance consumes independently;
   - total unlocked shop instances remain capped at 8;
   - unlock cadence and per-shop consumption cadence are otherwise unchanged.

Strategic implication: town demand is weaker overall, player oversupply should move markets more, and random duplicated shop compositions create meaningfully different economic regimes across seeds.

## Current Strategic Direction

The project is now intentionally **RL-centered**, but not raw primitive-action RL.

The learned policy should own meaningful decisions such as:

- production allocation;
- land/labor investment;
- crop/animal mix;
- task assignment;
- adaptation to random shop demand;
- opponent-aware production pivots;
- market order selection and timing.

Deterministic code should primarily provide:

- mechanical legality/action masking;
- pathfinding and execution of selected worker intents;
- task persistence;
- state normalization/bookkeeping;
- minimal safety/recovery where required to prevent meaningless invalid-action cascades.

The working design is a **hierarchical intent-level RL policy** rather than a hard-coded farm strategy with a learned market add-on.

Detailed design: [`research/RL_DESIGN.md`](research/RL_DESIGN.md).

## Why Not Raw Primitive PPO

The public RL experience reported by competitors is plausible and consistent with the engine structure:

- enormous combinatorial primitive action space;
- long delayed crop/animal rewards;
- small logistical mistakes can cascade into catastrophic losses;
- random exploration must first rediscover precise deterministic maintenance routines.

The response is not to abandon RL. The current plan is to remove deterministic navigation/legality burden from the policy, bootstrap from strong public traces, and fine-tune adaptive strategy with RL.

## Proposed RL Boundary

Current leading hypothesis:

1. **worker-task policy** selects intent-level tasks such as plant/water/harvest/feed/build for target entities/tiles;
2. **deterministic executor** compiles the selected task into primitive movement + interaction actions;
3. **market policy** emits ordered market actions with learned product/type/quantity decisions;
4. optional **daily/global strategy head** changes higher-level production/resource targets;
5. policy may be recurrent to infer opponent hidden inventory/sale behavior from history.

Decision frequency is not locked. Turn-level, event-driven semi-MDP, and hybrid schemes must be compared before implementation.

## Reward Direction

Primary competitive objective should align with W/L/T rather than only absolute bank.

Current leading reward plan:

- terminal `+1 / 0 / -1` for win/tie/loss;
- investigate mathematically consistent **potential-based shaping** using estimated liquidation/future economic value;
- avoid arbitrary direct rewards such as bonuses for watering or harvesting;
- use auxiliary prediction losses for market, production, opponent behavior, and win probability instead of distorting the reward.

Discounting must respect the terminal objective; `gamma=1.0` or values extremely close to one should be evaluated rather than assumed away.

## Imitation Bootstrap Direction

Strong deterministic public notebooks are training data, not just opponents.

Likely bootstrap:

1. archive public agents with immutable provenance;
2. roll them out over many 1.32.6 seeds and opponents;
3. collect full state/action trajectories;
4. map traces into the intent-level action representation;
5. behavior-clone initial competence;
6. use PPO/self-play to depart from those scripts and adapt to stochastic shops/opponents.

This should dramatically reduce the precision/credit-assignment problem compared with learning farming from random initialization.

## Current Experiments

No training experiments are active.

Planning experiments to specify before implementation:

- shop-regime outcome variance across seeds;
- sensitivity to opponent product mix and sale timing;
- primitive vs task-level vs event-driven action abstractions;
- potential-based reward sanity checks on fixed trajectories;
- recurrent-memory value for opponent hidden-state inference;
- simulator/vectorization throughput.

## Immediate Priorities — Next Week

1. Track upstream changes and verify that 1.32.6 behavior remains stable.
2. Freeze/document exact 1.32.6 source, package version, commit, and hashes when implementation begins.
3. Enumerate exact worker and market action schemas.
4. Design the intent/task candidate generator without embedding strategy into it.
5. Choose market order/quantity representation.
6. Specify observation entities/tensors and normalization.
7. Formalize candidate potential functions and reward invariants.
8. Design BC trajectory format from public agents.
9. Design PPO/self-play curriculum and evaluation gates.
10. Recheck the engine before any large implementation/training packet.

## Known Risks

- Additional engine/balance changes.
- Accidentally moving too much strategy into the deterministic executor and leaving RL only cosmetic decisions.
- Going too far the other direction and asking RL to relearn pathfinding/legality.
- Reward shaping changing the actual competitive objective.
- BC overfitting to time-indexed deterministic public scripts.
- Public baselines becoming stale under 1.32.6 economics.
- Insufficient simulator throughput for large PPO runs.
- Hidden opponent inventory requiring memory/inference.
- Chat-context loss causing stale configurations or duplicated compute.

## Do Not Forget

- Read `CURRENT_STATE.md`, `PLANS.md`, `DECISIONS.md`, `MECHANICS.md`, `research/RL_DESIGN.md`, and the latest section of `HISTORY.md` before substantial work.
- Before any expensive run, record the exact command, engine identity, agent hash, seeds, opponent pool, expected outputs, and stop conditions.
- After a substantial session, update continuity documents before switching chats.
- Do not treat discussion claims as confirmed mechanics without source or behavioral verification.
