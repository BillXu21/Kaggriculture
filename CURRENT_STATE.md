# Kaggriculture Current State

Last updated: 2026-08-20

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **game understanding and action/observation research before implementation**
- Competitive implementation/training: not started
- Latest confirmed upstream package: `kaggle-environments 1.32.7`
- Exact local/vendored engine lock: not yet established
- Training direction: **behavior cloning first, then PPO/RL refinement; self-play only after simpler controlled improvement is demonstrated**
- Primary competition goal: build a self-play/refinement pipeline that produces meaningful, measurable improvement over a competent starting policy across multiple promotions; a medal is not the primary objective.

## Current Research Direction

The project is intentionally not locking a neural architecture, observation schema, or action abstraction yet.

Current work is to understand:

- how strong public agents actually make money;
- what production/logistics decisions matter strategically;
- how the nonlinear shared market and randomized shops change those decisions;
- what information a learned policy genuinely needs;
- which decisions should remain learned versus handled mechanically;
- what public notebooks/replays can provide as BC demonstrations and evaluation opponents.

`research/ACTION_OBSERVATION_V0.md` contains earlier scratch design ideas and is **not an authoritative locked interface**.

## Initial RL Scope

The first RL milestone is deliberately simpler than the eventual competitive problem.

Start from a competent behavior-cloned policy and ask whether RL can improve management of:

- the acting player's own farm and private state;
- production and reinvestment;
- labor/logistics decisions represented by the eventual chosen action interface;
- the shared market;
- town/shop demand;
- endgame realization/liquidation.

Do **not** require explicit opponent modeling, opponent hidden-inventory inference, or deliberate adversarial market attacks in the first milestone. A controlled/frozen opponent may still affect the shared market naturally.

Opponent-board features may even be withheld from the earliest experiment if that produces a cleaner learning test. Rich opponent modeling is a later-stage extension after BC-to-RL improvement is demonstrated.

See `.agents/notes/implemented/2026-08-20-stage-initial-rl-without-opponent-modeling.md`.

## Training Progression

Current high-level direction:

1. Study and archive strong current public agents/replays.
2. Choose an observation/action representation only after understanding their decision structure.
3. Behavior-clone a competent starting policy.
4. Verify strong closed-loop BC rollouts, not only teacher-forced accuracy.
5. Demonstrate RL improvement against a frozen/controlled opponent over held-out seeds.
6. Expand to a frozen opponent panel.
7. Only after those stages learn reliably, introduce changing opponents/population/self-play and measure promotions with broad cross-play evaluation.

The project should not add another layer of RL complexity until the simpler stationary problem underneath it demonstrably learns.

## Current Engine/Economics

1.32.6:

- town center consumes once/day at flat 1x;
- town shops are sampled with replacement;
- duplicate shops consume independently;
- total shop instances are capped at 8.

1.32.7:

- carrot, tomato, and egg use scarcity-side `hinge` curves;
- these products can become extremely valuable only in favorable randomized shop/scarcity regimes;
- selling adds supply and therefore erodes scarcity, so spot price times inventory is not realizable liquidation value;
- opponent production affects the same shared market even if the early learner does not explicitly model the opponent.

See `MECHANICS.md` for exact parameters and regression points.

## Useful External Research

- `diffmap/kaggicultureRL` contains a serious Rust batch simulator, replay/BC infrastructure, parity/fuzz tests, and feature-engineering ideas. Its engine is currently pinned to 1.32.6 and lacks the 1.32.7 `hinge` market shape, so it cannot be adopted unchanged.
- Current public RL discussion shows non-transitive matchup structure among strong route/meta-agents, reinforcing the need for cross-play evaluation rather than a single scalar strength metric.

## Immediate Priorities

1. Continue building an intuitive strategic/economic understanding of strong Kaggriculture play.
2. Inspect current strong notebooks and replays to identify the real decision surfaces and action distributions.
3. Compare candidate action-control scopes and observation representations without locking one prematurely.
4. When implementation begins, evaluate whether to vendor/port the `diffmap/kaggicultureRL` Rust engine to 1.32.7 and require parity before training.
5. Preserve durable conclusions as Agent Notes as they become decisions rather than recording every theory-crafting idea.

## Known Risks

- Spending too long on architecture before proving that learning actually occurs.
- BC memorizing turn-indexed scripts rather than learning recoverable state-conditioned behavior.
- An action abstraction that is either too primitive for available compute or so high-level that RL becomes cosmetic.
- Observation state aliasing that creates apparently contradictory PPO gradients.
- Non-transitive opponent strategies making simple Elo/latest-checkpoint promotion misleading.
- Context loss across chats wasting major amounts of work; GitHub continuity remains mandatory.

## Do Not Forget

- Read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, relevant Agent Notes, `research/RL_DESIGN.md`, and the latest `HISTORY.md` entry before substantial work.
- Treat current action/observation documents as research unless a decision note explicitly locks them.
- Preserve raw data/configs/hashes so representations can change without rerunning expensive collection.
- Before any expensive run, record exact command/config, code+engine hashes, seeds, opponent pool, expected outputs, and stop conditions.
