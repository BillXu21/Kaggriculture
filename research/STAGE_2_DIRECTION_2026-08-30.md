# Stage 2 Direction — 2026-08-30

This document is the authoritative handoff for the next learning phase after the Stage 1 executor work and issue #28 promotion.

## Current control

Issue #28 is complete and merged through PR #29 at merge SHA `a86f228e1bb4d638c157fbbb091d588ad0f833a9`.

Final official `kaggle_environments==1.32.7` paired BC-E panel, seeds `7,17,42,123`, both seats, PASS opponent, aggressive sell-all + optional spare watering, immediate plant watering OFF vs ON, with turn trace:

- OFF mean / median bank: `69,589.625 / 68,375`
- ON mean / median bank: `72,227.375 / 76,963.5`
- paired mean / median delta: `+2,637.75 / +1,547.5`
- wins: `5/8`
- final weeds: `14.125 -> 7`
- weeds created: `51 -> 35.625`
- movement: `2,763 -> 2,647.75`
- PASS: `163.5 -> 157.75`
- unfinished work debt: `343.625 -> 267.625`
- animal escapes: `0 -> 0`
- fallback errors: `0 -> 0`

The repaired BC-E instant-sell executor at roughly `72k` official bank is now the primary control for Stage 2.

The submitted build appears functional and modestly stronger in live play, but does not materially clear the current ~600-rating region. Replay observation suggests the remaining weakness is primarily manager/economic strategy rather than executor mechanics.

Observed manager failures to track:

- retains too much idle cash instead of converting capital into useful production;
- frequently buys the third land plot on the final day, paying for capacity with essentially no remaining horizon to recover the cost.

Do **not** add executor heuristics such as a hard `no land after day X` rule or a generic cash-on-hand penalty. These are strategy/horizon-learning failures and should remain manager/RL-owned.

## Stage 2 primary objective

Build a self-play pipeline that produces repeated, measurable policy improvement over the repaired BC control. Medal placement is secondary. The important success criterion is repeated promotions and meaningful score/economic improvement from learning.

Immediate Stage 2 priorities:

1. Correct and version the E economic-history runner semantics, then retrain/evaluate checkpoints on the corrected feature distribution.
2. Add opponent information to the manager model architecture.
3. Move from stationary BC-only opposition toward current/current and current/snapshot self-play.
4. Run a bounded scratch-RL track in parallel with the BC->PPO mainline.
5. Keep exact official 1.32.7 evaluation as the promotion authority.
6. Audit fast-engine parity later; until then use fast only for throughput, smoke tests, and obvious regressions, not promotion magnitude.

## Corrected economic context

Old PPO/legacy checkpoints were trained with the historical runner-E bookkeeping bug where the current `(day, money)` pair was supplied as `economic_prev_start`, causing previous-day economic deltas to be invalid/zero. The runner-compatible submission builder deliberately reproduces that distribution for those old checkpoints.

Stage 2 should stop extending that legacy distribution:

- version the corrected economic-history input contract;
- supply the real prior-day start/bank delta and validity semantics;
- retrain BC/PPO checkpoints on the corrected distribution;
- preserve legacy runner-parity packaging only for old checkpoints that require it;
- evaluate corrected and legacy checkpoints as distinct policy identities.

## Horizon economics

The current policy's idle-cash behavior and last-day land purchases are useful diagnostics.

Make remaining horizon explicit in the manager input if it is not already sufficiently represented, e.g. `day`, `days_remaining`, and/or a normalized remaining-horizon feature. A land purchase on an early day and the same purchase on the final day must be easy for the network to distinguish.

Track late expansion as an evaluation metric, not an executor rule. Suggested diagnostic:

- count/value of `BUY_LAND` on days 27-29;
- cost of those purchases;
- resulting terminal-bank and competitive outcomes.

Do not train a generic penalty for holding cash. Cash is valuable at terminal; the desired behavior is learning opportunity cost and remaining-horizon ROI.

## Opponent-aware manager

Opponent information is a central Stage 2 model change.

Preferred shape:

`own-board/economics encoder + opponent encoder -> shared manager trunk -> daily-plan heads`

Start with public opponent information such as:

- opponent bank/money;
- unlocked land;
- farmer/hand count;
- crop occupancy/types;
- animal counts/types;
- weeds/productive occupancy;
- coarse spatial board representation where useful.

Keep the opponent pathway present in the architecture from the beginning. For economic scratch pretraining, mask/zero opponent inputs rather than changing the architecture later. This avoids an avoidable architecture/distribution break when competitive training begins.

## Scratch-RL track

A scratch manager is now considered viable because the deterministic executor and opening remove much of the low-level exploration burden. This is scratch at the daily-manager level, not primitive-action-from-scratch learning.

Run scratch in parallel with the BC->PPO mainline.

### Phase A — economic bootstrapping

- random-init manager;
- deterministic opening + current repaired executor;
- opponent encoder present but opponent inputs masked;
- optimize for economic performance/cash generation rather than W/L/T;
- no competitive terminal reward pressure initially;
- use cash/bank shaping strongly enough to learn basic capital deployment;
- first milestone: reliably reach roughly `50k-60k` economic performance.

The purpose is to test whether successive training/promotions can produce monotonic economic improvement from a random manager.

### Phase B — transition to competition

Around `50k-60k`:

- begin exposing real opponent features;
- introduce competitive terminal reward gradually;
- begin annealing dense cash shaping;
- retain explicit horizon/economic inputs;
- start current/snapshot opposition alongside stationary controls.

### Phase C — competitive objective

Around `70k-80k` economic performance:

- cash shaping should become small or zero;
- competitive terminal reward becomes the primary objective;
- use current/current and current/snapshot self-play;
- retain fixed BC control panels for regression and attribution.

The exact thresholds are curriculum gates to tune empirically, not hard mechanics rules.

## BC->PPO mainline

The reliable path remains corrected-distribution BC -> PPO. Maintain this track even while testing scratch.

Use the repaired ~72k BC agent as the frozen control. Evaluate retained checkpoints frequently and keep separate identities for:

- economic-best;
- competitive-best;
- latest;
- promoted snapshot.

Do not use hard KL rejection as the default PPO control; retain the previously preferred soft target-KL/epoch-stop behavior unless new evidence requires otherwise.

## Self-play progression

Once corrected-distribution training is functional:

- keep frozen BC-E as a stable control/opponent;
- add current/current economic games;
- add current/snapshot competitive games;
- promote snapshots based on official fixed panels and competitive evidence;
- let the league grow only as repeated promotions justify it.

The target is repeated self-play improvement, not one lucky checkpoint.

## Fast engine status

Issue #28 demonstrated that fast-engine score magnitude cannot currently be trusted for promotion decisions:

- initial fast OFF->ON mean delta was about `+16.9k`;
- corrected official 1.32.7 OFF->ON mean delta was about `+2.6k`;
- before the WHEAT reserve fix, official exposed six animal escapes that the fast panel did not expose.

Therefore:

- fast remains useful for throughput and bounded sanity checks;
- official 1.32.7 remains the promotion authority;
- run a dedicated closed-loop official-vs-fast parity audit later;
- locate the first observation/action/next-state divergence under a frozen deterministic policy;
- do not block immediate Stage 2 RL work on this audit unless a fast-only behavior contaminates training correctness.

## Stage 3 — learned low-level execution research

A separate BC model for primitive execution/task assignment is **not** a Stage 2 priority.

Potential future research direction:

- retain deterministic legality and hard safety constraints;
- train a learned foreman/ranker over legal candidate tasks/actions given the current primitive state and daily manager plan;
- learn worker-task assignment, routing/order preferences, and micro-efficiency from replay data;
- do not replace starvation FEED, crop-survival WATER, dependency legality, or other hard mechanical invariants with unconstrained BC.

This work is explicitly deferred to Stage 3. Revisit after the current Stage 2 manager/self-play work and after the planned coding-model/tooling refresh next week; do not start implementation now.

## Near-term execution order

1. Finish recording issue #28 as the new executor/control baseline.
2. Correct/version E economic history.
3. Add opponent pathway/masking to the manager architecture.
4. Retrain a corrected-distribution BC/PPO mainline.
5. Launch bounded scratch economic curriculum in parallel.
6. Add current/current + current/snapshot self-play and repeated promotions.
7. Audit fast-engine parity as a separate workstream.
8. Defer learned low-level BC/foreman research to Stage 3.
