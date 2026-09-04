# 2026-09-04 PPO curriculum findings and next decisions

This note records the durable conclusions from the recent scratch-PPO curriculum experiments, the executor land fix, and the new manager-intent telemetry. It is intended to be folded into the root durable-decision index when that file is next edited normally.

## D-054 - Persist manager intent separately from executor outcomes

- **Status:** active observability contract.
- **Decision:** Curriculum and PPO interpretation must include bounded manager-intent diagnostics in addition to realized farm state. Under low telemetry, retain one compact record per manager `(episode, seat, day)` and aggregate requested crop targets, per-species distributions, target-vector changes, action-max saturation, day/late-game intent, unresolved crop deficits, and requested-vs-realized crop shortfall. Do not infer that a good realized farm or improving bank means the manager itself issued a coherent plan.
- **Semantics:** action-max saturation uses authoritative `ManagerConfig.count_max`. Vector changes compare only within one `(episode, seat)` trajectory. `unresolved_generator` is the latest/end-of-day unresolved state rather than a cumulative count of every transient deficit. For pre-terminal days, `achieved_final` is the crop state observed at the next day boundary; day 29 is finalized explicitly from the terminal observation. Shortfall is neutral telemetry, not automatically executor failure.
- **Rationale:** the executor mechanically projects, reconciles, clips, schedules, and suppresses manager requests. Without intent telemetry, a random or wildly infeasible manager can look strategically competent because deterministic code salvages the plan.
- **Evidence:** commit `9e2369427fd4ba888b554759dbfd47e3d0fce4e4` added the bounded low-telemetry funnel. On the first `land<=3` rollout it showed requested crop total mean `258.39` / median `258`, mean `4.973` of 5 species requested, only `0.81%` of crop components exactly at action max, and zero rows with all five crop heads at max. Per-species medians were roughly `50-55` with broad p10/p90 ranges, so the failure was not simple maximum saturation. `99.965%` of manager rows had a remaining unresolved crop deficit, mean realized/requested crop fraction was `19.53%`, and requested totals stayed essentially flat from days `4-19` through `28-29` (~`257-260`) rather than showing late liquidation taper. This is consistent with poorly learned, diffuse absolute crop-count semantics rather than a coherent crop policy.
- **Revisit when:** the action representation no longer uses absolute targets, the executor no longer projects/reconciles manager intent, or richer accepted-action attribution supersedes these aggregates.

## D-055 - Unmask land directly after the executor fix; do not default to targeted land exploration

- **Status:** active curriculum rule.
- **Decision:** With the D-053 land-capital correction present, land curriculum stages should normally be opened by unmasking the next land count directly. Do not use targeted epsilon land exploration by default merely to make `BUY_LAND` happen. Keep targeted exploration as an explicit diagnostic when policy sampling itself is in question.
- **Rationale:** before D-053, manager land requests were often erased by executor capital/work-debt behavior, so policy-side exploration could not solve the physical conversion bottleneck. After the correction, the same latent policy immediately buys newly unmasked land early and nearly universally, showing that manager sampling was not the limiting factor.
- **Evidence:** old-executor stage-2 from the same severe-u15 parent and seed `43043` bought land 2 in only `19/768` seats (`2.47%`) at mean day `19.26`. With D-053 and otherwise matching parent/seed/curriculum, the first rollout bought land 2 in `766/768` seats (`99.74%`) at mean day `8.18` / median day `8`. Earlier targeted land-2 exploration sampled the target on roughly `54-58%` of manager rows while physical purchase still stayed around `1-2%`, confirming the old manager->executor conversion bottleneck. A later land-3 unmask likewise produced `766/768` three-plot farms with mean plot-3 purchase day `10.60`.
- **Revisit when:** a new executor/capital rule again breaks requested-land conversion, or a future policy architecture shows genuinely low land-target sampling under an otherwise functioning executor.

## D-056 - Use BC initialization as the main PPO line; treat scratch PPO as an ablation

- **Status:** active RL direction.
- **Decision:** Stop using scratch-initialized PPO as the main evidence that the full daily manager has learned coherent strategy. Preserve scratch runs as diagnostics/ablations, but return the primary self-play pipeline to the original project goal: initialize from a competent BC model, then use PPO to improve it. Do not interpret rising terminal bank alone as proof that every manager head has learned meaningful semantics.
- **Rationale:** constrained scratch PPO clearly learned some useful behavior, but D-054 telemetry shows that the executor can turn highly infeasible manager plans into competent-looking farms. This creates strong action aliasing: many different bad absolute requests compile to similar feasible behavior and receive similar terminal credit. Observed bank gains can therefore come from a subset of heads or executor-mediated economics while other heads remain effectively unlearned.
- **Evidence:** the severe one-plot run improved from roughly `40k` at u1 to `43-44k` by u15. Executor-fixed two-plot continuation recovered from mean bank `32,961` at u1 to `36,061` at u10 while keeping near-universal early land-2 purchase. Widening animal caps to `goose<=2,cow<=6,sheep<=4` for 10 updates improved bank to roughly `40-41k` but left final animals pinned around `4.91-4.95`. On the first three-plot rollout, `766/768` seats owned three plots, plot 3 was bought at mean day `10.60`, final crops reached `59.94`, but mean bank collapsed to `4,693` while crop requests remained grossly infeasible (D-054).
- **Interpretation boundary:** these results prove terminal-bank PPO can optimize some behavior under constrained curricula; they do not prove scratch PPO learned coherent crop counts, animal expansion, liquidation, or a complete farm plan.
- **Revisit when:** a scratch architecture with a materially smaller/local action space (for example delta targets), stronger representation, or another demonstrated credit-assignment fix learns manager intent that passes the D-054 feasibility diagnostics.

## D-057 - Keep BC-E PPO on its stored E contract before the next architecture change

- **Status:** active experiment sequencing.
- **Decision:** The next serious PPO quality experiments initialize from the real promoted BC-E checkpoint and use its stored own-only E architecture/config unchanged for the first comparison. Do not silently enable opponent-board tokens, J/JE joint decision tokens, or delta targets in that baseline. After the BC-E-initialized PPO baseline is measured with manager-intent telemetry, JE is the preferred next architecture candidate to investigate through a JAX port and E->JE transfer/distillation; delta targets remain a separate action-parameterization redesign requiring a compatible new training/transfer path.
- **Rationale:** current JAX PPO supports V0/E only and the RL seam intentionally enforces the own-only E contract. The BC-E checkpoint gives meaningful initialization for the existing absolute heads; changing inputs and decoder structure at the same time would confound whether BC initialization itself solves the scratch semantic/exploration failure. JE is attractive because it adds cross-task coordination while preserving the existing output semantics, whereas delta targets change the action contract itself.
- **Evidence:** D-027 records that J/JE remain PyTorch-only while BC-E is the JAX production target; current PPO uses independent crop/animal/fertilizer/care categorical heads. The 2026-09-04 scratch diagnostics above show that the present absolute heads can remain poorly learned from terminal-bank PPO even while realized bank improves through the executor.
- **Revisit when:** BC-E manager-intent telemetry is itself incoherent, BC-E PPO fails to improve under a controlled run, or a tested JE/delta transfer path is available with exact checkpoint/input compatibility.

## Supporting artifact identities

- Executor land-priority fix: `b4d9e600738b114373d738cefbf06233b61768ef`.
- Manager crop-intent telemetry: `9e2369427fd4ba888b554759dbfd47e3d0fce4e4`.
- Severe scratch parent retained at `F_severe_s1_u15_no_targetkl/final.npz`.
- Executor-fixed land-2 continuation retained at `J_land2_from_severe_u15_executorfix/final.npz`.
- Wider-animal land-2 continuation retained at `K_land2_animals_2g6c4s_from_Ju10/final.npz`.
- Land-3 continuation is diagnostic evidence; do not promote it solely from the first rollout.
