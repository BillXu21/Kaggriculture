# 2026-09-05 BC-E PPO P lineage, elite-personality panel, and submission checkpoint

This note records the endpoint of the first successful BC-E -> PPO improvement line before Stage 2.5 representation/strategy work. It extends D-058/D-059 without changing their historical conclusions.

## D-060 - Promote the N -> O -> P BC-E PPO lineage as the current RL reference

- **Status:** promoted experimental reference for training/self-play; not automatically the best deployment policy.
- **Decision:** Preserve the BC-E-initialized N -> O -> P lineage as the current RL baseline. N established that `lr=1e-5` can improve economics without destroying BC semantics; O extended that stable regime into a plateau; P increased the continuation learning rate modestly to `3e-5` and remained economically coherent for five updates. Do not replace P with scratch-derived parents.
- **Artifacts:**
  - N: `N_bce_fullspace_lr1e5_s43047`, 10 updates.
  - O: `O_bce_fullspace_lr1e5_from_Nu10_s43048`, 10 updates from N u10.
  - P: `P_bce_fullspace_lr3e5_from_Ou10_s43049`, 5 updates from O u10.
- **N evidence:** update 1 -> update 10 moved mean bank from about `59.6k` to `63.6k`, median from about `60.7k` to `64.7k`, and p10 from about `42.6k` to `48.5k`, while crop requests stayed near the BC physical scale.
- **O evidence:** the continuation remained stable but largely plateaued. O u10 had mean bank about `64.0k`; first-five -> last-five improvement was only about `+462` mean / `+210` median / `+1,278` p10. Peak productive squares averaged about `56.94` with median `57`, indicating a real capacity/utilization plateau rather than continued large farm-scale growth.
- **P evidence:** the `3e-5` continuation survived five updates without the catastrophic semantic blow-up seen at `3e-4`. P u5 produced a stochastic rollout maximum final bank of `104,378`. Aggregate farm behavior remained in the three-plot regime and productive occupancy stayed coherent; however the large maximum is not itself treated as a robust strategy result (D-061).
- **Interpretation:** the first BC-init PPO line is competent and stable enough to serve as the control for architecture/action-space changes. The main remaining bottleneck is no longer basic semantic collapse; it is strategy/capacity headroom around roughly the high-50s productive squares. D-064 adds a deployment boundary: the strong stochastic policy distribution must not be silently replaced by per-head argmax at submission time.

## D-061 - Elite stochastic traces mostly regress to the same strong policy regime

- **Status:** active interpretation of stochastic PPO quality.
- **Decision:** Do not select/promote PPO checkpoints from single spectacular stochastic harvests. Evaluate elite stochastic row-ID streams across fresh environment seeds. The current BC-E PPO lineage is substantially more consistent than the old scratch policy, where one lucky fixed stream could score roughly twice the ordinary policy level.
- **Method:** for each of `N_u5`, `N_u10`, `O_u5`, `O_u10`, and `P_u5`, take the top two harvest-bank stochastic traces and replay the same stochastic personality on common fresh CPU seeds. Preserve checkpoint, source episode index, seat, and policy identity so manager sampling uses the same row-ID random stream while environment seeds vary.
- **Fresh-seed panel results (12 fresh seeds per personality):**

| trace | rank | TPU harvest | CPU anchor | fresh mean | median | p10 | min | max | mean peak productive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O u10 | 2 | 95,434 | 91,842 | 70,646 | 71,391 | 55,834 | 52,646 | 94,957 | 57.8 |
| P u5 | 1 | 104,378 | 63,738 | 69,411 | 67,090 | 61,735 | 55,215 | 84,861 | 59.0 |
| P u5 | 2 | 94,299 | 94,296 | 69,062 | 67,478 | 51,206 | 42,988 | 95,015 | 56.2 |
| N u10 | 1 | 94,329 | 94,328 | 66,853 | 64,718 | 49,880 | 47,078 | 86,608 | 58.1 |
| O u5 | 1 | 95,068 | 95,130 | 65,304 | 65,403 | 55,448 | 47,799 | 78,738 | 56.6 |
| N u10 | 2 | 90,752 | 81,805 | 65,099 | 63,609 | 49,736 | 41,048 | 84,447 | 57.1 |
| O u10 | 1 | 97,028 | 96,248 | 63,860 | 64,748 | 44,125 | 32,716 | 86,303 | 57.8 |
| O u5 | 2 | 94,606 | 75,418 | 63,782 | 67,911 | 46,648 | 39,628 | 79,722 | 56.3 |
| N u5 | 2 | 93,168 | 90,789 | 63,062 | 62,537 | 49,642 | 44,847 | 82,666 | 54.4 |
| N u5 | 1 | 94,852 | 91,938 | 56,152 | 56,512 | 35,162 | 30,815 | 91,089 | 54.8 |

- **Stage-level interpretation:** averaging the two selected personalities per stage gives fresh-seed means of roughly `59.6k` (N u5), `66.0k` (N u10), `64.5k` (O u5), `67.3k` (O u10), and `69.2k` (P u5). The corresponding average p10 rises from roughly `42.4k` at N u5 to roughly `56.5k` at P u5. This is weakly/non-monotonically sampled evidence, but it is directionally consistent with stronger and better-compressed stochastic behavior through the lineage.
- **Main conclusion:** elite harvest selection no longer uncovers a hidden lottery-ticket policy whose normal behavior is far weaker. Most selected personalities regress toward a common `~60-70k` economic regime. This is a positive stability result for self-play/promotion: improvements appear distributed across stochastic behavior rather than concentrated only in rare traces.
- **Capacity conclusion:** fresh elite peak productivity remains mostly around `55-59`, so bank improvement is not explained by a breakthrough to a much denser `65-75` productive-square farm. Timing, composition, labor/capital efficiency, and liquidation likely account for much of the economic gain.

## D-062 - Cross-device stochastic anchors are not guaranteed exact even with exact checkpoint identity

- **Status:** diagnostic caveat.
- **Decision:** Do not require exact TPU -> CPU reproduction of a stochastic trace as a validity condition once checkpoint identity, row-ID identity, and initial trajectory agreement are independently verified. Use one device/runtime consistently for comparative fresh-seed panels.
- **Evidence:** N u5 rank-1 used `ppo_update_000003.npz` (post-u4/pre-u5), state step `624`, and reconstructed fingerprint `6f3397df72736c6ee4f7b6cefa34b017984f45161ba3b429e7051aac3aeccd8c`, exactly matching the update-4 learner fingerprint in the original log. Original and CPU replay matched daily bank, land, and productive state through day 24, then first diverged in the day-25 state; final bank was `94,852` TPU vs `91,938` CPU. P u5 rank-1 showed a much larger `104,378` TPU vs `63,738` CPU anchor gap, while P u5 rank-2 reproduced almost exactly (`94,299` vs `94,296`).
- **Interpretation:** this is not a generic CPU performance penalty. The mixed anchor behavior is consistent with occasional device-level floating-point differences crossing a stochastic categorical decision boundary, after which state trajectories diverge. The `104,378` P harvest remains a real observed rollout but is not evidence of a robust `100k` policy regime.

## D-063 - P post-u5 deterministic Torch deployment passed runtime smoke but was not a valid competitive proxy

- **Status:** corrected deployment interpretation.
- **Original decision:** submit the post-update-5 P manager (`P_bce_fullspace_lr3e5_from_Ou10_s43049/final.npz`, PPO step `3900`) through the legacy runner-compatible deployment path, exporting only `state.params["base"]` to a normal Torch E checkpoint while preserving `E_LEGACY`, `strict=True`, `aggressive_sell_all=False`, optional spare watering, and the vendored market dependency.
- **Export verification:** Torch export round-tripped exactly through the repository's strict Torch -> JAX converter. Sell-quantity parameters remained equal to the frozen BC-E snapshot as required by the PPO contract.
- **PASS smoke:** six exact-archive official-engine games against a PASS opponent, both seat orientations on seeds `17`, `42`, and `2026`, all ended `DONE`; banks were `68,913`, `66,868`, `67,045`, `74,103`, `77,188`, and `74,845`, mean `71,493.67`.
- **What this actually proved:** archive/runtime health only. It did not establish competitive policy strength because PASS removes most shared-market competition and the deployed Torch manager used deterministic per-head argmax rather than PPO's stochastic action distribution.
- **Competitive exact-archive result:** P-vs-P on eight seeds (`16` seat-results) produced mean `42,626.56`, median `41,045`, p10 `26,675`, min `22,723`, max `68,862`. The first live Kaggle validation then scored only about `24k`, which is consistent with the local deterministic competitive tail rather than evidence of a mysterious sandbox failure.
- **Boundary:** never again treat a strong PASS-opponent archive smoke as evidence that the submitted competitive policy matches the stochastic PPO policy being promoted.

## D-064 - Match PPO deployment semantics to the stochastic policy that was actually optimized

- **Status:** active deployment contract.
- **Decision:** For PPO checkpoints trained/evaluated in stochastic mode, deployment must preserve stochastic sampling semantics (or separately prove that deterministic mode is competitive). Do not silently replace the learned distribution with independent argmax/threshold decisions at submission time.
- **Source fact:** PPO stochastic mode samples 17 categorical decisions plus 54 sell-presence Bernoullis. Deterministic mode uses categorical argmax and sell-presence `logit > 0`. The Torch BC manager decoder likewise uses argmax for crop/animal/land/fertilizer/care and a fixed presence threshold, so a plain Torch export changes the effective policy even when every weight is bit-exact.
- **Why this matters:** PPO optimized expected return under samples from `pi(a|s)`, not the return of the joint per-head mode. With many conditionally independent heads, taking every marginal mode simultaneously can create a joint plan that is uncommon under the training distribution. Shared-market interactions also make two rigid deterministic copies materially harsher than stochastic self-play.
- **Evidence:** stochastic elite personalities from N/O/P commonly generalized in the `60-70k` regime, while post-u5 P deterministic exact-archive self-play averaged only `42.6k`. The live `~24k` validation is therefore a policy-contract mismatch first, not evidence that the P weights failed to export.
- **Next implementation rule:** submission testing must explicitly name the inference contract: deterministic argmax, fresh stochastic sampling, or fixed stochastic personality. Promotion tables must not compare them as if they were interchangeable.

## D-065 - Require competitive exact-archive smoke after runtime smoke

- **Status:** active release gate.
- **Decision:** Keep the existing fresh-subprocess exact-archive runtime smoke (including both seats, lazy market dependency, strict mode, and status-history checks), but add a competitive exact-archive panel before any serious submission. PASS-opponent games are necessary for packaging/runtime validation but insufficient for strategy validation.
- **Minimum evidence:** exact tarball, fresh subprocess with repository paths excluded, both seat orientations or self-play, multiple seeds, all statuses ACTIVE/DONE, and competitive bank mean/median/tail reported. A policy expected to live near `60k` should not be submitted from a PASS-only `70k` smoke if its competitive exact-archive mean is near `40k`.
- **Reason:** the market is shared; opponent buying/selling materially changes prices/inventory and therefore realized economics. PASS produces an artificially easy market and can hide brittle deployment behavior.

## D-066 - Use O-u10 rank-2 fixed stochastic personality as the practical deployment fallback

- **Status:** current practical submission candidate; live score pending.
- **Decision:** For the immediate submission fallback, prefer the already-tested O-u10 rank-2 stochastic personality over the spectacular P `104,378` harvest. The goal is a boring robust `~60k+` deployment, not reproducing a single lucky maximum.
- **Source policy:** `O_bce_fullspace_lr1e5_from_Nu10_s43048`; update-10 economics were generated by the post-u9 checkpoint `ppo_update_000008.npz`. The selected rank-2 personality had original harvest `95,434`, CPU anchor `91,842`, and prior fresh-seed panel mean `70,646`, median `71,391`, p10 `55,834`, min `52,646`.
- **Deployment method:** export the post-u9 JAX base exactly to Torch, then preserve the selected fixed stochastic row-ID personality by precomputing its JAX Gumbel noise (categoricals) and uniform variates (sell-presence Bernoullis) for seat/day rows and embedding those small arrays in the archive. Torch inference adds the fixed Gumbel noise before argmax and uses `u < sigmoid(logit)` for sell presence. Sell quantities remain the frozen-E behavior.
- **Construction checks:** JAX categorical sampling was decomposed exactly as `argmax(logits + gumbel)` for the tested keys; Bernoulli sampling was decomposed exactly as `uniform < sigmoid(logit)`; the O checkpoint Torch export round-tripped exactly back to the JAX base parameters.
- **Competitive exact-archive panel:** 12 seeds / 24 seat-results produced mean `61,926.33`, median `60,833`, p10 `46,264`, min `38,284`, max `83,368`. This clears the immediate `~60k` local target and is materially stronger than the deterministic P archive under competitive self-play.
- **Submission boundary:** this is a pragmatic fixed-personality deployment, not the final stochastic-inference architecture. Live Kaggle result remains to be recorded separately.

## Stage 2.5 direction

- Use P as the main **training/self-play control checkpoint**, but do not assume its deterministic Torch mode is the best deployable policy.
- Treat O-u10 rank-2 fixed stochastic personality as the current practical submission fallback until live evidence or a better stochastic deployment path supersedes it.
- Mine full Fourth Quadrant replays as the primary strategic reference until a clearly stronger public/dominant agent appears.
- Focus replay mining on the observed policy bottleneck: land timing, crop/animal count changes, productive-square trajectory and peak density, care/fertilizer execution, hiring, selling/liquidation, and daily cash/productivity curves.
- Use those replay distributions to inform the hybrid action redesign rather than choosing delta supports heuristically. Current working hypothesis: crop targets should become local/signed deltas; land should become a local HOLD/BUY_NEXT decision; animal representation should be chosen after replay contraction/expansion analysis; care and fertilizer remain absolute/daily quantities unless replay evidence supports a simpler coverage-style action.
- Separately, redesign the production inference/deployment contract so stochastic PPO performance is measured and shipped under the same semantics.

## Supporting artifact identities

- Prior durable BC-E PPO note: `.agents/notes/implemented/2026-09-04-ppo-curriculum-findings.md`.
- P run directory: `P_bce_fullspace_lr3e5_from_Ou10_s43049`.
- P post-u5 deployment source: `P_bce_fullspace_lr3e5_from_Ou10_s43049/final.npz`.
- Policy that generated the P update-5 stochastic rollout is the pre-u5 checkpoint `P_bce_fullspace_lr3e5_from_Ou10_s43049/ppo_update_000003.npz`; do not confuse that rollout policy with post-u5 `final.npz`.
- O fixed-personality deployment source: `O_bce_fullspace_lr1e5_from_Nu10_s43048/ppo_update_000008.npz` (post-u9 / pre-update-10 rollout policy).
- Legacy-E submission builder baseline: `tools/build_runner_compatible_submission.py`.
- First P deterministic exact archive was useful as a runtime/argmax diagnostic but is not the preferred competitive submission artifact.
