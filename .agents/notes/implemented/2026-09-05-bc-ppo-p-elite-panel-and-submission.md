# 2026-09-05 BC-E PPO P lineage, elite-personality panel, and submission checkpoint

This note records the endpoint of the first successful BC-E -> PPO improvement line before Stage 2.5 representation/strategy work. It extends D-058/D-059 without changing their historical conclusions.

## D-060 - Promote the N -> O -> P BC-E PPO lineage as the current RL reference

- **Status:** promoted experimental reference.
- **Decision:** Preserve the BC-E-initialized N -> O -> P lineage as the current RL baseline. N established that `lr=1e-5` can improve economics without destroying BC semantics; O extended that stable regime into a plateau; P increased the continuation learning rate modestly to `3e-5` and remained economically coherent for five updates. Do not replace P with scratch-derived parents.
- **Artifacts:**
  - N: `N_bce_fullspace_lr1e5_s43047`, 10 updates.
  - O: `O_bce_fullspace_lr1e5_from_Nu10_s43048`, 10 updates from N u10.
  - P: `P_bce_fullspace_lr3e5_from_Ou10_s43049`, 5 updates from O u10.
- **N evidence:** update 1 -> update 10 moved mean bank from about `59.6k` to `63.6k`, median from about `60.7k` to `64.7k`, and p10 from about `42.6k` to `48.5k`, while crop requests stayed near the BC physical scale.
- **O evidence:** the continuation remained stable but largely plateaued. O u10 had mean bank about `64.0k`; first-five -> last-five improvement was only about `+462` mean / `+210` median / `+1,278` p10. Peak productive squares averaged about `56.94` with median `57`, indicating a real capacity/utilization plateau rather than continued large farm-scale growth.
- **P evidence:** the `3e-5` continuation survived five updates without the catastrophic semantic blow-up seen at `3e-4`. P u5 produced a stochastic rollout maximum final bank of `104,378`. Aggregate farm behavior remained in the three-plot regime and productive occupancy stayed coherent; however the large maximum is not itself treated as a robust strategy result (D-061).
- **Interpretation:** the first BC-init PPO line is now competent and stable enough to serve as the control for architecture/action-space changes. The main remaining bottleneck is no longer basic semantic collapse; it is strategy/capacity headroom around roughly the high-50s productive squares.

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

## D-063 - P post-u5 deterministic Torch deployment passed exact-archive official smoke

- **Status:** submission artifact validated locally; live leaderboard result pending.
- **Decision:** Submit the post-update-5 P manager (`P_bce_fullspace_lr3e5_from_Ou10_s43049/final.npz`, PPO step `3900`) through the legacy runner-compatible deployment path. Export only `state.params["base"]` to a normal `bc_manager_checkpoint_v1` Torch E checkpoint; the PPO value head is not a deployment component. Preserve `E_LEGACY`, hardcoded `strict=True`, `aggressive_sell_all=False`, optional spare watering, and vendored market pricing dependency.
- **Export verification:** Torch export round-trips through the repository's strict Torch -> JAX converter exactly back to the P base parameters. Sell-quantity parameters remain equal to the frozen BC-E snapshot as required by the current PPO contract.
- **Archive verification:** the exact built tarball was extracted into a fresh temporary directory and loaded without repository-source imports. The historical lazy `BUY_PRODUCT` branch resolved through `executor_v0/_submission_market.py`; a forced WHEAT purchase probe returned cost `125.0`.
- **Official-engine smoke:** six exact-archive games against a PASS opponent, both seat orientations on seeds `17`, `42`, and `2026`, all ended `DONE` with zero runtime/status failures. Candidate banks were `68,913`, `66,868`, `67,045`, `74,103`, `77,188`, and `74,845`; mean `71,493.67`, median `71,508`, min `66,868`, max `77,188`.
- **Boundary:** these PASS-opponent banks validate runtime/deployment health, not competitive strength. The live Kaggle leaderboard result must be recorded separately once observed.

## Stage 2.5 direction

- Use P as the main control/reference checkpoint for subsequent representation work.
- Mine full Fourth Quadrant replays as the primary strategic reference until a clearly stronger public/dominant agent appears.
- Focus replay mining on the observed P bottleneck: land timing, crop/animal count changes, productive-square trajectory and peak density, care/fertilizer execution, hiring, selling/liquidation, and daily cash/productivity curves.
- Use those replay distributions to inform the hybrid action redesign rather than choosing delta supports heuristically. Current working hypothesis: crop targets should become local/signed deltas; land should become a local HOLD/BUY_NEXT decision; animal representation should be chosen after replay contraction/expansion analysis; care and fertilizer remain absolute/daily quantities unless replay evidence supports a simpler coverage-style action.

## Supporting artifact identities

- Prior durable BC-E PPO note: `.agents/notes/implemented/2026-09-04-ppo-curriculum-findings.md`.
- P run directory: `P_bce_fullspace_lr3e5_from_Ou10_s43049`.
- P post-u5 deployment source: `P_bce_fullspace_lr3e5_from_Ou10_s43049/final.npz`.
- Policy that generated the P update-5 stochastic rollout is the pre-u5 checkpoint `ppo_update_000003.npz`; do not confuse that rollout policy with post-u5 `final.npz`.
- Submission builder for these legacy-E PPO checkpoints: `tools/build_runner_compatible_submission.py`.
