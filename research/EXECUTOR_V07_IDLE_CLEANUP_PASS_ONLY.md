# Executor V0.7 Stage 6/7: PASS-Only Idle Cleanup Evidence

- Date: 2026-08-27
- Branch: `executor-v07-fixed-plan`
- Evidence HEAD: `1a0ac65` (`executor: distinguish water-only and weed-first idle cleanup modes`)
- Engine: official Kaggriculture backend `1.32.7`
- Checkpoint: BC-E variant E, `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`
- Panel: `standard_mixed`, PASS opponent, seeds
  `7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`, seats `0,1`,
  prior-debt suppression ON, aggressive selling ON, 24 games per arm
- Artifacts: `artifacts/executor-v07-official-off.json`,
  `artifacts/executor-v07-official-water-only.json`, and
  `artifacts/executor-v07-official-weed-water.json`
- Status: **evidence recorded; cleanup remains OFF by default; independent review required**

## Why the earlier optional result was not valid promotion evidence

The earlier optional-watering experiment was a flawed mixed-pool comparison:
OFF ended at mean bank `63,592.3`, while the optional arm ended at `60,948.5`
(delta `-2,643.8`, `8W/16L`). Optional tasks were appended to the normal
foreman task pool, so the same assignment pass could let optional work compete
with and alter ordinary work. In particular, the foreman's first **underfoot**
selection is part of normal dispatch; treating optional tasks as ordinary
foreman candidates allowed the optional arm to change normal worker assignment
rather than merely replace an otherwise literal PASS. That result cannot
answer the intended question, and is not used below as a quality estimate.

## PASS-only flow now measured

Each primitive turn follows this sequence:

1. Regenerate the normal task set from the current observation, including the
   existing survival filter and normal priority ordering.
2. Run the normal foreman first, with no optional cleanup candidates.
3. If enabled, regenerate optional candidates from that same current
   observation and apply them only to workers whose normal result is exactly
   `PASS`; a normal non-PASS action is authoritative and cannot be replaced.
4. Discard cleanup claims after the turn. They are not persisted, added to
   normal task accounting, used for hiring, or used to create work debt.

The candidate set is recomputed every turn. Weed cleanup is ordered ahead of
optional watering within the optional layer. Normal maintenance/productive
work, starvation preemption, and the `WATER:<y>,<x>` Stage 4 dependency remain
unchanged; optional watering uses the isolated `WATER_OPTIONAL:<y>,<x>` key.

### Mode definitions and flags

| Mode | Selection flag | Candidates |
|---|---|---|
| OFF | both flags false (production default) | none |
| WATER-only | `optional_spare_watering=True` | safe-to-defer optional WATER only |
| WEED+WATER | `optional_idle_cleanup=True` | optional DIG of weeds first, then safe-to-defer WATER |

The weed-first mode is the superset when both flags are supplied. Diagnostics
record the resolved `cleanup_mode` as `none`, `water_only`, or `weed_water`,
alongside the two input/resolved configuration flags. No source behavior was
changed by this documentation pass; the implementation is already committed
in `0b596ff`, `b082a4d`, and `1a0ac65`.

## A/B/C result summary

Arm A is OFF, B is WATER-only, and C is WEED+WATER. All three arms used the
same official backend, checkpoint, opening, opponent, seed/seat pairs, prior-
debt setting, and aggressive-sell setting.

| arm | mode | mean bank | median bank | min | max | paired mean delta vs A | paired median delta vs A | wins | ties | losses | worst delta | best delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | OFF | 63,592.3 | 65,509.5 | 47,290 | 74,151 | — | — | — | — | — | — | — |
| B | WATER-only | 67,712.6 | 65,242.0 | 56,611 | 85,879 | +4,120.3 | -267.5 | 16 | 0 | 8 | -15,820 | +30,696 |
| C | WEED+WATER | 66,886.3 | 66,662.5 | 55,843 | 80,504 | +3,294.0 | +1,153.0 | 16 | 0 | 8 | -16,669 | +27,821 |

The paired result table is the complete 24-game bank table. A positive delta
means the optional arm beat the OFF game with the same seed and seat.

| seed | seat | A OFF | B WATER-only | B-A | C WEED+WATER | C-A |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 50,655 | 69,914 | +19,259 | 63,743 | +13,088 |
| 7 | 1 | 47,290 | 77,986 | +30,696 | 75,111 | +27,821 |
| 17 | 0 | 63,139 | 68,290 | +5,151 | 68,735 | +5,596 |
| 17 | 1 | 63,956 | 68,266 | +4,310 | 66,618 | +2,662 |
| 42 | 0 | 58,652 | 64,608 | +5,956 | 67,614 | +8,962 |
| 42 | 1 | 64,092 | 62,468 | -1,624 | 64,610 | +518 |
| 123 | 0 | 49,304 | 58,638 | +9,334 | 62,299 | +12,995 |
| 123 | 1 | 49,304 | 58,638 | +9,334 | 62,299 | +12,995 |
| 2026 | 0 | 69,206 | 77,407 | +8,201 | 74,252 | +5,046 |
| 2026 | 1 | 71,315 | 76,610 | +5,295 | 74,581 | +3,266 |
| 1013 | 0 | 72,512 | 60,280 | -12,232 | 55,843 | -16,669 |
| 1013 | 1 | 72,425 | 56,611 | -15,814 | 56,272 | -16,153 |
| 1022 | 0 | 57,309 | 64,636 | +7,327 | 62,883 | +5,574 |
| 1022 | 1 | 59,247 | 65,848 | +6,601 | 61,368 | +2,121 |
| 1003 | 0 | 66,927 | 62,651 | -4,276 | 66,707 | -220 |
| 1003 | 1 | 68,461 | 63,921 | -4,540 | 63,690 | -4,771 |
| 1026 | 0 | 61,348 | 83,777 | +22,429 | 80,504 | +19,156 |
| 1026 | 1 | 63,732 | 85,879 | +22,147 | 80,365 | +16,633 |
| 1011 | 0 | 70,073 | 63,018 | -7,055 | 67,919 | -2,154 |
| 1011 | 1 | 70,073 | 63,018 | -7,055 | 67,919 | -2,154 |
| 1024 | 0 | 74,151 | 58,331 | -15,820 | 61,288 | -12,863 |
| 1024 | 1 | 68,459 | 72,092 | +3,633 | 60,107 | -8,352 |
| 1019 | 0 | 67,632 | 71,788 | +4,156 | 73,051 | +5,419 |
| 1019 | 1 | 66,953 | 70,428 | +3,475 | 67,493 | +540 |

## Cleanup telemetry and invariants

The telemetry aggregates below sum all 24 games in each arm. Replacement rate
is `cleanup_replacements / baseline_pass_worker_actions`.

| arm | baseline PASS worker actions | cleanup replacements | replacement rate | remaining PASS | cleanup movement actions | optional WATER interactions | weed DIG interactions | normal non-PASS changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A OFF | 4,035 | 0 | 0.00% | 4,035 | 0 | 0 | 0 | 0 |
| B WATER-only | 3,748 | 3,409 | 90.96% | 339 | 2,973 | 436 | 0 | 0 |
| C WEED+WATER | 3,558 | 3,298 | 92.69% | 260 | 2,930 | 210 | 158 | 0 |

The per-game diagnostics also report `cleanup_mode`, mode flags, assignments,
pending/debt accounting, market accounting, and survival data. Across the
panel, unaffordable-market-order counts were A/B/C = `0/174/168`, and animal-
loss event counts were `0/1/0`. All 72 games completed their 719 transitions;
there was no fallback/day/status error, and `normal_non_pass_actions_changed`
was zero in every arm. These are mechanical invariants, not a claim that the
strategic result is robust.

## First-divergence examples from official panel timelines

These are the first per-turn timeline-state differences from A for the named
paired game; they are not official-engine parity divergences. The panel
artifact records the state summary at the first difference, while the complete
ignored artifacts remain available for follow-up trace inspection.

1. **Seed 7, seat 0, d4h15 (step 112):** A and B had the same money (`285`),
   no market order, no feed shortage, and no animal change. B nevertheless had
   `available_wheat 4` and `carried_wheat 4` after the turn, versus A's zero;
   B finished at `69,914` versus A's `50,655` (`+19,259`). C has the same
   first state divergence (`available_wheat/carried_wheat 0 -> 4`) but finished
   at `63,743` (`+13,088`). The surprising point is that the first divergence
   is a carried-feed state change with no money or animal change, not a direct
   sale or purchase.
2. **Seed 1013, seat 0, d4h15 (step 112):** the first B and C differences
   are again the same `0 -> 4` available/carried WHEAT change with unchanged
   money (`287`), no market order, and no feed shortage at that row. Here the
   eventual effect reverses: B finished `60,280` versus A `72,512`
   (`-12,232`), while C finished `55,843` (`-16,669`). This is a direct
   example of the same mechanically safe-looking idle replacement having
   seed-dependent strategic value.

## Interpretation and decision

Both optional arms have positive mean deltas and win a majority of paired
games, but each also has eight losses and a materially worse tail near
`-15k`/`-16.7k`. WATER-only has a slightly negative median delta (`-267.5`),
whereas WEED+WATER has a positive median delta (`+1,153.0`). Neither arm
meets the strict strong-evidence promotion rule cleanly: positive mean and
median, majority wins, no materially worse tail, and no survival regression.

The layer is mechanically safe as an idle replacement layer: normal
non-PASS actions were never changed, normal task/debt/hiring/market accounting
was isolated, and no broad normal-work regression was observed. Its strategic
value remains mixed and seed-dependent, as shown by the loss tails and the
paired first-divergence examples. **Leave cleanup OFF by default and do not
promote it without independent review.** If the strategic effect of watering
itself needs to be separated from weed reclamation, retain the distinct
WATER-only and WEED+WATER modes rather than collapsing them.

Next step: independent review of this evidence and the ignored official traces;
do not attach this layer to PPO or run a new optimization cycle as a substitute
for review.
