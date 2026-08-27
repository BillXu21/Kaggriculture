# Issue #12 comment draft

Stage 6/7 cleanup evidence is recorded in
`research/EXECUTOR_V07_IDLE_CLEANUP_PASS_ONLY.md`.

The earlier optional result was invalid promotion evidence: optional tasks
were in the normal foreman pool, so ordinary assignment selection (including
underfoot selection) could change. The corrected layer runs normal dispatch
first and only replaces literal normal `PASS` actions; candidates are
recomputed each turn and never persist or enter hiring/debt/market accounting.

Official 24-game A/B/C panel: same BC-E checkpoint, official 1.32.7 backend,
`standard_mixed`, PASS, seeds
`7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`, both seats, prior-debt
suppression ON, aggressive selling ON.

| arm | mean | median | paired result | worst/best delta |
|---|---:|---:|---|---:|
| OFF | 63,592.3 | 65,509.5 | — | — |
| WATER-only | 67,712.6 | 65,242.0 | +4,120.3 / -267.5; 16W/0T/8L | -15,820 / +30,696 |
| WEED+WATER | 66,886.3 | 66,662.5 | +3,294.0 / +1,153.0; 16W/0T/8L | -16,669 / +27,821 |

Telemetry: baseline PASS `4035/3748/3558` (OFF/WATER/WEED), replacements
`0/3409/3298`, rates `0/90.96%/92.69%`, remaining PASS `4035/339/260`,
cleanup movements `0/2973/2930`, WATER interactions `0/436/210`, weed DIG
interactions `0/0/158`, and normal non-PASS changes `0/0/0`. Unaffordable
orders were `0/174/168`; animal-loss events were `0/1/0`; all games completed
719 transitions without fallback/day/status errors.

Conclusion: the layer is mechanically isolated, but strategic value is mixed
and seed-dependent. WATER-only has a slightly negative median delta;
WEED+WATER has a positive median, but both have eight losses and materially
worse tails. Neither meets the strict strong-evidence promotion rule. Cleanup
stays OFF by default; please independently review the note and ignored traces
before any promotion. Do not substitute PPO work for this review. Keep
WATER-only and WEED+WATER distinct if watering effects need separate analysis.
