# Stage 2.5 strip executor and scratch self-play — durable note

Date: 2026-09-15

Detailed status: `research/STAGE25_STRIP_EXECUTOR_STATUS_2026-09-15.md`.

## Durable decisions

1. **Executor philosophy:** the strip executor is a mechanical realization layer, not a strategic economic safety layer. It should execute manager requests efficiently and legally even when those requests are economically bad. Do not reintroduce broad debt/capital/overextension protections just to preserve behavior of policies trained under the legacy executor.

2. **Routing geometry:** keep fixed horizontal five-tile row ownership for now. `StripRoute` remains generic so a later route generator can replace rows if controlled evidence shows a routing bottleneck. New replay evidence suggesting compact/vertical-first territories is not yet sufficient to trigger redesign.

3. **Selling ownership:** selling is temporarily executor-owned through strip `aggressive_sell_all`. This is a bridge, not the final strategic selling architecture. The immediate correction is to insta-sell finished outputs only and retain WHEAT and FERTILIZER.

4. **Old-policy evaluation:** P-final under strip is a compatibility/plumbing smoke, not a promotion benchmark. P-final was trained behind legacy executor safeguards, so severe economic degradation under the new executor does not by itself imply a routing defect.

5. **Known open executor seams:** worker-carried product delivery/deposit, bootstrap timing/same-day liquidity, and eventual route geometry remain candidates for later work. Do not implement them without a measured failure that isolates the seam.

6. **Near-term training direction:** because Stage 2.5 BC/outcome-proxy initialization is not ready and TPU access is time-sensitive, a bounded scratch Stage 2.5 self-play run is acceptable as an experiment. It is a plumbing/learnability/TPU-utilization run, not the canonical competitive baseline. Preserve exact source/executor/action/checkpoint identity and start with a short smoke before a long allocation.

## Evidence snapshot

- Strip Packet 5C source baseline: `8c9dd94291ee4ea92cdf6e88828e27aa09022f5f`.
- Pre-insta-sell P-final strip smoke: four games, all candidate banks `0`; historical executor arm mean bank `90,710.5` on the same two seeds / both seats.
- Per-day diagnostics showed assigned routes generally completing; the immediate collapse was cash -> hiring -> route coverage rather than obvious assigned-route traversal failure.
- Packet 5C one-game smoke confirmed actual aggressive sales (4 FERTILIZER, 7 WHEAT, 15 MELON) but still ended at bank `0`.
- Therefore insta-sell fixes a real missing liquidity seam but does not restore legacy-policy economics.

## Revisit conditions

Revisit the executor philosophy only if the manager/executor contract explicitly moves a strategic decision back into the executor. Revisit row geometry when controlled full-game or day-slice evidence shows material unfinished assigned work, excess travel, or worker-idle imbalance that is attributable to geometry rather than missing labor/cash. Revisit scratch-vs-BC initialization after the first bounded scratch run and once the BC pipeline is ready.
