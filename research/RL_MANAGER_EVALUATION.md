# RL Manager Evaluation Contract

Issue #18 owns the evaluation summary and snapshot promotion decision in
`rl_manager.evaluation`. `summarize_evaluation` accepts real
`EpisodeResult`-like records, preserves raw paired margins, and reports
candidate/opponent bank distributions, margin tails, W/L/T, and both seat
orientations.

Evaluation health is separated into:

- `fatal_anomalies`: non-DONE or incomplete/invalid results, missing or
  duplicate panel results, and explicit runtime/fallback errors;
- `opening_diagnostics`: opening guard/delegation details retained for review;
- `executor_diagnostics`: compact executor/provider warning records;
- `warnings`: reserved for nonfatal notices.

The default `PromotionConfig` requires `W - L >= 6`, mean margin `> 0`, median
margin `>= 0`, and zero fatal anomalies. Informational opening/executor
diagnostics do not veto that default gate. Optional candidate mean/median bank
floors and strict diagnostic gates are configurable. `PromotionDecision`
contains all conditions, observed values, exact failed reasons, and available
policy/opponent/seed-set provenance.

The CLI applies this helper after its fixed-seed, both-seat panel and writes
the decision in the evaluation JSON. It also prints a single EVAL line with
`gate=PASS` or `gate=HOLD` and the complete reason list.
