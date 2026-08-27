# Executor V0.7 Stage 7 Issue #12C: Opportunistic Watering Audit

- Date: 2026-08-27
- Branch: `executor-v07-fixed-plan`
- Audited repository HEAD: `c676fcf53d4064667fb454bfc20b7d5d59b7f6d6`
- Status: **mechanically correct; disabled by default; defer behavior change**
- Scope: documentation and decision only; no source, test, tool, or config patch

## Requirement-by-requirement result

| Requirement | Result | Current evidence |
|---|---|---|
| Keep hard watering distinct from opportunistic watering | **PASS** | `executor_v0/tasks.py:178-220,356-422` classifies weed-boundary WATER as `MAINTENANCE`, yield-window WATER as `PRODUCTIVE`, and safe-to-defer plants separately. |
| Generate only safe optional candidates | **PASS** | `executor_v0/tasks.py:237-278` excludes watered plants, locked/non-plant tiles, malformed plants, nonzero drought counters, and `_water_urgency(...)` yield/maintenance cases. |
| Preserve priority and starvation behavior | **PASS** | `executor_v0/tasks.py:59-64` places OPTIONAL after priorities 0–3; `executor_v0/foreman.py:320-372` makes priority dominate distance; `executor_v0/agent.py:819-838` retains survival filtering before the explicitly gated append. |
| Remain off in production by default | **PASS** | `AgentConfig.optional_spare_watering` is `False` at `executor_v0/agent.py:78-90`; dispatch appends candidates only when the flag is true at `executor_v0/agent.py:833-838`. |
| Isolate accounting and preserve traceability | **PASS** | Hiring receives `normal_dispatch_tasks` only (`executor_v0/agent.py:886-890`); focused tests verify no hire/debt/pending/missed-maintenance pollution and preserve the optional source in trace (`tests/test_executor_v0_optional_water.py:188-215`). |
| Avoid a Stage 4 dependency/key regression | **PASS** | Yield-positive harvest dependencies remain `WATER:<y>,<x>` (`executor_v0/tasks.py:436-450`); optional candidates use `WATER_OPTIONAL:<y>,<x>` only as their isolated dispatch key (`executor_v0/tasks.py:274-277`) and do not replace Stage 4 keys. |

## Test evidence

The focused optional-watering suite covers default-off behavior, enabled spare-worker
dispatch, every existing priority class, nearest selection, distinct workers,
malformed/yield exclusions, and debt isolation/traceability:

```text
python -m pytest tests/test_executor_v0_optional_water.py
```

The suite is the mechanical evidence for this audit; no full 24-game panel was run
for Stage 7.

## Bounded risk and decision

The implementation has no maximum-distance clamp. With the flag enabled, a lone
eligible optional plant can therefore receive a Manhattan route and consume walking
turns; the foreman score adds distance only after priority (`executor_v0/foreman.py:351-372`).
That is a real bounded strategic risk, not a demonstrated mechanical defect. A
distance threshold is a strategy/tuning choice, and there is no A/B evidence for
choosing one. **Do not enable optional watering or add a distance clamp tonight.**

## Exact deferred A/B task (not run tonight)

Run the same fixed 24-game panel twice from the same source/checkpoint/configuration:
seeds `7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019`, seats `0,1`,
`standard_mixed`, PASS opponent, fast backend, prior-debt suppression ON,
aggressive selling ON, turn trace ON, and 719 transitions. The only difference is
`optional_spare_watering=False` versus `True`:

```text
python -m tools.run_executor_v07_panel --checkpoint "<BC-E best.pt>" --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --output "artifacts/executor-v07-stage7-issue12c-off.json" --label "executor-v07-stage7-issue12c-off"
python -m tools.run_executor_v07_panel --checkpoint "<BC-E best.pt>" --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --optional-spare-watering --output "artifacts/executor-v07-stage7-issue12c-on.json" --label "executor-v07-stage7-issue12c-on"
```

Compare paired final banks, movement counts, missed maintenance, end-of-day debt,
fallback/per-day errors, status anomalies, and unaffordable orders. Consider a
distance clamp only after those results establish a repeatable cost/benefit; do not
choose its threshold before evidence.

**Conclusion:** the bounded implementation passes the mechanical audit and remains
default-off. Stage 7 issue #12C is deferred as strategic experimentation, with no
behavior patch in this commit.
