# Stage 2.5 prior-work-debt expansion ablation

This ablation reuses `AgentConfig.suppress_expansion_from_prior_debt`. The
historical default remains enabled (`on`). With the explicit
`--suppress-expansion-from-prior-debt off` evaluator setting, prior-day
unfinished work no longer suppresses requested `BUY_ANIMAL`, `BUILD_COOP`, or
`BUILD_PASTURE` tasks, nor `PLACE` tasks that depend on those constructions.

The executor still applies ordinary task priorities, dependencies, legality,
inventory, affordability, and market-order limits. Outstanding work remains
in the normal debt and turn diagnostics. The current-turn starvation/feed-
shortage veto is a separate mechanism and is unchanged by this setting; hiring
economics are also unchanged. The Stage 2.5 evaluator applies the ablation to
the candidate seat only, keeps the opponent on the historical default, and
records the effective setting in the manifest and capture executor identity.
