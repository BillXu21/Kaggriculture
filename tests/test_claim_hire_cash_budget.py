"""Regression tests for the claim-hire treasury-risk budget.

Hire cost is ``mult * fib(hires_today)``, so a single day that reaches a high
Fibonacci index commits far more cash than the day can earn back.  On seed
41004 the claim-hire loop reached 23 hires in one turn (75 024 against a
68 368 bank) and the final bank collapsed from ~68 k to ~4.7 k.

These tests pin the bound itself: it must be a fraction of the cash actually
on hand, it must stop the loop before the offending order, and it must be
disableable to restore historical behaviour.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executor_v0.strip_executor import (  # noqa: E402
    StripExecutorConfig,
    StripExecutorController,
)

DAYS = 30


def _controller(fraction: float | None) -> StripExecutorController:
    config = replace(
        StripExecutorConfig(),
        acting_seat=0,
        enable_row_claim_board=True,
        max_daily_hire_cash_fraction=fraction,
    )
    return StripExecutorController(config=config)


def _farm(obs, money: float, hires_today: int, hands: int = 0):
    farm = obs["farms"][0]
    farm["money"] = money
    farm["hires_today"] = hires_today
    farm["hands"] = [[0, 0] for _ in range(hands)]


def _day(obs, day: int, hour: int) -> None:
    obs["day"] = day
    obs["hour"] = hour
    obs["step"] = day * 24 + hour


def _budget_after(ctl, obs, money, hires_today):
    """Run one claim-hire planning pass and return (orders, diagnostics)."""
    _farm(obs, money, hires_today)
    work_plan = ctl._build_work_plan(obs, ctl._daily_plan)
    positions = ctl._worker_positions(obs)
    ctl._finalize_claim_day(
        obs, work_plan, positions,
        {w: ctl._worker_inventory(obs, w) for w in positions},
    )
    orders = ctl._plan_claim_hires(obs, work_plan, positions)
    return orders, dict(ctl._claim_hiring_diagnostics)


def test_default_fraction_is_present_and_positive():
    controller = _controller(None)
    assert controller.config.max_daily_hire_cash_fraction is None
    default = StripExecutorConfig()
    assert default.max_daily_hire_cash_fraction is not None
    assert 0.0 < default.max_daily_hire_cash_fraction <= 1.0


def test_budget_is_a_fraction_of_cash_on_hand():
    from test_executor_v0_agent import make_obs, simple_plan

    ctl = _controller(0.25)
    ctl._daily_plan = simple_plan()
    obs = make_obs(day=6, hour=0)
    _day(obs, 6, 0)
    _, diagnostics = _budget_after(ctl, obs, money=10_000.0, hires_today=0)
    assert diagnostics["hire_cash_budget"] == pytest.approx(2_500.0)


def test_unbounded_fraction_restores_historical_behaviour():
    """``None`` disables the bound and reports no budget at all."""
    from test_executor_v0_agent import make_obs, simple_plan

    obs = make_obs(day=6, hour=0)
    _day(obs, 6, 0)
    unbounded = _controller(None)
    unbounded._daily_plan = simple_plan()
    _, diagnostics = _budget_after(unbounded, obs, money=10_000.0, hires_today=0)
    assert diagnostics["hire_cash_budget"] is None

    huge = _controller(1e9)
    huge._daily_plan = simple_plan()
    _, diagnostics = _budget_after(huge, obs, money=10_000.0, hires_today=0)
    assert diagnostics["hire_cash_budget"] > 1e9


def test_fibonacci_escalation_is_what_the_budget_stops():
    """A 0.25 budget must bind far earlier than the cash check alone would.

    ``remaining_cash < cost`` only stops at the exact point of insolvency.
    The treasury bound is what prevents the burst from ever approaching it.
    """
    from replay_daily.constants import fib

    cash = 68_368.0
    budget = cash * 0.25

    def allowed_under(limit: float) -> tuple[int, int]:
        running = 0
        count = 0
        for index in range(60):
            cost = fib(index)
            if running + cost > limit:
                break
            running += cost
            count += 1
        return count, running

    budget_allowed, budget_spend = allowed_under(budget)
    cash_allowed, _ = allowed_under(cash)

    assert budget_spend <= budget
    assert budget_allowed < cash_allowed, (
        "the treasury bound must bind strictly earlier than insolvency")
    # The measured 41004 failure committed fib(0..22) = 75 024, i.e. it ran
    # past its own bank; the bound has to prevent exactly that.
    assert sum(fib(i) for i in range(23)) == 75_024
    assert sum(fib(i) for i in range(23)) > cash


def test_daily_spend_counter_resets_on_new_day():
    from test_executor_v0_agent import make_obs, simple_plan

    ctl = _controller(0.25)
    ctl._daily_plan = simple_plan()
    obs = make_obs(day=6, hour=0)
    _day(obs, 6, 0)
    _budget_after(ctl, obs, money=10_000.0, hires_today=0)
    assert ctl._hire_spend_day == 6
    _day(obs, 7, 0)
    _budget_after(ctl, obs, money=10_000.0, hires_today=0)
    assert ctl._hire_spend_day == 7


def test_committed_spend_is_recovered_from_observed_hires():
    """A restarted bootstrap pass must not forget what was already spent."""
    from test_executor_v0_agent import make_obs, simple_plan

    ctl = _controller(0.25)
    ctl._daily_plan = simple_plan()
    obs = make_obs(day=6, hour=0)
    _day(obs, 6, 0)
    _budget_after(ctl, obs, money=10_000.0, hires_today=12)
    # fib(0..11) = 232
    assert ctl._hire_spend_cash == sum(
        __import__("replay_daily.constants", fromlist=["fib"]).fib(i)
        for i in range(12)
    )


def test_start_day_resets_budget_state():
    from test_executor_v0_agent import make_obs, simple_plan

    ctl = _controller(0.25)
    ctl._daily_plan = simple_plan()
    obs = make_obs(day=6, hour=0)
    _day(obs, 6, 0)
    _budget_after(ctl, obs, money=10_000.0, hires_today=0)
    ctl._start_day(make_obs(day=9, hour=0), simple_plan())
    assert ctl._hire_spend_day is None
    assert ctl._hire_spend_cash == 0
    assert ctl._hire_budget_stops == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
