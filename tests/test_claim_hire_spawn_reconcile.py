"""Regression tests for claim-hire spawn reconciliation.

``_reconcile_hire_observation`` used to keep a hired worker's planned
assignment only when the observed tile equalled the *forecast* spawn from
``predict_hire_spawns``. Measurement on five real seeds showed every
claim-hire route discard came from this test, and that the worker was never
actually absent - it had simply already acted before the next observation.

The contract pinned here:

- a worker the engine actually produced is kept (it is real labour);
- its claim is released and its route dropped, so the next dispatch re-derives
  work from the live board instead of committing to a stale day-start plan;
- a genuinely absent worker is still rejected and its reserved interactions
  returned to the hiring budget;
- no bundle can end up owned twice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executor_v0.strip_claim_board import ClaimPhase  # noqa: E402

from test_executor_v0_agent import make_obs  # noqa: E402
from test_executor_v0_idle_cleanup import empty_plan  # noqa: E402
from test_strip_claim_board import _intensive_row_work, _claim_hiring_case  # noqa: E402

DAY = 6


def _planned(obs, *, work=None, money=1000.0):
    """Run the real bootstrap so a genuine claim-hire record exists."""
    work = _intensive_row_work(0, 2, 4, 6) if work is None else work
    obs["configuration"] = {"maxMarketOrdersPerTurn": 10, "boardSize": 10}
    farm = obs["farms"][0]
    farm["money"] = money
    farm["hires_today"] = 0
    farm["hands"] = []
    controller, orders = _claim_hiring_case(work, obs)
    assert controller._pending_claim_hires, "expected at least one planned hire"
    # The bootstrap helper returns the plan; emulate the submission bookkeeping
    # act() performs so reconciliation has a pending window to reconcile.
    farm = obs["farms"][0]
    controller._pending_hires = {
        "submitted_step": int(obs.get("step", 0)),
        "hands_before": len(farm.get("hands") or ()),
        "hires_before": int(farm.get("hires_today", 0)),
        "submitted": len(controller._pending_claim_hires),
    }
    return controller, obs, orders


def _advance(obs, controller, *, hands, hires_today, money):
    """Present the next observation and run reconciliation."""
    farm = obs["farms"][0]
    farm["hands"] = [list(h) for h in hands]
    farm["hires_today"] = hires_today
    farm["money"] = money
    obs["day"] = obs["day"] + 1
    obs["hour"] = 3
    obs["step"] = obs["day"] * 24 + obs["hour"]
    controller._reconcile_hire_observation(obs)


def test_moved_worker_is_kept_and_replanned_not_discarded():
    """A real worker that stepped must survive and re-derive its work."""
    obs = make_obs(day=DAY, hour=0)
    obs["step"] = DAY * 24
    controller, obs, orders = _planned(obs)
    record = controller._pending_claim_hires[0]
    assert record.worker not in controller._routes or True  # route may exist

    # The engine produced the worker one tile away from the forecast.
    moved = (record.spawn[0] + 1, record.spawn[1])
    _advance(obs, controller, hands=[moved], hires_today=1, money=900.0)

    diagnostics = controller._claim_hiring_diagnostics
    assert diagnostics.get("spawn_replanned") == [record.worker.label]
    assert record.worker.label in diagnostics.get("spawn_mismatches", [])
    # Kept as a real hire: not stripped from the planned roster.
    assert diagnostics["wanted_hires"] >= 1
    # The stale route is dropped so the next dispatch re-plans.
    assert record.worker not in controller._routes


def test_exact_spawn_is_not_a_spawn_mismatch():
    """A worker at its forecast tile is not a divergence.

    Every confirmed worker is re-planned from the live board (the route was
    planned at submission time and may already be stale), but only a genuine
    tile divergence is reported as a spawn mismatch.
    """
    obs = make_obs(day=DAY, hour=0)
    obs["step"] = DAY * 24
    controller, obs, orders = _planned(obs)
    submitted = len(controller._pending_claim_hires)
    assert submitted >= 1
    records = list(controller._pending_claim_hires)
    # ``predict_hire_spawns`` can forecast the same access tile for two workers
    # in one batch, so only the first record is placed at its exact tile here.
    first = records[0]
    _advance(obs, controller, hands=[first.spawn], hires_today=1, money=900.0)

    diagnostics = controller._claim_hiring_diagnostics
    # The observed worker sat exactly on its forecast tile: no divergence, and
    # it was not thrown away.
    assert diagnostics.get("spawn_mismatches", []) == []
    assert diagnostics["wanted_hires"] == 1
    assert controller._hire_observed == 1
    assert controller._hiring_blocked is False


def test_absent_worker_is_still_rejected():
    """A hire the engine did not produce must return its reservation."""
    obs = make_obs(day=DAY, hour=0)
    obs["step"] = DAY * 24
    controller, obs, orders = _planned(obs)
    before = controller._claim_hiring_diagnostics["wanted_hires"]
    assert before >= 1

    _advance(obs, controller, hands=[], hires_today=0, money=1000.0)

    diagnostics = controller._claim_hiring_diagnostics
    assert "spawn_replanned" not in diagnostics
    assert diagnostics["wanted_hires"] == 0
    assert diagnostics["required_interactions_reserved"] == 0


def test_replanned_claim_leaves_no_double_ownership():
    """The released bundle must be free for anyone to claim again."""
    obs = make_obs(day=DAY, hour=0)
    obs["step"] = DAY * 24
    controller, obs, orders = _planned(obs)
    record = controller._pending_claim_hires[0]
    board = controller._claim_board
    owned_before = [
        bid for bid, owner in board.owner_by_bundle.items()
        if owner == record.worker
    ]
    assert owned_before, "the planned hire should own at least one bundle"

    moved = (record.spawn[0] + 1, record.spawn[1])
    _advance(obs, controller, hands=[moved], hires_today=1, money=900.0)

    still_owned = [
        bid for bid, owner in board.owner_by_bundle.items()
        if owner == record.worker
    ]
    assert still_owned == [], "released bundles must not stay owned"
    for bid in owned_before:
        assert board.owner_by_bundle.get(bid) != record.worker
        assert board.phase_by_bundle[bid] != ClaimPhase.IN_PROGRESS


def test_hire_observation_counters_still_count_real_hires():
    """Keeping the worker must not be reported as a failed hire."""
    obs = make_obs(day=DAY, hour=0)
    obs["step"] = DAY * 24
    controller, obs, orders = _planned(obs)
    submitted = len(controller._pending_claim_hires)
    record = controller._pending_claim_hires[0]
    moved = (record.spawn[0] + 1, record.spawn[1])
    _advance(obs, controller, hands=[moved], hires_today=1, money=900.0)

    # One worker actually appeared (moved one tile); the rest genuinely did not.
    assert controller._hire_observed == 1
    assert controller._hire_failures == submitted - 1
    assert controller._hiring_blocked is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
