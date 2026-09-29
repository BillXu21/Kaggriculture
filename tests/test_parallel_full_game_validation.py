"""Focused contracts for the checkpoint-backed full-game validation panel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.agent_match import MatchResult
from scripts import parallel_full_game_validation as validation


def test_seed_spec_accepts_csv_and_inclusive_ranges():
    assert validation.parse_seed_spec("7,10..12") == (7, 10, 11, 12)


@pytest.mark.parametrize("seed_spec", ["", "7,,8", "9..7", "7,7", "-1"])
def test_seed_spec_rejects_ambiguous_or_invalid_values(seed_spec):
    with pytest.raises(ValueError):
        validation.parse_seed_spec(seed_spec)


def test_symmetric_panel_yields_one_game_per_seed():
    assert list(validation._task_iter((7, 10))) == [(0, 7, 0), (1, 10, 0)]


def test_asymmetric_panel_keeps_both_seat_orientations():
    assert list(validation._task_iter((7, 10), both_seats=True)) == [
        (0, 7, 0), (1, 7, 1), (2, 10, 0), (3, 10, 1),
    ]


def test_row_claim_flag_reaches_the_strip_executor_config():
    for enabled in (False, True):
        factory = validation._StripControllerFactory(
            checkpoint_path="policy.npz",
            checkpoint_sha256="a" * 64,
            episode_index=0,
            seed=41001,
            enable_row_claim_board=enabled,
        )
        assert factory._strip_config.enable_row_claim_board is enabled
        assert factory._strip_config.aggressive_sell_all is True
        # The harness always records the effective flag so a report can prove
        # whether row-claim ran, even when the opt-in schema omits the key.
        profile = factory.provenance["executor_profile"]
        assert profile["enable_row_claim_board"] is enabled
        # rl_manager._strip_config_json drops the opt-in key when disabled.
        serialized = profile["strip_config"]
        if enabled:
            assert serialized["enable_row_claim_board"] is True
        else:
            assert "enable_row_claim_board" not in serialized


def test_row_claim_and_no_row_claim_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        validation.main([
            "--checkpoint", "policy.npz",
            "--row-claim", "--no-row-claim",
        ])


def test_opponent_mode_rejects_unknown_value():
    # Argument validation runs before checkpoint resolution, so no file needed.
    with pytest.raises(ValueError, match="opponent must be one of"):
        validation.run_validation(
            checkpoint_path=Path(__file__).with_name("missing-policy.npz"),
            opponent="mirror",
        )


def _opening_diagnostics():
    return {
        "turns_replayed": validation.OPENING_TURNS,
        "divergence": {"occurred": False, "reason": None},
        "handoff": {
            "turn": list(validation.HANDOFF_TURN),
            "clean_handoff": True,
        },
    }


def test_full_game_gate_requires_terminal_horizon_provider_and_opening_coverage():
    valid = validation._validation_failures(
        terminated=True,
        turns=validation.FULL_GAME_TURNS,
        statuses=["DONE", "DONE"],
        controller_errors=[],
        backend_errors=[],
        primitive_actions=validation.FULL_GAME_TURNS,
        interaction_turns=3,
        observed_days=list(range(validation.TOTAL_DAYS)),
        manager_days=list(range(
            validation.MANAGER_START_DAY, validation.TOTAL_DAYS)),
        opening_diagnostics=_opening_diagnostics(),
    )
    assert valid == []

    failures = validation._validation_failures(
        terminated=False,
        turns=48,
        statuses=["ACTIVE", "ACTIVE"],
        controller_errors=[{"type": "RuntimeError"}],
        backend_errors=[],
        primitive_actions=48,
        interaction_turns=0,
        observed_days=[0, 1],
        manager_days=[],
        opening_diagnostics=None,
    )
    assert len(failures) >= 8


def test_match_summary_extracts_full_game_diagnostics():
    manager_days = list(range(
        validation.MANAGER_START_DAY, validation.TOTAL_DAYS))
    days = {
        str(day): {"route_diagnostics": [{"interaction_turns": 1}]}
        for day in manager_days
    }
    result = MatchResult(
        episode_index=0,
        seed=7,
        composition="checkpoint_vs_pass",
        orientation="checkpoint_vs_pass",
        controller_a_seat=0,
        final_banks=[100.0, 200.0],
        margin=-100.0,
        winner_seat=1,
        outcome="L",
        statuses=["DONE", "DONE"],
        terminated=True,
        turns=validation.FULL_GAME_TURNS,
        runtime_seconds=3.0,
        trace_digest="d" * 64,
        opening_diagnostics=[{
            "seat": 0,
            "detail": _opening_diagnostics(),
        }],
        executor_diagnostics=[{
            "seat": 0,
            "detail": {
                "agent": {"days": days},
                "validation": {
                    "primitive_actions": validation.FULL_GAME_TURNS,
                    "observed_days": list(range(validation.TOTAL_DAYS)),
                    "stage25_manager_days": manager_days,
                    "automatic_sell_orders_by_product": {
                        "WHEAT": 9, "FERTILIZER": 2,
                    },
                },
            },
        }],
    )

    report = validation._summarize_match(result, worker_pid=123)

    assert report["orientation"] == "checkpoint_seat_0_vs_pass"
    assert report["symmetric"] is False
    assert report["validation"]["passed"] is True
    assert report["validation"]["active_days"] == validation.TOTAL_DAYS
    assert report["validation"]["stage25_manager_days"] == (
        validation.MANAGER_ACTIVE_DAYS)
    assert report["validation"]["interaction_turns"] == (
        validation.MANAGER_ACTIVE_DAYS)
    assert report["validation"]["automatic_sell_orders_by_product"] == {
        "WHEAT": 9, "FERTILIZER": 2,
    }
    assert report["worker_pid"] == 123


def test_symmetric_summary_requires_and_reports_both_seats():
    manager_days = list(range(
        validation.MANAGER_START_DAY, validation.TOTAL_DAYS))
    days = {
        str(day): {"route_diagnostics": [{"interaction_turns": 1}]}
        for day in manager_days
    }

    def _strip_detail():
        return {
            "agent": {"days": days},
            "validation": {
                "primitive_actions": validation.FULL_GAME_TURNS,
                "observed_days": list(range(validation.TOTAL_DAYS)),
                "stage25_manager_days": manager_days,
                "automatic_sell_orders_by_product": {"WHEAT": 3},
            },
        }

    result = MatchResult(
        episode_index=0,
        seed=41001,
        composition="symmetric_same_policy_both_seats",
        orientation="symmetric_same_policy_both_seats",
        controller_a_seat=0,
        final_banks=[400.0, 410.0],
        margin=-10.0,
        winner_seat=1,
        outcome="L",
        statuses=["DONE", "DONE"],
        terminated=True,
        turns=validation.FULL_GAME_TURNS,
        runtime_seconds=12.0,
        trace_digest="e" * 64,
        opening_diagnostics=[
            {"seat": seat, "detail": _opening_diagnostics()}
            for seat in (0, 1)
        ],
        executor_diagnostics=[
            {"seat": seat, "detail": _strip_detail()} for seat in (0, 1)
        ],
    )

    report = validation._summarize_match(
        result, worker_pid=9, require_both_seats=True)

    assert report["validation"]["passed"] is True, report["validation"]["failures"]
    assert report["symmetric"] is True
    assert report["orientation"] == "checkpoint_symmetric_both_seats"
    assert set(report["validation"]["strips_by_seat"]) == {"0", "1"}
    assert all(
        seat["passed"] for seat in report["validation"]["strips_by_seat"].values()
    )


def test_symmetric_summary_fails_when_one_seat_is_missing():
    manager_days = list(range(
        validation.MANAGER_START_DAY, validation.TOTAL_DAYS))
    days = {
        str(day): {"route_diagnostics": [{"interaction_turns": 1}]}
        for day in manager_days
    }
    result = MatchResult(
        episode_index=0,
        seed=41001,
        composition="symmetric_same_policy_both_seats",
        orientation="symmetric_same_policy_both_seats",
        controller_a_seat=0,
        final_banks=[400.0, 400.0],
        margin=0.0,
        winner_seat=-1,
        outcome="T",
        statuses=["DONE", "DONE"],
        terminated=True,
        turns=validation.FULL_GAME_TURNS,
        runtime_seconds=12.0,
        trace_digest="f" * 64,
        opening_diagnostics=[{"seat": 0, "detail": _opening_diagnostics()}],
        executor_diagnostics=[{
            "seat": 0,
            "detail": {
                "agent": {"days": days},
                "validation": {
                    "primitive_actions": validation.FULL_GAME_TURNS,
                    "observed_days": list(range(validation.TOTAL_DAYS)),
                    "stage25_manager_days": manager_days,
                    "automatic_sell_orders_by_product": {},
                },
            },
        }],
    )

    report = validation._summarize_match(
        result, worker_pid=9, require_both_seats=True)

    assert report["validation"]["passed"] is False
    assert any(
        "seat 1" in failure for failure in report["validation"]["failures"])


def test_cli_emits_strict_json_success_and_failure(monkeypatch, capsys):
    validation_name = "stage25_native_checkpoint_parallel_full_game_v1"
    valid_report = {"status": "ok", "validation": validation_name}
    monkeypatch.setattr(validation, "run_validation", lambda **_kwargs: valid_report)

    assert validation.main([
        "--checkpoint", "policy.npz", "--seeds", "7", "--workers", "2",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == valid_report

    failed_report = {"status": "failed", "validation": validation_name}
    monkeypatch.setattr(validation, "run_validation", lambda **_kwargs: failed_report)
    assert validation.main(["--checkpoint", "policy.npz"]) == 1
    assert json.loads(capsys.readouterr().out) == failed_report


def test_checkpoint_path_must_be_existing_native_npz():
    missing = Path(__file__).with_name("missing-policy.npz")
    with pytest.raises(ValueError, match="unavailable"):
        validation._resolve_checkpoint_path(missing)

    with pytest.raises(ValueError, match=r"native inference \.npz"):
        validation._resolve_checkpoint_path(__file__)
