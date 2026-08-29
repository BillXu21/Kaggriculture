"""Focused evaluation taxonomy, economics, and promotion-gate tests."""

from __future__ import annotations

from types import SimpleNamespace

from rl_manager.evaluation import PromotionConfig, evaluate_promotion
from rl_manager.evaluation import summarize_evaluation


def _result(
    seed: int,
    composition: str,
    candidate_bank: float,
    opponent_bank: float,
    *,
    statuses: tuple[str, str] = ("DONE", "DONE"),
    terminated: bool = True,
    opening: list[dict] | None = None,
    executor: list[dict] | None = None,
) -> SimpleNamespace:
    banks = ([candidate_bank, opponent_bank]
             if composition == "candidate_vs_frozen"
             else [opponent_bank, candidate_bank])
    return SimpleNamespace(
        seed=seed,
        composition=composition,
        final_banks=banks,
        statuses=list(statuses),
        terminated=terminated,
        opening_diagnostics=opening or [],
        executor_diagnostics=executor or [],
        trace_digest="trace",
    )


def test_completed_64_zero_panel_ignores_benign_opening_diagnostics():
    results = [
        _result(
            seed,
            "candidate_vs_frozen" if seat == 0 else "frozen_vs_candidate",
            20_000 + seed,
            10_000,
            opening=[{
                "fallback_active": True,
                "delegated_permanently": True,
                "divergence": {"occurred": True, "reason": "guard"},
            }],
        )
        for seed in range(32)
        for seat in (0, 1)
    ]
    summary = summarize_evaluation(
        results,
        expected_seeds=range(32),
        provenance={"seed_set": "holdout"},
    )

    decision = evaluate_promotion(summary)

    assert decision.passed
    assert decision.failed_reasons == ()
    assert summary["wlt"] == {"W": 64, "L": 0, "T": 0}
    assert len(summary["opening_diagnostics"]) == 64
    assert summary["fatal_anomalies"] == []
    assert decision.eval_seed_set == "holdout"


def test_economic_summary_is_exact_and_seat_orientations_are_paired():
    results = [
        _result(1, "candidate_vs_frozen", 100, 50),
        _result(1, "frozen_vs_candidate", 1_000, 500),
        _result(2, "candidate_vs_frozen", 10_000, 9_000),
        _result(2, "frozen_vs_candidate", 20_000, 30_000),
    ]

    summary = summarize_evaluation(results)

    assert summary["banks"]["candidate"] == {
        "mean": 7_775.0,
        "median": 5_500.0,
        "p10": 370.0,
        "p25": 775.0,
        "p75": 12_500.0,
        "p90": 17_000.0,
        "min": 100.0,
        "max": 20_000.0,
        "frac_below_1k": 0.25,
        "frac_below_10k": 0.5,
    }
    assert summary["banks"]["opponent"]["mean"] == 9_887.5
    assert summary["banks"]["opponent"]["median"] == 4_750.0
    assert summary["margins"]["mean"] == -2_112.5
    assert summary["margins"]["median"] == 275.0
    assert summary["margins"]["min"] == -10_000.0
    assert summary["margins"]["max"] == 1_000.0
    assert summary["per_orientation"]["candidate_vs_frozen"] == {
        "games": 2, "W": 2, "L": 0, "T": 0}
    assert summary["per_orientation"]["frozen_vs_candidate"] == {
        "games": 2, "W": 1, "L": 1, "T": 0}
    assert summary["seat_splits"] == {
        "0": {"games": 2, "W": 2, "L": 0, "T": 0},
        "1": {"games": 2, "W": 1, "L": 1, "T": 0},
    }


def test_non_done_and_runtime_fallback_are_fatal():
    results = [
        _result(1, "candidate_vs_frozen", 20_000, 10_000,
                statuses=("DONE", "ACTIVE")),
        _result(2, "frozen_vs_candidate", 20_000, 10_000, executor=[{
            "seat": 1,
            "fallback_errors": [{"error": "provider exploded"}],
        }]),
    ]

    summary = summarize_evaluation(results)
    decision = evaluate_promotion(summary)

    assert {entry["kind"] for entry in summary["fatal_anomalies"]} == {
        "statuses", "runtime_fallback"}
    assert len(summary["executor_diagnostics"]) == 1
    assert decision.passed is False
    assert "fatal_anomalies=2" in decision.failed_reasons


def test_missing_and_duplicate_results_are_reported():
    results = [
        _result(1, "candidate_vs_frozen", 20_000, 10_000),
        _result(1, "candidate_vs_frozen", 20_000, 10_000),
        _result(1, "frozen_vs_candidate", 20_000, 10_000),
    ]

    summary = summarize_evaluation(
        results, expected_seeds=(1, 2),
    )
    kinds = {entry["kind"] for entry in summary["fatal_anomalies"]}

    assert kinds == {"duplicate_result", "missing_result"}
    assert len([entry for entry in summary["fatal_anomalies"]
                if entry["kind"] == "missing_result"]) == 2


def test_each_default_gate_failure_has_an_exact_reason():
    results = [
        _result(1, "candidate_vs_frozen", 100, 200),
        _result(1, "frozen_vs_candidate", 200, 100),
    ]
    summary = summarize_evaluation(results)

    decision = evaluate_promotion(summary)

    assert decision.passed is False
    assert decision.failed_reasons == (
        "w_minus_l 0 < 6",
        "mean_margin 0.0 <= 0.0",
    )

    mean_fail = evaluate_promotion(
        summarize_evaluation([_result(1, "candidate_vs_frozen", 101, 100)]),
        PromotionConfig(min_w_minus_l=0, min_mean_margin=2),
    )
    median_fail = evaluate_promotion(
        summarize_evaluation([_result(1, "candidate_vs_frozen", 101, 100),
                              _result(2, "candidate_vs_frozen", 100, 100)]),
        PromotionConfig(min_w_minus_l=0, min_median_margin=2),
    )
    assert "mean_margin 1.0 <= 2" in mean_fail.failed_reasons
    assert "median_margin 0.5 < 2" in median_fail.failed_reasons


def test_promotion_gate_is_independent_of_result_order():
    results = [
        _result(1, "candidate_vs_frozen", 12_000, 10_000),
        _result(1, "frozen_vs_candidate", 10_500, 10_000),
        _result(2, "candidate_vs_frozen", 9_000, 10_000),
        _result(2, "frozen_vs_candidate", 11_000, 10_000),
    ]

    first = evaluate_promotion(summarize_evaluation(results))
    second = evaluate_promotion(summarize_evaluation(list(reversed(results))))

    assert first == second
