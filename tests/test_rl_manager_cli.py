"""CLI planning/aggregation tests for the issue #9 Stage B CLIs.

Tests cover ONLY the parser, plan validation, and the fixed evaluation
output schema on synthetic records. The train/eval commands are NEVER
executed here (they require the real BC-E checkpoint and would launch
games/updates).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from rl_manager.cli import (
    CONFIRM_FLAG,
    DEV_SEEDS,
    HOLDOUT_SEEDS,
    SMOKE_SEEDS,
    build_parser,
    plan_evaluation,
    plan_training,
    summarize_evaluation,
)


def _train_args(**overrides):
    values = {
        "command": "train",
        "e_checkpoint": "artifacts/local/bc-v1-E/best.pt",
        "executor_factory": "executor_v0@stage-a-v1",
        "backend": "fast",
        "master_seed": 17,
        "num_workers": 1,
        "num_envs": 1,
        "num_threads": 1,
        "episodes_per_update": 8,
        "updates": 1,
        "epochs": 4,
        "minibatch_size": 64,
        "lr": 3e-4,
        "kl_to_frozen_coef": 0.0,
        "output_dir": "artifacts/local/ppo-smoke",
        "checkpoint": "artifacts/local/ppo-smoke/ppo.npz",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _eval_args(**overrides):
    values = {
        "command": "eval",
        "checkpoint": "artifacts/local/ppo-smoke/ppo.npz",
        "e_checkpoint": "artifacts/local/bc-v1-E/best.pt",
        "executor_factory": "executor_v0@stage-a-v1",
        "backend": "fast",
        "num_workers": 1,
        "num_envs": 1,
        "num_threads": 1,
        "seed_set": "smoke",
        "output_json": "artifacts/local/ppo-smoke/eval.json",
        "confirm_expensive": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


# ------------------------------------------------------------------ parser


def test_parser_defaults_are_safe_single_process():
    args = build_parser().parse_args([
        "train", "--e-checkpoint", "ck.pt",
        "--executor-factory", "executor_v0@stage-a-v1",
        "--master-seed", "17", "--output-dir", "out", "--checkpoint",
        "out/ppo.npz"])
    assert (args.num_workers, args.num_envs, args.num_threads) == (1, 1, 1)
    assert args.backend == "fast"

    ev = build_parser().parse_args([
        "eval", "--checkpoint", "ppo.npz", "--e-checkpoint", "ck.pt",
        "--executor-factory", "executor_v0@stage-a-v1",
        "--seed-set", "smoke", "--output-json", "eval.json"])
    assert (ev.num_workers, ev.num_envs, ev.num_threads) == (1, 1, 1)
    assert ev.confirm_expensive is False


def test_parser_requires_explicit_executor_and_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "train", "--e-checkpoint", "ck.pt", "--master-seed", "17",
            "--output-dir", "out", "--checkpoint", "out/ppo.npz"])
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# ----------------------------------------------------------- train planning


def test_train_plan_fails_loud_when_checkpoint_missing():
    with pytest.raises(FileNotFoundError, match="e-checkpoint"):
        plan_training(_train_args())


def test_train_plan_rejects_unsafe_knobs(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    base = {"e_checkpoint": str(checkpoint)}
    with pytest.raises(NotImplementedError, match="num-workers"):
        plan_training(_train_args(**base, num_workers=96))
    with pytest.raises(ValueError, match="threads"):
        plan_training(_train_args(**base, num_threads=0))
    with pytest.raises(ValueError, match="master-seed"):
        plan_training(_train_args(**base, master_seed=-1))


def test_train_plan_full_fields(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    plan = plan_training(_train_args(e_checkpoint=str(checkpoint)))
    assert plan["mode"] == "train"
    assert plan["master_seed"] == 17
    assert plan["knobs"] == {"num_workers": 1, "num_envs": 1,
                             "num_threads": 1}
    assert plan["executor_factory"] == "executor_v0@stage-a-v1"
    assert plan["ppo"]["epochs"] == 4
    assert Path(plan["e_checkpoint"]) == checkpoint


# ------------------------------------------------------------ eval planning


def test_seed_sets_match_issue_specification():
    assert SMOKE_SEEDS == (17, 42, 2026)
    assert DEV_SEEDS == tuple(range(200, 264))
    assert HOLDOUT_SEEDS == tuple(range(5000, 5032))
    assert CONFIRM_FLAG == "--confirm-expensive"


def test_eval_plan_smoke_plans_both_seats_without_confirmation(
        tmp_path, capsys):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    plan = plan_evaluation(_eval_args(e_checkpoint=str(checkpoint)))
    assert plan["planned_games"] == len(SMOKE_SEEDS) * 2
    assert plan["seat_orientations"] == ["candidate_vs_frozen",
                                         "frozen_vs_candidate"]
    assert "both seat orientations" in capsys.readouterr().out


@pytest.mark.parametrize("seed_set,games", [
    ("dev", len(DEV_SEEDS) * 2),
    ("holdout", len(HOLDOUT_SEEDS) * 2),
])
def test_eval_dev_holdout_require_explicit_confirmation(
        seed_set, games, tmp_path, capsys):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    with pytest.raises(SystemExit, match="confirm-expensive"):
        plan_evaluation(_eval_args(seed_set=seed_set,
                                   e_checkpoint=str(checkpoint)))
    capsys.readouterr()
    confirmed = _eval_args(seed_set=seed_set, confirm_expensive=True,
                           e_checkpoint=str(checkpoint))
    assert plan_evaluation(confirmed)["planned_games"] == games
    capsys.readouterr()


# ------------------------------------------------------------- aggregation


def _result(seed, composition, banks, statuses=("DONE", "DONE"),
            opening=None):
    margin = banks[0] - banks[1]
    winner = 0 if margin > 0 else (1 if margin < 0 else -1)
    rewards = [0.0, 0.0]
    if winner >= 0:
        rewards[winner] = 1.0
        rewards[1 - winner] = -1.0
    return types.SimpleNamespace(
        episode_index=seed, seed=seed, composition=composition,
        final_banks=banks, margin=margin, winner_seat=winner,
        rewards=rewards, statuses=list(statuses), transitions=52,
        terminated=statuses == ("DONE", "DONE"),
        opening_diagnostics=opening or [], trace_digest="x" * 64,
        rollout=None, timing_seconds={}, policy_identities=({}, {}))


def test_summarize_evaluation_fixed_schema():
    results = [
        _result(17, "candidate_vs_frozen", [3000.0, 1000.0]),
        _result(17, "frozen_vs_candidate", [1200.0, 800.0]),
        _result(42, "candidate_vs_frozen", [900.0, 900.0]),
        _result(42, "frozen_vs_candidate", [800.0, 1200.0]),
    ]
    summary = summarize_evaluation(results)
    # Candidate-perspective paired margins: +2000 (seat 0), -400 (seat 1),
    # 0 tie, +400 (seat 1).
    assert summary["games"] == 4
    assert summary["wlt"] == {"W": 2, "L": 1, "T": 1}
    assert summary["win_rate"] == 0.5
    assert summary["paired_margins"] == [2000.0, -400.0, 0.0, 400.0]
    assert summary["median_margin"] == 200.0
    assert summary["mean_margin"] == 500.0
    assert summary["per_orientation"]["candidate_vs_frozen"] == {
        "games": 2, "W": 1, "L": 0, "T": 1}
    assert summary["worst_seeds"][0]["margin"] == -400.0
    assert summary["anomalies"] == []


def test_summarize_evaluation_flags_anomalies():
    results = [
        _result(7, "candidate_vs_frozen", [10.0, 20.0],
                statuses=("DONE", "ACTIVE")),
        _result(9, "frozen_vs_candidate", [5.0, 5.0],
                opening=[{"delegated_permanently": True}]),
    ]
    summary = summarize_evaluation(results)
    kinds = {entry["kind"] for entry in summary["anomalies"]}
    assert kinds == {"statuses", "opening"}
