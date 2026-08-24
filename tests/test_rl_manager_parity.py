"""Official-vs-fast parity gate tests (issue #9 A4).

The fast-vs-fast seam is proven on the cached deterministic full-game pair
shared with `test_rl_manager_runner.py` (no additional complete games):
matched comparison, first-divergence localization for every compared phase
(manager inputs / plans / primitive actions / final banks), and fast-seam
provenance. The official 1.32.7 leg is gated: without `kaggle_environments`
in the interpreter the test skips with the exact reproduction command and
never installs anything.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from rl_manager.parity import (
    OFFICIAL_BLOCKER_COMMAND,
    compare_rollouts,
    official_backend_available,
)
from test_rl_manager_runner import full_game_pair


def test_fast_vs_fast_parity_matches_on_deterministic_pair():
    game = full_game_pair()
    result_a, result_b = game["results"]
    comparison = compare_rollouts(result_a, result_b)
    assert comparison.matched is True
    assert comparison.report is None
    # Opening handoff (2) + manager inputs (52) + plans (52) + actions
    # (719 turns x 2 seats) + banks were all actually compared.
    assert comparison.checks >= 2 + 52 + 52 + 2


def _mutated_result(result, mutate_rollout=None, **fields):
    rollout = copy.deepcopy(result.rollout)
    if mutate_rollout is not None:
        mutate_rollout(rollout)
    return replace(result, rollout=rollout, **fields)


def test_injected_manager_input_mismatch_localizes_seat_day():
    game = full_game_pair()
    result_a, result_b = game["results"]

    def inject(rollout):
        rollout.manager_input_digests[(1, 7)] = "0" * 64

    mutated = _mutated_result(result_b, inject)
    comparison = compare_rollouts(result_a, mutated)
    assert comparison.matched is False
    report = comparison.report
    assert report.phase == "manager_inputs"
    assert (report.seat, report.day) == (1, 7)
    assert report.value_a != report.value_b
    assert "manager_inputs[1,7]" in report.field_path


def test_injected_plan_mismatch_localizes_seat_day_and_value():
    game = full_game_pair()
    result_a, result_b = game["results"]

    def inject(rollout):
        plan = dict(rollout.plans[(0, 10)])
        plan["_injected_divergence"] = True
        rollout.plans[(0, 10)] = plan

    mutated = _mutated_result(result_b, inject)
    comparison = compare_rollouts(result_a, mutated)
    assert comparison.matched is False
    report = comparison.report
    assert report.phase == "plans"
    assert (report.seat, report.day) == (0, 10)
    assert "_injected_divergence" in report.field_path
    assert report.render().startswith("first rollout divergence phase=plans")


def test_injected_action_mismatch_localizes_turn_seat_hour():
    game = full_game_pair()
    result_a, result_b = game["results"]
    turn_index = 120  # inside the manager-controlled window (day >= 4)

    def inject(rollout):
        _step, day, hour, seat0, seat1 = rollout.joint_actions[turn_index]
        seat1 = dict(seat1)
        seat1["_injected_divergence"] = True
        rollout.joint_actions[turn_index] = (_step, day, hour, seat0, seat1)

    mutated = _mutated_result(result_b, inject)
    comparison = compare_rollouts(result_a, mutated)
    assert comparison.matched is False
    report = comparison.report
    assert report.phase == "actions"
    assert report.turn_index == turn_index
    assert report.seat == 1
    _step, day, hour, _s0, _s1 = result_b.rollout.joint_actions[turn_index]
    assert (report.day, report.hour) == (day, hour)
    assert report.action_a is not None and report.action_b is not None


def test_injected_bank_mismatch_localizes_final_phase():
    game = full_game_pair()
    result_a, result_b = game["results"]
    banks = list(result_b.final_banks)
    banks[1] += 1.0
    mutated = _mutated_result(result_b, final_banks=banks,
                              margin=banks[0] - banks[1])
    comparison = compare_rollouts(result_a, mutated)
    assert comparison.matched is False
    report = comparison.report
    assert report.phase == "banks"
    assert report.field_path.startswith("final")


def test_compare_rollouts_requires_recorded_rollouts_and_paired_seeds():
    game = full_game_pair()
    result_a, result_b = game["results"]
    unrecorded = replace(result_a, rollout=None)
    with pytest.raises(ValueError, match="record_rollout"):
        compare_rollouts(unrecorded, result_b)
    with pytest.raises(ValueError, match="same seed/composition"):
        compare_rollouts(result_a, replace(result_b, seed=result_b.seed + 1))


def test_official_engine_parity_gated_with_exact_blocker():
    """Official 1.32.7 confirmation runs ONLY when the dependency imports in
    this interpreter; otherwise skip with the exact reproduction command."""
    if not official_backend_available():
        pytest.skip(
            "kaggle_environments 1.32.7 unavailable in this interpreter; "
            f"reproduce with: {OFFICIAL_BLOCKER_COMMAND}")
    from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
        build_episode_spec

    game = full_game_pair()
    official_runner = SelfPlayRunner(RunnerConfig(
        backend_name="official",
        backend_configuration={"numThreads": 1},
        record_rollout=True))
    spec = build_episode_spec(
        0, game["results"][0].seed, "e_vs_e", game["policy"], game["policy"])
    official_result = official_runner.run([spec])[0]
    comparison = compare_rollouts(game["results"][0], official_result)
    assert comparison.matched is True, comparison.report.render()


def test_fast_seam_provenance_recorded():
    game = full_game_pair()
    result = game["results"][0]
    assert result.rollout.backend_name == "fast"
    provenance = game["provenance"]
    assert provenance["backend"]["backend"] == "fast"
    assert provenance["backend"]["configuration"]["numThreads"] == 1
    engine_module = provenance["backend"]["engine_module"]
    # The fast seam records the exact loaded engine binary (Rust .pyd here).
    assert engine_module and "fast_env" in engine_module
    assert provenance["opening"]["name"] == "standard_mixed"
    assert len(provenance["opening"]["digest"]) == 64
