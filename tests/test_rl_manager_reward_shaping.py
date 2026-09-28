from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rl_manager.runner import SelfPlayRunner
from rl_manager.reward import (
    MAX_TOTAL_SHAPING_WEIGHT,
    BehaviorShapingConfig,
    BehaviorShapingFeature,
    RewardConfig,
    TERMINAL_OWN_BANK,
    TERMINAL_OWN_BANK_LINEAR,
    normalized_saturated_potential,
    terminal_rewards,
)
from rl_manager.stage25_ppo_cli import (
    _reward_config,
    _validate_rollout_controls,
    _parser,
)
from rl_manager.stage25_mechanics import PhysicalContext
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer
from tests.test_stage25_trajectory import _row


def feature(target: int, weight: float) -> BehaviorShapingFeature:
    return BehaviorShapingFeature(target=target, weight=weight)


def test_disabled_config_preserves_existing_terminal_rewards() -> None:
    disabled = RewardConfig()
    assert not disabled.behavior_shaping.enabled
    assert terminal_rewards([100.0, 99.0], disabled) == [1.0, -1.0]
    assert terminal_rewards([100.0, 100.0], disabled) == [0.0, 0.0]
    assert terminal_rewards([99.0, 100.0], disabled) == [-1.0, 1.0]
    economic = RewardConfig(mode=TERMINAL_OWN_BANK)
    assert terminal_rewards([3000.0, 53000.0], economic) == [0.0, math.tanh(1.0)]


def test_linear_own_bank_reward_is_unclipped_and_wlt_is_unchanged() -> None:
    linear = RewardConfig(
        mode=TERMINAL_OWN_BANK_LINEAR,
        bank_baseline=3000.0,
        bank_scale=100000.0,
    )
    assert terminal_rewards(
        [3000.0, 53000.0], linear) == [0.0, 0.5]
    assert terminal_rewards([103000.0, 203000.0], linear) == [1.0, 2.0]
    assert terminal_rewards([0.0, 3000.0], linear) == pytest.approx(
        [-0.03, 0.0])

    tanh = RewardConfig(
        mode=TERMINAL_OWN_BANK,
        bank_baseline=3000.0,
        bank_scale=100000.0,
    )
    assert terminal_rewards([203000.0, 0.0], tanh) == [
        math.tanh(2.0), math.tanh(-0.03)]
    assert terminal_rewards([53000.0, 3000.0], RewardConfig()) == [1.0, -1.0]


@pytest.mark.parametrize(("count", "expected"), [
    (0, 0.0), (3, 0.5), (6, 1.0), (12, 1.0), (-2, 0.0),
])
def test_normalized_potential_is_saturated(count: int, expected: float) -> None:
    assert normalized_saturated_potential(count, 6) == expected


def test_signed_potential_differences_telescope_across_reacquisition() -> None:
    shaping = BehaviorShapingConfig(cow=feature(6, 0.1))
    counts = [0, 3, 0, 3]
    deltas = [
        shaping.contributions({"cow": left}, {"cow": right})["cow"]
        for left, right in zip(counts, counts[1:])
    ]
    assert deltas == pytest.approx([0.05, -0.05, 0.05])
    assert math.fsum(deltas) == pytest.approx(
        0.1 * (normalized_saturated_potential(3, 6)
               - normalized_saturated_potential(0, 6)))
    assert shaping.contributions({"cow": 3}, {"cow": 3})["cow"] == 0.0


def test_feature_contributions_sum_and_config_serializes_stably() -> None:
    shaping = BehaviorShapingConfig(
        goose=feature(8, 0.05), cow=feature(5, 0.075),
        wheat=feature(25, 0.1))
    contributions = shaping.contributions(
        {"goose": 0, "cow": 1, "wheat": 0},
        {"goose": 4, "cow": 3, "wheat": 10})
    assert contributions == pytest.approx({
        "goose": 0.025, "cow": 0.03, "wheat": 0.04})
    assert math.fsum(contributions.values()) == pytest.approx(0.095)
    serialized = shaping.to_json_dict()
    assert list(serialized) == ["goose", "cow", "wheat"]
    assert BehaviorShapingConfig.from_json_dict(serialized) == shaping


def test_weight_bound_and_zero_weight_canonicalization() -> None:
    accepted = BehaviorShapingConfig(
        cow=feature(6, 0.1), wheat=feature(30, 0.15))
    assert math.fsum(item.weight for _, item in accepted.active_features()) \
        == MAX_TOTAL_SHAPING_WEIGHT
    assert not BehaviorShapingConfig(cow=feature(4, 0.0)).enabled
    with pytest.raises(ValueError, match="total behavior-shaping weight"):
        BehaviorShapingConfig(
            cow=feature(8, 0.2), wheat=feature(30, 0.2))


@pytest.mark.parametrize("weight", [-0.1, math.nan, math.inf, -math.inf])
def test_invalid_weights_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        feature(5, weight)


@pytest.mark.parametrize("target", [0, -1, 101])
def test_out_of_range_targets_rejected(target: int) -> None:
    with pytest.raises(ValueError, match="target must be in"):
        feature(target, 0.1)


@pytest.mark.parametrize("target", [True, 1.5, "3"])
def test_non_integer_targets_rejected(target: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        feature(target, 0.1)  # type: ignore[arg-type]


def test_maximum_valid_shaping_cannot_change_win_or_loss_sign() -> None:
    shaping = BehaviorShapingConfig(
        goose=feature(8, 0.05), cow=feature(6, 0.1), wheat=feature(30, 0.1))
    largest = shaping.potential({
        "goose": 8, "cow": 6, "wheat": 30})
    assert largest <= MAX_TOTAL_SHAPING_WEIGHT
    assert 1.0 - largest >= 0.75
    assert -1.0 + largest <= -0.75


@pytest.mark.parametrize(
    "reward_mode", (TERMINAL_OWN_BANK, TERMINAL_OWN_BANK_LINEAR))
def test_cli_requires_complete_pairs_and_terminal_wlt(reward_mode) -> None:
    incomplete = _parser().parse_args([
        "--scratch", "--output-dir", "out",
        "--shape-cow-target", "6"])
    with pytest.raises(ValueError, match="supplied together"):
        _reward_config(incomplete)

    own_bank = _parser().parse_args([
        "--scratch", "--output-dir", "out",
        "--reward-mode", reward_mode,
        "--shape-cow-target", "6", "--shape-cow-weight", "0.1"])
    with pytest.raises(ValueError, match="requires reward mode terminal_wlt"):
        _reward_config(own_bank)


def test_resume_override_requires_resume_source() -> None:
    args = _parser().parse_args([
        "--scratch", "--output-dir", "out",
        "--allow-shaping-change-on-resume"])
    with pytest.raises(ValueError, match="requires --resume"):
        _validate_rollout_controls(args)


def test_disabled_runner_path_does_not_compute_realized_counts(monkeypatch) -> None:
    import rl_manager.runner as runner_module

    runner = object.__new__(SelfPlayRunner)
    runner.config = SimpleNamespace(reward_config=RewardConfig())
    state = SimpleNamespace(behavior_shaping_stats={})

    def unexpected_count(*_args, **_kwargs):
        raise AssertionError("disabled shaping must not count game state")

    monkeypatch.setattr(
        runner_module, "realized_behavior_counts_from_context", unexpected_count)
    monkeypatch.setattr(
        runner_module, "realized_behavior_counts_from_observation", unexpected_count)
    runner._record_behavior_shaping_boundary(state, 0, 4, object())
    assert runner._finish_behavior_shaping(state, terminated=True) == {}


def _physical_context(*, cow: int = 0, wheat: int = 0) -> PhysicalContext:
    # The runner's current manager context is the single canonical board scan.
    return PhysicalContext(
        observed_land=1,
        crop_build_cells_by_land=(25, 50, 75, 100),
        placed_animals=(0, cow, 0),
        observed_crop_counts=(wheat, 0, 0, 0, 0),
    )


def _seat_shaping_stats() -> dict[str, object]:
    return {
        "previous_counts": None,
        "initial_counts": None,
        "final_counts": None,
        "last_day": None,
        "terminal_finalized": False,
        "feature_rewards": {"cow": 0.0},
    }


def test_runner_places_manager_and_final_deltas_once_per_learner_seat(
        monkeypatch) -> None:
    import rl_manager.runner as runner_module

    shaping = BehaviorShapingConfig(cow=feature(6, 0.1))
    buffer = Stage25TrajectoryBuffer(6)
    runner = object.__new__(SelfPlayRunner)
    runner.config = SimpleNamespace(
        reward_config=RewardConfig(behavior_shaping=shaping))
    runner.stage25_trajectory = buffer
    state = SimpleNamespace(
        behavior_shaping_stats={0: _seat_shaping_stats(),
                                1: _seat_shaping_stats()},
        transition_index={},
        obs=[{}, {}],
    )

    # The first boundary is a baseline only. Seat 0 and seat 1 each use the
    # PhysicalContext prepared from their own view.
    runner._record_behavior_shaping_boundary(state, 0, 4, _physical_context())
    runner._record_behavior_shaping_boundary(
        state, 1, 4, _physical_context(cow=4))
    buffer.append(_row(seat=0, day=4, row_id="s0-d4"))
    state.transition_index[(0, 4)] = 0
    buffer.append(_row(seat=1, day=4, row_id="s1-d4"))
    state.transition_index[(1, 4)] = 1
    assert [float(row.reward) for row in buffer.rows] == [0.0, 0.0]

    # The next manager boundary patches the outgoing day-4 decision once.
    runner._record_behavior_shaping_boundary(
        state, 0, 5, _physical_context(cow=3))
    runner._record_behavior_shaping_boundary(
        state, 1, 5, _physical_context(cow=5))
    assert float(buffer.rows[0].reward) == pytest.approx(0.05)
    assert float(buffer.rows[1].reward) == pytest.approx(0.1 / 6)
    with pytest.raises(ValueError, match="must advance"):
        runner._record_behavior_shaping_boundary(
            state, 0, 5, _physical_context(cow=3))

    for seat in (0, 1):
        next_row = _row(seat=seat, day=5, row_id=f"s{seat}-d5")
        buffer.close_outgoing(
            episode_index=1, seat=seat, next_day=5,
            next_inputs=next_row.inputs,
            next_crop_capacity=np.asarray([1, 2, 3, 4, 5], dtype=np.int16))
    buffer.append(_row(seat=0, day=5, row_id="s0-d5"))
    state.transition_index[(0, 5)] = 2
    buffer.append(_row(seat=1, day=5, row_id="s1-d5"))
    state.transition_index[(1, 5)] = 3

    # Reaching another manager boundary with the same realized count earns
    # zero, even if the manager's hidden absolute request hypothetically grew.
    runner._record_behavior_shaping_boundary(
        state, 0, 6, _physical_context(cow=3))
    runner._record_behavior_shaping_boundary(
        state, 1, 6, _physical_context(cow=5))
    assert float(buffer.rows[2].reward) == 0.0
    assert float(buffer.rows[3].reward) == 0.0
    for seat in (0, 1):
        next_row = _row(seat=seat, day=6, row_id=f"s{seat}-d6")
        buffer.close_outgoing(
            episode_index=1, seat=seat, next_day=6,
            next_inputs=next_row.inputs,
            next_crop_capacity=np.asarray([1, 2, 3, 4, 5], dtype=np.int16))
    buffer.append(_row(seat=0, day=6, row_id="s0-d6"))
    state.transition_index[(0, 6)] = 4
    buffer.append(_row(seat=1, day=6, row_id="s1-d6"))
    state.transition_index[(1, 6)] = 5

    # The final observed state is seat-specific. Its boundary-to-terminal
    # delta joins W/L on the final row, while the other seat stays untouched.
    final_counts = {0: {"cow": 5}, 1: {"cow": 5}}
    monkeypatch.setattr(
        runner_module, "realized_behavior_counts_from_observation",
        lambda obs, seat: final_counts[seat])
    state.obs = [object(), object()]
    deltas = runner._finish_behavior_shaping(state, terminated=True)
    assert deltas == pytest.approx({0: 2 * 0.1 / 6, 1: 0.0})
    buffer.patch_terminal(4, np.float32(1.0 + deltas[0]))
    buffer.patch_terminal(5, np.float32(-1.0 + deltas[1]))
    rows = buffer.rows
    assert float(rows[4].reward) == pytest.approx(1.0 + 2 * 0.1 / 6)
    assert float(rows[5].reward) == -1.0
    with pytest.raises(RuntimeError, match="already finalized"):
        runner._finish_behavior_shaping(state, terminated=True)
