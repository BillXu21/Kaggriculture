"""Behavioral contract for the native multi-environment fast backend."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fast_env import BatchedFastEnv, FastKaggricultureEnv
from fast_env.api import _encode_actions
from fast_env._kaggriculture_env import ACTION_SLOTS
from oracle.batched_backend import make_batched_backend


def _actions(turn: int) -> list[dict]:
    return [
        {
            "farmer": [["PASS", "EAST", "WEST"][turn % 3]],
            "hands": [],
            "market": (
                [["BUY_SEED", "WHEAT", 1]] if turn % 4 == 1 else []
            ),
        },
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WHEAT", 2]] if turn % 5 == 2 else [],
        },
    ]


def test_batch_reset_and_steps_match_scalar_fast() -> None:
    seeds = [7, 19]
    scalar = [FastKaggricultureEnv({"seed": seed, "numThreads": 1})
              for seed in seeds]
    batch = BatchedFastEnv(2, {"numThreads": 1})
    scalar_observations = [env.reset() for env in scalar]
    batch_observations = batch.reset(seeds)
    assert scalar_observations == batch_observations

    for turn in range(40):
        action_pair = _actions(turn)
        scalar_steps = [env.step(action_pair) for env in scalar]
        batch_steps = batch.step([action_pair, copy.deepcopy(action_pair)])
        for index in range(2):
            observations, rewards, statuses = scalar_steps[index]
            assert observations == batch_steps[0][index]
            assert rewards == batch_steps[1][index].tolist()
            assert statuses == batch.statuses(index)


def test_batch_action_buffer_is_one_native_tensor_and_matches_scalar_encoder() -> None:
    batch = BatchedFastEnv(2, {"numThreads": 1})
    action_batch = [_actions(3), _actions(4)]
    encoded = batch.encode_actions_into(action_batch)
    assert encoded.shape == (2, 2, ACTION_SLOTS, 3)
    assert encoded.dtype == np.int64
    expected = np.concatenate(
        [_encode_actions(actions) for actions in action_batch], axis=0
    )
    assert np.array_equal(encoded, expected)
    assert batch.encode_actions_into(action_batch) is batch.action_buffer


def test_batch_private_views_are_seat_local_and_adapter_uses_canonical_farms() -> None:
    batch = BatchedFastEnv(1, {"numThreads": 1})
    observations = batch.reset([7])[0]
    observations[0]["private"]["shed"]["WHEAT"] = 999
    assert observations[1]["private"]["shed"]["WHEAT"] != 999

    adapter = make_batched_backend("fast", 1, {"numThreads": 1})
    canonical = adapter.reset([7])[0]
    assert canonical[0]["farms"] == canonical[1]["farms"]
    assert all(
        "age" not in tile
        for farm in canonical[0]["farms"]
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )


def test_batch_terminal_statuses_and_shape_validation() -> None:
    scalar = FastKaggricultureEnv({"seed": 7, "episodeSteps": 3,
                                   "numThreads": 1})
    batch = BatchedFastEnv(1, {"episodeSteps": 3, "numThreads": 1})
    scalar.reset()
    batch.reset([7])
    action_pair = _actions(0)
    for _ in range(2):
        scalar_observations, scalar_rewards, scalar_statuses = scalar.step(
            action_pair
        )
        observations, rewards, statuses = batch.step([action_pair])
    assert observations[0] == scalar_observations
    assert rewards[0].tolist() == scalar_rewards
    assert statuses[0].tolist() == [1, 1]
    assert batch.statuses(0) == scalar_statuses == ["DONE", "DONE"]

    with pytest.raises(ValueError, match="seeds"):
        batch.reset([7, 8])
    with pytest.raises(ValueError, match="action batch"):
        batch.step([])
