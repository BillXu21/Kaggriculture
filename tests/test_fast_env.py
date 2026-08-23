from __future__ import annotations

import sys

import numpy as np

from fast_env import FastKaggricultureEnv, market_price
from fast_env._kaggriculture_env import OBS_SIZE, RustBatchEnv


def pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def test_fast_import_does_not_load_kaggle_registry() -> None:
    assert "kaggle_environments" not in sys.modules
    assert "open_spiel" not in sys.modules


def test_hinge_prices_match_1327_anchors() -> None:
    assert market_price("CARROT", 9_550) == 70
    assert market_price("CARROT", 9_500) == 77
    assert market_price("TOMATO", 9_800) == 84
    assert market_price("TOMATO", 9_750) == 102
    assert market_price("EGG", 9_668) == 70
    assert market_price("EGG", 9_600) == 81


def test_reset_step_schema_and_private_observation() -> None:
    env = FastKaggricultureEnv({"seed": 7})
    observations = env.reset()
    assert [observation["player"] for observation in observations] == [0, 1]
    assert observations[0]["farms"] == observations[1]["farms"]
    assert observations[0]["private"] != observations[1]["private"] or observations[0]["private"]["seeds"]["WHEAT"] == 0

    actions = [
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]},
        pass_action(),
    ]
    observations, rewards, statuses = env.step(actions)
    assert observations[0]["private"]["seeds"]["WHEAT"] == 2
    assert observations[1]["private"]["seeds"]["WHEAT"] == 0
    assert rewards == [0.0, 0.0]
    assert statuses == ["ACTIVE", "ACTIVE"]
    assert observations[0]["market"] == observations[1]["market"]


def test_reset_is_repeatable() -> None:
    env = FastKaggricultureEnv({"seed": 7})
    first = env.reset()
    env.step([pass_action(), pass_action()])
    second = env.reset()
    assert first == second


def test_batch_api_fixed_layout_and_equal_seed_determinism() -> None:
    assert OBS_SIZE == 5630
    backend = RustBatchEnv(2, 720)
    observations, statuses = backend.reset(np.asarray([11, 11], dtype=np.uint64))
    assert observations.shape == (2, 2, OBS_SIZE)
    assert statuses.shape == (2, 2)
    assert np.array_equal(observations[0], observations[1])
    actions = np.zeros((2, 2, 27, 3), dtype=np.int64)
    stepped, rewards, step_statuses = backend.step(actions)
    assert stepped.shape == (2, 2, OBS_SIZE)
    assert rewards.shape == (2, 2)
    assert step_statuses.shape == (2, 2)
    assert np.array_equal(stepped[0], stepped[1])


def test_terminal_rewards_and_status() -> None:
    # Official 1.32.7 core.py marks agents DONE once observation.step >=
    # episodeSteps - 1 (reset is step 0), so episodeSteps=3 allows exactly
    # two agent steps: ACTIVE after the first, DONE after the second.
    env = FastKaggricultureEnv({"seed": 7, "episodeSteps": 3})
    env.reset()
    _, _, statuses = env.step([pass_action(), pass_action()])
    assert statuses == ["ACTIVE", "ACTIVE"]
    _, rewards, statuses = env.step([pass_action(), pass_action()])
    assert statuses == ["DONE", "DONE"]
    assert rewards == [3000.0, 3000.0]
