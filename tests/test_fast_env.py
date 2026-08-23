from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from fast_env import FastKaggricultureEnv, market_price
from fast_env._kaggriculture_env import OBS_SIZE, RustBatchEnv


def pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def test_fast_import_does_not_load_kaggle_registry() -> None:
    if importlib.util.find_spec("kaggle_environments") is not None:
        # Official oracle runs legitimately import kaggle_environments in the
        # same pytest process (module-level provenance guard); the in-process
        # assertion is then vacuous. The fresh-process isolation tests in
        # tests/test_oracle_import_isolation.py own this guarantee there.
        pytest.skip("kaggle_environments installed; fresh-process isolation test covers this")
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


def test_wire_unit_operations_translate_to_internal_op_codes() -> None:
    # Regression: wire vocabulary ids were once sent untranslated, so e.g.
    # "PLANT" (wire 8) landed in the Rust core's build-structure arm
    # (internal op 8). Every operation must map to the code that
    # apply_unit_action actually dispatches on.
    from fast_env.api import _unit_row

    assert _unit_row(["PASS"]) == (0, 0, 0)
    assert _unit_row(["NORTH"]) == (1, 0, 0)
    assert _unit_row(["SOUTH"]) == (2, 0, 0)
    assert _unit_row(["EAST"]) == (3, 0, 0)
    assert _unit_row(["WEST"]) == (4, 0, 0)
    assert _unit_row(["PLANT", "WHEAT"]) == (5, 0, 0)
    assert _unit_row(["HARVEST"]) == (6, 0, 0)
    assert _unit_row(["FEED"]) == (7, 0, 0)
    assert _unit_row(["BUILD_COOP"]) == (8, 0, 0)
    assert _unit_row(["BUILD_PASTURE"]) == (8, 1, 0)
    assert _unit_row(["PLACE", "WHEAT", 2]) == (9, 4, 2)
    assert _unit_row(["WATER"]) == (10, 0, 0)
    assert _unit_row(["PICKUP", "WHEAT", 2]) == (11, 4, 2)
    assert _unit_row(["FERTILIZE"]) == (12, 0, 0)
    assert _unit_row(["CARE"]) == (13, 0, 0)
    assert _unit_row(["COLLECT_FERTILIZER"]) == (14, 0, 0)
    assert _unit_row(["DROP"]) == (15, 0, 0)
    assert _unit_row(["DIG"]) == (17, 0, 0)


def test_plant_action_reaches_rust_plant_arm() -> None:
    # Behavioral companion to the translation table: a submitted PLANT must
    # create a crop (internal op 5), not a structure (internal op 8).
    env = FastKaggricultureEnv({"seed": 7})
    env.reset()
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        pass_action(),
    ])
    observations, _, _ = env.step([
        {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        pass_action(),
    ])
    farm = observations[0]["farms"][0]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    assert tile is not None and tile["kind"] == "PLANT" and tile["crop"] == "WHEAT"


def test_decode_inverts_fixed_season_steps_not_configured_episode_steps() -> None:
    # Regression: the Rust observation writer normalizes the primitive step
    # and plant lifespan by the FIXED generated_protocol::SEASON_STEPS (720),
    # never by the configured episodeSteps. Decoding must invert exactly 720
    # or step/day/hour/lifespan break whenever episodeSteps != 720.
    from fast_env.api import SEASON_STEPS

    assert SEASON_STEPS == 720
    env = FastKaggricultureEnv({"seed": 7, "episodeSteps": 5})
    env.reset()
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        pass_action(),
    ])
    observations, _, statuses = env.step([
        {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        pass_action(),
    ])
    # If decoding inverted episodeSteps=5, raw step 2/720 would decode as 0.
    assert observations[0]["step"] == 2
    assert (observations[0]["day"], observations[0]["hour"]) == (0, 2)
    assert statuses == ["ACTIVE", "ACTIVE"]
    farm = observations[0]["farms"][0]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    assert tile is not None and tile["kind"] == "PLANT"
    # Lifespan is encoded on the same fixed season scale: it must exceed the
    # configured episodeSteps of 5 (any decay day is at least a full day).
    assert tile["max_lifespan_step"] > 5
