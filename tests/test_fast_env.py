from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from fast_env import FastKaggricultureEnv, market_price
from fast_env._kaggriculture_env import ACTION_SLOTS, MASK_SIZE, MAX_HANDS, OBS_SIZE, RustBatchEnv


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
    # Exact default-contract capacity: 10 market orders/turn * 24 turns/day,
    # one HIRE per atomic order, hands cleared at day reset.
    assert MAX_HANDS == 240
    assert ACTION_SLOTS == MAX_HANDS + 1 + 10
    # Observation layout: two MAX_HANDS-scaled blocks move together; the
    # reserved gaps stay fixed (authoritative source: the protocol generator).
    assert OBS_SIZE == 8766
    # Masks: (farmer + hands) unit slots + 10 market slots.
    assert MASK_SIZE == (MAX_HANDS + 1) * 136 + 10 * 125
    backend = RustBatchEnv(2, 720)
    observations, statuses = backend.reset(np.asarray([11, 11], dtype=np.uint64))
    assert observations.shape == (2, 2, OBS_SIZE)
    assert statuses.shape == (2, 2)
    assert np.array_equal(observations[0], observations[1])
    actions = np.zeros((2, 2, ACTION_SLOTS, 3), dtype=np.int64)
    stepped, rewards, step_statuses = backend.step(actions)
    assert stepped.shape == (2, 2, OBS_SIZE)
    assert rewards.shape == (2, 2)
    assert step_statuses.shape == (2, 2)
    assert np.array_equal(stepped[0], stepped[1])


def test_preallocated_into_calls_use_new_shapes() -> None:
    backend = RustBatchEnv(3, 720)
    seeds = np.asarray([5, 5, 5], dtype=np.uint64)
    observations = np.zeros((3, 2, OBS_SIZE), dtype=np.float32)
    masks = np.zeros((3, 2, MASK_SIZE), dtype=np.uint8)
    backend.reset(seeds)
    backend.observe_into(observations)
    backend.action_masks_into(masks)
    # Equal seeds produce identical observation and mask buffers.
    assert np.array_equal(observations[0], observations[2])
    assert np.array_equal(masks[0], masks[2])
    actions = np.zeros((3, 2, ACTION_SLOTS, 3), dtype=np.int64)
    rewards = np.zeros((3, 2), dtype=np.float32)
    statuses = np.zeros((3, 2), dtype=np.uint8)
    backend.step_into(actions, observations, rewards, statuses)
    assert rewards.shape == (3, 2) and statuses.shape == (3, 2)
    # Wrong shapes are rejected loudly, never silently reinterpreted.
    with pytest.raises(ValueError):
        backend.observe_into(np.zeros((3, 2, 5630), dtype=np.float32))
    with pytest.raises(ValueError):
        backend.action_masks_into(np.zeros((3, 2, 3552), dtype=np.uint8))
    with pytest.raises(ValueError):
        backend.step(np.zeros((3, 2, 27, 3), dtype=np.int64))


def test_memory_size_sanity_per_env_buffers() -> None:
    # Fixed contiguous per-env buffer sizes at MAX_HANDS=240 (f32=4B, i64=8B):
    # observations 2*OBS_SIZE*4, actions 2*ACTION_SLOTS*3*8, masks 2*MASK_SIZE.
    assert OBS_SIZE * 4 == 35064
    assert ACTION_SLOTS * 3 * 8 == 6024
    assert MASK_SIZE == 34026


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


# ---------------------------------------------------------------------------
# >16 simultaneous hands (MAX_HANDS=240 exact default capacity)
# ---------------------------------------------------------------------------


def _fibonacci(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_hands(env: FastKaggricultureEnv, counts: list[int]) -> None:
    """Submit HIRE-only turns; ``counts`` lists hires per turn."""
    for count in counts:
        actions = [
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * count},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        env.step(actions)


def test_scalar_api_supports_23_simultaneous_hands() -> None:
    # 10 orders/turn x 3 turns reaches 23 hands with startingMoney=100000
    # (total Fibonacci hire cost F(25)-1 = 75024). The old 16-slot layout
    # silently capped here.
    env = FastKaggricultureEnv({"seed": 7, "startingMoney": 100000})
    env.reset()
    _hire_hands(env, [10, 10, 3])
    observations = env.state_snapshot()
    for player in range(2):
        farm = observations[player]["farms"][0]
        assert len(farm["hands"]) == 23
        assert farm["hires_today"] == 23
        assert all(0 <= x < 10 and 0 <= y < 10 for x, y in farm["hands"])
        # Hands spawn on the fixed 4-tile farmhouse access set (least
        # occupied tile wins), so duplicates are expected beyond 4 hands.
        assert all(x in (4, 5) and y in (4, 5) for x, y in farm["hands"])
    # Private privacy at 23 hands: seat 0 decodes farmer + 23 hand
    # inventories; seat 1 still sees only its own farmer inventory.
    assert len(observations[0]["private"]["inventories"]) == 24
    assert len(observations[1]["private"]["inventories"]) == 1
    assert observations[0]["private"]["inventories"] == [{"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0, "EGG": 0, "MILK": 0, "WOOL": 0, "FERTILIZER": 0, "GOOSE": 0, "COW": 0, "SHEEP": 0}] * 24


def test_hand_actions_reach_all_23_hands() -> None:
    env = FastKaggricultureEnv({"seed": 7, "startingMoney": 100000})
    env.reset()
    _hire_hands(env, [10, 10, 3])
    hands = [[["NORTH"]] * 23, []]
    observations, _, statuses = env.step([
        {"farmer": ["PASS"], "hands": hands[0], "market": []},
        {"farmer": ["PASS"], "hands": hands[1], "market": []},
    ])
    assert statuses == ["ACTIVE", "ACTIVE"]
    farm = observations[0]["farms"][0]
    assert len(farm["hands"]) == 23


def test_day_end_resets_from_23_hands_and_rehire_restarts_fibonacci() -> None:
    env = FastKaggricultureEnv({"seed": 7, "startingMoney": 100000})
    env.reset()
    _hire_hands(env, [10, 10, 3])
    # Pass through hour 23 of day 0; the next step crosses into day 1.
    for _ in range(20):
        env.step([pass_action(), pass_action()])
    observations = env.state_snapshot()
    assert (observations[0]["day"], observations[0]["hour"]) == (0, 23)
    assert len(observations[0]["farms"][0]["hands"]) == 23
    _, _, statuses = env.step([pass_action(), pass_action()])
    assert statuses == ["ACTIVE", "ACTIVE"]
    observations = env.state_snapshot()
    assert (observations[0]["day"], observations[0]["hour"]) == (1, 0)
    assert len(observations[0]["farms"][0]["hands"]) == 0
    assert observations[0]["farms"][0]["hires_today"] == 0
    # First day-1 hire restarts the Fibonacci schedule at cost 1.
    money_before = observations[0]["farms"][0]["money"]
    _hire_hands(env, [1])
    observations = env.state_snapshot()
    assert len(observations[0]["farms"][0]["hands"]) == 1
    assert observations[0]["farms"][0]["hires_today"] == 1
    assert observations[0]["farms"][0]["money"] == money_before - 1


def test_hire_mask_matches_official_reachable_semantics() -> None:
    # HIRE is available in market slot 0 iff hand_count < MAX_HANDS and
    # money >= fibonacci(hires_today) -- exactly the official gate. The
    # observation money/hand_count/hires_today feeding the expectation are
    # the same decoded fields the real-official oracle proves identical.
    unit_mask_width = 18 + 17 + 101
    market_mask_width = 7 + 17 + 101
    market_base = (MAX_HANDS + 1) * unit_mask_width

    def hire_mask_bit(env: FastKaggricultureEnv) -> int:
        masks = np.zeros((1, 2, MASK_SIZE), dtype=np.uint8)
        env._backend.action_masks_into(masks)
        return int(masks[0, 0, market_base + 5])

    env = FastKaggricultureEnv({"seed": 7, "startingMoney": 100000})
    observations = env.reset()
    farm = observations[0]["farms"][0]
    # Open side: the initial farm can afford the first hire (cost fib(0)=1),
    # so the official formula evaluates to 1 and the mask must be open.
    assert int(farm["money"] >= _fibonacci(farm["hires_today"])) == 1
    assert hire_mask_bit(env) == 1
    _hire_hands(env, [10, 10, 3])
    observations = env.state_snapshot()
    farm = observations[0]["farms"][0]
    assert len(farm["hands"]) == 23
    # Closed side: 23 hires cost fib(24)-1 = 75024 of the 100000 starting
    # money, leaving 24976 < fib(23) = 46368 for the next hire -- exactly the
    # official affordability gate, so the mask must close even though
    # hand_count is still far below MAX_HANDS=240.
    assert farm["money"] == 100000 - (_fibonacci(24) - 1)
    assert farm["money"] < _fibonacci(farm["hires_today"])
    assert hire_mask_bit(env) == int(
        len(farm["hands"]) < MAX_HANDS
        and farm["money"] >= _fibonacci(farm["hires_today"])
    )
    assert hire_mask_bit(env) == 0
    # The same closure holds mid-day with the maximal hiring pressure: keep
    # submitting 10 HIRE orders per turn on day 0 until the Fibonacci cost
    # exceeds the remaining money; the mask closes at exactly that state.
    # 23 steps stay inside day 0 (hour 23); a 24th step would cross the day
    # reset, which clears hands and reopens the mask legitimately.
    big_env = FastKaggricultureEnv({"seed": 7, "startingMoney": 100000})
    big_env.reset()
    for _ in range(23):
        big_env.step([
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 10},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ])
    observations = big_env.state_snapshot()
    farm = observations[0]["farms"][0]
    assert (observations[0]["day"], observations[0]["hour"]) == (0, 23)
    assert len(farm["hands"]) == farm["hires_today"] == 23
    assert farm["money"] < _fibonacci(farm["hires_today"])
    assert hire_mask_bit(big_env) == 0
