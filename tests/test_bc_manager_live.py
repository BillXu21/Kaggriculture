"""Hard parity tests: live observation encoder vs canonical-record -> adapter.

The reference path is the authoritative one: raw observation -> extractor
canonical state helpers -> full canonical record -> `records_to_table` ->
`bc_manager.adapter.table_to_arrays`. The live path is
`bc_manager.live.encode_live_inputs`. Every input array must match exactly
(discrete and float alike; both paths are deterministic), with identical key
sets, shapes, and dtypes.
"""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bc_manager.adapter import table_to_arrays
from bc_manager.live import encode_live_inputs, validate_previous_execution
from replay_daily.constants import PRODUCTS, SCHEMA_VERSION, SELL_BIN_ANCHORS
from replay_daily.extractor import (
    empty_events,
    opponent_public_state,
    self_state,
    shared_state,
)
from replay_daily.storage import records_to_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "samples" / "2026-08-20" / "94735084.json"

# Exact metadata field names of the canonical storage schema (storage._METADATA).
_METADATA_FIELDS = (
    "episode_id", "source_dataset", "partition_date", "source_path", "seat",
    "player", "opponent", "seed", "module_version", "avg_score", "min_score",
    "max_score", "sum_score", "final_rewards", "final_bank_self",
    "final_bank_opponent",
)


# ------------------------------------------------------- synthetic fixtures


def big_tiles() -> list[list[Any]]:
    """10x10 board exercising every tile shape the encoder must handle."""
    tiles: list[list[Any]] = [[None] * 10 for _ in range(10)]
    for x in range(5, 10):
        tiles[0][x] = "LOCKED"  # bare-string locked sentinel
    # WHEAT: harvestable, fertilized (full derived timing).
    tiles[1][1] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1,
                   "yield_units": 3, "watered_today": True,
                   "fertilized_until_day": 5, "max_lifespan_step": -1,
                   "consecutive_unwatered": 0}
    # TOMATO ongoing but unwatered: nullable days_until_next_harvest -> NaN.
    tiles[2][2] = {"kind": "PLANT", "crop": "TOMATO", "planted_day": 2,
                   "yield_units": 0, "watered_today": False,
                   "fertilized_until_day": -1, "max_lifespan_step": -1,
                   "consecutive_unwatered": 1}
    # Fed goose with product ready.
    tiles[3][3] = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0,
                   "yield_units": 2, "consecutive_unfed": 0,
                   "fed_today": True}
    # Starving sheep: starving=True, nullable days_until_next_product.
    tiles[4][4] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 1,
                   "yield_units": 0, "consecutive_unfed": 2,
                   "fed_today": False}
    return tiles


def make_farm10(shift: int = 0) -> dict:
    return {
        "farmer": [0 + shift, 1],
        "hands": [[2, 2], [3, 4]],
        "hires_today": 1 + shift,
        "money": 3000.0 + shift,
        "tiles": big_tiles(),
        "unlocked_quadrants": ["NW"],
    }


def make_obs10(seat: int, *, day: int = 3, hour: int = 0, step: int = 100) -> dict:
    return {
        "day": day,
        "hour": hour,
        "step": step,
        "player": seat,
        "farms": [make_farm10(), make_farm10(shift=1)],
        "market": {"inventory": {"WHEAT": 12, "EGG": 3},
                   "prices": {"WHEAT": 25, "MILK": 40}},
        # Duplicate shop instances must survive as a multiset.
        "town": {"unlocked_shops": ["BAKERY", "BAKERY", "PIZZA_SHOP"]},
        "private": {"shed": {"WHEAT": 4, "FERTILIZER": 1},
                    "seeds": {"CARROT": 2},
                    "inventories": [{"EGG": 1}, {}]},
        "remainingOverageTime": 60,  # framework-only field; must be ignored
    }


NONZERO_PREV = {"workers_hired": 3, "hire_cost": 4}


# ------------------------------------------------------------ reference path


def minimal_targets() -> dict:
    return {
        "crop_composition_end": {},
        "animal_counts_end": {},
        "unlocked_quadrants_end": ["NW"],
        "land_expansion": {"expanded": False, "new_quadrants": []},
        "fertilizer_by_crop": {},
        "care_by_animal": {"GOOSE": 0, "COW": 0, "SHEEP": 0},
        "sell_quantity": {str(a): {p: 0 for p in PRODUCTS}
                          for a in SELL_BIN_ANCHORS},
    }


def reference_record(obs: dict, seat: int, prev: dict) -> dict:
    """Full canonical record built exactly like replay_daily.extractor does."""
    day, hour, step = obs["day"], obs["hour"], obs["step"]
    start_self = self_state(obs, seat, day, step)
    end_self = self_state(obs, seat, day, step)
    start_public = opponent_public_state(obs, seat, day, step)
    end_public = opponent_public_state(obs, seat, day, step)
    shared = shared_state(obs)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {name: None for name in _METADATA_FIELDS},
        "day": day,
        "start": {
            "day": day, "hour": hour,
            "self": start_self,
            "opponent_public": start_public,
            **shared,
            "previous_execution": dict(prev),
        },
        "events": {**empty_events(),
                   "hires": {"submitted": 0, "realized": dict(prev)}},
        "targets": minimal_targets(),
        "end": {
            "boundary": "terminal", "day": day, "hour": hour,
            "self": end_self,
            "opponent_public": end_public,
            **shared,
        },
    }


def reference_inputs(obs: dict, seat: int, prev: dict,
                     include_opponent: bool) -> dict[str, np.ndarray]:
    inputs, _, _ = table_to_arrays(
        records_to_table([reference_record(obs, seat, prev)]),
        include_opponent=include_opponent)
    return inputs


def assert_parity(live: dict, ref: dict) -> None:
    assert set(live) == set(ref), (
        f"key mismatch: live-only={sorted(set(live) - set(ref))}, "
        f"ref-only={sorted(set(ref) - set(live))}")
    for key in sorted(ref):
        a, b = live[key], ref[key]
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), key
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        # Both paths are deterministic; exact equality also equates NaNs that
        # sit at identical positions (nullable derived timing channels).
        np.testing.assert_array_equal(a, b, err_msg=key)


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("include_opponent", [False, True])
@pytest.mark.parametrize("seat", [0, 1])
def test_synthetic_parity_both_seats_and_opponent_flag(seat, include_opponent):
    obs = make_obs10(seat)
    live = encode_live_inputs(obs, seat, NONZERO_PREV,
                              include_opponent=include_opponent)
    ref = reference_inputs(obs, seat, NONZERO_PREV, include_opponent)
    assert_parity(live, ref)


def test_day0_default_previous_execution_is_deterministic_zeros():
    obs = make_obs10(0)
    live = encode_live_inputs(obs, 0)
    ref = reference_inputs(obs, 0, {"workers_hired": 0, "hire_cost": 0},
                           include_opponent=False)
    assert_parity(live, ref)
    money = obs["farms"][0]["money"]
    hires_today = obs["farms"][0]["hires_today"]
    np.testing.assert_array_equal(
        live["scalars"][0], [money, hires_today, 0.0, 0.0])


def test_nonzero_previous_labor_is_carried_into_scalars():
    obs = make_obs10(1)
    live = encode_live_inputs(obs, 1, NONZERO_PREV)
    money = obs["farms"][1]["money"]
    hires_today = obs["farms"][1]["hires_today"]
    np.testing.assert_array_equal(
        live["scalars"][0], [money, hires_today,
                             float(NONZERO_PREV["workers_hired"]),
                             float(NONZERO_PREV["hire_cost"])])


def test_previous_execution_validation():
    assert validate_previous_execution(None) == \
        {"workers_hired": 0, "hire_cost": 0}
    assert validate_previous_execution(NONZERO_PREV) == NONZERO_PREV
    # numpy integers are accepted and normalized to Python ints.
    out = validate_previous_execution({"workers_hired": np.int64(2),
                                       "hire_cost": np.int32(7)})
    assert out == {"workers_hired": 2, "hire_cost": 7}
    assert all(type(v) is int for v in out.values())
    for bad in (
        {"workers_hired": -1, "hire_cost": 0},
        {"workers_hired": 0, "hire_cost": -2},
        {"workers_hired": True, "hire_cost": 0},
        {"workers_hired": 1.5, "hire_cost": 0},
        {"workers_hired": 1},
        {"hire_cost": 1},
        {"workers_hired": 1, "hire_cost": 1, "extra": 0},
        ["workers_hired", 1],
    ):
        with pytest.raises(ValueError):
            validate_previous_execution(bad)


@pytest.mark.parametrize("missing",
                         ["farms", "market", "town", "day", "hour"])
def test_missing_required_obs_fields_fail_clearly(missing):
    obs = make_obs10(0)
    del obs[missing]
    with pytest.raises(ValueError, match=missing):
        encode_live_inputs(obs, 0)


def test_missing_step_fails_without_explicit_override():
    obs = make_obs10(0)
    del obs["step"]
    with pytest.raises(ValueError, match="step"):
        encode_live_inputs(obs, 0)
    # Explicit override resolves the same lifecycle timing as obs["step"].
    assert_parity(encode_live_inputs(make_obs10(0), 0, NONZERO_PREV, step=100),
                  reference_inputs(make_obs10(0), 0, NONZERO_PREV, False))


def test_invalid_seat_day_hour_fail():
    with pytest.raises(ValueError):
        encode_live_inputs(make_obs10(0), 2)
    obs = make_obs10(0)
    obs["day"] = -1
    with pytest.raises(ValueError):
        encode_live_inputs(obs, 0)
    obs = make_obs10(0)
    obs["hour"] = "5"
    with pytest.raises(ValueError):
        encode_live_inputs(obs, 0)


def test_opponent_private_sentinels_never_leak_into_arrays():
    obs = make_obs10(0)
    baseline = encode_live_inputs(obs, 0, NONZERO_PREV, include_opponent=True)
    tampered = copy.deepcopy(obs)
    other = 1
    tampered["farms"][other]["shed"] = {"WOOL": 999}
    tampered["farms"][other]["seeds"] = {"MELON": 999}
    tampered["farms"][other]["inventories"] = [{"MILK": 999}]
    tampered["farms"][other]["private"] = {"shed": {"WOOL": 999}}
    tampered_live = encode_live_inputs(tampered, 0, NONZERO_PREV,
                                       include_opponent=True)
    assert_parity(tampered_live, baseline)


def test_no_metadata_or_result_fields_are_model_inputs():
    obs = make_obs10(0)
    live = encode_live_inputs(obs, 0, NONZERO_PREV, include_opponent=True)
    ref = reference_inputs(obs, 0, NONZERO_PREV, include_opponent=True)
    forbidden = ("metadata", "episode_id", "avg_score", "min_score",
                 "final_bank_self", "final_bank_opponent", "player",
                 "opponent", "source_path")
    assert not (set(live) & set(forbidden))
    assert_parity(live, ref)


@pytest.mark.skipif(not SAMPLE.exists(), reason="local real sample not present")
def test_real_replay_hour0_parity_both_seats():
    """Real 1.32.7 replay: live encoder == canonical-record -> adapter path."""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    records = {}
    from replay_daily.extractor import extract_replay

    for rec in extract_replay(raw, partition_date="2026-08-20"):
        records[(rec["metadata"]["seat"], rec["day"])] = rec

    checked = 0
    for seat in (0, 1):
        for day in (0, 7):
            rec = records[(seat, day)]
            # First observation of this seat's day at hour 0 (extractor rule).
            index = next(
                i for i, step in enumerate(raw["steps"])
                if step[seat]["observation"]["day"] == day
                and step[seat]["observation"]["hour"] == 0)
            obs = raw["steps"][index][seat]["observation"]
            resolved_step = int(obs.get("step", index))
            prev = rec["start"]["previous_execution"]

            ref = table_to_arrays(records_to_table([rec]),
                                  include_opponent=True)[0]
            live = encode_live_inputs(obs, seat, prev,
                                      include_opponent=True,
                                      step=resolved_step)
            assert_parity(live, ref)
            checked += 1
    assert checked == 4  # episode 94735084, seats {0,1} x days {0,7}
