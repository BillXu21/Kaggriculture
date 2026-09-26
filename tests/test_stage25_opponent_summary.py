"""Contract tests for the shared coarse opponent public-state summary."""

from __future__ import annotations

import numpy as np
import pytest

from bc_manager.adapter import (
    MAX_PUBLIC_WORKERS,
    OPPONENT_SUMMARY_ORDER,
    opponent_summary_arrays,
    opponent_summary_from_public_state,
)
from bc_manager.economics import signed_log_cash


def _tile(kind: str, *, crop: str | None = None,
          animal: str | None = None) -> dict[str, str | None]:
    return {"tile_kind": kind, "crop": crop, "animal": animal}


def _public_state() -> dict:
    board = [[_tile("EMPTY") for _ in range(10)] for _ in range(10)]
    for (y, x), crop in (
            ((0, 0), "WHEAT"), ((0, 1), "WHEAT"),
            ((0, 2), "CARROT"), ((0, 3), "STRAWBERRY")):
        board[y][x] = _tile("PLANT", crop=crop)
    board[1][0] = _tile("COOP", animal="GOOSE")
    board[1][1] = _tile("PASTURE", animal="COW")
    board[1][2] = _tile("PASTURE", animal="COW")
    board[1][3] = _tile("COOP", animal="SHEEP")
    return {
        "money": -250.0,
        "board": board,
        "unlocked_quadrants": ["NW", "NE"],
        "farmer": [0, 0],
        "hands": [[1, 1], [2, 2]],
        "hires_today": 3,
    }


def test_shared_summary_has_exact_feature_order_and_public_normalization():
    expected_order = (
        "cash", "land", "workers", "WHEAT", "CARROT", "TOMATO",
        "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP",
    )
    assert OPPONENT_SUMMARY_ORDER == expected_order
    expected = np.asarray([
        signed_log_cash(-250.0),
        2 / 4,
        3 / MAX_PUBLIC_WORKERS,
        2 / 100,
        1 / 100,
        0,
        1 / 100,
        0,
        1 / 100,
        2 / 100,
        1 / 100,
    ], dtype=np.float32)
    actual = opponent_summary_from_public_state(_public_state())
    assert actual.shape == (11,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        opponent_summary_arrays([_public_state()]), expected[None, :])


@pytest.mark.parametrize("private_key", ["shed", "seeds", "inventories", "private"])
def test_public_summary_rejects_private_opponent_fields(private_key: str):
    state = _public_state()
    state[private_key] = {"hidden": 123}
    with pytest.raises(ValueError, match="unexpected"):
        opponent_summary_from_public_state(state)
