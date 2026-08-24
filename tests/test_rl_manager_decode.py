"""Decode parity tests: rl_manager NumPy decode vs the authoritative torch
decoder in `executor_v0.manager` (issue #9 A1/B2 action representation).

The NumPy path exists so RL rollouts never need torch; it must mirror
argmax counts, land argmax + 1, sigmoid presence > 0.5, and round-half-up
expm1 quantity forced to 0 when absent — exactly.
"""

import numpy as np
import torch

from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
)
from executor_v0.manager import decode_daily_plan as torch_decode_daily_plan
from rl_manager.decode import (
    ACTION_TENSOR_SHAPES,
    LOGPROB_GROUPS,
    decode_outputs_to_action_tensors,
    plans_from_action_tensors,
    round_half_up,
)

COUNT_MAX = 100


def _random_logits(batch: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "crop_logits": rng.normal(size=(batch, NUM_CROPS, COUNT_MAX + 1)),
        "animal_logits": rng.normal(size=(batch, NUM_ANIMALS, COUNT_MAX + 1)),
        "land_logits": rng.normal(size=(batch, 4)),
        "fertilizer_logits": rng.normal(
            size=(batch, NUM_CROPS, COUNT_MAX + 1)),
        "care_logits": rng.normal(size=(batch, NUM_ANIMALS, COUNT_MAX + 1)),
        "sell_presence_logits": rng.normal(
            size=(batch, NUM_PRODUCTS, SELL_BIN_COUNT)),
        "sell_quantity_log1p": rng.normal(
            size=(batch, NUM_PRODUCTS, SELL_BIN_COUNT)),
    }


def test_action_tensor_schema_matches_model_constants():
    assert ACTION_TENSOR_SHAPES["crop"] == (NUM_CROPS,)
    assert ACTION_TENSOR_SHAPES["animal"] == (NUM_ANIMALS,)
    assert ACTION_TENSOR_SHAPES["sell_presence"] == \
        (NUM_PRODUCTS, SELL_BIN_COUNT)
    # sell_quantity stays a frozen regression output: no logprob slot.
    assert "sell_quantity" not in LOGPROB_GROUPS
    assert set(LOGPROB_GROUPS) == {
        "crop", "animal", "land", "fertilizer", "care", "sell_presence"}


def test_numpy_decode_matches_torch_decoder_every_row():
    outputs = _random_logits(batch=4, seed=11)
    tensors = decode_outputs_to_action_tensors(outputs)
    for row in range(4):
        single = {key: torch.tensor(np.asarray(value)[row][None])
                  for key, value in outputs.items()}
        expected = torch_decode_daily_plan(single, count_max=COUNT_MAX)
        ours = plans_from_action_tensors(
            {name: value[row:row + 1]
             for name, value in tensors.items()})[0]
        assert ours.to_json_dict() == expected.to_json_dict()


def test_absent_sell_presence_forces_zero_quantity():
    outputs = _random_logits(batch=1, seed=3)
    # Force presence logits clearly negative (absent) with positive quantity.
    outputs["sell_presence_logits"][:] = -10.0
    outputs["sell_quantity_log1p"][:] = 3.0
    tensors = decode_outputs_to_action_tensors(outputs)
    assert int(tensors["sell_quantity"].sum()) == 0
    assert int(tensors["sell_presence"].sum()) == 0


def test_round_half_up_rule():
    assert round_half_up(2.5) == 3
    assert round_half_up(2.4) == 2
    assert round_half_up(-0.5) == 0


def test_negative_quantity_log1p_is_clamped_to_zero_like_torch():
    outputs = _random_logits(batch=1, seed=5)
    outputs["sell_presence_logits"][:] = 10.0
    outputs["sell_quantity_log1p"][:] = -2.0
    tensors = decode_outputs_to_action_tensors(outputs)
    assert int(tensors["sell_quantity"].sum()) == 0
    single = {key: torch.tensor(np.asarray(value)) for key, value in
              outputs.items()}
    expected = torch_decode_daily_plan(single, count_max=COUNT_MAX)
    ours = plans_from_action_tensors(tensors)[0]
    assert ours.to_json_dict() == expected.to_json_dict()
