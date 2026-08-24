"""Batched JAX E policy wrapper tests (issue #9 architecture reqs. 8/11).

`JaxEPlanPolicy` must issue exactly ONE `bc_manager_jax.forward(..., "E")`
call per request batch (never per environment), decode exactly like the
issue-#8 path, and enforce the own-only E input contract loudly.
"""

import numpy as np
import pytest

from bc_manager_jax.model import (
    forward as jax_forward,
    init_params,
    tiny_manager_config,
)
from rl_manager.decode import LOGPROB_GROUPS, decode_outputs_to_action_tensors
from rl_manager.policy import JaxEPlanPolicy, params_fingerprint


def _encoded(day: int = 4, money: float = 3000.0):
    import bc_manager.live as live

    farm = {"farmer": [0, 0], "hands": [], "hires_today": 0,
            "money": money, "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"]}
    obs = {"day": day, "hour": 0, "step": day * 24, "player": 0,
           "farms": [farm, dict(farm)],
           "market": {"inventory": {}, "prices": {}},
           "town": {"unlocked_shops": []},
           "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}
    return live.encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=day * 24,
        economic_prev_start=(day - 1, money))


@pytest.fixture(scope="module")
def tiny_e():
    config = tiny_manager_config()
    params = init_params(config, seed=3, model_variant="E")
    return params, config


def test_params_fingerprint_stable_and_discriminating(tiny_e):
    params, _ = tiny_e
    fingerprint = params_fingerprint(params)
    assert fingerprint == params_fingerprint(params)
    other = init_params(tiny_manager_config(), seed=4, model_variant="E")
    assert fingerprint != params_fingerprint(other)


def test_rejects_non_e_variant(tiny_e):
    params, config = tiny_e
    with pytest.raises(ValueError, match="own-only E"):
        JaxEPlanPolicy(params, config, model_variant="V0")


def test_batch_of_two_through_eight_single_call_exact_decode(tiny_e):
    """One batched call for the whole batch; row decode equals a direct
    issue-#8 batched forward + decode of the same stacked inputs."""
    params, config = tiny_e
    policy = JaxEPlanPolicy(params, config)
    rows = [_encoded(day=day, money=1000.0 + 100.0 * day)
            for day in (4, 5, 6, 7, 8, 9, 10, 11)]
    for batch_size in (2, 5, 8):
        batch = {key: np.concatenate(
            [row[key] for row in rows[:batch_size]], axis=0)
            for key in rows[0]}
        outputs = policy.plan_batch(batch, f"prng/{batch_size}")
        assert outputs.batch_size == batch_size
        expected = decode_outputs_to_action_tensors(
            jax_forward(params, batch, config, model_variant="E"))
        for name, array in expected.items():
            assert np.array_equal(np.asarray(outputs.action_tensors[name]),
                                  array), name
        # PPO-ready slots exist per group plus total and value.
        assert set(outputs.logprob_groups) == set(LOGPROB_GROUPS)
        assert np.asarray(outputs.logprob_total).shape == (batch_size,)
        assert np.asarray(outputs.value).shape == (batch_size,)
    # Exactly one wrapper call per plan_batch invocation above.
    assert policy.call_count == 3
    assert policy.batch_size_history == [2, 5, 8]


def test_batched_rows_match_single_row_within_float_tolerance(tiny_e):
    """Batching may only move logits by float non-associativity (~1e-6);
    decoded actions stay identical whenever no near-exact logit tie."""
    params, config = tiny_e
    single = _encoded(day=6)
    batch = {key: np.concatenate([single[key], single[key]], axis=0)
             for key in single}
    out_single = jax_forward(params, single, config, model_variant="E")
    out_batch = jax_forward(params, batch, config, model_variant="E")
    decoded_batch = decode_outputs_to_action_tensors(out_batch)
    decoded_single = decode_outputs_to_action_tensors(out_single)
    logit_to_action = {
        "crop_logits": "crop", "animal_logits": "animal",
        "land_logits": "land", "fertilizer_logits": "fertilizer",
        "care_logits": "care", "sell_presence_logits": "sell_presence",
        "sell_quantity_log1p": "sell_quantity",
    }
    for key in out_batch:
        left = np.asarray(out_batch[key])[0]
        right = np.asarray(out_single[key])[0]
        assert np.allclose(left, right, atol=1e-5, rtol=0.0), key
        top2 = np.sort(np.abs(right).reshape(-1))[-2:]
        if top2[1] - top2[0] > 1e-4:  # no near-tie in this head
            action_key = logit_to_action[key]
            assert np.array_equal(decoded_batch[action_key][0],
                                  decoded_single[action_key][0]), action_key


def test_own_only_contract_rejects_leaked_opponent_keys(tiny_e):
    params, config = tiny_e
    policy = JaxEPlanPolicy(params, config)
    leaked = dict(_encoded(day=4))
    leaked["opp_board_kind"] = np.zeros((1, 100), dtype=np.int16)
    with pytest.raises(Exception):
        policy.plan_batch(leaked, "prng/leak")


def test_prng_id_must_be_explicit_string(tiny_e):
    params, config = tiny_e
    policy = JaxEPlanPolicy(params, config)
    with pytest.raises(ValueError, match="prng_id"):
        policy.plan_batch(_encoded(day=4), "")
