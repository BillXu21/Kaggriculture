"""Fixed-physical-batch stored-logprob audit tests.

The production TPU path runs rollout inference at a fixed physical batch
size; the pre-update audit must recompute at that SAME physical shape,
because different matmul shapes can differ at float-noise level. These tests
use shape-recording fakes plus the real tiny PPO policy (CPU only, no TPU).
"""

import jax
import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.parallel import pad_batch_to_physical
from rl_manager.ppo_adapter import recompute_stored_action_logprobs
from rl_manager.ppo_policy import CurriculumMaskConfig, PPOPolicy


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


class _RecordingEvaluator:
    """Fake policy: row-identity logprobs while recording call shapes."""

    def __init__(self):
        self.call_sizes: list[int] = []
        self.padded_calls: list[dict] = []

    def evaluate_actions(self, inputs, action_tensors):
        arrays = {name: np.asarray(array) for name, array in inputs.items()}
        n = int(next(iter(arrays.values())).shape[0])
        self.call_sizes.append(n)
        marker = np.asarray(arrays["marker"])[:, 0].astype(np.float64)
        self.padded_calls.append(
            {"size": n, "marker": marker.copy(),
             "action": np.asarray(action_tensors["land"]).copy()})
        return {"logprob_total": marker}


def _fake_batch(rows: int):
    inputs = {"marker": np.arange(rows, dtype=np.float32).reshape(rows, 1),
              "other": np.zeros((rows, 3), dtype=np.float32)}
    actions = {"land": np.arange(rows, dtype=np.int16),
               "animal": np.zeros((rows, 3), dtype=np.int16)}
    return inputs, actions


def test_chunks_use_exact_physical_size_in_order():
    policy = _RecordingEvaluator()
    inputs, actions = _fake_batch(10)
    result = recompute_stored_action_logprobs(
        policy, inputs, actions, physical_batch_size=4)
    assert policy.call_sizes == [4, 4, 4]
    assert np.array_equal(result, np.arange(10, dtype=np.float64))


def test_partial_chunk_pads_with_chunk_row_zero_and_drops_padding():
    policy = _RecordingEvaluator()
    inputs, actions = _fake_batch(10)
    result = recompute_stored_action_logprobs(
        policy, inputs, actions, physical_batch_size=4)
    assert result.shape == (10,)
    final = policy.padded_calls[-1]
    assert final["size"] == 4
    # Short final chunk [8, 9] padded by repeating chunk row 0 (value 8).
    assert np.array_equal(final["marker"], [8.0, 9.0, 8.0, 8.0])
    assert np.array_equal(final["action"][:4], [8, 9, 8, 8])


def test_none_physical_size_evaluates_whole_batch_once():
    policy = _RecordingEvaluator()
    inputs, actions = _fake_batch(7)
    result = recompute_stored_action_logprobs(policy, inputs, actions)
    assert policy.call_sizes == [7]
    assert np.array_equal(result, np.arange(7, dtype=np.float64))


def test_shape_sensitive_evaluator_never_sees_logical_batch():
    policy = _RecordingEvaluator()
    inputs, actions = _fake_batch(10)
    recompute_stored_action_logprobs(
        policy, inputs, actions, physical_batch_size=4)
    assert max(policy.call_sizes) == 4
    assert 10 not in policy.call_sizes


def test_pad_helper_matches_rollout_semantics():
    batch = {"x": np.array([[1.0], [2.0], [3.0]])}
    padded, count = pad_batch_to_physical(batch, 5)
    assert count == 2
    assert np.array_equal(padded["x"][:, 0], [1.0, 2.0, 3.0, 1.0, 1.0])
    same, count = pad_batch_to_physical(batch, 3)
    assert count == 0
    assert np.array_equal(same["x"], batch["x"])
    with pytest.raises(ValueError, match="exceeds physical"):
        pad_batch_to_physical(batch, 2)
    with pytest.raises(ValueError, match="positive int"):
        pad_batch_to_physical(batch, 0)


def test_rejects_bad_shapes_and_sizes():
    policy = _RecordingEvaluator()
    inputs, actions = _fake_batch(4)
    bad_actions = dict(actions)
    bad_actions["land"] = np.zeros(3, dtype=np.int16)
    with pytest.raises(ValueError, match="rows"):
        recompute_stored_action_logprobs(policy, inputs, bad_actions,
                                         physical_batch_size=2)
    with pytest.raises(ValueError, match="positive int"):
        recompute_stored_action_logprobs(policy, inputs, actions,
                                         physical_batch_size=0)
    with pytest.raises(ValueError, match="at least one"):
        recompute_stored_action_logprobs(
            policy, {"x": np.zeros((0, 1))}, {"land": np.zeros(0)},
            physical_batch_size=4)


def test_real_policy_chunked_matches_full_within_audit_tolerance():
    config = tiny_manager_config()
    params = init_params(config, seed=3, model_variant="E")
    policy = PPOPolicy(params, config, seed=11)
    rows = [_encoded(day=day, money=1000.0 + 100.0 * day)
            for day in (4, 5, 6, 7)]
    batch = {key: np.concatenate([row[key] for row in rows], axis=0)
             for key in rows[0]}
    sampled = policy.act(batch, rng=jax.random.PRNGKey(4))
    full = recompute_stored_action_logprobs(
        policy, batch, sampled["action_tensors"])
    chunked = recompute_stored_action_logprobs(
        policy, batch, sampled["action_tensors"], physical_batch_size=2)
    assert np.allclose(full, chunked, rtol=0.0, atol=1e-5)


def test_curriculum_invalid_actions_still_fail_loudly():
    config = tiny_manager_config()
    params = init_params(config, seed=3, model_variant="E")
    curriculum = CurriculumMaskConfig(max_land=1, max_goose=0)
    policy = PPOPolicy(params, config, seed=11, curriculum=curriculum)
    rows = [_encoded(day=day) for day in (4, 5)]
    batch = {key: np.concatenate([row[key] for row in rows], axis=0)
             for key in rows[0]}
    sampled = policy.act(batch, rng=jax.random.PRNGKey(4))
    invalid = {name: np.array(value, copy=True)
               for name, value in sampled["action_tensors"].items()}
    invalid["land"][0] = 2
    with pytest.raises(ValueError, match="max_land"):
        recompute_stored_action_logprobs(policy, batch, invalid)
    with pytest.raises(ValueError, match="max_land"):
        recompute_stored_action_logprobs(
            policy, batch, invalid, physical_batch_size=1)
