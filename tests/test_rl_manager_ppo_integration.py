"""Issue #9 Stage B2 integration tests: adapter, tiny live PPO smoke, CLIs.

Complete-game budget: exactly ONE complete fast-engine game is executed in
this file (module-scoped cache, numThreads=1). Its fixed-seed stochastic
rollout serves as the PPO data; all pre/post comparisons otherwise run on
synthetic fixed batches or stored trajectory rows — no additional games,
no quality claim (tiny random-init E, plumbing only).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bc_manager_jax.model import init_params, tiny_manager_config
from rl_manager.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    build_integration_diagnostics,
    write_diagnostics,
)
from rl_manager.decode import ACTION_TENSOR_SHAPES, LOGPROB_GROUPS
from rl_manager.gae import compute_gae
from rl_manager.ppo import build_ppo_batch, init_train_state, ppo_update
from rl_manager.ppo_adapter import (
    PPOBatchedPolicy,
    ppo_batched_policy_from_state,
    prng_key_from_id,
    select_ppo_subset,
)
from rl_manager.ppo_checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from rl_manager.ppo_policy import PPO_GROUPS, PPOConfig, PPOPolicy
from rl_manager.policy import JaxEPlanPolicy, params_fingerprint
from rl_manager.runner import (
    GAME_TURNS,
    RunnerConfig,
    SelfPlayRunner,
    build_episode_spec,
)
from rl_manager.seeds import SeedStream
from rl_manager.trajectory import TrajectoryBuffer, e_input_spec
from rl_manager.types import CANDIDATE_VS_FROZEN

MASTER_SEED = 17
NUM_MANAGER_DAYS = 26
TOTAL_TRANSITIONS = 2 * NUM_MANAGER_DAYS
REAL_E_CHECKPOINT = Path("artifacts/local/bc-v1-E/best.pt")


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


def _fixed_batch():
    rows = [_encoded(day=day, money=1000.0 + 100.0 * day)
            for day in (4, 5, 6, 7)]
    return {key: np.concatenate([row[key] for row in rows], axis=0)
            for key in rows[0]}


def _leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


# --------------------------------------------------------------------------
# The single authorized complete fast game + full pipeline (cached once).
# --------------------------------------------------------------------------

_SMOKE_CACHE: dict[str, object] = {}


class _Smoke:
    """Everything the end-to-end assertions need, built exactly once."""

    def __init__(self) -> None:
        self.config = tiny_manager_config()
        self.frozen_params = init_params(self.config, seed=11,
                                         model_variant="E")
        self.ppo_config = PPOConfig(minibatch_size=4, epochs=1)
        self.state = init_train_state(
            self.frozen_params, self.config, seed=42,
            ppo_config=self.ppo_config)
        self.candidate = ppo_batched_policy_from_state(
            self.state, self.config, ppo_config=self.ppo_config,
            name="ppo_candidate", version="ppo-v0-smoke")
        self.pre_fingerprint = self.candidate.identity.fingerprint
        self.frozen_policy = JaxEPlanPolicy(
            self.frozen_params, self.config, name="frozen_e")

        self.buffer = TrajectoryBuffer(capacity=64,
                                       input_spec=e_input_spec())
        self.runner = SelfPlayRunner(
            RunnerConfig(
                backend_name="fast",
                backend_configuration={"seed": 0, "numThreads": 1},
                max_turns=GAME_TURNS, num_envs=1),
            trajectory_buffer=self.buffer, master_seed=MASTER_SEED)
        spec = build_episode_spec(
            0, SeedStream(MASTER_SEED).episode_seed(0),
            CANDIDATE_VS_FROZEN, self.candidate, self.frozen_policy)
        (self.result,) = self.runner.run([spec])
        self.arrays = self.buffer.finalize()
        self.batch = build_ppo_batch(
            self.arrays, gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda)


@pytest.fixture(scope="module")
def smoke() -> _Smoke:
    if "smoke" not in _SMOKE_CACHE:
        _SMOKE_CACHE["smoke"] = _Smoke()
    return _SMOKE_CACHE["smoke"]  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Adapter unit behavior (no games).
# --------------------------------------------------------------------------


def test_adapter_exact_output_fields_and_single_batched_call(smoke):
    policy = PPOPolicy(smoke.frozen_params, smoke.config, seed=7)
    adapter = PPOBatchedPolicy(policy, name="adapter", version="v-test")
    batch = _fixed_batch()
    out = adapter.plan_batch(batch, "episode=0/day=4/policy=adapter")

    assert out.batch_size == 4
    assert set(out.logprob_groups) == set(LOGPROB_GROUPS) == set(PPO_GROUPS)
    assert out.logprob_total.shape == (4,)
    assert out.value.shape == (4,)
    for group in LOGPROB_GROUPS:
        assert out.logprob_groups[group].shape == (4,)
        assert bool(np.all(np.isfinite(out.logprob_groups[group])))
    for name, shape in ACTION_TENSOR_SHAPES.items():
        assert np.asarray(out.action_tensors[name]).shape == (4,) + shape
    assert adapter.identity.name == "adapter"
    assert adapter.identity.version == "v-test"
    assert adapter.identity.fingerprint == params_fingerprint(policy.params)
    # ONE batched call for the whole request batch — never per row/env.
    assert adapter.call_count == 1
    assert adapter.batch_size_history == [4]


def test_adapter_stochastic_seeds_deterministic_in_prng_id(smoke):
    policy = PPOPolicy(smoke.frozen_params, smoke.config, seed=9)
    adapter = PPOBatchedPolicy(policy)
    batch = _fixed_batch()
    again = adapter.plan_batch(batch, "episode=3/day=9/policy=x")
    repeat = adapter.plan_batch(batch, "episode=3/day=9/policy=x")
    other = adapter.plan_batch(batch, "episode=3/day=10/policy=x")
    for name in ACTION_TENSOR_SHAPES:
        assert np.array_equal(again.action_tensors[name],
                              repeat.action_tensors[name]), name
    assert any(not np.array_equal(again.action_tensors[name],
                                  other.action_tensors[name])
               for name in ACTION_TENSOR_SHAPES)
    # Root key derivation itself is stable and validated.
    key_a = prng_key_from_id("same-id")
    key_b = prng_key_from_id("same-id")
    assert np.array_equal(np.asarray(key_a), np.asarray(key_b))
    with pytest.raises(ValueError, match="prng_id"):
        prng_key_from_id("")


def test_row_aware_sampling_is_invariant_to_neighbors_and_padding(smoke):
    policy = PPOPolicy(smoke.frozen_params, smoke.config, seed=19)
    adapter = PPOBatchedPolicy(policy, name="row-aware", version="v-test")
    batch = _fixed_batch()
    target_index = 1
    target_id = "episode=42/seat=1/day=5/policy=row-aware"
    target = {key: np.ascontiguousarray(value[target_index:target_index + 1])
              for key, value in batch.items()}
    alone = adapter.plan_batch_with_row_ids(
        target, [target_id], "scheduler/group-alone")
    composed = {
        key: np.ascontiguousarray(np.concatenate(
            [value[target_index:target_index + 1], value[0:1]], axis=0))
        for key, value in batch.items()}
    together = adapter.plan_batch_with_row_ids(
        composed, [target_id, "neighbor"], "scheduler/group-together")
    padded = {
        key: np.ascontiguousarray(np.concatenate(
            [value[target_index:target_index + 1], value[0:1], value[0:1]],
            axis=0))
        for key, value in batch.items()}
    with_padding = adapter.plan_batch_with_row_ids(
        padded, [target_id, "neighbor", "padding/0"],
        "scheduler/group-padding")
    for name in ACTION_TENSOR_SHAPES:
        assert np.array_equal(alone.action_tensors[name],
                              together.action_tensors[name][0:1]), name
        assert np.array_equal(alone.action_tensors[name],
                              with_padding.action_tensors[name][0:1]), name
    for name in LOGPROB_GROUPS:
        assert np.array_equal(alone.logprob_groups[name],
                              together.logprob_groups[name][0:1]), name
        assert np.array_equal(alone.logprob_groups[name],
                              with_padding.logprob_groups[name][0:1]), name
    assert np.array_equal(alone.logprob_total, together.logprob_total[0:1])
    assert np.allclose(alone.value, together.value[0:1], rtol=0, atol=1e-6)


def test_adapter_deterministic_mode_reproduces_frozen_e_exactly(smoke):
    greedy = ppo_batched_policy_from_state(
        smoke.state, smoke.config, ppo_config=smoke.ppo_config,
        deterministic=True)
    batch = _fixed_batch()
    reference = smoke.frozen_policy.plan_batch(batch, "prng/ref")
    result = greedy.plan_batch(batch, "prng/ref")
    for name in ACTION_TENSOR_SHAPES:
        assert np.array_equal(result.action_tensors[name],
                              np.asarray(reference.action_tensors[name])), name


def test_adapter_rejects_leaked_non_own_inputs(smoke):
    adapter = ppo_batched_policy_from_state(
        smoke.state, smoke.config, ppo_config=smoke.ppo_config)
    leaked = dict(_fixed_batch())
    leaked["opponent_public_board"] = \
        np.zeros((4,) + leaked["board_kind"].shape[1:], dtype=np.int32)
    with pytest.raises(ValueError, match="own-only E contract"):
        adapter.plan_batch(leaked, "prng/x")


# --------------------------------------------------------------------------
# Tiny end-to-end smoke over the ONE cached complete game.
# --------------------------------------------------------------------------


def test_complete_game_rollout_shape_and_provenance(smoke):
    assert smoke.result.statuses == ["DONE", "DONE"]
    assert smoke.result.terminated is True
    assert smoke.result.transitions == TOTAL_TRANSITIONS
    assert len(smoke.buffer) == TOTAL_TRANSITIONS
    arrays = smoke.arrays
    candidate = ((arrays["valid"] == 1) & (arrays["trainable"] == 1))
    assert int(candidate.sum()) == NUM_MANAGER_DAYS          # seat 0 only
    assert int(arrays["seat"][candidate].max()) == 0
    assert int(arrays["seat"][~candidate].max()) == 1        # frozen seat
    assert bool(np.all(np.isfinite(arrays["logprob_total"][candidate])))
    assert bool(np.all(np.isfinite(arrays["value"][candidate])))
    # Terminal ±1 reward landed on the candidate's final manager row.
    last_day_rows = np.flatnonzero(arrays["day"] == 29)
    rewards = {(int(arrays["seat"][r])): float(arrays["reward"][r])
               for r in last_day_rows}
    assert set(rewards) == {0, 1}
    assert tuple(sorted(rewards.values())) in ((-1.0, 1.0), (0.0, 0.0))


def test_trajectory_to_gae_to_subset_to_update_pipeline(smoke, tmp_path):
    batch = smoke.batch
    # GAE ran over the FULL candidate trajectory and normalized advantages.
    assert batch.size == NUM_MANAGER_DAYS
    assert abs(float(np.mean(batch.advantages))) < 1e-6
    gae_reference = compute_gae(
        rewards=smoke.arrays["reward"][
            smoke.arrays["trainable"] == 1],
        values=smoke.arrays["value"][smoke.arrays["trainable"] == 1],
        terminated=smoke.arrays["terminated"][
            smoke.arrays["trainable"] == 1],
        truncated=smoke.arrays["truncated"][
            smoke.arrays["trainable"] == 1],
        episode_index=smoke.arrays["episode_index"][
            smoke.arrays["trainable"] == 1],
        seat=smoke.arrays["seat"][smoke.arrays["trainable"] == 1],
        day=smoke.arrays["day"][smoke.arrays["trainable"] == 1],
        gamma=smoke.ppo_config.gamma,
        gae_lambda=smoke.ppo_config.gae_lambda, normalize=True)
    assert np.allclose(gae_reference["advantages"], batch.advantages)

    subset = select_ppo_subset(batch, 4)
    assert subset.size == 4
    # Evenly-spaced deterministic selection over 26 rows -> indices 0,8,17,25.
    expected_indices = np.round(np.linspace(0, 25, 4)).astype(np.int64)
    assert np.array_equal(
        np.asarray(subset.old_logprob),
        np.asarray(batch.old_logprob)[expected_indices])
    assert np.array_equal(
        np.asarray(subset.advantages),
        np.asarray(batch.advantages)[expected_indices])
    # Exact stored-action logprob recompute under the UNCHANGED rollout
    # params: no resampling anywhere.
    policy = ppo_batched_policy_from_state(
        smoke.state, smoke.config, ppo_config=smoke.ppo_config)._policy
    recomputed = policy.evaluate_actions(subset.inputs, subset.action_tensors)
    assert np.array_equal(recomputed["logprob_total"], subset.old_logprob)

    # Exactly ONE PPO update call / one epoch / one 4-row minibatch.
    new_state, metrics = ppo_update(smoke.state, subset, smoke.config,
                                    smoke.ppo_config)
    assert new_state.step == 1
    for key, value in metrics.items():
        assert np.isfinite(float(value)), key
    assert float(metrics["clip_fraction"]) >= 0.0

    # Trainable/value params changed; immutable frozen snapshot did not;
    # sell quantities always derive from that unchanged snapshot.
    before = _leaves(smoke.state.params)
    after = _leaves(new_state.params)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    assert not any(not np.array_equal(a, b) for a, b in zip(
        _leaves(smoke.state.frozen_params), _leaves(new_state.frozen_params)))
    value_changed = not np.array_equal(
        np.asarray(smoke.state.params["value"]["kernel"]),
        np.asarray(new_state.params["value"]["kernel"]))
    assert value_changed

    # Checkpoint save/load: every leaf including explicit PRNG state.
    path = save_ppo_checkpoint(tmp_path / "smoke.npz", new_state,
                               smoke.config, smoke.ppo_config,
                               provenance={"master_seed": MASTER_SEED})
    loaded, meta = load_ppo_checkpoint(path, config=smoke.config,
                                       ppo_config=smoke.ppo_config)
    assert meta["step"] == 1
    for got, want in zip(_leaves(loaded.params), _leaves(new_state.params)):
        assert np.array_equal(got, want)
    assert np.array_equal(np.asarray(loaded.rng), np.asarray(new_state.rng))

    # Loaded deterministic eval identical to pre-save deterministic eval.
    greedy_pre = ppo_batched_policy_from_state(
        new_state, smoke.config, ppo_config=smoke.ppo_config,
        deterministic=True)
    greedy_loaded = ppo_batched_policy_from_state(
        loaded, smoke.config, ppo_config=smoke.ppo_config,
        deterministic=True)
    fixed = _fixed_batch()
    pre_out = greedy_pre.plan_batch(fixed, "eval/fixed")
    post_out = greedy_loaded.plan_batch(fixed, "eval/fixed")
    for name in ACTION_TENSOR_SHAPES:
        assert np.array_equal(pre_out.action_tensors[name],
                              post_out.action_tensors[name]), name
    assert np.array_equal(pre_out.value, post_out.value)

    # Explicit PRNG/rollout state resumes bit-identically.
    continued_a, _ = ppo_update(new_state, subset, smoke.config,
                                smoke.ppo_config)
    continued_b, _ = ppo_update(loaded, subset, smoke.config,
                                smoke.ppo_config)
    for got, want in zip(_leaves(continued_b.params),
                         _leaves(continued_a.params)):
        assert np.array_equal(got, want)

    # Post-update drift is real but bounded-plumbing only (no quality read).
    post_fingerprint = params_fingerprint(new_state.params)
    assert post_fingerprint != smoke.pre_fingerprint
    # One tiny-lr step moves the distribution only marginally; the analytic
    # KL sits at float noise around zero — assert finiteness + near-zero,
    # never a quality direction.
    kl_frozen = float(metrics["kl_to_frozen"])
    assert abs(kl_frozen) < 1e-3


def test_diagnostics_artifact_written_json_safe(smoke, tmp_path):
    updated_state, metrics = ppo_update(
        smoke.state, select_ppo_subset(smoke.batch, 4),
        smoke.config, smoke.ppo_config)
    prov = {key: value for key, value in smoke.runner.provenance.items()
            if key != "executor_factory"}
    prov["executor_factory"] = \
        smoke.runner.provenance["executor_factory_version"]
    payload = build_integration_diagnostics(
        result=smoke.result,
        runner_timing=smoke.runner.timing_totals,
        sidecar_records=smoke.buffer.sidecar_records,
        update_metrics=metrics,
        provenance=prov,
        pre_update_fingerprint=smoke.pre_fingerprint,
        post_update_fingerprint=params_fingerprint(updated_state.params),
        checkpoint_path=str(tmp_path / "smoke.npz"),
    )
    assert payload["diagnostics_schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert payload["rollout"]["manager_steps"] == TOTAL_TRANSITIONS
    assert set(payload["timing_seconds"]) == {
        "env", "executor", "policy", "orchestration"}
    assert set(payload["ppo_metrics"]["entropy_by_group"]) == set(PPO_GROUPS)
    assert abs(payload["ppo_metrics"]["action_drift_kl_to_frozen"]) < 1e-3
    assert payload["fingerprints"]["pre_update"] != \
        payload["fingerprints"]["post_update"]
    assert payload["provenance"]["opening"]["name"] == "standard_mixed"

    path = write_diagnostics(tmp_path / "diagnostics.json", payload)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == json.loads(json.dumps(payload))  # strictly JSON-safe
    for key, reason in payload["missing"].items():
        section = parsed
        for token in key.split("."):
            section = section[token]
        assert section is None and reason


def test_missing_sections_are_honest_nulls_with_reasons(tmp_path):
    payload = build_integration_diagnostics()
    assert payload["rollout"] is None
    assert payload["timing_seconds"] is None
    assert payload["ppo_metrics"] is None
    assert payload["missing"]["rollout"] == "no episode result recorded"
    path = write_diagnostics(tmp_path / "empty.json", payload)
    assert json.loads(path.read_text(encoding="utf-8"))["anomalies"] == []


# --------------------------------------------------------------------------
# Gated skips (official engine dependency / real BC-E checkpoint absent).
# --------------------------------------------------------------------------


def test_official_engine_gate():
    from rl_manager.parity import OFFICIAL_BLOCKER_COMMAND, \
        official_backend_available

    if official_backend_available():
        pytest.skip("official engine present locally; gate not exercised")
    pytest.skip(
        "official kaggle_environments dependency absent in this interpreter; "
        f"rerun where available: {OFFICIAL_BLOCKER_COMMAND}")


def test_real_e_checkpoint_gate():
    if REAL_E_CHECKPOINT.is_file():
        pytest.skip("real BC-E checkpoint present locally; gate not exercised")
    pytest.skip(
        f"real BC-E checkpoint absent at {REAL_E_CHECKPOINT}; rerun after "
        f"placing it there: python -m rl_manager.cli train --e-checkpoint "
        f"{REAL_E_CHECKPOINT} ...")
