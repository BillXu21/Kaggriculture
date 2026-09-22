"""Packet 5A integration checks for the centralized Stage 2.5 owner."""

from __future__ import annotations

from dataclasses import fields, replace
from queue import Queue
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from importlib.util import find_spec

from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.parallel_worker import RemotePlanPolicy
from rl_manager.parallel_protocol import (
    Stage25BootstrapRequest,
    Stage25InferenceRequest,
    Stage25InferenceResponse,
    Stage25RequestIdentity,
)
from rl_manager.runner import RunnerConfig
from rl_manager.runner import build_episode_spec
from rl_manager.seeds import SeedStream
from rl_manager.stage25_provider import (
    Stage25DecisionKey,
    Stage25PlanProvider,
    Stage25ProviderError,
    Stage25TerminalError,
)
from rl_manager.stage25_types import (
    Stage25BehaviorIdentity, Stage25PolicyOutputs, stage25_row_token,
)
from rl_manager.stage25_trajectory import Stage25TrajectoryBuffer

from test_stage25_provider import HOLD, _obs


IDENTITY = Stage25BehaviorIdentity(
    name="stage25-test", version="v1", parameter_fingerprint="f" * 64,
    observation_schema_version="e_v1", policy_schema_version="stage25_policy_v1",
    e_history_version="E_CORRECTED_V1", curriculum_version="stage25_curriculum_v1",
    curriculum_fingerprint="7282cd9c883618c0258b2105f40e3d7996c969853a5adc6db07a57a12da9d1ec")


class _Stage25Policy:
    identity = IDENTITY
    behavior_identity = IDENTITY

    def __init__(self) -> None:
        self.row_ids: list[list[str]] = []
        self.row_tokens: list[list[int]] = []
        self.prng_ids: list[str] = []
        self.supports_precomputed_row_tokens = True

    def infer_batch(self, *, inputs, crop_capacity, physical_contexts,
                    supports, row_ids, prng_id, row_tokens=None):
        del inputs, supports
        self.row_ids.append(list(row_ids))
        self.row_tokens.append(list(row_tokens) if row_tokens is not None else [])
        self.prng_ids.append(prng_id)
        batch = len(row_ids)
        classes = np.zeros((batch, 9), dtype=np.int16)
        for row, context in enumerate(physical_contexts):
            classes[row, 0] = context.observed_land - 1
            classes[row, 1:4] = np.asarray(context.placed_animals, dtype=np.int16)
            classes[row, 4:] = 100
        return Stage25PolicyOutputs(
            classes=classes,
            component_logprobs=np.zeros((batch, 9), dtype=np.float32),
            joint_logprob=np.zeros(batch, dtype=np.float32),
            value=np.asarray(crop_capacity[:, 0], dtype=np.float32),
            decoded_goals=np.asarray(crop_capacity, dtype=np.int16),
            valid=np.ones(batch, dtype=np.bool_),
            policy_identity=IDENTITY, batch_size=batch)

    def bootstrap_value(self, **kwargs):
        inputs = kwargs["inputs"]
        batch = int(np.asarray(next(iter(inputs.values()))).shape[0])
        return np.full(batch, 3.25, dtype=np.float32)


def _request(index: int, day: int = 4) -> Stage25InferenceRequest:
    provider = Stage25PlanProvider(
        index, 0, day, behavior_identity=IDENTITY)
    prepared = provider.prepare_inference_context(
        _obs(day=day), behavior_identity=IDENTITY)
    identity = Stage25RequestIdentity(index, 0, day, IDENTITY)
    return Stage25InferenceRequest(
        identity=identity, worker_id=0, prng_id="test-prng",
        row_token=stage25_row_token(identity.request_id),
        inputs={name: value for name, value in prepared.inputs.items()
                if name != "crop_capacity"},
        crop_capacity=np.asarray([prepared.crop_capacity], dtype=np.int16),
        physical_context=prepared.physical_context, support=prepared.support,
        queued_at=0.0)


def _bootstrap_request(index: int, day: int = 4) -> Stage25BootstrapRequest:
    provider = Stage25PlanProvider(
        index, 0, day, behavior_identity=IDENTITY)
    prepared = provider.prepare_bootstrap_context(_obs(day=day))
    identity = Stage25RequestIdentity(index, 0, day, IDENTITY)
    return Stage25BootstrapRequest(
        identity=identity, worker_id=0,
        inputs={name: value for name, value in prepared.inputs.items()
                if name != "crop_capacity"},
        crop_capacity=np.asarray([prepared.crop_capacity], dtype=np.int16),
        physical_context=prepared.physical_context, queued_at=0.0)


def _runner(size: int = 4) -> ParallelSelfPlayRunner:
    return ParallelSelfPlayRunner(
        RunnerConfig(stage25_enabled=True,
                     stage25_fixed_inference_batch_size=size), num_workers=1)


def test_stage25_wire_identity_is_routing_only_and_request_has_one_payload_copy():
    request = _request(17)
    assert {field.name for field in fields(Stage25RequestIdentity)} == {
        "episode_index", "seat", "day", "behavior_identity"}
    assert "crop_capacity" not in request.inputs
    assert request.identity.behavior_identity == IDENTITY
    assert request.crop_capacity.shape == (1, 5)
    assert request.physical_context is not None
    assert request.support is not None


def test_stage25_response_routes_without_echoing_physical_payload():
    request = _request(18)
    output = _Stage25Policy().infer_batch(
        inputs=request.inputs, crop_capacity=request.crop_capacity,
        physical_contexts=(request.physical_context,),
        supports=(request.support,), row_ids=(request.request_id,),
        prng_id=request.prng_id)
    response = Stage25InferenceResponse(
        request.request_id, request.identity, output)
    assert {field.name for field in fields(response)} == {
        "request_id", "identity", "outputs"}
    assert not hasattr(response.identity, "physical_context")
    assert not hasattr(response.identity, "crop_capacity")
    assert not hasattr(response.identity, "support")
    wrong = replace(output, policy_identity=replace(
        IDENTITY, name="wrong-policy"))
    with pytest.raises(ValueError, match="behavior identity"):
        Stage25InferenceResponse(request.request_id, request.identity, wrong)


def test_stage25_provider_materializes_support_only_in_strict(monkeypatch):
    calls = []
    for mode in ("strict", "fast", "none"):
        provider = Stage25PlanProvider(
            19, 0, 4, behavior_identity=IDENTITY,
            validation_mode=mode)
        original = provider._support_payload

        def counted(context, initial, *, _original=original):
            calls.append(mode)
            return _original(context, initial)

        monkeypatch.setattr(provider, "_support_payload", counted)
        context = provider.prepare_inference_context(
            _obs(day=4), behavior_identity=IDENTITY)
        assert (context.support is not None) is (mode == "strict")
    assert calls == ["strict"]


@pytest.mark.parametrize("mode", ("fast", "none"))
def test_stage25_compact_worker_request_omits_support_and_capacity_from_inputs(mode):
    provider = Stage25PlanProvider(
        21, 0, 4, behavior_identity=IDENTITY, validation_mode=mode)
    context = provider.prepare_inference_context(
        _obs(day=4), behavior_identity=IDENTITY)

    class RequestQueue:
        def __init__(self):
            self.message = None

        def put(self, message):
            self.message = message

    request_queue = RequestQueue()

    class ResponseQueue:
        def get(self):
            request = request_queue.message
            output = _Stage25Policy().infer_batch(
                inputs=request.inputs, crop_capacity=request.crop_capacity,
                physical_contexts=(request.physical_context,), supports=None,
                row_ids=(request.request_id,), prng_id=request.prng_id)
            return Stage25InferenceResponse(
                request.request_id, request.identity, output)

    response_queue = ResponseQueue()
    policy = RemotePlanPolicy(
        IDENTITY, request_queue, response_queue, 0,
        validation_mode=mode)
    policy.set_request_context([context])
    policy._stage25_plan_batch(context.inputs, "compact", [context])
    assert request_queue.message.support is None
    assert request_queue.message.row_token == stage25_row_token(
        request_queue.message.request_id)
    assert "crop_capacity" not in request_queue.message.inputs


def test_stage25_bootstrap_never_materializes_support(monkeypatch):
    provider = Stage25PlanProvider(
        20, 0, 4, behavior_identity=IDENTITY, validation_mode="strict")
    monkeypatch.setattr(
        provider, "_support_payload",
        lambda *_args: pytest.fail("bootstrap constructed diagnostic support"))
    context = provider.prepare_bootstrap_context(_obs(day=4))
    assert context.support is None


def test_stage25_parent_padding_has_no_extra_responses_and_stable_rows():
    class NonzeroPolicy(_Stage25Policy):
        def infer_batch(self, **kwargs):
            output = super().infer_batch(**kwargs)
            output.component_logprobs[:, 0] = -1.0
            output.joint_logprob[:] = -1.0
            return output

    policy = NonzeroPolicy()
    requests = [_request(9), _request(2)]
    queue = Queue()
    runner = _runner(4)
    runner._dispatch(IDENTITY, requests, 0.0, {IDENTITY: policy}, [queue])

    assert len(policy.row_ids) == 1
    assert policy.row_ids[0][0] == requests[1].request_id
    assert policy.row_ids[0][1] == requests[0].request_id
    assert policy.row_ids[0][2].startswith("padding/")
    assert policy.row_tokens[0] == [
        requests[1].row_token, requests[0].row_token, requests[1].row_token,
        requests[1].row_token]
    assert len([queue.get_nowait() for _ in range(2)]) == 2
    assert runner.inference_metrics["real_requests"] == 2
    assert runner.inference_metrics["padding_rows"] == 2


def test_stage25_worker_import_boundary_is_accelerator_free():
    script = """
import sys
import rl_manager.parallel_worker
assert not any(name == 'jax' or name.startswith('jax.') or
               name == 'torch' or name.startswith('torch.')
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stage25_row_reordering_does_not_change_owner_row_rng_tokens():
    first_policy = _Stage25Policy()
    first_requests = [_request(9), _request(2)]
    _runner(4)._dispatch(
        IDENTITY, first_requests, 0.0, {IDENTITY: first_policy}, [Queue()])
    second_policy = _Stage25Policy()
    second_requests = list(reversed([_request(9), _request(2)]))
    _runner(4)._dispatch(
        IDENTITY, second_requests, 0.0, {IDENTITY: second_policy}, [Queue()])
    assert first_policy.row_ids == second_policy.row_ids


def test_stage25_runner_and_parent_use_the_same_rng_namespace():
    from rl_manager.runner import SelfPlayRunner
    from rl_manager.stage25_types import stage25_rng_namespace

    policy = _Stage25Policy()
    request = _request(3)
    runner = _runner(1)
    runner._dispatch(IDENTITY, [request], 0.0, {IDENTITY: policy}, [Queue()])
    parent_namespace = policy.prng_ids[-1]
    SelfPlayRunner._stage25_policy_batch(
        policy, request.inputs, request.crop_capacity,
        (request.physical_context,), (request.support,),
        (request.request_id,), stage25_rng_namespace(IDENTITY))
    assert parent_namespace == policy.prng_ids[-1]


def test_stage25_inference_internal_type_error_is_not_retried():
    from rl_manager.runner import SelfPlayRunner

    class RaisingPolicy(_Stage25Policy):
        def infer_batch(self, **kwargs):
            del kwargs
            raise TypeError("owner implementation failure")

    request = _request(4)
    with pytest.raises(TypeError, match="owner implementation failure"):
        SelfPlayRunner._stage25_policy_batch(
            RaisingPolicy(), request.inputs, request.crop_capacity,
            (request.physical_context,), (request.support,),
            (request.request_id,), "direct-test")


def test_stage25_mixed_decision_and_bootstrap_batch_is_routed_by_type():
    from rl_manager.parallel_protocol import (
        Stage25BootstrapResponse,
        Stage25InferenceResponse,
    )

    for requests in (
            [_request(9, 4), _bootstrap_request(9, 4)],
            [_bootstrap_request(9, 4), _request(9, 4)]):
        policy = _Stage25Policy()
        queue = Queue()
        runner = _runner(2)
        runner._dispatch(IDENTITY, requests, 0.0, {IDENTITY: policy}, [queue])
        types = sorted(type(queue.get_nowait()).__name__ for _ in range(2))
        assert types == [Stage25BootstrapResponse.__name__,
                         Stage25InferenceResponse.__name__]
        assert runner.inference_metrics["mixed_request_batches"] == 1


def test_stage25_provider_binds_curriculum_to_behavior_identity():
    from rl_manager.stage25_config import Stage25CurriculumConfig
    from rl_manager.stage25_inference import curriculum_fingerprint

    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    identity = Stage25BehaviorIdentity(
        name="stage25-test", version="v1", parameter_fingerprint="f" * 64,
        observation_schema_version="e_v1",
        policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1",
        curriculum_version=enabled.version,
        curriculum_fingerprint=curriculum_fingerprint(enabled))
    provider = Stage25PlanProvider(
        7, 0, 4, behavior_identity=identity, curriculum=enabled)
    context = provider.prepare_inference_context(
        _obs(day=4), behavior_identity=identity)
    violating = (0, 1, 0, 1, 105, 100, 100, 100, 100)
    with pytest.raises(Stage25ProviderError, match="curriculum"):
        provider.accept_inference_response(
            context, violating, behavior_identity=identity)


def test_stage25_worker_wire_adopts_checkpoint_curriculum_without_override():
    from types import SimpleNamespace
    from rl_manager.stage25_config import Stage25CurriculumConfig
    from rl_manager.stage25_inference import curriculum_fingerprint

    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    identity = Stage25BehaviorIdentity(
        name="stage25-test", version="v1", parameter_fingerprint="f" * 64,
        observation_schema_version="e_v1", policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1", curriculum_version=enabled.version,
        curriculum_fingerprint=curriculum_fingerprint(enabled))
    checkpoint = SimpleNamespace(load_config=lambda: SimpleNamespace(
        curriculum=enabled))
    provider = Stage25PlanProvider(
        7, 0, 4, native_policy=checkpoint, behavior_identity=identity)
    assert provider.effective_curriculum() == enabled


def test_stage25_curriculum_mismatch_is_rejected_before_provider_mutation():
    from types import SimpleNamespace
    from rl_manager.stage25_config import Stage25CurriculumConfig
    from rl_manager.stage25_inference import curriculum_fingerprint

    enabled = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=1)
    identity = Stage25BehaviorIdentity(
        name="stage25-test", version="v1", parameter_fingerprint="f" * 64,
        observation_schema_version="e_v1", policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1", curriculum_version=enabled.version,
        curriculum_fingerprint=curriculum_fingerprint(enabled))
    checkpoint = SimpleNamespace(load_config=lambda: SimpleNamespace(
        curriculum=enabled))
    provider = Stage25PlanProvider(
        7, 0, 4, native_policy=checkpoint,
        curriculum=Stage25CurriculumConfig(), behavior_identity=identity)
    before = (provider.crop_capacity, provider.last_accepted_decision,
              provider._bound_curriculum)
    with pytest.raises(Stage25ProviderError, match="curriculum"):
        provider.effective_curriculum()
    assert (provider.crop_capacity, provider.last_accepted_decision,
            provider._bound_curriculum) == before


def test_stage25_unsupported_manager_start_day_is_rejected_at_startup():
    with pytest.raises(ValueError, match="manager_start_day=4"):
        RunnerConfig(stage25_enabled=True, manager_start_day=3)


def test_stage25_response_identity_mismatch_leaves_provider_unchanged():
    provider = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    prepared = provider.prepare_inference_context(
        _obs(day=4), behavior_identity=IDENTITY)
    before = provider.export_state()
    wrong = Stage25BehaviorIdentity(
        name="other", version="v1", parameter_fingerprint="a" * 64,
        observation_schema_version="e_v1", policy_schema_version="stage25_policy_v1",
        e_history_version="E_CORRECTED_V1", curriculum_version="stage25_curriculum_v1",
        curriculum_fingerprint="c" * 64)
    with pytest.raises(Stage25ProviderError, match="identity"):
        provider.accept_inference_response(
            prepared, HOLD, behavior_identity=wrong)
    assert provider.export_state() == before


def test_stage25_acceptance_reuses_one_frozen_prepared_context(monkeypatch):
    provider = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    obs = _obs(day=4, money=3000.0)
    calls = 0
    original = provider._stage_observation

    def counted(observation, previous_execution):
        nonlocal calls
        calls += 1
        return original(observation, previous_execution)

    monkeypatch.setattr(provider, "_stage_observation", counted)
    prepared = provider.prepare_inference_context(obs, behavior_identity=IDENTITY)
    assert calls == 1
    assert prepared.inputs["board_kind"].flags.writeable is False
    with pytest.raises(ValueError):
        prepared.inputs["board_kind"][0, 0] = 99
    with pytest.raises(TypeError):
        prepared.support["land"] = ()

    # The caller's live observation may advance while inference is outstanding;
    # acceptance must use the retained daily-start and encoded premises.
    obs["farms"][0]["money"] = 123.0
    monkeypatch.setattr(
        provider, "_stage_observation",
        lambda *_args: pytest.fail("response acceptance re-staged observation"),
    )
    provider.accept_inference_response(prepared, HOLD, behavior_identity=IDENTITY)
    assert calls == 1
    assert provider.e_history == (4, 3000.0)


def test_stage25_prepared_context_rejects_stale_or_wrong_identity_without_commit():
    provider = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    prepared = provider.prepare_inference_context(_obs(day=4), behavior_identity=IDENTITY)
    before = provider.export_state()
    stale_owner = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    with pytest.raises(Stage25ProviderError, match="pending request"):
        stale_owner.accept_inference_response(
            prepared, HOLD, behavior_identity=IDENTITY)
    for wrong_key in (
            replace(prepared,
                    decision_key=Stage25DecisionKey(7, 1, 4, "wrong-seat")),
            replace(prepared,
                    decision_key=Stage25DecisionKey(7, 0, 5, "wrong-day")),
    ):
        with pytest.raises(Stage25ProviderError, match="pending request"):
            provider.accept_inference_response(
                wrong_key, HOLD, behavior_identity=IDENTITY)
    assert provider.export_state() == before

    provider.accept_inference_response(prepared, HOLD, behavior_identity=IDENTITY)
    with pytest.raises(Stage25ProviderError, match="pending request"):
        provider.accept_inference_response(prepared, HOLD, behavior_identity=IDENTITY)
    assert provider.last_accepted_decision == prepared.decision_key


def test_stage25_reset_and_terminal_delivery_invalidate_pending_context():
    provider = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    prepared = provider.prepare_inference_context(_obs(day=4), behavior_identity=IDENTITY)
    with pytest.raises(Stage25TerminalError):
        provider.prepare_inference_context(
            {**_obs(day=4), "terminal": True}, behavior_identity=IDENTITY)
    provider.reset()
    with pytest.raises(Stage25ProviderError, match="pending request"):
        provider.accept_inference_response(prepared, HOLD, behavior_identity=IDENTITY)


def test_stage25_provider_transitions_k_once_and_rejects_terminal_delivery():
    provider = Stage25PlanProvider(7, 0, 4, behavior_identity=IDENTITY)
    prepared = provider.prepare_inference_context(
        _obs(day=4), behavior_identity=IDENTITY)
    provider.accept_inference_response(prepared, HOLD, behavior_identity=IDENTITY)
    assert provider.crop_capacity == (1, 0, 0, 0, 0)
    with pytest.raises(Stage25ProviderError):
        provider.accept_classes(_obs(day=4), HOLD)
    snapshot = provider.export_state()
    with pytest.raises(Stage25TerminalError):
        provider.accept_classes(_obs(day=5), HOLD, terminal=True)
    assert provider.export_state() == snapshot


def test_stage25_config_defaults_to_fixed_smoke_physical_batch():
    assert RunnerConfig(stage25=True).stage25_enabled
    assert RunnerConfig(stage25_enabled=True).stage25_fixed_inference_batch_size == 16


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_stage25_spawned_worker_fast_engine_smoke(tmp_path):
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config

    model_config = tiny_stage25_config()
    policy = Stage25InferenceAdapter(
        params=init_stage25_params(model_config, seed=23),
        config=model_config, mode="stochastic", name="stage25-smoke")
    config = RunnerConfig(
        stage25_enabled=True, stage25_mode="stochastic",
        manager_start_day=4, max_turns=144,
        openings=("none", "none"), low_telemetry=True,
        stage25_fixed_inference_batch_size=2)
    specs = [build_episode_spec(index, SeedStream(17).episode_seed(index),
                                 "e_vs_e", policy, policy)
             for index in range(6)]
    parallel = ParallelSelfPlayRunner(
        config, num_workers=2, inference_batch_wait_seconds=0.01,
        stage25_trajectory_buffer=Stage25TrajectoryBuffer(64))
    results = parallel.run(specs)
    assert len(results) == 6
    assert [result.episode_index for result in results] == list(range(6))
    assert all(result.transitions == 4 for result in results)
    metrics = parallel.inference_metrics
    assert metrics["real_requests"] == 24
    assert metrics["bootstrap_requests"] > 0
    assert metrics["physical_batch_sizes"]
    assert set(metrics["physical_batch_sizes"]) == {2}
    assert metrics["physical_rows"] >= metrics["real_requests"]
    assert metrics["animal_placement_rows"] == 24
    print("stage25 smoke inference metrics:", {
        key: metrics[key] for key in (
            "real_requests", "logical_requests", "bootstrap_requests",
            "physical_inference_calls", "real_batch_sizes",
            "physical_batch_sizes", "physical_rows", "padding_rows",
            "occupancy", "animal_placement_nonzero_classes")})
    assert len(parallel.stage25_trajectory) == 24
    arrays = parallel.stage25_trajectory.finalize()
    assert np.sum(arrays["terminated"] | arrays["truncated"]) == 12
    end_rows = arrays["terminated"] | arrays["truncated"]
    assert np.all(arrays["day"][end_rows] == arrays["day"][end_rows].max())
    assert np.all(arrays["bootstrap_patched"] == arrays["truncated"])
    path = parallel.stage25_trajectory.save(tmp_path / "smoke")
    reloaded, _ = Stage25TrajectoryBuffer.load(path)
    assert len(reloaded) == 24


@pytest.mark.skipif(find_spec("fast_env._kaggriculture_env") is None,
                    reason="native fast_env extension is unavailable")
def test_stage25_local_and_parallel_sampling_agree(tmp_path):
    from rl_manager.stage25_inference import Stage25InferenceAdapter
    from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config

    model_config = tiny_stage25_config()
    policy = Stage25InferenceAdapter(
        params=init_stage25_params(model_config, seed=7), config=model_config,
        name="stage25_learner", version="ppo-native-v1", seed=17,
        mode="stochastic")

    def rollout(workers: int) -> dict:
        config = RunnerConfig(
            stage25_enabled=True, stage25_mode="stochastic",
            manager_start_day=4, max_turns=144,
            openings=("none", "none"), low_telemetry=True,
            stage25_fixed_inference_batch_size=1)
        specs = [build_episode_spec(
            index, 100 + index, "candidate_vs_frozen", policy, policy)
            for index in range(2)]
        buffer = Stage25TrajectoryBuffer(64)
        ParallelSelfPlayRunner(
            config, num_workers=workers, inference_batch_wait_seconds=0.01,
            stage25_trajectory_buffer=buffer).run(specs)
        return {(row.episode_id, row.seat, row.day): (
                    row.row_id, row.classes.copy(),
                    row.component_logprobs.copy(), float(row.joint_logprob),
                    float(row.value))
                for row in buffer.rows}

    local = rollout(1)
    parallel = rollout(2)
    assert local.keys() == parallel.keys()
    assert local[(0, 0, 4)][0] == parallel[(0, 0, 4)][0]
    for key in local:
        left, right = local[key], parallel[key]
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_allclose(left[2], right[2], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(left[3:], right[3:], atol=1e-6, rtol=1e-6)


def test_stage25_runner_trajectory_closes_truncation_without_extra_plan(monkeypatch):
    from test_rl_manager_runner import _TraceBackend, _TraceExecutorFactory

    class NonzeroPolicy(_Stage25Policy):
        def infer_batch(self, **kwargs):
            output = super().infer_batch(**kwargs)
            output.component_logprobs[:, 0] = -1.0
            output.joint_logprob[:] = -1.0
            return output

    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: _TraceBackend(configuration))
    policy = NonzeroPolicy()
    trajectory = Stage25TrajectoryBuffer(2)
    runner = __import__("rl_manager.runner", fromlist=["SelfPlayRunner"]).SelfPlayRunner(
        RunnerConfig(stage25_enabled=True, max_turns=98,
                     openings=("none", "none")),
        executor_factory=_TraceExecutorFactory(),
        stage25_trajectory_buffer=trajectory)
    spec = build_episode_spec(0, 17, "e_vs_e", policy, policy)
    result = runner.run([spec])[0]
    assert result.terminated is False
    assert len(trajectory) == 2
    arrays = trajectory.finalize()
    assert arrays["truncated"].tolist() == [1, 1]
    assert arrays["bootstrap_patched"].tolist() == [1, 1]
    assert arrays["bootstrap_value"].tolist() == [3.25, 3.25]
    assert len(policy.row_ids) == 1


def test_stage25_runner_closes_true_terminal_once(monkeypatch):
    from test_rl_manager_runner import _TraceBackend, _TraceExecutorFactory

    class NonzeroPolicy(_Stage25Policy):
        def infer_batch(self, **kwargs):
            output = super().infer_batch(**kwargs)
            output.component_logprobs[:, 0] = -1.0
            output.joint_logprob[:] = -1.0
            return output

    class TerminalBackend(_TraceBackend):
        @property
        def statuses(self):
            return ["DONE", "DONE"] if self._step >= 97 else [
                "ACTIVE", "ACTIVE"]

    monkeypatch.setattr(
        "rl_manager.runner.make_backend",
        lambda name, configuration: TerminalBackend(configuration))
    trajectory = Stage25TrajectoryBuffer(2)
    runner = __import__("rl_manager.runner", fromlist=["SelfPlayRunner"]).SelfPlayRunner(
        RunnerConfig(stage25_enabled=True, max_turns=120,
                     openings=("none", "none")),
        executor_factory=_TraceExecutorFactory(),
        stage25_trajectory_buffer=trajectory)
    policy = NonzeroPolicy()
    spec = build_episode_spec(0, 18, "e_vs_e", policy, policy)
    result = runner.run([spec])[0]
    assert result.terminated is True
    arrays = trajectory.finalize()
    assert arrays["terminated"].tolist() == [1, 1]
    assert arrays["truncated"].tolist() == [0, 0]
    assert arrays["bootstrap_patched"].tolist() == [0, 0]
    assert arrays["reward"].tolist() == [0.0, 0.0]
    assert len(policy.row_ids) == 1
