"""Issue #17 process topology and deterministic routing tests."""

from __future__ import annotations

import multiprocessing as mp
from queue import Queue
import sys
from importlib.util import find_spec

import numpy as np
import pytest

from executor_v0.agent import AgentConfig
from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.parallel import ParallelRolloutError, _factory_wire
from rl_manager.parallel_protocol import InferenceRequest, policy_row_request_id
from rl_manager.parallel_worker import FORBIDDEN_ACCELERATOR_MODULES, \
    _factory_from_wire
from rl_manager.executor_factory import make_default_executor_factory
from rl_manager.runner import RunnerConfig, SelfPlayRunner, \
    build_episode_spec
from rl_manager.seeds import SeedStream
from rl_manager.types import E_VS_E, PolicyIdentity, PolicyOutputs
from rl_manager.decode import ACTION_TENSOR_SHAPES


def _spawn_import_probe(queue):
    loaded = sorted(
        name for name in FORBIDDEN_ACCELERATOR_MODULES
        if name in sys.modules or any(
            module_name.startswith(name + ".") for module_name in sys.modules))
    queue.put(loaded)


def _spawn_factory_config_probe(wire, queue):
    factory = _factory_from_wire(wire)
    config = factory.agent_config
    queue.put({
        "tasks_per_worker": config.tasks_per_worker,
        "hire_cost_mult": config.hire_cost_mult,
        "shed_capacity": config.shed_capacity,
        "strict": config.strict,
        "record_turn_snapshot": config.record_turn_snapshot,
        "optional_spare_watering": config.optional_spare_watering,
        "max_carried_item_types": config.foreman.max_carried_item_types,
    })


class _ConstantPlanPolicy:
    def __init__(self, name: str = "parallel-fake") -> None:
        self.identity = PolicyIdentity(name, "test-v1", "fixed-policy")
        self.calls: list[tuple[int, str]] = []

    def plan_batch(self, inputs, prng_id):
        batch = int(np.asarray(inputs["day"]).shape[0])
        self.calls.append((batch, prng_id))
        actions = {
            name: np.zeros((batch,) + shape, dtype=np.int16)
            for name, shape in ACTION_TENSOR_SHAPES.items()}
        actions["land"] = np.ones(batch, dtype=np.int16)
        zeros = np.zeros(batch, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=actions,
            logprob_groups={name: zeros.copy() for name in (
                "crop", "animal", "land", "fertilizer", "care",
                "sell_presence")},
            logprob_total=zeros.copy(), value=zeros.copy(), batch_size=batch)


class _RecordingRowAwarePolicy(_ConstantPlanPolicy):
    def __init__(self, name: str = "parallel-row-aware") -> None:
        super().__init__(name)
        self.row_ids: list[list[str]] = []
        self.physical_days: list[list[int]] = []

    def plan_batch_with_row_ids(self, inputs, row_ids, prng_id):
        del prng_id
        days = np.asarray(inputs["day"]).reshape(-1).astype(np.int16)
        self.calls.append((len(row_ids), "row-aware"))
        self.row_ids.append(list(row_ids))
        self.physical_days.append(days.tolist())
        outputs = super().plan_batch(inputs, "row-aware")
        outputs.action_tensors["crop"][:, 0] = days
        return outputs


def _dispatch_request(policy, *, episode: int, seat: int, day: int,
                      worker_id: int = 0) -> InferenceRequest:
    identity = policy.identity
    return InferenceRequest(
        request_id=policy_row_request_id(episode, seat, day, identity),
        worker_id=worker_id, episode_index=episode, seat=seat, day=day,
        policy_identity=identity, prng_id="worker-prng",
        inputs={"day": np.asarray([[day]], dtype=np.int16)},
        queued_at=0.0)


def _dispatch_direct(config, policy, requests):
    runner = ParallelSelfPlayRunner(config, num_workers=1)
    response_queues = [Queue()]
    runner._dispatch(
        policy.identity if config.inference_batch_scope == "policy" else
        (policy.identity, requests[0].day), requests, 0.0,
        {policy.identity: policy}, response_queues)
    responses = {}
    while not response_queues[0].empty():
        response = response_queues[0].get()
        responses[response.request_id] = response.outputs
    return runner, responses


class _FailingExecutorFactory:
    name = "test-failing-executor"
    version = "test-failing-v1"

    def create(self, *, backend_name, seat, configuration, provider):
        del backend_name, seat, configuration, provider
        raise RuntimeError("intentional worker crash")


def _config(**overrides):
    values = {
        "backend_name": "fast",
        "backend_configuration": {"seed": 0, "numThreads": 1},
        "max_turns": 130,
        "num_envs": 1,
    }
    values.update(overrides)
    return RunnerConfig(**values)


def test_request_identity_is_stable_and_policy_neutral():
    identity = PolicyIdentity("e", "v1", "f" * 64)
    assert policy_row_request_id(4, 1, 9, identity) == \
        "episode=4/seat=1/day=9/policy=e@v1:ffffffffffff"


def test_worker_import_boundary_has_no_accelerator_modules():
    context = mp.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_spawn_import_probe, args=(result,))
    process.start()
    assert result.get(timeout=10) == []
    process.join(timeout=10)
    assert process.exitcode == 0
    result.close()


def test_parallel_requires_positive_worker_count():
    with pytest.raises(ValueError, match="num_workers"):
        ParallelSelfPlayRunner(_config(), num_workers=0)


@pytest.mark.parametrize("config", [
    pytest.param(None, id="default"),
    pytest.param(AgentConfig(strict=True, record_turn_snapshot=False),
                 id="low-telemetry"),
    pytest.param(AgentConfig(tasks_per_worker=3, shed_capacity=17,
                             aggressive_sell_all=True), id="custom"),
])
def test_default_factory_wire_preserves_exact_agent_config(config):
    factory = make_default_executor_factory(config)
    wire = _factory_wire(factory)
    rebuilt = _factory_from_wire(wire)
    assert rebuilt.agent_config == factory.agent_config


def test_default_factory_enables_pass_only_preventive_watering():
    factory = make_default_executor_factory()
    assert factory.agent_config.optional_spare_watering is True
    assert factory.agent_config.optional_idle_cleanup is False


def test_default_factory_wire_preserves_config_in_spawned_worker():
    config = AgentConfig(
        tasks_per_worker=3, hire_cost_mult=2, shed_capacity=17,
        strict=True, record_turn_snapshot=False)
    wire = _factory_wire(make_default_executor_factory(config))
    context = mp.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_spawn_factory_config_probe,
                              args=(wire, result))
    process.start()
    assert result.get(timeout=10) == {
        "tasks_per_worker": 3,
        "hire_cost_mult": 2,
        "shed_capacity": 17,
        "strict": True,
        "record_turn_snapshot": False,
        "optional_spare_watering": False,
        "max_carried_item_types": 2,
    }
    process.join(timeout=10)
    assert process.exitcode == 0
    result.close()


def test_default_policy_day_scope_does_not_mix_days():
    policy = _RecordingRowAwarePolicy()
    requests = [_dispatch_request(policy, episode=17, seat=0, day=8),
                _dispatch_request(policy, episode=42, seat=1, day=15)]
    runner, _ = _dispatch_direct(
        _config(inference_batch_scope="policy_day"), policy, requests[:1])
    runner._dispatch((policy.identity, requests[1].day), requests[1:], 0.0,
                     {policy.identity: policy}, [Queue()])
    assert [days for days in policy.physical_days] == [[8], [15]]


def test_policy_scope_coalesces_mixed_days_and_routes_real_rows_only():
    policy = _RecordingRowAwarePolicy()
    requests = [_dispatch_request(policy, episode=93, seat=0, day=21),
                _dispatch_request(policy, episode=17, seat=0, day=8),
                _dispatch_request(policy, episode=42, seat=1, day=15)]
    runner, responses = _dispatch_direct(
        _config(inference_batch_scope="policy"), policy, requests)
    assert policy.physical_days == [[8, 15, 21]]
    assert len(responses) == len(requests)
    assert runner.inference_metrics["real_batch_sizes"] == [3]
    assert runner.inference_metrics["physical_batch_sizes"] == [3]
    assert runner.inference_metrics["padding_rows"] == 0
    assert runner.inference_metrics["occupancy"] == 1.0
    for request in requests:
        output = responses[request.request_id]
        assert int(output.action_tensors["crop"][0, 0]) == request.day


def test_fixed_physical_batch_pads_without_routing_or_recording_padding():
    policy = _RecordingRowAwarePolicy()
    requests = [_dispatch_request(policy, episode=17, seat=0, day=8),
                _dispatch_request(policy, episode=42, seat=1, day=15)]
    runner, responses = _dispatch_direct(
        _config(inference_batch_scope="policy", fixed_inference_batch_size=4),
        policy, requests)
    assert policy.physical_days == [[8, 15, 8, 8]]
    assert len(responses) == 2
    assert all("padding/" not in request_id for request_id in responses)
    assert runner.inference_metrics["physical_batch_sizes"] == [4]
    assert runner.inference_metrics["real_batch_sizes"] == [2]
    assert runner.inference_metrics["padding_rows"] == 2
    assert runner.inference_metrics["occupancy"] == 0.5


def test_fixed_batch_chunks_after_canonical_sorting():
    policy = _RecordingRowAwarePolicy()
    requests = [_dispatch_request(policy, episode=episode, seat=0, day=4)
                for episode in (4, 1, 3, 2, 6)]
    runner, responses = _dispatch_direct(
        _config(inference_batch_scope="policy", fixed_inference_batch_size=2),
        policy, requests)
    assert policy.physical_days == [[4, 4], [4, 4], [4, 4]]
    assert [[int(row) for row in days]
            for days in policy.physical_days] == [[4, 4], [4, 4], [4, 4]]
    assert policy.row_ids[:2] == [
        [requests[1].request_id, requests[3].request_id],
        [requests[2].request_id, requests[0].request_id],
    ]
    assert policy.row_ids[2][0] == requests[4].request_id
    assert policy.row_ids[2][1].startswith("padding/")
    assert len(responses) == 5
    assert runner.inference_metrics["physical_inference_calls"] == 3
    assert runner.inference_metrics["padding_rows"] == 1


def test_different_policy_identities_never_share_dispatch():
    first = _RecordingRowAwarePolicy("first")
    second = _RecordingRowAwarePolicy("second")
    requests = [_dispatch_request(first, episode=1, seat=0, day=8),
                _dispatch_request(second, episode=2, seat=0, day=8)]
    runner = ParallelSelfPlayRunner(_config(inference_batch_scope="policy"),
                                    num_workers=1)
    queues = [Queue()]
    runner._dispatch(first.identity, [requests[0]], 0.0,
                     {first.identity: first, second.identity: second}, queues)
    runner._dispatch(second.identity, [requests[1]], 0.0,
                     {first.identity: first, second.identity: second}, queues)
    assert first.physical_days == [[8]]
    assert second.physical_days == [[8]]
    assert runner.inference_metrics["physical_inference_calls"] == 2


@pytest.mark.skipif(
    find_spec("fast_env._kaggriculture_env") is None,
    reason="native fast_env extension is unavailable")
def test_parallel_truncated_e_vs_e_routes_and_normalizes_results():
    policy = _ConstantPlanPolicy()
    seeds = SeedStream(17)
    specs = [build_episode_spec(
        index, seeds.episode_seed(index), E_VS_E, policy, policy)
        for index in range(4)]
    integrated_config = _config(
        num_envs=2, batch_backend=True, low_telemetry=True,
        read_only_agent_observations=True)
    serial = SelfPlayRunner(integrated_config, master_seed=17).run(specs)
    policy.calls.clear()
    destination = __import__("rl_manager.trajectory", fromlist=[
        "TrajectoryBuffer", "e_input_spec"])
    parallel_buffer = destination.TrajectoryBuffer(
        capacity=32, input_spec=destination.e_input_spec())
    parallel = ParallelSelfPlayRunner(
        integrated_config, num_workers=2, master_seed=17,
        trajectory_buffer=parallel_buffer,
        inference_batch_wait_seconds=1.0).run(specs)
    assert [result.episode_index for result in parallel] == list(range(4))
    assert [result.seed for result in parallel] == [result.seed for result in serial]
    assert [result.transitions for result in parallel] == [
        result.transitions for result in serial]
    assert [result.trace_digest for result in parallel] == [
        result.trace_digest for result in serial]
    assert len(parallel_buffer) == 4 * 2 * 2  # d4 and d5, both seats
    arrays = parallel_buffer.finalize()
    assert sorted((int(row), int(seat), int(day)) for row, seat, day in zip(
        arrays["episode_index"][:len(parallel_buffer)],
        arrays["seat"][:len(parallel_buffer)],
        arrays["day"][:len(parallel_buffer)])) \
        == sorted((index, seat, day) for index in range(4)
                  for seat in (0, 1) for day in (4, 5))
    assert len(policy.calls) == 2
    assert sorted(size for size, _ in policy.calls) == [8, 8]


@pytest.mark.skipif(
    find_spec("fast_env._kaggriculture_env") is None,
    reason="native fast_env extension is unavailable")
def test_parallel_worker_failure_is_immediate_and_loud():
    policy = _ConstantPlanPolicy()
    spec = build_episode_spec(0, SeedStream(17).episode_seed(0), E_VS_E,
                              policy, policy)
    with pytest.raises(ParallelRolloutError, match="intentional worker crash"):
        ParallelSelfPlayRunner(
            _config(), num_workers=2, executor_factory=_FailingExecutorFactory()
        ).run([spec])
