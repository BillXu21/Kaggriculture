"""Issue #17 process topology and deterministic routing tests."""

from __future__ import annotations

import multiprocessing as mp
import sys
from importlib.util import find_spec

import numpy as np
import pytest

from rl_manager.parallel import ParallelSelfPlayRunner
from rl_manager.parallel import ParallelRolloutError
from rl_manager.parallel_protocol import policy_row_request_id
from rl_manager.parallel_worker import FORBIDDEN_ACCELERATOR_MODULES
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
