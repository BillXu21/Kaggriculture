"""Issue #36 Rust executor seam and first-divergence harness tests."""

from __future__ import annotations

import copy
import pickle

from executor_v0.agent import AgentConfig
from executor_v0.differential import compare_action_sequences
from executor_v0.rust_backend import (
    _native_type,
    make_rust_executor_factory,
)
from rl_manager.executor_factory import RUST_EXECUTOR_FACTORY_VERSION


def test_action_comparator_reports_first_exact_difference_with_context():
    observations = [{"step": 0}, {"step": 1}]
    divergence = compare_action_sequences(
        [{"farmer": ["PASS"]}, {"farmer": ["NORTH"]}],
        [{"farmer": ["PASS"]}, {"farmer": ["EAST"]}],
        observations=observations,
        plans=["p0", "p1"],
        histories=[[], ["PASS"]],
    )
    assert divergence is not None
    assert divergence.turn == 1
    assert divergence.python_action == {"farmer": ["NORTH"]}
    assert divergence.rust_action == {"farmer": ["EAST"]}
    assert divergence.observation_digest
    assert divergence.plan_digest
    assert divergence.history_digest


def test_action_comparator_accepts_only_equal_complete_sequences():
    actions = [(0, 0, 0, {"farmer": ["PASS"]}, {})]
    assert compare_action_sequences(actions, copy.deepcopy(actions)) is None
    assert compare_action_sequences(actions, actions + [(1,)]) is not None


def test_rust_factory_is_explicit_and_spawn_pickleable():
    config = AgentConfig(strict=True, record_turn_snapshot=False)
    factory = make_rust_executor_factory(config)
    assert factory.name == "executor_v0_rust"
    assert factory.version == RUST_EXECUTOR_FACTORY_VERSION
    assert factory.agent_config == config
    rebuilt = pickle.loads(pickle.dumps(factory))
    assert rebuilt.agent_config == config


def test_native_rust_seam_forwards_callable_when_extension_is_built():
    native_type = _native_type()
    if native_type is None:
        return

    class Echo:
        def __call__(self, observation):
            return {"step": observation["step"]}

        def diagnostics_json(self):
            return {"ok": True}

    native = native_type(Echo())
    assert native({"step": 3}) == {"step": 3}
    assert native.calls == 1
    assert native.diagnostics_json() == {"ok": True}
