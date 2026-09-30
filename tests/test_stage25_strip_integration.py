"""Regression coverage for the native Stage 2.5 strip-executor seam."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

from executor_v0.strip_executor import StripExecutorController
from rl_manager.executor_factory import (
    Stage25StripExecutorAgent,
    make_stage25_executor_factory,
)
from rl_manager.parallel import ParallelSelfPlayRunner, _factory_wire
from rl_manager.parallel_worker import _factory_from_wire
from rl_manager.runner import (
    RunnerConfig, SelfPlayRunner, _executor_factory_provenance,
    build_episode_spec,
)
from rl_manager.stage25_provider import Stage25PlanProvider
from rl_manager.types import E_VS_E

from test_executor_v0_agent import make_obs, recording_provider, simple_plan
from test_stage25_packet5a_parallel import _Stage25Policy
from test_stage25_provider import HOLD, _obs


def test_stage25_factory_constructs_strip_controller_and_truthful_profile():
    factory = make_stage25_executor_factory()
    agent = factory.create(
        backend_name="fast", seat=1, configuration={},
        provider=Stage25PlanProvider(7, 1, 3),
    )

    assert isinstance(agent, Stage25StripExecutorAgent)
    assert isinstance(agent.controller, StripExecutorController)
    assert agent.config.acting_seat == 1
    assert factory.name == "stage25_strip_executor"
    assert factory.version.startswith("strip_executor_v1@")
    profile = _executor_factory_provenance(factory)
    assert profile["effective_profile"]["controller"].endswith(
        "StripExecutorController")
    assert profile["effective_profile"]["aggressive_sell_all"] is True
    assert "strategic_protection" not in profile["effective_profile"]
    json.dumps(profile, allow_nan=False)


def test_strip_adapter_accepts_once_and_reuses_plan_for_primitive_turns():
    source = Stage25PlanProvider(7, 0, 3)
    source.accept_classes(_obs(day=3), HOLD)

    calls = []

    class CountingProvider:
        def daily_plan(self, obs, seat):
            calls.append((int(obs["day"]), seat))
            return source.daily_plan(obs, seat)

        def diagnostics_json(self):
            return source.diagnostics_json()

    provider = CountingProvider()
    agent = make_stage25_executor_factory().create(
        backend_name="fast", seat=0, configuration={}, provider=provider)

    first = agent(make_obs(day=3, hour=0, step=72, unlocked=("NW",)))
    second = agent(make_obs(day=3, hour=1, step=73, unlocked=("NW",)))

    assert set(first) == {"farmer", "hands", "market"}
    assert set(second) == {"farmer", "hands", "market"}
    assert calls == [(3, 0)]
    assert agent.diagnostics_json()["provider_diagnostics"][
        "decision_identity"].endswith("day=3")


def test_stage25_factory_wire_reconstructs_strip_factory():
    factory = make_stage25_executor_factory()
    wire = _factory_wire(factory)
    rebuilt = _factory_from_wire(wire)

    assert wire[0] == "stage25_strip_executor@config:v2"
    assert wire[2] is False
    assert rebuilt.name == factory.name
    assert rebuilt.version == factory.version
    assert rebuilt.strip_config == factory.strip_config
    assert isinstance(
        rebuilt.create(backend_name="fast", seat=0, configuration={},
                       provider=Stage25PlanProvider(8, 0, 3)),
        Stage25StripExecutorAgent)


def test_low_telemetry_preserves_actions_and_marks_reduced_diagnostics(
        monkeypatch):
    plan = simple_plan()
    obs = make_obs(day=3, hour=0, step=72, unlocked=("NW",))
    full = make_stage25_executor_factory().create(
        backend_name="fixture", seat=0, configuration={},
        provider=recording_provider(plan),
    )
    low_factory = make_stage25_executor_factory(low_telemetry=True)
    low = low_factory.create(
        backend_name="fixture", seat=0, configuration={},
        provider=recording_provider(plan),
    )

    full_action = full(obs)
    def unexpected_diagnostics(*_args, **_kwargs):
        raise AssertionError("low telemetry must not serialize controller diagnostics")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            low.controller, "_diagnostics", unexpected_diagnostics)
        patcher.setattr(
            "rl_manager.executor_factory.copy.deepcopy",
            unexpected_diagnostics,
        )
        assert low(obs) == full_action
    full_diagnostics = full.diagnostics_json()
    low_diagnostics = low.diagnostics_json()
    assert "telemetry_mode" not in full_diagnostics
    assert full_diagnostics["days"]["3"]
    assert low_diagnostics["telemetry_mode"] == "reduced"
    assert low_diagnostics["diagnostics_reduced"] is True
    assert low_diagnostics["days"] == {}
    assert low.controller.diagnostics == {
        "schema_version": 1,
        "telemetry_mode": "reduced",
        "diagnostics_reduced": True,
    }
    json.dumps(low_diagnostics, allow_nan=False)
    assert low_factory.version == make_stage25_executor_factory().version
    assert low_factory.effective_profile == make_stage25_executor_factory().effective_profile


def test_runner_low_telemetry_is_overridden_by_full_diagnostic_capture():
    low = SelfPlayRunner(
        RunnerConfig(stage25_enabled=True, low_telemetry=True),
        master_seed=17,
    )
    capture = SelfPlayRunner(
        RunnerConfig(
            stage25_enabled=True,
            low_telemetry=True,
            record_executor_full_diagnostics=True,
        ),
        master_seed=17,
    )
    parallel_low = ParallelSelfPlayRunner(
        RunnerConfig(stage25_enabled=True, low_telemetry=True),
        num_workers=1,
    )
    parallel_capture = ParallelSelfPlayRunner(
        RunnerConfig(
            stage25_enabled=True,
            low_telemetry=True,
            record_executor_full_diagnostics=True,
        ),
        num_workers=1,
    )

    assert low.executor_factory.low_telemetry is True
    assert capture.executor_factory.low_telemetry is False
    assert parallel_low.executor_factory.low_telemetry is True
    assert parallel_capture.executor_factory.low_telemetry is False
    low_wire = _factory_wire(
        parallel_low.executor_factory,
        low_telemetry=parallel_low.config.low_telemetry,
    )
    full_wire = _factory_wire(
        parallel_capture.executor_factory,
        low_telemetry=(
            parallel_capture.config.low_telemetry
            and not parallel_capture.config.record_executor_full_diagnostics
        ),
    )
    assert _factory_from_wire(low_wire).low_telemetry is True
    assert _factory_from_wire(full_wire).low_telemetry is False
    assert low.executor_factory.effective_profile == (
        capture.executor_factory.effective_profile)


def test_strip_factory_wire_reconstruction_stays_accelerator_free():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import rl_manager.stage25_ppo_cli
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.parallel import _factory_wire
from rl_manager.parallel_worker import _factory_from_wire
factory = _factory_from_wire(_factory_wire(make_stage25_executor_factory()))
factory.create(backend_name='fast', seat=0, configuration={}, provider=object())
assert not any(name == 'jax' or name.startswith('jax.') or
               name == 'libtpu' or name.startswith('libtpu.') or
               name == 'optax' or name.startswith('optax.')
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_strip_aggressive_sale_profile_includes_all_sellable_products():
    assert make_stage25_executor_factory().strip_config.aggressive_sell_all
    # The product allow-list is owned by strip_market; this test exercises the
    # configured controller path without duplicating its implementation rules.
    from executor_v0.strip_market import _AGGRESSIVE_SELL_PRODUCTS

    assert set(_AGGRESSIVE_SELL_PRODUCTS) == {
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
        "WOOL", "FERTILIZER",
    }
def test_low_telemetry_never_materializes_per_turn_diagnostics():
    """Reduced telemetry must not build a per-day diagnostics tree at all.

    Under the frozen executor contract ``low_telemetry`` is strictly cheaper
    than full telemetry: the controller returns a small reduced payload and the
    adapter retains no per-day snapshots. ``finalize_diagnostics`` stays a safe
    no-op in that mode, and full telemetry keeps the per-turn snapshot tree.
    """
    source = Stage25PlanProvider(7, 0, 3)
    source.accept_classes(_obs(day=3), HOLD)
    low = make_stage25_executor_factory().with_low_telemetry(True)
    agent = low.create(
        backend_name="fast", seat=0, configuration={}, provider=source)
    obs0 = make_obs(day=3, hour=0, step=72, unlocked=("NW",))
    obs1 = make_obs(day=3, hour=1, step=73, unlocked=("NW",))
    obs2 = make_obs(day=4, hour=0, step=96, unlocked=("NW",))

    with patch.object(agent.controller, "_diagnostics",
                      wraps=agent.controller._diagnostics) as diagnostics:
        agent(obs0)
        agent(obs1)
        source.accept_classes(_obs(day=4), HOLD)
        agent(obs2)
        assert diagnostics.call_count == 0
        requested = agent.diagnostics_json()
        assert diagnostics.call_count == 0
        assert requested["days"] == {}
        assert requested["telemetry_mode"] == "reduced"
        agent.finalize_diagnostics(obs2, 0)
        assert diagnostics.call_count == 0
        assert agent.diagnostics_json()["days"] == {}


def test_full_telemetry_keeps_per_turn_diagnostic_semantics():
    source = Stage25PlanProvider(8, 0, 3)
    source.accept_classes(_obs(day=3), HOLD)
    agent = make_stage25_executor_factory().create(
        backend_name="fast", seat=0, configuration={}, provider=source)
    with patch.object(agent.controller, "_diagnostics",
                      wraps=agent.controller._diagnostics) as diagnostics:
        agent(make_obs(day=3, hour=0, step=72, unlocked=("NW",)))
        agent(make_obs(day=3, hour=1, step=73, unlocked=("NW",)))
    assert diagnostics.call_count == 2
    assert agent.diagnostics_json()["days"]["3"]["day"] == 3


def _run_stage25_mode(*, seed: int, low_telemetry: bool,
                      max_turns: int = 720):
    policy = _Stage25Policy()
    config = RunnerConfig(
        backend_name="fast",
        backend_configuration={"seed": 0, "numThreads": 1},
        opening="standard_mixed",
        max_turns=max_turns,
        low_telemetry=low_telemetry,
        stage25_enabled=True,
        record_rollout=True,
        record_executor_full_diagnostics=True,
    )
    runner = SelfPlayRunner(
        config, executor_factory=make_stage25_executor_factory(),
        master_seed=25)
    spec = build_episode_spec(0, seed, E_VS_E, policy, policy)
    return runner.run([spec])[0]


def test_stage25_low_telemetry_preserves_full_game_actions_and_metadata():
    full = _run_stage25_mode(seed=144368101, low_telemetry=False)
    low = _run_stage25_mode(seed=144368101, low_telemetry=True)
    assert low.trace_digest == full.trace_digest
    assert low.final_banks == full.final_banks
    assert low.statuses == full.statuses
    assert low.terminated == full.terminated
    assert low.terminated is True
    # The full game must actually run its economics on both seats. This is not
    # a pinned bank value: the absolute number depends on the executor build,
    # while the invariant under test is that reduced telemetry leaves the
    # executed economy identical to full telemetry (asserted above).
    assert len(low.final_banks) == 2
    assert all(bank > 0.0 for bank in low.final_banks), low.final_banks
    assert low.final_banks[0] == low.final_banks[1]
    assert low.executor_full_diagnostics == full.executor_full_diagnostics
    assert low.executor_diagnostics == full.executor_diagnostics
    assert low.opening_diagnostics == full.opening_diagnostics


def test_stage25_low_telemetry_preserves_truncation_and_opening_handoff():
    full = _run_stage25_mode(seed=144368102, low_telemetry=False,
                             max_turns=98)
    low = _run_stage25_mode(seed=144368102, low_telemetry=True,
                            max_turns=98)
    assert low.trace_digest == full.trace_digest
    assert low.final_banks == full.final_banks
    assert low.statuses == full.statuses
    assert low.terminated is False
    assert low.opening_diagnostics == full.opening_diagnostics
    assert low.executor_full_diagnostics == full.executor_full_diagnostics
