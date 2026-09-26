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
from rl_manager.parallel import _factory_wire
from rl_manager.parallel_worker import _factory_from_wire
from rl_manager.runner import (
    RunnerConfig, SelfPlayRunner, _executor_factory_provenance,
    build_episode_spec,
)
from rl_manager.stage25_provider import Stage25PlanProvider
from rl_manager.types import E_VS_E

from test_executor_v0_agent import make_obs
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

    assert wire[0] == "stage25_strip_executor@config:v1"
    assert rebuilt.name == factory.name
    assert rebuilt.version == factory.version
    assert rebuilt.strip_config == factory.strip_config
    assert isinstance(
        rebuilt.create(backend_name="fast", seat=0, configuration={},
                       provider=Stage25PlanProvider(8, 0, 3)),
        Stage25StripExecutorAgent)


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
def test_low_telemetry_materializes_only_at_boundaries_and_explicit_request():
    source = Stage25PlanProvider(7, 0, 3)
    source.accept_classes(_obs(day=3), HOLD)
    full = make_stage25_executor_factory()
    low = full.with_low_telemetry(True)
    agent = low.create(
        backend_name="fast", seat=0, configuration={}, provider=source)
    obs0 = make_obs(day=3, hour=0, step=72, unlocked=("NW",))
    obs1 = make_obs(day=3, hour=1, step=73, unlocked=("NW",))
    obs2 = make_obs(day=4, hour=0, step=96, unlocked=("NW",))

    with patch.object(agent.controller, "_diagnostics",
                      wraps=agent.controller._diagnostics) as diagnostics:
        agent(obs0)
        agent(obs1)
        assert diagnostics.call_count == 0
        requested = agent.diagnostics_json()
        assert diagnostics.call_count == 1
        assert requested["days"]["3"]["day"] == 3
        source.accept_classes(_obs(day=4), HOLD)
        agent(obs2)
        assert diagnostics.call_count == 2
        agent.finalize_diagnostics(obs2, 0)
        assert diagnostics.call_count == 3

    final = agent.diagnostics_json()
    assert set(final["days"]) == {"3", "4"}
    assert final["days"]["3"]["market_diagnostics"]
    assert final["days"]["4"]["routes_finalized"] is not None


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
    assert low.final_banks == [0.0, 0.0]
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
