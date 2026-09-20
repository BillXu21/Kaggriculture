"""Regression coverage for the native Stage 2.5 strip-executor seam."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from executor_v0.strip_executor import StripExecutorController
from rl_manager.executor_factory import (
    Stage25StripExecutorAgent,
    make_stage25_executor_factory,
)
from rl_manager.parallel import _factory_wire
from rl_manager.parallel_worker import _factory_from_wire
from rl_manager.runner import _executor_factory_provenance
from rl_manager.stage25_provider import Stage25PlanProvider

from test_executor_v0_agent import make_obs
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
