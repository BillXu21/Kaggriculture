"""Focused coverage for the Stage 2.5 PPO executor selector."""

from __future__ import annotations

import json

from executor_v0.agent import AgentConfig
from executor_v0.plan import DailyPlan

from rl_manager import stage25_checkpoint
from rl_manager import stage25_ppo_cli as cli
from rl_manager.parallel import _factory_wire
from rl_manager.parallel_worker import _factory_from_wire
from rl_manager.runner import _executor_factory_provenance
from rl_manager.stage25_provider import Stage25PlanProvider

from test_executor_v0_agent import make_obs
from test_stage25_provider import HOLD, _obs


def test_parser_defaults_to_strip_executor() -> None:
    args = cli._parser().parse_args(["--scratch", "--output-dir", "out"])

    assert args.executor == "strip"


def test_legacy_executor_uses_exact_effective_agent_config() -> None:
    factory = cli._resolve_executor_factory("legacy")

    assert factory.name == "executor_v0"
    assert factory.agent_config == AgentConfig(
        strict=True,
        optional_spare_watering=True,
        record_turn_snapshot=False,
        aggressive_sell_all=True,
    )
    assert factory.agent_config.heuristic_care is False
    assert factory.agent_config.heuristic_fertilizer is False
    assert factory.agent_config.wheat_harvest_threshold is False


def test_strip_and_legacy_executor_provenance_differ() -> None:
    strip = _executor_factory_provenance(cli._resolve_executor_factory("strip"))
    legacy = _executor_factory_provenance(cli._resolve_executor_factory("legacy"))

    assert strip != legacy
    assert strip["name"] == "stage25_strip_executor"
    assert legacy["name"] == "executor_v0"


def test_legacy_executor_provenance_distinguishes_aggressive_sell_profile() -> None:
    from rl_manager.executor_factory import make_default_executor_factory

    old = _executor_factory_provenance(
        make_default_executor_factory(AgentConfig(
            strict=True, optional_spare_watering=True,
            record_turn_snapshot=False)))
    current = _executor_factory_provenance(
        cli._resolve_executor_factory("legacy"))

    assert old != current
    assert old["effective_profile"]["agent_config"]["aggressive_sell_all"] is False
    assert current["effective_profile"]["agent_config"]["aggressive_sell_all"] is True
    json.dumps(current, allow_nan=False)


def test_resume_rejects_checkpoint_from_other_executor(monkeypatch) -> None:
    args = cli._parser().parse_args([
        "--resume", "checkpoint.npz", "--executor", "legacy",
        "--output-dir", "out",
    ])
    config = cli._config(args)
    strip_provenance = _executor_factory_provenance(
        cli._resolve_executor_factory("strip"))

    def fake_load(*checkpoint_args, **kwargs):
        del checkpoint_args
        return None, kwargs["optimizer_state_template"], None, {
            "training_contract": cli._training_contract(args),
            "executor": strip_provenance,
        }, None

    monkeypatch.setattr(stage25_checkpoint,
                        "load_stage25_ppo_checkpoint", fake_load)
    try:
        cli._new_state(args, config)
    except ValueError as exc:
        assert "executor provenance does not match" in str(exc)
    else:
        raise AssertionError("cross-executor resume was accepted")


def test_resume_rejects_checkpoint_with_old_legacy_sell_profile(monkeypatch) -> None:
    from rl_manager.executor_factory import make_default_executor_factory

    args = cli._parser().parse_args([
        "--resume", "checkpoint.npz", "--executor", "legacy",
        "--output-dir", "out",
    ])
    config = cli._config(args)
    old_provenance = _executor_factory_provenance(
        make_default_executor_factory(AgentConfig(
            strict=True, optional_spare_watering=True,
            record_turn_snapshot=False)))

    def fake_load(*checkpoint_args, **kwargs):
        del checkpoint_args
        return None, kwargs["optimizer_state_template"], None, {
            "training_contract": cli._training_contract(args),
            "executor": old_provenance,
        }, None

    monkeypatch.setattr(stage25_checkpoint,
                        "load_stage25_ppo_checkpoint", fake_load)
    try:
        cli._new_state(args, config)
    except ValueError as exc:
        assert "executor provenance does not match" in str(exc)
    else:
        raise AssertionError("old legacy sell profile was accepted")


def test_legacy_factory_survives_parallel_wire_reconstruction() -> None:
    factory = cli._resolve_executor_factory("legacy")
    wire = _factory_wire(factory)
    rebuilt = _factory_from_wire(wire)

    assert wire[0] == "executor_v0@config"
    assert rebuilt.name == factory.name
    assert rebuilt.version == factory.version
    assert rebuilt.agent_config == factory.agent_config
    assert _executor_factory_provenance(rebuilt) == \
        _executor_factory_provenance(factory)
    assert rebuilt.create(
        backend_name="fast", seat=0, configuration={},
        provider=Stage25PlanProvider(7, 0, 3),
    ).config == factory.agent_config


def test_stage25_provider_lowers_daily_plan_for_legacy_executor() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    provider.accept_classes(_obs(day=3), HOLD)
    agent = cli._resolve_executor_factory("legacy").create(
        backend_name="fast", seat=0, configuration={}, provider=provider)

    actions = agent(make_obs(day=3, hour=0, step=72, unlocked=("NW",)))

    assert isinstance(provider.cached_plan, DailyPlan)
    assert set(actions) == {"farmer", "hands", "market"}
