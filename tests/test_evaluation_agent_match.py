"""Focused Issue #35 controller and subprocess contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from evaluation.agent_match import normalize_action, run_match, run_panel
from evaluation.cli import build_parser
from evaluation.external import ExternalControllerFactory, bundle_digest
from evaluation.internal import (
    InternalControllerFactory,
    PassControllerFactory,
    load_internal_factory,
    make_agent_config,
)


FIXTURE = Path(__file__).parent / "fixtures" / "issue35_external_pass.py"


def test_pass_panel_is_deterministic_and_seat_swapped():
    first = run_panel(
        PassControllerFactory("A"), PassControllerFactory("B"), seeds=[7], max_turns=719
    )
    second = run_panel(
        PassControllerFactory("A"), PassControllerFactory("B"), seeds=[7], max_turns=719
    )

    def deterministic_fields(item):
        value = item.to_json_dict()
        value.pop("runtime_seconds")
        value["timing_seconds"].pop("controllers")
        value["timing_seconds"].pop("environment")
        return value

    assert [deterministic_fields(item) for item in first] == [
        deterministic_fields(item) for item in second
    ]
    assert [item.orientation for item in first] == [
        "candidate_vs_frozen",
        "frozen_vs_candidate",
    ]
    assert all(item.statuses == ["DONE", "DONE"] for item in first)


def test_normalize_action_is_shape_conservative():
    action = normalize_action(
        {
            "farmer": ("PASS",),
            "hands": [("PASS",)],
            "market": [],
        }
    )
    assert action == {
        "farmer": ["PASS"],
        "hands": [["PASS"]],
        "market": [],
    }


def test_external_child_returns_legal_action_without_parent_import():
    module_name = "kaggriculture_external_agent"
    sys.modules.pop(module_name, None)
    factory = ExternalControllerFactory(
        FIXTURE, display_name="fixture", timeout_seconds=5
    )
    controller = factory.create(seat=0, configuration={"marker": "seat0"})
    try:
        action = controller.act(
            {
                "player": 0,
                "farms": [{"hands": []}, {"hands": []}],
                "private": {"shed": {"WHEAT": 1}},
            }
        )
    finally:
        controller.close()
    assert action == {"farmer": ["PASS"], "hands": [], "market": []}
    assert module_name not in sys.modules


def test_external_one_argument_callable_gets_structified_observation():
    factory = ExternalControllerFactory(
        FIXTURE.with_name("issue35_external_one_arg.py"), timeout_seconds=5
    )
    controller = factory.create(seat=1, configuration={})
    try:
        action = controller.act(
            {
                "player": 1,
                "farms": [{"hands": []}, {"hands": []}],
            }
        )
    finally:
        controller.close()
    assert action["farmer"] == ["PASS"]


def test_each_controller_receives_only_its_own_private_view():
    seen = []

    class Controller:
        observation_mode = "raw"

        def __init__(self, expected):
            self.expected = expected

        def act(self, observation):
            seen.append(observation)
            assert observation["private"]["shed"]["WHEAT"] == self.expected
            assert "opponent_private" not in observation
            return {"farmer": ["PASS"], "hands": [], "market": []}

        def close(self):
            pass

    class Factory:
        def __init__(self, expected):
            self.expected = expected
            self.provenance = {"display_name": str(expected)}

        def create(self, *, seat, configuration):
            del seat, configuration
            return Controller(self.expected)

    result = run_match(Factory(0), Factory(0), seed=7, max_turns=1)
    assert result.controller_errors == []
    assert len(seen) == 2


def test_external_digest_is_stable():
    expected = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert bundle_digest(FIXTURE) == expected
    assert bundle_digest(FIXTURE) == bundle_digest(FIXTURE)


def test_external_exception_is_reported_as_game_failure(tmp_path):
    source = tmp_path / "agent.py"
    source.write_text(
        "def agent(obs):\n    raise RuntimeError('fixture failure')\n", encoding="utf-8"
    )
    result = run_match(
        PassControllerFactory(),
        ExternalControllerFactory(source, timeout_seconds=5),
        seed=7,
        max_turns=719,
    )
    assert result.controller_errors
    assert result.controller_errors[0]["type"] == "ExternalAgentError"
    assert "fixture failure" in result.controller_errors[0]["detail"]["message"]
    assert result.terminated is False


def test_agent_config_serialization_distinguishes_aggressive_seat():
    normal = make_agent_config()
    aggressive = make_agent_config(aggressive_sell_all=True)
    assert normal.aggressive_sell_all is False
    assert aggressive.aggressive_sell_all is True
    assert json.dumps(normal.__dict__, default=str)


def test_internal_factory_selects_explicit_opening_per_seat():
    class Executor:
        def __call__(self, observation):
            return {"farmer": ["PASS"], "hands": [], "market": []}

    class Factory:
        name = "test-executor"
        version = "test-v1"

        def create(self, **kwargs):
            del kwargs
            return Executor()

    factory = InternalControllerFactory(
        policy=object(),
        executor_config=make_agent_config(),
        opening_names=("fourth_quadrant_s0", "fourth_quadrant_s1"),
        executor_factory=Factory(),
    )
    assert factory.create(seat=0, configuration={}).opening._identity \
        == "fourth_quadrant_s0"
    assert factory.create(seat=1, configuration={}).opening._identity \
        == "fourth_quadrant_s1"
    provenance = factory.provenance["opening"]
    assert [item["name"] for item in provenance["by_seat"]] == [
        "fourth_quadrant_s0", "fourth_quadrant_s1"]


def test_evaluation_cli_exposes_optional_per_seat_opening_flags():
    args = build_parser().parse_args([
        "--a-opening-s0", "fourth_quadrant_s0",
        "--a-opening-s1", "fourth_quadrant_s1",
        "--output", "panel.json",
    ])
    assert (args.a_opening_s0, args.a_opening_s1) == (
        "fourth_quadrant_s0", "fourth_quadrant_s1")


def test_internal_bc_identity_and_executor_provenance_are_stable():
    checkpoint = Path("artifacts/local/bc-v1-E/best.pt")
    if not checkpoint.is_file():
        pytest.skip("local BC-E checkpoint is not available")
    normal = load_internal_factory(
        "bc", checkpoint, executor_config=make_agent_config(), display_name="BC-E"
    )
    aggressive = load_internal_factory(
        "bc",
        checkpoint,
        executor_config=make_agent_config(aggressive_sell_all=True),
        display_name="BC-E",
    )
    first = normal.provenance
    second = normal.provenance
    assert first["policy"] == second["policy"]
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert first["executor_factory"]["config"]["aggressive_sell_all"] is False
    assert (
        aggressive.provenance["executor_factory"]["config"]["aggressive_sell_all"]
        is True
    )
    assert first["identity"] != aggressive.provenance["identity"]
