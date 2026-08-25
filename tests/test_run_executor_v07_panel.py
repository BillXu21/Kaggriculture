"""Focused tests for the local full-game evaluator plumbing."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from opening_book.trace import load_built_in_trace
from tools.run_executor_v07_panel import (
    EvaluatorError,
    main,
    run_panel,
)


def _empty_plan():
    from executor_v0.plan import DailyPlan, SELL_BIN_ANCHORS
    from replay_daily.constants import PRODUCTS

    return DailyPlan.create(
        crop_targets={name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        animal_targets={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        land_count=1,
        fertilizer_by_crop={name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
        care_by_animal={name: 0 for name in ("GOOSE", "COW", "SHEEP")},
        sell_quantities={product: {anchor: 0 for anchor in SELL_BIN_ANCHORS} for product in PRODUCTS},
    )


class FakeProvider:
    model_variant = "E"

    def __init__(self, records):
        self.records = records

    def daily_plan(self, obs, seat, previous_execution=None):
        self.records.append((int(obs["day"]), seat, copy.deepcopy(previous_execution)))
        return _empty_plan()

    def diagnostics_json(self):
        return {"model_variant": "E", "records": [], "aggregate": {}}


class FakeBackend:
    name = "fast"

    def __init__(self, configuration):
        self.seed = configuration["seed"]
        self.step_number = 0
        self.terminal_step = 2
        self._statuses = ["ACTIVE", "ACTIVE"]
        self._rewards = [0.0, 0.0]
        self.trace = load_built_in_trace("standard_mixed")

    @property
    def statuses(self):
        return list(self._statuses)

    @property
    def rewards(self):
        return list(self._rewards)

    def _observation(self, seat):
        day, hour = divmod(self.step_number, 24)
        expected = self.trace["turns"][self.step_number]["action"] if self.step_number < 96 else {
            "hands": [],
        }
        hands = [list(position) for position in ([0, 0],) * len(expected["hands"])]
        animal = (
            {"kind": "PASTURE", "animal": "GOOSE", "placed_day": 0,
             "yield_units": 0, "consecutive_unfed": 1, "fed_today": False,
             "cared_today": False, "fertilizer_available": False,
             "pending_care_bonus": 0}
            if self.step_number == 0 else None
        )
        tiles = [[None] * 10 for _ in range(10)]
        tiles[4][4] = animal
        farm = {
            "money": 3000.0,
            "tiles": tiles,
            "farmer": [0, 0],
            "hands": hands,
            "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
            "hires_today": 0,
        }
        private = {
            "shed": {"WHEAT": 2}, "seeds": {},
            "inventories": [{} for _ in range(1 + len(hands))],
        }
        return {
            "day": day, "hour": hour, "step": self.step_number,
            "player": seat, "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
            "private": private,
            "market": {"inventory": {"WHEAT": 100}, "prices": {"WHEAT": 10}},
            "town": {"unlocked_shops": []},
        }

    def reset(self):
        self.step_number = 0
        self._statuses = ["ACTIVE", "ACTIVE"]
        return [self._observation(0), self._observation(1)]

    def step(self, actions):
        del actions
        self.step_number += 1
        if self.step_number >= self.terminal_step:
            self._statuses = ["DONE", "DONE"]
        return [self._observation(0), self._observation(1)], self.rewards, self.statuses

    def canonical_state(self):
        return {
            "step": self.step_number,
            "day": self.step_number // 24,
            "hour": self.step_number % 24,
            "farms": [self._observation(0)["farms"][0], self._observation(1)["farms"][1]],
            "privates": [self._observation(0)["private"], self._observation(1)["private"]],
            "market": self._observation(0)["market"],
            "town": {"unlocked_shops": []},
            "rewards": self.rewards,
            "statuses": self.statuses,
        }


def _run(tmp_path: Path, *, trace=False, seeds=(17,), seats=(0,), max_transitions=2,
         terminal_step=2, provider_records=None, backend_records=None, **kwargs):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    provider_records = provider_records if provider_records is not None else []
    backend_records = backend_records if backend_records is not None else []

    def provider_factory(path):
        del path
        provider_records.append(object())
        return FakeProvider([])

    def backend_factory(name, config):
        assert name == "fast"
        backend = FakeBackend(config)
        backend.terminal_step = terminal_step
        backend_records.append(backend)
        return backend

    return run_panel(
        checkpoint, list(seeds), list(seats),
        turn_trace=trace, max_transitions=max_transitions,
        provider_factory=provider_factory, backend_factory=backend_factory,
        **kwargs,
    )


def test_deterministic_bytes_and_animal_drop_event(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = _run(tmp_path, trace=True, output_path=first_path)
    second = _run(tmp_path, trace=True, output_path=second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    game = first["games"][0]
    assert game["animal_total_decrease_events"][0]["before"] == 1
    assert game["animal_total_decrease_events"][0]["after"] == 0
    assert len(game["tested_action_trace_sha256"]) == 64


def test_fresh_backend_and_provider_per_game(tmp_path):
    providers = []
    backends = []
    artifact = _run(tmp_path, seeds=(1, 2), provider_records=providers, backend_records=backends)
    assert len(artifact["games"]) == 2
    assert len(providers) == len(backends) == 2
    assert backends[0] is not backends[1]


def test_trace_toggle_preserves_actions_and_final_result(tmp_path):
    traced = _run(tmp_path, trace=True, max_transitions=98, terminal_step=98)
    plain = _run(tmp_path, trace=False, max_transitions=98, terminal_step=98)
    for key in ("tested_action_trace_sha256", "opponent_action_trace_sha256", "final", "status"):
        assert traced["games"][0][key] == plain["games"][0][key]


def test_prior_debt_toggle_propagates_to_executor_config(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")

    def provider_factory(path):
        del path
        return FakeProvider([])

    def backend_factory(name, config):
        return FakeBackend(config)

    # The opening owns d0-d3; d4h0 is the first executor call.
    artifact = run_panel(
        checkpoint, [17], [0], max_transitions=98,
        prior_debt_suppression=False,
        provider_factory=provider_factory,
        backend_factory=lambda name, config: _backend_to_98(backend_factory, name, config),
    )
    assert artifact["games"][0]["executor_diagnostics"]["config"][
        "suppress_expansion_from_prior_debt"
    ] is False


def _backend_to_98(factory, name, config):
    backend = factory(name, config)
    backend.terminal_step = 98
    return backend


def test_invalid_checkpoint_and_output_overwrite_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_panel(tmp_path / "missing.pt", [17], [0], max_transitions=1,
                  provider_factory=lambda path: FakeProvider([]),
                  backend_factory=lambda name, config: FakeBackend(config))
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")
    output = tmp_path / "artifact.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_panel(checkpoint, [17], [0], max_transitions=1, output_path=output,
                  provider_factory=lambda path: FakeProvider([]),
                  backend_factory=lambda name, config: FakeBackend(config))


def test_invalid_seed_and_seat_fail_loudly(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")
    kwargs = {
        "provider_factory": lambda path: FakeProvider([]),
        "backend_factory": lambda name, config: FakeBackend(config),
        "max_transitions": 1,
    }
    with pytest.raises(EvaluatorError, match="seed"):
        run_panel(checkpoint, [-1], [0], **kwargs)
    with pytest.raises(EvaluatorError, match="seat"):
        run_panel(checkpoint, [17], [2], **kwargs)


def test_cli_rejects_invalid_seed_before_running(tmp_path):
    assert main([
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--seeds=-1", "--seats=0", "--max-transitions", "1",
        "--output", str(tmp_path / "artifact.json"),
    ]) == 2
