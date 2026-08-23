"""Focused offline tests for the opening_book evaluation CLI (issue #4, stage 3).

Default Python only; the official engine is faked at the single stable
``_official_runner``/``engine_make`` boundary. No real engine, no weights.
"""

from __future__ import annotations

import json

import pytest

import opening_book.eval as ob_eval
from opening_book.eval import (
    ENVELOPES,
    EXIT_ENGINE_UNAVAILABLE,
    EXIT_OK,
    EXIT_USAGE,
    adapt_one_arg,
    build_opponents,
    evaluate_handoff_envelope,
    make_checkpoint_downstream_factory,
    run_opening_game,
    run_paired_comparison,
)

HORIZON = [(d, h) for d in range(4) for h in range(24)]


# ---------------------------------------------------------------------------
# Fakes at stable boundaries
# ---------------------------------------------------------------------------

class FakeState:
    def __init__(self, status="ACTIVE", reward=100, farms_money=(3000.0, 3000.0)):
        self.status = status
        self.reward = reward
        self.observation = type("Obs", (), {})()
        self.observation.farms = [{"money": m} for m in farms_money]


class FakeEnv:
    """Drives both agents through d0h0..d4h0 with real trace hand counts."""

    def __init__(self, envelope_summary_by_seat=None):
        self.envelope = envelope_summary_by_seat or {}
        self.recorded_agents = None
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def run(self, agents):
        self.recorded_agents = list(agents)
        trace_cache = {}
        steps = []
        phases = HORIZON + [(4, 0)]
        for day, hour in phases:
            step_states = []
            for seat in (0, 1):
                obs = self._obs_for(seat, day, hour, trace_cache)
                action = agents[seat](obs, {"seed": 1})
                assert set(action) == {"farmer", "hands", "market"}
                step_states.append(FakeState())
            steps.append(step_states)
        # terminal step: DONE statuses and final banks
        final_farms = self.envelope.get("final_banks", (29.0, 3000.0))
        steps.append([FakeState(status="DONE", reward=r, farms_money=final_farms)
                      for r in self.envelope.get("rewards", (100, 50))])
        return steps

    def _obs_for(self, seat, day, hour, cache):
        if "trace" not in cache:
            from opening_book.trace import load_built_in_trace
            cache["trace"] = load_built_in_trace("standard_mixed")
        trace = cache["trace"]
        idx = day * 24 + hour
        n_hands = len(trace["turns"][idx]["action"]["hands"]) \
            if idx < 96 else 5
        farms = [{}, {}]
        for s in (0, 1):
            summary = self.envelope.get(s, {})
            farms[s] = {
                "money": summary.get("money", 30.0),
                "tiles": summary.get("tiles", []),
                "unlocked_quadrants": ["NE"],
                "hands": [None] * n_hands,
            }
        obs = {"day": day, "hour": hour, "farms": farms, "player": seat}
        shed = self.envelope.get(seat, {}).get("shed_wheat")
        if shed is not None:
            obs["private"] = {"shed": {"WHEAT": shed}}
        return obs


def standard_envelope_state():
    """FakeEnv state whose d4h0 farm matches the standard_mixed envelope."""
    tiles = [[{"kind": "PLANT", "crop": "WHEAT"}] * 7
             + [{"kind": "PLANT", "crop": "MELON"}] * 12,
             [{"animal": "COW"}] * 3 + [{"animal": "SHEEP"}] * 2]
    return {0: {"money": 29.0, "tiles": tiles, "shed_wheat": 0}, 1: {}}


@pytest.fixture()
def gated_env(monkeypatch):
    """Patch engine detection/provenance/runner; record call order."""
    order = []
    monkeypatch.setattr(ob_eval, "detect_engine",
                        lambda: {"available": True, "version": "1.32.7",
                                 "reason": ""})
    monkeypatch.setattr(ob_eval, "_official_runner",
                        lambda: order.append("runner")
                        or (lambda env_id, configuration=None: FakeEnv(
                            standard_envelope_state())))
    monkeypatch.setattr(ob_eval, "verify_official_provenance",
                        lambda: order.append("provenance")
                        or {"installed_version": "1.32.7"})
    return order


# ---------------------------------------------------------------------------
# Envelope evaluation
# ---------------------------------------------------------------------------

def test_envelope_standard_pass_and_fail():
    good = {"money": 29.0,
            "crops": {"WHEAT": 7, "MELON": 12},
            "animals": {"COW": 3, "SHEEP": 2},
            "land_count": 1, "shed_wheat": 0}
    result = evaluate_handoff_envelope("standard_mixed", good)
    assert result["ok"] is True and result["failed_reasons"] == []

    bad = dict(good)
    bad["crops"] = {"WHEAT": 6, "MELON": 12}
    result = evaluate_handoff_envelope("standard_mixed", bad)
    assert result["ok"] is False
    assert any("WHEAT" in r for r in result["failed_reasons"])

    bad = dict(good)
    bad["money"] = 5000.0
    result = evaluate_handoff_envelope("standard_mixed", bad)
    assert result["ok"] is False
    assert any("money" in c["check"] for c in result["checks"]
               if not c["ok"])


def test_envelope_pasture_pass_and_missing_fields():
    good = {"money": 213.0,
            "crops": {"WHEAT": 6, "MELON": 4, "STRAWBERRY": 3},
            "animals": {"COW": 1, "SHEEP": 4},
            "land_count": 1, "shed_wheat": 8}
    assert evaluate_handoff_envelope("pasture_heavy", good)["ok"] is True

    empty = {"money": None, "crops": {}, "animals": {},
             "land_count": None, "shed_wheat": None}
    result = evaluate_handoff_envelope("pasture_heavy", empty)
    assert result["ok"] is False
    assert len(result["failed_reasons"]) >= 3  # crops/animals/land


def test_envelopes_carry_source_references():
    assert ENVELOPES["standard_mixed"]["source_reference"]["episode"] == 95515912
    assert ENVELOPES["pasture_heavy"]["source_reference"]["episode"] == 95055022
    for envelope in ENVELOPES.values():
        assert envelope["money_justification"]
        assert envelope["shed_justification"]


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def test_adapt_one_arg_preserves_observation_identity():
    seen = []

    def agent(obs):
        seen.append(obs)
        return {"farmer": ["PASS"], "hands": [], "market": []}

    obs = {"day": 9}
    adapt_one_arg(agent)(obs, {"config": 1})
    assert seen == [obs]


def test_mirror_opponent_is_independent_wrapper_instance():
    opponents, mirror = build_opponents("mirror", "standard_mixed", 0)
    assert mirror is not None and mirror.seat == 1
    assert mirror is not build_opponents("mirror", "standard_mixed", 0)[1]
    # independent instance has its own diagnostics object
    diag = mirror.diagnostics_json()
    assert diag["seat"] == 1 and diag["opening"] == "standard_mixed"
    opponents2, mirror_pass = build_opponents("pass", "standard_mixed", 1)
    assert mirror_pass is None
    assert callable(opponents2[0])


def test_checkpoint_factory_builds_fresh_instances(tmp_path):
    calls = []

    class FakeModule:
        @staticmethod
        def make_agent(*args, **kwargs):
            calls.append(kwargs)
            return object()

    import sys
    saved = sys.modules.get("executor_v0.agent")
    sys.modules["executor_v0.agent"] = FakeModule
    try:
        factory = make_checkpoint_downstream_factory("best.pt", "cpu", 1)
        a, b = factory(), factory()
        assert a is not b  # fresh per game
        assert calls[0]["checkpoint"] == "best.pt"
        assert calls[0]["seat"] == 1 and calls[0]["device"] == "cpu"
        assert len(calls) == 2
    finally:
        if saved is None:
            sys.modules.pop("executor_v0.agent", None)
        else:
            sys.modules["executor_v0.agent"] = saved


# ---------------------------------------------------------------------------
# Game runner against the fake engine boundary
# ---------------------------------------------------------------------------

def test_run_opening_game_full_acceptance_record():
    env = FakeEnv(standard_envelope_state())

    def fake_make(env_id, configuration=None):
        assert env_id == "kaggriculture"
        assert configuration == {"seed": 1146601720}
        return env

    record = run_opening_game(fake_make, opening="standard_mixed", seat=0,
                              seed=1146601720, opponent_kind="pass",
                              downstream_factory=lambda: ob_eval.pass_action)
    assert record["passed"] is True, record["failure_reasons"]
    assert record["opening_diagnostics"]["turns_replayed"] == 96
    assert record["envelope"]["ok"] is True
    assert record["source_provenance"]["episode"] == 95515912
    assert record["final_rewards"] == [100, 50]
    assert record["final_banks"] == [29.0, 3000.0]
    assert record["status_anomalies"] == []
    # tested wrapper sat in seat 0; PASS responder served seat 1
    assert env.recorded_agents[0].__closure__ is not None


def test_run_opening_game_propagates_status_anomaly_failure():
    class AnomalyEnv(FakeEnv):
        def run(self, agents):
            steps = super().run(agents)
            steps[3][0].status = "INVALID"
            return steps

    record = run_opening_game(lambda env_id, configuration=None: AnomalyEnv(
        standard_envelope_state()), opening="standard_mixed", seat=0,
        seed=1, opponent_kind="pass",
        downstream_factory=lambda: ob_eval.pass_action)
    assert record["passed"] is False
    assert any(a["status"] == "INVALID" for a in record["status_anomalies"])
    assert record["failure_reasons"]


def test_run_opening_game_validates_mirror_diagnostics():
    record = run_opening_game(
        lambda env_id, configuration=None: FakeEnv({
            0: standard_envelope_state()[0], 1: standard_envelope_state()[0]}),
        opening="standard_mixed", seat=0, seed=2, opponent_kind="mirror",
        downstream_factory=lambda: ob_eval.pass_action)
    assert record["passed"] is True, record["failure_reasons"]
    mirror_diag = record["mirror_diagnostics"]
    assert mirror_diag["turns_replayed"] == 96
    assert mirror_diag["seat"] == 1
    assert mirror_diag["divergence"]["occurred"] is False


def test_paired_comparison_runs_both_arms_with_same_settings():
    envs = []

    def fake_make(env_id, configuration=None):
        env = FakeEnv(standard_envelope_state())
        envs.append((env_id, configuration))
        return env

    result = run_paired_comparison(
        fake_make, opening="standard_mixed", seat=1, seed=42,
        opponent_kind="pass",
        downstream_factory=lambda: ob_eval.pass_action)
    assert result["mode"] == "paired"
    assert result["baseline"]["mode"] == "paired_baseline_bc_only"
    assert result["opener"]["mode"] == "paired_opener_to_bc"
    assert result["opener"]["opening_diagnostics"]["turns_replayed"] == 96
    # identical paired settings across arms
    assert envs[0][1]["seed"] == envs[1][1]["seed"] == 42
    assert result["seed"] == 42 and result["seat"] == 1


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        ob_eval.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--opening" in out and "--seeds" in out and "--mode" in out


def test_cli_usage_errors():
    assert ob_eval.main([]) == EXIT_USAGE  # no seeds
    assert ob_eval.main(["--seed", "1", "--downstream", "checkpoint"]) \
        == EXIT_USAGE  # missing --checkpoint
    assert ob_eval.main(["--seed", "1", "--downstream", "checkpoint",
                         "--checkpoint", "missing.pt"]) == EXIT_USAGE
    assert ob_eval.main(["--seed", "1", "--mode", "paired"]) == EXIT_USAGE
    assert ob_eval.main(["--seed", "1", "--mode", "paired",
                         "--downstream", "checkpoint",
                         "--checkpoint", "x.pt", "--opponent", "mirror"]) \
        == EXIT_USAGE


def test_cli_engine_unavailable_exit(monkeypatch, capsys):
    monkeypatch.setattr(ob_eval, "detect_engine",
                        lambda: {"available": False, "version": None,
                                 "reason": "not installed"})
    assert ob_eval.main(["--seed", "1"]) == EXIT_ENGINE_UNAVAILABLE
    err = capsys.readouterr().err
    assert "engine unavailable" in err


def test_cli_provenance_gate_runs_before_runner(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(ob_eval, "detect_engine",
                        lambda: {"available": True, "version": "1.32.7",
                                 "reason": ""})
    monkeypatch.setattr(ob_eval, "verify_official_provenance",
                        lambda: order.append("provenance")
                        or {"installed_version": "1.32.7"})
    monkeypatch.setattr(ob_eval, "_official_runner",
                        lambda: order.append("runner")
                        or (lambda env_id, configuration=None: FakeEnv(
                            standard_envelope_state())))
    out = tmp_path / "r.jsonl"
    code = ob_eval.main(["--seed", "1146601720", "--out", str(out)])
    assert code == EXIT_OK
    assert order == ["provenance", "runner"]

    # provenance mismatch short-circuits before the runner with exit 3
    from oracle.provenance import ProvenanceError
    order.clear()

    def boom():
        order.append("provenance")
        raise ProvenanceError("bad wheel")

    monkeypatch.setattr(ob_eval, "verify_official_provenance", boom)
    code = ob_eval.main(["--seed", "1", "--out", str(tmp_path / "r2.jsonl")])
    assert code == EXIT_ENGINE_UNAVAILABLE
    assert order == ["provenance"]


def test_cli_writes_jsonl_records_with_metadata(gated_env, tmp_path):
    out = tmp_path / "records.jsonl"
    argv = ["--opening", "standard_mixed", "--seat", "0",
            "--seeds", "1146601720", "1979016230",
            "--opponent", "pass", "--out", str(out)]
    assert ob_eval.main(argv) == EXIT_OK
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [r["seed"] for r in records] == [1146601720, 1979016230]
    for record in records:
        assert record["passed"] is True
        assert record["provenance"]["installed_version"] == "1.32.7"
        assert record["engine_version"] == "1.32.7"
        assert record["command"]["argv"] == argv
        assert record["source_provenance"]["episode"] == 95515912
        json.dumps(record)  # fully serializable
