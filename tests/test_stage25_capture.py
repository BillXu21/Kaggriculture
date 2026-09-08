"""Stage 2.5 paired capture + audit validation (no trained weights needed).

- capture off/on equivalence runs a short deterministic fast-engine game
  through the REAL executor/center loop with a scripted constant-plan policy;
- episode-ID preservation is asserted against the original panel formula;
- the capture writer round-trips synthetic artifacts and keeps partials;
- the audit is validated on a tiny fixture with known movement/task
  transitions, including divergence detection and cautious language.

Real-checkpoint (P final vs BC-E, official 1.32.7, full 16-game) validation
is deferred to the Kaggle notebook and marked pending there.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from executor_v0.agent import AgentConfig, make_agent
from executor_v0.manager import FixedPlanProvider
from rl_manager.debug_trace import TraceRecorder
from rl_manager.decode import ACTION_TENSOR_SHAPES
from rl_manager.runner import (
    RunnerConfig, SelfPlayRunner, build_episode_spec,
)
from rl_manager.stage25_capture import (
    game_dir, iter_captured_games, read_json_gz, write_game_capture,
)
from rl_manager.types import E_VS_E, PolicyIdentity, PolicyOutputs
from test_executor_v0_tasks import make_obs, make_plan
from tools.audit_stage25_capture import analyze_game, load_game, run_audit
from tools.evaluate_stage25_upkeep import (
    UpkeepFactory, episode_id_for, parse_game_filter,
)


# ------------------------------------------------- deterministic scripted policy


class _ConstantPlanPolicy:
    """Zero-weight stand-in: fixed decodable plan, instrumented calls."""

    def __init__(self, name: str) -> None:
        self.identity = PolicyIdentity(
            name=name, version="fake-v1", fingerprint=f"fake-{name}")
        self.calls: list[tuple[str, int, int]] = []

    def plan_batch(self, inputs, prng_id):
        batch_size = int(np.asarray(inputs["board_kind"]).shape[0])
        day = int(np.asarray(inputs["day"]).ravel()[0])
        self.calls.append((self.identity.identity_id(), day, batch_size))
        action_tensors = {
            name: np.zeros((batch_size,) + shape, dtype=np.int16)
            for name, shape in ACTION_TENSOR_SHAPES.items()}
        action_tensors["land"] = np.ones((batch_size,), dtype=np.int16)
        zeros = np.zeros(batch_size, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=action_tensors,
            logprob_groups={group: zeros.copy() for group in (
                "crop", "animal", "land", "fertilizer", "care",
                "sell_presence")},
            logprob_total=zeros.copy(),
            value=zeros.copy(),
            batch_size=batch_size)


class _ScriptedExecutorFactory:
    """Real ExecutorAgent both seats; capture toggles snapshots only."""

    name = "stage25_scripted"
    version = "test-v1"

    def __init__(self, capture: bool) -> None:
        self.capture = capture

    def create(self, *, backend_name, seat, configuration, provider):
        del backend_name, configuration
        return make_agent(provider=provider, seat=seat, config=AgentConfig(
            strict=True, optional_spare_watering=True,
            record_turn_snapshot=self.capture))


def _run_short_game(capture: bool):
    policy = _ConstantPlanPolicy("scripted")
    config = RunnerConfig(
        backend_name="fast",
        backend_configuration={"seed": 0, "numThreads": 1},
        max_turns=5 * 24,  # d0..d4: opening plus first manager day
        record_rollout=True,
        record_debug_trace=capture,
        record_executor_full_diagnostics=capture,
        record_official_replay=capture)
    runner = SelfPlayRunner(
        config, executor_factory=_ScriptedExecutorFactory(capture),
        master_seed=25)
    spec = build_episode_spec(100, 144368101, E_VS_E, policy, policy)
    result = runner.run([spec])[0]
    return result, policy


def test_capture_off_on_identical_actions_and_outcome():
    first, policy_off = _run_short_game(capture=False)
    second, policy_on = _run_short_game(capture=True)
    assert first.statuses == second.statuses
    assert first.final_banks == second.final_banks
    assert first.trace_digest == second.trace_digest
    assert first.transitions == second.transitions
    off_actions = [(step, day, hour, a0, a1)
                   for step, day, hour, a0, a1 in first.rollout.joint_actions]
    on_actions = [(step, day, hour, a0, a1)
                  for step, day, hour, a0, a1 in second.rollout.joint_actions]
    assert off_actions == on_actions
    # No extra policy inference introduced by capture.
    assert policy_off.calls == policy_on.calls
    # Capture-only fields stay empty when capture is off...
    assert first.debug_trace is None
    assert first.executor_full_diagnostics is None
    assert first.official_replay is None
    assert first.status_history is None
    # ...and populate (gracefully) when capture is on. Fast backend has no
    # official toJSON/status seam, so replay fields stay None by design.
    assert second.debug_trace is not None
    assert second.debug_trace["turns"]
    assert second.executor_full_diagnostics is not None
    assert len(second.executor_full_diagnostics) == 2
    assert second.official_replay is None
    assert second.status_history is None


def test_executor_snapshot_flag_does_not_change_actions():
    plan = make_plan()
    obs = make_obs(day=4, hour=0, step=96)
    plain = make_agent(provider=FixedPlanProvider(plan), seat=0,
                       config=AgentConfig(strict=True,
                                          record_turn_snapshot=False))
    traced = make_agent(provider=FixedPlanProvider(plan), seat=0,
                        config=AgentConfig(strict=True,
                                           record_turn_snapshot=True))
    assert plain(copy.deepcopy(obs)) == traced(copy.deepcopy(obs))
    assert traced.debug_trace_turn is not None
    assert plain.debug_trace_turn is None


# ------------------------------------------------------- episode identity


def test_episode_ids_match_original_panel_and_survive_filtering():
    seeds = [144368101, 2112243121]
    assert episode_id_for(25, len(seeds), 0, 0) == 100
    assert episode_id_for(25, len(seeds), 0, 1) == 101
    assert episode_id_for(25, len(seeds), 1, 0) == 102
    assert episode_id_for(25, len(seeds), 1, 1) == 103
    # Variant subset never enters the formula.
    for variant in ("baseline", "care", "fertilizer", "fertilizer_wheat3",
                    "combined", "combined_wheat3"):
        UpkeepFactory(0, variant)  # must not raise
    # Identity-preserving single-game selection keeps original numbering.
    selected = parse_game_filter(["144368101:1"], seeds)
    assert selected == {(0, 1)}
    index, seat = next(iter(selected))
    assert episode_id_for(25, len(seeds), index, seat) == 101
    with pytest.raises(ValueError):
        parse_game_filter(["999:0"], seeds)


def test_factory_versions_distinguish_capture_without_changing_defaults():
    plain = UpkeepFactory(0, "combined")
    assert plain.version == "v1:combined:candidate-seat-0"
    assert plain.capture is False
    captured = UpkeepFactory(0, "combined", capture=True)
    assert captured.version != plain.version

    class Provider:
        def daily_plan(self, obs, seat, previous_execution=None):
            return make_plan()

    for acting_seat in (0, 1):
        agent = captured.create(backend_name="fast", seat=acting_seat,
                                configuration={}, provider=Provider())
        # Both sides traced and labeled by seat; heuristics stay
        # candidate-only.
        assert agent.config.record_turn_snapshot is True
        assert agent.config.heuristic_care == (acting_seat == 0)
        assert agent.config.heuristic_fertilizer == (acting_seat == 0)


def test_new_executor_controls_are_candidate_only():
    class Provider:
        def daily_plan(self, obs, seat, previous_execution=None):
            return make_plan()

    factory = UpkeepFactory(
        0, "combined", underfoot_first=True,
        deadline_safe_planting=True, deadline_safe_hiring=True,
        persistent_worker_queues=True, schedule_informed_hiring=True)
    candidate = factory.create(backend_name="fast", seat=0,
                               configuration={}, provider=Provider())
    opponent = factory.create(backend_name="fast", seat=1,
                              configuration={}, provider=Provider())
    assert candidate.config.foreman.underfoot_first is True
    assert candidate.config.deadline_safe_planting is True
    assert candidate.config.deadline_safe_hiring is True
    assert candidate.config.persistent_worker_queues is True
    assert candidate.config.schedule_informed_hiring is True
    assert opponent.config.foreman.underfoot_first is False
    assert opponent.config.deadline_safe_planting is False
    assert opponent.config.deadline_safe_hiring is False
    assert opponent.config.persistent_worker_queues is False
    assert opponent.config.schedule_informed_hiring is False


# ------------------------------------------------------- capture writer


def _tile(*, plant: bool = True) -> dict:
    if plant:
        return {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                "max_lifespan_step": 100, "yield_units": 1,
                "watered_today": False, "consecutive_unwatered": 0,
                "fertilized_until_day": 0}
    return {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0,
            "yield_units": 1, "consecutive_unfed": 0, "fed_today": False,
            "cared_today": False, "fertilizer_available": False,
            "pending_care_bonus": 0}


def _state(step: int, day: int = 0, hour: int = 0,
           money: float = 3000.0) -> dict:
    farm = {"money": money, "tiles": [[_tile(), "LOCKED"]],
            "farmer": [0, 0], "hands": [], "unlocked_quadrants": ["NW"],
            "hires_today": 0}
    return {"step": step, "day": day, "hour": hour,
            "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
            "privates": [
                {"shed": {"WHEAT": 0}, "seeds": {"WHEAT": 1},
                 "inventories": [{}]},
                {"shed": {"WHEAT": 0}, "seeds": {"WHEAT": 1},
                 "inventories": [{}]}],
            "market": {"inventory": {"WHEAT": 10}, "prices": {"WHEAT": 5}},
            "town": {"unlocked_shops": []},
            "rewards": [0.0, 0.0], "statuses": ["ACTIVE", "ACTIVE"]}


class _StubRollout:
    def __init__(self) -> None:
        self.seed = 1
        self.backend_name = "fast"
        self.composition = E_VS_E
        self.joint_actions = [
            (0, 0, 0, {"farmer": ["PASS"]}, {"farmer": ["PASS"]})]
        self.manager_input_digests = {(0, 4): "abc"}
        self.plans = {(0, 4): {"land_count": 1}}
        self.opening_handoff = [{"seat": 0}]


def _meta() -> dict:
    return {"variant": "baseline", "seed": 1, "seat": 0, "episode_id": 100,
            "master_seed": 25, "candidate_seat": 0,
            "composition": "candidate_vs_frozen",
            "final_banks": [5000.0, 4000.0], "margin": 1000.0,
            "winner_seat": 0, "rewards": [1.0, -1.0],
            "statuses": ["DONE", "DONE"], "terminated": True,
            "trace_digest": "ff"}


def _recorder() -> TraceRecorder:
    recorder = TraceRecorder({"seed": 1, "view": "joint",
                              "backend": "fast", "engine": "x",
                              "provenance": {}})
    recorder.append_turn(step=0, day=0, hour=0, canonical_state=_state(0),
                         joint_actions={"0": {"farmer": ["PASS"]},
                                        "1": {"farmer": ["PASS"]}},
                         executor_debug={"0": {}, "1": {}})
    recorder.append_turn(step=1, day=0, hour=1, canonical_state=_state(1, hour=1))
    return recorder


def test_capture_writer_roundtrip_and_partial(tmp_path: Path):
    directory = game_dir(tmp_path, "baseline", 100, 1, 0)
    report = write_game_capture(
        directory, meta=_meta(), debug_trace=_recorder().build(),
        rollout=_StubRollout(),
        executor_full_diagnostics=[{"seat": 0}, {"seat": 1}],
        official_replay=None, status_history=[["ACTIVE", "ACTIVE"]])
    assert report["complete"] is True
    assert (directory / "meta.json").is_file()
    assert (directory / "replay.json.gz").exists() is False
    assert read_json_gz(directory / "rollout.json.gz")["plans"] == {
        "0/4": {"land_count": 1}}
    assert [g["meta"]["variant"] for g in
            iter_captured_games(tmp_path)] == ["baseline"]

    # Unserializable payload fails one file but keeps earlier partials.
    bad_dir = game_dir(tmp_path, "care", 100, 1, 0)
    bad = write_game_capture(
        bad_dir, meta=_meta(), debug_trace=_recorder().build(),
        rollout=_StubRollout(),
        executor_full_diagnostics=[{"seat": 0}],
        official_replay={"nan": float("nan")}, status_history=None)
    assert bad["complete"] is False
    assert (bad_dir / "capture_error.json").is_file()
    assert (bad_dir / "debug_trace.json.gz").is_file()
    assert (bad_dir / "meta.json").is_file()
    assert not any(g["meta"]["variant"] == "care"
                   for g in iter_captured_games(tmp_path))


def test_audit_runs_on_real_scripted_capture(tmp_path: Path):
    """Audit must tolerate real executor snapshot shapes (truncated game)."""
    from rl_manager.stage25_capture import write_game_capture as _write

    result, _ = _run_short_game(capture=True)
    directory = game_dir(tmp_path / "captures", "baseline", 100,
                         144368101, 0)
    meta = {"variant": "baseline", "seed": 144368101, "seat": 0,
            "episode_id": 100, "master_seed": 25, "candidate_seat": 0,
            "composition": E_VS_E,
            "final_banks": [float(b) for b in result.final_banks],
            "margin": float(result.margin),
            "winner_seat": int(result.winner_seat),
            "rewards": [float(r) for r in result.rewards],
            "statuses": list(result.statuses),
            "terminated": bool(result.terminated),
            "trace_digest": str(result.trace_digest)}
    report = _write(
        directory, meta=meta, debug_trace=result.debug_trace,
        rollout=result.rollout,
        executor_full_diagnostics=result.executor_full_diagnostics,
        official_replay=result.official_replay,
        status_history=result.status_history)
    assert report["complete"] is True
    summary = run_audit(capture_dir=tmp_path / "captures",
                        output_dir=tmp_path / "audit",
                        baseline_variant="baseline", focus_pairs=4)
    assert summary["games"] == 1
    assert summary["day_rows"] >= 5  # d0..d4 covered by the short game
    game = analyze_game(load_game(directory))
    assert game["game"]["cand_worker_turns"] > 0
    assert (tmp_path / "audit" / "audit.md").is_file()


# ------------------------------------------------------- audit fixture


def _debug(seat_actions: dict[str, list[dict]], tasks: list[dict],
           unassigned: list[str]) -> dict:
    return {seat: {"tasks": tasks if seat == "0" else [],
                   "assignments": actions,
                   "unassigned": {"task_keys": (unassigned if seat == "0"
                                                else []), "reasons": {}},
                   "market": {"submitted": [], "unaffordable": [],
                              "skipped": []}}
            for seat, actions in seat_actions.items()}


def _fixture_game(tmp_path: Path, variant: str, diverge: bool,
                  banks: list[float]) -> Path:
    water = {"key": "WATER:0,0", "kind": "WATER"}
    recorder = TraceRecorder({"seed": 7, "view": "joint", "backend": "fast",
                              "engine": "x", "provenance": {}})
    turn1_action = ["NORTH"] if diverge else ["WATER"]
    recorder.append_turn(
        step=0, day=0, hour=0, canonical_state=_state(0),
        joint_actions={"0": {"farmer": ["NORTH"], "hands": [], "market": []},
                       "1": {"farmer": ["PASS"], "hands": [], "market": []}},
        executor_debug=_debug(
            {"0": [{"worker_index": 0, "task_key": "WATER:0,0",
                    "reason": "closest", "action": ["NORTH"],
                    "target": [0, 0]}],
             "1": [{"worker_index": 0, "task_key": None,
                    "reason": "idle", "action": ["PASS"],
                    "target": None}]},
            [water], []))
    recorder.append_turn(
        step=1, day=0, hour=1, canonical_state=_state(1, hour=1),
        joint_actions={"0": {"farmer": turn1_action, "hands": [],
                             "market": []},
                       "1": {"farmer": ["PASS"], "hands": [], "market": []}},
        executor_debug=_debug(
            {"0": [{"worker_index": 0, "task_key": "WATER:0,0",
                    "reason": "closest", "action": turn1_action,
                    "target": [0, 0]}],
             "1": [{"worker_index": 0, "task_key": None,
                    "reason": "idle", "action": ["PASS"],
                    "target": None}]},
            [water], []))
    recorder.append_turn(
        step=2, day=0, hour=2,
        canonical_state=_state(2, hour=2,
                               money=3001.0 if diverge else 3000.0),
        joint_actions={"0": {"farmer": ["PASS"], "hands": [], "market": []},
                       "1": {"farmer": ["PASS"], "hands": [], "market": []}},
        executor_debug=_debug(
            {"0": [{"worker_index": 0, "task_key": None,
                    "reason": "idle", "action": ["PASS"],
                    "target": None}],
             "1": [{"worker_index": 0, "task_key": None,
                    "reason": "idle", "action": ["PASS"],
                    "target": None}]},
            [], [] if diverge else ["FEED:1,1"]))
    meta = {"variant": variant, "seed": 7, "seat": 0, "episode_id": 100,
            "master_seed": 25, "candidate_seat": 0,
            "composition": "candidate_vs_frozen",
            "final_banks": banks, "margin": banks[0] - banks[1],
            "winner_seat": 0, "rewards": [1.0, -1.0],
            "statuses": ["DONE", "DONE"], "terminated": True,
            "trace_digest": "ff"}
    directory = game_dir(tmp_path / "captures", variant, 100, 7, 0)

    class _Rollout:
        seed = 7
        backend_name = "fast"
        composition = "candidate_vs_frozen"
        joint_actions = []
        manager_input_digests = {}
        plans = {(0, 4): {"land_count": 1 if not diverge else 2}}
        opening_handoff = []

    report = write_game_capture(
        directory, meta=meta, debug_trace=recorder.build(),
        rollout=_Rollout(),
        executor_full_diagnostics=[{"seat": 0, "days": {}}, {"seat": 1}],
        official_replay=None, status_history=None)
    assert report["complete"] is True
    return directory


def test_audit_fixture_movement_completion_idle_and_divergence(
        tmp_path: Path):
    _fixture_game(tmp_path, "baseline", diverge=False,
                  banks=[5000.0, 4000.0])
    _fixture_game(tmp_path, "combined", diverge=True,
                  banks=[4800.0, 4100.0])
    capture_dir = tmp_path / "captures"
    games = {Path(g["directory"]).parent.name: load_game(Path(g["directory"]))
             for g in iter_captured_games(capture_dir)}
    assert len(games) == 2
    base = analyze_game(games["baseline"])
    assert base["game"]["cand_movement"] == 1
    assert base["game"]["cand_productive"] == 1
    assert base["game"]["cand_completed"] == {"WATER": 1}
    assert base["game"]["cand_idle_with_queued"] == 1
    assert base["game"]["cand_assign_changes"] == 1
    assert base["game"]["cand_changes_after_interaction"] == 1
    assert base["game"]["cand_changes_without_interaction"] == 0

    summary = run_audit(capture_dir=capture_dir,
                        output_dir=tmp_path / "audit",
                        baseline_variant="baseline", focus_pairs=4)
    assert summary["games"] == 2
    divergences = json.loads(
        (tmp_path / "audit" / "divergences.json").read_text(
            encoding="utf-8"))
    combined = next(d for d in divergences if d["variant"] == "combined")
    assert combined["first_action_turn"] == 1
    assert combined["first_state_turn"] == 2
    assert combined["first_plan_seat"] == 0
    assert combined["first_plan_day"] == 4
    assert combined["bank_delta"] == pytest.approx(-200.0)
    assert combined["opponent_bank_delta"] == pytest.approx(100.0)
    assert combined["margin_delta"] == pytest.approx(-300.0)
    markdown = (tmp_path / "audit" / "audit.md").read_text(encoding="utf-8")
    assert "idle with queued work" in markdown
    assert "hypotheses only" in markdown
    assert "not causal" in markdown
    for name in ("games.csv", "days.csv", "pairs.csv",
                 "divergences.csv", "audit.md"):
        assert (tmp_path / "audit" / name).is_file()
