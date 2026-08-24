"""Stage-3 tests: live E/JE provider history parity, diagnostic-only
exposure through provider/agent/opening harness, and the four-variant panel
orchestration against a mocked ``run_opening_game`` boundary.

No real corpus/checkpoints: tiny real-format checkpoints are synthesized;
the official engine is faked at existing stable boundaries only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_bc_manager_v1_economics import make_record  # noqa: E402
from test_bc_manager_v1_economics import _live_obs  # noqa: E402

import bc_manager.ablation as ablation  # noqa: E402
from bc_manager.adapter import load_dataset  # noqa: E402
from bc_manager.economics import ECONOMIC_CONTEXT_KEY  # noqa: E402
from bc_manager.model import DailyManagerTransformer, tiny_manager_config  # noqa: E402
from bc_manager.training import TrainingConfig, save_checkpoint  # noqa: E402
from executor_v0.agent import AgentConfig, ExecutorAgent  # noqa: E402
from executor_v0.manager import CheckpointPlanProvider  # noqa: E402
from opening_book.eval import run_opening_game as real_run_opening_game  # noqa: E402
from replay_daily.storage import write_parquet  # noqa: E402


# ------------------------------------------------------- tiny checkpoints


@pytest.fixture(scope="module")
def checkpoint_paths(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("ckpts")
    paths = {}
    for variant in ("V0", "J", "E", "JE"):
        torch.manual_seed(11)
        model = DailyManagerTransformer(tiny_manager_config(),
                                        model_variant=variant)
        path = root / f"{variant}.pt"
        save_checkpoint(path, kind="best", epoch=1, model=model,
                        model_config=tiny_manager_config(),
                        training_config=TrainingConfig(checkpoint_dir=None),
                        validation_metrics={"total": 1.5},
                        model_variant=variant)
        paths[variant] = path
    return paths


# ------------------------------------- 1. live E/JE history + batch parity


def _capture_provider_batches(provider):
    captured = []

    def spy(module, inputs):
        batch = inputs[0]
        captured.append(
            {k: v.clone() for k, v in batch.items()})

    handle = provider.model.register_forward_pre_hook(spy)
    return captured, handle


def test_e_provider_history_matches_batch_derivation(checkpoint_paths,
                                                     tmp_path):
    money_by_day = {0: 100.0, 1: 250.0, 2: 250.0, 3: 400.0}
    provider = CheckpointPlanProvider(checkpoint_paths["E"], device="cpu")
    assert provider.model_variant == "E"
    captured, handle = _capture_provider_batches(provider)
    try:
        for day in sorted(money_by_day):
            plan = provider.daily_plan(_live_obs(0, day, money_by_day[day]),
                                       0, None)
            assert plan is not None
    finally:
        handle.remove()

    # Batch reference over equivalent canonical rows.
    records = [make_record(900, 0, day, money=money_by_day[day])
               for day in sorted(money_by_day)]
    path = tmp_path / "ref.parquet"
    write_parquet(records, path)
    data = load_dataset(path, dates=["2026-08-17"], min_score=2950.0,
                        with_economic_context=True)
    by_day = {m["day"]: i for i, m in enumerate(data["meta"])}
    for pos, day in enumerate(sorted(money_by_day)):
        np.testing.assert_array_equal(
            captured[pos][ECONOMIC_CONTEXT_KEY].numpy()[0],
            data["inputs"][ECONOMIC_CONTEXT_KEY][by_day[day]])
    # Day 0 invalid, later days valid with exact deltas.
    assert captured[0][ECONOMIC_CONTEXT_KEY].numpy()[0][13] == 0.0
    assert captured[1][ECONOMIC_CONTEXT_KEY].numpy()[0][13] == 1.0
    assert captured[1][ECONOMIC_CONTEXT_KEY].numpy()[0][12] == \
        pytest.approx(np.float32(np.sign(150.0)
                                 * np.log1p(abs(150.0) * 1e-4)), abs=1e-6)


def test_je_provider_also_feeds_history_and_v0_j_do_not(checkpoint_paths):
    for variant in ("V0", "J"):
        provider = CheckpointPlanProvider(checkpoint_paths[variant])
        assert provider.model_variant == variant
        assert provider._economic_history is None
        captured, handle = _capture_provider_batches(provider)
        try:
            provider.daily_plan(_live_obs(0, 0, 3000.0), 0, None)
        finally:
            handle.remove()
        assert ECONOMIC_CONTEXT_KEY not in captured[0]
    provider = CheckpointPlanProvider(checkpoint_paths["JE"])
    assert provider.model_variant == "JE"
    assert provider._economic_history is not None
    captured, handle = _capture_provider_batches(provider)
    try:
        provider.daily_plan(_live_obs(0, 0, 3000.0), 0, None)
    finally:
        handle.remove()
    assert ECONOMIC_CONTEXT_KEY in captured[0]


def test_provider_tracker_resets_across_gap_days(checkpoint_paths):
    provider = CheckpointPlanProvider(checkpoint_paths["E"])
    captured, handle = _capture_provider_batches(provider)
    try:
        provider.daily_plan(_live_obs(0, 0, 100.0), 0, None)
        provider.daily_plan(_live_obs(0, 1, 150.0), 0, None)
        provider.daily_plan(_live_obs(0, 5, 400.0), 0, None)  # gap
    finally:
        handle.remove()
    econ = [c[ECONOMIC_CONTEXT_KEY].numpy()[0] for c in captured]
    assert econ[0][13] == 0.0
    assert econ[1][13] == 1.0
    assert econ[2][13] == 0.0 and econ[2][12] == 0.0  # gap -> invalid


# ------------------------- 2. diagnostic exposure is JSON-safe, inert


def test_provider_diagnostics_json_safe_and_complete(checkpoint_paths):
    provider = CheckpointPlanProvider(checkpoint_paths["E"])
    for day, money in ((0, 100.0), (1, 30.0)):  # day1: cash<=0? no, 30>0
        provider.daily_plan(_live_obs(0, day, money), 0, None)
    report = provider.diagnostics_json()
    json.dumps(report, allow_nan=False)  # must never emit Infinity/NaN
    assert report["model_variant"] == "E"
    assert len(report["records"]) == 2
    aggregate = report["aggregate"]
    for key in ("days", "lower_bound_cost_mean", "lower_bound_cost_median",
                "over_1x_rate", "over_2x_rate",
                "zero_cash_positive_cost_days", "land_regression_rate"):
        assert key in aggregate
    record = report["records"][0]
    assert record["ratio"] is not None or record["zero_cash"] is True
    assert isinstance(record["over_1x"], bool)


def test_zero_cash_positive_cost_record_is_flagged_not_infinite(
        checkpoint_paths):
    provider = CheckpointPlanProvider(checkpoint_paths["V0"])
    # A random tiny model will request positive counts; cash 0 forces the
    # zero-cash flag path regardless of the sampled plan.
    provider.daily_plan(_live_obs(0, 0, 0.0), 0, None)
    record = provider.diagnostics_json()["records"][0]
    assert record["cash"] == 0.0
    if record["lower_bound_cost"] > 0:
        assert record["zero_cash"] is True and record["ratio"] is None
        assert record["over_1x"] and record["over_2x"]
    else:
        assert record["ratio"] == 0.0


def test_agent_actions_identical_with_and_without_diagnostics_access(
        checkpoint_paths):
    obs_seq = [_live_obs(0, day, 3000.0 - day) for day in range(3)]

    def build_agent():
        provider = CheckpointPlanProvider(checkpoint_paths["V0"])
        return ExecutorAgent(provider, seat=0, config=AgentConfig())

    agent_a, agent_b = build_agent(), build_agent()
    actions_a = [agent_a(obs) for obs in obs_seq[:2]]
    agent_a.diagnostics_json()  # diagnostics access between calls
    actions_a.append(agent_a(obs_seq[2]))
    actions_b = [agent_b(obs) for obs in obs_seq]
    assert actions_a == actions_b


def test_fixed_provider_agent_has_no_provider_diagnostics_key():
    from executor_v0.manager import FixedPlanProvider
    from executor_v0.plan import DailyPlan
    plan = DailyPlan.create(
        crop_targets={"WHEAT": 1, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0,
                      "MELON": 0},
        animal_targets={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        land_count=1,
        fertilizer_by_crop={"WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 0, "COW": 0, "SHEEP": 0},
        sell_quantities={p: {a: 0 for a in (0, 4, 8, 12, 16, 20)}
                         for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                                   "MELON", "EGG", "MILK", "WOOL",
                                   "FERTILIZER")})
    agent = ExecutorAgent(FixedPlanProvider(plan), seat=0)
    diag = agent.diagnostics_json()
    assert "provider_diagnostics" not in diag


# ------------------------ 3. opening record downstream diagnostics additive


class _FakeState:
    def __init__(self, reward=100, farms_money=(29.0, 3000.0)):
        self.status = "ACTIVE"
        self.reward = reward
        self.observation = type("Obs", (), {})()
        self.observation.farms = [{"money": m} for m in farms_money]


class _FakeEnv:
    """Minimal standard_mixed horizon driving the wrapper to d4h0."""

    def reset(self):
        pass

    def run(self, agents):
        sys.path.insert(0, str(Path(__file__).parent))
        from opening_book.trace import load_built_in_trace
        from test_opening_book_eval import HORIZON, standard_envelope_state
        envelope = standard_envelope_state()
        trace = load_built_in_trace("standard_mixed")
        steps = []
        for day, hour in HORIZON + [(4, 0)]:
            idx = day * 24 + hour
            n_hands = len(trace["turns"][idx]["action"]["hands"]) \
                if idx < 96 else 5
            farms = [{}, {}]
            for s in (0, 1):
                summary = envelope.get(s, {})
                farms[s] = {
                    "money": summary.get("money", 30.0),
                    "tiles": summary.get("tiles", []),
                    "unlocked_quadrants": ["NE"],
                    "hands": [None] * n_hands,
                }
            obs = {"day": day, "hour": hour, "farms": farms, "player": 0}
            agents[0](obs, {"seed": 1})
            agents[1](obs, {"seed": 1})
            steps.append([_FakeState(), _FakeState()])
        steps.append([_FakeState(reward=100), _FakeState(reward=50)])
        return steps


class _DiagDownstream:
    def __init__(self):
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def diagnostics_json(self):
        return {"marker": True, "calls": self.calls}


def test_opening_record_carries_downstream_diagnostics_additively():
    downstream = _DiagDownstream()
    record = real_run_opening_game(
        lambda env_id, configuration=None: _FakeEnv(),
        opening="standard_mixed", seat=0, seed=1146601720,
        opponent_kind="pass", downstream_factory=lambda: downstream)
    assert record["downstream_diagnostics"] == \
        {"marker": True, "calls": record["downstream_diagnostics"]["calls"]}
    assert record["passed"] is True, record["failure_reasons"]
    assert record["opening_diagnostics"]["turns_replayed"] == 96


def test_opening_pass_downstream_yields_null_diagnostics():
    from opening_book.eval import pass_action
    record = real_run_opening_game(
        lambda env_id, configuration=None: _FakeEnv(),
        opening="standard_mixed", seat=0, seed=7, opponent_kind="pass",
        downstream_factory=lambda: pass_action)
    assert record["downstream_diagnostics"] is None
    assert record["passed"] is True, record["failure_reasons"]


# ------------------------------- 4/5. panel orchestration (mocked boundary)


class _FakeGameRecord(dict):
    """Build a record shaped exactly like run_opening_game output."""

    def __init__(self, seed, seat, bank_self, *, passed=True,
                 diverged=False):
        super().__init__(
            mode="panel_X", seed=seed, seat=seat, opponent="pass",
            passed=passed,
            failure_reasons=[] if passed else ["forced failure"],
            final_banks=[float(bank_self), 3000.0],
            opening_diagnostics={
                "divergence": {"occurred": diverged},
                "fallback_active": diverged,
            },
            downstream_diagnostics={
                "fallback_errors": [],
                "days": {"0": {"unfinished_tasks": ["A"],
                               "missed_maintenance": ["WATER:A"]}},
                "provider_diagnostics": {
                    "aggregate": {"days": 4, "over_1x_rate": 0.25,
                                  "over_2x_rate": 0.0},
                },
            },
            status_anomalies=[],
        )


@pytest.fixture()
def panel_checkpoints(checkpoint_paths):
    return {v: str(p) for v, p in checkpoint_paths.items()}


def test_panel_full_matrix_aggregation_and_ranking(monkeypatch,
                                                   panel_checkpoints):
    seen = []

    def fake_run(engine_make, *, opening, seat, seed, opponent_kind,
                 downstream_factory, mode):
        variant = mode.removeprefix("panel_")
        seen.append((variant, seed, seat))
        # Deterministic per-(variant, seed) bank independent of seat pairing.
        bank = 100.0 + sorted(("V0", "J", "E", "JE")).index(variant) * 10.0 \
            + (seed % 7)
        if variant == "V0" and seed == 2026:
            bank = 1000.0
        if variant == "JE" and seed == 2026:
            bank = 500.0
        if variant == "J" and seed == 17:
            bank = 50.0  # collapse flag
        return _FakeGameRecord(seed, seat, bank)

    monkeypatch.setattr(ablation, "detect_engine",
                        lambda: {"available": True, "version": "1.32.7",
                                 "reason": ""})
    report = run_panel_with_mock(fake_run, panel_checkpoints)

    assert report["status"] == "complete"
    assert len(seen) == 4 * 5 * 2
    assert sorted(set(seen)) == sorted(
        (v, s, seat) for v in ("V0", "J", "E", "JE")
        for s in (7, 17, 42, 123, 2026) for seat in (0, 1))
    by_variant = report["by_variant"]
    assert all(by_variant[v]["games"] == 10 for v in by_variant)
    # Ranking uses closed-loop bank medians, not teacher-forced totals.
    ranking = report["ranking"]
    medians = [entry["bank_median"] for entry in ranking]
    assert medians == sorted(medians, reverse=True)
    assert ranking[0]["selection_criterion"] == \
        "closed_loop_final_bank_median_then_mean"
    # Seed 17 collapse flag transparently reported with raw bank.
    j_summary = by_variant["J"]
    assert j_summary["seed17_final_bank"] == 50.0
    assert j_summary["seed17_collapse"] is True
    assert j_summary["seed17_collapse_threshold"] == 100.0
    # Seed 2026 upside retention relative to V0.
    assert by_variant["V0"]["seed2026_final_bank"] == 1000.0
    assert by_variant["JE"]["seed2026_upside_retention_vs_v0"] == \
        pytest.approx(0.5)
    # Teacher-forced metrics carried as prerequisites, not ranking inputs.
    for variant, summary in by_variant.items():
        assert summary["teacher_forced_validation_total"] == 1.5
    # Executor summaries aggregated.
    assert by_variant["V0"]["unfinished_tasks_total"] == 10
    assert by_variant["V0"]["missed_maintenance_total"] == 10
    json.dumps(report, allow_nan=False)


def run_panel_with_mock(fake_run, checkpoints):
    saved = real_run_opening_game  # noqa: F841 - documentation only
    import opening_book.eval as ob_eval
    saved_fn = ob_eval.run_opening_game
    ob_eval.run_opening_game = fake_run
    try:
        return ablation.run_panel(lambda *a, **k: None,
                                  checkpoints=checkpoints)
    finally:
        ob_eval.run_opening_game = saved_fn


def test_panel_fails_loudly_on_failed_or_missing_games(monkeypatch,
                                                       panel_checkpoints):
    def fake_run(engine_make, *, opening, seat, seed, opponent_kind,
                 downstream_factory, mode):
        variant = mode.removeprefix("panel_")
        if variant == "E" and seed == 42 and seat == 0:
            return _FakeGameRecord(seed, seat, 10.0, passed=False)
        return _FakeGameRecord(seed, seat, 100.0)

    import opening_book.eval as ob_eval
    monkeypatch.setattr(ob_eval, "run_opening_game", fake_run)
    report = ablation.run_panel(lambda *a, **k: None,
                                checkpoints=panel_checkpoints)
    assert report["status"] == "partial"
    assert any(f["variant"] == "E" and f["seed"] == 42 and f["seat"] == 0
               for f in report["failed_games"])

    class Boom(Exception):
        pass

    def exploding_run(engine_make, *, opening, seat, seed, opponent_kind,
                      downstream_factory, mode):
        raise Boom("engine exploded")

    monkeypatch.setattr(ob_eval, "run_opening_game", exploding_run)
    report = ablation.run_panel(lambda *a, **k: None,
                                checkpoints=panel_checkpoints)
    assert report["status"] == "failed"
    assert report["errors"][0]["error_type"] == "Boom"


def test_panel_rejects_mismatched_or_metricless_checkpoints(tmp_path):
    torch.manual_seed(0)
    model = DailyManagerTransformer(tiny_manager_config())
    wrong_variant = tmp_path / "wrong.pt"
    save_checkpoint(wrong_variant, kind="best", epoch=1, model=model,
                    model_config=tiny_manager_config(),
                    training_config=TrainingConfig(checkpoint_dir=None),
                    validation_metrics={"total": 1.0}, model_variant="V0")
    with pytest.raises(ValueError, match="model_variant"):
        ablation.load_panel_checkpoint(wrong_variant, "E")

    no_metrics = tmp_path / "nometrics.pt"
    save_checkpoint(no_metrics, kind="best", epoch=1, model=model,
                    model_config=tiny_manager_config(),
                    training_config=TrainingConfig(checkpoint_dir=None),
                    validation_metrics={}, model_variant="V0")
    with pytest.raises(ValueError, match="validation_metrics"):
        ablation.load_panel_checkpoint(no_metrics, "V0")

    with pytest.raises(FileNotFoundError, match="not found"):
        ablation.load_panel_checkpoint(tmp_path / "missing.pt", "V0")


def test_parse_checkpoint_mapping_completeness():
    mapping = ablation.parse_checkpoint_mapping(
        ["V0=a.pt", "j=b.pt", "E=c.pt", "JE=d.pt"])
    assert sorted(mapping) == ["E", "J", "JE", "V0"]  # case-normalized
    with pytest.raises(ValueError, match="incomplete"):
        ablation.parse_checkpoint_mapping(["V0=a.pt"])
    with pytest.raises(ValueError, match="unknown model_variant"):
        ablation.parse_checkpoint_mapping(
            [f"{v}=x.pt" for v in ("V0", "J", "E", "JE")] + ["XX=y.pt"])
    with pytest.raises(ValueError, match="VARIANT=PATH"):
        ablation.parse_checkpoint_mapping(["V0a.pt"])
    with pytest.raises(ValueError, match="model_variant"):
        ablation.parse_checkpoint_mapping(
            ["ZZ=x.pt"] + [f"{v}=x.pt" for v in ("V0", "J", "E")])


# ------------------------------------------------------------ 6. CLI


def test_cli_help_mentions_exact_contract(capsys):
    with pytest.raises(SystemExit) as exc:
        ablation.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for token in ("--checkpoint", "VARIANT=PATH", "--validate-only",
                  "--seeds", "standard_mixed"):
        assert token in out


def test_cli_validate_only_preflight_without_engine(
        checkpoint_paths, tmp_path, monkeypatch):
    def forbidden_engine():
        raise AssertionError("validate-only must not touch the engine")

    monkeypatch.setattr(ablation, "detect_engine", forbidden_engine)
    argv: list[str] = []
    for v, p in checkpoint_paths.items():
        argv += ["--checkpoint", f"{v}={p}"]
    argv += ["--validate-only", "--out", str(tmp_path / "preflight.json")]
    assert ablation.main(argv) == 0
    payload = json.loads((tmp_path / "preflight.json").read_text("utf-8"))
    assert payload["status"] == "validated"
    assert payload["mode"] == "validate_only"
    assert payload["panel"]["expected_games"] == 40
    assert set(payload["checkpoints"]) == {"V0", "J", "E", "JE"}
    json.dumps(payload, allow_nan=False)


def test_cli_usage_errors(checkpoint_paths):
    # Incomplete mapping.
    assert ablation.main(["--checkpoint", "V0=a.pt", "--validate-only"]) == 2
    # Stored variant mismatch against its mapping slot fails the preflight.
    swapped = [
        "--checkpoint", f"V0={checkpoint_paths['E']}",
        "--checkpoint", f"J={checkpoint_paths['J']}",
        "--checkpoint", f"E={checkpoint_paths['V0']}",
        "--checkpoint", f"JE={checkpoint_paths['JE']}",
        "--validate-only",
    ]
    assert ablation.main(swapped) == 2
