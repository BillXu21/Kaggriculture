"""Focused contracts for native Stage 2.5 packaging and runtime behavior."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

import jax
import numpy as np
import pytest

from bc_manager.economics import E_HISTORY_CORRECTED_V1
from bc_manager_jax.train import TrainConfig, init_opt_state
from opening_book.trace import action_for, load_built_in_trace
from replay_daily.constants import total_hire_cost
from rl_manager.stage25_checkpoint import (
    Stage25CheckpointError,
    load_stage25_inference_checkpoint,
    save_stage25_bc_checkpoint,
    save_stage25_inference_checkpoint,
)
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_provider import Stage25PlanProvider
from rl_manager.stage25_submission import (
    RealizedLaborTracker,
    Stage25SubmissionAgent,
)
from rl_manager.stage25_submission_observation import (
    canonicalize_official_observation,
    executor_observation,
)
from tools.build_stage25_submission import build_submission
from tools.verify_stage25_submission import (
    VerificationError,
    extract_fresh,
    verify_archive,
)


ROOT = Path(__file__).resolve().parents[1]
HOLD = (0, 1, 0, 1, 100, 100, 100, 100, 100)


@pytest.fixture(scope="module")
def inference_checkpoint(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("stage25-submission") / "tiny.npz"
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=117)
    save_stage25_inference_checkpoint(path, params, config, seed=117)
    return path


def _farm(money: float = 3000.0, hires_today: int = 0) -> dict:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "max_lifespan_step": -1, "yield_units": 0,
        "watered_today": True, "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
    }
    tiles[0][1] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 0, "consecutive_unfed": 0, "fed_today": True,
        "cared_today": True, "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    tiles[0][2] = {
        "kind": "PASTURE", "animal": "SHEEP", "placed_day": 0,
        "yield_units": 0, "consecutive_unfed": 0, "fed_today": True,
        "cared_today": True, "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    return {
        "money": money, "tiles": tiles, "farmer": [0, 0], "hands": [],
        "unlocked_quadrants": ["NW"], "hires_today": hires_today,
    }


def _observation(day: int, hour: int = 0, *, seat: int = 0,
                 hands: int = 0, hires_today: int = 0,
                 money: float = 3000.0) -> dict:
    own = _farm(money, hires_today)
    own["hands"] = [[index, 0] for index in range(hands)]
    other = _farm(money, hires_today)
    other["hands"] = [[index, 0] for index in range(hands)]
    return {
        "player": seat, "day": day, "hour": hour,
        "step": day * 24 + hour,
        "farms": [own, other],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": []},
    }


def test_archive_reproducibility_manifest_and_safe_members(
        inference_checkpoint: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    report = build_submission(
        inference_checkpoint, first, label="tiny-validation")
    build_submission(inference_checkpoint, second, label="tiny-validation")

    assert first.read_bytes() == second.read_bytes()
    assert report["archive_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert report["archive_size_bytes"] == first.stat().st_size
    with __import__("tarfile").open(first, "r:gz") as archive:
        names = archive.getnames()
        assert {"main.py", "stage25.npz", "submission_manifest.json"} <= set(names)
        for name in names:
            member = PurePosixPath(name)
            assert not member.is_absolute() and ".." not in member.parts
            assert "tests/" not in name and ".git/" not in name
        assert not any(name.startswith("fast_env/") for name in names)
        manifest = json.load(archive.extractfile("submission_manifest.json"))
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        inference_checkpoint.read_bytes()).hexdigest()
    assert manifest["checkpoint_payload_kind"] == "stage25_inference_params_v1"
    assert manifest["e_history_version"] == E_HISTORY_CORRECTED_V1
    assert manifest["inference_mode"] == "deterministic"
    assert manifest["executor_profile"]["aggressive_sell_all"] is True
    assert manifest["jax_versions_at_build"]["jax"] == jax.__version__


def test_builder_accepts_native_inference_and_rejects_training_state(
        inference_checkpoint: Path, tmp_path: Path) -> None:
    _, metadata = load_stage25_inference_checkpoint(inference_checkpoint)
    assert metadata["payload_kind"] == "stage25_inference_params_v1"
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=117)
    training_path = tmp_path / "training.npz"
    optimizer = init_opt_state(params, TrainConfig())
    save_stage25_bc_checkpoint(
        training_path, params, optimizer, np.asarray([1, 2], dtype=np.uint32),
        config, seed=117, step=4, epoch=1, optimizer_config=TrainConfig(),
        data_order_position={"epoch": 1, "batch": 0, "seed": 117},
    )
    with pytest.raises(Stage25CheckpointError, match="payload kind"):
        build_submission(training_path, tmp_path / "should-not-exist.tar.gz",
                         label="training-state")


def test_builder_rejects_malformed_and_wrong_version_checkpoints(
        inference_checkpoint: Path, tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not an npz")
    with pytest.raises(Stage25CheckpointError, match="corrupt or unreadable"):
        build_submission(malformed, tmp_path / "bad.tar.gz", label="bad")

    wrong_version = tmp_path / "wrong-version.npz"
    with np.load(inference_checkpoint, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(arrays["__meta__"].tobytes().decode("utf-8"))
    meta["format"] = "stage25_native_checkpoint_v0"
    arrays["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with wrong_version.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(Stage25CheckpointError, match="version-incompatible"):
        build_submission(wrong_version, tmp_path / "wrong.tar.gz", label="wrong")


def test_clean_extract_loads_kaggle_callable_without_checkout_fallback(
        inference_checkpoint: Path, tmp_path: Path) -> None:
    archive = tmp_path / "submission.tar.gz"
    build_submission(inference_checkpoint, archive, label="clean-import")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    extract_fresh(archive, extracted)
    code = r'''
import json, sys
from pathlib import Path
from kaggle_environments.agent import get_last_callable
root = Path(sys.argv[1]).resolve()
main_path = root / "main.py"
candidate = get_last_callable(main_path.read_text(encoding="utf-8"), path=str(main_path))
import rl_manager
assert Path(rl_manager.__file__).resolve().is_relative_to(root)
assert "jax" not in sys.modules
from opening_book.trace import action_for, load_built_in_trace
trace = load_built_in_trace("standard_mixed")
expected = action_for(trace, 0, 0)
hands = len(expected["hands"])
farm = {"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
        "farmer": [0, 0], "hands": [[i, 0] for i in range(hands)],
        "unlocked_quadrants": ["NW"], "hires_today": 0}
obs = {"player": 0, "day": 0, "hour": 0, "step": 0,
       "farms": [farm, farm], "market": {"inventory": {}, "prices": {}},
       "town": {"unlocked_shops": []},
       "private": {"shed": {}, "seeds": {}, "inventories": []}}
actual = candidate(obs)
assert actual == expected
json.dumps(actual, allow_nan=False)
assert "jax" not in sys.modules
assert not any(name == "optax" or name.startswith("optax.") for name in sys.modules)
print(json.dumps({"action": actual, "agent_origin": str(Path(rl_manager.__file__).resolve())}))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(extracted)], cwd=extracted,
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout[-1000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert Path(report["agent_origin"]).is_relative_to(extracted)


def test_fresh_archive_runs_day4_inference_without_bc_manager_jax(
        inference_checkpoint: Path, tmp_path: Path) -> None:
    archive = tmp_path / "stage25-no-bc-manager-jax.tar.gz"
    build_submission(inference_checkpoint, archive, label="no-bc-manager-jax")
    extracted = tmp_path / "stage25-no-bc-manager-jax"
    extracted.mkdir()
    members = extract_fresh(archive, extracted)

    assert not any(name.startswith("bc_manager_jax/") for name in members)
    vendored_model = extracted / "rl_manager" / "_submission_bc_manager_jax_model.py"
    assert vendored_model.read_bytes() == (ROOT / "bc_manager_jax" / "model.py").read_bytes()
    manifest = json.loads((extracted / "submission_manifest.json").read_text())
    assert "bc_manager_jax" not in manifest["runtime_packages"]
    assert manifest["vendored_bc_manager_jax_model"]["helper_member"] == (
        "rl_manager/_submission_bc_manager_jax_model.py")

    observation = _observation(4)
    code = r'''
import importlib.abc
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repository_root = Path(sys.argv[2]).resolve()
observation = json.loads(sys.argv[3])
def under(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    return path == parent or parent in path.parents
sys.path[:] = [entry for entry in sys.path if not entry or
               not under(entry, repository_root)]
sys.path.insert(0, str(root))
assert not (root / "bc_manager_jax").exists()
class BlockBcManagerJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "bc_manager_jax" or fullname.startswith("bc_manager_jax."):
            raise ModuleNotFoundError(f"blocked package: {fullname}")
        return None
sys.meta_path.insert(0, BlockBcManagerJax())
from kaggle_environments.agent import get_last_callable
main_path = root / "main.py"
candidate = get_last_callable(main_path.read_text(encoding="utf-8"),
                              path=str(main_path))
assert "bc_manager_jax" not in sys.modules
action = candidate(observation)
runtime_agent = candidate.__globals__["_agent"]
assert runtime_agent.provider._native_policy._loaded
assert runtime_agent.provider._native_policy.mode == "deterministic"
diagnostics = runtime_agent.diagnostics_json()
assert diagnostics["manager_inference_latency_s"]
vendored = sys.modules[
    "rl_manager._submission_bc_manager_jax_model"]
assert Path(vendored.__file__).resolve().is_relative_to(root)
assert not any(name == "bc_manager_jax" or
               name.startswith("bc_manager_jax.") for name in sys.modules)
print(json.dumps({"action": action, "diagnostics": diagnostics,
                  "checkpoint_loaded": runtime_agent.provider._native_policy._loaded,
                  "bc_manager_jax_modules": sorted(
                      name for name in sys.modules
                      if name == "bc_manager_jax" or
                      name.startswith("bc_manager_jax."))}))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(extracted), str(ROOT),
         json.dumps(observation)],
        cwd=extracted, env=environment, capture_output=True, text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["checkpoint_loaded"] is True
    assert report["diagnostics"]["manager_inference_latency_s"]
    assert report["bc_manager_jax_modules"] == []


@pytest.mark.parametrize("seat", [0, 1])
def test_opening_handoff_and_realized_labor_rollover(seat: int,
                                                      tmp_path: Path) -> None:
    agent = Stage25SubmissionAgent(tmp_path / "lazy-checkpoint.npz", seat=seat)
    trace = load_built_in_trace("standard_mixed")
    captured: list[dict] = []

    class ProviderSpy:
        def daily_plan(self, observation, observed_seat, previous_execution):
            assert int(observation["day"]) == 4
            assert observed_seat == seat
            captured.append(dict(previous_execution))
            return object()

        def diagnostics_json(self):
            return {}

    class ControllerSpy:
        def act(self, observation, plan):
            class Result:
                @staticmethod
                def action_dict():
                    return {"farmer": ["PASS"], "hands": [], "market": []}
            return Result()

    agent.provider = ProviderSpy()
    agent.controller = ControllerSpy()
    for day in range(4):
        for hour in range(24):
            expected = action_for(trace, day, hour)
            hires = (0 if day < 3 else 1 if hour >= 2 else 0)
            if day == 3 and hour >= 5:
                hires = 2
            obs = _observation(
                day, hour, seat=seat,
                hands=len(expected["hands"]), hires_today=hires)
            obs["manager_intents"] = {"workers_hired": 999}
            action = agent(obs)
            assert action == expected
    assert captured == []

    handoff = agent(_observation(4, 0, seat=seat))
    assert handoff == {"farmer": ["PASS"], "hands": [], "market": []}
    assert captured == [{
        "workers_hired": 2,
        "hire_cost": total_hire_cost(2),
    }]
    opening = agent.opening.diagnostics_json()
    assert opening["turns_replayed"] == 96
    assert opening["handoff"]["clean_d4h0_handoff"] is True
    assert opening["divergence"]["occurred"] is False


def test_labor_tracker_uses_max_observed_hires_not_intents() -> None:
    tracker = RealizedLaborTracker(seat=1)
    first = {"day": 2, "farms": [_farm(), _farm(hires_today=1)],
             "hire_intents": 100}
    second = {"day": 2, "farms": [_farm(), _farm(hires_today=3)],
              "hire_intents": 0}
    rollover = {"day": 3, "farms": [_farm(), _farm(hires_today=0)],
                "hire_intents": 99}
    tracker.observe(first)
    tracker.observe(second)
    tracker.observe(rollover)
    assert tracker.previous_execution == {
        "workers_hired": 3, "hire_cost": total_hire_cost(3),
    }


def test_submission_reuses_corrected_e_history_provider_semantics() -> None:
    class FixedPolicy:
        def act(self, inputs, context, *, row_id, mode, seed):
            assert mode == "deterministic"
            return HOLD

    provider = Stage25PlanProvider(
        "submission-test", 0, 4, native_policy=FixedPolicy(),
        mode="deterministic")
    provider.daily_plan(_observation(4, money=1000.0), 0,
                        {"workers_hired": 0, "hire_cost": 0})
    provider.daily_plan(_observation(5, money=1250.0), 0,
                        {"workers_hired": 2,
                         "hire_cost": total_hire_cost(2)})
    assert provider.e_history_version == E_HISTORY_CORRECTED_V1
    encoded = provider.encoded_inputs
    assert encoded is not None
    assert encoded["economic_context"][0, 13] == 1.0
    assert encoded["economic_context"][0, 12] > 0.0


def test_official_observation_adapter_matches_closed_loop_exactly() -> None:
    from oracle.closed_loop import _executor_observation

    official = _observation(5, 7, seat=1)
    official["private"] = {
        "shed": {"EGG": 3}, "seeds": {"WHEAT": 2},
        "inventories": [{"MILK": 1}],
    }
    assert canonicalize_official_observation(official) == \
        _executor_observation(official, from_fast=False)
    fast = _observation(5, 7, seat=1)
    fast["private"] = {
        "shed": {"EGG": 3}, "seeds": {"WHEAT": 2},
        "inventories": [{"MILK": 1}],
    }
    fast["farms"][1]["tiles"][0][0]["age"] = 123
    animal = fast["farms"][1]["tiles"][0][1]
    animal.pop("placed_day")
    animal["age"] = 5
    assert executor_observation(fast, from_fast=True) == \
        _executor_observation(fast, from_fast=True)


def test_canonical_strip_profile_and_vendored_lazy_market_pricing() -> None:
    from executor_v0 import strip_market
    from executor_v0.strip_executor import StripExecutorConfig
    from fast_env.market import market_price
    from rl_manager.executor_factory import make_stage25_executor_factory

    factory = make_stage25_executor_factory()
    assert isinstance(factory.strip_config, StripExecutorConfig)
    assert factory.effective_profile["name"] == "stage25_strip_executor"
    assert factory.effective_profile["aggressive_sell_all"] is True
    assert factory.strip_config.aggressive_sell_all is True
    # This follows the executor module's imported pricing function, which the
    # archive builder rewrites to executor_v0._submission_market.
    assert strip_market.market_price("WHEAT", 10_000) == market_price(
        "WHEAT", 10_000)


def test_submission_startup_has_no_training_only_imports() -> None:
    code = """
import sys
import rl_manager.stage25_submission
assert 'jax' not in sys.modules
assert 'torch' not in sys.modules
assert 'optax' not in sys.modules
assert not any(name.startswith('bc_manager_jax.train') for name in sys.modules)
assert not any(name.startswith('rl_manager.ppo') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def _write_minimal_verifier_archive(path: Path) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in (
                ("main.py", b"def agent(obs): return {}\n"),
                ("stage25.npz", b"test-checkpoint"),
                ("submission_manifest.json", b"{}\n")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _fake_child_report(parity_game: dict, *, archive: bool) -> dict:
    report = {
        "checkpoint_sha256": parity_game["checkpoint_sha256"],
        "parity_game": dict(parity_game),
        "runtime_import_origins": {},
        "jax_runtime": {"import_success": True},
        "official_engine_version": "1.32.7",
        "checkpoint_metadata": {},
        "training_only_modules_imported": [],
        "vendored_market_price_probe": 25 if archive else None,
        "games": [{"manager_decision_reached": True}],
        "bc_manager_jax_import_blocked": archive,
        "bc_manager_jax_modules_loaded": [],
        "manifest": {},
    }
    if archive:
        # A stale self-comparison from the archive process must never stand in
        # for the independently executed source report.
        report["source_vs_archive_trace_parity"] = {
            "exact_match": True,
            "source_action_trace_sha256": parity_game["action_trace_sha256"],
            "archive_action_trace_sha256": parity_game["action_trace_sha256"],
        }
    return report


def _install_fake_verifier_children(monkeypatch, archive_game: dict,
                                    source_game: dict) -> list[str]:
    from tools import verify_stage25_submission as verifier

    reports = {
        "archive": _fake_child_report(archive_game, archive=True),
        "source": _fake_child_report(source_game, archive=False),
    }
    modes: list[str] = []

    def run_child(_root, _repository_root, *, mode, **_kwargs):
        modes.append(mode)
        return reports[mode]

    monkeypatch.setattr(verifier, "_run_child", run_child)
    return modes


def test_verifier_rejects_mismatched_source_archive_trace(
        tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "mismatch.tar.gz"
    _write_minimal_verifier_archive(archive_path)
    checkpoint_hash = "c" * 64
    archive_game = {
        "seed": 7, "seat": 0, "checkpoint_sha256": checkpoint_hash,
        "action_trace_sha256": "a" * 64,
    }
    source_game = {
        **archive_game,
        "action_trace_sha256": "b" * 64,
    }
    modes = _install_fake_verifier_children(
        monkeypatch, archive_game, source_game)

    with pytest.raises(VerificationError, match="source/archive parity mismatch"):
        verify_archive(archive_path, repository_root=tmp_path, seeds=(7,))

    assert modes == ["archive", "source"]


def test_verifier_accepts_exact_source_archive_trace_parity(
        tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "exact.tar.gz"
    _write_minimal_verifier_archive(archive_path)
    parity_game = {
        "seed": 7, "seat": 0, "checkpoint_sha256": "c" * 64,
        "action_trace_sha256": "a" * 64,
    }
    modes = _install_fake_verifier_children(
        monkeypatch, parity_game, parity_game)

    report = verify_archive(
        archive_path, repository_root=tmp_path, seeds=(7,))

    assert modes == ["archive", "source"]
    assert report["source_vs_archive_trace_parity"] == {
        "seed": 7,
        "seat": 0,
        "checkpoint_sha256": "c" * 64,
        "source_action_trace_sha256": "a" * 64,
        "archive_action_trace_sha256": "a" * 64,
        "exact_match": True,
    }
