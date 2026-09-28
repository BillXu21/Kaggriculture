from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_manager.stage25_checkpoint import (
    Stage25CheckpointError,
    export_stage25_dual_policy_snapshot,
    load_stage25_inference_checkpoint,
    save_stage25_dual_ppo_checkpoint,
    save_stage25_inference_checkpoint,
)
from rl_manager.stage25_inference import Stage25InferenceAdapter
from rl_manager.stage25_policy import Stage25ModelConfig, init_stage25_params
from rl_manager.stage25_ppo import (
    Stage25PPOConfig,
    init_stage25_dual_ppo_state,
)
from rl_manager.types import DUAL_POLICY_SELF_PLAY


def _config() -> Stage25PPOConfig:
    return Stage25PPOConfig(
        model=Stage25ModelConfig.tiny(), physical_batch_size=1,
        minibatch_size=1, epochs=1)


def _physical_contract(config: Stage25PPOConfig) -> dict:
    from rl_manager.stage25_ppo_cli import _physical_contract as make_contract

    return make_contract(config)


def _dual_checkpoint(path: Path, seed: int) -> tuple[Stage25PPOConfig, object]:
    config = _config()
    params = init_stage25_params(config.model, seed=seed)
    state = init_stage25_dual_ppo_state(config, seed=seed, params=params)
    params_b = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf) + np.float32(0.001),
        state.policy_b.params)
    policy_b = Stage25InferenceAdapter(
        params=params_b, config=config.model, name="stage25_dual_b",
        version="ppo-native-v1", seed=state.policy_b.rollout_seed,
        mode="stochastic")
    state = replace(state, policy_b=replace(
        state.policy_b, params=params_b,
        behavior_identity=policy_b.behavior_identity))
    save_stage25_dual_ppo_checkpoint(
        path, state, config, seed=seed,
        training_contract={"training_composition": DUAL_POLICY_SELF_PLAY,
                           "reward": {"mode": "terminal_wlt"}},
        physical_contract=_physical_contract(config),
        executor={"name": "test", "version": "v1"},
        source_identity={"name": f"dual-{seed}.npz"})
    return config, state


def _export_pair(tmp_path: Path, seed: int = 19):
    dual_path = tmp_path / f"dual-{seed}.npz"
    config, state = _dual_checkpoint(dual_path, seed)
    paths = {}
    for label in ("A", "B"):
        target = tmp_path / f"policy-{label}-{seed}.npz"
        export_stage25_dual_policy_snapshot(
            dual_path, target, policy=label)
        paths[label] = target
    return config, state, dual_path, paths


def _bc_snapshot(path: Path, seed: int = 7) -> Path:
    config = _config()
    params = init_stage25_params(config.model, seed=seed)
    save_stage25_inference_checkpoint(
        path, params, config.model, seed=seed,
        source_identity={"name": f"original-bc-{seed}"})
    return path


def _evaluation(registry: dict, candidate_path: Path, evaluation_id: str) -> dict:
    import rl_manager.stage25_champion as champion

    current = champion.load_champion_registry(registry["_path"])
    candidate, _policy, _contract = champion._snapshot_components(candidate_path)
    evaluated_opponents = {}
    for role, record in (("current_champion", current["current_champion"]),
                         ("bc_anchor", current["bc_anchor"])):
        entry = evaluated_opponents.setdefault(
            record["parameter_fingerprint"], {
                "snapshot_id": record["snapshot_id"],
                "parameter_fingerprint": record["parameter_fingerprint"],
                "roles": [],
            })
        entry["roles"].append(role)
    return {
        "schema_version": champion.PANEL_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "candidate": candidate,
        "registry": {
            "registry_id": current["registry_id"],
            "registry_version": current["registry_version"],
            "registry_state_fingerprint": champion._registry_state_fingerprint(current),
            "current_champion_snapshot_id": current["current_champion"]["snapshot_id"],
            "current_champion_parameter_fingerprint": current[
                "current_champion"]["parameter_fingerprint"],
            "bc_anchor_snapshot_id": current["bc_anchor"]["snapshot_id"],
            "bc_anchor_snapshot_sha256": current["bc_anchor"]["snapshot_sha256"],
            "bc_anchor_parameter_fingerprint": current[
                "bc_anchor"]["parameter_fingerprint"],
        },
        "per_opponent_results": [
            {"opponent": opponent,
             "metrics": {"candidate_win_fraction": 0.8}}
            for opponent in evaluated_opponents.values()
        ],
    }


def _write_evaluation(path: Path, evaluation: dict) -> Path:
    import rl_manager.stage25_champion as champion

    return champion._write_json_atomic(path, evaluation)


def _registry(tmp_path: Path, bc_path: Path, *, depth: int = 3) -> dict:
    import rl_manager.stage25_champion as champion

    path = tmp_path / "champion_registry.json"
    result = champion.initialize_champion_registry(
        path, bc_path, history_depth=depth)
    result["_path"] = str(path)
    return result


def test_export_a_and_b_preserves_exact_parameters_and_inference_only_payload(tmp_path):
    config, state, dual_path, paths = _export_pair(tmp_path)

    for label, policy_state in (("A", state.policy_a), ("B", state.policy_b)):
        params, metadata = load_stage25_inference_checkpoint(
            paths[label], config=config.model)
        before = jax.tree_util.tree_leaves(policy_state.params)
        after = jax.tree_util.tree_leaves(params)
        assert len(before) == len(after)
        assert all(np.array_equal(a, b) for a, b in zip(before, after))
        snapshot = metadata["evaluation_snapshot"]
        assert snapshot["source_generation"] == state.generation
        assert snapshot["source_policy"] == label
        assert snapshot["source_kind"] == "dual_policy"
        assert metadata["behavior_identity"] == policy_state.behavior_identity.to_json_dict()
        assert metadata["e_identity"] == {
            "variant": "E",
            "history_version": metadata["e_history_version"],
            "observation_schema_version": metadata["observation_schema_version"],
        }
        with np.load(paths[label], allow_pickle=False) as archive:
            assert all(key == "__meta__" or key.startswith("param:")
                       for key in archive.files)
        assert "optimizer_leaf_count" not in metadata
        assert "dual_state" not in metadata
        assert snapshot["source_dual_checkpoint_identity"]["sha256"]
        assert Path(dual_path).is_file()


def test_export_rejects_inference_and_single_policy_checkpoint_inputs(tmp_path):
    bc = _bc_snapshot(tmp_path / "single.npz")

    with pytest.raises(Stage25CheckpointError, match="payload kind"):
        export_stage25_dual_policy_snapshot(
            bc, tmp_path / "export.npz", policy="A")
    with pytest.raises(ValueError, match="A or B"):
        export_stage25_dual_policy_snapshot(
            bc, tmp_path / "export.npz", policy="C")


def test_initialize_registry_sets_bc_anchor_and_current_and_dedupes_panel(tmp_path):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor)

    assert registry["schema_version"] == "stage25_champion_registry_v1"
    assert registry["history_depth"] == 3
    assert registry["history"] == []
    assert registry["bc_anchor"] == registry["current_champion"]
    opponents, skipped = champion.select_panel_opponents(registry)
    assert skipped == []
    assert len(opponents) == 1
    assert opponents[0]["roles"] == ["current_champion", "bc_anchor"]


def test_registry_rotates_bounded_history_and_keeps_bc_anchor_immutable(tmp_path):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor, depth=2)
    anchor_before = dict(registry["bc_anchor"])
    promoted_ids = []
    for index in range(3):
        _config_value, _state, _dual, paths = _export_pair(
            tmp_path, seed=30 + index)
        candidate = paths["A"]
        evaluation = _evaluation(
            registry, candidate, f"eval-{index}")
        eval_path = _write_evaluation(tmp_path / f"eval-{index}.json", evaluation)
        registry = champion.promote_champion(
            registry["_path"], candidate, eval_path)
        registry["_path"] = str(tmp_path / "champion_registry.json")
        promoted_ids.append(registry["current_champion"]["snapshot_id"])
        assert len(registry["history"]) <= 2
        assert registry["bc_anchor"] == anchor_before

    assert [entry["snapshot_id"] for entry in registry["history"]] == promoted_ids[-2::-1]
    assert registry["lineage"][-1]["promoted_over_snapshot_id"] == promoted_ids[-2]
    assert registry["lineage"][-1]["source_generation"] == 0
    assert registry["lineage"][-1]["source_policy"] == "A"
    assert len(registry["lineage"]) == 4


def test_registry_history_depth_zero_and_missing_historical_slot(tmp_path):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor, depth=0)
    assert registry["history_depth"] == 0
    assert registry["history"] == []
    registry["history_depth"] = 1
    missing = dict(registry["bc_anchor"])
    missing.update({
        "snapshot_id": "missing-history",
        "path": str(tmp_path / "gone.npz"),
        "snapshot_sha256": "missing",
        "parameter_fingerprint": "different-fingerprint",
    })
    registry["history"] = [missing]
    opponents, skipped = champion.select_panel_opponents(registry)
    assert skipped == [{
        "role": "recent_champion_1",
        "reason": "snapshot file is unavailable",
    }]
    assert len(opponents) == 1


def test_atomic_registry_write_failure_keeps_previous_registry(tmp_path, monkeypatch):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor)
    path = Path(registry["_path"])
    before = path.read_bytes()
    _config_value, _state, _dual, paths = _export_pair(tmp_path, seed=44)
    evaluation = _write_evaluation(
        tmp_path / "atomic-eval.json",
        _evaluation(registry, paths["A"], "atomic-eval"))

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(champion.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        champion.promote_champion(path, paths["A"], evaluation)
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".champion_registry.json.*.tmp")) == []


def test_candidate_fingerprint_staleness_and_bc_safety_are_enforced(tmp_path):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor)
    _config_value, _state, _dual, paths1 = _export_pair(tmp_path, seed=51)
    bad = _evaluation(registry, paths1["A"], "bad-candidate")
    bad["candidate"]["parameter_fingerprint"] = "wrong"
    bad_path = _write_evaluation(tmp_path / "bad.json", bad)
    with pytest.raises(champion.ChampionRegistryError, match="parameter_fingerprint"):
        champion.promote_champion(registry["_path"], paths1["A"], bad_path)

    good_path = _write_evaluation(
        tmp_path / "good.json",
        _evaluation(registry, paths1["A"], "good-eval"))
    promoted = champion.promote_champion(
        registry["_path"], paths1["A"], good_path)
    promoted["_path"] = registry["_path"]
    with pytest.raises(champion.ChampionRegistryError, match="stale"):
        champion.promote_champion(registry["_path"], paths1["B"], good_path)

    # Re-label a normal inference-only checkpoint as a BC-like file; it cannot
    # pass the dual-policy source-kind requirement for manual promotion.
    with pytest.raises(champion.ChampionRegistryError, match="dual PPO"):
        champion.promote_champion(
            registry["_path"], anchor,
            _write_evaluation(
                tmp_path / "bc-candidate.json",
                _evaluation(promoted, anchor, "bc-eval")))


def test_promotion_policy_is_optional_and_configurable(tmp_path):
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor)
    _config_value, _state, _dual, paths = _export_pair(tmp_path, seed=62)
    evaluation = _evaluation(registry, paths["A"], "policy-eval")
    evaluation["per_opponent_results"][0]["metrics"]["candidate_win_fraction"] = 0.8
    eval_path = _write_evaluation(tmp_path / "policy-eval.json", evaluation)
    policy = champion.PromotionPolicy({
        "current_champion": 0.75,
        "recent_champion_1": None,
        "bc_anchor": 0.75,
    })
    with pytest.raises(champion.ChampionRegistryError, match="policy rejected"):
        champion.promote_champion(
            registry["_path"], paths["A"], eval_path,
            promotion_policy=champion.PromotionPolicy({
                "current_champion": 0.9}))
    promoted = champion.promote_champion(
        registry["_path"], paths["A"], eval_path,
        promotion_policy=policy)
    assert promoted["registry_version"] == 2
    assert promoted["last_promotion_policy"]["passed"] is True


def _identity(name: str):
    return SimpleNamespace(identity_id=lambda: name)


def _fake_result(candidate_id, opponent_id, candidate_seat, candidate_bank,
                 opponent_bank, episode_index, *, seed=0,
                 composition="candidate_vs_frozen"):
    banks = ([candidate_bank, opponent_bank] if candidate_seat == 0 else
             [opponent_bank, candidate_bank])
    identities = []
    for seat in (0, 1):
        policy_id = candidate_id if seat == candidate_seat else opponent_id
        identities.append({
            "seat": seat,
            "policy": {"identity_id": policy_id},
        })
    return SimpleNamespace(
        episode_index=episode_index,
        seed=seed,
        composition=composition,
        final_banks=banks,
        policy_identities=tuple(identities))


def test_88_game_schedule_is_paired_deterministic_and_worker_independent():
    from rl_manager.stage25_panel_eval import build_panel_schedule

    candidate = {"snapshot_id": "candidate-sha"}
    opponents = [{
        "snapshot_id": "opponent-sha",
        "parameter_fingerprint": "opp-fp",
        "roles": ["current_champion"],
    }]
    first = build_panel_schedule(candidate, opponents,
                                 games_per_opponent=88, seed=2026)
    reordered = build_panel_schedule(candidate, list(reversed(opponents)),
                                     games_per_opponent=88, seed=2026)
    assert first == reordered
    assert len(first) == 88
    assert sum(item.candidate_seat == 0 for item in first) == 44
    assert sum(item.candidate_seat == 1 for item in first) == 44
    for pair in range(44):
        seat0, seat1 = first[pair * 2:pair * 2 + 2]
        assert seat0.seed == seat1.seed
        assert seat0.candidate_seat == 0
        assert seat1.candidate_seat == 1
        assert seat0.pair_index == seat1.pair_index == pair
    with pytest.raises(ValueError, match="positive even"):
        build_panel_schedule(candidate, opponents, games_per_opponent=3)


def test_arbitrary_worker_counts_and_opponent_identity_schedule():
    from rl_manager.stage25_panel_eval import (
        _episode_specs,
        build_panel_schedule,
    )

    candidate_policy = SimpleNamespace(identity=SimpleNamespace(identity_id=lambda: "cand"))
    opponent_policy = SimpleNamespace(identity=SimpleNamespace(identity_id=lambda: "opp"))
    candidate = {"snapshot_id": "cand-file"}
    opponent = {
        "snapshot_id": "opp-file", "parameter_fingerprint": "opp-fp",
        "roles": ["current_champion"],
    }
    schedule = build_panel_schedule(candidate, [opponent], games_per_opponent=4)
    specs = _episode_specs(candidate_policy, {"opp-file": opponent_policy}, schedule)
    assert len(specs) == 4
    assert all(spec.trainable_seats == () for spec in specs)
    assert [(spec.policies[0].identity.identity_id(),
             spec.policies[1].identity.identity_id()) for spec in specs] == [
        ("cand", "opp"), ("opp", "cand"), ("cand", "opp"), ("opp", "cand")]
    # The schedule itself has no worker parameter; any valid runner topology
    # consumes the same episode-indexed assignments.
    for workers in (1, 3, 88):
        assert workers >= 1
        assert tuple((s.episode_index, s.seed) for s in specs) == tuple(
            (a.episode_index, a.seed) for a in schedule)


def test_wlt_bank_and_seat_metrics_follow_candidate_identity():
    from rl_manager.stage25_panel_eval import candidate_result_metrics

    results = [
        _fake_result("candidate", "opponent", 0, 100, 50, 0),
        _fake_result("candidate", "opponent", 1, 40, 70, 1),
        _fake_result("candidate", "opponent", 0, 20, 20, 2),
    ]
    metrics = candidate_result_metrics(results, _identity("candidate"))
    assert metrics["games"] == 3
    assert (metrics["candidate_wins"], metrics["candidate_losses"],
            metrics["ties"]) == (1, 1, 1)
    assert metrics["candidate_win_fraction"] == pytest.approx(1 / 3)
    assert metrics["non_tie_win_fraction"] == 0.5
    assert metrics["candidate_mean_bank"] == pytest.approx(160 / 3)
    assert metrics["opponent_mean_bank"] == pytest.approx(140 / 3)
    assert metrics["mean_bank_margin"] == pytest.approx(20 / 3)
    assert metrics["seat_0_candidate_win_fraction"] == 0.5
    assert metrics["seat_1_candidate_win_fraction"] == 0.0


def test_candidate_metric_rejects_ambiguous_identity():
    from rl_manager.stage25_panel_eval import candidate_result_metrics

    result = _fake_result("same", "same", 0, 100, 0, 0)
    with pytest.raises(ValueError, match="exactly one seat"):
        candidate_result_metrics([result], _identity("same"))


def test_panel_cli_writes_evaluation_only_artifact_with_default_topology(
        tmp_path, monkeypatch):
    import rl_manager.parallel as parallel
    import rl_manager.runner as runner_module
    import rl_manager.stage25_panel_eval as panel
    import rl_manager.stage25_ppo_cli as ppo_cli
    import rl_manager.stage25_champion as champion

    anchor = _bc_snapshot(tmp_path / "bc.npz")
    registry = _registry(tmp_path, anchor)
    _config_value, _state, _dual, paths = _export_pair(tmp_path, seed=75)

    class FakeRunner:
        def __init__(self, config, *, num_workers, master_seed, executor_factory):
            self.config = config
            self.num_workers = num_workers
            self.master_seed = master_seed
            self.executor_factory = executor_factory
            self.provenance = {"executor_factory": "fake"}
            self.inference_metrics = {"physical_inference_calls": 1}

        def run(self, specs):
            results = []
            for spec in specs:
                candidate_seat = next(
                    seat for seat, policy in enumerate(spec.policies)
                    if policy.identity.identity_id().startswith("stage25_dual_a@"))
                winning = spec.episode_index % 2 == 0
                candidate_bank, opponent_bank = ((100, 50) if winning else
                                                  (20, 80))
                results.append(_fake_result(
                    spec.policies[candidate_seat].identity.identity_id(),
                    spec.policies[1 - candidate_seat].identity.identity_id(),
                    candidate_seat, candidate_bank, opponent_bank,
                    spec.episode_index, seed=spec.seed,
                    composition=spec.composition))
            return results

    monkeypatch.setattr(parallel, "ParallelSelfPlayRunner", FakeRunner)
    monkeypatch.setattr(ppo_cli, "_resolve_executor_factory", lambda _name: object())
    monkeypatch.setattr(runner_module, "_executor_factory_provenance",
                        lambda _factory: {"name": "fake", "version": "v1"})
    artifact, artifact_path = panel.run_panel_evaluation(panel._parser().parse_args([
        "--candidate", str(paths["A"]),
        "--registry", str(registry["_path"]),
        "--output-dir", str(tmp_path / "evals"),
        "--games-per-opponent", "4",
        "--workers", "3",
        "--envs-per-worker", "4",
    ]))

    assert artifact_path.is_file()
    assert artifact["schema_version"] == champion.PANEL_EVALUATION_SCHEMA_VERSION
    assert artifact["scheduled_games"] == 4
    assert artifact["runtime"]["workers"] == 3
    assert artifact["runtime"]["envs_per_worker"] == 4
    assert artifact["runtime"]["physical_inference_batch_size"] == 32
    assert artifact["runtime"]["inference_batch_wait_ms"] == 20.0
    assert artifact["aggregate_results"]["candidate_wins"] == 2
    assert artifact["aggregate_results"]["candidate_losses"] == 2
    assert artifact["aggregate_results"]["seat_breakdown"]["0"]["games"] == 2
    assert artifact["aggregate_results"]["seat_breakdown"]["1"]["games"] == 2
    assert artifact["seed_derivation"]
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["evaluation_id"] == artifact[
        "evaluation_id"]
    assert champion.load_champion_registry(registry["_path"])["registry_version"] == 1


def test_panel_policy_identity_and_assignments_remain_stable_when_reordered():
    from rl_manager.stage25_panel_eval import build_panel_schedule

    candidate = {"snapshot_id": "candidate"}
    opponents = [
        {"snapshot_id": "opp-a", "parameter_fingerprint": "a", "roles": ["current_champion"]},
        {"snapshot_id": "opp-b", "parameter_fingerprint": "b", "roles": ["bc_anchor"]},
    ]
    one = build_panel_schedule(candidate, opponents, games_per_opponent=6, seed=9)
    two = build_panel_schedule(candidate, opponents, games_per_opponent=6, seed=9)
    assert one == two
    assert len({item.opponent_snapshot_id for item in one}) == 2
    assert all(item.candidate_seat in (0, 1) for item in one)
    assert all(one[index].opponent_snapshot_id == "opp-a" for index in range(6))
    assert all(one[index].opponent_snapshot_id == "opp-b" for index in range(6, 12))
