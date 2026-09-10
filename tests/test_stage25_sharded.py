"""Focused, checkpoint-free tests for the Stage 2.5 sharding wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_stage25_upkeep_sharded as sharded


SEEDS = list(range(16))


def _rows_for_filters(command: list[str]) -> list[dict]:
    variants = command[command.index("--variants") + 1:command.index("--backend")]
    seeds_start = command.index("--seeds") + 1
    seeds_end = command.index("--master-seed")
    seeds = [int(value) for value in command[seeds_start:seeds_end]]
    master = int(command[command.index("--master-seed") + 1])
    filters_start = command.index("--game-filter") + 1
    filters = command[filters_start:]
    rows = []
    for variant in variants:
        for value in filters:
            seed_text, seat_text = value.split(":")
            seed, seat = int(seed_text), int(seat_text)
            index = seeds.index(seed)
            episode = sharded.episode_id_for(master, len(seeds), index, seat)
            rows.append({
                "variant": variant, "seed": seed, "seat": seat,
                "episode_id": episode, "bank": float(1000 + episode),
                "opponent_bank": 1000.0, "margin": float(episode),
                "statuses": ["DONE", "DONE"],
                "telemetry": {"scheduler": {"runtime_ms": 2.0},
                              "missed_maintenance": ["WATER:0,0"]},
            })
    return rows


class _FakeProcess:
    def __init__(self, output_dir: Path, command: list[str], returncode: int = 0):
        self.output_dir = output_dir
        self.command = command
        self.returncode = returncode

    def wait(self):
        if self.returncode == 0:
            self.output_dir.mkdir(parents=True)
            with (self.output_dir / "games.jsonl").open("w") as stream:
                for row in _rows_for_filters(self.command):
                    stream.write(json.dumps(row) + "\n")
        return self.returncode


def test_identity_formula_and_full_seed_list_preflight(tmp_path: Path):
    output = tmp_path / "preflight"
    result = sharded.run_sharded(
        checkpoint=None, e_checkpoint=None, seeds=SEEDS, output_dir=output,
        backend="fast", preflight_only=True,
    )
    assert result["status"] == "preflight_only"
    payload = json.loads((output / "preflight.json").read_text())
    assert payload["ordered_seeds"] == SEEDS
    assert payload["coverage"]["expected_identities"] == 16 * 2 * 6
    assert result["config"]["starvation_workload_visibility_repair"] is False
    assert result["planned_games"] == 32
    assert payload["expected_episode_ids"][0] == sharded.episode_id_for(25, 16, 0, 0)
    assert payload["expected_episode_ids"][-1] == sharded.episode_id_for(25, 16, 15, 1)


def test_four_shards_are_deterministic_nonoverlapping_and_keep_seats_together():
    pairs = [(index, seat) for index in range(16) for seat in (0, 1)]
    first = sharded.shard_game_pairs(pairs, 4)
    second = sharded.shard_game_pairs(pairs, 4)
    assert first == second
    assert all(len(shard.games) == 8 for shard in first)
    assert sorted(pair for shard in first for pair in shard.games) == pairs
    for shard in first:
        for index in {index for index, _ in shard.games}:
            assert {seat for candidate, seat in shard.games if candidate == index} == {0, 1}


def test_child_command_propagates_executor_controls():
    command = sharded._child_command(
        checkpoint=Path("ppo"), e_checkpoint=Path("e"),
        output_dir=Path("out"), capture_dir=None, seeds=SEEDS,
        master_seed=25, variants=["baseline"], backend="fast",
        e_history_version="E_LEGACY", game_pairs=[(0, 0)],
        underfoot_first=True, deadline_safe_planting=True,
        deadline_safe_hiring=True, persistent_worker_queues=True,
        queue_ownership_repair=True,
        schedule_informed_hiring=True,
        starvation_workload_visibility_repair=True)
    for flag in (
            "--underfoot-first", "--deadline-safe-planting",
            "--deadline-safe-hiring", "--persistent-worker-queues",
            "--queue-ownership-repair",
            "--schedule-informed-hiring",
            "--starvation-workload-visibility-repair"):
        assert flag in command


def test_queue_repair_is_ignored_without_persistent_queues():
    command = sharded._child_command(
        checkpoint=Path("ppo"), e_checkpoint=Path("e"),
        output_dir=Path("out"), capture_dir=None, seeds=SEEDS,
        master_seed=25, variants=["baseline"], backend="fast",
        e_history_version="E_LEGACY", game_pairs=[(0, 0)],
        queue_ownership_repair=True)
    assert "--queue-ownership-repair" not in command
    assert "--starvation-workload-visibility-repair" not in command


def test_preflight_manifest_records_normalized_starvation_visibility(tmp_path: Path):
    output = tmp_path / "preflight"
    result = sharded.run_sharded(
        checkpoint=None, e_checkpoint=None, seeds=SEEDS, output_dir=output,
        backend="fast", preflight_only=True,
        starvation_workload_visibility_repair=True,
    )
    assert result["config"]["starvation_workload_visibility_repair"] is True
    assert result["config"]["schedule_informed_hiring"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["starvation_workload_visibility_repair"] is True


def test_fake_children_merge_canonically_and_emit_pair_bootstrap(tmp_path: Path):
    checkpoint = tmp_path / "ppo.ckpt"
    e_checkpoint = tmp_path / "e.ckpt"
    checkpoint.write_bytes(b"ppo")
    e_checkpoint.write_bytes(b"e")
    output = tmp_path / "run"

    def popen(command, **kwargs):
        del kwargs
        child_output = Path(command[command.index("--output-dir") + 1])
        return _FakeProcess(child_output, command)

    result = sharded.run_sharded(
        checkpoint=checkpoint, e_checkpoint=e_checkpoint, seeds=SEEDS,
        variants=["baseline"], output_dir=output, backend="fast",
        popen_factory=popen,
    )
    assert result["status"] == "complete"
    rows = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
    assert [(row["seed"], row["seat"]) for row in rows] == [
        (seed, seat) for seed in SEEDS for seat in (0, 1)
    ]
    groups = json.loads((output / "bootstrap_groups.json").read_text())["baseline"]
    assert all([item["seats"][0]["seat"], item["seats"][1]["seat"]] == [0, 1] for item in groups)
    telemetry = json.loads((output / "telemetry.json").read_text())["fields"]
    assert telemetry["scheduler_runtime"]["sum"] == 64.0
    assert telemetry["hiring_cost"]["available"] is False

    manifests = list((output / "shards").glob("shard_*.manifest.json"))
    assert len(manifests) == 4
    for path in manifests:
        shard_manifest = json.loads(path.read_text())
        assert shard_manifest["ordered_seeds"] == SEEDS
        assert set(("config", "source", "patch", "checkpoints", "engine")) <= set(shard_manifest)


def test_child_failure_preserves_partial_outputs(tmp_path: Path):
    checkpoint = tmp_path / "ppo.ckpt"
    e_checkpoint = tmp_path / "e.ckpt"
    checkpoint.write_bytes(b"ppo")
    e_checkpoint.write_bytes(b"e")
    output = tmp_path / "failed"
    calls = []

    def popen(command, **kwargs):
        del kwargs
        calls.append(command)
        child_output = Path(command[command.index("--output-dir") + 1])
        return _FakeProcess(child_output, command, returncode=7 if len(calls) == 2 else 0)

    with pytest.raises(RuntimeError, match="partial outputs preserved"):
        sharded.run_sharded(
            checkpoint=checkpoint, e_checkpoint=e_checkpoint, seeds=SEEDS,
            variants=["baseline"], output_dir=output, backend="fast",
            popen_factory=popen,
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "child_failed"
    assert (output / "shards" / "shard_1.manifest.json").is_file()


def test_resume_reuses_validated_completed_shards(tmp_path: Path):
    checkpoint = tmp_path / "ppo.ckpt"
    e_checkpoint = tmp_path / "e.ckpt"
    checkpoint.write_bytes(b"ppo")
    e_checkpoint.write_bytes(b"e")
    output = tmp_path / "resume"

    def popen(command, **kwargs):
        del kwargs
        child_output = Path(command[command.index("--output-dir") + 1])
        return _FakeProcess(child_output, command)

    first = sharded.run_sharded(
        checkpoint=checkpoint, e_checkpoint=e_checkpoint, seeds=SEEDS,
        variants=["baseline"], output_dir=output, backend="fast",
        popen_factory=popen,
    )
    assert first["status"] == "complete"

    def should_not_launch(*args, **kwargs):
        raise AssertionError("validated shards should be reused")

    resumed = sharded.run_sharded(
        checkpoint=checkpoint, e_checkpoint=e_checkpoint, seeds=SEEDS,
        variants=["baseline"], output_dir=output, backend="fast", resume=True,
        popen_factory=should_not_launch,
    )
    assert resumed["status"] == "complete"
    assert resumed["reused_shards"] == [0, 1, 2, 3]


def test_merge_rejects_duplicate_and_omitted_rows():
    pairs = [(0, 0), (0, 1)]
    row = {"variant": "baseline", "seed": 0, "seat": 0,
           "episode_id": 50, "statuses": ["DONE", "DONE"],
           "bank": 1, "opponent_bank": 0, "margin": 1}
    with pytest.raises(ValueError, match="duplicate"):
        sharded.merge_rows({0: [row], 1: [row]}, seeds=[0], master_seed=25,
                           variants=["baseline"], game_pairs=pairs)
    with pytest.raises(ValueError, match="omitted"):
        sharded.merge_rows({0: [row]}, seeds=[0], master_seed=25,
                           variants=["baseline"], game_pairs=pairs)
