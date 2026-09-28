"""Deterministic, seat-balanced candidate-vs-champion panel evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rl_manager.runner import EpisodeSpec, RunnerConfig


SEED_DERIVATION = (
    "sha256('stage25-panel-v1|' + base_seed + '|' + candidate_snapshot_id + "
    "'|' + opponent_snapshot_id + '|' + pair_index), first 4 bytes little-endian, "
    "reduced modulo 2^31-1; both candidate seat orientations share this seed"
)


@dataclass(frozen=True)
class PanelGameAssignment:
    episode_index: int
    opponent_snapshot_id: str
    opponent_parameter_fingerprint: str
    opponent_roles: tuple[str, ...]
    pair_index: int
    candidate_seat: int
    seed: int
    composition: str


def _assignment_seed(
    base_seed: int, candidate_snapshot_id: str,
    opponent_snapshot_id: str, pair_index: int,
) -> int:
    payload = (f"stage25-panel-v1|{int(base_seed)}|{candidate_snapshot_id}|"
               f"{opponent_snapshot_id}|{int(pair_index)}").encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return value % (2**31 - 1)


def build_panel_schedule(
    candidate: Mapping[str, Any],
    opponents: Sequence[Mapping[str, Any]],
    *,
    games_per_opponent: int = 88,
    seed: int = 0,
) -> tuple[PanelGameAssignment, ...]:
    """Make a stable schedule before worker allocation, in paired orientations."""
    if (isinstance(games_per_opponent, bool)
            or not isinstance(games_per_opponent, int)
            or games_per_opponent < 2 or games_per_opponent % 2):
        raise ValueError("games_per_opponent must be a positive even integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    candidate_id = candidate.get("snapshot_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate snapshot_id is required")
    assignments: list[PanelGameAssignment] = []
    episode_index = 0
    for opponent in opponents:
        opponent_id = opponent.get("snapshot_id")
        fingerprint = opponent.get("parameter_fingerprint")
        roles = opponent.get("roles")
        if not isinstance(opponent_id, str) or not opponent_id:
            raise ValueError("opponent snapshot_id is required")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("opponent parameter_fingerprint is required")
        if not isinstance(roles, Sequence) or isinstance(roles, str):
            raise ValueError("opponent roles must be a sequence")
        pair_seeds = [_assignment_seed(seed, candidate_id, opponent_id, pair)
                      for pair in range(games_per_opponent // 2)]
        if len(set(pair_seeds)) != len(pair_seeds):
            raise RuntimeError("deterministic panel seed collision")
        for pair_index, pair_seed in enumerate(pair_seeds):
            for candidate_seat in (0, 1):
                assignments.append(PanelGameAssignment(
                    episode_index=episode_index,
                    opponent_snapshot_id=opponent_id,
                    opponent_parameter_fingerprint=fingerprint,
                    opponent_roles=tuple(str(role) for role in roles),
                    pair_index=pair_index,
                    candidate_seat=candidate_seat,
                    seed=pair_seed,
                    composition=("candidate_vs_frozen" if candidate_seat == 0
                                 else "frozen_vs_candidate"),
                ))
                episode_index += 1
    return tuple(assignments)


def candidate_result_metrics(results: Sequence[Any], candidate_identity: Any) -> dict[str, Any]:
    """Summarize final-bank W/L/T by candidate policy identity, not seat order."""
    identity_id = candidate_identity.identity_id()
    wins = losses = ties = 0
    candidate_banks: list[float] = []
    opponent_banks: list[float] = []
    margins: list[float] = []
    seats: dict[int, dict[str, Any]] = {
        seat: {"games": 0, "wins": 0, "losses": 0, "ties": 0,
               "candidate_banks": [], "opponent_banks": []}
        for seat in (0, 1)
    }
    for result in results:
        identities = getattr(result, "policy_identities", None)
        if not isinstance(identities, Sequence) or len(identities) != 2:
            raise ValueError("evaluation result must contain two policy identities")
        candidate_seats = [
            int(record["seat"]) for record in identities
            if record.get("policy", {}).get("identity_id") == identity_id
        ]
        if len(candidate_seats) != 1:
            raise ValueError(
                "candidate identity must occupy exactly one seat in each result")
        candidate_seat = candidate_seats[0]
        opponent_seat = 1 - candidate_seat
        opponent_identity = identities[opponent_seat]["policy"].get("identity_id")
        if opponent_identity == identity_id:
            raise ValueError("candidate and opponent identities must be distinct")
        banks = [float(value) for value in result.final_banks]
        if len(banks) != 2 or not all(math.isfinite(value) for value in banks):
            raise ValueError("evaluation result final_banks must be finite [seat0, seat1]")
        candidate_bank = banks[candidate_seat]
        opponent_bank = banks[opponent_seat]
        margin = candidate_bank - opponent_bank
        outcome = "wins" if margin > 0 else "losses" if margin < 0 else "ties"
        if outcome == "wins":
            wins += 1
        elif outcome == "losses":
            losses += 1
        else:
            ties += 1
        candidate_banks.append(candidate_bank)
        opponent_banks.append(opponent_bank)
        margins.append(margin)
        bucket = seats[candidate_seat]
        bucket["games"] += 1
        bucket[outcome] += 1
        bucket["candidate_banks"].append(candidate_bank)
        bucket["opponent_banks"].append(opponent_bank)

    def summarize(wins_count: int, losses_count: int, ties_count: int,
                  candidate_values: Sequence[float],
                  opponent_values: Sequence[float], margin_values: Sequence[float]) -> dict[str, Any]:
        games = wins_count + losses_count + ties_count
        non_ties = wins_count + losses_count
        return {
            "games": games,
            "candidate_wins": wins_count,
            "candidate_losses": losses_count,
            "ties": ties_count,
            "candidate_win_fraction": wins_count / games if games else None,
            "non_tie_win_fraction": (wins_count / non_ties
                                     if non_ties else None),
            "candidate_mean_bank": (math.fsum(candidate_values) / len(candidate_values)
                                    if candidate_values else None),
            "opponent_mean_bank": (math.fsum(opponent_values) / len(opponent_values)
                                   if opponent_values else None),
            "mean_bank_margin": (math.fsum(margin_values) / len(margin_values)
                                 if margin_values else None),
        }

    aggregate = summarize(wins, losses, ties, candidate_banks,
                          opponent_banks, margins)
    seat_metrics: dict[str, Any] = {}
    for seat, bucket in seats.items():
        summary = summarize(
            bucket["wins"], bucket["losses"], bucket["ties"],
            bucket["candidate_banks"], bucket["opponent_banks"],
            [left - right for left, right in zip(
                bucket["candidate_banks"], bucket["opponent_banks"])])
        seat_metrics[str(seat)] = summary
    aggregate["seat_0_candidate_win_fraction"] = seat_metrics["0"][
        "candidate_win_fraction"]
    aggregate["seat_1_candidate_win_fraction"] = seat_metrics["1"][
        "candidate_win_fraction"]
    aggregate["seat_breakdown"] = seat_metrics
    return aggregate


def _episode_specs(
    candidate_policy: Any,
    opponent_policies: Mapping[str, Any],
    assignments: Sequence[PanelGameAssignment],
) -> tuple[EpisodeSpec, ...]:
    specs = []
    for assignment in assignments:
        opponent = opponent_policies[assignment.opponent_snapshot_id]
        policies = ((candidate_policy, opponent) if assignment.candidate_seat == 0
                    else (opponent, candidate_policy))
        specs.append(EpisodeSpec(
            episode_index=assignment.episode_index,
            seed=assignment.seed,
            composition=assignment.composition,
            policies=policies,
            trainable_seats=(),
            controlled_seat=None,
        ))
    return tuple(specs)


def _parser() -> argparse.ArgumentParser:
    from rl_manager.stage25_champion import DEFAULT_PANEL_GAMES_PER_OPPONENT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="native inference snapshot; dual PPO checkpoints must be exported first")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("panel-evaluations"))
    parser.add_argument("--games-per-opponent", type=int,
                        default=DEFAULT_PANEL_GAMES_PER_OPPONENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=("fast", "official"), default="fast")
    parser.add_argument("--executor", choices=("strip", "legacy"), default="strip")
    parser.add_argument("--opening", default="standard_mixed")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--batch-backend", action="store_true")
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=20.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.games_per_opponent < 2 or args.games_per_opponent % 2:
        raise ValueError("games-per-opponent must be a positive even integer")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.workers < 1 or args.envs_per_worker < 1:
        raise ValueError("workers and envs-per-worker must be positive")
    if args.physical_batch_size < 1:
        raise ValueError("physical-batch-size must be positive")
    if not math.isfinite(args.inference_batch_wait_ms) or args.inference_batch_wait_ms < 0:
        raise ValueError("inference-batch-wait-ms must be finite and nonnegative")
    if args.batch_backend and args.engine != "fast":
        raise ValueError("batch-backend requires --engine fast")


def run_panel_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Run one deterministic evaluation and atomically write its JSON artifact."""
    from rl_manager.stage25_champion import (
        PANEL_EVALUATION_SCHEMA_VERSION,
        _digest,
        _git_sha,
        _now,
        _snapshot_components,
        _write_json_atomic,
        load_champion_registry,
        select_panel_opponents,
    )

    _validate_args(args)
    registry = load_champion_registry(args.registry)
    registry_state_fingerprint = _digest(registry)
    opponent_records, skipped = select_panel_opponents(registry)
    candidate, candidate_policy, candidate_contract = _snapshot_components(
        args.candidate)
    opponent_policies: dict[str, Any] = {}
    for record in opponent_records:
        policy_record, policy, contract = _snapshot_components(record["path"])
        if policy_record["snapshot_sha256"] != record["snapshot_sha256"]:
            raise ValueError("registered opponent file identity changed")
        if policy_record["contract_fingerprint"] != candidate["contract_fingerprint"]:
            raise ValueError(
                f"candidate contract is incompatible with opponent roles {record['roles']}")
        if contract != candidate_contract:
            raise ValueError("candidate and opponent evaluation contracts differ")
        if policy.behavior_identity.identity_id() == candidate_policy.behavior_identity.identity_id():
            raise ValueError("candidate and opponent behavior identities must be distinct")
        opponent_policies[record["snapshot_id"]] = policy

    assignments = build_panel_schedule(
        candidate, opponent_records,
        games_per_opponent=args.games_per_opponent, seed=args.seed)
    specs = _episode_specs(candidate_policy, opponent_policies, assignments)
    from rl_manager.parallel import ParallelSelfPlayRunner
    from rl_manager.runner import _executor_factory_provenance
    from rl_manager.stage25_ppo_cli import _resolve_executor_factory

    executor_factory = _resolve_executor_factory(args.executor)
    runner_config = RunnerConfig(
        backend_name=args.engine,
        backend_configuration={"seed": args.seed, "numThreads": 1},
        opening=args.opening,
        num_envs=args.envs_per_worker,
        batch_backend=args.batch_backend,
        low_telemetry=True,
        stage25_enabled=True,
        stage25_mode="deterministic",
        stage25_fixed_inference_batch_size=args.physical_batch_size,
        inference_batch_wait_seconds=args.inference_batch_wait_ms / 1000.0,
    )
    runner = ParallelSelfPlayRunner(
        runner_config, num_workers=args.workers, master_seed=args.seed,
        executor_factory=executor_factory)
    results = runner.run(specs)
    if len(results) != len(assignments):
        raise RuntimeError(
            f"runner returned {len(results)} results for {len(assignments)} assignments")
    assignment_by_episode = {item.episode_index: item for item in assignments}
    results_by_episode = {int(item.episode_index): item for item in results}
    if len(results_by_episode) != len(results):
        raise RuntimeError("runner returned duplicate episode indices")
    if set(results_by_episode) != set(assignment_by_episode):
        raise RuntimeError("runner episode indices do not match the panel schedule")

    per_opponent = []
    all_ordered_results = []
    for opponent in opponent_records:
        group = [results_by_episode[item.episode_index]
                 for item in assignments
                 if item.opponent_snapshot_id == opponent["snapshot_id"]]
        expected_opponent_id = opponent_policies[
            opponent["snapshot_id"]].behavior_identity.identity_id()
        expected_episodes = [item for item in assignments
                             if item.opponent_snapshot_id == opponent["snapshot_id"]]
        if len(group) != len(expected_episodes):
            raise RuntimeError("panel result count does not match its opponent schedule")
        for result, expected in zip(group, expected_episodes):
            if int(result.seed) != expected.seed or result.composition != expected.composition:
                raise RuntimeError("panel result seed or seat assignment changed")
            if getattr(result, "statuses", ["DONE", "DONE"]) != ["DONE", "DONE"]:
                raise RuntimeError("panel game did not reach DONE for both players")
            if not bool(getattr(result, "terminated", True)):
                raise RuntimeError("panel game did not terminate")
            reported_ids = {
                record["policy"].get("identity_id")
                for record in result.policy_identities
            }
            if reported_ids != {
                    candidate_policy.behavior_identity.identity_id(),
                    expected_opponent_id}:
                raise RuntimeError("panel result policy identities do not match its assignment")
        metrics = candidate_result_metrics(group, candidate_policy.behavior_identity)
        per_opponent.append({
            "opponent": {
                key: opponent[key] for key in (
                    "snapshot_id", "parameter_fingerprint", "behavior_identity",
                    "source_kind", "source_generation", "source_policy", "roles")
            },
            "metrics": metrics,
        })
        all_ordered_results.extend(group)
    aggregate = candidate_result_metrics(
        all_ordered_results, candidate_policy.behavior_identity)

    candidate_record = dict(candidate)
    candidate_record.update({
        "source_generation": candidate["source_generation"],
        "source_policy": candidate["source_policy"],
    })
    registry_ref = {
        "registry_id": registry["registry_id"],
        "registry_version": registry["registry_version"],
        "registry_state_fingerprint": registry_state_fingerprint,
        "current_champion_snapshot_id": registry["current_champion"]["snapshot_id"],
        "current_champion_parameter_fingerprint": (
            registry["current_champion"]["parameter_fingerprint"]),
        "bc_anchor_snapshot_id": registry["bc_anchor"]["snapshot_id"],
        "bc_anchor_snapshot_sha256": registry["bc_anchor"]["snapshot_sha256"],
        "bc_anchor_parameter_fingerprint": registry["bc_anchor"]["parameter_fingerprint"],
    }
    runtime = {
        "engine": args.engine,
        "executor": args.executor,
        "opening": args.opening,
        "workers": args.workers,
        "envs_per_worker": args.envs_per_worker,
        "batch_backend": bool(args.batch_backend),
        "physical_inference_batch_size": args.physical_batch_size,
        "inference_batch_wait_ms": args.inference_batch_wait_ms,
        "executor_factory": _executor_factory_provenance(runner.executor_factory),
        "backend_provenance": runner.provenance.get("backend"),
    }
    reproducibility = {
        "candidate_snapshot_sha256": candidate["snapshot_sha256"],
        "registry_state_fingerprint": registry_state_fingerprint,
        "seed": args.seed,
        "games_per_opponent": args.games_per_opponent,
        "runtime": runtime,
        "assignments": [
            (item.opponent_snapshot_id, item.pair_index,
             item.candidate_seat, item.seed)
            for item in assignments
        ],
    }
    evaluation_id = f"panel-{_digest(reproducibility)[:20]}"
    artifact = {
        "schema_version": PANEL_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "timestamp": _now(),
        "git_sha": _git_sha(),
        "candidate": candidate_record,
        "registry": registry_ref,
        "runtime": runtime,
        "games_per_opponent": args.games_per_opponent,
        "seed": args.seed,
        "seed_derivation": SEED_DERIVATION,
        "scheduled_games": len(assignments),
        "skipped_historical_slots": skipped,
        "per_opponent_results": per_opponent,
        "aggregate_results": aggregate,
        "raw_wlt_counts": {
            "candidate_wins": aggregate["candidate_wins"],
            "candidate_losses": aggregate["candidate_losses"],
            "ties": aggregate["ties"],
        },
        "seat_breakdown": aggregate["seat_breakdown"],
        "assignments": [
            {
                "episode_index": item.episode_index,
                "opponent_snapshot_id": item.opponent_snapshot_id,
                "opponent_roles": list(item.opponent_roles),
                "pair_index": item.pair_index,
                "candidate_seat": item.candidate_seat,
                "seed": item.seed,
            }
            for item in assignments
        ],
        "inference_metrics": runner.inference_metrics,
    }
    output_path = Path(args.output_dir) / f"panel_eval_{evaluation_id}.json"
    _write_json_atomic(output_path, artifact)
    return artifact, output_path


def main(argv: Sequence[str] | None = None) -> int:
    try:
        artifact, path = run_panel_evaluation(_parser().parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise SystemExit(f"stage25 panel evaluation failed: {exc}") from exc
    summary = artifact["aggregate_results"]
    print(json.dumps({
        "evaluation_id": artifact["evaluation_id"],
        "artifact": str(path),
        "games": summary["games"],
        "candidate_wins": summary["candidate_wins"],
        "candidate_losses": summary["candidate_losses"],
        "ties": summary["ties"],
        "candidate_win_fraction": summary["candidate_win_fraction"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
