"""BC-anchor and global champion snapshot registry for Stage 2.5.

Registry reads/writes and scheduling stay framework-neutral. JAX is imported
only inside snapshot-loading and export paths, so spawned rollout workers do
not acquire an accelerator dependency from importing this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid


CHAMPION_REGISTRY_SCHEMA_VERSION = "stage25_champion_registry_v1"
PANEL_EVALUATION_SCHEMA_VERSION = "stage25_panel_evaluation_v1"
PROMOTION_POLICY_SCHEMA_VERSION = "stage25_promotion_policy_v1"
DEFAULT_CHAMPION_HISTORY_DEPTH = 3
DEFAULT_PANEL_GAMES_PER_OPPONENT = 88


class ChampionRegistryError(ValueError):
    """Raised when champion, evaluation, or snapshot metadata is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_or_zero_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChampionRegistryError(f"{what} must be a nonnegative integer")
    return value


def _snapshot_components(path: str | Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Load a strict native inference snapshot and return record, policy, contract."""
    from rl_manager.stage25_checkpoint import (
        _config_from_json,
        load_stage25_inference_checkpoint,
    )
    from rl_manager.stage25_inference import (
        Stage25InferenceAdapter,
        parameter_fingerprint,
    )

    snapshot_path = Path(path).expanduser().resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"snapshot does not exist: {snapshot_path}")
    try:
        params, metadata = load_stage25_inference_checkpoint(
            snapshot_path, expected_e_history_version=None,
            allow_legacy_e=True)
    except Exception as exc:  # noqa: BLE001 - contract boundary
        raise ChampionRegistryError(
            f"{snapshot_path} is not a valid native inference snapshot: {exc}") from exc
    config = _config_from_json(metadata["config"])
    fingerprint = parameter_fingerprint(params)
    saved_identity = metadata.get("behavior_identity")
    if saved_identity is not None and not isinstance(saved_identity, Mapping):
        raise ChampionRegistryError("snapshot behavior_identity must be an object")
    name = (str(saved_identity["name"]) if saved_identity else
            f"stage25_snapshot_{fingerprint[:12]}")
    version = (str(saved_identity["version"]) if saved_identity else
               "native-inference-v1")
    policy = Stage25InferenceAdapter(
        params=params, config=config, name=name, version=version,
        seed=int(metadata.get("seed", 0)), mode="deterministic",
        e_history_version=metadata["e_history_version"])
    identity = policy.behavior_identity.to_json_dict()
    if identity["parameter_fingerprint"] != fingerprint:
        raise ChampionRegistryError("snapshot policy identity fingerprint is invalid")
    if saved_identity is not None and dict(saved_identity) != identity:
        raise ChampionRegistryError(
            "snapshot behavior_identity does not match its parameters or contract")

    evaluation = metadata.get("evaluation_snapshot", {})
    if evaluation and not isinstance(evaluation, Mapping):
        raise ChampionRegistryError("evaluation_snapshot metadata must be an object")
    if evaluation and evaluation.get("schema_version") != "stage25_evaluation_snapshot_v1":
        raise ChampionRegistryError("unsupported evaluation snapshot metadata version")
    source_kind = str(evaluation.get("source_kind", "native_inference"))
    source_generation = evaluation.get("source_generation")
    if source_generation is not None:
        source_generation = _positive_or_zero_int(
            source_generation, "snapshot source_generation")
    source_policy = evaluation.get("source_policy")
    if source_kind == "dual_policy" and source_policy not in {"A", "B"}:
        raise ChampionRegistryError(
            "dual-policy snapshot source_policy must be A or B")
    contract = {
        "architecture_version": metadata["architecture_version"],
        "observation_schema_version": metadata["observation_schema_version"],
        "action_schema_version": metadata["action_schema_version"],
        "persistent_ledger_version": metadata["persistent_ledger_version"],
        "physical_support_version": metadata["physical_support_version"],
        "action_vocabulary": metadata["action_vocabulary"],
        "action_class_counts": metadata["action_class_counts"],
        "observation_vocabulary": metadata["observation_vocabulary"],
        "model_config": metadata["config"],
        "curriculum": metadata["curriculum"],
        "e_identity": metadata["e_identity"],
        "behavior_contract": {
            key: identity[key] for key in (
                "observation_schema_version", "policy_schema_version",
                "e_history_version", "curriculum_version",
                "curriculum_fingerprint", "physical_support_version")
        },
    }
    file_sha = _file_sha256(snapshot_path)
    source_identity = metadata.get("source_identity", {})
    source_dual_identity = evaluation.get("source_dual_checkpoint_identity", {})
    source_generation = evaluation.get("source_generation", source_generation)
    source_policy = evaluation.get("source_policy", source_policy)
    record = {
        "snapshot_id": f"stage25-snapshot:{file_sha}",
        "path": str(snapshot_path),
        "snapshot_sha256": file_sha,
        "parameter_fingerprint": fingerprint,
        "behavior_identity": identity,
        "contract_fingerprint": _digest(contract),
        "source_kind": source_kind,
        "source_generation": source_generation,
        "source_policy": source_policy,
        "source_dual_checkpoint_identity": dict(source_dual_identity),
        "promoted_at": None,
        "promotion_evaluation_id": None,
    }
    if not isinstance(source_identity, Mapping):
        raise ChampionRegistryError("snapshot source_identity must be an object")
    return record, policy, contract


def make_snapshot_record(path: str | Path, *, source_kind: str | None = None) -> dict[str, Any]:
    """Inspect a native inference snapshot and create a registry record."""
    record, _policy, _contract = _snapshot_components(path)
    if source_kind is not None:
        record["source_kind"] = str(source_kind)
        if source_kind == "bc_anchor":
            record["source_generation"] = None
            record["source_policy"] = None
            record["source_dual_checkpoint_identity"] = {}
    record["promoted_at"] = _now()
    return record


def _validate_record(record: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ChampionRegistryError(f"{what} must be an object")
    required = {
        "snapshot_id", "path", "snapshot_sha256", "parameter_fingerprint",
        "behavior_identity", "contract_fingerprint", "source_kind",
        "source_generation", "source_policy", "promoted_at",
        "promotion_evaluation_id",
    }
    missing = required - set(record)
    if missing:
        raise ChampionRegistryError(f"{what} missing fields {sorted(missing)}")
    for key in ("snapshot_id", "path", "snapshot_sha256",
                "parameter_fingerprint", "contract_fingerprint", "source_kind"):
        if not isinstance(record[key], str) or not record[key]:
            raise ChampionRegistryError(f"{what}.{key} must be a nonempty string")
    if not isinstance(record["behavior_identity"], Mapping):
        raise ChampionRegistryError(f"{what}.behavior_identity must be an object")
    if record["source_generation"] is not None:
        _positive_or_zero_int(record["source_generation"],
                              f"{what}.source_generation")
    if record["source_policy"] not in (None, "A", "B"):
        raise ChampionRegistryError(f"{what}.source_policy must be A, B, or null")
    return dict(record)


def _validate_registry(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise ChampionRegistryError("registry root must be an object")
    if registry.get("schema_version") != CHAMPION_REGISTRY_SCHEMA_VERSION:
        raise ChampionRegistryError("unsupported champion registry schema_version")
    if not isinstance(registry.get("registry_id"), str) or not registry["registry_id"]:
        raise ChampionRegistryError("registry_id must be a nonempty string")
    version = _positive_or_zero_int(registry.get("registry_version"),
                                    "registry_version")
    if version < 1:
        raise ChampionRegistryError("registry_version must be at least 1")
    depth = _positive_or_zero_int(registry.get("history_depth"), "history_depth")
    anchor = _validate_record(registry.get("bc_anchor"), what="bc_anchor")
    current = _validate_record(
        registry.get("current_champion"), what="current_champion")
    if anchor["source_kind"] != "bc_anchor":
        raise ChampionRegistryError("bc_anchor source_kind must be bc_anchor")
    history = registry.get("history")
    if not isinstance(history, list) or len(history) > depth:
        raise ChampionRegistryError(
            "history must be a list no longer than configured history_depth")
    history = [_validate_record(item, what=f"history[{index}]")
               for index, item in enumerate(history)]
    lineage = registry.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ChampionRegistryError("lineage must be a nonempty list")
    for index, event in enumerate(lineage):
        if not isinstance(event, Mapping) or not isinstance(
                event.get("snapshot_id"), str):
            raise ChampionRegistryError(f"lineage[{index}] is invalid")
    result = dict(registry)
    result["bc_anchor"] = anchor
    result["current_champion"] = current
    result["history"] = history
    result["lineage"] = [dict(item) for item in lineage]
    return result


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def load_champion_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - persisted boundary
        raise ChampionRegistryError(
            f"cannot read champion registry {registry_path}: {exc}") from exc
    return _validate_registry(payload)


def initialize_champion_registry(
    registry_path: str | Path,
    bc_anchor_path: str | Path,
    *,
    history_depth: int = DEFAULT_CHAMPION_HISTORY_DEPTH,
) -> dict[str, Any]:
    """Create a young registry whose immutable BC anchor is also champion."""
    _positive_or_zero_int(history_depth, "history_depth")
    destination = Path(registry_path).expanduser().resolve()
    if destination.exists():
        raise ChampionRegistryError(
            f"registry already exists; refusing to replace its BC anchor: {destination}")
    anchor, _policy, _contract = _snapshot_components(bc_anchor_path)
    anchor.update({
        "source_kind": "bc_anchor",
        "source_generation": None,
        "source_policy": None,
        "source_dual_checkpoint_identity": {},
        "promoted_at": _now(),
        "promotion_evaluation_id": None,
    })
    registry = {
        "schema_version": CHAMPION_REGISTRY_SCHEMA_VERSION,
        "registry_id": str(uuid.uuid4()),
        "registry_version": 1,
        "history_depth": int(history_depth),
        "created_at": _now(),
        "bc_anchor": dict(anchor),
        "current_champion": dict(anchor),
        "history": [],
        "lineage": [{
            "promotion_number": 0,
            "snapshot_id": anchor["snapshot_id"],
            "parameter_fingerprint": anchor["parameter_fingerprint"],
            "source_kind": "bc_anchor",
            "source_generation": None,
            "source_policy": None,
            "promoted_over_snapshot_id": None,
            "promotion_evaluation_id": None,
            "promoted_at": anchor["promoted_at"],
        }],
    }
    _write_json_atomic(destination, _validate_registry(registry))
    return load_champion_registry(destination)


def _registry_state_fingerprint(registry: Mapping[str, Any]) -> str:
    return _digest(dict(registry))


def select_panel_opponents(registry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return unique snapshot fingerprints with all logical panel roles."""
    state = _validate_registry(registry)
    ordered: list[tuple[str, dict[str, Any]]] = [
        ("current_champion", state["current_champion"]),
    ]
    ordered.extend((f"recent_champion_{index + 1}", record)
                   for index, record in enumerate(state["history"]))
    ordered.append(("bc_anchor", state["bc_anchor"]))
    unique: list[dict[str, Any]] = []
    by_fingerprint: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for role, record in ordered:
        snapshot_path = Path(record["path"])
        if not snapshot_path.is_file() and role.startswith("recent_champion_"):
            skipped.append({"role": role, "reason": "snapshot file is unavailable"})
            continue
        if not snapshot_path.is_file():
            raise ChampionRegistryError(
                f"required panel snapshot for {role} is unavailable: {snapshot_path}")
        actual_sha = _file_sha256(snapshot_path)
        if actual_sha != record["snapshot_sha256"]:
            raise ChampionRegistryError(
                f"registered snapshot changed on disk for {role}: {snapshot_path}")
        existing = by_fingerprint.get(record["parameter_fingerprint"])
        if existing is not None:
            if existing["contract_fingerprint"] != record["contract_fingerprint"]:
                raise ChampionRegistryError(
                    "duplicate panel parameter fingerprints have incompatible contracts")
            existing["roles"].append(role)
            continue
        loaded, _policy, _contract = _snapshot_components(snapshot_path)
        if (loaded["parameter_fingerprint"] != record["parameter_fingerprint"]
                or loaded["contract_fingerprint"] != record["contract_fingerprint"]):
            raise ChampionRegistryError(
                f"registered snapshot identity changed on disk for {role}")
        value = dict(record)
        value["roles"] = [role]
        unique.append(value)
        by_fingerprint[record["parameter_fingerprint"]] = value
    return unique, skipped


@dataclass(frozen=True)
class PromotionPolicy:
    """Optional minimum win fractions keyed by logical panel role.

    All thresholds default to ``None``. This policy object never promotes on
    its own; it only validates an explicit manual promotion request.
    """

    minimum_win_fraction: Mapping[str, float | None]

    def __post_init__(self) -> None:
        allowed_prefixes = {"current_champion", "bc_anchor"}
        if not isinstance(self.minimum_win_fraction, Mapping):
            raise TypeError("minimum_win_fraction must be a mapping")
        for role, value in self.minimum_win_fraction.items():
            if role not in allowed_prefixes and not role.startswith("recent_champion_"):
                raise ValueError(f"unsupported promotion-policy role {role!r}")
            if role.startswith("recent_champion_"):
                suffix = role.removeprefix("recent_champion_")
                if not suffix.isdigit() or int(suffix) < 1:
                    raise ValueError(f"invalid recent champion role {role!r}")
            if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0):
                raise ValueError(
                    f"promotion threshold for {role} must be in [0, 1] or null")

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PromotionPolicy":
        if payload.get("schema_version") != PROMOTION_POLICY_SCHEMA_VERSION:
            raise ChampionRegistryError("unsupported promotion policy schema_version")
        thresholds = payload.get("minimum_win_fraction", {})
        if not isinstance(thresholds, Mapping):
            raise ChampionRegistryError("minimum_win_fraction must be an object")
        return cls(dict(thresholds))

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
            "minimum_win_fraction": dict(self.minimum_win_fraction),
        }


def _check_promotion_policy(
    evaluation: Mapping[str, Any], policy: PromotionPolicy,
) -> dict[str, Any]:
    opponents = evaluation.get("per_opponent_results")
    if not isinstance(opponents, list):
        raise ChampionRegistryError("evaluation has no per_opponent_results list")
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in opponents:
        if not isinstance(row, Mapping):
            raise ChampionRegistryError("evaluation opponent result is invalid")
        opponent = row.get("opponent")
        metrics = row.get("metrics")
        if not isinstance(opponent, Mapping) or not isinstance(metrics, Mapping):
            raise ChampionRegistryError("evaluation opponent identity or metrics are invalid")
        roles = opponent.get("roles", [])
        if not isinstance(roles, list):
            raise ChampionRegistryError("evaluation opponent roles must be a list")
        for role in roles:
            by_role[str(role)] = metrics
    conditions = []
    failures = []
    for role, threshold in sorted(policy.minimum_win_fraction.items()):
        if threshold is None:
            continue
        metrics = by_role.get(role)
        observed = None if metrics is None else metrics.get("candidate_win_fraction")
        passed = observed is not None and float(observed) >= float(threshold)
        conditions.append({
            "role": role, "observed_win_fraction": observed,
            "minimum_win_fraction": float(threshold), "passed": passed,
        })
        if not passed:
            failures.append(
                f"{role} win fraction {observed!r} is below {float(threshold)}")
    if failures:
        raise ChampionRegistryError(
            "promotion policy rejected candidate: " + "; ".join(failures))
    return {"conditions": conditions, "passed": True}


def _load_evaluation(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise ChampionRegistryError(f"cannot read evaluation artifact: {exc}") from exc
    if not isinstance(evaluation, dict) or evaluation.get("schema_version") != PANEL_EVALUATION_SCHEMA_VERSION:
        raise ChampionRegistryError("unsupported panel evaluation artifact")
    return evaluation


def promote_champion(
    registry_path: str | Path,
    candidate_path: str | Path,
    evaluation_path: str | Path,
    *,
    promotion_policy: PromotionPolicy | None = None,
) -> dict[str, Any]:
    """Manually promote an evaluated dual-policy snapshot with stale-state checks."""
    registry_file = Path(registry_path).expanduser().resolve()
    registry = load_champion_registry(registry_file)
    evaluation = _load_evaluation(evaluation_path)
    registry_ref = evaluation.get("registry")
    if not isinstance(registry_ref, Mapping):
        raise ChampionRegistryError("evaluation does not reference a registry")
    expected_ref = {
        "registry_id": registry["registry_id"],
        "registry_version": registry["registry_version"],
        "registry_state_fingerprint": _registry_state_fingerprint(registry),
        "current_champion_snapshot_id": registry["current_champion"]["snapshot_id"],
        "current_champion_parameter_fingerprint": (
            registry["current_champion"]["parameter_fingerprint"]),
        "bc_anchor_snapshot_id": registry["bc_anchor"]["snapshot_id"],
        "bc_anchor_snapshot_sha256": registry["bc_anchor"]["snapshot_sha256"],
        "bc_anchor_parameter_fingerprint": registry["bc_anchor"]["parameter_fingerprint"],
    }
    for key, expected in expected_ref.items():
        if registry_ref.get(key) != expected:
            raise ChampionRegistryError(
                f"evaluation is stale or references another registry ({key})")

    anchor_record, _anchor_policy, anchor_contract = _snapshot_components(
        registry["bc_anchor"]["path"])
    if (anchor_record["snapshot_sha256"] != registry["bc_anchor"]["snapshot_sha256"]
            or anchor_record["parameter_fingerprint"]
            != registry["bc_anchor"]["parameter_fingerprint"]
            or anchor_record["contract_fingerprint"]
            != registry["bc_anchor"]["contract_fingerprint"]):
        raise ChampionRegistryError("registered BC anchor changed; promotion refused")
    current_record, _current_policy, _current_contract = _snapshot_components(
        registry["current_champion"]["path"])
    if (current_record["snapshot_sha256"]
            != registry["current_champion"]["snapshot_sha256"]
            or current_record["parameter_fingerprint"]
            != registry["current_champion"]["parameter_fingerprint"]
            or current_record["contract_fingerprint"]
            != registry["current_champion"]["contract_fingerprint"]):
        raise ChampionRegistryError(
            "registered current champion changed; promotion refused")

    evaluated_roles: dict[str, Mapping[str, Any]] = {}
    opponent_rows = evaluation.get("per_opponent_results")
    if not isinstance(opponent_rows, list):
        raise ChampionRegistryError("evaluation has no per_opponent_results list")
    for row in opponent_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("opponent"), Mapping):
            raise ChampionRegistryError("evaluation opponent record is invalid")
        for role in row.get("opponent", {}).get("roles", []):
            evaluated_roles[str(role)] = row["opponent"]
    for role, record in (
            ("current_champion", registry["current_champion"]),
            ("bc_anchor", registry["bc_anchor"])):
        evaluated = evaluated_roles.get(role)
        if (evaluated is None
                or evaluated.get("snapshot_id") != record["snapshot_id"]
                or evaluated.get("parameter_fingerprint")
                != record["parameter_fingerprint"]):
            raise ChampionRegistryError(
                f"evaluation does not contain the current {role} identity")

    candidate_file = Path(candidate_path).expanduser().resolve()
    candidate, _candidate_policy, _candidate_contract = _snapshot_components(candidate_file)
    if candidate["source_kind"] != "dual_policy":
        raise ChampionRegistryError(
            "promotion candidates must be exported from a dual PPO checkpoint")
    evaluation_candidate = evaluation.get("candidate")
    if not isinstance(evaluation_candidate, Mapping):
        raise ChampionRegistryError("evaluation candidate identity is missing")
    for key in ("snapshot_id", "snapshot_sha256", "parameter_fingerprint",
                "contract_fingerprint", "source_generation", "source_policy"):
        if evaluation_candidate.get(key) != candidate.get(key):
            raise ChampionRegistryError(
                f"candidate {key} does not match the evaluated snapshot")
    if candidate["contract_fingerprint"] != anchor_record["contract_fingerprint"]:
        raise ChampionRegistryError(
            "candidate model/observation/action/E-history/curriculum contract "
            "is incompatible with the BC anchor")
    if candidate["contract_fingerprint"] != registry["current_champion"]["contract_fingerprint"]:
        raise ChampionRegistryError("candidate contract is incompatible with current champion")
    known_fingerprints = {
        registry["current_champion"]["parameter_fingerprint"],
        registry["bc_anchor"]["parameter_fingerprint"],
        *(record["parameter_fingerprint"] for record in registry["history"]),
    }
    if candidate["parameter_fingerprint"] in known_fingerprints:
        raise ChampionRegistryError(
            "candidate parameters already appear in the champion lineage")

    policy_result = ({"conditions": [], "passed": True}
                     if promotion_policy is None else
                     _check_promotion_policy(evaluation, promotion_policy))
    old_current = dict(registry["current_champion"])
    promoted = dict(candidate)
    promoted_at = _now()
    evaluation_id = evaluation.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ChampionRegistryError("evaluation_id must be a nonempty string")
    promoted["promoted_at"] = promoted_at
    promoted["promotion_evaluation_id"] = evaluation_id
    promoted["promoted_over"] = {
        "snapshot_id": old_current["snapshot_id"],
        "parameter_fingerprint": old_current["parameter_fingerprint"],
    }
    history = [old_current, *registry["history"]]
    history = history[:registry["history_depth"]]
    promotion_number = len(registry["lineage"])
    registry["current_champion"] = promoted
    registry["history"] = history
    registry["registry_version"] += 1
    registry["lineage"].append({
        "promotion_number": promotion_number,
        "snapshot_id": promoted["snapshot_id"],
        "parameter_fingerprint": promoted["parameter_fingerprint"],
        "source_kind": promoted["source_kind"],
        "source_generation": promoted["source_generation"],
        "source_policy": promoted["source_policy"],
        "promoted_over_snapshot_id": old_current["snapshot_id"],
        "promoted_over_parameter_fingerprint": old_current["parameter_fingerprint"],
        "promotion_evaluation_id": evaluation_id,
        "promoted_at": promoted_at,
    })
    registry["last_promotion_policy"] = {
        "schema_version": PROMOTION_POLICY_SCHEMA_VERSION,
        **policy_result,
        "configured_minimum_win_fraction": (
            {} if promotion_policy is None else
            dict(promotion_policy.minimum_win_fraction)),
    }
    _write_json_atomic(registry_file, _validate_registry(registry))
    return load_champion_registry(registry_file)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a BC-anchor champion registry")
    init.add_argument("--registry", type=Path, required=True)
    init.add_argument("--bc-anchor", type=Path, required=True)
    init.add_argument("--history-depth", type=int,
                      default=DEFAULT_CHAMPION_HISTORY_DEPTH)
    export = commands.add_parser("export", help="export A or B from a dual PPO checkpoint")
    export.add_argument("--dual-checkpoint", type=Path, required=True)
    export.add_argument("--policy", choices=("A", "B"), required=True)
    export.add_argument("--output", type=Path, required=True)
    promote = commands.add_parser("promote", help="manually promote an evaluated candidate")
    promote.add_argument("--registry", type=Path, required=True)
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--evaluation", type=Path, required=True)
    promote.add_argument("--policy-config", type=Path,
                         help="optional explicit promotion-policy JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_champion_registry(
                args.registry, args.bc_anchor, history_depth=args.history_depth)
        elif args.command == "export":
            from rl_manager.stage25_checkpoint import (
                export_stage25_dual_policy_snapshot)
            path = export_stage25_dual_policy_snapshot(
                args.dual_checkpoint, args.output, policy=args.policy)
            result = {"snapshot": str(path)}
        else:
            policy = None
            if args.policy_config is not None:
                with args.policy_config.open("r", encoding="utf-8") as handle:
                    policy = PromotionPolicy.from_json(json.load(handle))
            result = promote_champion(
                args.registry, args.candidate, args.evaluation,
                promotion_policy=policy)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise SystemExit(f"stage25 champion {args.command} failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
