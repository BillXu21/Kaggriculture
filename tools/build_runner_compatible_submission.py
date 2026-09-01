'''Build Kaggriculture archives compatible with legacy RL runner E semantics.

This builder is intentionally for checkpoints trained/evaluated with the
legacy rl_manager.runner economic-context behavior present at integration SHA
45f88001b6cc14f802f10668179a68f6fe3c2bf5. That runner passes the current
(day, money) pair as economic_prev_start, so bc_manager.live marks the cash
delta invalid and emits a zero/invalid previous-day delta.

Issue20 added a second deployment contract that must remain explicit:
PPO sell quantities come from the FULL immutable frozen-E network while the
other action heads come from the mutable PPO network. A single-network Torch
archive is therefore not equivalent after the PPO trunk drifts.

The standard_mixed opening also hides the executor during d0..d3, so the
first d4 manager call must receive the same realized d3 labor history used by
training: 5 hires, total hire cost 12.

Do not use this compatibility mode for checkpoints trained after the runner
bookkeeping is corrected and versioned.
'''

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

LEGACY_RUNNER_SHA = "45f88001b6cc14f802f10668179a68f6fe3c2bf5"

POLICY_CONTRACT_SINGLE = "single_network"
POLICY_CONTRACT_PPO_FROZEN_SELL = "ppo_frozen_sell_quantity"
POLICY_CONTRACTS = (
    POLICY_CONTRACT_SINGLE,
    POLICY_CONTRACT_PPO_FROZEN_SELL,
)

OPENING_HANDOFF_PREVIOUS_EXECUTION = {
    "workers_hired": 5,
    "hire_cost": 12,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MARKET_SOURCE = _REPO_ROOT / "fast_env" / "market.py"
_MARKET_DESTINATION = Path("executor_v0/_submission_market.py")
_OLD_MARKET_IMPORT = "from fast_env.market import market_price"
_NEW_MARKET_IMPORT = "from executor_v0._submission_market import market_price"

_MUTABLE_DESTINATION = Path("best.pt")
_FROZEN_DESTINATION = Path("frozen_e.pt")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vendor_market_helper(root: Path) -> None:
    '''Make executor market pricing self-contained inside its existing package.'''
    if not _MARKET_SOURCE.is_file():
        raise FileNotFoundError(
            f"required market helper missing from repository: {_MARKET_SOURCE}"
        )

    agent_path = root / "executor_v0" / "agent.py"
    if not agent_path.is_file():
        raise FileNotFoundError(
            f"base archive is missing executor runtime: {agent_path}"
        )

    agent_text = agent_path.read_text(encoding="utf-8")
    replacements = agent_text.count(_OLD_MARKET_IMPORT)
    if replacements != 1:
        raise RuntimeError(
            "expected exactly one fast_env.market import in packaged "
            f"executor_v0/agent.py, found {replacements}"
        )
    agent_path.write_text(
        agent_text.replace(_OLD_MARKET_IMPORT, _NEW_MARKET_IMPORT),
        encoding="utf-8",
    )

    destination = root / _MARKET_DESTINATION
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_MARKET_SOURCE, destination)


def _submission_main(*, aggressive_sell_all: bool, policy_contract: str) -> str:
    if policy_contract not in POLICY_CONTRACTS:
        raise ValueError(
            f"unknown policy_contract {policy_contract!r}; "
            f"expected one of {POLICY_CONTRACTS}"
        )
    aggressive = "True" if aggressive_sell_all else "False"
    dual = policy_contract == POLICY_CONTRACT_PPO_FROZEN_SELL
    frozen_argument = "_FROZEN_CHECKPOINT" if dual else "None"

    return f'''\"\"\"Kaggriculture submission matching legacy Issue20 runner semantics.\"\"\"

from pathlib import Path

import numpy as np
import torch

import executor_v0
from executor_v0 import AgentConfig, make_agent
from executor_v0.manager import CheckpointPlanProvider, decode_daily_plan
from bc_manager.live import encode_live_inputs
from opening_book.agent import make_opening_agent
from oracle.closed_loop import _executor_observation

_ROOT = Path(executor_v0.__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / "best.pt"
_FROZEN_CHECKPOINT = _ROOT / "frozen_e.pt"

_POLICY_CONTRACT = {policy_contract!r}
_OPENING_PREVIOUS_EXECUTION = {{
    "workers_hired": 5,
    "hire_cost": 12,
}}

_agent = None


class RunnerParityProvider(CheckpointPlanProvider):
    \"\"\"Reproduce Issue20 train-time E/labor and PPO action semantics.\"\"\"

    def __init__(self, checkpoint_path, frozen_checkpoint_path=None, device="cpu"):
        super().__init__(
            checkpoint_path,
            device=device,
            expected_e_history_version="E_LEGACY",
        )
        self._frozen = None
        if frozen_checkpoint_path is not None:
            self._frozen = CheckpointPlanProvider(
                frozen_checkpoint_path,
                device=device,
                expected_e_history_version="E_LEGACY",
            )
            if self.model_variant != self._frozen.model_variant:
                raise RuntimeError(
                    "mutable/frozen model_variant mismatch: "
                    f"{{self.model_variant!r}} vs {{self._frozen.model_variant!r}}"
                )
            if self.model_config != self._frozen.model_config:
                raise RuntimeError("mutable/frozen ManagerConfig mismatch")
            if self.include_opponent_board != self._frozen.include_opponent_board:
                raise RuntimeError("mutable/frozen opponent-board contract mismatch")
        self._first_manager_call = True

    def daily_plan(self, obs, seat, previous_execution=None):
        day = int(obs["day"])

        # During training the runner observed the deterministic opening and
        # supplied realized d3 labor to the first d4 manager decision. The
        # deployed executor is hidden behind opening_book until d4, so seed
        # the exact audited standard_mixed handoff here.
        if self._first_manager_call:
            self._first_manager_call = False
            if day == 4:
                previous_execution = dict(_OPENING_PREVIOUS_EXECUTION)

        economic_prev_start = None
        if self.uses_economic_context:
            # E_LEGACY intentionally sees zero/invalid previous-day cash.
            economic_prev_start = (
                day,
                float(obs["farms"][seat]["money"]),
            )

        inputs = encode_live_inputs(
            obs,
            seat,
            previous_execution,
            include_opponent=self.include_opponent_board,
            economic_prev_start=economic_prev_start,
        )
        batch = {{
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in inputs.items()
        }}

        with torch.no_grad():
            outputs = self.model(batch)
            if self._frozen is not None:
                frozen_outputs = self._frozen.model(batch)
                # Critical PPO contract: quantity is produced by the FULL
                # immutable frozen-E network, not by its head on the mutable
                # PPO representation.
                outputs = dict(outputs)
                outputs["sell_quantity_log1p"] = (
                    frozen_outputs["sell_quantity_log1p"]
                )

        self._record_coherence(obs, inputs, outputs)
        return decode_daily_plan(
            outputs,
            count_max=self.model_config.count_max,
        )


def agent(obs, configuration=None):
    del configuration
    global _agent

    adapted = _executor_observation(obs, from_fast=False)
    if _agent is None:
        seat = int(adapted["player"])
        provider = RunnerParityProvider(
            _CHECKPOINT,
            frozen_checkpoint_path={frozen_argument},
            device="cpu",
        )
        downstream = make_agent(
            provider=provider,
            seat=seat,
            config=AgentConfig(
                strict=True,
                suppress_expansion_from_prior_debt=True,
                aggressive_sell_all={aggressive},
                optional_spare_watering=True,
            ),
        )
        _agent = make_opening_agent(
            opening="standard_mixed",
            downstream=downstream,
            seat=seat,
        )
    return _agent(adapted)


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)
'''


def _manifest_file_records(root: Path) -> list[dict[str, object]]:
    records = []
    for member in sorted(path for path in root.rglob("*") if path.is_file()):
        rel = member.relative_to(root).as_posix()
        if rel == "submission_manifest.json":
            continue
        payload = member.read_bytes()
        records.append({
            "path": rel,
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
        })
    return records


def build_submission(
    *,
    base_archive: Path,
    checkpoint: Path,
    frozen_checkpoint: Path | None,
    output: Path,
    aggressive_sell_all: bool,
    label: str,
    policy_contract: str,
) -> dict[str, object]:
    '''Build one explicit deployment contract; ambiguous PPO packaging fails.'''
    if policy_contract not in POLICY_CONTRACTS:
        raise ValueError(
            f"policy_contract must be one of {POLICY_CONTRACTS}, "
            f"got {policy_contract!r}"
        )
    if not base_archive.is_file():
        raise FileNotFoundError(base_archive)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    dual = policy_contract == POLICY_CONTRACT_PPO_FROZEN_SELL
    if dual and frozen_checkpoint is None:
        raise ValueError(
            "ppo_frozen_sell_quantity requires --frozen-checkpoint; "
            "refusing to fabricate a single-network PPO submission"
        )
    if not dual and frozen_checkpoint is not None:
        raise ValueError(
            "single_network does not accept --frozen-checkpoint; "
            "choose the explicit PPO contract instead"
        )
    if frozen_checkpoint is not None and not frozen_checkpoint.is_file():
        raise FileNotFoundError(frozen_checkpoint)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(base_archive, "r:gz") as archive:
            archive.extractall(root)

        shutil.copy2(checkpoint, root / _MUTABLE_DESTINATION)
        if dual:
            assert frozen_checkpoint is not None
            shutil.copy2(frozen_checkpoint, root / _FROZEN_DESTINATION)
        else:
            (root / _FROZEN_DESTINATION).unlink(missing_ok=True)

        (root / "main.py").write_text(
            _submission_main(
                aggressive_sell_all=aggressive_sell_all,
                policy_contract=policy_contract,
            ),
            encoding="utf-8",
        )
        _vendor_market_helper(root)

        manifest_path = root / "submission_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        manifest["checkpoint_sha256"] = _sha256_bytes(
            (root / _MUTABLE_DESTINATION).read_bytes()
        )
        manifest["submission_variant"] = label
        manifest["submission_fix"] = "issue20_runner_deployment_parity_v2"
        fixes = [
            "legacy_runner_economic_context_parity",
            "standard_mixed_d4_labor_handoff",
        ]
        if dual:
            fixes.append("ppo_full_frozen_network_sell_quantity")
        manifest["submission_fixes"] = fixes
        manifest["policy_contract"] = policy_contract
        manifest["e_history_version"] = "E_LEGACY"
        manifest["legacy_runner_sha"] = LEGACY_RUNNER_SHA
        manifest["aggressive_sell_all"] = aggressive_sell_all
        manifest["optional_spare_watering"] = True
        manifest["opening_handoff_previous_execution"] = dict(
            OPENING_HANDOFF_PREVIOUS_EXECUTION
        )
        manifest["vendored_runtime_dependencies"] = [
            _MARKET_DESTINATION.as_posix()
        ]
        manifest["patched_runtime_imports"] = {
            _OLD_MARKET_IMPORT: _NEW_MARKET_IMPORT,
        }
        manifest["ppo_deployment_contract"] = (
            {
                "mutable_network": _MUTABLE_DESTINATION.as_posix(),
                "mutable_outputs": [
                    "crop_logits",
                    "animal_logits",
                    "land_logits",
                    "fertilizer_logits",
                    "care_logits",
                    "sell_presence_logits",
                ],
                "frozen_network": _FROZEN_DESTINATION.as_posix(),
                "frozen_outputs": ["sell_quantity_log1p"],
            }
            if dual else None
        )
        if dual:
            manifest["frozen_e_checkpoint_sha256"] = _sha256_bytes(
                (root / _FROZEN_DESTINATION).read_bytes()
            )
        else:
            manifest.pop("frozen_e_checkpoint_sha256", None)

        # Rebuild the file list from the actual archive tree. This guarantees
        # newly vendored helpers and frozen_e.pt cannot be omitted silently.
        manifest["files"] = _manifest_file_records(root)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        members = sorted(path for path in root.rglob("*") if path.is_file())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    for member in members:
                        arcname = member.relative_to(root).as_posix()
                        info = archive.gettarinfo(str(member), arcname)
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with member.open("rb") as handle:
                            archive.addfile(info, handle)

    result = {
        "archive": str(output),
        "archive_sha256": _sha256_bytes(output.read_bytes()),
        "checkpoint_sha256": _sha256_bytes(checkpoint.read_bytes()),
        "aggressive_sell_all": aggressive_sell_all,
        "optional_spare_watering": True,
        "label": label,
        "legacy_runner_sha": LEGACY_RUNNER_SHA,
        "policy_contract": policy_contract,
        "opening_handoff_previous_execution": dict(
            OPENING_HANDOFF_PREVIOUS_EXECUTION
        ),
        "vendored_runtime_dependencies": [_MARKET_DESTINATION.as_posix()],
        "patched_market_import": _NEW_MARKET_IMPORT,
    }
    if dual:
        assert frozen_checkpoint is not None
        result["frozen_e_checkpoint_sha256"] = _sha256_bytes(
            frozen_checkpoint.read_bytes()
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--policy-contract",
        required=True,
        choices=POLICY_CONTRACTS,
        help=(
            "Deployment semantics. PPO Issue20 checkpoints require "
            "ppo_frozen_sell_quantity plus --frozen-checkpoint."
        ),
    )
    parser.add_argument("--aggressive-sell-all", action="store_true")
    args = parser.parse_args()

    result = build_submission(
        base_archive=args.base_archive,
        checkpoint=args.checkpoint,
        frozen_checkpoint=args.frozen_checkpoint,
        output=args.output,
        aggressive_sell_all=args.aggressive_sell_all,
        label=args.label,
        policy_contract=args.policy_contract,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
