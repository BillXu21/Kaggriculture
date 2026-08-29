"""Build Kaggriculture archives compatible with legacy RL runner E semantics.

This builder is intentionally for checkpoints trained/evaluated with the
legacy rl_manager.runner economic-context behavior present at integration SHA
45f88001b6cc14f802f10668179a68f6fe3c2bf5. That runner passes the current
(day, money) pair as economic_prev_start, so bc_manager.live marks the cash
delta invalid and emits a zero/invalid previous-day delta. The normal Torch
CheckpointPlanProvider uses EconomicHistory and therefore supplies the real
previous-day delta; using it for those checkpoints creates a severe train /
deploy feature-distribution mismatch.

Do not use this compatibility mode for checkpoints trained after the runner
bookkeeping is corrected and versioned.
"""

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
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_DEPENDENCIES = (
    Path("fast_env/__init__.py"),
    Path("fast_env/market.py"),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle_runtime_dependencies(root: Path) -> None:
    """Copy submission-time pure-Python dependencies absent from old archives.

    executor_v0 imports ``fast_env.market.market_price`` lazily in the
    survival-feed affordability path. Local archive tests can accidentally hide
    the missing package when the repository is already importable, while the
    Kaggle submission sandbox contains only the archive. Bundle the two
    pure-Python files required for this import; do not bundle the native fast
    engine extension or its API modules.
    """
    for relative in _RUNTIME_DEPENDENCIES:
        source = _REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(
                f"required submission runtime dependency missing: {source}"
            )
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _submission_main(*, aggressive_sell_all: bool) -> str:
    aggressive = "True" if aggressive_sell_all else "False"
    return f'''"""Kaggriculture submission matching legacy RL-runner E economics."""

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
_agent = None


class RunnerParityProvider(CheckpointPlanProvider):
    """Reproduce the E economic-context semantics used by the legacy runner."""

    def daily_plan(self, obs, seat, previous_execution=None):
        economic_prev_start = None
        if self.uses_economic_context:
            # Intentional compatibility bug: current day is supplied as the
            # purported previous start. encode_live_inputs therefore treats
            # the delta as invalid and emits the legacy zero/invalid value.
            economic_prev_start = (
                int(obs["day"]),
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
        provider = RunnerParityProvider(_CHECKPOINT, device="cpu")
        downstream = make_agent(
            provider=provider,
            seat=seat,
            config=AgentConfig(
                strict=True,
                suppress_expansion_from_prior_debt=True,
                aggressive_sell_all={aggressive},
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


def build_submission(
    *,
    base_archive: Path,
    checkpoint: Path,
    output: Path,
    aggressive_sell_all: bool,
    label: str,
) -> dict[str, object]:
    if not base_archive.is_file():
        raise FileNotFoundError(base_archive)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(base_archive, "r:gz") as archive:
            archive.extractall(root)

        shutil.copy2(checkpoint, root / "best.pt")
        (root / "main.py").write_text(
            _submission_main(aggressive_sell_all=aggressive_sell_all),
            encoding="utf-8",
        )
        _bundle_runtime_dependencies(root)

        manifest_path = root / "submission_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoint_sha256"] = _sha256_bytes(
                (root / "best.pt").read_bytes()
            )
            manifest["submission_variant"] = label
            manifest["submission_fix"] = "legacy_runner_economic_context_parity"
            manifest["legacy_runner_sha"] = LEGACY_RUNNER_SHA
            manifest["aggressive_sell_all"] = aggressive_sell_all
            manifest["bundled_runtime_dependencies"] = [
                path.as_posix() for path in _RUNTIME_DEPENDENCIES
            ]
            for record in manifest.get("files", []):
                member = root / record["path"]
                if member.exists():
                    payload = member.read_bytes()
                    record["sha256"] = _sha256_bytes(payload)
                    record["bytes"] = len(payload)
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

    return {
        "archive": str(output),
        "archive_sha256": _sha256_bytes(output.read_bytes()),
        "checkpoint_sha256": _sha256_bytes(checkpoint.read_bytes()),
        "aggressive_sell_all": aggressive_sell_all,
        "label": label,
        "legacy_runner_sha": LEGACY_RUNNER_SHA,
        "bundled_runtime_dependencies": [
            path.as_posix() for path in _RUNTIME_DEPENDENCIES
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--aggressive-sell-all", action="store_true")
    args = parser.parse_args()

    result = build_submission(
        base_archive=args.base_archive,
        checkpoint=args.checkpoint,
        output=args.output,
        aggressive_sell_all=args.aggressive_sell_all,
        label=args.label,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
