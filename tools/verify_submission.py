"""Fresh-extract and raw-load verifier for the BC-E V0.7 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

EXPECTED_BANK = 54439.0
EXPECTED_TRACE_FINGERPRINT = (
    "516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40"
)
EXPECTED_STATUSES = {"ACTIVE", "DONE"}
REQUIRED_PACKAGES = (
    "executor_v0",
    "bc_manager",
    "opening_book",
    "oracle",
    "replay_daily",
    "fast_env",
)


class VerificationError(RuntimeError):
    """Raised when archive extraction, dependency, or trajectory checks fail."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise VerificationError(f"unsafe archive member path: {name!r}")
    normalized = "/".join(path.parts)
    if normalized != name:
        raise VerificationError(f"non-normalized archive member path: {name!r}")
    return normalized


def extract_fresh(archive_path: str | Path, destination: str | Path) -> list[str]:
    """Extract regular files into a fresh empty directory without traversal."""
    archive = Path(archive_path).resolve()
    target = Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    if target.exists():
        if any(target.iterdir()):
            raise VerificationError(f"extraction target is not empty: {target}")
    else:
        target.mkdir(parents=True)

    names: list[str] = []
    with tarfile.open(archive, mode="r:gz") as source:
        for member in source.getmembers():
            name = _safe_member_name(member.name)
            if name in names:
                raise VerificationError(f"duplicate archive member: {name}")
            if not member.isfile():
                raise VerificationError(f"archive contains non-regular member: {name}")
            payload = source.extractfile(member)
            if payload is None:
                raise VerificationError(f"archive member has no payload: {name}")
            destination_path = target.joinpath(*PurePosixPath(name).parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(payload.read())
            names.append(name)
    return names


def _child_code() -> str:
    return r'''
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

ARCHIVE_ROOT = Path(sys.argv[1]).resolve()
REPOSITORY_ROOT = Path(sys.argv[2]).resolve()
VENV_ROOT = REPOSITORY_ROOT / ".venv"
REQUIRED_PACKAGES = ("executor_v0", "bc_manager", "opening_book", "oracle", "replay_daily", "fast_env")

def under(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents

def repository_source_path(path):
    return under(path, REPOSITORY_ROOT) and not under(path, VENV_ROOT)

def fail(message):
    raise RuntimeError(message)

sys.path[:] = [entry for entry in sys.path if not entry or not repository_source_path(entry)]
for entry in sys.path:
    if entry and repository_source_path(entry):
        fail(f"repository root present on sys.path: {entry}")

# This is deliberately before the game: fast_env.market is the historical
# lazy dependency whose omission otherwise survives until a late task path.
for name in REQUIRED_PACKAGES:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if not origin or not under(origin, ARCHIVE_ROOT):
        fail(f"runtime package {name} did not load from extracted archive: {origin!r}")
market = importlib.import_module("fast_env.market")
if not callable(getattr(market, "market_price", None)):
    fail("fast_env.market.market_price is not callable")
for module_name, module in list(sys.modules.items()):
    origin = getattr(module, "__file__", None)
    if origin and repository_source_path(origin):
        fail(f"repository import origin present: {module_name} -> {origin}")

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

source = (ARCHIVE_ROOT / "main.py").read_text(encoding="utf-8")
candidate = get_last_callable(source, path=str(ARCHIVE_ROOT / "main.py"))

from oracle.provenance import verify_official_provenance
provenance = verify_official_provenance()

trace = []
def pas(obs, configuration=None):
    seat = int(obs.get("player", 0))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (obs.get("farms", [])[seat].get("hands") or [])],
        "market": [],
    }

def recording_candidate(obs, configuration=None):
    action = candidate(obs, configuration)
    trace.append(copy.deepcopy(action))
    return action

env = make("kaggriculture", configuration={"seed": 7}, debug=True)
env.reset()
env.run([pas, recording_candidate])

anomalies = []
for step_index, states in enumerate(env.steps):
    for seat, state in enumerate(states):
        status = str(state.status)
        if status not in {"ACTIVE", "DONE"}:
            anomalies.append({"step": step_index, "seat": seat, "status": status})

bank = float(env.state[1].observation["farms"][1]["money"])
trace_bytes = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
fingerprint = hashlib.sha256(trace_bytes).hexdigest()
result = {
    "bank": bank,
    "status_history_entries": len(env.steps),
    "status_anomaly_count": len(anomalies),
    "status_anomalies": anomalies,
    "trace_actions": len(trace),
    "trace_fingerprint": fingerprint,
    "provenance": provenance,
    "sys_path_repository_root_present": any(entry and repository_source_path(entry) for entry in sys.path),
}
print(json.dumps(result, sort_keys=True))
if result["sys_path_repository_root_present"]:
    fail("repository root appeared on sys.path")
if anomalies:
    fail(f"status history anomalies: {anomalies}")
if bank != 54439.0:
    fail(f"reference bank mismatch: expected 54439.0, got {bank}")
'''


def _run_extracted(extracted: Path, repository_root: Path) -> dict[str, object]:
    environment = dict(__import__("os").environ)
    environment.pop("PYTHONPATH", None)
    environment["KAGGRICULTURE_SUBMISSION_STRICT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _child_code(), str(extracted), str(repository_root)],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise VerificationError(
            "raw extracted verification failed\n"
            f"command cwd: {extracted}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise VerificationError(f"raw verifier produced no JSON output; stderr:\n{result.stderr}")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise VerificationError(
            f"raw verifier output was not JSON: {result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    if EXPECTED_TRACE_FINGERPRINT and report["trace_fingerprint"] != EXPECTED_TRACE_FINGERPRINT:
        raise VerificationError(
            "trajectory fingerprint mismatch: "
            f"expected {EXPECTED_TRACE_FINGERPRINT}, got {report['trace_fingerprint']}"
        )
    return report


def verify_archive(archive_path: str | Path, *, repository_root: str | Path | None = None) -> dict[str, object]:
    """Fresh-extract, raw-load, and run the pinned official reference game."""
    archive = Path(archive_path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="bc-e-v07-verify-") as temp:
        extracted = Path(temp)
        members = extract_fresh(archive, extracted)
        required = {"main.py", "best.pt", "submission_manifest.json"}
        missing = sorted(required - set(members))
        if missing:
            raise VerificationError(f"archive missing required members: {missing}")
        report = _run_extracted(extracted, root)
    return {
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "member_count": len(members),
        "members": members,
        **report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_archive(args.archive)
    except (FileNotFoundError, VerificationError, OSError) as error:
        print(f"VERIFY FAILURE: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
