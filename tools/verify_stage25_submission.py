"""Fresh-extract verifier for native Stage 2.5 submission archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

OFFICIAL_ENGINE_VERSION = "1.32.7"
SMOKE_SEEDS = (7, 42)
PARITY_SEED = 7
PARITY_SEAT = 0


class VerificationError(RuntimeError):
    """Raised when an archive violates submission runtime expectations."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (path.is_absolute() or windows_path.is_absolute() or windows_path.drive
            or ".." in path.parts or not path.parts):
        raise VerificationError(f"unsafe archive member path: {name!r}")
    normalized = "/".join(path.parts)
    if "\\" in name or normalized != name:
        raise VerificationError(f"non-normalized archive member path: {name!r}")
    return normalized


def extract_fresh(archive_path: str | Path, destination: str | Path) -> list[str]:
    """Safely extract regular files into a fresh empty directory."""
    archive_path = Path(archive_path).resolve()
    target = Path(destination).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise VerificationError(f"extraction target is not empty: {target}")
    names: list[str] = []
    with tarfile.open(archive_path, mode="r:gz") as source:
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
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time

ROOT = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
CHECKPOINT = Path(sys.argv[3]).resolve()
SEEDS = [int(x) for x in sys.argv[4].split(",") if x]
PARITY_SEED = int(sys.argv[5])
PARITY_SEAT = int(sys.argv[6])
REPOSITORY_ROOT = Path(sys.argv[7]).resolve()

def under(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents

def fail(message):
    raise RuntimeError(message)

if MODE == "archive":
    sys.path[:] = [entry for entry in sys.path if not entry or (
        not under(entry, ROOT) and not under(entry, REPOSITORY_ROOT))]
    sys.path.insert(0, str(ROOT))
elif MODE == "source":
    sys.path[:] = [entry for entry in sys.path
                   if not entry or not under(entry, ROOT)]
    sys.path.insert(0, str(ROOT))
else:
    fail(f"unknown verifier mode: {MODE}")

required = ("executor_v0", "bc_manager", "bc_manager_jax", "opening_book",
            "replay_daily", "rl_manager")

def module_origins():
    result = {}
    for name in required:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if not origin or not under(origin, ROOT):
            fail(f"runtime package {name} did not load under {ROOT}: {origin!r}")
        result[name] = str(Path(origin).resolve())
    return result

def ensure_no_repository_modules(repository_root):
    if MODE != "archive":
        return
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin and under(origin, repository_root):
            fail(f"repository module leaked into archive run: {name} -> {origin}")

def forbidden_training_modules():
    prefixes = ("optax", "torch", "bc_manager.training", "bc_manager.loss",
                "bc_manager.model", "bc_manager_jax.train", "bc_manager_jax.loss",
                "bc_manager_jax.sharding", "rl_manager.stage25_ppo_cli",
                "rl_manager.ppo", "rl_manager.ppo_")
    return sorted(name for name in sys.modules
                  if any(name == prefix or name.startswith(prefix + ".")
                         for prefix in prefixes))

if MODE == "archive":
    from kaggle_environments.agent import get_last_callable
    main_path = ROOT / "main.py"
    if not main_path.is_file():
        fail("archive main.py is missing")
    candidate = get_last_callable(main_path.read_text(encoding="utf-8"),
                                  path=str(main_path))
else:
    from rl_manager.stage25_submission import make_stage25_submission_agent
    candidate = None

startup_jax_loaded = "jax" in sys.modules
startup_training_modules = forbidden_training_modules()
if startup_jax_loaded:
    fail("ordinary submission startup imported JAX before the first manager decision")
if startup_training_modules:
    fail(f"training-only modules imported during ordinary startup: {startup_training_modules}")

# Probe JAX before any checkpoint loader imports it so an unavailable native
# runtime is reported explicitly and cannot be mistaken for an archive failure.
jax_report = {"import_success": False, "jax_version": None,
              "jaxlib_version": None, "devices": [], "backend": None,
              "platforms": [], "error": None}
try:
    import jax
    jax_report["import_success"] = True
    jax_report["jax_version"] = jax.__version__
    try:
        import jaxlib
        jax_report["jaxlib_version"] = jaxlib.__version__
    except Exception as exc:
        jax_report["error"] = f"jaxlib version unavailable: {type(exc).__name__}: {exc}"
    devices = jax.devices()
    jax_report["devices"] = [str(device) for device in devices]
    jax_report["platforms"] = sorted({str(device.platform) for device in devices})
    jax_report["backend"] = jax.default_backend()
except Exception as exc:
    jax_report["error"] = f"{type(exc).__name__}: {exc}"

if not jax_report["import_success"]:
    report = {"ok": False, "mode": MODE, "jax_runtime": jax_report,
              "message": "native Stage 2.5 deployment is unavailable in this runtime: JAX import failed",
              "runtime_import_origins": {}}
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(2)

origins = module_origins()
if MODE == "archive" and "fast_env" in sys.modules:
    fail("fast_env imported on the native submission runtime path")
if forbidden_training_modules():
    fail(f"training-only modules imported during ordinary startup: {forbidden_training_modules()}")

# Validate checkpoint bytes, strict metadata and native payload from the active
# source root before gameplay.
from rl_manager.stage25_checkpoint import load_stage25_inference_checkpoint
checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
params, checkpoint_meta = load_stage25_inference_checkpoint(CHECKPOINT)
del params
manifest = None
if MODE == "archive":
    manifest_path = ROOT / "submission_manifest.json"
    if not manifest_path.is_file():
        fail("archive submission_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        fail("extracted checkpoint SHA-256 differs from manifest")
    if checkpoint_meta.get("payload_kind") != manifest.get("checkpoint_payload_kind"):
        fail("checkpoint payload kind differs from manifest")
    if checkpoint_meta.get("architecture_version") != manifest.get("architecture_version"):
        fail("checkpoint architecture differs from manifest")
    records = manifest.get("members")
    if not isinstance(records, list):
        fail("manifest member inventory is missing")
    expected_paths = set()
    for record in records:
        member_name = record.get("path")
        if not isinstance(member_name, str) or member_name in expected_paths:
            fail(f"invalid or duplicate manifest member path: {member_name!r}")
        manifest_path = PurePosixPath(member_name)
        windows_path = PureWindowsPath(member_name)
        if (manifest_path.is_absolute() or windows_path.is_absolute()
                or windows_path.drive or ".." in manifest_path.parts
                or "\\" in member_name or "/".join(manifest_path.parts) != member_name):
            fail(f"unsafe manifest member path: {member_name!r}")
        expected_paths.add(member_name)
        member_path = ROOT.joinpath(*manifest_path.parts)
        if not member_path.is_file():
            fail(f"manifest member is missing from extraction: {member_name}")
        payload = member_path.read_bytes()
        if len(payload) != record.get("bytes") or hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            fail(f"archive member differs from manifest: {member_name}")
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*") if path.is_file()
        and path.name != "submission_manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual_paths != expected_paths:
        fail("extracted archive member set differs from manifest inventory: "
             f"missing={sorted(expected_paths - actual_paths)}, "
             f"extra={sorted(actual_paths - expected_paths)}")

if MODE == "archive":
    market_module = importlib.import_module("executor_v0.strip_market")
    vendored_market = importlib.import_module("executor_v0._submission_market")
    price = vendored_market.market_price("WHEAT", 10_000)
    if not isinstance(price, int) or price <= 0:
        fail("vendored lazy market/resource pricing path returned invalid price")
    if market_module.market_price("WHEAT", 10_000) != price:
        fail("strip executor did not route lazy pricing through vendored market helper")
    if "fast_env" in sys.modules:
        fail("market pricing imported fast_env instead of its self-contained helper")

from kaggle_environments import make
import kaggle_environments
if str(getattr(kaggle_environments, "__version__", "")) != "1.32.7":
    fail(f"official engine version must be 1.32.7, got {getattr(kaggle_environments, '__version__', None)!r}")

def pass_agent(obs, configuration=None):
    del configuration
    seat = int(obs.get("player", 0))
    hands = obs.get("farms", [])[seat].get("hands") or []
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}

def play(seed, seat, want_trace=True):
    if MODE == "archive":
        candidate.__globals__["_agent"] = None
        agent = candidate
    else:
        agent = make_stage25_submission_agent(CHECKPOINT, seat=seat)
    trace = []
    valid_actions = 0
    def recording(obs, configuration=None):
        nonlocal valid_actions
        action = agent(obs, configuration)
        if not isinstance(action, dict):
            fail(f"agent action is not a JSON object: {type(action).__name__}")
        try:
            json.dumps(action, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            fail(f"agent emitted invalid JSON-shaped action: {exc}")
        valid_actions += 1
        trace.append(copy.deepcopy(action))
        return action
    agents = [pass_agent, pass_agent]
    agents[seat] = recording
    started = time.perf_counter()
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.reset()
    env.run(agents)
    runtime_s = time.perf_counter() - started
    statuses = [str(state.status) for states in env.steps for state in states]
    anomalies = [status for status in statuses if status not in {"ACTIVE", "DONE"}]
    final_status = [str(state.status) for state in env.state]
    banks = [float(state.observation["farms"][i]["money"])
             for i, state in enumerate(env.state)]
    if anomalies:
        fail(f"official game status anomalies seed={seed} seat={seat}: {anomalies[:10]}")
    if any(not math.isfinite(value) for value in banks):
        fail(f"official game ended with non-finite banks: {banks}")
    if any(status not in {"DONE", "TERMINAL", "FINISHED"} for status in final_status):
        fail(f"official game did not reach normal terminal state: {final_status}")
    trace_bytes = json.dumps(trace, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    runtime_agent = (agent.__globals__.get("_agent") if MODE == "archive"
                     else agent)
    diagnostics = getattr(runtime_agent, "diagnostics_json", None)
    diagnostics = diagnostics() if callable(diagnostics) else {}
    inference_latencies = diagnostics.get("manager_inference_latency_s", [])
    if forbidden_training_modules():
        fail(f"training-only modules imported by first inference: {forbidden_training_modules()}")
    if MODE == "archive":
        ensure_no_repository_modules(REPOSITORY_ROOT)
    return {
        "seed": seed,
        "seat": seat,
        "final_banks": banks,
        "status_history_anomaly_count": len(anomalies),
        "action_count": len(trace),
        "valid_json_action_count": valid_actions,
        "status_history_entries": len(env.steps),
        "action_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "runtime_seconds": runtime_s,
        "first_call_latency_seconds": diagnostics.get("first_call_latency_s"),
        "manager_inference_latency_seconds": inference_latencies,
        "final_status": final_status,
        "terminal": True,
    }

games = []
if MODE == "archive":
    for seed in SEEDS:
        for seat in (0, 1):
            games.append(play(seed, seat))
    if PARITY_SEED not in SEEDS:
        games.append(play(PARITY_SEED, PARITY_SEAT))
    parity_archive = next(
        row for row in games
        if row["seed"] == PARITY_SEED and row["seat"] == PARITY_SEAT)
    parity_source = play(PARITY_SEED, PARITY_SEAT)
    source_vs_archive_parity = {
        "seed": PARITY_SEED,
        "seat": PARITY_SEAT,
        "source_action_trace_sha256": parity_source["action_trace_sha256"],
        "archive_action_trace_sha256": parity_archive["action_trace_sha256"],
        "exact_match": parity_source["action_trace_sha256"] == parity_archive["action_trace_sha256"],
    }
    if not source_vs_archive_parity["exact_match"]:
        fail(f"source/archive action trace mismatch: {source_vs_archive_parity}")
else:
    games.append(play(PARITY_SEED, PARITY_SEAT))
    source_vs_archive_parity = None

if MODE == "archive":
    ensure_no_repository_modules(Path(sys.argv[7]).resolve())
report = {
    "ok": True,
    "mode": MODE,
    "official_engine_version": kaggle_environments.__version__,
    "runtime_import_origins": origins,
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_metadata": {
        key: checkpoint_meta.get(key) for key in
        ("format", "payload_kind", "architecture_version",
         "observation_schema_version", "action_schema_version", "e_history_version")
    },
    "jax_runtime": jax_report,
    "training_only_modules_imported": forbidden_training_modules(),
    "vendored_market_price_probe": price if MODE == "archive" else None,
    "games": games,
    "source_vs_archive_trace_parity": source_vs_archive_parity,
    "manifest": manifest,
}
print(json.dumps(report, sort_keys=True, allow_nan=False))
'''


def _run_child(
    extracted: Path,
    repository_root: Path,
    *,
    mode: str,
    checkpoint: Path,
    seeds: tuple[int, ...],
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("KAGGRICULTURE_SUBMISSION_STRICT", None)
    command = [
        sys.executable, "-c", _child_code(), str(extracted), mode,
        str(checkpoint), ",".join(map(str, seeds)), str(PARITY_SEED),
        str(PARITY_SEAT), str(repository_root),
    ]
    result = subprocess.run(
        command, cwd=extracted, env=env, capture_output=True, text=True,
        timeout=timeout_seconds,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    child_report: dict[str, Any] | None = None
    if lines:
        try:
            parsed = json.loads(lines[-1])
            if isinstance(parsed, dict):
                child_report = parsed
        except json.JSONDecodeError:
            child_report = None
    if result.returncode != 0:
        detail = {
            "return_code": result.returncode,
            "report": child_report,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
        }
        if child_report and not child_report.get("jax_runtime", {}).get("import_success", True):
            message = (
                "native Stage 2.5 deployment is unavailable in this runtime: "
                "JAX import failed")
        else:
            message = "fresh-extract verifier child failed"
        raise VerificationError(f"{message}: {json.dumps(detail, sort_keys=True)}")
    if child_report is None:
        raise VerificationError(
            "verifier child produced no machine-readable JSON report; "
            f"stdout={result.stdout[-4000:]!r}; stderr={result.stderr[-4000:]!r}")
    return child_report


def verify_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    seeds: tuple[int, ...] = SMOKE_SEEDS,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    root = Path(repository_root or Path(__file__).resolve().parent.parent).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    if not seeds or any(isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    with tempfile.TemporaryDirectory(prefix="stage25-submission-verify-") as tmp:
        extracted = Path(tmp).resolve()
        members = extract_fresh(archive, extracted)
        required = {"main.py", "stage25.npz", "submission_manifest.json"}
        missing = sorted(required - set(members))
        if missing:
            raise VerificationError(f"archive missing required members: {missing}")
        member_set = set(members)
        for name in members:
            if _safe_member_name(name) != name:
                raise VerificationError(f"archive contains unsafe path: {name!r}")
        archive_report = _run_child(
            extracted, root, mode="archive",
            checkpoint=extracted / "stage25.npz", seeds=seeds)
        # Source parity uses the same freshly extracted checkpoint bytes while
        # importing repository source modules explicitly from the checkout.
        source_report = _run_child(
            root, root, mode="source",
            checkpoint=extracted / "stage25.npz", seeds=())
        if archive_report.get("checkpoint_sha256") != source_report.get("checkpoint_sha256"):
            raise VerificationError("source/archive parity used different checkpoint bytes")
        parity = archive_report.get("source_vs_archive_trace_parity")
        if not isinstance(parity, dict) or not parity.get("exact_match"):
            raise VerificationError(f"source/archive action trace mismatch: {parity!r}")
        return {
            "ok": True,
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "archive_size_bytes": archive.stat().st_size,
            "member_count": len(members),
            "members": sorted(member_set),
            "unsafe_member_paths": [],
            "fresh_extract_import_origins": archive_report["runtime_import_origins"],
            "jax_runtime": archive_report["jax_runtime"],
            "official_engine_version": archive_report["official_engine_version"],
            "checkpoint_sha256": archive_report["checkpoint_sha256"],
            "checkpoint_metadata": archive_report["checkpoint_metadata"],
            "training_only_modules_imported": archive_report[
                "training_only_modules_imported"],
            "vendored_market_price_probe": archive_report[
                "vendored_market_price_probe"],
            "games": archive_report["games"],
            "source_vs_archive_trace_parity": parity,
            "manifest": archive_report["manifest"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SMOKE_SEEDS))
    args = parser.parse_args(argv)
    try:
        report = verify_archive(args.archive, seeds=tuple(args.seeds))
    except (FileNotFoundError, VerificationError, OSError, ValueError,
            subprocess.TimeoutExpired) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
