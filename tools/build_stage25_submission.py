"""Build a reproducible native Stage 2.5 Kaggle submission archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import tarfile
from typing import Any
import gzip

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from bc_manager.economics import E_HISTORY_CORRECTED_V1  # noqa: E402
from rl_manager.executor_factory import make_stage25_executor_factory  # noqa: E402
from rl_manager.provenance import opening_provenance  # noqa: E402
from rl_manager.stage25_checkpoint import (  # noqa: E402
    INFERENCE_PAYLOAD_KIND,
    STAGE25_CHECKPOINT_VERSION,
    load_stage25_inference_checkpoint,
)

BUILDER_VERSION = "stage25_submission_builder_v1"
MANIFEST_VERSION = "kaggriculture_stage25_submission_v1"
RUNTIME_PACKAGES = (
    "executor_v0",
    "bc_manager",
    "bc_manager_jax",
    "opening_book",
    "replay_daily",
    "rl_manager",
)
EXTERNAL_RUNTIME_MODULES = ("jax", "jaxlib", "numpy", "pyarrow")
_MARKET_IMPORT = "from fast_env.market import market_price"
_VENDORED_IMPORT = "from executor_v0._submission_market import market_price"


class BuildError(ValueError):
    """Raised when a checkpoint or runtime source violates the archive contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildError(f"cannot identify source repository revision: {exc}") from exc
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
        raise BuildError(f"git returned an invalid source SHA: {revision!r}")
    return revision.lower(), bool(status.strip())


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for package in RUNTIME_PACKAGES:
        package_root = root / package
        if not package_root.is_dir():
            raise FileNotFoundError(f"runtime package is missing: {package_root}")
        for path in package_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix == ".py" or (
                    package == "opening_book" and path.suffix == ".json"):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _member_name(name: str) -> str:
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (path.is_absolute() or windows_path.is_absolute() or windows_path.drive
            or not path.parts or ".." in path.parts):
        raise BuildError(f"unsafe archive member path: {name!r}")
    normalized = "/".join(path.parts)
    if "\\" in name or normalized != name:
        raise BuildError(f"archive member path is not normalized: {name!r}")
    return normalized


def _tar_info(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _write_archive(output: Path, members: dict[str, bytes]) -> None:
    names = [_member_name(name) for name in members]
    if len(names) != len(set(names)):
        raise BuildError("archive contains duplicate normalized member names")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(members):
                    archive.addfile(
                        _tar_info(name, members[name]), io.BytesIO(members[name]))


def _vendor_market_imports(members: dict[str, bytes], root: Path) -> dict[str, str]:
    helper = root / "fast_env" / "market.py"
    if not helper.is_file():
        raise FileNotFoundError(f"pure-Python market helper is missing: {helper}")
    helper_member = "executor_v0/_submission_market.py"
    patched: list[str] = []
    for member in ("executor_v0/agent.py", "executor_v0/strip_market.py"):
        if member not in members:
            raise BuildError(f"runtime archive is missing {member}")
        try:
            source = members[member].decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BuildError(f"runtime source is not UTF-8: {member}") from exc
        count = source.count(_MARKET_IMPORT)
        if count != 1:
            raise BuildError(
                f"expected one lazy fast_env market import in {member}, got {count}")
        members[member] = source.replace(
            _MARKET_IMPORT, _VENDORED_IMPORT).encode("utf-8")
        patched.append(member)
    members[helper_member] = helper.read_bytes()
    return {
        "source": _MARKET_IMPORT,
        "replacement": _VENDORED_IMPORT,
        "patched_members": patched,
        "helper_member": helper_member,
    }


def build_submission(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    label: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and package one native inference checkpoint."""
    root = Path(repo_root or Path(__file__).resolve().parent.parent).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    if not label or not label.strip():
        raise BuildError("label must be nonempty")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    params, metadata = load_stage25_inference_checkpoint(checkpoint)
    del params  # strict loading validates every payload array and metadata field
    if metadata.get("format") != STAGE25_CHECKPOINT_VERSION:
        raise BuildError("checkpoint format is not native Stage 2.5")
    if metadata.get("payload_kind") != INFERENCE_PAYLOAD_KIND:
        raise BuildError("checkpoint payload is not inference parameters")
    if metadata.get("e_history_version") != E_HISTORY_CORRECTED_V1:
        raise BuildError("checkpoint does not use corrected Stage 2.5 E history")

    template = root / "tools" / "stage25_submission_main.py"
    if not template.is_file():
        raise FileNotFoundError(f"submission entrypoint template is missing: {template}")
    files = _source_files(root)
    members = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in files
    }
    vendor_info = _vendor_market_imports(members, root)
    members["main.py"] = template.read_bytes()
    members["stage25.npz"] = checkpoint.read_bytes()

    source_sha, source_dirty = _source_identity(root)
    factory = make_stage25_executor_factory()
    curriculum = metadata["curriculum"]
    model_config = metadata["model_config"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": MANIFEST_VERSION,
        "label": label.strip(),
        "source_repository_sha": source_sha,
        "source_worktree_dirty_at_build": source_dirty,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_member": "stage25.npz",
        "checkpoint_sha256": sha256_bytes(members["stage25.npz"]),
        "checkpoint_native_format": metadata["format"],
        "checkpoint_payload_kind": metadata["payload_kind"],
        "architecture_version": metadata["architecture_version"],
        "observation_schema_version": metadata["observation_schema_version"],
        "action_schema_version": metadata["action_schema_version"],
        "e_history_version": metadata["e_history_version"],
        "curriculum": {
            "identity": curriculum.get("version"),
            "config": curriculum,
        },
        "model_config": model_config,
        "model_architecture_dimensions": {
            key: model_config.get(key)
            for key in ("d_model", "num_layers", "num_heads", "ffn_dim",
                        "dropout", "manager_config")
            if key in model_config
        },
        "executor_profile": factory.effective_profile,
        "opening": opening_provenance("standard_mixed"),
        "inference_mode": "deterministic",
        "archive_builder_version": BUILDER_VERSION,
        "runtime_packages": list(RUNTIME_PACKAGES),
        "required_external_runtime_modules": list(EXTERNAL_RUNTIME_MODULES),
        "jax_versions_at_build": _jax_versions(),
        "vendored_market_imports": vendor_info,
        "members": [
            {"path": name, "sha256": sha256_bytes(payload),
             "bytes": len(payload)}
            for name, payload in sorted(members.items())
        ],
    }
    members["submission_manifest.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_archive(output, members)
    return {
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_sha": source_sha,
        "model_config_summary": {
            "d_model": model_config.get("d_model"),
            "num_layers": model_config.get("num_layers"),
            "num_heads": model_config.get("num_heads"),
            "ffn_dim": model_config.get("ffn_dim"),
            "manager_config": model_config.get("manager_config"),
        },
        "executor_identity": factory.effective_profile,
        "member_count": len(members),
    }


def _jax_versions() -> dict[str, str | None]:
    import jax
    try:
        import jaxlib
        jaxlib_version: str | None = jaxlib.__version__
    except Exception:  # noqa: BLE001
        jaxlib_version = None
    return {"jax": jax.__version__, "jaxlib": jaxlib_version}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    try:
        result = build_submission(
            args.checkpoint, args.output, label=args.label)
    except (BuildError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
