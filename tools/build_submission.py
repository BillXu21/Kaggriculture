"""Build a deterministic, small BC-E V0.7 Kaggle submission archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from typing import Iterable

REQUIRED_PACKAGES = (
    "executor_v0",
    "bc_manager",
    "opening_book",
    "oracle",
    "replay_daily",
    "fast_env",
)
CHECKPOINT_SHA256 = (
    "F4B029D3E463ABA1DB0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2"
).lower()
MANIFEST_FORMAT = "bc_e_v07_submission_v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for package in REQUIRED_PACKAGES:
        package_root = repo_root / package
        if not package_root.is_dir():
            raise FileNotFoundError(f"required runtime package is missing: {package_root}")
        for path in package_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix == ".py" or (
                package == "opening_book"
                and path.parent.name == "data"
                and path.suffix == ".json"
            ):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def _local_imports(paths: Iterable[Path], repo_root: Path) -> set[str]:
    roots = {
        path.name for path in repo_root.iterdir() if path.is_dir()
    }
    found: set[str] = set()
    for path in paths:
        if path.suffix != ".py":
            continue
        # A few existing runtime files carry a UTF-8 BOM; Python accepts it
        # when loading source, while ast.parse on a decoded string does not.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".", 1)[0])
    return {name for name in found if name in roots}


def _validate_runtime_import_closure(repo_root: Path, source_files: list[Path]) -> None:
    template = repo_root / "tools" / "submission_main.py"
    imports = _local_imports([template, *source_files], repo_root)
    unexpected = sorted(imports - set(REQUIRED_PACKAGES))
    missing = sorted(set(REQUIRED_PACKAGES) - imports)
    if unexpected:
        raise ValueError(
            "submission source imports unlisted local packages: "
            + ", ".join(unexpected)
        )
    if missing:
        raise ValueError(
            "required runtime packages are not represented in the import graph: "
            + ", ".join(missing)
        )


def _tar_info(name: str, size: int, *, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(output: Path, members: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        import gzip

        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(members):
                    payload = members[name]
                    archive.addfile(_tar_info(name, len(payload)), __import__("io").BytesIO(payload))


def build_submission(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    expected_checkpoint_sha256: str | None = CHECKPOINT_SHA256,
) -> dict[str, object]:
    """Build one archive and return its deterministic manifest summary."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    root = root.resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    checkpoint_digest = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is not None and checkpoint_digest != expected_checkpoint_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 {checkpoint_digest} does not match pinned BC-E "
            f"identity {expected_checkpoint_sha256.lower()}"
        )

    template = root / "tools" / "submission_main.py"
    if not template.is_file():
        raise FileNotFoundError(f"tracked submission template not found: {template}")
    source_files = _source_files(root)
    _validate_runtime_import_closure(root, source_files)

    members: dict[str, bytes] = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in source_files
    }
    members["main.py"] = template.read_bytes()
    members["best.pt"] = checkpoint.read_bytes()
    file_manifest = [
        {"path": name, "sha256": sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in sorted(members.items())
    ]
    manifest = {
        "format": MANIFEST_FORMAT,
        "archive_root": True,
        "runtime_packages": list(REQUIRED_PACKAGES),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_member": "best.pt",
        "entrypoint_member": "main.py",
        "native_extensions": [],
        "files": file_manifest,
    }
    members["submission_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    output = Path(output_path).resolve()
    _write_archive(output, members)
    return {
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "checkpoint_sha256": checkpoint_digest,
        "member_count": len(members),
        "members": sorted(members),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local/submissions/bc-e-v07.tar.gz"),
    )
    args = parser.parse_args(argv)
    try:
        result = build_submission(args.checkpoint, args.output)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"BUILD FAILURE: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
