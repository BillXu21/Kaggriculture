from __future__ import annotations

import gzip
import io
from pathlib import Path
import tarfile

import pytest

from tools.build_submission import build_submission
from tools.verify_submission import VerificationError, extract_fresh, verify_archive


ROOT = Path(__file__).resolve().parents[1]


def _archive_bytes(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(members):
                    payload = members[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))


def test_builder_is_deterministic_and_excludes_junk(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    first = tmp_path / "one.tar.gz"
    second = tmp_path / "two.tar.gz"

    first_report = build_submission(
        checkpoint, first, repo_root=ROOT, expected_checkpoint_sha256=None
    )
    second_report = build_submission(
        checkpoint, second, repo_root=ROOT, expected_checkpoint_sha256=None
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["archive_sha256"] == second_report["archive_sha256"]
    members = _archive_bytes(first)
    assert members["best.pt"] == b"test-checkpoint"
    assert "main.py" in members
    assert "submission_manifest.json" in members
    assert all("__pycache__" not in name for name in members)
    assert all(not name.endswith((".pyc", ".pyo", ".pyd")) for name in members)
    assert all("Kaggriculture-executor-v07" not in name for name in members)
    assert "artifacts/local/submissions" not in members


def test_builder_requires_pinned_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        build_submission(checkpoint, tmp_path / "submission.tar.gz", repo_root=ROOT)


def test_fresh_extraction_rejects_nonempty_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    archive = tmp_path / "submission.tar.gz"
    build_submission(
        checkpoint, archive, repo_root=ROOT, expected_checkpoint_sha256=None
    )
    target = tmp_path / "extract"
    target.mkdir()
    (target / "leftover").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(VerificationError, match="not empty"):
        extract_fresh(archive, target)


def test_omitting_fast_env_fails_before_late_game_behavior(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    original = tmp_path / "submission.tar.gz"
    build_submission(
        checkpoint, original, repo_root=ROOT, expected_checkpoint_sha256=None
    )
    members = {
        name: payload
        for name, payload in _archive_bytes(original).items()
        if not name == "fast_env" and not name.startswith("fast_env/")
    }
    omitted = tmp_path / "omitted-fast-env.tar.gz"
    _write_archive(omitted, members)

    with pytest.raises(VerificationError, match="fast_env") as error:
        verify_archive(omitted, repository_root=ROOT)
    assert "ModuleNotFoundError" in str(error.value)
