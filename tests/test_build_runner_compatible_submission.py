from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from tools.build_runner_compatible_submission import build_submission


def _write_base_archive(path: Path) -> None:
    members = {
        "main.py": b"# old entrypoint\n",
        "best.pt": b"old-checkpoint",
        "executor_v0/__init__.py": b"# package\n",
        "executor_v0/agent.py": (
            b"def _probe():\n"
            b"    from fast_env.market import market_price\n"
            b"    return market_price('WHEAT', 10000)\n"
        ),
    }
    manifest = {
        "format": "test",
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in sorted(members.items())
        ],
    }
    members["submission_manifest.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for name, payload in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = 0
                    archive.addfile(info, __import__("io").BytesIO(payload))


def _extract(archive_path: Path, output: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output)


def test_builder_hardcodes_runner_compat_and_strict_runtime(tmp_path: Path) -> None:
    base = tmp_path / "base.tar.gz"
    checkpoint = tmp_path / "ppo.pt"
    output = tmp_path / "fixed.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    _write_base_archive(base)
    checkpoint.write_bytes(b"new-ppo-checkpoint")

    result = build_submission(
        base_archive=base,
        checkpoint=checkpoint,
        output=output,
        aggressive_sell_all=False,
        label="rl-u50",
    )
    _extract(output, extracted)

    main = (extracted / "main.py").read_text()
    executor_agent = (extracted / "executor_v0" / "agent.py").read_text()
    manifest = json.loads((extracted / "submission_manifest.json").read_text())

    assert "class RunnerParityProvider" in main
    assert "economic_prev_start = (" in main
    assert "strict=True" in main
    assert "aggressive_sell_all=False" in main
    assert "KAGGRICULTURE_SUBMISSION_STRICT" not in main
    assert (extracted / "best.pt").read_bytes() == checkpoint.read_bytes()

    vendored = extracted / "executor_v0" / "_submission_market.py"
    assert vendored.is_file()
    assert "def market_price(" in vendored.read_text()
    assert "from fast_env.market import market_price" not in executor_agent
    assert (
        "from executor_v0._submission_market import market_price"
        in executor_agent
    )

    assert manifest["submission_variant"] == "rl-u50"
    assert manifest["submission_fix"] == "legacy_runner_economic_context_parity"
    assert manifest["aggressive_sell_all"] is False
    assert manifest["vendored_runtime_dependencies"] == [
        "executor_v0/_submission_market.py",
    ]
    assert manifest["patched_runtime_imports"] == {
        "from fast_env.market import market_price":
            "from executor_v0._submission_market import market_price",
    }
    assert result["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()


def test_builder_can_emit_instant_sell_control(tmp_path: Path) -> None:
    base = tmp_path / "base.tar.gz"
    checkpoint = tmp_path / "bc.pt"
    output = tmp_path / "insta.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    _write_base_archive(base)
    checkpoint.write_bytes(b"bc-checkpoint")

    build_submission(
        base_archive=base,
        checkpoint=checkpoint,
        output=output,
        aggressive_sell_all=True,
        label="bc-e-instant-sell",
    )
    _extract(output, extracted)

    main = (extracted / "main.py").read_text()
    executor_agent = (extracted / "executor_v0" / "agent.py").read_text()
    manifest = json.loads((extracted / "submission_manifest.json").read_text())

    assert "aggressive_sell_all=True" in main
    assert (extracted / "executor_v0" / "_submission_market.py").is_file()
    assert "from fast_env.market import market_price" not in executor_agent
    assert manifest["aggressive_sell_all"] is True
