from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from tools.build_runner_compatible_submission import (
    POLICY_CONTRACT_PPO_FROZEN_SELL,
    POLICY_CONTRACT_SINGLE,
    build_submission,
)


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


def test_builder_emits_dual_ppo_contract_and_d4_handoff(tmp_path: Path) -> None:
    base = tmp_path / "base.tar.gz"
    mutable = tmp_path / "ppo.pt"
    frozen = tmp_path / "frozen.pt"
    output = tmp_path / "fixed.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    _write_base_archive(base)
    mutable.write_bytes(b"mutable-ppo-checkpoint")
    frozen.write_bytes(b"frozen-bc-checkpoint")

    result = build_submission(
        base_archive=base,
        checkpoint=mutable,
        frozen_checkpoint=frozen,
        output=output,
        aggressive_sell_all=False,
        label="rl-u10",
        policy_contract=POLICY_CONTRACT_PPO_FROZEN_SELL,
    )
    _extract(output, extracted)

    main = (extracted / "main.py").read_text()
    executor_agent = (extracted / "executor_v0" / "agent.py").read_text()
    manifest = json.loads((extracted / "submission_manifest.json").read_text())

    # Generated entrypoint must be syntactically valid.
    compile(main, "main.py", "exec")

    assert "class RunnerParityProvider" in main
    assert "economic_prev_start = (" in main
    assert "strict=True" in main
    assert "aggressive_sell_all=False" in main
    assert "optional_spare_watering=True" in main
    assert "KAGGRICULTURE_SUBMISSION_STRICT" not in main

    # Critical Issue20 PPO deployment contract: use the full frozen network.
    assert '_FROZEN_CHECKPOINT = _ROOT / "frozen_e.pt"' in main
    assert "frozen_outputs = self._frozen.model(batch)" in main
    assert 'outputs["sell_quantity_log1p"] = ' in main
    assert 'frozen_outputs["sell_quantity_log1p"]' in main

    # Critical opening -> manager handoff parity.
    assert '"workers_hired": 5' in main
    assert '"hire_cost": 12' in main
    assert "previous_execution = dict(_OPENING_PREVIOUS_EXECUTION)" in main

    assert (extracted / "best.pt").read_bytes() == mutable.read_bytes()
    assert (extracted / "frozen_e.pt").read_bytes() == frozen.read_bytes()

    vendored = extracted / "executor_v0" / "_submission_market.py"
    assert vendored.is_file()
    assert "def market_price(" in vendored.read_text()
    assert "from fast_env.market import market_price" not in executor_agent
    assert (
        "from executor_v0._submission_market import market_price"
        in executor_agent
    )

    assert manifest["submission_variant"] == "rl-u10"
    assert manifest["submission_fix"] == "issue20_runner_deployment_parity_v2"
    assert manifest["policy_contract"] == POLICY_CONTRACT_PPO_FROZEN_SELL
    assert manifest["aggressive_sell_all"] is False
    assert manifest["optional_spare_watering"] is True
    assert manifest["opening_handoff_previous_execution"] == {
        "workers_hired": 5,
        "hire_cost": 12,
    }
    assert manifest["ppo_deployment_contract"] == {
        "mutable_network": "best.pt",
        "mutable_outputs": [
            "crop_logits",
            "animal_logits",
            "land_logits",
            "fertilizer_logits",
            "care_logits",
            "sell_presence_logits",
        ],
        "frozen_network": "frozen_e.pt",
        "frozen_outputs": ["sell_quantity_log1p"],
    }
    assert "ppo_full_frozen_network_sell_quantity" in manifest["submission_fixes"]
    assert manifest["frozen_e_checkpoint_sha256"] == hashlib.sha256(
        frozen.read_bytes()
    ).hexdigest()

    file_paths = {record["path"] for record in manifest["files"]}
    assert "best.pt" in file_paths
    assert "frozen_e.pt" in file_paths
    assert "executor_v0/_submission_market.py" in file_paths

    assert result["policy_contract"] == POLICY_CONTRACT_PPO_FROZEN_SELL
    assert result["opening_handoff_previous_execution"] == {
        "workers_hired": 5,
        "hire_cost": 12,
    }
    assert result["checkpoint_sha256"] == hashlib.sha256(
        mutable.read_bytes()
    ).hexdigest()
    assert result["frozen_e_checkpoint_sha256"] == hashlib.sha256(
        frozen.read_bytes()
    ).hexdigest()


def test_builder_single_network_contract_stays_explicit(tmp_path: Path) -> None:
    base = tmp_path / "base.tar.gz"
    checkpoint = tmp_path / "bc.pt"
    output = tmp_path / "single.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    _write_base_archive(base)
    checkpoint.write_bytes(b"bc-checkpoint")

    build_submission(
        base_archive=base,
        checkpoint=checkpoint,
        frozen_checkpoint=None,
        output=output,
        aggressive_sell_all=True,
        label="bc-e-instant-sell",
        policy_contract=POLICY_CONTRACT_SINGLE,
    )
    _extract(output, extracted)

    main = (extracted / "main.py").read_text()
    manifest = json.loads((extracted / "submission_manifest.json").read_text())

    compile(main, "main.py", "exec")
    assert "frozen_checkpoint_path=None" in main
    assert "aggressive_sell_all=True" in main
    assert not (extracted / "frozen_e.pt").exists()
    assert manifest["policy_contract"] == POLICY_CONTRACT_SINGLE
    assert manifest["ppo_deployment_contract"] is None
    assert "frozen_e_checkpoint_sha256" not in manifest


def test_builder_refuses_ppo_contract_without_frozen_checkpoint(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.tar.gz"
    checkpoint = tmp_path / "ppo.pt"
    output = tmp_path / "bad.tar.gz"

    _write_base_archive(base)
    checkpoint.write_bytes(b"ppo")

    with pytest.raises(ValueError, match="requires --frozen-checkpoint"):
        build_submission(
            base_archive=base,
            checkpoint=checkpoint,
            frozen_checkpoint=None,
            output=output,
            aggressive_sell_all=False,
            label="rl-u10",
            policy_contract=POLICY_CONTRACT_PPO_FROZEN_SELL,
        )


def test_builder_refuses_frozen_checkpoint_under_single_contract(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.tar.gz"
    checkpoint = tmp_path / "bc.pt"
    frozen = tmp_path / "frozen.pt"
    output = tmp_path / "bad.tar.gz"

    _write_base_archive(base)
    checkpoint.write_bytes(b"bc")
    frozen.write_bytes(b"frozen")

    with pytest.raises(ValueError, match="does not accept --frozen-checkpoint"):
        build_submission(
            base_archive=base,
            checkpoint=checkpoint,
            frozen_checkpoint=frozen,
            output=output,
            aggressive_sell_all=False,
            label="bc",
            policy_contract=POLICY_CONTRACT_SINGLE,
        )
