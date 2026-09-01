"""Fail-closed Issue20 PPO JAX -> Torch submission parity audit.

This audit exists because the Issue20 PPO policy is not a single neural
network at deployment time:

- crop/animal/land/fertilizer/care/sell-presence: mutable PPO network
- sell-quantity: FULL immutable frozen-E network

A prior submission copied only the frozen sell-quantity head onto the mutable
PPO trunk. The checkpoint serialized successfully and most decoded plan fields
looked correct, but sell quantities collapsed toward zero in live play.

Run this after exporting Torch checkpoints and before submitting:

    python tools/audit_issue20_ppo_submission.py \
      --ppo-checkpoint /path/to/final.npz \
      --mutable-torch /path/to/ppo.pt \
      --frozen-torch /path/to/frozen_bc_e.pt \
      --archive /path/to/submission.tar.gz

The audit uses a deterministic synthetic E_LEGACY manager batch and checks:
1. mutable JAX outputs ~= mutable Torch outputs for every head;
2. frozen JAX outputs ~= frozen Torch outputs for every head;
3. the deployed combined contract ~= the JAX PPO combined contract;
4. if an archive is supplied, it explicitly declares and contains the dual
   network contract and its checkpoint hashes match.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


OUTPUT_KEYS = (
    "crop_logits",
    "animal_logits",
    "land_logits",
    "fertilizer_logits",
    "care_logits",
    "sell_presence_logits",
    "sell_quantity_log1p",
)

DEFAULT_ATOL = 2e-4
DEFAULT_RTOL = 2e-4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_inputs(batch_size: int) -> dict[str, np.ndarray]:
    """Build a deterministic valid own-only E batch from the canonical spec."""
    from rl_manager.trajectory import e_input_spec

    spec = e_input_spec()
    inputs = {
        name: np.zeros((batch_size,) + shape, dtype=dtype)
        for name, (shape, dtype) in spec.items()
    }

    # Add non-zero context without relying on game-engine state. Keeping board
    # categorical ids at zero makes this robust to enum changes while the
    # global/economic channels still produce a non-trivial representation.
    if "day" in inputs:
        inputs["day"][:] = 4
    if "days_remaining" in inputs:
        inputs["days_remaining"][:] = 25
    if "scalars" in inputs:
        inputs["scalars"][:, 0] = np.asarray(3000.0, dtype=inputs["scalars"].dtype)
        if inputs["scalars"].shape[-1] > 1:
            inputs["scalars"][:, 1] = np.asarray(5, dtype=inputs["scalars"].dtype)
        if inputs["scalars"].shape[-1] > 2:
            inputs["scalars"][:, 2] = np.asarray(5, dtype=inputs["scalars"].dtype)
        if inputs["scalars"].shape[-1] > 3:
            inputs["scalars"][:, 3] = np.asarray(12, dtype=inputs["scalars"].dtype)
    if "unlocked" in inputs and inputs["unlocked"].shape[-1]:
        inputs["unlocked"][:, 0] = 1
    if "market_prices" in inputs:
        inputs["market_prices"][:] = np.asarray(
            100, dtype=inputs["market_prices"].dtype
        )
    if "market_inventory" in inputs:
        inputs["market_inventory"][:] = np.asarray(
            7, dtype=inputs["market_inventory"].dtype
        )
    if "shed_counts" in inputs and inputs["shed_counts"].shape[-1]:
        inputs["shed_counts"][:, 0] = 3
    if "seed_counts" in inputs and inputs["seed_counts"].shape[-1]:
        inputs["seed_counts"][:, 0] = 4
    if "economic_context" in inputs and inputs["economic_context"].shape[-1]:
        # E_LEGACY uses zero/invalid history in production, but a few bounded
        # non-zero audited channels make trunk parity more discriminating.
        width = min(4, inputs["economic_context"].shape[-1])
        inputs["economic_context"][:, :width] = np.asarray(
            [0.1, -0.2, 0.3, 1.0][:width],
            dtype=inputs["economic_context"].dtype,
        )

    return inputs


def _as_numpy_outputs(outputs) -> dict[str, np.ndarray]:
    result = {}
    for key in OUTPUT_KEYS:
        value = outputs[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        result[key] = np.asarray(value, dtype=np.float32)
    return result


def _compare_group(
    label: str,
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float]]:
    metrics = {}
    failures = []
    for key in OUTPUT_KEYS:
        a = expected[key]
        b = actual[key]
        if a.shape != b.shape:
            failures.append(f"{key}: shape {a.shape} != {b.shape}")
            continue
        diff = np.abs(a - b)
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean()) if diff.size else 0.0
        metrics[key] = {"max_abs": max_abs, "mean_abs": mean_abs}
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            failures.append(
                f"{key}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
            )
    if failures:
        raise RuntimeError(
            f"{label} parity FAILED:\n  " + "\n  ".join(failures)
        )
    return metrics


def _audit_archive(
    archive_path: Path,
    *,
    mutable_sha256: str,
    frozen_sha256: str,
) -> dict:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        required = {
            "main.py",
            "best.pt",
            "frozen_e.pt",
            "submission_manifest.json",
            "executor_v0/_submission_market.py",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(
                f"archive is missing required dual-PPO members: {missing}"
            )

        manifest_file = archive.extractfile("submission_manifest.json")
        main_file = archive.extractfile("main.py")
        mutable_file = archive.extractfile("best.pt")
        frozen_file = archive.extractfile("frozen_e.pt")
        if None in (manifest_file, main_file, mutable_file, frozen_file):
            raise RuntimeError("archive member extraction failed")

        manifest = json.loads(manifest_file.read().decode("utf-8"))
        main = main_file.read().decode("utf-8")
        archive_mutable_sha = hashlib.sha256(mutable_file.read()).hexdigest()
        archive_frozen_sha = hashlib.sha256(frozen_file.read()).hexdigest()

    if manifest.get("policy_contract") != "ppo_frozen_sell_quantity":
        raise RuntimeError(
            "archive manifest does not declare policy_contract="
            "'ppo_frozen_sell_quantity'"
        )

    contract = manifest.get("ppo_deployment_contract")
    expected_contract = {
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
    if contract != expected_contract:
        raise RuntimeError(
            "archive PPO deployment contract mismatch:\n"
            f"expected={expected_contract!r}\nactual={contract!r}"
        )

    if manifest.get("opening_handoff_previous_execution") != {
        "workers_hired": 5,
        "hire_cost": 12,
    }:
        raise RuntimeError("archive lacks the audited d4 opening labor handoff")

    required_main_snippets = (
        "frozen_outputs = self._frozen.model(batch)",
        'outputs["sell_quantity_log1p"] = (',
        "previous_execution = dict(_OPENING_PREVIOUS_EXECUTION)",
    )
    for snippet in required_main_snippets:
        if snippet not in main:
            raise RuntimeError(
                f"archive main.py is missing required contract code: {snippet!r}"
            )

    if archive_mutable_sha != mutable_sha256:
        raise RuntimeError(
            "archive best.pt SHA-256 does not match supplied mutable Torch export"
        )
    if archive_frozen_sha != frozen_sha256:
        raise RuntimeError(
            "archive frozen_e.pt SHA-256 does not match supplied frozen checkpoint"
        )
    if manifest.get("checkpoint_sha256") != mutable_sha256:
        raise RuntimeError("manifest checkpoint_sha256 does not match mutable export")
    if manifest.get("frozen_e_checkpoint_sha256") != frozen_sha256:
        raise RuntimeError(
            "manifest frozen_e_checkpoint_sha256 does not match frozen checkpoint"
        )

    return {
        "archive": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "policy_contract": manifest["policy_contract"],
        "opening_handoff_previous_execution":
            manifest["opening_handoff_previous_execution"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--mutable-torch", type=Path, required=True)
    parser.add_argument("--frozen-torch", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()

    for path in (args.ppo_checkpoint, args.mutable_torch, args.frozen_torch):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    import torch

    from bc_manager.economics import E_HISTORY_LEGACY
    from bc_manager_jax.model import ManagerConfig, forward
    from executor_v0.manager import CheckpointPlanProvider
    from rl_manager.ppo_checkpoint import load_ppo_checkpoint

    state, meta = load_ppo_checkpoint(
        args.ppo_checkpoint,
        expected_e_history_version=E_HISTORY_LEGACY,
    )
    config = ManagerConfig(**meta["model_config"])

    mutable = CheckpointPlanProvider(
        args.mutable_torch,
        device="cpu",
        expected_e_history_version=E_HISTORY_LEGACY,
    )
    frozen = CheckpointPlanProvider(
        args.frozen_torch,
        device="cpu",
        expected_e_history_version=E_HISTORY_LEGACY,
    )

    if mutable.model_config != frozen.model_config:
        raise RuntimeError("mutable/frozen Torch ManagerConfig mismatch")
    if dataclasses.asdict(mutable.model_config) != dataclasses.asdict(config):
        raise RuntimeError("Torch/JAX ManagerConfig mismatch")
    if mutable.model_variant != frozen.model_variant:
        raise RuntimeError("mutable/frozen Torch model_variant mismatch")

    inputs = _synthetic_inputs(args.batch_size)

    jax_mutable = _as_numpy_outputs(
        forward(
            state.params["base"],
            inputs,
            config,
            model_variant="E",
        )
    )
    jax_frozen = _as_numpy_outputs(
        forward(
            state.frozen_params,
            inputs,
            config,
            model_variant="E",
        )
    )

    batch = {
        key: torch.from_numpy(np.ascontiguousarray(value))
        for key, value in inputs.items()
    }
    with torch.no_grad():
        torch_mutable = _as_numpy_outputs(mutable.model(batch))
        torch_frozen = _as_numpy_outputs(frozen.model(batch))

    mutable_metrics = _compare_group(
        "mutable JAX -> Torch",
        jax_mutable,
        torch_mutable,
        atol=args.atol,
        rtol=args.rtol,
    )
    frozen_metrics = _compare_group(
        "frozen JAX -> Torch",
        jax_frozen,
        torch_frozen,
        atol=args.atol,
        rtol=args.rtol,
    )

    jax_combined = dict(jax_mutable)
    jax_combined["sell_quantity_log1p"] = jax_frozen["sell_quantity_log1p"]
    torch_combined = dict(torch_mutable)
    torch_combined["sell_quantity_log1p"] = torch_frozen["sell_quantity_log1p"]

    combined_metrics = _compare_group(
        "Issue20 combined deployment",
        jax_combined,
        torch_combined,
        atol=args.atol,
        rtol=args.rtol,
    )

    mutable_sha = _sha256_file(args.mutable_torch)
    frozen_sha = _sha256_file(args.frozen_torch)

    archive_result = None
    if args.archive is not None:
        archive_result = _audit_archive(
            args.archive,
            mutable_sha256=mutable_sha,
            frozen_sha256=frozen_sha,
        )

    result = {
        "status": "PASS",
        "ppo_checkpoint": str(args.ppo_checkpoint),
        "mutable_torch": str(args.mutable_torch),
        "mutable_torch_sha256": mutable_sha,
        "frozen_torch": str(args.frozen_torch),
        "frozen_torch_sha256": frozen_sha,
        "e_history_version": meta["e_history_version"],
        "mutable_parity": mutable_metrics,
        "frozen_parity": frozen_metrics,
        "combined_parity": combined_metrics,
        "archive": archive_result,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
