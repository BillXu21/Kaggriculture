"""Stage-2 focused tests: benchmark metadata, honesty rules, and CLI.

Local CPU smoke only — these numbers are plumbing evidence, never TPU
throughput.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from bc_manager.model import DailyManagerTransformer, tiny_manager_config \
    as torch_tiny_config
from bc_manager.training import TrainingConfig, save_checkpoint as \
    torch_save_checkpoint
from bc_manager_jax.benchmark import build_parser, main, run_benchmark
from bc_manager_jax.model import tiny_manager_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _smoke_argv(**overrides):
    argv = ["--model-config", "tiny", "--device-counts", "1",
            "--batch-sizes", "8", "--warmup", "1", "--iterations", "2",
            "--regimes", "both"]
    argv.extend(overrides.pop("extra", []))
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


def _smoke_args(**overrides):
    args = build_parser().parse_args(_smoke_argv(**overrides))
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    return args


INFERENCE_FIELDS = (
    "inference_compile_seconds",
    "inference_examples_per_second_mean",
    "inference_examples_per_second_best",
)
TRAIN_FIELDS = (
    "train_compile_seconds",
    "train_examples_per_second_mean",
    "train_examples_per_second_best",
)


def test_benchmark_smoke_metadata_and_results(tmp_path):
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    args = _smoke_args()
    report = run_benchmark(args)
    # Output writing lives behind the CLI entrypoint; exercise it directly.
    from bc_manager_jax.benchmark import _write_outputs
    _write_outputs(report, str(json_path), str(csv_path))

    meta = report["metadata"]
    for key in ("jax_version", "backend", "device_count_visible",
                "device_descriptions", "process_count", "sync_policy",
                "reduction_policy", "honesty_note"):
        assert key in meta, key
    assert meta["backend"] == "cpu"

    ok_rows = [r for r in report["results"] if r["status"] == "ok"]
    assert ok_rows, "expected at least one successful benchmark row"
    regimes = {r["regime"] for r in ok_rows}
    assert regimes == {"own", "opponent"}
    for row in ok_rows:
        assert row["token_count"] in (106, 206)
        assert row["param_count"] > 0
        assert row["per_device_batch"] == row["global_batch"]
        assert row["dtype_mode"] == "f32"
        assert row["iterations"] == 2 and row["warmup"] == 1
        # BOTH metric families present and positive on successful rows.
        for field in INFERENCE_FIELDS + TRAIN_FIELDS:
            assert field in row, field
            assert row[field] > 0.0, (field, row[field])
        # Inference must not be silently aliased to training: forward-only
        # work is strictly cheaper per example than forward+backward+update.
        assert (row["inference_examples_per_second_mean"]
                > row["train_examples_per_second_mean"]), row
        assert "mode" not in row  # no ambiguous single-mode field

    # JSON round-trip preserves both metric families.
    payload = json.loads(json_path.read_text("utf-8"))
    assert len(payload["results"]) == len(report["results"])
    for row in payload["results"]:
        if row["status"] == "ok":
            for field in INFERENCE_FIELDS + TRAIN_FIELDS:
                assert field in row and row[field] > 0.0, field

    # CSV round-trip carries both families as columns.
    import csv as csv_module
    with open(csv_path, newline="", encoding="utf-8") as handle:
        csv_rows = list(csv_module.DictReader(handle))
    assert len(csv_rows) == len(ok_rows)
    for row in csv_rows:
        assert row["status"] == "ok"
        for field in INFERENCE_FIELDS + TRAIN_FIELDS:
            assert field in row
            assert float(row[field]) > 0.0, (field, row[field])
        assert float(row["inference_examples_per_second_mean"]) > \
            float(row["train_examples_per_second_mean"])


def test_skipped_row_reports_null_metrics_and_reason():
    from bc_manager_jax.benchmark import run_case

    args = type("Args", (), {})()
    args.dtype = "f32"
    args.warmup = 0
    args.iterations = 1
    args.seed = 0
    row = run_case("own", tiny_manager_config(), None, "random init",
                   device_count=2, global_batch=3, args=args)
    assert row["status"] == "skipped"
    assert "not divisible" in row["reason"]
    for field in INFERENCE_FIELDS + TRAIN_FIELDS:
        assert field in row and row[field] is None, field


def test_benchmark_checkpoint_mode_and_missing_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = DailyManagerTransformer(torch_tiny_config())
    checkpoint = tmp_path / "best.pt"
    torch_save_checkpoint(checkpoint, kind="best", epoch=2, model=model,
                          model_config=torch_tiny_config(),
                          training_config=TrainingConfig(),
                          validation_metrics={"total": 1.0})

    args = _smoke_args(extra=["--checkpoint", str(checkpoint)])
    report = run_benchmark(args)
    rows = [r for r in report["results"] if r["status"] == "ok"]
    assert rows
    # The checkpoint config fixes exactly one regime (own-only here).
    assert {r["regime"] for r in rows} == {"own"}
    assert all(r["token_count"] == 106 for r in rows)
    assert all("bc_manager_checkpoint_v1" in r["source"] for r in rows)

    missing = tmp_path / "does_not_exist.pt"
    args = _smoke_args(extra=["--checkpoint", str(missing)])
    try:
        run_benchmark(args)
        raise AssertionError("missing checkpoint must fail loudly")
    except ValueError as error:
        assert "does not exist" in str(error)


def test_cli_help_documents_kaggle_command():
    completed = subprocess.run(
        [sys.executable, "-m", "bc_manager_jax.benchmark", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0
    help_text = completed.stdout
    assert "/kaggle/working/bc-v0-score2950/best.pt" in help_text
    assert "--device-counts 8" in help_text
    assert "--output-json" in help_text
