"""Focused BC-E teacher -> JE student distillation tests."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_bc_manager import _records  # noqa: E402

from bc_manager.economics import E_HISTORY_LEGACY, ECONOMIC_DIM  # noqa: E402
from bc_manager.loss import (  # noqa: E402
    GROUP_NAMES,
    mixed_manager_loss,
    teacher_student_loss,
)
from bc_manager.model import DailyManagerTransformer, tiny_manager_config  # noqa: E402
from bc_manager.training import (  # noqa: E402
    TrainingConfig,
    load_checkpoint,
    run_training,
    save_checkpoint,
)
from replay_daily.storage import write_parquet  # noqa: E402


def _write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.parquet"
    write_parquet(_records(), path)
    return path


def _write_teacher(tmp_path: Path, *, variant: str = "E",
                   config=None, history: str = E_HISTORY_LEGACY) -> Path:
    config = config or tiny_manager_config()
    model = DailyManagerTransformer(config, model_variant=variant)
    path = tmp_path / f"teacher-{variant}-{config.count_max}.pt"
    save_checkpoint(
        path, kind="best", epoch=1, model=model, model_config=config,
        training_config=TrainingConfig(), validation_metrics={},
        model_variant=variant,
        e_history_version=history if variant in ("E", "JE") else None)
    return path


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "crop_logits": torch.randn(batch, 5, 5),
        "animal_logits": torch.randn(batch, 3, 5),
        "land_logits": torch.randn(batch, 4),
        "fertilizer_logits": torch.randn(batch, 5, 5),
        "care_logits": torch.randn(batch, 3, 5),
        "sell_presence_logits": torch.randn(batch, 9, 6),
        "sell_quantity_log1p": torch.randn(batch, 9, 6),
    }


def test_distillation_loss_zero_when_outputs_match_and_student_gets_gradients():
    teacher = _inputs()
    student = {key: value.detach().clone().requires_grad_()
               for key, value in teacher.items()}
    total, groups = teacher_student_loss(student, teacher, temperature=2.0)
    assert all(float(groups[name].detach()) == pytest.approx(0.0, abs=2e-6)
               for name in GROUP_NAMES
               if name not in ("sell_presence",))
    assert float(total.detach()) > 0.0  # matching BCE is its entropy minimum
    total.backward()
    assert all(value.grad is not None for value in student.values())


def test_sell_presence_and_quantity_matching_are_minima():
    teacher = _inputs(1)
    student = {key: value.detach().clone().requires_grad_()
               for key, value in teacher.items()}
    _, groups = teacher_student_loss(student, teacher)
    assert float(groups["sell_presence"].detach()) == pytest.approx(
        float(torch.nn.functional.binary_cross_entropy_with_logits(
            teacher["sell_presence_logits"],
            torch.sigmoid(teacher["sell_presence_logits"]))), abs=1e-7)
    assert float(groups["sell_quantity"].detach()) == pytest.approx(
        0.0, abs=1e-7)


def test_mixed_objective_uses_requested_weight():
    hard = torch.tensor(2.0, requires_grad=True)
    distill = torch.tensor(6.0, requires_grad=True)
    assert mixed_manager_loss(hard, distill, 0.25).item() == pytest.approx(3.0)


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_invalid_distill_weight_rejected(weight):
    with pytest.raises(ValueError, match="distill_weight"):
        run_training("does-not-exist.parquet", distill_weight=weight)


def test_invalid_temperature_and_missing_teacher_rejected():
    with pytest.raises(ValueError, match="temperature"):
        run_training("does-not-exist.parquet", distill_temperature=0.0)
    with pytest.raises(ValueError, match="requires"):
        run_training("does-not-exist.parquet", model_variant="JE",
                     distill_weight=0.5)


def test_teacher_variant_must_be_e(tmp_path):
    path = _write_corpus(tmp_path)
    teacher = _write_teacher(tmp_path, variant="JE")
    with pytest.raises(ValueError, match="checkpoint variant.*E"):
        run_training(path, model_config=tiny_manager_config(),
                     model_variant="JE", e_history_version=E_HISTORY_LEGACY,
                     teacher_checkpoint=teacher, distill_weight=0.5,
                     train_dates=("2026-08-17",),
                     val_dates=("2026-08-21",), log=lambda _: None)


def test_teacher_student_contract_rejects_history_count_and_opponent_mismatch(
        tmp_path):
    path = _write_corpus(tmp_path)
    teacher = _write_teacher(tmp_path)
    base = dict(model_variant="JE", teacher_checkpoint=teacher,
                distill_weight=0.5, train_dates=("2026-08-17",),
                val_dates=("2026-08-21",), log=lambda _: None)
    with pytest.raises(ValueError, match="e_history_version"):
        run_training(path, model_config=tiny_manager_config(),
                     e_history_version="E_CORRECTED_V1", **base)

    with pytest.raises(ValueError, match="count_max"):
        run_training(path, model_config=tiny_manager_config(count_max=10),
                     e_history_version=E_HISTORY_LEGACY, **base)

    with pytest.raises(ValueError, match="opponent-board"):
        run_training(path, model_config=tiny_manager_config(
            include_opponent_board=True), e_history_version=E_HISTORY_LEGACY,
                     **base)


def test_teacher_no_grad_student_distillation_gradients(tmp_path):
    config = tiny_manager_config()
    teacher = DailyManagerTransformer(config, model_variant="E").eval()
    student = DailyManagerTransformer(config, model_variant="JE")
    batch = {
        "board_kind": torch.zeros(2, 100, dtype=torch.int16),
        "board_crop": torch.zeros(2, 100, dtype=torch.int8),
        "board_animal": torch.zeros(2, 100, dtype=torch.int8),
        "board_numeric": torch.zeros(2, 100, 11),
        "board_bool": torch.zeros(2, 100, 8, dtype=torch.bool),
        "board_mask": torch.zeros(2, 100, 4, dtype=torch.uint8),
        "scalars": torch.zeros(2, 4), "shed_counts": torch.zeros(2, 12),
        "seed_counts": torch.zeros(2, 5), "carried_counts": torch.zeros(2, 12),
        "unlocked": torch.ones(2, 4, dtype=torch.uint8),
        "market_inventory": torch.zeros(2, 9), "market_prices": torch.zeros(2, 9),
        "shop_counts": torch.zeros(2, 9), "day": torch.zeros(2, dtype=torch.int16),
        "days_remaining": torch.full((2,), 29, dtype=torch.int16),
        "economic_context": torch.zeros(2, ECONOMIC_DIM),
    }
    teacher_outputs = teacher(batch)
    student_outputs = student(batch)
    loss, _ = teacher_student_loss(student_outputs, teacher_outputs)
    loss.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_tiny_je_distillation_smoke_and_metadata(tmp_path):
    corpus = _write_corpus(tmp_path)
    teacher = _write_teacher(tmp_path)
    checkpoint_dir = tmp_path / "student"
    result = run_training(
        corpus, model_config=tiny_manager_config(
            d_model=32, ffn_dim=64, num_heads=1),
        training_config=TrainingConfig(batch_size=2, epochs=1,
                                       checkpoint_dir=str(checkpoint_dir)),
        model_variant="JE", e_history_version=E_HISTORY_LEGACY,
        teacher_checkpoint=teacher, distill_weight=0.5,
        distill_temperature=2.0, train_dates=("2026-08-17",),
        val_dates=("2026-08-21",), log=lambda _: None)
    assert result["history"][0]["train"]["hard_total"] >= 0.0
    assert result["history"][0]["train"]["distill_total"] >= 0.0
    assert result["history"][0]["train"]["combined_total"] >= 0.0
    payload = load_checkpoint(checkpoint_dir / "best.pt")
    metadata = payload["distillation"]
    assert metadata["enabled"] is True
    assert metadata["teacher_checkpoint"] == teacher.name
    assert metadata["teacher_variant"] == "E"
    assert metadata["teacher_e_history_version"] == E_HISTORY_LEGACY
    assert metadata["distill_weight"] == 0.5
    assert metadata["distill_temperature"] == 2.0
    assert metadata["teacher_model_config"] == {
        "d_model": 16, "num_layers": 1, "num_heads": 1, "ffn_dim": 32,
        "dropout": 0.0, "count_max": 100, "include_opponent_board": False,
    }


def test_non_distillation_path_records_zero_soft_metrics(tmp_path):
    result = run_training(
        _write_corpus(tmp_path), model_config=tiny_manager_config(),
        training_config=TrainingConfig(batch_size=2, epochs=1),
        train_dates=("2026-08-17",), val_dates=("2026-08-21",),
        log=lambda _: None)
    train = result["history"][0]["train"]
    assert train["hard_total"] == pytest.approx(train["combined_total"])
    assert train["distill_total"] == pytest.approx(0.0)
    assert result["distillation"]["enabled"] is False
