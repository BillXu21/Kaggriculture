"""BC training/evaluation loop for the daily-manager Transformer.

In-RAM pipeline over the accepted schema-v2 Arrow adapter:
Parquet -> compact NumPy (once) -> compact torch tensors -> DataLoader ->
AdamW epochs with gradient clipping and optional CUDA AMP -> sparse
diagnostics beside the train-split-only day baseline -> best/last
checkpoints. No scheduler, DDP, sweep, or download infrastructure.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapter import load_train_val
from .baseline import DayBaseline, evaluate_baseline
from .constants import (
    ANIMAL_ORDER,
    MIN_SCORE_DEFAULT,
    TRAIN_DATES_DEFAULT,
    VAL_DATES_DEFAULT,
)
from .loss import GROUP_NAMES, ManagerLossConfig, manager_loss
from .metrics import group_metrics, nonzero_recall, sell_presence_accuracy
from .model import (
    DailyManagerTransformer,
    ManagerConfig,
    predict_counts,
    predict_land,
)

CHECKPOINT_FORMAT = "bc_manager_checkpoint_v1"
BEST_CHECKPOINT_NAME = "best.pt"
LAST_CHECKPOINT_NAME = "last.pt"


@dataclass
class TrainingConfig:
    """Simple fixed training configuration (no sweeps/schedulers)."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    epochs: int = 30
    gradient_clip: float = 1.0
    seed: int = 0
    num_workers: int = 0
    # None = automatic: enabled only when the resolved device is CUDA.
    use_amp: bool | None = None
    early_stopping_patience: int | None = None
    checkpoint_dir: str | None = None
    save_last: bool = True
    loss_weights: ManagerLossConfig = field(default_factory=ManagerLossConfig)

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if self.gradient_clip <= 0.0:
            raise ValueError(
                f"gradient_clip must be positive, got {self.gradient_clip}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.early_stopping_patience is not None \
                and self.early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience must be >= 1 when set, got "
                f"{self.early_stopping_patience}")


# ------------------------------------------------------------ in-RAM data


def arrays_to_tensors(
    inputs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    include_opponent_board: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert adapter arrays once into exact model-input/target tensors."""
    input_keys = sorted(inputs.keys())
    target_keys = ("crop_target", "animal_target", "land_count",
                   "fertilizer_target", "care_target", "sell_presence",
                   "sell_quantity_log1p")
    if not include_opponent_board:
        # Opponent PUBLIC arrays are simply excluded when disabled.
        input_keys = [k for k in input_keys if not k.startswith("opp_")]
    missing = [k for k in target_keys if k not in targets]
    if missing:
        raise ValueError(f"adapter targets missing keys: {missing}")

    def convert(arrays: Mapping[str, np.ndarray], keys: Sequence[str]):
        return {
            key: torch.from_numpy(np.ascontiguousarray(arrays[key]))
            for key in keys
        }

    return convert(inputs, input_keys), convert(targets, target_keys)


class ManagerTorchDataset(Dataset):
    """In-RAM slice view over converted tensors; no rich dicts per item."""

    def __init__(self, inputs: Mapping[str, torch.Tensor],
                 targets: Mapping[str, torch.Tensor]) -> None:
        if not inputs:
            raise ValueError("dataset requires at least one input array")
        n = next(iter(inputs.values())).shape[0]
        for key, value in inputs.items():
            if value.shape[0] != n:
                raise ValueError(f"input row counts differ for {key}")
        for key, value in targets.items():
            if value.shape[0] != n:
                raise ValueError(f"target row counts differ for {key}")
        self.inputs = dict(inputs)
        self.targets = dict(targets)
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor],
                                               dict[str, torch.Tensor]]:
        return ({key: value[index] for key, value in self.inputs.items()},
                {key: value[index] for key, value in self.targets.items()})


# ------------------------------------------------------------- evaluation


def evaluate(model: DailyManagerTransformer, loader: DataLoader,
             device: torch.device,
             loss_config: ManagerLossConfig) -> dict[str, Any]:
    """Deterministic validation pass: group losses + sparse diagnostics."""
    model.eval()
    totals = {name: 0.0 for name in GROUP_NAMES}
    total_loss = 0.0
    rows = 0
    preds: dict[str, list[np.ndarray]] = {}
    trues: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(inputs)
            loss, groups = manager_loss(outputs, targets, loss_config)
            batch = inputs["day"].shape[0]
            rows += batch
            total_loss += float(loss) * batch
            for name in GROUP_NAMES:
                totals[name] += float(groups[name]) * batch
            for task in ("crop_logits", "animal_logits",
                         "fertilizer_logits", "care_logits"):
                preds.setdefault(task, []).append(
                    predict_counts(outputs[task]).cpu().numpy())
                trues.setdefault(task, []).append(
                    targets[task.replace("_logits", "_target")].cpu().numpy())
            preds.setdefault("land", []).append(
                predict_land(outputs["land_logits"]).cpu().numpy())
            trues.setdefault("land", []).append(
                targets["land_count"].cpu().numpy())
            presence_pred = (torch.sigmoid(outputs["sell_presence_logits"])
                             > 0.5).cpu().numpy()
            preds.setdefault("presence", []).append(presence_pred)
            trues.setdefault("presence", []).append(
                targets["sell_presence"].cpu().numpy())
            preds.setdefault("quantity", []).append(
                outputs["sell_quantity_log1p"].cpu().numpy())
            trues.setdefault("quantity", []).append(
                targets["sell_quantity_log1p"].cpu().numpy())

    report: dict[str, Any] = {
        "rows": rows,
        "total": total_loss / max(rows, 1),
        **{f"group.{name}": totals[name] / max(rows, 1)
           for name in GROUP_NAMES},
    }
    # count-task diagnostics
    for task, name in (("crop_logits", "crop"), ("animal_logits", "animal"),
                       ("fertilizer_logits", "fertilizer"),
                       ("care_logits", "care")):
        p = np.concatenate(preds[task])
        t = np.concatenate(trues[task])
        metrics = group_metrics(t, p)
        for key, value in metrics.items():
            report[f"{name}_{key}"] = float(value)
    # GOOSE zero-collapse visibility: per-animal nonzero diagnostics.
    for k, animal in enumerate(ANIMAL_ORDER):
        p = np.concatenate(preds["animal_logits"])[:, k]
        t = np.concatenate(trues["animal_logits"])[:, k]
        report[f"animal_{animal.lower()}_nonzero_recall"] = \
            nonzero_recall(t, p)
        report[f"animal_{animal.lower()}_pred_nonzero_rate"] = \
            float(np.mean(p != 0))

    land_p = np.concatenate(preds["land"])
    land_t = np.concatenate(trues["land"])
    report["land_exact_accuracy"] = float(np.mean(land_p == land_t))
    report["land_mae"] = float(np.mean(np.abs(land_p - land_t)))

    pres_p = np.concatenate(preds["presence"])
    pres_t = np.concatenate(trues["presence"])
    qty_p = np.concatenate(preds["quantity"])
    qty_t = np.concatenate(trues["quantity"])
    report["sell_presence_accuracy"] = sell_presence_accuracy(pres_t, pres_p)
    report["sell_presence_true_rate"] = float(np.mean(pres_t))
    report["sell_presence_pred_rate"] = float(np.mean(pres_p))
    report["sell_presence_nonzero_recall"] = nonzero_recall(pres_t, pres_p)
    positive = pres_t > 0
    report["sell_positive_quantity_log_mae"] = (
        float(np.mean(np.abs(qty_p[positive] - qty_t[positive])))
        if positive.any() else 0.0)
    return report


def train_one_epoch(model: DailyManagerTransformer,
                    loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, training_config: TrainingConfig,
                    scaler: torch.amp.GradScaler | None) -> dict[str, float]:
    """One shuffled training epoch with clipping (+ optional AMP)."""
    model.train()
    totals = {name: 0.0 for name in GROUP_NAMES}
    total_loss = 0.0
    rows = 0
    started = time.perf_counter()
    for inputs, targets in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            enabled=scaler is not None):
            outputs = model(inputs)
            loss, groups = manager_loss(outputs, targets,
                                        training_config.loss_weights)
        batch = inputs["day"].shape[0]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           training_config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           training_config.gradient_clip)
            optimizer.step()
        rows += batch
        total_loss += float(loss.detach()) * batch
        for name in GROUP_NAMES:
            totals[name] += float(groups[name].detach()) * batch
    elapsed = time.perf_counter() - started
    return {
        "total": total_loss / max(rows, 1),
        **{f"group.{name}": totals[name] / max(rows, 1)
           for name in GROUP_NAMES},
        "elapsed_seconds": elapsed,
        "examples_per_second": rows / max(elapsed, 1e-9),
    }


# ------------------------------------------------------------- checkpoints


def save_checkpoint(path: str | Path, *, kind: str, epoch: int,
                    model: DailyManagerTransformer,
                    model_config: ManagerConfig,
                    training_config: TrainingConfig,
                    validation_metrics: Mapping[str, Any]) -> None:
    """Atomic best/last checkpoint write inside the caller's directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": kind,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "validation_metrics": _jsonable(validation_metrics),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: unrecognized checkpoint format "
            f"{payload.get('format')!r}; expected {CHECKPOINT_FORMAT!r}")
    return payload


def load_model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu",
) -> tuple[DailyManagerTransformer, dict[str, Any]]:
    """Reconstruct the model from the serialized config and state."""
    payload = load_checkpoint(path)
    model_config = ManagerConfig(**payload["model_config"])
    model = DailyManagerTransformer(model_config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) or isinstance(value, int) \
            or isinstance(value, str) or value is None or isinstance(value, bool):
        return value
    return str(value)


# -------------------------------------------------------------- orchestration


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec not in ("cuda", "cpu"):
        raise ValueError(f"unknown device spec {spec!r}; use auto/cuda/cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(spec)


def run_training(
    paths: str | Path | Sequence[str | Path],
    *,
    model_config: ManagerConfig | None = None,
    training_config: TrainingConfig | None = None,
    train_dates: Sequence[str] = TRAIN_DATES_DEFAULT,
    val_dates: Sequence[str] = VAL_DATES_DEFAULT,
    min_score: float = MIN_SCORE_DEFAULT,
    device_spec: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Full in-RAM BC run: load -> tensors -> baseline -> epochs -> ckpts."""
    model_config = model_config if model_config is not None else ManagerConfig()
    training_config = training_config if training_config is not None \
        else TrainingConfig()

    data = load_train_val(paths, train_dates=train_dates,
                          val_dates=val_dates, min_score=min_score,
                          include_opponent=model_config.include_opponent_board)
    if len(data["train"]["meta"]) == 0:
        raise ValueError(
            f"empty train split: no rows selected for dates "
            f"{list(train_dates)} at min_score={min_score}; refusing to "
            f"train on nothing")
    if len(data["val"]["meta"]) == 0:
        raise ValueError(
            f"empty validation split: no rows selected for dates "
            f"{list(val_dates)} at min_score={min_score}; refusing to "
            f"evaluate on nothing")

    seed = training_config.seed
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    device = resolve_device(device_spec)
    amp_enabled = training_config.use_amp
    if amp_enabled is None:
        amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled) \
        if amp_enabled else None

    train_inputs, train_targets = arrays_to_tensors(
        data["train"]["inputs"], data["train"]["targets"],
        include_opponent_board=model_config.include_opponent_board)
    val_inputs, val_targets = arrays_to_tensors(
        data["val"]["inputs"], data["val"]["targets"],
        include_opponent_board=model_config.include_opponent_board)
    train_dataset = ManagerTorchDataset(train_inputs, train_targets)
    val_dataset = ManagerTorchDataset(val_inputs, val_targets)
    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True,
        num_workers=training_config.num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=False,
        num_workers=training_config.num_workers)

    # Train-split-only day baseline, evaluated once on held-out validation.
    baseline = DayBaseline().fit(
        data["train"]["inputs"]["day"], data["train"]["targets"])
    baseline_report = evaluate_baseline(
        baseline, data["val"]["inputs"]["day"], data["val"]["targets"])

    model = DailyManagerTransformer(model_config).to(device)
    param_count = model.trainable_parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr,
                                  weight_decay=training_config.weight_decay)

    log(f"device={device} amp={bool(amp_enabled)} "
        f"params={param_count} "
        f"train_rows={len(train_dataset)} val_rows={len(val_dataset)}")
    log(f"model_config={asdict(model_config)}")
    log(f"training_config={_jsonable(asdict(training_config))}")

    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_total = float("inf")
    best_path = last_path = None
    if training_config.checkpoint_dir:
        ckpt_dir = Path(training_config.checkpoint_dir)
        best_path = ckpt_dir / BEST_CHECKPOINT_NAME
        last_path = ckpt_dir / LAST_CHECKPOINT_NAME if \
            training_config.save_last else None

    started = time.perf_counter()
    stopped_early = False
    for epoch in range(1, training_config.epochs + 1):
        train_report = train_one_epoch(model, train_loader, optimizer,
                                       device, training_config, scaler)
        val_report = evaluate(model, val_loader, device,
                              training_config.loss_weights)
        record = {
            "epoch": epoch,
            "lr": training_config.lr,
            "train": train_report,
            "validation": val_report,
            "baseline_validation": baseline_report,
            "best_epoch": best_epoch if best_epoch > 0 else epoch,
        }
        improved = val_report["total"] < best_total
        if improved:
            best_total = val_report["total"]
            best_epoch = epoch
            record["best_epoch"] = epoch
            if best_path is not None:
                save_checkpoint(best_path, kind="best", epoch=epoch,
                                model=model, model_config=model_config,
                                training_config=training_config,
                                validation_metrics=val_report)
        if last_path is not None:
            save_checkpoint(last_path, kind="last", epoch=epoch, model=model,
                            model_config=model_config,
                            training_config=training_config,
                            validation_metrics=val_report)
        history.append(record)
        groups = " ".join(f"{name}={val_report[f'group.{name}']:.4f}"
                          for name in GROUP_NAMES)
        log(f"epoch={epoch}/{training_config.epochs} "
            f"train_total={train_report['total']:.4f} "
            f"val_total={val_report['total']:.4f} {groups} "
            f"crop_exact={val_report['crop_exact_accuracy']:.3f} "
            f"goose_recall={val_report['animal_goose_nonzero_recall']:.3f} "
            f"land_acc={val_report['land_exact_accuracy']:.3f} "
            f"sell_pres_acc={val_report['sell_presence_accuracy']:.3f} "
            f"ex_s={train_report['examples_per_second']:.0f} "
            f"elapsed={time.perf_counter() - started:.1f}s "
            f"lr={training_config.lr:g} best_epoch={best_epoch}"
            + (" *best*" if improved else ""))

        patience = training_config.early_stopping_patience
        if patience is not None and epoch - best_epoch >= patience:
            stopped_early = True
            log(f"early stopping at epoch {epoch}: no validation-total "
                f"improvement for {patience} epoch(s); best={best_epoch}")
            break

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_total": best_total,
        "stopped_early": stopped_early,
        "baseline_validation": baseline_report,
        "trainable_parameters": param_count,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "model_config": asdict(model_config),
        "training_config": _jsonable(asdict(training_config)),
        "best_checkpoint": str(best_path) if best_path else None,
        "last_checkpoint": str(last_path) if last_path else None,
    }
    log(f"done: best_epoch={best_epoch} "
        f"best_val_total={best_total:.4f} "
        f"best_checkpoint={result['best_checkpoint']} "
        f"last_checkpoint={result['last_checkpoint']}")
    return result
