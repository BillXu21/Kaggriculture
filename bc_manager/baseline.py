"""Train-split-only empirical day baseline for the daily farm manager.

For each day, learns from TRAIN rows only:
- modal (most frequent) crop/animal/fertilizer/CARE count vectors;
- modal resulting unlocked-land count;
- cellwise-modal sell presence [9, 6];
- one typical positive bounded sell quantity statistic (mean of the true
  bounded quantities over cells with presence, rounded to int >= 0).

Prediction is a per-day lookup. Unseen days fall back to a documented global
statistic computed across all training rows (all days 0..29 are normally
present). `fit` never receives or retains validation targets; `predict`
takes only day indices.
"""

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import group_metrics, sell_metrics

_TARGET_KEYS = (
    "crop_target", "animal_target", "land_count", "fertilizer_target",
    "care_target", "sell_presence", "sell_quantity_bounded",
)


def _modal_vector(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows)
    if values.shape[0] == 0:
        raise ValueError("cannot fit a baseline with zero rows")
    if values.ndim == 1:
        counts = Counter(values.tolist())
        return np.asarray(counts.most_common(1)[0][0])
    counts = Counter(map(tuple, values.tolist()))
    return np.asarray(counts.most_common(1)[0][0], dtype=values.dtype)


def _cellwise_modal_presence(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=bool)
    if values.shape[0] == 0:
        raise ValueError("cannot fit a baseline with zero rows")
    # Ties deliberately choose False: a positive sell must win a strict
    # majority, avoiding fabricated positive actions on a 50/50 split.
    return np.count_nonzero(values, axis=0) > (values.shape[0] / 2)


class DayBaseline:
    """Deterministic per-day empirical predictor (train split only)."""

    def __init__(self) -> None:
        self._per_day: dict[int, dict[str, Any]] = {}
        self._global: dict[str, Any] | None = None

    # ------------------------------------------------------------- fit

    def fit(self, days: Sequence[int],
            targets: Mapping[str, np.ndarray]) -> "DayBaseline":
        missing = [k for k in _TARGET_KEYS if k not in targets]
        if missing:
            raise ValueError(f"fit targets missing keys: {missing}")
        days_arr = np.asarray(days, dtype=np.int16)
        if len(days_arr) != len(targets[_TARGET_KEYS[0]]):
            raise ValueError("days and targets row counts differ")
        if len(days_arr) == 0:
            raise ValueError("cannot fit a baseline with zero rows")
        for key in _TARGET_KEYS[1:]:
            if len(targets[key]) != len(days_arr):
                raise ValueError(f"target row count differs for {key}")

        per_day: dict[int, list[int]] = {}
        for i, d in enumerate(days_arr.tolist()):
            per_day.setdefault(d, []).append(i)

        stats: dict[int, dict[str, Any]] = {}
        for d, idxs in per_day.items():
            sel = np.asarray(idxs, dtype=np.intp)
            qty = targets["sell_quantity_bounded"][sel]
            presence = targets["sell_presence"][sel]
            positive = qty[presence]
            typical_qty = int(round(float(np.mean(positive)))) \
                if positive.size else 0
            stats[d] = {
                "crop": _modal_vector(targets["crop_target"][sel]),
                "animal": _modal_vector(targets["animal_target"][sel]),
                "land": _modal_vector(targets["land_count"][sel]),
                "fertilizer": _modal_vector(targets["fertilizer_target"][sel]),
                "care": _modal_vector(targets["care_target"][sel]),
                "presence": _cellwise_modal_presence(presence),
                "typical_positive_sell_quantity": max(typical_qty, 0),
            }

        global_qty = targets["sell_quantity_bounded"][
            targets["sell_presence"]]
        self._global = {
            "crop": _modal_vector(targets["crop_target"]),
            "animal": _modal_vector(targets["animal_target"]),
            "land": _modal_vector(targets["land_count"]),
            "fertilizer": _modal_vector(targets["fertilizer_target"]),
            "care": _modal_vector(targets["care_target"]),
            "presence": _cellwise_modal_presence(targets["sell_presence"]),
            "typical_positive_sell_quantity":
                int(round(float(np.mean(global_qty)))) if global_qty.size
                else 0,
        }
        self._per_day = stats
        return self

    # --------------------------------------------------------- predict

    def predict(self, days: Sequence[int]) -> dict[str, np.ndarray]:
        if self._global is None:
            raise RuntimeError("DayBaseline.predict called before fit")
        days_arr = np.asarray(days, dtype=np.int16)
        n = len(days_arr)
        out: dict[str, np.ndarray] = {
            "crop_target": np.zeros((n, len(self._global["crop"])),
                                    dtype=np.int32),
            "animal_target": np.zeros((n, len(self._global["animal"])),
                                      dtype=np.int32),
            "land_count": np.zeros(n, dtype=np.int32),
            "fertilizer_target": np.zeros((n, len(self._global["fertilizer"])),
                                          dtype=np.int32),
            "care_target": np.zeros((n, len(self._global["care"])),
                                    dtype=np.int32),
            "sell_presence": np.zeros_like(self._global["presence"],
                                           dtype=bool)[None].repeat(n, 0),
            "sell_quantity_bounded": np.zeros(
                (n,) + self._global["presence"].shape, dtype=np.int32),
        }
        for i, d in enumerate(days_arr.tolist()):
            stats = self._per_day.get(d, self._global)  # documented fallback
            out["crop_target"][i] = stats["crop"]
            out["animal_target"][i] = stats["animal"]
            out["land_count"][i] = stats["land"]
            out["fertilizer_target"][i] = stats["fertilizer"]
            out["care_target"][i] = stats["care"]
            out["sell_presence"][i] = stats["presence"]
            qty = np.zeros_like(out["sell_quantity_bounded"][i])
            qty[stats["presence"]] = stats["typical_positive_sell_quantity"]
            out["sell_quantity_bounded"][i] = qty
        out["sell_quantity_log1p"] = np.log1p(
            out["sell_quantity_bounded"].astype(np.float64)).astype(np.float32)
        return out


def evaluate_baseline(baseline: DayBaseline, days: Sequence[int],
                      targets: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Group metrics + sell metrics for baseline predictions on one split."""
    pred = baseline.predict(days)
    report: dict[str, Any] = {}
    for key in ("crop_target", "animal_target", "fertilizer_target",
                "care_target"):
        report[key] = group_metrics(targets[key], pred[key])
    report["land_count"] = group_metrics(np.asarray(targets["land_count"]),
                                         pred["land_count"])
    report["sells"] = sell_metrics(targets["sell_quantity_bounded"],
                                   pred["sell_quantity_bounded"])
    return report
