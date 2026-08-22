"""Small reusable NumPy metrics for sparse daily-manager targets.

All functions accept array-likes and return plain floats. Zero-denominator
behavior is explicit: `nonzero_recall` with no true-nonzero elements returns
`zero_denominator_value` (default 0.0); positive-cell quantity MAE/log-MAE
with no positive true cells return 0.0. No balancing/focal research here.
"""

from typing import Any

import numpy as np


def _pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    return t, p


def exact_match_rate(y_true: Any, y_pred: Any) -> float:
    """Fraction of rows whose entire target vector is predicted exactly."""
    t, p = _pair(y_true, y_pred)
    if t.shape[0] == 0:
        return 0.0
    return float(np.mean(np.all(t == p, axis=tuple(range(1, t.ndim)))))


def exact_accuracy(y_true: Any, y_pred: Any) -> float:
    """Elementwise exact count accuracy, including zero-valued cells."""
    t, p = _pair(y_true, y_pred)
    if t.size == 0:
        return 0.0
    return float(np.mean(t == p))


def mae(y_true: Any, y_pred: Any) -> float:
    t, p = _pair(y_true, y_pred)
    if t.size == 0:
        return 0.0
    return float(np.mean(np.abs(t.astype(np.float64) - p.astype(np.float64))))


def true_nonzero_rate(y_true: Any) -> float:
    t = np.asarray(y_true)
    if t.size == 0:
        return 0.0
    return float(np.mean(t != 0))


def pred_nonzero_rate(y_pred: Any) -> float:
    p = np.asarray(y_pred)
    if p.size == 0:
        return 0.0
    return float(np.mean(p != 0))


def nonzero_recall(y_true: Any, y_pred: Any,
                   zero_denominator_value: float = 0.0) -> float:
    """Elementwise recall over true-nonzero cells.

    With zero true-nonzero elements the denominator is zero; the function
    returns `zero_denominator_value` (default 0.0) instead of dividing.
    """
    t, p = _pair(y_true, y_pred)
    positives = int(np.count_nonzero(t))
    if positives == 0:
        return float(zero_denominator_value)
    hits = int(np.count_nonzero((t != 0) & (p != 0)))
    return hits / positives


def group_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Full metric set for one count-target group [N, K] (or [N])."""
    t = np.asarray(y_true)
    if t.ndim == 1:
        t = t[:, None]
        p1 = np.asarray(y_pred)[:, None]
    else:
        p1 = y_pred
    return {
        "exact_accuracy": exact_accuracy(t, p1),
        "exact_match": exact_match_rate(t, p1),
        "mae": mae(t, p1),
        "true_nonzero_rate": true_nonzero_rate(t),
        "pred_nonzero_rate": pred_nonzero_rate(p1),
        "nonzero_recall": nonzero_recall(t, p1),
    }


def sell_presence_accuracy(presence_true: Any, presence_pred: Any) -> float:
    t, p = _pair(presence_true, presence_pred)
    if t.size == 0:
        return 0.0
    return float(np.mean(t == p))


def positive_cell_mae(qty_true_bounded: Any, qty_pred: Any) -> float:
    """MAE over cells where the true bounded quantity is > 0; else 0.0."""
    t, p = _pair(qty_true_bounded, qty_pred)
    mask = t > 0
    if not mask.any():
        return 0.0
    diff = np.abs(t[mask].astype(np.float64) - p[mask].astype(np.float64))
    return float(np.mean(diff))


def positive_cell_log_mae(qty_true_bounded: Any, qty_pred: Any) -> float:
    """log1p-space MAE over true-positive cells; else 0.0."""
    t, p = _pair(qty_true_bounded, qty_pred)
    mask = t > 0
    if not mask.any():
        return 0.0
    diff = np.abs(np.log1p(t[mask].astype(np.float64))
                  - np.log1p(p[mask].astype(np.float64)))
    return float(np.mean(diff))


def sell_metrics(qty_true_bounded: Any, qty_pred_bounded: Any) -> dict[str, float]:
    """Presence accuracy + positive-cell quantity MAE/log-MAE for sells."""
    t, p = _pair(qty_true_bounded, qty_pred_bounded)
    return {
        "presence_accuracy": sell_presence_accuracy(t > 0, p > 0),
        "positive_cell_mae": positive_cell_mae(t, p),
        "positive_cell_log_mae": positive_cell_log_mae(t, p),
    }
