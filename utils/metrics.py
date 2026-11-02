"""ERMA classification and TATP trajectory metrics."""

import numpy as np
from sklearn.metrics import f1_score

N_CLASSES = 7


def classification(y_true, y_pred):
    """Accuracy, weighted-F1 and macro-F1 in percent; absent classes count as 0."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    labels = list(range(N_CLASSES))
    return {
        "acc": 100.0 * float((y_true == y_pred).mean()),
        "weighted_f1": 100.0 * f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "n": int(len(y_true)),
    }


def iou(pred, target):
    x1 = np.maximum(pred[..., 0], target[..., 0])
    y1 = np.maximum(pred[..., 1], target[..., 1])
    x2 = np.minimum(pred[..., 2], target[..., 2])
    y2 = np.minimum(pred[..., 3], target[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area(pred) + area(target) - inter
    return np.where(union > 1e-8, inter / np.maximum(union, 1e-8), 0.0)


def trajectory(pred, target):
    """ADE/FDE/sADE/sFDE/AIoU/FIoU on x1y1x2y2 pixels.

    ``pred`` is (N, T, 4) or (N, K, T, 4); K samples are averaged first.
    Displacement is the L2 over the 4-D box; s* divide by the true box diagonal.
    """
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    if pred.ndim == 4:
        pred = pred.mean(axis=1)
    disp = np.linalg.norm(pred - target, axis=-1)
    diag = np.hypot(target[..., 2] - target[..., 0], target[..., 3] - target[..., 1])
    valid = diag > 1.0
    scaled = np.where(valid, disp / np.where(valid, diag, 1.0), np.nan)
    ious = iou(pred, target)
    return {
        "ADE": float(disp.mean()), "FDE": float(disp[:, -1].mean()),
        "sADE": float(np.nanmean(scaled)), "sFDE": float(np.nanmean(scaled[:, -1])),
        "AIOU": float(ious.mean()), "FIOU": float(ious[:, -1].mean()),
        "n": int(len(target)),
    }
