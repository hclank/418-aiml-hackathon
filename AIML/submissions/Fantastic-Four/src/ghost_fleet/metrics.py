from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def binary_metrics_at_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    predictions = scores >= threshold
    positives = targets == 1
    negatives = targets == 0

    tp = int(np.logical_and(predictions, positives).sum())
    fp = int(np.logical_and(predictions, negatives).sum())
    tn = int(np.logical_and(~predictions, negatives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())

    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    tpr = recall
    tnr = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (tpr + tnr)

    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        balanced_accuracy=float(balanced_accuracy),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def sweep_best_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    resolution: int = 101,
    min_threshold: float = 0.01,
) -> BinaryMetrics:
    thresholds = np.linspace(min_threshold, 1.0, resolution)
    best = binary_metrics_at_threshold(scores, targets, 0.5)
    for threshold in thresholds:
        current = binary_metrics_at_threshold(scores, targets, float(threshold))
        if (
            current.f1 > best.f1
            or (current.f1 == best.f1 and current.balanced_accuracy > best.balanced_accuracy)
        ):
            best = current
    return best
