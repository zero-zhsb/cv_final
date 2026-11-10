"""Exact pixel AUROC through score/count histograms, without downsampling."""
import numpy as np


class PixelAUROC:
    def __init__(self, memory_budget_gib=8):
        self.values, self.positive, self.negative = [], [], []
        self.entries = 0
        self.budget = memory_budget_gib * 1024**3

    def add(self, scores, mask):
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape or scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError("Pixel scores and truth must be finite 2D arrays of equal shape")
        values, inverse, counts = np.unique(scores.reshape(-1), return_inverse=True, return_counts=True)
        positive = np.bincount(inverse[mask.reshape(-1)], minlength=len(values)).astype(np.uint32)
        negative = counts.astype(np.uint32) - positive
        self.entries += len(values)
        # Sorting/reduction need temporary arrays in addition to stored histograms.
        if self.entries * 72 > self.budget:
            raise MemoryError("Exact pixel AUROC exceeds the selected memory budget; increase it in a suitable CPU slot")
        self.values.append(values)
        self.positive.append(positive)
        self.negative.append(negative)

    def compute(self):
        if not self.values:
            raise ValueError("No pixel observations")
        values = np.concatenate(self.values)
        positive = np.concatenate(self.positive)
        negative = np.concatenate(self.negative)
        self.values.clear(); self.positive.clear(); self.negative.clear()
        order = np.argsort(values)
        sorted_values = values[order]
        positive = positive[order]
        negative = negative[order]
        del values, order
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        del sorted_values
        positive = np.add.reduceat(positive, starts, dtype=np.uint64)
        negative = np.add.reduceat(negative, starts, dtype=np.uint64)
        total_positive, total_negative = int(positive.sum()), int(negative.sum())
        if not total_positive or not total_negative:
            raise ValueError("AUROC requires both normal and defective pixels")
        negatives_below = np.cumsum(negative, dtype=np.float64) - negative
        favorable = np.dot(positive.astype(np.float64), negatives_below + negative * 0.5)
        return float(favorable / (total_positive * total_negative))
