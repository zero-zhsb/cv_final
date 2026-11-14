"""Exact CPU float32 AUROC: integer key counts, no quantization or downsampling.

Nonnegative float32 bit patterns preserve score ordering. Bounds are scanned
before allocation; counts are chunked for reduction, avoiding global sorting.
"""
import numpy as np


class FloatKeyAUROC:
    def __init__(self, low, high, budget_gib=3):
        self.low, self.high = low, high
        span = 0 if low is None else high - low + 1
        if span * 8 > budget_gib * 1024**3:
            raise MemoryError("Exact float-key interval exceeds CPU budget")
        self.total = np.zeros(span, dtype=np.uint32)
        self.positive = np.zeros(span, dtype=np.uint32)
        self.zero_positive = self.zero_negative = 0
        self.pixels = 0

    def add(self, scores, mask):
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape or scores.ndim != 2 or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Expected matching finite nonnegative float32 maps")
        self.pixels += scores.size
        if self.pixels > np.iinfo(np.uint32).max:
            raise ValueError("Counter overflow would occur")
        valid = scores != 0
        self.zero_positive += int((mask & ~valid).sum())
        self.zero_negative += int((~mask & ~valid).sum())
        if not valid.any():
            return
        keys = scores[valid].view(np.uint32)
        if self.low is None or int(keys.min()) < self.low or int(keys.max()) > self.high:
            raise ValueError("Score lies outside scanned bounds")
        values, counts = np.unique(keys, return_counts=True)
        self.total[values.astype(np.int64) - self.low] += counts.astype(np.uint32)
        positive_values, positive_counts = np.unique(keys[mask[valid]], return_counts=True)
        self.positive[positive_values.astype(np.int64) - self.low] += positive_counts.astype(np.uint32)

    def compute(self):
        favorable = 0.5 * self.zero_positive * self.zero_negative
        positives, negatives = self.zero_positive, self.zero_negative
        for begin in range(0, len(self.total), 1_000_000):
            total = self.total[begin:begin + 1_000_000]
            occupied = total != 0
            positive = self.positive[begin:begin + 1_000_000][occupied].astype(np.int64)
            negative = total[occupied].astype(np.int64) - positive
            below = negatives + np.cumsum(negative, dtype=np.float64) - negative
            favorable += float(np.dot(positive.astype(np.float64), below + 0.5 * negative))
            positives += int(positive.sum()); negatives += int(negative.sum())
        if not positives or not negatives:
            raise ValueError("AUROC requires both pixel classes")
        return favorable / (positives * negatives)


def bounds(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Unsupported float scores")
    nonzero = scores[scores != 0].view(np.uint32)
    return (int(nonzero.min()), int(nonzero.max())) if len(nonzero) else (None, None)
