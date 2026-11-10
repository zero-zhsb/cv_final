"""Exact pixel AUROC through score/count histograms, without downsampling."""
import numpy as np
import heapq


class PixelAUROC:
    def __init__(self, memory_budget_gib=8):
        self.levels = [[]]
        self.budget = memory_budget_gib * 1024**3

    def _merge(self, parts, retained_entries=0):
        entries = sum(len(part[0]) for part in parts)
        if entries * 72 + retained_entries * 20 > self.budget:
            raise MemoryError("Exact pixel AUROC merge exceeds memory budget; increase it in a suitable CPU slot")
        values = np.concatenate([part[0] for part in parts])
        positive = np.concatenate([part[1] for part in parts])
        negative = np.concatenate([part[2] for part in parts])
        order = np.argsort(values, kind="stable")
        values = values[order]
        positive = positive[order]
        negative = negative[order]
        del order
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        result_values = values[starts]
        positive = np.add.reduceat(positive, starts, dtype=np.uint64)
        negative = np.add.reduceat(negative, starts, dtype=np.uint64)
        if max(int(positive.max()), int(negative.max())) <= np.iinfo(np.uint32).max:
            positive = positive.astype(np.uint32)
            negative = negative.astype(np.uint32)
        return result_values, positive, negative

    def add(self, scores, mask):
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape or scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError("Pixel scores and truth must be finite 2D arrays of equal shape")
        values, inverse, counts = np.unique(scores.reshape(-1), return_inverse=True, return_counts=True)
        positive = np.bincount(inverse[mask.reshape(-1)], minlength=len(values)).astype(np.uint32)
        negative = counts.astype(np.uint32) - positive
        carry = (values, positive, negative)
        level = 0
        while True:
            if level == len(self.levels):
                self.levels.append([])
            self.levels[level].append(carry)
            if len(self.levels[level]) < 4:
                break
            parts, self.levels[level] = self.levels[level], []
            retained = sum(len(part[0]) for bucket in self.levels for part in bucket)
            carry = self._merge(parts, retained)
            del parts
            level += 1

    def compute(self):
        parts = [part for bucket in self.levels for part in bucket]
        if not parts:
            raise ValueError("No pixel observations")
        self.levels = [[]]
        heap = [(len(part[0]), index, part) for index, part in enumerate(parts)]
        del parts
        heapq.heapify(heap)
        serial = len(heap)
        while len(heap) > 1:
            _, _, first = heapq.heappop(heap)
            _, _, second = heapq.heappop(heap)
            retained = sum(entry[0] for entry in heap)
            merged = self._merge([first, second], retained)
            del first, second
            heapq.heappush(heap, (len(merged[0]), serial, merged))
            serial += 1
        _, _, (_, positive, negative) = heap[0]
        total_positive, total_negative = int(positive.sum()), int(negative.sum())
        if not total_positive or not total_negative:
            raise ValueError("AUROC requires both normal and defective pixels")
        negatives_below = np.cumsum(negative, dtype=np.float64) - negative
        favorable = np.dot(positive.astype(np.float64), negatives_below + negative * 0.5)
        return float(favorable / (total_positive * total_negative))
