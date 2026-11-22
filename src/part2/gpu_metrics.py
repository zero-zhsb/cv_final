"""Exact float32 AUROC via bounded GPU integer counts, without quantization.

Positive finite float32 bit patterns preserve numeric ordering. Zero is counted
separately so the large bit-pattern gap between zero and tiny scores is not stored.
"""
import numpy as np


def gpu_pixel_auc(pairs_factory, device="cuda:0", memory_fraction=0.1):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pixel evaluation requested without CUDA")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    minimum, maximum = None, None
    positives = negatives = zero_positive = zero_negative = 0
    for scores, mask in pairs_factory():
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("GPU float-key counting requires equal-shape finite nonnegative scores")
        valid = scores != 0
        positives += int(mask.sum()); negatives += int(mask.size - mask.sum())
        zero_positive += int((mask & ~valid).sum())
        zero_negative += int((~mask & ~valid).sum())
        if valid.any():
            keys = scores[valid].view(np.uint32)
            lo, hi = int(keys.min()), int(keys.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
    if not positives or not negatives:
        raise ValueError("Pixel AUROC requires both classes")
    if positives + negatives > np.iinfo(np.int32).max:
        raise ValueError("Pixel count exceeds the exact int32 histogram capacity")
    if minimum is None:
        return 0.5, {"backend": "cuda_exact_float32_keys_zero_only", "histogram_bytes": 0}
    span = maximum - minimum + 1
    histogram_bytes = span * 8
    total_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory, _ = torch.cuda.mem_get_info(device)
    if histogram_bytes + 128 * 1024**2 > min(total_memory * memory_fraction, free_memory):
        raise MemoryError("Exact GPU key histograms exceed the selected cap; use the CPU exact evaluator")
    torch.cuda.reset_peak_memory_stats(device)
    totals = torch.zeros(span, dtype=torch.int32, device=device)
    positive_counts = torch.zeros_like(totals)
    for scores, mask in pairs_factory():
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        valid = scores != 0
        keys = scores[valid].view(np.uint32).astype(np.int64) - minimum
        indices = torch.as_tensor(keys, dtype=torch.int64, device=device)
        totals.index_add_(0, indices, torch.ones(indices.numel(), dtype=torch.int32, device=device))
        selected = torch.as_tensor(keys[mask[valid]], dtype=torch.int64, device=device)
        positive_counts.index_add_(0, selected, torch.ones(selected.numel(), dtype=torch.int32, device=device))
    favorable = zero_positive * zero_negative * 0.5
    negatives_below = zero_negative
    counted_positive, counted_negative = zero_positive, zero_negative
    for begin in range(0, span, 1_000_000):
        total = totals[begin:begin+1_000_000].cpu().numpy().astype(np.int64)
        positive = positive_counts[begin:begin+1_000_000].cpu().numpy().astype(np.int64)
        occupied = total != 0
        positive = positive[occupied]
        negative = total[occupied] - positive
        if len(positive):
            below = negatives_below + np.cumsum(negative, dtype=np.float64) - negative
            favorable += np.dot(positive.astype(np.float64), below + negative * 0.5)
            negatives_below += int(negative.sum())
            counted_positive += int(positive.sum()); counted_negative += int(negative.sum())
    if (counted_positive, counted_negative) != (positives, negatives):
        raise RuntimeError("Pixel counts differ between the scan and counting passes")
    details = {"backend": "cuda_exact_float32_keys", "histogram_bytes": histogram_bytes,
               "peak_evaluator_gpu_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
               "positive_pixels": positives, "negative_pixels": negatives,
               "gpu_fraction_cap": memory_fraction}
    return float(favorable / (positives * negatives)), details


def gpu_pixel_auc_positive_keys(pairs_factory, device="cuda:0", memory_fraction=0.1):
    """Count exact ranks in bins defined by the observed positive float32 keys."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pixel evaluation requested without CUDA")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    positive_parts = []
    negatives = zero_negatives = 0
    for scores, mask in pairs_factory():
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("GPU float-key counting requires equal-shape finite nonnegative scores")
        positives = scores[mask]
        positives[positives == 0] = 0  # Give IEEE signed zeros one rank.
        positive_parts.append(positives.view(np.int32))
        negatives += mask.size - positives.size
        zero_negatives += int(((scores == 0) & ~mask).sum())

    positive_keys, positive_counts = np.unique(np.concatenate(positive_parts), return_counts=True)
    del positive_parts
    positive_total = int(positive_counts.sum())
    if not positive_total or not negatives:
        raise ValueError("Pixel AUROC requires both classes")
    if negatives > np.iinfo(np.int32).max:
        raise ValueError("Negative pixel count exceeds exact int32 capacity")

    key_count = len(positive_keys)
    histogram_bytes = (2 * key_count + 1) * np.dtype(np.int32).itemsize
    key_bytes = (key_count + 1) * np.dtype(np.int32).itemsize
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    if histogram_bytes + key_bytes + 128 * 1024**2 > min(total_memory * memory_fraction, free_memory):
        raise MemoryError("Positive-key AUROC exceeds the selected GPU cap")
    torch.cuda.reset_peak_memory_stats(device)
    keys = torch.as_tensor(positive_keys, device=device)
    padded_keys = torch.cat((keys, keys.new_full((1,), torch.iinfo(torch.int32).max)))
    counts = torch.zeros(2 * key_count + 1, dtype=torch.int32, device=device)

    # Even bins contain negatives between positive keys. Odd bins contain exact ties.
    for scores, mask in pairs_factory():
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        nonzero_negative = scores[(scores != 0) & ~mask].view(np.int32)
        for start in range(0, len(nonzero_negative), 262_144):
            values = torch.as_tensor(nonzero_negative[start:start + 262_144], device=device)
            ranks = torch.searchsorted(keys, values)
            bins = ranks.mul(2).add_(padded_keys[ranks] == values)
            counts.index_add_(0, bins, torch.ones_like(values))

    negative_counts = counts.cpu().numpy().astype(np.int64)
    negative_counts[1 if positive_keys[0] == 0 else 0] += zero_negatives
    if int(negative_counts.sum()) != negatives:
        raise RuntimeError("Negative pixel counts differ between evaluation passes")
    tied = negative_counts[1::2]
    below = np.cumsum(negative_counts, dtype=np.int64)[:-1:2]
    favorable = np.dot(positive_counts.astype(np.float64), below + tied * 0.5)
    return float(favorable / (positive_total * negatives)), {
        "backend": "cuda_exact_positive_float32_keys",
        "histogram_bytes": histogram_bytes,
        "positive_unique_keys": key_count,
        "positive_pixels": positive_total,
        "negative_pixels": negatives,
        "peak_evaluator_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "gpu_fraction_cap": memory_fraction,
    }
