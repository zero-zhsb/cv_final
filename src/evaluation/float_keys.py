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


class CUDAFloatKeyAUROC(FloatKeyAUROC):
    """Optional exact counting acceleration; inference still runs on CPU.

Uses CuPy RawKernel atomic integer counts, then identical CPU AUROC reduction.
https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html
"""
    def __init__(self, low, high, budget_gib=3):
        import cupy as cp
        self.cp, self.low, self.high = cp, low, high
        span = 0 if low is None else high-low+1
        free, _ = cp.cuda.runtime.memGetInfo()
        if span*8 + 128*1024**2 > min(free, budget_gib*1024**3):
            raise MemoryError("Exact CUDA counter exceeds selected budget/free memory")
        cp.get_default_memory_pool().set_limit(size=int((budget_gib+.25)*1024**3))
        self.total = cp.zeros(span, dtype=cp.uint32)
        self.positive = cp.zeros_like(self.total)
        self.zero_positive = self.zero_negative = self.pixels = 0
        self.kernel = cp.RawKernel(r'''
        extern "C" __global__ void count_keys(const unsigned int* keys,
          const unsigned char* mask, unsigned int* total, unsigned int* pos,
          unsigned int low, unsigned long long n) {
          unsigned long long i = (unsigned long long)blockDim.x*blockIdx.x+threadIdx.x;
          if(i<n) { unsigned int k=keys[i]-low; atomicAdd(total+k,1u);
            if(mask[i]) atomicAdd(pos+k,1u); }
        }''', 'count_keys')

    def add(self, scores, mask):
        cp = self.cp
        scores = np.asarray(scores,dtype=np.float32)
        mask = np.asarray(mask,dtype=bool)
        if scores.shape != mask.shape or scores.ndim != 2 or not np.isfinite(scores).all() or (scores<0).any():
            raise ValueError("Expected matching finite nonnegative float32 maps")
        self.pixels += scores.size
        if self.pixels > np.iinfo(np.uint32).max:
            raise ValueError("Counter overflow")
        valid=scores!=0
        self.zero_positive += int((mask & ~valid).sum())
        self.zero_negative += int((~mask & ~valid).sum())
        if not valid.any():
            return
        keys=scores[valid].view(np.uint32)
        if self.low is None or int(keys.min())<self.low or int(keys.max())>self.high:
            raise ValueError("Out-of-scan-range float key")
        device_keys=cp.asarray(keys); device_mask=cp.asarray(mask[valid],dtype=cp.uint8)
        self.kernel(((len(keys)+255)//256,),(256,),
                    (device_keys,device_mask,self.total,self.positive,np.uint32(self.low),np.uint64(len(keys))))
        cp.cuda.get_current_stream().synchronize()

    def compute(self):
        favorable=.5*self.zero_positive*self.zero_negative
        positives,negatives=self.zero_positive,self.zero_negative
        for begin in range(0,len(self.total),1_000_000):
            total=self.total[begin:begin+1_000_000].get()
            positive=self.positive[begin:begin+1_000_000].get()
            occupied=total!=0
            positive=positive[occupied].astype(np.int64)
            negative=total[occupied].astype(np.int64)-positive
            below=negatives+np.cumsum(negative,dtype=np.float64)-negative
            favorable+=float(np.dot(positive.astype(np.float64),below+.5*negative))
            positives+=int(positive.sum());negatives+=int(negative.sum())
        if not positives or not negatives:
            raise ValueError("AUROC requires both classes")
        return favorable/(positives*negatives)

    def close(self):
        self.total=self.positive=None
        self.cp.get_default_memory_pool().free_all_blocks()
