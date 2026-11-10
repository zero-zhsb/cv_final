"""Exhaustive float32 GPU squared-L2 search over CPU-normalized DINO features."""
import numpy as np
import torch


class ExactGPUIndex:
    def __init__(self, bank, device="cuda:0", chunk_size=8192):
        bank = np.ascontiguousarray(bank, dtype=np.float32)
        if bank.ndim != 2 or not len(bank) or not np.isfinite(bank).all():
            raise ValueError("Invalid reference feature bank")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.device = device
        self.chunk_size = chunk_size
        self.bank = torch.from_numpy(bank).to(device)
        self.norms = (self.bank * self.bank).sum(dim=1)
        self.ntotal, self.d = bank.shape

    @torch.inference_mode()
    def search(self, queries, k=1):
        if k != 1:
            raise ValueError("This course adapter supports exact k=1 search")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.d:
            raise ValueError("Query feature dimensions differ from the bank")
        if len(queries) == 0:
            return np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.int64)
        query = torch.from_numpy(queries).to(self.device)
        norms = (query * query).sum(dim=1, keepdim=True)
        nearest = torch.full((len(query),), torch.inf, device=self.device, dtype=torch.float32)
        matches = torch.zeros(len(query), device=self.device, dtype=torch.int64)
        for begin in range(0, self.ntotal, self.chunk_size):
            end = min(begin + self.chunk_size, self.ntotal)
            distance = query @ self.bank[begin:end].T
            distance.mul_(-2).add_(norms).add_(self.norms[begin:end].unsqueeze(0)).clamp_(min=0)
            candidate, indices = distance.min(dim=1)
            better = candidate < nearest
            matches = torch.where(better, indices + begin, matches)
            nearest = torch.minimum(nearest, candidate)
        return nearest.cpu().numpy()[:, None], matches.cpu().numpy()[:, None]
