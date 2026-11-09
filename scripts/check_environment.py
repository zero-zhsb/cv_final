"""Report environment information without exposing environment variables or credentials."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for package in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn", "faiss-cpu", "faiss-gpu"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        report["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        report["nvidia_smi"] = "unavailable"
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = cuda
        report["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
        report["cuda_available"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_cuda and not cuda:
        parser.exit(1, "CUDA is required but torch cannot access it.\n")


if __name__ == "__main__":
    main()
