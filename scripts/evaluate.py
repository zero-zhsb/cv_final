"""Common entrypoint: evaluate Part 1/2 with identical completeness and AUROC code."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_part2 import main

if __name__ == "__main__":
    main()
