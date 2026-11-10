"""Evaluate completed configurations serially while one inference process runs."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--memory-budget-gib", type=float, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    while True:
        metadata = json.loads((args.run / "run.json").read_text())
        completed = metadata.get("configurations", [])
        for configuration in completed:
            key = configuration["configuration"]
            result = args.run / "metrics" / (key.replace("/", "__") + ".json")
            if result.exists():
                continue
            (args.run / "evaluation_status.json").write_text(json.dumps({"status": "running", "configuration": key}))
            subprocess.run(["nice", "-n", "19", sys.executable, str(root / "scripts/evaluate_part2.py"),
                "--run", str(args.run), "--data-root", str(args.data_root), "--configuration", key,
                "--memory-budget-gib", str(args.memory_budget_gib)], check=True)
            reports = [json.loads(p.read_text()) for p in sorted((args.run / "metrics").glob("*.json"))]
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(reports[0]))
            writer.writeheader(); writer.writerows(reports)
            temporary = args.run / "summary_configurations.csv.tmp"
            temporary.write_text(buffer.getvalue())
            temporary.replace(args.run / "summary_configurations.csv")
            print(f"METRICS_READY {len(reports)} configurations", flush=True)
        if metadata["status"] != "running":
            expected = len(metadata["categories"]) * len(metadata["shots"]) * len(metadata["seeds"])
            reports = list((args.run / "metrics").glob("*.json")) if (args.run / "metrics").exists() else []
            status = "completed" if metadata["status"] == "completed" and len(reports) == expected else "incomplete"
            (args.run / "evaluation_status.json").write_text(json.dumps({"status": status, "evaluated": len(reports), "expected": expected}))
            print(f"EVALUATION_{status.upper()} {len(reports)}/{expected}", flush=True)
            if status != "completed":
                raise SystemExit(1)
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
