"""Bounded evaluation pool; adopts a previous evaluator without repeating its work."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--pixel-device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    args.run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    lock = (args.run / "evaluation_pool.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    # The preceding serial controller can be stopped while its current child
    # finishes normally. Match only this user's exact run and evaluator script.
    for text in subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid="], text=True).split():
        pid = int(text)
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\x00").split("\x00")
            if not any(Path(a).name == "evaluate_part2.py" for a in argv): continue
            if Path(argv[argv.index("--run") + 1]).resolve() != args.run: continue
            key = argv[argv.index("--configuration") + 1]
            active[key] = {"pid": pid, "process": None}
        except (OSError, ValueError, IndexError):
            continue
    print(f"Adopted {len(active)} existing evaluator(s)", flush=True)
    log_root = args.run / "evaluation_logs"
    log_root.mkdir(exist_ok=True)
    previous_count = -1
    while True:
        meta = json.loads((args.run / "run.json").read_text())
        completed = [x["configuration"] for x in meta.get("configurations", [])]
        reports = {p.stem.replace("__", "/"): p for p in (args.run / "metrics").glob("*.json")} if (args.run / "metrics").exists() else {}
        for key, record in list(active.items()):
            if record["process"] is not None:
                code = record["process"].poll()
                alive = code is None
            else:
                try:
                    path = Path(f"/proc/{record['pid']}")
                    argv = (path / "cmdline").read_bytes()
                    alive = (b"evaluate_part2.py" in argv and str(args.run).encode() in argv
                             and key.encode() in argv and (path / "stat").read_text().split()[2] != "Z")
                except OSError:
                    alive = False
                code = 0 if (args.run / "metrics" / (key.replace("/", "__") + ".json")).is_file() else 1
            if alive: continue
            if code != 0 or not (args.run / "metrics" / (key.replace("/", "__") + ".json")).is_file():
                atomic_json(args.run / "evaluation_status.json", {"status": "failed", "configuration": key, "pid": record["pid"]})
                raise SystemExit(f"Evaluator failed for {key}; inspect {log_root}")
            print(f"EVALUATED {key}", flush=True)
            del active[key]
        for key in completed:
            if len(active) >= args.workers: break
            if key in reports or key in active: continue
            # A child may have atomically published its report after the snapshot
            # above and then exited. Recheck before scheduling the same key again.
            if (args.run / "metrics" / (key.replace("/", "__") + ".json")).is_file(): continue
            with (log_root / (key.replace("/", "__") + ".log")).open("ab") as log:
                process = subprocess.Popen(["nice", "-n", "19", sys.executable, str(root / "scripts/evaluate_part2.py"),
                    "--run", str(args.run), "--data-root", str(args.data_root), "--configuration", key,
                    "--memory-budget-gib", "8", "--pixel-device", args.pixel_device], stdout=log, stderr=log)
            active[key] = {"pid": process.pid, "process": process}
            print(f"EVALUATING {key} pid={process.pid}", flush=True)
        if len(reports) != previous_count and reports:
            values = [json.loads(path.read_text()) for _, path in sorted(reports.items())]
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
            writer.writeheader(); writer.writerows(values)
            temporary = args.run / "summary_configurations.csv.tmp"
            temporary.write_text(buffer.getvalue()); temporary.replace(args.run / "summary_configurations.csv")
            previous_count = len(reports)
        expected = len(meta["categories"]) * len(meta["shots"]) * len(meta["seeds"])
        status = "completed" if meta["status"] == "completed" and len(reports) == expected and not active else "running"
        atomic_json(args.run / "evaluation_status.json", {"status": status, "evaluated": len(reports), "expected": expected,
            "workers": args.workers, "active_configurations": list(active)})
        if status == "completed":
            print(f"EVALUATION_COMPLETED {len(reports)}/{expected}", flush=True)
            break
        if meta["status"] not in {"running", "completed"} and not active:
            raise SystemExit(f"Inference ended with {meta['status']}; evaluated {len(reports)}/{expected}")
        time.sleep(3)


if __name__ == "__main__":
    main()
