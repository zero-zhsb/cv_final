"""One-off study postprocessing; writes artifacts_ready, never claims scientific review."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = (ROOT / "runs/part3_finalize.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / "runs/part3_finalize_status.json"
    def save(stage, **kwargs):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(), **kwargs}, indent=2) + "\n")
        temp.replace(path)
    try:
        save("waiting_for_full_study")
        deadline = time.monotonic() + 12 * 3600
        while True:
            state = json.loads((ROOT / "runs/part3_study_status.json").read_text())
            if state["stage"] == "failed":
                raise RuntimeError("Study failed: " + state.get("error", "unknown"))
            if state["stage"] == "completed":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Study is still incomplete; inspect before rerunning finalization")
            time.sleep(30)
        for script in ["summarize_part3.py", "analyze_part3_cases.py", "benchmark_part3.py"]:
            save("postprocessing", script=script)
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
        save("artifacts_ready", review_required=["Inspect all five paired cases", "Verify positive and negative per-category results", "Write evidence-based conclusions", "Commit and publish reviewed results"])
    except Exception as exc:
        save("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
