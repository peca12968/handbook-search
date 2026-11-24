"""Compare two calibration policies on identical, chronologically separated data."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def cli(*args):
    return json.loads(subprocess.check_output([sys.executable, ROOT / "triage.py", *args], cwd=ROOT))


def main():
    reports = {}
    with TemporaryDirectory() as directory:
        for mode in ("global", "per-queue"):
            model = Path(directory) / f"{mode}.json"
            calibration = cli("fit", "examples/train.jsonl", "examples/calibration.jsonl",
                              "--threshold-mode", mode, "--min-precision", "0.9",
                              "--min-calibration-cases", "3", "--model", model)
            holdout = cli("evaluate", "examples/holdout.jsonl", "--model", model)
            reports[mode] = {
                "policy": calibration.get("queue_policies", calibration.get("margin_threshold")),
                "calibration_assigned": calibration["calibration"]["assigned"],
                "holdout_assigned": holdout["assigned"], "holdout_correct": holdout["correct_assignments"],
                "holdout_review": holdout["review"], "holdout_by_candidate_queue": holdout["by_candidate_queue"],
                "holdout_without_unique_candidate": holdout["no_unique_candidate"],
                "assigned_ids": [row["case_id"] for row in holdout["details"] if row["action"] == "assign"],
            }
    shared = set(reports["global"]["assigned_ids"])
    local = set(reports["per-queue"]["assigned_ids"])
    reports["changed_assignments"] = {"now_review": sorted(shared - local), "now_assigned": sorted(local - shared)}
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
