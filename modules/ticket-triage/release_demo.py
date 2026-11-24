"""Generate the three reproducible deployment decisions used by the portfolio demo."""
from fractions import Fraction
import json
from pathlib import Path
import sys

from release_gate import Limits, compare
from triage import fit, read_cases


ROOT = Path(__file__).resolve().parent


def demonstrate():
    train, calibration, holdout = (read_cases(ROOT / "examples" / f"{name}.jsonl")
                                   for name in ("train", "calibration", "holdout"))
    baseline = fit(train, calibration, Fraction("0.9"), min_cases=3)
    careful = fit(train, calibration, Fraction("0.9"), min_cases=3, threshold_mode="per-queue")
    aggressive = fit(train, calibration, Fraction("0.7"), min_cases=3)
    choices = (
        ("strict", careful, Limits(11, 2)),
        ("staffed", careful, Limits(11, 2, max_extra_reviews=3, max_extra_reviews_per_queue=1)),
        ("aggressive", aggressive, Limits(11, 2)),
    )
    result = []
    for name, candidate, limits in choices:
        report = compare(baseline, candidate, holdout, limits)
        fields = ("passed", "totals", "candidate", "changes")
        result.append({"scenario": name, **{field: report[field] for field in fields},
                       "failed_checks": [check for check in report["checks"] if not check["passed"]]})
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(demonstrate(), ensure_ascii=False, indent=2))
