"""Compare two ticket routers on one independent, labeled holdout period."""
import argparse
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

from triage import measure, parse_cases, parse_model, require_later, timestamp


@dataclass(frozen=True)
class Limits:
    min_cases: int
    min_queue_cases: int = 1
    min_assignments: int = 1
    min_precision: Fraction = Fraction(9, 10)
    max_new_misroutes: int = 0
    max_extra_reviews: int = 0
    max_extra_reviews_per_queue: int = 0

    def __post_init__(self):
        if min(self.min_cases, self.min_queue_cases, self.min_assignments) < 1 or not 0 < self.min_precision <= 1:
            raise ValueError("case minimums and precision must be positive; precision cannot exceed one")
        if min(self.max_new_misroutes, self.max_extra_reviews, self.max_extra_reviews_per_queue) < 0:
            raise ValueError("maximums must be nonnegative")


def outcome(row):
    if row["action"] == "review":
        return "review"
    return "correct" if row["queue"] == row["expected"] else "misroute"


def summarize(pairs):
    baseline_reviews = candidate_reviews = new_misroutes = 0
    for baseline, candidate in pairs:
        before, after = outcome(baseline), outcome(candidate)
        baseline_reviews += before == "review"
        candidate_reviews += after == "review"
        new_misroutes += after == "misroute" and candidate["queue"] != baseline["queue"]
    return {"cases": len(pairs), "baseline_reviews": baseline_reviews, "candidate_reviews": candidate_reviews,
            "extra_reviews": candidate_reviews - baseline_reviews, "new_misroutes": new_misroutes}


def check(name, observed, limit, passed, queue=None):
    entry = {"check": name, "observed": observed, "limit": limit, "passed": passed}
    if queue is not None:
        entry["queue"] = queue
    return entry


def compare(baseline, candidate, cases, limits):
    reports = []
    for name, model in (("baseline", baseline), ("candidate", candidate)):
        try:
            require_later(cases, model["used_ids"], timestamp(model["used_through"]))
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        reports.append(measure(model, cases))

    before, after = reports
    paired = list(zip(before["details"], after["details"], strict=True))
    queues = {queue: [] for queue in sorted(baseline["centers"].keys() | candidate["centers"].keys())}
    for old, new in paired:
        queues.setdefault(new["expected"], []).append((old, new))
    groups = [{"queue": queue, **summarize(group)} for queue, group in queues.items()]
    total = summarize(paired)
    assigned = after["assigned"]
    correct = after["correct_assignments"]
    precision = Fraction(correct, assigned) if assigned else None
    checks = [
        check("minimum_cases", total["cases"], limits.min_cases, total["cases"] >= limits.min_cases),
        check("minimum_assignments", assigned, limits.min_assignments, assigned >= limits.min_assignments),
        check("minimum_precision", str(precision) if assigned else None, str(limits.min_precision),
              precision is not None and precision >= limits.min_precision),
        check("maximum_new_misroutes", total["new_misroutes"], limits.max_new_misroutes,
              total["new_misroutes"] <= limits.max_new_misroutes),
        check("maximum_extra_reviews", total["extra_reviews"], limits.max_extra_reviews,
              total["extra_reviews"] <= limits.max_extra_reviews),
    ]
    for group in groups:
        queue = group["queue"]
        if queue is None:
            continue
        checks.extend([
            check("minimum_queue_cases", group["cases"], limits.min_queue_cases,
                  group["cases"] >= limits.min_queue_cases, queue),
            check("maximum_extra_reviews_per_queue", group["extra_reviews"], limits.max_extra_reviews_per_queue,
                  group["extra_reviews"] <= limits.max_extra_reviews_per_queue, queue),
        ])

    changes = [
        {"case_id": new["case_id"], "expected_queue": new["expected"],
         "baseline_queue": old["queue"], "candidate_queue": new["queue"],
         "baseline_outcome": outcome(old), "candidate_outcome": outcome(new)}
        for old, new in paired if old["queue"] != new["queue"]
    ]
    return {"passed": all(row["passed"] for row in checks), "checks": checks, "totals": total,
            "by_expected_queue": groups, "changes": changes,
            "baseline": {field: before[field] for field in ("assigned", "correct_assignments", "assignment_precision")},
            "candidate": {field: after[field] for field in ("assigned", "correct_assignments", "assignment_precision")}}


def evaluation_model(data):
    model = parse_model(data.decode("utf-8"))
    centers = model["centers"]
    if type(model["format"]) is not int or not isinstance(centers, dict) or len(centers) < 2:
        raise ValueError("expected a fitted model with at least two queues")
    for weights in (model["idf"], *centers.values()):
        if not isinstance(weights, dict) or not weights or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights.values()):
            raise ValueError("model contains invalid TF-IDF weights")
    ids = model["used_ids"]
    if not isinstance(ids, list) or not ids or any(not isinstance(case, str) or not case.strip() for case in ids):
        raise ValueError("model must record the training and calibration case IDs")
    return model


def audit(baseline_path, candidate_path, holdout_path, limits):
    original, proposed, holdout = (path.read_bytes() for path in (baseline_path, candidate_path, holdout_path))
    baseline = evaluation_model(original)
    candidate = evaluation_model(proposed)
    cases = parse_cases(holdout.decode("utf-8-sig"), holdout_path)
    report = compare(baseline, candidate, cases, limits)
    digests = {name: sha256(data).hexdigest() for name, data in
               (("baseline", original), ("candidate", proposed), ("holdout", holdout))}
    first, last = min(row[0] for row in cases.values()), max(row[0] for row in cases.values())
    return {"version": 1, "input_sha256": digests,
            "holdout_utc": {"first": first.isoformat(), "last": last.isoformat()}, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--min-cases", type=int, required=True)
    parser.add_argument("--min-queue-cases", type=int, default=1)
    parser.add_argument("--min-assignments", type=int, default=1)
    parser.add_argument("--min-precision", default="0.9")
    parser.add_argument("--max-new-misroutes", type=int, default=0)
    parser.add_argument("--max-extra-reviews", type=int, default=0)
    parser.add_argument("--max-extra-reviews-per-queue", type=int, default=0)
    args = parser.parse_args()
    try:
        limits = Limits(min_cases=args.min_cases, min_queue_cases=args.min_queue_cases,
                        min_assignments=args.min_assignments, min_precision=Fraction(args.min_precision),
                        max_new_misroutes=args.max_new_misroutes, max_extra_reviews=args.max_extra_reviews,
                        max_extra_reviews_per_queue=args.max_extra_reviews_per_queue)
        report = audit(args.baseline, args.candidate, args.holdout, limits)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
