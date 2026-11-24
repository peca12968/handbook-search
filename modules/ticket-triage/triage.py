"""Fit, route and evaluate a local selective classifier for support tickets."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import groupby
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import unicodedata


def timestamp(value):
    moment = datetime.fromisoformat(value)
    if moment.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return moment.astimezone(timezone.utc)


def read_cases(path):
    return parse_cases(path.read_text(encoding="utf-8-sig"), path)


def parse_cases(text, path):
    cases = {}
    for number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
            case, message, label = row["case_id"], row["text"], row["queue"]
            if not isinstance(case, str) or not case.strip() or case in cases:
                raise ValueError("expected a unique nonempty case_id")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("expected nonempty text")
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise ValueError("queue must be a nonempty string or null")
            cases[case] = (timestamp(row["at"]), message, label)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}, line {number}: {error}") from error
    if not cases:
        raise ValueError(f"empty dataset: {path}")
    return cases


def require_later(cases, earlier_ids, cutoff):
    shared = set(cases).intersection(earlier_ids)
    if shared:
        raise ValueError("case_id reused across periods: " + min(shared))
    if min(row[0] for row in cases.values()) <= cutoff:
        raise ValueError("evaluation period must begin strictly after the preceding period")


def grams(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return [normalized[i:i+n] for n in (2, 3) for i in range(len(normalized)-n+1)]


def unit(weights):
    norm = math.sqrt(math.fsum(x*x for x in weights.values()))
    return {k: v / norm for k, v in weights.items()} if norm else {}


def vector(counts, idf):
    return unit({key: n * idf[key] for key, n in counts.items() if key in idf})


def learn(cases):
    docs = []
    frequency = Counter()
    for _, (_, text, label) in sorted(cases.items()):
        counts = Counter(grams(text))
        if label is None or not counts:
            raise ValueError("training cases require a queue and at least one character bigram")
        docs.append((label, counts))
        frequency.update(counts.keys())
    idf = {k: math.log((1 + len(docs)) / (1 + df)) + 1 for k, df in frequency.items()}
    totals = defaultdict(Counter)
    for label, counts in docs:
        totals[label].update(vector(counts, idf))
    if len(totals) < 2:
        raise ValueError("training requires at least two queues")
    return {"idf": idf, "centers": {label: unit(values) for label, values in totals.items()}}


def scores(model, text):
    query = vector(Counter(grams(text)), model["idf"])
    return {label: math.fsum(weight * center.get(key, 0) for key, weight in query.items())
            for label, center in model["centers"].items()}


def candidate(model, text):
    return ranked_candidate(scores(model, text))


def ranked_candidate(values):
    ranked = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))
    label, top = ranked[0]
    margin = round(top - ranked[1][1], 9)
    return (label if margin > 0 else None), margin


def calibrate(observations, target, min_cases):
    # Every member of a tied margin group is accepted or rejected together.
    threshold = None
    peak = None
    peak_precision = None
    correct = accepted = 0
    for margin, group in groupby(sorted(observations, reverse=True), key=lambda pair: pair[0]):
        for _, hit in group:
            correct += hit
            accepted += 1
        if accepted < min_cases:
            continue
        precision = Fraction(correct, accepted)
        if peak_precision is None or precision >= peak_precision:
            peak_precision = precision
            peak = {"margin": margin, "correct": correct, "accepted": accepted}
        if precision >= target:
            threshold = margin
    if threshold is not None:
        status = "enabled"
    elif accepted < min_cases:
        status = "insufficient_candidates"
    else:
        status = "precision_target_unmet"
    return {"threshold": threshold, "status": status, "candidates": accepted, "best_supported": peak}


def choose_threshold(observations, target, min_cases=1):
    return calibrate(observations, target, min_cases)["threshold"]


def threshold_for(model, label):
    if model["format"] == 1:
        return model["margin_threshold"]
    return model["queue_policies"][label]["threshold"] if label is not None else None


def decide(model, label, margin):
    threshold = threshold_for(model, label)
    if threshold is None and (label is not None or model["format"] == 1):
        reason = "no_validated_threshold"
    elif label is None:
        reason = "no_unique_match"
    elif margin < threshold:
        reason = "below_threshold"
    else:
        reason = "accepted"
    accepted = reason == "accepted"
    return {"action": "assign" if accepted else "review", "queue": label if accepted else None,
            "reason": reason, "margin": margin, "margin_threshold": threshold}


def route(model, text):
    label, margin = candidate(model, text)
    return decision_record(model, label, margin)


def decision_record(model, label, margin):
    result = decide(model, label, margin)
    if model["format"] == 2:
        result["candidate_queue"] = label
    return result


def measure(model, cases):
    details = []
    correct = assigned = 0
    no_candidate = 0
    queues = {label: {"margin_threshold": threshold_for(model, label), "candidates": 0,
                      "assigned": 0, "review": 0, "correct_assignments": 0}
              for label in sorted(model["centers"])}
    for case, (_, text, truth) in sorted(cases.items()):
        label, margin = candidate(model, text)
        result = decide(model, label, margin)
        if label is None:
            no_candidate += 1
        else:
            queues[label]["candidates"] += 1
        if result["action"] == "assign":
            assigned += 1
            hit = label == truth
            correct += hit
            queues[label]["assigned"] += 1
            queues[label]["correct_assignments"] += hit
        elif label is not None:
            queues[label]["review"] += 1
        details.append({"case_id": case, "expected": truth, "candidate_queue": label, **result})
    for stats in queues.values():
        stats["assignment_precision"] = stats["correct_assignments"] / stats["assigned"] if stats["assigned"] else None
    return {"cases": len(cases), "assigned": assigned, "review": len(cases) - assigned,
            "correct_assignments": correct, "coverage": assigned / len(cases),
            "assignment_precision": correct / assigned if assigned else None,
            "by_candidate_queue": queues, "no_unique_candidate": no_candidate, "details": details}


def fit(train, calibration, target, *, threshold_mode="global", min_cases=1):
    if not 0 < target <= 1:
        raise ValueError("minimum precision must be greater than zero and at most one")
    if threshold_mode not in ("global", "per-queue") or type(min_cases) is not int or min_cases < 1:
        raise ValueError("expected global or per-queue calibration and a positive minimum case count")
    require_later(calibration, train, max(row[0] for row in train.values()))
    model = learn(train)
    observations = defaultdict(list)
    for _, text, truth in calibration.values():
        label, margin = candidate(model, text)
        if label is not None:
            observations[label].append((margin, label == truth))
    if threshold_mode == "global":
        pooled = [item for group in observations.values() for item in group]
        model.update(format=1, margin_threshold=choose_threshold(pooled, target, min_cases))
    else:
        model.update(format=2, queue_policies={
            label: calibrate(observations[label], target, min_cases) for label in sorted(model["centers"])})
    model.update(minimum_precision=str(target), minimum_calibration_cases=min_cases,
                 used_ids=sorted(set(train) | set(calibration)),
                 used_through=max(row[0] for row in calibration.values()).isoformat())
    return model


def load_model(path):
    return parse_model(path.read_text(encoding="utf-8"))


def parse_model(text):
    model = json.loads(text)
    if model["format"] == 1:
        thresholds = [model["margin_threshold"]]
    elif model["format"] == 2:
        policy = model["queue_policies"]
        if not isinstance(policy, dict) or not isinstance(model["centers"], dict) or policy.keys() != model["centers"].keys():
            raise ValueError("queue thresholds must cover exactly the trained queues")
        thresholds = [row["threshold"] for row in policy.values()]
    else:
        raise ValueError("unknown model format")
    if any(v is not None and (type(v) not in (int, float) or not 0 < v <= 1) for v in thresholds):
        raise ValueError("invalid margin threshold")
    return model


def write_model(path, model):
    temp = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    staging = Path(temp.name)
    try:
        with temp:
            json.dump(model, temp, ensure_ascii=False, sort_keys=True, allow_nan=False)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("fit")
    build.add_argument("train", type=Path)
    build.add_argument("calibration", type=Path)
    build.add_argument("--min-precision", default="0.9")
    build.add_argument("--threshold-mode", choices=["global", "per-queue"], default="global")
    build.add_argument("--min-calibration-cases", type=int, default=1,
                       help="minimum number of accepted calibration predictions per threshold")
    build.add_argument("--model", type=Path, required=True)
    predict = commands.add_parser("route")
    predict.add_argument("text")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("holdout", type=Path)
    for command in (predict, evaluate):
        command.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fit":
            if args.model.resolve() in {args.train.resolve(), args.calibration.resolve()}:
                raise ValueError("model output also names a dataset")
            train, calibration = read_cases(args.train), read_cases(args.calibration)
            model = fit(train, calibration, Fraction(args.min_precision),
                        threshold_mode=args.threshold_mode, min_cases=args.min_calibration_cases)
            policy = "margin_threshold" if model["format"] == 1 else "queue_policies"
            report = {policy: model[policy], "calibration": measure(model, calibration)}
            write_model(args.model, model)
        else:
            model = load_model(args.model)
            if args.command == "route":
                report = route(model, args.text)
            else:
                cases = read_cases(args.holdout)
                require_later(cases, model["used_ids"], timestamp(model["used_through"]))
                report = measure(model, cases)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
