from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from release_demo import demonstrate
from triage import fit, read_cases, write_model


ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "examples/holdout.jsonl"


class ReleaseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = read_cases(ROOT / "examples/train.jsonl")
        calibration = read_cases(ROOT / "examples/calibration.jsonl")
        cls.original = fit(train, calibration, Fraction("0.9"), min_cases=3)
        cls.local = fit(train, calibration, Fraction("0.9"), min_cases=3, threshold_mode="per-queue")
        cls.aggressive = fit(train, calibration, Fraction("0.7"), min_cases=3)

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.baseline, self.candidate = self.directory / "baseline.json", self.directory / "candidate.json"
        write_model(self.baseline, self.original)
        write_model(self.candidate, self.local)

    def gate(self, *args, holdout=HOLDOUT):
        return subprocess.run([sys.executable, ROOT / "release_gate.py", self.baseline, self.candidate, holdout,
                               "--min-cases", "11", "--min-queue-cases", "2", *args],
                              capture_output=True, encoding="utf-8", timeout=15)

    def test_unchanged_models_pass_with_exact_evidence_hashes(self):
        write_model(self.candidate, self.original)
        inputs = {"baseline": self.baseline, "candidate": self.candidate, "holdout": HOLDOUT}
        expected = {name: sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["changes"], [])
        self.assertEqual(report["holdout_utc"], {"first": "2025-03-01T01:00:00+00:00",
                                                  "last": "2025-03-11T01:00:00+00:00"})
        self.assertEqual(report["input_sha256"], expected)
        self.assertEqual({name: sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}, expected)

    def test_queue_policy_only_passes_with_sufficient_review_capacity(self):
        blocked = self.gate()
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        report = json.loads(blocked.stdout)
        self.assertEqual((report["totals"]["extra_reviews"], report["totals"]["new_misroutes"]), (3, 0))
        self.assertEqual([row["case_id"] for row in report["changes"]], ["T-026", "T-027", "T-031"])
        self.assertTrue(all(row["candidate_outcome"] == "review" for row in report["changes"]))
        sufficient = self.gate("--max-extra-reviews", "3", "--max-extra-reviews-per-queue", "1")
        self.assertEqual(sufficient.returncode, 0, sufficient.stderr)
        support = self.gate("--max-extra-reviews", "3", "--max-extra-reviews-per-queue", "1", "--min-assignments", "2")
        self.assertEqual(support.returncode, 1, support.stderr)
        self.assertEqual([row["check"] for row in json.loads(support.stdout)["checks"] if not row["passed"]],
                         ["minimum_assignments"])

    def test_new_misroutes_are_visible_even_when_manual_load_drops(self):
        write_model(self.candidate, self.aggressive)
        result = self.gate("--min-precision", "7/9")
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["totals"]["extra_reviews"], -5)
        failures = [row["check"] for row in report["checks"] if not row["passed"]]
        self.assertEqual(failures, ["maximum_new_misroutes"])
        misrouted = [row for row in report["changes"] if row["candidate_outcome"] == "misroute"]
        self.assertEqual([(row["case_id"], row["expected_queue"]) for row in misrouted],
                         [("T-034", "配送"), ("T-036", None)])

    def test_extra_reviews_in_one_queue_are_visible_when_total_decreases(self):
        threshold = self.original["margin_threshold"]
        policies = {q: {"threshold": t} for q, t in (("配送", 0.001), ("帳號", None), ("付款", threshold))}
        write_model(self.candidate, {**self.original, "format": 2, "queue_policies": policies})
        result = self.gate("--min-precision", "1/10", "--max-new-misroutes", "11")
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertLess(report["totals"]["extra_reviews"], 0)
        self.assertEqual([row for row in report["checks"] if not row["passed"]], [
            {"check": "maximum_extra_reviews_per_queue", "observed": 1, "limit": 0, "passed": False, "queue": "帳號"}])

    def test_unrepresented_queue_and_precision_without_dispatch_fail(self):
        missing = self.directory / "missing.jsonl"
        rows = [row for row in HOLDOUT.read_text(encoding="utf-8").splitlines() if json.loads(row)["queue"] != "配送"]
        missing.write_text("\n".join(rows), encoding="utf-8")
        write_model(self.candidate, {**self.original, "margin_threshold": None})
        result = self.gate("--min-cases", "1", "--max-extra-reviews", "11", "--max-extra-reviews-per-queue", "11", holdout=missing)
        self.assertEqual(result.returncode, 1, result.stderr)
        failures = [row for row in json.loads(result.stdout)["checks"] if not row["passed"]]
        self.assertEqual({(row["check"], row.get("queue"), row["observed"]) for row in failures},
                         {("minimum_assignments", None, 0), ("minimum_precision", None, None), ("minimum_queue_cases", "配送", 0)})

    def test_rejects_leakage_for_either_model_and_invalid_limits(self):
        write_model(self.baseline, {**self.original, "used_ids": [*self.original["used_ids"], "T-028"]})
        reused = self.gate()
        self.assertEqual(reused.returncode, 2)
        self.assertIn("baseline: case_id reused", reused.stderr)
        write_model(self.baseline, self.original)
        write_model(self.candidate, {**self.local, "used_through": "2025-03-01T09:00:00+08:00"})
        overlap = self.gate()
        self.assertEqual(overlap.returncode, 2)
        self.assertIn("candidate: evaluation period must begin strictly after", overlap.stderr)
        self.assertEqual(self.gate("--min-precision", "0").returncode, 2)
        self.assertEqual(self.gate("--max-new-misroutes", "-1").returncode, 2)
        broken = {**self.local, "idf": {**self.local["idf"], "壞值": float("nan")}}
        self.candidate.write_text(json.dumps(broken), encoding="utf-8")
        failure = self.gate()
        self.assertEqual(failure.returncode, 2)
        self.assertIn("invalid TF-IDF weights", failure.stderr)

    def test_demo_is_recreated_from_the_same_paired_evaluation(self):
        fixture = ROOT / "examples/release-demo.json"
        data = demonstrate()
        self.assertEqual(json.loads(fixture.read_text(encoding="utf-8")), data)
        self.assertEqual([(row["scenario"], row["passed"]) for row in data],
                         [("strict", False), ("staffed", True), ("aggressive", False)])


if __name__ == "__main__":
    unittest.main()
