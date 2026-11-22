import json
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import triage


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"


class RoutingTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model.json"

    def cli(self, *args, success=True):
        command = [sys.executable, str(ROOT / "triage.py"), *map(str, args), "--model", str(self.model)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0 if success else 2, result.stderr)
        if success:
            return json.loads(result.stdout)
        self.assertEqual(result.stdout, b"")
        self.assertNotIn(b"Traceback", result.stderr)

    def train(self):
        return self.cli("fit", EXAMPLES / "train.jsonl", EXAMPLES / "calibration.jsonl")

    def test_cli_trains_routes_and_evaluates_without_rewriting_model(self):
        calibration = self.train()["calibration"]
        self.assertEqual((calibration["cases"], calibration["assigned"], calibration["correct_assignments"]), (10, 7, 7))
        original = self.model.read_bytes()
        report = self.cli("evaluate", EXAMPLES / "holdout.jsonl")
        self.assertEqual((report["cases"], report["assigned"], report["correct_assignments"], report["review"]),
                         (11, 4, 4, 7))
        self.assertAlmostEqual(report["coverage"], 4 / 11)
        self.assertEqual(self.cli("route", "無法登入帳號，重設密碼連結過期")["queue"], "帳號")
        self.assertEqual(self.cli("route", "🐙")["reason"], "no_unique_match")
        self.assertEqual(self.cli("route", "宅配地址需要更正")["reason"], "below_threshold")
        self.assertEqual(self.model.read_bytes(), original)

    def test_cosines_match_independent_sklearn_calculation(self):
        training = triage.read_cases(EXAMPLES / "train.jsonl")
        model = triage.learn(training)
        cases = list(training.values())
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                     preprocessor=lambda t: " ".join(unicodedata.normalize("NFKC", t).casefold().split()))
        matrix = vectorizer.fit_transform([s for _, s, _ in cases])
        labels = sorted(model["centers"])
        centers = [normalize(matrix[[i for i, r in enumerate(cases) if r[2] == label]].sum(axis=0).A)[0]
                   for label in labels]
        queries = list(triage.read_cases(EXAMPLES / "calibration.jsonl").values())
        queries += list(triage.read_cases(EXAMPLES / "holdout.jsonl").values())
        for _, text, _ in queries:
            actual = triage.scores(model, text)
            q = vectorizer.transform([text])
            for label, center in zip(labels, centers):
                self.assertAlmostEqual(actual[label], (q @ center)[0], places=12)

    def test_threshold_matches_exhaustive_search_with_ties_and_nonmonotone_precision(self):
        rng = random.Random(3187)
        for _ in range(250):
            items = [(rng.randrange(1, 10) / 10, rng.choice([False, True])) for _ in range(rng.randrange(0, 20))]
            target = Fraction(rng.randrange(1, 5), 4)
            choices = []
            for margin in set(x for x, _ in items):
                included = [ok for x, ok in items if x >= margin]
                if Fraction(sum(included), len(included)) >= target:
                    choices.append((len(included), margin))
            expected = max(choices)[1] if choices else None
            self.assertEqual(triage.choose_threshold(items, target), expected)
        self.assertEqual(triage.choose_threshold([(0.9, False), (0.8, True)], Fraction(1, 2)), 0.8)

    def test_unseen_features_tied_classes_and_unattainable_target_go_to_review(self):
        train = {k: (triage.timestamp("2025-01-01T00:00:00Z"), "共同範例", k) for k in ("one", "two")}
        calibration = {"v": (triage.timestamp("2025-02-01T00:00:00Z"), "校正資料", "one")}
        model = triage.fit(train, calibration, Fraction(1))
        self.assertNotIn("校正", model["idf"])
        self.assertIsNone(model["margin_threshold"])
        self.assertEqual(triage.candidate(model, "共同範例"), (None, 0))
        self.assertEqual(triage.measure(model, calibration)["assignment_precision"], None)
        training = triage.read_cases(EXAMPLES / "train.jsonl")
        unsupported = {"v": (triage.timestamp("2025-02-01T00:00Z"), "信用卡扣款失敗", None)}
        strict = triage.fit(training, unsupported, Fraction(1))
        self.assertIsNone(strict["margin_threshold"])
        self.assertEqual(triage.route(strict, "信用卡扣款失敗")["reason"], "no_validated_threshold")

    def test_rejects_overlapping_periods_reused_ids_and_bad_input_without_replacing_model(self):
        self.train()
        original = self.model.read_bytes()
        sample = json.loads((EXAMPLES / "calibration.jsonl").read_text(encoding="utf-8").splitlines()[0])
        trial = self.root / "invalid.jsonl"
        for changes in [{"at": "2025-01-15T01:00:00Z"}, {"case_id": "T-001"}, {"at": "2025-02-20T09:00:00"}]:
            trial.write_text(json.dumps(sample | changes), encoding="utf-8")
            self.cli("fit", EXAMPLES / "train.jsonl", trial, success=False)
        duplicate = json.dumps(sample)
        trial.write_text(duplicate + "\n" + duplicate, encoding="utf-8")
        self.cli("fit", EXAMPLES / "train.jsonl", trial, success=False)
        self.cli("evaluate", EXAMPLES / "calibration.jsonl", success=False)
        trial.write_text(json.dumps(sample | {"at": "2025-05-01T00:00Z"}), encoding="utf-8")
        self.cli("evaluate", trial, success=False)
        self.assertEqual(self.model.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
