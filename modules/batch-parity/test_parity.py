import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check import compare, session
from model_demo import build, SAMPLES


class InferenceParity(unittest.TestCase):
    def test_compares_real_scores_with_and_without_padding_or_other_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.onnx"
            for variant, failed in [("masked_mean", set()), ("padded_mean", {"padding"}),
                                    ("batch_centered", {"batch"}), ("batch_prefix", {"batch", "order"})]:
                build(model, variant)
                report = compare(session(model), SAMPLES)
                self.assertEqual(report["batch_width"], 5)
                self.assertEqual(len(report["checks"]), 6)
                self.assertEqual({c["check"] for c in report["checks"] if not c["passed"]}, failed)
                self.assertEqual(report["passed"], not failed)
                if variant == "padded_mean":
                    check = next(c for c in report["checks"] if c["sample"] == "returns" and c["check"] == "padding")
                    self.assertEqual(check["expected_score"], 4.0)
                    self.assertAlmostEqual(check["actual_score"], 4/5)
                    self.assertFalse(check["argmax_changed"])
                if variant == "batch_centered":
                    self.assertTrue(any(c["argmax_changed"] for c in report["checks"]))
                if variant == "batch_prefix":
                    order_checks = [c for c in report["checks"] if c["check"] == "order"]
                    self.assertTrue(all(c["failed_classes"] for c in order_checks))

    def test_cli_exit_status_separates_measured_drift_from_unusable_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.onnx"
            inputs = Path(tmp) / "tickets.json"
            inputs.write_text(json.dumps(SAMPLES))
            cmd = [sys.executable, str(Path(__file__).with_name("check.py")), model, inputs]
            build(model, "padded_mean")
            drift = subprocess.run(cmd, capture_output=True, timeout=30)
            self.assertEqual(drift.returncode, 1, drift.stderr)
            self.assertFalse(json.loads(drift.stdout)["passed"])
            invalid = subprocess.run([*cmd, "--output", "missing"], capture_output=True, timeout=30)
            self.assertEqual((invalid.returncode, invalid.stdout), (2, b""))
            self.assertNotIn(b"Traceback", invalid.stderr)
            build(model, "masked_mean")
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["passed"])
