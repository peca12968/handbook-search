"""Compare exported class scores, calibrated decisions and alternate batch layouts."""
from fractions import Fraction
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from triage import fit, read_cases, route, scores, write_model
from triage_onnx import BatchRouter


def exercise(checker):
    examples = Path(__file__).with_name("examples")
    train, calibration, holdout = [read_cases(examples / f"{name}.jsonl") for name in ("train", "calibration", "holdout")]
    texts = [text for _, text, _ in (*train.values(), *calibration.values(), *holdout.values())]
    texts.extend(["", "！", "退款" * 30, "ＡＢＣ 配送", "付款退款物流付款退款"])
    cli = Path(__file__).with_name("triage_onnx.py")
    results = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for mode in ("global", "per-queue"):
            model = fit(train, calibration, Fraction("0.9"), threshold_mode=mode, min_cases=2)
            source, target = root / "trained.json", root / "deployed.onnx"
            write_model(source, model)
            subprocess.run([sys.executable, cli, "export", source, target], check=True)
            router = BatchRouter(target)
            error = 0.0
            for text, values in zip(texts, router.scores(texts), strict=True):
                reference = scores(model, text)
                assert values.keys() == reference.keys()
                error = max(error, *(abs(values[label] - reference[label]) for label in reference))
            assert error < 1e-12
            expected = [route(model, text) for text in texts]
            assert router.route(texts) == expected
            actual = json.loads(subprocess.check_output([sys.executable, cli, "route", target, *texts[-5:]]))
            assert actual == expected[-5:]
            samples = [{"id": str(i), "tokens": router.encode(text)} for i, text in enumerate(texts)]
            data = root / "samples.json"
            data.write_text(json.dumps(samples))
            # A real vocabulary ID at masked positions catches a mask ignored by the graph.
            report = json.loads(subprocess.check_output([sys.executable, checker, target, data, "--pad", "1",
                                                        "--atol", "1e-12", "--rtol", "1e-12"]))
            assert report["passed"] and len(report["checks"]) == 3 * len(texts)
            results.append({"mode": mode, "tickets": len(texts), "max_score_error": error,
                            "matching_decisions": len(expected), "layout_checks": len(report["checks"])})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_checker", type=Path)
    print(json.dumps(exercise(parser.parse_args().batch_checker.resolve()), indent=2))
