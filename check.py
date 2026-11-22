"""Compare a token classifier's scores for the same input across batch layouts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session(model):
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 1
    settings.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), settings, providers=["CPUExecutionProvider"])


def compare(runtime, samples, *, pad=0, extra=2, atol=1e-5, rtol=1e-5, output="logits"):
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)) or extra < 1:
        raise ValueError("Require finite nonnegative tolerances and at least one extra pad token")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Supply at least two labeled token sequences")
    sequences = {}
    for row in samples:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in sequences:
            raise ValueError("Expected a unique nonempty sample id")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(v) is not int or not 0 <= v < 2**63 for v in tokens):
            raise ValueError("Expected a nonempty list of nonnegative int64 token IDs")
        sequences[row["id"]] = tokens
    if type(pad) is not int or not 0 <= pad < 2**63:
        raise ValueError("Invalid padding token ID")

    def run(order, width):
        ids = np.full((len(order), width), pad, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, key in enumerate(order):
            values = sequences[key]
            ids[i, :len(values)] = values
            mask[i, :len(values)] = 1
        scores, = runtime.run([output], {"input_ids": ids, "attention_mask": mask})
        if (scores.ndim != 2 or scores.shape[0] != len(order) or not scores.shape[1] or
                not np.issubdtype(scores.dtype, np.floating) or not np.isfinite(scores).all()):
            raise ValueError("Expected finite floating-point scores shaped [batch, classes]")
        return dict(zip(order, scores.astype(np.float64), strict=True))

    order = list(sequences)
    width = max(map(len, sequences.values())) + extra
    mixed = run(order, width)
    rotated_batch = run(order[1:] + order[:1], width)
    checks = []
    for key in order:
        single = run([key], len(sequences[key]))[key]
        padded = run([key], width)[key]
        for name, ref, actual in [("padding", single, padded), ("batch", padded, mixed[key]),
                                  ("order", mixed[key], rotated_batch[key])]:
            if actual.shape != ref.shape:
                raise ValueError("Class dimension changed between layouts")
            difference = np.abs(actual - ref)
            failing = np.flatnonzero(difference > atol + rtol*np.abs(ref))
            index = int(difference.argmax())
            checks.append({"sample": key, "check": name, "passed": not failing.size,
                           "failed_classes": failing.tolist(),
                           "max_abs_error": float(difference[index]), "class_at_max_error": index,
                           "expected_score": float(ref[index]), "actual_score": float(actual[index]),
                           "argmax_changed": bool(actual.argmax() != ref.argmax())})
    return {"passed": all(c["passed"] for c in checks), "batch_width": width, "batch_size": len(samples),
            "atol": atol, "rtol": rtol, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", default="logits")
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--extra", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        result = compare(session(args.model), json.loads(args.samples.read_text(encoding="utf-8-sig")),
                         pad=args.pad, extra=args.extra, atol=args.atol, rtol=args.rtol, output=args.output)
    except Exception as exc:
        # ONNX Runtime's native status classes do not share a Python-specific base.
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
