"""Build four small executable ONNX classifiers and inspect their observed scores."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper

from check import compare, session

SAMPLES = [{"id": "returns", "tokens": [1]}, {"id": "delivery", "tokens": [2, 2, 1]}]


def build(path, variant):
    weights = numpy_helper.from_array(np.array([[0, 0], [4, 1], [0, 6]], np.float32), "weights")
    axes = [numpy_helper.from_array(np.array([i], np.int64), f"axis{i}") for i in (0, 1, 2)]
    axes.append(numpy_helper.from_array(np.array(0, np.int64), "batch_axis"))
    nodes = [
        h.make_node("Gather", ["weights", "input_ids"], ["embedded"]),
        h.make_node("Cast", ["attention_mask"], ["mask"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["mask", "axis2"], ["mask3"]),
        h.make_node("Mul", ["embedded", "mask3"], ["masked"]),
    ]
    if variant == "padded_mean":
        nodes.append(h.make_node("ReduceMean", ["masked", "axis1"], ["logits"], keepdims=0))
    else:
        nodes.extend([h.make_node("ReduceSum", ["masked", "axis1"], ["totals"], keepdims=0),
                      h.make_node("ReduceSum", ["mask", "axis1"], ["lengths"], keepdims=1)])
        if variant == "masked_mean":
            nodes.append(h.make_node("Div", ["totals", "lengths"], ["logits"]))
        elif variant == "batch_centered":
            nodes.extend([h.make_node("Div", ["totals", "lengths"], ["mean"]),
                          h.make_node("ReduceMean", ["mean", "axis0"], ["batchmean"], keepdims=1),
                          h.make_node("Sub", ["mean", "batchmean"], ["logits"])])
        elif variant == "batch_prefix":
            nodes.extend([h.make_node("Div", ["totals", "lengths"], ["mean"]),
                          h.make_node("CumSum", ["mean", "batch_axis"], ["logits"])])
        else:
            raise ValueError(variant)
    inputs = [h.make_tensor_value_info(name, T.INT64, ["batch", "tokens"]) for name in ("input_ids", "attention_mask")]
    graph = h.make_graph(nodes, variant, inputs, [h.make_tensor_value_info("logits", T.FLOAT, ["batch", 2])],
                         [weights, *[axis for axis in axes if any(axis.name in node.input for node in nodes)]])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    with TemporaryDirectory() as temporary:
        for variant in ("masked_mean", "padded_mean", "batch_centered", "batch_prefix"):
            path = Path(temporary) / f"{variant}.onnx"
            build(path, variant)
            result = compare(session(path), SAMPLES)
            failures = [c for c in result["checks"] if not c["passed"]]
            print(json.dumps({"model": variant, "failed_checks": sorted({c["check"] for c in failures}),
                              "max_abs_error": max(c["max_abs_error"] for c in result["checks"]),
                              "changed_labels": sum(c["argmax_changed"] for c in failures)}))
