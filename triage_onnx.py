"""Export the fitted ticket scorer and route batches using ONNX Runtime on CPU."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
import onnxruntime as ort

from triage import decision_record, grams, load_model, ranked_candidate


def export(model, destination):
    vocabulary = sorted(model["idf"])
    labels = sorted(model["centers"])
    if not vocabulary or len(labels) < 2:
        raise ValueError("Expected a nonempty vocabulary and at least two queues")
    indices = {gram: i for i, gram in enumerate(vocabulary, 1)}
    idf = np.array([0, *(model["idf"][key] for key in vocabulary)], dtype=np.float64)
    centers = np.zeros((len(idf), len(labels)), np.float64)
    for col, label in enumerate(labels):
        for gram, value in model["centers"][label].items():
            centers[indices[gram], col] = value
    if not np.isfinite(idf).all() or not np.isfinite(centers).all():
        raise ValueError("Expected finite model weights")
    init = {"width": np.array([len(idf)], np.int64), "idf": idf, "centers": centers,
            "axis1": np.array([1], np.int64), "zero": np.array(0., np.float64), "one": np.array(1., np.float64)}
    nodes = [
        h.make_node("Shape", ["input_ids"], ["batch"], start=0, end=1),
        h.make_node("Concat", ["batch", "width"], ["shape"], axis=0),
        h.make_node("ConstantOfShape", ["shape"], ["empty"], value=h.make_tensor("empty_count", T.DOUBLE, [1], [0.])),
        h.make_node("Cast", ["attention_mask"], ["increments"], to=T.DOUBLE),
        h.make_node("ScatterElements", ["empty", "input_ids", "increments"], ["counts"], axis=1, reduction="add"),
        h.make_node("Mul", ["counts", "idf"], ["weighted"]),
        h.make_node("ReduceL2", ["weighted", "axis1"], ["norm"], keepdims=1),
        h.make_node("Equal", ["norm", "zero"], ["unknown"]),
        h.make_node("Where", ["unknown", "one", "norm"], ["denominator"]),
        h.make_node("Div", ["weighted", "denominator"], ["unit"]),
        h.make_node("MatMul", ["unit", "centers"], ["logits"]),
    ]
    inputs = [h.make_tensor_value_info(name, T.INT64, ["batch", "tokens"]) for name in ("input_ids", "attention_mask")]
    output = h.make_tensor_value_info("logits", T.DOUBLE, ["batch", len(labels)])
    graph = h.make_graph(nodes, "ticket_tfidf", inputs, [output], [numpy_helper.from_array(v, k) for k, v in init.items()])
    artifact = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9)
    if model["format"] == 1:
        policy = {"format": 1, "margin_threshold": model["margin_threshold"]}
    else:
        policy = {"format": 2, "queue_policies": {k: {"threshold": v["threshold"]} for k, v in model["queue_policies"].items()}}
    h.set_model_props(artifact, {"ticket_triage_v1": json.dumps({"vocabulary": vocabulary, "labels": labels, "policy": policy},
                                                                 ensure_ascii=False, allow_nan=False)})
    onnx.checker.check_model(artifact)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        staged = Path(stream.name)
    try:
        onnx.save(artifact, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


class BatchRouter:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.runtime = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        config = json.loads(self.runtime.get_modelmeta().custom_metadata_map["ticket_triage_v1"])
        self.vocabulary = {gram: i for i, gram in enumerate(config["vocabulary"], 1)}
        self.labels, self.policy = config["labels"], config["policy"]

    def encode(self, text):
        return [self.vocabulary[gram] for gram in grams(text) if gram in self.vocabulary] or [0]

    def scores(self, texts):
        if not texts:
            raise ValueError("Expected at least one ticket")
        rows = [self.encode(text) for text in texts]
        ids = np.zeros((len(rows), max(map(len, rows))), np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        values, = self.runtime.run(["logits"], {"input_ids": ids, "attention_mask": mask})
        return [dict(zip(self.labels, row.tolist(), strict=True)) for row in values]

    def route(self, texts):
        return [decision_record(self.policy, *ranked_candidate(row)) for row in self.scores(texts)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("export")
    build.add_argument("source", type=Path)
    build.add_argument("output", type=Path)
    route = commands.add_parser("route")
    route.add_argument("model", type=Path)
    route.add_argument("texts", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "export":
            if args.source.resolve() == args.output.resolve():
                raise ValueError("Use a separate destination for the ONNX file")
            export(load_model(args.source), args.output)
        else:
            print(json.dumps(BatchRouter(args.model).route(args.texts), ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(2, f"{exc}\n")
