"""Build reproducible synthetic item vectors for two handbook locales."""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from sieve import evaluate


def dataset():
    rng = np.random.default_rng(12680)
    dimensions = 32
    items, queries = [], []
    for topic in range(16):
        center = np.zeros(dimensions)
        center[topic] = 20
        for i in range(128):
            vector = center + rng.normal(0, 0.08, dimensions)
            group = "regional" if i >= 120 else "general"
            if group == "regional":
                vector[-1] += 2
            items.append({"id": f"topic-{topic}-p{i}", "group": group, "vector": vector.tolist()})
        for group in ("general", "regional"):
            queries.append({"id": f"q{topic}-{group}", "group": group, "vector": center.tolist()})
    return {"items": items, "queries": queries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    data = dataset()
    if args.export:
        args.export.write_text(json.dumps(data), encoding="utf-8")
    faiss.omp_set_num_threads(1)
    for budget in (32, 128):
        rows = evaluate(data, candidates=budget)
        for group in ("general", "regional"):
            sample = [row for row in rows if row["group"] == group]
            recall = {method: sum(row[method]["recall"] for row in sample) / len(sample)
                      for method in ("global_then_filter", "partition_first")}
            print(json.dumps({"candidates": budget, "group": group, "queries": len(sample), "mean_recall": recall}))
