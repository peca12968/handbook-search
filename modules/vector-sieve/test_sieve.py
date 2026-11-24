import itertools
import math
import unittest

import faiss

from demo import dataset
from sieve import evaluate, grade


def item(key, group, coords):
    return {"id": key, "group": group, "vector": coords}


class FilteredRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1)

    def test_synthetic_catalog_against_independent_exhaustive_search(self):
        data = dataset()
        report = evaluate(data)
        for query, row in zip(data["queries"], report, strict=True):
            allowed = {doc["id"]: doc for doc in data["items"] if doc["group"] == query["group"]}
            # Independently rank the generator's original doubles with scalar Euclidean distance.
            winners = sorted(allowed, key=lambda key: (math.dist(allowed[key]["vector"], query["vector"]), key))[:5]
            self.assertEqual(row["exact_ids"], winners)
            for method in ("partition_first", "global_then_filter"):
                ids = row[method]["ids"]
                self.assertTrue(set(ids) <= set(allowed))
                self.assertEqual(row[method]["recall"], len(set(winners).intersection(ids)) / 5)
            if query["group"] == "regional":
                self.assertEqual(row["global_then_filter"]["ids"], [])
                self.assertEqual(row["partition_first"]["ids"], winners)

    def test_cutoff_ties_against_all_admissible_top_k_sets(self):
        distances = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 4.0}
        for k in range(1, 6):
            cutoff = sorted(distances.values())[k - 1]
            choices = list(itertools.combinations(distances, k))
            minimum = min(sum(distances[i] for i in choice) for choice in choices)
            optimal = [set(choice) for choice in choices if sum(distances[i] for i in choice) == minimum]
            for count in range(k+1):
                for selected in itertools.combinations(distances, count):
                    expected = max(len(set(selected) & answer) for answer in optimal) / k
                    self.assertEqual(grade(selected, distances, k, cutoff), expected)

    def test_fewer_than_k_and_empty_partition(self):
        data = {"items": [item("x", "x", [0]), item("b", "y", [0]), item("a", "y", [0])],
                "queries": [item("y", "y", [0]), item("z", "absent", [0])]}
        short, empty = evaluate(data, k=5, candidates=8)
        self.assertEqual(short["exact_ids"], ["a", "b"])
        self.assertEqual(short["partition_first"]["recall"], 1)
        for method in ("global_then_filter", "partition_first"):
            self.assertEqual(empty[method]["ids"], [])
            self.assertIsNone(empty[method]["recall"])

    def test_invalid_vectors_and_options(self):
        base = item("1", "g", [1, 2])
        for bad in ([base, base], [item("x", "g", [float("nan"), 1])], [item("x", "g", [1e25, 1])],
                    [item("x", "g", [1])], [item("x", "g", ["2", 1])]):
            with self.assertRaises(ValueError):
                evaluate({"items": bad, "queries": [base]})
        with self.assertRaises(ValueError):
            evaluate({"items": [base], "queries": [base]}, k=5, candidates=2)
