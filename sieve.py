"""Measure filtered HNSW retrieval against an exhaustive squared-L2 reference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import faiss
import numpy as np


def unpack(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of vector records")
    ids, groups, vectors = [], [], []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in ("id", "group")):
            raise ValueError("Each record needs a nonempty id and group")
        vector = row.get("vector")
        if not isinstance(vector, list) or not vector or any(type(x) not in (int, float) for x in vector):
            raise ValueError("Expected a nonempty numeric vector")
        ids.append(row["id"])
        groups.append(row["group"])
        vectors.append(vector)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record id")
    matrix = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Vectors must be finite float32 coordinates")
    # HNSW uses float32 squared distances; reserve headroom even if every
    # coordinate differs by twice the largest permitted magnitude.
    if np.abs(matrix).max() > np.sqrt(float(np.finfo(np.float32).max) / (8 * matrix.shape[1])):
        raise ValueError("Coordinates exceed safe range for squared L2 distance")
    return ids, groups, matrix


def graph(vectors, expansion):
    index = faiss.IndexHNSWFlat(vectors.shape[1], 16)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = expansion
    index.add(vectors)
    return index


def neighbors(index, query, count):
    _, keys = index.search(query.reshape(1, -1), min(count, index.ntotal))
    return [int(key) for key in keys[0] if key >= 0]


def grade(selected, distances, target, cutoff):
    if not target:
        return None
    mandatory = sum(distance < cutoff for distance in distances.values())
    inside = sum(distances[key] < cutoff for key in selected)
    tied = sum(distances[key] == cutoff for key in selected)
    return (inside + min(target - mandatory, tied)) / target


def evaluate(data, k=5, candidates=32, expansion=64):
    if k < 1 or candidates < k or expansion < 1:
        raise ValueError("Require candidates >= k >= 1 and a positive expansion")
    ids, groups, vectors = unpack(data["items"])
    query_ids, filters, queries = unpack(data["queries"])
    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("Query and item dimensions differ")
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    global_index = graph(vectors, expansion)
    requested_groups = set(filters)
    partitions = {group: graph(vectors[keys], expansion) for group, keys in members.items() if group in requested_groups}
    results = []
    for query_id, group, query in zip(query_ids, filters, queries):
        allowed = members.get(group, [])
        target = min(k, len(allowed))
        delta = vectors[allowed].astype(np.float64) - query
        distances = dict(zip(allowed, np.einsum("ij,ij->i", delta, delta).tolist()))
        exact = sorted(allowed, key=lambda i: (distances[i], ids[i]))
        cutoff = distances[exact[target - 1]] if target else None
        global_keys = neighbors(global_index, query, candidates)
        partition_keys = [allowed[i] for i in neighbors(partitions[group], query, candidates)] if allowed else []
        methods = {}
        for name, keys in [("global_then_filter", global_keys), ("partition_first", partition_keys)]:
            eligible = [i for i in keys if i in distances]
            selected = sorted(eligible, key=lambda i: (distances[i], ids[i]))[:target]
            methods[name] = {"candidate_count": len(keys), "eligible_candidates": len(eligible),
                             "ids": [ids[i] for i in selected], "recall": grade(selected, distances, target, cutoff)}
        results.append({"query": query_id, "group": group, "eligible": len(allowed),
                        "exact_ids": [ids[i] for i in exact[:target]], "cutoff_squared_l2": cutoff, **methods})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=64)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    try:
        data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
        print(json.dumps(evaluate(data, args.k, args.candidates, args.expansion), indent=2))
    except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
