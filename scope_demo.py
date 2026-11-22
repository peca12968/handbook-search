"""Compare source-scoped text and vector retrieval using the same labeled passages."""
import argparse
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

from handbook import build, evaluate, search, tokens


def exercise(sieve):
    topics = [
        ("付款重送 payment retry", "沿用原請求鍵。 Reuse the original request key."),
        ("備份還原 backup restore", "先核對摘要。 Verify the recorded digest first."),
    ]
    with TemporaryDirectory() as temp:
        root = Path(temp)
        corpus = root / "handbook"
        for section in ("archive", "current"):
            (corpus / section).mkdir(parents=True)
        cases = []
        for i, (query, answer) in enumerate(topics):
            for version in range(12):
                (corpus / "archive" / f"{i}-{version}.md").write_text(f"# {query}\n{query}\n", encoding="utf-8")
            path = f"current/topic-{i}.md"
            (corpus / path).write_text(f"# {query}\n{answer}\n", encoding="utf-8")
            cases.append({"query": query, "within": "current", "relevant": [f"{path}#L2"]})
        labels = root / "labels.json"
        labels.write_text(json.dumps(cases), encoding="utf-8")
        with closing(sqlite3.connect(":memory:")) as db:
            build(db, corpus)
            assert evaluate(db, labels, 1)["mean_recall"] == 1
            items = []
            for path, heading, line, body in db.execute("select path,heading,start_line,passage from handbook_passages"):
                items.append((f"{path}#L{line}", path.split("/")[0], heading + " " + body))
            for case, (_, text) in zip(cases, topics, strict=True):
                hit, = search(db, case["query"], 1, within=case["within"])
                assert hit["id"] == case["relevant"][0] and hit["passage"] == text
                assert (corpus / hit["path"]).read_text(encoding="utf-8").splitlines()[hit["start_line"]-1] == text
        vocab = sorted({word for _, _, content in items for word in tokens(content)})
        def record(key, group, text):
            counts = Counter(tokens(text))
            norm = math.sqrt(sum(n*n for n in counts.values()))
            return {"id": key, "group": group, "vector": [counts[word]/norm for word in vocab]}
        vectors = {"items": [record(*row) for row in items], "queries": [
            record(f"q{i}", case["within"], case["query"]) for i, case in enumerate(cases)]}
        dataset = root / "vectors.json"
        dataset.write_text(json.dumps(vectors), encoding="utf-8")
        rows = json.loads(subprocess.check_output([sys.executable, sieve, dataset, "--k", "1", "--candidates", "4"]))
        assert len(rows) == len(cases)
        for row, case in zip(rows, cases, strict=True):
            assert row["global_then_filter"]["ids"] == []
            assert row["partition_first"]["ids"] == row["exact_ids"] == case["relevant"]
        return {"passages": len(items), "queries": len(cases), "scoped_fts_recall_at_1": 1,
                "global_vector_hits": 0, "scoped_vector_hits": len(rows),
                "cited_lines": [row["exact_ids"][0] for row in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vector_sieve", type=Path)
    print(json.dumps(exercise(parser.parse_args().vector_sieve.resolve()), indent=2))
