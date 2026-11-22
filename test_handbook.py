import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from handbook import build, evaluate, passages, search

ROOT = Path(__file__).resolve().parent


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.corpus = self.root / "docs"
        self.corpus.mkdir()
        self.db = sqlite3.connect(self.root / "index.db")
        self.addCleanup(self.db.close)

    def put(self, name, text):
        path = self.corpus / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_bilingual_tokenization_normalization_and_source_lines(self):
        self.put("guide.md", "# Retry\n\n重送訂單使用相同鍵。\nUse the ＳＡＭＥ key.\n")
        build(self.db, self.corpus)
        for query in ["重送", "鍵", "same", "重送 same"]:
            hit = search(self.db, query, 3)[0]
            self.assertEqual((hit["id"], hit["end_line"]), ("guide.md#L3", 4))
            self.assertIn("相同鍵", hit["passage"])

    def test_markdown_fences_keep_fake_headings_out_of_search(self):
        text = "## Guide\nFirst\n\n````py\n# Fake\n```\nStill code\n````\nLast\n"
        self.assertEqual(list(passages(text, "file")),
                         [("Guide", 2, 2, "First"), ("Guide", 9, 9, "Last")])

    def test_title_weight_stable_ties_and_no_match(self):
        for name in ["b.md", "a.md"]:
            self.put(name, "# cache\nlookup\n")
        self.put("c.md", "# lookup\ncache\n")
        build(self.db, self.corpus)
        hits = search(self.db, "CACHE", 3)
        self.assertEqual([row["path"] for row in hits], ["a.md", "b.md", "c.md"])
        self.assertEqual(search(self.db, "absentterm", 3), [])

    def test_rebuild_replaces_deleted_passages(self):
        old = self.put("old.md", "# First\noldterm\n")
        build(self.db, self.corpus)
        old.unlink()
        self.put("new.md", "# Second\nnewterm\n")
        self.assertEqual(build(self.db, self.corpus), {"passages": 1})
        self.assertEqual(search(self.db, "oldterm", 3), [])

    def test_invalid_file_or_empty_corpus_leaves_last_good_index(self):
        old = self.put("a.md", "# Guide\npreviousversion\n")
        build(self.db, self.corpus)
        old.write_text("replacement", encoding="utf-8")
        invalid = self.corpus / "z.md"
        invalid.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "z.md"):
            build(self.db, self.corpus)
        invalid.unlink()
        old.unlink()
        with self.assertRaisesRegex(ValueError, "No searchable"):
            build(self.db, self.corpus)
        with closing(sqlite3.connect(self.root / "index.db")) as reopened:
            self.assertEqual(search(reopened, "previousversion", 1)[0]["id"], "a.md#L2")

    def test_metric_math_handles_partial_recall_and_late_or_missing_first_hit(self):
        for letter in "abc":
            self.put(letter + ".md", "# Entry\ncommon\n")
        build(self.db, self.corpus)
        cases = self.root / "labels.json"
        cases.write_text(json.dumps([
            {"query": "common", "relevant": ["b.md#L2", "c.md#L2"]},
            {"query": "unseen", "relevant": ["a.md#L2"]},
        ]), encoding="utf-8")
        result = evaluate(self.db, cases, 2)
        self.assertEqual(result["mean_recall"], 0.25)
        self.assertEqual(result["mrr"], 0.25)
        cases.write_text('[{"query":"test","relevant":["deleted.md#L7"]}]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing labeled passage"):
            evaluate(self.db, cases, 2)

    def test_cli_round_trip_and_quality_gate(self):
        def run(*args):
            return subprocess.run([sys.executable, str(ROOT / "handbook.py"), "--db",
                                   str(self.root / "cli.db"), *map(str, args)],
                                  capture_output=True, encoding="utf-8", timeout=15)
        self.assertEqual(run("query", "unindexed").returncode, 2)
        self.assertFalse((self.root / "cli.db").exists())
        indexed = run("index", ROOT / "examples/handbook")
        self.assertEqual(indexed.returncode, 0, indexed.stderr)
        self.assertEqual(json.loads(indexed.stdout)["passages"], 11)
        self.assertEqual(json.loads(run("query", "付款重送").stdout)[0]["id"], "checkout.md#L7")
        report = json.loads(run("eval", ROOT / "examples/queries.json", "-k", "3").stdout)
        self.assertEqual(report["queries"], 13)
        self.assertGreaterEqual(report["mean_recall"], 0.90)
        self.assertGreaterEqual(report["mrr"], 0.90)
        for args in [("query", "..."), ("query", "term", "-k", "0")]:
            failed = run(*args)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(failed.stdout, "")
