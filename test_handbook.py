import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from handbook import build, evaluate, load_glossary, passages, search, verify_sources

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

    def test_citation_demo_matches_cli_with_an_old_index(self):
        demo = json.loads((ROOT / "examples" / "citations-demo.json").read_text(encoding="utf-8"))
        original = (ROOT / "examples" / "handbook" / "checkout.md").read_text(encoding="utf-8")
        source = self.put("checkout.md", original)
        build(self.db, self.corpus)
        # Older indexes have the original passages but no source manifest.
        self.db.execute("DROP TABLE handbook_sources_v1")
        self.db.commit()
        shifted = "# 維運補充\n\n本文在近期更新過。\n\n" + original
        scenarios = [
            ("current", original),
            ("moved", shifted),
            ("ambiguous", shifted + "\n" + original),
            ("changed", shifted.replace("沿用原請求鍵", "先查詢既有請求的狀態")),
            ("missing", None),
        ]
        args = [sys.executable, ROOT / "handbook.py", "--db", self.root / "index.db",
                "query", demo["query"], "-k", "1", "--within", "checkout.md"]
        for (expected, text), scenario in zip(scenarios, demo["scenarios"], strict=True):
            with self.subTest(status=expected):
                if text is None:
                    source.unlink()
                else:
                    source.write_text(text, encoding="utf-8")
                result = subprocess.run([*args, "--verify-sources", self.corpus],
                                        capture_output=True, encoding="utf-8", timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                hit, = json.loads(result.stdout)
                self.assertEqual(scenario["status"], expected)
                self.assertEqual(hit.pop("source_check"), scenario)
                self.assertEqual(hit, demo["hit"])
        plain = subprocess.run(args, capture_output=True, encoding="utf-8", timeout=15)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertEqual(json.loads(plain.stdout), [demo["hit"]])

    def test_source_verification_marks_bad_text_or_unsafe_paths_unavailable(self):
        source = self.put("guide.md", "same text")
        build(self.db, self.corpus)
        hit = search(self.db, "same", 1)[0]
        source.write_bytes(b"\xff")
        outside = self.root / "outside.md"
        outside.write_text("same text", encoding="utf-8")
        checks = verify_sources([hit, {**hit, "path": "../outside.md"}], self.corpus)
        self.assertEqual([row["source_check"] for row in checks],
                         [{"status": "unavailable", "locations": [], "matches": 0}] * 2)

    def test_glossary_matches_whole_words_and_expands_a_literal_phrase_once(self):
        self.put("match.md", "# Deploy\nrelease rollback\n")
        self.put("partial.md", "# Deploy\nrelease requires a rollback\n")
        self.put("chain.md", "# Deploy\nsecondary\n")
        build(self.db, self.corpus)
        path = self.root / "rules.json"
        path.write_text(json.dumps([{"from": "Roll Back", "to": "release rollback"},
                                    {"from": "release rollback", "to": "secondary"}]), encoding="utf-8")
        glossary = load_glossary(path)
        hits = search(self.db, "ＲＯＬＬ back", 5, glossary)
        self.assertEqual([hit["path"] for hit in hits], ["match.md"])
        self.assertEqual(search(self.db, "enroll backing", 5, glossary), [])

    def test_invalid_glossary_reports_rule_number(self):
        path = self.root / "rules.json"
        for rules, message in [
            ([], "nonempty"),
            ([{"from": "alias"}], "rule 1"),
            ([{"from": "!!", "to": "word"}], "rule 1"),
            ([{"from": "Ａ", "to": "first"}, {"from": "a", "to": "second"}], "rule 2"),
        ]:
            with self.subTest(rules=rules):
                path.write_text(json.dumps(rules), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_glossary(path)

    def test_example_glossary_handles_other_queries_in_both_directions(self):
        build(self.db, ROOT / "examples/handbook")
        glossary = load_glossary(ROOT / "examples/glossary.json")
        for query, expected in [
            ("如何設定工作租約", "operations.md#L7"),
            ("版本回退", "operations.md#L13"),
            ("資料檢查點", "pipeline.md#L13"),
            ("restore drill", "operations.md#L10"),
        ]:
            with self.subTest(query=query):
                self.assertEqual(search(self.db, query, 3, glossary)[0]["id"], expected)

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

    def test_scope_is_literal_and_applied_before_limit_and_metrics(self):
        for folder in ("docs", "docs-old", "docs_100%", "docsX100X"):
            (self.corpus / folder).mkdir()
            self.put(f"{folder}/a.md", "# Guide\nparcel\n")
        build(self.db, self.corpus)
        target = "docs_100%/a.md#L2"
        self.assertNotEqual(search(self.db, "parcel", 1)[0]["id"], target)
        self.assertEqual([r["id"] for r in search(self.db, "parcel", 4, within="docs")], ["docs/a.md#L2"])
        self.assertEqual(search(self.db, "parcel", 1, within="docs_100%")[0]["id"], target)
        self.assertEqual(search(self.db, "parcel", 3, within="absent"), [])
        for invalid in ("", "../docs", "docs/", "/docs", "docs//file"):
            with self.assertRaisesRegex(ValueError, "Scope"):
                search(self.db, "parcel", 1, within=invalid)
        cases = self.root / "scoped.json"
        case = {"query": "parcel", "within": "docs_100%", "relevant": [target]}
        cases.write_text(json.dumps([case]))
        report = evaluate(self.db, cases, 1)
        self.assertEqual(report["details"][0]["within"], case["within"])
        self.assertEqual(report["mean_recall"], 1)
        cases.write_text(json.dumps([{**case, "within": "docs-old"}]))
        with self.assertRaisesRegex(ValueError, "outside requested scope"):
            evaluate(self.db, cases, 1)

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
        scope = run("query", "付款重送 payment retry", "-k", "1", "--within", "pipeline.md")
        self.assertEqual(json.loads(scope.stdout)[0]["id"], "pipeline.md#L10")
        report = json.loads(run("eval", ROOT / "examples/queries.json", "-k", "3").stdout)
        self.assertEqual(report["queries"], 13)
        self.assertGreaterEqual(report["mean_recall"], 0.90)
        self.assertGreaterEqual(report["mrr"], 0.90)
        glossary_args = ["--glossary", ROOT / "examples/glossary.json"]
        improved = json.loads(run("eval", ROOT / "examples/queries.json", *glossary_args).stdout)
        for before, after in zip(report["details"], improved["details"], strict=True):
            with self.subTest(query=before["query"]):
                self.assertEqual(after["query"], before["query"])
                if before["query"] == "背景執行者當機由誰接手":
                    self.assertEqual(before["recall"], 0)
                    self.assertEqual(after["retrieved"][0], "operations.md#L7")
                else:
                    self.assertEqual(after["retrieved"], before["retrieved"])
        self.assertEqual((improved["mean_recall"], improved["mrr"]), (1, 1))
        hit = json.loads(run("query", "背景執行者當機由誰接手", *glossary_args).stdout)[0]
        self.assertEqual(hit["id"], "operations.md#L7")
        self.assertIn("When its lease expires", hit["passage"])
        for args in [("query", "..."), ("query", "term", "-k", "0")]:
            failed = run(*args)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(failed.stdout, "")
