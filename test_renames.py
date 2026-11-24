from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from handbook import build, checked_sources, indexed_digests, search
from rename_demo import CLI, demo


class RenamedSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "docs"
        self.corpus.mkdir()
        self.db_path = self.root / "index.db"
        self.source = self.corpus / "old.md"
        self.source.write_bytes(b"# Payment\n\nPayment retry uses the same key.\n")
        with closing(sqlite3.connect(self.db_path)) as db:
            build(db, self.corpus)

    def cli(self, *flags):
        return subprocess.run([sys.executable, CLI, "--db", self.db_path, "query", "payment retry", *flags],
                              capture_output=True, encoding="utf-8", timeout=20)

    def test_demo_preserves_old_location_and_handles_the_ambiguous_copy(self):
        report = demo()
        fixture = json.loads(Path(__file__).with_name("examples").joinpath("renamed-citation-demo.json").read_bytes())
        self.assertEqual(report, fixture)
        original, followed, ambiguous, scoped = (report[name] for name in ("original", "followed", "ambiguous", "scoped"))
        self.assertEqual(original["source_check"]["status"], "missing")
        self.assertEqual(followed["id"], "ops/checkout.md#L3")
        self.assertEqual(followed["citation"], {"id": "ops/payment-retry.md#L3", "path": "ops/payment-retry.md",
                                               "start_line": 3, "end_line": 3})
        self.assertEqual(ambiguous["source_check"]["status"], "ambiguous")
        self.assertEqual(ambiguous["source_check"]["matches"], 2)
        self.assertEqual(followed, scoped)

    def test_edited_or_out_of_scope_replacement_and_legacy_index_remain_unverified(self):
        archive = self.corpus / "archive"
        archive.mkdir()
        moved = self.source.rename(archive / "new.md")
        original = moved.read_bytes()
        flags = ("--verify-sources", self.corpus, "--follow-renames")
        for extra, status in [((), "renamed"), (("--within", "old.md"), "missing")]:
            response = self.cli(*flags, *extra)
            self.assertEqual(response.returncode, 0, response.stderr)
            self.assertEqual(json.loads(response.stdout)[0]["source_check"]["status"], status)
        moved.write_bytes(original.replace(b"\n", b"\r\n"))
        response = self.cli(*flags, "--verified-only")
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertEqual(json.loads(response.stdout), {"hits": [], "scanned": 1, "skipped": {"missing": 1}, "stopped": "exhausted"})
        moved.write_bytes(original)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("DROP TABLE handbook_sources_v1")
        response = self.cli(*flags)
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertEqual(json.loads(response.stdout)[0]["source_check"]["status"], "missing")
        self.assertEqual(self.cli("--follow-renames").returncode, 2)

    def test_duplicate_old_citations_collapse_and_scan_failure_protects_uniqueness(self):
        copy = self.corpus / "duplicate.md"
        copy.write_bytes(self.source.read_bytes())
        with closing(sqlite3.connect(self.db_path)) as db:
            build(db, self.corpus)
            hits = search(db, "payment retry", 2)
            digests = indexed_digests(db, hits)
        with patch("handbook.renamed_sources", side_effect=AssertionError("unexpected scan")):
            checks = list(checked_sources(hits, self.corpus, source_digests=digests))
            self.assertEqual([hit["source_check"]["status"] for hit in checks], ["current", "current"])

        self.source.rename(self.corpus / "moved.md")
        copy.unlink()
        with patch("handbook.markdown_files", side_effect=PermissionError("cannot complete scan")) as walk:
            checks = list(checked_sources(hits, self.corpus, source_digests=digests))
            self.assertEqual([hit["source_check"]["status"] for hit in checks], ["unavailable", "unavailable"])
            walk.assert_called_once()
        response = self.cli("-k", "2", "--verify-sources", self.corpus, "--follow-renames", "--verified-only")
        self.assertEqual(response.returncode, 0, response.stderr)
        report = json.loads(response.stdout)
        self.assertEqual(report["scanned"], 2)
        self.assertEqual(report["skipped"], {"duplicate": 1})
        self.assertEqual(report["stopped"], "exhausted")
        hit, = report["hits"]
        self.assertEqual((hit["rank"], hit["citation"]["id"]), (1, "moved.md#L3"))
