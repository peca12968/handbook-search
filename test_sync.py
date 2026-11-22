"""Incremental indexing invariants across real SQLite connections and CLI processes."""
from contextlib import closing
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import handbook


ROOT = Path(__file__).resolve().parent


class SyncTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "docs"
        self.root.mkdir()
        self.path = self.root.parent / "handbook.db"
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(text, encoding="utf-8")
        return path

    def sync(self):
        return handbook.build(self.db, self.root, incremental=True)

    def test_change_move_removal_and_unchanged_rows_match_full_rebuild(self):
        self.put("keep.md", "# Keep\nshared\n\nkeepterm\n")
        changed = self.put("edit.md", "# Edit\noldterm\n")
        moved = self.put("sub/old.md", "# Move\nshared movedterm\n")
        removed = self.put("retired.md", "obsolete")
        self.put("empty.md", "# Empty\n")
        self.assertEqual(self.sync(), {"mode": "bootstrap", "passages": 5, "added": 5,
                                       "changed": 0, "removed": 0, "unchanged": 0})
        old_ids = list(self.db.execute("SELECT rowid FROM handbook_passages WHERE path='keep.md' ORDER BY rowid"))
        # Same size and timestamp still count as changed.
        stamp = changed.stat()
        self.put("edit.md", "# Edit\nnewterm\n")
        os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        moved.rename(self.root / "sub/new.md")
        removed.unlink()
        self.put("fresh.md", "another shared paragraph")
        with patch.object(handbook, "insert_passages", wraps=handbook.insert_passages) as insert:
            self.assertEqual(self.sync(), {"mode": "incremental", "passages": 5, "added": 2,
                                           "changed": 1, "removed": 2, "unchanged": 2})
            self.assertCountEqual([call.args[2] for call in insert.call_args_list], ["edit.md", "sub/new.md", "fresh.md"])
        self.assertEqual(list(self.db.execute("SELECT rowid FROM handbook_passages WHERE path='keep.md' ORDER BY rowid")), old_ids)
        with closing(sqlite3.connect(":memory:")) as clean:
            handbook.build(clean, self.root)
            for query in ("oldterm", "obsolete", "newterm", "movedterm", "keepterm", "shared"):
                self.assertEqual(handbook.search(clean, query, 10), handbook.search(self.db, query, 10))
        before = self.db.total_changes
        with patch.object(handbook, "insert_passages", side_effect=AssertionError("unchanged source parsed")):
            self.assertEqual(self.sync()["unchanged"], 5)
        self.assertEqual(self.db.total_changes, before)

    def test_original_fts_only_index_migrates_atomically(self):
        with self.db:
            self.db.execute(handbook.SCHEMA)
            self.db.execute("INSERT INTO handbook_passages VALUES (?,?,?,?,?,?,?)",
                            ("", "legacy", "gone.md", "Gone", 2, 2, "legacy"))
        self.put("a.md", "current")
        bad = self.root / "z.md"
        bad.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.sync()
        self.assertEqual(handbook.search(self.db, "legacy", 1)[0]["path"], "gone.md")
        self.assertIsNone(self.db.execute("SELECT name FROM sqlite_schema WHERE name='handbook_sources_v1'").fetchone())
        bad.unlink()
        self.assertEqual(self.sync()["mode"], "bootstrap")
        self.assertEqual(handbook.search(self.db, "legacy", 1), [])
        self.assertEqual(self.sync()["unchanged"], 1)

    def test_reader_sees_previous_version_while_fts_and_manifest_change(self):
        self.put("a.md", "oldversion")
        self.sync()
        self.put("a.md", "newversion")
        original = handbook.insert_passages
        observed = []

        def inspect_after_insert(db, raw, path):
            ids = original(db, raw, path)
            with closing(sqlite3.connect(self.path)) as reader:
                observed.append((len(handbook.search(reader, "oldversion", 1)),
                                 len(handbook.search(reader, "newversion", 1))))
            return ids

        with patch.object(handbook, "insert_passages", side_effect=inspect_after_insert):
            self.sync()
        self.assertEqual(observed, [(1, 0)])
        self.assertEqual(handbook.search(self.db, "oldversion", 1), [])
        self.assertEqual(handbook.search(self.db, "newversion", 1)[0]["path"], "a.md")

    def test_failed_edit_rolls_back_metadata_and_last_paragraph_removal(self):
        file = self.put("a.md", "original")
        self.sync()
        manifest = self.db.execute("SELECT * FROM handbook_sources_v1").fetchall()
        self.put("a.md", "replacement")
        bad = self.root / "z.md"
        bad.write_bytes(b"\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.sync()
        self.assertEqual(self.db.execute("SELECT * FROM handbook_sources_v1").fetchall(), manifest)
        bad.unlink()
        file.unlink()
        with self.assertRaisesRegex(ValueError, "No searchable"):
            self.sync()
        self.assertEqual(handbook.search(self.db, "original", 1)[0]["path"], "a.md")
        self.put("a.md", "replacement")
        self.assertEqual(self.sync()["changed"], 1)
        self.assertEqual(handbook.search(self.db, "replacement", 1)[0]["path"], "a.md")

    def test_two_cli_processes_share_one_committed_update(self):
        self.put("a.md", "first")
        handbook.build(self.db, self.root)
        self.put("a.md", "second")
        cmd = [sys.executable, str(ROOT / "handbook.py"), "--db", str(self.path), "sync", str(self.root)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as a, \
             subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as b:
            reports = []
            for process in (a, b):
                out, err = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, err)
                reports.append(json.loads(out))
        self.assertEqual(sorted((r["changed"], r["unchanged"]) for r in reports), [(0, 1), (1, 0)])
        self.assertEqual(len(handbook.search(self.db, "second", 10)), 1)

    def test_32_successive_editions_match_a_clean_index(self):
        rng = random.Random(1841)
        self.put("anchor.md", "anchor and stable reference")
        words = ["release", "recovery", "運送", "交班"]
        for edition in range(32):
            for _ in range(4):
                name = f"team/page-{rng.randrange(8)}.md"
                if rng.randrange(4) == 0:
                    (self.root / name).unlink(missing_ok=True)
                else:
                    self.put(name, "# Guide\n" + "\n\n".join(
                        f"{rng.choice(words)} edition {edition}" for _ in range(rng.randrange(5))))
            stats = self.sync()
            with closing(sqlite3.connect(":memory:")) as clean:
                self.assertEqual(stats["passages"], handbook.build(clean, self.root)["passages"])
                for term in words + ["anchor"]:
                    self.assertEqual(handbook.search(self.db, term, 50), handbook.search(clean, term, 50), edition)


if __name__ == "__main__":
    unittest.main()
