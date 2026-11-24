"""Index, search and evaluate a local Markdown handbook."""
import argparse
from collections import defaultdict
from contextlib import closing
import fnmatch
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

HAN = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]+)")
WORD = re.compile(r"[^\W_]+")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
SCHEMA = """CREATE VIRTUAL TABLE IF NOT EXISTS handbook_passages USING fts5(
    title_terms, body_terms, path UNINDEXED, heading UNINDEXED,
    start_line UNINDEXED, end_line UNINDEXED, passage UNINDEXED)"""


def tokens(text, *, for_index=False):
    text = unicodedata.normalize("NFKC", text).casefold()
    for index, chunk in enumerate(HAN.split(text)):
        if index % 2:
            if for_index or len(chunk) == 1:
                yield from chunk
            yield from (chunk[i:i + 2] for i in range(len(chunk) - 1))
        else:
            yield from WORD.findall(chunk)


def passages(text, default_heading):
    heading = default_heading
    fence = None
    lines = []
    start = 0
    for number, line in enumerate([*text.splitlines(), ""], 1):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = None
            continue
        title = HEADING.match(line)
        if title or marker or not line.strip():
            if lines:
                yield heading, start, number - 1, "\n".join(lines)
                lines.clear()
            if title:
                heading = title[1]
            if marker:
                fence = marker[1]
        else:
            if not lines:
                start = number
            lines.append(line)


def markdown_files(root):
    # An unreadable subtree must abort the scan, or its files would appear deleted.
    def raise_error(error):
        raise error

    for directory, folders, names in os.walk(root, onerror=raise_error, followlinks=False):
        folders.sort()
        for name in sorted(fnmatch.filter(names, "*.md")):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def insert_passages(db, raw, path):
    try:
        source = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: Markdown must be UTF-8") from exc
    first = last = None
    for heading, start, end, body in passages(source, Path(path).stem):
        row = db.execute("INSERT INTO handbook_passages VALUES (?,?,?,?,?,?,?)", (
            " ".join(tokens(heading, for_index=True)), " ".join(tokens(body, for_index=True)),
            path, heading, start, end, body))
        last = row.lastrowid
        if first is None:
            first = last
    return first, last


def current_passages(root, path):
    try:
        validate_scope(path)
        candidate = root / path
        if not candidate.is_relative_to(root):
            return "unavailable", {}
        resolved = candidate.resolve(strict=True)
        if candidate != resolved or not resolved.is_relative_to(root) or not resolved.is_file():
            return "unavailable", {}
        text = resolved.read_bytes().decode("utf-8-sig")
    except (FileNotFoundError, NotADirectoryError):
        return "missing", {}
    except (OSError, RuntimeError, ValueError):
        return "unavailable", {}

    ranges = defaultdict(list)
    for heading, start, end, body in passages(text, resolved.stem):
        ranges[heading, body].append((start, end))
    return "changed", ranges


def checked_sources(hits, directory):
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Missing handbook directory: {root}")
    scanned = {}
    for hit in hits:
        path = hit["path"]
        if path not in scanned:
            scanned[path] = current_passages(root, path)
        status, passages_by_text = scanned[path]
        matches = passages_by_text.get((hit["heading"], hit["passage"]), [])
        original = (hit["start_line"], hit["end_line"])
        if original in matches:
            status, matches = "current", [original]
        elif len(matches) == 1:
            status = "moved"
        elif matches:
            status = "ambiguous"
        hit["source_check"] = {
            "status": status,
            "locations": [{"id": f"{path}#L{start}", "start_line": start, "end_line": end}
                          for start, end in matches[:50]],
            "matches": len(matches),
        }
        yield hit


def verify_sources(hits, directory):
    for _ in checked_sources(hits, directory):
        pass
    return hits


def confirmed_hits(candidates, directory, k, scan_limit):
    hits = []
    seen = set()
    skipped = defaultdict(int)
    scanned = 0
    for scanned, hit in enumerate(checked_sources(islice(candidates, scan_limit), directory), 1):
        check = hit["source_check"]
        if check["status"] not in ("current", "moved"):
            skipped[check["status"]] += 1
            continue
        citation = check["locations"][0]
        if citation["id"] in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(citation["id"])
        hits.append({**hit, "rank": scanned, "citation": citation})
        if len(hits) == k:
            break
    stopped = "filled" if len(hits) == k else "scan_limit" if len(candidates) > scan_limit else "exhausted"
    return {"hits": hits, "scanned": scanned, "skipped": dict(sorted(skipped.items())), "stopped": stopped}


def build(db, root, *, incremental=False):
    if not root.is_dir():
        raise ValueError(f"Missing handbook directory: {root}")
    with db:
        db.execute("BEGIN IMMEDIATE")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
        reuse = incremental and {"handbook_passages", "handbook_sources_v1"} <= tables
        db.execute(SCHEMA)
        # The writer inserts each file's passages consecutively inside this
        # transaction. Its rowid range lets us remove just that file from FTS5.
        # Advance the manifest version if passage or tokenizer semantics change.
        db.execute("""CREATE TABLE IF NOT EXISTS handbook_sources_v1 (
            path TEXT PRIMARY KEY, digest BLOB NOT NULL, first_id INTEGER, last_id INTEGER
        ) WITHOUT ROWID""")
        if not reuse:
            db.execute("DELETE FROM handbook_passages")
            db.execute("DELETE FROM handbook_sources_v1")
        remaining = {path: (digest, first, last) for path, digest, first, last in db.execute(
            "SELECT path,digest,first_id,last_id FROM handbook_sources_v1")}
        stats = {"mode": "incremental" if reuse else "bootstrap", "added": 0, "changed": 0,
                 "removed": 0, "unchanged": 0}
        count = 0
        for file in markdown_files(root):
            path = file.relative_to(root).as_posix()
            raw = file.read_bytes()
            digest = hashlib.sha256(raw).digest()
            old = remaining.pop(path, None)
            if old and digest == old[0]:
                _, first, last = old
                stats["unchanged"] += 1
            else:
                if old:
                    db.execute("DELETE FROM handbook_passages WHERE rowid BETWEEN ? AND ?", old[1:])
                stats["changed" if old else "added"] += 1
                first, last = insert_passages(db, raw, path)
                db.execute("INSERT OR REPLACE INTO handbook_sources_v1 VALUES (?,?,?,?)",
                           (path, digest, first, last))
            if first is not None:
                count += last - first + 1
        for path, (_, first, last) in remaining.items():
            db.execute("DELETE FROM handbook_passages WHERE rowid BETWEEN ? AND ?", (first, last))
            db.execute("DELETE FROM handbook_sources_v1 WHERE path=?", (path,))
        stats["removed"] = len(remaining)
        if not count:
            raise ValueError("No searchable Markdown paragraphs found")
    return {"passages": count, **(stats if incremental else {})}


def alias_units(text):
    """Match Han characters inside unspaced text and other words as whole units."""
    for index, chunk in enumerate(HAN.split(unicodedata.normalize("NFKC", text).casefold())):
        yield from (chunk if index % 2 else WORD.findall(chunk))


def load_glossary(path):
    rules = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rules, list) or not rules:
        raise ValueError("Expected a nonempty list of glossary rules")
    by_length = defaultdict(dict)
    for index, rule in enumerate(rules, 1):
        if not isinstance(rule, dict) or not all(isinstance(rule.get(key), str) for key in ("from", "to")):
            raise ValueError(f"Glossary rule {index}: expected 'from' and 'to' strings")
        source = tuple(alias_units(rule["from"]))
        target = " ".join(tokens(rule["to"]))
        if not source or not target:
            raise ValueError(f"Glossary rule {index}: empty source or target after tokenization")
        bucket = by_length[len(source)]
        if source in bucket:
            raise ValueError(f"Glossary rule {index}: duplicate normalized source")
        bucket[source] = target
    return dict(by_length)


def expansion_phrases(query, glossary):
    units = tuple(alias_units(query))
    for size, rules in glossary.items():
        for start in range(len(units) - size + 1):
            phrase = rules.get(units[start:start + size])
            if phrase:
                yield phrase


def validate_scope(scope):
    if scope is not None and (not isinstance(scope, str) or "\\" in scope or "\0" in scope or
                              any(part in ("", ".", "..") for part in scope.split("/"))):
        raise ValueError("Scope must be a relative indexed path using / between components")


def search(db, query, k, glossary=None, *, within=None):
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    return ranked_hits(db, query, k, glossary, within=within)


def ranked_hits(db, query, limit, glossary=None, *, within=None):
    words = list(dict.fromkeys(tokens(query)))
    if not 1 <= len(words) <= 64:
        raise ValueError("Expected 1 to 64 distinct search terms")
    phrases = dict.fromkeys(words)
    if glossary:
        phrases.update(dict.fromkeys(expansion_phrases(query, glossary)))
    expression = " OR ".join(f'"{phrase}"' for phrase in phrases)
    validate_scope(within)
    scope_clause = "" if within is None else " AND (path = ? OR instr(path, ? || '/') = 1)"
    bindings = [expression] if within is None else [expression, within, within]
    rows = db.execute(f"""
        SELECT path, heading, start_line, end_line, passage FROM handbook_passages
        WHERE handbook_passages MATCH ? {scope_clause}
        ORDER BY bm25(handbook_passages, 3, 1), path, start_line LIMIT ?
    """, (*bindings, limit))
    return [{"id": f"{path}#L{start}", "path": path, "heading": heading, "start_line": start,
             "end_line": end, "passage": body} for path, heading, start, end, body in rows]


def evaluate(db, dataset, k, glossary=None):
    cases = json.loads(dataset.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty list of evaluation cases")
    known = {f"{path}#L{line}": path for path, line in db.execute("SELECT path,start_line FROM handbook_passages")}
    details = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("query"), str):
            raise ValueError(f"Case {index}: missing query")
        labels = case.get("relevant")
        if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
            raise ValueError(f"Case {index}: expected a nonempty list of relevant passage IDs")
        relevant = set(labels)
        if missing := relevant - known.keys():
            raise ValueError(f"Case {index}: missing labeled passage {sorted(missing)}")
        scope = case.get("within")
        validate_scope(scope)
        if scope is not None and any(known[label] != scope and not known[label].startswith(scope + "/") for label in relevant):
            raise ValueError(f"Case {index}: relevant passage outside requested scope")
        ids = [row["id"] for row in search(db, case["query"], k, glossary, within=scope)]
        ranks = [rank for rank, key in enumerate(ids, 1) if key in relevant]
        details.append({"query": case["query"], "retrieved": ids, "recall": len(ranks) / len(relevant),
                        "reciprocal_rank": 1 / ranks[0] if ranks else 0})
        if scope is not None:
            details[-1]["within"] = scope
    return {"k": k, "queries": len(cases), "mean_recall": sum(x["recall"] for x in details) / len(cases),
            "mrr": sum(x["reciprocal_rank"] for x in details) / len(cases), "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("handbook.db"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index").add_argument("directory", type=Path)
    commands.add_parser("sync", help="update only new or changed Markdown files").add_argument("directory", type=Path)
    query = commands.add_parser("query")
    query.add_argument("text")
    query.add_argument("-k", type=int, default=3)
    query.add_argument("--within", help="Restrict retrieval to an indexed file or directory")
    query.add_argument("--verify-sources", type=Path, help="check matched citations against the current Markdown directory")
    query.add_argument("--verified-only", action="store_true", help="return only citations confirmed in the source directory")
    query.add_argument("--scan-limit", type=int, help="verified-only candidate limit, default 100, up to 500")
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("-k", type=int, default=3)
    for command in (query, evaluation):
        command.add_argument("--glossary", type=Path, help="JSON file of directional phrase aliases")
    args = parser.parse_args()
    try:
        if args.command in ("index", "sync"):
            with closing(sqlite3.connect(args.db, timeout=15)) as db:
                result = build(db, args.directory, incremental=args.command == "sync")
        else:
            glossary = load_glossary(args.glossary) if args.glossary else None
            with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                with db:
                    db.execute("BEGIN")
                    if args.command == "eval":
                        result = evaluate(db, args.dataset, args.k, glossary)
                    elif args.verified_only:
                        if args.verify_sources is None:
                            raise ValueError("--verified-only requires --verify-sources")
                        if not 1 <= args.k <= 50:
                            raise ValueError("k must be between 1 and 50")
                        scan_limit = 100 if args.scan_limit is None else args.scan_limit
                        if not args.k <= scan_limit <= 500:
                            raise ValueError("--scan-limit must be between k and 500")
                        result = ranked_hits(db, args.text, scan_limit + 1, glossary, within=args.within)
                    else:
                        if args.scan_limit is not None:
                            raise ValueError("--scan-limit requires --verified-only")
                        result = search(db, args.text, args.k, glossary, within=args.within)
            if args.command == "query" and args.verify_sources is not None:
                result = (confirmed_hits(result, args.verify_sources, args.k, scan_limit) if args.verified_only
                          else verify_sources(result, args.verify_sources))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
