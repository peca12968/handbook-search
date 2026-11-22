"""Index, search and evaluate a local Markdown handbook."""
import argparse
from contextlib import closing
import json
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


def build(db, root):
    if not root.is_dir():
        raise ValueError(f"Missing handbook directory: {root}")
    count = 0
    with db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(SCHEMA)
        db.execute("DELETE FROM handbook_passages")
        for path in sorted(root.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                source = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: Markdown must be UTF-8") from exc
            for heading, start, end, body in passages(source, path.stem):
                db.execute("INSERT INTO handbook_passages VALUES (?,?,?,?,?,?,?)", (
                    " ".join(tokens(heading, for_index=True)), " ".join(tokens(body, for_index=True)),
                    path.relative_to(root).as_posix(), heading, start, end, body))
                count += 1
        if not count:
            raise ValueError("No searchable Markdown paragraphs found")
    return {"passages": count}


def search(db, query, k):
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    words = list(dict.fromkeys(tokens(query)))
    if not 1 <= len(words) <= 64:
        raise ValueError("Expected 1 to 64 distinct search terms")
    expression = " OR ".join(f'"{word}"' for word in words)
    rows = db.execute("""
        SELECT path, heading, start_line, end_line, passage FROM handbook_passages
        WHERE handbook_passages MATCH ?
        ORDER BY bm25(handbook_passages, 3, 1), path, start_line LIMIT ?
    """, (expression, k))
    return [{"id": f"{path}#L{start}", "path": path, "heading": heading, "start_line": start,
             "end_line": end, "passage": body} for path, heading, start, end, body in rows]


def evaluate(db, dataset, k):
    cases = json.loads(dataset.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty list of evaluation cases")
    known = {f"{path}#L{line}" for path, line in db.execute("SELECT path,start_line FROM handbook_passages")}
    details = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("query"), str):
            raise ValueError(f"Case {index}: missing query")
        labels = case.get("relevant")
        if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
            raise ValueError(f"Case {index}: expected a nonempty list of relevant passage IDs")
        relevant = set(labels)
        if relevant - known:
            raise ValueError(f"Case {index}: missing labeled passage {sorted(relevant - known)}")
        ids = [row["id"] for row in search(db, case["query"], k)]
        ranks = [rank for rank, key in enumerate(ids, 1) if key in relevant]
        details.append({"query": case["query"], "retrieved": ids, "recall": len(ranks) / len(relevant),
                        "reciprocal_rank": 1 / ranks[0] if ranks else 0})
    return {"k": k, "queries": len(cases), "mean_recall": sum(x["recall"] for x in details) / len(cases),
            "mrr": sum(x["reciprocal_rank"] for x in details) / len(cases), "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("handbook.db"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index").add_argument("directory", type=Path)
    query = commands.add_parser("query")
    query.add_argument("text")
    query.add_argument("-k", type=int, default=3)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "index":
            with closing(sqlite3.connect(args.db, timeout=15)) as db:
                result = build(db, args.directory)
        else:
            with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                with db:
                    db.execute("BEGIN")
                    result = (search(db, args.text, args.k) if args.command == "query"
                              else evaluate(db, args.dataset, args.k))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
