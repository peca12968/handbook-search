"""Rehearse a handbook revision in a temporary directory using the public CLI."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    project = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        corpus = base / "docs"
        shutil.copytree(project / "examples/handbook", corpus)
        retired = corpus / "retired.md"
        retired.write_text("# Retired\nlegacyprocedure\n", encoding="utf-8")

        def cli(*args):
            raw = subprocess.check_output([sys.executable, str(project / "handbook.py"),
                                           "--db", str(base / "index.db"), *map(str, args)])
            return json.loads(raw)

        before = cli("sync", corpus)
        with (corpus / "checkout.md").open("a", encoding="utf-8") as stream:
            stream.write("\n## Operator approval\n\nRequire supervisorapproval before release.\n")
        (corpus / "operations.md").rename(corpus / "oncall.md")
        retired.unlink()
        after = cli("sync", corpus)
        replay = cli("sync", corpus)
        approval = cli("query", "supervisorapproval")[0]["id"]
        retired_hits = len(cli("query", "legacyprocedure"))
        if (after != {"passages": 12, "mode": "incremental", "added": 1, "changed": 1,
                      "removed": 2, "unchanged": 1} or replay["unchanged"] != 3 or retired_hits):
            raise RuntimeError("Index changes did not match the rehearsal")
        print(json.dumps({"initial": before, "revision": after, "repeat": replay,
                          "approval_passage": approval, "retired_hits": retired_hits}, indent=2))


if __name__ == "__main__":
    main()
