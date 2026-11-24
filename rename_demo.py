"""Replay a file rename, a duplicate copy, and a scoped citation lookup."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


CLI = Path(__file__).with_name("handbook.py")


def demo():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        handbook = folder / "docs"
        operations = handbook / "ops"
        operations.mkdir(parents=True)
        source = operations / "checkout.md"
        content = "# 付款\n\n付款重送時使用同一個冪等鍵。\n"
        source.write_bytes(content.encode("utf-8"))

        def run(*args):
            response = subprocess.run([sys.executable, CLI, "--db", folder / "index.db", *args], check=True,
                                      capture_output=True, encoding="utf-8")
            return json.loads(response.stdout)

        run("index", handbook)
        source.rename(operations / "payment-retry.md")
        query = "付款重送"
        common = ("query", query, "-k", "1", "--verify-sources", handbook)
        original = run(*common)[0]
        followed = run(*common, "--follow-renames", "--verified-only")["hits"][0]

        archive = handbook / "ops-old"
        archive.mkdir()
        (archive / "copy.md").write_bytes(content.encode("utf-8"))
        ambiguous = run(*common, "--follow-renames")[0]
        scoped = run(*common, "--follow-renames", "--verified-only", "--within", "ops")["hits"][0]
        return {"query": query, "original": original, "followed": followed, "ambiguous": ambiguous, "scoped": scoped}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
