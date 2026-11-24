"""Replay a stale citation, a moved section and a bounded verified lookup."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def demo():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sources = root / "docs"
        sources.mkdir()
        command = [sys.executable, Path(__file__).with_name("handbook.py"), "--db", root / "index.db"]
        body = "# 付款重送\n逾時後使用相同請求鍵。\n"
        for name in ("alpha.md", "bravo.md", "charlie.md"):
            (sources / name).write_text(body, encoding="utf-8")

        def run(*args):
            result = subprocess.run([*command, *args], check=True, capture_output=True, encoding="utf-8", timeout=15)
            return json.loads(result.stdout)

        run("index", sources)
        (sources / "alpha.md").write_text("# 付款重送\n先向值班主管查核付款結果。\n", encoding="utf-8")
        (sources / "bravo.md").write_text("# 上線前檢查\n\n先記錄版本。\n\n" + body, encoding="utf-8")
        query = "付款重送"
        common = ("query", query, "-k", "2", "--verify-sources", sources)
        return {"query": query, "ordinary": run(*common),
                "limited": run(*common, "--verified-only", "--scan-limit", "2"),
                "verified": run(*common, "--verified-only", "--scan-limit", "3")}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
