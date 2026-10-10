"""Run the example request through the LiteRT graphs and print the response.

    python examples/run_example.py [--dir .] [--window 384] [--accel cpu|gpu] [--form int8emb|embeds] [--check]

--check compares the answers with examples/run_example.expected.json (desktop CPU, window 384, int8emb), ignoring latency_ms.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from phocinae_litert import PhocinaeLiteRT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT))
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--form", choices=["int8emb", "embeds"], default="int8emb")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    host = PhocinaeLiteRT(a.dir, a.window, a.accel, a.form)
    req = json.loads((ROOT / "examples" / "request.json").read_text())
    out = host.decide(req)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if a.check:
        exp = json.loads((ROOT / "examples" / "run_example.expected.json").read_text())
        got = {k: v for k, v in out.items() if k != "latency_ms"}
        ok = got == exp
        print("CHECK", "PASS" if ok else "FAIL (the answers differ from the expected file)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
