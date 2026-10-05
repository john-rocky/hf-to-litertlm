"""The two Ls256 request files the timed request legs ran on the phone (shared_timing on the Ls256 pair, with and without
GPU constant tensor sharing): the own_order_06 (2 questions) and own_email_03 (3 questions) entries of
r14_device_rows_req.py's requests_Ls256_Lq64_<share|noshare>_req.json, unchanged and in that file's order, with
timing_requests set to those two (the app times the requests named in timing_requests).

    python3 scripts/r18_req2.py [--src device/r14] [--out device/r14]

Writes <out>/requests_Ls256_Lq64_<share|noshare>_req2.json (never overwrites a file with other content)."""
import argparse
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
POINTS = ["own_order_06", "own_email_03"]


def write(path, doc):
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    if path.exists():
        assert path.read_text() == text, f"{path} exists with other content"
    else:
        path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="device/r14", help="the folder of r14_device_rows_req.py's files")
    ap.add_argument("--out", default="device/r14")
    a = ap.parse_args()
    src_dir, out_dir = K / a.src, K / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("share", "noshare"):
        src = src_dir / f"requests_Ls256_Lq64_{mode}_req.json"
        d = json.loads(src.read_text())
        reqs = [r for r in d["requests"] if r["request"] in POINTS]
        assert [r["request"] for r in reqs] == POINTS, [r["request"] for r in reqs]
        doc = {**{k: v for k, v in d.items() if k not in ("requests", "timing_requests", "note")},
               "timing_requests": POINTS, "requests": reqs,
               "note": f"request block v2 (Ls256 points only), from {src.name}"}
        p = write(out_dir / f"requests_Ls256_Lq64_{mode}_req2.json", doc)
        print(p.relative_to(K), p.stat().st_size)


if __name__ == "__main__":
    main()
