"""One table over the shipped graphs of a form: per file its bytes, sha256 (from the export record), op count, FC
weight dtypes, the five largest runtime buffers by distinct shape (DEQUANTIZE outputs excluded: the GPU takes the
float16 constants as weights), and the count of risky operand pairs (rank_dump: runtime rank 3 + constant rank 4).

  $EXPORT conversion/bucket_table.py [--form fp16fc] [--json results/bucket_table.json]
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def main():
    form = sys.argv[sys.argv.index("--form") + 1] if "--form" in sys.argv else "fp16fc"
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    import rank_dump
    from tflite_scan_d1 import scan

    gdir = os.path.join(WORK, "out/graphs")
    rows = []
    for name in sorted(os.listdir(gdir), key=lambda n: (("tower" in n), [int(x) for x in re.findall(r"\d+", n)])):
        if not name.endswith(f"_{form}.tflite"):
            continue
        path = os.path.join(gdir, name)
        tag = name[len("d1-2b_"):-len(f"_{form}.tflite")]
        kind, _, rest = tag.partition("_")
        rec_path = os.path.join(WORK, f"results/export_{kind}_{rest}.json")
        rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
        s = scan(path, with_sha=False)
        seen, top = set(), []
        for t in s["largest_activations"]:
            key = tuple(t["shape"])
            if key in seen:
                continue
            seen.add(key)
            top.append({"shape": t["shape"], "MiB": round(t["bytes"] / 2 ** 20, 1)})
            if len(top) == 5:
                break
        sys.argv = ["rank_dump.py", path]
        buf = io.StringIO()
        with redirect_stdout(buf):
            pats = rank_dump.main()
        risk = sum(p["count"] for p in pats if p["risk"].startswith("RISK"))
        rows.append({"file": os.path.relpath(path, WORK), "bytes": os.path.getsize(path),
                     "sha256": (rec.get(form) or {}).get("sha256"), "ops": s["operator_count"],
                     "fc_weight_dtypes": s["fc_weight_dtypes"], "custom_ops": s["custom_ops"],
                     "rank_gt4": s["rank_gt4_tensor_count"], "int64": s["int64_tensor_count"],
                     "largest_runtime_buffers": top, "risk_pairs": risk,
                     "mixed_rank_patterns": [f"{p['op']} {' + '.join(p['inputs'])} x{p['count']}" for p in pats],
                     "peak_rss_gb": round(rec.get("peak_rss_bytes", 0) / 2 ** 30, 1) if rec else None,
                     "cpu_vs_module": rec.get("fp32_cpu_vs_module_real_max_abs") or rec.get("fp32_cpu_vs_module_max_abs")})
        r = rows[-1]
        print(f"| `{name}` | {r['bytes']:,} | {r['ops']} | {r['fc_weight_dtypes']} | "
              + "; ".join(f"{t['shape']} {t['MiB']}" for t in top) + f" | {risk} |")
    if out_json:
        json.dump(rows, open(os.path.join(WORK, out_json), "w"), indent=1)


if __name__ == "__main__":
    main()
