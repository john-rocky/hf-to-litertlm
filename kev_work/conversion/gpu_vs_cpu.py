"""GPU-vs-CPU comparison of one file after both parity runs exist (used when the GPU run finished before the CPU run
of the same file, so its parity json says gpu_vs_cpu_same_file.available = false).

    python gpu_vs_cpu.py --L 1024 --variant v2_fp16fc_i8emb --tag gpu_f32 [--model 4b] [--cache <dir>] [--cpu-sfx _smoke50]

Reads results/litert_{tag}_rows_L{L}[_4b]_{variant}.json + <cache>/hsel_{tag}_L{L}[_4b]_{variant}.npz (GPU) and the CPU
run's rows json + npz of the same file; writes results/litert_{tag}_vs_cpu_L{L}[_4b]_{variant}[{cpu-sfx}].json (never
overwritten). --cpu-sfx _smoke50 compares with a CPU run of the first N questions (litert_parity.py --limit N); only
the questions both runs have are compared."""
import argparse
import json

import numpy as np

from r2_common import MODEL, RESULT_SUFFIX, K, add_model_arg, cache_dir, dump_json
from r3_common import gpu_vs_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tag", choices=["gpu_f32", "gpu_f16"], required=True)
    ap.add_argument("--cache", default="", help="dir of the h_sel npz (default cache/r3 for 0.8B, cache/r4b for 4B)")
    ap.add_argument("--cpu-sfx", default="", help="suffix of a partial CPU run (e.g. _smoke50)")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    cache = (K / a.cache) if a.cache else cache_dir("cache/r3")
    sfx = f"{RESULT_SUFFIX}_{a.variant}"
    out = K / f"results/litert_{a.tag}_vs_cpu_L{a.L}{sfx}{a.cpu_sfx}.json"
    assert not out.exists(), f"refusing to overwrite {out.name}"
    rows = json.loads((K / f"results/litert_{a.tag}_rows_L{a.L}{sfx}.json").read_text())["rows"]
    g = np.load(cache / f"hsel_{a.tag}_L{a.L}{sfx}.npz")
    cpu_rows_path = K / f"results/litert_cpu_rows_L{a.L}{sfx}{a.cpu_sfx}.json"
    if a.cpu_sfx:   # partial CPU run: compare only the questions it has
        have = {r["key"] for r in json.loads(cpu_rows_path.read_text())["rows"]}
        rows = [r for r in rows if r["key"] in have]
    doc = gpu_vs_cpu(rows, cpu_rows_path, {k: g[k] for k in g.files}, cache / f"hsel_cpu_L{a.L}{sfx}{a.cpu_sfx}.npz")
    doc = {"what": f"{a.tag} vs cpu, same file (L={a.L} {a.variant}, model {MODEL})",
           "gpu_parity_file": f"results/litert_{a.tag}_parity_L{a.L}{sfx}.json", **doc}
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "top10_by_dp"}, indent=1))


if __name__ == "__main__":
    main()
