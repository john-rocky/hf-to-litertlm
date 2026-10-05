"""The Kev-4B shared-state pair on Metal at float32 with GPU constant tensor sharing on and off, the same file and
questions: the probabilities (host and direct hand-over) and the read-out hidden states h_sel, compared bit for bit (on
Kev-0.8B the two are bit-equal as well).

    python r15_share_compare.py --form R64+sp+ec+dd+vs6+in1+fn5     -> results/r15_share_vs_noshare_4b_<tag>.json

Reads the outputs of r15_shared_parity.py's two Metal runs (--accel gpu --f32 with and without --share) for Ls 128 and
Ls 256:
  results/litert_shared_gpu_f32[_share]_rows_Ls<Ls>_Lq64_4b_<variant>.json
  cache/r15/hsel_shared_gpu_f32[_share]_Ls<Ls>_Lq64_4b_<variant>.npz
Never overwrites."""
import argparse
import json
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
V = "v2_fp16fc_i8emb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    a = ap.parse_args()
    tag = "r15" + a.form.replace("+", "-")
    out = K / f"results/r15_share_vs_noshare_4b_{tag}.json"
    assert not out.exists(), out
    doc = {"what": "Metal enforce_f32, constant tensor sharing on vs off, same pair file and questions", "form": a.form,
           "pairs": {}}
    for P in ("Ls128_Lq64", "Ls256_Lq64"):
        sfx = f"{P}_4b_{V}_{tag}"
        on_rows, off_rows = (K / f"results/litert_shared_gpu_f32{s}_rows_{sfx}.json" for s in ("_share", ""))
        on_h, off_h = (K / f"cache/r15/hsel_shared_gpu_f32{s}_{sfx}.npz" for s in ("_share", ""))
        if not (on_rows.exists() and off_rows.exists()):
            doc["pairs"][P] = {"available": False}
            continue
        on = {r["key"]: r for r in json.loads(on_rows.read_text())["rows"]}
        off = {r["key"]: r for r in json.loads(off_rows.read_text())["rows"]}
        keys = sorted(set(on) & set(off))
        res = {"available": True, "questions": len(keys), "only_on": len(set(on) - set(off)), "only_off": len(set(off) - set(on)),
               "rows_files": [str(on_rows.relative_to(K)), str(off_rows.relative_to(K))]}
        for field in ("probs", "probs_direct"):
            d = [np.abs(np.asarray(on[k][field], np.float64) - np.asarray(off[k][field], np.float64)).max() for k in keys]
            res[f"{field}_bit_equal"] = int(sum(np.array_equal(np.asarray(on[k][field]), np.asarray(off[k][field])) for k in keys))
            res[f"{field}_max_abs_diff"] = float(max(d))
        res["argmax_equal"] = int(sum(on[k]["argmax_key"] == off[k]["argmax_key"] for k in keys))
        if on_h.exists() and off_h.exists():
            zo, zf = np.load(on_h), np.load(off_h)
            hk = sorted(set(zo.keys()) & set(zf.keys()))
            res["h_sel_questions"] = len(hk)
            res["h_sel_bit_equal"] = int(sum(np.array_equal(zo[k], zf[k]) for k in hk))
            res["h_sel_max_abs_diff"] = float(max(np.abs(zo[k].astype(np.float64) - zf[k]).max() for k in hk))
        doc["pairs"][P] = res
        print(P, json.dumps(res))
    out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
