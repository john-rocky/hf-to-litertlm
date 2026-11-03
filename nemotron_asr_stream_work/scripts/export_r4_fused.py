#!/usr/bin/env python3
"""r4: export the fused pre-encode + conformer graph (graphs_r4.ConformerPre, one signature `conformer_pre`) with
litert-torch (.venv-lt), CONTRACT v1 weights (graphs.load_all):
  out/r4/graphs/conformer_pre_fp32_raw.tflite    LayerNorms as is (the P0 form after the lift)
  out/r4/graphs/conformer_pre_fp32s_raw.tflite   results/r1/fix.json's LayerNorm 2^-k (half.py's source of
                                                 conformer_pre_fp16ws, as conformer_fp32s is conformer_fp16ws's)
The caller lifts each raw file (scripts/r2_fb.py lift, the 48 runtime rank-3 BATCH_MATMULs -> rank 4, the same step as for the r1 graphs)
and removes the raw file.
Torch check before converting (exit 3 = stop): the fused module against the split path (PreFirst / Pre + the same
Conformer) on the first --check-chunks chunks of the first --check-utts ls-S32 utterances, from the P0 mel dumps
(out/p0/fp32_cpu_chunks.npz), each path with its own caches: max|d| of enc_proj and of the seven state outputs; stops
when enc_proj's max|d| relative to max|ref| exceeds 1e-6.
Each file's bytes / sha256 / convert seconds / peak RSS -> results/r4/export.jsonl.
usage: export_r4_fused.py [--only conformer_pre_fp32,conformer_pre_fp32s] [--check-chunks 6] [--check-utts 3]
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs as G  # noqa: E402
import graphs_r4 as R4  # noqa: E402

OUT = "out/r4/graphs"
REC = "results/r4/export.jsonl"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fused_kwargs():
    m0, m1 = R4.masks(1)
    return {"mel": torch.zeros(1, G.T_NEXT, G.MEL),
            "cache0": torch.zeros(1, 1, 1, 131), "cache1": torch.zeros(1, G.SUB_CH, 1, 68),
            "cache2": torch.zeros(1, G.SUB_CH, 1, 36),
            "pre_m0": torch.from_numpy(m0), "pre_m1": torch.from_numpy(m1),
            "neg_mask": torch.from_numpy(G.neg_mask(0)),
            "k_cache": torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
            "v_cache": torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
            "conv_cache": torch.zeros(G.NLAYERS, G.HID, G.CONV_K - 1),
            "one_hot": torch.from_numpy(G.one_hot(0))}


def check(sub, conf, fused, n_chunks, n_utts, dump="out/p0/fp32_cpu_chunks.npz"):
    """the split path (PreFirst / Pre -> conf) against the fused module, each with its own caches."""
    D = np.load(dump)
    ids = [json.loads(x)["id"] for x in open("data/sets/ls-S32/manifest.jsonl")][:n_utts]
    pf, pr = G.PreFirst(sub).eval(), G.Pre(sub).eval()
    oh = torch.from_numpy(G.one_hot(0))
    worst = {k: 0.0 for k in ("enc_proj", "k_out", "v_out", "conv_out", "cache0_out", "cache1_out", "cache2_out")}
    ref_max, n = 0.0, 0
    with torch.no_grad():
        for u in ids:
            mels = [np.asarray(D[f"ls-S32/{u}/mel0"], np.float32)[None]] + \
                   [np.asarray(x, np.float32)[None] for x in D[f"ls-S32/{u}/mel"]]
            z = lambda *s: torch.zeros(*s)  # noqa: E731
            ra = [z(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM), z(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
                  z(G.NLAYERS, G.HID, G.CONV_K - 1)]
            fa = [x.clone() for x in ra] + [z(1, 1, 1, 131), z(1, G.SUB_CH, 1, 68), z(1, G.SUB_CH, 1, 36)]
            pc = None
            for c, m in enumerate(mels[:n_chunks]):
                mt = torch.from_numpy(m)
                if c == 0:
                    e, c0, c1, c2 = pf(mt)
                    fm = torch.from_numpy(R4.first_mel(m))
                else:
                    e, c0, c1, c2 = pr(mt, *pc)
                    fm = mt
                pc = (c0, c1, c2)
                nm = torch.from_numpy(G.neg_mask(c))
                enc, ko, vo, co = conf(e, nm, ra[0], ra[1], ra[2], oh)
                ra = [ko, vo, co]
                m0, m1 = (torch.from_numpy(x) for x in R4.masks(c))
                out = fused(fm, fa[3], fa[4], fa[5], m0, m1, nm, fa[0], fa[1], fa[2], oh)
                fa = [out[1], out[2], out[3], out[4], out[5], out[6]]
                ref_max = max(ref_max, float(enc.abs().max()))
                for k, x, y in zip(worst, (enc, ko, vo, co, c0, c1, c2), out):
                    worst[k] = max(worst[k], float((x - y).abs().max()))
                n += 1
    return {"utts": ids, "chunks": n, "maxabs": worst, "enc_max_ref": ref_max,
            "enc_rel": worst["enc_proj"] / max(ref_max, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="conformer_pre_fp32,conformer_pre_fp32s")
    ap.add_argument("--check-chunks", type=int, default=6)
    ap.add_argument("--check-utts", type=int, default=3)
    a = ap.parse_args()
    import litert_torch
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(REC), exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    fix = json.load(open("results/r1/fix.json"))
    for name in [x for x in a.only.split(",") if x]:
        sub, conf, _, _ = G.load_all()
        if name == "conformer_pre_fp32s":
            conf.apply_ln_k(fix["ln_k"])
        elif name != "conformer_pre_fp32":
            raise SystemExit(f"unknown form {name}")
        fused = R4.ConformerPre(sub, conf).eval()
        chk = check(sub, conf, fused, a.check_chunks, a.check_utts)
        print("FUSED_CHECK", name, json.dumps(chk), flush=True)
        if chk["enc_rel"] > 1e-6:
            raise SystemExit(3)
        path = f"{OUT}/{name}_raw.tflite"
        t0 = time.time()
        em = litert_torch.signature("conformer_pre", R4.ExConformerPre(fused).eval(),
                                    sample_kwargs=fused_kwargs()).convert(enable_x64=False)
        t1 = time.time()
        em.export(path)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": name, "path": path, "check": chk,
               "litert_torch": litert_torch.__version__, "torch": torch.__version__,
               "convert_s": round(t1 - t0, 1), "write_s": round(time.time() - t1, 1),
               "bytes": os.path.getsize(path), "sha256": sha256(path),
               "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)}
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("EXPORTED", json.dumps(rec), flush=True)
        del em, fused, conf, sub
    print("EXPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
