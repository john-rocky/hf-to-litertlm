#!/usr/bin/env python3
"""Export the r1 fp32 LiteRT graphs with litert-torch (.venv-lt). Run from the work dir after check_torch.py (it writes
out/graphs/melfb.npy). One file per stage; files with two signatures share their weights:
  out/graphs/mel.tflite              mel_first (pcm [1,4040]) / mel (pcm [1,5520])
  out/graphs/pre.tflite              pre_first (mel [1,25,128]) / pre (mel [1,32,128] + cache0/1/2)
  out/graphs/conformer_fp32.tflite   conformer (x, neg_mask, k_cache, v_cache, conv_cache, one_hot)
  out/graphs/dec_fp32.tflite         predict (token int32, h, c) / joint (dec_out, enc_frame)
  out/graphs/conformer_fp32s.tflite  the same conformer with the LayerNorm 2^-k of results/r1/fix.json (--fixed)
Each file's bytes / sha256 / convert seconds / peak RSS go to results/r1/export.jsonl (one line per file).
usage: export_graphs.py [--only mel,pre,conformer_fp32,dec_fp32,conformer_fp32s] [--fixed]
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
import litert_torch  # noqa: E402

OUT = "out/graphs"
REC = "results/r1/export.jsonl"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def conformer_kwargs():
    return {"x": torch.zeros(1, G.Q, G.HID), "neg_mask": torch.from_numpy(G.neg_mask(0)),
            "k_cache": torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
            "v_cache": torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
            "conv_cache": torch.zeros(G.NLAYERS, G.HID, G.CONV_K - 1),
            "one_hot": torch.from_numpy(G.one_hot(0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mel,pre,conformer_fp32,dec_fp32")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(8)
    mf = np.load(G.MELFB)
    sub, conf, pred, joint = G.load_all()
    only = [x for x in a.only.split(",") if x]
    plan = {
        "mel": lambda: litert_torch.signature(
            "mel_first", G.ExMelFirst(G.MelFirst(mf)).eval(), sample_kwargs={"pcm": torch.zeros(1, G.N_FIRST)}).signature(
            "mel", G.ExMel(G.MelNext(mf)).eval(), sample_kwargs={"pcm": torch.zeros(1, G.N_NEXT)}).convert(enable_x64=False),
        "pre": lambda: litert_torch.signature(
            "pre_first", G.ExPreFirst(G.PreFirst(sub)).eval(), sample_kwargs={"mel": torch.zeros(1, G.T_FIRST, G.MEL)}).signature(
            "pre", G.ExPre(G.Pre(sub)).eval(), sample_kwargs={"mel": torch.zeros(1, G.T_NEXT, G.MEL),
                                                     "cache0": torch.zeros(1, 1, 1, 131),
                                                     "cache1": torch.zeros(1, G.SUB_CH, 1, 68),
                                                     "cache2": torch.zeros(1, G.SUB_CH, 1, 36)}).convert(
            enable_x64=False),
        "conformer_fp32": lambda: litert_torch.signature(
            "conformer", G.ExConformer(conf).eval(), sample_kwargs=conformer_kwargs()).convert(enable_x64=False),
        "dec_fp32": lambda: litert_torch.signature(
            "predict", G.ExPredict(pred).eval(), sample_kwargs={"token": torch.tensor([[G.BLANK]], dtype=torch.int32),
                                                   "h": torch.zeros(G.DEC_NL, 1, G.DEC_HID),
                                                   "c": torch.zeros(G.DEC_NL, 1, G.DEC_HID)}).signature(
            "joint", G.ExJoint(joint).eval(), sample_kwargs={"dec_out": torch.zeros(1, G.DEC_HID),
                                                  "enc_frame": torch.zeros(1, G.DEC_HID)}).convert(enable_x64=False),
    }
    if "conformer_fp32s" in only:
        fix = json.load(open("results/r1/fix.json"))
        conf_s = G.load_all()[1]
        conf_s.apply_ln_k(fix["ln_k"])
        plan["conformer_fp32s"] = lambda: litert_torch.signature(
            "conformer", G.ExConformer(conf_s).eval(), sample_kwargs=conformer_kwargs()).convert(enable_x64=False)
    for name in only:
        path = f"{OUT}/{name}.tflite"
        t0 = time.time()
        em = plan[name]()
        t1 = time.time()
        em.export(path)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": name, "path": path,
               "litert_torch": litert_torch.__version__, "torch": torch.__version__,
               "convert_s": round(t1 - t0, 1), "write_s": round(time.time() - t1, 1),
               "bytes": os.path.getsize(path), "sha256": sha256(path),
               "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)}
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("EXPORTED", json.dumps(rec), flush=True)
        del em
    print("EXPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
