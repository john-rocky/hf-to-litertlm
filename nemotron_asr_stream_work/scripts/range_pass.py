#!/usr/bin/env python3
"""r1 range pass for the fp16ws form (.venv-ref, Mac CPU torch fp32; no transformers): the streaming conformer of
scripts/graphs.py, chunk by chunk with its caches exactly as the exported graph runs (mel graphs on contract.pcm_chunks,
pre_first / pre, neg_mask(i)), over ls-S32 (82 utterances) and the --n-fl longest utterances of each FLEURS set.

Per LayerNorm site (24 layers x n_ff1 / n_att / n_conv / n_cm / n_ff2 / n_out): max |x - mean| and the max row
variance; k = the larger of ceil(log2(dev / 32)) (|x - mean| 2^-k <= 32, the squared difference <= 1,024) and
ceil(log2(1,024 x var / 16,384) / 2) (the row sum of the squared difference 4x under 65,504 in case a GPU reduction
holds it in fp16), at least 0 (the "rule" line of fix.json). Every other recorded point
(Conformer._st: residual stream, FF hidden, q / k / v, attention terms, GLU, depthwise out, head) with max|x| above
2^14 = 16,384 is listed under "other_above_2^14"; the attention logits are judged without the -30,000 key bias.
-> results/r1/fix.json (ln_k = dict "L<i>.<site>" -> k, read by export_graphs.py --only conformer_fp32s) and the
per-point table results/r1/range.json.
usage: range_pass.py [--n-fl 4] [--threads 8]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_dwconv  # noqa: E402
import graphs as G  # noqa: E402

fast_dwconv.install()   # the depthwise conv1d as K shifted multiply-adds (PyTorch's Mac CPU kernel: 76 ms per call)

TARGET = 32.0
ROWSUM_LIMIT = 16384.0
OTHER_LIMIT = 2.0 ** 14
PROMPTS = {"en-US": 0, "de-DE": 9, "fr-FR": 8, "ru-RU": 11, "ja-JP": 10}


def decide(probe):
    rows, ln_k = [], {}
    for key in sorted(probe, key=lambda s: (int(s.split(".")[0][1:]), G.LN_SITES.index(s.split(".")[1]))):
        dev, var = probe[key]
        k_x2 = max(0, math.ceil(math.log2(dev / TARGET))) if dev > TARGET else 0
        k_sum = max(0, math.ceil(0.5 * math.log2(1024.0 * var / ROWSUM_LIMIT))) if 1024.0 * var > ROWSUM_LIMIT else 0
        k = max(k_x2, k_sum)
        ln_k[key] = k
        rows.append({"site": key, "max_abs_dev": round(dev, 4), "max_row_var": round(var, 4), "k_x2": k_x2,
                     "k_rowsum": k_sum, "k": k})
    return ln_k, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fl", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    mf = np.load(G.MELFB)
    sub, conf, _, _ = G.load_all()
    mel_first, mel_next = G.MelFirst(mf).eval(), G.MelNext(mf).eval()
    pre_first, pre = G.PreFirst(sub).eval(), G.Pre(sub).eval()
    items = [("ls-S32", "en-US", json.loads(x)) for x in open("data/sets/ls-S32/manifest.jsonl")]
    for name, prompt in (("fleurs-de", "de-DE"), ("fleurs-fr", "fr-FR"), ("fleurs-ru", "ru-RU"),
                         ("fleurs-ja", "ja-JP")):
        rows = [json.loads(x) for x in open(f"data/sets/{name}/manifest.jsonl")]
        rows.sort(key=lambda r: (-r["audio_length_s"], r["id"]))
        items += [(name, prompt, r) for r in rows[:a.n_fl]]
    conf.ln_probe, conf.stats = {}, {}
    t0, chunks, used = time.time(), 0, []
    with torch.no_grad():
        for name, prompt, r in items:
            audio, sr = sf.read(r["path"], dtype="float32", always_2d=True)
            assert sr == 16000
            pcm = G.pcm_chunks(audio[:, 0])
            oh = torch.from_numpy(G.one_hot(PROMPTS[prompt]))
            kc = torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM)
            vc = kc.clone()
            cc = torch.zeros(G.NLAYERS, G.HID, G.CONV_K - 1)
            c0 = c1 = c2 = None
            for i, p in enumerate(pcm):
                m = (mel_first if i == 0 else mel_next)(torch.from_numpy(p)[None])
                if i == 0:
                    e, c0, c1, c2 = pre_first(m)
                else:
                    e, c0, c1, c2 = pre(m, c0, c1, c2)
                conf.stats["pre.embeds"] = max(conf.stats.get("pre.embeds", 0.0), float(e.abs().max()))
                conf.stats["mel"] = max(conf.stats.get("mel", 0.0), float(m.abs().max()))
                _, kc, vc, cc = conf(e, torch.from_numpy(G.neg_mask(i)), kc, vc, cc, oh)
                chunks += 1
            used.append(f"{name}/{r['id']}")
            print(f"{time.strftime('%H:%M:%S')} {name} {r['id']} chunks {len(pcm)}", flush=True)
    ln_k, sites = decide(conf.ln_probe)
    other = []
    for k, v in conf.stats.items():
        if k.endswith(":unmasked"):
            continue
        val = conf.stats.get(k + ":unmasked", v) if k.endswith("att_logits") else v
        if val > OTHER_LIMIT:
            other.append({"point": k, "max_abs": round(val, 3)})
    fix = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": "per LayerNorm site, k = max(0, ceil(log2(max|x - mean| / target)), ceil(log2(1024 * max_row_var / rowsum_limit) / 2))",
           "target": TARGET, "rowsum_limit": ROWSUM_LIMIT, "other_limit": OTHER_LIMIT, "utterances": len(used),
           "chunks": chunks, "seconds": round(time.time() - t0, 1), "n_sites": len(ln_k),
           "n_sites_rescaled": sum(1 for v in ln_k.values() if v > 0), "k_max": max(ln_k.values()),
           "ln_k": ln_k, "sites": sites, "other_above_2^14": other, "items": used}
    os.makedirs("results/r1", exist_ok=True)
    json.dump(fix, open("results/r1/fix.json", "w"), indent=1)
    json.dump({"stats": {k: round(v, 4) for k, v in sorted(conf.stats.items())}}, open("results/r1/range.json", "w"),
              indent=1)
    print("RANGE DONE", json.dumps({k: fix[k] for k in ("utterances", "chunks", "n_sites_rescaled", "k_max",
                                                         "other_above_2^14")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
