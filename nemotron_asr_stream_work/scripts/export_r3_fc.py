#!/usr/bin/env python3
"""The CPU-only conformer with the relative-position term written as one FULLY_CONNECTED per layer (XNNPACK v2.3.0's
FORCE_FP16 mode refuses the 24 BATCH_MATMULs whose second input is a constant, so the original layout cannot run in
float16 arithmetic on the CPU). litert-torch (.venv-lt), CONTRACT v1 weights (graphs.load_all). The GPU and NPU files
keep the original layout.
  out/r3/graphs/conformer_fp32_fc.tflite    graphs.Conformer, the rel-pos term rewritten (the P0 check's form)
  out/r3/graphs/conformer_fp32s_fc.tflite   the same with results/r1/fix.json's LayerNorm 2^-k (half.py's source of
                                            conformer_fp16ws_fc, as conformer_fp32s is conformer_fp16ws's)
Rewrite (graphs.Conformer.forward otherwise copied as is): per layer
  graphs.py   q = q_proj(hn) -> [1, H, Q, D]; bd = (q + bias_v) @ relative_k_proj(pos_emb).view(1, P, H, D).permute(0, 2, 3, 1)
              -> after folding: BATCH_MATMUL [8,4,128] x constant [8,128,119]
  here        bd = F.linear(hn, W_bd @ W_q, W_bd @ bias_v.view(HID)).view(1, Q, H, P).transpose(1, 2)
              -> FULLY_CONNECTED [1,4,1024] x constant [952,1024] (+ bias [952]) + RESHAPE + TRANSPOSE, reading the
              attention LayerNorm's output like k_proj / v_proj (q_proj keeps its one consumer, the ac term)
with W_bd [H*P, H*D] block diagonal: W_bd[h*P + j, h*D + d] = relative_k_proj(pos_emb)[j, h*D + d], zero off the
blocks (P = 2KV - 1 = 119, H = 8, D = 128); W_q = q_proj's weight (no bias); the two products in float64, stored fp32.
Why one FC (the form with fewer ops that stays one XNNPACK partition under flags 7): 1 FC + 1 RESHAPE + 1 TRANSPOSE
per layer against 8 SLICE + 8 FC + a CONCAT tree of <= 5 inputs (the v2.3.0 delegate takes CONCATENATION with 2..5
inputs; 8 would fall out of XNNPACK and split the partition) + RESHAPEs. A block-diagonal FC on q was exported first
and the converter duplicated q_proj in every layer: F.linear(q_flat + bias_v, W_bd) -> the ADD fused into a second
q_proj with the same weight buffer; F.linear(q_flat, W_bd, W_bd @ bias_v) -> still a second q_proj, now with its own
weight copy (+100 MB); both 292 FCs instead of 268 = 24 more 1024 x 1024 GEMMs per chunk.
Folding W_q into the FC removes the q_proj from this path; the FC is dense, with the block-diagonal one's size: 952 x
1024 weights per layer instead of the table's 119 x 1024 (+852,992 per layer, +20.5 M over 24 layers = +3.5 % of the
conformer's 583.9 M FC weights; the MACs per chunk grow by the same count). Numerics: the same sum of products,
associated differently ((hn W_q^T) W_bd^T against hn (W_bd W_q)^T) = equal up to rounding.
Torch check before converting: the rewritten forward against graphs.Conformer.forward on the first --check-chunks
chunks of the first ls-S32 utterance (the CONTRACT v1 mel / pre path, each with its own caches): max|d| of enc and of
the three cache outputs, relative to max|ref|; stops (exit 3) when the relative max|d| of enc exceeds 1e-5.
Each file's bytes / sha256 / convert seconds / peak RSS -> results/r3/export_fc.jsonl.
usage: export_r3_fc.py [--only conformer_fp32_fc,conformer_fp32s_fc] [--check-chunks 8]
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs as G  # noqa: E402

OUT = "out/r3/graphs"
REC = "results/r3/export_fc.jsonl"
P = 2 * G.KV - 1


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class ConformerFC(G.Conformer):
    """graphs.Conformer with the relative-position term as one FC on hn (blk.rel_w / rel_b, built by add_rel_bd)."""

    def forward(self, x, neg_mask, k_cache, v_cache, conv_cache, one_hot):
        k_outs, v_outs, c_outs = [], [], []
        for i, blk in enumerate(self.layers):
            h = self._ln(i, "n_ff1", blk.n_ff1, x)
            x = x + 0.5 * blk.ff1_l2(F.silu(blk.ff1_l1(h)))
            hn = self._ln(i, "n_att", blk.n_att, x)
            q = blk.q_proj(hn).view(1, G.Q, G.HEADS, G.HDIM).transpose(1, 2)            # [1, H, Q, D]
            k_new = blk.k_proj(hn).view(1, G.Q, G.HEADS, G.HDIM).transpose(1, 2)
            v_new = blk.v_proj(hn).view(1, G.Q, G.HEADS, G.HDIM).transpose(1, 2)
            k = torch.cat([k_cache[i:i + 1], k_new], dim=2)                           # [1, H, KV, D]
            v = torch.cat([v_cache[i:i + 1], v_new], dim=2)
            k_outs.append(k[:, :, G.Q:, :])
            v_outs.append(v[:, :, G.Q:, :])
            q_u = q + blk.bias_u.view(1, G.HEADS, 1, G.HDIM)
            # the rewrite (graphs.py: q_v = q + bias_v; bd = q_v @ rel_k.permute(0, 2, 3, 1))
            bd = F.linear(hn, blk.rel_w, blk.rel_b).view(1, G.Q, G.HEADS, P).transpose(1, 2)  # [1, H, Q, P]
            bd = blk._rel_shift(bd)[..., :G.KV] * blk.scaling + neg_mask
            ac = q_u @ k.transpose(-2, -1) * blk.scaling
            w = torch.softmax(ac + bd, dim=-1)
            attn = (w @ v).transpose(1, 2).reshape(1, G.Q, G.HID)
            x = x + blk.o_proj(attn)
            hc = self._ln(i, "n_conv", blk.n_conv, x)
            g = blk.pw1(hc)
            g = g[..., :G.HID] * torch.sigmoid(g[..., G.HID:])
            gp = torch.cat([conv_cache[i:i + 1], g.transpose(1, 2)], dim=-1)
            c_outs.append(gp[:, :, G.Q:])
            y = blk.dw(gp).transpose(1, 2)
            y = F.silu(self._ln(i, "n_cm", blk.n_cm, y))
            x = x + blk.pw2(y)
            h = self._ln(i, "n_ff2", blk.n_ff2, x)
            x = x + 0.5 * blk.ff2_l2(F.silu(blk.ff2_l1(h)))
            x = self._ln(i, "n_out", blk.n_out, x)
        w1 = self.prompt_linear_1.weight
        p = F.linear(one_hot, w1[:, G.HID:], self.prompt_linear_1.bias)
        hx = F.linear(x, w1[:, :G.HID]) + p[:, None, :]
        enc = self.projector(self.prompt_linear_2(torch.relu(hx)))
        return enc, torch.cat(k_outs, 0), torch.cat(v_outs, 0), torch.cat(c_outs, 0)


def add_rel_bd(conf):
    """Give every block its rel_w = W_bd @ W_q and rel_b = W_bd @ bias_v (float64 products, fp32 buffers) from its
    relative_k_proj, q_proj, bias_v and the conformer's pos_emb; -> the new class."""
    with torch.no_grad():
        for blk in conf.layers:
            rel = blk.relative_k_proj(conf.pos_emb).view(P, G.HEADS, G.HDIM).double()   # [P, H, D]
            wbd = torch.zeros(G.HEADS * P, G.HEADS * G.HDIM, dtype=torch.float64)
            for h in range(G.HEADS):
                wbd[h * P:(h + 1) * P, h * G.HDIM:(h + 1) * G.HDIM] = rel[:, h, :]
            blk.register_buffer("rel_w", (wbd @ blk.q_proj.weight.double()).float().contiguous(), persistent=False)
            blk.register_buffer("rel_b", (wbd @ blk.bias_v.reshape(G.HID).double()).float(), persistent=False)
    conf.__class__ = ConformerFC
    return conf


def check(conf_ref, conf_fc, n_chunks):
    """Both conformers on the first n chunks of the first ls-S32 utterance, each with its own caches."""
    man = [json.loads(x) for x in open("data/sets/ls-S32/manifest.jsonl")]
    r = man[0]
    audio, _ = sf.read(r["path"], dtype="float32", always_2d=True)
    mf = np.load(G.MELFB)
    sub = G.load_all()[0]
    mel_first, mel_next = G.MelFirst(mf).eval(), G.MelNext(mf).eval()
    pre_first, pre = G.PreFirst(sub).eval(), G.Pre(sub).eval()
    oh = torch.from_numpy(G.one_hot(0))
    caches = []
    for _ in range(2):
        caches.append([torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM), torch.zeros(G.NLAYERS, G.HEADS, G.CACHE, G.HDIM),
                       torch.zeros(G.NLAYERS, G.HID, G.CONV_K - 1)])
    worst = {"enc": 0.0, "k": 0.0, "v": 0.0, "conv": 0.0}
    ref_max = 0.0
    c0 = None
    with torch.no_grad():
        for c, pcm in enumerate(G.pcm_chunks(audio[:, 0])):
            if c >= n_chunks:
                break
            m = (mel_first if c == 0 else mel_next)(torch.from_numpy(pcm)[None])
            if c == 0:
                e, c0, c1, c2 = pre_first(m)
            else:
                e, c0, c1, c2 = pre(m, c0, c1, c2)
            nm = torch.from_numpy(G.neg_mask(c))
            outs = []
            for conf, cc in ((conf_ref, caches[0]), (conf_fc, caches[1])):
                enc, ko, vo, co = conf(e, nm, cc[0], cc[1], cc[2], oh)
                cc[0], cc[1], cc[2] = ko, vo, co
                outs.append((enc, ko, vo, co))
            (a0, a1, a2, a3), (b0, b1, b2, b3) = outs
            ref_max = max(ref_max, float(a0.abs().max()))
            for key, x, y in (("enc", a0, b0), ("k", a1, b1), ("v", a2, b2), ("conv", a3, b3)):
                worst[key] = max(worst[key], float((x - y).abs().max()))
    return {"utt": r["id"], "chunks": min(n_chunks, c + 1), "maxabs": worst, "enc_max_ref": ref_max,
            "enc_rel": worst["enc"] / max(ref_max, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="conformer_fp32_fc,conformer_fp32s_fc")
    ap.add_argument("--check-chunks", type=int, default=8)
    a = ap.parse_args()
    import litert_torch
    from export_graphs import conformer_kwargs
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(REC), exist_ok=True)
    torch.set_num_threads(8)
    fix = json.load(open("results/r1/fix.json"))
    for name in [x for x in a.only.split(",") if x]:
        conf_ref = G.load_all()[1].eval()
        conf_fc = G.load_all()[1].eval()
        if name == "conformer_fp32s_fc":
            conf_ref.apply_ln_k(fix["ln_k"])
            conf_fc.apply_ln_k(fix["ln_k"])
        elif name != "conformer_fp32_fc":
            raise SystemExit(f"unknown form {name}")
        add_rel_bd(conf_fc)
        chk = check(conf_ref, conf_fc, a.check_chunks)
        print("FC_CHECK", name, json.dumps(chk), flush=True)
        if chk["enc_rel"] > 1e-5:
            raise SystemExit(3)
        del conf_ref
        path = f"{OUT}/{name}.tflite"
        t0 = time.time()
        em = litert_torch.signature("conformer", G.ExConformer(conf_fc).eval(),
                                    sample_kwargs=conformer_kwargs()).convert(enable_x64=False)
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
        del em, conf_fc
    print("EXPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
