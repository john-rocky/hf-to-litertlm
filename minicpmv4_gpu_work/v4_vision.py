#!/usr/bin/env python3
"""v4_vision.py selftest|export|tflcheck ...  — MiniCPM-V-4's vision tower (navit SigLIP 27 layers + Resampler) re-authored
in plain torch for LiteRT-LM's `minicpmv4` contract (runtime v0.19.0, minicpmv_data_processor.cc), one slice per call:

  inputs   images        [1,1088,588] f32  patch-flattened slice, (ch*14+py)*14+px; zero rows past the slice
           positions_xy  [1,1088,2]   i32  (x, y) on the slice grid (x = i % tgt_w, y = i / tgt_w); -1 on pad rows
           vit_positions [1,1088,2]   i32  bucketized (w, h) cell of the 70x70 ViT position table; -1 on pad rows
  output   features      [1,64,2560]  f32  the Resampler's 64 tokens (LLM width)

GPU-minded rewrite (same arithmetic as the vendor code, checked by `selftest` against the vendor modules):
  - every lookup is a one-hot product, no GATHER: onehot(v) = relu(1 - |v - iota70|) (exact for the integer
    coordinates, also in fp16; -1 gives an all-zero row); the ViT table row y*70+x = onehot(y) (x) onehot(x) flattened
    to 4900 and multiplied by the table; the Resampler's 2D sin-cos key embedding is separable, so it is
    concat(onehot(x) @ T, onehot(y) @ T) with T[p] = the vendor's 1-D sin-cos row of p (1280 wide)
  - pad rows: an additive key mask of -1e4 (valid = relu(v+1) - relu(v)), in the ViT and in the Resampler
  - the Resampler's query side (ln_q(query), in-projection, 1/sqrt(d)) is input-independent: folded into one fully
    connected layer on the keys with a block-diagonal weight (no BATCH_MATMUL with a constant operand)
  - Conv2d patch embedding = a Linear on the flattened patch (weight [1152, 3*14*14])

  selftest --src SRC_DIR --ref REF_VISION_DIR [--out JSON]
  export   --src SRC_DIR --out-dir DIR [--quant none,fp16,...]
  tflcheck --model FILE.tflite --src SRC_DIR --ref REF_VISION_DIR [--out JSON]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

L_PAD = 1088
SIDE = 70
PATCH = 14
VIT_D, VIT_H, VIT_LAYERS, VIT_MLP = 1152, 16, 27, 4304
RD, RH, NQ = 2560, 20, 64
MASK_NEG = -1.0e4


def load_state(src):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for name, shard in idx.items():
        if name.startswith(("vpm.", "resampler.")):
            sd[name] = None
    for shard in sorted(set(idx.values())):
        with safe_open(os.path.join(src, shard), "pt") as f:
            for name in f.keys():
                if name in sd:
                    sd[name] = f.get_tensor(name).to(torch.float32)
    assert all(v is not None for v in sd.values())
    return sd


def sincos_1d(dim, n):
    """The vendor's get_1d_sincos_pos_embed_from_grid_new for positions 0..n-1 (numpy float32, as in resampler.py)."""
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(n, dtype=np.float32)
    out = np.einsum("p,d->pd", pos, omega)
    return torch.from_numpy(np.concatenate([np.sin(out), np.cos(out)], axis=-1)).float()


class Layer(nn.Module):
    def __init__(self, sd, i):
        super().__init__()
        p = f"vpm.encoder.layers.{i}."
        self.ln1 = nn.LayerNorm(VIT_D, eps=1e-6)
        self.ln2 = nn.LayerNorm(VIT_D, eps=1e-6)
        self.q = nn.Linear(VIT_D, VIT_D)
        self.k = nn.Linear(VIT_D, VIT_D)
        self.v = nn.Linear(VIT_D, VIT_D)
        self.o = nn.Linear(VIT_D, VIT_D)
        self.fc1 = nn.Linear(VIT_D, VIT_MLP)
        self.fc2 = nn.Linear(VIT_MLP, VIT_D)
        for mod, name in ((self.ln1, "layer_norm1"), (self.ln2, "layer_norm2"), (self.q, "self_attn.q_proj"),
                          (self.k, "self_attn.k_proj"), (self.v, "self_attn.v_proj"), (self.o, "self_attn.out_proj"),
                          (self.fc1, "mlp.fc1"), (self.fc2, "mlp.fc2")):
            mod.weight.data.copy_(sd[p + name + ".weight"])
            mod.bias.data.copy_(sd[p + name + ".bias"])

    def forward(self, x, mask):
        n = x.shape[1]
        hd = VIT_D // VIT_H
        h = self.ln1(x)
        q = self.q(h).reshape(1, n, VIT_H, hd).permute(0, 2, 1, 3)
        k = self.k(h).reshape(1, n, VIT_H, hd).permute(0, 2, 1, 3)
        v = self.v(h).reshape(1, n, VIT_H, hd).permute(0, 2, 1, 3)
        s = torch.matmul(q, k.transpose(2, 3)) * (hd ** -0.5) + mask
        a = torch.matmul(torch.softmax(s, dim=-1), v)
        x = x + self.o(a.permute(0, 2, 1, 3).reshape(1, n, VIT_D))
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


class ScaledLN(nn.Module):
    """LayerNorm(x) computed as LayerNorm(x * s) with eps * s^2: the same function in exact arithmetic (LayerNorm is
    scale-invariant up to eps), but the squared deviations the graph keeps (MEAN / SQUARED_DIFFERENCE) shrink by s^2,
    so they stay under fp16's 65504 (an optional fp16 form; the shipped files use the plain export)."""

    def __init__(self, ln, s):
        super().__init__()
        self.ln, self.s = ln, float(s)

    def forward(self, x):
        return F.layer_norm(x * self.s, self.ln.normalized_shape, self.ln.weight, self.ln.bias,
                            self.ln.eps * self.s * self.s)


class V4Vision(nn.Module):
    def __init__(self, sd, ln_k=None, score_div=1.0, head_div=None):
        """ln_k: {module name: k} = a pre-scale 2^-k on those LayerNorms (ScaledLN); score_div: the Resampler's folded
        score weight divided by this (its logits / score_div = a softmax temperature, NOT exact; measured per slice);
        head_div: {head: c} = the same division for the score rows of those Resampler heads only."""
        super().__init__()
        self.patch = nn.Linear(3 * PATCH * PATCH, VIT_D)
        self.patch.weight.data.copy_(sd["vpm.embeddings.patch_embedding.weight"].reshape(VIT_D, -1))
        self.patch.bias.data.copy_(sd["vpm.embeddings.patch_embedding.bias"])
        self.pos = nn.Linear(SIDE * SIDE, VIT_D, bias=False)
        self.pos.weight.data.copy_(sd["vpm.embeddings.position_embedding.weight"].t())
        self.layers = nn.ModuleList([Layer(sd, i) for i in range(VIT_LAYERS)])
        self.post_ln = nn.LayerNorm(VIT_D, eps=1e-6)
        self.post_ln.weight.data.copy_(sd["vpm.post_layernorm.weight"])
        self.post_ln.bias.data.copy_(sd["vpm.post_layernorm.bias"])
        # Resampler
        self.kv_proj = nn.Linear(VIT_D, RD, bias=False)
        self.kv_proj.weight.data.copy_(sd["resampler.kv_proj.weight"])
        self.ln_kv = nn.LayerNorm(RD, eps=1e-6)
        self.ln_kv.weight.data.copy_(sd["resampler.ln_kv.weight"])
        self.ln_kv.bias.data.copy_(sd["resampler.ln_kv.bias"])
        self.ln_post = nn.LayerNorm(RD, eps=1e-6)
        self.ln_post.weight.data.copy_(sd["resampler.ln_post.weight"])
        self.ln_post.bias.data.copy_(sd["resampler.ln_post.bias"])
        w_in, b_in = sd["resampler.attn.in_proj_weight"], sd["resampler.attn.in_proj_bias"]
        wq, wk, wv = w_in.chunk(3, 0)
        bq, bk, bv = b_in.chunk(3, 0)
        qn = F.layer_norm(sd["resampler.query"], (RD,), sd["resampler.ln_q.weight"], sd["resampler.ln_q.bias"], 1e-6)
        # scores of all 20 heads x 64 queries as ONE fully connected layer on the keys: a block-diagonal constant
        # weight [RH*NQ, RD] (head h's 64 query rows, pre-scaled by 1/sqrt(128), over its 128 key channels), so no
        # BATCH_MATMUL takes a constant operand on the GPU
        hd = RD // RH
        qh = (qn @ wq.t() + bq).reshape(NQ, RH, hd).permute(1, 0, 2) * (hd ** -0.5)
        w_s = torch.zeros(RH * NQ, RD)
        for h in range(RH):
            w_s[h * NQ:(h + 1) * NQ, h * hd:(h + 1) * hd] = qh[h]
        self.score = nn.Linear(RD, RH * NQ, bias=False)
        self.score.weight.data.copy_(w_s)
        self.k_proj = nn.Linear(RD, RD)
        self.k_proj.weight.data.copy_(wk)
        self.k_proj.bias.data.copy_(bk)
        self.v_proj = nn.Linear(RD, RD)
        self.v_proj.weight.data.copy_(wv)
        self.v_proj.bias.data.copy_(bv)
        self.out_proj = nn.Linear(RD, RD)
        self.out_proj.weight.data.copy_(sd["resampler.attn.out_proj.weight"])
        self.out_proj.bias.data.copy_(sd["resampler.attn.out_proj.bias"])
        self.proj = nn.Linear(RD, RD, bias=False)
        self.proj.weight.data.copy_(sd["resampler.proj"].t())
        self.sincos = nn.Linear(SIDE, RD // 2, bias=False)
        self.sincos.weight.data.copy_(sincos_1d(RD // 2, SIDE).t())
        self.register_buffer("iota", torch.arange(SIDE, dtype=torch.float32))
        if score_div != 1.0:
            self.score.weight.data.div_(score_div)
        for h, c in (head_div or {}).items():
            self.score.weight.data[int(h) * NQ:(int(h) + 1) * NQ].div_(float(c))
        for name, k in (ln_k or {}).items():
            parent, _, attr = name.rpartition(".")
            owner = self.get_submodule(parent) if parent else self
            setattr(owner, attr, ScaledLN(getattr(owner, attr), 2.0 ** -k))

    def onehot(self, v):
        return torch.relu(1.0 - torch.abs(v - self.iota))

    def forward(self, images, positions_xy, vit_positions):
        n = images.shape[1]
        vp = vit_positions.to(torch.float32)
        vx, vy = vp[..., 0:1], vp[..., 1:2]
        valid = torch.relu(vx + 1.0) - torch.relu(vx)
        oh2 = (self.onehot(vy).unsqueeze(-1) * self.onehot(vx).unsqueeze(-2)).reshape(1, n, SIDE * SIDE)
        x = self.patch(images) + self.pos(oh2)
        mask = ((valid - 1.0) * -MASK_NEG).reshape(1, 1, 1, n)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.post_ln(x)
        # Resampler (one slice): queries are constants; keys carry the 2D sin-cos embedding of (x, y)
        kv = self.ln_kv(self.kv_proj(x))
        pxy = positions_xy.to(torch.float32)
        rx, ry = pxy[..., 0:1], pxy[..., 1:2]
        rvalid = torch.relu(rx + 1.0) - torch.relu(rx)
        pos = torch.cat([self.sincos(self.onehot(rx)), self.sincos(self.onehot(ry))], dim=-1)
        hd = RD // RH
        s = self.score(self.k_proj(kv + pos)).reshape(1, n, RH, NQ).permute(0, 2, 3, 1)
        v = self.v_proj(kv).reshape(1, n, RH, hd).permute(0, 2, 1, 3)
        rmask = ((rvalid - 1.0) * -MASK_NEG).reshape(1, 1, 1, n)
        s = s + rmask
        a = torch.matmul(torch.softmax(s, dim=-1), v).permute(0, 2, 1, 3).reshape(1, NQ, RD)
        out = self.proj(self.ln_post(self.out_proj(a)))
        return {"features": out}


def slice_inputs(pixel_values, tgt_size):
    """One vendor slice (pixel strip [3,14,L*14], tgt (h,w)) -> the runtime's three inputs (BuildSliceImage +
    PadSliceToSignatureLength in minicpmv_data_processor.cc; vit positions = the vendor's bucketized ids)."""
    h, w = int(tgt_size[0]), int(tgt_size[1])
    n = h * w
    strip = pixel_values.to(torch.float32)
    assert strip.shape == (3, PATCH, n * PATCH), strip.shape
    patches = strip.reshape(3, PATCH, n, PATCH).permute(2, 0, 1, 3).reshape(n, 3 * PATCH * PATCH)
    images = torch.zeros(1, L_PAD, 3 * PATCH * PATCH)
    images[0, :n] = patches
    boundaries = torch.arange(1 / SIDE, 1.0, 1 / SIDE)
    bh = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / h), boundaries, right=True)
    bw = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / w), boundaries, right=True)
    ids = (bh[:, None] * SIDE + bw).flatten()
    vit = torch.full((1, L_PAD, 2), -1, dtype=torch.int32)
    vit[0, :n, 0] = (ids % SIDE).to(torch.int32)
    vit[0, :n, 1] = (ids // SIDE).to(torch.int32)
    pxy = torch.full((1, L_PAD, 2), -1, dtype=torch.int32)
    i = torch.arange(n)
    pxy[0, :n, 0] = (i % w).to(torch.int32)
    pxy[0, :n, 1] = (i // w).to(torch.int32)
    return images, pxy, vit


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def ref_slices(ref_dir):
    for name in sorted(os.listdir(ref_dir)):
        if name.endswith(".pt"):
            r = torch.load(os.path.join(ref_dir, name))
            for j, (pv, ts) in enumerate(zip(r["pixel_values"], r["tgt_sizes"])):
                yield f"{name[:-3]}.{j}", pv, ts, r["resampler"][j]


def cmd_selftest(a):
    torch.set_grad_enabled(False)
    m, ln_k, head_div = model_from_args(a)
    res = {}
    for key, pv, ts, want in ref_slices(a.ref):
        got = m(*slice_inputs(pv, ts))["features"][0]
        res[key] = {"tgt": [int(ts[0]), int(ts[1])], "rel_l2": rel_l2(got, want),
                    "max_abs": float((got - want).abs().max())}
        print(key, res[key], flush=True)
    worst = max(v["rel_l2"] for v in res.values())
    print(f"SELFTEST worst rel L2 {worst:.3e} over {len(res)} slices")
    if a.out:
        json.dump({"worst_rel_l2": worst, "slices": res}, open(a.out, "w"), indent=1)
    return 0


def fp16safe_config(range_json, headroom=4.0):
    """The LayerNorm pre-scales of the fp16-safe form (optional, not in the shipped files): k per LayerNorm whose largest row sum of squared
    deviations on the reference slices (a range scan, not in this directory) times `headroom` passes 65504."""
    st = json.load(open(range_json))["stats"]
    ln_k = {}
    for name, v in st.items():
        rs = v.get("max_row_sum_sq_dev")
        if rs is not None and rs * headroom > 65504.0:
            ln_k[name] = max(1, math.ceil(math.log2(math.sqrt(rs * headroom / 65504.0))))
    return ln_k


def model_from_args(a):
    ln_k, head_div = {}, {}
    if getattr(a, "fp16safe", None):
        ln_k = fp16safe_config(a.fp16safe)
    if getattr(a, "head_div", None):
        head_div = {int(h): float(c) for h, c in (x.split(":") for x in a.head_div.split(","))}
    return V4Vision(load_state(a.src), ln_k=ln_k, head_div=head_div).eval(), ln_k, head_div


def cmd_export(a):
    import litert_torch
    torch.set_grad_enabled(False)
    m, ln_k, head_div = model_from_args(a)
    print(f"export: {len(ln_k)} LayerNorm pre-scales (k {sorted(set(ln_k.values()))}), head_div {head_div}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    kw = {"images": torch.rand(1, L_PAD, 3 * PATCH * PATCH),
          "positions_xy": torch.zeros(1, L_PAD, 2, dtype=torch.int32),
          "vit_positions": torch.zeros(1, L_PAD, 2, dtype=torch.int32)}
    path = os.path.join(a.out_dir, "vision_encoder_fp32.tflite")
    if hasattr(litert_torch, "signature"):
        edge = litert_torch.signature("vision_1088", m, sample_kwargs=kw).convert()
    else:
        edge = litert_torch.convert(m, sample_args=(), sample_kwargs=kw)
    edge.export(path)
    print(f"wrote {path} {os.path.getsize(path)} B", flush=True)
    for recipe in [r for r in a.quant.split(",") if r and r != "none"]:
        qpath = os.path.join(a.out_dir, {"fp16": "vision_encoder_fp16.tflite",
                                         "dynamic_wi8_afp32": "vision_encoder_wi8.tflite"}[recipe])
        quantize(path, qpath, recipe)
        print(f"wrote {qpath} {os.path.getsize(qpath)} B ({recipe})", flush=True)
    return 0


def quantize(src, dst, recipe):
    """fp16 = float_casting of FULLY_CONNECTED + CONV_2D weights (as minicpmocr_work/scripts/vision_export.py quant_fp16);
    dynamic_wi8_afp32 = the stock recipe."""
    import copy
    from ai_edge_quantizer import quantizer
    import ai_edge_quantizer.recipe as r
    if recipe == "dynamic_wi8_afp32":
        q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
    else:
        base = copy.deepcopy(r.dynamic_wi8_afp32()[0])
        base["algorithm_key"] = "float_casting"
        base["op_config"]["weight_tensor_config"] = {
            "num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"}
        base["op_config"]["compute_precision"] = "FLOAT"
        base["op_config"]["explicit_dequantize"] = False
        recipes = []
        for op in ("FULLY_CONNECTED", "CONV_2D"):
            rr = copy.deepcopy(base)
            rr["operation"] = op
            recipes.append(rr)
        q = quantizer.Quantizer(src, recipes)
    q.quantize().export_model(dst)


def run_tflite(path, images, pxy, vit, accel="cpu"):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    if accel == "gpu32":  # GPU with fp32 activations enforced = what LiteRT-LM's vision executor runs when the
        # section declares no activation type (LiteRT-LM then runs the vision encoder at fp32 on the GPU); plain "gpu" = fp16
        from ai_edge_litert.options import GpuOptions, Options
        model = CompiledModel.from_file(path, options=Options(hardware_accelerators=HardwareAccelerator.GPU,
                                                              gpu_options=GpuOptions(enforce_f32=True)))
    else:
        hw = {"cpu": HardwareAccelerator.CPU, "gpu": HardwareAccelerator.GPU}[accel]
        model = CompiledModel.from_file(path, hardware_accel=hw)
    sig = list(model.get_signature_list().keys())[0]
    feeds = {"images": images.numpy().astype(np.float32), "positions_xy": pxy.numpy().astype(np.int32),
             "vit_positions": vit.numpy().astype(np.int32)}
    ins = {}
    for name, arr in feeds.items():
        b = model.create_input_buffer_by_name(sig, name)
        b.write(np.ascontiguousarray(arr))
        ins[name] = b
    ob = model.create_output_buffer_by_name(sig, "features")
    model.run_by_name(sig, ins, {"features": ob})
    return np.array(ob.read(NQ * RD, np.float32), dtype=np.float32).reshape(NQ, RD)


def cmd_tflcheck(a):
    res = {}
    for key, pv, ts, want in ref_slices(a.ref):
        got = run_tflite(a.model, *slice_inputs(pv, ts), accel=a.accel)
        res[key] = {"rel_l2": rel_l2(got, want.numpy()), "max_abs": float(np.abs(got - want.numpy()).max())}
        print(key, res[key], flush=True)
    worst = max(v["rel_l2"] for v in res.values())
    print(f"TFLCHECK {a.accel} {os.path.basename(a.model)} worst rel L2 {worst:.3e} over {len(res)} slices")
    if a.out:
        json.dump({"model": a.model, "accel": a.accel, "worst_rel_l2": worst, "slices": res}, open(a.out, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--src", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out")
    p.add_argument("--fp16safe")
    p.add_argument("--head-div")
    p = sub.add_parser("export")
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quant", default="none")
    p.add_argument("--fp16safe", help="a JSON of per-LayerNorm input ranges -> LayerNorm pre-scales (optional fp16-safe form)")
    p.add_argument("--head-div", help="Resampler heads whose score logits are divided, e.g. 6:8 (optional fp16-safe form)")
    p = sub.add_parser("tflcheck")
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--accel", default="cpu")
    p.add_argument("--out")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "export": cmd_export, "tflcheck": cmd_tflcheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
