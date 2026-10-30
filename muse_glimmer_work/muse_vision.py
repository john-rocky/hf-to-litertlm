#!/usr/bin/env python3
"""muse_vision.py — Muse-Glimmer-30B vision tower as two static graphs for LiteRT-LM's fast_vlm contract.

  VISION_ENCODER : image NHWC [1,G,G,3] in [0,1] -> [1,(G/28)^2,6144]   (ViT + ln_post + 2x2 pixel shuffle)
  VISION_ADAPTER : [1,N,6144] -> [1,N,6656]                             (adapter MLP, projection, weightless RMSNorm)

G is a multiple of 448 (the 32x32-patch attention window): G = 448 is one window (window attention == full
attention, the window order is the identity), G = 896 is 2x2 windows, done with reshape/transpose (no gather).
Everything that depends on the grid is computed once with HF's own helpers at build time and stored as
constants: the interpolated position embedding, the 2D axial RoPE cos/sin, the window order. The patch
embedding (a Linear over a temporally duplicated [2,3,14,14] patch) is folded into a stride-14 Conv2d with
the two temporal kernels summed. The pixel shuffle is four strided slices stacked on the last axis (no gather).

Subcommands:
  selftest --config SNAP --layers 4 --g 448,896   random-weight HF tower vs the static module (torch, fp32)
  ref      --src SNAP --image PNG --g 448 --out ref.npz     HF reference (processor path) on a GxG image
  export   --src SNAP --g 448 --out-dir DIR [--fp16]        convert both graphs with litert_torch
  check    --enc E.tflite --adp A.tflite --image PNG --ref ref.npz   CompiledModel CPU vs the reference
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

WINDOW = 32  # patches per window side (pos_emb_height * patch_size / patch_size)


def hf_classes():
    from transformers.models.muse_glimmer import modeling_muse_glimmer as m
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    return m, MuseGlimmerConfig


def grid_constants(vision, g):
    """HF's own grid tensors for one GxG image: pos_embeds [N,D] (raster), cos/sin [N,hd] (raster)."""
    from transformers import vision_utils as vu
    cfg = vision.config
    gh = gw = g // cfg.patch_size
    grid = torch.tensor([[1, gh, gw]])
    pe = vision.patch_embedder
    idx, wts = vu.get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=pe.num_grid_per_side, mode=pe.interpolation_mode,
        align_corners=pe.interpolation_align_corners, spatial_merge_size=1, padding=pe.interpolation_padding)
    with torch.no_grad():
        pos = (pe.position_embedding_table(idx) * wts[:, :, None]).sum(1).float()
        position_ids = vu.get_vision_position_ids(grid, spatial_merge_size=1).flip(-1) + 1
        dummy = torch.zeros(1, dtype=torch.float32)
        cos, sin = vision.rotary_emb(dummy, position_ids)
    window_index, cu = vu.get_vision_window_index(grid, spatial_merge_size=1, window_size=vision.window_size,
                                                  patch_size=cfg.patch_size)
    return pos, cos.float(), sin.float(), window_index, cu


def window_order(gh, gw):
    """The window-major order HF builds with get_vision_window_index, as an index list (for the selftest)."""
    nh, nw = gh // WINDOW, gw // WINDOW
    idx = torch.arange(gh * gw).reshape(nh, WINDOW, nw, WINDOW).permute(0, 2, 1, 3).reshape(-1)
    return idx


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class StaticVisionEncoder(nn.Module):
    def __init__(self, vision, g):
        super().__init__()
        cfg = vision.config
        self.g, self.p = g, cfg.patch_size
        self.gh = self.gw = g // cfg.patch_size
        assert self.gh % WINDOW == 0 and self.gh == self.gw, f"G={g} must be a multiple of {WINDOW * self.p}"
        self.nh, self.nw = self.gh // WINDOW, self.gw // WINDOW
        self.heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.hd = self.dim // self.heads
        self.layer_types = list(cfg.layer_types)
        pos, cos, sin, window_index, cu = grid_constants(vision, g)
        if self.nh * self.nw == 1:
            assert torch.equal(window_index, torch.arange(self.gh * self.gw)), "one window: identity order"
        else:
            assert torch.equal(window_index, window_order(self.gh, self.gw)), "window order differs from HF"
        # constants kept in window order, as HF's layers see them
        order = window_order(self.gh, self.gw)
        self.register_buffer("pos", pos[order].clone(), persistent=False)
        self.register_buffer("cos", cos[order].clone(), persistent=False)
        self.register_buffer("sin", sin[order].clone(), persistent=False)
        w = vision.patch_embedder.patch_embedding.weight.detach().float()
        w = w.view(self.dim, cfg.patch_temporal, 3, self.p, self.p).sum(1)
        self.register_buffer("conv_w", w.contiguous(), persistent=False)
        self.ln_pre = vision.ln_pre
        self.layers = vision.layers
        self.ln_post = vision.ln_post

    def to_windows(self, x):
        # [gh*gw, D] raster -> window-major order (HF's window_index), by reshape/transpose
        if self.nh * self.nw == 1:
            return x
        x = x.reshape(self.nh, WINDOW, self.nw, WINDOW * self.dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(self.gh * self.gw, self.dim)

    def from_windows(self, x):
        if self.nh * self.nw == 1:
            return x
        x = x.reshape(self.nh, self.nw, WINDOW, WINDOW * self.dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(self.gh * self.gw, self.dim)

    def attention(self, attn, x, windowed):
        s = x.shape[0]
        q = attn.q_proj(x).reshape(s, self.heads, self.hd)
        k = attn.k_proj(x).reshape(s, self.heads, self.hd)
        v = attn.v_proj(x).reshape(s, self.heads, self.hd)
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        nwin = self.nh * self.nw if windowed else 1
        L = s // nwin
        # [nwin, heads, L, hd]
        q = q.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        k = k.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        v = v.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        o = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(s, self.dim)
        return attn.proj(o)

    def forward(self, image):
        x = image * 2.0 - 1.0                       # rescale 1/255 done by the runtime; mean = std = 0.5
        x = x.permute(0, 3, 1, 2)                   # NHWC -> NCHW
        x = F.conv2d(x, self.conv_w, stride=self.p)  # [1, D, gh, gw]
        x = x.permute(0, 2, 3, 1).reshape(self.gh * self.gw, self.dim)
        x = self.to_windows(x) + self.pos
        x = self.ln_pre(x)
        for layer, kind in zip(self.layers, self.layer_types):
            windowed = kind == "window_attention"
            x = x + self.attention(layer.attn, layer.norm1(x), windowed)
            x = x + layer.mlp(layer.norm2(x))
        x = self.from_windows(x)
        x = self.ln_post(x)
        # pixel shuffle: merged row (i, j) = [x(2i,2j), x(2i,2j+1), x(2i+1,2j), x(2i+1,2j+1)] interleaved per channel
        x = x.reshape(self.gh, self.gw, self.dim)
        parts = [x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]]
        x = torch.stack(parts, dim=-1)              # [gh/2, gw/2, D, 4]
        n = (self.gh // 2) * (self.gw // 2)
        return x.reshape(1, n, self.dim * 4)


class StaticVisionAdapter(nn.Module):
    def __init__(self, adapter, projection, eps):
        super().__init__()
        self.adapter = adapter
        self.projection = projection
        self.eps = eps

    def forward(self, feats):
        x = self.projection(self.adapter(feats))
        return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)


def build_hf(src=None, layers=None, seed=0, cfg_dir=None):
    """HF vision tower + adapter + projection in fp32: from the checkpoint, or random (selftest, config from cfg_dir)."""
    m, MuseGlimmerConfig = hf_classes()
    if not (src or cfg_dir):
        raise SystemExit("pass --config DIR (a directory holding the model's config.json, e.g. the HF snapshot)")
    cfg = MuseGlimmerConfig.from_pretrained(src or cfg_dir)
    vcfg = cfg.vision_config
    if layers:
        vcfg.num_hidden_layers = layers
        vcfg.layer_types = ["full_attention" if (i + 1) % 4 == 0 or i == layers - 1 else "window_attention"
                            for i in range(layers)]
    vcfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    vision = m.MuseGlimmerVisionModel(vcfg).float().eval()
    adapter = m.MuseGlimmerVisionAdapter(cfg).float().eval()
    projection = nn.Linear(cfg.projector_hidden_size, cfg.text_config.hidden_size, bias=False).float().eval()
    if src:
        load_weights(src, vision, adapter, projection)
    else:
        with torch.no_grad():
            for mod in (vision, adapter, projection):
                for _, p in mod.named_parameters():
                    p.copy_(torch.randn_like(p) * (0.1 if p.dim() == 1 else p.shape[-1] ** -0.5))
    return cfg, vision, adapter, projection


def load_weights(src, vision, adapter, projection):
    from safetensors import safe_open
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    groups = {"model.vision_tower.": vision, "model.vision_adapter.": adapter, "model.vision_projection.": projection}
    want = {}
    for prefix, mod in groups.items():
        for name, p in mod.state_dict().items():
            want[prefix + name] = (mod, name)
    by_file = {}
    for key in want:
        by_file.setdefault(index[key], []).append(key)
    for fname, keys in by_file.items():
        with safe_open(os.path.join(src, fname), framework="pt") as f:
            for key in keys:
                mod, name = want[key]
                t = f.get_tensor(key).float()
                with torch.no_grad():
                    mod.state_dict()[name].copy_(t)
    print(f"loaded {len(want)} vision tensors")


def hf_forward(vision, adapter, projection, eps, pixel_values, grid):
    with torch.no_grad():
        feats = vision(pixel_values=pixel_values, grid_thw=grid).last_hidden_state
        out = projection(adapter(feats))
        out = out * torch.pow(out.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return feats, out


def hf_pixels(image_u8, patch, temporal):
    """The processor's pixel layout for an image already at GxG (no resize): [N, temporal*3*p*p]."""
    x = torch.from_numpy(image_u8).float().permute(2, 0, 1)[None] / 255.0
    x = (x - 0.5) / 0.5
    _, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    x = x.view(1, c, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5)
    x = x.unsqueeze(3).expand(-1, -1, -1, temporal, -1, -1, -1).reshape(1, gh * gw, temporal * c * patch * patch)
    return x[0], torch.tensor([[1, gh, gw]])


def cmd_selftest(a):
    cfg, vision, adapter, projection = build_hf(layers=a.layers, seed=a.seed, cfg_dir=a.config)
    eps = cfg.text_config.rms_norm_eps
    for g in [int(x) for x in a.g.split(",")]:
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(g, g, 3), dtype=np.uint8)
        pv, grid = hf_pixels(img, vision.config.patch_size, vision.config.patch_temporal)
        ref_feats, ref_out = hf_forward(vision, adapter, projection, eps, pv, grid)
        enc = StaticVisionEncoder(vision, g).eval()
        adp = StaticVisionAdapter(adapter, projection, eps).eval()
        with torch.no_grad():
            got_feats = enc(torch.from_numpy(img).float()[None] / 255.0)
            got_out = adp(got_feats)
        df = (got_feats[0] - ref_feats).abs().max().item()
        do = (got_out[0] - ref_out).abs().max().item()
        rel = ((got_out[0] - ref_out).norm() / ref_out.norm()).item()
        print(f"G={g} layers={a.layers} feats {tuple(got_feats.shape)} max|d| {df:.3e}; "
              f"out {tuple(got_out.shape)} max|d| {do:.3e} rel {rel:.3e}")
    return 0


def load_image(path, g):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    assert im.size == (g, g), f"{path} is {im.size}, expected {g}x{g} (resize it first)"
    return np.asarray(im, dtype=np.uint8)


def cmd_ref(a):
    cfg, vision, adapter, projection = build_hf(src=a.src, layers=a.layers, seed=a.seed)
    eps = cfg.text_config.rms_norm_eps
    if a.image:
        img = load_image(a.image, a.g)
    else:
        img = np.random.default_rng(1).integers(0, 256, size=(a.g, a.g, 3), dtype=np.uint8)
    pv, grid = hf_pixels(img, vision.config.patch_size, vision.config.patch_temporal)
    feats, out = hf_forward(vision, adapter, projection, eps, pv, grid)
    np.savez_compressed(a.out, image=img, feats=feats.numpy(), out=out.numpy())
    print(f"ref feats {tuple(feats.shape)} out {tuple(out.shape)} -> {a.out}")
    return 0


def cmd_export(a):
    import litert_torch
    cfg, vision, adapter, projection = build_hf(src=a.src, layers=a.layers, seed=a.seed)
    eps = cfg.text_config.rms_norm_eps
    enc = StaticVisionEncoder(vision, a.g).eval()
    adp = StaticVisionAdapter(adapter, projection, eps).eval()
    os.makedirs(a.out_dir, exist_ok=True)
    n = (a.g // 28) ** 2
    sample_img = (torch.rand(1, a.g, a.g, 3),)
    sample_feats = (torch.randn(1, n, enc.dim * 4),)
    tag = f"g{a.g}" + (f"_l{a.layers}" if a.layers else "")
    for name, mod, sample in (("vision_encoder", enc, sample_img), ("vision_adapter", adp, sample_feats)):
        path = os.path.join(a.out_dir, f"{name}_{tag}_fp32.tflite")
        with torch.no_grad():
            m = litert_torch.convert(mod, sample)
        m.export(path)
        print(f"wrote {path} {os.path.getsize(path)} B")
        for recipe in [r for r in a.quant.split(",") if r and r != "none"]:
            # litert-torch 0.10.0: quantize_model lives in export_lib and writes <name>_quantized.tflite
            from litert_torch.generative.export_hf.core import export_lib
            short = {"dynamic_wi8_afp32": "wi8"}.get(recipe, recipe)
            qpath = os.path.join(a.out_dir, f"{name}_{tag}_{short}.tflite")
            os.replace(export_lib.quantize_model(path, recipe), qpath)
            print(f"wrote {qpath} {os.path.getsize(qpath)} B ({recipe})")
    return 0


def run_tflite(path, inputs):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    model = CompiledModel.from_file(path, hardware_accel=HardwareAccelerator.CPU)
    sig = list(model.get_signature_list().keys())[0]
    in_det = model.get_input_tensor_details(sig)
    out_det = model.get_output_tensor_details(sig)
    assert len(in_det) == 1 and len(out_det) == 1, (in_det.keys(), out_det.keys())
    (iname, idet), (oname, odet) = next(iter(in_det.items())), next(iter(out_det.items()))
    ib = model.create_input_buffer_by_name(sig, iname)
    ib.write(np.ascontiguousarray(inputs.astype(np.float32)))
    ob = model.create_output_buffer_by_name(sig, oname)
    model.run_by_name(sig, {iname: ib}, {oname: ob})
    shape = odet["shape"]
    return np.array(ob.read(int(np.prod(shape)), np.float32), dtype=np.float32).reshape(shape)


def cmd_check(a):
    ref = np.load(a.ref)
    img = ref["image"].astype(np.float32)[None] / 255.0
    feats = run_tflite(a.enc, img)
    out = run_tflite(a.adp, feats)
    out_ref_feats = run_tflite(a.adp, ref["feats"][None])
    res = {}
    for name, got, want in (("encoder", feats[0], ref["feats"]), ("encoder+adapter", out[0], ref["out"]),
                            ("adapter on ref feats", out_ref_feats[0], ref["out"])):
        d = np.abs(got - want)
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        cos = float(np.mean(np.sum(got * want, -1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))))
        res[name] = {"max_abs": float(d.max()), "rel_l2": rel, "mean_cos": cos}
        print(f"{name:22s} max|d| {d.max():.3e} rel L2 {rel:.3e} mean cos {cos:.6f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"enc": a.enc, "adp": a.adp, "ref": a.ref, **res}, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--config", required=True, help="a directory holding the model's config.json (the HF snapshot)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--g", default="448,896")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_selftest)
    for name, fn in (("ref", cmd_ref), ("export", cmd_export)):
        p = sub.add_parser(name)
        p.add_argument("--src", help="checkpoint dir; omitted = random weights")
        p.add_argument("--layers", type=int, help="truncate the tower (a quick check)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--g", type=int, default=448)
        if name == "ref":
            p.add_argument("--image")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--out-dir", required=True)
            p.add_argument("--quant", default="none", help="comma list of quantizer recipes after the fp32 file")
        p.set_defaults(fn=fn)
    p = sub.add_parser("check")
    p.add_argument("--enc", required=True)
    p.add_argument("--adp", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
