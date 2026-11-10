#!/usr/bin/env python3
# Copied from wevisone_work/tools/vision_taps.py (WeVisOne-VL-2B): dimensions read from the snapshot's config.json
# (tools/wv_dims.py), LayerNorm scales calibrated on the three public pictures, the modes that need the source
# package's reference taps (check, export) left out.
"""WeVisOne-VL-4B's vision tower re-authored in plain torch at a fixed 512x512 input: the four merger inputs (after
ViT blocks 23 / 5 / 11 / 17) concatenated in raster order, [1, 1024, 4096]; the wide adapter emits merger(h23) and the
three deepstack mergers concatenated, [1, 256, 4 * OUT_HID] = [1, 256, 10240] (tools/wv_vision.py appends the image
flag and three zeros).

Static single-image rewrite (northmv_work/convert_northmv_vision.py, qwen35vl_work/convert_qwen35_vision.py):
  - Conv3d(temporal 2) -> Conv2d with the summed temporal kernel (the processor repeats a still image in both frames).
  - The learned 48x48 position table resampled to the 32x32 grid in RASTER order and the 2-D vision rotary, both
    computed once in the reference environment (transformers 5.12.1 helpers) and stored as constants (consts mode).
  - Patches stay in raster order through the (permutation-equivariant) encoder, full attention, rank >= 3 everywhere.
  - Encoder output = concat(h23, h5, h11, h17) [1, 1024, 4096]; wide adapter = concat(merger(h23), deepstack
    merger 0 (h5), 1 (h11), 2 (h17)) [1, 256, 10240] (2x2 merges as four strided slices + concat).
  - Normalization (x - 0.5) / 0.5 and NHWC -> NCHW inside (the runtime feeds [0, 1] NHWC).
  - LayerNorm pre-scaled by a power of two S (exact in real arithmetic, fp16 range) when FP16SAFE=1: the published
    scales when LN_SCALES points at ln_scales_512.json, else calibrated on CALIB_ROWS.

Modes (reference environment: torch 2.11.0, transformers 5.12.1, Pillow 12.2.0):
  consts  out/vision512/consts.npz, and ref_<row>.npz with the 512x512 picture of the first calibration row (the
          export traces with it); of every calibration row when LN_SCALES is unset
  scales  the LayerNorm scales calibrated again on CALIB_ROWS -> out/vision512/ln_scales_check.json, compared with
          LN_SCALES when it is set (exit 1 on a difference)
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wv_dims", Path(__file__).resolve().parent / "wv_dims.py")
DIMS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DIMS)
SNAP = DIMS.SNAP
OUT = K / "out" / "vision512"
IMG, PATCH, MERGE = 512, DIMS.V_PATCH, DIMS.V_MERGE
GRID = IMG // PATCH                 # 32
N = GRID * GRID                     # 1024 patches
NTOK = N // (MERGE * MERGE)         # 256 merged tokens
HID, HEADS, DEPTH = DIMS.V_HID, DIMS.V_HEADS, DIMS.V_DEPTH      # vision tower: 1024, 16, 24 (2B and 4B)
V_MLP = DIMS.V_MLP                  # 4096
OUT_HID = DIMS.OUT_HID              # merger output = text hidden (2B 2048, 4B 2560)
LAST = DEPTH - 1                    # 23: the merger input
DEEPSTACK = DIMS.DEEPSTACK          # (5, 11, 17)
TAPS = tuple(sorted((*DEEPSTACK, LAST)))
# the three public pictures of WeVisOne-VL-2B's calibration rows (its fourth row, a crop of a figure of the source
# package, is not redistributed and is not used here)
CALIB_ROWS = ("syn_a_det", "nasa_jungle_det", "nasa_truck_det")
FP16SAFE = os.environ.get("FP16SAFE", "1") == "1"
# LN_SCALES=<json>: take the published LayerNorm scales (ln_scales_512.json) instead of calibrating on CALIB_ROWS
LN_SCALES = os.environ.get("LN_SCALES", "")


def load_visual_weights():
    from safetensors import safe_open
    w = {}
    with safe_open(str(SNAP / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                w[k[len("model.visual."):]] = f.get_tensor(k).float()
    return w


def fixture_image(row_id, side=IMG):
    from PIL import Image
    rows = {r["id"]: r for r in json.loads((DIMS.FIX / "fixtures.json").read_text())["rows"]}
    with Image.open(DIMS.FIX / rows[row_id]["image"]) as im:
        return im.convert("RGB").resize((side, side), Image.Resampling.BICUBIC)


def ln(x, w, b, eps, S=1.0):
    if not FP16SAFE or S <= 1.0:
        return F.layer_norm(x, (x.shape[-1],), w, b, eps)
    xs = x * (1.0 / S)
    d = xs - xs.mean(-1, keepdim=True)
    var = (d * d).mean(-1, keepdim=True)
    return d * torch.rsqrt(var + eps / (S * S)) * w + b


def rot_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def merge2x2(f):
    """[1, N, C] raster -> [1, N/4, 4C] in the merger's block order (intra (0,0), (0,1), (1,0), (1,1))."""
    f = f.reshape(1, GRID, GRID, -1)
    m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :], f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
    return m.reshape(1, NTOK, -1)


def taps_from_enc(enc):
    """enc [1, 1024, 4096] (h23 | h5 | h11 | h17, raster) -> {layer: [1, 1024, 32, 32] row-major NCHW}."""
    out = {}
    for j, layer in enumerate((LAST, *DEEPSTACK)):
        out[layer] = enc[:, :, j * HID:(j + 1) * HID].reshape(1, GRID, GRID, HID).permute(0, 3, 1, 2)
    return out


class VisionTaps(torch.nn.Module):
    """images [1, 512, 512, 3] in [0, 1] -> enc [1, 1024, 4096] = the merger inputs h23 | h5 | h11 | h17, raster."""

    def __init__(self, w, consts, ln_scale=None):
        super().__init__()
        p = lambda t: torch.nn.Parameter(t.clone(), requires_grad=False)
        w3 = w["patch_embed.proj.weight"]                              # [1024, 3, 2, 16, 16]
        self.conv = torch.nn.Conv2d(3, HID, PATCH, stride=PATCH, bias=True)
        self.conv.weight = p(w3[:, :, 0] + w3[:, :, 1])
        self.conv.bias = p(w["patch_embed.proj.bias"])
        self.register_buffer("pos", torch.from_numpy(consts["pos_raster"]).reshape(1, N, HID), persistent=False)
        self.register_buffer("cos", torch.from_numpy(consts["cos_raster"]).reshape(1, 1, N, -1), persistent=False)
        self.register_buffer("sin", torch.from_numpy(consts["sin_raster"]).reshape(1, 1, N, -1), persistent=False)
        self.blocks = torch.nn.ModuleList()
        for i in range(DEPTH):
            blk = torch.nn.Module()
            for name in ("norm1", "norm2"):
                m = torch.nn.LayerNorm(HID, eps=1e-6)
                m.weight, m.bias = p(w[f"blocks.{i}.{name}.weight"]), p(w[f"blocks.{i}.{name}.bias"])
                setattr(blk, name, m)
            blk.qkv = torch.nn.Linear(HID, 3 * HID)
            blk.qkv.weight, blk.qkv.bias = p(w[f"blocks.{i}.attn.qkv.weight"]), p(w[f"blocks.{i}.attn.qkv.bias"])
            blk.proj = torch.nn.Linear(HID, HID)
            blk.proj.weight, blk.proj.bias = p(w[f"blocks.{i}.attn.proj.weight"]), p(w[f"blocks.{i}.attn.proj.bias"])
            blk.fc1 = torch.nn.Linear(HID, V_MLP)
            blk.fc1.weight, blk.fc1.bias = p(w[f"blocks.{i}.mlp.linear_fc1.weight"]), p(w[f"blocks.{i}.mlp.linear_fc1.bias"])
            blk.fc2 = torch.nn.Linear(V_MLP, HID)
            blk.fc2.weight, blk.fc2.bias = p(w[f"blocks.{i}.mlp.linear_fc2.weight"]), p(w[f"blocks.{i}.mlp.linear_fc2.bias"])
            self.blocks.append(blk)
        self.ln_scale = ln_scale or {}
        self.calib = None

    def attn(self, blk, x):
        qkv = blk.qkv(x)                                                # [1, N, 3C]
        qkv = qkv.reshape(N, 3, HEADS, HID // HEADS).permute(1, 2, 0, 3)  # [3, H, N, d]
        q, k, v = qkv[0:1], qkv[1:2], qkv[2:3]                          # [1, H, N, d]
        q = q * self.cos + rot_half(q) * self.sin
        k = k * self.cos + rot_half(k) * self.sin
        a = torch.softmax((q * (HID // HEADS) ** -0.5) @ k.transpose(-2, -1), dim=-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(1, N, HID)
        return blk.proj(o)

    def forward(self, images):
        x = (images.permute(0, 3, 1, 2) - 0.5) / 0.5
        h = self.conv(x).permute(0, 2, 3, 1).reshape(1, N, HID) + self.pos   # raster order
        taps = {}
        for i, blk in enumerate(self.blocks):
            if self.calib is not None:
                self.calib[f"b{i}.n1"] = max(self.calib.get(f"b{i}.n1", 0.0), float(h.abs().max()))
            h = h + self.attn(blk, ln(h, blk.norm1.weight, blk.norm1.bias, 1e-6, self.ln_scale.get(f"b{i}.n1", 1.0)))
            if self.calib is not None:
                self.calib[f"b{i}.n2"] = max(self.calib.get(f"b{i}.n2", 0.0), float(h.abs().max()))
            n2 = ln(h, blk.norm2.weight, blk.norm2.bias, 1e-6, self.ln_scale.get(f"b{i}.n2", 1.0))
            h = h + blk.fc2(F.gelu(blk.fc1(n2), approximate="tanh"))
            if i in TAPS:
                taps[i] = h
        return torch.cat([taps[LAST], *(taps[i] for i in DEEPSTACK)], dim=-1)    # [1, 1024, 4 * HID]


class WideAdapter(torch.nn.Module):
    """enc [1, 1024, 4096] -> concat(merger(h23), deepstack_merger_0(h5), _1(h11), _2(h17)) [1, 256, 4 * OUT_HID]."""

    def __init__(self, w, ln_scale=None):
        super().__init__()
        p = lambda t: torch.nn.Parameter(t.clone(), requires_grad=False)
        self.ln_scale = ln_scale or {}

        def mk(prefix, norm_dim):
            m = torch.nn.Module()
            m.norm_w, m.norm_b = p(w[f"{prefix}.norm.weight"]), p(w[f"{prefix}.norm.bias"])
            m.norm_dim = norm_dim
            m.fc1 = torch.nn.Linear(*reversed(w[f"{prefix}.linear_fc1.weight"].shape))      # [4096 <- 4096]
            m.fc1.weight, m.fc1.bias = p(w[f"{prefix}.linear_fc1.weight"]), p(w[f"{prefix}.linear_fc1.bias"])
            m.fc2 = torch.nn.Linear(*reversed(w[f"{prefix}.linear_fc2.weight"].shape))      # [OUT_HID <- 4096]
            assert m.fc2.out_features == OUT_HID, (prefix, m.fc2.out_features)
            m.fc2.weight, m.fc2.bias = p(w[f"{prefix}.linear_fc2.weight"]), p(w[f"{prefix}.linear_fc2.bias"])
            return m
        self.merger = mk("merger", HID)
        self.ds = torch.nn.ModuleList([mk(f"deepstack_merger_list.{i}", 4 * HID) for i in range(len(DEEPSTACK))])

    def forward(self, enc):
        h23, h5, h11, h17 = enc.split(HID, dim=-1)
        m = self.merger
        x = merge2x2(ln(h23, m.norm_w, m.norm_b, 1e-6, self.ln_scale.get("final", 1.0)))
        parts = [m.fc2(F.gelu(m.fc1(x)))]
        for d, h, key in zip(self.ds, (h5, h11, h17), ("ds0", "ds1", "ds2")):
            x = ln(merge2x2(h), d.norm_w, d.norm_b, 1e-6, self.ln_scale.get(key, 1.0))
            parts.append(d.fc2(F.gelu(d.fc1(x))))
        return torch.cat(parts, dim=-1)


def calib_scale(absmax):
    return float(2 ** max(0, int(math.ceil(math.log2(max(absmax, 1e-6) / 8.0)))))


def mode_consts():
    """Reference environment: the position and rotary constants, and the picture(s) the export reads."""
    from transformers import AutoModelForImageTextToText
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = AutoModelForImageTextToText.from_pretrained(str(SNAP), dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    visual = model.model.visual
    grid = torch.tensor([[1, GRID, GRID]])
    idx, wts = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=visual.num_grid_per_side,
                                                       spatial_merge_size=1)
    with torch.no_grad():
        pos_raster = (visual.pos_embed.weight[idx] * wts[:, :, None]).sum(0)          # [1024, 1024] raster
        pid = torch.arange(N)
        rpe = visual.rotary_pos_emb(torch.stack([pid // GRID, pid % GRID], dim=-1))   # [N, 32]
        emb = torch.cat((rpe, rpe), dim=-1)
    np.savez(OUT / "consts.npz", pos_raster=pos_raster.numpy().astype(np.float32),
             cos_raster=emb.cos().numpy().astype(np.float32), sin_raster=emb.sin().numpy().astype(np.float32))
    rows = CALIB_ROWS[:1] if LN_SCALES else CALIB_ROWS
    for row in rows:
        np.savez(OUT / f"ref_{row}.npz", image=np.asarray(fixture_image(row), dtype=np.uint8))
    import transformers
    (OUT / "consts.json").write_text(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                                                  "pictures": list(rows)}) + "\n")
    print("consts", OUT / "consts.npz", "pictures", list(rows), flush=True)
    return 0


def build(w=None):
    w = w or load_visual_weights()
    consts = dict(np.load(OUT / "consts.npz"))
    enc = VisionTaps(w, consts).eval()
    scale = {}
    if FP16SAFE and LN_SCALES:
        scale = {k: float(v) for k, v in json.loads(Path(LN_SCALES).read_text())["ln_scales"].items()}
        enc.ln_scale = scale
    elif FP16SAFE:
        enc.calib = {}
        with torch.no_grad():
            for row in CALIB_ROWS:
                im = np.load(OUT / f"ref_{row}.npz")["image"]
                enc(torch.from_numpy(im).float()[None] / 255.0)
        scale = {k: calib_scale(v) for k, v in enc.calib.items()}
        enc.calib = None
        enc.ln_scale = scale
    ada = WideAdapter(w).eval()
    return enc, ada, scale


def mode_scales():
    """The LayerNorm scales calibrated on CALIB_ROWS (each picture resized here, as consts does) against LN_SCALES."""
    torch.set_num_threads(1)
    enc = VisionTaps(load_visual_weights(), dict(np.load(OUT / "consts.npz"))).eval()
    enc.calib = {}
    with torch.no_grad():
        for row in CALIB_ROWS:
            im = np.asarray(fixture_image(row), dtype=np.uint8)
            enc(torch.from_numpy(im).float()[None] / 255.0)
    res = {"rows": list(CALIB_ROWS), "absmax": dict(sorted(enc.calib.items())),
           "scales": {k: calib_scale(v) for k, v in sorted(enc.calib.items())}}
    if LN_SCALES:
        want = {k: float(v) for k, v in json.loads(Path(LN_SCALES).read_text())["ln_scales"].items()}
        res["ln_scales_file"] = LN_SCALES
        res["differ"] = sorted(k for k in set(want) | set(res["scales"]) if want.get(k) != res["scales"].get(k))
    (OUT / "ln_scales_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("LN_SCALES", sum(v > 1 for v in res["scales"].values()), "of", len(res["scales"]), "> 1;",
          "differ from the file:", res.get("differ", "(LN_SCALES not set)"), flush=True)
    return 1 if res.get("differ") else 0


if __name__ == "__main__":
    sys.exit({"consts": mode_consts, "scales": mode_scales}[sys.argv[1]]())
