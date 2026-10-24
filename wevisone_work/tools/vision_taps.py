#!/usr/bin/env python3
"""Question 3 (a): WeVisOne-VL-2B's vision tower re-authored in plain torch at a fixed 512x512 input. One encoder output
serves both paths (the fast_vlm adapter and the pixel decoder's taps): the four merger inputs (after ViT blocks 23 / 5 / 11 / 17)
concatenated in raster order, [1, 1024, 4096]; the mask-decoder taps are its 1024-wide slices reshaped to row-major
[1, 1024, 32, 32]; the wide adapter emits merger(h23) and the three deepstack mergers concatenated, [1, 256, 8192].

Static single-image rewrite (northmv_work/convert_northmv_vision.py, qwen35vl_work/convert_qwen35_vision.py):
  - Conv3d(temporal 2) -> Conv2d with the summed temporal kernel (the processor repeats a still image in both frames).
  - The learned 48x48 position table resampled to the 32x32 grid in RASTER order and the 2-D vision rotary, both
    computed once in the reference venv (transformers 5.12.1 helpers) and stored as constants (consts mode).
  - Patches stay in raster order through the (permutation-equivariant) encoder, full attention, rank >= 3 everywhere.
    Raster order makes each tap a plain reshape + permute to NCHW: no GATHER for the block-major -> row-major undo.
  - Encoder output = concat(h23, h5, h11, h17) [1, 1024, 4096]; wide adapter = concat(merger(h23), deepstack
    merger 0 (h5), 1 (h11), 2 (h17)) [1, 256, 8192] (2x2 merges as four strided slices + concat).
  - Normalization (x - 0.5) / 0.5 and NHWC -> NCHW inside (the runtime feeds [0, 1] NHWC).
  - LayerNorm pre-scaled by a calibrated power of two S (exact in real arithmetic, fp16 range) when FP16SAFE=1.

Modes:
  consts  (reference venv)  out/vision512/consts.npz + ref_<row>.npz (HF taps via the package's VisualFeatureCapture
                            and premerge_tokens_to_2d, the HF merger / deepstack outputs) for the calibration rows
  check   (reference venv)  torch re-author vs the HF references -> ref/p3a_torch.json
  export  (export venv)     litert_torch.convert -> out/vision512/vision_encoder.tflite + wide_adapter.tflite, op
                            tables, CPU interpreter vs the torch re-author and vs HF -> ref/p3a_export.json
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("WEVISONE_SNAPSHOT", "~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/"
                           "snapshots/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf")).expanduser()
OUT = K / "out" / "vision512"
IMG, PATCH, MERGE = 512, 16, 2
GRID = IMG // PATCH                 # 32
N = GRID * GRID                     # 1024 patches
NTOK = N // (MERGE * MERGE)         # 256 merged tokens
HID, HEADS, DEPTH = 1024, 16, 24
TAPS = (5, 11, 17, 23)
DEEPSTACK = (5, 11, 17)
CALIB_ROWS = ("syn_a_det", "nasa_jungle_det", "nasa_truck_det", "cs_matting_daikin")
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
    rows = {r["id"]: r for r in json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]}
    with Image.open(K / "fixtures" / rows[row_id]["image"]) as im:
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
    for j, layer in enumerate((23, 5, 11, 17)):
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
            blk.fc1 = torch.nn.Linear(HID, 4096)
            blk.fc1.weight, blk.fc1.bias = p(w[f"blocks.{i}.mlp.linear_fc1.weight"]), p(w[f"blocks.{i}.mlp.linear_fc1.bias"])
            blk.fc2 = torch.nn.Linear(4096, HID)
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
        return torch.cat([taps[23], taps[5], taps[11], taps[17]], dim=-1)        # [1, 1024, 4096]


class WideAdapter(torch.nn.Module):
    """enc [1, 1024, 4096] -> concat(merger(h23), deepstack_merger_0(h5), _1(h11), _2(h17)) [1, 256, 8192]."""

    def __init__(self, w, ln_scale=None):
        super().__init__()
        p = lambda t: torch.nn.Parameter(t.clone(), requires_grad=False)
        self.ln_scale = ln_scale or {}

        def mk(prefix, norm_dim):
            m = torch.nn.Module()
            m.norm_w, m.norm_b = p(w[f"{prefix}.norm.weight"]), p(w[f"{prefix}.norm.bias"])
            m.norm_dim = norm_dim
            m.fc1 = torch.nn.Linear(4096, 4096)
            m.fc1.weight, m.fc1.bias = p(w[f"{prefix}.linear_fc1.weight"]), p(w[f"{prefix}.linear_fc1.bias"])
            m.fc2 = torch.nn.Linear(4096, 2048)
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
    """Reference venv: constants and HF references."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = AutoModelForImageTextToText.from_pretrained(str(SNAP), dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    visual = model.model.visual
    proc = AutoProcessor.from_pretrained(str(SNAP), min_pixels=IMG * IMG, max_pixels=IMG * IMG)
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
    if LN_SCALES:
        img = fixture_image(CALIB_ROWS[0])
        np.savez(OUT / f"ref_{CALIB_ROWS[0]}.npz", image=np.asarray(img, dtype=np.uint8))
        print("consts only (LN_SCALES set): the trace image", CALIB_ROWS[0], flush=True)
        return
    sys.path.insert(0, str(K / "src" / "WeVisOne-VL"))
    from wevisone_vl.feature_bridge import VisualFeatureCapture, premerge_tokens_to_2d
    for row in CALIB_ROWS:
        img = fixture_image(row)
        pp = proc.image_processor(images=[img], return_tensors="pt")
        assert pp["image_grid_thw"].tolist() == [[1, GRID, GRID]]
        with torch.no_grad(), VisualFeatureCapture(model, feature_layers=TAPS) as cap:
            vo = visual(pp["pixel_values"], grid_thw=pp["image_grid_thw"])
        taps = np.stack([premerge_tokens_to_2d(cap.premerge_features[i], (GRID, GRID), MERGE)[0].numpy()
                         for i in TAPS])                                                # [4, 1024, 32, 32]
        wide = torch.cat([vo.pooler_output, *vo.deepstack_features], dim=-1).numpy()          # [256, 8192]
        np.savez(OUT / f"ref_{row}.npz", image=np.asarray(img, dtype=np.uint8), taps=taps.astype(np.float32),
                 wide=wide.astype(np.float32))
        print("ref", row, taps.shape, wide.shape, flush=True)
    import transformers
    (OUT / "consts.json").write_text(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                                                  "rows": CALIB_ROWS}) + "\n")


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


def compare(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return {"max_abs": float(np.abs(a - b).max()), "ref_absmax": float(np.abs(b).max()),
            "corr": float(np.corrcoef(a, b)[0, 1])}


def mode_check():
    torch.set_num_threads(1)
    enc, ada, scale = build()
    out = {"fp16safe": FP16SAFE, "ln_scales_gt1": {k: v for k, v in sorted(scale.items()) if v > 1}, "rows": {}}
    for row in CALIB_ROWS:
        ref = np.load(OUT / f"ref_{row}.npz")
        x = torch.from_numpy(ref["image"]).float()[None] / 255.0
        with torch.no_grad():
            e = enc(x)
            maps = taps_from_enc(e)
            wide = ada(e)[0]
        out["rows"][row] = {f"tap{t}": compare(maps[t][0].numpy(), ref["taps"][i]) for i, t in enumerate(TAPS)}
        out["rows"][row]["wide"] = compare(wide.numpy(), ref["wide"])
        out["rows"][row]["wide_merger_part"] = compare(wide[:, :2048].numpy(), ref["wide"][:, :2048])
        print(row, json.dumps(out["rows"][row]), flush=True)
    out["worst_tap_max_abs"] = max(v["max_abs"] for r in out["rows"].values() for k, v in r.items() if k.startswith("tap"))
    out["worst_wide_max_abs"] = max(r["wide"]["max_abs"] for r in out["rows"].values())
    (K / "ref" / "p3a_torch.json").write_text(json.dumps(out, indent=1) + "\n")


def op_table(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("wv_op_table", Path(__file__).resolve().parent / "op_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.op_table(path)


def run_tflite(path, inputs):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], np.ascontiguousarray(x, dtype=d["dtype"]))
    t0 = time.perf_counter()
    it.invoke()
    dt = time.perf_counter() - t0
    outs = sorted(it.get_output_details(), key=lambda d: d["name"])
    return [it.get_tensor(d["index"]) for d in it.get_output_details()], dt, [d["name"] for d in it.get_output_details()]


def mode_export():
    import litert_torch
    torch.set_num_threads(1)
    enc, ada, scale = build()
    res = {"litert_torch": getattr(litert_torch, "__version__", "?"), "torch": torch.__version__, "fp16safe": FP16SAFE,
           "ln_scales_gt1": {k: v for k, v in sorted(scale.items()) if v > 1}, "rows": {}}
    ref0 = np.load(OUT / f"ref_{CALIB_ROWS[0]}.npz")
    x0 = torch.from_numpy(ref0["image"]).float()[None] / 255.0
    with torch.no_grad():
        e0 = enc(x0)
    t0 = time.time()
    litert_torch.convert(enc, (x0,)).export(str(OUT / "vision_encoder.tflite"))
    res["enc_convert_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    litert_torch.convert(ada, (e0,)).export(str(OUT / "wide_adapter.tflite"))
    res["ada_convert_s"] = round(time.time() - t0, 1)
    res["enc_ops"] = op_table(OUT / "vision_encoder.tflite")
    res["ada_ops"] = op_table(OUT / "wide_adapter.tflite")
    for row in CALIB_ROWS:
        ref = np.load(OUT / f"ref_{row}.npz")
        x = torch.from_numpy(ref["image"]).float()[None] / 255.0
        with torch.no_grad():
            e_t = enc(x)
            w_t = ada(e_t)
        (enc_out,), dt, _ = run_tflite(OUT / "vision_encoder.tflite", [x.numpy()])
        (wide_out,), adt, _ = run_tflite(OUT / "wide_adapter.tflite", [enc_out])
        maps = taps_from_enc(torch.from_numpy(enc_out))
        r = {"enc_cpu_s": round(dt, 2), "ada_cpu_s": round(adt, 2),
             "enc_vs_torch": compare(enc_out, e_t.numpy()),
             "taps_vs_hf": {f"tap{t}": compare(maps[t][0].numpy(), ref["taps"][i]) for i, t in enumerate(TAPS)},
             "wide_vs_torch": compare(wide_out, w_t.numpy()),
             "wide_vs_hf": compare(wide_out[0], ref["wide"])}
        res["rows"][row] = r
        print(row, json.dumps({k: v for k, v in r.items()}), flush=True)
    (K / "ref" / "p3a_export.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("enc flagged", res["enc_ops"]["flagged"], "rank>4", res["enc_ops"]["n_rank_gt4"],
          "| ada flagged", res["ada_ops"]["flagged"], flush=True)


if __name__ == "__main__":
    {"consts": mode_consts, "check": mode_check, "export": mode_export}[sys.argv[1]]()
