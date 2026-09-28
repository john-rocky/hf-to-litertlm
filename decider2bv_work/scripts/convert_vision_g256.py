"""Export decider-2b-vision's vision path (qwen3_5 ViT, 24 x 1024, NO deepstack) at a static 256x256 input as two
fp32 tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,256,256,3] in [0,1] -> patch features [1,256,1024]   (16x16 patches, raster order)
  VISION_ADAPTER: features [1,256,1024]               -> soft tokens  [1,64,2048]     (2x2 merge, window order)

Copied from qwen35vl_work/convert_qwen35_vision.py (same static-single-image rewrite) and changed for this lane:
  - MODEL = the pinned snapshot dir, IMG = 256 (16x16 patches -> 64 tokens), output out/vision_g256/.
  - Position-embedding taps come from transformers 5.17.0 (out/vision_g256/interp_5170_g256.npz, written by
    scripts/vision_interp_5170.py): the parity reference is 5.17.0 and 5.14.1's helper rounds the same bilinear
    weights differently (max |dw| 3.6e-6).
  - fp16-safe LayerNorm scales are recalibrated from zero (LN_S.clear()) on every fixture 256 PNG of v1 + v2
    (the exact graph inputs in out/fixtures_resized/g256/) plus one uniform-noise image.
  - fp32 only (no int8 / fp16 stage: fp16 is round 4). The 5.17.0 parity lives in scripts/vision_parity_g256.py;
    this script's own eager check compares against the 5.14.1 tower it loads (sanity, not the gate).

Rewrite (unchanged from the precedent): Conv3d(temporal 2) -> Conv2d with the summed temporal kernel (exact for a
still image); learned pos_embed resampled to the static grid and precomputed in raster order; 2-D rope precomputed
in raster order; explicit full attention; patches stay in raster order through the encoder and the 2x2 merge is 4
strided slices + concat in the adapter (no GATHER_ND on litert-converter 0.4.0); (x - 0.5) / 0.5 and NHWC -> NCHW
baked in; every activation keeps a leading batch dim (rank >= 3); LN -> FC clamp barrier; fp16-safe LayerNorm with
calibrated power-of-two pre-scales.

    PYTHONDONTWRITEBYTECODE=1 $PY_VISION -B scripts/convert_vision_g256.py     (from decider2bv_work/)
"""
import hashlib
import json
import os
import sys
import time
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch
from PIL import Image

import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "out/src/decider-2b-vision"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/vision_g256")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "256"))
PATCH = 16
MERGE = 2
assert IMG % (PATCH * MERGE) == 0, "IMG must be a multiple of 32 (patch16 x merge2)"
GRID = IMG // PATCH
N_PATCH = GRID * GRID
N_TOK = N_PATCH // (MERGE * MERGE)
MEAN, STD = 0.5, 0.5
BARRIER = True
FP16SAFE = True
LN_S = {}  # per-LN pre-scale for the fp16-safe LayerNorm (filled by calibration)
FIXTURES_V2 = os.path.join(ROOT, "fixtures/fixtures_v2.json")
INTERP = os.path.join(OUT, f"interp_5170_g{IMG}.npz")


def sha256_file(p):
  with open(p, "rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n": len(h), "ops": dict(sorted(h.items())),
          "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
          "custom": sorted(k for k in h if "CUSTOM" in k.upper()),
          "gather": sorted(k for k in h if "GATHER" in k.upper())}


def tfl_run(p, x):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  d = it.get_input_details()[0]
  it.set_tensor(d["index"], x.detach().cpu().numpy().astype(d["dtype"]))
  it.invoke()
  o = it.get_output_details()[0]
  return it.get_tensor(o["index"])


def fixture_inputs():
  """Every image row of fixtures v1 + v2 (all fixture rows): the GxG PNG the oracle and the graph consume."""
  rows = json.load(open(FIXTURES_V2))["rows"]
  out = []
  for r in rows:
    if r["image"] is None:
      continue
    p = os.path.join(ROOT, f"out/fixtures_resized/g{IMG}/{r['id']}.png")
    assert os.path.exists(p), p
    out.append((r["id"], r["tier"], p))
  return out


def load01(path):
  im = Image.open(path).convert("RGB")
  assert im.size == (IMG, IMG), (path, im.size)
  return (torch.from_numpy(np.asarray(im)).float() / 255.0).unsqueeze(0)       # [1,IMG,IMG,3]


def main():
  t0 = time.monotonic()
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK, "model": MODEL,
         "transformers": transformers.__version__, "torch": torch.__version__}
  try:
    import importlib.metadata as md
    res["packages"] = {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert", "numpy")}
    model, info = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager", output_loading_info=True)
    model.eval()
    res["loading_info"] = {k: sorted(v) if isinstance(v, (list, set)) else str(v) for k, v in info.items()}
    res["missing_visual_keys"] = [k for k in info.get("missing_keys", []) if "visual" in k]
    assert not res["missing_visual_keys"], res["missing_visual_keys"]
    processor = AutoProcessor.from_pretrained(MODEL)
    visual = model.model.visual
    visual.config._attn_implementation = "eager"
    vcfg = visual.config
    assert vcfg.patch_size == PATCH and vcfg.spatial_merge_size == MERGE
    assert vcfg.temporal_patch_size == 2
    assert not list(getattr(vcfg, "deepstack_visual_indexes", [])), "expected NO deepstack"
    hidden = vcfg.hidden_size

    # Conv3d(temporal=2) -> Conv2d with the summed temporal kernel.
    pe = visual.patch_embed
    w3d, b = pe.proj.weight, pe.proj.bias                          # [1024,3,2,16,16]
    w2d = (w3d[:, :, 0] + w3d[:, :, 1]).contiguous()               # [1024,3,16,16]
    conv2d = torch.nn.Conv2d(3, hidden, PATCH, stride=PATCH, bias=True)
    conv2d.weight.data = w2d.detach().clone()
    conv2d.bias.data = b.detach().clone()

    # learned pos_embed resampled to the static grid, RASTER order, with 5.17.0's taps
    tap = np.load(INTERP)
    assert list(tap["grid"]) == [1, GRID, GRID] and int(tap["num_grid_per_side"]) == visual.num_grid_per_side
    res["interp_taps"] = {"file": os.path.relpath(INTERP, ROOT), "sha256": sha256_file(INTERP),
                          "transformers": str(tap["transformers"])}
    idx = torch.from_numpy(tap["indices"])                          # [N,4] int64
    wts = torch.from_numpy(tap["weights"])                          # [N,4] float32
    with torch.no_grad():
      pos_embed = (visual.pos_embed(idx) * wts[:, :, None]).sum(1)  # [N,1024] (5.17.0's own expression)

    # 2-D rope, RASTER order
    pid = torch.arange(N_PATCH)
    pos = torch.stack([pid // GRID, pid % GRID], dim=-1)   # [N,2]
    rpe = visual.rotary_pos_emb(pos)                       # [N, head_dim//2]
    emb = torch.cat((rpe, rpe), dim=-1)
    rcos, rsin = emb.cos(), emb.sin()

    def _rot_half(x):
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2:]
      return torch.cat((-x2, x1), dim=-1)

    # [1,N,C] throughout; rank >= 3 everywhere (Metal rank-2 elementwise trap).
    def _attn(self, hidden_states, cos, sin):        # hidden_states [1,N,C]
      L = hidden_states.shape[1]
      qkv = self.qkv(hidden_states)                     # [1,N,3C]
      qkv = qkv.reshape(L, 3, self.num_heads, -1).permute(1, 2, 0, 3)  # [3,H,N,d]
      q, k, v = qkv[0].unsqueeze(0), qkv[1].unsqueeze(0), qkv[2].unsqueeze(0)  # [1,H,N,d]
      q = q * cos + _rot_half(q) * sin
      k = k * cos + _rot_half(k) * sin
      attn = (q * self.scaling) @ k.transpose(-2, -1)  # [1,H,N,N]
      attn = attn.softmax(dim=-1)
      o = (attn @ v).permute(0, 2, 1, 3).reshape(1, L, -1)  # [1,N,C]
      return self.proj(o)

    # LN->FC fold barrier (int8 range protection; semantically neutral clamp).
    def _barrier(x):
      return torch.clamp(x, -65504.0, 65504.0) if BARRIER else x

    # fp16-safe LayerNorm: pre-scale by a calibrated power of two S (exact in
    # real arithmetic) so (x-m)^2 stays inside fp16 range on GPU delegates.
    def _ln_safe(x, weight, bias, eps, S):
      if not FP16SAFE or S <= 1.0:
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
      xs = x * (1.0 / S)
      d = xs - xs.mean(-1, keepdim=True)
      var = (d * d).mean(-1, keepdim=True)
      y = d * torch.rsqrt(var + eps / (S * S))
      return y * weight + bias

    def _calib_S(absmax):
      return float(2 ** max(0, int(np.ceil(np.log2(max(absmax, 1e-6) / 8.0)))))

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conv = conv2d
        self.blocks = visual.blocks
        self.register_buffer("pos_embed", pos_embed.detach().clone().reshape(1, N_PATCH, -1), persistent=False)
        self.register_buffer("cos", rcos.reshape(1, 1, N_PATCH, -1), persistent=False)
        self.register_buffer("sin", rsin.reshape(1, 1, N_PATCH, -1), persistent=False)

      def forward(self, images, calib=None):        # [1,IMG,IMG,3] in [0,1]
        x = (images.permute(0, 3, 1, 2) - MEAN) / STD
        p = self.conv(x)                            # [1,1024,GRID,GRID]
        # NHWC-order flatten (permute then reshape) -- NOT flatten(2).transpose:
        # the converter's TRANSPOSE->RESHAPE[C,N] form is miscomputed on Metal.
        h = p.permute(0, 2, 3, 1).reshape(1, N_PATCH, -1)  # [1,N,1024] raster
        h = h + self.pos_embed
        for i, blk in enumerate(self.blocks):
          if calib is not None:
            calib[f"blk{i}.norm1"] = max(calib.get(f"blk{i}.norm1", 0.0), float(h.abs().max()))
          n1 = _ln_safe(h, blk.norm1.weight, blk.norm1.bias, blk.norm1.eps, LN_S.get(f"blk{i}.norm1", 1.0))
          h = h + _attn(blk.attn, _barrier(n1), self.cos, self.sin)
          if calib is not None:
            calib[f"blk{i}.norm2"] = max(calib.get(f"blk{i}.norm2", 0.0), float(h.abs().max()))
          n2 = _ln_safe(h, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps, LN_S.get(f"blk{i}.norm2", 1.0))
          h = h + blk.mlp(_barrier(n2))
        if calib is not None:
          calib["final"] = max(calib.get("final", 0.0), float(h.abs().max()))
        return h                                     # [1,N,1024] raster order

    def merge2x2(f):                                 # [1,N,C] raster -> [1,N/4,4C] window order
      f = f.reshape(1, GRID, GRID, -1)
      m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                     f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
      return m.reshape(1, N_TOK, -1)

    class Adapter(torch.nn.Module):
      """merger: LN per patch -> 2x2 merge -> fc1 -> GELU -> fc2."""

      def __init__(self):
        super().__init__()
        self.merger = visual.merger

      def forward(self, feats):                     # [1,N,1024]
        S = LN_S.get("final", 1.0)
        n = _ln_safe(feats, self.merger.norm.weight, self.merger.norm.bias, self.merger.norm.eps, S)
        return self.merger.linear_fc2(self.merger.act_fn(self.merger.linear_fc1(merge2x2(n))))

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # calibrate the fp16-safe LN scales from zero on every fixture GxG PNG (v1 + v2) + noise
    res["stage"] = "calibrate"
    LN_S.clear()
    calib = {}
    fx = fixture_inputs()
    res["calibration_images"] = {"n": len(fx), "n_internal": sum(t != "public" for _, t, _ in fx),
                                 "rows": [rid for rid, _, _ in fx], "noise": "torch.manual_seed(1); torch.rand(1,IMG,IMG,3)"}
    with torch.no_grad():
      for _, _, fpath in fx:
        enc_m(load01(fpath), calib=calib)
      torch.manual_seed(1)
      enc_m(torch.rand(1, IMG, IMG, 3), calib=calib)
    for k, v in calib.items():
      LN_S[k] = _calib_S(v)
    res["ln_scales"] = {k: v for k, v in sorted(LN_S.items()) if v > 1.0}
    res["ln_absmax"] = dict(sorted(calib.items()))
    res["ln_absmax_max"] = max(calib.values())
    print("fp16-safe LN scales (>1):", res["ln_scales"], "| absmax max:",
          round(res["ln_absmax_max"], 1), flush=True)

    # sanity: the eager rewrite vs the 5.14.1 tower it was built from, on the first fixture image
    res["stage"] = "eager-check"
    rid0, _, p0 = fx[0]
    pil = Image.open(p0).convert("RGB")
    pp = processor.image_processor(images=[pil], return_tensors="pt")
    pv, pgrid = pp["pixel_values"], pp["image_grid_thw"]
    assert list(pgrid[0]) == [1, GRID, GRID], f"processor grid {pgrid} != static {GRID}"
    img01 = load01(p0)
    with torch.no_grad():
      ref = visual(pv, grid_thw=pgrid).pooler_output               # [N/4,2048]
      feat = enc_m(img01)
      emb_out = adp_m(feat).squeeze(0)                             # [N/4,2048]
    res["eager_vs_5141"] = {"row": rid0,
                            "corr": float(np.corrcoef(emb_out.flatten().numpy(), ref.flatten().numpy())[0, 1]),
                            "max_abs_diff": float((emb_out - ref).abs().max()), "ref_absmax": float(ref.abs().max())}
    print("eager adapter vs 5.14.1 tower:", res["eager_vs_5141"], flush=True)

    res["stage"] = "convert-encoder"
    enc_path = os.path.join(OUT, "vision_encoder.tflite")
    adp_path = os.path.join(OUT, "vision_adapter.tflite")
    litert_torch.convert(enc_m, (img01,)).export(enc_path)
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp_m, (feat,)).export(adp_path)

    res["stage"] = "tflite-check"
    e = tfl_run(enc_path, img01)
    a = tfl_run(adp_path, torch.from_numpy(e))
    got = a.astype("float64").reshape(-1)
    rf = ref.numpy().astype("float64").reshape(-1)
    res["enc_ops"] = op_hist(enc_path)
    res["adp_ops"] = op_hist(adp_path)
    res["tflite_vs_5141"] = {"row": rid0, "corr": float(np.corrcoef(got, rf)[0, 1]),
                             "max_abs_diff": float(np.max(np.abs(got - rf)))}
    res["files"] = {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
                    for p in (enc_path, adp_path)}
    print("tflite vs 5.14.1 tower:", res["tflite_vs_5141"])
    print("enc ops", res["enc_ops"], "\nadp ops", res["adp_ops"], flush=True)
    for k in ("enc_ops", "adp_ops"):
      assert not (res[k]["flex"] or res[k]["custom"] or res[k]["gather"]), (k, res[k])
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:400]
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(traceback.format_exc())
    print("ERROR", res["error_type"], res["error_head"])
  res["wall_seconds_contended"] = time.monotonic() - t0
  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k not in ("enc_ops", "adp_ops", "ln_absmax", "loading_info", "calibration_images")}))


if __name__ == "__main__":
  main()
