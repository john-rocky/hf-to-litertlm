"""Export Qwen3.5's vision path (qwen3_5 ViT: the Qwen3-VL structure at 1024 dims,
NO deepstack) as TWO tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,IMG,IMG,3] in [0,1] -> patch features [1,N,1024]
  VISION_ADAPTER: features [1,N,1024]               -> soft tokens  [1,N/4,2048]

Static-single-image rewrite = northmv_work/convert_qwen35_vision.py's recipe
(measured there on the same graph family; transformers here is 5.14.1 so the
patch embed is a plain Conv3d and the pos-embed helper is
get_vision_bilinear_indices_and_weights returning [4,N]):
  - Conv3d(temporal 2) -> Conv2d with the summed temporal kernel (+ bias); the
    processor duplicates a still image into both frames, so this is exact.
  - Learned 48x48 pos_embed bilinearly resampled to the static grid, precomputed
    in RASTER order (spatial_merge_size=1 makes the helper's reorder an identity).
  - 2-D rope precomputed in raster order; explicit full attention (no varlen split).
  - Patches stay in RASTER order through the (permutation-equivariant) encoder;
    the 2x2 merge happens in the adapter with 4 strided slices + concat -> NO
    GATHER_ND (mobile GPU delegates cannot compile it).
  - Normalization (x-0.5)/0.5 and NHWC->NCHW baked in (runtime feeds [0,1] NHWC).
  - Every activation keeps a leading batch dim (rank >= 3): Metal computes rank-2
    elementwise x rank-2 const silently wrong.
  - LN->FC fold barrier clamp (int8 range protection) + fp16-safe LayerNorm with
    calibrated power-of-two pre-scales.

IMG must be a multiple of 32 (patch16 x merge2). Default 512 -> 32x32 grid ->
1024 patches -> 256 soft tokens.

    IMG=512 .venv-092/bin/python qwen35vl_work/convert_qwen35_vision.py [out_dir]
"""
import json
import os
import sys
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.vision_utils import get_vision_bilinear_indices_and_weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/qwen35vl-vision")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "512"))
PATCH = 16
MERGE = 2
assert IMG % (PATCH * MERGE) == 0, "IMG must be a multiple of 32 (patch16 x merge2)"
GRID = IMG // PATCH
N_PATCH = GRID * GRID
N_TOK = N_PATCH // (MERGE * MERGE)
MEAN, STD = 0.5, 0.5
BARRIER = os.environ.get("BARRIER", "1") == "1"
FP16SAFE = os.environ.get("FP16SAFE", "1") == "1"
LN_S = {}  # per-LN pre-scale for the fp16-safe LayerNorm (filled by calibration)
NBLOCKS = int(os.environ["NBLOCKS"]) if "NBLOCKS" in os.environ else None
FIXTURES = os.path.join(ROOT, "northmv_work", "fixtures")


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n": len(h), "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
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


def _quant_int8(src, dst):
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
  q.quantize().export_model(dst)
  return round(os.path.getsize(dst) / 1e6, 1)


def main():
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK}
  try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").eval()
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

    grid = torch.tensor([[1, GRID, GRID]], dtype=torch.long)
    # learned pos_embed resampled to the static grid, RASTER order
    # (spatial_merge_size=1 -> the helper's merge reorder is the identity)
    idx, wts = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, spatial_merge_size=1)
    pos_embed = (visual.pos_embed.weight[idx] * wts[:, :, None]).sum(0)  # [N,1024]

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
        for i, blk in enumerate(self.blocks if NBLOCKS is None else self.blocks[:NBLOCKS]):
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

    # calibrate the fp16-safe LN scales on real images + noise, in plain fp32
    res["stage"] = "calibrate"
    LN_S.clear()
    calib = {}
    fx = [os.path.join(FIXTURES, f) for f in ("cats_512.png", "kitchen1_512.png", "kitchen2_512.png")]
    with torch.no_grad():
      for fpath in [f for f in fx if os.path.exists(f)]:
        im = Image.open(fpath).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        enc_m((torch.from_numpy(np.asarray(im)).float() / 255.0).unsqueeze(0), calib=calib)
      torch.manual_seed(1)
      enc_m(torch.rand(1, IMG, IMG, 3), calib=calib)
    for k, v in calib.items():
      LN_S[k] = _calib_S(v)
    res["ln_scales"] = {k: v for k, v in sorted(LN_S.items()) if v > 1.0}
    res["ln_absmax_max"] = max(calib.values())
    print("fp16-safe LN scales (>1):", res["ln_scales"], "| absmax max:",
          round(res["ln_absmax_max"], 1), flush=True)

    # reference: the model's OWN visual path on the same IMGxIMG input
    res["stage"] = "reference"
    torch.manual_seed(0)
    img_u8 = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img_u8.numpy(), mode="RGB")
    pp = processor.image_processor(images=[pil], return_tensors="pt")
    pv, pgrid = pp["pixel_values"], pp["image_grid_thw"]
    assert list(pgrid[0]) == [1, GRID, GRID], f"processor grid {pgrid} != static {GRID}"
    img01 = (img_u8.float() / 255.0).unsqueeze(0)

    with torch.no_grad():
      vo = visual(pv, grid_thw=pgrid)
      ref = vo.pooler_output                            # [N/4,2048]
      feat = enc_m(img01)
      emb_out = adp_m(feat).squeeze(0)                  # [N/4,2048]

    res["adp_eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), ref.flatten().numpy())[0, 1])
    res["adp_eager_maxdiff"] = float((emb_out - ref).abs().max())
    res["ref_absmax"] = float(ref.abs().max())
    print("eager adapter corr", res["adp_eager_corr"], "maxdiff", res["adp_eager_maxdiff"],
          "ref absmax", res["ref_absmax"], flush=True)

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc_m, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    if NBLOCKS is not None:
      print("DEBUG NBLOCKS export done", NBLOCKS); return
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp_m, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    e = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    a = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(e))
    got = a.astype("float64").reshape(-1)
    rf = ref.numpy().astype("float64").reshape(-1)
    res["enc_ops"] = op_hist(os.path.join(OUT, "vision_encoder.tflite"))
    res["adp_ops"] = op_hist(os.path.join(OUT, "vision_adapter.tflite"))
    res["end2end_corr"] = float(np.corrcoef(got, rf)[0, 1])
    res["end2end_maxdiff"] = float(np.max(np.abs(got - rf)))
    res["enc_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_encoder.tflite")) / 1e6, 1)
    res["adp_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_adapter.tflite")) / 1e6, 1)
    print("end2end corr", res["end2end_corr"], "maxdiff", res["end2end_maxdiff"])
    print("enc ops", res["enc_ops"], "\nadp ops", res["adp_ops"], flush=True)

    res["stage"] = "quant-int8"
    res["enc_int8_mb"] = _quant_int8(
        os.path.join(OUT, "vision_encoder.tflite"), os.path.join(OUT, "vision_encoder_int8.tflite"))
    res["adp_int8_mb"] = _quant_int8(
        os.path.join(OUT, "vision_adapter.tflite"), os.path.join(OUT, "vision_adapter_int8.tflite"))
    e8 = tfl_run(os.path.join(OUT, "vision_encoder_int8.tflite"), img01)
    a8 = tfl_run(os.path.join(OUT, "vision_adapter_int8.tflite"), torch.from_numpy(e8))
    res["end2end_int8_corr"] = float(np.corrcoef(a8.astype("float64").reshape(-1), rf)[0, 1])
    res["end2end_int8_maxdiff"] = float(np.max(np.abs(a8.astype("float64").reshape(-1) - rf)))
    print("int8 end2end corr", res["end2end_int8_corr"], "maxdiff", res["end2end_int8_maxdiff"],
          "enc", res["enc_int8_mb"], "adp", res["adp_int8_mb"], "MB")

    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:400]
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(traceback.format_exc())
    print("ERROR", res["error_type"], res["error_head"])

  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k not in ("enc_ops", "adp_ops")}))


if __name__ == "__main__":
  main()
