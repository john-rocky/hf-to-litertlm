"""Export North-Micro-Vision-Instruct's vision path (CohereCompass = Qwen3-VL-structure
ViT at SigLIP2-SO400M dims) as TWO tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,IMG,IMG,3] in [0,1] -> patch features [1,N,C]
  VISION_ADAPTER: features [1,N,C]                  -> soft tokens  [1,N/4,2048]

Static-single-image rewrite (the Qwen2-VL-2B ride's recipe, qwen2vl_work/convert_qwen2vl_vision.py):
  - Conv3d(temporal 2) -> Conv2d with the summed temporal kernel (+ bias); the processor
    duplicates a still image into both frames, so this is exact.
  - Learned 48x48 pos_embed bilinearly resampled (align_corners) to the static grid,
    precomputed in RASTER order (transformers.vision_utils, spatial_merge_size=1).
  - 2-D rope precomputed in raster order; explicit full attention (no FLEX/varlen).
  - Patches stay in RASTER order through the (permutation-equivariant) encoder; the
    2x2 merge happens in the adapter with 4 strided slices + concat -> NO GATHER_ND
    (the mobile GPU delegate cannot compile it; Qwen2-VL lesson).
  - Normalization (x-0.5)/0.5 and NHWC->NCHW baked in (runtime feeds [0,1] NHWC).

DEEPSTACK env decides what the single embedding carries (Phase 0 ablation decides):
  drop : soft tokens = merger(h_27)                              encoder out C=1152
  fold : soft tokens = merger(h_27) + sum_i deepstack_merger_i(h_{8,16,24})
         encoder out = concat(h_27, h_8, h_16, h_24) along features, C=4608

IMG must be a multiple of 32 (patch16 x merge2). Default 512 -> 32x32 grid ->
1024 patches -> 256 soft tokens .

    IMG=512 DEEPSTACK=drop .venv-vl0930-t515/bin/python northmv_work/convert_northmv_vision.py [out_dir]
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
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("MODEL", "CohereLabs/North-Micro-Vision-Instruct")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/northmv-vision")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "512"))
DEEPSTACK = os.environ.get("DEEPSTACK", "drop")
assert DEEPSTACK in ("drop", "fold")
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
NBLOCKS = int(os.environ["NBLOCKS"]) if "NBLOCKS" in os.environ else None  # debug: truncate encoder, export only it


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
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK,
         "deepstack": DEEPSTACK}
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
    ds_idx = list(vcfg.deepstack_visual_indexes)  # [8,16,24]
    hidden = vcfg.hidden_size

    # Conv3d(temporal=2) -> Conv2d with the summed temporal kernel. transformers >=5.16
    # may have fused the Conv3d into a Linear over the flattened [C][T][py][px] patch
    # (config.fusion_config.patch_embeddings) -- same weights, different view.
    pe = visual.patch_embed
    if hasattr(pe, "proj"):
      w3d, b = pe.proj.weight, pe.proj.bias                        # [1152,3,2,16,16]
    else:
      w3d = pe.linear_proj.weight.view(hidden, 3, 2, PATCH, PATCH)  # fused Linear view
      b = pe.linear_proj.bias
    w2d = (w3d[:, :, 0] + w3d[:, :, 1]).contiguous()  # [1152,3,16,16]
    conv2d = torch.nn.Conv2d(3, hidden, PATCH, stride=PATCH, bias=True)
    conv2d.weight.data = w2d.detach().clone()
    conv2d.bias.data = b.detach().clone()

    grid = torch.tensor([[1, GRID, GRID]], dtype=torch.long)
    # learned pos_embed resampled to the static grid, RASTER order
    idx, wts = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=1)
    pos_embed = (visual.pos_embed.weight[idx] * wts[:, :, None]).sum(1)  # [N,1152]

    # 2-D rope, RASTER order
    pid = torch.arange(N_PATCH)
    pos = torch.stack([pid // GRID, pid % GRID], dim=-1)   # [N,2]
    rpe = visual.rotary_pos_emb(pos)                       # [N, head_dim//2]
    emb = torch.cat((rpe, rpe), dim=-1)
    rcos, rsin = emb.cos(), emb.sin()

    # Every activation is kept >= 3-D with a leading batch of 1 (LLM-decoder shapes).
    # The Mac Metal delegate computes rank-2 elementwise ops with a rank-2 constant
    # WRONG (backend_diff: ADD([1024,1152], const) rel 1.2; the same add as [1,1024,1152]
    # is bit-exact), so the 2-D [N,C] formulation of the Qwen2-VL ride is not used here.
    def _attn(self, hidden_states, cos, sin):        # hidden_states [1,N,C]
      L = hidden_states.shape[1]
      qkv = self.qkv(hidden_states)                     # [1,N,3C]
      qkv = qkv.reshape(L, 3, self.num_heads, -1).permute(1, 2, 0, 3)  # [3,H,N,d]
      q, k, v = qkv[0].unsqueeze(0), qkv[1].unsqueeze(0), qkv[2].unsqueeze(0)  # [1,H,N,d]
      # rope: cos/sin [1,1,N,d]
      q = q * cos + _rot_half(q) * sin
      k = k * cos + _rot_half(k) * sin
      attn = (q * self.scaling) @ k.transpose(-2, -1)  # [1,H,N,N]
      attn = attn.softmax(dim=-1)
      o = (attn @ v).permute(0, 2, 1, 3).reshape(1, L, -1)  # [1,N,C]
      return self.proj(o)

    def _rot_half(x):
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2:]
      return torch.cat((-x2, x1), dim=-1)

    # The TFLite converter folds a LayerNorm's gamma MUL into a single-consumer FC that
    # follows it (norm1->qkv, norm2->fc1). Per-output-channel int8 of W*diag(gamma) then
    # spends its range on the few large-gamma input columns and destroys the small ones
    # (measured: int8 encoder corr 0.91-0.97 folded vs 0.998 unfolded). A clamp to the fp16
    # range between LN and FC is a fold barrier the converter cannot see through, and is
    # semantically neutral (activations are far inside +-65504) -- also a guard for fp16 GPUs.
    def _barrier(x):
      return torch.clamp(x, -65504.0, 65504.0) if BARRIER else x

    # fp16-safe LayerNorm. From block 9 on, the residual stream carries "massive
    # activations" (|x| ~ 1900-2100 on real images; register-style outlier dims). A plain
    # LN computes (x-mean)^2 -> 3.8e6, which overflows fp16 -> inf -> NaN, and every mobile
    # GPU delegate (Mac Metal / Pixel CL / iPhone Metal) runs the vision executor in fp16
    # -> "This image is a blank canvas" (measured). Pre-scaling by a per-LN constant S keeps
    # every intermediate inside fp16 range and is EXACTLY LN(x) in real arithmetic:
    #   (x-m)/sqrt(var+eps) == (x/S - m/S) / sqrt(var/S^2 + eps/S^2)
    # S is a power of two per LN chosen from a calibration pass (absmax/8, min 1).
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
        p = self.conv(x)                            # [1,1152,GRID,GRID]
        # NHWC-order flatten (permute then reshape). NOT flatten(2).transpose(0,1): the
        # converter turns that into TRANSPOSE->RESHAPE[C,N]->ADD(pos^T)->TRANSPOSE and the
        # Mac Metal delegate computes it WRONG (backend_diff rel 1.2 vs bit-exact here).
        h = p.permute(0, 2, 3, 1).reshape(1, N_PATCH, -1)  # [1,N,1152] raster (NO gather)
        h = h + self.pos_embed
        taps = []
        for i, blk in enumerate(self.blocks if NBLOCKS is None else self.blocks[:NBLOCKS]):
          if calib is not None:
            calib[f"blk{i}.norm1"] = max(calib.get(f"blk{i}.norm1", 0.0), float(h.abs().max()))
          n1 = _ln_safe(h, blk.norm1.weight, blk.norm1.bias, blk.norm1.eps, LN_S.get(f"blk{i}.norm1", 1.0))
          h = h + _attn(blk.attn, _barrier(n1), self.cos, self.sin)
          if calib is not None:
            calib[f"blk{i}.norm2"] = max(calib.get(f"blk{i}.norm2", 0.0), float(h.abs().max()))
          n2 = _ln_safe(h, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps, LN_S.get(f"blk{i}.norm2", 1.0))
          h = h + blk.mlp(_barrier(n2))
          if DEEPSTACK == "fold" and i in ds_idx:
            taps.append(h)
        if calib is not None:
          calib["final"] = max(calib.get("final", 0.0), float(h.abs().max()))
        if DEEPSTACK == "fold":
          h = torch.cat([h] + taps, dim=-1)          # [1, N, 4*1152]
        return h                                     # [1,N,C] raster order

    def merge2x2(f):                                 # [1,N,C] raster -> [1,N/4,4C] block order
      f = f.reshape(1, GRID, GRID, -1)
      m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                     f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
      return m.reshape(1, N_TOK, -1)

    class Adapter(torch.nn.Module):
      """merger (LN per patch -> 2x2 merge -> fc1 -> GELU -> fc2), optionally plus the
      three deepstack mergers (2x2 merge -> LN(4C) -> fc1 -> GELU -> fc2)."""

      def __init__(self):
        super().__init__()
        self.merger = visual.merger
        self.ds = visual.deepstack_merger_list

      def forward(self, feats):                     # [1,N,C]
        h27 = feats[:, :, :hidden]
        S = LN_S.get("final", 1.0)
        n = _ln_safe(h27, self.merger.norm.weight, self.merger.norm.bias, self.merger.norm.eps, S)
        out = self.merger.linear_fc2(self.merger.act_fn(self.merger.linear_fc1(merge2x2(n))))
        if DEEPSTACK == "fold":
          for j, m in enumerate(self.ds):
            hj = feats[:, :, (j + 1) * hidden:(j + 2) * hidden]
            nj = _ln_safe(merge2x2(hj), m.norm.weight, m.norm.bias, m.norm.eps, S)
            out = out + m.linear_fc2(m.act_fn(m.linear_fc1(nj)))
        return out                                  # [1,N/4,2048]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # calibrate the fp16-safe LN scales on real images (fixtures) + noise, in plain fp32
    res["stage"] = "calibrate"
    LN_S.clear()
    calib = {}
    fx = [os.path.join(HERE, "fixtures", f) for f in ("cats_512.png", "kitchen1_512.png", "kitchen2_512.png")]
    with torch.no_grad():
      for fpath in [f for f in fx if os.path.exists(f)]:
        im = Image.open(fpath).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        enc_m((torch.from_numpy(np.asarray(im)).float() / 255.0).unsqueeze(0), calib=calib)
      torch.manual_seed(1)
      enc_m(torch.rand(1, IMG, IMG, 3), calib=calib)
    for k, v in calib.items():
      LN_S[k] = _calib_S(v)
    res["ln_scales"] = {k: LN_S[k] for k in sorted(LN_S)}
    print("fp16-safe LN scales:", res["ln_scales"], flush=True)

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
      if DEEPSTACK == "fold":
        ref = ref + sum(vo.deepstack_features)
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
