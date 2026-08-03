"""Export microsoft/Mage-VL's vision path as TWO tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,IMG,IMG,3] in [0,1] -> patch features [1,N,1024]
  VISION_ADAPTER: features [1,N,1024]               -> soft tokens  [1,N/4,2560]

Static-single-image rewrite validated by magevl_work/precheck_magevl_vision.py
(random-init op-precheck: 0 FLEX/custom, corr 1.0). This script uses REAL weights:

  - vision tower loaded directly from the safetensors shards (model.visual.*),
    no full-VL instantiation (text side untouched, ~1.2 GB fp32 tower only)
  - bakes OpenAI-CLIP normalization ((x-mean)/std; runtime feeds [0,1] NHWC)
  - validates patch ordering + normalization AGAINST THE REAL PROCESSOR
    (Qwen2VLImageProcessor pixel_values == our raster reshape + block perm) and
    patch_positions against the repo's build_patch_positions
  - raster-order encoder (no merge-order GATHER_ND — mobile-GPU killer),
    constant 3D rope (4:6:6, t=0), full attention, strided-slice 2x2 merge
  - reference = vendored MageVLVisionPretrainedModel forward on processor output

IMG must be a multiple of 32 (patch16 x merge2). Default 448 (native config
image_size) -> 28x28 grid -> 784 patches -> 196 soft tokens.

    IMG=448 .venv-092/bin/python magevl_work/convert_magevl_vision.py [out_dir]
"""
import glob
import json
import os
import sys
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from vendor.configuration_mage_vl import MageVLVisionConfig  # noqa: E402
from vendor.modeling_mage_vl import (  # noqa: E402
    MageVLVisionPretrainedModel,
    apply_rotary_pos_emb,
)

MODEL = os.environ.get("MODEL", os.path.join(ROOT, "src_models/mage-vl"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/magevl-vision")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "448"))
assert IMG % 32 == 0, "IMG must be a multiple of 32 (patch16 x merge2)"
PATCH = 16
MERGE = 2
GRID = IMG // PATCH
N_PATCH = GRID * GRID
N_TOK = N_PATCH // (MERGE * MERGE)

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n_op_types": len(h),
          "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
          "custom": sorted(k for k in h if "CUSTOM" in k.upper())}


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


def merge_perm():
  """block-order position i -> raster index (processor 2x2 merge-block order)."""
  idx = []
  for hb in range(GRID // MERGE):
    for wb in range(GRID // MERGE):
      for mh in range(MERGE):
        for mw in range(MERGE):
          idx.append((hb * MERGE + mh) * GRID + (wb * MERGE + mw))
  return torch.tensor(idx, dtype=torch.long)


def load_vision_tower():
  from safetensors import safe_open
  with open(os.path.join(MODEL, "config.json")) as f:
    vcfg_dict = json.load(f)["vision_config"]
  vcfg_dict.pop("model_type", None)
  cfg = MageVLVisionConfig(**vcfg_dict)
  cfg._attn_implementation = "eager"
  model = MageVLVisionPretrainedModel(cfg).eval().float()

  sd = {}
  for shard in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
    with safe_open(shard, framework="pt") as f:
      for k in f.keys():
        if k.startswith("model.visual."):
          sd[k[len("model.visual."):]] = f.get_tensor(k).float()
  missing, unexpected = model.load_state_dict(sd, strict=False)
  missing = [m for m in missing if "inv_freq" not in m]  # non-persistent buffers
  assert not missing, f"missing vision keys: {missing[:5]}"
  assert not unexpected, f"unexpected vision keys: {unexpected[:5]}"
  for name in ("inv_freq_t", "inv_freq_h", "inv_freq_w"):
    assert float(getattr(model.video_rope, name).abs().min()) > 0, f"{name} zeroed"
  print(f"vision tower loaded: {len(sd)} tensors")
  return cfg, model


def main():
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK}
  try:
    torch.manual_seed(0)
    cfg, model = load_vision_tower()
    perm = merge_perm()
    pid = torch.arange(N_PATCH)
    pos_raster = torch.stack(
        [torch.zeros_like(pid), pid // GRID, pid % GRID], dim=-1)
    pos_block = pos_raster[perm]

    # --- validate ordering + normalization against the REAL processor ---
    res["stage"] = "processor-check"
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=True)
    img_u8 = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img_u8.numpy(), mode="RGB")
    pp = proc(images=[pil], return_tensors="pt")
    pv, grid_thw = pp["pixel_values"].float(), pp["image_grid_thw"]
    assert list(grid_thw[0]) == [1, GRID, GRID], f"processor grid {grid_thw} != static {GRID}"

    img01 = (img_u8.float() / 255.0).unsqueeze(0)          # [1,IMG,IMG,3]
    x_norm = (img01.permute(0, 3, 1, 2) - MEAN) / STD
    patches_raster = (x_norm.squeeze(0)
                      .reshape(3, GRID, PATCH, GRID, PATCH)
                      .permute(1, 3, 0, 2, 4)
                      .reshape(N_PATCH, 3, PATCH, PATCH))
    ours_block = patches_raster[perm].reshape(N_PATCH, -1)
    theirs = pv.reshape(N_PATCH, -1)
    res["processor_patch_maxdiff"] = float((ours_block - theirs).abs().max())
    assert res["processor_patch_maxdiff"] < 5e-3, \
        f"patch order/norm mismatch: {res['processor_patch_maxdiff']}"
    print("processor patch check maxdiff", res["processor_patch_maxdiff"])

    sys.path.insert(0, MODEL)
    from video_processing_mage_vl import build_patch_positions
    pp_ref = build_patch_positions(grid_thw, spatial_merge_size=MERGE)
    assert torch.equal(pp_ref, pos_block), "patch_positions != repo build_patch_positions"
    print("patch_positions check OK")

    # --- static modules (precheck-validated rewrite, real weights) ---
    with torch.no_grad():
      freqs = model.video_rope.forward_from_positions(pos_raster)
      freqs = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)

    conv = model.embeddings.patch_embedding
    ln_pre = model.layernorm_pre
    layers = model.encoder.layers
    scale = layers[0].self_attn.scale
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conv = conv
        self.ln_pre = ln_pre
        self.blocks = layers
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

      def forward(self, images):                    # [1,IMG,IMG,3] in [0,1]
        x = (images.permute(0, 3, 1, 2) - self.mean) / self.std
        p = self.conv(x)                            # [1,1024,G,G]
        h = p.flatten(2).transpose(1, 2)            # [1,N,1024] raster (NO gather)
        h = self.ln_pre(h)
        for blk in self.blocks:
          r = blk.layer_norm1(h)
          B, L, _ = r.shape
          q, k, v = (blk.self_attn.qkv(r)
                     .reshape(B, L, 3, n_heads, head_dim)
                     .permute(2, 0, 3, 1, 4).unbind(0))
          q, k = apply_rotary_pos_emb(q, k, self.freqs)
          attn = torch.matmul(q, k.transpose(2, 3)) * scale
          attn = attn.softmax(dim=-1)
          o = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
          h = h + blk.self_attn.proj(o)
          h = h + blk.mlp(blk.layer_norm2(h))
        return h                                     # [1,N,1024] raster order

    class Adapter(torch.nn.Module):
      """LN(raster) -> GPU-safe 2x2 strided-slice merge -> merger MLP."""

      def __init__(self):
        super().__init__()
        self.ln_q = model.merger.ln_q
        self.mlp = model.merger.mlp

      def forward(self, feats):                     # [1,N,1024] raster
        f = self.ln_q(feats).reshape(1, GRID, GRID, -1)
        m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                       f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(1, N_TOK, -1)
        return self.mlp(m)                           # [1,N/4,2560]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # --- reference: vendored model's OWN forward on the processor's output ---
    res["stage"] = "reference"
    with torch.no_grad():
      ref = model(hidden_state=pv.reshape(N_PATCH, 3, PATCH, PATCH),
                  grid_thw=grid_thw, patch_positions=pp_ref).last_hidden_state
      ref = ref.reshape(N_TOK, -1)
      feat = enc_m(img01)
      emb_out = adp_m(feat).reshape(N_TOK, -1)

    res["eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), ref.flatten().numpy())[0, 1])
    res["eager_maxdiff"] = float((emb_out - ref).abs().max())
    print("eager corr", res["eager_corr"], "maxdiff", res["eager_maxdiff"])

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc_m, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
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
    print("enc flex/custom", res["enc_ops"]["flex"], res["enc_ops"]["custom"])
    print("adp flex/custom", res["adp_ops"]["flex"], res["adp_ops"]["custom"])

    res["stage"] = "quant-int8"
    res["enc_int8_mb"] = _quant_int8(
        os.path.join(OUT, "vision_encoder.tflite"), os.path.join(OUT, "vision_encoder_int8.tflite"))
    res["adp_int8_mb"] = _quant_int8(
        os.path.join(OUT, "vision_adapter.tflite"), os.path.join(OUT, "vision_adapter_int8.tflite"))
    e8 = tfl_run(os.path.join(OUT, "vision_encoder_int8.tflite"), img01)
    a8 = tfl_run(os.path.join(OUT, "vision_adapter_int8.tflite"), torch.from_numpy(e8))
    res["end2end_int8_corr"] = float(np.corrcoef(a8.astype("float64").reshape(-1), rf)[0, 1])
    print("int8 end2end corr", res["end2end_int8_corr"],
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
