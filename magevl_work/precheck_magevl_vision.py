"""PRECHECK: does microsoft/Mage-VL's vision encoder (mage_vl_vision) export via litert-torch?

Op-coverage precheck with RANDOM-INIT weights (no checkpoint download): instantiates
MageVLVisionPretrainedModel from the vendored remote code and tests the static
single-image rewrite end to end against the model's own forward. Weights don't
matter for exportability; using one shared random-init instance makes the
raster-vs-block-order equivalence check exact.

Static-single-image rewrite (Qwen2-VL-2B family pattern, qwen2vl_work/convert_qwen2vl_vision.py):
  - temporal_patch_size=1 -> patch_embedding IS a stride-16 Conv2d; run it on the
    whole image in RASTER order (no gather; merge-order GATHER_ND kills mobile GPU).
  - 3-D rope (4:6:6 t:h:w split, t=0 for a single image) precomputed from raster
    patch positions via the model's own forward_from_positions -> constant buffer.
  - Single image => one cu_seqlens chunk == FULL attention; replicate the eager
    matmul-softmax path explicitly (interleaved rotate_half via vendored
    apply_rotary_pos_emb).
  - Adapter: ln_q -> GPU-safe 2x2 strided-slice merge (raster -> processor
    merge-block order) -> merger MLP -> [1, N/4, 2560].
  - Reference: vendored model.forward with processor-style block-order patches,
    grid_thw=[[1,G,G]], block-order patch_positions.

IMG must be a multiple of 32 (patch16 x merge2). Default 448 (native) -> 28x28
grid -> 784 patches -> 196 soft tokens.

    IMG=448 .venv-092/bin/python magevl_work/precheck_magevl_vision.py [out_dir]
"""
import json
import os
import sys
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from vendor.configuration_mage_vl import MageVLVisionConfig  # noqa: E402
from vendor.modeling_mage_vl import (  # noqa: E402
    MageVLVisionPretrainedModel,
    apply_rotary_pos_emb,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/magevl-precheck")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "448"))
assert IMG % 32 == 0, "IMG must be a multiple of 32 (patch16 x merge2)"
PATCH = 16
MERGE = 2
GRID = IMG // PATCH
N_PATCH = GRID * GRID
N_TOK = N_PATCH // (MERGE * MERGE)


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n_op_types": len(h), "hist": h,
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


def main():
  res = {"ok": False, "stage": "init", "img": IMG, "grid": GRID, "n_tok": N_TOK}
  try:
    torch.manual_seed(0)
    with open(os.path.join(HERE, "vendor/config.json")) as f:
      vcfg_dict = json.load(f)["vision_config"]
    vcfg_dict.pop("model_type", None)
    cfg = MageVLVisionConfig(**vcfg_dict)
    cfg._attn_implementation = "eager"
    model = MageVLVisionPretrainedModel(cfg).eval().float()

    # rope sanity (tf5.x meta-load trap: inv_freq buffers must be non-zero)
    for name in ("inv_freq_t", "inv_freq_h", "inv_freq_w"):
      buf = getattr(model.video_rope, name)
      assert float(buf.abs().min()) > 0, f"{name} is zeroed"

    perm = merge_perm()
    pid = torch.arange(N_PATCH)
    pos_raster = torch.stack(
        [torch.zeros_like(pid), pid // GRID, pid % GRID], dim=-1)   # [N,3] t=0
    pos_block = pos_raster[perm]

    # precompute rope freqs in RASTER order as a constant
    with torch.no_grad():
      freqs = model.video_rope.forward_from_positions(pos_raster)   # [N, half]
      freqs = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)        # [1,N,head_dim]

    conv = model.embeddings.patch_embedding                          # Conv2d 16/16
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

      def forward(self, images):                    # [1,IMG,IMG,3] normalized
        x = images.permute(0, 3, 1, 2)
        p = self.conv(x)                            # [1,1024,G,G]
        h = p.flatten(2).transpose(1, 2)            # [1,N,1024] raster (NO gather)
        h = self.ln_pre(h)
        for blk in self.blocks:
          r = blk.layer_norm1(h)
          B, L, _ = r.shape
          q, k, v = (blk.self_attn.qkv(r)
                     .reshape(B, L, 3, n_heads, head_dim)
                     .permute(2, 0, 3, 1, 4).unbind(0))             # (B,H,L,D)
          q, k = apply_rotary_pos_emb(q, k, self.freqs)
          attn = torch.matmul(q, k.transpose(2, 3)) * scale
          attn = attn.softmax(dim=-1)
          o = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
          h = h + blk.self_attn.proj(o)
          h = h + blk.mlp(blk.layer_norm2(h))
        return h                                     # [1,N,1024] raster order

    class Adapter(torch.nn.Module):
      """LN(raster) -> GPU-safe 2x2 strided-slice merge -> merger MLP.
      Replaces the merger's `view(-1, 1024*4)` (which assumes block-order input)
      with 4 strided slices + concat over the raster grid (all <=4D)."""

      def __init__(self):
        super().__init__()
        self.ln_q = model.merger.ln_q
        self.mlp = model.merger.mlp

      def forward(self, feats):                     # [1,N,1024] raster
        f = self.ln_q(feats).reshape(1, GRID, GRID, -1)
        m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                       f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(1, N_TOK, -1)                  # [1,N/4,4096] block order
        return self.mlp(m)                           # [1,N/4,2560]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # reference: the vendored model's OWN forward on block-order patches
    res["stage"] = "reference"
    img = torch.randn(1, IMG, IMG, 3)               # stands in for normalized pixels
    x_chw = img.permute(0, 3, 1, 2)                 # [1,3,IMG,IMG]
    patches_raster = (x_chw
                      .reshape(3, GRID, PATCH, GRID, PATCH)
                      .permute(1, 3, 0, 2, 4)
                      .reshape(N_PATCH, 3, PATCH, PATCH))
    patches_block = patches_raster[perm]
    grid_thw = torch.tensor([[1, GRID, GRID]])

    with torch.no_grad():
      ref = model(hidden_state=patches_block, grid_thw=grid_thw,
                  patch_positions=pos_block).last_hidden_state
      ref = ref.reshape(N_TOK, -1)                   # [N/4,2560]
      feat = enc_m(img)
      emb_out = adp_m(feat).reshape(N_TOK, -1)

    res["eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), ref.flatten().numpy())[0, 1])
    res["eager_maxdiff"] = float((emb_out - ref).abs().max())
    print("eager corr", res["eager_corr"], "maxdiff", res["eager_maxdiff"])

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc_m, (img,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp_m, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    e = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img)
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
    e8 = tfl_run(os.path.join(OUT, "vision_encoder_int8.tflite"), img)
    a8 = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(e8))
    res["end2end_int8_corr"] = float(np.corrcoef(a8.astype("float64").reshape(-1), rf)[0, 1])
    print("int8 end2end corr", res["end2end_int8_corr"], "enc", res["enc_int8_mb"], "MB")

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
