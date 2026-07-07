"""Export PaddleOCR-VL-1.6's vision path as TWO tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,IMG,IMG,3] in [0,1] -> features [1,N,1152]
  VISION_ADAPTER: features [1,N,1152]               -> embeddings [1,N/4,1024]

Structure (probed from the 1.6 remote code):
  - NaViT-style dynamic-res SigLIP (so400m dims: hidden 1152, 27L, patch 14),
    full attention (the LM calls the tower with window_size=-1), 2D rope
    (SigLIPRotaryEmbedding on h/w ids), bilinear-interpolated learned pos-embed.
  - Processor packs patches in PURE RASTER order (no Qwen2-VL merge interleave),
    so patchify == Conv2d(k14,s14) on the whole NCHW image + raster flatten. No
    gather needed (unlike Ovis).
  - Projector (mlp_AR): LN(1152) -> 2x2 spatial merge -> Linear(4608,4608) ->
    GELU -> Linear(4608,1024). Merge is done GPU-safe with 4 strided slices +
    concat (all tensors <=4D) instead of the literal 6D rearrange.
  - Normalization mean/std 0.5 baked in: runtime feeds [0,1], graph does 2x-1.

IMG must be a multiple of 28 (patch 14 x merge 2). Default 560 -> grid 40x40 ->
1600 patches -> 400 soft tokens (attention 16x1600^2 fits mobile).

    .venv/bin/python paddleocr_work/convert_paddleocr_vision.py [out_dir]
    IMG=728 .venv/bin/python paddleocr_work/convert_paddleocr_vision.py out/x
"""
import json
import os
import sys
import traceback

import litert_torch  # noqa: F401  (must import before transformers submodules)
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pocr_compat import load_pocr  # noqa: E402
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "src_models/paddleocr-vl-1.6")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/paddleocr-vision")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "560"))
assert IMG % 28 == 0, "IMG must be a multiple of 28 (patch14 x merge2)"
GRID = IMG // 14
N_PATCH = GRID * GRID
N_TOK = N_PATCH // 4


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n": len(h), "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
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


def _quantize_int8(src, dst):
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
  q.quantize().export_model(dst)
  return round(os.path.getsize(dst) / 1e6, 1)


def main():
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK}
  try:
    model = load_pocr(MODEL, dtype=torch.float32, attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vt = model.visual.vision_model      # PaddleOCRVisionTransformer
    emb = vt.embeddings
    enc = vt.encoder
    projector = model.mlp_AR
    # PaddleOCRAttention reads self.config._attn_implementation
    model.visual.config._attn_implementation = "eager"

    # transformers 5.12 meta-device init leaves the NON-PERSISTENT inv_freq
    # buffer of the remote SigLIPRotaryEmbedding ZEROED (its class has no
    # original_inv_freq so _init_weights skips it) -> rope becomes identity and
    # the whole vision tower silently loses positional information. Recompute.
    enc.rotary_pos_emb.rope_init()
    assert float(enc.rotary_pos_emb.inv_freq.abs().sum()) > 1.0, \
        "vision rotary inv_freq still zero"

    res["stage"] = "precompute"
    with torch.no_grad():
      # learned pos-embed bilinear-interpolated to the static grid (raster order)
      pe = emb.interpolate_pos_encoding(
          torch.zeros(1, 1, emb.embed_dim), GRID, GRID, is_after_patchify=True)
      # 2D rope over (h,w) ids, exactly as the encoder builds it for use_rope=True
      pid = torch.arange(N_PATCH)
      pids = torch.stack([pid // GRID, pid % GRID], dim=-1)
      freqs = enc.rotary_pos_emb(GRID)               # [GRID, head_dim//4]
      rope = freqs[pids].flatten(1).repeat(1, 2)     # [N, head_dim]
      rope_cos, rope_sin = rope.cos(), rope.sin()

    class Encoder(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.patch = emb.patch_embedding             # Conv2d(3,1152,14,stride14)
        self.layers = enc.layers
        self.post_ln = vt.post_layernorm
        self.register_buffer("pe", pe, persistent=False)
        self.register_buffer("cos", rope_cos, persistent=False)
        self.register_buffer("sin", rope_sin, persistent=False)
        self.register_buffer("cu", torch.tensor([0, N_PATCH], dtype=torch.int32),
                             persistent=False)

      def forward(self, images):                     # [1,IMG,IMG,3] in [0,1]
        x = images * 2.0 - 1.0                       # (x-0.5)/0.5
        x = x.permute(0, 3, 1, 2)                    # [1,3,IMG,IMG]
        p = self.patch(x)                            # [1,1152,GRID,GRID]
        h = p.flatten(2).transpose(1, 2)             # [1,N,1152] raster
        h = h + self.pe
        for layer in self.layers:
          h = layer(h, None, output_attentions=False, cu_seqlens=self.cu,
                    rope_emb=(self.cos, self.sin))[0]
        return self.post_ln(h)                       # [1,N,1152]

    class Adapter(torch.nn.Module):
      """LN -> GPU-safe 2x2 merge (4 strided slices, <=4D) -> MLP -> [1,N/4,1024]."""

      def __init__(self):
        super().__init__()
        self.pre_norm = projector.pre_norm
        self.linear_1 = projector.linear_1
        self.act = projector.act
        self.linear_2 = projector.linear_2

      def forward(self, features):                   # [1,N,1152]
        f = self.pre_norm(features)
        f = f.reshape(1, GRID, GRID, -1)             # 4D
        m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                       f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(1, N_TOK, -1)                  # [1,N/4,4608] == (p1 p2 d)
        h = self.linear_1(m)
        h = self.act(h)
        return self.linear_2(h)                      # [1,N/4,1024]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # --- reference: the model's OWN vision path on the same IMGxIMG input ---
    res["stage"] = "reference"
    torch.manual_seed(0)
    img_u8 = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img_u8.numpy(), mode="RGB")
    ip = processor.image_processor
    out_pp = ip(images=[pil], return_tensors="pt")
    pv, grid_thw = out_pp["pixel_values"], out_pp["image_grid_thw"]
    assert list(grid_thw[0]) == [1, GRID, GRID], f"processor grid {grid_thw} != static"
    img01 = (img_u8.float() / 255.0).unsqueeze(0)    # [1,IMG,IMG,3]

    with torch.no_grad():
      vision_outputs = model.visual(
          pixel_values=pv.unsqueeze(0),
          image_grid_thw=[(1, GRID, GRID)],
          position_ids=torch.arange(N_PATCH) % N_PATCH,
          vision_return_embed_list=True,
          interpolate_pos_encoding=True,
          sample_indices=torch.zeros(N_PATCH, dtype=torch.int64),
          cu_seqlens=torch.tensor([0, N_PATCH], dtype=torch.int32),
          return_pooler_output=False,
          use_rope=True,
          window_size=-1,
      )
      feat_ref_list = vision_outputs.last_hidden_state
      emb_ref = torch.cat(projector(feat_ref_list, [(1, GRID, GRID)]), dim=0)

      res["stage"] = "eager"
      feat = enc_m(img01)                            # [1,N,1152]
      emb_out = adp_m(feat).squeeze(0)               # [N/4,1024]

    feat_ref = feat_ref_list[0] if isinstance(feat_ref_list, (list, tuple)) \
        else feat_ref_list.squeeze(0)
    res["enc_eager_maxdiff"] = float((feat.squeeze(0) - feat_ref).abs().max())
    res["enc_eager_corr"] = float(np.corrcoef(
        feat.flatten().numpy(), feat_ref.flatten().numpy())[0, 1])
    res["adp_eager_maxdiff"] = float((emb_out - emb_ref).abs().max())
    res["adp_eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), emb_ref.flatten().numpy())[0, 1])
    print("eager enc corr", res["enc_eager_corr"], "maxdiff", res["enc_eager_maxdiff"])
    print("eager adp corr", res["adp_eager_corr"], "maxdiff", res["adp_eager_maxdiff"])

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc_m, (img01,)).export(
        os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp_m, (feat,)).export(
        os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    enc_tfl = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    adp_tfl = tfl_run(os.path.join(OUT, "vision_adapter.tflite"),
                      torch.from_numpy(enc_tfl))
    ref = emb_ref.numpy().astype("float64").reshape(-1)
    got = adp_tfl.astype("float64").reshape(-1)
    res["enc_ops"] = op_hist(os.path.join(OUT, "vision_encoder.tflite"))
    res["adp_ops"] = op_hist(os.path.join(OUT, "vision_adapter.tflite"))
    res["end2end_corr"] = float(np.corrcoef(got, ref)[0, 1])
    res["end2end_maxdiff"] = float(np.max(np.abs(got - ref)))
    res["enc_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_encoder.tflite")) / 1e6, 1)
    res["adp_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_adapter.tflite")) / 1e6, 1)
    print("end2end corr", res["end2end_corr"], "maxdiff", res["end2end_maxdiff"])
    print("enc ops", res["enc_ops"], "\nadp ops", res["adp_ops"])

    res["stage"] = "quantize-int8"
    res["enc_int8_mb"] = _quantize_int8(
        os.path.join(OUT, "vision_encoder.tflite"),
        os.path.join(OUT, "vision_encoder_int8.tflite"))
    res["adp_int8_mb"] = _quantize_int8(
        os.path.join(OUT, "vision_adapter.tflite"),
        os.path.join(OUT, "vision_adapter_int8.tflite"))
    e8 = tfl_run(os.path.join(OUT, "vision_encoder_int8.tflite"), img01)
    a8 = tfl_run(os.path.join(OUT, "vision_adapter_int8.tflite"), torch.from_numpy(e8))
    got8 = a8.astype("float64").reshape(-1)
    res["end2end_int8_corr"] = float(np.corrcoef(got8, ref)[0, 1])
    print("int8 end2end corr", res["end2end_int8_corr"],
          "enc", res["enc_int8_mb"], "MB adp", res["adp_int8_mb"], "MB")

    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    tb = traceback.format_exc()
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:500]
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(tb)
    print("ERROR", res["error_type"], res["error_head"])

  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k not in ("enc_ops", "adp_ops")}))


if __name__ == "__main__":
  main()
