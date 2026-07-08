"""Export Qwen2-VL-2B's vision path as TWO tflites for the fast_vlm contract:

  VISION_ENCODER: image NHWC [1,IMG,IMG,3] in [0,1] -> patch features [1,N,1280]
  VISION_ADAPTER: features [1,N,1280]               -> soft tokens  [1,N/4,1536]

Qwen2-VL vision (probed): Conv3d patch-embed (temporal_patch_size=2), 2-D rope
over (row,col), FULL attention (no windowing — that's 2.5-VL), then a PatchMerger
(LN -> 2x2 group -> MLP) whose output is already the text hidden size (1536).

Static-single-image rewrite (same family as Ovis / PaddleOCR):
  - **Conv3d -> Conv2d fold.** For a single image the processor duplicates it into
    the 2 temporal frames, so Conv3d over 2 identical slices == Conv2d with the
    summed temporal kernel `w[:,:,0]+w[:,:,1]`. GPU-safe, no Conv3d.
  - Whole-image raster patch-conv, then a single **gather** reorders raster patches
    into the processor's merge-block order (4 consecutive patches = one 2x2 block,
    which the merger's `view(-1, 1280*4)` consumes). Provably identical; all ops <=4D.
  - Precomputed 2-D rope cos/sin (merge order) + explicit full attention (no FLEX).
  - Bakes OpenAI-CLIP normalization ((x-mean)/std, runtime feeds [0,1] NHWC).

IMG must be a multiple of 28 (patch14 x merge2). Default 672 -> 48x48 grid ->
2304 patches -> 576 soft tokens.

    IMG=672 .venv/bin/python qwen2vl_work/convert_qwen2vl_vision.py [out_dir]
"""
import json
import os
import sys
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "src_models/qwen2-vl-2b")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/qwen2vl-vision")
os.makedirs(OUT, exist_ok=True)

IMG = int(os.environ.get("IMG", "672"))
assert IMG % 28 == 0, "IMG must be a multiple of 28 (patch14 x merge2)"
GRID = IMG // 14              # patches per side
N_PATCH = GRID * GRID
MERGE = 2
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


def _quant_int8(src, dst):
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
  q.quantize().export_model(dst)
  return round(os.path.getsize(dst) / 1e6, 1)


def merge_perm():
  """raster index (h*GRID+w) -> processor merge-block order."""
  idx = []
  for hb in range(GRID // MERGE):
    for wb in range(GRID // MERGE):
      for mh in range(MERGE):
        for mw in range(MERGE):
          idx.append((hb * MERGE + mh) * GRID + (wb * MERGE + mw))
  return torch.tensor(idx, dtype=torch.long)


def main():
  res = {"ok": False, "stage": "load", "img": IMG, "grid": GRID, "n_tok": N_TOK}
  try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    visual = model.model.visual if hasattr(model.model, "visual") else model.visual
    visual.config._attn_implementation = "eager"
    perm = merge_perm()

    # fold Conv3d(temporal=2) -> Conv2d with summed temporal kernel
    w3d = visual.patch_embed.proj.weight            # [1280,3,2,14,14]
    w2d = (w3d[:, :, 0] + w3d[:, :, 1]).contiguous()  # [1280,3,14,14]
    conv2d = torch.nn.Conv2d(3, w2d.shape[0], 14, stride=14, bias=False)
    conv2d.weight.data = w2d

    # precompute 2-D rope in RASTER order (row=r//GRID, col=r%GRID). Keeping
    # patches in raster order through the (full-attention, permutation-equivariant)
    # encoder avoids a merge-order GATHER_ND (which the Pixel/ML-Drift GPU delegate
    # cannot compile); the 2x2 merge is instead done GPU-safe in the adapter with
    # strided slices + concat (the PaddleOCR/Ovis pattern).
    pid = torch.arange(N_PATCH)
    pos = torch.stack([pid // GRID, pid % GRID], dim=-1)   # [N,2] raster order
    rpe = visual.rotary_pos_emb(pos)                # [N, head_dim//2]
    emb = torch.cat((rpe, rpe), dim=-1)
    rcos, rsin = emb.cos(), emb.sin()

    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    def _attn(self, hidden_states, cos, sin):
      L = hidden_states.shape[0]
      q, k, v = self.qkv(hidden_states).reshape(
          L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
      q, k = apply_rotary_pos_emb_vision(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
      q, k = q.squeeze(0).transpose(0, 1), k.squeeze(0).transpose(0, 1)  # (heads,L,d)
      v = v.transpose(0, 1)
      attn = (q * self.scaling) @ k.transpose(-2, -1)
      attn = attn.softmax(dim=-1)
      o = (attn @ v).transpose(0, 1).reshape(L, -1)
      return self.proj(o)

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conv = conv2d
        self.blocks = visual.blocks
        self.register_buffer("cos", rcos, persistent=False)
        self.register_buffer("sin", rsin, persistent=False)

      def forward(self, images):                    # [1,IMG,IMG,3] in [0,1]
        x = (images.permute(0, 3, 1, 2) - MEAN) / STD
        p = self.conv(x)                            # [1,1280,GRID,GRID]
        h = p.flatten(2).squeeze(0).transpose(0, 1)  # [N,1280] raster (NO gather)
        for blk in self.blocks:
          h = h + _attn(blk.attn, blk.norm1(h), self.cos, self.sin)
          h = h + blk.mlp(blk.norm2(h))
        return h.unsqueeze(0)                        # [1,N,1280] raster order

    class Adapter(torch.nn.Module):
      """LN(raster) -> GPU-safe 2x2 strided-slice merge -> merger MLP -> [1,N/4,1536].
      Replaces the merger's `view(-1, 1280*4)` (which assumes a GATHER'd merge
      order) with 4 strided slices + concat over the raster grid (all <=4D)."""

      def __init__(self):
        super().__init__()
        self.ln_q = visual.merger.ln_q
        self.mlp = visual.merger.mlp

      def forward(self, feats):                     # [1,N,1280] raster
        f = self.ln_q(feats).reshape(1, GRID, GRID, -1)   # 4D
        m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                       f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(N_TOK, -1)                     # [N/4, 1280*4] merge order
        return self.mlp(m).unsqueeze(0)             # [1,N/4,1536]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # reference: the model's OWN visual path on the same IMGxIMG input
    res["stage"] = "reference"
    torch.manual_seed(0)
    img_u8 = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img_u8.numpy(), mode="RGB")
    pp = processor.image_processor(images=[pil], return_tensors="pt")
    pv, grid = pp["pixel_values"], pp["image_grid_thw"]
    assert list(grid[0]) == [1, GRID, GRID], f"processor grid {grid} != static {GRID}"
    img01 = (img_u8.float() / 255.0).unsqueeze(0)

    with torch.no_grad():
      ref = visual(pv, grid_thw=grid).pooler_output    # [N/4,1536]
      feat = enc_m(img01)
      emb_out = adp_m(feat).squeeze(0)                  # [N/4,1536]

    res["adp_eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), ref.flatten().numpy())[0, 1])
    res["adp_eager_maxdiff"] = float((emb_out - ref).abs().max())
    print("eager adapter corr", res["adp_eager_corr"], "maxdiff", res["adp_eager_maxdiff"])

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
    print("enc ops", res["enc_ops"], "\nadp ops", res["adp_ops"])

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
