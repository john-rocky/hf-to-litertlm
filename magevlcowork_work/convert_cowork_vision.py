"""Export microsoft/Mage-VL-Cowork's vision path at a non-square canvas (default 1920 x 1088) as two tflites for the
fast_vlm contract:

  VISION_ENCODER: image NHWC [1,H,W,3] in [0,1] -> patch features [1,N,1024]      (N = H/16 * W/16)
  VISION_ADAPTER: features [1,N,1024]           -> soft tokens  [1,N/4,2560]

  * IMG_H / IMG_W (default 1088 / 1920 = the 68 x 120 grid, 2,040 soft tokens): the grid is GH x GW everywhere
    (patch reshape, merge permutation, raster positions, 2x2 strided-slice merge);
  * query-chunked attention: the 8,160 queries of each layer are split into chunks of QCHUNK rows; every chunk sees
    all 8,160 keys, so each softmax row is the same row as in one call (the arithmetic is unchanged). QCHUNK default =
    the largest divisor of N whose score tensor [1, 16, Q, N] fp32 is <= 512 MiB (N = 8,160 -> Q = 1,020, 8 chunks,
    532,684,800 B = 508.0 MiB); one call would make a 16 x 8,160^2 x 4 B = 4.26 GB score, over a phone GPU's 1 GiB
    single-buffer limit. The shipped file uses QCHUNK=255 (32 chunks, 133,171,200 B = 127.0 MiB per score chunk), a
    smaller peak for the attention. q is scaled by head_dim^-0.5 before the matmul (the vendor's softmax_scale), so the
    large score tensor gets no extra MUL;
  * every tensor is at rank <= 4: the LiteRT-LM 0.19.0 GPU vision executor refuses rank-5 tensors ("SLICE: Slicing is
    unsupported for input with rank > 4", "TRANSPOSE: Permutation for transpose is invalid", then "Failed to modify
    graph with delegate"), and the vendor's qkv split (reshape [B,L,3,heads,dim] -> permute(2,0,3,1,4) -> unbind) and
    its interleaved rotate_half (a stack) are rank 5. Here, with the same arithmetic:
      - q / k / v = three last-axis slices of the qkv output (rank 3; the vendor's layout is (3, heads, head_dim) on
        that axis), each reshaped to [B,L,heads,dim] and transposed (0,2,1,3);
      - rotate_half = the same interleave on [B*heads, L, dim/2, 2] (rank 4): out[2i] = -x[2i+1], out[2i+1] = x[2i]
        (checked equal to the vendor's on a random tensor); cos / sin of the constant rope are constants [1,1,N,dim]
        (the vendor's freqs.cos().unsqueeze(1));
      - the graph census counts the ops that touch a tensor of rank > 4 (must be 0);
  * the tower is loaded straight from the shards (`visual.*`, no full-VL instantiation); OpenAI-CLIP normalization is
    baked in (the runtime feeds [0,1] NHWC); raster-order encoder (no GATHER_ND); constant 3D rope (4:6:6, t = 0); the
    vendor's apply_rotary_pos_emb arithmetic (its interleaved rotate_half); 2x2 strided-slice merge in the adapter;
  * reference = the repository's LlavaOnevision2VisionPretrainedModel.forward on the processor's pixel_values with
    grid_thw and raster patch_positions (the vendor forward turns them into its 2x2-block rope itself), on the --example
    picture plus one seeded noise picture; processor patches vs the static graph's patches are compared per picture;
    the vendor reference outputs and the inputs are saved (OUT/ref/*.npy);
  * the op histogram lists every op (STABLEHLO_COMPOSITE counted next to FLEX / custom) and the largest non-constant
    tensor of each graph (bytes, from the flatbuffer: a tensor without buffer data);
  * no quantization here (fp32 weights).

--example PICTURE: a screenshot. The script applies its EXIF orientation, converts it to RGB and resizes it to
IMG_W x IMG_H (bicubic, no padding), as cowork_step.py does. It is the tracing input of the encoder (its encoder output
is the adapter's) and the first parity picture. Only its shape and dtype reach the converted graphs: litert_torch.convert
lowers a traced input as a graph input, and the graphs' constants come from the modules' parameters and buffers (the
rope tables and the normalization constants are computed from positions, not from pixels).

    IMG_H=1088 IMG_W=1920 QCHUNK=255 TFL_THREADS=4 MODEL=<checkpoint> \\
      python magevlcowork_work/convert_cowork_vision.py OUT --example PICTURE
Writes OUT/vision_encoder.tflite, OUT/vision_adapter.tflite, OUT/result.json (ok, the census, the parity numbers),
OUT/ref/ and OUT/vendor/ (the repository's configuration / modeling files, imported from there).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import litert_torch  # noqa: F401  import before transformers submodules
import numpy as np
import torch
from PIL import Image

_ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
_ap.add_argument("out", help="output directory")
_ap.add_argument("--example", required=True,
                 help="a screenshot: the tracing input and the first parity picture (resized to IMG_W x IMG_H)")
_A = _ap.parse_args()
if not os.environ.get("MODEL"):
  sys.exit("set MODEL to the microsoft/Mage-VL-Cowork checkpoint directory")
MODEL = os.path.abspath(os.environ["MODEL"])
OUT = os.path.abspath(_A.out)
EXAMPLE = os.path.abspath(_A.example)
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "ref"), exist_ok=True)

# The repo's modeling code imports its config relatively: copy both files as a package (bytes checked).
VENDOR = os.path.join(OUT, "vendor", "cowork_vendor")
os.makedirs(VENDOR, exist_ok=True)
open(os.path.join(VENDOR, "__init__.py"), "w").close()
VENDOR_SHA = {}
for _f in ("configuration_llava_onevision2.py", "modeling_llava_onevision2.py"):
  shutil.copyfile(os.path.join(MODEL, _f), os.path.join(VENDOR, _f))
  _a = hashlib.sha256(open(os.path.join(MODEL, _f), "rb").read()).hexdigest()
  _b = hashlib.sha256(open(os.path.join(VENDOR, _f), "rb").read()).hexdigest()
  assert _a == _b
  VENDOR_SHA[_f] = _a
sys.path.insert(0, os.path.dirname(VENDOR))

from cowork_vendor.configuration_llava_onevision2 import LlavaOnevision2VisionConfig  # noqa: E402
from cowork_vendor.modeling_llava_onevision2 import (  # noqa: E402
    LlavaOnevision2VisionPretrainedModel,
    apply_rotary_pos_emb,
)

IMG_H = int(os.environ.get("IMG_H", "1088"))
IMG_W = int(os.environ.get("IMG_W", "1920"))
assert IMG_H % 32 == 0 and IMG_W % 32 == 0, "IMG_H / IMG_W must be multiples of 32 (patch16 x merge2)"
PATCH = 16
MERGE = 2
GH, GW = IMG_H // PATCH, IMG_W // PATCH
N_PATCH = GH * GW
N_TOK = N_PATCH // (MERGE * MERGE)
SCORE_CAP = 512 * 2**20   # bytes of one fp32 score chunk [1, heads, Q, N]
THREADS = int(os.environ.get("TFL_THREADS", "4"))

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def auto_qchunk(n_heads):
  best = None
  for q in range(1, N_PATCH + 1):
    if N_PATCH % q == 0 and n_heads * q * N_PATCH * 4 <= SCORE_CAP:
      best = q
  return best


def rank_census(p):
  """Ops whose input or output tensor has rank > 4 (the GPU delegate's limit), by op name, plus a few examples."""
  from ai_edge_litert import schema_py_generated as S
  buf = open(p, "rb").read()
  m = S.Model.GetRootAsModel(buf, 0)
  names = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
  hist, ex, max_rank = {}, [], 0
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      oc = m.OperatorCodes(op.OpcodeIndex())
      code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
      ranks = []
      for idx in [op.Inputs(j) for j in range(op.InputsLength())] + [op.Outputs(j) for j in range(op.OutputsLength())]:
        if idx >= 0:
          ranks.append(sg.Tensors(idx).ShapeLength())
      r = max(ranks) if ranks else 0
      max_rank = max(max_rank, r)
      if r > 4:
        nm = names.get(code, str(code))
        hist[nm] = hist.get(nm, 0) + 1
        if len(ex) < 5:
          ex.append({"op": nm, "ranks": ranks})
  return {"rank_gt4_ops": sum(hist.values()), "by_op": hist, "examples": ex, "max_rank": max_rank}


def rotate_half4(x):
  """The vendor's interleaved rotate_half ((x1, x2, x3, x4) -> (-x2, x1, -x4, x3)) at rank 4: x [B, heads, L, dim]."""
  b, hh, n, d = x.shape
  y = x.reshape(b * hh, n, d // 2, 2)                # pairs (2i, 2i+1) on the last axis
  return torch.cat([-y[..., 1:2], y[..., 0:1]], dim=-1).reshape(b, hh, n, d)


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n_op_types": len(h), "ops": dict(sorted(h.items(), key=lambda kv: -kv[1])),
          "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
          "custom": sorted(k for k in h if "CUSTOM" in k.upper()),
          "composite": sorted(k for k in h if "COMPOSITE" in k.upper())}


def largest_tensors(p, top=5):
  """Non-constant tensors (no buffer data) of every subgraph, largest first: (bytes, name, shape, dtype)."""
  from ai_edge_litert import schema_py_generated as S
  buf = open(p, "rb").read()
  m = S.Model.GetRootAsModel(buf, 0)
  size = {S.TensorType.FLOAT32: 4, S.TensorType.FLOAT16: 2, S.TensorType.INT32: 4, S.TensorType.INT8: 1,
          S.TensorType.UINT8: 1, S.TensorType.INT64: 8, S.TensorType.BOOL: 1, S.TensorType.INT16: 2}
  names = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}
  rows = []
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for ti in range(sg.TensorsLength()):
      t = sg.Tensors(ti)
      b = m.Buffers(t.Buffer())
      const = b is not None and (b.DataLength() > 0 or b.Size() > 0)
      if const:
        continue
      shape = [int(t.Shape(j)) for j in range(t.ShapeLength())]
      n = int(np.prod(shape)) if shape else 1
      rows.append((n * size.get(t.Type(), 4), t.Name().decode(), shape, names.get(t.Type(), t.Type())))
  rows.sort(key=lambda r: -r[0])
  return [{"bytes": r[0], "mib": round(r[0] / 2**20, 1), "name": r[1][:120], "shape": r[2], "dtype": r[3]}
          for r in rows[:top]]


def tfl_run(p, x):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p, num_threads=THREADS)
  it.allocate_tensors()
  d = it.get_input_details()[0]
  it.set_tensor(d["index"], x.detach().cpu().numpy().astype(d["dtype"]))
  t0 = time.time()
  it.invoke()
  dt = time.time() - t0
  o = it.get_output_details()[0]
  return it.get_tensor(o["index"]).copy(), dt


def merge_perm():
  """block-order position i -> raster index (processor 2x2 merge-block order) on the GH x GW grid."""
  idx = []
  for hb in range(GH // MERGE):
    for wb in range(GW // MERGE):
      for mh in range(MERGE):
        for mw in range(MERGE):
          idx.append((hb * MERGE + mh) * GW + (wb * MERGE + mw))
  return torch.tensor(idx, dtype=torch.long)


def load_vision_tower():
  from safetensors import safe_open
  with open(os.path.join(MODEL, "config.json")) as f:
    vcfg_dict = json.load(f)["vision_config"]
  vcfg_dict.pop("model_type", None)
  cfg = LlavaOnevision2VisionConfig(**vcfg_dict)
  cfg._attn_implementation = "eager"
  model = LlavaOnevision2VisionPretrainedModel(cfg).eval().float()

  sd = {}
  for shard in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
    with safe_open(shard, framework="pt") as f:
      for k in f.keys():
        if k.startswith("visual."):
          sd[k[len("visual."):]] = f.get_tensor(k).float()
  missing, unexpected = model.load_state_dict(sd, strict=False)
  missing = [m for m in missing if "inv_freq" not in m]  # non-persistent buffers
  assert not missing, f"missing vision keys: {missing[:5]}"
  assert not unexpected, f"unexpected vision keys: {unexpected[:5]}"
  for name in ("inv_freq_t", "inv_freq_h", "inv_freq_w"):
    assert float(getattr(model.video_rope, name).abs().min()) > 0, f"{name} zeroed"
  print(f"vision tower loaded: {len(sd)} tensors")
  return cfg, model


def pictures(example):
  """[(label, path, u8 HxWx3)]: the example picture (EXIF orientation applied, RGB, resized to IMG_W x IMG_H with
  bicubic, as cowork_step.py does), plus a noise picture."""
  from PIL import ImageOps
  im = ImageOps.exif_transpose(Image.open(example)).convert("RGB")
  im = im.resize((IMG_W, IMG_H), Image.BICUBIC)
  assert im.size == (IMG_W, IMG_H), (example, im.size)
  out = [{"label": "example", "path": example, "made_here": False,
          "sha256": hashlib.sha256(open(example, "rb").read()).hexdigest(), "u8": np.asarray(im)}]
  g = torch.Generator().manual_seed(0)
  noise = (torch.rand(IMG_H, IMG_W, 3, generator=g) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
  out.append({"label": "noise", "path": None, "made_here": True, "sha256": None, "u8": noise})
  return out


def corr(a, b):
  a = np.asarray(a, dtype=np.float64).reshape(-1)
  b = np.asarray(b, dtype=np.float64).reshape(-1)
  return float(np.corrcoef(a, b)[0, 1])


def main():
  res = {"ok": False, "stage": "load", "img_h": IMG_H, "img_w": IMG_W, "grid": [GH, GW], "n_patch": N_PATCH,
         "n_tok": N_TOK, "torch": torch.__version__, "tfl_threads": THREADS}
  try:
    import importlib.metadata as md
    res["versions"] = {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert",
                                                  "transformers", "torch", "numpy", "pillow")}
    torch.manual_seed(0)
    t_load = time.time()
    cfg, model = load_vision_tower()
    res["load_s"] = round(time.time() - t_load, 1)
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    qchunk = int(os.environ.get("QCHUNK", "0")) or auto_qchunk(n_heads)
    assert N_PATCH % qchunk == 0, (N_PATCH, qchunk)
    res["qchunk"] = qchunk
    res["n_chunks"] = N_PATCH // qchunk
    res["score_chunk_bytes"] = n_heads * qchunk * N_PATCH * 4
    res["score_full_bytes"] = n_heads * N_PATCH * N_PATCH * 4
    print(f"grid {GH}x{GW} N {N_PATCH} tokens {N_TOK} qchunk {qchunk} ({N_PATCH // qchunk} chunks, "
          f"score chunk {res['score_chunk_bytes']} B = {res['score_chunk_bytes'] / 2**20:.1f} MiB)", flush=True)

    perm = merge_perm()
    pid = torch.arange(N_PATCH)
    pos_raster = torch.stack([torch.zeros_like(pid), pid // GW, pid % GW], dim=-1)

    # --- processor + positions ---
    res["stage"] = "processor-check"
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=True)
    res["processor_class"] = type(proc).__name__
    grid_thw = torch.tensor([[1, GH, GW]])
    with torch.no_grad():
      assert torch.equal(model.video_rope.forward_from_positions(pos_raster), model.video_rope.forward(grid_thw)), \
          "row-major positions != VisionRotaryEmbedding.forward(grid_thw)"
    res["rope_raster_equals_grid_thw"] = True
    res["vendor_sha256"] = VENDOR_SHA

    with torch.no_grad():
      freqs = model.video_rope.forward_from_positions(pos_raster)
      freqs = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)          # [1, N, head_dim]
      rope_cos = freqs.cos().unsqueeze(1).float()                     # [1, 1, N, head_dim] (vendor apply_rotary)
      rope_sin = freqs.sin().unsqueeze(1).float()
      # self-check: the rank-4 rotate_half == the vendor's on a random tensor (bit for bit: data movement + NEG)
      _x = torch.randn(1, n_heads, 8, head_dim, generator=torch.Generator().manual_seed(1))
      from cowork_vendor.modeling_llava_onevision2 import rotate_half as _vendor_rotate_half
      assert torch.equal(rotate_half4(_x), _vendor_rotate_half(_x)), "rotate_half4 != vendor rotate_half"
      res["rotate_half4_equals_vendor"] = True

    conv = model.embeddings.patch_embedding
    ln_pre = model.layernorm_pre
    layers = model.encoder.layers
    scale = layers[0].self_attn.scale
    assert abs(scale - head_dim ** -0.5) < 1e-12
    bounds = [(a, a + qchunk) for a in range(0, N_PATCH, qchunk)]

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conv = conv
        self.ln_pre = ln_pre
        self.blocks = layers
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

      def forward(self, images):                    # [1,H,W,3] in [0,1]
        x = (images.permute(0, 3, 1, 2) - self.mean) / self.std
        p = self.conv(x)                            # [1,1024,GH,GW]
        h = p.flatten(2).transpose(1, 2)            # [1,N,1024] raster (NO gather)
        h = self.ln_pre(h)
        for blk in self.blocks:
          r = blk.layer_norm1(h)
          B, L, _ = r.shape
          qkv = blk.self_attn.qkv(r)                # [1,N,3*D]: (3, heads, head_dim) on the last axis
          D = n_heads * head_dim
          q, k, v = (qkv[:, :, i * D:(i + 1) * D].reshape(B, L, n_heads, head_dim).transpose(1, 2)
                     for i in range(3))                # rank <= 4 (the vendor's split is a rank-5 reshape/permute/unbind)
          q = q * self.rope_cos + rotate_half4(q) * self.rope_sin
          k = k * self.rope_cos + rotate_half4(k) * self.rope_sin
          q = q * scale
          kt = k.transpose(2, 3)                    # [1,heads,head_dim,N]
          outs = []
          for a, b in bounds:                       # query chunks: each row sees every key
            s = torch.matmul(q[:, :, a:b, :], kt)   # [1,heads,Q,N]
            outs.append(torch.matmul(s.softmax(dim=-1), v))
          o = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
          o = o.transpose(1, 2).reshape(B, L, -1)
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
        f = self.ln_q(feats).reshape(1, GH, GW, -1)
        m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                       f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(1, N_TOK, -1)
        return self.mlp(m)                           # [1,N/4,2560]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    # --- per picture: processor patches, vendor reference, eager static graph ---
    res["stage"] = "reference"
    pics = pictures(EXAMPLE)
    res["pictures"] = []
    for pic in pics:
      u8 = pic["u8"]
      pp = proc(images=[Image.fromarray(u8, mode="RGB")], return_tensors="pt")
      pv, gthw = pp["pixel_values"].float(), pp["image_grid_thw"]
      assert gthw.tolist() == [[1, GH, GW]], f"processor grid {gthw.tolist()} != static {[1, GH, GW]}"
      img01 = (torch.from_numpy(u8.copy()).float() / 255.0).unsqueeze(0)          # [1,H,W,3]
      x_norm = (img01.permute(0, 3, 1, 2) - MEAN) / STD
      patches_raster = (x_norm.squeeze(0)
                        .reshape(3, GH, PATCH, GW, PATCH)
                        .permute(1, 3, 0, 2, 4)
                        .reshape(N_PATCH, 3, PATCH, PATCH))
      ours_block = patches_raster[perm].reshape(N_PATCH, -1)
      rec = {"label": pic["label"], "path": pic["path"], "sha256": pic["sha256"], "made_here": pic["made_here"],
             "processor_patch_maxdiff": float((ours_block - pv.reshape(N_PATCH, -1)).abs().max())}
      assert rec["processor_patch_maxdiff"] < 5e-3, f"patch order/norm mismatch: {rec}"
      t0 = time.time()
      with torch.no_grad():
        ref = model(hidden_state=pv.reshape(N_PATCH, 3, PATCH, PATCH),
                    grid_thw=gthw, patch_positions=pos_raster).last_hidden_state.reshape(N_TOK, -1)
      rec["vendor_forward_s"] = round(time.time() - t0, 1)
      t0 = time.time()
      with torch.no_grad():
        feat = enc_m(img01)
        emb_out = adp_m(feat).reshape(N_TOK, -1)
      rec["eager_static_s"] = round(time.time() - t0, 1)
      rec["eager_corr"] = corr(emb_out, ref)
      rec["eager_maxdiff"] = float((emb_out - ref).abs().max())
      rec["ref_absmax"] = float(ref.abs().max())
      np.save(os.path.join(OUT, "ref", f"{pic['label']}_img01.npy"), img01.numpy())
      np.save(os.path.join(OUT, "ref", f"{pic['label']}_ref.npy"), ref.numpy())
      res["pictures"].append(rec)
      print("picture", json.dumps(rec), flush=True)
      if pic["label"] == pics[0]["label"]:
        example_img, example_feat = img01, feat
    res["ref_dir"] = os.path.join(OUT, "ref")

    res["stage"] = "convert-encoder"
    t0 = time.time()
    litert_torch.convert(enc_m, (example_img,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["convert_encoder_s"] = round(time.time() - t0, 1)
    res["stage"] = "convert-adapter"
    t0 = time.time()
    litert_torch.convert(adp_m, (example_feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))
    res["convert_adapter_s"] = round(time.time() - t0, 1)

    res["stage"] = "graph-census"
    enc_p, adp_p = os.path.join(OUT, "vision_encoder.tflite"), os.path.join(OUT, "vision_adapter.tflite")
    res["enc_ops"] = op_hist(enc_p)
    res["adp_ops"] = op_hist(adp_p)
    res["enc_largest"] = largest_tensors(enc_p)
    res["enc_rank"] = rank_census(enc_p)
    res["adp_rank"] = rank_census(adp_p)
    res["rank_gt4_ok"] = res["enc_rank"]["rank_gt4_ops"] == 0 and res["adp_rank"]["rank_gt4_ops"] == 0
    print("rank census enc", json.dumps(res["enc_rank"]), "\nrank census adp", json.dumps(res["adp_rank"]), flush=True)
    res["adp_largest"] = largest_tensors(adp_p)
    for k, p in (("enc", enc_p), ("adp", adp_p)):
      res[f"{k}_bytes"] = os.path.getsize(p)
      res[f"{k}_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("enc ops", json.dumps(res["enc_ops"]), "\nadp ops", json.dumps(res["adp_ops"]), flush=True)
    print("enc largest", json.dumps(res["enc_largest"][:3]), "\nadp largest", json.dumps(res["adp_largest"][:3]),
          flush=True)

    res["stage"] = "parity"
    for rec in res["pictures"]:
      img01 = torch.from_numpy(np.load(os.path.join(OUT, "ref", f"{rec['label']}_img01.npy")))
      ref = np.load(os.path.join(OUT, "ref", f"{rec['label']}_ref.npy"))
      e, te = tfl_run(enc_p, img01)
      a, ta = tfl_run(adp_p, torch.from_numpy(e))
      rec["tfl_fp32_corr"] = corr(a, ref)
      rec["tfl_fp32_maxdiff"] = float(np.max(np.abs(a.reshape(N_TOK, -1).astype(np.float64) - ref)))
      rec["tfl_enc_s"], rec["tfl_adp_s"] = round(te, 2), round(ta, 2)
      np.save(os.path.join(OUT, "ref", f"{rec['label']}_tfl_fp32_feat.npy"), e)
      print("tflite fp32", rec["label"], rec["tfl_fp32_corr"], rec["tfl_fp32_maxdiff"], f"enc {te:.1f}s", flush=True)
    real = [r for r in res["pictures"] if r["label"] != "noise"]
    res["tfl_fp32_corr_min_fixtures"] = min(r["tfl_fp32_corr"] for r in real)
    res["max_intermediate_ok"] = max(res["enc_largest"][0]["bytes"], res["adp_largest"][0]["bytes"]) <= SCORE_CAP
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:400]
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(traceback.format_exc())
    print("ERROR", res["error_type"], res["error_head"], flush=True)

  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k not in ("enc_ops", "adp_ops", "pictures", "enc_largest", "adp_largest")}))
  print("VISION_DONE ok=%s" % res["ok"], flush=True)


if __name__ == "__main__":
  main()
