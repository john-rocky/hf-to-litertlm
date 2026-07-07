"""Export Ovis2.5-2B's vision path as TWO tflites matching the fast_vlm bundle contract:

  VISION_ENCODER: image NHWC [1,512,512,3] in [0,1]  ->  features [1,256,4608]
  VISION_ADAPTER: features [1,256,4608]              ->  embeddings [1,256,2048]

The runtime StbImagePreprocessor feeds the encoder a [0,1] NHWC image (already /255,
no mean/std), so we BAKE Ovis's Siglip normalization ((x-0.5)/0.5 -> [-1,1]) + the
patchify into the encoder graph.

Patchify is done GPU-friendly (all reshapes <=4D): apply the vit's patch_embedding
Conv2d (kernel=stride=16, valid) DIRECTLY to the full NCHW image -> per-patch
embeddings in RASTER order, then a single GATHER reorders them into Ovis's
hidden-stride "merge" order (hb,wb,hi,wj). This is provably identical to the model's
own preprocess+embeddings (validated numerically below), but avoids the 8-D
reshape/transpose the literal patchify would need (>5-D ops break GPU delegates and
lack tflite kernels).

Vision tower is made export-able by ovis_work/ovis_static.install_static_vision
(NaViT dynamic-res -> static single-512; see NEXT-SESSION-ovis-bundle.md).

ADAPTER = the Ovis visual-tokenizer tail:
  head (Linear 4608->65532 + LayerNorm) -> softmax -> pad 4 zeros -> vte matmul
  (VisualEmbedding 65536x2048) -> [256, 2048].

    ~/clipconv/bin/python scripts/convert_ovis_vision.py [out_dir]
"""
import sys, os, json, traceback
import types as _types


class _StubLeaf:
  def __getattr__(self, n):
    return lambda *a, **k: None

  def __call__(self, *a, **k):
    return None


def _scipy_healthy():
  try:
    import scipy.sparse.linalg._propack  # noqa: F401
    import scipy.optimize  # noqa: F401
    return True
  except Exception:
    return False


if not _scipy_healthy():
  _pp = _types.ModuleType("scipy.sparse.linalg._propack")
  _pp.__file__ = "<stub>"; _pp.__spec__ = None
  for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _StubLeaf())
  sys.modules["scipy.sparse.linalg._propack"] = _pp
  _opt = _types.ModuleType("scipy.optimize")
  _opt.__file__ = "<stub>"; _opt.__spec__ = None
  _opt.linear_sum_assignment = lambda *a, **k: None
  sys.modules["scipy.optimize"] = _opt

import litert_torch  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Ovis remote code predates transformers 5.12: install the same 3 compat shims used
# by ovis_work/run_verify.py so from_pretrained succeeds.
import transformers.modeling_utils as _mu  # noqa: E402
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(_mu.PreTrainedModel, "is_parallelizable"):
  _mu.PreTrainedModel.is_parallelizable = False
from transformers import AutoModelForCausalLM  # noqa: E402

sys.path.insert(0, "ovis_work")
import ovis_static  # noqa: E402

MID = "AIDC-AI/Ovis2.5-2B"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/ovis-vision"
os.makedirs(OUT, exist_ok=True)

GRID_H = GRID_W = 32          # 512 / patch_size(16)
N_PATCH = GRID_H * GRID_W     # 1024
HS = 2                        # hidden_stride
N_TOK = N_PATCH // (HS * HS)  # 256 soft tokens
N_IND = 4                     # len(INDICATOR_IDS): 4 zero-pad columns


def _load():
  def load():
    return AutoModelForCausalLM.from_pretrained(
        MID, trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
  try:
    return load()
  except TypeError as e:
    if "tie_weights" not in str(e):
      raise
    for name, mod in list(sys.modules.items()):
      if "modeling_ovis2_5" in name and hasattr(mod, "Ovis2_5"):
        cls = mod.Ovis2_5; _o = cls.tie_weights
        cls.tie_weights = lambda self, *a, **k: _o(self)
    return load()


def _find_apply_rope():
  for name, mod in sys.modules.items():
    if "modeling_ovis2_5" in name and hasattr(mod, "apply_rotary_pos_emb_vision"):
      return mod.apply_rotary_pos_emb_vision
  raise RuntimeError("apply_rotary_pos_emb_vision not found in sys.modules")


def _install_manual_attn(enc, apply_rope):
  """Replace the static SDPA attention with an explicit matmul+softmax (<=3-D
  tensors, no FLEX op) so the encoder converts clean for GPU/XNNPACK."""
  def _attn(self, hidden_states, cu_seqlens=None, position_embeddings=None):
    L, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(L, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(L, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(L, self.num_heads, self.head_dim)
    if self.use_rope:
      cos, sin = position_embeddings
      q, k = apply_rope(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
      q, k = q.squeeze(0), k.squeeze(0)
    q = q.transpose(0, 1)               # (heads, L, d)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn = (q * self.scale) @ k.transpose(-2, -1)   # (heads, L, L)
    attn = attn.softmax(dim=-1)
    o = (attn @ v).transpose(0, 1).reshape(L, self.embed_dim)
    return self.out_proj(o)
  for layer in enc.layers:
    layer.self_attn.forward = _attn.__get__(layer.self_attn, type(layer.self_attn))


def _merge_perm():
  """Index that reorders RASTER patch rows (r = h*32 + w) into Ovis's hidden-stride
  merge order m = ((hb*16 + wb)*2 + hi)*2 + wj, where h = hb*2+hi, w = wb*2+wj."""
  idx = []
  for hb in range(GRID_H // HS):
    for wb in range(GRID_W // HS):
      for hi in range(HS):
        for wj in range(HS):
          idx.append((hb * HS + hi) * GRID_W + (wb * HS + wj))
  return torch.tensor(idx, dtype=torch.long)


def op_hist(p):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p); it.allocate_tensors()
  h = {}
  for d in it._get_ops_details():
    h[d["op_name"]] = h.get(d["op_name"], 0) + 1
  return {"n": len(h), "flex": sorted(k for k in h if k.upper().startswith("FLEX")),
          "custom": sorted(k for k in h if "CUSTOM" in k.upper())}


def tfl_run(p, x):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p); it.allocate_tensors()
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
  res = {"ok": False, "stage": "load", "n_tok": N_TOK}
  try:
    model = _load()
    vt = model.visual_tokenizer
    vm = vt.vit.vision_model
    emb = vm.embeddings
    assert isinstance(emb.patch_embedding, torch.nn.Conv2d), \
        f"expected Conv2d patch_embedding, got {type(emb.patch_embedding)}"

    # --- make the vision tower static/export-able, then swap SDPA -> manual attn ---
    ovis_static.install_static_vision(model, GRID_H, GRID_W)
    enc = vm.encoder
    _install_manual_attn(enc, _find_apply_rope())
    perm = _merge_perm()

    # --- reference patchify: run the model's OWN preprocess on a 512x512 image ---
    res["stage"] = "ref-preprocess"
    torch.manual_seed(0)
    img_u8 = (torch.rand(GRID_H * 16, GRID_W * 16, 3) * 255).round().clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img_u8.numpy(), mode="RGB")
    pv_ref, grid_ref = vt.preprocess(image=pil, min_pixels=512 * 512, max_pixels=512 * 512)
    assert list(grid_ref[0]) == [1, GRID_H, GRID_W], f"unexpected grid {grid_ref}"
    assert list(pv_ref.shape) == [N_PATCH, 3 * 16 * 16], f"unexpected pv {pv_ref.shape}"
    img01 = (img_u8.float() / 255.0).unsqueeze(0)   # [1,512,512,3] in [0,1]

    class Encoder(torch.nn.Module):
      """NHWC [1,512,512,3] [0,1] -> Siglip norm -> raster patch-conv -> merge-order
      gather -> +pos-embed -> static Siglip2 encoder -> features [1,256,4608]."""
      def __init__(self):
        super().__init__()
        self.patch = emb.patch_embedding            # Conv2d(3,1152,16,stride16)
        self.enc = enc                              # static encoder (patched)
        self.register_buffer("pe_const", emb._pe_const, persistent=False)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("grid", torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.long),
                             persistent=False)

      def forward(self, images):
        x = (images - 0.5) / 0.5                    # [-1,1]  [1,512,512,3]
        x = x.permute(0, 3, 1, 2)                   # [1,3,512,512]
        p = self.patch(x)                           # [1,1152,32,32]
        p = p.flatten(2).transpose(1, 2).reshape(N_PATCH, -1)   # [1024,1152] raster
        p = p.index_select(0, self.perm) + self.pe_const        # [1024,1152] merge
        h, states = self.enc(p, self.grid, output_hidden_states=True)
        feat = states[-1].reshape(N_TOK, -1)        # [256,4608]
        return feat.unsqueeze(0)                    # [1,256,4608]

    class Adapter(torch.nn.Module):
      """features [1,256,4608] -> head+LN -> softmax -> pad -> vte -> [1,256,2048]."""
      def __init__(self):
        super().__init__()
        self.head = vt.head                         # Sequential(Linear 4608->65532, LayerNorm)
        self.vte = model.vte                        # VisualEmbedding(65536,2048)

      def forward(self, features):
        f = features.squeeze(0)                      # [256,4608]
        logits = self.head(f)                        # [256,65532]
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        pad = torch.zeros(tokens.shape[0], N_IND, dtype=tokens.dtype)
        tokens = torch.cat((tokens, pad), dim=1)     # [256,65536]
        e = self.vte(tokens)                         # [256,2048]
        return e.unsqueeze(0)                        # [1,256,2048]

    enc_m = Encoder().eval()
    adp_m = Adapter().eval()

    res["stage"] = "eager"
    with torch.no_grad():
      feat = enc_m(img01)
      emb_out = adp_m(feat)
      # ground truth for the whole vision path: vte(visual_tokenizer(pv_ref, grid))
      feat_ref = vt._encode(pv_ref, grid_ref).unsqueeze(0)        # [1,256,4608]
      soft_ref = vt(pv_ref, grid_ref)                             # [256,65536]
      emb_ref = model.vte(soft_ref).unsqueeze(0)                  # [1,256,2048]
    res["enc_out"] = list(feat.shape)
    res["adp_out"] = list(emb_out.shape)
    res["enc_eager_maxdiff"] = float((feat - feat_ref).abs().max())
    res["enc_eager_corr"] = float(np.corrcoef(
        feat.flatten().numpy(), feat_ref.flatten().numpy())[0, 1])
    res["adp_eager_maxdiff"] = float((emb_out - emb_ref).abs().max())
    res["adp_eager_corr"] = float(np.corrcoef(
        emb_out.flatten().numpy(), emb_ref.flatten().numpy())[0, 1])
    print("eager enc corr", res["enc_eager_corr"], "maxdiff", res["enc_eager_maxdiff"])
    print("eager adp corr", res["adp_eager_corr"], "maxdiff", res["adp_eager_maxdiff"])

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc_m, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp_m, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    enc_tfl = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    adp_tfl = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(enc_tfl))
    ref = emb_ref.detach().cpu().numpy().astype("float64").reshape(-1)
    got = adp_tfl.astype("float64").reshape(-1)
    n = min(len(ref), len(got))
    res["enc_ops"] = op_hist(os.path.join(OUT, "vision_encoder.tflite"))
    res["adp_ops"] = op_hist(os.path.join(OUT, "vision_adapter.tflite"))
    res["end2end_corr"] = float(np.corrcoef(got[:n], ref[:n])[0, 1])
    res["end2end_maxdiff"] = float(np.max(np.abs(got[:n] - ref[:n])))
    res["enc_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_encoder.tflite")) / 1e6, 1)
    res["adp_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_adapter.tflite")) / 1e6, 1)
    print("end2end corr", res["end2end_corr"], "maxdiff", res["end2end_maxdiff"])
    print("enc ops", res["enc_ops"], "\nadp ops", res["adp_ops"])

    res["stage"] = "quantize-int8"
    res["enc_int8_mb"] = _quantize_int8(
        os.path.join(OUT, "vision_encoder.tflite"), os.path.join(OUT, "vision_encoder_int8.tflite"))
    res["adp_int8_mb"] = _quantize_int8(
        os.path.join(OUT, "vision_adapter.tflite"), os.path.join(OUT, "vision_adapter_int8.tflite"))
    # int8 end-to-end parity
    e8 = tfl_run(os.path.join(OUT, "vision_encoder_int8.tflite"), img01)
    a8 = tfl_run(os.path.join(OUT, "vision_adapter_int8.tflite"), torch.from_numpy(e8))
    got8 = a8.astype("float64").reshape(-1)
    res["end2end_int8_corr"] = float(np.corrcoef(got8[:n], ref[:n])[0, 1])
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
  print("RESULT " + json.dumps({k: v for k, v in res.items() if k not in ("enc_ops", "adp_ops")}))


if __name__ == "__main__":
  main()
