"""Export InternVL3 vision path as TWO tflites matching the fast_vlm bundle contract
(verified from FastVLM-0.5B.litertlm):

  VISION_ENCODER: image NHWC [1,448,448,3] in [0,1]  ->  features [1,256,4096]
  VISION_ADAPTER: features [1,256,4096]              ->  embeddings [1,256,1536]

The runtime StbImagePreprocessor feeds the encoder a [0,1] NHWC image with NO
mean/std (stb_image_preprocessor.cc:285 `/255.0f`), so we BAKE InternVL's ImageNet
normalization (mean/std) + the NCHW permute into the encoder graph.

    ~/clipconv/bin/python scripts/convert_internvl_vision_split.py [model_dir] [out_dir]
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
from transformers import AutoModel  # noqa: E402
import transformers.modeling_utils as _mu  # noqa: E402
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/internvl3-2b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/internvl-vision-split"
os.makedirs(OUT, exist_ok=True)

# ImageNet normalization (InternVL preprocessor_config.json), applied to a [0,1] image.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def main():
  res = {"ok": False, "stage": "load"}
  try:
    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True).eval()
    # Optional C12 fix: InternViT attention reshapes qkv to a 5D tensor
    # (`reshape(B,N,3,H,d)`), which GPU delegates reject ("RESHAPE dims must be < 5").
    # Split qkv into q/k/v FIRST (each [B,N,C]) then reshape each to 4D — numerically
    # identical, no 5D intermediate. Gated by env GPU4D=1.
    if os.environ.get("GPU4D"):
      import types as _t
      def _naive_attn_4d(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [B,N,C]
        H = self.num_heads; d = C // H
        q = q.reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B,H,N,d] 4D
        k = k.reshape(B, N, H, d).permute(0, 2, 1, 3)
        v = v.reshape(B, N, H, d).permute(0, 2, 1, 3)
        if self.qk_normalization:
          B_, H_, N_, D_ = q.shape
          q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
          k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
      n = 0
      for mod in model.modules():
        if type(mod).__name__ == "InternAttention":
          type(mod)._naive_attn = _naive_attn_4d
          n += 1
      res["gpu4d_patched_attn_classes"] = n
      print("GPU4D: patched InternAttention._naive_attn -> 4D-clean")

    vm = model.vision_model
    mlp1 = model.mlp1
    ds = model.downsample_ratio
    sel = model.select_layer
    pixel_shuffle = model.pixel_shuffle  # bound method

    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 1, 3)

    class Encoder(torch.nn.Module):
      """NHWC [0,1] image -> InternViT -> drop CLS -> pixel_shuffle -> [1,256,4096].
      Bakes ImageNet normalize + NCHW permute (runtime feeds [0,1] NHWC, no mean/std)."""
      def __init__(self):
        super().__init__()
        self.vm = vm
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

      def forward(self, images):  # [1,448,448,3] in [0,1]
        x = (images - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)  # NCHW
        if sel == -1:
          vit = self.vm(pixel_values=x, output_hidden_states=False,
                        return_dict=True).last_hidden_state
        else:
          vit = self.vm(pixel_values=x, output_hidden_states=True,
                        return_dict=True).hidden_states[sel]
        vit = vit[:, 1:, :]  # drop CLS -> [1,1024,1024]
        h = w = int(vit.shape[1] ** 0.5)
        vit = vit.reshape(vit.shape[0], h, w, -1)
        vit = pixel_shuffle(vit, scale_factor=ds)  # [1,16,16,4096]
        vit = vit.reshape(vit.shape[0], -1, vit.shape[-1])  # [1,256,4096]
        return vit

    class Adapter(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.mlp1 = mlp1

      def forward(self, soft_tokens):  # [1,256,4096] -> [1,256,1536]
        return self.mlp1(soft_tokens)

    enc = Encoder().eval()
    adp = Adapter().eval()

    img01 = torch.rand(1, 448, 448, 3)  # [0,1] NHWC
    res["stage"] = "eager"
    with torch.no_grad():
      feat = enc(img01)
      emb = adp(feat)
      # reference via the model's own extract_feature on the equivalently-normalized NCHW
      nchw = ((img01 - mean) / std).permute(0, 3, 1, 2)
      ref = model.extract_feature(nchw)
    res["enc_out"] = list(feat.shape)
    res["adp_out"] = list(emb.shape)
    res["ref_out"] = list(ref.shape)
    res["eager_pipeline_vs_extract_feature_maxdiff"] = float((emb - ref).abs().max())
    print("shapes enc", feat.shape, "adp", emb.shape, "ref", ref.shape,
          "eager maxdiff", res["eager_pipeline_vs_extract_feature_maxdiff"])

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    enc_tfl = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    adp_tfl = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(enc_tfl))
    ref_f = ref.detach().cpu().numpy().astype("float64").reshape(-1)
    got = adp_tfl.astype("float64").reshape(-1)
    n = min(len(ref_f), len(got))
    res["enc_ops"] = op_hist(os.path.join(OUT, "vision_encoder.tflite"))
    res["adp_ops"] = op_hist(os.path.join(OUT, "vision_adapter.tflite"))
    res["end2end_corr"] = float(np.corrcoef(got[:n], ref_f[:n])[0, 1])
    res["end2end_maxdiff"] = float(np.max(np.abs(got[:n] - ref_f[:n])))
    res["enc_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_encoder.tflite")) / 1e6, 1)
    res["adp_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_adapter.tflite")) / 1e6, 1)
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    tb = traceback.format_exc()
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:500]
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(tb)

  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("RESULT " + json.dumps(res))


if __name__ == "__main__":
  main()
