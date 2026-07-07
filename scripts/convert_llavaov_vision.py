"""Export LLaVA-OneVision-0.5B vision path as the fast_vlm VISION_ENCODER + VISION_ADAPTER.

Single base 384x384 image (no anyres):
  VISION_ENCODER: image NHWC [1,384,384,3] in [0,1]  ->  SigLIP  ->  [1,729,1152]
  VISION_ADAPTER: [1,729,1152] -> multi_modal_projector -> [1,729,896]
                  -> append the learned `image_newline` token -> [1,730,896]

Runtime feeds [0,1] NHWC (stb_image_preprocessor /255.0f), so we BAKE the OpenAI-CLIP
normalization + NCHW transpose into the encoder graph.

    ~/clipconv/bin/python scripts/convert_llavaov_vision.py [model_dir] [out_dir]
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
  _pp = _types.ModuleType("scipy.sparse.linalg._propack"); _pp.__file__ = "<stub>"; _pp.__spec__ = None
  for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _StubLeaf())
  sys.modules["scipy.sparse.linalg._propack"] = _pp
  _opt = _types.ModuleType("scipy.optimize"); _opt.__file__ = "<stub>"; _opt.__spec__ = None
  _opt.linear_sum_assignment = lambda *a, **k: None
  sys.modules["scipy.optimize"] = _opt

import litert_torch  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/llava-ov-0.5b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/llavaov-vision"
os.makedirs(OUT, exist_ok=True)

# OpenAI-CLIP normalization (LLaVA-OneVision image processor), applied to a [0,1] image.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG = 384


def _get(m, name):
  """vision_tower/multi_modal_projector/image_newline live on `model` or `model.model`."""
  for base in (getattr(m, "model", None), m):
    if base is not None and hasattr(base, name):
      return getattr(base, name)
  raise AttributeError(f"{name} not found on model or model.model")


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
  it.set_tensor(d["index"], x.detach().cpu().numpy().astype(d["dtype"])); it.invoke()
  o = it.get_output_details()[0]
  return it.get_tensor(o["index"])


def main():
  res = {"ok": False, "stage": "load"}
  try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    vision_tower = _get(model, "vision_tower")
    projector = _get(model, "multi_modal_projector")
    image_newline = _get(model, "image_newline")  # [hidden] learned param
    res["newline_shape"] = list(image_newline.shape)

    mean = torch.tensor(CLIP_MEAN).view(1, 1, 1, 3)
    std = torch.tensor(CLIP_STD).view(1, 1, 1, 3)

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.vt = vision_tower
        self.register_buffer("mean", mean); self.register_buffer("std", std)

      def forward(self, images):  # [1,384,384,3] in [0,1]
        x = (images - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)  # NCHW
        return self.vt(pixel_values=x).last_hidden_state  # [1,729,1152]

    class Adapter(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.proj = projector
        self.register_buffer("newline", image_newline.detach().reshape(1, 1, -1))

      def forward(self, feats):  # [1,729,1152] -> [1,730,896]
        p = self.proj(feats)
        return torch.cat([p, self.newline.to(p.dtype)], dim=1)

    enc = Encoder().eval(); adp = Adapter().eval()
    img01 = torch.rand(1, IMG, IMG, 3)

    res["stage"] = "eager"
    with torch.no_grad():
      feat = enc(img01)
      emb = adp(feat)
    res["enc_out"] = list(feat.shape); res["adp_out"] = list(emb.shape)

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    enc_tfl = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    adp_tfl = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(enc_tfl))
    ref = emb.detach().cpu().numpy().astype("float64").reshape(-1)
    got = adp_tfl.astype("float64").reshape(-1)
    n = min(len(ref), len(got))
    res["enc_ops"] = op_hist(os.path.join(OUT, "vision_encoder.tflite"))
    res["adp_ops"] = op_hist(os.path.join(OUT, "vision_adapter.tflite"))
    res["end2end_corr"] = float(np.corrcoef(got[:n], ref[:n])[0, 1])
    res["end2end_maxdiff"] = float(np.max(np.abs(got[:n] - ref[:n])))
    res["enc_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_encoder.tflite")) / 1e6, 1)
    res["adp_mb"] = round(os.path.getsize(os.path.join(OUT, "vision_adapter.tflite")) / 1e6, 1)
    res["ok"] = True; res["stage"] = "done"
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
