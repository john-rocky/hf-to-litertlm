"""Export SmolVLM2-500M vision path as the fast_vlm VISION_ENCODER + VISION_ADAPTER.

  VISION_ENCODER: image NHWC [1,512,512,3] in [0,1] -> SigLIP -> [1,1024,768]
  VISION_ADAPTER: [1,1024,768] -> connector (pixel-shuffle x4 + Linear) -> [1,64,960]

The SmolVLM vision embeddings compute position_ids dynamically (bucketize on a patch
mask) — data-dependent, fights torch.export. For one full 512x512 image every patch
is valid, so position_ids == arange(1024): we monkeypatch the embeddings to that
static path. Norm = (x-0.5)/0.5 ([-1,1]); baked in + NCHW transpose.

    ~/clipconv/bin/python scripts/convert_smolvlm2_vision.py [model_dir] [out_dir]
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/smolvlm2-500m"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/smolvlm2-vision"
os.makedirs(OUT, exist_ok=True)
# image size read from the model's vision config (500M=512, 2.2B=384); env override wins.
IMG = int(os.environ.get("IMG_SIZE", "0")) or None


def _get(m, name):
  for base in (getattr(m, "model", None), m):
    if base is not None and hasattr(base, name):
      return getattr(base, name)
  raise AttributeError(name)


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
  global IMG
  res = {"ok": False, "stage": "load"}
  try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    vision_model = _get(model, "vision_model")
    connector = _get(model, "connector")
    if IMG is None:
      IMG = int(getattr(model.config.vision_config, "image_size", 512))
    res["image_size"] = IMG

    # Static-position monkeypatch: bypass the bucketize/mask dynamic position logic.
    emb = vision_model.embeddings
    npatch = emb.num_patches if hasattr(emb, "num_patches") else (IMG // 16) ** 2

    def _static_emb_forward(self, pixel_values, patch_attention_mask=None, **kw):
      pe = self.patch_embedding(pixel_values)            # [B,768,32,32]
      e = pe.flatten(2).transpose(1, 2)                  # [B,1024,768]
      ids = torch.arange(e.shape[1], device=e.device)
      return e + self.position_embedding(ids)

    type(emb).forward = _static_emb_forward
    res["num_patches"] = npatch

    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.vm = vision_model

      def forward(self, images):  # [1,512,512,3] in [0,1]
        x = (images - 0.5) / 0.5
        x = x.permute(0, 3, 1, 2)
        return self.vm(pixel_values=x).last_hidden_state  # [1,1024,768]

    class Adapter(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conn = connector

      def forward(self, feats):  # [1,1024,768] -> [1,64,960]
        return self.conn(feats)

    enc = Encoder().eval(); adp = Adapter().eval()
    img01 = torch.rand(1, IMG, IMG, 3)

    res["stage"] = "eager"
    with torch.no_grad():
      feat = enc(img01)
      out = adp(feat)
    res["enc_out"] = list(feat.shape); res["adp_out"] = list(out.shape)

    res["stage"] = "convert-encoder"
    litert_torch.convert(enc, (img01,)).export(os.path.join(OUT, "vision_encoder.tflite"))
    res["stage"] = "convert-adapter"
    litert_torch.convert(adp, (feat,)).export(os.path.join(OUT, "vision_adapter.tflite"))

    res["stage"] = "parity"
    enc_tfl = tfl_run(os.path.join(OUT, "vision_encoder.tflite"), img01)
    adp_tfl = tfl_run(os.path.join(OUT, "vision_adapter.tflite"), torch.from_numpy(enc_tfl))
    ref = out.detach().cpu().numpy().astype("float64").reshape(-1)
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
