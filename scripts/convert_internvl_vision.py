"""De-risk probe: convert InternVL3 vision path (InternViT + pixel_shuffle + mlp1)
to a single tflite, for the fast_vlm-ride bundle.

`InternVLChatModel.extract_feature(pixel_values)` IS the combined vision encoder +
projector: vision_model -> drop CLS -> pixel_shuffle(downsample_ratio) -> mlp1.
For a single 448x448 image it lowers to a fixed [1, 256, 1536] (256 soft tokens at
the Qwen2.5-1.5B hidden dim). This is the novel converter risk (does InternViT export
cleanly); the decoder is a standard Qwen2.

    ~/clipconv/bin/python scripts/convert_internvl_vision.py [model_dir] [out_dir]

Records the tflite op histogram (flex/custom?) + CPU float parity vs eager.
"""
import sys, os, json, traceback

# --- conditional scipy stub (clipconv env has the Accelerate-rebuilt scipy; only
#     stub when it is actually broken, else the stub breaks the real import chain).
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
  _pp.__file__ = "<stub:scipy._propack>"
  _pp.__spec__ = None
  for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _StubLeaf())
  sys.modules["scipy.sparse.linalg._propack"] = _pp
  _opt = _types.ModuleType("scipy.optimize")
  _opt.__file__ = "<stub:scipy.optimize>"
  _opt.__spec__ = None
  _opt.linear_sum_assignment = lambda *a, **k: None
  sys.modules["scipy.optimize"] = _opt

# Import litert_torch FIRST (before any transformers model submodule) to avoid
# corrupting litert_converter's lazy MLIR dialect loading.
import litert_torch  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from transformers import AutoModel  # noqa: E402

# The remote InternVLChatModel code predates transformers 5.12's weight-tying
# refactor (modeling_utils.py:1402/1425 expect `all_tied_weights_keys`). Provide a
# class-level default ({}) so models whose custom __init__ doesn't set it load.
# Safe: InternVL3's LLM is tie=False and the composite has no tied params.
import transformers.modeling_utils as _mu  # noqa: E402
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/internvl3-2b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/internvl-vision"
os.makedirs(OUT, exist_ok=True)


def read_op_histogram(tflite_path):
  from ai_edge_litert.interpreter import Interpreter
  interp = Interpreter(model_path=tflite_path)
  interp.allocate_tensors()
  hist = {}
  for d in interp._get_ops_details():
    op = d["op_name"]
    hist[op] = hist.get(op, 0) + 1
  flex = sorted(k for k in hist if k.upper().startswith("FLEX"))
  custom = sorted(k for k in hist if "CUSTOM" in k.upper())
  return {"n_types": len(hist), "hist": dict(sorted(hist.items(), key=lambda x: -x[1])),
          "flex": flex, "custom": custom}


def run_parity(tflite_path, px, eager_out):
  from ai_edge_litert.interpreter import Interpreter
  interp = Interpreter(model_path=tflite_path)
  interp.allocate_tensors()
  ins = interp.get_input_details()
  interp.set_tensor(ins[0]["index"], px.detach().cpu().numpy().astype(ins[0]["dtype"]))
  interp.invoke()
  od = interp.get_output_details()[0]
  tfl = interp.get_tensor(od["index"]).astype("float64").reshape(-1)
  ref = eager_out.detach().cpu().numpy().astype("float64").reshape(-1)
  n = min(len(tfl), len(ref))
  tfl, ref = tfl[:n], ref[:n]
  return {"n": n, "max_abs_diff": float(np.max(np.abs(tfl - ref))),
          "corr": float(np.corrcoef(tfl, ref)[0, 1]) if n > 1 else 1.0,
          "out_shape": list(eager_out.shape)}


def main():
  res = {"ok": False, "stage": None}
  try:
    res["stage"] = "load"
    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True).eval()
    # Force eager attention everywhere (InternViT remote code may carry use_flash_attn).
    n_off = 0
    for mod in model.modules():
      if hasattr(mod, "use_flash_attn") and getattr(mod, "use_flash_attn"):
        mod.use_flash_attn = False
        n_off += 1
    res["flash_off"] = n_off
    res["select_layer"] = getattr(model.config, "select_layer", None)
    res["downsample_ratio"] = getattr(model.config, "downsample_ratio", None)

    class VisionWrap(torch.nn.Module):
      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, pixel_values):  # [1,3,448,448] -> [1,256,1536]
        return self.m.extract_feature(pixel_values)

    wrap = VisionWrap(model).eval()
    px = torch.randn(1, 3, 448, 448)

    res["stage"] = "eager-forward"
    with torch.no_grad():
      out = wrap(px)
    res["eager_out_shape"] = list(out.shape)
    print("eager out:", out.shape)

    res["stage"] = "convert"
    edge = litert_torch.convert(wrap, (px,))
    tfl = os.path.join(OUT, "internvl_vision.tflite")
    edge.export(tfl)
    res["tflite"] = tfl
    res["tflite_mb"] = round(os.path.getsize(tfl) / 1e6, 1)

    res["stage"] = "read-ops"
    res["ops"] = read_op_histogram(tfl)
    res["stage"] = "parity"
    res["parity"] = run_parity(tfl, px, out)
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    tb = traceback.format_exc()
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or [tb.strip().splitlines()[-1]])[0][:600]
    for ln in tb.splitlines():
      s = ln.strip()
      if ("While executing" in s) or s.startswith("%") or ("aten." in s) or ("Lowering" in s):
        res["failing_op"] = s[:400]
        break
    with open(os.path.join(OUT, "trace.txt"), "w") as f:
      f.write(tb)

  with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("PROBE_RESULT " + json.dumps({k: v for k, v in res.items() if k != "ops"}))
  if res.get("ops"):
    print("OPS " + json.dumps(res["ops"]))


if __name__ == "__main__":
  main()
