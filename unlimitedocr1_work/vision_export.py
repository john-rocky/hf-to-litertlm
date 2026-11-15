#!/usr/bin/env python3
"""The fast_vlm vision TFLite pair of Unlimited-OCR-2 (base mode, one 1024x1024 page) from the pure-torch re-author
uocr_vision.py: fixtures, HF references, module identity, export, op histogram, intermediate tensor bytes, fp32 parity,
LayerNorm absmax / fp16 pre-scales, quantization A/B.

  MODEL=<checkpoint dir> FIXTURES=<dir> FIXTURE_FILES=a.jpg,b.png,... \
  python vision_export.py OUT --stage fixtures|ref|identity|calib|convert|quant|all [--force]
                          [--rel-form bias|aug] [--qsplit N] [--fp16safe]      (litert-torch 0.9.3)

  OUT/fix/<name>_p1024.png      the fixtures: _uocr_build_inputs' base view (ImageOps.pad to 1024^2, colour 127)
  OUT/ref/<name>.npz            img01 (u8 * (1/255), float32 NHWC = what the runtime hands the encoder),
                                feats32 / out32 (HF deepencoder.py classes, fp32, pixels = ToTensor + Normalize),
                                feats64 / out64 (the same in float64 = the truth)
  OUT/vision_encoder.tflite     image NHWC [1,1024,1024,3] in [0,1] -> [1,256,2048]
  OUT/vision_adapter.tflite     [1,256,2048] -> [1,273,1280]
  OUT/*_fp16.tflite / *_int8.tflite  float_casting of FULLY_CONNECTED + CONV_2D weights / dynamic_wi8_afp32
  OUT/result.json, OUT/tflite_inspect.json

Parity = the TFLite pair on the XNNPACK CPU (ai-edge-litert Interpreter, 4 threads) against the HF fp32 rows that
masked_scatter_ writes into the 273 <image> slots; the bar is corr >= 0.9999 (fp32) and >= 0.999 for a quantized
candidate. The first fixture is also the sample input of the conversion (only its shape and dtype reach the graph). The float64 rows give each output's rel_rms distance to the truth next to the HF fp32's own.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("uocr_vision", os.path.join(HERE, "uocr_vision.py"))
uv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uv)

# Document pictures for the parity check: FIXTURES (a directory) + FIXTURE_FILES (comma-separated names in it); a
# fixture's name is its file name without the extension, '-' read as '_'. The ship script passes the four example
# pictures of xiaomi-research/Xiaomi-OCR-0 (receipt, dense equations, handwritten formula, calligraphy).
_FIX_DIR = os.environ.get("FIXTURES", "")
FIXTURES = [(os.path.splitext(f)[0].replace("-", "_"), os.path.join(_FIX_DIR, f))
            for f in os.environ.get("FIXTURE_FILES", "").split(",") if f]
THREADS = 4
PARTS = {"enc": "vision_encoder", "adp": "vision_adapter"}


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def tfl_inspect(path):
  """Op histogram, I/O, and the largest tensors of every subgraph from the flatbuffer (bytes = elements x dtype)."""
  from ai_edge_litert import schema_py_generated as S
  buf = open(path, "rb").read()
  m = S.Model.GetRootAs(buf, 0)
  opname = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
  tname = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}
  size = {"FLOAT32": 4, "FLOAT16": 2, "INT8": 1, "INT32": 4, "INT64": 8, "BOOL": 1, "INT4": 0.5, "UINT8": 1,
          "INT16": 2}
  out = {"bytes": os.path.getsize(path), "subgraphs": []}
  hist = Counter()
  wtypes = Counter()
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    const = set()
    for ti in range(sg.TensorsLength()):
      t = sg.Tensors(ti)
      b = m.Buffers(t.Buffer())
      if b is None:
        continue
      if b.DataLength() > 0 or (hasattr(b, "Size") and b.Size() > 0):
        const.add(ti)
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      oc = m.OperatorCodes(op.OpcodeIndex())
      name = opname.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
      if name == "CUSTOM":
        name = "CUSTOM:" + (oc.CustomCode() or b"").decode()
      hist[name] += 1
      if name in ("FULLY_CONNECTED", "CONV_2D", "BATCH_MATMUL") and op.InputsLength() > 1:
        wi = op.Inputs(1)
        if wi >= 0:
          wtypes[f"{name}:{tname.get(sg.Tensors(wi).Type())}"] += 1
    acts = []
    for ti in range(sg.TensorsLength()):
      if ti in const:
        continue
      t = sg.Tensors(ti)
      shp = t.ShapeAsNumpy().tolist() if t.ShapeLength() else []
      dt = tname.get(t.Type(), str(t.Type()))
      n = int(np.prod(shp)) if shp else 1
      acts.append((n * size.get(dt, 4), (t.Name() or b"").decode()[:90], shp, dt))
    acts.sort(key=lambda r: -r[0])
    io = {"inputs": [[sg.Tensors(t).ShapeAsNumpy().tolist(), tname.get(sg.Tensors(t).Type())]
                     for t in sg.InputsAsNumpy().tolist()],
          "outputs": [[sg.Tensors(t).ShapeAsNumpy().tolist(), tname.get(sg.Tensors(t).Type())]
                      for t in sg.OutputsAsNumpy().tolist()]}
    out["subgraphs"].append({"name": (sg.Name() or b"").decode(), "io": io,
                             "largest_activations": [{"bytes": int(b), "name": nm, "shape": s, "dtype": d}
                                                     for b, nm, s, d in acts[:6]]})
  out["n_ops"] = sum(hist.values())
  out["ops"] = dict(sorted(hist.items()))
  out["weight_types"] = dict(sorted(wtypes.items()))
  out["bad_ops"] = sorted(k for k in hist if any(w in k.upper() for w in ("GATHER", "FLEX", "CUSTOM", "STABLEHLO")))
  return out


class Runner:
  def __init__(self, path, threads=THREADS):
    from ai_edge_litert.interpreter import Interpreter
    self.it = Interpreter(model_path=path, num_threads=threads)
    self.it.allocate_tensors()
    self.inp = self.it.get_input_details()[0]
    self.out = self.it.get_output_details()[0]

  def __call__(self, x):
    self.it.set_tensor(self.inp["index"], np.ascontiguousarray(x, dtype=self.inp["dtype"]))
    self.it.invoke()
    return self.it.get_tensor(self.out["index"]).copy()


def metrics(got, ref, truth=None):
  g = np.asarray(got, dtype=np.float64).reshape(-1, np.asarray(ref).shape[-1])
  r = np.asarray(ref, dtype=np.float64).reshape(g.shape)
  cos = (g * r).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  m = {"corr": float(np.corrcoef(g.reshape(-1), r.reshape(-1))[0, 1]), "cos_mean": float(cos.mean()),
       "cos_min": float(cos.min()), "argmin_row": int(cos.argmin()), "maxdiff": float(np.abs(g - r).max()),
       "rel_rms": float(np.sqrt(((g - r) ** 2).mean() / (r ** 2).mean()))}
  if truth is not None:
    t = np.asarray(truth, dtype=np.float64).reshape(g.shape)
    m["rel_rms_to_f64"] = float(np.sqrt(((g - t) ** 2).mean() / (t ** 2).mean()))
  return m


def quant_int8(src, dst):
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
  q.quantize().export_model(dst)


def quant_fp16(src, dst):
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  base = copy.deepcopy(r.dynamic_wi8_afp32()[0])
  base["algorithm_key"] = "float_casting"
  base["op_config"]["weight_tensor_config"] = {
      "num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"}
  base["op_config"]["compute_precision"] = "FLOAT"
  base["op_config"]["explicit_dequantize"] = False
  recipes = []
  for op in ("FULLY_CONNECTED", "CONV_2D"):
    rr = copy.deepcopy(base)
    rr["operation"] = op
    recipes.append(rr)
  q = quantizer.Quantizer(src, recipes)
  q.quantize().export_model(dst)


def stage_fixtures(out, res):
  from PIL import Image
  d = os.path.join(out, "fix")
  os.makedirs(d, exist_ok=True)
  res["fixtures"] = {}
  for name, src in FIXTURES:
    im = Image.open(src)
    padded = uv.author_pad(im)
    p = os.path.join(d, f"{name}_p1024.png")
    padded.save(p)
    res["fixtures"][name] = {"src": src, "src_sha256": sha256(src), "src_size": list(im.size),
                             "file": p, "sha256": sha256(p)}
    print("fixture", name, im.size, "->", p, flush=True)


def load_fix(out, name):
  from PIL import Image
  return Image.open(os.path.join(out, "fix", f"{name}_p1024.png")).convert("RGB")


def stage_ref(out, res, names):
  d = os.path.join(out, "ref")
  os.makedirs(d, exist_ok=True)
  tensors = uv.load_vision_tensors()
  ref32 = uv.HFReference(torch.float32, tensors=tensors)
  ref64 = uv.HFReference(torch.float64, tensors=tensors)
  res["ref"] = {}
  for n in names:
    im = load_fix(out, n)
    u8 = np.asarray(im, dtype=np.uint8)
    img01 = (u8.astype(np.float32) * np.float32(1.0 / 255.0))[None]          # what LiteRT-LM hands the encoder
    pix = uv.hf_pixels(im)                                                     # ToTensor + Normalize, fp32
    t0 = time.time()
    with torch.no_grad():
      f32, o32 = ref32(pix)
      f64, o64 = ref64(pix.double())
    np.savez(os.path.join(d, f"{n}.npz"), img01=img01, feats32=f32[0].numpy(), out32=o32.numpy(),
             feats64=f64[0].numpy(), out64=o64.numpy(), pix=pix[0].numpy())
    res["ref"][n] = {"seconds": round(time.time() - t0, 1), "out_shape": list(o32.shape),
                     "hf32_vs_f64": metrics(o32.numpy(), o64.numpy())}
    print("ref", n, json.dumps(res["ref"][n]), flush=True)
  return tensors


def refs_of(out, names):
  return {n: np.load(os.path.join(out, "ref", f"{n}.npz")) for n in names}


def stage_identity(out, res, names, opts, tensors=None):
  """Re-author vs HF classes: float64 on the HF pixels (module identity) and float32 from img01 (the graph's own
  u8 floor step) against HF fp32."""
  tensors = tensors if tensors is not None else uv.load_vision_tensors()
  e64, a64, info = uv.build(opts, torch.float64, tensors=tensors)
  e32, a32, _ = uv.build(opts, torch.float32, tensors=tensors)
  res["build_info"] = info
  refs = refs_of(out, names)
  res["identity"] = {}
  for n in names[:2]:
    r = refs[n]
    pix = torch.from_numpy(r["pix"]).permute(1, 2, 0)[None].double()         # NHWC normalized
    img01 = torch.from_numpy(r["img01"])
    with torch.no_grad():
      o64 = a64(e64(None, pre_normalized=pix))[0].numpy()
      norm32 = e32.normalize(img01)
      o32 = a32(e32(img01))[0].numpy()
    res["identity"][n] = {
        "reauthor64_vs_hf64": metrics(o64, r["out64"]),
        "norm_step_max_abs_vs_hf_pixels": float((norm32[0].permute(2, 0, 1) - torch.from_numpy(r["pix"])).abs().max()),
        "reauthor32_vs_hf32": metrics(o32, r["out32"], r["out64"]),
    }
    print("identity", n, json.dumps(res["identity"][n]), flush=True)


def stage_calib(out, res, names, tensors=None):
  tensors = tensors if tensors is not None else uv.load_vision_tensors()
  e32, _, _ = uv.build(uv.Opts(), torch.float32, tensors=tensors)
  refs = refs_of(out, names)
  calib = {}
  for n in names:
    with torch.no_grad():
      e32(torch.from_numpy(refs[n]["img01"]), calib=calib)
  S = {k: uv.calib_S(v) for k, v in calib.items()}
  res["ln_absmax"] = {k: round(v, 3) for k, v in sorted(calib.items())}
  res["ln_S"] = S
  over = {k: v for k, v in calib.items() if v > 256.0}
  res["ln_absmax_over_256"] = {k: round(v, 1) for k, v in sorted(over.items())}
  json.dump({"S": S, "absmax": calib}, open(os.path.join(out, "ln_scales.json"), "w"), indent=1)
  print("ln absmax max", max(calib.values()), "over 256:", len(over), flush=True)
  return S


def stage_convert(out, res, names, opts, force, tensors=None):
  import litert_torch
  tensors = tensors if tensors is not None else uv.load_vision_tensors()
  enc, adp, info = uv.build(opts, torch.float32, tensors=tensors)
  res["export_opts"] = opts.as_dict()
  refs = refs_of(out, names)
  x0 = torch.from_numpy(refs[names[0]]["img01"])
  with torch.no_grad():
    f0 = enc(x0)
  P = lambda nm: os.path.join(out, nm)  # noqa: E731
  for key, mod, sample in (("enc", enc, x0), ("adp", adp, f0)):
    path = P(f"{PARTS[key]}.tflite")
    if force or not os.path.exists(path):
      t0 = time.time()
      litert_torch.convert(mod, (sample,)).export(path)
      res[f"{key}_convert_seconds"] = round(time.time() - t0, 1)
      print(f"converted {path} in {res[f'{key}_convert_seconds']} s", flush=True)
  del enc, adp
  insp = {}
  for key in PARTS:
    insp[f"{PARTS[key]}.tflite"] = tfl_inspect(P(f"{PARTS[key]}.tflite"))
    print(key, json.dumps({k: v for k, v in insp[f"{PARTS[key]}.tflite"].items() if k != "subgraphs"}), flush=True)
  renc, radp = Runner(P("vision_encoder.tflite")), Runner(P("vision_adapter.tflite"))
  res["parity_fp32"] = {}
  for n in names:
    r = refs[n]
    e = renc(r["img01"])
    a = radp(e)
    res["parity_fp32"][n] = {"enc_vs_hf32": metrics(e[0], r["feats32"], r["feats64"]),
                             "out_vs_hf32": metrics(a[0], r["out32"], r["out64"]),
                             "hf32_rel_rms_to_f64": metrics(r["out32"], r["out64"])["rel_rms"]}
    print("fp32", n, json.dumps(res["parity_fp32"][n]["out_vs_hf32"]), flush=True)
  return insp


def stage_quant(out, res, names, force):
  P = lambda nm: os.path.join(out, nm)  # noqa: E731
  insp = {}
  variants = {"enc_fp16": ("vision_encoder.tflite", "vision_encoder_fp16.tflite", quant_fp16),
              "enc_int8": ("vision_encoder.tflite", "vision_encoder_int8.tflite", quant_int8),
              "adp_int8": ("vision_adapter.tflite", "vision_adapter_int8.tflite", quant_int8),
              "adp_fp16": ("vision_adapter.tflite", "vision_adapter_fp16.tflite", quant_fp16)}
  for key, (src, dst, fn) in variants.items():
    if force or not os.path.exists(P(dst)):
      t0 = time.time()
      fn(P(src), P(dst))
      print(f"{key}: {dst} in {time.time() - t0:.1f} s", flush=True)
    insp[dst] = tfl_inspect(P(dst))
  refs = refs_of(out, names)
  res["quant_ab"] = {}
  for enc_name, adp_name in (("vision_encoder_fp16.tflite", "vision_adapter_int8.tflite"),
                             ("vision_encoder_int8.tflite", "vision_adapter_int8.tflite"),
                             ("vision_encoder.tflite", "vision_adapter_int8.tflite"),
                             ("vision_encoder_fp16.tflite", "vision_adapter_fp16.tflite")):
    renc, radp = Runner(P(enc_name)), Runner(P(adp_name))
    key = f"{enc_name[:-7]}+{adp_name[:-7]}"
    res["quant_ab"][key] = {}
    for n in names:
      a = radp(renc(refs[n]["img01"]))
      res["quant_ab"][key][n] = metrics(a[0], refs[n]["out32"], refs[n]["out64"])
    worst = min(v["cos_min"] for v in res["quant_ab"][key].values())
    corr = min(v["corr"] for v in res["quant_ab"][key].values())
    res["quant_ab"][key]["_min_corr"] = corr
    res["quant_ab"][key]["_worst_row_cos"] = worst
    print(f"A/B {key}: min corr {corr:.6f}, worst row cos {worst:.6f}", flush=True)
  return insp


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--stage", default="all",
                  choices=("fixtures", "ref", "identity", "calib", "convert", "quant", "all"))
  ap.add_argument("--rel-form", default="bias", choices=("bias", "aug"))
  ap.add_argument("--qsplit", type=int, default=1)
  ap.add_argument("--fp16safe", action="store_true")
  ap.add_argument("--force", action="store_true")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  torch.set_grad_enabled(False)
  torch.set_num_threads(4)
  rpath, ipath = os.path.join(args.out, "result.json"), os.path.join(args.out, "tflite_inspect.json")
  res = json.load(open(rpath)) if os.path.exists(rpath) else {}
  insp = json.load(open(ipath)) if os.path.exists(ipath) else {}
  import importlib.metadata as md
  res["versions"] = {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert",
                                                "ai-edge-quantizer", "torch", "torchvision", "pillow")}
  res["python"] = sys.version.split()[0]
  names = [n for n, _ in FIXTURES]
  st = args.stage
  tensors = None

  def save():
    json.dump(insp, open(ipath, "w"), indent=1)
    json.dump(res, open(rpath, "w"), indent=1)

  if st in ("fixtures", "all"):
    stage_fixtures(args.out, res)
    save()
  if st in ("ref", "all"):
    tensors = stage_ref(args.out, res, names)
    save()
  S = None
  if st in ("calib", "all"):
    S = stage_calib(args.out, res, names, tensors)
    save()
  if args.fp16safe and S is None:
    S = json.load(open(os.path.join(args.out, "ln_scales.json")))["S"]
  opts = uv.Opts(fp16safe=args.fp16safe, ln_s=S, rel_form=args.rel_form, qsplit=args.qsplit)
  if st in ("identity", "all"):
    stage_identity(args.out, res, names, opts, tensors)
    save()
  if st in ("convert", "all"):
    insp.update(stage_convert(args.out, res, names, opts, args.force, tensors))
    save()
  if st in ("quant", "all"):
    insp.update(stage_quant(args.out, res, names, args.force))
  res["files"] = {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
                  for p in sorted(glob.glob(os.path.join(args.out, "*.tflite")))}
  save()
  print("DONE", rpath, flush=True)


if __name__ == "__main__":
  main()
