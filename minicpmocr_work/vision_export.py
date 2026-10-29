#!/usr/bin/env python3
"""The fast_vlm vision TFLite pair of MiniCPM-OCR-1.0 (4x path, GxG) from the pure-torch re-author
(ocr_vision.py): export, op histogram, parity against the HF references of vision_check.py, fp16 / int8 variants.

  python vision_export.py OUT --snap SNAPSHOT --refs REF_DIR --ln-scales REF_DIR/ln_scales.json [--g 448]
                          [--stage convert|quant|all] [--force]          (litert-torch 0.9.3 environment)

  OUT/vision_encoder.tflite   image NHWC [1,G,G,3] in [0,1] -> [1,(G/14)^2,1152]
  OUT/vision_adapter.tflite   [1,(G/14)^2,1152] -> [1,(G/28)^2,1024]
  + _fp16 (float_casting of FULLY_CONNECTED + CONV_2D weights) and _int8 (dynamic_wi8_afp32), the recipes of the
    MiniCPM-V-4.7-1B conversion (minicpmv47_work/vision_export.py in hf-to-litertlm).
The module uses the fp16-safe LayerNorm with the S table and the LN->FC barrier. Parity = the TFLite pair on the
XNNPACK CPU against HF get_image_features (pooler_output) for every ref_*.npz. Writes OUT/result.json and
OUT/tflite_inspect.json (op histograms parsed from the flatbuffer).
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter

import litert_torch  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_vision as ov  # noqa: E402

THREADS = 8
PARTS = {"enc": "vision_encoder", "adp": "vision_adapter"}


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def tfl_inspect(path):
  from ai_edge_litert import schema_py_generated as S
  buf = open(path, "rb").read()
  m = S.Model.GetRootAs(buf, 0)
  opname = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
  tname = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}
  sg = m.Subgraphs(0)

  def tinfo(t):
    tt = sg.Tensors(t)
    shape = tt.ShapeAsNumpy().tolist() if tt.ShapeLength() else []
    return shape, tname.get(tt.Type(), str(tt.Type()))

  hist = Counter()
  for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    oc = m.OperatorCodes(op.OpcodeIndex())
    name = opname.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
    if name == "CUSTOM":
      name = "CUSTOM:" + (oc.CustomCode() or b"").decode()
    hist[name] += 1
  io = {"inputs": [{"shape": tinfo(t)[0], "dtype": tinfo(t)[1]} for t in sg.InputsAsNumpy().tolist()],
        "outputs": [{"shape": tinfo(t)[0], "dtype": tinfo(t)[1]} for t in sg.OutputsAsNumpy().tolist()]}
  bad = sorted(k for k in hist if any(w in k.upper() for w in ("GATHER", "FLEX", "CUSTOM", "STABLEHLO")))
  return {"bytes": os.path.getsize(path), "n_ops": sum(hist.values()), "ops": dict(sorted(hist.items())),
          "bad_ops": bad, "io": io}


class Runner:
  def __init__(self, path):
    from ai_edge_litert.interpreter import Interpreter
    self.it = Interpreter(model_path=path, num_threads=THREADS)
    self.it.allocate_tensors()
    self.inp = self.it.get_input_details()[0]
    self.out = self.it.get_output_details()[0]

  def __call__(self, x):
    self.it.set_tensor(self.inp["index"], np.ascontiguousarray(x, dtype=self.inp["dtype"]))
    self.it.invoke()
    return self.it.get_tensor(self.out["index"]).copy()


def metrics(got, ref):
  g = np.asarray(got, dtype=np.float64).reshape(-1, np.asarray(ref).shape[-1])
  r = np.asarray(ref, dtype=np.float64).reshape(g.shape)
  cos = (g * r).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
          "corr": float(np.corrcoef(g.reshape(-1), r.reshape(-1))[0, 1]), "maxdiff": float(np.abs(g - r).max())}


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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--snap", required=True)
  ap.add_argument("--refs", required=True)
  ap.add_argument("--ln-scales", required=True)
  ap.add_argument("--g", type=int, default=448)
  ap.add_argument("--stage", default="all", choices=("convert", "quant", "all"))
  ap.add_argument("--force", action="store_true")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  torch.set_grad_enabled(False)
  P = lambda name: os.path.join(args.out, name)  # noqa: E731
  rpath, ipath = P("result.json"), P("tflite_inspect.json")
  res = json.load(open(rpath)) if os.path.exists(rpath) else {}
  insp = json.load(open(ipath)) if os.path.exists(ipath) else {}
  import importlib.metadata as md
  res["versions"] = {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert",
                                                "ai-edge-quantizer", "torch")}
  res["g"] = args.g
  refs = {os.path.basename(p)[4:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(args.refs, "ref_*.npz")))}
  assert refs, args.refs
  names = list(refs)

  if args.stage in ("convert", "all"):
    S = json.load(open(args.ln_scales))["S"]
    res["ln_scales_gt1"] = {k: v for k, v in sorted(S.items()) if v > 1.0}
    enc, adp, info = ov.build(args.snap, ov.Opts(barrier=True, fp16safe=True, ln_s=S), args.g)
    res["build_info"] = info
    x0 = torch.from_numpy(refs[names[0]]["img01"])[None]
    feat0 = enc(x0)
    for key, mod, sample in (("enc", enc, x0), ("adp", adp, feat0)):
      path = P(f"{PARTS[key]}.tflite")
      if args.force or not os.path.exists(path):
        t0 = time.time()
        litert_torch.convert(mod, (sample,)).export(path)
        res[f"{key}_convert_seconds"] = round(time.time() - t0, 1)
        print(f"converted {path} in {res[f'{key}_convert_seconds']} s", flush=True)
    del enc, adp
    for key in PARTS:
      insp[f"{PARTS[key]}.tflite"] = tfl_inspect(P(f"{PARTS[key]}.tflite"))
      print(key, insp[f"{PARTS[key]}.tflite"], flush=True)
    renc, radp = Runner(P("vision_encoder.tflite")), Runner(P("vision_adapter.tflite"))
    res["parity_fp32"] = {}
    for n in names:
      e = renc(refs[n]["img01"][None])
      a = radp(e)
      res["parity_fp32"][n] = {"enc": metrics(e[0], refs[n]["enc_ref"]), "adp": metrics(a[0], refs[n]["pooler_ref"])}
      print("fp32", n, json.dumps(res["parity_fp32"][n]["adp"]), flush=True)

  if args.stage in ("quant", "all"):
    variants = {"enc_fp16": ("vision_encoder.tflite", "vision_encoder_fp16.tflite", quant_fp16),
                "enc_int8": ("vision_encoder.tflite", "vision_encoder_int8.tflite", quant_int8),
                "adp_fp16": ("vision_adapter.tflite", "vision_adapter_fp16.tflite", quant_fp16)}
    for key, (src, dst, fn) in variants.items():
      if args.force or not os.path.exists(P(dst)):
        t0 = time.time()
        fn(P(src), P(dst))
        print(f"{key}: {dst} in {time.time() - t0:.1f} s", flush=True)
      insp[dst] = tfl_inspect(P(dst))
    res["quant_ab"] = {}
    for enc_name, adp_name in (("vision_encoder_fp16.tflite", "vision_adapter_fp16.tflite"),
                               ("vision_encoder_int8.tflite", "vision_adapter_fp16.tflite")):
      renc, radp = Runner(P(enc_name)), Runner(P(adp_name))
      key = f"{enc_name.split('_')[-1][:-7]}/{adp_name.split('_')[-1][:-7]}"
      res["quant_ab"][key] = {}
      for n in names:
        a = radp(renc(refs[n]["img01"][None]))
        res["quant_ab"][key][n] = metrics(a[0], refs[n]["pooler_ref"])
      worst = min(v["cos_min"] for v in res["quant_ab"][key].values())
      corr = min(v["corr"] for v in res["quant_ab"][key].values())
      print(f"A/B {key}: worst token cos {worst:.6f}, min corr {corr:.6f}", flush=True)

  res["files"] = {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
                  for p in sorted(glob.glob(P("*.tflite")))}
  json.dump(insp, open(ipath, "w"), indent=1)
  json.dump(res, open(rpath, "w"), indent=1)
  print("DONE", rpath, flush=True)


if __name__ == "__main__":
  main()
