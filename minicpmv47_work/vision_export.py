#!/usr/bin/env python3
"""r2 step 4: the fast_vlm vision tflites of MiniCPM-V-4.7-1B from the pure-torch re-author
(mv47_vision.py), op histogram, tflite parity vs HF, fp16 / int8 variants and the quant A/B.

  python vision_export.py OUT --src SRC --ref W/ref \
      --fixtures W/fixtures --ln-scales ln_scales.json --r2 W --stage all   (litert-torch 0.9.3 environment)

  VISION_ENCODER  OUT/vision_encoder.tflite      image NHWC [1,448,448,3] in [0,1] -> [1,256,1152]
  VISION_ADAPTER  OUT/vision_adapter.tflite      [1,256,1152] -> [1,64,1024]
  + _int8 (ai_edge_quantizer dynamic_wi8_afp32) and _fp16 (float_casting of FULLY_CONNECTED +
    CONV_2D weights; the recipe_manager dynamic-config route silently no-ops for float_casting
    and BATCH_MATMUL is weightless + rejected = qwen35vl_work/vision_quant_ab.py).

--stage convert : export fp32 (skipped if the file exists, unless --force), inspect, fp32 parity
--stage quant   : int8 + fp16 variants, inspect, A/B on cats/kitchen1/kitchen2/noise
--stage all     : both
The module is built with the export configuration of the eager parity (fp16-safe LN with the
S table of --ln-scales + LN->FC barrier). Writes OUT/result.json (precedent keys + more),
R2/quant_ab.md, R2/FILES.md (bytes / sha256), R2/tflite_inspect.json (full op histograms).
Inspection parses the flatbuffer itself (schema_py_generated), so the histogram is the graph's,
not a delegate's view.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter

import litert_torch  # noqa: F401
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mv47_vision as mv  # noqa: E402

NAMES = ("cats", "kitchen1", "kitchen2", "noise")
THREADS = 8
ELEMENTWISE = {"ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "POW",
               "RSQRT", "SQRT", "TANH", "LOGISTIC", "EXP", "GELU", "NEG", "ABS"}
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
    b = m.Buffers(tt.Buffer()) if tt.Buffer() < m.BuffersLength() else None
    const = b is not None and (b.DataLength() > 0 or b.Offset() > 1)
    nbytes = (b.DataLength() if b.DataLength() > 0 else b.Size()) if const else 0
    return shape, tname.get(tt.Type(), str(tt.Type())), const, nbytes

  hist, low_rank, ew_rank2, produced = Counter(), [], [], set()
  for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    oc = m.OperatorCodes(op.OpcodeIndex())
    name = opname.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
    if name == "CUSTOM":
      name = "CUSTOM:" + (oc.CustomCode() or b"").decode()
    hist[name] += 1
    for t in op.OutputsAsNumpy().tolist():
      produced.add(t)
      shape, dt, _, _ = tinfo(t)
      if dt in ("FLOAT32", "FLOAT16") and (len(shape) < 3 or shape[0] != 1):
        low_rank.append({"op": i, "name": name, "shape": shape})
    if name in ELEMENTWISE:
      for t in op.InputsAsNumpy().tolist():
        if t < 0:
          continue
        shape, dt, const, _ = tinfo(t)
        if len(shape) == 2:
          ew_rank2.append({"op": i, "name": name, "shape": shape, "const": const})
  const_types, const_bytes = Counter(), Counter()
  for t in range(sg.TensorsLength()):
    shape, dt, const, nb = tinfo(t)
    if const:
      const_types[dt] += 1
      const_bytes[dt] += nb
  sig = []
  for s in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(s)
    sig.append({"key": sd.SignatureKey().decode(),
                "inputs": [sd.Inputs(j).Name().decode() for j in range(sd.InputsLength())],
                "outputs": [sd.Outputs(j).Name().decode() for j in range(sd.OutputsLength())]})
  io = {"inputs": [{"shape": tinfo(t)[0], "dtype": tinfo(t)[1]} for t in sg.InputsAsNumpy().tolist()],
        "outputs": [{"shape": tinfo(t)[0], "dtype": tinfo(t)[1]} for t in sg.OutputsAsNumpy().tolist()]}
  bad = sorted(k for k in hist if any(w in k.upper() for w in ("GATHER", "FLEX", "CUSTOM", "STABLEHLO")))
  return {"bytes": os.path.getsize(path), "n_ops": sum(hist.values()), "n_types": len(hist),
          "ops": dict(sorted(hist.items())), "bad_ops": bad,
          "gather": sorted(k for k in hist if "GATHER" in k.upper()),
          "flex": sorted(k for k in hist if "FLEX" in k.upper()),
          "custom": sorted(k for k in hist if "CUSTOM" in k.upper()),
          "stablehlo": sorted(k for k in hist if "STABLEHLO" in k.upper()),
          "low_rank_float_activations": low_rank[:50], "n_low_rank_float_activations": len(low_rank),
          "elementwise_rank2_inputs": ew_rank2[:50], "n_elementwise_rank2_inputs": len(ew_rank2),
          "const_tensor_types": dict(const_types), "const_bytes_by_type": dict(const_bytes),
          "n_subgraphs": m.SubgraphsLength(), "signatures": sig, "io": io}


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
          "corr": float(np.corrcoef(g.reshape(-1), r.reshape(-1))[0, 1]),
          "maxdiff": float(np.abs(g - r).max())}


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
  ap.add_argument("--src", required=True)
  ap.add_argument("--ref", required=True)
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--ln-scales", required=True)
  ap.add_argument("--r2", required=True)
  ap.add_argument("--stage", default="all", choices=("convert", "quant", "all"))
  ap.add_argument("--force", action="store_true")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  torch.set_grad_enabled(False)
  P = lambda name: os.path.join(args.out, name)  # noqa: E731
  rpath = P("result.json")
  res = json.load(open(rpath)) if os.path.exists(rpath) else {}
  res.update({"img": mv.IMG, "grid": mv.GRID, "n_tok": mv.N_TOK, "mask_neg": mv.MASK_NEG,
              "versions": {"torch": torch.__version__}})
  try:
    import importlib.metadata as md
    for p in ("litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer"):
      res["versions"][p] = md.version(p)
    imgs = {n: (np.asarray(Image.open(os.path.join(args.fixtures, f"{n}_448.png")).convert("RGB"),
                           dtype=np.float32) / 255.0)[None] for n in NAMES}
    refs = {n: np.load(os.path.join(args.ref, f"{n}.npz")) for n in NAMES}
    insp_path = os.path.join(args.r2, "tflite_inspect.json")
    insp = json.load(open(insp_path)) if os.path.exists(insp_path) else {}

    if args.stage in ("convert", "all"):
      res["stage"] = "build"
      ln = json.load(open(args.ln_scales))["scales"]
      S = {k: v["S"] for k, v in ln.items()}
      res["ln_scales"] = {k: v for k, v in sorted(S.items()) if v > 1.0}
      res["ln_absmax_max"] = max(v["absmax"] for v in ln.values())
      opts = mv.Opts(barrier=True, fp16safe=True, ln_s=S)
      t0 = time.time()
      enc, adp, info = mv.build(args.src, opts)
      res["build_seconds"] = round(time.time() - t0, 1)
      x0 = torch.from_numpy(imgs["noise"])
      feat0 = enc(x0)
      eager = {}
      for n in NAMES:
        e = enc(torch.from_numpy(imgs[n]))
        eager[n] = (e.numpy(), adp(e).numpy())
      res["adp_eager_corr"] = float(np.corrcoef(eager["noise"][1].reshape(-1).astype(np.float64),
                                                refs["noise"]["feat"].reshape(-1).astype(np.float64))[0, 1])
      res["adp_eager_maxdiff"] = float(np.abs(eager["noise"][1].reshape(-1) - refs["noise"]["feat"].reshape(-1)).max())
      res["ref_absmax"] = float(np.abs(refs["noise"]["feat"]).max())

      for key, mod, sample in (("enc", enc, x0), ("adp", adp, feat0)):
        path = P(f"{PARTS[key]}.tflite")
        if args.force or not os.path.exists(path):
          res["stage"] = f"convert-{key}"
          t0 = time.time()
          litert_torch.convert(mod, (sample,)).export(path)
          res[f"{key}_convert_seconds"] = round(time.time() - t0, 1)
          print(f"converted {path} in {res[f'{key}_convert_seconds']} s", flush=True)
      del enc, adp

      res["stage"] = "inspect-fp32"
      for key in PARTS:
        insp[f"{PARTS[key]}.tflite"] = tfl_inspect(P(f"{PARTS[key]}.tflite"))
      json.dump(insp, open(insp_path, "w"), indent=1)
      res["enc_ops"] = {k: insp["vision_encoder.tflite"][k] for k in ("n_types", "n_ops", "flex", "custom", "gather", "stablehlo")}
      res["adp_ops"] = {k: insp["vision_adapter.tflite"][k] for k in ("n_types", "n_ops", "flex", "custom", "gather", "stablehlo")}
      for key in PARTS:
        i = insp[f"{PARTS[key]}.tflite"]
        print(key, "ops", i["ops"], "bad", i["bad_ops"], "low-rank float act", i["n_low_rank_float_activations"],
              "ew rank2 inputs", i["n_elementwise_rank2_inputs"], "io", i["io"], flush=True)

      res["stage"] = "parity-fp32"
      renc, radp = Runner(P("vision_encoder.tflite")), Runner(P("vision_adapter.tflite"))
      par = {}
      for n in NAMES:
        t0 = time.time()
        e = renc(imgs[n])
        t_enc = time.time() - t0
        a = radp(e)
        a_hf = radp(refs[n]["enc_last_hidden"])
        par[n] = {"encoder_vs_hf": metrics(e, refs[n]["enc_last_hidden"]),
                  "end2end_vs_hf": metrics(a, refs[n]["feat"]),
                  "adapter_on_hf_encoder": metrics(a_hf, refs[n]["feat"]),
                  "encoder_vs_eager_maxdiff": float(np.abs(e - eager[n][0]).max()),
                  "end2end_vs_eager_maxdiff": float(np.abs(a - eager[n][1]).max()),
                  "encoder_seconds": round(t_enc, 2)}
        print("fp32 tflite", n, {k: (round(v["corr"], 8), round(v["cos_min"], 8)) if isinstance(v, dict) else v
                                for k, v in par[n].items()}, flush=True)
      del renc, radp
      res["parity_fp32"] = par
      res["end2end_corr"] = min(par[n]["end2end_vs_hf"]["corr"] for n in NAMES)
      res["end2end_cos_min"] = min(par[n]["end2end_vs_hf"]["cos_min"] for n in NAMES)
      res["end2end_maxdiff"] = max(par[n]["end2end_vs_hf"]["maxdiff"] for n in NAMES)
      res["encoder_corr"] = min(par[n]["encoder_vs_hf"]["corr"] for n in NAMES)
      res["encoder_cos_min"] = min(par[n]["encoder_vs_hf"]["cos_min"] for n in NAMES)
      res["enc_mb"] = round(os.path.getsize(P("vision_encoder.tflite")) / 1e6, 1)
      res["adp_mb"] = round(os.path.getsize(P("vision_adapter.tflite")) / 1e6, 1)
      json.dump(res, open(rpath, "w"), indent=2)

    if args.stage in ("quant", "all"):
      for key in PARTS:
        src = P(f"{PARTS[key]}.tflite")
        for kind, fn in (("int8", quant_int8), ("fp16", quant_fp16)):
          dst = P(f"{PARTS[key]}_{kind}.tflite")
          if args.force or not os.path.exists(dst):
            res["stage"] = f"quant-{kind}-{key}"
            t0 = time.time()
            fn(src, dst)
            res[f"{key}_{kind}_seconds"] = round(time.time() - t0, 1)
            print(f"{kind} {dst} in {res[f'{key}_{kind}_seconds']} s", flush=True)
          res[f"{key}_{kind}_mb"] = round(os.path.getsize(dst) / 1e6, 1)
          insp[os.path.basename(dst)] = tfl_inspect(dst)
      json.dump(insp, open(insp_path, "w"), indent=1)

      res["stage"] = "ab"
      enc_out, ab_enc = {}, {}
      for kind in ("fp32", "fp16", "int8"):
        name = "vision_encoder.tflite" if kind == "fp32" else f"vision_encoder_{kind}.tflite"
        r = Runner(P(name))
        enc_out[kind] = {n: r(imgs[n]) for n in NAMES}
        ab_enc[kind] = {n: metrics(enc_out[kind][n], refs[n]["enc_last_hidden"]) for n in NAMES}
        del r
      ab, ab_adp = {}, {}
      for akind in ("fp32", "fp16", "int8"):
        name = "vision_adapter.tflite" if akind == "fp32" else f"vision_adapter_{akind}.tflite"
        r = Runner(P(name))
        ab_adp[akind] = {n: metrics(r(refs[n]["enc_last_hidden"]), refs[n]["feat"]) for n in NAMES}
        for ekind in ("fp32", "fp16", "int8"):
          ab[f"{ekind}/{akind}"] = {n: metrics(r(enc_out[ekind][n]), refs[n]["feat"]) for n in NAMES}
        del r
      res["ab"] = ab
      res["ab_encoder_only"] = ab_enc
      res["ab_adapter_on_hf_encoder"] = ab_adp
      res["end2end_int8_corr"] = min(ab["int8/int8"][n]["corr"] for n in NAMES)
      res["end2end_int8_maxdiff"] = max(ab["int8/int8"][n]["maxdiff"] for n in NAMES)
      res["ship_fp16enc_int8adp_corr_min"] = min(ab["fp16/int8"][n]["corr"] for n in NAMES)
      for c in ("fp32/fp32", "fp16/int8", "int8/int8", "fp16/fp16"):
        print("AB", c, {n: (round(ab[c][n]["corr"], 6), round(ab[c][n]["cos_min"], 6)) for n in NAMES}, flush=True)

      L = ["# r2 quant A/B — MiniCPM-V-4.7 vision tflites vs HF fp32 `get_image_features`", "",
           "CPU `ai_edge_litert.Interpreter` (XNNPACK, %d threads), fixtures = `W/fixtures/*_448.png`, "
           "reference = `W/ref/<name>.npz` (HF fp32 eager). Pearson corr over all elements / per-token cos min. "
           "Generated by `vision_export.py --stage quant`." % THREADS, "",
           "## the shipped forms (2026-10-13)", "",
           "| form | enc / adp | " + " | ".join(NAMES) + " | min corr | min cos |", "|---|---|" + "---|" * (len(NAMES) + 2)]
      for label, c in (("ship (desktop / iPhone Metal)", "fp16/fp16"), ("phone (Mali; fp16 encoder reboots Mali-CL)", "int8/fp16"),
                       ("record only, not shipped (dynamic int8 adapter)", "fp16/int8")):
        row = ab[c]
        L.append(f"| {label} | {c} | " + " | ".join(f"{row[n]['corr']:.6f} / {row[n]['cos_min']:.6f}" for n in NAMES)
                 + f" | {min(row[n]['corr'] for n in NAMES):.6f} | {min(row[n]['cos_min'] for n in NAMES):.6f} |")
      L += ["",
           "## end2end (encoder variant / adapter variant) -> `[64,1024]` vs HF `pooler_output[0]`", "",
           "| enc / adp | " + " | ".join(NAMES) + " | min corr | min cos |", "|---|" + "---|" * (len(NAMES) + 2)]
      order = ["fp32/fp32", "fp16/int8", "int8/int8", "fp16/fp16", "fp32/int8", "fp32/fp16", "int8/fp32",
               "int8/fp16", "fp16/fp32"]
      for c in order:
        row = ab[c]
        L.append(f"| {c} | " + " | ".join(f"{row[n]['corr']:.6f} / {row[n]['cos_min']:.6f}" for n in NAMES)
                 + f" | {min(row[n]['corr'] for n in NAMES):.6f} | {min(row[n]['cos_min'] for n in NAMES):.6f} |")
      L += ["", "cells = corr / per-token cos min.", "",
            "## encoder only -> `[1,256,1152]` vs HF `last_hidden_state` (post_layernorm)", "",
            "| encoder | " + " | ".join(NAMES) + " | min corr |", "|---|" + "---|" * (len(NAMES) + 1)]
      for k in ("fp32", "fp16", "int8"):
        L.append(f"| {k} | " + " | ".join(f"{ab_enc[k][n]['corr']:.6f} / {ab_enc[k][n]['cos_min']:.6f}" for n in NAMES)
                 + f" | {min(ab_enc[k][n]['corr'] for n in NAMES):.6f} |")
      L += ["", "## adapter only, on HF's own encoder output", "",
            "| adapter | " + " | ".join(NAMES) + " | min corr |", "|---|" + "---|" * (len(NAMES) + 1)]
      for k in ("fp32", "fp16", "int8"):
        L.append(f"| {k} | " + " | ".join(f"{ab_adp[k][n]['corr']:.6f} / {ab_adp[k][n]['cos_min']:.6f}" for n in NAMES)
                 + f" | {min(ab_adp[k][n]['corr'] for n in NAMES):.6f} |")
      L += ["", "## files", "", "| file | MB |", "|---|---|"]
      for key in PARTS:
        for kind in ("", "_fp16", "_int8"):
          f = f"{PARTS[key]}{kind}.tflite"
          L.append(f"| {f} | {os.path.getsize(P(f)) / 1e6:.1f} |")
      open(os.path.join(args.r2, "quant_ab.md"), "w").write("\n".join(L) + "\n")

      rows = ["# r2 files — bytes / sha256", "", "Written by `vision_export.py` (%s)." % time.strftime("%Y-%m-%d %H:%M %Z"),
              "", "| path | bytes | sha256 |", "|---|---|---|"]
      for key in PARTS:
        for kind in ("", "_fp16", "_int8"):
          f = P(f"{PARTS[key]}{kind}.tflite")
          rows.append(f"| `{os.path.relpath(f, os.path.dirname(args.r2))}` | {os.path.getsize(f):,} | `{sha256(f)}` |")
      for f in sorted(os.listdir(args.fixtures)):
        p = os.path.join(args.fixtures, f)
        rows.append(f"| `{os.path.relpath(p, os.path.dirname(args.r2))}` | {os.path.getsize(p):,} | `{sha256(p)}` |")
      for f in ("ln_scales.json", "parity_eager.json", "quant_ab.md", "tflite_inspect.json"):
        p = os.path.join(args.r2, f)
        if os.path.exists(p):
          rows.append(f"| `{os.path.relpath(p, os.path.dirname(args.r2))}` | {os.path.getsize(p):,} | `{sha256(p)}` |")
      open(os.path.join(args.r2, "FILES.md"), "w").write("\n".join(rows) + "\n")
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    res["ok"] = False
    res["error_type"] = type(e).__name__
    res["error_head"] = (str(e).strip().splitlines() or ["?"])[0][:400]
    with open(P("trace.txt"), "w") as f:
      f.write(traceback.format_exc())
    print("ERROR", res["error_type"], res["error_head"], flush=True)
  json.dump(res, open(rpath, "w"), indent=2)
  print("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k not in ("parity_fp32", "ab", "ab_encoder_only", "ab_adapter_on_hf_encoder")}),
        flush=True)


if __name__ == "__main__":
  main()
