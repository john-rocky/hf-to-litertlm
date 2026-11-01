#!/usr/bin/env python3
"""tower_tfl.py - check the LFM3-VL vision graphs that export_vision.py writes: ops, ranks, and the CPU outputs
against the source model's image features.

  python -I tower_tfl.py --tower-dir OUTDIR/tower --ref-dir REF/ref_npy --out tower_tfl_g512.json [--g 512]
      [--forms fp32,fp16] [--fixtures lfm3vl_work/fixtures] [--bar 0.999]

TOWER-DIR = export_vision.py's output: vision_{encoder,adapter}_g<G>_v2.tflite (fp32 weights) and *_v2_fp16.tflite
(fp16 weights); an *_v2_int8.tflite pair is checked too when present and named in --forms. REF-DIR = hf_ref.py --npy's
ref_npy folder (<REF-DIR>/g<G>/<picture>.{feat,hid}.npy, fp32 Transformers with sdpa attention).
Per file: op histogram, ops that a GPU or NPU delegate commonly refuses (GATHER, GATHER_ND, FLEX, CUSTOM, STABLEHLO,
BROADCAST_TO, RELU_0_TO_1; any op name with those words), max tensor rank, float activations of rank < 3 or without a
leading batch axis, elementwise ops with a rank-2 input, constant bytes by dtype, signatures and io.
Outputs (CPU, ai_edge_litert Interpreter with XNNPACK, 8 threads), on the fixture pictures as the runtime feeds them
(the G x G PNG as float(u8) / 255, NHWC): every encoder form -> every adapter form against HF `feat` (per-token cosine
minimum, max |d| / max |ref|); the encoder alone against HF `hid` (its raster order put back into HF's merge-block
order); the adapter alone on HF's own `hid`. A pair passes when its worst per-token cosine over the pictures is at
least --bar. Exit 0 = every pair that contains no fp16 / int8 encoder passed (those are reported, not required).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lfm3_vision as lv  # noqa: E402

THREADS = 8
BANNED = ("GATHER", "FLEX", "CUSTOM", "STABLEHLO", "BROADCAST_TO", "RELU_0_TO_1")
ELEMENTWISE = {"ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "POW", "RSQRT", "SQRT",
               "TANH", "LOGISTIC", "EXP", "GELU", "NEG", "ABS", "SUM", "MEAN"}


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def inspect(path):
  from ai_edge_litert import schema_py_generated as S
  m = S.Model.GetRootAs(open(path, "rb").read(), 0)
  opname = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
  tname = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}
  hist, low, ew2, max_rank, rank0 = Counter(), Counter(), Counter(), 0, []
  const_bytes = Counter()
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)

    def info(t):
      tt = sg.Tensors(t)
      shape = tt.ShapeAsNumpy().tolist() if tt.ShapeLength() else []
      b = m.Buffers(tt.Buffer()) if tt.Buffer() < m.BuffersLength() else None
      const = b is not None and (b.DataLength() > 0 or b.Offset() > 1)
      nb = (b.DataLength() if b.DataLength() > 0 else b.Size()) if const else 0
      return shape, tname.get(tt.Type(), str(tt.Type())), const, nb

    for t in range(sg.TensorsLength()):
      shape, dt, const, nb = info(t)
      max_rank = max(max_rank, len(shape))
      if const:
        const_bytes[dt] += nb
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      oc = m.OperatorCodes(op.OpcodeIndex())
      name = opname.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
      if name == "CUSTOM":
        name = "CUSTOM:" + (oc.CustomCode() or b"").decode()
      hist[name] += 1
      ins = [info(t) for t in op.InputsAsNumpy().tolist() if t >= 0]
      for s, dt, const, _ in ins:
        if not const and dt in ("FLOAT32", "FLOAT16") and len(s) == 0:
          rank0.append(name)
      for t in op.OutputsAsNumpy().tolist():
        s, dt, _, _ = info(t)
        if dt in ("FLOAT32", "FLOAT16") and (len(s) < 3 or s[0] != 1):
          low[f"{name} {s}"] += 1
      if name in ELEMENTWISE and any(len(s) == 2 for s, _, _, _ in ins):
        ew2[f"{name} " + " x ".join(f"{'C' if c else 'A'}{s}" for s, _, c, _ in ins)] += 1
  sg = m.Subgraphs(0)
  io = {"inputs": [sg.Tensors(t).ShapeAsNumpy().tolist() for t in sg.InputsAsNumpy().tolist()],
        "outputs": [sg.Tensors(t).ShapeAsNumpy().tolist() for t in sg.OutputsAsNumpy().tolist()]}
  sigs = []
  for s in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(s)
    sigs.append({"key": sd.SignatureKey().decode(), "inputs": [sd.Inputs(j).Name().decode() for j in range(sd.InputsLength())],
                 "outputs": [sd.Outputs(j).Name().decode() for j in range(sd.OutputsLength())]})
  banned = sorted(k for k in hist if any(w in k.upper() for w in BANNED))
  return {"bytes": os.path.getsize(path), "sha256": sha256(path), "n_ops": sum(hist.values()),
          "ops": dict(sorted(hist.items())), "banned_ops": banned, "max_rank": max_rank,
          "rank0_float_inputs": sorted(set(rank0)), "n_low_rank_float_outputs": sum(low.values()),
          "low_rank_float_outputs": dict(sorted(low.items())), "elementwise_rank2": dict(sorted(ew2.items())),
          "const_bytes_by_dtype": dict(const_bytes), "n_subgraphs": m.SubgraphsLength(), "signatures": sigs, "io": io}


class Runner:
  def __init__(self, path):
    from ai_edge_litert.interpreter import Interpreter
    self.it = Interpreter(model_path=path, num_threads=THREADS)
    self.it.allocate_tensors()
    self.i = self.it.get_input_details()[0]
    self.o = self.it.get_output_details()[0]

  def __call__(self, x):
    self.it.set_tensor(self.i["index"], np.ascontiguousarray(x, dtype=self.i["dtype"]))
    self.it.invoke()
    return self.it.get_tensor(self.o["index"]).copy()


def metrics(got, ref):
  g = np.asarray(got, dtype=np.float64).reshape(-1, ref.shape[-1])
  r = np.asarray(ref, dtype=np.float64).reshape(g.shape)
  cos = (g * r).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-30)
  md = float(np.abs(g - r).max())
  return {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "maxdiff": md,
          "rel": md / float(np.abs(r).max())}


def tower_files(tower_dir, g, form):
  t = "" if form == "fp32" else f"_{form}"
  return {k: os.path.join(tower_dir, f"vision_{k}_g{g}_v2{t}.tflite") for k in ("encoder", "adapter")}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tower-dir", required=True)
  ap.add_argument("--ref-dir", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--g", type=int, default=512)
  ap.add_argument("--forms", default="fp32,fp16")
  ap.add_argument("--fixtures", default=os.path.join(HERE, "..", "fixtures"))
  ap.add_argument("--bar", type=float, default=0.999)
  args = ap.parse_args()
  torch.set_grad_enabled(False)
  g, forms, fxdir = args.g, args.forms.split(","), os.path.abspath(args.fixtures)
  paths = {f: tower_files(args.tower_dir, g, f) for f in forms}
  res = {"g": g, "forms": forms, "bar": args.bar}
  import importlib.metadata as md
  res["versions"] = {p: md.version(p) for p in ("torch", "ai-edge-litert")}
  insp = {}
  for f, v in paths.items():
    for k, p in v.items():
      i = inspect(p)
      insp[os.path.basename(p)] = i
      print(f"{os.path.basename(p)}: {i['bytes']:,} B sha256 {i['sha256'][:16]} ops {i['n_ops']} banned {i['banned_ops']} "
            f"max_rank {i['max_rank']} rank0 {i['rank0_float_inputs']} io {i['io']}", flush=True)
  res["inspect"] = insp
  fx = json.load(open(os.path.join(fxdir, "fixtures.json")))
  ref_dir = os.path.join(args.ref_dir, f"g{g}")
  gh = g // 16
  imgs = {iid: (np.asarray(Image.open(os.path.join(fxdir, rec[f"g{g}"])).convert("RGB"), dtype=np.float32) / 255.0)[None]
          for iid, rec in fx["images"].items()}
  refs = {iid: {k: np.load(os.path.join(ref_dir, f"{iid}.{k}.npy")) for k in ("feat", "hid")} for iid in imgs}
  enc_out, enc_m = {}, {}
  for f in paths:
    r = Runner(paths[f]["encoder"])
    enc_out[f], enc_m[f] = {}, {}
    for iid, x in imgs.items():
      e = r(x)
      enc_out[f][iid] = e
      hid = lv.raster_to_merge(torch.from_numpy(e), gh, gh)[0].numpy()
      enc_m[f][iid] = metrics(hid, refs[iid]["hid"])
    del r
  pairs, adp_m = {}, {}
  for fa in paths:
    r = Runner(paths[fa]["adapter"])
    adp_m[fa] = {iid: metrics(r(lv.merge_to_raster(torch.from_numpy(refs[iid]["hid"])[None], gh, gh).numpy()),
                              refs[iid]["feat"]) for iid in imgs}
    for fe in paths:
      pairs[f"{fe}/{fa}"] = {iid: metrics(r(enc_out[fe][iid]), refs[iid]["feat"]) for iid in imgs}
    del r
  summ, required_ok = {}, True
  for c, row in pairs.items():
    cmin = min(v["cos_min"] for v in row.values())
    worst = min(row, key=lambda i: row[i]["cos_min"])
    ok = cmin >= args.bar
    summ[c] = {"cos_min": cmin, "worst_picture": worst, "rel_max": max(v["rel"] for v in row.values()), "pass": ok}
    if c.split("/")[0] == "fp32":
      required_ok &= ok
    print(f"G{g} encoder/adapter {c}: cos_min {cmin:.10f} ({worst}) rel_max {summ[c]['rel_max']:.2e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
  res["end2end"] = summ
  res["end2end_rows"] = pairs
  res["encoder_vs_hf_hid"] = {f: {"cos_min": min(v["cos_min"] for v in m.values()),
                                  "rel_max": max(v["rel"] for v in m.values())} for f, m in enc_m.items()}
  res["adapter_on_hf_hid"] = {f: {"cos_min": min(v["cos_min"] for v in m.values()),
                                  "rel_max": max(v["rel"] for v in m.values())} for f, m in adp_m.items()}
  print("encoder vs HF hid", json.dumps(res["encoder_vs_hf_hid"]), flush=True)
  print("adapter on HF hid", json.dumps(res["adapter_on_hf_hid"]), flush=True)
  res["ok"] = bool(required_ok)
  json.dump(res, open(args.out, "w"), indent=1)
  print("TOWER_TFL_DONE", g, "PASS" if required_ok else "FAIL", flush=True)
  return 0 if required_ok else 1


if __name__ == "__main__":
  sys.exit(main())
