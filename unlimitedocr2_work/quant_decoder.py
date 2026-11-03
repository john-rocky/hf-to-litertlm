#!/usr/bin/env python3
"""Quantize the float decoder / embedder TFLite of Unlimited-OCR-2 with ai-edge-quantizer and census the result.

  python quant_decoder.py SRC DST --form FORM [--block 32] [--json OUT.json]
  python quant_decoder.py --census FILE.tflite [--json OUT.json]

Forms (the latest matching recipe entry wins in ai-edge-quantizer, so the general entry comes first):
  int8        every FULLY_CONNECTED int8 channelwise, INTEGER compute (dynamic-range: CPU quantizes the activations
              per token at run time)                                                    = dynamic_wi8_afp32
  int8w       the same weights, FLOAT compute + explicit DEQUANTIZE (weight-only; a control for the CPU path)
  int4e       int8 as above, then the routed experts' two fused FCs (ExpertsGateUp / ExpertsDown in the op scope)
              int4 BLOCKWISE_<block> symmetric, INTEGER compute
  int4ew      int4e with FLOAT compute + explicit DEQUANTIZE everywhere (weight-only control; ai-edge-quantizer 0.9.0
              refuses int4 BLOCKWISE with FLOAT compute = not buildable)
  fp16w       every FULLY_CONNECTED weight float16 (float_casting), FLOAT compute
  emb8       EMBEDDING_LOOKUP int8 channelwise (the embedder TFLite)
Every form leaves the DASH FCs float (a last no_quantize entry for the op scopes RopeTable / DashQ / DashK: the
one-hot cos / sin table and the router q / k of uocr_dash.py). Every blockwise int4 scale is floored at 2^-14 before
the weights are quantized (a block of zero weights would get a zero scale); the count of floored scales is reported. The census lists, per signature, the op histogram, the FULLY_CONNECTED weight types split by the
module class in the op's name (ExpertsGateUp / ExpertsDown / Router / LMHead / GateUp / Down / attention), and any op
whose name contains GATHER / FLEX / CUSTOM / STABLEHLO.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

FLOOR = np.float32(2.0 ** -14)
STATS = {"blockwise4_calls": 0, "floored_tensors": 0, "floored_scales": 0}


def install_floor():
  from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
  orig = u.tensor_zp_scale_from_min_max

  def clamped(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs):
    zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs)
    if u.is_blockwise(granularity) and num_bits == 4:
      STATS["blockwise4_calls"] += 1
      if not symmetric:
        raise ValueError("asymmetric blockwise int4 is not covered by the floor")
      low = scale < FLOOR
      n = int(np.count_nonzero(low))
      if n:
        scale = np.where(low, FLOOR, scale).astype(scale.dtype)
        STATS["floored_tensors"] += 1
        STATS["floored_scales"] += n
    return zp, scale

  u.tensor_zp_scale_from_min_max = clamped


def entry(regex, op, bits, gran, compute="INTEGER", explicit=False):
  return {"regex": regex, "operation": op, "algorithm_key": "min_max_uniform_quantize",
          "op_config": {"weight_tensor_config": {"num_bits": bits, "symmetric": True, "granularity": gran,
                                                 "dtype": "INT"},
                        "compute_precision": compute, "explicit_dequantize": explicit, "skip_checks": False,
                        "min_weight_elements": 0}}


KEEP_FLOAT = {"regex": ".*(RopeTable|DashQ|DashK).*", "operation": "FULLY_CONNECTED", "algorithm_key": "no_quantize",
              "op_config": {}}


def recipe(form, block):
  bw = f"BLOCKWISE_{block}"
  experts = ".*Experts(GateUp|Down).*"
  if form == "int8":
    return [entry(".*", "FULLY_CONNECTED", 8, "CHANNELWISE"), KEEP_FLOAT]
  if form == "int8w":
    return [entry(".*", "FULLY_CONNECTED", 8, "CHANNELWISE", "FLOAT", True), KEEP_FLOAT]
  if form == "int4e":
    return [entry(".*", "FULLY_CONNECTED", 8, "CHANNELWISE"), entry(experts, "FULLY_CONNECTED", 4, bw), KEEP_FLOAT]
  if form == "int4ew":
    return [entry(".*", "FULLY_CONNECTED", 8, "CHANNELWISE", "FLOAT", True),
            entry(experts, "FULLY_CONNECTED", 4, bw, "FLOAT", True), KEEP_FLOAT]
  if form == "fp16w":
    # every FC weight float16 (float_casting), FLOAT compute
    return [{"regex": ".*", "operation": "FULLY_CONNECTED", "algorithm_key": "float_casting",
             "op_config": {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE",
                                                    "dtype": "FLOAT"},
                           "compute_precision": "FLOAT", "explicit_dequantize": False}}, KEEP_FLOAT]
  if form == "emb8":
    return [entry(".*", "EMBEDDING_LOOKUP", 8, "CHANNELWISE")]
  raise SystemExit(f"unknown form {form}")


CLASSES = ("ExpertsGateUp", "ExpertsDown", "RopeTable", "DashQ", "DashK", "Router", "LMHead", "GateUp", "Down",
           "q_proj", "k_proj", "v_proj", "o_proj")


def fc_class(name):
  for c in CLASSES:
    if re.search(rf"\b{c}\b|{c}[_;/]|\.{c}", name):
      return c
  return "other"


def census(path):
  from ai_edge_litert import schema_py_generated as S
  buf = open(path, "rb").read()
  m = S.Model.GetRootAs(buf, 0)
  opname = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
  tname = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}
  sig_of = {}
  for i in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(i)
    sig_of[sd.SubgraphIndex()] = sd.SignatureKey().decode()
  out = {"bytes": os.path.getsize(path), "signatures": {}}
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    hist = Counter()
    fc = defaultdict(Counter)
    blocks = Counter()
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      oc = m.OperatorCodes(op.OpcodeIndex())
      name = opname.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
      if name == "CUSTOM":
        name = "CUSTOM:" + (oc.CustomCode() or b"").decode()
      hist[name] += 1
      if name == "FULLY_CONNECTED":
        wt = sg.Tensors(op.Inputs(1))
        oname = (sg.Tensors(op.Outputs(0)).Name() or b"").decode()
        wtype = tname.get(wt.Type())
        q = wt.Quantization()
        if q is not None and q.DetailsType() != 0:
          blocks["blockwise"] += 1
        fc[fc_class(oname)][wtype] += 1
    io = {}
    for kind, idxs in (("inputs", sg.InputsAsNumpy().tolist()), ("outputs", sg.OutputsAsNumpy().tolist())):
      io[kind] = {(sg.Tensors(t).Name() or b"").decode(): [sg.Tensors(t).ShapeAsNumpy().tolist(),
                                                           tname.get(sg.Tensors(t).Type())] for t in idxs}
    bad = sorted(k for k in hist if any(w in k.upper() for w in ("GATHER", "FLEX", "CUSTOM", "STABLEHLO")))
    out["signatures"][sig_of.get(si, f"subgraph_{si}")] = {
        "n_ops": sum(hist.values()), "ops": dict(sorted(hist.items())),
        "fc_weights": {k: dict(v) for k, v in sorted(fc.items())}, "fc_blockwise_details": dict(blocks),
        "bad_ops": bad, "io": io}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src", nargs="?")
  ap.add_argument("dst", nargs="?")
  ap.add_argument("--form")
  ap.add_argument("--block", type=int, default=32)
  ap.add_argument("--census")
  ap.add_argument("--json")
  a = ap.parse_args()
  if a.census:
    res = census(a.census)
  else:
    install_floor()
    from ai_edge_quantizer import quantizer
    rec = recipe(a.form, a.block)
    q = quantizer.Quantizer(a.src, rec)
    q.quantize().export_model(a.dst, overwrite=True)
    res = {"form": a.form, "block": a.block, "recipe": rec, "floor": dict(STATS), "census": census(a.dst)}
    print("QUANT_DONE", a.form, json.dumps(STATS), flush=True)
  if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
  else:
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
  sys.exit(main())
