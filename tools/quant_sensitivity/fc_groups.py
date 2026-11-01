#!/usr/bin/env python3
"""fc_groups.py - group a float decoder graph's FULLY_CONNECTED ops by the Linear they come from, and build forms that
quantize one group at a time (int8, the rest fp16), with recipes that match each group's ops exactly.

  python -I fc_groups.py list --float FLOAT.tflite --classes CLASSES.json --out fc_ops.json
  python -I fc_groups.py recipe --ops fc_ops.json --classes CLASSES.json --int8 GROUP[,GROUP] --out RECIPE.json
  python -I fc_groups.py build --float FLOAT.tflite --recipe RECIPE.json --ops fc_ops.json --classes CLASSES.json
      --out FORM.tflite [--record FORM.json]
  python -I fc_groups.py quantize --src IN.tflite --recipe RECIPE.json --out OUT.tflite     (any graph, any recipe)

list: every FULLY_CONNECTED op of every signature (read through mmap), its weight buffer and shape and its quantizer
    scope (ai-edge-quantizer's op scope = the op's output tensor names joined by ';' plus a trailing ';'). The converter
    fuses names, so the first Linear name in a scope is not always the op's own (one signature's v_proj scope starts
    with v_proj, another's with self_attn; w3's scope carries w1 after it): each op is classified by the ordered rules
    of CLASSES.json (first match wins), its layer read with "layer_regex", its weight shape checked against "shapes".
    Every weight buffer must get one class and one layer only (the quantizer gives every op that shares a weight the
    same config, so a buffer with two classes cannot be split): the command stops otherwise.
recipe: base = every FULLY_CONNECTED fp16 (float_casting); then one entry per int8 group (min_max_uniform_quantize
    int8 symmetric channelwise, compute INTEGER = the fully connected entry of dynamic_wi8_afp32). A group's regex is
    the exact scopes of its ops, `^(?:<scope>|...)$` (re.escape), so it never catches a neighbour through a fused name.
    Groups: the names under "groups" in CLASSES.json (sets of classes), L<a>-<b> (every FC of layers a..b) and
    <group or class>@L<a>-<b>. Writes RECIPE.json and RECIPE.info.json (ops, weights and parameters per group).
build: ai-edge-quantizer on the float graph with the recipe, written to FORM.tflite; then the bytes of every constant
    by dtype and the dtype of every FC weight per class in the decode signature, read back from the built graph (a
    group the quantizer skipped shows here as fp16).
CLASSES.json: {"layer_regex": "...(\\d+)...", "rules": [{"class": C, one of "first_name_contains": S |
    "linear_any": NAME | "linear_first": NAME | "linear_only": [NAMES]}, ...], "linear_regex": "Linear_(\\w+?)(?=;|$)",
    "shapes": {C: [out, in]}, "groups": {G: [C, ...]}}. Example: lfm3vl_work/quant_sens_classes.json (LFM3).
Needs ai-edge-litert (the flatbuffer schema) and ai-edge-quantizer.
"""
import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import time

FP16 = {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"}
INT8 = {"num_bits": 8, "symmetric": True, "granularity": "CHANNELWISE", "dtype": "INT"}


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def classifier(classes):
  lin_re = re.compile(classes.get("linear_regex", r"Linear_(\w+?)(?=;|$)"))
  rules = classes["rules"]

  def classify(scope):
    lin = lin_re.findall(scope)
    first = scope.split(";")[0]
    for r in rules:
      if ("first_name_contains" in r and r["first_name_contains"] in first) or \
         ("linear_any" in r and r["linear_any"] in lin) or \
         ("linear_first" in r and lin and lin[0] == r["linear_first"]) or \
         ("linear_only" in r and lin == r["linear_only"]):
        return r["class"]
    raise ValueError(f"unclassified FC scope: {scope[:200]}")
  return classify


def read_model(path):
  from ai_edge_litert import schema_py_generated as schema
  f = open(path, "rb")
  buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  m = schema.Model.GetRootAsModel(buf, 0)
  op_names = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}
  codes = [op_names.get(max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode()))
           for i in range(m.OperatorCodesLength())]
  sig = {m.SignatureDefs(i).SubgraphIndex(): m.SignatureDefs(i).SignatureKey().decode()
         for i in range(m.SignatureDefsLength())}
  return schema, m, codes, sig


def list_ops(path, classes):
  _schema, m, codes, sig = read_model(path)
  classify = classifier(classes)
  layer_re = re.compile(classes["layer_regex"])
  ops = []
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      if codes[op.OpcodeIndex()] != "FULLY_CONNECTED":
        continue
      w = sg.Tensors(op.Inputs(1))
      names = [sg.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()) if op.Outputs(j) != -1]
      scope = ";".join(n for n in names if n) + ";"
      cls = classify(scope)
      shape = [w.Shape(j) for j in range(w.ShapeLength())]
      want = classes.get("shapes", {}).get(cls)
      if want is not None and shape != want:
        raise ValueError(f"{cls} shape {shape} != {want}: {scope[:200]}")
      layer = layer_re.search(scope.split(";")[0])
      ops.append({"signature": sig.get(si, str(si)), "op": oi, "class": cls, "layer": int(layer.group(1)) if layer else None,
                  "buffer": w.Buffer(), "shape": shape, "scope": scope})
  by_buf = {}
  for o in ops:
    by_buf.setdefault(o["buffer"], set()).add((o["class"], o["layer"]))
  bad = {b: sorted(map(str, v)) for b, v in by_buf.items() if len(v) != 1}
  if bad:
    raise ValueError(f"weight buffers with more than one class or layer: {bad}")
  return ops


def select(ops, group, groups):
  m = re.fullmatch(r"(?:(\w+)@)?L(\d+)-(\d+)", group)
  if m:
    cls, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    allowed = set(groups.get(cls, [cls])) if cls else None
    return [o for o in ops if o["layer"] is not None and lo <= o["layer"] <= hi and (allowed is None or o["class"] in allowed)]
  if group in groups:
    return [o for o in ops if o["class"] in set(groups[group])]
  if any(o["class"] == group for o in ops):
    return [o for o in ops if o["class"] == group]
  raise SystemExit(f"unknown group {group}")


def entry(regex, cfg, algo="min_max_uniform_quantize"):
  return {"regex": regex, "operation": "FULLY_CONNECTED", "algorithm_key": algo,
          "op_config": {"weight_tensor_config": cfg, "compute_precision": "FLOAT" if cfg["dtype"] == "FLOAT" else "INTEGER",
                        "explicit_dequantize": False, "skip_checks": False, "min_weight_elements": 0}}


def scopes_regex(sel):
  return "^(?:" + "|".join(sorted({re.escape(o["scope"]) for o in sel})) + ")$"


def make_recipe(a, classes):
  ops = json.load(open(a.ops))["ops"]
  rec = [entry(".*", FP16, "float_casting")]
  info = {"int8": []}
  for g in [x for x in a.int8.split(",") if x]:
    sel = select(ops, g, classes.get("groups", {}))
    if not sel:
      raise SystemExit(f"group {g} selects no op")
    rec.append(entry(scopes_regex(sel), INT8))
    uniq = {o["buffer"]: o for o in sel}
    info["int8"].append({"group": g, "ops": len(sel), "weights": len(uniq),
                         "params": sum(o["shape"][0] * o["shape"][1] for o in uniq.values())})
  json.dump(rec, open(a.out, "w"), indent=1)
  json.dump(info, open(a.out[:-5] + ".info.json" if a.out.endswith(".json") else a.out + ".info.json", "w"), indent=1)
  print(f"RECIPE {a.out}: " + json.dumps(info), flush=True)
  return 0


def const_bytes(path):
  schema, m, _codes, _sig = read_model(path)
  tn = {v: k for k, v in schema.TensorType.__dict__.items() if not k.startswith("_")}
  seen, out = set(), {}
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for t in range(sg.TensorsLength()):
      tt = sg.Tensors(t)
      b = tt.Buffer()
      if b == 0 or b in seen or b >= m.BuffersLength():
        continue
      bb = m.Buffers(b)
      n = bb.DataLength() if bb.DataLength() > 0 else (bb.Size() if bb.Offset() > 1 else 0)
      if n:
        seen.add(b)
        k = tn.get(tt.Type(), str(tt.Type()))
        out[k] = out.get(k, 0) + n
  return out


def fc_weight_dtypes(path, classes, signature="decode"):
  """Built graph: dtype of every FC weight constant per class (through DEQUANTIZE when the FC reads one)."""
  schema, m, codes, sig = read_model(path)
  classify = classifier(classes)
  layer_re = re.compile(classes["layer_regex"])
  tn = {v: k for k, v in schema.TensorType.__dict__.items() if not k.startswith("_")}
  out = {}
  for si in range(m.SubgraphsLength()):
    if sig.get(si) != signature:
      continue
    sg = m.Subgraphs(si)
    producer = {}
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      for j in range(op.OutputsLength()):
        producer[op.Outputs(j)] = op
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      if codes[op.OpcodeIndex()] != "FULLY_CONNECTED":
        continue
      t = op.Inputs(1)
      w = sg.Tensors(t)
      if w.Buffer() == 0 and t in producer and codes[producer[t].OpcodeIndex()] == "DEQUANTIZE":
        w = sg.Tensors(producer[t].Inputs(0))
      names = [sg.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()) if op.Outputs(j) != -1]
      cls = classify(";".join(names) + ";")
      layer = layer_re.search(names[0])
      out[f"{cls}@L{layer.group(1)}" if layer else cls] = tn.get(w.Type(), str(w.Type()))
  return out


def quantize(src, recipe, dst):
  from ai_edge_quantizer import quantizer as quantizer_lib
  t0 = time.time()
  qt = quantizer_lib.Quantizer(src)
  qt.load_quantization_recipe(recipe)
  qt.quantize().export_model(dst, overwrite=True)
  return round(time.time() - t0, 1)


def build(a, classes):
  qs = quantize(a.float, a.recipe, a.out)
  dts = fc_weight_dtypes(a.out, classes)
  per_class = {}
  for key, dt in dts.items():
    c = per_class.setdefault(key.split("@")[0], {})
    c[dt] = c.get(dt, 0) + 1
  info_path = a.recipe[:-5] + ".info.json" if a.recipe.endswith(".json") else a.recipe + ".info.json"
  res = {"recipe": a.recipe, "recipe_sha256": sha256(a.recipe),
         "info": json.load(open(info_path)) if os.path.exists(info_path) else None, "source": a.float,
         "output": a.out, "bytes": os.path.getsize(a.out), "sha256": sha256(a.out), "seconds": qs,
         "const_dtype_bytes": const_bytes(a.out), "fc_weight_dtype_per_class": per_class, "fc_weight_dtype_per_layer": dts}
  if a.record:
    json.dump(res, open(a.record, "w"), indent=1)
  print(f"BUILT {a.out} {res['bytes']} B {qs:.0f}s per class (decode) {json.dumps(per_class)}", flush=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("stage", choices=("list", "recipe", "build", "quantize"))
  ap.add_argument("--float")
  ap.add_argument("--src")
  ap.add_argument("--classes")
  ap.add_argument("--ops")
  ap.add_argument("--int8", default="")
  ap.add_argument("--recipe")
  ap.add_argument("--out", required=True)
  ap.add_argument("--record")
  a = ap.parse_args()
  if a.stage == "quantize":
    s = quantize(a.src, a.recipe, a.out)
    print(f"QUANTIZED {a.out} {os.path.getsize(a.out)} B sha256 {sha256(a.out)} {s}s", flush=True)
    return 0
  classes = json.load(open(a.classes))
  if a.stage == "list":
    ops = list_ops(a.float, classes)
    cnt = {}
    for o in ops:
      cnt.setdefault(o["signature"], {}).setdefault(o["class"], 0)
      cnt[o["signature"]][o["class"]] += 1
    json.dump({"source": a.float, "counts": cnt, "ops": ops}, open(a.out, "w"))
    per_class_buf = {}
    for o in {o["buffer"]: o for o in ops}.values():
      per_class_buf[o["class"]] = per_class_buf.get(o["class"], 0) + 1
    print(f"LISTED {len(ops)} FC ops in {len(cnt)} signatures, {len({o['buffer'] for o in ops})} weight buffers "
          f"{json.dumps(dict(sorted(per_class_buf.items())))}", flush=True)
    return 0
  return make_recipe(a, classes) if a.stage == "recipe" else build(a, classes)


if __name__ == "__main__":
  sys.exit(main())
