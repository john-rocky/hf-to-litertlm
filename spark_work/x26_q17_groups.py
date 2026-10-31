#!/usr/bin/env python3
"""Spark-X2.6: the FC groups of the 1.7B prefill/decode graph for the Q17 (`_int4-mlp`) mixed recipe.

  classify GRAPH.tflite [--json OUT]
      Every FULLY_CONNECTED / EMBEDDING_LOOKUP of every signature, classified by its weight shape (Spark-X2.6-1.7B:
      q_k_v_proj [3072, 2048], g_proj [8, 2048], out_proj [2048, 2048] = attn; gate_proj / up_proj [6656, 2048] =
      mlp_gu; down_proj [2048, 6656] = mlp_down; [131072, 2048] = lm_head), never by name. Prints the op scope
      (output tensor names joined by ';' + a trailing ';', ai_edge_quantizer's get_op_scope) of a few ops per group
      and checks that each group regex in GROUP_RX matches (re.search) exactly its own group's scopes in every
      signature. Exit 1 on a mismatch.
  census GRAPH.tflite --plan attn=8,mlp_gu=4,mlp_down=4,lm_head=8 [--json OUT]
      After quantization: per group the weight dtype (INT8 / INT4), granularity (per-channel scales vs blockwise) and the
      distinct weight buffers, against the plan. Exit 1 on a mismatch.
"""
import argparse
import collections
import json
import mmap
import re
import sys

from ai_edge_litert import schema_py_generated as s

OPN = {v: k for k, v in s.BuiltinOperator.__dict__.items() if not k.startswith("_")}
TTN = {v: k for k, v in s.TensorType.__dict__.items() if not k.startswith("_")}
SHAPE_GROUP = {(3072, 2048): "attn", (8, 2048): "attn", (2048, 2048): "attn", (6656, 2048): "mlp_gu",
               (2048, 6656): "mlp_down", (131072, 2048): "lm_head"}
# Scope regexes per group (re.search, as ai-edge-quantizer), anchored on the FIRST location of the op's output name
# (`^[^;]*...;`): fused names carry other modules' locations after the first `;` — up_proj's scope is
# `.../Spark2_5MLP_mlp/<Linear>up_proj;;.../Spark2_5MLP_mlp/<Linear>gate_proj;<n>;` and g_proj's starts with the
# attention module itself (`.../Spark2_5Attention_self_attn;;...q_k_v_proj;;...g_proj;<n>;`). Read off the int4 b32
# graph's names (06:2x) and checked by `classify` on every FC of every signature, never trusted by name alone.
_LIN = r"/torch\.nn\.modules\.linear\.Linear_"
GROUP_RX = {
    "attn": r"^[^;]*Spark2_5Attention_self_attn(?:" + _LIN + r"(?:q_k_v_proj|out_proj))?;",
    "mlp_gu": r"^[^;]*Spark2_5MLP_mlp" + _LIN + r"(?:gate_proj|up_proj);",
    "mlp_down": r"^[^;]*Spark2_5MLP_mlp" + _LIN + r"down_proj;",
    "lm_head": r"^decode_logits_output;",
}


def load(path):
  f = open(path, "rb")
  mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  return s.Model.GetRootAs(mm, 0)


def ops_of(m):
  for si in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(si)
    key = sd.SignatureKey().decode()
    sg = m.Subgraphs(sd.SubgraphIndex())
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      oc = m.OperatorCodes(op.OpcodeIndex())
      name = OPN.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
      if name not in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
        continue
      outs = [sg.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength())]
      scope = "".join(o + ";" for o in outs)
      wi = op.Inputs(1) if name == "FULLY_CONNECTED" else op.Inputs(1)
      w = sg.Tensors(wi)
      shape = tuple(int(x) for x in w.ShapeAsNumpy()) if w.ShapeLength() else ()
      yield key, i, name, scope, w, shape, m.Buffers(w.Buffer())


def classify(a):
  m = load(a.graph)
  rows = collections.defaultdict(list)
  bad = []
  for key, i, name, scope, w, shape, buf in ops_of(m):
    g = "embedding" if name == "EMBEDDING_LOOKUP" else SHAPE_GROUP.get(shape)
    if g is None:
      bad.append((key, i, name, shape, scope))
      continue
    rows[g].append((key, i, scope, shape))
  out = {"per_signature": {}, "examples": {}, "unclassified": bad, "regex_check": {}}
  per = collections.defaultdict(collections.Counter)
  for g, lst in rows.items():
    for key, i, scope, shape in lst:
      per[key][g] += 1
    out["examples"][g] = [x[2] for x in lst[:3]] + [x[2] for x in lst if x[0] == "decode"][:3]
  out["per_signature"] = {k: dict(v) for k, v in per.items()}
  ok = not bad
  for g, rx in GROUP_RX.items():
    hits_own = sum(1 for x in rows.get(g, []) if re.search(rx, x[2]))
    hits_other = sum(1 for g2, lst in rows.items() if g2 != g for x in lst if re.search(rx, x[2]))
    out["regex_check"][g] = {"regex": rx, "own": len(rows.get(g, [])), "own_hit": hits_own, "other_hit": hits_other}
    ok = ok and hits_own == len(rows.get(g, [])) and hits_other == 0
  for k, v in sorted(out["per_signature"].items()):
    print(k, v)
  for g, ex in out["examples"].items():
    print(f"== {g}")
    for e in ex:
      print("   ", e[:300])
  for g, r in out["regex_check"].items():
    print(f"REGEX {g}: own {r['own_hit']}/{r['own']} other {r['other_hit']}  {r['regex']}")
  print("unclassified", len(bad), bad[:3])
  if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
  return 0 if ok else 1


def census(a):
  plan = dict(kv.split("=") for kv in a.plan.split(","))
  m = load(a.graph)
  res = collections.defaultdict(lambda: {"dtype": collections.Counter(), "blockwise": collections.Counter(),
                                         "buffers": set(), "ops": 0})
  for key, i, name, scope, w, shape, buf in ops_of(m):
    g = "embedding" if name == "EMBEDDING_LOOKUP" else SHAPE_GROUP.get(shape, "?")
    q = w.Quantization()
    blockwise = bool(q and q.DetailsType() == s.QuantizationDetails.BlockwiseQuantization)
    r = res[g]
    r["dtype"][TTN.get(w.Type(), w.Type())] += 1
    r["blockwise"][blockwise] += 1
    r["buffers"].add(w.Buffer())
    r["ops"] += 1
  ok = True
  out = {}
  for g, r in res.items():
    want = plan.get(g)
    dts = set(r["dtype"])
    if want == "4":
      good = dts == {"INT4"} and set(r["blockwise"]) == {True}
    elif want == "8":
      good = dts == {"INT8"} and set(r["blockwise"]) == {False}
    else:
      good = True
    ok = ok and good
    out[g] = {"ops": r["ops"], "dtype": dict(r["dtype"]), "blockwise": {str(k): v for k, v in r["blockwise"].items()},
              "buffers": len(r["buffers"]), "plan": want, "ok": good}
    print(f"CENSUS {g}: ops {r['ops']} dtype {dict(r['dtype'])} blockwise {dict(r['blockwise'])} buffers "
          f"{len(r['buffers'])} plan {want} {'ok' if good else 'MISMATCH'}")
  print(f"CENSUS {'PASS' if ok else 'FAIL'}")
  if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
  return 0 if ok else 1


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("classify")
  c.add_argument("graph")
  c.add_argument("--json")
  c2 = sub.add_parser("census")
  c2.add_argument("graph")
  c2.add_argument("--plan", required=True)
  c2.add_argument("--json")
  a = ap.parse_args()
  sys.exit(classify(a) if a.cmd == "classify" else census(a))
