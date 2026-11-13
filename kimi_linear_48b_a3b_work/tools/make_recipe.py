#!/usr/bin/env python3
"""make_recipe.py EXPORT.tflite OUT_RECIPE.json - the litert_quantizer recipe of the GPU file: the KDA and MLA projections
and the output layer int8 per channel, chosen one fully connected op at a time.

The converter fuses op locations, so an op's output name also holds the module paths of its neighbours (the outputs of
KDA's f_a_proj and b_proj end in `...Lin_q_proj;`, and MLA's kv_c / kv_r / q_rot outputs hold `Lin_q_nope_proj`), and
litert_quantizer matches a recipe entry's regex against an op's scope: its output names joined by ";", with a trailing
";". A regex like `Lin_q_proj` would make KDA's small gate projections int8 too. So a FULLY_CONNECTED op with a constant
weight is chosen by a weight shape that only the targets have, together with the module name in its scope:
  KDA q / k / v [4096, 2304] and o [2304, 4096]                                  (the scope holds kimi_ext.KDA_attn)
  MLA q_nope [4096, 2304], q_rot [2048, 2304], kv_c [512, 2304], kv_r [64, 2304],
      o [2304, 4096]                                                             (the scope holds kimi_ext.MLA_attn)
  the output layer [163840, 2304]                                                (the scope is `decode_logits_output;`)
Everything else stays fp32: KDA's f_a / g_a [128, 2304], f_b / g_b [4096, 128] and b [32, 2304]; the router
[256, 2304]; the shared experts [1024, 2304] / [2304, 1024]; the first layer's dense MLP [9216, 2304] / [2304, 9216];
the 16 x 16 chunk masks; the fully connected ops whose weight is a run-time tensor (the attention over the MLA cache).
The recipe lists the chosen ops' exact scopes, anchored and escaped; a scope shared by a chosen and an unchosen op stops
the tool. The `moe` op is left as it is (no_quantize): tools/graft_kimi.py writes the routed experts afterwards.
Writes OUT_RECIPE.json and OUT_RECIPE.json.select.json (per signature: the int8 and fp32 counts by weight shape; the
scopes). Reads the file through mmap with ai_edge_litert's flatbuffer schema.
"""
import collections
import json
import mmap
import re
import sys

TARGETS = {
    (4096, 2304): ("KDA_attn", "MLA_attn"),
    (2304, 4096): ("KDA_attn", "MLA_attn"),
    (2048, 2304): ("MLA_attn",),
    (512, 2304): ("MLA_attn",),
    (64, 2304): ("MLA_attn",),
    (163840, 2304): (None,),
}
INT8 = {"num_bits": 8, "symmetric": True, "granularity": "CHANNELWISE", "dtype": "INT"}


def main():
  if len(sys.argv) != 3:
    print(__doc__.splitlines()[0], file=sys.stderr)
    return 2
  src, out = sys.argv[1], sys.argv[2]
  from ai_edge_litert import schema_py_generated as S
  f = open(src, "rb")
  mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  m = S.Model.GetRootAs(mm, 0)
  codes = {v: k for k, v in S.BuiltinOperator.__dict__.items() if isinstance(v, int)}
  opc = [max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())
         for i in range(m.OperatorCodesLength())]
  chosen, unchosen = set(), set()
  per_sig = {}
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    name = sg.Name().decode() if sg.Name() else str(si)
    rec = {"int8": collections.Counter(), "float": collections.Counter(), "runtime_weight": collections.Counter()}
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      if codes[opc[op.OpcodeIndex()]] != "FULLY_CONNECTED":
        continue
      wt = sg.Tensors(op.Inputs(1))
      shape = tuple(int(x) for x in wt.ShapeAsNumpy())
      outs = [sg.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()) if op.Outputs(j) >= 0]
      scope = ";".join(n for n in outs if n) + ";"
      buf = m.Buffers(wt.Buffer())
      const = buf is not None and (buf.DataLength() > 0 or buf.Size() > 0)
      key = "x".join(map(str, shape))
      if not const:
        rec["runtime_weight"][key] += 1
        continue
      mods = TARGETS.get(shape)
      if mods == (None,):
        hit = scope == "decode_logits_output;"
      else:
        hit = mods is not None and any(f"kimi_ext.{x}" in scope for x in mods)
      (chosen if hit else unchosen).add(scope)
      rec["int8" if hit else "float"][key] += 1
    per_sig[name] = {k: dict(v) for k, v in rec.items()}
  both = chosen & unchosen
  if both:
    print("SCOPE SHARED by an int8 target and a float op:", sorted(both)[:5], file=sys.stderr)
    return 1
  regex = "^(?:" + "|".join(re.escape(s) for s in sorted(chosen)) + ")$"
  for s in unchosen:
    assert not re.search(regex, s), s
  recipe = [
      {"regex": regex, "operation": "FULLY_CONNECTED", "algorithm_key": "min_max_uniform_quantize",
       "op_config": {"weight_tensor_config": INT8, "compute_precision": "INTEGER", "explicit_dequantize": False,
                     "skip_checks": False, "min_weight_elements": 0}},
      {"regex": ".*", "operation": "CUSTOM_OP", "algorithm_key": "no_quantize"},
  ]
  json.dump(recipe, open(out, "w"), indent=1)
  json.dump({"tflite": src, "signatures": per_sig, "chosen_scopes": sorted(chosen), "float_scopes": sorted(unchosen)},
            open(out + ".select.json", "w"), indent=1)
  for name, rec in per_sig.items():
    print(name, json.dumps(rec))
  print("RECIPE", out, "chosen scopes", len(chosen), "float scopes", len(unchosen))
  return 0


if __name__ == "__main__":
  sys.exit(main())
