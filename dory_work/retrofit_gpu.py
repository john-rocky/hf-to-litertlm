"""retrofit_gpu.py SRC.tflite DST.tflite REPORT.json [--dry-run] - the GPU KV composites (odml.cache_update and
odml.runtime_bmm, the form litert-torch's --apply_gpu_composites produces) put into Dory's prefill/decode TFLite by this
repository's tools/gpu_graph/gpu_graph_retrofit.py, with one addition for Dory's sliding-window mask. The tool's file is
read, never written.

Why the addition: the tool's --mask add_bcast accepts a mask input that is repeated over the GQA groups by
CONCATENATION(mask x g, axis 2) before ADD(logits, .), or a mask whose consumers are all ADD on a [bk, g, T, C] view.
Dory's export has a second mask built inside the graph: export_hf adds the sliding band (create_sliding_mask, [1, T, C],
0 / -1e4) to the mask input with one ADD, and that sum is repeated over the groups by its own CONCATENATION for the
sliding blocks, while the full-attention blocks repeat the mask input itself. The tool alone stops at the band ADD
("mask consumer op 8 (ADD) is not CONCATENATION(mask x g, axis 2)").

What this adds (nothing else): the tool is loaded by path (importlib, no bytecode written) and its module-level
rewrite_mask is replaced by a wrapper that the tool's retrofit() then calls (it resolves the name at call time):
  1. derived masks = the ADD consumers of the mask input whose output has the mask's shape and is read only by
     CONCATENATION(out x g, axis 2); the band ADD itself stays in the graph.
  2. the tool's own rewrite_mask runs on the mask input with those band ADDs hidden from its consumer list, then once
     per derived mask with that tensor standing in for the mask input = the tool's add_bcast rewrite on both chains
     (the CONCATENATIONs removed; each ADD(logits, concat) -> RESHAPE [bk, g, T, C] -> ADD(view, mask) -> RESHAPE back,
     the same values).
Only --mask add_bcast is accepted. find_layers, both composites, their attributes, the decompositions (exporter), the
buffer layout and the buffer identity check are the tool's code, unchanged; every weight buffer keeps its bytes.

The report = the tool's report + `survey` (per signature, read before the rewrite on a separate load: the KV layers the
tool finds, BATCH_MATMULs per layer, the consumers of the mask input and of each derived mask, the ADDs fed by each
group CONCATENATION) + `extension` (tool path, sha256 before / after, seconds). On an error the report is still
written with the survey and the error, and the exit code is 1.

Usage: python -B dory_work/retrofit_gpu.py <src .tflite> <dst .tflite> <report json> [--dry-run]
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gpu_graph",
                    "gpu_graph_retrofit.py")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def load_tool():
  spec = importlib.util.spec_from_file_location("gpu_graph_retrofit_tool", TOOL)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class ConsumerView:
  """The tool's Graph with some consumer lists replaced (read by the tool's rewrite_mask only)."""

  def __init__(self, g, override):
    self._g = g
    self.consumers = dict(g.consumers)
    self.consumers.update(override)

  def __getattr__(self, name):
    return getattr(self._g, name)


def mask_chain(T, sg, g, mask_t):
  """-> (derived [(band ADD op index, derived tensor)], record) for one signature's mask input."""
  OP = T.OP

  def group_concat(oi, t):
    op = sg.operators[oi]
    return (g.opcode(op) == OP.CONCATENATION and len(op.inputs) > 1 and all(x == t for x in op.inputs)
            and op.builtinOptions is not None and op.builtinOptions.axis == 2)

  def describe(oi):
    op = sg.operators[oi]
    return dict(op=oi, type=T.BN.get(g.opcode(op)), inputs=[g.shape(t) if t >= 0 else None for t in op.inputs],
                outputs=[g.shape(t) for t in op.outputs])

  def concat_record(t):
    rows = []
    for c in sorted(set(g.consumers.get(t, []))):
      r = describe(c)
      if group_concat(c, t):
        out = sg.operators[c].outputs[0]
        fed = sorted(set(g.consumers.get(out, [])))
        r["groups"] = len(sg.operators[c].inputs)
        r["fed_ops"] = dict(collections.Counter(T.BN.get(g.opcode(sg.operators[a])) for a in fed))
        r["fed_add_mask_operand"] = dict(collections.Counter(
            ("inputs[1]" if sg.operators[a].inputs[1] == out else "inputs[0]" if sg.operators[a].inputs[0] == out
             else "other") for a in fed if g.opcode(sg.operators[a]) == OP.ADD))
      rows.append(r)
    return rows

  mshape = g.shape(mask_t)
  derived = []
  for c in sorted(set(g.consumers.get(mask_t, []))):
    op = sg.operators[c]
    if g.opcode(op) != OP.ADD or len(op.outputs) != 1 or g.shape(op.outputs[0]) != mshape:
      continue
    out = op.outputs[0]
    oc = sorted(set(g.consumers.get(out, [])))
    if oc and out not in sg.outputs and all(group_concat(x, out) for x in oc):
      derived.append((c, out))
  rec = dict(mask=dict(tensor=mask_t, shape=mshape, consumers=concat_record(mask_t)), derived=[])
  for c, out in derived:
    band = [t for t in sg.operators[c].inputs if t != mask_t]
    rec["derived"].append(dict(band_add=describe(c), band_operand_shapes=[g.shape(t) for t in band],
                               band_operand_is_const=[T.is_const(g.model, sg, t) for t in band], tensor=out,
                               consumers=concat_record(out)))
  return derived, rec


def make_rewrite_mask(T):
  orig = T.rewrite_mask

  def rewrite_mask(model, sg, g, key, sd, mode, new_ops_before, remove, replace, rep):
    if mode != "add_bcast":
      raise RuntimeError(f"{key}: retrofit_gpu.py takes --mask add_bcast only (got {mode})")
    mask_t = next(tm.tensorIndex for tm in sd.inputs if T.name_of(tm.name) == "mask")
    derived, _ = mask_chain(T, sg, g, mask_t)
    hide = {c for c, _ in derived}
    view = ConsumerView(g, {mask_t: [c for c in g.consumers.get(mask_t, []) if c not in hide]})
    parts = []
    sub = {}
    orig(model, sg, view, key, sd, mode, new_ops_before, remove, replace, sub)
    parts.append(dict(source="mask input", tensor=mask_t, **sub["mask"]))
    for c, out in derived:
      sdd = types.SimpleNamespace(inputs=[types.SimpleNamespace(name="mask", tensorIndex=out)])
      subd = {}
      orig(model, sg, g, key, sdd, mode, new_ops_before, remove, replace, subd)
      parts.append(dict(source=f"derived mask = output of ADD op {c} (sliding band + mask input)", tensor=out,
                        **subd["mask"]))
    rep["mask"] = dict(mode=mode, extension="band ADD -> derived mask, the tool's add_bcast on each", parts=parts,
                       mask_adds_rewritten=sum(p.get("add", 0) for p in parts),
                       mask_concats_removed=sum(p.get("mask_concats", 0) for p in parts), band_adds_kept=sorted(hide))

  return rewrite_mask


def survey(T, src):
  """Read-only pass on a separate load: what the tool and the addition will meet, per signature."""
  model, _, _ = T.load_tflite(src)
  out = {}
  for sd in model.signatureDefs:
    key = T.name_of(sd.signatureKey)
    if not (key.startswith("prefill") or key == "decode"):
      continue
    layers, other_dus, g = T.find_layers(model, sd.subgraphIndex, sd)
    sg = model.subgraphs[sd.subgraphIndex]
    mask_t = next((tm.tensorIndex for tm in sd.inputs if T.name_of(tm.name) == "mask"), None)
    rec = dict(subgraph=sd.subgraphIndex, n_ops=len(sg.operators), kv_layers=len(layers), other_dus=other_dus,
               bmm_per_layer=dict(collections.Counter(str(len(L["bmm"])) for L in layers)),
               layers_without_bmm=[L["layer"] for L in layers if not L["bmm"]],
               cache_shape=dict(bk=layers[0]["bk"], C=layers[0]["C"], H=layers[0]["H"], T=layers[0]["T"]) if layers else None,
               expected=dict(cache_update=len(layers), runtime_bmm=sum(len(L["bmm"]) for L in layers)))
    if mask_t is not None:
      derived, chain = mask_chain(T, sg, g, mask_t)
      rec["mask_chain"] = chain
      rec["expected"]["mask_adds"] = sum(
          sum(r.get("fed_ops", {}).get("ADD", 0) for r in part["consumers"])
          for part in [chain["mask"]] + chain["derived"])
    out[key] = rec
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("report")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  t0 = time.time()
  tool_sha0 = sha256(TOOL)
  T = load_tool()
  rec = dict(extension=dict(tool=TOOL, tool_sha256_before=tool_sha0, script=os.path.abspath(__file__),
                            src=os.path.abspath(args.src), dst=os.path.abspath(args.dst), dry_run=args.dry_run,
                            options=dict(decomp="exporter", indices="keep", rhs_cache_update=True, mask="add_bcast")))
  rc = 0
  try:
    rec["survey"] = survey(T, args.src)
    for key, s in rec["survey"].items():
      ch = s.get("mask_chain", {})
      print(f"SURVEY {key}: ops {s['n_ops']} kv_layers {s['kv_layers']} bmm/layer {s['bmm_per_layer']} "
            f"no-bmm layers {s['layers_without_bmm']} expected {s['expected']} derived masks {len(ch.get('derived', []))}",
            flush=True)
    T.rewrite_mask = make_rewrite_mask(T)
    rep = T.retrofit_tflite(args.src, args.dst, decomp="exporter", indices="keep", rhs_cache_update=True,
                            mask="add_bcast", dry_run=args.dry_run)
    rec["retrofit"] = rep
    for key, s in rep["signatures"].items():
      m = s["mask"]
      print(f"RETROFIT {key}: layers {s['layers']} bmm/layer {s['bmm_per_layer']} v_producer {s['v_producer']} "
            f"reordered {s['reordered']} mask adds {m['mask_adds_rewritten']} concats removed {m['mask_concats_removed']} "
            f"band ADDs kept {m['band_adds_kept']}", flush=True)
    print(f"COMPOSITES {rep['composites_added']}", flush=True)
    if not args.dry_run:
      bi = rep["buffer_identity"]
      print(f"BUFFER_IDENTITY ok={bi['ok']} inline {bi['inline_same']} external {bi['external_same_size']} "
            f"empty {bi['empty']} mismatch {len(bi['mismatch'])} region_identical {bi.get('region_identical')} "
            f"dst {rep['dst']['bytes']} B", flush=True)
  except Exception as ex:  # pylint: disable=broad-except
    rec["error"] = f"{type(ex).__name__}: {ex}"
    rec["traceback"] = traceback.format_exc()[-4000:]
    print(f"RETROFIT_GPU_ERROR {rec['error']}", flush=True)
    rc = 1
  rec["extension"]["tool_sha256_after"] = sha256(TOOL)
  rec["extension"]["tool_unchanged"] = rec["extension"]["tool_sha256_after"] == tool_sha0
  rec["extension"]["seconds"] = round(time.time() - t0, 1)
  with open(args.report, "w") as f:
    json.dump(rec, f, indent=1, default=str)
  print(f"RETROFIT_GPU {'OK' if rc == 0 else 'FAIL'} tool_unchanged={rec['extension']['tool_unchanged']} "
        f"{rec['extension']['seconds']} s", flush=True)
  return rc


if __name__ == "__main__":
  sys.exit(main())
