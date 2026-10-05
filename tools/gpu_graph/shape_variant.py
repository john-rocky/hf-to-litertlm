#!/usr/bin/env python3
"""A prefill-bucket variant of a shipped bundle — same weights, fewer prefill signatures.

  shape_variant.py build SRC.litertlm DST.litertlm --keep prefill_256,decode [--drop-subgraphs] --report R.json
      SignatureDefs whose key is not in --keep are removed from the TF_LITE_PREFILL_DECODE section. The runtime
      builds its prefill runner set from the SignatureDefs (LiteRT-LM v0.17.1
      litert_compiled_model_executor_utils.cc GetPrefillRunnerSetFromModel), so it sees only the kept lengths.
      Without --drop-subgraphs the removed signatures' subgraphs stay in the flatbuffer (variant G1). With it,
      every subgraph not reachable from a kept signature through StableHLOComposite decomposition indices is
      removed and the remaining subgraph indices are renumbered (G1s). Buffers are never touched: the external
      weight region is copied verbatim (gpu_graph_retrofit.write_tflite) and gpu_graph_retrofit.buffer_identity
      must hold; every kept subgraph must pack to the same bytes as its source subgraph (composite indices mapped
      back). Bundle unpack / pack and the other-sections-byte-identical assertion = gpu_graph_retrofit.retrofit_bundle,
      with its tflite step swapped for this one (the tool itself is not changed).
  shape_variant.py inspect BUNDLE.litertlm [...]
      SignatureDefs (key, subgraph, prefill length = the input_pos dimension), subgraph count, external-region
      sha256 of the TF_LITE_PREFILL_DECODE section, read in place from the bundle.
"""
import argparse
import copy
import hashlib
import json
import mmap
import os
import shutil
import sys

import flatbuffers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_graph_retrofit as R  # noqa: E402  load_tflite / write_tflite / buffer_identity / bundle_sections / retrofit_bundle
from ai_edge_litert import schema_py_generated as fb  # noqa: E402

BO, BO2 = fb.BuiltinOptions, fb.BuiltinOptions2
# option types that carry a subgraph index; only StableHLOComposite is handled, the rest must be absent
OTHER_SUBGRAPH_REFS = {
    ("1", BO.WhileOptions), ("1", BO.IfOptions), ("1", BO.CallOnceOptions),
    ("2", BO2.StablehloReduceOptions), ("2", BO2.StablehloScatterOptions), ("2", BO2.StablehloSortOptions),
    ("2", BO2.StablehloWhileOptions), ("2", BO2.StablehloCaseOptions), ("2", BO2.StablehloReduceWindowOptions),
}


def name_of(x):
  return R.name_of(x)


def composite_ref(op):
  if op.builtinOptions2Type == BO2.StableHLOCompositeOptions:
    return op.builtinOptions2.decompositionSubgraphIndex
  if ("1", op.builtinOptionsType) in OTHER_SUBGRAPH_REFS or ("2", op.builtinOptions2Type) in OTHER_SUBGRAPH_REFS:
    raise RuntimeError(f"op with a non-composite subgraph reference (options {op.builtinOptionsType} / "
                       f"{op.builtinOptions2Type}); this tool only renumbers StableHLOComposite decompositions")
  return None


def reachable(model, roots):
  seen, stack = set(), list(roots)
  while stack:
    si = stack.pop()
    if si in seen:
      continue
    seen.add(si)
    for op in model.subgraphs[si].operators or []:
      r = composite_ref(op)
      if r is not None:
        stack.append(r)
  return seen


def sg_bytes(sg, remap=None):
  """Packed bytes of one SubGraphT; composite decomposition indices pass through remap first."""
  s = copy.deepcopy(sg)
  if remap is not None:
    for op in s.operators or []:
      if composite_ref(op) is not None:
        op.builtinOptions2.decompositionSubgraphIndex = remap[op.builtinOptions2.decompositionSubgraphIndex]
  b = flatbuffers.Builder(1 << 16)
  b.Finish(s.Pack(b))
  return bytes(b.Output())


def sig_rows(model):
  rows = []
  for sd in model.signatureDefs or []:
    sg = model.subgraphs[sd.subgraphIndex]
    plen = None
    for tm in sd.inputs or []:
      if name_of(tm.name) in ("input_pos", "positions", "input_positions"):
        shp = [int(x) for x in sg.tensors[tm.tensorIndex].shape]
        plen = shp[-1]
    rows.append({"key": name_of(sd.signatureKey), "subgraph": int(sd.subgraphIndex),
                 "subgraph_name": name_of(sg.name), "inputs": len(sd.inputs or []), "outputs": len(sd.outputs or []),
                 "prefill_len": plen})
  return rows


def trim_tflite(src, dst, keep=(), drop_subgraphs=False, dry_run=False):
  """Drop-in for gpu_graph_retrofit.retrofit_tflite inside retrofit_bundle (same call shape)."""
  model, data_start, size = R.load_tflite(src)
  src_model, _, _ = R.load_tflite(src)  # untouched copy for the per-subgraph comparison
  keys = [name_of(sd.signatureKey) for sd in model.signatureDefs]
  missing = [k for k in keep if k not in keys]
  if missing:
    raise RuntimeError(f"--keep names a signature the source does not have: {missing} (has {keys})")
  rep = {"src_signatures": sig_rows(model), "src_subgraphs": len(model.subgraphs), "keep": list(keep),
         "drop_subgraphs": bool(drop_subgraphs)}
  model.signatureDefs = [sd for sd in model.signatureDefs if name_of(sd.signatureKey) in keep]
  n_comp = sum(composite_ref(op) is not None for sg in model.subgraphs for op in sg.operators or [])
  rep["composite_ops_src"] = n_comp
  if drop_subgraphs:
    live = sorted(reachable(model, [sd.subgraphIndex for sd in model.signatureDefs]))
    new_of = {old: new for new, old in enumerate(live)}
    model.subgraphs = [model.subgraphs[i] for i in live]
    for sd in model.signatureDefs:
      sd.subgraphIndex = new_of[sd.subgraphIndex]
    for sg in model.subgraphs:
      for op in sg.operators or []:
        if composite_ref(op) is not None:
          op.builtinOptions2.decompositionSubgraphIndex = new_of[op.builtinOptions2.decompositionSubgraphIndex]
    rep["live_subgraphs_src_index"] = live
  else:
    live = list(range(len(model.subgraphs)))
  rep["dst_signatures"] = sig_rows(model)
  rep["dst_subgraphs"] = len(model.subgraphs)
  rep["removed_subgraphs"] = rep["src_subgraphs"] - rep["dst_subgraphs"]
  removed_names = {}
  for i in sorted(set(range(rep["src_subgraphs"])) - set(live)):
    nm = name_of(src_model.subgraphs[i].name)
    base = nm.split(".impl")[0] + ".impl" if ".impl" in nm else nm
    removed_names[base] = removed_names.get(base, 0) + 1
  rep["removed_by_name"] = removed_names
  rep["src"] = {"path": src, "bytes": size}
  if dry_run:
    return rep
  rep["layout"] = R.write_tflite(model, src, data_start, size, dst)
  rep["dst"] = {"path": dst, "bytes": os.path.getsize(dst)}
  rep["buffer_identity"] = R.buffer_identity(src, dst)
  if not rep["buffer_identity"]["ok"]:
    raise RuntimeError(f"buffer identity failed: {rep['buffer_identity']}")
  # every kept subgraph = its source subgraph, byte for byte after packing (composite indices mapped back)
  out_model, _, _ = R.load_tflite(dst)
  back = {new: old for new, old in enumerate(live)}
  same = 0
  for new, old in enumerate(live):
    if sg_bytes(out_model.subgraphs[new], back) != sg_bytes(src_model.subgraphs[old]):
      raise RuntimeError(f"subgraph {new} (source {old}, {name_of(src_model.subgraphs[old].name)}) changed")
    same += 1
  rep["kept_subgraphs_identical"] = f"{same}/{len(live)}"
  for a, b in zip([s for s in src_model.signatureDefs if name_of(s.signatureKey) in keep], out_model.signatureDefs):
    if name_of(a.signatureKey) != name_of(b.signatureKey) or back[b.subgraphIndex] != a.subgraphIndex:
      raise RuntimeError("signature order / subgraph mapping changed")
    ta = [(name_of(t.name), int(t.tensorIndex)) for t in a.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in a.outputs]
    tb = [(name_of(t.name), int(t.tensorIndex)) for t in b.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in b.outputs]
    if ta != tb:
      raise RuntimeError(f"signature {name_of(a.signatureKey)} inputs/outputs changed")
  rep["kept_signature_io_identical"] = True
  ka = [name_of(c.customCode) + str(max(c.builtinCode, c.deprecatedBuiltinCode)) for c in src_model.operatorCodes]
  kb = [name_of(c.customCode) + str(max(c.builtinCode, c.deprecatedBuiltinCode)) for c in out_model.operatorCodes]
  rep["operator_codes_identical"] = ka == kb
  rep["metadata_identical"] = ([(name_of(m.name), m.buffer) for m in src_model.metadata or []]
                               == [(name_of(m.name), m.buffer) for m in out_model.metadata or []])
  return rep


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for blk in iter(lambda: f.read(1 << 24), b""):
      h.update(blk)
  return h.hexdigest()


def inspect(path):
  secs = R.bundle_sections(path)
  pd = [s for s in secs if s["type"] == "TFLiteModel" and s["items"].get("model_type", "").lower().endswith("prefill_decode")][0]
  with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = fb.Model.GetRootAsModel(memoryview(mm)[pd["begin"]:pd["end"]], 0)
    sigs = []
    for i in range(m.SignatureDefsLength()):
      sd = m.SignatureDefs(i)
      sg = m.Subgraphs(sd.SubgraphIndex())
      plen = None
      for k in range(sd.InputsLength()):
        tm = sd.Inputs(k)
        if tm.Name().decode() in ("input_pos", "positions", "input_positions"):
          t = sg.Tensors(tm.TensorIndex())
          plen = t.Shape(t.ShapeLength() - 1)
      sigs.append({"key": sd.SignatureKey().decode(), "subgraph": sd.SubgraphIndex(), "prefill_len": plen,
                   "subgraph_ops": sg.OperatorsLength()})
    start, n_ext = R.external_region(m)
    n_sub = m.SubgraphsLength()
    # the reader objects keep views of mm alive; the map is released when they are collected
  region = R.sha256_range(path, pd["begin"] + start, pd["end"]) if start else None
  return {"path": path, "bytes": os.path.getsize(path), "signatures": sigs, "subgraphs": n_sub,
          "external_buffers": n_ext, "prefill_decode_section": {"begin": pd["begin"], "bytes": pd["bytes"],
          "sha256": pd["sha256"], "external_region_start": start, "external_region_sha256": region},
          "sections": [{k: s[k] for k in ("type", "items", "bytes", "sha256")} for s in secs]}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  b = sub.add_parser("build")
  b.add_argument("src")
  b.add_argument("dst")
  b.add_argument("--keep", required=True, help="comma-separated SignatureDef keys to keep, e.g. prefill_256,decode")
  b.add_argument("--drop-subgraphs", action="store_true")
  b.add_argument("--report", required=True)
  b.add_argument("--work")
  i = sub.add_parser("inspect")
  i.add_argument("bundles", nargs="+")
  args = ap.parse_args()
  if args.cmd == "inspect":
    print(json.dumps([inspect(p) for p in args.bundles], indent=1))
    return 0
  keep = tuple(x for x in args.keep.split(",") if x)
  work = args.work or (os.path.splitext(args.dst)[0] + "_work")
  os.makedirs(work, exist_ok=True)
  R.retrofit_tflite = trim_tflite  # retrofit_bundle looks the name up at call time
  rep = R.retrofit_bundle(args.src, args.dst, work, keep=keep, drop_subgraphs=args.drop_subgraphs)
  shutil.rmtree(work, ignore_errors=True)
  rep["bundle"] = {"src": args.src, "dst": args.dst, "src_bytes": os.path.getsize(args.src),
                   "dst_bytes": os.path.getsize(args.dst), "src_sha256": sha256_file(args.src),
                   "dst_sha256": sha256_file(args.dst)}
  rep["inspect_src"] = inspect(args.src)
  rep["inspect_dst"] = inspect(args.dst)
  rep["weight_region_identical"] = (rep["inspect_src"]["prefill_decode_section"]["external_region_sha256"]
                                    == rep["inspect_dst"]["prefill_decode_section"]["external_region_sha256"])
  with open(args.report, "w") as f:
    json.dump(rep, f, indent=1, default=str)
  short = {k: rep[k] for k in ("dst_signatures", "src_subgraphs", "dst_subgraphs", "removed_subgraphs", "removed_by_name",
                               "kept_subgraphs_identical", "kept_signature_io_identical", "operator_codes_identical",
                               "metadata_identical", "weight_region_identical", "bundle")}
  short["buffer_identity_ok"] = rep["buffer_identity"]["ok"]
  short["region_identical"] = rep["buffer_identity"].get("region_identical")
  short["sections"] = [(s["type"], s["items_src"].get("model_type"), s["sha256_src"][:12], s["sha256_dst"][:12], s["replaced"])
                       for s in rep["bundle_sections"]]
  print(json.dumps(short, indent=1, default=str))
  return 0


if __name__ == "__main__":
  sys.exit(main())
