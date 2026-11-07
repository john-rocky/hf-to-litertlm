#!/usr/bin/env python3
"""aot_compare.py A.litertlm B.litertlm - read-only: is A the same computation as B, for two LiteRT-LM files compiled ahead of
time for an NPU? Section by section in order (type, header items, size, sha256 of the bytes); a TFLite section whose bytes
differ is compared as a model: operator codes, subgraphs (names, inputs, outputs, every tensor's name, type, shape,
quantization and constant bytes, every operator's code, inputs, outputs and options; a dispatch operator's custom options
without its bytecode offset), signatures, metadata, and the compiled NPU code (each dispatch operator's bytecode, the
bytes at its offset and size) byte for byte. Prints each difference; the last line is the verdict:
  SAME BYTES        every section byte for byte
  SAME COMPUTATION  every section equal but compiled TFLite sections that differ only in where their parts sit and in
                    the LiteRtStamp metadata, with every operator, tensor, constant and compiled bytecode equal
  DIFFERENT         anything else (exit 1)
The file header's uuid and creation time are not section bytes and are not compared. With a third argument OUT.json, the
result (sections, differences with every differing byte run, verdict) is also written there. Needs litert-lm-builder,
ai-edge-litert (its TFLite schema), flatbuffers and numpy: mnn_pocket_work/npu/requirements.txt, or the stage-4 container."""
import hashlib
import json
import mmap
import struct
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers
from litert_lm_builder import litertlm_header_schema_py_generated as H

TYPE = {v: k for k, v in vars(H.AnySectionDataType).items() if isinstance(v, int)}
OPS = {v: k for k, v in vars(S.BuiltinOperator).items() if isinstance(v, int)}
TT = {v: k for k, v in vars(S.TensorType).items() if isinstance(v, int)}


def sections(path):
  f = open(path, "rb")
  mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  header_end = struct.unpack("<Q", mm[24:32])[0]
  md = H.LiteRTLMMetaData.GetRootAs(mm[:header_end], 32)
  objs = md.SectionMetadata()
  out = []
  for i in range(objs.ObjectsLength()):
    o = objs.Objects(i)
    items = {}
    for j in range(o.ItemsLength()):
      it = o.Items(j)
      key = it.Key().decode() if it.Key() else ""
      if it.ValueType() == H.VData.StringValue:
        v = H.StringValue()
        v.Init(it.Value().Bytes, it.Value().Pos)
        items[key] = v.Value().decode()
      else:
        items[key] = f"<value type {it.ValueType()}>"
    b, e = o.BeginOffset(), o.EndOffset()
    out.append({"i": i, "type": TYPE.get(o.DataType(), str(o.DataType())), "items": items, "size": e - b,
                "sha256": hashlib.sha256(mm[b:e]).hexdigest(), "bytes": mm[b:e]})
  return out


def arr(fn, n):
  return [fn(k) for k in range(n)]


def plain(o):
  """A schema object (the generated object API) -> nested tuples of its fields, arrays as bytes, for comparison."""
  if o is None or isinstance(o, (bool, int, float, str, bytes)):
    return o
  if hasattr(o, "tobytes"):
    return o.tobytes()
  if isinstance(o, (list, tuple)):
    return tuple(plain(x) for x in o)
  return tuple(sorted((k, plain(v)) for k, v in vars(o).items()))


def model(buf):
  """A TFLite flatbuffer -> a comparable description, and its dispatch operators' bytecodes (name -> bytes)."""
  m = S.Model.GetRootAs(buf, 0)
  codes = []
  for k in range(m.OperatorCodesLength()):
    oc = m.OperatorCodes(k)
    code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
    codes.append(OPS.get(code, str(code)) + (f":{oc.CustomCode().decode()}" if oc.CustomCode() else ""))

  def buffer_bytes(idx):
    b = m.Buffers(idx)
    if b.Offset():  # data outside the flatbuffer
      return bytes(buf[b.Offset():b.Offset() + b.Size()])
    return b.DataAsNumpy().tobytes() if b.DataLength() else b""

  desc, codes_by_name = {"operator_codes": codes, "subgraphs": [], "signatures": [], "metadata": {}}, {}
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    tensors = []
    for ti in range(sg.TensorsLength()):
      t = sg.Tensors(ti)
      q = t.Quantization()
      qd = None
      if q is not None and (q.ScaleLength() or q.ZeroPointLength()):
        qd = (q.ScaleAsNumpy().tobytes() if q.ScaleLength() else b"", q.ZeroPointAsNumpy().tobytes() if q.ZeroPointLength() else b"",
              q.QuantizedDimension())
      data = buffer_bytes(t.Buffer()) if t.Buffer() else b""
      tensors.append((t.Name().decode(), TT.get(t.Type(), t.Type()), tuple(arr(t.Shape, t.ShapeLength())),
                      tuple(arr(t.ShapeSignature, t.ShapeSignatureLength())), qd, hashlib.sha256(data).hexdigest() if data else None))
    ops = []
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      name = codes[op.OpcodeIndex()]
      custom = None
      if op.CustomOptionsLength():
        raw = op.CustomOptionsAsNumpy().tobytes()
        try:
          custom = flexbuffers.Loads(raw)
        except Exception:  # noqa: BLE001 - not a flexbuffer: compared as bytes
          custom = raw
      if isinstance(custom, dict) and "bytecode_offset" in custom:
        off, size = int(custom["bytecode_offset"]), int(custom["bytecode_size"])
        codes_by_name[f"{sg.Name().decode()}/{custom.get('name')}"] = bytes(buf[off:off + size])
        custom = {k: v for k, v in custom.items() if k != "bytecode_offset"}
      opts = plain(S.OperatorT.InitFromObj(op).builtinOptions) if op.BuiltinOptionsType() else None
      ops.append((name, tuple(arr(op.Inputs, op.InputsLength())), tuple(arr(op.Outputs, op.OutputsLength())),
                  op.BuiltinOptionsType(), opts, custom))
    desc["subgraphs"].append({"name": sg.Name().decode() if sg.Name() else "", "inputs": tuple(arr(sg.Inputs, sg.InputsLength())),
                              "outputs": tuple(arr(sg.Outputs, sg.OutputsLength())), "tensors": tensors, "operators": ops})
  for k in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(k)
    desc["signatures"].append((sd.SignatureKey().decode(), sd.SubgraphIndex(),
                               tuple((sd.Inputs(j).Name().decode(), sd.Inputs(j).TensorIndex()) for j in range(sd.InputsLength())),
                               tuple((sd.Outputs(j).Name().decode(), sd.Outputs(j).TensorIndex()) for j in range(sd.OutputsLength()))))
  for k in range(m.MetadataLength()):
    md = m.Metadata(k)
    desc["metadata"][md.Name().decode()] = buffer_bytes(md.Buffer())
  return desc, codes_by_name


def diff_bytes(a, b):
  """The number of differing bytes (a length difference counts in full) and the runs of differing offsets [first, last]."""
  n = min(len(a), len(b))
  idx = np.flatnonzero(np.frombuffer(a, np.uint8, n) != np.frombuffer(b, np.uint8, n))
  runs = []
  if idx.size:
    cut = np.flatnonzero(np.diff(idx) > 1)
    runs = list(zip(np.concatenate(([idx[0]], idx[cut + 1])).tolist(), np.concatenate((idx[cut], [idx[-1]])).tolist()))
  return int(idx.size) + abs(len(a) - len(b)), runs


def main(argv):
  if len(argv) not in (3, 4):
    print(__doc__)
    return 2
  sa, sb = sections(argv[1]), sections(argv[2])
  diffs, notes, compiled_same, code_runs = [], [], True, {}
  if [s["type"] for s in sa] != [s["type"] for s in sb]:
    diffs.append(f"section types: {[s['type'] for s in sa]} vs {[s['type'] for s in sb]}")
  for a, b in zip(sa, sb):
    tag = f"section {a['i']} {a['type']} {a['items'].get('model_type', '')}".strip()
    if a["items"] != b["items"]:
      diffs.append(f"{tag}: header items {a['items']} vs {b['items']}")
    if a["sha256"] == b["sha256"]:
      print(f"{tag}: {a['size']} B, sha256 {a['sha256']}: equal bytes")
      continue
    print(f"{tag}: bytes differ ({a['size']} B {a['sha256'][:16]}... vs {b['size']} B {b['sha256'][:16]}...)")
    if a["type"] != "TFLiteModel":
      diffs.append(f"{tag}: bytes differ")
      continue
    da, ca = model(a["bytes"])
    db, cb = model(b["bytes"])
    for key in ("operator_codes", "signatures"):
      if da[key] != db[key]:
        diffs.append(f"{tag}: {key} differ")
    if len(da["subgraphs"]) != len(db["subgraphs"]):
      diffs.append(f"{tag}: {len(da['subgraphs'])} vs {len(db['subgraphs'])} subgraphs")
    for ga, gb in zip(da["subgraphs"], db["subgraphs"]):
      for key in ("name", "inputs", "outputs"):
        if ga[key] != gb[key]:
          diffs.append(f"{tag} subgraph {ga['name']}: {key} differ")
      if ga["tensors"] != gb["tensors"]:
        bad = [i for i, (x, y) in enumerate(zip(ga["tensors"], gb["tensors"])) if x != y]
        diffs.append(f"{tag} subgraph {ga['name']}: {len(bad)} tensors differ (first {bad[:5]}), {len(ga['tensors'])} vs {len(gb['tensors'])}")
      if ga["operators"] != gb["operators"]:
        bad = [i for i, (x, y) in enumerate(zip(ga["operators"], gb["operators"])) if x != y]
        diffs.append(f"{tag} subgraph {ga['name']}: {len(bad)} operators differ (first {bad[:5]}), {len(ga['operators'])} vs {len(gb['operators'])}")
      print(f"  subgraph {ga['name']}: {len(ga['operators'])} operators ({', '.join(sorted(set(o[0] for o in ga['operators'])))}), "
            f"{len(ga['tensors'])} tensors: {'equal' if ga == gb else 'DIFFER'}")
    ma, mb = da["metadata"], db["metadata"]
    for k in sorted(set(ma) | set(mb)):
      if ma.get(k) != mb.get(k):
        (notes if k == "LiteRtStamp" else diffs).append(f"{tag}: metadata {k} differs ({len(ma.get(k) or b'')} vs {len(mb.get(k) or b'')} B)")
    for k in sorted(set(ca) | set(cb)):
      x, y = ca.get(k, b""), cb.get(k, b"")
      if x == y:
        print(f"  compiled code {k}: {len(x)} B, sha256 {hashlib.sha256(x).hexdigest()}: equal bytes")
      else:
        n, runs = diff_bytes(x, y)
        compiled_same = False
        code_runs[f"{tag}: {k}"] = {"bytes_a": len(x), "bytes_b": len(y), "differing": n, "runs": runs}
        diffs.append(f"{tag}: compiled code {k} differs: {len(x)} vs {len(y)} B, {n} bytes differ in {len(runs)} runs "
                     f"(first {[(r[0], r[1] - r[0] + 1) for r in runs[:8]]}); sha256 {hashlib.sha256(x).hexdigest()[:16]}... vs {hashlib.sha256(y).hexdigest()[:16]}...")
  for n in notes:
    print("note: " + n)
  for d in diffs:
    print("DIFF: " + d)
  if all(a["sha256"] == b["sha256"] for a, b in zip(sa, sb)) and len(sa) == len(sb):
    verdict = "SAME BYTES"
  elif not diffs and compiled_same:
    verdict = "SAME COMPUTATION"
  else:
    verdict = "DIFFERENT"
  if len(argv) == 4:
    with open(argv[3], "w") as fh:
      json.dump({"a": argv[1], "b": argv[2], "verdict": verdict, "notes": notes, "diffs": diffs, "compiled_code_runs": code_runs,
                 "sections": [[{k: v for k, v in s.items() if k != "bytes"} for s in x] for x in (sa, sb)]}, fh, indent=1)
  print(verdict)
  return 0 if verdict != "DIFFERENT" else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv))
