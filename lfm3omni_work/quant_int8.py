#!/usr/bin/env python3
"""quant_int8.py SRC_DIR OUT_DIR - dynamic int8 for the float32 decoder (SRC_DIR/model.tflite) and embedder
(SRC_DIR/embedder.tflite) of export_lm.py, with ai-edge-quantizer's dynamic_wi8_afp32 recipe: the weights of every
FULLY_CONNECTED op (the tied output head included) and the embedder's table int8 per channel, the activations float32
(the CPU path quantizes each FULLY_CONNECTED input per row at run time). Afterwards every FULLY_CONNECTED weight of every
signature must be INT8, or the script stops. Writes OUT_DIR/model.tflite and OUT_DIR/embedder.tflite.
"""
import collections
import mmap
import pathlib
import sys


def fc_weight_dtypes(path):
  """-> {signature: Counter(weight dtype of each FULLY_CONNECTED op)}; a weight that comes out of a DEQUANTIZE op counts
  as the dtype of the constant it reads."""
  from ai_edge_litert import schema_py_generated as schema  # pylint: disable=g-import-not-at-top
  type_names = {v: k for k, v in schema.TensorType.__dict__.items() if not k.startswith("_")}
  op_names = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}
  with open(path, "rb") as f:
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  m = schema.Model.GetRootAsModel(buf, 0)
  codes = [max(m.OperatorCodes(i).BuiltinCode(), m.OperatorCodes(i).DeprecatedBuiltinCode())
           for i in range(m.OperatorCodesLength())]
  out = {}
  for i in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(i)
    sg = m.Subgraphs(sd.SubgraphIndex())
    producer = {}
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      for j in range(op.OutputsLength()):
        producer[op.Outputs(j)] = op
    c = collections.Counter()
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      if op_names.get(codes[op.OpcodeIndex()]) != "FULLY_CONNECTED":
        continue
      wi = op.Inputs(1)
      src = producer.get(wi)
      if src is not None and op_names.get(codes[src.OpcodeIndex()]) == "DEQUANTIZE":
        wi = src.Inputs(0)
      c[type_names.get(sg.Tensors(wi).Type())] += 1
    out[sd.SignatureKey().decode()] = c
  return out


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  from ai_edge_quantizer import quantizer  # pylint: disable=g-import-not-at-top
  import ai_edge_quantizer.recipe as rlib  # pylint: disable=g-import-not-at-top
  src, dst = (pathlib.Path(p).resolve() for p in sys.argv[1:])
  dst.mkdir(parents=True, exist_ok=True)
  recipe = rlib.dynamic_wi8_afp32()
  for name in ("model.tflite", "embedder.tflite"):
    out = dst / name
    tmp = dst / (name + ".part")
    quantizer.Quantizer(str(src / name), recipe).quantize().export_model(str(tmp))
    tmp.rename(out)
    print(f"{out} {out.stat().st_size} B", flush=True)
  counts = fc_weight_dtypes(dst / "model.tflite")
  for sig, c in counts.items():
    print(f"  {sig}: FULLY_CONNECTED weights {dict(c)}", flush=True)
  if not all(set(c) == {"INT8"} for c in counts.values()):
    print("FAIL: a FULLY_CONNECTED weight is not INT8", flush=True)
    return 1
  print("every FULLY_CONNECTED weight INT8", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
