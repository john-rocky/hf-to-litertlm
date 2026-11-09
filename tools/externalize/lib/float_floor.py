#!/usr/bin/env python3
"""The float weights of one TFLite section, and the threshold that keeps them in the graph (--keep-float-inline).

  float_floor.py SECTION.tflite

Prints one line, `<elements> <TAB> <float weights> <TAB> <integer weights kept>`: the largest element count among the
section's constant FLOAT16 / FLOAT32 tensors read as input 1 of EMBEDDING_LOOKUP, FULLY_CONNECTED, CONV_2D or
DEPTHWISE_CONV_2D (the only tensors LiteRT's externalize_tflite_flatbuffer.py moves), how many such tensors the section
has (counted per subgraph), and how many integer tensors of the same kind with more than 256 elements are at or under
that count (they stay in the graph too); `0 <TAB> 0 <TAB> 0` when the section has no float weight. The tool keeps a
tensor inline when its element count is at most --num_elements_threshold, so a threshold of at least <elements> keeps
every float weight in the graph.

Why: LiteRT's WebGPU streamed loading takes only INT8 / INT4 / INT2 external weights, so a float weight in the weights
section gains nothing there; and an LFM2 int4 file whose FLOAT32 ShortConv filters (DEPTHWISE_CONV_2D input 1) went to
the weights section gave other greedy replies than the original on the Mac GPU (LiteRT-LM 0.19.0), while the same file
with those filters in the graph and every other weight external gave the original's replies.

Reads the flatbuffer only (ai_edge_litert's generated schema, lazy API). Exit 0, or 2 on bad input.
"""
import sys

from ai_edge_litert import schema_py_generated as S

FLOAT_TYPES = {S.TensorType.FLOAT16, S.TensorType.FLOAT32}
WEIGHT_OPS = {S.BuiltinOperator.EMBEDDING_LOOKUP, S.BuiltinOperator.FULLY_CONNECTED, S.BuiltinOperator.CONV_2D,
              S.BuiltinOperator.DEPTHWISE_CONV_2D}


def op_code(m, opcode_index):
    c = m.OperatorCodes(opcode_index)
    code = c.BuiltinCode()
    if code == 0 and c.DeprecatedBuiltinCode() != 0:
        code = c.DeprecatedBuiltinCode()
    return code


def elements(t):
    n = 1
    for k in range(t.ShapeLength()):
        n *= max(int(t.Shape(k)), 0)
    return n


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    try:
        buf = open(sys.argv[1], "rb").read()
        m = S.Model.GetRootAs(buf, 0)
    except (OSError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2
    floats, ints = {}, {}
    for sg_i in range(m.SubgraphsLength()):
        sg = m.Subgraphs(sg_i)
        for o_i in range(sg.OperatorsLength()):
            o = sg.Operators(o_i)
            if op_code(m, o.OpcodeIndex()) not in WEIGHT_OPS or o.InputsLength() < 2:
                continue
            ti = o.Inputs(1)
            if ti < 0:
                continue
            t = sg.Tensors(ti)
            b = t.Buffer()
            if b <= 0 or b >= m.BuffersLength():
                continue
            bb = m.Buffers(b)
            if not (bb.DataLength() or bb.Offset() > 1):  # a computed tensor (e.g. DEQUANTIZE's output) has no data
                continue
            (floats if t.Type() in FLOAT_TYPES else ints)[(sg_i, ti)] = elements(t)
    most = max(floats.values(), default=0)
    kept = sum(1 for n in ints.values() if 256 < n <= most)
    print(f"{most}\t{len(floats)}\t{kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
