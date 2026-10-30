#!/usr/bin/env python3
"""The DEPTHWISE_CONV_2D filters of one TFLite section that must stay in the graph, and the threshold that keeps them.

  dw_floor.py SECTION.tflite

Prints one line, `<elements> <TAB> <filters>`: the largest element count among the section's DEPTHWISE_CONV_2D
filters (input 1) that are INT8 or UINT8 with per-channel scales (more than one scale), and how many such filter
tensors the section has (counted per subgraph); `0 <TAB> 0` when it has none. LiteRT's GPU delegate refuses such a
filter once it is an external buffer: ML Drift's operation parser reads it as a non-constant tensor and its
PreCheckReadValueByTensorIdx returns "Unsupported quantization scale size", so no GPU engine is built (LFM2 ShortConv
int8 files, LiteRT-LM 0.19.0, OpenCL and WebGPU). LiteRT's externalize_tflite_flatbuffer.py keeps a tensor inline when
its element count is at most --num_elements_threshold, so a threshold of at least <elements> keeps these filters in
the graph. Reads the flatbuffer only (ai_edge_litert's generated schema, lazy API). Exit 0, or 2 on bad input.
"""
import sys

from ai_edge_litert import schema_py_generated as S

INT8_TYPES = {S.TensorType.INT8, S.TensorType.UINT8}


def op_code(m, opcode_index):
    c = m.OperatorCodes(opcode_index)
    code = c.BuiltinCode()
    if code == 0 and c.DeprecatedBuiltinCode() != 0:
        code = c.DeprecatedBuiltinCode()
    return code


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
    most = 0
    filters = set()
    for sg_i in range(m.SubgraphsLength()):
        sg = m.Subgraphs(sg_i)
        for o_i in range(sg.OperatorsLength()):
            o = sg.Operators(o_i)
            if op_code(m, o.OpcodeIndex()) != S.BuiltinOperator.DEPTHWISE_CONV_2D or o.InputsLength() < 2:
                continue
            ti = o.Inputs(1)
            if ti < 0:
                continue
            t = sg.Tensors(ti)
            q = t.Quantization()
            if t.Type() not in INT8_TYPES or q is None or q.ScaleLength() <= 1:
                continue
            n = 1
            for k in range(t.ShapeLength()):
                n *= max(int(t.Shape(k)), 0)
            filters.add((sg_i, ti))
            most = max(most, n)
    print(f"{most}\t{len(filters)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
