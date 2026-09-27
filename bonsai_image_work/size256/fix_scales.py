"""Replace zero blockwise-quantization scales in a bare .tflite with an epsilon.

Same patch as ../../bonsai_work/fix_zero_block_scales.py::patch_tflite (which
wraps it for .litertlm bundles): a ternary weight block can be all zeros, so
min-max blockwise int4 emits scale = 0 for it and XNNPACK refuses to prepare
("unsupported scale value (0.000000) ... for INT4 tensor"). The quantized
values in such a block are all 0, so any positive scale dequantizes the same;
we substitute the tensor's smallest nonzero scale.

Usage: fix_scales.py <in.tflite> <out.tflite>
"""
import shutil
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as schema


def patch_tflite(path):
    data = bytearray(open(path, "rb").read())
    model = schema.Model.GetRootAsModel(data, 0)
    n_fixed = n_tensors = 0
    seen_bufs = set()
    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        for i in range(sg.TensorsLength()):
            q = sg.Tensors(i).Quantization()
            if q is None or q.DetailsType() != \
                    schema.QuantizationDetails.BlockwiseQuantization:
                continue
            bq = schema.BlockwiseQuantization()
            tab = q.Details()
            bq.Init(tab.Bytes, tab.Pos)
            st = sg.Tensors(bq.Scales())
            bidx = st.Buffer()
            if bidx in seen_bufs:
                continue
            seen_bufs.add(bidx)
            buf = model.Buffers(bidx)
            arr = buf.DataAsNumpy()
            if isinstance(arr, np.ndarray):
                f16 = arr.view(np.float16)  # view into `data` (inline buffer)
            else:
                # >2GB-style serialization: buffer data lives out-of-band at an
                # absolute file offset (Buffer.offset/size), not in the vector.
                off, size = buf.Offset(), buf.Size()
                if not size:
                    continue
                f16 = np.ndarray(size // 2, dtype=np.float16, buffer=data,
                                 offset=off)
            zeros = f16 == 0
            if zeros.any():
                nzmin = f16[~zeros].min() if (~zeros).any() else np.float16(1e-4)
                f16[zeros] = nzmin  # writes into `data`
                n_fixed += int(zeros.sum())
                n_tensors += 1
    print(f"patched {n_fixed} zero scales across {n_tensors} scale tensors")
    open(path, "wb").write(data)
    return n_fixed


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copyfile(src, dst)
    patch_tflite(dst)
    print("wrote", dst)
