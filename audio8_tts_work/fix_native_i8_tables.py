"""Native-int8 codec fix-up: litert_torch's PT2E DYNAMIC conversion also quantizes the unannotated RVQ codebook
tables (asymmetric int8), which TFLite's EMBEDDING_LOOKUP kernel refuses (zero_point must be 0). Put the original
fp32 codebooks back (matched by shape + dequantized content) and rewrite the flatbuffer.
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/fix_native_i8_tables.py <in.tflite> <out.tflite>
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arktts_port as P
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as U
from ai_edge_litert import schema_py_generated as S

src, dst = sys.argv[1], sys.argv[2]
codec, _ = P.load_codec()
tables = [codec.quantizer.semantic_quantizer.quantizers[0].codebook.weight.detach().numpy()] + \
         [q.codebook.weight.detach().numpy() for q in codec.quantizer.quantizer.quantizers]
m = U.read_model(src); sg = m.subgraphs[0]
EMB = S.BuiltinOperator.EMBEDDING_LOOKUP
n_fixed = 0
for op in sg.operators:
    if m.operatorCodes[op.opcodeIndex].builtinCode != EMB:
        continue
    t = sg.tensors[op.inputs[1]]
    if t.type != S.TensorType.INT8:
        continue
    q = t.quantization; scale, zp = float(q.scale[0]), int(q.zeroPoint[0])
    raw = np.frombuffer(m.buffers[t.buffer].data, dtype=np.int8).reshape(list(t.shape))
    deq = (raw.astype(np.float32) - zp) * scale
    cands = [(np.abs(tb - deq).max(), i) for i, tb in enumerate(tables) if tb.shape == tuple(t.shape)]
    err, idx = min(cands)
    assert err < 2 * scale, f"no matching fp32 table for {t.name} (best max|d| {err} vs scale {scale})"
    m.buffers[t.buffer].data = tables[idx].astype(np.float32).tobytes()
    t.type = S.TensorType.FLOAT32
    t.quantization = None
    n_fixed += 1
U.write_model(m, dst) if "write_model" in dir(U) else None
print(f"replaced {n_fixed} int8 codebook tables with fp32 originals -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
