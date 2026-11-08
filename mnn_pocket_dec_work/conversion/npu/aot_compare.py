"""Compare a compiled *_SM8850.tflite file with the published one (conversion/README.md, step 8).

    venv-export/bin/python conversion/npu/aot_compare.py work/<name>_SM8850.tflite <name>_SM8850.tflite

Each file is a TFLite flatbuffer with one DISPATCH_OP, whose options (bytecode_offset, bytecode_size) point at the QNN context binary
stored after the flatbuffer. The context binary holds some 16-bit values in 32-bit words and the compiler does not write the upper 16
bits of those words, so they keep whatever its memory held: two compiles of the same graph can differ there and nowhere else.

Prints "same bytes" when the files are equal. Prints "same apart from unwritten bits" when they have the same size, the same bytes
before the context binary (the flatbuffer with the DISPATCH_OP and its options), and every 32-bit word that differs keeps its lower 16
bits. Otherwise prints "different" and the first place that breaks those rules. Exit code 0 for the first two, 1 otherwise."""
import hashlib
import mmap
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from flatbuffers import flexbuffers


def sha256(m):
    h = hashlib.sha256()
    for off in range(0, len(m), 1 << 24):
        h.update(m[off:off + (1 << 24)])
    return h.hexdigest()


def dispatch_options(m):
    fb = schema.Model.GetRootAsModel(m, 0)
    sg = fb.Subgraphs(0)
    if fb.SubgraphsLength() != 1 or sg.OperatorsLength() != 1:
        return None
    op = sg.Operators(0)
    if fb.OperatorCodes(op.OpcodeIndex()).CustomCode() != b"DISPATCH_OP":
        return None
    return flexbuffers.Loads(bytes(op.CustomOptionsAsNumpy().tobytes()))


def main(a_path, b_path):
    fa, fb = open(a_path, "rb"), open(b_path, "rb")
    a = mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ)
    b = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
    print(f"{a_path}: {len(a):,} bytes, SHA-256 {sha256(a)}")
    print(f"{b_path}: {len(b):,} bytes, SHA-256 {sha256(b)}")
    if len(a) == len(b) and a[:] == b[:]:
        print("same bytes")
        return 0
    if len(a) != len(b):
        print(f"different: sizes {len(a):,} and {len(b):,}")
        return 1
    oa, ob = dispatch_options(a), dispatch_options(b)
    if oa is None or oa != ob:
        print(f"different: not one DISPATCH_OP with the same options ({oa} and {ob})")
        return 1
    start = oa["bytecode_offset"]
    if a[:start] != b[:start]:
        print(f"different: the {start:,} bytes before the context binary differ")
        return 1
    x = np.frombuffer(a, dtype=np.uint8)
    y = np.frombuffer(b, dtype=np.uint8)
    words, n = 0, len(a) - len(a) % 4
    for off in range(0, n, 1 << 24):
        wx = x[off:min(off + (1 << 24), n)].view("<u4")
        wy = y[off:min(off + (1 << 24), n)].view("<u4")
        d = np.nonzero(wx != wy)[0]
        bad = d[(wx[d] & 0xFFFF) != (wy[d] & 0xFFFF)]
        if bad.size:
            print(f"different: the 32-bit word at byte {off + 4 * int(bad[0]):,} differs in its lower 16 bits")
            return 1
        words += int(d.size)
    if x[n:].tobytes() != y[n:].tobytes():
        print("different: the last bytes differ")
        return 1
    print(f"same apart from unwritten bits: {words:,} 32-bit words of the context binary differ, each in its upper 16 bits only")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
