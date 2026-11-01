#!/usr/bin/env python3
"""Checks of tfl_rank4.py --unique-names on a small graph written here (no download): a RoPE-style MUL by a [N,1,d]
constant, an ADD of a position table, two axis-0 STRIDED_SLICEs and a FULLY_CONNECTED. The plain rewrite repeats five
names on it (two `<name>/r4` and r4_begin / r4_end / r4_strides). With --unique-names: no name repeats, the file is the
one tfl_unique_names.py writes from the plain rewrite (byte for byte), tfl_same_fn.py reports SAME_FN
against the plain rewrite, the CPU output equals the input graph's, and ai-edge-quantizer takes the file where it refuses the plain
rewrite (that check is skipped when ai_edge_quantizer does not import).

    python tools/gpu_graph/test_tfl_rank4_names.py
"""
import collections
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

HERE = pathlib.Path(__file__).resolve().parent


def tool(name, *args):
    return subprocess.run([sys.executable, str(HERE / name), *map(str, args)], capture_output=True, text=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_") + "_by_path", HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_graph(path):
    """x[4,3,8] -> MUL rope[4,1,8] -> ADD pos[4,3,8] -> two axis-0 STRIDED_SLICEs [0:2], [2:4] -> ADD -> FC[5,8]."""
    rng = np.random.default_rng(1)
    m = s.ModelT()
    m.version = 3
    m.buffers = [s.BufferT()]
    m.operatorCodes = []
    sg = s.SubGraphT()
    sg.tensors, sg.operators = [], []

    def opcode(name):
        code = getattr(s.BuiltinOperator, name)
        for i, oc in enumerate(m.operatorCodes):
            if oc.builtinCode == code:
                return i
        oc = s.OperatorCodeT()
        oc.builtinCode, oc.deprecatedBuiltinCode, oc.version = code, min(code, 127), 1
        m.operatorCodes.append(oc)
        return len(m.operatorCodes) - 1

    def tensor(name, shape, ttype=s.TensorType.FLOAT32, data=None):
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name.encode(), list(shape), ttype, 0
        if data is not None:
            b = s.BufferT()
            b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), np.uint8)
            m.buffers.append(b)
            t.buffer = len(m.buffers) - 1
        sg.tensors.append(t)
        return len(sg.tensors) - 1

    def op(name, ins, outs, otype=None, opts=None):
        o = s.OperatorT()
        o.opcodeIndex, o.inputs, o.outputs = opcode(name), list(ins), list(outs)
        if otype is not None:
            o.builtinOptionsType, o.builtinOptions = otype, opts
        sg.operators.append(o)

    def sslice_opts():
        o = s.StridedSliceOptionsT()
        o.beginMask = o.endMask = o.ellipsisMask = o.newAxisMask = o.shrinkAxisMask = 0
        return o

    x = tensor("x", [4, 3, 8])
    rope = tensor("rope", [4, 1, 8], data=rng.random((4, 1, 8), dtype=np.float32))
    a = tensor("a", [4, 3, 8])
    pos = tensor("pos", [4, 3, 8], data=rng.random((4, 3, 8), dtype=np.float32))
    b = tensor("b", [4, 3, 8])
    op("MUL", [x, rope], [a], s.BuiltinOptions.MulOptions, s.MulOptionsT())
    op("ADD", [a, pos], [b], s.BuiltinOptions.AddOptions, s.AddOptionsT())
    halves = []
    for k, (lo, hi) in enumerate(((0, 2), (2, 4))):
        bg = tensor(f"begin{k}", [3], s.TensorType.INT32, np.array([lo, 0, 0], np.int32))
        en = tensor(f"end{k}", [3], s.TensorType.INT32, np.array([hi, 3, 8], np.int32))
        st = tensor(f"strides{k}", [3], s.TensorType.INT32, np.array([1, 1, 1], np.int32))
        h = tensor(f"half{k}", [2, 3, 8])
        op("STRIDED_SLICE", [b, bg, en, st], [h], s.BuiltinOptions.StridedSliceOptions, sslice_opts())
        halves.append(h)
    c = tensor("c", [2, 3, 8])
    op("ADD", halves, [c], s.BuiltinOptions.AddOptions, s.AddOptionsT())
    w = tensor("w", [5, 8], data=rng.standard_normal((5, 8)).astype(np.float32))
    y = tensor("y", [6, 5])
    fc = s.FullyConnectedOptionsT()
    fc.keepNumDims = False
    op("FULLY_CONNECTED", [c, w, -1], [y], s.BuiltinOptions.FullyConnectedOptions, fc)
    sg.inputs, sg.outputs = [x], [y]
    m.subgraphs = [sg]
    sig = s.SignatureDefT()
    sig.signatureKey, sig.subgraphIndex = b"serving_default", 0
    sig.inputs, sig.outputs = [], []
    for lst, name, ti in ((sig.inputs, b"x", x), (sig.outputs, b"y", y)):
        tm = s.TensorMapT()
        tm.name, tm.tensorIndex = name, ti
        lst.append(tm)
    m.signatureDefs = [sig]
    fb = flatbuffers.Builder(1 << 12)
    fb.Finish(m.Pack(fb), file_identifier=b"TFL3")
    pathlib.Path(path).write_bytes(bytes(fb.Output()))


def names(path):
    m = s.ModelT.InitFromPackedBuf(pathlib.Path(path).read_bytes(), 0)
    return [t.name for t in m.subgraphs[0].tensors]


def repeated(path):
    return {n: c for n, c in collections.Counter(names(path)).items() if c > 1}


class UniqueNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls.tmp.name)
        cls.src, cls.plain, cls.uniq, cls.renamed = (d / n for n in ("src.tflite", "plain.tflite", "uniq.tflite",
                                                                        "plain_u.tflite"))
        write_graph(cls.src)
        for out, extra in ((cls.plain, []), (cls.uniq, ["--unique-names", "--report", d / "uniq.json"])):
            r = tool("tfl_rank4.py", "rewrite", cls.src, out, *extra)
            assert r.returncode == 0, r.stderr
        cls.uniq_stdout = r.stdout
        cls.report = __import__("json").loads((d / "uniq.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_input_names_unique(self):
        self.assertEqual(repeated(self.src), {})

    def test_plain_rewrite_repeats_names(self):
        self.assertEqual(repeated(self.plain), {b"a/r4": 2, b"b/r4": 2, b"r4_begin": 2, b"r4_end": 2,
                                                b"r4_strides": 2})

    def test_unique_names(self):
        self.assertEqual(repeated(self.uniq), {})
        self.assertEqual(len(self.report["renamed"]), 5)
        self.assertIn("renamed 5", self.uniq_stdout)
        self.assertEqual(sorted(r["to"] for r in self.report["renamed"]),
                         ["a/r4#2", "b/r4#2", "r4_begin#2", "r4_end#2", "r4_strides#2"])

    def test_only_names_differ_from_plain(self):
        a, b = names(self.plain), names(self.uniq)
        self.assertEqual(len(a), len(b))
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 5)

    def test_same_bytes_as_tfl_unique_names(self):
        r = tool("tfl_unique_names.py", self.plain, self.renamed)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("renamed 5", r.stdout)
        self.assertEqual(self.renamed.read_bytes(), self.uniq.read_bytes())

    def test_same_function(self):
        r = tool("tfl_same_fn.py", self.plain, self.uniq)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SAME_FN", r.stdout)
        run_cpu = load("tfl_same_fn.py").run_cpu
        y_src, _ = run_cpu(str(self.src), 0)
        y_new, _ = run_cpu(str(self.uniq), 0)
        self.assertEqual(float(np.abs(y_src - y_new).max()), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("ai_edge_quantizer"), "ai_edge_quantizer not installed")
    def test_quantizer_takes_unique_refuses_plain(self):
        d = pathlib.Path(self.tmp.name)
        ok = tool("tfl_int8_weights.py", self.uniq, d / "uniq_i8.tflite")
        self.assertEqual(ok.returncode, 0, ok.stderr[-2000:])
        self.assertIn("QUANT_DONE", ok.stdout)
        bad = tool("tfl_int8_weights.py", self.plain, d / "plain_i8.tflite")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("not unique", bad.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
