#!/usr/bin/env python3
"""Checks of tfl_rank4.py's rule promote (--promote-lower-rank) on small graphs written here (no download).

The graph: inputs x [6,4] and y [6,1];
  op 0  MUL(x, const [1,1,4])       -> [1,6,4]   promote: runtime rank 2 < 3, 24 elements both
  op 1  ADD(const [1,6,4], x)       -> [1,6,4]   promote (constant first; x to [1,6,4] again: the same RESHAPE)
  op 2  MUL(op 0, op 1)             -> [1,6,4]   both operands runtime: left alone
  op 3  ADD(x, const [6,4])         -> [6,4]     the same rank: left alone
  op 4  MUL(y, const [1,1,4])       -> [1,6,4]   a broadcast that adds elements (6 -> 24): left alone
  op 5  SUB(const [1,4], const [1,1,4]) -> [1,1,4]  both operands constant: left alone
  op 6  DIV(op 3, const [1,1,4])    -> [1,6,4]   promote
  outputs: op 2, op 4, op 5, op 6.
Rule promote takes ops 0, 1 and 6 (two RESHAPEs, x shared) and nothing else, and the CPU outputs stay byte for byte;
with the default rules added (--promote-lower-rank) as well. An int8 operand with quantization parameters is left
alone. Without the rule the tool writes the bytes it wrote before the rule existed (tools/gpu_graph/tfl_rank4.py at
commit 62de36d, read with git; skipped when git cannot read it), on this graph and on test_tfl_rank4_names.py's.

    python tools/gpu_graph/test_tfl_rank4_promote.py
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

HERE = pathlib.Path(__file__).resolve().parent
BEFORE = "62de36d"  # the last commit of tfl_rank4.py without rule promote


def tool(path, *args):
    return subprocess.run([sys.executable, str(path), *map(str, args)], capture_output=True, text=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_") + "_by_path", HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Builder:
    def __init__(self):
        self.m = s.ModelT()
        self.m.version = 3
        self.m.buffers = [s.BufferT()]
        self.m.operatorCodes = []
        self.sg = s.SubGraphT()
        self.sg.tensors, self.sg.operators = [], []

    def opcode(self, name):
        code = getattr(s.BuiltinOperator, name)
        for i, oc in enumerate(self.m.operatorCodes):
            if oc.builtinCode == code:
                return i
        oc = s.OperatorCodeT()
        oc.builtinCode, oc.deprecatedBuiltinCode, oc.version = code, min(code, 127), 1
        self.m.operatorCodes.append(oc)
        return len(self.m.operatorCodes) - 1

    def tensor(self, name, shape, ttype=s.TensorType.FLOAT32, data=None, scale=None):
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name.encode(), list(shape), ttype, 0
        if data is not None:
            b = s.BufferT()
            b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), np.uint8)
            self.m.buffers.append(b)
            t.buffer = len(self.m.buffers) - 1
        if scale is not None:
            t.quantization = s.QuantizationParametersT()
            t.quantization.scale, t.quantization.zeroPoint = [float(scale)], [0]
        self.sg.tensors.append(t)
        return len(self.sg.tensors) - 1

    def op(self, name, ins, outs):
        o = s.OperatorT()
        o.opcodeIndex, o.inputs, o.outputs = self.opcode(name), list(ins), list(outs)
        o.builtinOptionsType = getattr(s.BuiltinOptions, name.capitalize() + "Options")
        o.builtinOptions = getattr(s, name.capitalize() + "OptionsT")()
        self.sg.operators.append(o)

    def save(self, path, inputs, outputs):
        self.sg.inputs, self.sg.outputs = list(inputs), list(outputs)
        self.m.subgraphs = [self.sg]
        sig = s.SignatureDefT()
        sig.signatureKey, sig.subgraphIndex = b"serving_default", 0
        sig.inputs, sig.outputs = [], []
        for lst, ids in ((sig.inputs, inputs), (sig.outputs, outputs)):
            for ti in ids:
                tm = s.TensorMapT()
                tm.name, tm.tensorIndex = self.sg.tensors[ti].name, ti
                lst.append(tm)
        self.m.signatureDefs = [sig]
        fb = flatbuffers.Builder(1 << 12)
        fb.Finish(self.m.Pack(fb), file_identifier=b"TFL3")
        pathlib.Path(path).write_bytes(bytes(fb.Output()))


def write_graph(path):
    rng = np.random.default_rng(2)
    b = Builder()

    def const(name, shape):
        return b.tensor(name, shape, data=(rng.random(shape, dtype=np.float32) + 0.5).astype(np.float32))

    x, y = b.tensor("x", [6, 4]), b.tensor("y", [6, 1])
    a, c, d = b.tensor("a", [1, 6, 4]), b.tensor("c", [1, 6, 4]), b.tensor("d", [1, 6, 4])
    e, f, g, h = b.tensor("e", [6, 4]), b.tensor("f", [1, 6, 4]), b.tensor("g", [1, 1, 4]), b.tensor("h", [1, 6, 4])
    b.op("MUL", [x, const("w0", [1, 1, 4])], [a])
    b.op("ADD", [const("w1", [1, 6, 4]), x], [c])
    b.op("MUL", [a, c], [d])
    b.op("ADD", [x, const("w3", [6, 4])], [e])
    b.op("MUL", [y, const("w4", [1, 1, 4])], [f])
    b.op("SUB", [const("w5", [1, 4]), const("w6", [1, 1, 4])], [g])
    b.op("DIV", [e, const("w7", [1, 1, 4])], [h])
    b.save(path, [x, y], [d, f, g, h])


def write_quantized_graph(path):
    """xq int8 [6,4] (scale 0.05) x const int8 [1,1,4] (scale 0.02) -> int8 [1,6,4] (scale 0.1)."""
    b = Builder()
    xq = b.tensor("xq", [6, 4], s.TensorType.INT8, scale=0.05)
    w = b.tensor("wq", [1, 1, 4], s.TensorType.INT8, data=np.array([1, 2, 3, 4], np.int8).reshape(1, 1, 4), scale=0.02)
    yq = b.tensor("yq", [1, 6, 4], s.TensorType.INT8, scale=0.1)
    b.op("MUL", [xq, w], [yq])
    b.save(path, [xq], [yq])


def graph(path):
    m = s.ModelT.InitFromPackedBuf(pathlib.Path(path).read_bytes(), 0)
    sg = m.subgraphs[0]
    names = {v: k for k, v in s.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    ops = [(names[max(m.operatorCodes[o.opcodeIndex].builtinCode, m.operatorCodes[o.opcodeIndex].deprecatedBuiltinCode)],
            [sg.tensors[i].name.decode() for i in o.inputs], [sg.tensors[i].name.decode() for i in o.outputs])
           for o in sg.operators]
    shapes = {t.name.decode(): [int(v) for v in t.shape] for t in sg.tensors}
    return ops, shapes


def run_cpu(path, feeds):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import Options
    model = CompiledModel.from_file(str(path), options=Options(hardware_accelerators=HardwareAccelerator.CPU))
    sig = list(model.get_signature_list().keys())[0]
    ins = {}
    for n in model.get_input_tensor_details(sig):
        ins[n] = model.create_input_buffer_by_name(sig, n)
        ins[n].write(np.ascontiguousarray(feeds[n]))
    det = model.get_output_tensor_details(sig)
    outs = {n: model.create_output_buffer_by_name(sig, n) for n in det}
    model.run_by_name(sig, ins, outs)
    res = {n: np.array(outs[n].read(int(np.prod(d["shape"])), np.float32), np.float32).reshape(d["shape"])
           for n, d in det.items()}
    del model
    return res


class Promote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = pathlib.Path(cls.tmp.name)
        cls.src = d / "src.tflite"
        write_graph(cls.src)
        cls.runs = {}
        for tag, extra in (("promote", ["--rules", "promote"]), ("all", ["--promote-lower-rank"]),
                           ("default", [])):
            out, rep = d / f"{tag}.tflite", d / f"{tag}.json"
            r = tool(HERE / "tfl_rank4.py", "rewrite", cls.src, out, "--report", rep, *extra)
            assert r.returncode == 0, r.stderr
            cls.runs[tag] = (out, json.loads(rep.read_text()), r.stdout)
        rng = np.random.default_rng(3)
        cls.feeds = {"x": rng.standard_normal((6, 4)).astype(np.float32),
                     "y": rng.standard_normal((6, 1)).astype(np.float32)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sites(self):
        _, rep, stdout = self.runs["promote"]
        self.assertEqual(rep["rules"], ["promote"])
        self.assertEqual([(r["op"], r["name"], r["runtime"], r["const"], r["out"]) for r in rep["sites"]],
                         [(0, "MUL", [6, 4], [1, 1, 4], [1, 6, 4]), (1, "ADD", [6, 4], [1, 6, 4], [1, 6, 4]),
                          (6, "DIV", [6, 4], [1, 1, 4], [1, 6, 4])])
        self.assertIn("sites {'promote': 3}", stdout)

    def test_reshapes(self):
        ops, shapes = graph(self.runs["promote"][0])
        self.assertEqual([o[0] for o in ops], ["RESHAPE", "MUL", "ADD", "MUL", "ADD", "MUL", "SUB", "RESHAPE", "DIV"])
        self.assertEqual(ops[0][1][0], "x")
        self.assertEqual(ops[0][2], ["x/promoted"])
        self.assertEqual(ops[1][1][0], "x/promoted")
        self.assertEqual(ops[2][1][1], "x/promoted")
        self.assertEqual(ops[7][1][0], "e")
        self.assertEqual(ops[7][2], ["e/promoted"])
        self.assertEqual(ops[8][1][0], "e/promoted")
        self.assertEqual(shapes["x/promoted"], [1, 6, 4])
        self.assertEqual(shapes["e/promoted"], [1, 6, 4])

    def test_others_left_alone(self):
        src_ops, _ = graph(self.src)
        new_ops = [o for o in graph(self.runs["promote"][0])[0] if o[0] != "RESHAPE"]
        for k in (2, 3, 4, 5):
            self.assertEqual(new_ops[k], src_ops[k], k)

    def test_cpu_output_same_bytes(self):
        ref = run_cpu(self.src, self.feeds)
        for tag in ("promote", "all"):
            got = run_cpu(self.runs[tag][0], self.feeds)
            self.assertEqual(sorted(got), sorted(ref))
            for n in ref:
                self.assertEqual(got[n].tobytes(), ref[n].tobytes(), (tag, n))

    def test_flag_adds_rule_to_defaults(self):
        _, rep, stdout = self.runs["all"]
        self.assertEqual(rep["rules"], ["bmm", "elem", "mean", "promote", "softmax", "sslice"])
        self.assertEqual([(r["rule"], r["op"]) for r in rep["sites"]],
                         [("promote", 0), ("promote", 1), ("elem", 3), ("promote", 6)])
        self.assertIn("sites {'promote': 3, 'elem': 1}", stdout)

    def test_default_has_no_promote(self):
        _, rep, _ = self.runs["default"]
        self.assertNotIn("promote", rep["rules"])
        self.assertEqual([(r["rule"], r["op"]) for r in rep["sites"]], [("elem", 1), ("elem", 3)])

    def test_quantized_operand_left_alone(self):
        q, out, rep = self.d / "q.tflite", self.d / "q_out.tflite", self.d / "q.json"
        write_quantized_graph(q)
        r = tool(HERE / "tfl_rank4.py", "rewrite", q, out, "--rules", "promote", "--report", rep)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(rep.read_text())["sites"], [])

    def test_default_bytes_as_before(self):
        old = self.d / "tfl_rank4_before.py"
        r = subprocess.run(["git", "-C", str(HERE), "show", f"{BEFORE}:tools/gpu_graph/tfl_rank4.py"],
                           capture_output=True)
        if r.returncode != 0:
            self.skipTest(f"git cannot read {BEFORE}")
        old.write_bytes(r.stdout)
        names_src = self.d / "names_src.tflite"
        load("test_tfl_rank4_names.py").write_graph(names_src)
        for src in (self.src, names_src):
            for extra in ([], ["--unique-names"]):
                a, b = self.d / "before.tflite", self.d / "after.tflite"
                ra = tool(old, "rewrite", src, a, *extra)
                rb = tool(HERE / "tfl_rank4.py", "rewrite", src, b, *extra)
                self.assertEqual((ra.returncode, rb.returncode), (0, 0), ra.stderr + rb.stderr)
                self.assertEqual(a.read_bytes(), b.read_bytes(), (src.name, extra))


if __name__ == "__main__":
    unittest.main(verbosity=2)
