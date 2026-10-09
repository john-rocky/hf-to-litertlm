"""Round 3 helpers shared by graph_build.py, quant_forms.py and litert_gate.py (exporter venv ~/venvs/lt094dev).

- runtime_log_verbose(): copied verbatim from d1_3b_work/scripts/d1_check.py `runtime_log_verbose` (d1a round 2,
  2026-10-08; read-only source). On the Mac the CompiledModel wrapper runs the runtime linked into
  libpywrap_litert_common.dylib and the default log level hides the `Replacing N out of M node(s) with delegate` line;
  this switches the runtime's VERBOSE lines on through the library's local symbols (checked against the file first).
- capture_fd2(): this process's fd 2 -> a log file (the runtime writes its lines there), restored on exit.
- delegation_from_log(): the Replacing / Partitioned lines parsed into numbers, plus every key line verbatim.
- scan(): static flatbuffer scan (ai_edge_litert.schema_py_generated, mmapped, never runs the model): op histogram,
  tensor rank / dtype histograms, INT64 and rank > 4 tensors, GPU-risk op sites, BATCH_MATMUL shapes, FULLY_CONNECTED
  weight dtype through a DEQUANTIZE producer, EMBEDDING_LOOKUP table dtype / shape / bytes, constant bytes by dtype,
  signatures. Built from d1_3b_work/scripts/tflite_scan.py `scan` and kev_work/scripts/quantize_kev.py `quant_scan`.
- open_compiled() / Runner: one CompiledModel (CPU XNNPACK with N threads, or GPU = Metal with GpuOptions
  enforce_f32 for precision fp32), buffers created by signature name, the six inputs written by name, `scores` read.

Round 4 additions (nothing above changes behaviour): open_compiled(share=True) = GpuOptions(constant_tensor_sharing);
scan() also reports the constant bytes per dtype counted once per flatbuffer buffer (`constant_unique_buffer_*`: a
multi-signature file whose subgraphs share a weight buffer counts it once there, once per subgraph tensor in
`constant_bytes_by_dtype`) and each EMBEDDING_LOOKUP table's buffer index; memory() = this process's ru_maxrss and
proc_pid_rusage(RUSAGE_INFO_V4) phys_footprint / lifetime max (the value `vmmap -summary` prints as Physical footprint)
plus `ps -o rss`.
"""
from __future__ import annotations

import collections
import contextlib
import ctypes
import hashlib
import json
import mmap
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("ids", "prefix", "media", "pad", "keep_right", "qtype_onehot")
LINE_KEYS = re.compile(r"(?i)replacing|partition|delegat|fail|error|unsupported|not supported|abort|fallback|reject")
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions")
PARTITIONED = re.compile(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions")
GPU_RISK_OPS = ("GATHER_ND", "GATHER", "CAST", "SELECT_V2", "SELECT", "BROADCAST_TO", "MAXIMUM", "RESIZE_BILINEAR",
                "RESIZE_NEAREST_NEIGHBOR", "EMBEDDING_LOOKUP", "PAD", "PADV2")
NP_OF = {"FLOAT32": np.float32, "INT32": np.int32, "INT64": np.int64, "FLOAT16": np.float16, "INT8": np.int8,
         "UINT8": np.uint8, "BOOL": np.bool_, "INT16": np.int16}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def dump_json(path, doc, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- runtime log (copied from d1a, verbatim body)

def runtime_log_verbose() -> dict:
    """Let the runtime print its VERBOSE lines (the TFLite `Replacing N out of M node(s) with delegate` line, LiteRT's
    `Partitioned subgraph` line). The CompiledModel wrapper runs the runtime linked into libpywrap_litert_common.dylib
    (libLiteRt.dylib is not loaded on this path, and the common library does not export its logger API), so the local
    symbols are used: their `nm` values plus the image's slide from dyld. Before any use, each address is checked
    against the file (code: the first 16 bytes; data: the 4-byte initial value, which must be INFO = 1); on any mismatch
    nothing is touched. Then both MinimalLogger severity words are set to VERBOSE (0) and the LiteRT default logger's
    minimum severity to VERBOSE (LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), 0))."""
    import subprocess

    import ai_edge_litert.compiled_model  # noqa: F401  (loads the library)

    lib = (Path(ai_edge_litert.compiled_model.__file__).parent / "libpywrap_litert_common.dylib").resolve()
    want = {"get_logger": "_LiteRtGetDefaultLogger", "set_min": "_LiteRtSetMinLoggerSeverity",
            "get_min": "_LiteRtGetMinLoggerSeverity",
            "tflite_min": "__ZN6tflite16logging_internal13MinimalLogger21minimum_log_severity_E",
            "anon_min": "__ZN12_GLOBAL__N_113MinimalLogger21minimum_log_severity_E"}
    syms = {}
    for line in subprocess.run(["nm", str(lib)], capture_output=True, text=True, check=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in want.values():
            syms[parts[2]] = int(parts[0], 16)
    sysdl = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    sysdl._dyld_image_count.restype = ctypes.c_uint32
    sysdl._dyld_get_image_name.restype = ctypes.c_char_p
    sysdl._dyld_get_image_vmaddr_slide.restype = ctypes.c_long
    idx = [i for i in range(sysdl._dyld_image_count())
           if Path(sysdl._dyld_get_image_name(i).decode()).resolve() == lib]
    doc = {"library": str(lib), "symbols_found": sorted(k for k, v in want.items() if v in syms), "image_matches": len(idx)}
    if len(idx) != 1 or len(syms) != len(want):
        doc["applied"] = False
        return doc
    slide = sysdl._dyld_get_image_vmaddr_slide(idx[0])
    blob = lib.read_bytes()
    addr = {k: syms[v] + slide for k, v in want.items()}
    code_ok = {k: ctypes.string_at(addr[k], 16) == blob[syms[want[k]]: syms[want[k]] + 16]
               for k in ("get_logger", "set_min", "get_min")}
    data_init = {k: int.from_bytes(blob[syms[want[k]]: syms[want[k]] + 4], "little", signed=True)
                 for k in ("tflite_min", "anon_min")}
    data_now = {k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")}
    doc.update(code_bytes_match=code_ok, data_initial=data_init, data_before=data_now)
    if not all(code_ok.values()) or any(v != 1 for v in data_init.values()) or any(v != 1 for v in data_now.values()):
        doc["applied"] = False
        return doc
    for k in ("tflite_min", "anon_min"):
        ctypes.c_int.from_address(addr[k]).value = 0
    get_logger = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr["get_logger"])
    set_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)(addr["set_min"])
    get_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(addr["get_min"])
    logger = get_logger()
    before, after = ctypes.c_int(0), ctypes.c_int(0)
    st_before = get_min(logger, ctypes.byref(before))
    st_set = set_min(logger, 0)
    st_after = get_min(logger, ctypes.byref(after))
    doc.update(applied=True, data_after={k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")},
               litert_logger={"status_get_before": st_before, "before_low_byte": before.value & 0xFF, "status_set": st_set,
                              "status_get_after": st_after, "after_low_byte": after.value & 0xFF})
    return doc


@contextlib.contextmanager
def capture_fd2(log_path):
    """fd 2 of this process -> log_path for the duration (the runtime's C++ lines go there)."""
    log_path = Path(log_path)
    sys.stderr.flush()
    f = open(log_path, "w")
    saved = os.dup(2)
    os.dup2(f.fileno(), 2)
    try:
        yield log_path
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        f.close()


def delegation_from_log(log_path, max_lines=120):
    lines = Path(log_path).read_text(errors="replace").splitlines()
    text = "\n".join(lines)
    return {
        "runtime_log": str(Path(log_path).relative_to(K)) if Path(log_path).is_relative_to(K) else str(log_path),
        "runtime_log_lines": len(lines),
        "key_lines": [ln for ln in lines if LINE_KEYS.search(ln)][:max_lines],
        "replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                      for m in REPLACING.findall(text)],
        "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                        for m in PARTITIONED.findall(text)],
    }


# ---------------------------------------------------------------- static scan

def scan(path, with_sha=True):
    from ai_edge_litert import schema_py_generated as schema

    path = Path(path)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "UNKNOWN")
        if name == "CUSTOM":
            name = "CUSTOM:" + (c.CustomCode().decode() if c.CustomCode() else "")
        codes.append(name)

    def buffer_array(tensor):
        b = model.Buffers(tensor.Buffer())
        dtype = NP_OF.get(type_names.get(tensor.Type()))
        if dtype is None:
            return None
        if b.DataLength() > 0:
            raw = b.DataAsNumpy().tobytes()
        elif b.Offset() > 1:
            raw = mm[b.Offset(): b.Offset() + b.Size()]
        else:
            return None
        shp = tensor.ShapeAsNumpy().tolist() if tensor.ShapeLength() else []
        return np.frombuffer(raw, dtype=dtype).reshape(shp)

    hist, ranks, dtypes = collections.Counter(), collections.Counter(), collections.Counter()
    const_bytes, const_count = collections.Counter(), collections.Counter()
    uniq_bytes, uniq_count, seen_buffers = collections.Counter(), collections.Counter(), set()
    risk = collections.defaultdict(list)
    pads, bmm_groups, customs = [], collections.Counter(), collections.Counter()
    int64_t, rank_gt4 = [], []
    fc_rows, emb_rows = [], []
    fc_weight_src, fc_weight_direct, fc_in_rank, fc_bias = (collections.Counter() for _ in range(4))
    deq_in_out = collections.Counter()
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        shape = lambda t: tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else []
        tname = lambda t: tensors[t].Name().decode() if t >= 0 else None
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {}
        for oi, op in enumerate(ops):
            for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []):
                producer[int(t)] = oi
        graph_inputs = {int(x) for x in g.InputsAsNumpy()} if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in producer and t not in graph_inputs

        def nbytes(t):
            b = model.Buffers(tensors[t].Buffer())
            return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

        def quant(t):
            q = tensors[t].Quantization()
            if q is None or q.ScaleLength() == 0:
                return None
            return {"scales": q.ScaleLength(), "quantized_dimension": q.QuantizedDimension(),
                    "zero_points_nonzero": int((q.ZeroPointAsNumpy() != 0).sum()) if q.ZeroPointLength() else 0}

        for ti in range(len(tensors)):
            s = shape(ti)
            ranks[len(s)] += 1
            dtypes[ttype(ti)] += 1
            if is_const(ti):
                const_bytes[ttype(ti)] += nbytes(ti)
                const_count[ttype(ti)] += 1
                buf = tensors[ti].Buffer()
                if buf not in seen_buffers and nbytes(ti) > 0:
                    seen_buffers.add(buf)
                    uniq_bytes[ttype(ti)] += nbytes(ti)
                    uniq_count[ttype(ti)] += 1
            if ttype(ti) == "INT64":
                int64_t.append({"tensor": tname(ti), "shape": s, "constant": is_const(ti)})
            if len(s) > 4:
                rank_gt4.append({"tensor": tname(ti), "shape": s})
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
            outs = [int(x) for x in op.OutputsAsNumpy()] if op.OutputsLength() else []
            site = {"subgraph": gi, "op_index": oi, "outputs": [tname(t) for t in outs],
                    "inputs": [{"shape": shape(t), "dtype": ttype(t), "constant": is_const(t)} for t in ins if t >= 0],
                    "output_shapes": [shape(t) for t in outs]}
            if name in GPU_RISK_OPS:
                risk[name].append(site)
            if name.startswith("CUSTOM"):
                customs[name] += 1
            if name == "DEQUANTIZE":
                deq_in_out[f"{ttype(ins[0])}->{ttype(outs[0])}"] += 1
            if name in ("PAD", "PADV2", "MIRROR_PAD"):
                x, p = ins[0], ins[1]
                arr = buffer_array(tensors[p]) if is_const(p) else None
                axes = None if arr is None else [int(a) for a in range(arr.shape[0]) if (arr[a] != 0).any()]
                pads.append({"op": name, "op_index": oi, "input_shape": shape(x), "rank": len(shape(x)),
                             "paddings": None if arr is None else arr.tolist(), "padded_axes": axes})
            if name == "BATCH_MATMUL":
                key = json.dumps([shape(t) for t in ins] + [["const" if is_const(t) else "act" for t in ins]]
                                 + [shape(outs[0])])
                bmm_groups[key] += 1
            if name == "FULLY_CONNECTED":
                w = ins[1]
                direct = ttype(w) if is_const(w) else None
                src, via = direct, ("constant" if is_const(w) else "activation")
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    dq_in = int(ops[producer[w]].InputsAsNumpy()[0])
                    src, via = ttype(dq_in), "DEQUANTIZE(" + ttype(dq_in) + ")"
                fc_weight_src[str(src)] += 1
                fc_weight_direct[str(direct)] += 1
                fc_in_rank[len(shape(ins[0]))] += 1
                b = ins[2] if len(ins) > 2 else -1
                fc_bias[ttype(b) if b >= 0 else "none"] += 1
                fc_rows.append({"op_index": oi, "out": tname(outs[0]), "input_shape": shape(ins[0]),
                                "weight_shape": shape(w), "weight_source_dtype": src, "weight_via": via,
                                "weight_quant": quant(w) if is_const(w) else None})
            if name == "EMBEDDING_LOOKUP":
                tbl = ins[1]
                src_t, via = tbl, ("constant" if is_const(tbl) else "activation")
                if not is_const(tbl) and tbl in producer and codes[ops[producer[tbl]].OpcodeIndex()] == "DEQUANTIZE":
                    src_t = int(ops[producer[tbl]].InputsAsNumpy()[0])
                    via = "DEQUANTIZE(" + ttype(src_t) + ")"
                emb_rows.append({"subgraph": gi, "table_buffer": tensors[src_t].Buffer(),
                                 "op_index": oi, "table_dtype": ttype(tbl), "table_shape": shape(tbl),
                                 "table_constant": is_const(tbl), "table_quant": quant(tbl) if is_const(tbl) else None,
                                 "table_bytes": nbytes(tbl) if is_const(tbl) else None,
                                 "table_source_dtype": ttype(src_t), "table_via": via,
                                 "table_source_bytes": nbytes(src_t) if is_const(src_t) else None,
                                 "ids_dtype": ttype(ins[0]), "ids_shape": shape(ins[0]),
                                 "out_dtype": ttype(outs[0]), "out_shape": shape(outs[0])})
    sigs = []
    for si in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(si)
        g = model.Subgraphs(sd.SubgraphIndex())
        entry = {"key": sd.SignatureKey().decode(), "subgraph": sd.SubgraphIndex()}
        for side, n, get in (("inputs", sd.InputsLength(), sd.Inputs), ("outputs", sd.OutputsLength(), sd.Outputs)):
            entry[side] = []
            for k in range(n):
                tm = get(k)
                t = g.Tensors(tm.TensorIndex())
                entry[side].append({"name": tm.Name().decode(), "tensor_index": tm.TensorIndex(),
                                    "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [],
                                    "dtype": type_names.get(t.Type(), str(t.Type()))})
        sigs.append(entry)
    n_sub = model.SubgraphsLength()
    mm.close()
    f.close()
    bmm = [{"inputs": json.loads(k)[:-2], "kinds": json.loads(k)[-2], "output": json.loads(k)[-1], "count": v}
           for k, v in sorted(bmm_groups.items(), key=lambda kv: -kv[1])]
    return {
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path) if with_sha else None,
        "subgraphs": n_sub, "signatures": sigs, "operator_count": sum(hist.values()),
        "op_histogram": dict(sorted(hist.items())),
        "tensor_rank_histogram": {str(k): v for k, v in sorted(ranks.items())},
        "max_tensor_rank": max(ranks) if ranks else None,
        "tensor_dtype_histogram": dict(sorted(dtypes.items())),
        "constant_bytes_by_dtype": dict(sorted(const_bytes.items())),
        "constant_count_by_dtype": dict(sorted(const_count.items())),
        "constant_unique_buffer_bytes_by_dtype": dict(sorted(uniq_bytes.items())),
        "constant_unique_buffer_count_by_dtype": dict(sorted(uniq_count.items())),
        "int64_tensor_count": len(int64_t), "int64_tensors": int64_t[:50],
        "rank_gt4_tensor_count": len(rank_gt4), "rank_gt4_tensors": rank_gt4[:50],
        "gpu_risk_counts": {k: len(risk.get(k, [])) for k in GPU_RISK_OPS},
        "gpu_risk_sites": {k: v[:40] for k, v in risk.items() if k not in ("EMBEDDING_LOOKUP",)},
        "custom_ops": dict(customs), "custom_op_count": sum(customs.values()),
        "pad_count": len(pads), "pads": pads,
        "batch_matmul_count": hist.get("BATCH_MATMUL", 0), "batch_matmul_shape_groups": bmm,
        "batch_matmul_ranks": sorted({len(s) for g in bmm for s in g["inputs"]}),
        "fully_connected_count": len(fc_rows), "fc_weight_source_dtype": dict(fc_weight_src),
        "fc_weight_tensor_dtype_direct": dict(fc_weight_direct), "fc_input_rank": dict(fc_in_rank),
        "fc_bias_dtype": dict(fc_bias),
        "fc_weight_shapes": dict(collections.Counter(json.dumps(r["weight_shape"]) for r in fc_rows)),
        "fc_rows_first3": fc_rows[:3],
        "dequantize_count": hist.get("DEQUANTIZE", 0), "dequantize_in_out": dict(deq_in_out),
        "embedding_lookup": emb_rows,
    }


# ---------------------------------------------------------------- CompiledModel

def open_compiled(path, backend, precision="fp32", threads=8, share=False):
    """-> (CompiledModel, options description). backend cpu = XNNPACK (num_threads); gpu = Metal with
    GpuOptions(enforce_f32 = precision == 'fp32'); precision 'default' = the GPU's default (fp16 activations).
    share (GPU, round 4) = GpuOptions(constant_tensor_sharing=True): one weight copy for every signature of the file."""
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options

    if backend == "gpu":
        gopt = (GpuOptions(enforce_f32=(precision == "fp32"), constant_tensor_sharing=True) if share
                else GpuOptions(enforce_f32=(precision == "fp32")))
        opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gopt)
        desc = {"accelerator": "GPU (Metal)", "precision": precision, "gpu_options": gopt._as_flat_kwargs()}
    else:
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        desc = {"accelerator": "CPU (XNNPACK)", "threads": threads}
    return CompiledModel.from_file(str(path), options=opts), desc


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
        "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


def memory(pid=None):
    """Bytes: ru_maxrss (this process), proc_pid_rusage(RUSAGE_INFO_V4) phys_footprint / lifetime max / resident size
    (pid, default this process; phys_footprint = `vmmap -summary`'s Physical footprint, Metal buffers included on
    Apple silicon) and `ps -o rss` (KiB -> bytes)."""
    import resource
    import subprocess

    pid = os.getpid() if pid is None else pid
    doc = {"pid": pid, "ru_maxrss_self": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = _RUsageInfoV4()
    rc = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 4, ctypes.byref(info))
    if rc == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint),
                   resident_size=int(info.ri_resident_size))
    else:
        doc["proc_pid_rusage_rc"] = rc
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        doc["ps_rss_bytes"] = int(out) * 1024 if out else None
    except Exception as e:  # informational
        doc["ps_rss_error"] = repr(e)
    return doc


class Runner:
    """Buffers by signature name; __call__(inputs dict of numpy) -> scores float32 [L]."""

    def __init__(self, model, signature):
        self.model, self.sig = model, signature
        self.in_det = model.get_input_tensor_details(signature)
        self.out_det = model.get_output_tensor_details(signature)
        assert sorted(self.in_det) == sorted(INPUT_NAMES), self.in_det
        assert list(self.out_det) == ["scores"], self.out_det
        self.L = int(list(self.out_det["scores"]["shape"])[1])
        self.ins = {n: model.create_input_buffer_by_name(signature, n) for n in self.in_det}
        self.outs = {"scores": model.create_output_buffer_by_name(signature, "scores")}

    def __call__(self, x):
        for n in INPUT_NAMES:
            v = x[n]
            want = np.int32 if n == "ids" else np.float32
            assert v.dtype == want, (n, v.dtype)
            self.ins[n].write(np.ascontiguousarray(v))
        self.model.run_by_name(self.sig, self.ins, self.outs)
        return np.asarray(self.outs["scores"].read(self.L, np.float32), dtype=np.float32).reshape(self.L).copy()

    def timed(self, x):
        """Round 4: the same call as __call__, timed -> (scores, start wall clock ms since the epoch, ms of write +
        run + read back, ms of run only). Inputs must already be contiguous with the right dtypes."""
        import time

        wall = time.time() * 1000.0
        t = time.perf_counter()
        for n in INPUT_NAMES:
            self.ins[n].write(x[n])
        tr = time.perf_counter()
        self.model.run_by_name(self.sig, self.ins, self.outs)
        run_ms = (time.perf_counter() - tr) * 1000.0
        s = self.outs["scores"].read(self.L, np.float32)
        total_ms = (time.perf_counter() - t) * 1000.0
        return np.asarray(s, dtype=np.float32).reshape(self.L), wall, total_ms, run_ms

    def close(self):
        for b in list(self.ins.values()) + list(self.outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
