"""Round 5: a stand-in for the d1-omni gate app's process (android/d1omni_gate GateActivity), on the Mac = d1_3b_work's
scripts/d1_fake_runner.py (round 4) with the d1-omni graph contract (six inputs -> scores [1, L]).

    # started by scripts/fake_adb.py for `am start -n com.mlboydaisuke.d1omni.gate/.GateActivity --es ... --ei ...`
    # standalone = the Mac CPU same-file baseline of a device leg (same graph file, same rows file):
    cd d1_omni_work; ~/venvs/lt094dev/bin/python scripts/fake_runner.py --files-dir <dir with the graph, rows, prefix files> \
        --graph <file> --rows <file> --report <tag>.json --mode gate --accel cpu --threads 8

Same extras as the app (graph, rows, report, sig, mode, accel, precision, L, limit, threads, warmup, rest_ms, reps,
cool_ms, keep_scores, cpu_cache, gpu_src_quant), same files in <files dir>: <report> (written as <report>.partial, then
renamed), sel_<stem>.f32 (little-endian float32, the K scores at P + markers[k] per row, rows in file order),
full_<stem>.f32 with keep_scores=1 (the whole [L] scores per row); files/STOP ends a gate after the current row and a
timing set after the current round. A row is laid out as the app's RowCodec does ([prefix rows | ids | pad], media / pad
/ keep_right from P and n, qtype one-hot; the prefix from <prefix_file>, little-endian float32 [P, 1024]); every row is
also checked bit-equal against the host's build_inputs (host/d1_host.py) and the count lands in the report
(`layout_equals_host_build_inputs`). Timing mode: per set whose L equals the graph's L, `warmup` calls then `reps`
rounds, each call's [wall clock ms at its start, ms write + run + read, ms run only]. L and D are read from the graph's
signature (the extra `sig`, else decide_<rows L>, else the first decide_<bucket> of the file), as the app does; the
same checks fail the same way (status FAILED + error).
The graph runs on the Mac's CompiledModel CPU (ai-edge-litert 2.2.0, `threads` threads) whatever accel / precision the
extras ask for: the report keeps the request and says what ran in `stand_in`. Memory fields hold the Mac process's
ru_maxrss (not a phone's VmHWM). Under the fake device (--fake-root) it also writes /proc/<pid>/status for the chain's
2 s sampler, appends app-style log lines (tag D1OMNI_GATE, and a `Replacing N out of M node(s)` line in the runtime's
words, N = M = the graph's operator count, delegate FAKE_STAND_IN) to the fake logcat and, like the app, stays alive
after the report until `am force-stop`; with --lowmem it plays an app that never gets past the compile while the
phone's MemAvailable is low (the chain's memory guard must stop it).
Round 9: mode=generic / generic_timing = the app's generic mode (GenericCodec) on the same Mac CPU: a generic rows file
(kind generic: signature, inputs [{name, dtype, shape, file}], outputs, rows [{key, index}], sets), every declared
shape / dtype checked against the graph's signature and the declared names against all of its inputs / outputs; one
call per row with slice `index` of every stacked input file; every output's whole tensor appended to
out_<stem>.f32 (little-endian float32, rows in file order, outputs in declared order); generic_timing times the sets.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import d1_host as H  # noqa: E402  (numpy only at import: build_inputs is the reference layout)

OUTPUT = "scores"
INPUTS = ("ids", "prefix", "media", "pad", "keep_right", "qtype_onehot")
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
WARMUP, REPS = 5, 20
LITERT_VERSION = "2.2.0"


class Fake:
    """The fake device's side channels (none when standalone)."""

    def __init__(self, root: str | None):
        self.root = Path(root) if root else None
        self.pid = os.getpid()

    def log(self, text: str, tag: str = "D1OMNI_GATE") -> None:
        if not self.root:
            return
        d = self.root / "_logcat"
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"
        with open(d / f"{self.pid}.log", "a") as f:
            f.write(f"{stamp} {self.pid:5d} {self.pid:5d} I {tag}: {text} (fake)\n")

    def status_loop(self, stop: threading.Event) -> None:
        if not self.root:
            return
        p = self.root / "proc" / str(self.pid)
        p.mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024   # macOS: bytes
            (p / "status").write_text(f"Name:\tcom.mlboydaisuke.d1omni.gate\nPid:\t{self.pid}\nVmHWM:\t{kb:8d} kB\n"
                                      f"VmRSS:\t{kb:8d} kB\nVmSwap:\t       0 kB\n")
            stop.wait(1.0)

    def gpu_state(self):
        if not self.root:
            return None
        try:
            g = self.root / "sys/class/kgsl/kgsl-3d0"
            return {"max_clock_mhz": int((g / "max_clock_mhz").read_text().strip()), "temp_mc": int((g / "temp").read_text().strip())}
        except (OSError, ValueError):
            return None


def memory_now() -> dict:
    return {"stand_in": "mac", "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def median(xs):
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def stats(xs):
    return {"median": median(xs), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)}


def op_count(path: Path):
    """The graph's operator count (flatbuffer, subgraph 0) for the fake Replacing line; None when unreadable."""
    import mmap

    try:
        from ai_edge_litert import schema_py_generated as sp

        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            return int(sp.Model.GetRootAsModel(m, 0).Subgraphs(0).OperatorsLength())
    except Exception:
        return None


class Graph:
    """The app's view of one decision graph: CompiledModel + its signature's L and D (CPU, `threads` threads)."""

    def __init__(self, path: Path, threads: int, sig: str | None, rows_L: int | None, want_L: int):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options

        options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        self.model = CompiledModel.from_file(str(path), options=options)
        sigs = list(self.model.get_signature_list())
        if sig:
            cands = [sig]
        else:
            cands = ([f"decide_{rows_L}"] if rows_L else []) + ([f"decide_{want_L}"] if want_L else []) + [f"decide_{b}" for b in BUCKETS]
        tried = []
        self.sig = None
        for c in dict.fromkeys(cands):
            tried.append(c)
            if c in sigs:
                self.sig = c
                break
        if self.sig is None:
            raise ValueError(f"no usable signature among {tried} (the file has {sigs})")
        self.tried = tried
        sl = self.model.get_signature_list()[self.sig]
        if sorted(sl.get("inputs", [])) != sorted(INPUTS) or list(sl.get("outputs", [])) != [OUTPUT]:
            raise ValueError(f"unexpected signature {sl}")
        ins = self.model.get_input_tensor_details(self.sig)
        out = self.model.get_output_tensor_details(self.sig)[OUTPUT]
        dims = {n: [int(x) for x in ins[n]["shape"]] for n in INPUTS}
        dims[OUTPUT] = [int(x) for x in out["shape"]]
        ids = dims["ids"]
        if len(ids) != 2 or ids[0] != 1 or ids[1] <= 0:
            raise ValueError(f"an input of shape [1, L] expected, got ids {ids}")
        self.L = ids[1]
        if self.sig != f"decide_{self.L}":
            raise ValueError(f"signature {self.sig} but ids have L={self.L}")
        for n in ("media", "pad", "keep_right", OUTPUT):
            if dims[n] != [1, self.L]:
                raise ValueError(f"{n} of shape [1, {self.L}] expected, got {dims[n]}")
        if dims["qtype_onehot"] != [1, 3]:
            raise ValueError(f"qtype_onehot of shape [1, 3] expected, got {dims['qtype_onehot']}")
        p = dims["prefix"]
        if len(p) != 3 or p[0] != 1 or p[1] != self.L or p[2] <= 0:
            raise ValueError(f"prefix of shape [1, {self.L}, D] expected, got {p}")
        self.d = p[2]
        self.dims = dims
        self.inputs = {n: self.model.create_input_buffer_by_name(self.sig, n) for n in INPUTS}
        self.outputs = {OUTPUT: self.model.create_output_buffer_by_name(self.sig, OUTPUT)}

    def call(self, x: dict):
        t0 = time.perf_counter_ns()
        for n in INPUTS:
            self.inputs[n].write(x[n])
        t1 = time.perf_counter_ns()
        self.model.run_by_name(self.sig, self.inputs, self.outputs)
        t2 = time.perf_counter_ns()
        scores = np.asarray(self.outputs[OUTPUT].read(self.L, np.float32), dtype=np.float32).reshape(self.L).copy()
        t3 = time.perf_counter_ns()
        ms = lambda a, b: (b - a) / 1e6
        return scores, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)

    def close(self):
        for b in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        try:
            self.model.close()
        except Exception:
            pass


_PREFIX = {}


def prefix_rows(files: Path, name: str, P: int, d: int) -> np.ndarray:
    k = (name, P, d)
    if k not in _PREFIX:
        raw = (files / name).read_bytes()
        if len(raw) != P * d * 4:
            raise ValueError(f"prefix file has {len(raw)} bytes, P x D x 4 = {P * d * 4}")
        _PREFIX[k] = np.frombuffer(raw, dtype="<f4").reshape(P, d).astype(np.float32)
    return _PREFIX[k]


class Row:
    """One row laid out as the app's RowCodec (and checked against the host's build_inputs)."""

    def __init__(self, row: dict, g: Graph, pad_id: int, files: Path):
        self.key = row["key"]
        ids = [int(x) for x in row["ids"]]
        self.n, self.P, self.K = len(ids), int(row.get("P", 0)), int(row["K"])
        self.markers = [int(m) for m in row["markers"]]
        L, d = g.L, g.d
        if self.n < 1 or self.P < 0 or self.P + self.n > L:
            raise ValueError(f"row of {self.P} + {self.n} positions does not fit L={L}")
        if self.K < 1 or len(self.markers) < self.K or not all(0 <= m < self.n for m in self.markers[: self.K]):
            raise ValueError(f"{self.key}: markers outside the row")
        qt = int(row["qtype"])
        if qt not in (0, 1, 2):
            raise ValueError(f"qtype {qt} is not 0 (choice), 1 (score) or 2 (noul)")
        x_ids = np.full((1, L), pad_id, np.int32)
        x_ids[0, self.P: self.P + self.n] = ids
        pre = np.zeros((1, L, d), np.float32)
        rows_p = None
        if self.P:
            name = row.get("prefix_file") or ""
            if not name:
                raise ValueError(f"{self.key}: P={self.P} but no prefix_file")
            rows_p = prefix_rows(files, name, self.P, d)
            pre[0, : self.P] = rows_p
        t = np.arange(L)
        oh = np.zeros((1, 3), np.float32)
        oh[0, qt] = 1.0
        self.x = {"ids": x_ids, "prefix": pre, "media": (t < self.P).astype(np.float32)[None],
                  "pad": (t < self.P + self.n).astype(np.float32)[None],
                  "keep_right": np.where((t == self.P - 1) & (self.P > 0), 0.0, 1.0).astype(np.float32)[None],
                  "qtype_onehot": oh}
        # the app's layout must equal the host's build_inputs (+ the one-hot), bit for bit
        if pad_id == 0:
            ref = H.build_inputs(ids, rows_p, L)
            self.host_equal = all(np.array_equal(ref[k], self.x[k]) and ref[k].dtype == self.x[k].dtype for k in ref)
        else:
            self.host_equal = None

    def select(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray([scores[self.P + m] for m in self.markers[: self.K]], dtype=np.float32)


def gate(g: Graph, doc, a, files: Path, out: dict, fake: Fake):
    if int(doc["L"]) != g.L:
        raise ValueError(f"rows file is for L={doc['L']}, the graph's L={g.L}")
    pad = int(doc["pad_id"])
    rows = doc["rows"]
    count = min(a.limit, len(rows)) if a.limit > 0 else len(rows)
    stem = a.report[:-5] if a.report.endswith(".json") else a.report
    sel_path, full_path = files / f"sel_{stem}.f32", files / f"full_{stem}.f32"
    records, totals, runs = [], [], []
    sel_bytes = full_bytes = nonfinite_rows = 0
    host_equal = host_checked = 0
    full = open(full_path, "wb") if a.keep_scores else None
    try:
        with open(sel_path, "wb") as sel:
            for i in range(count):
                if (files / "STOP").exists():
                    out["stopped_early"] = True
                    break
                r = Row(rows[i], g, pad, files)
                if r.host_equal is not None:
                    host_checked += 1
                    host_equal += int(r.host_equal)
                wall = int(time.time() * 1000)
                scores, w_ms, r_ms, rd_ms, total = g.call(r.x)
                s = r.select(scores).astype("<f4")
                b = s.tobytes()
                sel.write(b)
                sel_bytes += len(b)
                if full is not None:
                    fb = scores.astype("<f4").tobytes()
                    full.write(fb)
                    full_bytes += len(fb)
                finite = bool(np.isfinite(s).all())
                real = r.P + r.n
                nf_real = int((~np.isfinite(scores[:real])).sum())
                nf_all = nf_real + int((~np.isfinite(scores[real:])).sum())
                nonfinite_rows += 0 if finite else 1
                totals.append(total)
                runs.append(r_ms)
                records.append({"key": r.key, "n": r.n, "P": r.P, "K": r.K, "finite": finite, "nonfinite_real": nf_real,
                                "nonfinite_all": nf_all, "t_start_ms": wall, "write_ms": w_ms, "run_ms": r_ms,
                                "read_ms": rd_ms, "write_run_read_ms": total})
                if i % 25 == 0 or i == count - 1:
                    fake.log(f"row {i + 1} / {count}: {total:.1f} ms")
    finally:
        if full is not None:
            full.close()
    if not totals:
        raise RuntimeError("stopped before the first row")
    warm_t = totals[WARMUP:] if len(totals) > WARMUP else totals
    warm_r = runs[WARMUP:] if len(runs) > WARMUP else runs
    out["sel_file"], out["sel_bytes"] = sel_path.name, sel_bytes
    if a.keep_scores:
        out["full_file"], out["full_bytes"] = full_path.name, full_bytes
    out["layout_equals_host_build_inputs"] = {"rows_checked": host_checked, "rows_equal": host_equal}
    out["summary"] = {"count": len(totals), "finite_rows": len(totals) - nonfinite_rows, "nonfinite_rows": nonfinite_rows,
                      "first_call_write_run_read_ms": totals[0], "first_call_run_ms": runs[0], "warm_rows": len(warm_t),
                      "warm_median_write_run_read_ms": median(warm_t), "warm_median_run_ms": median(warm_r),
                      "warm_min_write_run_read_ms": min(warm_t), "warm_max_write_run_read_ms": max(warm_t)}
    out["rows"] = records


def cool_down(a, base, fake: Fake) -> dict:
    rec = {"cool_ms": a.cool_ms}
    if a.cool_ms <= 0:
        return {**rec, "mode": "none"}
    t0 = time.time()
    if base is None or fake.gpu_state() is None:
        time.sleep(a.cool_ms / 1000)
        return {**rec, "mode": "fixed", "waited_ms": int((time.time() - t0) * 1000)}
    now = fake.gpu_state()
    while now and (now["max_clock_mhz"] < base["max_clock_mhz"] or now["temp_mc"] > base["temp_mc"] + 5000) \
            and (time.time() - t0) * 1000 < a.cool_ms:
        time.sleep(0.5)
        now = fake.gpu_state()
    ok = bool(now and now["max_clock_mhz"] >= base["max_clock_mhz"] and now["temp_mc"] <= base["temp_mc"] + 5000)
    return {**rec, "mode": "kgsl", "waited_ms": int((time.time() - t0) * 1000), "base": {"readable": True, **base},
            "end": {"readable": True, **now} if now else {"readable": False}, "recovered": ok}


def timing(g: Graph, doc, a, files: Path, out: dict, fake: Fake, base):
    pad = int(doc["pad_id"])
    results = {}
    for s in doc["sets"]:
        if int(s["L"]) != g.L:
            continue
        if (files / "STOP").exists():
            out["stopped_early"] = True
            break
        prepared = [Row(r, g, pad, files) for r in s["rows"]]
        fake.log(f"timing {s['name']}: {len(prepared)} row(s)")
        cool = cool_down(a, base, fake)
        warm, warm_calls, timed = [], [], []
        for w in range(a.warmup):
            r = prepared[w % len(prepared)]
            t = int(time.time() * 1000)
            _, _, r_ms, _, total = g.call(r.x)
            warm.append(total)
            warm_calls.append([t, total, r_ms])
        req_t, req_r, per_t, per_r, finite, stopped = [], [], [], [], True, False
        for _ in range(a.reps):
            if (files / "STOP").exists():
                stopped = True
                break
            tt = rr = 0.0
            for r in prepared:
                t = int(time.time() * 1000)
                scores, _, r_ms, _, total = g.call(r.x)
                timed.append([t, total, r_ms])
                tt += total
                rr += r_ms
                per_t.append(total)
                per_r.append(r_ms)
                finite = finite and bool(np.isfinite(r.select(scores)).all())
            req_t.append(tt)
            req_r.append(rr)
            if a.rest_ms > 0:
                time.sleep(a.rest_ms / 1000)
        if stopped:
            out["stopped_early"] = True
        if not per_t:
            break
        res = {"kind": s["kind"], "rows": len(prepared), "keys": [r.key for r in prepared], "tokens": [r.n for r in prepared],
               "prefix_rows": [r.P for r in prepared], "warmup_ms_write_run_read": warm,
               "per_call_ms_write_run_read": stats(per_t), "per_call_ms_run_only": stats(per_r), "finite_markers": finite,
               "warmup_calls": warm_calls, "cool": cool, "timed_calls": timed,
               "calls_format": "[device wall clock ms at the call's start, ms write + run + read, ms run only]; a request set's calls in row order"}
        if len(prepared) > 1:
            res["request_ms_write_run_read"], res["request_ms_run_only"] = stats(req_t), stats(req_r)
        if stopped:
            res["stopped_early"] = True
        results[s["name"]] = res
        if stopped:
            break
    if not results:
        raise RuntimeError(f"no timing set for L={g.L}")
    out["timing"] = results


# ---------------------------------------------------------------- generic mode (round 9)

GENERIC_MODES = ("generic", "generic_timing")
DTYPES = {"float32": "<f4", "int32": "<i4"}


def plain_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f'"{name}" is not a plain file name')
    return name


class GenericGraph:
    """The app's generic view of one single-signature graph (GateActivity.genericIO + GenericCodec), Mac CPU."""

    def __init__(self, path: Path, threads: int, sig: str | None, doc: dict, files: Path):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options

        if doc.get("kind") != "generic":
            raise ValueError("mode generic needs a generic rows file (kind generic)")
        options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        self.model = CompiledModel.from_file(str(path), options=options)
        self.sig = sig or doc["signature"]
        sl = self.model.get_signature_list()
        if self.sig not in sl:
            raise ValueError(f"no signature {self.sig} (the file has {list(sl)})")
        det = {"inputs": self.model.get_input_tensor_details(self.sig), "outputs": self.model.get_output_tensor_details(self.sig)}

        def specs(kind):
            out = []
            for d in doc[kind]:
                name, dtype, shape = d["name"], d["dtype"], [int(v) for v in d["shape"]]
                if dtype not in DTYPES:
                    raise ValueError(f"dtype {dtype} is not float32 or int32")
                if not shape or any(v < 1 for v in shape):
                    raise ValueError(f"shape {shape} has a dimension < 1")
                if name not in det[kind]:
                    raise ValueError(f"{name} is not an {kind[:-1]} of {self.sig}")
                g = det[kind][name]
                if [int(v) for v in g["shape"]] != shape:
                    raise ValueError(f"{name}: the rows file says {shape}, the graph has {g['shape']}")
                if g["dtype"] != dtype:
                    raise ValueError(f"{name} is {g['dtype']} in the graph, the rows file says {dtype}")
                s = {"name": name, "dtype": dtype, "shape": shape, "count": int(np.prod(shape))}
                if kind == "inputs":
                    f = files / plain_name(d["file"])
                    if not f.is_file():
                        raise FileNotFoundError(f"missing {f.name}")
                    size, slice_bytes = f.stat().st_size, s["count"] * 4
                    if size < slice_bytes or size % slice_bytes:
                        raise ValueError(f"a file of {size} bytes is not a whole number of slices of {s['count']} values")
                    s.update(file=f, slices=size // slice_bytes)
                elif dtype != "float32":
                    raise ValueError(f"output {name}: only float32 outputs are read")
                out.append(s)
            names = [s["name"] for s in out]
            if not names or len(set(names)) != len(names) or sorted(names) != sorted(det[kind]):
                raise ValueError(f"{kind} declared {names}, the signature has {sorted(det[kind])}")
            return out

        self.inputs, self.outputs = specs("inputs"), specs("outputs")
        self.in_buf = {s["name"]: self.model.create_input_buffer_by_name(self.sig, s["name"]) for s in self.inputs}
        self.out_buf = {s["name"]: self.model.create_output_buffer_by_name(self.sig, s["name"]) for s in self.outputs}

    def io_json(self):
        return {"inputs": [{"name": s["name"], "dtype": s["dtype"], "shape": s["shape"], "file": s["file"].name,
                            "file_bytes": s["file"].stat().st_size, "slices": s["slices"]} for s in self.inputs],
                "outputs": [{"name": s["name"], "dtype": s["dtype"], "shape": s["shape"]} for s in self.outputs]}

    def slice(self, s, index: int) -> np.ndarray:
        if not 0 <= index < s["slices"]:
            raise ValueError(f"row index {index} outside the file's {s['slices']} slices")
        a = np.fromfile(s["file"], dtype=DTYPES[s["dtype"]], count=s["count"], offset=index * s["count"] * 4)
        return a.astype(np.float32 if s["dtype"] == "float32" else np.int32).reshape(s["shape"])

    def prepare(self, row: dict, position: int):
        index = int(row.get("index", position))
        return row["key"], index, [self.slice(s, index) for s in self.inputs]

    def call(self, values):
        t0 = time.perf_counter_ns()
        for s, v in zip(self.inputs, values):
            self.in_buf[s["name"]].write(v)
        t1 = time.perf_counter_ns()
        self.model.run_by_name(self.sig, self.in_buf, self.out_buf)
        t2 = time.perf_counter_ns()
        outs = [np.asarray(self.out_buf[s["name"]].read(s["count"], np.float32), np.float32).reshape(-1).copy()
                for s in self.outputs]
        t3 = time.perf_counter_ns()
        ms = lambda a, b: (b - a) / 1e6
        return outs, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)

    def close(self):
        for b in list(self.in_buf.values()) + list(self.out_buf.values()):
            try:
                b.destroy()
            except Exception:
                pass
        try:
            self.model.close()
        except Exception:
            pass


def generic_gate(g: GenericGraph, doc, a, files: Path, out: dict, fake: Fake):
    rows = doc["rows"]
    count = min(a.limit, len(rows)) if a.limit > 0 else len(rows)
    stem = a.report[:-5] if a.report.endswith(".json") else a.report
    out_path = files / f"out_{stem}.f32"
    records, totals, runs, out_bytes, nonfinite_rows = [], [], [], 0, 0
    with open(out_path, "wb") as f:
        for i in range(count):
            if (files / "STOP").exists():
                out["stopped_early"] = True
                break
            key, index, values = g.prepare(rows[i], i)
            wall = int(time.time() * 1000)
            outs, w_ms, r_ms, rd_ms, total = g.call(values)
            b = b"".join(o.astype("<f4").tobytes() for o in outs)
            f.write(b)
            out_bytes += len(b)
            nf = {s["name"]: int((~np.isfinite(o)).sum()) for s, o in zip(g.outputs, outs)}
            nonfinite_rows += int(any(nf.values()))
            totals.append(total)
            runs.append(r_ms)
            records.append({"key": key, "index": index, "nonfinite": nf, "t_start_ms": wall, "write_ms": w_ms, "run_ms": r_ms,
                            "read_ms": rd_ms, "write_run_read_ms": total})
            if i % 5 == 0 or i == count - 1:
                fake.log(f"row {i + 1} / {count}: {total:.1f} ms")
    if not totals:
        raise RuntimeError("stopped before the first row")
    warm_t = totals[WARMUP:] if len(totals) > WARMUP else totals
    warm_r = runs[WARMUP:] if len(runs) > WARMUP else runs
    out["out_file"], out["out_bytes"] = out_path.name, out_bytes
    out["out_bytes_per_row"] = sum(s["count"] * 4 for s in g.outputs)
    out["summary"] = {"count": len(totals), "finite_rows": len(totals) - nonfinite_rows, "nonfinite_rows": nonfinite_rows,
                      "first_call_write_run_read_ms": totals[0], "first_call_run_ms": runs[0], "warm_rows": len(warm_t),
                      "warm_median_write_run_read_ms": median(warm_t), "warm_median_run_ms": median(warm_r),
                      "warm_min_write_run_read_ms": min(warm_t), "warm_max_write_run_read_ms": max(warm_t)}
    out["rows"] = records


def generic_timing(g: GenericGraph, doc, a, files: Path, out: dict, fake: Fake, base):
    rows = doc["rows"]
    position = {r["key"]: i for i, r in enumerate(rows)}
    results = {}
    for s in doc["sets"]:
        if (files / "STOP").exists():
            out["stopped_early"] = True
            break
        missing = [k for k in s["rows"] if k not in position]
        if missing:
            raise ValueError(f"set {s['name']} names row {missing[0]}, which the rows file does not have")
        prepared = [g.prepare(rows[position[k]], position[k]) for k in s["rows"]]
        if not prepared:
            raise ValueError(f"set {s['name']} has no row")
        fake.log(f"timing {s['name']}: {len(prepared)} row(s)")
        cool = cool_down(a, base, fake)
        warm, warm_calls, timed = [], [], []
        for w in range(a.warmup):
            t = int(time.time() * 1000)
            _, _, r_ms, _, total = g.call(prepared[w % len(prepared)][2])
            warm.append(total)
            warm_calls.append([t, total, r_ms])
        req_t, req_r, per_t, per_r, finite, stopped = [], [], [], [], True, False
        for _ in range(a.reps):
            if (files / "STOP").exists():
                stopped = True
                break
            tt = rr = 0.0
            for _, _, values in prepared:
                t = int(time.time() * 1000)
                outs, _, r_ms, _, total = g.call(values)
                timed.append([t, total, r_ms])
                tt += total
                rr += r_ms
                per_t.append(total)
                per_r.append(r_ms)
                finite = finite and all(bool(np.isfinite(o).all()) for o in outs)
            req_t.append(tt)
            req_r.append(rr)
            if a.rest_ms > 0:
                time.sleep(a.rest_ms / 1000)
        if stopped:
            out["stopped_early"] = True
        if not per_t:
            break
        res = {"kind": s.get("kind", "single"), "rows": len(prepared), "keys": [p[0] for p in prepared],
               "indices": [p[1] for p in prepared], "warmup_ms_write_run_read": warm,
               "per_call_ms_write_run_read": stats(per_t), "per_call_ms_run_only": stats(per_r), "finite_outputs": finite,
               "warmup_calls": warm_calls, "cool": cool, "timed_calls": timed,
               "calls_format": "[device wall clock ms at the call's start, ms write every input + run + read every output, "
                               "ms run only]; a request set's calls in row order"}
        if len(prepared) > 1:
            res["request_ms_write_run_read"], res["request_ms_run_only"] = stats(req_t), stats(req_r)
        if stopped:
            res["stopped_early"] = True
        results[s["name"]] = res
        if stopped:
            break
    if not results:
        raise RuntimeError("no timing set ran")
    out["timing"] = results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-dir", required=True)
    ap.add_argument("--fake-root", default=None)
    ap.add_argument("--graph", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--report", default="d1omni_gate_report.json")
    ap.add_argument("--sig", default="")
    ap.add_argument("--mode", default="gate")
    ap.add_argument("--accel", default="gpu")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--rest_ms", type=int, default=0)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--cool_ms", type=int, default=0)
    ap.add_argument("--keep_scores", type=int, default=0)
    ap.add_argument("--cpu_cache", type=int, default=0)
    ap.add_argument("--gpu_src_quant", type=int, default=-1)
    ap.add_argument("--lowmem", action="store_true", help="play an app stuck in the compile under low memory")
    a, unknown = ap.parse_known_args()   # the app ignores extras it does not know
    files = Path(a.files_dir)
    fake = Fake(a.fake_root)
    stop = threading.Event()
    threading.Thread(target=fake.status_loop, args=(stop,), daemon=True).start()
    out = {"graph": a.graph, "accel": a.accel, "precision": a.precision, "rows_file": a.rows, "limit": a.limit,
           "mode": a.mode, "litert": LITERT_VERSION, "device": "stand-in (Mac)", "android": None,
           "stand_in": f"Mac CompiledModel CPU, {a.threads} threads (requested: {a.accel} {a.precision})"}
    if a.accel == "cpu":
        out.update({"threads": a.threads, "cpu_weight_cache": bool(a.cpu_cache)})
    if a.accel == "gpu" and a.gpu_src_quant >= 0:
        out["gpu_allow_src_quantized_fc_conv_ops"] = bool(a.gpu_src_quant)
    if a.mode in ("timing", "generic_timing"):
        out.update({"warmup_calls_setting": a.warmup, "rest_ms": a.rest_ms, "reps": a.reps, "cool_ms": a.cool_ms})
    if a.mode == "gate":
        out["keep_scores"] = bool(a.keep_scores)
    if unknown:
        out["extras_ignored"] = unknown
    try:
        (files / "STOP").unlink()
    except FileNotFoundError:
        pass
    g = None
    try:
        if a.lowmem:             # fault injection: an app that never gets past its start; am force-stop ends it
            fake.log(f"compiling {a.graph} on {a.accel} {a.precision} (low memory: stuck)")
            while True:
                time.sleep(1)
        if not a.graph or not a.rows:
            raise ValueError("extras graph and rows are required")
        if a.mode not in ("gate", "timing") + GENERIC_MODES:
            raise ValueError(f"unknown mode {a.mode}")
        if a.accel not in ("gpu", "cpu"):
            raise ValueError(f"unknown accel {a.accel}")
        if a.accel == "gpu" and a.precision not in ("fp32", "fp16acc32", "fp16", "default"):
            raise ValueError(f"unknown precision {a.precision}")
        graph = files / a.graph
        if not graph.is_file():
            raise FileNotFoundError(f"missing {graph.name}")
        out["graph_bytes"] = graph.stat().st_size
        doc = json.loads((files / a.rows).read_text())
        base = fake.gpu_state() if a.cool_ms > 0 else None
        if a.cool_ms > 0:
            out["gpu_state_before_compile"] = {"readable": True, **base} if base else {"readable": False}
        out["memory_before_compile"] = memory_now()
        fake.log(f"compiling {a.graph} on {a.accel} {a.precision}")
        t0 = time.perf_counter_ns()
        if a.mode in GENERIC_MODES:
            g = GenericGraph(graph, a.threads, a.sig or None, doc, files)
            out["compile_ms"] = (time.perf_counter_ns() - t0) / 1e6
            out["memory_after_compile"] = memory_now()
            nops = op_count(graph)
            if nops:
                fake.log(f"Replacing {nops} out of {nops} node(s) with delegate (FAKE_STAND_IN) node, yielding 1 partitions", tag="tflite")
            fake.log(f"D1OMNI_GATE compiled {a.graph} {a.accel} {a.precision} in {out['compile_ms']} ms")
            out["signature"], out["signature_from"] = g.sig, "extra sig" if a.sig else "rows file"
            out["io"] = g.io_json()
            out["signature_input_count"], out["signature_output_count"] = len(g.inputs), len(g.outputs)
            if a.mode == "generic_timing":
                generic_timing(g, doc, a, files, out, fake, base)
            else:
                generic_gate(g, doc, a, files, out, fake)
        else:
            g = Graph(graph, a.threads, a.sig or None, int(doc["L"]) if "L" in doc else None, a.L)
            out["compile_ms"] = (time.perf_counter_ns() - t0) / 1e6
            out["memory_after_compile"] = memory_now()
            nops = op_count(graph)
            if nops:
                fake.log(f"Replacing {nops} out of {nops} node(s) with delegate (FAKE_STAND_IN) node, yielding 1 partitions", tag="tflite")
            fake.log(f"D1OMNI_GATE compiled {a.graph} {a.accel} {a.precision} in {out['compile_ms']} ms")
            out["signature"], out["signature_from"], out["signatures_tried"] = g.sig, "extra sig" if a.sig else "resolved", g.tried
            out["L"], out["hidden"], out["io_dims"] = g.L, g.d, g.dims
            if a.L not in (0, g.L):
                raise ValueError(f"extra L={a.L}, the graph's L={g.L}")
            if "hidden" in doc and int(doc["hidden"]) != g.d:
                raise ValueError(f"rows file says hidden {doc['hidden']}, the graph's D={g.d}")
            if a.mode == "timing":
                timing(g, doc, a, files, out, fake, base)
            else:
                gate(g, doc, a, files, out, fake)
        out["memory_at_end"] = memory_now()
        out["status"] = "DONE"
    except Exception as e:  # noqa: BLE001 - the app's catch-all, written into the report
        out["status"], out["error"] = "FAILED", f"{type(e).__name__}: {e}"
        fake.log(f"failed: {out['error']}")
    finally:
        if g is not None:
            g.close()
    tmp = files / (a.report + ".partial")
    tmp.write_text(json.dumps(out))
    tmp.rename(files / a.report)
    fake.log(f"D1OMNI_GATE report {a.report} {out['status']}")
    if fake.root:
        # The app never finishes its Activity: on the phone its process stays, in front, until `am force-stop`.
        while True:
            time.sleep(1)
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
