"""LiteRT backends of the host (ai-edge-litert CompiledModel): the tower per patch-count bucket, the decision graph per length
bucket. They pad and pick the smallest bucket that holds the request. Options: cpu = XNNPACK with N threads; gpu_f32 = the platform's GPU with
GpuOptions(enforce_f32=True); gpu_default = the delegate's default precision. A kept graph is an LRU slot: the sets run in bucket
order, so a small `keep` means few compiles."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from . import formulas as F


def options(acc: str, threads: int = 8, sharing: bool = False, weight_cache: str = ""):
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if acc == "cpu":
        cpu = CpuOptions(num_threads=threads, xnnpack_weight_cache_path=weight_cache) if weight_cache else CpuOptions(num_threads=threads)
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu)
    if acc not in ("gpu_f32", "gpu_default"):
        raise ValueError(acc)
    kw = {"enforce_f32": acc == "gpu_f32"}
    if sharing:
        kw["constant_tensor_sharing"] = True
    return Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(**kw))


class Graph:
    """One signature of one .tflite, buffers made once."""

    def __init__(self, path, signature: str | None = None, acc: str = "cpu", threads: int = 8, sharing: bool = False,
                 weight_cache: str = ""):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.acc = str(path), acc
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(self.path, options=options(acc, threads, sharing, weight_cache))
        self.load_s = time.perf_counter() - t0
        sigs = self.model.get_signature_list()
        self.signature = signature or next(iter(sigs))
        sig = sigs[self.signature]
        self.in_names, self.out_names = list(sig["inputs"]), list(sig["outputs"])
        self.in_shapes = {n: [int(x) for x in self.model.get_input_tensor_details(self.signature)[n]["shape"]] for n in self.in_names}
        self.out_shapes = {n: [int(x) for x in self.model.get_output_tensor_details(self.signature)[n]["shape"]] for n in self.out_names}
        t1 = time.perf_counter()
        self.inputs = {n: self.model.create_input_buffer_by_name(self.signature, n) for n in self.in_names}
        self.outputs = {n: self.model.create_output_buffer_by_name(self.signature, n) for n in self.out_names}
        self.buffer_s = time.perf_counter() - t1
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:
            self.fully_accelerated = None
        self.last_ms = self.write_ms = self.read_ms = None

    def write(self, **arrays) -> None:
        t = time.perf_counter()
        for n in self.in_names:
            a = np.ascontiguousarray(arrays[n], dtype=np.float32)
            assert list(a.shape) == self.in_shapes[n], (n, a.shape, self.in_shapes[n])
            self.inputs[n].write(a)
        self.write_ms = (time.perf_counter() - t) * 1000

    def run(self) -> None:
        t = time.perf_counter()
        self.model.run_by_name(self.signature, self.inputs, self.outputs)
        self.last_ms = (time.perf_counter() - t) * 1000

    def read(self) -> dict:
        t = time.perf_counter()
        out = {}
        for n in self.out_names:
            shape = self.out_shapes[n]
            out[n] = np.asarray(self.outputs[n].read(int(np.prod(shape)), np.float32), dtype=np.float32).reshape(shape)
        self.read_ms = (time.perf_counter() - t) * 1000
        return out

    def __call__(self, **arrays) -> dict:
        self.write(**arrays)
        self.run()
        return self.read()

    def close(self) -> None:
        for b in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class _Buckets:
    def __init__(self, files: dict, signature: str, acc: str, threads: int, keep: int, sharing: bool, weight_cache_dir: str):
        assert files, "no graph file"
        self.files = {int(k): Path(v) for k, v in sorted(files.items())}
        self.signature, self.acc, self.threads, self.keep, self.sharing = signature, acc, threads, keep, sharing
        self.weight_cache_dir = weight_cache_dir
        self.graphs: dict[int, Graph] = {}
        self.load_log: list = []
        self.last_bucket = None
        self.last = {}

    def graph(self, size: int) -> Graph:
        if size in self.graphs:
            self.graphs[size] = self.graphs.pop(size)         # most recent last
            return self.graphs[size]
        while len(self.graphs) >= self.keep:
            old = next(iter(self.graphs))
            self.graphs.pop(old).close()
        wc = ""
        if self.acc == "cpu" and self.weight_cache_dir:
            wc = str(Path(self.weight_cache_dir) / (self.files[size].stem + ".xnnpack_cache"))
        g = Graph(self.files[size], self.signature, self.acc, self.threads, self.sharing, wc)
        self.load_log.append({"size": size, "load_s": round(g.load_s, 2), "buffer_s": round(g.buffer_s, 2), "fully_accelerated": g.fully_accelerated})
        self.graphs[size] = g
        return g

    def bucket(self, n: int) -> int:
        for size in self.files:
            if n <= size:
                return size
        raise ValueError(f"{n} is above the top bucket {max(self.files)}")

    def close(self) -> None:
        for g in self.graphs.values():
            g.close()
        self.graphs = {}


class LiteRTTower(_Buckets):
    """files = {N: path}. __call__ as the host's tower interface."""

    def __init__(self, files: dict, acc: str = "cpu", threads: int = 8, keep: int = 8, sharing: bool = False, weight_cache_dir: str = ""):
        super().__init__(files, "tower", acc, threads, keep, sharing, weight_cache_dir)
        self.max_n = max(self.files)

    def __call__(self, patches, pos, cos, sin) -> np.ndarray:
        n = patches.shape[0]
        N = self.bucket(n)
        g = self.graph(N)
        feed = {"patches": np.zeros((1, N, patches.shape[1]), np.float32), "valid": np.zeros((1, N), np.float32),
                "pos": np.zeros((1, N, pos.shape[1]), np.float32), "cos": np.ones((1, N, cos.shape[1]), np.float32),
                "sin": np.zeros((1, N, sin.shape[1]), np.float32)}
        feed["patches"][0, :n] = patches
        feed["valid"][0, :n] = 1.0
        feed["pos"][0, :n] = pos
        feed["cos"][0, :n] = cos
        feed["sin"][0, :n] = sin
        rows = g(**feed)["rows"][0, :n // 4].copy()
        self.last_bucket = N
        self.last = {"run": g.last_ms, "write": g.write_ms, "read": g.read_ms}
        return rows


class LiteRTDecide(_Buckets):
    """files = {L: path}. __call__ as the host's decide interface."""

    def __init__(self, files: dict, acc: str = "cpu", threads: int = 8, keep: int = 1, sharing: bool = False, weight_cache_dir: str = ""):
        super().__init__(files, "decide", acc, threads, keep, sharing, weight_cache_dir)
        self.max_len = max(self.files)

    def __call__(self, embeds, cos, sin, planes) -> np.ndarray:
        T = embeds.shape[0]
        L = self.bucket(T)
        g = self.graph(L)
        e = np.zeros((1, L, embeds.shape[1]), np.float32)
        e[0, :T] = embeds
        v = np.zeros((1, L), np.float32)
        v[0, :T] = 1.0
        c, s = F.mrope_tables(F.padded_planes(planes, L))       # pads continue the planes
        c[0, :T], s[0, :T] = cos, sin
        hidden = g(embeds=e, valid=v, cos=c, sin=s)["hidden"][0, :T].copy()
        self.last_bucket = L
        self.last = {"run": g.last_ms, "write": g.write_ms, "read": g.read_ms}
        return hidden


class LiteRTMultiDecide:
    """One .tflite holding every length bucket as a signature `decide_L<L>[_qb<Q>]` on ONE CompiledModel: buffers per signature on first use and kept.
    __call__ as the host's decide interface."""

    def __init__(self, path, acc: str = "cpu", threads: int = 8, sharing: bool = False, weight_cache: str = ""):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.acc = str(path), acc
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(self.path, options=options(acc, threads, sharing, weight_cache))
        self.load_s = time.perf_counter() - t0
        self.load_log = [{"size": "multi", "load_s": round(self.load_s, 2)}]
        self.sigs = {}
        for name, sig in self.model.get_signature_list().items():
            m = re.match(r"^decide_L(\d+)", name)
            if m:
                self.sigs[int(m.group(1))] = name
        self.sigs = dict(sorted(self.sigs.items()))
        self.max_len = max(self.sigs)
        self.bufs = {}
        self.last_bucket = None
        self.last = {}

    def _buffers(self, name):
        if name not in self.bufs:
            sig = self.model.get_signature_list()[name]
            self.bufs[name] = ({n: self.model.create_input_buffer_by_name(name, n) for n in sig["inputs"]},
                               {n: self.model.create_output_buffer_by_name(name, n) for n in sig["outputs"]})
        return self.bufs[name]

    def __call__(self, embeds, cos, sin, planes) -> np.ndarray:
        T = embeds.shape[0]
        L = next(s for s in self.sigs if T <= s)
        name = self.sigs[L]
        ins, outs = self._buffers(name)
        t0 = time.perf_counter()
        e = np.zeros((1, L, embeds.shape[1]), np.float32)
        e[0, :T] = embeds
        v = np.zeros((1, L), np.float32)
        v[0, :T] = 1.0
        c, s = F.mrope_tables(F.padded_planes(planes, L))
        c[0, :T], s[0, :T] = cos, sin
        for n, a in (("embeds", e), ("valid", v), ("cos", c), ("sin", s)):
            ins[n].write(a)
        t1 = time.perf_counter()
        self.model.run_by_name(name, ins, outs)
        t2 = time.perf_counter()
        hidden = np.asarray(outs["hidden"].read(L * embeds.shape[1], np.float32), dtype=np.float32).reshape(L, embeds.shape[1])[:T].copy()
        t3 = time.perf_counter()
        self.last_bucket = L
        self.last = {"run": (t2 - t1) * 1000, "write": (t1 - t0) * 1000, "read": (t3 - t2) * 1000}
        return hidden

    def close(self) -> None:
        for ins, outs in self.bufs.values():
            for b in list(ins.values()) + list(outs.values()):
                try:
                    b.destroy()
                except Exception:
                    pass
        self.bufs = {}
        self.model.close()


class LiteRTMultiTower:
    """One .tflite holding every patch-count bucket as a signature `tower_N<N>[_qb<Q>]` on ONE CompiledModel: buffers per signature on first use and kept.
    __call__ as the host's tower interface; `files` = the buckets, as LiteRTTower's."""

    def __init__(self, path, acc: str = "cpu", threads: int = 8, sharing: bool = False, weight_cache: str = ""):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.acc = str(path), acc
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(self.path, options=options(acc, threads, sharing, weight_cache))
        self.load_s = time.perf_counter() - t0
        self.load_log = [{"size": "multi", "load_s": round(self.load_s, 2)}]
        self.sigs = {}
        for name in self.model.get_signature_list():
            m = re.match(r"^tower_N(\d+)", name)
            if m:
                self.sigs[int(m.group(1))] = name
        self.sigs = dict(sorted(self.sigs.items()))
        self.files = self.sigs
        self.max_n = max(self.sigs)
        self.bufs = {}
        self.last_bucket = None
        self.last = {}

    def _buffers(self, name):
        if name not in self.bufs:
            sig = self.model.get_signature_list()[name]
            self.bufs[name] = ({n: self.model.create_input_buffer_by_name(name, n) for n in sig["inputs"]},
                               {n: self.model.create_output_buffer_by_name(name, n) for n in sig["outputs"]})
        return self.bufs[name]

    def __call__(self, patches, pos, cos, sin) -> np.ndarray:
        n = patches.shape[0]
        N = next(s for s in self.sigs if n <= s)
        name = self.sigs[N]
        ins, outs = self._buffers(name)
        t0 = time.perf_counter()
        feed = {"patches": np.zeros((1, N, patches.shape[1]), np.float32), "valid": np.zeros((1, N), np.float32),
                "pos": np.zeros((1, N, pos.shape[1]), np.float32), "cos": np.ones((1, N, cos.shape[1]), np.float32),
                "sin": np.zeros((1, N, sin.shape[1]), np.float32)}
        feed["patches"][0, :n] = patches
        feed["valid"][0, :n] = 1.0
        feed["pos"][0, :n] = pos
        feed["cos"][0, :n] = cos
        feed["sin"][0, :n] = sin
        for k, v in feed.items():
            ins[k].write(v)
        t1 = time.perf_counter()
        self.model.run_by_name(name, ins, outs)
        t2 = time.perf_counter()
        m = N // 4
        rows = np.asarray(outs["rows"].read(m * 2048, np.float32), dtype=np.float32).reshape(m, 2048)[:n // 4].copy()
        t3 = time.perf_counter()
        self.last_bucket = N
        self.last = {"run": (t2 - t1) * 1000, "write": (t1 - t0) * 1000, "read": (t3 - t2) * 1000}
        return rows

    def close(self) -> None:
        for ins, outs in self.bufs.values():
            for b in list(ins.values()) + list(outs.values()):
                try:
                    b.destroy()
                except Exception:
                    pass
        self.bufs = {}
        self.model.close()


def discover(graph_dir, name: str, form: str, sizes=None) -> dict:
    """{size: path} of `<name>_{L|N}<size>[_qb<Q>]_<form>.tflite` under graph_dir."""
    pat = re.compile(rf"^{name}_[LN](\d+)(?:_qb\d+)?_{re.escape(form)}\.tflite$")
    out = {}
    for p in Path(graph_dir).iterdir():
        m = pat.match(p.name)
        if m and (sizes is None or int(m.group(1)) in sizes):
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))
