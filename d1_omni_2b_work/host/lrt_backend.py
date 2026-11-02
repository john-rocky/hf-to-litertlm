"""LiteRT backends for the d1-Omni-2B host (ai-edge-litert CompiledModel): the decision graph per bucket, the picture
tower per canvas and the audio encoder per length bucket (LiteRTAudio). CPU = XNNPACK with N threads; GPU = the platform's GPU accelerator (Metal on the Mac) with float32
(`GpuOptions(enforce_f32=True)`) or the delegate's default precision.

    dec = LiteRTDecide({L: path, ...}, accelerator="gpu", precision="fp32")
    hidden = dec(embeds [T, 2048], state_len, question)      # the host's DecideBackend interface
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d1_layout import from_canvas, make_sees, to_canvas  # noqa: E402  (numpy only: no torch at run time)


def compiled_options(accelerator: str, precision: str = "fp32", threads: int = 8, sharing: bool = False,
                     weight_cache: str = ""):
    """`sharing` = GpuOptions.constant_tensor_sharing (the GPU keeps one copy of a constant several signatures read);
    `weight_cache` = CpuOptions.xnnpack_weight_cache_path. Both off by default (= the options of the single-file checks)."""
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if accelerator == "gpu":
        if precision not in ("fp32", "default"):
            raise ValueError("precision is 'fp32' or 'default'")
        gpu = GpuOptions(enforce_f32=precision == "fp32", constant_tensor_sharing=True) if sharing else \
            GpuOptions(enforce_f32=precision == "fp32")
        return Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gpu)
    if accelerator == "cpu":
        cpu = CpuOptions(num_threads=threads, xnnpack_weight_cache_path=weight_cache) if weight_cache else \
            CpuOptions(num_threads=threads)
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu)
    raise ValueError(accelerator)


class MultiGraph:
    """Every signature of one .tflite on ONE CompiledModel (the seven-bucket decide file): buffers are made per
    signature on first use and kept."""

    def __init__(self, path: str, accelerator: str = "cpu", precision: str = "fp32", threads: int = 8,
                 sharing: bool = False, weight_cache: str = ""):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.accelerator, self.precision = path, accelerator, precision
        t0 = time.time()
        self.model = CompiledModel.from_file(path, options=compiled_options(accelerator, precision, threads, sharing,
                                                                            weight_cache))
        self.load_s = time.time() - t0
        self.signatures = self.model.get_signature_list()
        self.bufs = {}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:
            self.fully_accelerated = None

    def _buffers(self, sig):
        if sig not in self.bufs:
            s = self.signatures[sig]
            ins, outs = list(s["inputs"]), list(s["outputs"])
            shapes = {n: list(self.model.get_output_tensor_details(sig)[n]["shape"]) for n in outs}
            ishapes = {n: list(self.model.get_input_tensor_details(sig)[n]["shape"]) for n in ins}
            self.bufs[sig] = ({n: self.model.create_input_buffer_by_name(sig, n) for n in ins},
                              {n: self.model.create_output_buffer_by_name(sig, n) for n in outs}, ishapes, shapes)
        return self.bufs[sig]

    def __call__(self, sig: str, **arrays) -> dict:
        inputs, outputs, ishapes, shapes = self._buffers(sig)
        for n, buf in inputs.items():
            a = np.ascontiguousarray(arrays[n], dtype=np.float32)
            assert list(a.shape) == ishapes[n], (sig, n, a.shape, ishapes[n])
            buf.write(a)
        self.model.run_by_name(sig, inputs, outputs)
        return {n: np.asarray(outputs[n].read(int(np.prod(shapes[n])), np.float32), dtype=np.float32).reshape(shapes[n])
                for n in outputs}

    def close(self):
        for ins, outs, _, _ in self.bufs.values():
            for b in list(ins.values()) + list(outs.values()):
                try:
                    b.destroy()
                except Exception:
                    pass
        self.bufs = {}
        self.model.close()


class Graph:
    """One signature of one .tflite on the CompiledModel API, buffers made once."""

    def __init__(self, path: str, signature: str, accelerator: str = "cpu", precision: str = "fp32", threads: int = 8):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.signature, self.accelerator, self.precision = path, signature, accelerator, precision
        t0 = time.time()
        self.model = CompiledModel.from_file(path, options=compiled_options(accelerator, precision, threads))
        sig = self.model.get_signature_list()[signature]
        self.in_names, self.out_names = list(sig["inputs"]), list(sig["outputs"])
        self.in_shapes = {n: list(self.model.get_input_tensor_details(signature)[n]["shape"]) for n in self.in_names}
        self.out_shapes = {n: list(self.model.get_output_tensor_details(signature)[n]["shape"]) for n in self.out_names}
        self.inputs = {n: self.model.create_input_buffer_by_name(signature, n) for n in self.in_names}
        self.outputs = {n: self.model.create_output_buffer_by_name(signature, n) for n in self.out_names}
        self.load_s = time.time() - t0
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:
            self.fully_accelerated = None

    def __call__(self, **arrays) -> dict:
        for n in self.in_names:
            a = np.ascontiguousarray(arrays[n], dtype=np.float32)
            assert list(a.shape) == self.in_shapes[n], (n, a.shape, self.in_shapes[n])
            self.inputs[n].write(a)
        self.model.run_by_name(self.signature, self.inputs, self.outputs)
        out = {}
        for n in self.out_names:
            shape = self.out_shapes[n]
            size = int(np.prod(shape))
            out[n] = np.asarray(self.outputs[n].read(size, np.float32), dtype=np.float32).reshape(shape)
        return out

    def close(self):
        for b in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class LiteRTDecide:
    """The host's decide backend over bucket graphs {L: path}: pads one sequence to the smallest bucket that holds it.
    Graphs load on first use; at most `keep` stay loaded (least recently used closed first: each bucket file carries
    its own weights, ~6 GB once widened to float32 by Metal or XNNPACK); `max_len` = the largest bucket."""

    def __init__(self, paths: dict, accelerator: str = "cpu", precision: str = "fp32", threads: int = 8,
                 keep: int = 2):
        self.paths = {int(L): p for L, p in paths.items()}
        self.accelerator, self.precision, self.threads, self.keep = accelerator, precision, threads, max(1, int(keep))
        self.graphs = {}
        self.max_len = max(self.paths)
        self.last = {}
        self.loads = []

    def graph(self, L: int) -> Graph:
        if L in self.graphs:
            self.graphs[L] = self.graphs.pop(L)          # most recently used last
        else:
            while len(self.graphs) >= self.keep:
                old = next(iter(self.graphs))
                self.graphs.pop(old).close()
            g = Graph(self.paths[L], "decide", self.accelerator, self.precision, self.threads)
            self.graphs[L] = g
            self.loads.append({"L": L, "load_s": round(g.load_s, 2), "fully_accelerated": g.fully_accelerated})
        return self.graphs[L]

    def __call__(self, embeds: np.ndarray, state_len: int, question) -> np.ndarray:
        T = embeds.shape[0]
        fits = [L for L in sorted(self.paths) if L >= T]
        if not fits:
            raise ValueError(f"a sequence of {T} tokens is longer than the largest graph ({self.max_len})")
        L = fits[0]
        g = self.graph(L)
        e = np.zeros((1, L, embeds.shape[1]), np.float32)
        e[0, :T] = embeds
        q, k = make_sees(state_len, question, L)
        t0 = time.perf_counter()
        h = g(embeds=e, sees_q=q, sees_k=k)["hidden"]
        self.last = {"L": L, "T": T, "run_s": time.perf_counter() - t0}
        return h[0, :T]

    def close(self):
        for g in self.graphs.values():
            g.close()
        self.graphs.clear()


class LiteRTTower:
    """The host's tower backend over canvas graphs {(GH, GW): path}: each picture runs on the smallest-area canvas that
    holds its gh x gw patch grid (the picture at the top-left, the rest masked exactly); a picture no canvas holds is
    refused (its aspect ratio is never changed). Graphs load on first use and stay loaded (`keep`)."""

    def __init__(self, paths: dict, accelerator: str = "cpu", precision: str = "fp32", threads: int = 8,
                 keep: bool = True):
        self.paths = {tuple(int(x) for x in c): p for c, p in paths.items()}
        self.accelerator, self.precision, self.threads, self.keep = accelerator, precision, threads, keep
        self.graphs = {}
        self.last = {}

    def canvas(self, gh: int, gw: int) -> tuple[int, int]:
        fits = [c for c in self.paths if c[0] >= gh and c[1] >= gw]
        if not fits:
            raise ValueError(f"a picture of {gh} x {gw} patches fits no tower canvas ({sorted(self.paths)})")
        return min(fits, key=lambda c: (c[0] * c[1], c))

    def graph(self, canvas) -> Graph:
        if canvas not in self.graphs:
            if not self.keep:
                for g in self.graphs.values():
                    g.close()
                self.graphs.clear()
            self.graphs[canvas] = Graph(self.paths[canvas], "tower", self.accelerator, self.precision, self.threads)
        return self.graphs[canvas]

    def __call__(self, pixel_values: np.ndarray, grid) -> np.ndarray:
        gh, gw = int(grid[0]), int(grid[1])
        GH, GW = self.canvas(gh, gw)
        g = self.graph((GH, GW))
        patches, valid = to_canvas(pixel_values, gh, gw, GH, GW)
        t0 = time.perf_counter()
        out = g(patches=patches, valid=valid)["features"]
        self.last = {"canvas": (GH, GW), "grid": (gh, gw), "run_s": time.perf_counter() - t0}
        return from_canvas(out, gh, gw, GH, GW)

    def close(self):
        for g in self.graphs.values():
            g.close()
        self.graphs.clear()


class LiteRTAudio:
    """The host's audio backend over length-bucket graphs {T: path} (conversion/export_audio.py, signature `audio`):
    one clip's provider input_features [128, F] run on the smallest bucket with T >= F (mel [1, T, 128] zeros past F,
    valid [1, T, 1, 1] 1 on the clip's frames); the clip's tokens are the first ceil(F / 8) rows of `features`. A clip
    longer than the largest bucket is refused (`max_frames`: 3000 frames = 30 s). Graphs load on first use and stay
    loaded. numpy only (the host imports no torch)."""

    def __init__(self, paths: dict, accelerator: str = "cpu", precision: str = "fp32", threads: int = 8):
        self.paths = {int(T): p for T, p in paths.items()}
        if not self.paths:
            raise ValueError("no audio graph")
        self.accelerator, self.precision, self.threads = accelerator, precision, threads
        self.max_frames = max(self.paths)
        self.graphs = {}
        self.loads = []
        self.last = {}

    def graph(self, T: int) -> Graph:
        if T not in self.graphs:
            g = Graph(self.paths[T], "audio", self.accelerator, self.precision, self.threads)
            self.graphs[T] = g
            self.loads.append({"T": T, "load_s": round(g.load_s, 2), "fully_accelerated": g.fully_accelerated})
        return self.graphs[T]

    def __call__(self, input_features: np.ndarray) -> np.ndarray:
        x = np.asarray(input_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != 128:
            raise ValueError(f"input_features are [128, frames], not {x.shape}")
        F = x.shape[1]
        fits = [T for T in sorted(self.paths) if T >= F]
        if not fits:
            raise ValueError(f"a clip of {F} mel frames is longer than the largest audio graph ({self.max_frames} "
                             f"frames = {self.max_frames / 100:g} s)")
        T = fits[0]
        g = self.graph(T)
        mel = np.zeros((1, T, 128), np.float32)
        mel[0, :F] = x.T
        valid = np.zeros((1, T, 1, 1), np.float32)
        valid[0, :F] = 1.0
        n = -(F // -8)
        t0 = time.perf_counter()
        out = g(mel=mel, valid=valid)["features"]
        self.last = {"T": T, "frames": F, "tokens": n, "run_s": time.perf_counter() - t0}
        return out[0, :n]

    def close(self):
        for g in self.graphs.values():
            g.close()
        self.graphs.clear()


def audio_paths(gdir: str, form: str = "fp16fc") -> dict:
    """The audio encoder graphs {T: path} of one form found in `gdir` (export_audio.py's file names)."""
    import re

    out = {}
    for name in sorted(os.listdir(gdir)):
        m = re.fullmatch(r"d1-omni-2b_audio_T(\d+)_" + re.escape(form) + r"\.tflite", name)
        if m:
            out[int(m.group(1))] = os.path.join(gdir, name)
    return out


def graph_paths(gdir: str, form: str = "fp16fc") -> tuple[dict, dict]:
    """The decision graphs {L: path} and tower graphs {(GH, GW): path} of one form found in `gdir` (file names as
    export_decide.py / export_tower.py write them; a bucket's query-blocked file is the one shipped when present)."""
    import re

    dec, tow = {}, {}
    for name in sorted(os.listdir(gdir)):
        m = re.fullmatch(r"d1-omni-2b_decide_L(\d+)(_qb\d+)?_" + re.escape(form) + r"\.tflite", name)
        if m:
            L = int(m.group(1))
            if L not in dec or m.group(2):
                dec[L] = os.path.join(gdir, name)
        m = re.fullmatch(r"d1-omni-2b_tower_G(\d+)x(\d+)(_qb\d+)?_" + re.escape(form) + r"\.tflite", name)
        if m:
            c = (int(m.group(1)), int(m.group(2)))
            if c not in tow or m.group(3):
                tow[c] = os.path.join(gdir, name)
    return dec, tow

