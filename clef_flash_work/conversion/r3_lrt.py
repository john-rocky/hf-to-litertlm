"""Round 3 LiteRT runtime helpers (ai-edge-litert CompiledModel, lt094dev): open a file on the CPU (XNNPACK) or the
Mac GPU (Metal, fp32 precision or the default), run its one signature by tensor name, and capture the runtime's own
delegation lines (scripts/clef_rt.py: runtime_log_verbose + the `Replacing N out of M node(s)` / `Partitioned
subgraph` parsers) into a log file per compiled model.

    with RuntimeLog(K / "logs/r3_x.runtime.log") as rl:
        m = Model(path, "gpu_f32")
        y = m.run({"patches": x})["rows"]
    rl.delegation    # {"replacing": [...], "partitioned": [...]}, rl.key_lines

Accelerators: "cpu" = HardwareAccelerator.CPU, CpuOptions(num_threads); the XNNPACK weight cache path is left empty
(no cache file is written; the caller lists the directory before / after). "gpu_f32" = HardwareAccelerator.GPU,
GpuOptions(enforce_f32=True) (Metal fp32 precision, the gate); "gpu_default" = GpuOptions() (the default
precision: fp16 storage, recorded only)."""
from __future__ import annotations

import importlib.metadata
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_rt import LINE_KEYS, PARTITIONED, REPLACING, runtime_log_verbose  # noqa: E402

SIG = "serving_default"
_LOGGER = {}


def logger_on() -> dict:
    if not _LOGGER:
        _LOGGER.update(runtime_log_verbose())
    return _LOGGER


def delegation_of(text: str) -> dict:
    return {"replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                          for m in REPLACING.findall(text)],
            "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                            for m in PARTITIONED.findall(text)]}


class RuntimeLog:
    """fd 2 -> `path` while open (the runtime writes its lines to the process's stderr)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.delegation, self.key_lines, self.lines = None, [], 0

    def __enter__(self):
        logger_on()
        sys.stderr.flush()
        self.f = open(self.path, "w")
        self.saved = os.dup(2)
        os.dup2(self.f.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.f.close()
        text = self.path.read_text(errors="replace")
        self.lines = len(text.splitlines())
        self.key_lines = [ln for ln in text.splitlines() if LINE_KEYS.search(ln)][:80]
        self.delegation = delegation_of(text)
        return False


def options(accel: str, threads: int = 8):
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if accel == "cpu":
        cpu = CpuOptions(num_threads=threads)
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu), {
            "accelerator": "CPU (XNNPACK)", "cpu_options": cpu._as_flat_kwargs()}
    if accel in ("gpu_f32", "gpu_default"):
        g = GpuOptions(enforce_f32=True) if accel == "gpu_f32" else GpuOptions()
        return Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=g), {
            "accelerator": "GPU (Metal)", "gpu_options": g._as_flat_kwargs()}
    raise ValueError(accel)


class Model:
    def __init__(self, path: Path, accel: str, threads: int = 8):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.accel = Path(path), accel
        opts, self.options = options(accel, threads)
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(self.path), options=opts)
        self.compile_seconds = round(time.perf_counter() - t0, 3)
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception as e:  # informational
            self.fully_accelerated = f"unavailable: {type(e).__name__}: {e}"
        sigs = self.model.get_signature_list()
        assert list(sigs) == [SIG], sigs
        self.in_det = self.model.get_input_tensor_details(SIG)
        self.out_det = self.model.get_output_tensor_details(SIG)
        self.ins = {n: self.model.create_input_buffer_by_name(SIG, n) for n in self.in_det}
        self.outs = {n: self.model.create_output_buffer_by_name(SIG, n) for n in self.out_det}
        self.out_shape = {n: [int(x) for x in d["shape"]] for n, d in self.out_det.items()}

    def info(self) -> dict:
        return {"file": self.path.name, "accel": self.accel, "options": self.options, "compile_seconds": self.compile_seconds,
                "is_fully_accelerated": self.fully_accelerated,
                "inputs": {n: [int(x) for x in d["shape"]] for n, d in self.in_det.items()}, "outputs": self.out_shape,
                "ai_edge_litert": importlib.metadata.version("ai-edge-litert")}

    def write(self, feed: dict) -> None:
        for n, v in feed.items():
            self.ins[n].write(np.ascontiguousarray(v, dtype=np.float32))

    def invoke(self) -> None:
        self.model.run_by_name(SIG, self.ins, self.outs)

    def read(self, name: str | None = None) -> np.ndarray:
        name = name or next(iter(self.out_det))
        shp = self.out_shape[name]
        return np.asarray(self.outs[name].read(int(np.prod(shp)), np.float32), dtype=np.float32).reshape(shp)

    def run(self, feed: dict) -> dict:
        self.write(feed)
        self.invoke()
        return {n: self.read(n) for n in self.out_det}

    def close(self) -> None:
        for b in list(self.ins.values()) + list(self.outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


def listing(d: Path) -> list:
    return sorted(f"{p.name} {p.stat().st_size}" for p in Path(d).iterdir()) if Path(d).is_dir() else []


def ms_stats(ms: list) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "p90": float(np.percentile(a, 90))}
