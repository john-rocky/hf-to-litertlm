"""Replay and recording backends for the decision host (host/decide_host), and the generic rows files they read. The host's backend boundary:
    tower(patches [n, 1536], pos [n, 1024], cos [n, 64], sin [n, 64]) -> rows [n / 4, 2048]      tower.max_n
    decide(embeds [T, 2048], cos [T, 64], sin [T, 64], planes [3, T]) -> hidden [T, 2048]        decide.max_len

Every graph call returns the output another run wrote for the very same input bytes: a phone run (out_<tag>.f32), or a Mac run of the
same rows (mac_generic.py). A backend first lays the call's inputs out as the LiteRT backends do (host/decide_host/litert_backends.py:
the smallest bucket, the same pads), takes the sha256 of each input's bytes and finds the row whose recorded slices have the same
sha256. A call with no such row is an error: the replay never answers for an input the device did not see. Recording backends keep
every call's laid-out inputs: make_rows.py writes the rows from them.

Rows file:
  {"kind": "generic", "graph_kind": "tower" | "decide", "signature": "tower_N<N>[_qb<Q>]" | "decide_L<L>[_qb<Q>]" (the multi-signature
   file's name; a single-bucket file has `tower` / `decide`), "inputs": [{name, dtype, shape, file}],
   "outputs": [{name, dtype, shape}], "rows": [{key, index, record, sha256: {input: hex}, ...}], "sets": [...]}
Numbers only: the input files hold a picture's float values and the token rows' embeddings, so they stay under out/, never in this repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "host"))
from decide_host import formulas as F  # noqa: E402

L_BUCKETS = (256, 512, 1024, 2048, 4096)
N_BUCKETS = (400, 576, 1024, 1600, 2304, 4096)
QBLOCK = {4096: 1024}
HIDDEN, VIS_PATCH, VIS_POS, ROT = 2048, 1536, 1024, 64


def decide_sig(L: int) -> str:
    q = QBLOCK.get(L)
    return f"decide_L{L}_qb{q}" if q else f"decide_L{L}"


def tower_sig(N: int) -> str:
    q = QBLOCK.get(N)
    return f"tower_N{N}_qb{q}" if q else f"tower_N{N}"


def decide_file(L: int, form: str = "fp16fc") -> str:
    q = QBLOCK.get(L)
    return f"decide_L{L}_qb{q}_{form}.tflite" if q else f"decide_L{L}_{form}.tflite"


def tower_file(N: int, form: str = "fp16fc") -> str:
    q = QBLOCK.get(N)
    return f"tower_N{N}_qb{q}_{form}.tflite" if q else f"tower_N{N}_{form}.tflite"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def bucket_of(sizes, n: int) -> int:
    for s in sizes:
        if n <= s:
            return s
    raise ValueError(f"{n} is above the top bucket {max(sizes)}")


def tower_inputs(patches, pos, cos, sin) -> tuple[int, dict]:
    """LiteRTTower's layout of one call: (N, {patches, valid, pos, cos, sin}): pads zero, the cos pads 1."""
    n = patches.shape[0]
    N = bucket_of(N_BUCKETS, n)
    feed = {"patches": np.zeros((1, N, VIS_PATCH), np.float32), "valid": np.zeros((1, N), np.float32),
            "pos": np.zeros((1, N, VIS_POS), np.float32), "cos": np.ones((1, N, ROT), np.float32),
            "sin": np.zeros((1, N, ROT), np.float32)}
    feed["patches"][0, :n] = patches
    feed["valid"][0, :n] = 1.0
    feed["pos"][0, :n] = pos
    feed["cos"][0, :n] = cos
    feed["sin"][0, :n] = sin
    return N, feed


def decide_inputs(embeds, cos, sin, planes) -> tuple[int, dict]:
    """LiteRTDecide's layout of one call: (L, {embeds, valid, cos, sin}): the pads continue the planes."""
    T = embeds.shape[0]
    L = bucket_of(L_BUCKETS, T)
    e = np.zeros((1, L, HIDDEN), np.float32)
    e[0, :T] = embeds
    v = np.zeros((1, L), np.float32)
    v[0, :T] = 1.0
    c, s = F.mrope_tables(F.padded_planes(planes, L))
    c[0, :T], s[0, :T] = cos, sin
    return L, {"embeds": e, "valid": v, "cos": c, "sin": s}


def tower_spec(N: int, stem: str) -> tuple[list, list]:
    ins = [{"name": "patches", "dtype": "float32", "shape": [1, N, VIS_PATCH], "file": f"{stem}__patches.f32"},
           {"name": "valid", "dtype": "float32", "shape": [1, N], "file": f"{stem}__valid.f32"},
           {"name": "pos", "dtype": "float32", "shape": [1, N, VIS_POS], "file": f"{stem}__pos.f32"},
           {"name": "cos", "dtype": "float32", "shape": [1, N, ROT], "file": f"{stem}__cos.f32"},
           {"name": "sin", "dtype": "float32", "shape": [1, N, ROT], "file": f"{stem}__sin.f32"}]
    return ins, [{"name": "rows", "dtype": "float32", "shape": [1, N // 4, HIDDEN]}]


def decide_spec(L: int, stem: str) -> tuple[list, list]:
    ins = [{"name": "embeds", "dtype": "float32", "shape": [1, L, HIDDEN], "file": f"{stem}__embeds.f32"},
           {"name": "valid", "dtype": "float32", "shape": [1, L], "file": f"{stem}__valid.f32"},
           {"name": "cos", "dtype": "float32", "shape": [1, L, ROT], "file": f"{stem}__cos.f32"},
           {"name": "sin", "dtype": "float32", "shape": [1, L, ROT], "file": f"{stem}__sin.f32"}]
    return ins, [{"name": "hidden", "dtype": "float32", "shape": [1, L, HIDDEN]}]


def write_rows(path: str, graph_kind: str, signature: str, inputs: list, outputs: list, rows: list, arrays: list,
               extra: dict | None = None) -> list:
    """Write a generic rows file at `path` and its stacked input files beside it (one slice per row, in row order); `arrays[i]` =
    row i's {input name: array}. Returns [(sha256, file name)] of every file written (json last)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    written = []
    for spec in inputs:
        f = os.path.join(d, spec["file"])
        h = hashlib.sha256()
        with open(f + ".tmp", "wb") as fh:
            for a in arrays:
                b = np.ascontiguousarray(a[spec["name"]], dtype=np.float32)
                assert list(b.shape) == spec["shape"], (spec["name"], b.shape, spec["shape"])
                bb = b.tobytes()
                fh.write(bb)
                h.update(bb)
        os.replace(f + ".tmp", f)
        written.append((h.hexdigest(), spec["file"]))
    for i, (r, a) in enumerate(zip(rows, arrays)):
        r["index"] = i
        r["sha256"] = {spec["name"]: sha(a[spec["name"]]) for spec in inputs}
    extra = dict(extra or {})
    doc = {"kind": "generic", "graph_kind": graph_kind, "signature": signature, "inputs": inputs, "outputs": outputs,
           "rows": rows, "sets": extra.pop("sets", [])}
    doc.update(extra)
    with open(path + ".tmp", "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    with open(path, "rb") as fh:
        written.append((hashlib.sha256(fh.read()).hexdigest(), os.path.basename(path)))
    return written


class GenericRows:
    """One generic rows file (its stacked inputs checked slice by slice against the rows' sha256 unless told not to) and, optionally,
    the out file a run of it wrote: per row (file order) every output's whole tensor, little-endian float32, in declared order."""

    def __init__(self, rows_path: str, out_path: str | None = None, check_inputs: bool = True, rows_limit: int = 0):
        self.path, self.out_path = rows_path, out_path
        self.doc = json.load(open(rows_path))
        assert self.doc.get("kind") == "generic", rows_path
        self.dir = os.path.dirname(os.path.abspath(rows_path))
        self.rows = self.doc["rows"]
        self.inputs, self.outputs = self.doc["inputs"], self.doc["outputs"]
        self.counts = {o["name"]: int(np.prod(o["shape"])) for o in self.outputs}
        self.per_row = sum(self.counts.values())
        if check_inputs:
            for spec in self.inputs:
                n = int(np.prod(spec["shape"]))
                mm = np.memmap(os.path.join(self.dir, spec["file"]), dtype="<f4", mode="r")
                if mm.size % n:
                    raise ValueError(f"{spec['file']}: {mm.size} values, not a whole number of {n}-value slices")
                for r in self.rows:
                    got = sha(mm[r["index"] * n:(r["index"] + 1) * n])
                    if got != r["sha256"][spec["name"]]:
                        raise ValueError(f"{rows_path}: row {r['key']} input {spec['name']}: the file's slice {got[:12]} != "
                                         f"the row's {r['sha256'][spec['name']][:12]}")
        self.out = None
        self.n_out = 0
        if out_path is not None:
            mm = np.memmap(out_path, dtype="<f4", mode="r")
            n_rows = rows_limit or len(self.rows)
            if mm.size != n_rows * self.per_row:
                raise ValueError(f"{out_path}: {mm.size * 4} bytes, the rows file's {n_rows} rows need {n_rows * self.per_row * 4}")
            self.out = mm.reshape(n_rows, self.per_row)
            self.n_out = n_rows

    def output(self, position: int, name: str) -> np.ndarray:
        """Output `name` of the row at `position` (file order) as written by the run."""
        assert self.out is not None, f"{self.path}: no out file"
        off = 0
        for o in self.outputs:
            n = self.counts[o["name"]]
            if o["name"] == name:
                return np.asarray(self.out[position, off:off + n], np.float32).reshape(o["shape"])
            off += n
        raise KeyError(name)

    def input_slice(self, position: int, name: str) -> np.ndarray:
        spec = next(s for s in self.inputs if s["name"] == name)
        n = int(np.prod(spec["shape"]))
        mm = np.memmap(os.path.join(self.dir, spec["file"]), dtype="<f4", mode="r")
        idx = self.rows[position]["index"]
        return np.asarray(mm[idx * n:(idx + 1) * n], np.float32).reshape(spec["shape"])

    def by_sha(self) -> dict:
        names = [s["name"] for s in self.inputs]
        return {tuple(r["sha256"][n] for n in names): pos for pos, r in enumerate(self.rows)
                if self.out is None or pos < self.n_out}


class _Replay:
    names: tuple = ()

    def __init__(self, sources: list):
        self.sources = sources
        self.index = {}
        for src in sources:
            assert src.out is not None, f"{src.path}: a replay source needs its out file"
            assert tuple(s["name"] for s in src.inputs) == self.names, (src.path, self.names)
            for key, pos in src.by_sha().items():
                self.index.setdefault(key, (src, pos))
        self.used = []
        self.last = {}

    def _find(self, arrays: dict):
        key = tuple(sha(arrays[n]) for n in self.names)
        hit = self.index.get(key)
        if hit is None:
            raise KeyError(f"no replayed row has these {self.names} bytes ({key[0][:12]}...): the device never ran this input")
        src, pos = hit
        self.used.append((os.path.basename(src.path), src.rows[pos]["key"]))
        return src, pos


class ReplayTower(_Replay):
    names = ("patches", "valid", "pos", "cos", "sin")
    max_n = N_BUCKETS[-1]

    def __call__(self, patches, pos, cos, sin) -> np.ndarray:
        n = patches.shape[0]
        N, arrays = tower_inputs(patches, pos, cos, sin)
        src, p = self._find(arrays)
        rows = src.output(p, "rows")
        assert rows.shape == (1, N // 4, HIDDEN), (rows.shape, N)
        self.last = {"N": N, "n": n, "rows": os.path.basename(src.path), "key": src.rows[p]["key"]}
        return np.array(rows[0, :n // 4])


class ReplayDecide(_Replay):
    names = ("embeds", "valid", "cos", "sin")
    max_len = L_BUCKETS[-1]

    def __call__(self, embeds, cos, sin, planes) -> np.ndarray:
        T = embeds.shape[0]
        L, arrays = decide_inputs(np.asarray(embeds, np.float32), cos, sin, planes)
        src, p = self._find(arrays)
        h = src.output(p, "hidden")
        assert h.shape == (1, L, HIDDEN), (h.shape, L)
        self.last = {"L": L, "T": T, "rows": os.path.basename(src.path), "key": src.rows[p]["key"]}
        return np.array(h[0, :T])


class RecordingTower:
    """Keeps every call's laid-out inputs under the current tag; returns zeros."""
    max_n = N_BUCKETS[-1]

    def __init__(self):
        self.calls, self.tag = [], None

    def __call__(self, patches, pos, cos, sin) -> np.ndarray:
        n = patches.shape[0]
        N, arrays = tower_inputs(patches, pos, cos, sin)
        self.calls.append({"tag": self.tag, "N": N, "n": n, "arrays": arrays})
        return np.zeros((n // 4, HIDDEN), np.float32)


class RecordingDecide:
    """Keeps every call's T and planes under the current tag (a decide row is never static: its embeddings hold the run's own tower
    output, see make_rows.py pic); returns zeros."""
    max_len = L_BUCKETS[-1]

    def __init__(self):
        self.calls, self.tag = [], None

    def __call__(self, embeds, cos, sin, planes) -> np.ndarray:
        T = embeds.shape[0]
        self.calls.append({"tag": self.tag, "T": T, "L": bucket_of(L_BUCKETS, T)})
        return np.zeros((T, embeds.shape[1]), np.float32)
