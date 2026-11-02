"""Replay backends for the d1-2B host (host/d1_host.py), and the generic rows files they read.

The host runs on the Mac as in the Mac checks; every graph call returns the output another run wrote for the very same
input bytes: a phone leg of the test app's generic mode (out_<tag>.f32; the app is not published), or a Mac run of the
same rows. Before it returns anything, a backend lays the call's inputs out as the LiteRT backends do
(host/lrt_backend.py: the smallest bucket / canvas, zero padding, make_sees / to_canvas), takes the sha256 of each
input's bytes and finds the row whose recorded slices have the same sha256 (each slice's sha256 is checked against the
stacked input file when the rows file is opened). A call with no such row is an error: the replay never answers for an
input the device did not see.

    dec = ReplayDecide([GenericRows("dec_L128.json", rows_dir, out="R/out_A_L128.f32"), ...])
    tow = ReplayTower([GenericRows("twr_G24x24.json", rows_dir, out="R/out_A_tw_G24x24.f32"), ...])
    host = D1Host(SNAP, dec, tow, embed_table=...)

Recording backends (RecordingDecide / RecordingTower) keep every call's laid-out inputs and return zeros (or, for the
tower, a replayed output): make_rows.py writes the rows files from them.

Rows file (the test app's generic contract, plus keys of these scripts):
  {"kind": "generic", "graph_kind": "d1-2b_decide" | "d1-2b_tower", "signature": <the 7-signature file's name for the
   bucket, or "tower">, "inputs": [{"name", "dtype": "float32", "shape", "file"}], "outputs": [{"name", "dtype",
   "shape"}], "rows": [{"key", "index", "record", "call", "sha256": {input: hex}, ...}], "sets": [...]}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from d1_layout import from_canvas, make_sees, to_canvas  # noqa: E402  (numpy only)

BUCKETS = (128, 256, 512, 1024, 2048, 4096, 6144)
CANVASES = ((24, 24), (32, 48), (48, 32), (48, 64), (64, 48), (72, 96), (96, 72))
HIDDEN = 2048
# the signature of each bucket in the 7-signature file (results/export_decide_multi7.json); the single-bucket files
# have one signature, `decide` (the app takes it as its `sig` extra)
DECIDE_SIG = {128: "decide_L128", 256: "decide_L256", 512: "decide_L512", 1024: "decide_L1024",
              2048: "decide_L2048", 4096: "decide_L4096_qb1024", 6144: "decide_L6144_qb1024"}
DECIDE_FILE = {L: f"d1-2b_decide_L{L}{'_qb1024' if L >= 4096 else ''}_fp16fc.tflite" for L in BUCKETS}
TOWER_FILE = {c: f"d1-2b_tower_G{c[0]}x{c[1]}{'_qb1152' if c[0] * c[1] > 4096 else ''}_fp16fc.tflite"
              for c in CANVASES}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def bucket_of(T: int) -> int:
    for L in BUCKETS:
        if L >= T:
            return L
    raise ValueError(f"a sequence of {T} tokens is longer than the largest graph ({BUCKETS[-1]})")


def canvas_of(gh: int, gw: int) -> tuple[int, int]:
    fits = [c for c in CANVASES if c[0] >= gh and c[1] >= gw]
    if not fits:
        raise ValueError(f"a picture of {gh} x {gw} patches fits no tower canvas ({list(CANVASES)})")
    return min(fits, key=lambda c: (c[0] * c[1], c))


def decide_inputs(embeds: np.ndarray, state_len: int, question) -> tuple[int, dict]:
    """LiteRTDecide's layout of one call: (L, {embeds [1,L,2048], sees_q [1,L,66], sees_k [1,L,66]})."""
    T = embeds.shape[0]
    L = bucket_of(T)
    e = np.zeros((1, L, embeds.shape[1]), np.float32)
    e[0, :T] = embeds
    q, k = make_sees(state_len, question, L)
    return L, {"embeds": e, "sees_q": q, "sees_k": k}


def tower_inputs(pixel_values: np.ndarray, grid) -> tuple[tuple[int, int], dict]:
    """LiteRTTower's layout of one call: ((GH, GW), {patches [1,N,768], valid [1,N,1]})."""
    gh, gw = int(grid[0]), int(grid[1])
    c = canvas_of(gh, gw)
    patches, valid = to_canvas(pixel_values, gh, gw, c[0], c[1])
    return c, {"patches": patches, "valid": valid}


def decide_spec(L: int, stem: str) -> tuple[list, list]:
    ins = [{"name": "embeds", "dtype": "float32", "shape": [1, L, HIDDEN], "file": f"{stem}__embeds.f32"},
           {"name": "sees_q", "dtype": "float32", "shape": [1, L, 66], "file": f"{stem}__sees_q.f32"},
           {"name": "sees_k", "dtype": "float32", "shape": [1, L, 66], "file": f"{stem}__sees_k.f32"}]
    return ins, [{"name": "hidden", "dtype": "float32", "shape": [1, L, HIDDEN]}]


def tower_spec(c: tuple[int, int], stem: str) -> tuple[list, list]:
    n = c[0] * c[1]
    ins = [{"name": "patches", "dtype": "float32", "shape": [1, n, 768], "file": f"{stem}__patches.f32"},
           {"name": "valid", "dtype": "float32", "shape": [1, n, 1], "file": f"{stem}__valid.f32"}]
    return ins, [{"name": "features", "dtype": "float32", "shape": [1, n // 4, HIDDEN]}]


def write_rows(path: str, graph_kind: str, signature: str, inputs: list, outputs: list, rows: list,
               arrays: list[dict], extra: dict | None = None) -> list[tuple[str, str]]:
    """Write a generic rows file at `path` and its stacked input files beside it (one slice per row, in row order);
    `arrays[i]` = row i's {input name: array}. Returns [(sha256, file name)] of every file written (json last)."""
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
    """One generic rows file (its stacked inputs checked slice by slice against the rows' sha256) and, optionally, the
    out file a run of it wrote: per row (file order) every output's whole tensor, little-endian float32, in declared
    order (GenericCodec.rowBytes)."""

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
                        raise ValueError(f"{rows_path}: row {r['key']} input {spec['name']}: the file's slice "
                                         f"{got[:12]} != the row's {r['sha256'][spec['name']][:12]}")
        self.out = None
        if out_path is not None:
            mm = np.memmap(out_path, dtype="<f4", mode="r")
            n_rows = rows_limit or len(self.rows)
            if mm.size != n_rows * self.per_row:
                raise ValueError(f"{out_path}: {mm.size * 4} bytes, the rows file's {n_rows} rows need "
                                 f"{n_rows * self.per_row * 4}")
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

    def by_sha(self) -> dict:
        names = [s["name"] for s in self.inputs]
        return {tuple(r["sha256"][n] for n in names): pos for pos, r in enumerate(self.rows)
                if self.out is None or pos < self.n_out}


class _Replay:
    names: tuple = ()

    def __init__(self, sources: list[GenericRows]):
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
            raise KeyError(f"no replayed row has these {self.names} bytes ({key[0][:12]}...): the device never ran "
                           f"this input")
        src, pos = hit
        self.used.append((os.path.basename(src.path), src.rows[pos]["key"]))
        return src, pos


class ReplayDecide(_Replay):
    names = ("embeds", "sees_q", "sees_k")
    max_len = BUCKETS[-1]

    def __call__(self, embeds: np.ndarray, state_len: int, question) -> np.ndarray:
        T = embeds.shape[0]
        L, arrays = decide_inputs(np.asarray(embeds, np.float32), state_len, question)
        src, pos = self._find(arrays)
        h = src.output(pos, "hidden")
        assert h.shape == (1, L, HIDDEN), (h.shape, L)
        self.last = {"L": L, "T": T, "rows": os.path.basename(src.path), "key": src.rows[pos]["key"]}
        return np.array(h[0, :T])


class ReplayTower(_Replay):
    names = ("patches", "valid")

    def __call__(self, pixel_values: np.ndarray, grid) -> np.ndarray:
        gh, gw = int(grid[0]), int(grid[1])
        c, arrays = tower_inputs(pixel_values, grid)
        src, pos = self._find(arrays)
        f = src.output(pos, "features")
        self.last = {"canvas": c, "grid": (gh, gw), "rows": os.path.basename(src.path), "key": src.rows[pos]["key"]}
        return np.array(from_canvas(f, gh, gw, c[0], c[1]))


class RecordingDecide:
    """Keeps every call's laid-out inputs (bucket, arrays, state_len, T) under the current tag; returns zeros."""
    max_len = BUCKETS[-1]

    def __init__(self):
        self.calls = []
        self.tag = None

    def __call__(self, embeds: np.ndarray, state_len: int, question) -> np.ndarray:
        T = embeds.shape[0]
        L, arrays = decide_inputs(np.asarray(embeds, np.float32), state_len, question)
        self.calls.append({"tag": self.tag, "L": L, "T": T, "state_len": int(state_len), "arrays": arrays})
        return np.zeros((T, embeds.shape[1]), np.float32)


class RecordingTower:
    """Keeps every call's laid-out inputs (canvas, grid, arrays); returns zeros, or the replayed features when given a
    ReplayTower (the picture sequences' decide inputs then hold the run's own tower output)."""

    def __init__(self, replay: ReplayTower | None = None):
        self.calls = []
        self.tag = None
        self.replay = replay

    def __call__(self, pixel_values: np.ndarray, grid) -> np.ndarray:
        gh, gw = int(grid[0]), int(grid[1])
        c, arrays = tower_inputs(pixel_values, grid)
        self.calls.append({"tag": self.tag, "canvas": c, "grid": (gh, gw), "arrays": arrays})
        if self.replay is not None:
            return self.replay(pixel_values, grid)
        return np.zeros((gh * gw // 4, HIDDEN), np.float32)
