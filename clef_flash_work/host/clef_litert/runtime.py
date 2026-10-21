"""The LiteRT graphs of a clef-flash folder on ai-edge-litert's CompiledModel, and the host around them.

    with ClefLiteRT.from_dir("<folder>", accelerator="gpu") as clef:     # "gpu" = float32 precision, "cpu" = XNNPACK
        body = clef.decide(request, tile=448)                           # the SystemOne-compatible response

Folder layout (FOLDER below; every path relative to the folder):
    clef-flash_rowprefill_embeds_L{256,512,1024,2048,4096}_fp16fc.tflite   row graphs: embeds, valid, cos, sin -> hidden
    clef-flash_vision_tower_g{256,448}_fp16fc.tflite                        tower + merger: patches, pos -> rows
    tables/vision_pos_g{256,448}.safetensors                                the towers' position rows (float32, key "pos")
    tables/embed_tokens.safetensors, tables/lm_head.safetensors             the bf16 vocabulary tables
    head/joint_head.safetensors, head/joint_head_config.json                the checkpoint's joint schema head
    tokenizer/tokenizer.json                                                the checkpoint's tokenizer

Only the row graphs present are used: a request runs on the smallest present length that holds it, and a request
longer than the largest present one is refused (RequestTooLong), never cut. Graphs are compiled when a request first
needs them. One row graph stays compiled at a time (its fp16 weights are about 13.9 GB; on a Mac GPU at float32
precision a compiled row graph held about 37 GB): a request of another length closes it and compiles the next one.
The towers (about 0.9 GB each) stay compiled once used.

Accelerators: "gpu" = HardwareAccelerator.GPU with GpuOptions(enforce_f32=True) (Metal on a Mac; the graphs are
checked at float32 precision only: at the default precision (fp16 storage) the row graphs' norms overflow and the
towers' rows are wrong); "cpu" = HardwareAccelerator.CPU with CpuOptions(num_threads=threads) (XNNPACK; no weight
cache file is written). No other runtime option is set."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .head import JointHead
from .host import HIDDEN, ClefHost
from .pixels import TILES, tile_grid
from .positions import BUCKETS
from .tables import Bf16Table, read_header

SIGNATURE = "serving_default"
ACCELERATORS = ("gpu", "cpu")
FOLDER = {
    "row": "clef-flash_rowprefill_embeds_L{length}_fp16fc.tflite",
    "tower": "clef-flash_vision_tower_g{tile}_fp16fc.tflite",
    "tower_pos": "tables/vision_pos_g{tile}.safetensors",
    "embed_tokens": "tables/embed_tokens.safetensors",
    "lm_head": "tables/lm_head.safetensors",
    "joint_head": "head/joint_head.safetensors",
    "joint_head_config": "head/joint_head_config.json",
    "tokenizer": "tokenizer/tokenizer.json",
}
ROW_INPUTS = ("embeds", "valid", "cos", "sin")
TOWER_INPUTS = ("patches", "pos")


def load_f32_tensor(path: str | Path, key: str | None = None) -> np.ndarray:
    """One float32 tensor of a safetensors file (the towers' position rows), read without a tensor framework."""
    header, data_start = read_header(path)
    names = [k for k in header if k != "__metadata__"]
    if key is None:
        if len(names) != 1:
            raise ValueError(f"{Path(path).name} holds {len(names)} tensors; name the key")
        key = names[0]
    e = header[key]
    if e["dtype"] != "F32":
        raise ValueError(f"{key}: dtype {e['dtype']}, expected F32")
    start, end = e["data_offsets"]
    n = int(np.prod(e["shape"], dtype=np.int64))
    if end - start != 4 * n:
        raise ValueError(f"{key}: {end - start} bytes for {n} F32 values")
    with open(path, "rb") as f:
        f.seek(data_start + start)
        data = f.read(end - start)
    return np.frombuffer(data, dtype="<f4").reshape(e["shape"]).astype(np.float32)


def write_f32_tensor(path: str | Path, key: str, array: np.ndarray) -> dict:
    """A one-key safetensors file of a float32 array, header without __metadata__ (padded with spaces to 8 bytes, as the
    safetensors writer pads). Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    a = np.ascontiguousarray(array, dtype="<f4")
    hdr = json.dumps({key: {"dtype": "F32", "shape": list(a.shape), "data_offsets": [0, a.nbytes]}},
                     separators=(",", ":")).encode()
    hdr += b" " * (-len(hdr) % 8)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr)
        f.write(a.tobytes())
    tmp.rename(path)
    return {"file": path.name, "key": key, "dtype": "F32", "shape": list(a.shape), "bytes": path.stat().st_size}


def compile_options(accelerator: str, threads: int = 8):
    """-> (CompiledModel options, a plain description of them)."""
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if accelerator == "cpu":
        cpu = CpuOptions(num_threads=threads)
        return (Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu),
                {"accelerator": "cpu", "backend": "CPU (XNNPACK)", "num_threads": threads})
    if accelerator == "gpu":
        gpu = GpuOptions(enforce_f32=True)
        return (Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gpu),
                {"accelerator": "gpu", "backend": "GPU, float32 precision", "enforce_f32": True})
    raise ValueError(f"accelerator {accelerator!r}: one of {ACCELERATORS}")


class LiteRTGraph:
    """One .tflite file with one signature, compiled on `accelerator`; call it with its inputs by name."""

    def __init__(self, path: str | Path, accelerator: str = "gpu", threads: int = 8):
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.accelerator = Path(path), accelerator
        opts, self.options = compile_options(accelerator, threads)
        self.model = CompiledModel.from_file(str(self.path), options=opts)
        signatures = list(self.model.get_signature_list())
        if signatures != [SIGNATURE]:
            self.model.close()
            raise ValueError(f"{self.path.name}: signatures {signatures}, expected [{SIGNATURE!r}]")
        self.input_shapes = {n: tuple(int(x) for x in d["shape"])
                             for n, d in self.model.get_input_tensor_details(SIGNATURE).items()}
        self.output_shapes = {n: tuple(int(x) for x in d["shape"])
                              for n, d in self.model.get_output_tensor_details(SIGNATURE).items()}
        self.inputs = {n: self.model.create_input_buffer_by_name(SIGNATURE, n) for n in self.input_shapes}
        self.outputs = {n: self.model.create_output_buffer_by_name(SIGNATURE, n) for n in self.output_shapes}

    def fully_accelerated(self) -> bool:
        return bool(self.model.is_fully_accelerated())

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        if sorted(feeds) != sorted(self.input_shapes):
            raise ValueError(f"{self.path.name}: inputs {sorted(feeds)}, expected {sorted(self.input_shapes)}")
        for name, value in feeds.items():
            value = np.ascontiguousarray(value, dtype=np.float32)
            if value.shape != self.input_shapes[name]:
                raise ValueError(f"{self.path.name}: {name} {value.shape}, expected {self.input_shapes[name]}")
            self.inputs[name].write(value)
        self.model.run_by_name(SIGNATURE, self.inputs, self.outputs)
        return {name: np.asarray(self.outputs[name].read(int(np.prod(shape)), np.float32),
                                 dtype=np.float32).reshape(shape)
                for name, shape in self.output_shapes.items()}

    def close(self) -> None:
        for buffer in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                buffer.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class ClefGraphs:
    """The row graphs (one file per length) and the tower graphs (one per tile), compiled when first needed.
    row_graph(length, inputs) and tower(tile, patches) are the callables ClefHost.decide takes."""

    def __init__(self, row_files: dict, tower_files: dict | None = None, tower_pos: dict | None = None,
                 accelerator: str = "gpu", threads: int = 8):
        if accelerator not in ACCELERATORS:
            raise ValueError(f"accelerator {accelerator!r}: one of {ACCELERATORS}")
        self.row_files = {int(L): Path(p) for L, p in row_files.items()}
        if not self.row_files:
            raise ValueError("no row graph")
        self.tower_files = {int(t): Path(p) for t, p in (tower_files or {}).items()}
        self.tower_pos = {}
        for t, pos in (tower_pos or {}).items():
            self.tower_pos[int(t)] = load_f32_tensor(pos) if isinstance(pos, (str, Path)) else np.asarray(pos, np.float32)
        missing = sorted(set(self.tower_files) - set(self.tower_pos))
        if missing:
            raise ValueError(f"tower(s) {missing} without position rows")
        self.accelerator, self.threads = accelerator, threads
        self.row = None                     # (length, LiteRTGraph): the one compiled row graph
        self.towers: dict[int, LiteRTGraph] = {}
        self.compiled: list[dict] = []      # what was compiled, in order (file, accelerator)

    @classmethod
    def from_dir(cls, folder: str | Path, accelerator: str = "gpu", threads: int = 8, lengths=None) -> "ClefGraphs":
        """The graphs of a folder laid out as FOLDER (only the files present; `lengths` limits the row graphs)."""
        folder = Path(folder)
        rows = {L: folder / FOLDER["row"].format(length=L) for L in BUCKETS if lengths is None or L in lengths}
        rows = {L: p for L, p in rows.items() if p.is_file()}
        towers = {t: folder / FOLDER["tower"].format(tile=t) for t in TILES}
        towers = {t: p for t, p in towers.items() if p.is_file()}
        pos = {t: folder / FOLDER["tower_pos"].format(tile=t) for t in towers}
        return cls(rows, towers, pos, accelerator, threads)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(sorted(self.row_files))

    @property
    def tiles(self) -> tuple[int, ...]:
        return tuple(sorted(self.tower_files))

    def _row_graph(self, length: int) -> LiteRTGraph:
        if self.row is not None and self.row[0] == length:
            return self.row[1]
        if length not in self.row_files:
            raise ValueError(f"no row graph of length {length} (present: {list(self.lengths)})")
        if self.row is not None:            # one row graph at a time
            self.row[1].close()
            self.row = None
        graph = LiteRTGraph(self.row_files[length], self.accelerator, self.threads)
        if graph.input_shapes["embeds"] != (1, length, HIDDEN):
            graph.close()
            raise ValueError(f"{self.row_files[length].name}: embeds {graph.input_shapes['embeds']}")
        self.row = (length, graph)
        self.compiled.append({"file": self.row_files[length].name, "accelerator": self.accelerator})
        return graph

    def row_graph(self, length: int, inputs: dict) -> np.ndarray:
        """The row graph of `length` on the host's row_inputs -> hidden [1, L, 4096]."""
        graph = self._row_graph(int(length))
        return graph(**{n: inputs[n] for n in ROW_INPUTS})["hidden"]

    def _tower(self, tile: int) -> LiteRTGraph:
        if tile not in self.towers:
            if tile not in self.tower_files:
                raise ValueError(f"no tower graph for a {tile} px tile (present: {list(self.tiles)})")
            self.towers[tile] = LiteRTGraph(self.tower_files[tile], self.accelerator, self.threads)
            self.compiled.append({"file": self.tower_files[tile].name, "accelerator": self.accelerator})
        return self.towers[tile]

    def tower(self, tile: int, patches: np.ndarray) -> np.ndarray:
        """The tower of `tile` on the host's patches [4 G^2, 1536] -> picture rows [G^2, 4096]."""
        g = tile_grid(tile)
        rows = self._tower(tile)(patches=np.asarray(patches, np.float32)[None], pos=self.tower_pos[tile][None])["rows"]
        if rows.shape != (1, g * g, HIDDEN):
            raise ValueError(f"tower rows {rows.shape}, expected {(1, g * g, HIDDEN)}")
        return rows[0]

    def close(self) -> None:
        if self.row is not None:
            self.row[1].close()
            self.row = None
        for graph in self.towers.values():
            graph.close()
        self.towers = {}


class ClefLiteRT:
    """The host (tokenizer, tables, joint head, response) and the graphs of one folder."""

    def __init__(self, host: ClefHost, graphs: ClefGraphs):
        self.host, self.graphs = host, graphs

    @classmethod
    def from_dir(cls, folder: str | Path, accelerator: str = "gpu", threads: int = 8, lengths=None) -> "ClefLiteRT":
        from tokenizers import Tokenizer

        folder = Path(folder)
        graphs = ClefGraphs.from_dir(folder, accelerator, threads, lengths)
        host = ClefHost(Tokenizer.from_file(str(folder / FOLDER["tokenizer"])),
                        Bf16Table(folder / FOLDER["embed_tokens"]), Bf16Table(folder / FOLDER["lm_head"]),
                        JointHead.from_files(folder / FOLDER["joint_head"], folder / FOLDER["joint_head_config"]),
                        buckets=graphs.lengths)
        return cls(host, graphs)

    def decide(self, request: dict, tile: int = 448) -> dict:
        """request -> the SystemOne-compatible response (ClefHost.decide on this folder's graphs). A picture at a tile
        whose tower file is not in the folder is refused (UnsupportedMedia)."""
        tower = self.graphs.tower if tile in self.graphs.tiles else None
        return self.host.decide(request, self.graphs.row_graph, tower, tile=tile)

    def close(self) -> None:
        self.graphs.close()

    def __enter__(self) -> "ClefLiteRT":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
