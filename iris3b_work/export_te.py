"""Writes the twelve text-encoder graphs of iris-3b Turbo (float32 weights) with litert-torch.

    python export_te.py --out OUT [--text-encoder DIR] [--chunks 0-11]

Graph i runs decoder layers te_modules.CHUNKS[i] of the Qwen3-VL-4B-Instruct language stack (35 of its 36 layers in
all) and returns the hidden state of the 1-based layer the DiT reads (2, 5, ..., 35). Inputs: hidden [1, 334, 2560],
an additive attention mask [1, 32, 334, 334] and the rotary tables cos / sin [1, 1, 334, 128] (te_modules.py has the
rewrites). Each graph is traced with zeros of those shapes: the export keeps the shapes and dtypes, not the values.

Writes OUT/fp32/te/iris_te_c<NN>_f32.tflite and OUT/fp32/te/graphs.json (bytes, sha256 and seconds per file). A graph
whose file is already there is skipped. --text-encoder defaults to OUT/weights/Qwen3-VL-4B-Instruct.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import te_modules as T


def parse_range(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output root (default: out)")
    ap.add_argument("--text-encoder", default=None,
                    help="Qwen3-VL-4B-Instruct directory (default: OUT/weights/Qwen3-VL-4B-Instruct)")
    ap.add_argument("--chunks", default="0-11")
    args = ap.parse_args()
    root = Path(args.out)
    T.TE_DIR = Path(args.text_encoder) if args.text_encoder else root / "weights" / "Qwen3-VL-4B-Instruct"
    torch.set_num_threads(8)
    t0 = time.time()
    import litert_torch

    graphs = root / "fp32" / "te"
    graphs.mkdir(parents=True, exist_ok=True)
    log_path = graphs / "graphs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    shapes = [(1, T.SEQ, T.HIDDEN), (1, T.N_HEADS, T.SEQ, T.SEQ), (1, 1, T.SEQ, T.HEAD_DIM),
              (1, 1, T.SEQ, T.HEAD_DIM)]
    for ci in parse_range(args.chunks):
        out = graphs / T.chunk_name(ci, "f32")
        if out.exists() and out.stat().st_size > 0:
            print("SKIP", out.name, flush=True)
            continue
        tc = time.time()
        chunk = T.build_chunk(ci)
        sample = tuple(torch.zeros(s, dtype=torch.float32) for s in shapes)
        edge = litert_torch.convert(chunk, sample, quant_config=None)
        tmp = out.with_name(out.stem + ".part.tflite")
        edge.export(str(tmp))
        os.replace(tmp, out)
        ent = {"chunk": ci, "layers": list(T.CHUNKS[ci]), "bytes": out.stat().st_size, "sha256": T.sha256(out),
               "seconds": round(time.time() - tc, 1), "litert_torch": litert_torch.__version__,
               "torch": torch.__version__}
        log[out.name] = ent
        log_path.write_text(json.dumps(log, indent=1))
        print("WROTE", json.dumps({"file": out.name, **ent}), flush=True)
        del edge, chunk
    print("TE_GRAPHS_DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
