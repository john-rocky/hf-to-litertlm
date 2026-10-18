"""Step 5b: the token table of the host-lookup form (exports/tables/tok_embeddings_fp16.bin + .json; --f32 for the exact rows).

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer; $P scripts/tables.py [--f32]

`embeddings.tok_embeddings.weight` [256000, 768] as stored in model.safetensors (float32), written row-major little-endian:
fp16 = the rows ROUNDED to float16 (393.2 MB; the rounding is measured by the embeds-form gates, it is not free here as it
was for phocinae's fp16 checkpoint), f32 = the exact rows (786.4 MB). Pad row 0.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f32", action="store_true")
    a = ap.parse_args()
    path = W / "hf" / "model.safetensors"
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        info = header["embeddings.tok_embeddings.weight"]
        assert info["dtype"] == "F32" and info["shape"] == [256000, 768], info
        fh.seek(8 + n + info["data_offsets"][0])
        raw = fh.read(info["data_offsets"][1] - info["data_offsets"][0])
    f32 = np.frombuffer(raw, dtype="<f4").reshape(256000, 768)
    out_dir = W / "exports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.f32:
        bin_path = out_dir / "tok_embeddings_f32.bin"
        assert not bin_path.exists(), bin_path
        bin_path.write_bytes(raw)
        dtype, note = "float32", "the exact checkpoint rows"
    else:
        bin_path = out_dir / "tok_embeddings_fp16.bin"
        assert not bin_path.exists(), bin_path
        h16 = f32.astype("<f2")
        bin_path.write_bytes(h16.tobytes())
        back = h16.astype(np.float32)
        dtype = "float16"
        note = dict(rounding="the checkpoint's float32 rows rounded to float16", max_abs_rounding=float(np.abs(back - f32).max()),
                    max_rel_rounding=float((np.abs(back - f32) / np.maximum(np.abs(f32), 1e-12)).max()), rows_with_overflow=int((~np.isfinite(back)).any(axis=1).sum()))
    meta = dict(shape=[256000, 768], dtype=dtype, byte_order="little", layout="row-major", filename=bin_path.name, size_bytes=bin_path.stat().st_size,
                sha256=sha256_file(bin_path), source_repository="vllm-sr/Vela-1.0-Encoder-307M-Embedding", source_file="model.safetensors",
                source_checkpoint_sha256=sha256_file(path), source_tensor="embeddings.tok_embeddings.weight", source_dtype="F32", note=note,
                min=float(f32.min()), max=float(f32.max()), all_finite=bool(np.isfinite(f32).all()), pad_id=0,
                pad_policy="gather row 0 for every padded position (the graph masks padded keys and zeroes padded rows before the pooling)")
    (out_dir / f"{bin_path.stem}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
