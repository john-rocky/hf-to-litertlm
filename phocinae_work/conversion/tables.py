"""Step 5b: the token table of the host-lookup form (exports/tables/tok_embeddings_fp16.bin + .json).

    python tables.py

`encoder.embeddings.tok_embeddings.weight` [256000, 384] exactly as stored in model.safetensors (float16; the
publisher's runtime upcasts it to fp32, so these rows ARE the reference values), row-major little-endian, pad row 0.
The SDK's TokenTable (`table_dtype` float16) and the gate app read it by row id.
"""
import json
import struct

import numpy as np

from common import HF, W, dump_json, sha256_file


def main():
    path = HF / "model.safetensors"
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        info = header["encoder.embeddings.tok_embeddings.weight"]
        assert info["dtype"] == "F16" and info["shape"] == [256000, 384], info
        fh.seek(8 + n + info["data_offsets"][0])
        raw = fh.read(info["data_offsets"][1] - info["data_offsets"][0])
    table = np.frombuffer(raw, dtype="<f2").reshape(256000, 384)
    out_dir = W / "exports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "tok_embeddings_fp16.bin"
    bin_path.write_bytes(raw)
    f32 = table.astype(np.float32)
    meta = dict(shape=[256000, 384], dtype="float16", byte_order="little", layout="row-major", filename=bin_path.name, size_bytes=len(raw),
                sha256=sha256_file(bin_path), source_repository="Phocinae/Phocinae-Largha-150M-v1", source_file="model.safetensors",
                source_checkpoint_sha256=sha256_file(path), source_tensor="encoder.embeddings.tok_embeddings.weight",
                source_dtype="F16 (stored); the publisher's Engine upcasts to fp32 on the CPU, so the rows are exact reference values",
                min=float(f32.min()), max=float(f32.max()), all_finite=bool(np.isfinite(f32).all()), pad_id=0,
                pad_policy="gather row 0 for every padded position (the graph masks padded keys; the SDK's TokenTable does the same)")
    dump_json(out_dir / "tok_embeddings.json", meta)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
