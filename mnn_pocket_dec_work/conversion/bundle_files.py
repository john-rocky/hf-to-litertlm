"""Write the files the host reads, from the downloaded original repository (standard library only).

    python conversion/bundle_files.py --src <MNN-Pocket-0.3B-Dec download> --out <directory>

tokenizer.json = base/tokenizer.json, decision_head.safetensors, temperature.json and LICENSE = LICENSE.txt are copied
byte for byte. embedding.safetensors holds one tensor, `model.embed_tokens.weight`: the bf16 bytes of that key in
base/model.safetensors (65,536 x 896), with a one-tensor safetensors header and no metadata. Prints each file's SHA-256;
an existing file with other bytes stops the script."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

KEY = "model.embed_tokens.weight"


def embedding_bytes(checkpoint):
    with open(checkpoint, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[KEY]
        assert meta["dtype"] == "BF16" and meta["shape"] == [65536, 896], meta
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        data = f.read(end - start)
    header = json.dumps({KEY: {"dtype": "BF16", "shape": meta["shape"], "data_offsets": [0, len(data)]}},
                        separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="the downloaded taobao-mnn/MNN-Pocket-0.3B-Dec directory")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = {"tokenizer.json": (a.src / "base/tokenizer.json").read_bytes(),
             "decision_head.safetensors": (a.src / "decision_head.safetensors").read_bytes(),
             "temperature.json": (a.src / "temperature.json").read_bytes(),
             "LICENSE": (a.src / "LICENSE.txt").read_bytes(),
             "embedding.safetensors": embedding_bytes(a.src / "base/model.safetensors")}
    for name, data in files.items():
        path = a.out / name
        if path.exists() and path.read_bytes() != data:
            raise SystemExit(f"{path} exists with other bytes")
        path.write_bytes(data)
        print(f"{name}: {len(data):,} bytes, SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
