"""Round 5 acceptance 4: cut the host's two tables out of a d1-3B snapshot, reading only the bytes they need.

    python3 -I scripts/d1_tables.py <snapshot dir> --out <dir> [--ids <json file>] [--verify]
    python3 -I scripts/d1_tables.py --check-header [<snapshot dir>]

<snapshot dir>/model.safetensors is read by its header (8-byte little-endian length + JSON), then by seek + read at the
offsets the header gives; the file is never loaded whole (the bytes read are counted and recorded). Two outputs:
  readout_table.safetensors        `ids` int64 [n] + `rows` float32 [n, d]: rows of
                                   model.language_model.embed_tokens.weight (the tied embedding = lm_head) at the ids
                                   of host/contract.json `readout.table.ids` (1,234 on d1-3B; --ids <json list> for a
                                   tiny stand-in), sorted ascending as the contract lists them
  vision_position_table.safetensors `table` float32 [S, S, C]: model.vision_tower.vision_model.embeddings.
                                   position_embedding.weight [S * S, C] reshaped (S = 16, C = 1152 on d1-3B)
bfloat16 -> float32 is exact (the 16 bits become the high half of the float32); float32 is copied; float16 is widened.
<out>/tables_manifest.json records per file the shape, dtype, sha256, the source tensor's name, dtype, shape and
data offsets, the absolute byte ranges read, and the snapshot header's sha256 (the .safetensors files carry no metadata,
so their sha256 depends on the rows only).
--verify (needs torch + safetensors; tiny files or a real snapshot): every cut row and the position table compared bit
for bit with safetensors' own load of the whole tensor (`.to(torch.float32)`), an independent reader.
--check-header: the two tensors in evidence/hub_safetensors_header_da1fe36a.json (the Hub's header at the pinned
revision) against config.json (vocab x hidden, num_patches x vision hidden), dtype BF16, byte length = numel x 2, and,
with a snapshot dir, against that snapshot's own header (names, dtype, shape, data offsets equal) -> printed table +
results/tables_header_check.json (with a snapshot: results/tables_header_check_<--tag, default the snapshot dir's
name>.json; never overwritten); with --out it also refuses to cut from a snapshot whose two entries differ from the
evidence header.
Never overwrites an output. Standard library + numpy + safetensors (writing) only on the cutting path.
--full-embed (round 6c, with <snapshot dir> --out <dir>): <out>/embed_table.safetensors = the whole tied table
model.language_model.embed_tokens.weight [128000, 2048] kept in bfloat16 (its bytes copied as they are in the
snapshot, 524 MB; key `embed_tokens.weight`, dtype BF16, no __metadata__) for the embeds variant of the row graph: the
host looks up the row of every id and widens it to float32 (`EmbedTable`; exact, the 16 bits become the high half).
<out>/embed_table_manifest.json records the file's bytes and sha256, the source entry and the byte range read; with
--verify the file is read back with safetensors + torch (an independent reader) and compared bit for bit with the
snapshot's tensor, and `EmbedTable` with the float32 widening of that tensor. The existing table files are not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
CONTRACT = K / "host/contract.json"
EVIDENCE = K / "evidence/hub_safetensors_header_da1fe36a.json"
CONFIG = K / "hf_small/config.json"
READOUT = "model.language_model.embed_tokens.weight"
POSITION = "model.vision_tower.vision_model.embeddings.position_embedding.weight"
ITEM = {"BF16": 2, "F16": 2, "F32": 4}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> tuple[dict, int, bytes]:
    """-> (header dict, data start offset, the raw header bytes)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), 8 + n, raw


def to_f32(raw: bytes, dtype: str, count: int) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, "<u2", count).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, "<f4", count).astype(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, "<f2", count).astype(np.float32)
    raise ValueError(f"dtype {dtype} not handled")


def cut(snapshot: Path, out: Path, ids: list[int] | None, verify: bool) -> dict:
    from safetensors.numpy import save_file

    path = snapshot / "model.safetensors"
    header, start, raw_header = read_header(path)
    ev = json.loads(EVIDENCE.read_text())["header"]
    real = header.get(READOUT, {}).get("shape") == ev[READOUT]["shape"]
    if real:   # a real snapshot must carry the pinned revision's entries for both tensors
        for name in (READOUT, POSITION):
            assert header[name] == ev[name], (name, header[name], ev[name])
    if ids is None:
        ids = json.loads(CONTRACT.read_text())["readout"]["table"]["ids"]
    assert ids == sorted(set(ids)), "ids must be ascending and unique"
    targets = {}
    for name in (READOUT, POSITION):
        e = header[name]
        b, epos = e["data_offsets"]
        n = int(np.prod(e["shape"]))
        assert epos - b == n * ITEM[e["dtype"]], (name, e)
        targets[name] = e
    out.mkdir(parents=True, exist_ok=True)
    files = {"readout": out / "readout_table.safetensors", "position": out / "vision_position_table.safetensors",
             "manifest": out / "tables_manifest.json"}
    for p in files.values():
        assert not p.exists(), f"refusing to overwrite {p}"
    t0 = time.perf_counter()
    bytes_read, ranges = 0, {}
    with open(path, "rb") as f:
        # read-out rows: one seek + read per id
        e = targets[READOUT]
        vocab, d = e["shape"]
        assert all(0 <= i < vocab for i in ids), (min(ids), max(ids), vocab)
        row_bytes = d * ITEM[e["dtype"]]
        rows = np.empty((len(ids), d), np.float32)
        first = []
        for k, i in enumerate(ids):
            off = start + e["data_offsets"][0] + i * row_bytes
            f.seek(off)
            raw = f.read(row_bytes)
            assert len(raw) == row_bytes, (i, len(raw))
            rows[k] = to_f32(raw, e["dtype"], d)
            bytes_read += row_bytes
            if k < 3:
                first.append([off, off + row_bytes])
        ranges[READOUT] = {"rows": len(ids), "row_bytes": row_bytes, "first_ranges": first,
                           "tensor_range": [start + e["data_offsets"][0], start + e["data_offsets"][1]]}
        # position table: the whole tensor (S * S x C, 0.6 MB on d1-3B)
        e = targets[POSITION]
        n_pos, c = e["shape"]
        side = int(round(n_pos ** 0.5))
        assert side * side == n_pos, n_pos
        off = start + e["data_offsets"][0]
        f.seek(off)
        raw = f.read(e["data_offsets"][1] - e["data_offsets"][0])
        table = to_f32(raw, e["dtype"], n_pos * c).reshape(side, side, c)
        bytes_read += len(raw)
        ranges[POSITION] = {"range": [off, off + len(raw)]}
    seconds = time.perf_counter() - t0
    # no __metadata__ in the files: safetensors writes a metadata dict's keys in a random order (two cuts of the same
    # rows differed only there), and the provenance belongs to the manifest anyway; without it the bytes are the
    # tensors' only, so the sha256 repeats across runs and safetensors versions
    meta = {"source": str(path), "header_sha256": hashlib.sha256(raw_header).hexdigest()}
    save_file({"ids": np.asarray(ids, np.int64), "rows": rows}, str(files["readout"]))
    save_file({"table": np.ascontiguousarray(table)}, str(files["position"]))
    import safetensors

    doc = {"what": "round 5 acceptance 4: the host's tables cut from a snapshot (only the needed bytes read)",
           "writer": {"python": sys.version.split()[0], "numpy": np.__version__, "safetensors": safetensors.__version__},
           "snapshot": str(snapshot), "model_safetensors_bytes": path.stat().st_size, "header_bytes": start - 8,
           "header_sha256": meta["header_sha256"], "real_revision_header": real,
           "bytes_read_from_snapshot": bytes_read, "seconds_cut": round(seconds, 3),
           "files": {"readout_table.safetensors": {"keys": {"ids": f"int64 [{len(ids)}]",
                                                            "rows": f"float32 [{len(ids)}, {rows.shape[1]}]"},
                                                   "bytes": files["readout"].stat().st_size,
                                                   "sha256": sha256_file(files["readout"]),
                                                   "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                                   "source": {"tensor": READOUT, **targets[READOUT]}},
                     "vision_position_table.safetensors": {"keys": {"table": f"float32 [{side}, {side}, {c}]"},
                                                           "bytes": files["position"].stat().st_size,
                                                           "sha256": sha256_file(files["position"]),
                                                           "source": {"tensor": POSITION, **targets[POSITION]}}},
           "ranges_read": ranges}
    if verify:
        doc["verify"] = verify_cut(path, ids, rows, table)
    files["manifest"].write_text(json.dumps(doc, indent=1) + "\n")
    return doc


EMBED_FILE, EMBED_KEY = "embed_table.safetensors", "embed_tokens.weight"


class EmbedTable:
    """The host side of the embeds variant: embed_table.safetensors (bfloat16 [V, d], one tensor) memory-mapped; rows(ids)
    = the rows widened to float32 (bit-exact: the bfloat16 bits become the high half of the float32)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        header, start, _ = read_header(self.path)
        e = header[EMBED_KEY]
        assert e["dtype"] == "BF16" and len(e["shape"]) == 2, e
        self.vocab, self.hidden = e["shape"]
        b, epos = e["data_offsets"]
        assert epos - b == self.vocab * self.hidden * 2, e
        self._raw = np.memmap(self.path, dtype="<u2", mode="r", offset=start + b, shape=(self.vocab, self.hidden))

    def rows(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        assert ids.min() >= 0 and ids.max() < self.vocab, (int(ids.min()), int(ids.max()), self.vocab)
        return (np.asarray(self._raw[ids]).astype(np.uint32) << 16).view(np.float32)


def cut_full_embed(snapshot: Path, out: Path, verify: bool) -> dict:
    """--full-embed: the whole tied table in bfloat16, its bytes copied from the snapshot (one seek, block reads)."""
    path = snapshot / "model.safetensors"
    header, start, raw_header = read_header(path)
    ev = json.loads(EVIDENCE.read_text())["header"]
    e = header[READOUT]
    real = e.get("shape") == ev[READOUT]["shape"]
    if real:
        assert e == ev[READOUT], (e, ev[READOUT])
    assert e["dtype"] == "BF16", e
    vocab, d = e["shape"]
    b, epos = e["data_offsets"]
    n = epos - b
    assert n == vocab * d * 2, e
    out.mkdir(parents=True, exist_ok=True)
    dst, man = out / EMBED_FILE, out / "embed_table_manifest.json"
    for p in (dst, man):
        assert not p.exists(), f"refusing to overwrite {p}"
    # safetensors layout: 8-byte little-endian header length + JSON header (padded with spaces to 8 bytes) + data
    hdr = json.dumps({EMBED_KEY: {"dtype": "BF16", "shape": [vocab, d], "data_offsets": [0, n]}},
                     separators=(",", ":")).encode()
    hdr += b" " * (-len(hdr) % 8)
    t0 = time.perf_counter()
    read = 0
    with open(path, "rb") as f, open(dst, "wb") as g:
        g.write(struct.pack("<Q", len(hdr)) + hdr)
        f.seek(start + b)
        while read < n:
            chunk = f.read(min(1 << 24, n - read))
            assert chunk, (read, n)
            g.write(chunk)
            read += len(chunk)
    seconds = time.perf_counter() - t0
    doc = {"what": "round 6c: the whole tied table in bfloat16 for the embeds variant (host lookup, float32 widening)",
           "snapshot": str(snapshot), "header_sha256": hashlib.sha256(raw_header).hexdigest(),
           "real_revision_header": real, "bytes_read_from_snapshot": read, "seconds_cut": round(seconds, 3),
           "range_read": [start + b, start + epos],
           "file": {"name": EMBED_FILE, "keys": {EMBED_KEY: f"bfloat16 [{vocab}, {d}]"}, "bytes": dst.stat().st_size,
                    "sha256": sha256_file(dst), "data_offset_in_file": 8 + len(hdr),
                    "source": {"tensor": READOUT, **e}}}
    tbl = EmbedTable(dst)
    probe = [0, 1, vocab // 2, vocab - 1] + json.loads(CONTRACT.read_text())["readout"]["table"]["ids"][:8]
    doc["self_check"] = {"shape": [tbl.vocab, tbl.hidden], "probe_ids": probe,
                         "probe_rows_finite": bool(np.isfinite(tbl.rows(probe)).all())}
    if verify:
        import torch
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            want = f.get_tensor(READOUT)
        with safe_open(str(dst), framework="pt") as f:
            got = f.get_tensor(EMBED_KEY)
        ids = np.arange(vocab)
        doc["verify"] = {"reader": "safetensors.safe_open(...).get_tensor(...) on the snapshot and on the file",
                         "dtype_file": str(got.dtype), "shape_file": list(got.shape),
                         "file_bit_equal_snapshot": bool(torch.equal(want.view(torch.int16), got.view(torch.int16))),
                         "embedtable_float32_bit_equal": bool(np.array_equal(
                             want.to(torch.float32).numpy().view(np.uint32), tbl.rows(ids).view(np.uint32)))}
    man.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def verify_cut(path: Path, ids: list[int], rows: np.ndarray, table: np.ndarray) -> dict:
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        full = f.get_tensor(READOUT).to(torch.float32)
        pos = f.get_tensor(POSITION).to(torch.float32)
    want_rows = full[torch.tensor(ids)].numpy()
    want_table = pos.numpy().reshape(table.shape)
    return {"reader": "safetensors.safe_open(...).get_tensor(...).to(torch.float32), whole tensor",
            "readout_rows_bit_equal": bool(np.array_equal(want_rows.view(np.uint32), rows.view(np.uint32))),
            "position_table_bit_equal": bool(np.array_equal(want_table.view(np.uint32), table.view(np.uint32)))}


def check_header(snapshot: Path | None) -> dict:
    ev = json.loads(EVIDENCE.read_text())
    hdr, cfg = ev["header"], json.loads(CONFIG.read_text())
    start = 8 + ev["header_bytes"]
    want = {READOUT: [cfg["text_config"]["vocab_size"], cfg["text_config"]["hidden_size"]],
            POSITION: [cfg["vision_config"]["num_patches"], cfg["vision_config"]["hidden_size"]]}
    contract_ids = json.loads(CONTRACT.read_text())["readout"]["table"]["ids"]
    rows = []
    snap_hdr = read_header(snapshot / "model.safetensors")[0] if snapshot else None
    for name, shape in want.items():
        e = hdr[name]
        b, epos = e["data_offsets"]
        r = {"tensor": name, "dtype": e["dtype"], "shape": e["shape"], "config_shape": shape,
             "data_offsets": e["data_offsets"], "file_range": [start + b, start + epos],
             "bytes": epos - b, "numel_x_2": int(np.prod(e["shape"])) * 2,
             "ok": e["dtype"] == "BF16" and e["shape"] == shape and epos - b == int(np.prod(e["shape"])) * 2}
        if name == READOUT:
            row_bytes = shape[1] * 2
            r["readout_rows"] = {"n": len(contract_ids), "row_bytes": row_bytes, "bytes_to_read": len(contract_ids) * row_bytes,
                                 "max_id": max(contract_ids), "max_id_in_range": max(contract_ids) < shape[0],
                                 "first_id_file_range": [start + b + contract_ids[0] * row_bytes,
                                                         start + b + (contract_ids[0] + 1) * row_bytes],
                                 "last_id_file_range": [start + b + contract_ids[-1] * row_bytes,
                                                        start + b + (contract_ids[-1] + 1) * row_bytes]}
            r["ok"] = r["ok"] and r["readout_rows"]["max_id_in_range"]
        else:
            side = int(round(shape[0] ** 0.5))
            r["table_shape"] = [side, side, shape[1]]
            r["ok"] = r["ok"] and side * side == shape[0]
        if snap_hdr is not None:
            r["snapshot_entry_equal"] = snap_hdr.get(name) == e
            r["ok"] = r["ok"] and r["snapshot_entry_equal"]
        rows.append(r)
    return {"what": "round 5 acceptance 4: the two table tensors in the Hub header at the pinned revision",
            "evidence": str(EVIDENCE.relative_to(K)), "url": ev["url"], "header_bytes": ev["header_bytes"],
            "data_start": start, "snapshot": str(snapshot) if snapshot else None, "rows": rows,
            "pass": all(r["ok"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="?", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ids", default="", help="JSON file with an ascending id list (tiny stand-ins); default the contract")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check-header", action="store_true")
    ap.add_argument("--tag", default="", help="--check-header with a snapshot: the output's suffix (default the "
                                              "snapshot dir's name)")
    ap.add_argument("--full-embed", action="store_true", help="write <out>/embed_table.safetensors (the whole tied "
                                                              "table, bfloat16) instead of the two small tables")
    a = ap.parse_args()
    snap = (Path(a.snapshot) if Path(a.snapshot).is_absolute() else K / a.snapshot) if a.snapshot else None
    if a.check_header:
        r = check_header(snap)
        out = K / f"results/tables_header_check{'_' + (a.tag or snap.name) if snap else ''}.json"
        assert not out.exists(), f"refusing to overwrite {out}"
        out.write_text(json.dumps(r, indent=1) + "\n")
        print("| tensor | dtype | shape (header) | shape (config) | data_offsets | file bytes | numel x 2 | ok |")
        print("|---|---|---|---|---|---:|---:|---|")
        for x in r["rows"]:
            print(f"| {x['tensor']} | {x['dtype']} | {x['shape']} | {x['config_shape']} | {x['data_offsets']} | "
                  f"{x['bytes']:,} | {x['numel_x_2']:,} | {x['ok']} |")
        print(json.dumps({"pass": r["pass"], "data_start": r["data_start"],
                          "readout_rows": r["rows"][0]["readout_rows"], "file": str(out.relative_to(K))}))
        if not a.out:
            return 0 if r["pass"] else 1
    assert snap and a.out, "usage: d1_tables.py <snapshot dir> --out <dir>"
    out = Path(a.out) if Path(a.out).is_absolute() else K / a.out
    if a.full_embed:
        doc = cut_full_embed(snap, out, a.verify)
        print(json.dumps({k: doc[k] for k in ("bytes_read_from_snapshot", "seconds_cut", "real_revision_header", "file",
                                              "self_check")}, indent=1))
        if a.verify:
            print(json.dumps(doc["verify"]))
            return 0 if doc["verify"]["file_bit_equal_snapshot"] and doc["verify"]["embedtable_float32_bit_equal"] else 1
        return 0
    ids = json.loads((Path(a.ids) if Path(a.ids).is_absolute() else K / a.ids).read_text()) if a.ids else None
    doc = cut(snap, out, ids, a.verify)
    print(json.dumps({k: doc[k] for k in ("model_safetensors_bytes", "header_bytes", "bytes_read_from_snapshot",
                                          "seconds_cut", "real_revision_header")}, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("keys", "bytes", "sha256")} for k, v in doc["files"].items()}, indent=1))
    if a.verify:
        print(json.dumps(doc["verify"]))
        if not all(v for k, v in doc["verify"].items() if k.endswith("_bit_equal")):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
