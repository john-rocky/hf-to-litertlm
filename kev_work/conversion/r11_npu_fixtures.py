"""Inputs of the NPU runner (android/measure/kev_npu_runner.cc) for the 128- and 256-token row graphs, from the oracle rows
that fit L.

    python r11_npu_fixtures.py --L 128          (and --L 256)

Writes (never overwrites):
  device/r11/rows_L{L}.json        = the gate app's rows-file shape {L, pad_id, rows: [{key, ids, decide, opts}]}, oracle
                                     order, every question whose row fits L (red_arm_000 included when it fits)
  npu/fixtures_L{L}/<id3>_ids.f32   = int32 [1, L] little-endian bytes (the runner copies every input byte for byte; the
                                     .f32 name is the runner's fixture naming, the bytes are int32), right-padded 248044
  npu/fixtures_L{L}/<id3>_valid.f32 = float32 [1, L], 1.0 real / 0.0 pad
  npu/fixtures_L{L}/<id3>_sel.i32   = int32 [decide, *opts] (the read-out rows, in the oracle's order)
  npu/fixtures_L{L}/<id3>_n.i32     = int32 [row_len]
  npu/fixtures_L{L}/manifest.json   = id3 -> key, row_len, k (+ sha256 of the rows file)"""
import argparse
import json

import numpy as np

from r2_common import K, PAD_ID, dump_json, load_oracle, qkey, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    a = ap.parse_args()
    L = a.L
    oracle = load_oracle()
    fits = [q for q in oracle["questions"] if q["row_len"] <= L]
    rows_path = K / f"device/r11/rows_L{L}.json"
    fdir = K / f"npu/fixtures_L{L}"
    assert not rows_path.exists() and not fdir.exists(), "refusing to overwrite"
    fdir.mkdir(parents=True)
    rows = [{"key": qkey(q), "ids": q["row_ids"], "decide": q["decide_idx"], "opts": q["opt_idx"]} for q in fits]
    dump_json(rows_path, {"L": L, "pad_id": PAD_ID, "rows": rows})
    manifest = []
    for i, q in enumerate(fits):
        id3 = f"{i:03d}"
        n = q["row_len"]
        ids = np.full((1, L), PAD_ID, dtype="<i4")
        ids[0, :n] = np.asarray(q["row_ids"], dtype="<i4")
        valid = np.zeros((1, L), dtype="<f4")
        valid[0, :n] = 1.0
        sel = np.asarray([q["decide_idx"], *q["opt_idx"]], dtype="<i4")
        assert sel.max() < n
        ids.tofile(fdir / f"{id3}_ids.f32")
        valid.tofile(fdir / f"{id3}_valid.f32")
        sel.tofile(fdir / f"{id3}_sel.i32")
        np.asarray([n], dtype="<i4").tofile(fdir / f"{id3}_n.i32")
        manifest.append({"id3": id3, "key": qkey(q), "row_len": n, "k": len(q["opt_idx"])})
    assert len(fits) <= 1000
    dump_json(fdir / "manifest.json", {"L": L, "rows_file": str(rows_path.relative_to(K)),
                                       "rows_sha256": sha256_file(rows_path), "count": len(fits), "rows": manifest})
    print(json.dumps({"L": L, "count": len(fits), "rows_file": str(rows_path.relative_to(K)),
                      "red_arm_included": any(m["key"].startswith("red_arm_000/") for m in manifest)}))


if __name__ == "__main__":
    main()
