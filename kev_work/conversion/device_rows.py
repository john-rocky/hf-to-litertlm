"""Phone gate inputs: the token rows the debug app runs, cut from the 0.8B reference (no tokenizer run here).

    python3 scripts/device_rows.py

Writes (never overwrites):
  device/rows_L512.json, device/rows_L1024.json   every reference question whose row fits L, in reference file order
      (392 questions + red_arm_000 = 393 rows for both L; the 9 long own rows are > 1,024 tokens):
      {"L": L, "pad_id": 248044, "rows": [{"key": "<id>/<qid>", "ids": [...], "decide": d, "opts": [...]}]}
      The app pads ids with pad_id and builds valid (1.0 real / 0.0 pad) itself.
  device/rows_L512_cpu.json   CPU subset: the first 120 rows + the near-tie rows not among them
      (results/oracle_summary.json near_tie.ids) + red_arm_000, file order.
  device/rows_L512_tap.json   default-precision subset: the first 40 rows + the 18 rows whose hidden was all non-finite
      under fp16 activations on Mac Metal (results/litert_gpu_f16_parity_L1024_v2_fp16fc_i8emb.json), file order.
  device/timing_rows.json     the Mac timing rows (results/timing_mac_0.8b.json of timing_mac.py, jobs.card.rows):
      fiveq = own_fiveq_09's 5 question rows (L=512 file, one request = 5 calls),
      T300 = the first 300 tokens of own_long_log_10/first_failure's row (L=512 file, synthetic),
      T1000 = the first 1,000 tokens of the same row (L=1024 file, synthetic).
  device/rows_manifest.json   bytes + sha256 of every file above and of the sources they were cut from."""
import hashlib
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
OUT = K / "device"
ORACLE = K / "oracle/oracle_0.8b.json"
ORACLE_SUMMARY = K / "results/oracle_summary.json"
F16_PARITY = K / "results/litert_gpu_f16_parity_L1024_v2_fp16fc_i8emb.json"
TIMING_MAC = K / "results/timing_mac_0.8b.json"   # timing_mac.py --job 0.8b:card
PAD_ID = 248044
CPU_HEAD, TAP_HEAD = 120, 40


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write(name, doc):
    path = OUT / name
    assert not path.exists(), f"refusing to overwrite {path}"
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return path


def row_of(q):
    ids = [int(x) for x in q["row_ids"]]
    assert len(ids) == q["row_len"] and q["decide_idx"] == len(ids) - 1
    assert ids[q["decide_idx"]] == 248062 and all(ids[i] == 248050 for i in q["opt_idx"])
    assert PAD_ID not in ids, f"{q['id']}: pad id inside the row"
    return {"key": f"{q['id']}/{q['qid']}", "ids": ids, "decide": int(q["decide_idx"]),
            "opts": [int(i) for i in q["opt_idx"]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = json.loads(ORACLE.read_text())
    assert oracle["pad_token_id"] == PAD_ID
    questions = oracle["questions"]
    written = {}
    for L in (512, 1024):
        rows = [row_of(q) for q in questions if q["row_len"] <= L]
        assert sum(r["key"].startswith("red_arm_000/") for r in rows) == 1
        written[f"rows_L{L}.json"] = (write(f"rows_L{L}.json", {"L": L, "pad_id": PAD_ID, "rows": rows}), len(rows))
    rows512 = [row_of(q) for q in questions if q["row_len"] <= 512]
    keys = [r["key"] for r in rows512]
    near = json.loads(ORACLE_SUMMARY.read_text())["near_tie"]["ids"]
    assert len(near) == 15 and all(k in keys for k in near)
    cpu_keys = set(keys[:CPU_HEAD]) | set(near) | {k for k in keys if k.startswith("red_arm_000/")}
    cpu_rows = [r for r in rows512 if r["key"] in cpu_keys]
    written["rows_L512_cpu.json"] = (write("rows_L512_cpu.json", {
        "L": 512, "pad_id": PAD_ID, "subset": f"first {CPU_HEAD} rows + near-tie rows not among them + red_arm_000",
        "near_tie_keys": near, "rows": cpu_rows}), len(cpu_rows))
    nan_keys = [r["key"] for r in json.loads(F16_PARITY.read_text())["nonfinite"]["questions_with_nonfinite_h_sel"]]
    assert len(nan_keys) == 18 and all(k in keys for k in nan_keys)
    tap_keys = set(keys[:TAP_HEAD]) | set(nan_keys)
    tap_rows = [r for r in rows512 if r["key"] in tap_keys]
    written["rows_L512_tap.json"] = (write("rows_L512_tap.json", {
        "L": 512, "pad_id": PAD_ID,
        "subset": f"first {TAP_HEAD} rows + the 18 rows that were all non-finite under fp16 activations on Mac Metal",
        "mac_f16_nonfinite_keys": nan_keys, "rows": tap_rows}), len(tap_rows))
    mac = json.loads(TIMING_MAC.read_text())["jobs"]["card"]["rows"]
    fiveq = [q for q in questions if q["id"] == mac["fiveq"]["id"]]
    assert [q["row_len"] for q in fiveq] == mac["fiveq"]["row_lens"]
    src_id, src_qid = mac["T300"]["from"].split("/")
    longq = next(q for q in questions if q["id"] == src_id and q["qid"] == src_qid)
    assert mac["T1000"]["from"] == mac["T300"]["from"] and longq["row_len"] >= mac["T1000"]["tokens"]
    sets = [
        {"name": "fiveq", "L": mac["fiveq"]["file_L"], "kind": "request",
         "rows": [{"key": f"{q['id']}/{q['qid']}", "ids": [int(x) for x in q["row_ids"]]} for q in fiveq]},
        {"name": "T300", "L": mac["T300"]["file_L"], "kind": "single", "synthetic": True,
         "rows": [{"key": f"{mac['T300']['from']}[:{mac['T300']['tokens']}]",
                   "ids": [int(x) for x in longq["row_ids"][: mac["T300"]["tokens"]]]}]},
        {"name": "T1000", "L": mac["T1000"]["file_L"], "kind": "single", "synthetic": True,
         "rows": [{"key": f"{mac['T1000']['from']}[:{mac['T1000']['tokens']}]",
                   "ids": [int(x) for x in longq["row_ids"][: mac["T1000"]["tokens"]]]}]},
    ]
    written["timing_rows.json"] = (write("timing_rows.json", {
        "pad_id": PAD_ID, "protocol": "5 warm-up calls, then 20 timed calls (fiveq: 20 requests of 5 calls); "
                                      "ms = write + run + read-back of the full hidden output, and run only",
        "source": "results/timing_mac_0.8b.json jobs.card.rows (the rows of the Mac timing)", "sets": sets}), 7)
    manifest = {"sources": {str(p.relative_to(K)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in (ORACLE, ORACLE_SUMMARY, F16_PARITY, TIMING_MAC)},
                "files": {name: {"rows": n, "bytes": p.stat().st_size, "sha256": sha256(p)}
                          for name, (p, n) in written.items()}}
    mpath = OUT / "rows_manifest.json"
    assert not mpath.exists()
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
