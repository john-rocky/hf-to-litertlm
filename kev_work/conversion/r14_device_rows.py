"""Phone gate and phone timing inputs of the published Kev-0.8B files: the token rows the measurement app runs (the
activity in android/measure/), cut from the reference oracle/oracle_0.8b.json (no tokenizer run here). The row format is
device_rows.py's (device_rows.row_of asserts the readout ids).

    python3 scripts/r14_device_rows.py --form C --pairs 128:64 256:64

--form C names the published files (form R64+sp+ec+dd+vs6, file tag r14B-vs6; A and B are two kernel forms that were
compared before C was chosen). Writes into device/r14/ (never overwrites a file with other content):
  rows_L64.json (72 rows: the questions whose rows fit 64 tokens; the control row red_arm_000 does not), rows_L128.json
      (321 questions + the control row = 322 rows), rows_L256.json (385 + 1 = 386), rows_L512.json and rows_L1024.json
      (392 + 1 = 393), rows_L2048.json (401 + 1 = 402; the 9 rows of more than 1,024 tokens and the control row come
      before the others, so a gate stopped at a time limit still holds the rows that only this bucket can take) = the
      gate inputs (mode gate), every row that fits each bucket.
  rows_L2048_gate.json  the 30 rows of the earlier release's 2,048-token phone run (tv4_000 to tv4_019, the 9 long rows
      and the control row), only when device/r6/rows_L2048.json, that run's input, is present: the rows are rebuilt
      from the reference and must equal it. No published number comes from this file.
  timing_rows_r14.json  one timing set per bucket (the app runs the sets whose L equals the graph's L): L64 short64
      (tv4_010/answer, 64 tokens), L128 p50_80 (tv4_007/answer, 80), L256 fiveq (own_fiveq_09's 5 rows of 128 to 142
      tokens, one request = 5 calls), L512 T300 (the first 300 tokens of own_long_log_10/first_failure), L1024 T1000 (its
      first 1,000 tokens), L2048 long1805 (the whole row, 1,805 tokens).
  timing_rows_r14_cpu.json  the CPU sets: L128 p50_80 and L256 fiveq.
  requests_Ls<Ls>_Lq<Lq>_<share|noshare>.json and rows_Ls<Ls>_Lq<Lq>_full.json  the shared-state pair's inputs for the
      app's shared_gate / shared_timing modes: every request whose state fits Ls and whose questions fit Lq, the state
      and each question's own tokens, with GPU constant tensor sharing on or off (the state names come from the pair's
      export record results/export_sharedstate_Ls<Ls>_Lq<Lq>_r14B-vs6.json); the _full file lists the same questions as
      whole rows in row coordinates for r12_device_compare.py.
  inputs_r14_<form>.json  bytes and sha256 of the graphs (the form's row files and pair files under exports/) and of
      every file above."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
from device_rows import PAD_ID, row_of  # noqa: E402

OUT = K / "device/r14"
ORACLE = K / "oracle/oracle_0.8b.json"
TAGS = {"A": {64: "r14A-R64-sp-ec", 128: "r12R64f16safe2", 256: "r12R64f16safe2", 512: "r14A-R64-sp-ec",
              1024: "r14A-R64-sp-ec", 2048: "r14A-R64-sp-ec"},
        "B": {64: "r14B-R64-sp-ec-dd-vs8", 128: "r13R64-sp-ec-dd-vs8", 256: "r14B-R64-sp-ec-dd-vs8",
              512: "r14B-R64-sp-ec-dd-vs8", 1024: "r14B-R64-sp-ec-dd-vs8", 2048: "r14B-R64-sp-ec-dd-vs8"},
        "C": {L: "r14B-vs6" for L in (64, 128, 256, 512, 1024, 2048)}}   # the published files
PAIR_TAG = {"A": "r14A-R64-sp-ec", "B": "r14B-R64-sp-ec-dd-vs8", "C": "r14B-vs6"}
TIMING_REQ = ("own_fiveq_09", "own_ticket_01", "tv4_000")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write(name, doc):
    path = OUT / name
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    if path.exists():
        assert path.read_text() == text, f"{path} exists with other content"
    else:
        path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["A", "B", "C"], required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="Ls:Lq of this form's pair files to prepare")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = json.loads(ORACLE.read_text())
    assert oracle["pad_token_id"] == PAD_ID
    qs = oracle["questions"]
    by = {f"{q['id']}/{q['qid']}": q for q in qs}
    out = {}
    for L, want in ((64, 72), (128, 322), (256, 386), (512, 393), (1024, 393), (2048, 402)):
        fit = [q for q in qs if q["row_len"] <= L]
        if L == 2048:   # long rows (> 1,024 tokens) and the control row first, then the oracle order
            first = [q for q in fit if q["row_len"] > 1024] + [q for q in fit if f"{q['id']}/{q['qid']}" == "red_arm_000/answer"]
            assert len(first) == 10, len(first)
            fit = first + [q for q in fit if q not in first]
        rows = [row_of(q) for q in fit]
        assert len(rows) == want, (L, len(rows))
        out[f"rows_L{L}.json"] = (write(f"rows_L{L}.json", {"L": L, "pad_id": PAD_ID, "rows": rows}), len(rows))
    if (K / "device/r6/rows_L2048.json").exists():   # the earlier release's 30 phone rows: checked only when present
        r6 = json.loads((K / "device/r6/rows_L2048.json").read_text())
        keys = [r["key"] for r in r6["rows"]]
        assert len(keys) == 30 and sum(by[k]["row_len"] > 1024 for k in keys) == 9 and "red_arm_000/answer" in keys
        rows = [row_of(by[k]) for k in keys]
        assert rows == r6["rows"], "the earlier release's L2048 rows differ from the oracle rows"
        out["rows_L2048_gate.json"] = (write("rows_L2048_gate.json", {"L": 2048, "pad_id": PAD_ID, "rows": rows,
                                                                      "source": "device/r6/rows_L2048.json (same 30 rows)"}), 30)
    fiveq = [q for q in qs if q["id"] == "own_fiveq_09"]
    longq = by["own_long_log_10/first_failure"]
    assert [q["row_len"] for q in fiveq] == [132, 142, 132, 128, 128] and longq["row_len"] == 1805

    def one(key, ids):
        return {"key": key, "ids": [int(x) for x in ids]}

    sets = [{"name": "short64", "L": 64, "kind": "single", "rows": [one("tv4_010/answer", by["tv4_010/answer"]["row_ids"])]},
            {"name": "p50_80", "L": 128, "kind": "single", "rows": [one("tv4_007/answer", by["tv4_007/answer"]["row_ids"])]},
            {"name": "fiveq", "L": 256, "kind": "request",
             "rows": [one(f"{q['id']}/{q['qid']}", q["row_ids"]) for q in fiveq]},
            {"name": "T300", "L": 512, "kind": "single",
             "rows": [one("own_long_log_10/first_failure[:300]", longq["row_ids"][:300])]},
            {"name": "T1000", "L": 1024, "kind": "single",
             "rows": [one("own_long_log_10/first_failure[:1000]", longq["row_ids"][:1000])]},
            {"name": "long1805", "L": 2048, "kind": "single",
             "rows": [one("own_long_log_10/first_failure", longq["row_ids"])]}]
    protocol = ("5 warm-up calls, then 20 timed calls (fiveq: 20 requests of 5 calls); ms = write + run + read-back of "
                "the full hidden output, and run only")
    out["timing_rows_r14.json"] = (write("timing_rows_r14.json", {"pad_id": PAD_ID, "protocol": protocol, "sets": sets}),
                                   len(sets))
    out["timing_rows_r14_cpu.json"] = (write("timing_rows_r14_cpu.json", {
        "pad_id": PAD_ID, "protocol": protocol, "sets": [s for s in sets if s["L"] in (128, 256)]}), 2)
    graphs = {}
    for L, tag in TAGS[a.form].items():
        p = K / f"exports/kev08b_rowprefill_L{L}_v2_fp16fc_i8emb_{tag}.tflite"
        graphs[p.name] = {"L": L, "bytes": p.stat().st_size, "sha256": sha256(p)} if p.exists() else None
    reqs = {r["id"]: r for r in oracle["requests"]}
    for pair in a.pairs:
        Ls, Lq = (int(x) for x in pair.split(":"))
        rec = json.loads((K / f"results/export_sharedstate_Ls{Ls}_Lq{Lq}_{PAIR_TAG[a.form]}.json").read_text())
        sig = next(s for s in rec["signatures"] if s["key"] == f"state_prefill_{Ls}")
        names = [o["name"] for o in sig["outputs"]]
        assert len(names) == 48
        by_req, order = {}, []
        for q in qs:
            n = reqs[q["id"]]["state_tokens"]
            if n > Ls or q["row_len"] - n > Lq:
                continue
            if q["id"] not in by_req:
                by_req[q["id"]] = []
                order.append(q["id"])
            by_req[q["id"]].append(q)
        requests, full_rows = [], []
        for rid in order:
            n = reqs[rid]["state_tokens"]
            state = [int(x) for x in by_req[rid][0]["row_ids"][:n]]
            questions = []
            for q in by_req[rid]:
                ids = [int(x) for x in q["row_ids"]]
                assert ids[:n] == state and ids[n] == 248061 and PAD_ID not in ids
                assert ids[q["decide_idx"]] == 248062 and all(ids[i] == 248050 for i in q["opt_idx"])
                key = f"{q['id']}/{q['qid']}"
                questions.append({"key": key, "ids": ids[n:], "decide": q["decide_idx"] - n,
                                  "opts": [o - n for o in q["opt_idx"]], "row_len": q["row_len"]})
                full_rows.append({"key": key, "ids": ids, "decide": int(q["decide_idx"]),
                                  "opts": [int(o) for o in q["opt_idx"]]})
            requests.append({"request": rid, "state": state, "questions": questions})
        assert all(t in order for t in TIMING_REQ)
        for share in (True, False):
            name = f"requests_Ls{Ls}_Lq{Lq}_{'share' if share else 'noshare'}.json"
            out[name] = (write(name, {"Ls": Ls, "Lq": Lq, "pad_id": PAD_ID, "gpu_constant_tensor_sharing": share,
                                      "state_names": names, "timing_requests": list(TIMING_REQ),
                                      "requests": requests}), len(requests))
        name = f"rows_Ls{Ls}_Lq{Lq}_full.json"
        out[name] = (write(name, {"L": Ls, "pad_id": PAD_ID, "Lq": Lq, "rows": full_rows,
                                  "note": "device_compare rows format: full rows, row coordinates"}), len(full_rows))
        p = K / f"exports/kev08b_sharedstate_Ls{Ls}_Lq{Lq}_v2_fp16fc_i8emb_{PAIR_TAG[a.form]}.tflite"
        graphs[p.name] = {"Ls": Ls, "Lq": Lq, "bytes": p.stat().st_size, "sha256": sha256(p)}
    doc = {"step": f"S26 inputs, form {a.form} (scripts/r14_device_rows.py)",
           "oracle": {"path": str(ORACLE.relative_to(K)), "sha256": sha256(ORACLE)}, "graphs": graphs,
           "files": {name: {"rows_or_sets_or_requests": n, "bytes": p.stat().st_size, "sha256": sha256(p)}
                     for name, (p, n) in out.items()}}
    ip = OUT / f"inputs_r14_{a.form}.json"
    ip.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: (v if k != "files" else {n: f["rows_or_sets_or_requests"] for n, f in v.items()})
                      for k, v in doc.items()}, indent=1))


if __name__ == "__main__":
    main()
