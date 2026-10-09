"""Round 8: the picture path's entries of host/contract.json on the real weights; every other section stays as it is.

    $EXPORT scripts/d1_image_contract.py [--rehash] [--write] \
        [--ledger results/real_files.json] [--diff-out cache/real/image/contract_diff.txt]

Entries written (read from the files, never typed in):
  embeds_graph.buckets      the embeds row graphs (v2e_fp16fc) of every bucket present: L, file, bytes and sha256 from
                            the quantization record (--rehash: the file is hashed again and compared), the I/O from the
                            float32 export's record, the Mac gate per accelerator from the chain's check jsons
                            (d1_contract.gate_line). The top-level fields stay the L256 entry of round 6c.
  vision.verified.real_e2e  the end-to-end runs of host/test_d1_vision_real.py (results/real_vision_e2e_*.json): per run
                            the files, the accelerator, the exact checks, max / mean |dp|, argmax, the bar applied and
                            its verdict; the reference (results/real_vision_e2e_ref.json) and its npz sha256.
  vision.status             its clause "the picture path end to end on the real weights is not run yet (...)" replaced
                            by the round-8 result (the same entry).
Before any change the file is re-serialised as d1_contract.py writes it (json.dumps(indent=1, ensure_ascii=False) +
"\n") and must give the same bytes. After the change every top-level key but `embeds_graph` and `vision`, and every
`vision` key but `verified` and `status`, must equal the original, and `verified`'s other keys too; the unified diff's
hunks are printed (and written to --diff-out). --ledger appends the new v2e files to the ledger's `candidates` (the
entries already there stay as they are). Without --write: a dry run.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
sys.path.insert(0, str(K / "scripts"))
from d1_contract import ACCELS, gate_line, io_of, shipped_file  # noqa: E402

CONTRACT = K / "host/contract.json"
BUCKETS = (256, 512, 1024, 2048, 4096)
FORM = "v2e_fp16fc (fp16 FC, no table in the graph)"
GATE_KEYS = ("verdict", "questions", "max_abs_dp", "mean_abs_dp", "non_near_tie_argmax", "near_tie_argmax", "red_arms",
             "check")
STALE = "; the picture path end to end on the real weights is not run yet (needs the real embeds row graph)"


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def embeds_buckets(rehash: bool) -> list:
    out = []
    for L in BUCKETS:
        stem = f"real_rowprefill_embeds_L{L}_v2e_fp16fc"
        f = shipped_file(stem, rehash)
        if f is None:
            continue
        checks = {acc: K / f"results/{stem}_{acc}_check.json" for acc in ACCELS}
        io = io_of(K / f"results/real_export_embeds_L{L}.json", checks["gpu_f32"])
        gates = {acc: gate_line(p) for acc, p in checks.items() if p.exists()}
        out.append({"L": L, "file": f["file"], "bytes": f["bytes"], "sha256": f["sha256"], "form": FORM,
                    "inputs": io["inputs"], "outputs": io["outputs"], "record": f["record"],
                    "sha256_rehashed": f["sha256_rehashed"],
                    "gate": {acc: {k: x[k] for k in GATE_KEYS} for acc, x in gates.items()}})
    return out


def e2e_runs() -> dict:
    ref = json.loads((K / "results/real_vision_e2e_ref.json").read_text())
    runs = []
    for p in sorted((K / "results").glob("real_vision_e2e_*.json")):
        if p.name in ("real_vision_e2e_ref.json", "real_vision_e2e_table.json"):
            continue
        d = json.loads(p.read_text())
        s = d["summary"]
        runs.append({"run": str(p.relative_to(K)), "files": d["files"], "accel": d["accel"], "precision": d["precision"],
                     "requests": [r["name"] for r in d["rows"]], "L": {r["name"]: r["L"] for r in d["rows"]},
                     "exact_checks_pass": s["exact_checks_pass"],
                     "picture_tokens_vs_provider_rel_max": s["picture_tokens_vs_provider_rel_max"],
                     "hidden_slot_vs_provider_max": s["hidden_slot_vs_provider_max"], "max_abs_dp": s["max_abs_dp"],
                     "mean_abs_dp_all_options": s["mean_abs_dp_all_options"],
                     "argmax_equal_non_near_tie": s["argmax_equal_non_near_tie"], "nonfinite": s["nonfinite"],
                     "bar": s["bar_applied"], "pass": s["pass"]})
    return {"what": "the picture path end to end on the real weights: the host (host/d1_litert.py D1Host, item 9) against "
                    "the provider's float32 CPU plain pass, three requests of one question (one tile, 2 x 3 tiles + "
                    "thumbnail, two pictures)",
            "test": "host/test_d1_vision_real.py", "reference": "results/real_vision_e2e_ref.json",
            "reference_npz_sha256": ref["npz_sha256"], "files": {"fp32": "the float32 picture graphs + the float32 embeds "
                                                                         "row graph (path check, bar 1e-4)",
                                                                 "ship": "tower v2 + projector v2 + v2e (the FACTS section 7 bar)",
                                                                 "mixed": "the float32 picture graphs + v2e (recorded)",
                                                                 "torch": "the torch export forms (path check, bar 1e-4)"},
            "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehash", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--diff-out", default="")
    a = ap.parse_args()
    old_text = CONTRACT.read_text()
    old = json.loads(old_text)
    assert dump(old) == old_text, "host/contract.json does not round-trip through d1_contract.py's serialisation"
    new = json.loads(old_text)
    buckets = embeds_buckets(a.rehash)
    assert any(b["L"] == 256 for b in buckets) and buckets, [b["L"] for b in buckets]
    l256 = next(b for b in buckets if b["L"] == 256)
    assert (l256["file"], l256["sha256"]) == (old["embeds_graph"]["file"], old["embeds_graph"]["sha256"])
    new["embeds_graph"]["buckets"] = buckets
    new["embeds_graph"]["buckets_note"] = ("round 8: one embeds row graph per bucket present (the host takes the smallest "
                                           "that holds the row, pick_L); the top-level fields are the L256 entry of round 6c")
    new["vision"]["verified"]["real_e2e"] = e2e_runs()
    status = new["vision"]["status"]
    assert STALE in status, status
    new["vision"]["status"] = status.replace(STALE, "; round 8: the picture path end to end on the real weights through "
                                                    "the host (D1Host) on the Mac CPU and Metal (`verified.real_e2e`)")
    for k in old:
        if k not in ("embeds_graph", "vision"):
            assert new[k] == old[k], k
    for k in old["vision"]:
        if k not in ("verified", "status"):
            assert new["vision"][k] == old["vision"][k], k
    for k in old["vision"]["verified"]:
        assert new["vision"]["verified"][k] == old["vision"]["verified"][k], k
    for k in old["embeds_graph"]:
        assert new["embeds_graph"][k] == old["embeds_graph"][k], k
    new_text = dump(new)
    diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), "a/host/contract.json",
                                     "b/host/contract.json", n=0, lineterm=""))
    hunks = [ln for ln in diff if ln.startswith("@@")]
    summary = {"buckets": [b["L"] for b in buckets], "e2e_runs": len(new["vision"]["verified"]["real_e2e"]["runs"]),
               "hunks": hunks, "old_bytes": len(old_text.encode()), "new_bytes": len(new_text.encode()), "write": a.write}
    print(json.dumps(summary, indent=1))
    if a.diff_out:
        (K / a.diff_out).write_text("\n".join(diff) + "\n")
    if a.write:
        CONTRACT.write_text(new_text)
    if a.ledger:
        path = K / a.ledger
        led_text = path.read_text()
        led = json.loads(led_text)
        have = {c["file"] for c in led["candidates"]}
        added = []
        for b in buckets:
            f = f"exports/{b['file']}"
            if f in have:
                continue
            added.append({"file": f, "bytes": b["bytes"], "sha256": b["sha256"], "bucket": b["L"], "form": FORM,
                          "gate": {acc: x["verdict"] for acc, x in b["gate"].items()}, "gate_detail": b["gate"],
                          "record": b["record"]})
        led["candidates"] += added
        r8_del = [d["file"] + " (" + d["deleted_at"] + ")" for d in led.get("deleted", []) if "_embeds_L" in d["file"] and d["file"].endswith("_fp32.tflite") and not d["file"].endswith("L256_fp32.tflite")]
        led.setdefault("appended", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "by": "round 8 "
                                               "(scripts/d1_image_contract.py --ledger)", "candidates": [x["file"] for x in added],
                                               "note": "the deletions " + "; ".join(r8_del) + " are round 8's: their `why` "
                                                       "is the fixed text of scripts/d1_export.py --retire, which names round 6c"})
        print(json.dumps({"ledger": a.ledger, "candidates_added": [x["file"] for x in added]}, indent=1))
        if a.write:
            path.write_text(json.dumps(led, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
