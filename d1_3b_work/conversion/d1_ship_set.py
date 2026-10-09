"""Round 10 design 6: the shipped file set of d1-3B LiteRT, every file hashed again, with its role, its Mac
gate and its timing source -> results/ship_set.json (never overwritten).

    $EXPORT scripts/d1_ship_set.py

Ship = the text path unified on the embeds form (round 10): the embeds row graphs v2e (fp16 FC, no table in the graph)
of host/contract.json `embeds_graph.buckets`, the embeds shared-state pairs v2e of `shared_state.files`, the picture
graphs (tower v2 + projector v2, `vision.real_files`), the three host tables (`readout_table`, `embeds_graph.table`, the
vision position table), the repository's tokenizer.json, the host's three python files, host/contract.json itself and
the fixtures (the requests, their rows, the red arms, the token probes, their README and licence).
Every file is read and hashed here (sha256 of the bytes on disk) and compared with the record that made it (the export
/ quant json, the contract entry, the table manifest); a mismatch stops the script before anything is written. Per
file: path, bytes, sha256, role, the gate (per accelerator the check json, its verdict, max / mean |dp|, argmax and red
arms; the picture graphs: the end-to-end checks of round 8), the timing rows of results/timing_mac_r10.json (and the
round 8 picture timing) that time it. Totals: files, bytes (all, the graphs, the rest).
not_shipped: the ids forms (v2 = fp16 FC + an int8 table inside the graph: the row graphs L256..L4096 of `graphs` and
round 9's pairs) and the other files kept on disk, each with the reason; their numbers stay in the work
directory's records (the published conversion notes carry the numbers only).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
OUT = K / "results/ship_set.json"
CONTRACT = K / "host/contract.json"
TIMING = K / "results/timing_mac_r10.json"
TIMING_IMAGE = K / "results/timing_mac_image.json"
HOST_PY = ("host/d1_litert.py", "host/d1_shared_state.py", "host/d1_vision.py")
FIXTURES = ("fixtures/requests.json", "fixtures/rows.json", "fixtures/red_arms.json", "fixtures/token_probes.json",
            "fixtures/README.md", "fixtures/LICENSE-SemIf-MIT.txt")
TOKENIZER = "hf_small/tokenizer.json"
TABLE_DIR = "cache/real/tables"
ACCELS = ("cpu", "gpu_f32", "gpu_default")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def rec(path: str, role: str, record=None) -> dict:
    """A file entry; record = (bytes, sha256, where the record is) to compare with, or None (hashed only)."""
    p = K / path
    size, sha = p.stat().st_size, sha256(p)
    out = {"path": path, "bytes": size, "sha256": sha, "role": role}
    if record is not None:
        want_bytes, want_sha, src = record
        assert (size, sha) == (want_bytes, want_sha), f"{path} differs from {src}: {(size, sha)} vs {record[:2]}"
        out["record"] = src
    return out


def gate_of(stem: str, kind: str = "row") -> dict:
    """The Mac gate of a graph file (results/<stem>_<accel>_check.json, d1_check.py or d1_check_pair.py)."""
    g = {}
    for acc in ACCELS:
        p = K / f"results/{stem}_{acc}_check.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s = d["reference_parity"]["summary"] if kind == "row" else d["summary"]
        line = s["line"]
        g[acc] = {"file": str(p.relative_to(K)), "verdict": "PASS" if s["stop_bar_pass"] else "FAIL",
                  "questions": s["questions_run"], "max_abs_dp": s["max_abs_dp"],
                  "mean_abs_dp": s["mean_abs_dp_all_options"], "non_near_tie_argmax": s["non_near_tie_argmax"],
                  "near_tie_argmax": s["near_tie_argmax"], "red_arms": line["red_arms"],
                  "fully_accelerated": d.get("is_fully_accelerated")}
        if kind == "pair":
            g[acc]["handover_equal"] = s["handover_equal"]
    return g


def timing_of(file: str, timing: dict) -> list:
    return [{"set": r["set"], "form": r["form"], "accel": r["accel"], "handover": r.get("handover"),
             "median_ms": r["median_ms"], "n": r["n"], "label": r["label"]}
            for r in timing["rows"] if r["file"] == file]


def main() -> int:
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    t0 = time.time()
    contract = json.loads(CONTRACT.read_text())
    timing = json.loads(TIMING.read_text())
    timing_image = json.loads(TIMING_IMAGE.read_text())
    files = []
    # 1. the embeds row graphs
    for b in contract["embeds_graph"]["buckets"]:
        stem = Path(b["file"]).stem
        q = json.loads((K / b["record"]).read_text())["output"]
        assert (q["bytes"], q["sha256"]) == (b["bytes"], b["sha256"]), b["file"]
        e = rec(f"exports/{b['file']}", f"row graph, embeds form v2e (fp16 FC, the table on the host), rows of up to "
                                         f"{b['L']} tokens: embeds float32 [1, {b['L']}, 2048] + valid -> hidden",
                (q["bytes"], q["sha256"], b["record"]))
        e.update(kind="row_graph", L=b["L"], gate_mac=gate_of(stem), timing=timing_of(f"exports/{b['file']}", timing))
        files.append(e)
    # 2. the embeds shared-state pairs
    for f in contract["shared_state"]["files"]:
        stem = Path(f["file"]).stem
        q = json.loads((K / f"results/{stem}_quant.json").read_text())["output"]
        assert (q["bytes"], q["sha256"]) == (f["bytes"], f["sha256"]), f["file"]
        e = rec(f"exports/{f['file']}", f"shared-state pair, embeds form v2e: state_prefill_{f['Ls']} once per request "
                                         f"+ question_step_{f['Ls']}_{f['Lq']} per question (one file, two signatures)",
                (q["bytes"], q["sha256"], f"results/{stem}_quant.json"))
        e.update(kind="pair", Ls=f["Ls"], Lq=f["Lq"], gate_mac=gate_of(stem, "pair"),
                 timing=timing_of(f"exports/{f['file']}", timing))
        files.append(e)
    # 3. the picture graphs (round 6d / 8)
    vr = contract["vision"]["real_files"]
    real = {vr[g]["v2_fp16fc"]["file"]: vr[g]["v2_fp16fc"] for g in ("tower", "projector")}
    for name, role in (("real_vision_tower_v2_fp16fc.tflite", "picture tower v2 (fp16 FC), one call per tile"),
                       ("real_projector_v2_fp16fc.tflite", "picture projector v2 (fp16 FC), one call per tile")):
        x = real[name]
        e = rec(f"exports/{name}", role, (x["bytes"], x["sha256"], "host/contract.json vision.real_files"))
        e2e = {"cpu": "results/real_vision_e2e_ship_cpu.json", "gpu_f32": "results/real_vision_e2e_ship_gpu.json",
               "gpu_f32_round10_default_host": "results/real_vision_e2e_r10_ship_gpu.json"}
        e.update(kind="picture_graph",
                 gate_mac={k: {"file": p, "pass": json.loads((K / p).read_text())["summary"]["pass"],
                               "max_abs_dp": json.loads((K / p).read_text())["summary"]["max_abs_dp"],
                               "mean_abs_dp": json.loads((K / p).read_text())["summary"]["mean_abs_dp_all_options"]}
                           for k, p in e2e.items()},
                 timing=[{"set": r["set"], "accel": r["accel"], "total_median_ms": r["ms"]["total"]["median"],
                          "tower_median_ms": r["ms"]["tower"]["median"], "tiles": r["tiles"],
                          "source": str(TIMING_IMAGE.relative_to(K))} for r in timing_image.get("rows", [])])
        files.append(e)
    # 4. the host tables
    tables = {"readout_table.safetensors": "read-out table: float32 rows of the tied embedding at the ids a read-out "
                                           "uses (1,234 rows)",
              "embed_table.safetensors": "embed table: the whole tied embedding in bfloat16 [128000, 2048]; the host "
                                         "writes its rows as float32 into every embeds graph",
              "vision_position_table.safetensors": "the picture tower's position table, float32 [16, 16, 1152]"}
    t_rec = {name: (x["bytes"], x["sha256"], f"{TABLE_DIR}/tables_manifest.json")
             for name, x in json.loads((K / TABLE_DIR / "tables_manifest.json").read_text())["files"].items()}
    em = json.loads((K / TABLE_DIR / "embed_table_manifest.json").read_text())["file"]
    t_rec[em["name"]] = (em["bytes"], em["sha256"], f"{TABLE_DIR}/embed_table_manifest.json")
    for name, role in tables.items():
        e = rec(f"{TABLE_DIR}/{name}", role, t_rec.get(name))
        e["kind"] = "table"
        files.append(e)
    # 5. tokenizer, host, contract, fixtures
    tk = contract["tokenizer"]
    e = rec(TOKENIZER, "the repository's tokenizer.json (tokenizers.Tokenizer.from_file), unchanged")
    assert e["sha256"] == tk["sha256"], "hf_small/tokenizer.json differs from the contract's tokenizer.sha256"
    e["record"] = "host/contract.json tokenizer.sha256"
    e["kind"] = "tokenizer"
    files.append(e)
    for path, role in zip(HOST_PY, ("the host: request -> rows -> row graph / embeds graph -> read-out -> answer",
                                    "the shared-state host: the pairs and the pick by measured call time",
                                    "the picture path: preprocessing, tower and projector calls, the insertion")):
        e = rec(path, role)
        e["kind"] = "host"
        files.append(e)
    e = rec("host/contract.json", "the contract: graphs, signatures, buckets, pairs, pick, tables, gates")
    e["kind"] = "contract"
    files.append(e)
    rows_sha = contract["buckets"]["rows_json_sha256"]
    for path in FIXTURES:
        e = rec(path, "test fixtures: " + {"fixtures/requests.json": "the requests (fixture records)",
                                           "fixtures/rows.json": "their rows (the provider's render and ids)",
                                           "fixtures/red_arms.json": "the four red arms",
                                           "fixtures/token_probes.json": "tokenizer probes",
                                           "fixtures/README.md": "what the fixtures are and where they come from",
                                           "fixtures/LICENSE-SemIf-MIT.txt": "the SemIf licence"}[path])
        if path == "fixtures/rows.json":
            assert e["sha256"] == rows_sha, "fixtures/rows.json differs from the contract's rows_json_sha256"
            e["record"] = "host/contract.json buckets.rows_json_sha256"
        e["kind"] = "fixture"
        files.append(e)
    graphs = [x for x in files if x["kind"] in ("row_graph", "pair", "picture_graph")]
    # not shipped
    not_shipped = []
    for g in contract["graphs"]:
        p = K / "exports" / g["file"]
        not_shipped.append({"path": f"exports/{g['file']}", "bytes": p.stat().st_size if p.exists() else None,
                            "why": "the ids form (v2 = fp16 FC + an int8 table inside the graph): the text path is "
                                   "unified on the embeds form in round 10 (near-exact: the int8 table's max |dp| "
                                   "7.1e-3 is gone, 264 MB smaller, as fast); kept as the ladder's evidence"})
    for f in contract["shared_state"].get("files_ids_not_shipped", []):
        p = K / "exports" / f["file"]
        not_shipped.append({"path": f"exports/{f['file']}", "bytes": p.stat().st_size if p.exists() else None,
                            "why": "round 9's ids pair (v2): the embeds pair of the same shape replaces it (round 10); "
                                   "kept as the ladder's evidence"})
    for path, why in (("exports/real_rowprefill_L256_v1_wi8fc.tflite",
                       "dynamic int8 FC: fails the bar (CPU max |dp| 0.245); the quant lever's evidence"),
                      ("exports/real_vision_tower_fp32.tflite", "the float32 tower: the picture path's reference form"),
                      ("exports/real_projector_fp32.tflite", "the float32 projector: the reference form")):
        p = K / path
        if p.exists():
            not_shipped.append({"path": path, "bytes": p.stat().st_size, "why": why})
    doc = {"what": "round 10: the shipped file set of LiquidAI/d1-3B on LiteRT (Mac / desktop), every file re-hashed "
                   "(scripts/d1_ship_set.py)", "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "contract_sha256": sha256(CONTRACT), "timing": str(TIMING.relative_to(K)),
           "totals": {"files": len(files), "bytes": sum(x["bytes"] for x in files),
                      "graph_files": len(graphs), "graph_bytes": sum(x["bytes"] for x in graphs),
                      "other_bytes": sum(x["bytes"] for x in files if x not in graphs)},
           "files": files, "not_shipped": not_shipped,
           "not_shipped_bytes": sum(x["bytes"] or 0 for x in not_shipped), "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"totals": doc["totals"], "files": [(x["path"], x["bytes"], x["sha256"][:12]) for x in files],
                      "not_shipped": [(x["path"], x["bytes"]) for x in not_shipped]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
