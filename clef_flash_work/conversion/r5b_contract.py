#!/usr/bin/env python3
"""Round 5b: host/contract.json (round 5a's contract plus the folder: graph files, signatures, tables, accelerators,
what was checked end to end), written from the records (no number typed in by hand), and its copy in staging/upload.

    venv-host/bin/python scripts/r5b_contract.py      -> host/contract.json, staging/upload/contract.json

Reads: host/contract.json (round 5a's, kept as the base: its sections stay as they are unless named here),
host/clef_litert (FOLDER, version), results/r5b_stage.json (the staged files and their recorded sha256),
results/r5b_host_test.json (the host test on this package), results/r5b_e2e_*_part*.json (the graphs' signatures
and delegation as compiled), results/r5b_score_{gpu,cpu}.json, results/r5b_example_{gpu,cpu}.json."""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
UP = K / "staging/upload"
sys.path.insert(0, str(K / "host"))
import clef_litert as C  # noqa: E402


def load(p: str):
    f = K / p
    return json.loads(f.read_text()) if f.exists() else None


def public(doc: dict) -> dict:
    """The published copy: no paths of this machine, no round names; the checks named by what they compared."""
    import copy

    p = copy.deepcopy({k: v for k, v in doc.items() if k not in ("round_5a", "verified", "status")})
    p["status"] = "the graphs, the host package and the folder of this repository"
    for f in p["files"].values():
        if "made_by" in f:
            f["made_by"] = "conversion/r5a_tables.py (clef_litert.copy_bf16_tensor): the checkpoint's bytes, new name"
    p["row_graph"].pop("recorded_signature_r1", None)
    p["picture"]["tower_graph"] = ("patches [1, 4 G^2, 1536] and the position rows pos [1, 4 G^2, 1152] float32 in -> "
                                   "picture rows [1, G^2, 4096] float32 out (tower_graphs)")
    v = doc["verified"]
    p["checks"] = {
        "reference": "the provider's own code (joint_schema_model.py systemone() at the revision above) in float32 "
                     "on the CPU: 243 runs of fixtures/ (text, and pictures at the 256 / 448 px tiles), "
                     "fixtures/reference_probs.json",
        "line": "on every question that is not a near-tie (top-2 margin <= 0.02) the most likely option is the "
                "reference's; every probability within 0.02; the mean over runs of each run's mean |dp| <= 0.002",
        "end_to_end": {acc: ({k: e[k] for k in ("complete", "line", "red_arms", "pass")} if e else None)
                       for acc, e in v["e2e"].items()},
        "example": {acc: (e["pass"] if e else None) for acc, e in v["example"].items()},
        "delegation_as_compiled": {k: {kk: d[kk] for kk in ("compile_seconds", "fully_accelerated",
                                                             "one_partition_all_nodes", "replacing")}
                                   for k, d in v["delegation_as_compiled"].items()},
    }
    return p


def main() -> int:
    base = json.loads((K / "host/contract.json").read_text())
    if "folder" in base:                          # already a round-5b contract: start again from round 5a's sections
        base = base["round_5a"]
    stage = load("results/r5b_stage.json")
    items = {it["path"]: it for it in stage["core"]["items"]}
    test = load("results/r5b_host_test.json")
    parts = {}
    for f in sorted(glob.glob(str(K / "results/r5b_e2e_*_part*.json"))):
        d = json.loads(Path(f).read_text())
        for name, m in (d.get("models") or {}).items():
            parts.setdefault((d["accelerator"], m["file"]), m | {"part": Path(f).name})
    rows = []
    for L in C.BUCKETS:
        path = C.FOLDER["row"].format(length=L)
        sig = next((m for (acc, f), m in parts.items() if f == path), None)
        rows.append({"length": L, "file": path, "bytes": items[path]["bytes"], "sha256": items[path]["sha256"],
                     "source": items[path]["source"],
                     "inputs": sig["inputs"] if sig else None, "outputs": sig["outputs"] if sig else None})
    towers = []
    for t in C.TILES:
        path = C.FOLDER["tower"].format(tile=t)
        sig = next((m for (acc, f), m in parts.items() if f == path), None)
        pos = C.FOLDER["tower_pos"].format(tile=t)
        towers.append({"tile": t, "file": path, "bytes": items[path]["bytes"], "sha256": items[path]["sha256"],
                       "source": items[path]["source"], "inputs": sig["inputs"] if sig else None,
                       "outputs": sig["outputs"] if sig else None,
                       "position_rows": {"file": pos, "key": "pos", "dtype": "F32",
                                         "shape": list(C.load_f32_tensor(UP / pos).shape),
                                         "sha256": hashlib.sha256((UP / pos).read_bytes()).hexdigest(),
                                         "source": items[pos]["source"]}})
    delegation = {f"{acc}:{f}": {"compile_seconds": m["compile_seconds"], "fully_accelerated": m["fully_accelerated"],
                                 "one_partition_all_nodes": m["one_partition_all_nodes"],
                                 "replacing": m["delegation"]["replacing"], "record": m["part"]}
                  for (acc, f), m in sorted(parts.items())}
    doc = {
        "what": "clef-flash on LiteRT: the contract between the graphs and a host",
        "status": "round 5b (2026-10-21): the folder, its graphs and the LiteRT runtime of the host package",
        "model": base["model"],
        "host": {"package": "host/clef_litert", "version": C.__version__,
                 "requirements": "host/requirements-host.txt (Python 3.14.6, macOS 27.0): numpy, tokenizers, pillow and "
                                 "ai-edge-litert (imported when a graph is compiled); no torch, no transformers",
                 "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((K / "host/clef_litert").glob("*.py"))},
                 "entry": "ClefLiteRT.from_dir(folder, accelerator='gpu' | 'cpu').decide(request, tile=448 | 256)",
                 "steps": base["host"]["steps"]},
        "folder": {"layout": dict(C.FOLDER), "lengths": list(C.BUCKETS), "tiles": list(C.TILES),
                   "rule": "a request runs on the smallest present row graph that holds it; longer than the largest "
                           "present one = RequestTooLong; one row graph compiled at a time, the towers stay compiled",
                   "accelerators": {"gpu": "HardwareAccelerator.GPU, GpuOptions(enforce_f32=True) (Metal on a Mac)",
                                    "cpu": "HardwareAccelerator.CPU, CpuOptions(num_threads=8) (XNNPACK), no weight "
                                           "cache file"},
                   "not_supported": "the GPU's default precision (fp16 storage): the row graphs' norms overflow and "
                                    "the towers' rows are wrong (measured, not shipped)"},
        "row_graphs": rows, "tower_graphs": towers,
        "files": {C.FOLDER["tokenizer"]: base["files"]["tokenizer.json"],
                  C.FOLDER["joint_head"]: base["files"]["joint_head.safetensors"],
                  C.FOLDER["joint_head_config"]: base["files"]["joint_head_config.json"],
                  C.FOLDER["embed_tokens"]: base["files"]["embed_tokens.safetensors"],
                  C.FOLDER["lm_head"]: base["files"]["lm_head.safetensors"]},
        "token_ids": base["token_ids"], "request": base["request"], "ids": base["ids"],
        "row_graph": base["row_graph"] | {"files": [r["file"] for r in rows]},
        "positions": base["positions"], "picture": base["picture"], "head": base["head"], "response": base["response"],
        "verified": {
            "host_test": {"record": "results/r5b_host_test.json", "pass": test["pass"],
                          "sections": {k: v["pass"] for k, v in test["sections"].items()}} if test else None,
            "delegation_as_compiled": delegation,
            "e2e": {acc: (lambda s: {"record": f"results/r5b_score_{acc}.json", "complete": s["complete"],
                                     "line": {k: s["line"][k] for k in ("runs", "argmax_equal_non_near_tie",
                                                                        "near_tie", "max_abs_dp", "mean_of_run_means",
                                                                        "pass")},
                                     "red_arms": s["red_arms"]["moved"], "pass": s["pass"]})(s) if s else None
                    for acc, s in (("gpu", load("results/r5b_score_gpu.json")), ("cpu", load("results/r5b_score_cpu.json")))},
            "example": {acc: (lambda e: {"record": f"results/r5b_example_{acc}.json", "pass": e["pass"]})(e) if e else None
                        for acc, e in (("gpu", load("results/r5b_example_gpu.json")),
                                       ("cpu", load("results/r5b_example_cpu.json")))},
            "round_5a": base["verified"]},
        "round_5a": base,
    }
    (K / "host/contract.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    (UP / "contract.json").write_text(json.dumps(public(doc), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"row_graphs": [(r["length"], r["inputs"] is not None) for r in rows],
                      "towers": [(t["tile"], t["inputs"] is not None) for t in towers],
                      "verified": {k: (v if not isinstance(v, dict) else list(v)) for k, v in doc["verified"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
