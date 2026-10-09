"""Round 2 acceptance 3: export D1Prefill (float32, one window L) with litert_torch, then scan the file.

    $EXPORT scripts/d1_export.py --L 64              # tiny (default --source tiny)
    $EXPORT scripts/d1_export.py --L 64 --embeds     # D1PrefillEmbeds
    $EXPORT scripts/d1_export.py --source <snapshot> --tag real --L 1024   (round 3)

1. Guard: a results/{tag}_graph_check*.json that says pass and covers this L (acceptance 2; the graph code is the same).
2. litert_torch.convert(graph, sample_kwargs={"ids": int32 [1, L], "valid": float32 [1, L]}) (Kev's call; embeds:
   {"embeds": float32 [1, L, d], "valid"}) -> .export(exports/{tag}_rowprefill[_embeds]_L{L}_fp32.tflite). Input names
   = the kwargs keys, output name = the returned dict key.
3. Static scan (scripts/tflite_scan.py, ai_edge_litert's schema module) -> results/{tag}_opscan[_embeds]_L{L}.json;
   the summary goes to results/{tag}_export[_embeds]_L{L}.json with seconds, bytes, sha256, versions and the stops:
   CUSTOM op or a rank > 4 tensor = stop; BROADCAST_TO and INT64 tensors are counted (both must be 0 here).
Existing files are never overwritten. On an exception: logs/{tag}_export_L{L}.traceback.txt +
results/{tag}_export_attempt[_embeds]_L{L}.json, and the exception is raised again (the MLIR text is in the log).
--norm-scale <json or file> (round 5): d1_prefill_graph.apply_norm_scale before the export (the guard then needs a
graph check of the tag with the same k); the record goes to the json (`norm_scale`).
--retire (round 6c, after the gate of a bucket): no export; the float32 file exports/{tag}_rowprefill[_embeds]_L{L}_
fp32.tflite is hashed again, compared with its export record (bytes + sha256), deleted when equal, and the deletion is
written to results/{tag}[_embeds]_L{L}_fp32_deleted.json and appended to the ledger results/{tag}_files.json
(`deleted`). A file that differs from its record is not deleted (exit 1).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_prefill_graph as G  # noqa: E402
from d1_common import K, ROWS  # noqa: E402


def retire(tag: str, L: int, kind: str) -> int:
    """--retire (module docstring)."""
    import shutil

    from d1_common import sha256_file

    path = K / f"exports/{tag}_rowprefill{kind}_L{L}_fp32.tflite"
    rec_path = K / f"results/{tag}_export{kind}_L{L}.json"
    out = K / f"results/{tag}{kind}_L{L}_fp32_deleted.json"
    ledger = K / f"results/{tag}_files.json"
    assert not out.exists(), f"never overwrite {out}"
    rec = json.loads(rec_path.read_text())
    assert rec["file"] == str(path.relative_to(K)) and rec["status"] == "EXPORTED", (rec["file"], rec["status"])
    t0 = time.perf_counter()
    size, sha = path.stat().st_size, sha256_file(path)
    doc = {"file": rec["file"], "bytes": rec["bytes"], "sha256": rec["sha256"], "from": str(rec_path.relative_to(K)),
           "sha256_rehashed_before_delete": sha, "bytes_before_delete": size,
           "equal_to_record": size == rec["bytes"] and sha == rec["sha256"], "rehash_seconds": round(time.perf_counter() - t0, 1),
           "free_gib_before": round(shutil.disk_usage(K).free / 2**30, 1)}
    if not doc["equal_to_record"]:
        print(json.dumps(doc, indent=1))
        print(f"STOP: {path.name} differs from {rec_path.name}; not deleted")
        return 1
    path.unlink()
    doc.update(deleted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), present_after=path.exists(),
               free_gib_after=round(shutil.disk_usage(K).free / 2**30, 1),
               why="round 6c: the float32 export of a bucket, deleted after the bucket's gate (the v2 file stays)")
    out.write_text(json.dumps(doc, indent=1) + "\n")
    led = json.loads(ledger.read_text()) if ledger.exists() else {"what": "deleted row graph files (round 6c)", "deleted": []}
    led.setdefault("deleted", []).append(doc)
    ledger.write_text(json.dumps(led, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--source", default="tiny")
    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--embeds", action="store_true", help="D1PrefillEmbeds (embeds float32 [1, L, d] + valid)")
    ap.add_argument("--norm-scale", default="", help="per-site k (JSON or file); default none")
    ap.add_argument("--retire", action="store_true", help="delete the float32 file after its gate (docstring)")
    a = ap.parse_args()
    if a.retire:
        return retire(a.tag, a.L, "_embeds" if a.embeds else "")
    ks = G.parse_norm_scale(a.norm_scale or None)
    L, kind = a.L, ("_embeds" if a.embeds else "")
    path = K / f"exports/{a.tag}_rowprefill{kind}_L{L}_fp32.tflite"
    out = K / f"results/{a.tag}_export{kind}_L{L}.json"
    scan_out = K / f"results/{a.tag}_opscan{kind}_L{L}.json"
    for p in (path, out, scan_out):
        assert not p.exists(), f"never overwrite {p}"
    # the guard: a passing torch graph check of this tag that covers L (any window for the embeds graph: same code)
    found = sorted((K / "results").glob(f"{a.tag}_graph_check*.json"))
    same_k = lambda d: d.get("norm_scale", {}).get("k", {x: 0 for x in G.NORM_SITES}) == ks
    passing = [p for p in found if json.loads(p.read_text())["summary"]["pass"] and same_k(json.loads(p.read_text()))]
    covering = [p for p in passing if L in json.loads(p.read_text())["Ls"]] or (passing if a.embeds else [])
    assert covering, (f"no passing results/{a.tag}_graph_check*.json with norm scale {ks} covers L={L}: run "
                      f"d1_graph_check.py first")
    check_path = covering[0]
    check = json.loads(check_path.read_text())
    torch.set_num_threads(4)
    t0 = time.time()
    lm, info = G.load(a.source)
    norm_scale = G.apply_norm_scale(lm, ks)
    graph = G.graph(lm, L, embeds=a.embeds)
    pad_id = G.TINY_PAD_ID if a.source == "tiny" else 124893
    fold = (lambda ids: [i % 255 for i in ids]) if a.source == "tiny" else (lambda ids: list(ids))
    first = fold(json.loads(ROWS.read_text())["rows"][0]["ids"])[: L // 2 + 1]
    ids, valid = G.row_inputs(first, L, pad_id)
    with torch.no_grad():
        sample = {"embeds": lm.embed_tokens(ids), "valid": valid} if a.embeds else {"ids": ids, "valid": valid}
        ref = graph(**sample)["hidden"]
    record = {"L": L, "graph": "D1PrefillEmbeds" if a.embeds else "D1Prefill", "source": info, "status": "RUNNING",
              "norm_scale": norm_scale,
              "file": str(path.relative_to(K)), "guard": {"file": str(check_path.relative_to(K)),
                                                          "summary": check["summary"]},
              "sample_kwargs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                                for k, v in sample.items()},
              "convert_call": "litert_torch.convert(graph, sample_kwargs=" + ("{'embeds': ..., 'valid': ...}" if a.embeds
                                                                              else "{'ids': ..., 'valid': ...}") + ").export(path)",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")},
              "load_seconds": round(time.time() - t0, 1)}
    t1 = time.perf_counter()
    try:
        import litert_torch

        lrt = litert_torch.convert(graph, sample_kwargs=sample)
        record["convert_seconds"] = round(time.perf_counter() - t1, 1)
        t2 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t2, 1)
        from tflite_scan import scan

        s = scan(path)
        scan_out.write_text(json.dumps(s, indent=1) + "\n")
        d = info["hidden"]
        want_in = ([("embeds", [1, L, d], "FLOAT32"), ("valid", [1, L], "FLOAT32")] if a.embeds else
                   [("ids", [1, L], "INT32"), ("valid", [1, L], "FLOAT32")])
        sig = s["signatures"]
        sig_ok = (len(sig) == 1 and sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) == sorted(want_in)
                  and [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("hidden", [1, L, d], "FLOAT32")])
        stops = {"custom_op_count": s["custom_op_count"], "rank_gt4_tensor_count": s["rank_gt4_tensor_count"]}
        record.update(
            status="CUSTOM_OP_STOP" if s["custom_op_count"] else ("RANK5_STOP" if s["rank_gt4_tensor_count"] else "EXPORTED"),
            bytes=s["bytes"], sha256=s["sha256"], signatures=sig, signature_matches_contract=sig_ok, stops=stops,
            operator_count=s["operator_count"], op_histogram=s["op_histogram"],
            tensor_rank_histogram=s["tensor_rank_histogram"], tensor_dtype_histogram=s["tensor_dtype_histogram"],
            int64_tensor_count=s["int64_tensor_count"], broadcast_to_count=s["forbidden_counts"]["BROADCAST_TO"],
            forbidden_counts=s["forbidden_counts"], pad_count=s["pad_count"], pad_summary=s["pad_summary"],
            batch_matmul_count=s["batch_matmul_count"], batch_matmul_all_rank4=s["batch_matmul_all_rank4"],
            batch_matmul_shape_groups=s["batch_matmul_shape_groups"],
            fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
            gather_count=s["gather_count"], stablehlo_ops=s["stablehlo_ops"], opscan=str(scan_out.relative_to(K)),
            torch_sample_hidden_finite=bool(torch.isfinite(ref).all()))
    except BaseException:
        (K / f"logs/{a.tag}_export{kind}_L{L}.traceback.txt").write_text(traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t1, 1))
        (K / f"results/{a.tag}_export_attempt{kind}_L{L}.json").write_text(json.dumps(record, indent=1) + "\n")
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
    record["seconds_wall"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("op_histogram", "batch_matmul_shape_groups", "guard",
                                                                    "source", "signatures")}, indent=1))
    return 0 if record["status"] == "EXPORTED" and sig_ok else 1


if __name__ == "__main__":
    sys.exit(main())
