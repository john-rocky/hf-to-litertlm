"""Baseline: the merged fp32 checkpoint in stock transformers 5.14.1 (the export environment's modeling, no patch)
against the reference.

AutoModel.from_pretrained(merged fp32 checkpoint, dtype=float32, attn_implementation="eager") -> Qwen3_5TextModel,
torch 1 thread (the Gated DeltaNet depthwise conv is about 20x slower at 12 threads on this Mac), eval, no_grad. Every
question is one row, unpadded: input_ids=[row_ids], use_cache=False, no mask, no position_ids. last_hidden_state[0] ->
[decide, *opts] -> head.pt readout (numpy float32) -> compared with the reference.

Outputs (never overwritten): results/tf514_unpatched_parity[_4b].json (summary + per-question rows),
cache/r2/tf514_hidden_full.npz (cache/r4b/... for 4B; every position of every row, key "<id>/<qid>"; the torch-graph
check reads it; an intermediate).

  --workers N   rows sharded over N processes, 1 torch thread each (the merged fp32 shards are mmapped, so the
                processes share one copy of the weights).
  --phase compute --rows-from oracle/oracle_0.8b.json
                only the hidden states (no reference needed): the 0.8B reference's rows are the 4B rows too (same
                fixtures, same tokenizer) -> the hidden npz + a json with the row_ids sha256 per question and seconds.
  --phase compare
                after the reference exists: asserts row_ids / decide_idx / opt_idx of every question equal the rows the
                hidden states were computed on, then the readout and the comparison -> the results json.
  --phase all   (default) both in one run, rows from the model's own reference."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from r2_common import (CKPT, HIDDEN, MODEL, ORACLE_JSON, ORACLE_NPZ, RESULT_SUFFIX, K, Clock, Head, add_model_arg,
                       cache_dir, compare_question, dump_json, load_oracle, qkey, select, summarize)

STOP_HIDDEN = 1e-3


def rows_sha(q):
    return hashlib.sha256(json.dumps([q["row_ids"], q["decide_idx"], q["opt_idx"]]).encode()).hexdigest()


def shard_paths(k, suffix):
    d = cache_dir("cache/r2")
    return d / f"tf514_hidden_w{k}{suffix}.npz", d / f"tf514_secs_w{k}{suffix}.json"


def load_model():
    from transformers import AutoModel
    t = time.time()
    model = AutoModel.from_pretrained(str(CKPT), dtype=torch.float32, attn_implementation="eager").eval()
    load_s = round(time.time() - t, 1)
    assert type(model).__name__ == "Qwen3_5TextModel", type(model)
    assert model.config._attn_implementation == "eager"
    return model, load_s


def hidden_of(model, q):
    ids = torch.tensor([q["row_ids"]], dtype=torch.long)
    t = time.time()
    h = model(input_ids=ids, use_cache=False).last_hidden_state[0].float().numpy()
    dt = time.time() - t
    assert h.shape == (q["row_len"], HIDDEN) and np.isfinite(h).all()
    return h.astype(np.float32), dt


def question_list(rows_from, limit):
    doc = json.loads(Path(rows_from).read_text()) if rows_from else load_oracle()
    qs = doc["questions"]
    return qs[:limit] if limit else qs


def worker(k, workers, limit, suffix, rows_from):
    """Rows k, k+workers, ... -> shard npz (hidden of every position) + seconds json."""
    torch.set_num_threads(1)
    model, load_s = load_model()
    qs = question_list(rows_from, limit)[k::workers]
    store, secs = {}, {}
    with torch.no_grad():
        for q in qs:
            store[qkey(q)], secs[qkey(q)] = hidden_of(model, q)
    npz, js = shard_paths(k, suffix)
    np.savez(npz, **store)
    js.write_text(json.dumps({"worker": k, "load_seconds": load_s, "seconds": secs}))


def compute(a, smoke):
    """-> (hidden {key: [len, d]}, seconds {key: s}, load seconds, compute record)."""
    questions = question_list(a.rows_from, a.limit)
    t0 = time.time()
    if a.workers > 1:
        for k in range(a.workers):
            for p in shard_paths(k, RESULT_SUFFIX + smoke):
                assert not p.exists(), f"stale shard {p}"
        cmd = [sys.executable, __file__, "--model", MODEL, "--limit", str(a.limit), "--workers", str(a.workers)]
        if a.rows_from:
            cmd += ["--rows-from", a.rows_from]
        procs = [subprocess.Popen(cmd + ["--worker", str(k)]) for k in range(a.workers)]
        codes = [p.wait() for p in procs]
        assert codes == [0] * a.workers, codes
        hidden, secs, load_s = {}, {}, None
        for k in range(a.workers):
            npz, js = shard_paths(k, RESULT_SUFFIX + smoke)
            z = np.load(npz)
            hidden.update({key: z[key] for key in z.files})
            d = json.loads(js.read_text())
            secs.update(d["seconds"])
            load_s = d["load_seconds"]
        for k in range(a.workers):
            for p in shard_paths(k, RESULT_SUFFIX + smoke):
                p.unlink()
    else:
        torch.set_num_threads(1)
        model, load_s = load_model()
        hidden, secs = {}, {}
        with torch.no_grad():
            for n, q in enumerate(questions):
                hidden[qkey(q)], secs[qkey(q)] = hidden_of(model, q)
                if n % 50 == 0 or q["row_len"] > 1024:
                    print(json.dumps({"n": n, "key": qkey(q), "len": q["row_len"], "s": round(secs[qkey(q)], 3)}), flush=True)
    assert set(hidden) == {qkey(q) for q in questions}
    record = {"rows_from": a.rows_from or str(ORACLE_JSON.relative_to(K)),
              "questions": len(questions), "row_sha256": {qkey(q): rows_sha(q) for q in questions},
              "seconds": secs, "load_seconds": load_s, "workers": a.workers, "seconds_compute_wall": round(time.time() - t0, 1)}
    return hidden, secs, load_s, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="processes (1 torch thread each)")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--phase", choices=["all", "compute", "compare"], default="all")
    ap.add_argument("--rows-from", default="", help="phase compute: oracle json whose questions give the rows")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    smoke = f"_smoke{a.limit}" if a.limit else ""
    if a.worker >= 0:
        return worker(a.worker, a.workers, a.limit, RESULT_SUFFIX + smoke, a.rows_from)
    out = K / f"results/tf514_unpatched_parity{RESULT_SUFFIX}{smoke}.json"
    out_npz = cache_dir("cache/r2") / f"tf514_hidden_full{RESULT_SUFFIX}{smoke}.npz"
    out_rec = cache_dir("cache/r2") / f"tf514_compute{RESULT_SUFFIX}{smoke}.json"
    assert not out.exists()
    clock = Clock()
    started_at = clock.stamp()
    import transformers
    if a.phase in ("all", "compute"):
        assert not out_npz.exists() and not out_rec.exists()
        hidden_store, secs_by_key, load_s, record = compute(a, smoke)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **hidden_store)
        record.update(started_at=started_at, npz=str(out_npz.relative_to(K)))
        if a.phase == "compute":
            dump_json(out_rec, record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("row_sha256", "seconds")}, indent=1))
            return
    else:
        record = json.loads(out_rec.read_text())
        z = np.load(out_npz)
        hidden_store = {key: z[key] for key in z.files}
        secs_by_key, load_s = record["seconds"], record["load_seconds"]
    head = Head()
    oracle = load_oracle()
    ref = np.load(ORACLE_NPZ)
    questions = oracle["questions"][: a.limit] if a.limit else oracle["questions"]
    same_rows = {qkey(q): record["row_sha256"].get(qkey(q)) == rows_sha(q) for q in questions}
    assert all(same_rows.values()), [k for k, v in same_rows.items() if not v][:5]
    rows, full, secs = [], {}, []
    red_probs = None
    for n, q in enumerate(questions):
        h = hidden_store[qkey(q)]
        secs.append(secs_by_key[qkey(q)])
        h_sel = select(h, q)
        z_pre, z_post, probs = head(h_sel)
        row = compare_question(q, probs, z_post, h_sel, ref[qkey(q)])
        row["seconds"] = round(secs[-1], 4)
        rows.append(row)
        if qkey(q) + "/full" in ref.files:
            full[qkey(q)] = float(np.abs(h.astype(np.float64) - ref[qkey(q) + "/full"]).max())
        if q["id"] == "red_arm_000":
            red_probs = probs
    summary = summarize(rows, red_probs, oracle)
    summary["full_rows_max_abs_hidden"] = full
    summary["full_rows_max_abs_hidden_max"] = max(full.values()) if full else None
    hmax = max(summary["overall"]["h_sel_max_abs"], summary["full_rows_max_abs_hidden_max"] or 0.0)
    doc = {
        "step": "baseline: stock transformers 5.14.1, no patch, one unpadded row per question"
                + (f" (model {MODEL})" if RESULT_SUFFIX else ""),
        "started_at": started_at, "seconds_wall": clock.seconds(), "load_seconds": load_s,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()},
        "torch_threads": 1, "workers": record.get("workers", a.workers), "attn_implementation": "eager",
        "model_class": "Qwen3_5TextModel", "model": MODEL, "checkpoint": str(CKPT.relative_to(K)), "head": head.info,
        "call": "model(input_ids=[row_ids], use_cache=False).last_hidden_state[0]; no attention_mask, no position_ids",
        "rows_computed_from": record["rows_from"], "rows_identical_to_oracle": len(same_rows),
        "seconds_compute_wall": record.get("seconds_compute_wall"),
        "seconds_per_row": {"median": float(np.median(secs)), "min": float(np.min(secs)), "max": float(np.max(secs)),
                            "sum": float(np.sum(secs))},
        "hidden_max_abs_vs_oracle": hmax, "stop_threshold_hidden": STOP_HIDDEN, "stop": bool(hmax > STOP_HIDDEN),
        "summary": summary, "hidden_npz": str(out_npz.relative_to(K)), "rows": rows,
    }
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("rows", "summary")}, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("overall", "bar_pass", "red_arm", "flips", "full_rows_max_abs_hidden")}, indent=1))


if __name__ == "__main__":
    main()
