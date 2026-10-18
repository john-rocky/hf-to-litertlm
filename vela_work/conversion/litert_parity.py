"""Step 6: one exported graph on the Mac (ai-edge-litert CompiledModel) against the reference embeddings.

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer
    $P scripts/litert_parity.py --tflite exports/vela_L128_ids_v2_fp16fc_i8emb_shift.tflite --L 128 --accel cpu [--threads 8]
    $P scripts/litert_parity.py --tflite ... --accel gpu [--f32]           # Metal; --f32 = GpuOptions(enforce_f32=True)
    $P scripts/litert_parity.py --tflite exports/vela_L128_embeds_wfp16_shift.tflite --L 128 --table exports/tables/tok_embeddings_fp16.bin ...
    ... --timing     # per-call timing of a short row (the fixture-median pair sentence), ~32 / ~100 / ~200 / ~400-token rows
                     # and the window's longest row (median of 20 after 5 warm-ups); run it under quiet_hold.py (the Mac window)

Every fixture row that fits L (pair / stsb / long* groups; --groups, --limit select) is run alone: input_ids (or the host-
looked-up table rows as inputs_embeds), attention_mask -> embedding. The parity bar (embed_gate.py; REPRODUCE.md):
finite, cos vs the reference >= 0.999 on every row, top-1 retrieval agreement 100 % on the non-near-tie queries, pair-cosine
Spearman vs the reference >= 0.999, mask liveness (the longest row with 5 valid tokens dropped moves: cos < 0.999); the
task metrics (PAWS-X AP, STS-B v2 Spearman) next to the reference's on the same rows.
Output results/litert_{cpu|gpu_f32|gpu_f16}_parity_L{L}_{variant}{tag}.json (+ _rows). Never overwrites.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
from embed_gate import BAR, score  # noqa: E402

PAD_ID = 0
HIDDEN = 768


def open_model(path, accel, f32, threads):
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
    if accel == "gpu":
        opts = Options(HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=f32))
    else:
        opts = Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
    t0 = time.perf_counter()
    m = CompiledModel.from_file(str(path), options=opts)
    compile_s = time.perf_counter() - t0
    try:
        fully = bool(m.is_fully_accelerated())
    except Exception as e:  # noqa: BLE001
        fully = f"unavailable: {type(e).__name__}"
    sigs = m.get_signature_list()
    assert len(sigs) == 1, sigs
    key = next(iter(sigs))
    in_det, out_det = m.get_input_tensor_details(key), m.get_output_tensor_details(key)
    ins = {n: m.create_input_buffer_by_name(key, n) for n in in_det}
    outs = {n: m.create_output_buffer_by_name(key, n) for n in out_det}
    return m, key, ins, outs, dict(signature=key, compile_s=round(compile_s, 2), is_fully_accelerated=fully, inputs=sorted(in_det), outputs=sorted(out_det))


def load_table(path):
    p = Path(path)
    raw = np.fromfile(p, dtype=np.float16 if "fp16" in p.name else np.float32)
    assert raw.size % HIDDEN == 0
    return raw.reshape(-1, HIDDEN).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--table", default="", help="host table [vocab,768] (fp16 or f32 .bin) for the embeds form")
    ap.add_argument("--groups", default="pair,stsb,long128,long256,long512,edge512")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th fitting row (group coverage kept; the tag carries _evN)")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else W / a.tflite
    variant = path.stem.replace(f"vela_L{a.L}_", "")
    mode = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_f16")
    tag = a.tag + (f"_lim{a.limit}" if a.limit else "") + (f"_ev{a.every}" if a.every > 1 else "")
    out = W / "results" / f"litert_{mode}_parity_L{a.L}_{variant}{tag}.json"
    out_rows = W / "results" / f"litert_{mode}_rows_L{a.L}_{variant}{tag}.json"
    assert not out.exists(), f"never overwrite {out}"
    fx = json.loads((W / "fixtures" / "texts.json").read_text())
    ref = np.load(W / "oracle" / "embeddings.npz")
    oracle = json.loads((W / "oracle" / "oracle.json").read_text())
    groups = set(a.groups.split(","))
    rows = [r for r in fx["rows"] if r["group"] in groups and r["row_len"] <= a.L and r["key"] in ref.files]
    if a.every > 1:
        rows = rows[:: a.every]
    if a.limit:
        rows = rows[: a.limit]
    table = load_table(a.table if Path(a.table).is_absolute() else W / a.table) if a.table else None
    m, key, ins, outs, info = open_model(path, a.accel, a.f32, a.threads)
    assert ("inputs_embeds" in info["inputs"]) == (table is not None), (info["inputs"], a.table)
    print(json.dumps(info), flush=True)
    mem_after_compile = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
    L = a.L
    ids_buf = np.full((1, L), PAD_ID, dtype=np.int32)
    mask_buf = np.zeros((1, L), dtype=np.float32)
    oname = info["outputs"][0]

    def run_ids(ids, n):
        ids_buf[:] = PAD_ID
        ids_buf[0, :n] = np.asarray(ids[:n], dtype=np.int32)
        mask_buf[:] = 0.0
        mask_buf[0, :n] = 1.0
        t = time.perf_counter()
        if table is not None:
            ins["inputs_embeds"].write(np.ascontiguousarray(table[ids_buf[0]][None]))
        else:
            ins["input_ids"].write(ids_buf)
        ins["attention_mask"].write(mask_buf)
        tr = time.perf_counter()
        m.run_by_name(key, ins, outs)
        tr2 = time.perf_counter()
        v = np.asarray(outs[oname].read(HIDDEN, np.float32), dtype=np.float32).reshape(HIDDEN).copy()
        t2 = time.perf_counter()
        return v, dict(write=(tr - t) * 1e3, run=(tr2 - tr) * 1e3, read=(t2 - tr2) * 1e3, total=(t2 - t) * 1e3)

    recs_ms, vec = [], {}
    t0 = time.time()
    for i, r in enumerate(rows):
        v, ms = run_ids(r["ids"], r["row_len"])
        vec[r["key"]] = v
        recs_ms.append(dict(ms=round(ms["total"], 3), run_ms=round(ms["run"], 3)))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    done_s = time.time() - t0
    sc = score(vec, fx, oracle, ref)
    longest = max(rows, key=lambda r: r["row_len"])
    v_full = vec[longest["key"]]
    v_cut, _ = run_ids(longest["ids"], longest["row_len"] - 5)
    liveness = dict(key=longest["key"], row_len=longest["row_len"],
                    cos_full_vs_minus5=float(v_full.astype(np.float64) @ v_cut.astype(np.float64)) if np.isfinite(v_cut).all() and np.isfinite(v_full).all() else None)
    bar_pass = bool(sc["bar_core"] and liveness["cos_full_vs_minus5"] is not None and liveness["cos_full_vs_minus5"] < BAR["liveness_cos_below"])
    recs = sc.pop("per_row")
    for x, ms in zip(recs, recs_ms):
        x.update(ms)
    summary = dict(file=str(path.relative_to(W)), bytes=path.stat().st_size, L=L, accel=a.accel, gpu_enforce_f32=a.f32 if a.accel == "gpu" else None,
                   threads=a.threads if a.accel == "cpu" else None, table=a.table or None, model_info=info, groups=sorted(groups),
                   **{k: sc[k] for k in ("rows", "cos", "max_abs", "norm", "pairs", "top1", "metrics", "bar", "bar_core")},
                   liveness=liveness, bar_pass=bar_pass,
                   timing_note="per-row ms here are contended (Mac shared) and informational; --timing rows are the card numbers",
                   ms=dict(median=round(statistics.median([x["ms"] for x in recs]), 2), run_median=round(statistics.median([x["run_ms"] for x in recs]), 2)) if recs else None,
                   seconds=round(done_s, 1),
                   memory=dict(ru_maxrss_mb_after_compile=mem_after_compile, ru_maxrss_mb_end=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)),
                   env=dict(python=platform.python_version(), macos=platform.mac_ver()[0], machine=platform.machine(),
                            ai_edge_litert=__import__("importlib.metadata").metadata.version("ai-edge-litert")))
    if a.timing:
        pool = [r for r in fx["rows"] if r["group"] == "pair" and r["row_len"] <= L]
        lens = sorted(r["row_len"] for r in pool)
        picks = {"pair_median_row": min(pool, key=lambda r: abs(r["row_len"] - lens[len(lens) // 2]))}
        cands = [r for r in fx["rows"] if r["row_len"] <= L]
        for name, target in (("row_32", 32), ("row_100", 100), ("row_200", 200), ("row_400", 400)):
            if target <= L:
                picks[name] = min(cands, key=lambda r: abs(r["row_len"] - target))
        picks["row_max"] = max(cands, key=lambda r: r["row_len"])
        timing = {}
        for name, r in picks.items():
            for _ in range(5):
                run_ids(r["ids"], r["row_len"])
            samples = [run_ids(r["ids"], r["row_len"])[1] for _ in range(20)]
            timing[name] = dict(key=r["key"], row_len=r["row_len"], n=20, warmups=5,
                                total_ms=dict(median=round(statistics.median(s["total"] for s in samples), 2), min=round(min(s["total"] for s in samples), 2), max=round(max(s["total"] for s in samples), 2)),
                                run_ms=dict(median=round(statistics.median(s["run"] for s in samples), 2), min=round(min(s["run"] for s in samples), 2), max=round(max(s["run"] for s in samples), 2)),
                                write_ms_median=round(statistics.median(s["write"] for s in samples), 3), read_ms_median=round(statistics.median(s["read"] for s in samples), 3))
        lock = Path("~/code/coreai/_GPU_LOCK").expanduser()
        summary["timing"] = dict(definition="one text: write input(s) + run + read-back of the embedding; median of 20 after 5 warm-ups",
                                 lock=lock.read_text().strip() if lock.exists() else None, load1=os.getloadavg()[0], rows=timing)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str))
    out_rows.write_text(json.dumps(dict(file=str(path.relative_to(W)), rows=recs), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k in ("file", "accel", "gpu_enforce_f32", "model_info", "rows", "cos", "max_abs", "pairs", "top1", "metrics", "liveness", "bar_pass", "ms", "seconds", "memory", "timing")}, indent=1, default=str))


if __name__ == "__main__":
    main()
