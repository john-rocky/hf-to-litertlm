"""Step 6: one exported graph on the Mac (ai-edge-litert CompiledModel) against the publisher's reference.

    python litert_parity.py --tflite exports/phocinae_L512_ids_v2_fp16fc_i8emb.tflite --L 512 --accel cpu [--threads 8]
    python litert_parity.py --tflite ... --accel gpu [--f32]            # Metal; --f32 = GpuOptions(enforce_f32=True)
    python litert_parity.py --tflite exports/phocinae_L512_embeds_wfp16.tflite --L 512 --table exports/tables/tok_embeddings_fp16.bin ...
    ... --timing                                                        # adds per-call timing of the fixture-median row and a
                                                                        # 300-token row (median of 20 after 5 warm-ups);
                                                                        # run it under quiet_hold.py (Mac measurement window)

Every fixture row that fits L (groups test / flip / control; --rows limits) is run alone: input_ids (or the host-looked-up
fp16 table rows as inputs_embeds), attention_mask, qtype_onehot -> token_logits gathered at the markers, pooled_cls ->
act features -> the act head graph (always on the CPU) -> act_logits. Post-temperature probabilities vs the reference
(oracle/oracle.json): max / mean |dp|, argmax on non-near-tie rows, non-finite counts, act_probability |d|; the control
rows against the ORIGINAL rows' reference (must move > 0.02); the graph's own flip rates (rev / random mean / any).
Output: results/litert_{cpu|gpu_f32|gpu_f16}_parity_L{L}_{variant}.json (+ _rows). Never overwrites.
"""
import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np

from common import BAR, NEAR_TIE, W, act_features, act_probability, dump_json, probs_from_logits, sha256_file, temperatures

PAD_ID = 0
HIDDEN = 384


def footprint_mb():
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        class RUsage(ctypes.Structure):
            _fields_ = [("pad", ctypes.c_uint8 * 2048)]
        buf = RUsage()
        if libc.proc_pid_rusage(ctypes.c_int(__import__("os").getpid()), ctypes.c_int(4), ctypes.byref(buf)) == 0:
            # rusage_info_v4: ri_phys_footprint at offset 0x70? (keep ru_maxrss as the portable number)
            pass
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)


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
    return m, key, ins, outs, dict(compile_s=round(compile_s, 2), is_fully_accelerated=fully, inputs=sorted(in_det), outputs=sorted(out_det))


def load_table(path):
    raw = np.fromfile(path, dtype=np.float16)
    assert raw.size % HIDDEN == 0
    return raw.reshape(-1, HIDDEN).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--table", default="", help="fp16 [vocab,384] row-major table for the embeds form")
    ap.add_argument("--act", default="exports/phocinae_act_head_fp32.tflite")
    ap.add_argument("--rows", default="test,flip,control")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else W / a.tflite
    variant = path.stem.replace(f"phocinae_L{a.L}_", "")
    mode = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_f16")
    tag = a.tag + (f"_lim{a.limit}" if a.limit else "")
    out = W / "results" / f"litert_{mode}_parity_L{a.L}_{variant}{tag}.json"
    out_rows = W / "results" / f"litert_{mode}_rows_L{a.L}_{variant}{tag}.json"
    assert not out.exists(), f"never overwrite {out}"
    temps = temperatures()
    oracle = json.loads((W / "oracle" / "oracle.json").read_text())
    ref = {o["id"]: o for o in oracle["rows"]}
    fx = json.loads((W / "fixtures" / "rows_laya.json").read_text())["rows"]
    groups = set(a.rows.split(","))
    rows = [r for r in fx if r["group"] in groups and r["row_len"] <= a.L]
    if a.limit:
        rows = rows[: a.limit]
    table = load_table(W / a.table if not Path(a.table).is_absolute() else a.table) if a.table else None
    m, key, ins, outs, info = open_model(path, a.accel, a.f32, a.threads)
    assert sorted(info["outputs"]) == ["pooled_cls", "token_logits"], info
    assert ("inputs_embeds" in info["inputs"]) == (table is not None), (info["inputs"], a.table)
    print(json.dumps(info), flush=True)
    am, akey, ains, aouts, ainfo = open_model(W / a.act if not Path(a.act).is_absolute() else a.act, "cpu", False, 1)
    mem_after_compile = footprint_mb()

    L = a.L
    ids_buf = np.full((1, L), PAD_ID, dtype=np.int32)
    mask_buf = np.zeros((1, L), dtype=np.float32)
    onehot = np.zeros((1, 3), dtype=np.float32)

    def run_row(r, timed=False):
        n = r["row_len"]
        ids_buf[:] = PAD_ID
        ids_buf[0, :n] = np.asarray(r["ids"], dtype=np.int32)
        mask_buf[:] = 0.0
        mask_buf[0, :n] = 1.0
        onehot[:] = 0.0
        onehot[0, r["qtype"]] = 1.0
        t = time.perf_counter()
        if table is not None:
            ins["inputs_embeds"].write(np.ascontiguousarray(table[ids_buf[0]][None]))
        else:
            ins["input_ids"].write(ids_buf)
        ins["attention_mask"].write(mask_buf)
        ins["qtype_onehot"].write(onehot)
        tr = time.perf_counter()
        m.run_by_name(key, ins, outs)
        tr2 = time.perf_counter()
        tl = np.asarray(outs["token_logits"].read(L, np.float32), dtype=np.float32).reshape(L)
        pc = np.asarray(outs["pooled_cls"].read(HIDDEN, np.float32), dtype=np.float32).reshape(HIDDEN)
        t2 = time.perf_counter()
        raw = tl[r["markers"]].astype(np.float64)
        finite = bool(np.isfinite(tl[:n]).all() and np.isfinite(pc).all())
        if finite:
            feats = np.asarray([act_features(raw)], dtype=np.float32)
            ains["pooled_cls"].write(np.ascontiguousarray(pc[None]))
            ains["feats"].write(feats)
            am.run_by_name(akey, ains, aouts)
            al = np.asarray(aouts["act_logits"].read(2, np.float32), dtype=np.float64).reshape(2)
        else:
            al = np.full(2, np.nan)
        t3 = time.perf_counter()
        return dict(raw=raw, pc=pc, al=al, finite=finite, nonfinite_real=int((~np.isfinite(tl[:n])).sum()), nonfinite_all=int((~np.isfinite(tl)).sum()),
                    ms=dict(write=(tr - t) * 1e3, run=(tr2 - tr) * 1e3, read=(t2 - tr2) * 1e3, act=(t3 - t2) * 1e3, total=(t3 - t) * 1e3))

    recs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        o = ref[r["id"]]
        got = run_row(r)
        rec = dict(id=r["id"], group=r["group"], qtype=r["qtype"], K=r["K"], row_len=r["row_len"], finite=got["finite"],
                   nonfinite_real=got["nonfinite_real"], nonfinite_all=got["nonfinite_all"], ms=round(got["ms"]["total"], 2), run_ms=round(got["ms"]["run"], 2))
        if got["finite"]:
            p = probs_from_logits(got["raw"], r["qtype"], temps)
            dp = np.abs(np.asarray(p) - np.asarray(o["probs"]))
            rec.update(max_abs_dlogit=float(np.abs(got["raw"] - np.asarray(o["logits"])).max()), max_abs_dp=float(dp.max()), mean_abs_dp=float(dp.mean()),
                       argmax=int(np.argmax(p)), argmax_equal=int(np.argmax(p)) == o["argmax"], near_tie=o["near_tie"], top2_gap_oracle=o["top2_gap"],
                       act_dp=abs(act_probability(got["al"]) - o["act_probability"]), probs=[round(float(x), 6) for x in p])
            if r["group"] == "control":
                base = ref[r["id"].replace("ctrl/", "", 1)]
                rec["control_move_vs_original"] = float(np.abs(np.asarray(p) - np.asarray(base["probs"])).max())
        recs.append(rec)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    done_s = time.time() - t0

    test = [x for x in recs if x["group"] == "test" and x["finite"]]
    nf = [x for x in recs if not x["finite"]]
    non_near = [x for x in test if not x["near_tie"]]
    near = [x for x in test if x["near_tie"]]
    flip_recs = [x for x in recs if x["group"] == "flip" and x["finite"]]
    ctrl = [x for x in recs if x["group"] == "control" and x["finite"]]
    by_id = {x["id"]: x for x in recs}
    # flip rates of the graph itself (chosen option KEY vs the original order's choice)
    fx_by_id = {r["id"]: r for r in fx}
    flips = {}
    for suffix in ("rev", "s0", "s1", "s2"):
        n = f = 0
        for x in flip_recs:
            if not x["id"].endswith("#" + suffix):
                continue
            base = by_id.get(x["id"].split("#")[0])
            if base is None or not base["finite"]:
                continue
            n += 1
            f += fx_by_id[x["id"]]["option_keys"][x["argmax"]] != fx_by_id[base["id"]]["option_keys"][base["argmax"]]
        flips[suffix] = dict(n=n, flips=f, rate=round(f / max(1, n), 4))
    any_n = any_f = 0
    for x in flip_recs:
        if x["id"].endswith("#s0"):
            base = by_id.get(x["id"].split("#")[0])
            if base is None:
                continue
            b = fx_by_id[base["id"]]["option_keys"][base["argmax"]]
            any_n += 1
            any_f += any(fx_by_id[k]["option_keys"][by_id[k]["argmax"]] != b for k in (x["id"][:-2] + s for s in ("s0", "s1", "s2")) if k in by_id and by_id[k]["finite"])
    flips["random_any"] = dict(n=any_n, flips=any_f, rate=round(any_f / max(1, any_n), 4))
    flips["random_mean"] = round(sum(flips[s]["rate"] for s in ("s0", "s1", "s2")) / 3, 4)

    def agg(xs):
        if not xs:
            return {}
        return dict(rows=len(xs), argmax_equal=sum(x["argmax_equal"] for x in xs), max_abs_dp=max(x["max_abs_dp"] for x in xs),
                    mean_abs_dp=float(np.mean([x["mean_abs_dp"] for x in xs])), p95_max_abs_dp=float(np.percentile([x["max_abs_dp"] for x in xs], 95)),
                    max_abs_dlogit=max(x["max_abs_dlogit"] for x in xs), max_act_dp=max(x["act_dp"] for x in xs))

    overall = agg([x for x in recs if x["finite"]])
    summary = dict(file=str(path.relative_to(W)), sha256=sha256_file(path), bytes=path.stat().st_size, L=L, accel=a.accel, gpu_enforce_f32=a.f32 if a.accel == "gpu" else None,
                   threads=a.threads if a.accel == "cpu" else None, table=a.table or None, model_info=info, act_head=dict(file=a.act, **ainfo),
                   rows=dict(run=len(recs), groups=sorted(groups), nonfinite_rows=len(nf), nonfinite_ids=[x["id"] for x in nf][:20]),
                   overall=overall, test=agg(test), test_non_near_tie=agg(non_near), test_near_tie=agg(near), flip=agg(flip_recs), control=agg(ctrl),
                   control_moves=[dict(id=x["id"], move=x["control_move_vs_original"], max_abs_dp=x["max_abs_dp"]) for x in ctrl],
                   flip_rates=flips, reference_flip_rates=oracle["flip"],
                   bar=dict(**BAR, near_tie_definition=f"reference top-2 gap <= {NEAR_TIE}"),
                   bar_pass=bool(recs and not nf and non_near and all(x["argmax_equal"] for x in non_near)
                                 and overall["max_abs_dp"] <= BAR["max_abs_dp"] and overall["mean_abs_dp"] <= BAR["mean_abs_dp"]
                                 and (not ctrl or max(x["control_move_vs_original"] for x in ctrl) > BAR["control_row_moves"])),
                   timing_note="per-row ms here are contended (Mac shared) and informational; --timing rows are the card numbers",
                   ms=dict(median=round(statistics.median([x["ms"] for x in recs]), 2), run_median=round(statistics.median([x["run_ms"] for x in recs]), 2)) if recs else None,
                   seconds=round(done_s, 1), memory=dict(ru_maxrss_mb_after_compile=mem_after_compile, ru_maxrss_mb_end=footprint_mb()),
                   env=dict(python=platform.python_version(), macos=platform.mac_ver()[0], machine=platform.machine(),
                            ai_edge_litert=__import__("importlib.metadata").metadata.version("ai-edge-litert")))
    if a.timing:
        pool = [r for r in fx if r["group"] == "test" and r["row_len"] <= L]      # the timing rows come from every fitting test row, not the --limit subset
        lens = sorted(r["row_len"] for r in pool)
        med_len = lens[len(lens) // 2]
        picks = {"fixture_median_row": min(pool, key=lambda r: abs(r["row_len"] - med_len))}
        for name, target in (("row_300", 300), ("row_200", 200), ("row_128", 128)):
            if target <= L:
                picks[name] = min(pool, key=lambda r: abs(r["row_len"] - target))
        timing = {}
        for name, r in picks.items():
            for _ in range(5):
                run_row(r)
            samples = [run_row(r)["ms"] for _ in range(20)]
            timing[name] = dict(id=r["id"], row_len=r["row_len"], K=r["K"], n=20, warmups=5,
                                total_ms=dict(median=round(statistics.median(s["total"] for s in samples), 2), min=round(min(s["total"] for s in samples), 2), max=round(max(s["total"] for s in samples), 2)),
                                run_ms=dict(median=round(statistics.median(s["run"] for s in samples), 2), min=round(min(s["run"] for s in samples), 2), max=round(max(s["run"] for s in samples), 2)),
                                act_ms_median=round(statistics.median(s["act"] for s in samples), 3),
                                write_ms_median=round(statistics.median(s["write"] for s in samples), 3), read_ms_median=round(statistics.median(s["read"] for s in samples), 3))
        summary["timing"] = dict(definition="one question's row: write + run + read-back of the main graph + act features + act head; median of 20 after 5 warm-ups",
                                 lock=Path(__import__("os").path.expanduser("~/code/coreai/_GPU_LOCK")).read_text().strip() if Path(__import__("os").path.expanduser("~/code/coreai/_GPU_LOCK")).exists() else None,
                                 load1=__import__("os").getloadavg()[0], rows=timing)
    dump_json(out, summary)
    dump_json(out_rows, dict(file=str(path.relative_to(W)), rows=recs))
    print(json.dumps({k: v for k, v in summary.items() if k in ("file", "accel", "gpu_enforce_f32", "model_info", "rows", "overall", "test_non_near_tie", "test_near_tie", "control_moves", "flip_rates", "bar_pass", "ms", "seconds", "memory", "timing")}, indent=1, default=str))


if __name__ == "__main__":
    main()
