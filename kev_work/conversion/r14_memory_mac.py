"""Mac GPU memory of the Kev-0.8B files, one leg per new process: phys_footprint and ru_maxrss (timing_mac.memory,
bytes) before, after the compile, after one request (and after a second one with the state handed over through the
host) and the process maximum, with the compile seconds, fully accelerated and finite.

    python r14_memory_mac.py --leg pair --Ls 128 --share share --run 1
    python r14_memory_mac.py --leg row --L 128 --form C7 --run 1

One process per leg, with nothing else on the GPU: a model compiled earlier in the same process stays in its footprint
(in r14_timing_mac_req.py's run the pair files were compiled one after another in one process, so only the compile
that opened the process gave a new-process value).
pair: exports/kev08b_sharedstate_Ls<Ls>_Lq64_v2_fp16fc_i8emb_r14B-vs6.tflite on Metal with float32 activations
(GpuOptions(enforce_f32=True, constant_tensor_sharing=<share>), r14_timing_mac.PairS); the request = Ls128:
own_fiveq_09 (5 questions), Ls256: own_email_03 (3 questions), run once with the state handed over directly
(after_request) and once through the host (after_request_host).
row: the C7 L128 or C L512 row file with the row timing's options (timing_mac_shared.Row: enforce_f32), one call of
that bucket's timing row (p50_80 = tv4_007 / T300).
Output results/memory_mac_r14_pair_Ls<Ls>_<share|noshare>_run<k>.json or
results/memory_mac_r14_row_L<L>_<form>_run<k>.json; never overwritten."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import K, memory, stamp, swap_used, top_lines  # noqa: E402

V = "v2_fp16fc_i8emb"
REQ = {128: "own_fiveq_09", 256: "own_email_03"}
ROWFILE = {(128, "C7"): f"exports/kev08b_rowprefill_L128_{V}_r17C7-bkzq.tflite",
           (512, "C"): f"exports/kev08b_rowprefill_L512_{V}_r14B-vs6.tflite"}


def gb(doc):
    return {k: round(v / 1e9, 3) for k, v in doc.items()} if isinstance(doc, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["pair", "row"], required=True)
    ap.add_argument("--Ls", type=int, choices=[128, 256])
    ap.add_argument("--share", choices=["share", "noshare"])
    ap.add_argument("--L", type=int, choices=[128, 512])
    ap.add_argument("--form", choices=["C", "C7"])
    ap.add_argument("--run", type=int, required=True)
    a = ap.parse_args()
    name = (f"memory_mac_r14_pair_Ls{a.Ls}_{a.share}_run{a.run}" if a.leg == "pair" else
            f"memory_mac_r14_row_L{a.L}_{a.form}_run{a.run}")
    out = K / f"results/{name}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    doc = {"what": f"Mac GPU memory in a fresh process ({name}; scripts/r14_memory_mac.py)", "leg": a.leg, "run": a.run,
           "top_at_start": top_lines(), "swap_at_start": swap_used(), "started_at": stamp()}
    t0 = time.time()
    try:
        before = memory()
        if a.leg == "pair":
            from r14_timing_mac import PairS
            from r14_timing_mac_req import load_points
            path = K / f"exports/kev08b_sharedstate_Ls{a.Ls}_Lq64_{V}_r14B-vs6.tflite"
            req = load_points(a.Ls, 64, [REQ[a.Ls]])[REQ[a.Ls]]
            g = PairS(path, a.Ls, 64, "gpu_f32", 8, a.share == "share")
            after_compile = memory()
            ms, _, ok = g.request(req, "direct")
            after_request = memory()
            ms_h, _, ok_h = g.request(req, "host")
            after_host = memory()
            doc.update(file=str(path.relative_to(K)), bytes=path.stat().st_size, share=a.share == "share", Ls=a.Ls, Lq=64,
                       request=REQ[a.Ls], questions=len(req["questions"]), n_state=req["n_state"],
                       compile_s=round(g.compile_s, 2), fully_accelerated=g.fully, finite=bool(ok and ok_h),
                       request_ms_first_direct=round(ms, 1), request_ms_second_host=round(ms_h, 1),
                       before=before, after_compile=after_compile, after_request=after_request, after_request_host=after_host)
            g.close()
        else:
            from r14_timing_mac import row_sets
            from timing_mac_shared import Row, padded
            path = K / ROWFILE[(a.L, a.form)]
            set_name, _, rows = row_sets(json.loads((K / "oracle/oracle_0.8b.json").read_text()))[a.L]
            g = Row(path, "gpu_f32", 8)
            after_compile = memory()
            ids, valid = padded(rows[0], g.L)
            ms, ok = g.call(ids, valid, list(range(min(len(rows[0]), g.L))))
            after_request = memory()
            doc.update(file=str(path.relative_to(K)), bytes=path.stat().st_size, form=a.form, L=a.L, row_set=set_name,
                       row_tokens=len(rows[0]), compile_s=round(g.compile_s, 2), fully_accelerated=g.fully, finite=bool(ok),
                       call_ms_first=round(ms, 1), before=before, after_compile=after_compile, after_request=after_request)
            g.close()
        doc["after_close"] = memory()
        doc["gb"] = {k: gb(doc.get(k)) for k in ("before", "after_compile", "after_request", "after_request_host", "after_close")
                     if doc.get(k)}
        doc["status"] = "measured"
    finally:
        doc.update(top_at_end=top_lines(), swap_at_end=swap_used(), finished_at=stamp(), seconds_wall=round(time.time() - t0, 1))
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: doc.get(k) for k in ("status", "compile_s", "fully_accelerated", "finite")} | {
        "after_compile_gb": doc.get("gb", {}).get("after_compile", {}).get("phys_footprint"),
        "after_request_gb": doc.get("gb", {}).get("after_request", {}).get("phys_footprint"),
        "max_gb": doc.get("gb", {}).get("after_close", {}).get("lifetime_max_phys_footprint")}))


if __name__ == "__main__":
    main()
