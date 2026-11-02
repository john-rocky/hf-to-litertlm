"""Run the question set through the PyTorch-free host (host/d1_host.py) on the LiteRT graphs of one form, on one
accelerator, and write the candidate in the reference's form (conversion/ref_run.py), for conversion/score.py.

  $EXPORT conversion/run_gate.py --accel gpu --precision fp32 --form fp16fc \
      --files set/requests.json set/extra_g96x72.json set/red_arms.json \
      --media set/media.json set/media_extra.json --out out/gate/gpu_fp32.json

The d1-2B conversion's run_gate.py with this model's loader (conversion/records.py: audio media and spoken questions opened as the
provider takes them) and the audio encoder graphs (lrt_backend.LiteRTAudio over
out/graphs/d1-omni-2b_audio_T<T>_<form>.tflite). `--multi out/graphs/d1-omni-2b_decide_multi7_fp16fc.tflite`
runs every bucket on the seven-signature file (one CompiledModel, conversion/lrt_multi.py); `--sharing` turns
GpuOptions.constant_tensor_sharing on for every graph (with --multi).

Per record: `answers` (host.answer on host.run's values, as ref_run does it), `probs` (the input of api.answer),
`usage`, `seqs` (ids, state_len, question tags, slot positions, picture grids), `graphs` (the bucket / canvas / audio
bucket each sequence / picture / clip ran on), `timing` (host.last), `time_s`, `error`. The slot hidden states go to
<out>_slots.npz as `<id>::seq<j>`. Records run in the order of their longest sequence in the reference (one graph
load per bucket). Resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "host"))
SNAP = os.path.join(WORK, "snapshot")
TABLE = os.path.join(WORK, "tables/embed_table.npy")


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", default="fp16fc")
    ap.add_argument("--graphs", default="out/graphs")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--media", nargs="*", default=[])
    ap.add_argument("--ref", default="out/ref/ref_fp32.json", help="for the run order (longest sequence)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--min-bucket", type=int, default=0, help="use only decision graphs of L >= this")
    ap.add_argument("--multi", default="", help="the seven-signature decide file (one CompiledModel, all buckets)")
    ap.add_argument("--sharing", action="store_true", help="GpuOptions.constant_tensor_sharing (with --multi)")
    ap.add_argument("--weight-cache", default="", help="CpuOptions.xnnpack_weight_cache_path for the --multi file "
                    "(without it XNNPACK packs the shared weights once per signature, > 28 GiB)")
    ap.add_argument("--decide-keep", type=int, default=2, help="single-bucket decide graphs kept open (on "
                    "Metal one decide graph holds ~12-13 GiB, two open passed the 24 GiB watch)")
    ap.add_argument("--tower-keep", type=int, default=1, help="1 = every tower canvas stays open, 0 = one at a time")
    ap.add_argument("--save-every", type=int, default=20, help="records between saves of --out (a stopped run "
                    "resumes from the last save)")
    a = ap.parse_args()
    if a.sharing and not a.multi:
        ap.error("--sharing is implemented for --multi runs only")
    import numpy as np

    from d1_host import D1Host
    from lrt_backend import LiteRTAudio, LiteRTDecide, LiteRTTower, audio_paths, graph_paths
    from records import load_records, open_request

    gdir = os.path.join(WORK, a.graphs)
    dec_paths, tow_paths = graph_paths(gdir, a.form)
    aud_paths = audio_paths(gdir, a.form)
    dec_paths = {L: p for L, p in dec_paths.items() if L >= a.min_bucket}
    if a.multi:
        from lrt_multi import LiteRTMultiDecide, SharedAudio, SharedTower

        wc = os.path.join(WORK, a.weight_cache) if a.weight_cache else ""
        if wc:
            os.makedirs(os.path.dirname(wc), exist_ok=True)
        decide = LiteRTMultiDecide(os.path.join(WORK, a.multi), a.accel, a.precision, a.threads, sharing=a.sharing,
                                   weight_cache=wc)
        dec_paths = dict(decide.paths)
        tower = ((SharedTower(tow_paths, a.accel, a.precision, a.threads, keep=bool(a.tower_keep)) if a.sharing else
                  LiteRTTower(tow_paths, a.accel, a.precision, a.threads, keep=bool(a.tower_keep)))
                 if tow_paths else None)
        audio = ((SharedAudio(aud_paths, a.accel, a.precision, a.threads) if a.sharing else
                  LiteRTAudio(aud_paths, a.accel, a.precision, a.threads)) if aud_paths else None)
    else:
        decide = LiteRTDecide(dec_paths, a.accel, a.precision, a.threads, keep=a.decide_keep)
        tower = (LiteRTTower(tow_paths, a.accel, a.precision, a.threads, keep=bool(a.tower_keep))
                 if tow_paths else None)
        audio = LiteRTAudio(aud_paths, a.accel, a.precision, a.threads) if aud_paths else None
    host = D1Host(SNAP, decide, tower, embed_table=TABLE, audio=audio)
    import hashlib

    import d1_host as _h
    import lrt_backend as _b
    import records as _r
    code = {os.path.relpath(m.__file__, WORK): hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()
            for m in (_h, _b, _r, sys.modules[__name__])}
    for name in ("d1_audio_features", "d1_pictures", "d1_tokenizer", "lrt_multi"):
        if name in sys.modules and getattr(sys.modules[name], "__file__", None):
            f = sys.modules[name].__file__
            code[os.path.relpath(f, WORK)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    recs, media = load_records(a.files, a.media)
    if a.ids:
        want = set(a.ids.split(","))
        recs = [r for r in recs if r["id"] in want]
    # one load per bucket: records in the order of the bucket of their longest sequence (the reference's sequences)
    ref_path = os.path.join(WORK, a.ref)
    longest = {}
    if os.path.exists(ref_path):
        for rid, r in json.load(open(ref_path))["records"].items():
            longest[rid] = max((len(s["ids"]) for s in r.get("seqs") or []), default=0)

    def bucket_of(r):
        n = longest.get(r["id"], 0)
        return next((L for L in sorted(dec_paths) if L >= n), 10 ** 9)
    recs.sort(key=lambda r: (bucket_of(r), r["id"]))
    out_path = os.path.join(WORK, a.out)
    hid_path = out_path.replace(".json", "_slots.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = json.load(open(out_path)) if os.path.exists(out_path) else {"records": {}}
    arrays = dict(np.load(hid_path)) if os.path.exists(hid_path) else {}
    doc["run"] = {"accelerator": a.accel, "precision": a.precision, "form": a.form, "threads": a.threads,
                  "decide_graphs": {str(L): os.path.relpath(p, WORK) for L, p in sorted(dec_paths.items())},
                  "tower_graphs": {f"{c[0]}x{c[1]}": os.path.relpath(p, WORK) for c, p in sorted(tow_paths.items())},
                  "audio_graphs": {str(T): os.path.relpath(p, WORK) for T, p in sorted(aud_paths.items())},
                  "host": "host/d1_host.py", "code_sha256": code, "torch_imported": "torch" in sys.modules,
                  **({"multi": {"file": a.multi, "signatures": {str(L): s for L, s in sorted(decide.sigs.items())},
                                "sharing": a.sharing, "weight_cache": a.weight_cache or None,
                                "loads": decide.loads}} if a.multi else {}),
                  "decide_keep": a.decide_keep, "tower_keep": a.tower_keep, "save_every": a.save_every}
    # one entry per process that wrote this file (a run stopped by its memory watch resumes in a new process)
    doc.setdefault("attempts", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
                                           "records_at_start": len(doc["records"]), "code_sha256": code,
                                           "decide_keep": a.decide_keep, "tower_keep": a.tower_keep})

    calls = {"decide": [], "tower": [], "audio": []}
    dec_call = decide.__call__
    tow_call = tower.__call__ if tower else None
    aud_call = audio.__call__ if audio else None

    def decide_logged(embeds, state_len, question):
        h = dec_call(embeds, state_len, question)
        calls["decide"].append(dict(decide.last))
        return h

    def tower_logged(pv, grid):
        f = tow_call(pv, grid)
        calls["tower"].append({k: list(v) if isinstance(v, tuple) else v for k, v in tower.last.items()})
        return f

    def audio_logged(features):
        f = aud_call(features)
        calls["audio"].append(dict(audio.last))
        return f
    host.decide = decide_logged
    host.decide.max_len = decide.max_len if hasattr(decide, "max_len") else None
    if tower:
        host.tower = tower_logged
    if audio:
        audio_logged.max_frames = audio.max_frames
        host.audio = audio_logged
    t0 = time.time()
    for n, rec in enumerate(recs):
        rid = rec["id"]
        if rid in doc["records"]:
            continue
        for v in calls.values():
            v.clear()
        t1 = time.time()
        try:
            state, questions, items = open_request(rec, media)
            named = host.named(questions)
            done = host.run([(state, list(named.values()), items)])
            probs, read = done[0]
            answers = {name: host.answer(state, q, p) for (name, q), p in zip(named.items(), probs)}
            err = None
        except Exception as e:  # a refusal is a result too
            probs, read, answers, err = None, None, None, f"{type(e).__name__}: {e}"
        dt = time.time() - t1
        seqs = [{k: s.get(k) for k in ("first_question", "ids", "state_len", "question", "ends", "image_grid_thw",
                                        "audio")}
                for s in host.last.get("seqs", [])]
        row = {"answers": jsonable(answers), "probs": jsonable(probs), "usage": {"input_tokens": read,
                                                                                  "output_tokens": 0},
               "seqs": jsonable(seqs), "graphs": {k: jsonable(v) for k, v in calls.items()},
               "timing": jsonable(host.last.get("timing")), "time_s": round(dt, 4), "error": err}
        if err is None:
            for j, h in enumerate(host.last.get("slot_hidden", [])):
                arrays[f"{rid}::seq{j}"] = np.asarray(h, np.float32)
        doc["records"][rid] = row
        if n % max(1, a.save_every) == 0 or n == len(recs) - 1:
            json.dump(doc, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)
            np.savez(hid_path + ".tmp.npz", **arrays)
            os.replace(hid_path + ".tmp.npz", hid_path)
        print(f"{rid} {dt:.2f}s tokens={read} err={err}", flush=True)
    doc["wall_s"] = round(time.time() - t0, 2)
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["run"]["decide_loads"] = decide.loads
    doc["run"]["tower_loads"] = {f"{c[0]}x{c[1]}": {"load_s": round(g.load_s, 2) if hasattr(g, "load_s") else None,
                                                    "fully_accelerated": getattr(g, "fully_accelerated", None)}
                                 for c, g in (tower.graphs.items() if tower else [])}
    doc["run"]["audio_loads"] = audio.loads if audio else []
    json.dump(doc, open(out_path + ".tmp", "w"))
    os.replace(out_path + ".tmp", out_path)
    np.savez(hid_path + ".tmp.npz", **arrays)
    os.replace(hid_path + ".tmp.npz", hid_path)
    print("DONE", json.dumps(doc["run"]))


if __name__ == "__main__":
    main()
