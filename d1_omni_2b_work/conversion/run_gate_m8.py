"""The gate runs of d1-Omni-2B's eight-signature file: the d1-2B batch gate's group form on the Omni host
(host/d1_batch_omni.D1HostBatchedOmni: pictures and audio clips) with the multi8 decide backend
(host/multi8_omni.Multi8Decide: ONE CompiledModel, Metal float32 + constant_tensor_sharing, or the CPU with the
XNNPACK weight cache), the tower and the audio encoder = the single files WITHOUT sharing (the timing's and the
phone's form), written in run_gate.py's candidate form for score.py: per record `answers`, `probs`, `usage`, `seqs`
(with `audio`), `graphs`, `timing`, `time_s`, `error`; slot hidden states in <out>_slots.npz as `<id>::seq<j>`.

  --mode b8    the eight-sequence line: the records of --ids-file (conversion/ids_L128.json), up to --group (64)
               requests per `host.run` (the provider's `_batches` puts them in one batch), batch_min 1 = every L128
               sequence through decide_b8_L128, eight a call, padding rows for the rest; --b8single <file> = the
               B = 8 single file (host/b8_single_omni.B8SingleDecide, no sharing) as the decide instead
  --mode full  every record of --files one at a time (run_gate.py's form, a refusal is a result), the routing of the
               ship host (batch_min 8: b8 only for >= 8 sequences of <= 128 tokens in one batch, else the smallest
               bucket)

  $EXPORT conversion/run_gate_m8.py --mode b8 --accel gpu --out out/gate/b8_m8_gpu32.json
  $EXPORT conversion/run_gate_m8.py --mode b8 --accel cpu --weight-cache out/xnn_cache/multi8_fp16fc.xnncache \
      --out out/gate/b8_m8_cpu.json
  $EXPORT conversion/run_gate_m8.py --mode full --accel gpu --out out/gate/full_m8_gpu32.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORK, "snapshot")
TABLE = os.path.join(WORK, "tables/embed_table.npy")
MULTI8 = "out/graphs/d1-omni-2b_decide_multi8_fp16fc.tflite"
FILES = ["set/requests.json", "set/extra_g96x72.json", "set/red_arms.json", "set/requests_audio.json",
         "set/red_arms_audio.json", "set/card_omni.json"]
MEDIA = ["set/media.json", "set/media_extra.json", "set/media_audio.json"]


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("b8", "full"), required=True)
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--multi8", default=MULTI8)
    ap.add_argument("--weight-cache", default="", help="CPU: CpuOptions.xnnpack_weight_cache_path (a file)")
    ap.add_argument("--b8single", default="", help="b8 mode only: the decide = this B = 8 single file through "
                    "host/b8_single_omni.B8SingleDecide, no sharing and no cache (the speed window's b8single arm)")
    ap.add_argument("--ids-file", default="conversion/ids_L128.json")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    hb = load_by_path("d1_batch_omni", os.path.join(WORK, "host/d1_batch_omni.py"))
    m8 = load_by_path("multi8_omni", os.path.join(WORK, "host/multi8_omni.py"))
    import lrt_backend as _b
    import records as _r
    from lrt_backend import LiteRTAudio, LiteRTTower, audio_paths, graph_paths
    from records import load_records, open_request
    assert os.path.samefile(_b.__file__, os.path.join(WORK, "host/lrt_backend.py")), _b.__file__
    assert os.path.samefile(_r.__file__, os.path.join(WORK, "conversion/records.py")), _r.__file__

    gpu = a.accel == "gpu"
    wc = os.path.join(WORK, a.weight_cache) if a.weight_cache else ""
    if wc:
        os.makedirs(os.path.dirname(wc), exist_ok=True)
    if a.b8single:
        assert a.mode == "b8" and not wc, "--b8single answers the L128 set only, without a weight cache"
        bs = load_by_path("b8_single_omni", os.path.join(WORK, "host/b8_single_omni.py"))
        decide = bs.B8SingleDecide(os.path.join(WORK, a.b8single), a.accel, "fp32", 8, sharing=False)
    else:
        decide = m8.Multi8Decide(os.path.join(WORK, a.multi8), a.accel, "fp32", 8, sharing=gpu, weight_cache=wc,
                                 batch_min=1 if a.mode == "b8" else 8)
    gdir = os.path.join(WORK, "out/graphs")
    _, tow_paths = graph_paths(gdir, "fp16fc")
    aud_paths = audio_paths(gdir, "fp16fc")
    tower = LiteRTTower(tow_paths, a.accel, "fp32", 8, keep=False)
    audio = LiteRTAudio(aud_paths, a.accel, "fp32", 8)
    host = hb.D1HostBatchedOmni(SNAP, decide, tower, embed_table=TABLE, audio=audio)
    code = {}
    for name in ("d1_host", "lrt_backend", "records", "d1_audio_features", "d1_pictures", "d1_tokenizer", "d1_layout",
                 "d1_batch_omni", "multi8_omni", "b8_single_omni"):
        f = getattr(sys.modules.get(name), "__file__", None)
        if f:
            code[os.path.relpath(f, WORK)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    code[os.path.relpath(__file__, WORK)] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    if a.mode == "b8":
        ids = [i for i in json.load(open(os.path.join(WORK, a.ids_file)))["ids"] if i in recs]
        group = a.group
    else:
        ids = sorted(recs)
        group = 1
    out_path = os.path.join(WORK, a.out)
    hid_path = out_path.replace(".json", "_slots.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = json.load(open(out_path)) if os.path.exists(out_path) else {"records": {}, "groups": []}
    arrays = dict(np.load(hid_path)) if os.path.exists(hid_path) else {}
    doc.setdefault("attempts", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
                                           "records_at_start": len(doc["records"]), "code_sha256": code})
    tcalls = {"tower": [], "audio": []}
    tow_call, aud_call = tower.__call__, audio.__call__

    def tower_logged(pv, grid):
        f = tow_call(pv, grid)
        tcalls["tower"].append({k: list(v) if isinstance(v, tuple) else v for k, v in tower.last.items()})
        return f

    def audio_logged(features):
        f = aud_call(features)
        tcalls["audio"].append(dict(audio.last))
        return f
    audio_logged.max_frames = audio.max_frames
    host.tower, host.audio = tower_logged, audio_logged

    def save():
        json.dump(doc, open(out_path + ".tmp", "w"))
        os.replace(out_path + ".tmp", out_path)
        np.savez(hid_path + ".tmp.npz", **arrays)
        os.replace(hid_path + ".tmp.npz", hid_path)
    calls_all = []
    t_all = time.time()
    todo = [i for i in ids if i not in doc["records"]]
    for g0 in range(0, len(todo), group):
        gids = todo[g0:g0 + group]
        reqs, named, states = [], [], []
        decide.calls = []
        for v in tcalls.values():
            v.clear()
        t1 = time.time()
        try:                    # conversion/run_gate.py's try: opening, naming and running a record (a refusal is a result)
            for rid in gids:
                st, qs, items = open_request(recs[rid], media)
                nm = host.named(qs)
                named.append(nm)
                states.append(st)
                reqs.append((st, list(nm.values()), items))
            done, err = host.run(reqs), None
        except Exception as e:  # noqa: BLE001  (--mode full has one record a group; a b8 group must not fail)
            if group != 1:
                raise
            done, err = None, f"{type(e).__name__}: {e}"
        dt = time.time() - t1
        seqs = host.last.get("seqs", []) if err is None else []
        hidden = host.last.get("slot_hidden", []) if err is None else []
        calls = list(decide.calls)
        calls_all.extend(calls)
        doc["groups"].append({"records": len(gids), "sequences": len(seqs), "graph_calls": len(calls),
                              "rows_per_call": [c["rows"] for c in calls],
                              "calls_by_signature": dict(collections.Counter(c["sig"] for c in calls)),
                              "tower_calls": len(tcalls["tower"]), "audio_calls": len(tcalls["audio"]),
                              "wall_s": round(dt, 3), "timing": jsonable(host.last.get("timing")) if err is None else None})
        for i, rid in enumerate(gids):
            if err is not None:
                doc["records"][rid] = {"answers": None, "probs": None, "usage": {"input_tokens": None,
                                                                                  "output_tokens": 0},
                                       "seqs": [], "graphs": {"group": len(doc["groups"]) - 1}, "timing": None,
                                       "time_s": round(dt, 4), "error": err}
                continue
            probs, read = done[i]
            answers = {name: host.answer(states[i], q, p) for (name, q), p in zip(named[i].items(), probs)}
            own = [j for j, s in enumerate(seqs) if s["request"] == i]
            row = {"answers": jsonable(answers), "probs": jsonable(probs),
                   "usage": {"input_tokens": read, "output_tokens": 0},
                   "seqs": jsonable([{k: seqs[j].get(k) for k in ("first_question", "ids", "state_len", "question",
                                                                  "ends", "image_grid_thw", "audio")} for j in own]),
                   "graphs": {"group": len(doc["groups"]) - 1,
                              "decide": [{"sig": c["sig"], "rows": c["rows"]} for c in calls] if group == 1 else None,
                              "tower": jsonable(tcalls["tower"]) if group == 1 else None,
                              "audio": jsonable(tcalls["audio"]) if group == 1 else None},
                   "timing": jsonable(host.last.get("timing")) if group == 1 else None,
                   "time_s": round(dt / len(gids), 4), "error": None}
            for n, j in enumerate(own):
                arrays[f"{rid}::seq{n}"] = np.asarray(hidden[j], np.float32)
            doc["records"][rid] = row
        g = doc["groups"][-1]
        print(f"group {len(doc['groups'])}: {len(gids)} records, {len(seqs)} sequences, {g['graph_calls']} calls "
              f"{g['calls_by_signature']}, tower {g['tower_calls']}, audio {g['audio_calls']}, {dt:.2f}s err={err}",
              flush=True)
        if len(doc["groups"]) % max(1, a.save_every if group == 1 else 1) == 0:
            save()
    doc["run"] = {"mode": a.mode, "accelerator": a.accel, "precision": "fp32",
                  "multi8": None if a.b8single else a.multi8, "b8single": a.b8single or None,
                  "sharing": gpu and not a.b8single,
                  "sharing_on_tower_audio": False, "weight_cache": a.weight_cache or None,
                  "batch_min": getattr(decide, "batch_min", None), "group": group, "records": len(ids),
                  "host": "host/d1_batch_omni.py (host/d1_host.py)",
                  "backend": "host/b8_single_omni.py" if a.b8single else "host/multi8_omni.py",
                  "tower_graphs": {f"{c[0]}x{c[1]}": os.path.relpath(p, WORK) for c, p in sorted(tow_paths.items())},
                  "audio_graphs": {str(T): os.path.relpath(p, WORK) for T, p in sorted(aud_paths.items())},
                  "calls_by_signature_this_process": dict(collections.Counter(c["sig"] for c in calls_all)),
                  "loads": decide.loads, "audio_loads": audio.loads, "code_sha256": code,
                  "torch_imported": "torch" in sys.modules, "wall_s_this_process": round(time.time() - t_all, 2)}
    save()
    print("DONE", json.dumps({k: doc["run"][k] for k in ("mode", "accelerator", "records", "batch_min",
                                                           "calls_by_signature_this_process", "loads",
                                                           "torch_imported", "wall_s_this_process")}))


if __name__ == "__main__":
    main()
