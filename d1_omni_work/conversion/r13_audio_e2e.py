"""Round 13 (step A-7): end to end for the audio buckets T501 and T3001 against the provider's float32 run of the same
clips (results/audio_e2e_ref_T501_T3001.json, scripts/r13_audio_e2e_ref.py) — the phone's prefixes of round 12b
and the Mac through the files of ship/. Ship venv (ai-edge-litert 2.2.0, numpy, tokenizers, soundfile), from K, one
quiet window (the Metal runs):

    cd d1_omni_work
    ~/code/standup/tools/quiet/quiet_hold.py d1c-r13-audio-e2e -- venv-ship/bin/python scripts/r13_audio_e2e.py
    -> results/audio_e2e_T501_T3001_r13.json

Clips (rebuilt from the fixture wav files, int16 sha256 asserted equal to the reference's): aud_01 cut to 5 s
(T501, P 63) with aud_01's three questions; aud_01 + aud_02 + aud_03 (27.15 s, T3001, P 340) with card_topic's
question and aud_01's three questions. Runs, each through ship/host (D1Omni: prompt.encode() ids / markers asserted
equal to the reference's, build_inputs, the decision graph of the smallest bucket, readout_f64):
  0 phone      the Galaxy S26's prefixes (round 12b legs A5 / A8: OpenCL FP16_WITH_FP32_ACCUM on the fp16-safe audio
               files = this repository's files; device/r12b/<leg>.dump/000.f32 = the graph's whole output, rows
               0 .. P-1 used) -> the decision graph on the Mac CPU (8 threads) = the end-to-end form of round 12
  1 provider   the provider's own float32 prefix -> the same decision graph = the decision graph's own share
  2 mac_cpu    D1Omni(accelerator="cpu", threads=8): host mel, the audio graph and the decision graph on the CPU
  3 mac_metal  D1Omni(accelerator="gpu"): each graph at its contract.json precision (the audio graph at Metal's
               default precision, the decision graph at fp32) = the shipped Mac GPU pipeline
  4 mac_metal_audio_cpu_decision  the audio graph at Metal's default precision -> the decision graph on the CPU
               (the same split as run 0)
Statistics and bar = scripts/s26_score.py compare (FACTS §7: argmax outside near-ties, max |dp| <= 0.02, mean |dp|
<= 0.002, no non-finite value). Per run and clip: the prefix against the provider's (max |d|, rel_rms).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
SHIP = K / "ship"
sys.path.insert(0, str(SHIP / "host"))
import d1_audio_host as A  # noqa: E402  (ship/host, byte copies of K/host)
import d1_host as H  # noqa: E402
import d1_omni as O  # noqa: E402
import d1_prompt as Pm  # noqa: E402

sys.path.insert(1, str(K / "scripts"))
import s26_score as SS  # noqa: E402  (compare, BAR: the definitions of rounds 3-12)

REF = K / "results/audio_e2e_ref_T501_T3001.json"
REF_NPZ = K / "cache/r13/audio_e2e_ref_prefix.npz"
OUT = K / "results/audio_e2e_T501_T3001_r13.json"
DUMPS = {501: ("A5_gpu_fp16acc32_f16s_T501", "results/s26_npu_parity_A5_gpu_fp16acc32_f16s_T501_r12b.json"),
         3001: ("A8_gpu_fp16acc32_f16s_T3001", "results/s26_npu_parity_A8_gpu_fp16acc32_f16s_T3001_r12b.json")}
CLIP = {"aud_01_cut5": (501, "aud_01 cut to 5 s"), "long": (3001, "aud_01 + aud_02 + aud_03")}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def clips(records):
    def samples(rid):
        r = records[rid]
        path = K / r["media"]["ref"]
        assert sha_file(path) == r["media"]["sha256"], rid
        return read_wav(path)

    return {"aud_01_cut5": samples("aud_01")[:80000],
            "long": np.concatenate([samples(r) for r in ("aud_01", "aud_02", "aud_03")])}


def prefix_distance(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return {"max_abs": float(np.abs(a64 - b64).max()),
            "rel_rms": float(np.sqrt(((a64 - b64) ** 2).mean()) / np.sqrt((b64 ** 2).mean())),
            "bit_equal": bool(np.array_equal(a, b)), "nonfinite": int((~np.isfinite(a)).sum())}


def score_rows(model, ref_rec, prefix):
    """D1Omni's encode of the reference request with this prefix -> {key: (probs, logits)} on model.runner."""
    state = ref_rec["state"]
    qs = [ref_rec["questions_dicts"][q["qid"]] for q in ref_rec["questions"]]
    rows = model.rows(state, qs, prefix, "audio")
    out = {}
    for r, q in zip(rows, ref_rec["questions"]):
        assert r["ids"] == q["ids"] and r["markers"] == q["markers"] and r["P"] == q["prefix"], (ref_rec["id"], q["qid"])
        n = r["P"] + len(r["ids"])
        L = H.bucket_for(n, model.buckets)
        x = H.build_inputs(r["ids"], prefix, L)
        x["qtype_onehot"] = H.qtype_onehot(r["q"])
        scores = model.runner(x, L)
        K_ = r["q"].options
        logits = [float(scores[0, r["P"] + m]) for m in r["markers"][:K_]]
        probs = H.readout_f64(scores, r["P"], r["markers"], r["q"], r["calibrate"], model.temperatures)
        out[f"{ref_rec['id']}/{q['qid']}"] = ([float(v) for v in probs], logits, L)
    return out


def main():
    t0 = time.time()
    ref = json.loads(REF.read_text())
    reqs = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    qd = {"aud_01_cut5": reqs["aud_01"]["request"], "long_card_topic": reqs["card_topic"]["request"],
          "long_aud_01": reqs["aud_01"]["request"]}
    recs = {}
    for e in ref["records"]:
        e = dict(e)
        e["state"] = qd[e["id"]]["state"]
        e["questions_dicts"] = qd[e["id"]]["questions"]
        recs[e["id"]] = e
    cl = clips(reqs)
    for e in recs.values():
        x = cl[e["clip"]["name"]]
        assert hashlib.sha256(x.astype("<i2").tobytes()).hexdigest() == e["clip"]["int16_sha256"], e["id"]
    with np.load(REF_NPZ) as z:
        prov = {k.split(".")[0]: np.asarray(z[k], np.float32) for k in z.files}
    assert sha_file(REF_NPZ) == ref["prefix_npz"]["sha256"]
    by_clip = {}
    for rid, e in recs.items():
        by_clip.setdefault(e["clip"]["name"], []).append(rid)
    ref_rows = {f"{e['id']}/{q['qid']}": {"key": f"{e['id']}/{q['qid']}", "mode": "audio", "id": e["id"],
                                          "probs": q["probs"], "logits": q["logits_raw"], "near_tie": q["near_tie"]}
                for e in recs.values() for q in e["questions"]}
    refd = {k: (v["probs"], v["logits"]) for k, v in ref_rows.items()}

    cpu = O.D1Omni(SHIP, accelerator="cpu", threads=8)
    gpu = O.D1Omni(SHIP, accelerator="gpu")
    phone = {}
    for T, (leg, parity) in DUMPS.items():
        p = K / f"device/r12b/{leg}.dump/000.f32"
        rows_out = int(np.prod(json.loads((K / parity).read_text())["graph"]["out_shape"][1:2]))
        arr = np.fromfile(p, np.float32).reshape(rows_out, 1024)
        phone[T] = {"array": arr, "file": str(p.relative_to(K)), "sha256": sha_file(p), "leg": leg,
                    "scored_record": parity}

    def prefix_for(source, clip):
        T = CLIP[clip][0]
        P = recs[by_clip[clip][0]]["prefix"]
        if source == "phone":
            return phone[T]["array"][:P]
        if source == "provider":
            return prov[by_clip[clip][0]]
        if source == "mac_cpu":
            return cpu.audio_prefix(cl[clip])
        return gpu.audio_prefix(cl[clip])           # the audio graph at the contract's Metal precision

    runs_def = [("phone", "Galaxy S26 OpenCL GPU, FP16_WITH_FP32_ACCUM (audio graph, round 12b legs A5 / A8) + Mac CPU "
                          "XNNPACK 8 threads (decision graph)", "phone", cpu),
                ("provider", "the provider's float32 prefix + Mac CPU XNNPACK 8 threads (decision graph) = the decision "
                             "graph's own share", "provider", cpu),
                ("mac_cpu", "Mac CPU XNNPACK 8 threads (host mel, audio graph, decision graph: D1Omni accelerator=cpu)",
                 "mac_cpu", cpu),
                ("mac_metal", "Mac Metal, each graph at its contract.json precision (audio graph default precision, "
                              "decision graph fp32: D1Omni accelerator=gpu)", "mac_metal", gpu),
                ("mac_metal_audio_cpu_decision", "Mac Metal default precision (audio graph) + Mac CPU XNNPACK 8 threads "
                                                 "(decision graph)", "mac_metal", cpu)]
    runs = []
    for i, (name, backend, src, scorer) in enumerate(runs_def):
        run = {"index": i, "source": name, "backend": backend, "buckets": {}}
        for clip, (T, what) in CLIP.items():
            pre = prefix_for(src, clip)
            P = recs[by_clip[clip][0]]["prefix"]
            assert pre.shape == (P, 1024), (name, clip, pre.shape)
            lit, Ls = {}, {}
            for rid in by_clip[clip]:
                for k, (pr, lg, L) in score_rows(scorer, recs[rid], pre).items():
                    lit[k], Ls[k] = (pr, lg), L
            rows = [ref_rows[k] for k in lit]
            st = SS.compare(rows, lit, refd)
            sec = {"clip": what, "clip_seconds": recs[by_clip[clip][0]]["clip"]["seconds"], "T": T, "P": P,
                   "records": by_clip[clip], "decision_buckets": sorted(set(Ls.values())),
                   "prefix_vs_provider": prefix_distance(pre, prov[by_clip[clip][0]]),
                   "summary": {"rows": st["rows_compared"], "max_abs_dp": st["max_abs_dp"], "p95_abs_dp": st["p95_abs_dp"],
                               "mean_abs_dp": st["mean_abs_dp"], "max_abs_dlogit": st["max_abs_dlogit"],
                               "argmax_outside_near_tie": f"{st['argmax']['equal_outside_near_tie']}/"
                                                          f"{st['argmax']['rows_outside_near_tie']}",
                               "near_tie": f"{st['argmax']['near_tie_equal']}/{st['argmax']['near_tie_rows']}",
                               "cutoff_crossings": st["cutoff_crossings"], "nonfinite_rows": len(st["nonfinite_rows"]),
                               "bar_pass": st["bar_pass"]},
                   "comparison": st,
                   "per_row": [{"key": k, "L": Ls[k], "probs": lit[k][0], "logits": lit[k][1],
                                "probs_reference": ref_rows[k]["probs"],
                                "max_abs_dp": max(abs(a - b) for a, b in zip(lit[k][0], ref_rows[k]["probs"]))}
                               for k in lit]}
            if name == "phone":
                sec["prefix_file"] = {k: phone[T][k] for k in ("file", "sha256", "leg", "scored_record")}
            run["buckets"][str(T)] = sec
            s = sec["summary"]
            print(f"{name:30s} T{T}: rows {s['rows']} argmax {s['argmax_outside_near_tie']} max {s['max_abs_dp']:.3g} "
                  f"mean {s['mean_abs_dp']:.3g} -> {'PASS' if s['bar_pass'] else 'FAIL'}; prefix vs provider "
                  f"max {sec['prefix_vs_provider']['max_abs']:.3g} rel_rms {sec['prefix_vs_provider']['rel_rms']:.3g}",
                  flush=True)
        runs.append(run)
    gp = dict(gpu.graph_precision)
    cpu.close()
    gpu.close()
    import importlib.metadata as md

    doc = {"step": "round 13 step A-7: end to end of the audio buckets T501 and T3001 against the provider's float32 "
                   "run of the same clips",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
           "reference": {"file": str(REF.relative_to(K)), "sha256": sha_file(REF), "prefix_npz": ref["prefix_npz"]},
           "ship": {"contract_sha256": sha_file(SHIP / "contract.json"),
                    "audio_files": {T: {"file": cpu.audio_files[T].name, "sha256": sha_file(cpu.audio_files[T])}
                                    for T in (501, 3001)},
                    "decision_files": {L: {"file": cpu.text_files[L].name, "sha256": sha_file(cpu.text_files[L])}
                                       for L in (128, 512)},
                    "gpu_graph_precision": gp},
           "env": {"python": sys.version.split()[0], "ai_edge_litert": md.version("ai-edge-litert"),
                   "numpy": np.__version__},
           "bar": SS.BAR, "runs": runs}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"-> {OUT.relative_to(K)} ({doc['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
