"""Export FunAsrNanoAudioEncoder (funasr_port.py) as ONE single-signature fp32 .tflite for LiteRT-LM's
generic audio path, then gate it on CPU:

  signature `encode`
    input  `audio`    f32   [1, 504, 960]  raw 16 kHz PCM in [-1, 1), frame = hop = 960 (60 ms), 30.24 s
    output `features` f32   [1, 63, 1024]  adaptor rows 0..62 (audio_shrink_factor = 504 // 63 = 8)
    output `mask`     uint8 [1, 63]        1 for rows < fake_token_len = ceil(L / 8) (runtime GetValidCount)

Gate: op histogram (FLEX / CUSTOM must be 0), signature I/O names / dtypes / shapes, parity vs the port eager
on the same zero-padded windows (all 25 fixtures), first / warm invoke ms (8 threads), and the 25 transcripts
from the tflite features + lm_native greedy (same loop as port_eval.py) vs the funasr oracle.
Run with ~/venvs/lt094dev/bin/python. Writes out/audio_encoder/audio_encoder_504f_fp32.tflite,
tflite_parity.json, tflite_transcripts.json.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import funasr_port as P  # noqa: E402
from port_eval import Greedy, cmp  # noqa: E402

OUT_DIR = os.path.join(C.OUT, "audio_encoder")
TFL = os.path.join(OUT_DIR, "audio_encoder_504f_fp32.tflite")


def export(enc):
    import litert_torch
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = {"audio": torch.zeros(1, P.WIN_FRAMES, P.FRAME)}
    t0 = time.time()
    edge = litert_torch.signature("encode", enc, sample_kwargs=sample).convert()
    edge.export(TFL)
    dt = time.time() - t0
    print(f"exported {TFL} {os.path.getsize(TFL) / 1e6:.1f} MB in {dt:.0f}s", flush=True)
    return dt


def op_hist(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    ops = collections.Counter(d["op_name"] for d in it._get_ops_details())
    return dict(sorted(ops.items(), key=lambda kv: -kv[1]))


def valid_count(mask_row):
    """LiteRT-LM GetValidCount: index of the last non-zero element + 1 (0 if none)."""
    nz = np.nonzero(mask_row)[0]
    return int(nz[-1]) + 1 if len(nz) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(8)
    enc, n_t = P.load_encoder(os.path.join(C.OUT, "hf_vllm", "model.safetensors"))
    rep = {"encoder_tensors_loaded": n_t, "params": int(sum(p.numel() for p in enc.parameters()))}
    if not args.skip_export or not os.path.exists(TFL):
        rep["export_s"] = round(export(enc), 1)
    rep["file"] = os.path.relpath(TFL, C.WORK)
    rep["size_bytes"] = os.path.getsize(TFL)
    ops = op_hist(TFL)
    rep["ops"] = ops
    rep["n_ops"] = sum(ops.values())
    rep["flex_or_custom"] = {k: v for k, v in ops.items() if "FLEX" in k.upper() or "CUSTOM" in k.upper()}
    print("ops:", ops, flush=True)

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=TFL, num_threads=args.threads)
    sigs = it.get_signature_list()
    rep["signatures"] = sigs
    assert list(sigs) == ["encode"], sigs
    run = it.get_signature_runner("encode")
    ins, outs = run.get_input_details(), run.get_output_details()
    io = {"inputs": {k: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for k, v in ins.items()},
          "outputs": {k: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for k, v in outs.items()}}
    rep["io"] = io
    print("io:", io, flush=True)
    assert io["inputs"] == {"audio": {"dtype": "float32", "shape": [1, 504, 960]}}, io
    assert io["outputs"] == {"features": {"dtype": "float32", "shape": [1, 63, 1024]},
                             "mask": {"dtype": "uint8", "shape": [1, 63]}}, io

    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    oracle = {r["id"]: r for r in json.load(open(os.path.join(C.WORK, "oracle_transcripts.json")))["rows"]}
    lm = Greedy(os.path.join(C.OUT, "lm_native"))
    rows, trs, warm_ms = [], [], []
    first_ms = None
    for i, row in enumerate(meta):
        fid = row["id"]
        wav = P.read_wav_i16(os.path.join(C.WORK, row["file"]))
        audio = P.frame_window(wav)
        with torch.no_grad():
            ref = enc.forward_debug(audio)
        t0 = time.perf_counter()
        out = run(audio=audio.numpy())
        ms = (time.perf_counter() - t0) * 1e3
        if first_ms is None:
            first_ms = ms
        else:
            warm_ms.append(ms)
        feats, mask = out["features"], out["mask"]
        vt = valid_count(mask[0])
        ref_mask = ref["mask"].numpy()
        ftl_eager = int(ref["ftl"].item())
        o = oracle[fid]
        d_adp = np.load(os.path.join(C.OUT, "oracle_dumps", f"{fid}_adp.npy"))
        r = {"id": fid, "n_samples": len(wav), "valid_tokens": vt, "ftl_eager": ftl_eager, "ftl_oracle": o["fake_token_len"],
             "mask_equal_eager": bool(np.array_equal(mask, ref_mask)), "mask_dtype": str(mask.dtype),
             "vs_eager": cmp(feats[0, :vt], ref["features"][0, :vt].numpy()),
             "vs_eager_all63": cmp(feats[0], ref["features"][0].numpy()),
             "vs_oracle_ftl": cmp(feats[0, :min(vt, o['fake_token_len'])], d_adp[0, :min(vt, o['fake_token_len'])]),
             "invoke_ms": round(ms, 1)}
        rows.append(r)
        ids, text, text_tn = lm(torch.from_numpy(feats[0, :vt]))
        tr = {"id": fid, "oracle_text": o["text"], "text": text, "text_tn": text_tn, "gen_ids": ids,
              "text_equal": text == o["text"], "gen_ids_equal": ids == o["gen_ids"], "valid_tokens": vt}
        trs.append(tr)
        print(f"{fid:12s} vt {vt}/{o['fake_token_len']} mask== {r['mask_equal_eager']} | vs eager max {r['vs_eager']['max_abs']:.2e} "
              f"cos {r['vs_eager']['cos']:.8f} | vs oracle max {r['vs_oracle_ftl']['max_abs']:.2e} | {ms:.0f} ms | "
              f"{'==' if tr['text_equal'] and tr['gen_ids_equal'] else '!='} {text}", flush=True)

    # warm timing on one fixed window (5 more invokes)
    audio = P.frame_window(P.read_wav_i16(os.path.join(C.WORK, meta[0]["file"]))).numpy()
    rep_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        run(audio=audio)
        rep_ms.append((time.perf_counter() - t0) * 1e3)
    rep["timing"] = {"threads": args.threads, "first_invoke_ms": round(first_ms, 1),
                     "warm_invoke_ms_median_over_clips": round(float(np.median(warm_ms)), 1) if warm_ms else None,
                     "warm_invoke_ms_same_window_x5": [round(x, 1) for x in rep_ms],
                     "warm_invoke_ms_same_window_median": round(float(np.median(rep_ms)), 1)}
    rep["parity_summary"] = {
        "n": len(rows), "mask_equal_eager": sum(r["mask_equal_eager"] for r in rows),
        "valid_tokens_eq_oracle_ftl": sum(r["valid_tokens"] == r["ftl_oracle"] for r in rows),
        "vs_eager_ok": sum(r["vs_eager"]["ok"] for r in rows),
        "vs_eager_worst_max_abs": max(r["vs_eager"]["max_abs"] for r in rows),
        "vs_eager_worst_rel": max(r["vs_eager"]["rel"] for r in rows),
        "vs_eager_worst_cos": min(r["vs_eager"]["cos"] for r in rows),
        "vs_oracle_ok": sum(r["vs_oracle_ftl"]["ok"] for r in rows),
        "vs_oracle_fail_ids": [r["id"] for r in rows if not r["vs_oracle_ftl"]["ok"]],
    }
    with open(os.path.join(C.WORK, "tflite_parity.json"), "w") as f:
        json.dump({"report": rep, "rows": rows}, f, ensure_ascii=False, indent=1)
    by_id = {r["id"]: r for r in meta}
    wer = C.corpus_wer(trs, by_id)
    ts = {"text_match": sum(t["text_equal"] for t in trs), "gen_ids_match": sum(t["gen_ids_equal"] for t in trs), "n": len(trs),
          "wer_en": {k: wer[k] for k in ["errors", "words", "wer"]},
          "mismatches": [{"id": t["id"], "tflite": t["text"], "oracle": t["oracle_text"]} for t in trs if not t["text_equal"]]}
    with open(os.path.join(C.WORK, "tflite_transcripts.json"), "w") as f:
        json.dump({"summary": ts, "rows": trs}, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ["size_bytes", "n_ops", "flex_or_custom", "timing", "parity_summary"]}, indent=1), flush=True)
    print(json.dumps(ts, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
