"""Quantised variants of the fp32 audio encoder tflite (ai-edge-quantizer 0.9.0), gated on the 25 fixtures.

Variants (from out/audio_encoder/audio_encoder_504f_fp32.tflite):
  wi8fc       FULLY_CONNECTED dynamic int8 (weights int8 per-channel, activations quantised on the fly), EXCEPT the
              three constant frontend matmuls (DFT cos [512,257], DFT sin [512,257], mel [257,80]) which stay fp32.
              The exclusion is a negative look-ahead on the exact op scopes (= output tensor names + ";", the
              quantizer's scope rule) of those three ops, identified below from the fp32 file by weight shape AND
              by value (weights bit-equal to the port's dft_cos.T / dft_sin.T / mel_t.T).
  wi8fc_noadp wi8fc that also keeps the 14 adaptor FULLY_CONNECTED ops fp32 (only built if wi8fc misses the oracle).
  fp16        float casting of every supported weight to fp16 (FC / DEPTHWISE_CONV_2D here), explicit DEQUANTIZE,
              compute fp32 (the vibevoice_asr_work fp16 block).
Per variant: post-quantisation weight dtype audit of every FC op, op histogram, signature I/O (asserted equal to
the fp32 file), size, parity vs the fp32 tflite on the 25 runtime windows (features[:vt] max|diff| / cos, vt = the
fp32 file's valid count; mask equality), transcripts from the variant's features + lm_native greedy (port_eval.Greedy,
same loop as round 1) vs the funasr oracle (text, token ids, en WER), first / warm invoke ms (8 threads).
Selection rule (supervisor): ship candidate = the smallest variant whose transcripts equal the oracle 25/25 with the
same en WER.
Run with ~/venvs/lt094dev/bin/python. Writes out/audio_encoder/audio_encoder_504f_<variant>.tflite and
encoder_variants.json (+ quant_encoder.log via the caller's redirect).
"""
import argparse
import collections
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import funasr_port as P  # noqa: E402
from port_eval import Greedy  # noqa: E402

ENC_DIR = os.path.join(C.OUT, "audio_encoder")
FP32 = os.path.join(ENC_DIR, "audio_encoder_504f_fp32.tflite")
FRONT = "KaldiFbankLfrGraph_frontend"
ADAPTOR = "AudioAdaptor_audio_adaptor"


def path_of(v):
    return os.path.join(ENC_DIR, f"audio_encoder_504f_{v}.tflite")


def interp(path, threads=None):
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter(model_path=path, num_threads=threads) if threads else Interpreter(model_path=path)


def fc_table(path):
    """Every FULLY_CONNECTED op: index, scope (quantizer rule: output tensor names joined by ';' + ';'), weight
    tensor name / shape / dtype, and the weight values for the float ones we need to identify."""
    it = interp(path)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    # DELEGATE nodes (XNNPACK, present after allocate_tensors) list the dequantized weights among their outputs; keep
    # the real producers only (run 2 read 294 fp16 weights as float32 through this).
    producer = {int(i): o for o in ops if o["op_name"] != "DELEGATE" for i in o["outputs"]}
    rows = []
    for o in ops:
        if o["op_name"] != "FULLY_CONNECTED":
            continue
        outs = [td[i]["name"] for i in o["outputs"] if i >= 0 and td[i]["name"]]
        w = td[o["inputs"][1]]
        dtype = np.dtype(w["dtype"]).name
        src = producer.get(int(w["index"]))
        if src is not None and src["op_name"] == "DEQUANTIZE":   # fp16 casting: FC reads DEQUANTIZE(fp16 const)
            dtype = np.dtype(td[src["inputs"][0]]["dtype"]).name + "->dequantize"
        rows.append({"op_index": int(o["index"]), "scope": ";".join(outs) + ";", "weight_name": w["name"],
                     "weight_index": int(w["index"]), "weight_shape": [int(s) for s in w["shape"]],
                     "weight_dtype": dtype})
    return it, rows


def op_hist(path):
    it = interp(path)
    ops = collections.Counter(d["op_name"] for d in it._get_ops_details())
    return dict(sorted(ops.items(), key=lambda kv: -kv[1]))


def identify_frontend(it, fcs):
    """Return {role: row} for dft_cos / dft_sin / mel, asserting shape and bit-equal values."""
    g = P.KaldiFbankLfrGraph(P.WIN_SAMPLES)
    refs = {"dft_cos": g.dft_cos.T.contiguous().numpy(), "dft_sin": g.dft_sin.T.contiguous().numpy(),
            "mel": g.mel_t.T.contiguous().numpy()}
    found = {}
    for r in fcs:
        if tuple(r["weight_shape"]) not in [(257, 512), (80, 257)]:
            continue
        w = it.get_tensor(r["weight_index"])
        for role, ref in refs.items():
            if w.shape == ref.shape and np.array_equal(w, ref):
                assert role not in found, (role, r)
                found[role] = dict(r, value_check="bit-equal to funasr_port." + {"dft_cos": "dft_cos.T", "dft_sin": "dft_sin.T", "mel": "mel_t.T"}[role])
    assert set(found) == set(refs), found.keys()
    front_scopes = [r["scope"] for r in fcs if FRONT in r["scope"]]
    assert sorted(front_scopes) == sorted(f["scope"] for f in found.values()), (front_scopes, found)
    return found


def adaptor_rows(fcs):
    rows = [r for r in fcs if ADAPTOR in r["scope"]]
    assert len(rows) == 14, [r["scope"][:120] for r in rows]
    return rows


def negative_regex(scopes):
    return r"^(?!(?:" + "|".join(re.escape(s) for s in scopes) + r")$).*"


def build(variant, keep_fp32_scopes):
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    OP = qtyping.TFLOperationName
    out = path_of(variant)
    if os.path.exists(out):
        os.remove(out)
    rm = recipe_manager.RecipeManager()
    if variant.startswith("wi8fc"):
        regex = negative_regex(keep_fp32_scopes)
        rm.add_dynamic_config(regex=regex, operation_name=OP.FULLY_CONNECTED, num_bits=8)
    elif variant == "fp16":
        # algorithm_key must be FLOAT_CASTING: with the default (MIN_MAX_UNIFORM_QUANT) this 16-bit FLOAT config fails
        # the per-op check inside get_quantization_configs, every op falls back to NO_QUANTIZE and the "fp16" file is
        # the fp32 file (measured in run 1: 896.69 MiB -> 896.69 MiB, quant_encoder_run1.log).
        from ai_edge_quantizer import algorithm_manager
        regex = ".*"
        rm.add_quantization_config(
            regex=regex, operation_name=OP.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
            algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
    else:
        raise ValueError(variant)
    recipe = rm.get_quantization_recipe()
    qt = quantizer.Quantizer(FP32, recipe)
    assert not qt.need_calibration
    t0 = time.time()
    qt.quantize().export_model(out)
    dt = time.time() - t0
    print(f"[{variant}] {out} {os.path.getsize(out):,} B in {dt:.0f}s", flush=True)
    return out, recipe, regex, dt


def audit(variant, path, fcs_fp32, keep_fp32_scopes):
    """Weight dtype of every FC op after quantisation, matched to the fp32 table by op order."""
    _, fcs = fc_table(path)
    assert len(fcs) == len(fcs_fp32), (len(fcs), len(fcs_fp32))
    dt = collections.Counter()
    kept, wrong = [], []
    for a, b in zip(fcs_fp32, fcs):
        assert a["scope"] == b["scope"], (a["scope"], b["scope"])
        dt[b["weight_dtype"]] += 1
        if variant.startswith("wi8fc"):
            want = "float32" if a["scope"] in keep_fp32_scopes else "int8"
        else:
            want = "float16->dequantize"
        if b["weight_dtype"] != want:
            wrong.append({"scope": a["scope"][:200], "dtype": b["weight_dtype"], "want": want})
        if b["weight_dtype"] == "float32":
            kept.append(a["scope"])
    return {"fc_weight_dtypes": dict(dt), "fc_kept_fp32": kept, "fc_dtype_mismatch": wrong}


def sig_io(run):
    ins, outs = run.get_input_details(), run.get_output_details()
    return {"inputs": {k: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for k, v in ins.items()},
            "outputs": {k: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for k, v in outs.items()}}


def valid_count(mask_row):
    nz = np.nonzero(mask_row)[0]
    return int(nz[-1]) + 1 if len(nz) else 0


def cmp(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "rel": float(d.max() / (np.abs(b).max() + 1e-12)),
            "cos": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))}


def run_variant(name, path, windows, ref_feats, ref_masks, lm, oracle, meta, threads):
    it = interp(path, threads)
    assert list(it.get_signature_list()) == ["encode"], it.get_signature_list()
    run = it.get_signature_runner("encode")
    io = sig_io(run)
    assert io == {"inputs": {"audio": {"dtype": "float32", "shape": [1, 504, 960]}},
                  "outputs": {"features": {"dtype": "float32", "shape": [1, 63, 1024]}, "mask": {"dtype": "uint8", "shape": [1, 63]}}}, io
    rows, trs, warm = [], [], []
    first = None
    feats_all, masks_all = {}, {}
    for m in meta:
        fid = m["id"]
        t0 = time.perf_counter()
        out = run(audio=windows[fid])
        ms = (time.perf_counter() - t0) * 1e3
        if first is None:
            first = ms
        else:
            warm.append(ms)
        feats, mask = out["features"].copy(), out["mask"].copy()
        feats_all[fid], masks_all[fid] = feats, mask
        vt = valid_count(mask[0])
        r = {"id": fid, "valid_tokens": vt, "invoke_ms": round(ms, 1)}
        if ref_feats is not None:
            vref = valid_count(ref_masks[fid][0])
            r["valid_tokens_fp32"] = vref
            r["mask_equal_fp32"] = bool(np.array_equal(mask, ref_masks[fid]))
            r["vs_fp32"] = cmp(feats[0, :vref], ref_feats[fid][0, :vref])
        rows.append(r)
    # timing on one fixed window
    same = []
    w0 = windows[meta[0]["id"]]
    for _ in range(5):
        t0 = time.perf_counter()
        run(audio=w0)
        same.append((time.perf_counter() - t0) * 1e3)
    timing = {"threads": threads, "first_invoke_ms": round(first, 1),
              "warm_invoke_ms_median_over_clips": round(float(np.median(warm)), 1),
              "warm_invoke_ms_same_window_x5": [round(x, 1) for x in same],
              "warm_invoke_ms_same_window_median": round(float(np.median(same)), 1)}
    del run, it
    gc.collect()
    for m in meta:
        fid = m["id"]
        vt = valid_count(masks_all[fid][0])
        o = oracle[fid]
        ids, text, text_tn = lm(torch.from_numpy(feats_all[fid][0, :vt]))
        trs.append({"id": fid, "oracle_text": o["text"], "text": text, "text_tn": text_tn, "gen_ids": ids,
                    "text_equal": text == o["text"], "gen_ids_equal": ids == o["gen_ids"], "valid_tokens": vt})
        r = next(x for x in rows if x["id"] == fid)
        par = r.get("vs_fp32")
        print(f"[{name}] {fid:12s} vt {vt} " + (f"mask== {r['mask_equal_fp32']} max {par['max_abs']:.3e} cos {par['cos']:.7f} " if par else "")
              + f"{r['invoke_ms']:.0f} ms | {'==' if trs[-1]['text_equal'] and trs[-1]['gen_ids_equal'] else '!='} {text}"
              + ("" if trs[-1]["text_equal"] else f"\n      oracle: {o['text']}"), flush=True)
    wer = C.corpus_wer(trs, {m["id"]: m for m in meta})
    summ = {"file": os.path.relpath(path, C.WORK), "size_bytes": os.path.getsize(path), "io": io, "timing": timing,
            "text_match": sum(t["text_equal"] for t in trs), "gen_ids_match": sum(t["gen_ids_equal"] for t in trs),
            "n": len(trs), "wer_en": {k: wer[k] for k in ["errors", "words", "wer"]}, "wer_en_per_clip": wer["per_clip"],
            "mismatches": [{"id": t["id"], "variant": t["text"], "oracle": t["oracle_text"]} for t in trs if not t["text_equal"]]}
    if ref_feats is not None:
        summ["parity_vs_fp32"] = {
            "mask_equal": sum(r["mask_equal_fp32"] for r in rows),
            "valid_tokens_equal": sum(r["valid_tokens"] == r["valid_tokens_fp32"] for r in rows),
            "worst_max_abs": max(r["vs_fp32"]["max_abs"] for r in rows),
            "worst_rel": max(r["vs_fp32"]["rel"] for r in rows),
            "worst_cos": min(r["vs_fp32"]["cos"] for r in rows),
            "median_max_abs": float(np.median([r["vs_fp32"]["max_abs"] for r in rows])),
            "median_cos": float(np.median([r["vs_fp32"]["cos"] for r in rows]))}
    return summ, rows, trs, feats_all, masks_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="wi8fc,fp16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    rep = {"fp32_file": os.path.relpath(FP32, C.WORK), "fp32_size_bytes": os.path.getsize(FP32)}
    import ai_edge_quantizer
    import ai_edge_litert
    rep["versions"] = {"ai_edge_quantizer": getattr(ai_edge_quantizer, "__version__", None),
                       "ai_edge_litert": getattr(ai_edge_litert, "__version__", None), "torch": torch.__version__}
    try:
        from importlib.metadata import version
        rep["versions"]["ai_edge_quantizer_dist"] = version("ai-edge-quantizer")
        rep["versions"]["ai_edge_litert_dist"] = version("ai-edge-litert")
    except Exception as e:  # noqa: BLE001
        rep["versions"]["dist_error"] = repr(e)

    it32, fcs32 = fc_table(FP32)
    front = identify_frontend(it32, fcs32)
    adp = adaptor_rows(fcs32)
    del it32
    gc.collect()
    rep["fp32_fc_ops"] = len(fcs32)
    rep["frontend_fc"] = {k: {kk: v[kk] for kk in ["op_index", "scope", "weight_name", "weight_shape", "weight_dtype", "value_check"]}
                          for k, v in front.items()}
    rep["adaptor_fc_scopes"] = [r["scope"] for r in adp]
    print("frontend FC ops:", json.dumps(rep["frontend_fc"], indent=1), flush=True)
    front_scopes = [v["scope"] for v in front.values()]
    keep = {"wi8fc": front_scopes, "wi8fc_noadp": front_scopes + [r["scope"] for r in adp], "fp16": []}

    meta = C.load_meta()
    oracle = {r["id"]: r for r in json.load(open(os.path.join(C.WORK, "oracle_transcripts.json")))["rows"]}
    windows = {m["id"]: P.frame_window(P.read_wav_i16(os.path.join(C.WORK, m["file"]))).numpy() for m in meta}
    lm = Greedy(os.path.join(C.OUT, "lm_native"))

    results = {}
    print("=== fp32 (reference) ===", flush=True)
    s32, rows32, trs32, f32, m32 = run_variant("fp32", FP32, windows, None, None, lm, oracle, meta, args.threads)
    s32["ops"] = op_hist(FP32)
    results["fp32"] = {"summary": s32, "rows": rows32, "transcripts": trs32}

    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        print(f"=== {v} ===", flush=True)
        entry = {}
        if args.skip_build and os.path.exists(path_of(v)):
            path = path_of(v)
            entry["build"] = {"skipped": True}
        else:
            path, recipe, regex, dt = build(v, keep[v])
            entry["build"] = {"recipe": recipe, "regex": regex, "quantize_s": round(dt, 1),
                              "kept_fp32_scopes_requested": keep[v]}
        entry["audit"] = audit(v, path, fcs32, set(keep[v]))
        entry["ops"] = op_hist(path)
        print(f"[{v}] audit: {json.dumps({k: entry['audit'][k] for k in ['fc_weight_dtypes', 'fc_dtype_mismatch']})}", flush=True)
        assert not entry["audit"]["fc_dtype_mismatch"], entry["audit"]["fc_dtype_mismatch"]
        s, rows, trs, _, _ = run_variant(v, path, windows, f32, m32, lm, oracle, meta, args.threads)
        s["ops"] = entry["ops"]
        entry.update({"summary": s, "rows": rows, "transcripts": trs})
        results[v] = entry
        gc.collect()

    base_wer = results["fp32"]["summary"]["wer_en"]["errors"]
    ok = {k: (r["summary"]["text_match"] == r["summary"]["n"] and r["summary"]["gen_ids_match"] == r["summary"]["n"]
              and r["summary"]["wer_en"]["errors"] == base_wer) for k, r in results.items()}
    cands = sorted([k for k in results if k != "fp32" and ok[k]], key=lambda k: results[k]["summary"]["size_bytes"])
    table = [{"variant": k, "size_bytes": r["summary"]["size_bytes"], "text_match": r["summary"]["text_match"],
              "gen_ids_match": r["summary"]["gen_ids_match"], "wer_en_errors": r["summary"]["wer_en"]["errors"],
              "wer_en_words": r["summary"]["wer_en"]["words"],
              "mask_equal": r["summary"].get("parity_vs_fp32", {}).get("mask_equal"),
              "worst_max_abs_vs_fp32": r["summary"].get("parity_vs_fp32", {}).get("worst_max_abs"),
              "worst_cos_vs_fp32": r["summary"].get("parity_vs_fp32", {}).get("worst_cos"),
              "warm_ms": r["summary"]["timing"]["warm_invoke_ms_median_over_clips"],
              "meets_rule": ok[k]} for k, r in results.items()]
    rep["table"] = table
    rep["selection_rule"] = "smallest variant with transcripts (text and token ids) == oracle 25/25 and en WER == fp32/oracle"
    rep["ship_candidate"] = cands[0] if cands else None
    with open(os.path.join(C.WORK, "encoder_variants.json"), "w") as f:
        json.dump({"report": rep, "variants": results}, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(table, indent=1), flush=True)
    print("ship candidate:", rep["ship_candidate"], flush=True)
    print("QUANT_DONE", flush=True)


if __name__ == "__main__":
    main()
