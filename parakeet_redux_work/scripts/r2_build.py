#!/usr/bin/env python3
"""r2 files, part 1 (venv .venv-lt: litert-torch 0.9.4, torch 2.13, transformers 5.20.2): reference B's weights into
the three r2 graphs of r2_graphs.py, written as fp32 .tflite by litert_torch.convert (no quantization, no NeMo).

  out/r2/mel30.tflite           serving_default: audio [1,480000] f32, n_valid [1] i32 -> features, mel_mask, frame_mask
  out/r2/enc30_fp32_raw.tflite  serving_default: features [1,128,3000] f32, mel_mask [1,3000] f32 -> enc [1,375,640]
                                (pointwise convs as Linear, LayerNorm weight / bias as rank-3 constants; the rank-3
                                attention BATCH_MATMULs are lifted afterwards in the .tflite by r2_lift.py)
  out/r2/dec.tflite             predict: token [1,1] i32, h [2,1,640], c [2,1,640] -> dec_out [1,640], h_out, c_out
                                joint: enc_frame [1,640], dec_out [1,640] -> logits [1,8198]
  out/r2/enc30_r1.tflite        r1's encoder wrap (scripts/enc_smoke.py EncWrap: mel + valid length, masks built in the
                                graph, SELECT_V2) re-made as the comparison base of the rewrite (deleted after the check)

Before any conversion, torch checks with the real weights on the first 3 S32 30 s windows (out/s32_ii/*.npz, mode ii):
Enc30 vs B (get_audio_features with the mode-ii mask) and vs the stored r1 output, DecPredict / DecJoint vs B's
LSTM / projector / joint on 5 steps. A check above its bar stops the script before the conversion (exit 2).
Each file is skipped when it already exists (resumable). -> results/r2_build.json (appends one record per run)
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_graphs as G  # noqa: E402
import redux_ref as R  # noqa: E402

import litert_torch  # noqa: E402

OUT = "out/r2"
REC = "results/r2_build.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def torch_checks(model, enc, pred, joint):
    rep = {"enc": [], "dec": {}}
    files = sorted(glob.glob("out/s32_ii/*.npz"))[:3]
    for f in files:
        z = np.load(f)
        mel = torch.from_numpy(z["mel"])[None]
        n0 = int(z["valid_mel"])
        mm = (torch.arange(G.T_MEL)[None] < n0)
        with torch.no_grad():
            o = model.get_audio_features(input_features=mel.transpose(1, 2), attention_mask=mm,
                                         output_attention_mask=True)
            ref = o.pooler_output[0]
            ve = int(o.attention_mask.sum())
            out = enc(mel, mm.to(torch.float32))["enc"][0]
        stored = torch.from_numpy(z["enc_proj"])
        rep["enc"].append({"npz": os.path.basename(f), "valid_mel": n0, "valid_enc": ve,
                           "formula": G.enc_valid_frames(n0 * G.HOP) if n0 < G.T_MEL else None,
                           "max_abs_vs_B_now": (out[:ve] - ref[:ve]).abs().max().item(),
                           "max_abs_vs_r1_stored": (out[:ve] - stored[:ve]).abs().max().item(),
                           "B_now_vs_r1_stored": (ref[:ve] - stored[:ve]).abs().max().item(),
                           "finite": bool(torch.isfinite(out).all())})
    d = model.decoder
    g = torch.Generator().manual_seed(2)
    h = torch.zeros(2, 1, 640)
    c = torch.zeros(2, 1, 640)
    worst = {"dec_out": 0.0, "h": 0.0, "c": 0.0, "logits": 0.0}
    with torch.no_grad():
        for tok in (R.BLANK, 3, 1000, 8191, 17):
            t = torch.tensor([[tok]], dtype=torch.int32)
            o = pred(t, h, c)
            lo, (h2, c2) = d.lstm(d.embedding(t.long()), (h, c))
            ref = d.decoder_projector(lo)[:, 0]
            worst["dec_out"] = max(worst["dec_out"], (o["dec_out"] - ref).abs().max().item())
            worst["h"] = max(worst["h"], (o["h_out"] - h2).abs().max().item())
            worst["c"] = max(worst["c"], (o["c_out"] - c2).abs().max().item())
            ef = torch.randn(1, 640, generator=g)
            lg = joint(ef, ref)["logits"]
            lref = model.joint.head(model.joint.activation(ef + ref))
            worst["logits"] = max(worst["logits"], (lg - lref).abs().max().item())
            h, c = h2, c2
    rep["dec"] = worst
    bad = [r for r in rep["enc"] if r["max_abs_vs_B_now"] > 1e-4 or not r["finite"]]
    bad += [("dec", worst)] if max(worst.values()) > 1e-5 else []
    rep["pass"] = not bad
    return rep


def write(name, fn, rec):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        rec["files"][name] = {"skipped_exists": True, "bytes": os.path.getsize(path)}
        return
    t0 = time.time()
    em = fn()
    t1 = time.time()
    em.export(path)
    rec["files"][name] = {"convert_s": round(t1 - t0, 1), "write_s": round(time.time() - t1, 1),
                          "bytes": os.path.getsize(path), "sha256": sha256(path)}
    print(name, json.dumps(rec["files"][name]), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    R.set_threads(8)
    t0 = time.time()
    model, rep = R.build("B")
    proc = R.processor()
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_report": rep, "env": R.env_record(),
           "litert_torch": litert_torch.__version__, "load_s": round(time.time() - t0, 1), "files": {}}
    mel = G.Mel30(proc.feature_extractor.mel_filters).eval()
    enc = G.Enc30(model, pw_linear=True, ln_rank3=True).eval()
    pred = G.DecPredict(model).eval()
    joint = G.DecJoint(model).eval()
    rec["torch_checks"] = torch_checks(model, enc, pred, joint)
    print("torch checks", json.dumps(rec["torch_checks"]), flush=True)
    if not rec["torch_checks"]["pass"]:
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("TORCH CHECK FAIL: no conversion", flush=True)
        return 2
    zi = torch.tensor([G.N_AUDIO - 160], dtype=torch.int32)
    write("mel30.tflite", lambda: litert_torch.convert(
        mel, sample_kwargs={"audio": torch.zeros(1, G.N_AUDIO), "n_valid": zi}, enable_x64=False), rec)
    write("dec.tflite", lambda: litert_torch.signature(
        "predict", pred, sample_kwargs={"token": torch.tensor([[R.BLANK]], dtype=torch.int32),
                                        "h": torch.zeros(2, 1, 640), "c": torch.zeros(2, 1, 640)}).signature(
        "joint", joint, sample_kwargs={"enc_frame": torch.zeros(1, 640), "dec_out": torch.zeros(1, 640)}).convert(
        enable_x64=False), rec)
    write("enc30_fp32_raw.tflite", lambda: litert_torch.convert(
        enc, sample_kwargs={"features": torch.zeros(1, 128, G.T_MEL), "mel_mask": torch.ones(1, G.T_MEL)},
        enable_x64=False), rec)
    del enc
    import enc_smoke
    wrap = enc_smoke.EncWrap(model).eval()
    write("enc30_r1.tflite", lambda: litert_torch.convert(
        wrap, (torch.zeros(1, 128, G.T_MEL), torch.tensor([G.T_MEL], dtype=torch.int32)), enable_x64=False), rec)
    rec["total_s"] = round(time.time() - t0, 1)
    with open(REC, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("build done", rec["total_s"], "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
