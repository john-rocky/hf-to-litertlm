"""Round 7 checks of the d1-omni audio prefix graph against the provider's audio.py (pinned snapshot 414f8d64).

    cd d1_omni_work; Q=~/code/standup/tools/quiet
    venv-ref/bin/python scripts/audio_check.py --mel                          # step 1 -> results/audio_mel_check.json
    $Q/quiet_wait.py -- venv-ref/bin/python scripts/audio_check.py --eager    # step 2 -> results/audio_eager_check.json

Clips = the 7 audio records of the oracle (ref/records_ref.json v2): aud_01..03 (d1d's Kokoro clips), aud_reservation_01
/ aud_weather_02 / aud_food_03 (ours), card_topic (LibriSpeech flac, measurement only). Their npz (ref/npz/<id>.npz)
hold the provider's MelFrontend output `mel` [128, T] + `frames`, and the Audio module's output `prefix` [P, 1024].

--mel: the host's numpy front end (host/d1_audio_host.py, precision float32 and float64) against the provider's:
  the npz mel (the oracle run), a fresh provider MelFrontend run (bit-equal to the npz = same reader and torch), and the
  provider's own forward body re-run in float64 (the same torch ops with every tensor in float64; the fp32 Hann window
  and filterbank values cast up) = the provider's fp32 rounding floor. Stage by stage for every clip (preemphasis,
  |X|^2, mel, log, mean / std, output), worst (bin, frame), elements and frames above 1e-3. Plus the window, the
  filterbank, waveform() and the clip reader against the provider's, and the lengths / masks the host builds.
--eager: D1Audio (scripts/audio_graph.py, checkpoint weights, fp32 CPU, torch threads 12 like the oracle) at each
  clip's bucket vs the provider's Audio (same checkpoint, strict load) — see run_eager().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "host"))
import d1_src as S  # noqa: E402  (stdlib only)
import d1_audio_host as AH  # noqa: E402  (numpy only)

RECORDS = ("aud_01", "aud_02", "aud_03", "aud_reservation_01", "aud_weather_02", "aud_food_03", "card_topic")
MEL_BAR = 1e-3
THREADS = 12


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, doc):
    import os

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def provider_audio():
    return S.load_module("d1_provider_audio", S.SNAP / "audio.py")


def npz_of(rid):
    with np.load(K / f"ref/npz/{rid}.npz") as z:
        return {k: np.asarray(z[k]) for k in ("mel", "frames", "prefix")}


def diff(a, b, mask=None):
    """max / mean |a - b| (float64), the worst index, relative to max |b|; mask selects elements (bool, same shape)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    if mask is not None:
        d = np.where(mask, d, 0.0)
    i = np.unravel_index(int(np.argmax(d)), d.shape)
    ref = float(np.abs(b if mask is None else np.where(mask, b, 0.0)).max())
    return {"max_abs": float(d.max()), "mean_abs": float(d.sum() / (mask.sum() if mask is not None else d.size)),
            "at": [int(v) for v in i], "a_at": float(a[i]), "b_at": float(b[i]), "ref_absmax": ref,
            "max_rel_to_ref_absmax": float(d.max() / ref) if ref else None,
            "bit_equal": bool(np.array_equal(np.asarray(a), np.asarray(b))) if mask is None else
            bool((d == 0).all())}


# ---------------------------------------------------------------- step 1: mel

def provider_stages(A, wav, dtype, pre_f32=False):
    """The provider's MelFrontend.forward body, op for op, with every tensor in `dtype` (float32 = the provider).
    pre_f32: the preemphasis in float32 (as the provider and the host take it), then cast to `dtype`."""
    import torch

    x = wav.to(torch.float32 if pre_f32 else dtype)
    n = torch.tensor([x.shape[1]])
    frames = torch.floor_divide(n + 512 // 2 * 2 - 512, 160)
    pre = torch.cat((x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]), dim=1).to(dtype)
    window = torch.hann_window(400, periodic=False).to(dtype)
    spec = torch.stft(pre, n_fft=512, hop_length=160, win_length=400, center=True, window=window,
                      return_complex=True, pad_mode="constant")
    power = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1)).pow(2.0)
    fb = torch.from_numpy(A.slaney_filterbank(n_fft=512, n_mels=128))[None].to(dtype)
    lin = torch.matmul(fb, power)
    logm = torch.log(lin + 2 ** -24)
    valid = torch.arange(logm.shape[2])[None] < frames[:, None]
    count = valid.sum(1)
    mean = torch.where(valid[:, None], logm, 0.0).sum(2) / count[:, None]
    std = torch.sqrt(torch.where(valid[:, None], logm - mean[:, :, None], 0.0).pow(2).sum(2) / (count[:, None] - 1.0))
    out = (logm - mean[:, :, None]) / (std.masked_fill(std.isnan(), 0.0) + 1e-5)[:, :, None]
    out = out.masked_fill(~valid[:, None], 0.0)
    return {"pre": pre[0].numpy(), "power": power[0].numpy(), "lin": lin[0].numpy(), "log": logm[0].numpy(),
            "mean": mean[0].numpy(), "std": std[0].numpy(), "out": out[0].numpy(), "frames": int(frames[0])}


def frames_over(a, b, bar):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    over = d > bar
    return {"elements_over": int(over.sum()), "frames_over": int(over.any(0).sum()),
            "frame_list": [int(t) for t in np.nonzero(over.any(0))[0][:20]]}


def run_mel(a):
    import soundfile as sf
    import torch

    import audio_graph as G

    torch.set_num_threads(THREADS)
    A = provider_audio()
    t0 = time.time()
    w_t = torch.hann_window(400, periodic=False).numpy()
    w_h = AH.hann_window_f32()
    fb_p = A.slaney_filterbank()
    fb_h = AH.slaney_filterbank()
    doc = {"step": "round 7 step 1: host mel (host/d1_audio_host.py) vs the provider's MelFrontend", "written": now(),
           "provider_audio_py_sha256": S.sha256_file(S.SNAP / "audio.py"),
           "host_sha256": S.sha256_file(K / "host/d1_audio_host.py"),
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "soundfile": sf.__version__,
                        "python": sys.version.split()[0], "torch_threads": torch.get_num_threads()},
           "bar_elementwise": MEL_BAR,
           "window": {"host_vs_torch_hann_window": diff(w_h, w_t),
                      "differing_samples": int((w_h != w_t).sum())},
           "filterbank": {"host_vs_provider": diff(fb_h, fb_p), "shape": list(fb_h.shape), "dtype": str(fb_h.dtype)},
           "formulas": {
               "frames": "frames = (n + 512 // 2 * 2 - 512) // 160 = n // 160 (valid frames); T = n // 160 + 1 "
                         "(STFT frames, center=True); the last STFT frame is never valid",
               "bucket": "T_b = the smallest of (501, 1001, 2001, 3001) >= T; mel zero-padded on the right to T_b",
               "lengths": "L1 = (frames + 2 - 3) // 2 + 1, L2 = (L1 - 1) // 2 + 1, L3 = (L2 - 1) // 2 + 1 = P",
               "dims": "T1 = (T_b - 1) // 2 + 1, T2, T3 likewise (501 -> 251 / 126 / 63, 1001 -> 501 / 251 / 126, "
                       "2001 -> 1001 / 501 / 251, 3001 -> 1501 / 751 / 376)",
               "masks": "mel_valid[t] = t < frames (t < T_b); v1[t] = t < L1 (t < T1); v2[t] = t < L2; v3[t] = t < L3; "
                        "float32 0 / 1, shape [1, T_k]"},
           "clips": {}}
    worst = {"host_f32_vs_npz": 0.0, "host_f64_vs_npz": 0.0, "host_f64_vs_provider_f64": 0.0,
             "provider_f64_vs_npz": 0.0, "identity_host_f64_vs_provider_f64": 0.0}
    over = {"host_f32_vs_npz": 0, "host_f64_vs_npz": 0}
    for rid in RECORDS:
        z = npz_of(rid)
        x16, path = G.clip_samples(rid)
        x_sf, rate = sf.read(str(K / path), dtype="int16")
        wav_p = A.waveform(x_sf)                                   # torch [1, N]
        wav_h = AH.waveform(x16)
        fe = A.MelFrontend(128)
        with torch.no_grad():
            mel_p, frames_p = fe(wav_p)
            st32 = provider_stages(A, wav_p, torch.float32)
            st64 = provider_stages(A, wav_p, torch.float64)
            st64i = provider_stages(A, wav_p, torch.float64, pre_f32=True)
        h32 = AH.mel_stages(wav_h, "float32")
        h64 = AH.mel_stages(wav_h, "float64")
        h64i = AH.mel_stages(wav_h, "float64", window=w_t)
        T = z["mel"].shape[1]
        frames = int(z["frames"][0])
        valid = np.zeros((128, T), bool)
        valid[:, :frames] = True
        P = int(z["prefix"].shape[0])
        L = AH.lengths(frames)
        e = {"file": path, "samples": int(x16.shape[0]), "seconds": round(x16.shape[0] / 16000, 4),
             "reader_stdlib_wave_or_soundfile_equal_soundfile_int16": bool(np.array_equal(x16, x_sf)),
             "waveform_host_equal_provider": bool(np.array_equal(wav_h, wav_p[0].numpy())),
             "frames_npz": frames, "frames_host": h32["frames"], "frames_provider_now": int(frames_p[0]),
             "T_npz": T, "T_host": h32["T"], "frames_equal": frames == h32["frames"] == int(frames_p[0]),
             "T_equal": T == h32["T"], "bucket": AH.bucket_for(T), "L123_host": list(L), "P_npz_prefix_rows": P,
             "P_equal": L[2] == P,
             "provider_rerun_equal_npz_mel": bool(np.array_equal(mel_p[0].numpy(), z["mel"])),
             "provider_body_f32_equal_provider_module": bool(np.array_equal(st32["out"], mel_p[0].numpy())),
             "invalid_frames_zero": {"npz": bool((z["mel"][:, frames:] == 0).all()),
                                     "host_f32": bool((h32["out"][:, frames:] == 0).all()),
                                     "host_f64": bool((h64["out"][:, frames:] == 0).all())},
             "out": {"host_f32_vs_npz": diff(h32["out"], z["mel"]),
                     "host_f64_vs_npz": diff(h64["out"].astype(np.float32), z["mel"]),
                     "host_f64_vs_provider_f64": diff(h64["out"], st64["out"]),
                     "provider_f64_vs_npz": diff(st64["out"].astype(np.float32), z["mel"]),
                     "identity_host_f64_vs_provider_f64": diff(h64i["out"], st64i["out"])},
             "identity_note": "host float64 with torch's fp32 window vs the provider's body in float64 with the "
                              "preemphasis in float32 (both inputs equal): every stage in float64 = the same math",
             "identity_stages": {"power_rel_max": float(np.max(np.abs(h64i["power"] - st64i["power"])
                                                               / np.maximum(np.abs(st64i["power"]), 1e-300))),
                                 "log": diff(h64i["log"], st64i["log"]), "mean": diff(h64i["mean"], st64i["mean"]),
                                 "std": diff(h64i["std"], st64i["std"])},
             "over_bar": {"host_f32_vs_npz": frames_over(h32["out"], z["mel"], MEL_BAR),
                          "host_f64_vs_npz": frames_over(h64["out"].astype(np.float32), z["mel"], MEL_BAR)},
             "stages_host_f32_vs_provider_f32": {
                 "preemphasis": diff(h32["pre"], st32["pre"]),
                 "power": diff(h32["power"], st32["power"]),
                 "power_rel_max": float(np.max(np.abs(h32["power"].astype(np.float64) - st32["power"])
                                               / np.maximum(np.abs(st32["power"].astype(np.float64)), 1e-30))),
                 "mel_linear": diff(h32["lin"], st32["lin"]),
                 "log": diff(h32["log"], st32["log"]),
                 "mean": diff(h32["mean"], st32["mean"]), "std": diff(h32["std"], st32["std"]),
                 "std_min": float(st32["std"].min())},
             "stages_host_f64_vs_provider_f64": {
                 "preemphasis": diff(h64["pre"], st64["pre"].astype(np.float32)),
                 "power": diff(h64["power"], st64["power"]), "mel_linear": diff(h64["lin"], st64["lin"]),
                 "log": diff(h64["log"], st64["log"]), "mean": diff(h64["mean"], st64["mean"]),
                 "std": diff(h64["std"], st64["std"])},
             "stages_provider_f32_vs_f64": {
                 "power_rel_max": float(np.max(np.abs(st32["power"].astype(np.float64) - st64["power"])
                                               / np.maximum(np.abs(st64["power"]), 1e-30))),
                 "log": diff(st32["log"], st64["log"])}}
        doc["clips"][rid] = e
        for k in worst:
            worst[k] = max(worst[k], e["out"][k]["max_abs"])
        for k in over:
            over[k] += e["over_bar"][k]["frames_over"]
        print(f"{rid}: T {T} frames {frames} P {P} | host f32 vs npz {e['out']['host_f32_vs_npz']['max_abs']:.3g} "
              f"f64 vs npz {e['out']['host_f64_vs_npz']['max_abs']:.3g} f64 vs prov f64 "
              f"{e['out']['host_f64_vs_provider_f64']['max_abs']:.3g} prov f64 vs npz "
              f"{e['out']['provider_f64_vs_npz']['max_abs']:.3g} | rerun {e['provider_rerun_equal_npz_mel']}",
              flush=True)
    clips = doc["clips"].values()
    doc["summary"] = {
        "clips": len(doc["clips"]), "max_abs_over_clips": worst, "frames_over_bar_total": over,
        "frames_equal_all": all(c["frames_equal"] for c in clips), "T_equal_all": all(c["T_equal"] for c in clips),
        "P_equal_all": all(c["P_equal"] for c in clips),
        "provider_rerun_equal_npz_all": all(c["provider_rerun_equal_npz_mel"] for c in clips),
        "waveform_equal_all": all(c["waveform_host_equal_provider"] for c in clips),
        "reader_equal_all": all(c["reader_stdlib_wave_or_soundfile_equal_soundfile_int16"] for c in clips),
        "buckets": {rid: c["bucket"] for rid, c in doc["clips"].items()},
        "host_default_precision": AH.mel.__defaults__[0]}
    doc["seconds"] = round(time.time() - t0, 1)
    out = K / "results/audio_mel_check.json"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {out} (--overwrite)")
    write_json(out, doc)
    print(json.dumps(doc["summary"], indent=1))
    return 0


# ---------------------------------------------------------------- step 2: eager

def provider_model():
    """The provider's Audio(audio_config, 1024) with the checkpoint's audio.* tensors (strict load), eval."""
    import audio_graph as G

    A = provider_audio()
    cfg = S.config()
    m = A.Audio(cfg["audio_config"], cfg["text_config"]["hidden_size"]).eval()
    sd = {k[len("audio."):]: v for k, v in G.checkpoint_audio(S.WEIGHTS).items()}
    res = m.load_state_dict(sd, strict=True)
    return A, m, {"tensors": len(sd), "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}


def provider_prefix(m, mel, frames, keep=None):
    """Audio.forward minus the front end: encoder(mel, frames) -> adapter -> residual, valid rows [P, 1024].
    keep: dict -> the pre_encode / layer / adapter outputs and the BatchNorm inputs / outputs are stored in it."""
    import torch

    hooks = []
    if keep is not None:
        keep.update(sub=None, layers={}, bn={}, adapter=None)
        hooks.append(m.encoder.pre_encode.register_forward_hook(lambda mod, i, o: keep.__setitem__("sub", o[0])))
        for li, layer in enumerate(m.encoder.layers):
            hooks.append(layer.register_forward_hook(lambda mod, i, o, li=li: keep["layers"].__setitem__(li, o)))
            hooks.append(layer.conv.batch_norm.register_forward_hook(
                lambda mod, i, o, li=li: keep["bn"].__setitem__(li, (i[0].detach().clone(), o.detach().clone()))))
        hooks.append(m.adapter.register_forward_hook(lambda mod, i, o: keep.__setitem__("adapter", o)))
    try:
        with torch.no_grad():
            mel_t = torch.as_tensor(mel)[None].to(next(m.encoder.parameters()).dtype)
            x, lengths = m.encoder(mel_t, torch.tensor([int(frames)]))
            out = m.residual(m.adapter(x[:, :int(lengths[0])]))
    finally:
        for h in hooks:
            h.remove()
    return out[0].numpy(), int(lengths[0])


def mine_run(model, inputs, keep=None):
    """D1Audio on the host inputs -> prefix [T3, 1024] (all rows); keep: pre_encode / layer / adapter outputs."""
    import torch

    hooks = []
    if keep is not None:
        keep.update(sub=None, layers={}, adapter=None)
        hooks.append(model.pre_encode.register_forward_hook(lambda mod, i, o: keep.__setitem__("sub", o)))
        for li, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(lambda mod, i, o, li=li: keep["layers"].__setitem__(li, o)))
        hooks.append(model.adapter.register_forward_hook(lambda mod, i, o: keep.__setitem__("adapter", o)))
    dt = next(model.parameters()).dtype
    try:
        with torch.no_grad():
            out = model(**{k: torch.from_numpy(v).to(dt) for k, v in inputs.items()})["prefix"]
    finally:
        for h in hooks:
            h.remove()
    return out[0].numpy()


def oracle_audio_rows():
    oracle = json.loads((K / "ref/records_ref.json").read_text())
    rows = []
    for rec in oracle["records"]:
        if rec.get("mode") != "audio":
            continue
        for q in rec["questions"]:
            rows.append({"key": f"{rec['id']}/{q['qid']}", "id": rec["id"], "qid": q["qid"], "mode": "audio",
                         "P": int(q["prefix"]), "n": len(q["ids"]), "ids": q["ids"], "markers": q["markers"],
                         "K": int(q["K"]), "type": q["type"], "calibrate": bool(q["calibrate"]), "T": q.get("T"),
                         "temperature_key": q.get("temperature_key"),
                         "probs": [float(v) for v in q["probs"]], "logits_raw": [float(v) for v in q["logits_raw"]]})
    return rows, {"version": oracle.get("version"), "written": oracle.get("written")}


def text_inputs(row, prefix, L=256):
    import d1_host as H

    x = H.build_inputs(row["ids"], prefix, L)
    oh = np.zeros((1, 3), np.float32)
    oh[0, {"choice": 0, "score": 1, "noul": 2}[row["type"]]] = 1.0
    x["qtype_onehot"] = oh
    return x


def e2e_eager(rows, prefixes, text_model, L=256):
    """prefixes: {record id: [P, 1024]} -> {key: (probs, logits)} through the eager D1Decision (L256) + host readout."""
    import torch

    import litert_gate as LG

    out = {}
    for r in rows:
        x = text_inputs(r, prefixes[r["id"]], L)
        with torch.no_grad():
            s = text_model(**{k: torch.from_numpy(v) for k, v in x.items()})["scores"].numpy().reshape(-1)
        out[r["key"]] = LG.readout(s[:r["P"] + r["n"]], r)
    return out


def run_eager(a):
    import copy

    import torch

    import audio_graph as G
    import d1_graph as DG
    import litert_gate as LG

    torch.set_num_threads(THREADS)
    t0 = time.time()
    A, prov, prov_load = provider_model()
    assert not prov_load["missing"] and not prov_load["unexpected"], prov_load
    models, reports = {}, {}

    def mine(T):
        if T not in models:
            models[T], reports[T] = G.build(T)
        return models[T]

    doc = {"step": "round 7 step 2: D1Audio (scripts/audio_graph.py) eager fp32 vs the provider's Audio",
           "written": now(), "threads": THREADS,
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
           "provider_load": prov_load, "bar": {"prefix_max_abs_npz_mel": 1e-4, "float64_max_abs": 1e-12,
                                               "pad_content": "bit-equal valid rows"},
           "clips": {}}
    prefixes = {"mine_npz_mel": {}, "mine_host_mel": {}, "mine_host_mel_f64": {}, "npz": {}}
    rng = np.random.default_rng(7)
    for rid in RECORDS:
        z = npz_of(rid)
        frames, P = int(z["frames"][0]), int(z["prefix"].shape[0])
        T = z["mel"].shape[1]
        T_b = AH.bucket_for(T)
        model = mine(T_b)
        e = {"T": T, "frames": frames, "P": P, "bucket": T_b, "T123": list(G.dims(T_b))}
        # the provider again on the npz mel (= the oracle's prefix: same torch, same threads)
        kp = {}
        p_prov, lp = provider_prefix(prov, z["mel"], frames, keep=kp)
        e["provider_rerun_vs_npz_prefix"] = diff(p_prov, z["prefix"])
        assert lp == P
        # (i) the npz mel through D1Audio
        x = AH.build_inputs(z["mel"], frames, T_b)
        km = {}
        full = mine_run(model, x, keep=km)
        e["i_prefix_npz_mel_vs_npz_prefix"] = diff(full[:P], z["prefix"])
        e["i_prefix_npz_mel_vs_provider_rerun"] = diff(full[:P], p_prov)
        e["pad_rows_finite"] = bool(np.isfinite(full).all())
        prefixes["mine_npz_mel"][rid] = full[:P]
        prefixes["npz"][rid] = z["prefix"]
        # per stage: the valid rows of the provider's (at its own T) vs ours (at the bucket)
        e["per_stage_valid_rows_max_abs"] = {
            "pre_encode": float(np.abs(km["sub"][0, :P].numpy().astype(np.float64) - kp["sub"][0, :P].numpy()).max()),
            "layers": [float(np.abs(km["layers"][i][0, :P].numpy().astype(np.float64)
                                    - kp["layers"][i][0, :P].numpy()).max()) for i in range(G.N_LAYERS)],
            "adapter": float(np.abs(km["adapter"][0, :P].numpy().astype(np.float64) - kp["adapter"][0].numpy()).max())}
        # (v) BatchNorm as an affine, on the provider's own BatchNorm inputs (channels-first [1, 512, t])
        bn = []
        for li in range(G.N_LAYERS):
            xin, yout = kp["bn"][li]
            conv = model.layers[li].conv
            ya = xin * conv.bn_scale[None, :, None] + conv.bn_shift[None, :, None]
            yf = torch.nn.functional.batch_norm(xin, prov.encoder.layers[li].conv.batch_norm.running_mean,
                                                prov.encoder.layers[li].conv.batch_norm.running_var,
                                                prov.encoder.layers[li].conv.batch_norm.weight,
                                                prov.encoder.layers[li].conv.batch_norm.bias, False, 0.0, G.BN_EPS)
            bn.append({"layer": li, "affine_vs_provider_bn_max_abs": float((ya - yout).abs().max()),
                       "bit_equal": bool(torch.equal(ya, yout)), "F_batch_norm_equal_module": bool(torch.equal(yf, yout))})
        e["v_batch_norm_affine"] = {"max_abs": max(b["affine_vs_provider_bn_max_abs"] for b in bn),
                                    "layers_bit_equal": sum(b["bit_equal"] for b in bn), "per_layer": bn}
        # subsampling at the same T (the bucket): the provider's ConvSubsampling on the padded mel vs ours, all rows
        with torch.no_grad():
            sp, lsp = prov.encoder.pre_encode(torch.from_numpy(x["mel"]).transpose(1, 2), torch.tensor([frames]))
            sm = model.pre_encode(*(torch.from_numpy(x[k]) for k in G.INPUT_NAMES))
        e["subsampling_same_T_all_rows"] = {"bit_equal": bool(torch.equal(sp, sm)),
                                            "max_abs": float((sp - sm).abs().max()), "rows": int(sp.shape[1]),
                                            "lengths_provider": int(lsp[0])}
        # folded linear_pos(pos_emb) vs the provider's, same relative positions
        t_clip, T3 = kp["sub"].shape[1], model.T3
        off = T3 - t_clip
        pe_p = prov.encoder.pos_emb(t_clip, "cpu")
        pe_m = G.pos_table(T3)
        pos = {"pos_table_bit_equal": bool(torch.equal(pe_m[:, off:off + 2 * t_clip - 1], pe_p)), "layers": []}
        with torch.no_grad():
            for li in range(G.N_LAYERS):
                pp = prov.encoder.layers[li].self_attn.linear_pos(pe_p)                        # [1, 2t-1, 512]
                pm = model.layers[li].self_attn.p_t.permute(0, 3, 1, 2).reshape(1, 2 * T3 - 1, G.D_MODEL)
                d = (pm[:, off:off + 2 * t_clip - 1] - pp).abs().max()
                pos["layers"].append({"layer": li, "max_abs": float(d),
                                      "bit_equal": bool(torch.equal(pm[:, off:off + 2 * t_clip - 1], pp))})
        pos["max_abs"] = max(p["max_abs"] for p in pos["layers"])
        pos["layers_bit_equal"] = sum(p["bit_equal"] for p in pos["layers"])
        e["folded_pos_vs_provider_linear_pos"] = pos
        # (iii) pad-content: random mel on every invalid frame (t >= frames, including the bucket padding)
        y = {k: v.copy() for k, v in x.items()}
        y["mel"][0, :, frames:] = rng.standard_normal((128, T_b - frames)).astype(np.float32) * 5.0
        full2 = mine_run(model, y)
        e["iii_pad_content"] = {"valid_rows_bit_equal": bool(np.array_equal(full2[:P], full[:P])),
                                "valid_rows_max_abs": float(np.abs(full2[:P].astype(np.float64) - full[:P]).max()),
                                "pad_rows_changed": bool(not np.array_equal(full2[P:], full[P:]))}
        # (ii) the host's mel (float32 default; float64 as a variant)
        x16, _ = G.clip_samples(rid)
        for prec, key in (("float32", "mine_host_mel"), ("float64", "mine_host_mel_f64")):
            xh, info = AH.prepare(x16, precision=prec)
            assert info["T_b"] == T_b and info["P"] == P and info["frames"] == frames
            ph = mine_run(model, xh)[:P]
            prefixes[key][rid] = ph
            e[f"ii_prefix_host_mel_{prec}_vs_npz_prefix"] = diff(ph, z["prefix"])
            e[f"ii_prefix_host_mel_{prec}_vs_npz_mel_run"] = diff(ph, full[:P])
        doc["clips"][rid] = e
        print(f"{rid}: T_b {T_b} P {P} | (i) {e['i_prefix_npz_mel_vs_npz_prefix']['max_abs']:.3g} "
              f"(ii) {e['ii_prefix_host_mel_float32_vs_npz_prefix']['max_abs']:.3g} "
              f"pad {e['iii_pad_content']['valid_rows_bit_equal']} sub-same-T {e['subsampling_same_T_all_rows']['bit_equal']} "
              f"bn {e['v_batch_norm_affine']['max_abs']:.3g} pos {pos['max_abs']:.3g} "
              f"rerun {e['provider_rerun_vs_npz_prefix']['bit_equal']}", flush=True)
    # bucket dependence: aud_01 at 2001 / 3001 vs at 1001 (valid rows)
    z = npz_of("aud_01")
    frames, P = int(z["frames"][0]), int(z["prefix"].shape[0])
    ref1001 = prefixes["mine_npz_mel"]["aud_01"]
    doc["bucket_dependence_aud_01"] = {}
    for T_b in (2001, 3001):
        pb = mine_run(mine(T_b), AH.build_inputs(z["mel"], frames, T_b))[:P]
        doc["bucket_dependence_aud_01"][str(T_b)] = {"vs_T1001": diff(pb, ref1001), "vs_npz_prefix": diff(pb, z["prefix"])}
    # T501: aud_01 cut to 5 s (80,000 samples): the provider's own Audio forward on the cut clip vs ours at T501 / T1001
    x16, _ = G.clip_samples("aud_01", seconds=5.0)
    with torch.no_grad():
        p_cut = prov(x16)[0].numpy()                                     # the full provider module, front end included
    xh, info = AH.prepare(x16)
    assert info["T_b"] == 501 and info["P"] == p_cut.shape[0] == 63, (info, p_cut.shape)
    m501 = mine_run(mine(501), xh)
    m1001 = mine_run(mine(1001), AH.build_inputs(*AH.mel(AH.waveform(x16)), 1001))
    doc["T501_aud_01_cut_5s"] = {"samples": int(x16.shape[0]), "info": info,
                                 "prefix_T501_vs_provider": diff(m501[:63], p_cut),
                                 "prefix_T1001_first63_vs_T501": diff(m1001[:63], m501[:63]),
                                 "prefix_T1001_vs_provider": diff(m1001[:63], p_cut)}
    prefixes["mine_npz_mel"]["aud_01_cut5"] = m501[:63]
    np.savez(K / "out/r7_eager_prefix_T501_aud01cut5.npz", provider=p_cut, mine_T501=m501, mine_T1001=m1001)
    # (iv) float64: aud_01, both sides float64 (the npz mel cast up)
    prov64 = copy.deepcopy(prov).double()
    m64, _ = G.build(1001, dtype=torch.float64)
    z = npz_of("aud_01")
    frames, P = int(z["frames"][0]), int(z["prefix"].shape[0])
    p64, _ = provider_prefix(prov64, z["mel"].astype(np.float64), frames)
    x64 = {k: v.astype(np.float64) for k, v in AH.build_inputs(z["mel"], frames, 1001).items()}
    with torch.no_grad():
        o64 = m64(**{k: torch.from_numpy(v) for k, v in x64.items()})["prefix"][0, :P].numpy()
    doc["iv_float64_aud_01"] = diff(o64, p64)
    doc["iv_float64_aud_01"]["dtype"] = str(o64.dtype)
    del prov64, m64
    # end to end (eager): prefixes through the eager decision graph L256 -> host readout vs the oracle
    rows, ometa = oracle_audio_rows()
    cfg = S.config()
    tm = DG.D1Decision(cfg["text_config"], cfg["head_layers"], 256).eval()
    trep = DG.load_checkpoint(tm, S.WEIGHTS)
    sources = LG.record_sources()
    ref = {r["key"]: (r["probs"], r["logits_raw"]) for r in rows}
    e2e = {}
    for name in ("npz", "mine_npz_mel", "mine_host_mel", "mine_host_mel_f64"):
        lit = e2e_eager(rows, prefixes[name], tm)
        st = LG.compare(rows, lit, ref, sources)
        e2e[name] = {k: st[k] for k in ("rows_compared", "max_abs_dp", "p95_abs_dp", "mean_abs_dp", "max_abs_dlogit",
                                        "argmax", "cutoff_crossings", "nonfinite_rows", "bar_pass", "top10_by_dp")}
        e2e[name]["per_row"] = {k: v[0] for k, v in lit.items()}
    doc["e2e_eager_L256"] = {"oracle": ometa, "rows": len(rows), "text_model": "D1Decision L256 eager fp32 (d1_graph)",
                             "text_weights": {k: trep[k] for k in ("source_tensors", "loaded_parameters")},
                             "prefix_sources": {"npz": "the oracle's prefix (text graph alone)",
                                                "mine_npz_mel": "D1Audio on the oracle's mel",
                                                "mine_host_mel": "D1Audio on the host's float32 mel (the shipped path)",
                                                "mine_host_mel_f64": "D1Audio on the host's float64 mel"},
                             "bar": LG.BAR, "by_prefix": e2e}
    clips = doc["clips"].values()
    doc["summary"] = {
        "i_prefix_npz_mel_max_abs": max(c["i_prefix_npz_mel_vs_npz_prefix"]["max_abs"] for c in clips),
        "i_prefix_npz_mel_max_rel_to_absmax": max(c["i_prefix_npz_mel_vs_npz_prefix"]["max_rel_to_ref_absmax"]
                                                  for c in clips),
        "ii_prefix_host_mel_float32_max_abs": max(c["ii_prefix_host_mel_float32_vs_npz_prefix"]["max_abs"] for c in clips),
        "ii_prefix_host_mel_float64_max_abs": max(c["ii_prefix_host_mel_float64_vs_npz_prefix"]["max_abs"] for c in clips),
        "iii_pad_content_bit_equal": all(c["iii_pad_content"]["valid_rows_bit_equal"] for c in clips),
        "iv_float64_max_abs": doc["iv_float64_aud_01"]["max_abs"],
        "v_batch_norm_affine_max_abs": max(c["v_batch_norm_affine"]["max_abs"] for c in clips),
        "v_batch_norm_layers_bit_equal": f"{sum(c['v_batch_norm_affine']['layers_bit_equal'] for c in clips)}/"
                                         f"{len(doc['clips']) * G.N_LAYERS}",
        "subsampling_same_T_bit_equal": all(c["subsampling_same_T_all_rows"]["bit_equal"] for c in clips),
        "folded_pos_max_abs": max(c["folded_pos_vs_provider_linear_pos"]["max_abs"] for c in clips),
        "provider_rerun_equal_npz_prefix": all(c["provider_rerun_vs_npz_prefix"]["bit_equal"] for c in clips),
        "pad_rows_finite": all(c["pad_rows_finite"] for c in clips),
        "T501_vs_provider_max_abs": doc["T501_aud_01_cut_5s"]["prefix_T501_vs_provider"]["max_abs"],
        "T501_vs_T1001_first63": doc["T501_aud_01_cut_5s"]["prefix_T1001_first63_vs_T501"],
        "e2e_max_abs_dp": {k: v["max_abs_dp"] for k, v in e2e.items()},
        "e2e_bar_pass": {k: v["bar_pass"] for k, v in e2e.items()},
        "keymap_missing": sum(len(r["unmatched"]) + len(r["missing_parameters"]) + len(r["raw_missing"])
                              for r in reports.values())}
    s = doc["summary"]
    s["PASS"] = bool(s["i_prefix_npz_mel_max_abs"] <= 1e-4 and s["iii_pad_content_bit_equal"]
                     and s["iv_float64_max_abs"] <= 1e-12 and s["keymap_missing"] == 0)
    doc["seconds"] = round(time.time() - t0, 1)
    out = K / "results/audio_eager_check.json"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {out} (--overwrite)")
    write_json(out, doc)
    print(json.dumps(s, indent=1, default=str))
    return 0 if s["PASS"] else 1


# ---------------------------------------------------------------- step 4: LiteRT on the Mac

RUNS = K / "out/r7_runs"
BUCKET_OF = {"card_topic": 2001}          # every other clip lands in T1001 (<= 10 s); card_topic is 10.435 s
CUT = ("aud_01", 5.0)                       # the T501 clip: aud_01 cut to 80,000 samples (P 63)


def tag_of(backend, precision, form, long=False):
    t = f"audio_cpu_{form}" if backend == "cpu" else f"audio_gpu_{precision}_{form}"
    return t + "_T3001" if long else t


def label_of(backend, precision):
    return "cpu" if backend == "cpu" else f"gpu_{precision}"


def audio_file(T, form):
    return K / f"out/d1omni_audio_T{T}_{form}.tflite"


def run_lite(a):
    """Child body (GPU) or the whole run (CPU): every clip through the audio graph files of its bucket (host float32
    mel and the oracle's mel), pad-content per clip, the 5 s cut on T501 and T1001; prefixes -> out/r7_runs/<tag>.npz,
    the run record -> out/r7_runs/<tag>.json."""
    import importlib.metadata as md
    import os
    import resource

    import audio_graph as G
    import litert_run as R

    if a.long:
        return run_lite_long(a)
    tag = tag_of(a.backend, a.precision, a.form)
    RUNS.mkdir(parents=True, exist_ok=True)
    log = K / f"logs/r7_{tag}.runtime.log"
    info = {"tag": tag, "started": now(), "pid": os.getpid(), "backend": a.backend, "precision": a.precision,
            "form": a.form, "ai_edge_litert": md.version("ai-edge-litert"), "files": {}, "compile_seconds": {},
            "is_fully_accelerated": {}, "options": None, "status": "FAIL"}
    store = {}
    rng = np.random.default_rng(11)
    t0 = time.time()
    with R.capture_fd2(log):
        try:
            info["logger"] = R.runtime_log_verbose()
            runners = {}
            for T in (501, 1001, 2001):
                path = audio_file(T, a.form)
                info["files"][str(T)] = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size}
                t1 = time.time()
                cm, desc = R.open_compiled(path, a.backend, a.precision, threads=8)
                info["compile_seconds"][str(T)] = round(time.time() - t1, 2)
                info["options"] = desc
                try:
                    info["is_fully_accelerated"][str(T)] = bool(cm.is_fully_accelerated())
                except Exception as e:  # informational
                    info["is_fully_accelerated"][str(T)] = f"unavailable: {type(e).__name__}: {e}"
                runners[T] = (cm, G.AudioRunner(cm, f"audio_{T}"))
                print(f"compiled T{T} in {info['compile_seconds'][str(T)]} s", file=sys.stdout, flush=True)
            clips = {}
            for rid in RECORDS:
                z = npz_of(rid)
                frames, P = int(z["frames"][0]), int(z["prefix"].shape[0])
                T_b = BUCKET_OF.get(rid, 1001)
                run = runners[T_b][1]
                x16, _ = G.clip_samples(rid)
                xh, hinfo = AH.prepare(x16, bucket=T_b)
                assert hinfo["P"] == P and hinfo["frames"] == frames, (rid, hinfo)
                out_h = run(xh)
                out_n = run(AH.build_inputs(z["mel"], frames, T_b))
                y = {k: v.copy() for k, v in xh.items()}
                y["mel"][0, :, frames:] = rng.standard_normal((128, T_b - frames)).astype(np.float32) * 5.0
                out_p = run(y)
                store[f"{rid}__host"] = out_h[:P]
                store[f"{rid}__npzmel"] = out_n[:P]
                clips[rid] = {"T_b": T_b, "P": P, "frames": frames,
                              "nonfinite_all_rows_host": int((~np.isfinite(out_h)).sum()),
                              "nonfinite_all_rows_npzmel": int((~np.isfinite(out_n)).sum()),
                              "pad_content_valid_rows_bit_equal": bool(np.array_equal(out_p[:P], out_h[:P])),
                              "pad_content_valid_rows_max_abs": float(np.abs(out_p[:P].astype(np.float64)
                                                                             - out_h[:P]).max())}
            # the 5 s cut: T501 vs T1001 (first 63 rows), both from the host mel
            x16, _ = G.clip_samples(CUT[0], seconds=CUT[1])
            x501, i501 = AH.prepare(x16)
            assert i501["T_b"] == 501
            m, fr = AH.mel(AH.waveform(x16))
            o501 = runners[501][1](x501)
            o1001 = runners[1001][1](AH.build_inputs(m, fr, 1001))
            store["aud_01_cut5__T501"] = o501[:i501["P"]]
            store["aud_01_cut5__T1001"] = o1001[:i501["P"]]
            clips["aud_01_cut5"] = {"T_b": 501, "P": i501["P"], "frames": i501["frames"],
                                    "nonfinite_all_rows_host": int((~np.isfinite(o501)).sum()),
                                    "T1001_first_P_bit_equal_T501": bool(np.array_equal(o1001[:63], o501[:63])),
                                    "T1001_first_P_max_abs_vs_T501": float(np.abs(o1001[:63].astype(np.float64)
                                                                                  - o501[:63]).max())}
            info["clips"] = clips
            for T, (cm, run) in runners.items():
                run.close()
                if hasattr(cm, "close"):
                    cm.close()
            info["status"] = "OK"
        except BaseException as e:  # recorded
            import traceback

            info["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    info["seconds_wall"] = round(time.time() - t0, 1)
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["delegation"] = R.delegation_from_log(log)
    np.savez(RUNS / f"{tag}.npz", **store)
    write_json(RUNS / f"{tag}.json", info)
    print(json.dumps({"tag": tag, "status": info["status"], "error": info.get("error"),
                      "replacing": info["delegation"]["replacing"], "is_fully_accelerated": info["is_fully_accelerated"],
                      "compile_seconds": info["compile_seconds"], "seconds": info["seconds_wall"]}, indent=1))
    return 0 if info["status"] == "OK" else 1


def run_lite_long(a):
    """T3001: the 27.2 s concatenation (aud_01..03) through out/d1omni_audio_T3001_<form>.tflite (host mel and the
    provider's mel), pad-content -> out/r7_runs/<tag>_T3001.{npz,json}; reference = run_long()'s provider prefix."""
    import importlib.metadata as md
    import os
    import resource

    import audio_graph as G
    import litert_run as R

    tag = tag_of(a.backend, a.precision, a.form, True)
    RUNS.mkdir(parents=True, exist_ok=True)
    log = K / f"logs/r7_{tag}.runtime.log"
    info = {"tag": tag, "started": now(), "pid": os.getpid(), "backend": a.backend, "precision": a.precision,
            "form": a.form, "ai_edge_litert": md.version("ai-edge-litert"), "status": "FAIL"}
    store = {}
    t0 = time.time()
    with R.capture_fd2(log):
        try:
            info["logger"] = R.runtime_log_verbose()
            path = audio_file(3001, a.form)
            info["files"] = {"3001": {"file": str(path.relative_to(K)), "bytes": path.stat().st_size}}
            t1 = time.time()
            cm, desc = R.open_compiled(path, a.backend, a.precision, threads=8)
            info["compile_seconds"] = {"3001": round(time.time() - t1, 2)}
            info["options"] = desc
            info["is_fully_accelerated"] = {"3001": bool(cm.is_fully_accelerated())}
            run = G.AudioRunner(cm, "audio_3001")
            ref = np.load(LONG_REF)
            frames = int(ref["frames"][0])
            x16, name = G.long_samples()
            xh, hinfo = AH.prepare(x16)
            P = hinfo["P"]
            out_h = run(xh)
            out_n = run(AH.build_inputs(ref["provider_mel"], frames, 3001))
            y = {k: v.copy() for k, v in xh.items()}
            y["mel"][0, :, frames:] = np.random.default_rng(13).standard_normal((128, 3001 - frames)).astype(
                np.float32) * 5.0
            out_p = run(y)
            store["long__host"] = out_h[:P]
            store["long__npzmel"] = out_n[:P]
            info["clips"] = {"long": {"clip": name, "T_b": 3001, "P": P, "frames": frames,
                                      "nonfinite_all_rows_host": int((~np.isfinite(out_h)).sum()),
                                      "pad_content_valid_rows_bit_equal": bool(np.array_equal(out_p[:P], out_h[:P]))}}
            run.close()
            if hasattr(cm, "close"):
                cm.close()
            info["status"] = "OK"
        except BaseException as e:  # recorded
            import traceback

            info["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    info["seconds_wall"] = round(time.time() - t0, 1)
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["delegation"] = R.delegation_from_log(log)
    np.savez(RUNS / f"{tag}.npz", **store)
    write_json(RUNS / f"{tag}.json", info)
    print(json.dumps({"tag": tag, "status": info["status"], "error": info.get("error"),
                      "replacing": info["delegation"]["replacing"], "seconds": info["seconds_wall"]}, indent=1))
    return 0 if info["status"] == "OK" else 1


def run_lite_gpu_parent(a):
    """The GPU run in a child process; a delegate abort ends only the child and the parent writes the failure."""
    import os
    import signal
    import subprocess

    tag = tag_of(a.backend, a.precision, a.form, a.long)
    errlog = K / f"logs/r7_{tag}.child_stdio.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--lite", "--backend", "gpu", "--precision", a.precision,
           "--form", a.form, "--child"] + (["--long"] if a.long else [])
    t0 = time.time()
    with open(errlog, "w") as fo:
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, status, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    meta = RUNS / f"{tag}.json"
    rec = json.loads(meta.read_text()) if meta.exists() else {}
    if rec.get("pid") != child.pid:
        logs = K / f"logs/r7_{tag}.runtime.log"
        rec = {"tag": tag, "pid": child.pid, "status": "GPU_FAIL", "returncode": rc,
               "signal": signal.Signals(-rc).name if rc < 0 else None, "seconds_wall": round(time.time() - t0, 1),
               "runtime_log_tail": logs.read_text(errors="replace").splitlines()[-40:] if logs.exists() else [],
               "child_stdio_tail": errlog.read_text(errors="replace").splitlines()[-40:]}
    rec["child_ru_maxrss_bytes"] = int(ru.ru_maxrss)
    rec["child_returncode"] = rc
    write_json(meta, rec)
    print(f"gpu child pid={child.pid} rc={rc} status={rec.get('status')} maxrss={int(ru.ru_maxrss)} "
          f"seconds={time.time() - t0:.1f}")
    return rc


def score_all(a):
    """Every stored run: prefix parity vs the oracle's prefix (host mel and the oracle's mel), and end to end through
    the decision graph out/d1omni_decide_L256_fp16.tflite on CPU (XNNPACK 8 threads) -> probs vs the oracle (the 19
    audio rows, raw softmax), plus the oracle's own prefix through the same text graph (the text graph's share)
    -> results/audio_parity_<backend>_<precision>_<form>.json."""
    import litert_gate as LG
    import litert_run as R

    rows, ometa = oracle_audio_rows()
    ref = {r["key"]: (r["probs"], r["logits_raw"]) for r in rows}
    sources = LG.record_sources()
    text_file = K / "out/d1omni_decide_L256_fp16.tflite"
    cm, desc = R.open_compiled(text_file, "cpu", threads=8)
    sig = next(iter(cm.get_signature_list()))
    trun = R.Runner(cm, sig)
    assert trun.L == 256

    def e2e(prefixes):
        lit = {}
        for r in rows:
            x = text_inputs(r, prefixes[r["id"]], 256)
            s = trun(x)
            lit[r["key"]] = LG.readout(s[:r["P"] + r["n"]], r)
        return lit

    npz_prefix = {rid: npz_of(rid)["prefix"] for rid in RECORDS}
    base_lit = e2e(npz_prefix)
    base = LG.compare(rows, base_lit, ref, sources)
    cut_prov = np.load(K / "out/r7_eager_prefix_T501_aud01cut5.npz")["provider"]
    written = []
    long_ref = np.load(LONG_REF) if LONG_REF.exists() else None
    for meta in sorted(RUNS.glob("audio_*.json")):
        rec = json.loads(meta.read_text())
        tag = rec["tag"]
        if tag.endswith("_T3001"):
            continue
        backend = "cpu" if tag.startswith("audio_cpu_") else "gpu"
        precision = "fp32" if backend == "cpu" else tag.split("_")[2]
        form = tag.split("_")[-1]
        doc = {"step": f"round 7 step 4: Mac audio graph parity, {label_of(backend, precision)}, form {form}",
               "scored_at": now(), "tag": tag, "run": {k: v for k, v in rec.items() if k != "clips"},
               "oracle": ometa, "text_graph": {"file": str(text_file.relative_to(K)), "backend": desc, "L": 256},
               "bar_prefix_max_abs": 1e-4, "bar_e2e": LG.BAR}
        if rec.get("status") != "OK":
            doc["status"] = rec.get("status", "FAIL")
            write_json(K / f"results/audio_parity_{label_of(backend, precision)}_{form}.json", doc)
            written.append(doc["status"])
            continue
        with np.load(RUNS / f"{tag}.npz") as z:
            st = {k: np.asarray(z[k]) for k in z.files}
        clips = {}
        for rid in RECORDS:
            c = dict(rec["clips"][rid])
            c["prefix_host_mel_vs_npz"] = diff(st[f"{rid}__host"], npz_prefix[rid])
            c["prefix_npz_mel_vs_npz"] = diff(st[f"{rid}__npzmel"], npz_prefix[rid])
            clips[rid] = c
        cut = dict(rec["clips"]["aud_01_cut5"])
        cut["T501_vs_provider"] = diff(st["aud_01_cut5__T501"], cut_prov)
        cut["T1001_vs_provider"] = diff(st["aud_01_cut5__T1001"], cut_prov)
        clips["aud_01_cut5"] = cut
        lit_h = e2e({rid: st[f"{rid}__host"] for rid in RECORDS})
        lit_n = e2e({rid: st[f"{rid}__npzmel"] for rid in RECORDS})
        e_h = LG.compare(rows, lit_h, ref, sources)
        e_n = LG.compare(rows, lit_n, ref, sources)
        vs_base = max(max(abs(x - y) for x, y in zip(lit_h[k][0], base_lit[k][0])) for k in lit_h)
        keep = ("rows_compared", "max_abs_dp", "p95_abs_dp", "mean_abs_dp", "max_abs_dlogit", "argmax",
                "cutoff_crossings", "nonfinite_rows", "bar_pass", "top10_by_dp")
        doc.update(status="OK", clips=clips,
                   e2e={"host_mel": {k: e_h[k] for k in keep}, "oracle_mel": {k: e_n[k] for k in keep},
                        "oracle_prefix_same_text_graph": {k: base[k] for k in keep},
                        "max_abs_dp_host_mel_vs_oracle_prefix_run": vs_base},
                   per_row=[{"key": r["key"], "P": r["P"], "type": r["type"], "probs_oracle": r["probs"],
                             "probs_host_mel": lit_h[r["key"]][0], "probs_oracle_mel": lit_n[r["key"]][0],
                             "probs_oracle_prefix": base_lit[r["key"]][0]} for r in rows])
        cl = [clips[r] for r in RECORDS]
        doc["summary"] = {
            "prefix_host_mel_max_abs": max(c["prefix_host_mel_vs_npz"]["max_abs"] for c in cl),
            "prefix_host_mel_max_rel_to_absmax": max(c["prefix_host_mel_vs_npz"]["max_rel_to_ref_absmax"] for c in cl),
            "prefix_oracle_mel_max_abs": max(c["prefix_npz_mel_vs_npz"]["max_abs"] for c in cl),
            "e2e_host_mel_max_abs_dp": e_h["max_abs_dp"], "e2e_host_mel_mean_abs_dp": e_h["mean_abs_dp"],
            "e2e_argmax": f"{e_h['argmax']['equal_outside_near_tie']}/{e_h['argmax']['rows_outside_near_tie']} + "
                          f"near-tie {e_h['argmax']['near_tie_equal']}/{e_h['argmax']['near_tie_rows']}",
            "e2e_cutoff_crossings": e_h["cutoff_crossings"], "e2e_bar_pass": e_h["bar_pass"],
            "e2e_oracle_prefix_same_text_graph_max_abs_dp": base["max_abs_dp"],
            "nonfinite": sum(c["nonfinite_all_rows_host"] + c["nonfinite_all_rows_npzmel"] for c in cl),
            "pad_content_bit_equal": f"{sum(c['pad_content_valid_rows_bit_equal'] for c in cl)}/{len(cl)}",
            "T501_vs_provider_max_abs": cut["T501_vs_provider"]["max_abs"],
            "T1001_first63_bit_equal_T501": cut["T1001_first_P_bit_equal_T501"],
            "replacing": rec["delegation"]["replacing"], "is_fully_accelerated": rec["is_fully_accelerated"],
            "compile_seconds": rec["compile_seconds"]}
        if backend == "gpu":
            cpu_meta = RUNS / f"audio_cpu_{form}.npz"
            if cpu_meta.exists():
                with np.load(cpu_meta) as z:
                    cs = {k: np.asarray(z[k]) for k in z.files}
                doc["summary"]["gpu_vs_cpu_same_file_prefix_max_abs"] = max(
                    float(np.abs(st[k].astype(np.float64) - cs[k]).max()) for k in st if k in cs)
        lmeta = RUNS / f"{tag}_T3001.json"
        if lmeta.exists() and long_ref is not None:
            lrec = json.loads(lmeta.read_text())
            sec = {"run": {k: v for k, v in lrec.items() if k != "clips"}, "clips": lrec.get("clips"),
                   "reference": f"{LONG_REF.relative_to(K)} (the provider's Audio on aud_01 + aud_02 + aud_03, "
                                f"27.2 s; results/audio_eager_long.json)"}
            if lrec.get("status") == "OK":
                with np.load(RUNS / f"{tag}_T3001.npz") as z:
                    sec["prefix_host_mel_vs_provider"] = diff(z["long__host"], long_ref["provider_prefix"])
                    sec["prefix_provider_mel_vs_provider"] = diff(z["long__npzmel"], long_ref["provider_prefix"])
            doc["T3001_long"] = sec
            doc["summary"]["T3001_prefix_host_mel_max_abs"] = (sec.get("prefix_host_mel_vs_provider") or {}).get("max_abs")
            doc["summary"]["T3001_status"] = lrec.get("status")
        write_json(K / f"results/audio_parity_{label_of(backend, precision)}_{form}.json", doc)
        written.append(f"{tag}: prefix {doc['summary']['prefix_host_mel_max_abs']:.3g} e2e "
                       f"{doc['summary']['e2e_host_mel_max_abs_dp']:.3g} pass {doc['summary']['e2e_bar_pass']}")
    trun.close()
    print(json.dumps({"text_graph_oracle_prefix_max_abs_dp": base["max_abs_dp"], "written": written}, indent=1))
    return 0


# ---------------------------------------------------------------- step 5: ms on the Mac

WARMUP, REPS = 5, 20
SPEED_TMP = K / "out/r7_timing_tmp"
SPEED_CLIP = {501: ("aud_01", 5.0), 1001: ("aud_01", None), 2001: ("card_topic", None), 3001: ("long", None)}
PIPE_ROW = "aud_01/topic"                    # the 10 s audio question of round 4's workload (f)


def speed_samples(T):
    import audio_graph as G

    rid, cut = SPEED_CLIP[T]
    if rid == "long":
        return G.long_samples()
    return G.clip_samples(rid, cut)


def speed_reference(T):
    """The reference prefix of the timed clip (the oracle's npz prefix, the provider's prefix for the cut / long clip)."""
    rid, cut = SPEED_CLIP[T]
    if rid == "long":
        return np.load(LONG_REF)["provider_prefix"]
    if cut:
        return np.load(K / "out/r7_eager_prefix_T501_aud01cut5.npz")["provider"]
    return npz_of(rid)["prefix"]


def speed_child(a):
    """One fresh process: kind audio = one audio file (T, form) -> compile, memory, 5 warm-up + 20 timed calls (write 5
    inputs + run + read the prefix), the first timed output vs the reference; kind pipeline = the host mel of the
    5 / 10 / 27 s clips (numpy) and the 10 s question end to end (host mel -> audio graph T1001 -> prefix rows ->
    text inputs -> text graph L256 fp16 on the same backend -> readout), each part timed per call."""
    import os

    import audio_graph as G
    import litert_gate as LG
    import litert_run as R

    doc = {"pid": os.getpid(), "started": now(), "backend": a.backend, "form": a.form, "kind": a.kind, "T": a.T,
           "status": "FAIL"}
    backend = "cpu" if a.backend == "cpu" else "gpu"
    precision = "default" if a.backend == "gpu_default" else "fp32"
    try:
        doc["memory_before_compile"] = R.memory()
        if a.kind == "audio":
            path = audio_file(a.T, a.form)
            doc["file"], doc["file_bytes"] = str(path.relative_to(K)), path.stat().st_size
            t0 = time.perf_counter()
            cm, desc = R.open_compiled(path, backend, precision, threads=8)
            doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
            doc["options"] = desc
            doc["memory_after_compile"] = R.memory()
            doc["is_fully_accelerated"] = bool(cm.is_fully_accelerated())
            run = G.AudioRunner(cm, f"audio_{a.T}")
            x16, name = speed_samples(a.T)
            x, info = AH.prepare(x16, bucket=a.T)
            x = {k: np.ascontiguousarray(v) for k, v in x.items()}
            doc["clip"] = {"name": name, "samples": int(x16.shape[0]), "seconds": round(x16.shape[0] / 16000, 3),
                           **info}
            warm = [list(map(lambda v: round(v, 3), run.timed(x)[1:])) for _ in range(WARMUP)]
            calls, first, finite = [], None, True
            for _ in range(REPS):
                out, wall, tot, rn = run.timed(x)
                calls.append([round(wall, 3), round(tot, 3), round(rn, 3)])
                finite &= bool(np.isfinite(out).all())
                if first is None:
                    first = out[:info["P"]].copy()
            ref = speed_reference(a.T)
            doc["workload"] = {"what": f"audio graph T{a.T}, 1 call = write the 5 inputs + run + read the prefix "
                                       f"[1, {run.T3}, 1024]", "warmup_calls": warm, "calls": calls,
                               "call_columns": ["wall_clock_ms_at_start", "ms_write_run_read", "ms_run_only"],
                               "ms_write_run_read": __import__("timing_mac").stats([c[1] for c in calls]),
                               "ms_run_only": __import__("timing_mac").stats([c[2] for c in calls]),
                               "finite": finite, "first_call_prefix_vs_reference": diff(first, ref)}
            doc["memory_end"] = R.memory()
            run.close()
        else:
            import timing_mac as TM

            doc["mel"] = {}
            for T in (501, 1001, 3001):
                x16, name = speed_samples(T)
                for _ in range(WARMUP):
                    AH.prepare(x16)
                ms = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    AH.prepare(x16)
                    ms.append((time.perf_counter() - t0) * 1000.0)
                doc["mel"][f"{T}"] = {"clip": name, "seconds": round(x16.shape[0] / 16000, 3),
                                      "what": "host/d1_audio_host.prepare: waveform + float32 mel + the 5 inputs (numpy)",
                                      "ms": TM.stats(ms), "calls_ms": [round(v, 3) for v in ms]}
            apath, tpath = audio_file(1001, a.form), K / "out/d1omni_decide_L256_fp16.tflite"
            t0 = time.perf_counter()
            acm, desc = R.open_compiled(apath, backend, precision, threads=8)
            doc["compile_seconds_audio"] = round(time.perf_counter() - t0, 3)
            t0 = time.perf_counter()
            tcm, _ = R.open_compiled(tpath, backend, precision, threads=8)
            doc["compile_seconds_text"] = round(time.perf_counter() - t0, 3)
            doc["options"] = desc
            doc["files"] = {"audio": str(apath.relative_to(K)), "text": str(tpath.relative_to(K))}
            doc["memory_after_compile"] = R.memory()
            arun = G.AudioRunner(acm, "audio_1001")
            trun = R.Runner(tcm, next(iter(tcm.get_signature_list())))
            rows, _ = oracle_audio_rows()
            row = next(r for r in rows if r["key"] == PIPE_ROW)
            x16, _ = G.clip_samples(row["id"])

            def one():
                t0 = time.perf_counter()
                xa, info = AH.prepare(x16)
                t1 = time.perf_counter()
                out, _, _, _ = arun.timed({k: np.ascontiguousarray(v) for k, v in xa.items()})
                pre = out[:info["P"]]
                t2 = time.perf_counter()
                xt = text_inputs(row, pre, 256)
                t3 = time.perf_counter()
                s, _, _, _ = trun.timed(xt)
                t4 = time.perf_counter()
                p = LG.readout(s[:row["P"] + row["n"]], row)[0]
                t5 = time.perf_counter()
                return p, [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3,
                           (t5 - t0) * 1e3]
            for _ in range(WARMUP):
                one()
            parts, probs = [], None
            for _ in range(REPS):
                p, ms = one()
                parts.append([round(v, 3) for v in ms])
                probs = probs or p
            cols = ["mel_and_inputs", "audio_graph_T1001", "text_inputs", "text_graph_L256", "readout", "total"]
            doc["pipeline"] = {"row": PIPE_ROW, "what": "10 s audio, 1 question: host mel -> audio graph T1001 -> "
                                                        "prefix rows -> text inputs -> text graph L256 fp16 -> readout",
                               "columns": cols, "calls_ms": parts,
                               "ms": {c: TM.stats([q[i] for q in parts]) for i, c in enumerate(cols)},
                               "probs": probs, "probs_oracle": row["probs"],
                               "max_abs_dp_vs_oracle": max(abs(u - v) for u, v in zip(probs, row["probs"]))}
            doc["memory_end"] = R.memory()
            arun.close()
            trun.close()
        doc["status"] = "OK"
    except BaseException as e:  # recorded
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()[-3000:]
    doc["finished"] = now()
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["status"] == "OK" else 1


def speed_spawn(args, tag):
    import os
    import subprocess

    SPEED_TMP.mkdir(parents=True, exist_ok=True)
    out = SPEED_TMP / f"{tag}.json"
    if out.exists():
        out.unlink()
    log = K / f"logs/r7_timing_{tag}.child.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--speed-child", "--out", str(out)] + args
    t0 = time.time()
    with open(log, "w") as fo:
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, status, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    doc = json.loads(out.read_text()) if out.exists() else {"status": "FAIL", "error": "child wrote no record"}
    doc.update(child_returncode=rc, child_ru_maxrss_bytes=int(ru.ru_maxrss),
               child_seconds_wall=round(time.time() - t0, 1), child_log=str(log.relative_to(K)))
    if rc != 0:
        doc["child_log_tail"] = log.read_text(errors="replace").splitlines()[-30:]
    return doc


def speed_window(a):
    """One quiet window = one backend: per form, a contention gate then a fresh child per audio bucket, then the
    pipeline child (not for gpu_default: it fails parity; its audio sets are lever data)."""
    import timing_mac as TM

    forms = a.forms.split(",")
    buckets = [int(t) for t in a.buckets.split(",")]
    out = K / f"results/timing_mac_r7_{a.backend}.json"
    n = 2
    while out.exists():
        out = K / f"results/timing_mac_r7_{a.backend}_take{n}.json"
        n += 1
    doc = {"what": f"round 7 Mac timing, backend {a.backend}, audio forms {forms}, buckets {buckets}, one quiet window",
           "protocol": f"{WARMUP} warm-up calls then {REPS} timed calls per set; one fresh process per set; per call "
                       "[wall clock ms at start, ms write + run + read back, ms run only]",
           "gate_rule": "round 4: before each set no process outside this tree above 120 % CPU and CPU idle >= 50 % on "
                        "the second `top -l 2 -s 1` sample (10 s polls up to 300 s); load recorded",
           "backend": a.backend, "window_lock_line_at_start": TM.lock_line(), "started": now(), "sets": {}}
    rc = 0
    plan = [(f, "audio", T) for f in forms for T in buckets]
    if a.backend != "gpu_default":
        plan += [(f, "pipeline", 1001) for f in forms]
    for form, kind, T in plan:
        key = f"{form}_{kind}_T{T}"
        gate = TM.load_gate()
        st = {"load_gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["sets"][key] = st
            rc = 3
            break
        args = ["--backend", a.backend, "--form", form, "--kind", kind, "--T", str(T)]
        child = speed_spawn(args, f"{a.backend}_{key}")
        st["load_after"] = {"load": TM.load_avg(), "at": now()}
        w = child.get("workload") or {}
        sp = None
        if w:
            m = w["ms_write_run_read"]
            sp = round((m["median"] - m["min"]) / m["min"], 4)
            if a.backend == "cpu" and child.get("status") == "OK" and sp > TM.SPREAD_MAX:
                st["first_attempt"] = {"result": child, "spread": sp}
                st["retake_gate"] = TM.load_gate()
                child = speed_spawn(args, f"{a.backend}_{key}_retake")
                m = (child.get("workload") or {}).get("ms_write_run_read") or m
                sp = round((m["median"] - m["min"]) / m["min"], 4)
                st["retake_spread_still_over"] = sp > TM.SPREAD_MAX
        st.update(spread_median_over_min_minus_1=sp, result=child,
                  status="measured" if child.get("status") == "OK" else f"failed: {child.get('error')}")
        doc["sets"][key] = st
        print(f"{a.backend} {key}: {st['status']} "
              + (f"{w['ms_write_run_read']['median']} ms" if w else
                 (f"pipeline {child['pipeline']['ms']['total']['median']} ms" if child.get("pipeline") else "")),
              flush=True)
    doc["finished"] = now()
    doc["window_lock_line_at_end"] = TM.lock_line()
    write_json(out, doc)
    print(json.dumps({"out": str(out.relative_to(K)), "rc": rc}))
    return rc


def speed_aggregate(a):
    """results/timing_mac_r7.json from the window files (the last measured take per backend): the audio graph per
    bucket and form, the host mel, the 10 s question end to end, and the sum the launch names (mel + audio graph +
    round 4's text L256 row)."""
    import timing_mac as TM

    r4 = {}
    for be, fname in (("gpu_fp32", "timing_mac_r4_gpu_fp32_L256.json"), ("cpu", "timing_mac_r4_cpu_L256.json")):
        d = json.loads((K / "results" / fname).read_text())
        w = d["sets"]["fp16"]["result"]["workloads"]["f"]
        r4[be] = {"file": f"results/{fname}", "workload": "f: aud_01/topic text row at L256, fp16 file",
                  "ms_write_run_read": w["ms_write_run_read"]}
    out = {"what": "round 7 Mac timing of the audio prefix graph (ai-edge-litert 2.2.0 CompiledModel, Mac M4 Max)",
           "written": now(), "protocol": f"{WARMUP} warm-up then {REPS} timed calls; one fresh process per set; "
                                         "1 call = write the 5 inputs + run + read the prefix back",
           "windows": {}, "audio_graph": {}, "host_mel": {}, "pipeline_10s_question": {}, "sum_10s_question": {},
           "text_row_round4": r4}
    for p in sorted((K / "results").glob("timing_mac_r7_*.json")):
        if p.name == "timing_mac_r7.json":
            continue
        d = json.loads(p.read_text())
        be = d["backend"]
        out["windows"][p.name] = {"backend": be, "label_lock_line": d.get("window_lock_line_at_start"),
                                  "started": d.get("started"), "finished": d.get("finished")}
        for key, st in d["sets"].items():
            if st.get("status") != "measured":
                continue
            r = st["result"]
            gate = st["load_gate"]
            cond = {"load_at_start": gate.get("load_at_start"),
                    "idle_pct_at_start": (gate.get("contention") or {}).get("idle_pct"),
                    "spread_median_over_min_minus_1": st.get("spread_median_over_min_minus_1"),
                    "retake": "first_attempt" in st, "window_file": p.name}
            mem = r.get("memory_after_compile") or {}
            if r["kind"] == "audio":
                w = r["workload"]
                out["audio_graph"].setdefault(be, {}).setdefault(r["form"], {})[str(r["T"])] = {
                    "file": r["file"], "file_bytes": r["file_bytes"], "clip": r["clip"]["name"],
                    "clip_seconds": r["clip"]["seconds"], "P": r["clip"]["P"],
                    "ms_write_run_read": w["ms_write_run_read"], "ms_run_only": w["ms_run_only"],
                    "compile_seconds": r["compile_seconds"], "is_fully_accelerated": r.get("is_fully_accelerated"),
                    "phys_footprint_after_compile": mem.get("phys_footprint"),
                    "lifetime_max_phys_footprint": (r.get("memory_end") or {}).get("lifetime_max_phys_footprint"),
                    "child_ru_maxrss_bytes": r.get("child_ru_maxrss_bytes"), "finite": w["finite"],
                    "first_call_prefix_vs_reference_max_abs": w["first_call_prefix_vs_reference"]["max_abs"], **cond}
            else:
                for T, m in r["mel"].items():
                    out["host_mel"].setdefault(be, {}).setdefault(r["form"], {})[T] = {
                        "clip": m["clip"], "seconds": m["seconds"], "ms": m["ms"]}
                pl = r["pipeline"]
                out["pipeline_10s_question"].setdefault(be, {})[r["form"]] = {
                    "row": pl["row"], "ms_median": {c: pl["ms"][c]["median"] for c in pl["columns"]},
                    "ms_total": pl["ms"]["total"], "max_abs_dp_vs_oracle": pl["max_abs_dp_vs_oracle"],
                    "files": r["files"], "compile_seconds_audio": r["compile_seconds_audio"],
                    "compile_seconds_text": r["compile_seconds_text"],
                    "phys_footprint_after_compile": mem.get("phys_footprint"), **cond}
    for be in ("cpu", "gpu_fp32"):
        for form in ("fp32", "fp16"):
            ag = ((out["audio_graph"].get(be) or {}).get(form) or {}).get("1001")
            mel = ((out["host_mel"].get(be) or {}).get(form) or {}).get("1001")
            if ag and mel:
                s = mel["ms"]["median"] + ag["ms_write_run_read"]["median"] + r4[be]["ms_write_run_read"]["median"]
                out["sum_10s_question"].setdefault(be, {})[form] = {
                    "host_mel_ms": mel["ms"]["median"], "audio_graph_T1001_ms": ag["ms_write_run_read"]["median"],
                    "text_L256_row_ms_round4": r4[be]["ms_write_run_read"]["median"], "sum_ms": round(s, 3),
                    "note": "three medians from two windows (round 4's text row); the pipeline section is the "
                            "same question timed end to end in one process"}
    path = K / "results/timing_mac_r7.json"
    write_json(path, out)
    print(json.dumps({"audio_graph_ms": {be: {f: {T: v["ms_write_run_read"]["median"] for T, v in d2.items()}
                                              for f, d2 in d1.items()} for be, d1 in out["audio_graph"].items()},
                      "host_mel_ms": {be: {f: {T: v["ms"]["median"] for T, v in d2.items()} for f, d2 in d1.items()}
                                      for be, d1 in out["host_mel"].items()},
                      "pipeline": {be: {f: v["ms_median"] for f, v in d1.items()}
                                   for be, d1 in out["pipeline_10s_question"].items()},
                      "sum": out["sum_10s_question"]}, indent=1))
    return 0


LONG_REF = K / "out/r7_eager_prefix_T3001_long.npz"


def run_long(a):
    """The T3001 bucket has no oracle row: the provider's own Audio forward on aud_01 + aud_02 + aud_03 concatenated
    (27.2 s, the reference venv) is its reference -> out/r7_eager_prefix_T3001_long.npz (provider prefix and mel) and
    results/audio_eager_long.json (D1Audio eager at T3001 on the host mel and on the provider's mel, pad-content)."""
    import torch

    import audio_graph as G

    torch.set_num_threads(THREADS)
    t0 = time.time()
    A, prov, load = provider_model()
    x16, name = G.long_samples()
    keep = {}
    h = prov.frontend.register_forward_hook(lambda m, i, o: keep.__setitem__("fe", (o[0].detach().clone(), o[1])))
    with torch.no_grad():
        p_prov = prov(x16)[0].numpy()
    h.remove()
    mel_p, frames_p = keep["fe"][0][0].numpy(), int(keep["fe"][1][0])
    xh, info = AH.prepare(x16)
    assert info["T_b"] == 3001 and info["P"] == p_prov.shape[0] and info["frames"] == frames_p, (info, p_prov.shape)
    model, rep = G.build(3001)
    out_h = mine_run(model, xh)
    out_n = mine_run(model, AH.build_inputs(mel_p, frames_p, 3001))
    y = {k: v.copy() for k, v in xh.items()}
    y["mel"][0, :, frames_p:] = np.random.default_rng(5).standard_normal((128, 3001 - frames_p)).astype(np.float32) * 5
    out_p = mine_run(model, y)
    P = info["P"]
    np.savez(LONG_REF, provider_prefix=p_prov, provider_mel=mel_p, frames=np.array([frames_p]), mine_host=out_h[:P])
    doc = {"step": "round 7: the T3001 reference (no oracle row): the provider's Audio on a 27.2 s concatenation",
           "written": now(), "clip": name, "samples": int(x16.shape[0]), "seconds": round(x16.shape[0] / 16000, 4),
           "info": info, "provider_load": load, "torch": torch.__version__, "threads": THREADS,
           "host_mel_vs_provider_mel": diff(xh["mel"][0, :, :info["T"]], mel_p),
           "prefix_host_mel_vs_provider": diff(out_h[:P], p_prov),
           "prefix_provider_mel_vs_provider": diff(out_n[:P], p_prov),
           "pad_content_valid_rows_bit_equal": bool(np.array_equal(out_p[:P], out_h[:P])),
           "nonfinite_all_rows": int((~np.isfinite(out_h)).sum()), "reference_file": str(LONG_REF.relative_to(K)),
           "seconds_wall": round(time.time() - t0, 1)}
    out = K / "results/audio_eager_long.json"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {out}")
    write_json(out, doc)
    print(json.dumps({k: doc[k] for k in ("info", "host_mel_vs_provider_mel", "prefix_host_mel_vs_provider",
                                          "prefix_provider_mel_vs_provider", "pad_content_valid_rows_bit_equal")},
                     indent=1))
    return 0


FP16_MAX = 65504.0


def run_range(a):
    """Why Metal's default precision (fp16 activations) fails: the fp32 eager D1Audio on the 7 clips (host mel), the
    largest value each fp16 activation site would hold, valid rows only -> results/audio_fp16_range.json.
    Sites: every LayerNorm's input and its squared deviation (x - mean)^2 (the exported LayerNorm is MEAN /
    SQUARED_DIFFERENCE / RSQRT), every FFN hidden (linear1 output), the attention scores (ac + bd) * 0.125, the
    ConvModule's depthwise output, the subsampling conv outputs, the residual stream between layers."""
    import torch

    import audio_graph as G

    torch.set_num_threads(THREADS)
    sites = {}

    def note(name, v):
        v = float(v)
        e = sites.setdefault(name, {"max": 0.0, "clip": None})
        if v > e["max"]:
            e["max"], e["clip"] = v, cur["rid"]

    cur = {"rid": None, "P": None, "Pk": None}
    models = {}
    hooks = []
    for T in (1001, 2001):
        m, _ = G.build(T)
        models[T] = m
        for li, layer in enumerate(m.layers):
            for nm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
                def ln_hook(mod, inp, out, key=f"L{li:02d}.{nm}"):
                    x = inp[0][0, :cur["P"]].double()
                    dev = (x - x.mean(-1, keepdim=True)) ** 2
                    note(f"{key}.input_absmax", x.abs().max())
                    note(f"{key}.sq_dev_max", dev.max())
                    note(f"{key}.sq_dev_mean_max", dev.mean(-1).max())
                hooks.append(getattr(layer, nm).register_forward_hook(ln_hook))
            for nm in ("feed_forward1", "feed_forward2"):
                hooks.append(getattr(layer, nm).linear1.register_forward_hook(
                    lambda mod, i, o, key=f"L{li:02d}.{nm}.hidden": note(f"{key}_absmax", o[0, :cur["P"]].abs().max())))
            hooks.append(layer.conv.depthwise_conv.register_forward_hook(
                lambda mod, i, o, key=f"L{li:02d}.conv.depthwise": note(f"{key}_absmax", o[0, :, :cur["P"]].abs().max())))
            hooks.append(layer.register_forward_hook(
                lambda mod, i, o, key=f"L{li:02d}.output": note(f"{key}_absmax", o[0, :cur["P"]].abs().max())))
        for nm, mod in (("adapter.norm", m.adapter.norm), ("residual.ln", m.residual.ln)):
            def ln_hook2(mod, inp, out, key=nm):
                x = inp[0][0, :cur["P"]].double()
                dev = (x - x.mean(-1, keepdim=True)) ** 2
                note(f"{key}.input_absmax", x.abs().max())
                note(f"{key}.sq_dev_max", dev.max())
                note(f"{key}.sq_dev_mean_max", dev.mean(-1).max())
            hooks.append(mod.register_forward_hook(ln_hook2))
        for nm in ("conv0", "conv2", "conv3", "conv5", "conv6"):
            hooks.append(getattr(m.pre_encode, nm).register_forward_hook(
                lambda mod, i, o, key=f"pre_encode.{nm}": note(f"{key}_absmax", o.abs().max())))
        hooks.append(m.adapter.linear_1.register_forward_hook(
            lambda mod, i, o: note("adapter.linear_1_absmax", o[0, :cur["P"]].abs().max())))
        hooks.append(m.residual.register_forward_hook(
            lambda mod, i, o: note("prefix_absmax", o[0, :cur["P"]].abs().max())))

    # attention scores: recompute (ac + bd) * 0.125 for the valid block from the attention module's own tensors
    def attn_hook(mod, inp, out, li=None):
        x = inp[0]
        t = mod.t3
        with torch.no_grad():
            q = mod.linear_q(x).reshape(1, t, G.HEADS, G.D_K)
            k = mod.linear_k(x).reshape(1, t, G.HEADS, G.D_K).transpose(1, 2)
            ac = torch.matmul((q + mod.pos_bias_u).transpose(1, 2), k.transpose(2, 3))
            bd = torch.matmul((q + mod.pos_bias_v).transpose(1, 2), mod.p_t)
            bd = torch.nn.functional.pad(bd, (1, 0)).reshape(1, G.HEADS, 2 * t, t)[:, :, 1:].reshape(
                1, G.HEADS, t, 2 * t - 1)[:, :, :, :t]
            s = (ac + bd) * 0.125
            P = cur["P"]
            note(f"L{li:02d}.attn_scores_absmax", s[:, :, :P, :P].abs().max())
            note(f"L{li:02d}.attn_bd_unshifted_absmax", torch.matmul((q + mod.pos_bias_v).transpose(1, 2),
                                                                     mod.p_t)[:, :, :P].abs().max())
    for T, m in models.items():
        for li, layer in enumerate(m.layers):
            hooks.append(layer.self_attn.register_forward_hook(lambda mod, i, o, li=li: attn_hook(mod, i, o, li)))
    for rid in RECORDS:
        T_b = BUCKET_OF.get(rid, 1001)
        x16, _ = __import__("audio_graph").clip_samples(rid)
        xh, info = AH.prepare(x16, bucket=T_b)
        cur.update(rid=rid, P=info["P"])
        with torch.no_grad():
            models[T_b](**{k: torch.from_numpy(v) for k, v in xh.items()})
    for h in hooks:
        h.remove()
    over = {k: v for k, v in sites.items() if v["max"] > FP16_MAX}
    worst = sorted(sites.items(), key=lambda kv: -kv[1]["max"])[:25]
    doc = {"step": "round 7 step 4 (diagnosis): fp16 range of the audio graph's activation sites (eager fp32, host "
                   "mel, valid rows, 7 clips)", "written": now(), "fp16_max": FP16_MAX,
           "sites_measured": len(sites), "sites_over_fp16_max": len(over),
           "over": dict(sorted(over.items(), key=lambda kv: -kv[1]["max"])),
           "top25": [{"site": k, **v} for k, v in worst],
           "by_kind_max": {kind: max((v["max"] for k, v in sites.items() if kind in k), default=None)
                           for kind in ("sq_dev_max", "sq_dev_mean_max", "input_absmax", "hidden_absmax",
                                        "attn_scores_absmax", "depthwise_absmax", "output_absmax", "pre_encode")},
           "all": sites}
    out = K / "results/audio_fp16_range.json"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {out}")
    write_json(out, doc)
    print(json.dumps({k: doc[k] for k in ("sites_measured", "sites_over_fp16_max", "by_kind_max")}, indent=1))
    print(json.dumps(doc["top25"][:12], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--range", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--speed-child", action="store_true")
    ap.add_argument("--speed-agg", action="store_true")
    ap.add_argument("--forms", default="fp32,fp16")
    ap.add_argument("--buckets", default="501,1001,2001,3001")
    ap.add_argument("--kind", choices=("audio", "pipeline"))
    ap.add_argument("--T", type=int)
    ap.add_argument("--out")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--backend", choices=("cpu", "gpu", "gpu_fp32", "gpu_default"))
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.mel:
        return run_mel(a)
    if a.eager:
        return run_eager(a)
    if a.speed_child:
        return speed_child(a)
    if a.speed:
        return speed_window(a)
    if a.speed_agg:
        return speed_aggregate(a)
    if a.range:
        return run_range(a)
    if a.long and not a.lite:
        return run_long(a)
    if a.lite:
        if a.backend == "gpu" and not a.child:
            return run_lite_gpu_parent(a)
        return run_lite(a)
    if a.score:
        return score_all(a)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
