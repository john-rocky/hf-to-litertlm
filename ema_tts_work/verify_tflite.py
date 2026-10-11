"""Run the exported graphs on the oracle fixtures through the host bookkeeping and compare every stage.

  python verify_tflite.py --dir out/fp32 [--tags readme1,own1] [--wav out/fp32/wav]
Prints, per fixture: max|dh|, max|ddur|, frames equal?, max|dlat|, corr(lat), max|daudio|, corr(audio).
"""
import argparse, json
from pathlib import Path
import numpy as np
from ema_lightning.audio import write_wav
from ema_lightning.decoder import load_decoder
from ema_lightning.model import load_acoustic
from ema_litert import (Runner, bucket_for, decode_windows, decoder_masks, euler, p_bucket, sound_inputs, text_bucket, timeline)

HERE = Path(__file__).resolve().parent
W = HERE / "precheck" / "weights"
FX = HERE / "fixtures"


def corr(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.dot(a - a.mean(), b - b.mean()) / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/fp32")
    ap.add_argument("--tags", default="")
    ap.add_argument("--wav", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--suffix", default="fp32")
    ap.add_argument("--forms", default="", help="per-stage form, e.g. text=fp16,sound=fp32,decoder=fp16 (dir = out/<form>)")
    ap.add_argument("--label", default="")
    ap.add_argument("--split", action="store_true", help="use ema_align_* + ema_step_* (host Euler) instead of ema_sound_*")
    ap.add_argument("--multi", action="store_true", help="read each bucket as a signature of ema_<kind>_multi_<form>.tflite")
    a = ap.parse_args()
    D = HERE / a.dir
    forms = dict(kv.split("=") for kv in a.forms.split(",") if kv)
    m = load_acoustic(W / "ema.pt", "cpu")
    d = load_decoder(W / "decoder.pt", "cpu")
    manifest = json.loads((FX / "manifest.json").read_text())
    tags = a.tags.split(",") if a.tags else list(manifest["texts"])
    runners, report = {}, {}

    def run(name, **kw):
        if name not in runners:
            kind = name.split("_")[1]
            form = forms.get(kind, forms.get("sound", a.suffix) if kind in ("align", "step") else a.suffix)
            if a.multi:  # one file per kind, the bucket is a signature of it
                path = (HERE / "out" / form if forms else D) / f"ema_{kind}_multi_{form}.tflite"
                runners[name] = Runner(path, a.threads, signature=name)
            else:
                path = (HERE / "out" / form if forms else D) / f"{name}_{form}.tflite"
                runners[name] = Runner(path, a.threads)
        return runners[name](**kw)

    for tag in tags:
        fx = np.load(FX / f"{tag}.npz")
        ids, cw, wstart = fx["ids"], fx["cw"], fx["wstart"]
        L = len(ids)
        Lb = text_bucket(L)
        IDS = np.zeros((1, Lb), np.int32); IDS[0, :L] = ids
        MASK = np.zeros((1, Lb), np.float32); MASK[0, :L] = 1
        o = run(f"ema_text_L{Lb}", ids=IDS, mask=MASK)
        h, dur = o["output_0"][0, :L], o["output_1"][0, :L]
        dh, ddur = np.abs(h - fx["h"]).max(), np.abs(dur - fx["dur"]).max()
        fw, fp, cg, fg = timeline(dur, cw, wstart)
        T = len(fw)
        frames_ok = (T == len(fx["fw"])) and bool((fw == fx["fw"]).all())
        Lb2, Tb = bucket_for(L, T)
        kw = sound_inputs(h, cg, fg, cw, fw, Lb2, Tb, fx["noise"][:, :T] if T <= fx["noise"].shape[1] else fx["noise"])
        if a.split:
            form = forms.get("step", forms.get("sound", a.suffix)) if forms else a.suffix
            tab = np.load((HERE / "out" / form if forms else D) / "ema_step_table.npz")
            table = [(float(t), tab["bc"][k:k + 1], tab["sg"][k:k + 1]) for k, t in enumerate(tab["times"])]
            cond = run(f"ema_align_L{Lb2}_T{Tb}", h=kw["h"], cg=kw["cg"], fg=kw["fg"], allow=kw["allow"])["output_0"]
            step = lambda **s: run(f"ema_step_T{Tb}", **s)["output_0"]
            lat = euler(step, kw["noise"][0], cond, table, kw["fbias"], list(tab["times"]))[0, :T]
        else:
            lat = run(f"ema_sound_L{Lb2}_T{Tb}", **kw)["output_0"][0, :T]
        n = min(T, len(fx["latents"]))
        dlat, clat = np.abs(lat[:n] - fx["latents"][:n]).max(), corr(lat[:n], fx["latents"][:n])
        audio = []
        for (A, B, s, e) in decode_windows(T):
            nwin = B - A
            P = p_bucket(nwin)
            Z = np.zeros((1, d.pre.in_channels, P), np.float32); Z[0, :, :nwin] = lat[A:B].T
            y = run(f"ema_decoder_P{P}", z=Z, **decoder_masks(d, nwin, P))["output_0"][0]
            audio.append(y[(s - A) * d.hop:(e - A) * d.hop])
        audio = np.concatenate(audio)
        n = min(len(audio), len(fx["audio"]))
        diff = audio[:n] - fx["audio"][:n]
        daud, caud = np.abs(diff).max(), corr(audio[:n], fx["audio"][:n])
        snr = 10 * np.log10(float(np.sum(fx["audio"][:n] ** 2)) / max(float(np.sum(diff ** 2)), 1e-30))
        at = int(np.abs(diff).argmax()); wide = int((np.abs(diff) > daud / 4).sum())
        report[tag] = dict(L=L, T=T, frames_ok=frames_ok, dh=float(dh), ddur=float(ddur), dlat=float(dlat), corr_lat=clat,
                           daudio=float(daud), corr_audio=caud, snr_db=float(snr), argmax_s=at / 48000, wide=wide,
                           len_ok=len(audio) == len(fx["audio"]))
        print(f"{tag:8s} L={L:3d} T={T:3d} frames_ok={frames_ok} |dh|={dh:.2e} |ddur|={ddur:.2e} |dlat|={dlat:.2e} corr_lat={clat:.6f} "
              f"|daudio|={daud:.2e} corr_audio={caud:.6f} snr={snr:5.1f} dB (max at {at/48000:5.2f} s, {wide} smp > max/4) len_ok={len(audio) == len(fx['audio'])}")
        if a.wav:
            Path(HERE / a.wav).mkdir(parents=True, exist_ok=True)
            write_wav(HERE / a.wav / f"{tag}.{a.label or a.suffix}.wav", audio, 48000)
    label = a.label or ((a.forms.replace("=", "").replace(",", "_") if a.forms else a.suffix) + ("_split" if a.split else ""))
    outdir = (HERE / "out" / "mixed") if a.forms else D
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"verify_{label}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
