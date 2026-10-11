"""Text to 48 kHz wav through the LiteRT graphs: the host loop that ships with the model.

  python hostloop_e2e.py --forms text=fp16,sound=fp32,decoder=fp16 [--split] --text "Merhaba dünya." --out out/e2e/x.wav
  python hostloop_e2e.py --forms ... --fixtures            # the six fixture texts, RTF and first-audio per text

Frontend: normalizer-tr (Rust, via its Python wheel) + the package's alphabet step; chunking = the package's chunker.
Bookkeeping: ema_litert (numpy). Reports per piece: ms of text / align+steps (or sound) / each decoder window, the
time to the first window's audio (first-audio) and total time / audio seconds (RTF).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from ema_lightning.audio import write_wav
from ema_lightning.chunker import chunk
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic
from ema_litert import (Runner, bucket_for, decode_windows, decoder_masks, euler, noise_for, p_bucket, piece_ids,
                        sound_inputs, text_bucket, timeline)

HERE = Path(__file__).resolve().parent
W = HERE / "precheck" / "weights"
HOP, RATE = 1920, 48000


class EmaLiteRT:
    def __init__(self, forms, split, threads=4, multi=False):
        self.m = load_acoustic(W / "ema.pt", "cpu")  # vocab / stoi / times / latent_dim only; no torch inference here
        self.d = load_decoder(W / "decoder.pt", "cpu")  # stage lengths for the masks
        self.frontend = Frontend(self.m.vocab)
        self.forms, self.split, self.threads, self.multi = forms, split, threads, multi
        self.runners = {}
        tab = np.load(HERE / "out" / forms.get("step", forms.get("sound", "fp32")) / "ema_step_table.npz") if split else None
        self.table = [(float(t), tab["bc"][k:k + 1], tab["sg"][k:k + 1]) for k, t in enumerate(tab["times"])] if split else None
        self.times = [float(t) for t in self.m.times]

    def run_all(self, name, **kw):
        if name not in self.runners:
            kind = name.split("_")[1]
            form = self.forms.get(kind, self.forms.get("sound", "fp32") if kind in ("align", "step") else "fp32")
            if self.multi:  # one file per kind, the bucket is a signature named like the single-bucket file
                self.runners[name] = Runner(HERE / "out" / form / f"ema_{kind}_multi_{form}.tflite", self.threads, signature=name)
            else:
                self.runners[name] = Runner(HERE / "out" / form / f"{name}_{form}.tflite", self.threads)
        return self.runners[name](**kw)

    def run(self, name, **kw):
        return self.run_all(name, **kw)["output_0"]

    def piece(self, text, seed, speed=1.0):
        t = {}
        t0 = time.perf_counter()
        ids, cw, wstart = piece_ids(text, self.m.stoi)
        L = len(ids)
        Lb = text_bucket(L)
        IDS = np.zeros((1, Lb), np.int32); IDS[0, :L] = ids
        MASK = np.zeros((1, Lb), np.float32); MASK[0, :L] = 1
        out = self.run_all(f"ema_text_L{Lb}", ids=IDS, mask=MASK)
        h, dur = out["output_0"][0, :L], out["output_1"][0, :L]
        t["text_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        fw, fp, cg, fg = timeline(dur, cw, wstart, speed)
        T = len(fw)
        Lb2, Tb = bucket_for(L, T)
        noise = noise_for(seed, T, len(self.times), self.m.latent_dim)
        kw = sound_inputs(h, cg, fg, cw, fw, Lb2, Tb, noise)
        if self.split:
            cond = self.run(f"ema_align_L{Lb2}_T{Tb}", h=kw["h"], cg=kw["cg"], fg=kw["fg"], allow=kw["allow"])
            lat = euler(lambda **s: self.run(f"ema_step_T{Tb}", **s), kw["noise"][0], cond, self.table, kw["fbias"], self.times)[0, :T]
        else:
            lat = self.run(f"ema_sound_L{Lb2}_T{Tb}", **kw)[0, :T]
        t["sound_ms"] = (time.perf_counter() - t1) * 1e3
        audio, t["decode_ms"] = [], []
        for (A, B, s, e) in decode_windows(T):
            t2 = time.perf_counter()
            n = B - A
            P = p_bucket(n)
            Z = np.zeros((1, self.m.latent_dim, P), np.float32); Z[0, :, :n] = lat[A:B].T
            y = self.run(f"ema_decoder_P{P}", z=Z, **decoder_masks(self.d, n, P))[0]
            audio.append(y[(s - A) * HOP:(e - A) * HOP])
            t["decode_ms"].append((time.perf_counter() - t2) * 1e3)
            if len(audio) == 1:
                t["first_audio_ms"] = (time.perf_counter() - t0) * 1e3
        t["total_ms"] = (time.perf_counter() - t0) * 1e3
        return np.concatenate(audio), dict(L=L, T=T, Lb=Lb2, Tb=Tb, **t)

    def say(self, text, seed=0, speed=1.0):
        pieces = chunk(self.frontend(text), speed)
        out, stats = [], []
        for i, (piece, pause) in enumerate(pieces):
            audio, st = self.piece(piece, (seed * 1_000_003 + i) % 2**63, speed)
            st["text"] = piece
            out.append(audio)
            if pause:
                out.append(np.zeros(int(pause * RATE), np.float32))
            stats.append(st)
        return np.concatenate(out), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="text=fp32,sound=fp32,decoder=fp32")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--multi", action="store_true", help="read the buckets as signatures of ema_<kind>_multi_<form>.tflite")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--text", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--reps", type=int, default=3, help="timing repeats per text (the last one is reported)")
    a = ap.parse_args()
    forms = dict(kv.split("=") for kv in a.forms.split(",") if kv)
    tts = EmaLiteRT(forms, a.split, a.threads, a.multi)
    texts = {}
    if a.fixtures:
        texts = {k: v["text"] for k, v in json.loads((HERE / "fixtures" / "manifest.json").read_text())["texts"].items()}
    if a.text:
        texts["cli"] = a.text
    report = {"forms": forms, "split": a.split, "threads": a.threads, "texts": {}}
    tts.say("Isınma turu, bir iki üç.", 0)
    for tag, text in texts.items():
        for _ in range(a.reps):
            audio, stats = tts.say(text, a.seed)
        secs = len(audio) / RATE
        tot = sum(s["total_ms"] for s in stats)
        rec = dict(seconds=secs, total_ms=tot, rtf=tot / 1e3 / secs, first_audio_ms=stats[0]["first_audio_ms"], pieces=stats)
        report["texts"][tag] = rec
        print(f"{tag:8s} {secs:5.2f} s audio  total {tot:7.1f} ms  rtf {rec['rtf']:.3f}  first-audio {rec['first_audio_ms']:6.1f} ms  "
              f"(text {stats[0]['text_ms']:.1f} / sound {stats[0]['sound_ms']:.1f} / decode {' + '.join(f'{x:.0f}' for x in stats[0]['decode_ms'])} ms; L{stats[0]['L']} T{stats[0]['T']} -> L{stats[0]['Lb']} T{stats[0]['Tb']})")
        if a.out:
            Path(HERE / a.out).parent.mkdir(parents=True, exist_ok=True)
            write_wav(HERE / a.out if len(texts) == 1 else HERE / a.out.replace(".wav", f".{tag}.wav"), audio, RATE)
    if a.fixtures:
        lab = a.forms.replace("=", "").replace(",", "_") + ("_split" if a.split else "") + ("_multi" if a.multi else "")
        (HERE / "out" / "e2e").mkdir(parents=True, exist_ok=True)
        (HERE / "out" / "e2e" / f"hostloop_mac_{lab}_t{a.threads}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
