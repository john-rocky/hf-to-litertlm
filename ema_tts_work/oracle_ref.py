"""Reference run of the pip package (ema-lightning 1.1.2, CPU fp32, eager) on fixed texts and seeds.

Writes, per text: fixtures/<tag>.npz with every tensor that crosses a graph boundary (ids, cw, wstart, h, dur,
fw, fp, cg, fg, noise, latents, per-window decoder input/output) and fixtures/<tag>.ref.wav from the public
say() API, plus fixtures/manifest.json with the normalized text, frames, seconds and stage timings.
The weights are read from local files (not the shared HF cache), so the run is pinned by sha256.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from ema_lightning.api import EMA
from ema_lightning.audio import write_wav
from ema_lightning.decoder import load_decoder
from ema_lightning.engine import CONTEXT, windows
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic

HERE = Path(__file__).resolve().parent
W = HERE / "precheck" / "weights"
OUT = HERE / "fixtures"
TEXTS = {  # README "Listen" samples (seed 0 there) + three of our own, chosen to exercise numbers, dates and a long sentence
    "readme1": "Model sadece 8.6 milyon parametre ve çok hızlı.",
    "readme2": "15 Ekim 2026 Çarşamba günü saat 14:30'da, 1.250.000 TL tutarındaki ödemeniz hesabınıza yatırılacak.",
    "readme3": "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor, martılar ise teknelerin etrafında dönerek şanslarını deniyordu.",
    "own1": "Merhaba, bu ses telefonunuzun üzerinde üretiliyor.",
    "own2": "İstanbul'dan Ankara'ya trenle yaklaşık 4 saat 30 dakika sürüyor; bilet fiyatı 850 TL.",
    "own3": "Yapay zekâ modelleri artık cebimizde çalışıyor ve internet bağlantısı olmadan da konuşabiliyor.",
}
SEED = 0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    OUT.mkdir(exist_ok=True)
    model = load_acoustic(W / "ema.pt", "cpu")
    decoder = load_decoder(W / "decoder.pt", "cpu")
    tts = EMA._from_parts(model, decoder, Frontend(model.vocab), "cpu")
    eng = tts._engine
    manifest = {"weights": {"ema.pt": sha(W / "ema.pt"), "decoder.pt": sha(W / "decoder.pt")},
                "torch": torch.__version__, "threads": torch.get_num_threads(), "seed": SEED, "texts": {}}
    tts.say("Isınma turu.", seed=0)  # warm-up
    for tag, text in TEXTS.items():
        t0 = time.perf_counter()
        speech = tts.say(text, seed=SEED)
        t_say = time.perf_counter() - t0
        write_wav(OUT / f"{tag}.ref.wav", speech.audio, speech.sample_rate)
        pieces = tts._pieces(text, 1.0, SEED)
        rec = {"text": text, "normalized": [p.text for p in pieces], "pieces": len(pieces),
               "seconds": float(speech.duration), "say_s": t_say, "rtf_say": t_say / float(speech.duration)}
        assert len(pieces) == 1, (tag, len(pieces))  # the fixture texts are single pieces
        p = pieces[0]
        with torch.no_grad():
            t1 = time.perf_counter(); eng.plan([p], 1.0); t_plan = time.perf_counter() - t1
            noise = eng.noise(p)
            t1 = time.perf_counter(); eng.think([p]); t_think = time.perf_counter() - t1
            t1 = time.perf_counter()
            wins = [(max(0, s - CONTEXT), min(p.frames, e + CONTEXT), s, e) for s, e in windows(p.frames)]
            dec_in, dec_out, audio = [], [], []
            for a, b, s, e in wins:
                z = p.latents[a:b].T[None]
                y = decoder(z, None)
                dec_in.append(z[0].numpy()); dec_out.append(y[0].numpy())
                audio.append(y[0, (s - a) * decoder.hop:(e - a) * decoder.hop].numpy())
            t_dec = time.perf_counter() - t1
        audio = np.concatenate(audio)
        # the same bookkeeping the graphs will receive from the host (cg / fg as in sound_stage + aligner letter_pos)
        cw, wstart, dur, fw, fp = p.cw.numpy(), p.wstart.numpy(), p.dur.numpy(), p.fw.numpy(), p.fp.numpy()
        n_words = int(cw[-1]) + 1
        c = np.maximum(dur, 1e-4).astype(np.float32); done = np.cumsum(c); before = done - c
        total = np.zeros(n_words, np.float32); np.add.at(total, cw, c)
        cp = np.clip((done - before[wstart] - 0.5 * c) / np.maximum(total[cw], 1e-8), 0, 1).astype(np.float32)
        wlen = np.maximum(np.bincount(cw, minlength=n_words).astype(np.float32), 1); woff = np.cumsum(wlen) - wlen
        cg = (woff[cw] + cp * wlen[cw]).astype(np.float32); fg = (woff[fw] + fp * wlen[fw]).astype(np.float32)
        full = speech.audio
        n = min(len(full), len(audio))
        rec.update({"letters": int(p.letters), "frames": int(p.frames), "windows": [(a, b, s, e) for a, b, s, e in wins],
                    "plan_s": t_plan, "think_s": t_think, "decode_s": t_dec,
                    "decomposed_vs_say_maxabs": float(np.abs(full[:n] - audio[:n]).max()), "len_say": len(full), "len_dec": len(audio)})
        np.savez(OUT / f"{tag}.npz", ids=p.ids.numpy(), cw=cw, wstart=wstart, h=p.h.numpy(), dur=dur, fw=fw, fp=fp,
                 cg=cg, fg=fg, noise=noise.numpy(), latents=p.latents.numpy(), audio=audio,
                 **{f"dec_in_{i}": z for i, z in enumerate(dec_in)}, **{f"dec_out_{i}": y for i, y in enumerate(dec_out)})
        manifest["texts"][tag] = rec
        print(f"{tag:8s} L={p.letters:3d} T={p.frames:3d} {rec['seconds']:.2f}s say {t_say:.3f}s rtf {rec['rtf_say']:.3f} "
              f"plan {t_plan*1e3:.0f}ms think {t_think*1e3:.0f}ms dec {t_dec*1e3:.0f}ms  decomp-vs-say {rec['decomposed_vs_say_maxabs']:.2e}")
        print("   norm:", p.text)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
