"""Export the three graphs at fixed buckets with litert-torch (fp32; quantization is a separate step).

  python export_graphs.py --out out/fp32 --text 64 --sound 64x96 --decoder 48,120
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
import litert_torch as lt
from ema_lightning.decoder import load_decoder
from ema_lightning.model import load_acoustic
from ema_litert import (AlignGraph, DecoderGraph, SoundGraph, StepGraph, TextGraph, TextGraphEmb, embedding_table,
                        stage_lengths, step_table)

HERE = Path(__file__).resolve().parent
W = HERE / "precheck" / "weights"


def export(name, module, kwargs, out):
    t0 = time.perf_counter()
    with torch.no_grad():
        model = lt.signature(name, module, sample_kwargs=kwargs).convert()
    path = out / f"{name}_fp32.tflite"
    model.export(str(path))
    print(f"{path.name:36s} {path.stat().st_size / 1e6:6.2f} MB  {time.perf_counter() - t0:5.1f} s")


def export_multi(name, sigs, out):
    """One .tflite holding every bucket as its own signature (sigs: [(signature name, module, kwargs)]); the weights
    are shared constants, so the file is the size of one bucket."""
    t0 = time.perf_counter()
    conv = None
    with torch.no_grad():
        for sname, module, kwargs in sigs:
            conv = (conv or lt).signature(sname, module, sample_kwargs=kwargs)
        model = conv.convert()
    path = out / f"{name}_fp32.tflite"
    model.export(str(path))
    print(f"{path.name:36s} {path.stat().st_size / 1e6:6.2f} MB  {len(sigs)} signatures  {time.perf_counter() - t0:5.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fp32")
    ap.add_argument("--text", default="")      # e.g. 32,64,128,256
    ap.add_argument("--text-emb", default="")  # the same buckets with the letter embedding on the host (TextGraphEmb)
    ap.add_argument("--sound", default="")     # e.g. 32x48,64x96
    ap.add_argument("--decoder", default="")   # e.g. 48,120
    ap.add_argument("--align", default="")     # e.g. 32x48,64x96 (split form: aligner alone)
    ap.add_argument("--step", default="")      # e.g. 48,96,192,384,512 (split form: one DiT velocity pass)
    ap.add_argument("--multisig", action="store_true", help="also one multi-signature file per kind (text / align / step / decoder)")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    m = load_acoustic(W / "ema.pt", "cpu")
    d = load_decoder(W / "decoder.pt", "cpu")
    for L in [int(x) for x in a.text.split(",") if x]:
        export(f"ema_text_L{L}", TextGraph(m), dict(ids=torch.zeros(1, L, dtype=torch.int32), mask=torch.ones(1, L)), out)
    for L in [int(x) for x in a.text_emb.split(",") if x]:
        export(f"ema_textemb_L{L}", TextGraphEmb(m), dict(emb=torch.zeros(1, L, m.d), mask=torch.ones(1, L)), out)
    if a.text_emb:
        np.save(out / "ema_embedding_table.npy", embedding_table(m))
        print("ema_embedding_table.npy", embedding_table(m).shape)
    for spec in [x for x in a.sound.split(",") if x]:
        L, T = (int(x) for x in spec.split("x"))
        kw = dict(h=torch.zeros(1, L, m.d), cg=torch.zeros(1, L), fg=torch.zeros(1, T), allow=torch.ones(1, 1, T, L),
                  fbias=torch.zeros(1, 1, 1, T), noise=torch.zeros(1, len(m.times), T, m.latent_dim))
        export(f"ema_sound_L{L}_T{T}", SoundGraph(m), kw, out)
    for spec in [x for x in a.align.split(",") if x]:
        L, T = (int(x) for x in spec.split("x"))
        kw = dict(h=torch.zeros(1, L, m.d), cg=torch.zeros(1, L), fg=torch.zeros(1, T), allow=torch.ones(1, 1, T, L))
        export(f"ema_align_L{L}_T{T}", AlignGraph(m), kw, out)
    for T in [int(x) for x in a.step.split(",") if x]:
        kw = dict(x=torch.zeros(1, T, m.latent_dim), cond=torch.zeros(1, T, m.d), bc=torch.zeros(1, 6, m.d),
                  sg=torch.zeros(1, 2, m.d), fbias=torch.zeros(1, 1, 1, T))
        export(f"ema_step_T{T}", StepGraph(m), kw, out)
    if a.step:
        tab = step_table(m)
        np.savez(out / "ema_step_table.npz", times=np.array([t for t, _, _ in tab], np.float32),
                 bc=np.concatenate([b for _, b, _ in tab]), sg=np.concatenate([g for _, _, g in tab]))
        print("ema_step_table.npz", [t for t, _, _ in tab])
    for P in [int(x) for x in a.decoder.split(",") if x]:
        kw = {"z": torch.zeros(1, d.pre.in_channels, P)}
        kw.update({f"m{i}": torch.ones(1, 1, n) for i, n in enumerate(stage_lengths(d, P))})
        export(f"ema_decoder_P{P}", DecoderGraph(d), kw, out)
    if a.multisig:  # one file per kind, every bucket a signature named like the single-bucket files
        from ema_litert import LT_BUCKETS, L_BUCKETS, P_BUCKETS
        tg = TextGraph(m)
        export_multi("ema_text_multi", [(f"ema_text_L{L}", tg, dict(ids=torch.zeros(1, L, dtype=torch.int32), mask=torch.ones(1, L)))
                                        for L in L_BUCKETS], out)
        ag = AlignGraph(m)
        export_multi("ema_align_multi", [(f"ema_align_L{L}_T{T}", ag, dict(h=torch.zeros(1, L, m.d), cg=torch.zeros(1, L), fg=torch.zeros(1, T),
                                                                           allow=torch.ones(1, 1, T, L))) for L, T in LT_BUCKETS], out)
        sg_ = StepGraph(m)
        export_multi("ema_step_multi", [(f"ema_step_T{T}", sg_, dict(x=torch.zeros(1, T, m.latent_dim), cond=torch.zeros(1, T, m.d),
                                                                     bc=torch.zeros(1, 6, m.d), sg=torch.zeros(1, 2, m.d), fbias=torch.zeros(1, 1, 1, T)))
                                        for T in sorted({T for _, T in LT_BUCKETS})], out)
        dg = DecoderGraph(d)
        export_multi("ema_decoder_multi", [(f"ema_decoder_P{P}", dg, {"z": torch.zeros(1, d.pre.in_channels, P),
                                                                       **{f"m{i}": torch.ones(1, 1, n) for i, n in enumerate(stage_lengths(d, P))}})
                                           for P in P_BUCKETS], out)
        tab = step_table(m)
        np.savez(out / "ema_step_table.npz", times=np.array([t for t, _, _ in tab], np.float32),
                 bc=np.concatenate([b for _, b, _ in tab]), sg=np.concatenate([g for _, _, g in tab]))


if __name__ == "__main__":
    main()
