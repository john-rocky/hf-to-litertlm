"""Assemble the Hub repo directory from the chosen forms.

  python assemble_ship.py --out out/ship/EMA-Lightning-LiteRT --forms text=fp16,sound=fp32,decoder=fp16 --split
Copies the graphs of each kind (every bucket) with their form suffix, ema_step_table.npz (split form), the host loop
(ema_litert.py, hostloop_e2e.py with its weight-path dependence removed by config.json), ship_docs/README.md, and writes
config.json (vocab, buckets, times, hop, sample rate, decoder stage lengths, file list with sha256 / bytes).
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import torch
from ema_lightning.decoder import load_decoder
from ema_lightning.model import load_acoustic
from ema_litert import LT_BUCKETS, L_BUCKETS, P_BUCKETS, stage_lengths

HERE = Path(__file__).resolve().parent
W = HERE / "precheck" / "weights"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--multi", action="store_true", help="ship the multi-signature files (one per kind) instead of per-bucket files")
    a = ap.parse_args()
    forms = dict(kv.split("=") for kv in a.forms.split(",") if kv)
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    m = load_acoustic(W / "ema.pt", "cpu")
    d = load_decoder(W / "decoder.pt", "cpu")
    files = []

    def take(kind, name):
        form = forms.get(kind, forms.get("sound", "fp32") if kind in ("align", "step") else "fp32")
        src = HERE / "out" / form / f"{name}_{form}.tflite"
        dst = out / src.name
        shutil.copyfile(src, dst)
        files.append(dict(file=dst.name, kind=kind, form=form, bytes=dst.stat().st_size, sha256=sha(dst)))

    if a.multi:
        assert a.split, "the multi-signature set is the split form (align + step)"
        for kind in ("text", "align", "step", "decoder"):
            take(kind, f"ema_{kind}_multi")
        form = forms.get("step", forms.get("sound", "fp32"))
        shutil.copyfile(HERE / "out" / form / "ema_step_table.npz", out / "ema_step_table.npz")
        files.append(dict(file="ema_step_table.npz", kind="table", form="fp32", bytes=(out / "ema_step_table.npz").stat().st_size,
                          sha256=sha(out / "ema_step_table.npz")))
    for L in ([] if a.multi else L_BUCKETS):
        take("text", f"ema_text_L{L}")
    if a.multi:
        pass
    elif a.split:
        for L, T in LT_BUCKETS:
            take("align", f"ema_align_L{L}_T{T}")
        for T in sorted({T for _, T in LT_BUCKETS}):
            take("step", f"ema_step_T{T}")
        form = forms.get("step", forms.get("sound", "fp32"))
        shutil.copyfile(HERE / "out" / form / "ema_step_table.npz", out / "ema_step_table.npz")
        files.append(dict(file="ema_step_table.npz", kind="table", form="fp32", bytes=(out / "ema_step_table.npz").stat().st_size,
                          sha256=sha(out / "ema_step_table.npz")))
    else:
        for L, T in LT_BUCKETS:
            take("sound", f"ema_sound_L{L}_T{T}")
    for P in ([] if a.multi else P_BUCKETS):
        take("decoder", f"ema_decoder_P{P}")
    if a.multi:  # record each file's signatures (the bucket names) from the file itself
        from ai_edge_litert.interpreter import Interpreter
        for f in files:
            if f["file"].endswith(".tflite"):
                f["signatures"] = list(Interpreter(model_path=str(out / f["file"])).get_signature_list().keys())
    shutil.copyfile(HERE / "ship_docs" / "ema_host.py", out / "ema_host.py")  # the torch-free host loop (reads config.json)
    shutil.copyfile(HERE / "ship_docs" / "README.md", out / "README.md")
    cfg = dict(model="canberkkkkkk/ema-lightning", package="ema-lightning 1.1.2", sample_rate=48000, hop=int(d.hop), latent_dim=int(m.latent_dim),
               d=int(m.d), times=[float(t) for t in m.times], vocab=list(m.vocab), text_buckets=list(L_BUCKETS),
               sound_buckets=[list(x) for x in LT_BUCKETS], decoder_buckets=list(P_BUCKETS),
               decoder_stage_lengths={str(P): stage_lengths(d, P) for P in P_BUCKETS},
               aligner=dict(heads=int(m.aligner.h), pos_scale=float(m.aligner.pos_scale), lookback=int(m.aligner.lookback), lookahead=int(m.aligner.lookahead)),
               windows=dict(first=25, later=100, context=8), limits=dict(letters_per_piece=250, max_word_frames=250, max_frames=3000),
               split=a.split, multi=a.multi, forms=forms, files=files,
               source=dict(ema_pt_sha256=sha(W / "ema.pt"), decoder_pt_sha256=sha(W / "decoder.pt")))
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    total = sum(f["bytes"] for f in files)
    print(f"{out}: {len(files)} model files, {total / 1e6:.1f} MB")
    for f in files:
        print(f"  {f['file']:36s} {f['bytes'] / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
