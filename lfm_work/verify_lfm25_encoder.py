#!/usr/bin/env python3
"""Verify LFM2.5-Encoder LiteRT exports against the PyTorch reference.

    python verify_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m

Checks, per tflite (fp32 + wi8fc):
  1. Per-token last_hidden_state parity vs unpadded PyTorch eager on 16
     sentences covering all 15 supported languages (encode_128, right-padded)
     — max|diff|, per-token corr, mean-pooled-embedding cosine.
  2. Padding exactness: encode_64 vs encode_128 vs encode_256 on the same
     sentence must agree at valid positions (static-shape contract).
  3. Fill-mask E2E via the mlm_128 signature: top-5 tokens vs PyTorch top-5
     on en/fr/ja/de cloze prompts.

Reference = HF AutoModelForMaskedLM eager fp32, unpadded exact-length forward
(the semantic our padded export contracts to match at valid positions).
"""
import json
import os
import sys

import types


class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None

    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub>"
_pp.__spec__ = None
_pp.__getattr__ = lambda n: _D()  # noqa: E731
sys.modules.setdefault("scipy.sparse.linalg._propack", _pp)

import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-Encoder-230M"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "out/lfm25_encoder_230m"

SENTENCES = [  # 15 supported languages + 1 extra en
    ("en", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("en", "On-device machine learning keeps user data private and fast."),
    ("de", "Die Katze schläft den ganzen Nachmittag auf dem warmen Fensterbrett."),
    ("es", "El tren llega a la estación central a las nueve de la mañana."),
    ("fr", "Le musée du Louvre abrite des milliers d'œuvres d'art célèbres."),
    ("it", "La pizza napoletana si prepara con pomodoro e mozzarella fresca."),
    ("nl", "De fietsen staan naast de gracht in het centrum van Amsterdam."),
    ("pl", "Warszawa jest największym miastem i stolicą Polski."),
    ("pt", "O rio Amazonas atravessa a floresta tropical mais extensa do mundo."),
    ("ar", "تشرق الشمس كل صباح فوق الجبال العالية في الصحراء."),
    ("hi", "भारत की राजधानी नई दिल्ली एक बहुत बड़ा शहर है।"),
    ("ja", "東京の桜は毎年春になると美しく咲きます。"),
    ("ru", "Московский метрополитен считается одним из красивейших в мире."),
    ("tr", "İstanbul Boğazı, Asya ile Avrupa kıtalarını birbirine bağlar."),
    ("vi", "Phở là món ăn truyền thống nổi tiếng của Việt Nam."),
    ("zh", "长城是中国古代最伟大的建筑工程之一。"),
]

CLOZE = [
    ("en", "The capital of France is [MASK]."),
    ("fr", "La capitale de la France est [MASK]."),
    ("de", "Die Hauptstadt von Deutschland ist [MASK]."),
    ("ja", "日本の首都は[MASK]です。"),
]


def load_ref():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    mlm = AutoModelForMaskedLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    assert float(mlm.lfm2.rotary_emb.inv_freq.min()) > 0
    return tok, mlm


def runner(path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    return {name: it.get_signature_runner(name) for name in it.get_signature_list()}


def pad_to(ids, S):
    n = len(ids)
    assert n <= S
    return (
        np.array([ids + [0] * (S - n)], dtype=np.int32),
        np.array([[1] * n + [0] * (S - n)], dtype=np.int32),
    )


def pooled(h, n):
    v = h[0, :n].mean(axis=0)
    return v / np.linalg.norm(v)


def main():
    tok, mlm = load_ref()
    report = {"model": MODEL}

    refs = []  # (lang, ids, torch last_hidden [n,1024])
    for lang, text in SENTENCES:
        ids = tok(text)["input_ids"]
        t = torch.tensor([ids])
        with torch.no_grad():
            h = mlm.lfm2(input_ids=t, use_cache=False).last_hidden_state[0].numpy()
        refs.append((lang, ids, h))
    print(f"reference done: {len(refs)} sentences, "
          f"len {min(len(i) for _, i, _ in refs)}-{max(len(i) for _, i, _ in refs)} tokens")

    for kind in ("fp32", "fp16", "wi8fc"):
        path = os.path.join(OUT_DIR, f"encoder_{kind}.tflite")
        if not os.path.exists(path):
            continue
        sigs = runner(path)
        maxd, corrs, coss = [], [], []
        for lang, ids, ref in refs:
            x, m = pad_to(ids, 128)
            got = sigs["encode_128"](input_ids=x, attention_mask=m)
            got = list(got.values())[0][0, : len(ids)]
            maxd.append(float(np.abs(ref - got).max()))
            corrs.append(float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]))
            coss.append(float(pooled(got[None], len(ids)) @ pooled(ref[None], len(ids))))
        report[kind] = {
            "token_maxdiff_max": max(maxd),
            "token_corr_min": min(corrs),
            "pooled_cos_min": min(coss),
            "pooled_cos_mean": float(np.mean(coss)),
        }
        print(f"[{kind}] token max|diff| {max(maxd):.3e} | token corr min "
              f"{min(corrs):.6f} | pooled cos min/mean {min(coss):.6f}/{np.mean(coss):.6f}")

        # cross-signature (padding) consistency on the first sentence
        _, ids0, _ = refs[0]
        outs = []
        for S in (64, 128, 256):
            x, m = pad_to(ids0, S)
            o = sigs[f"encode_{S}"](input_ids=x, attention_mask=m)
            outs.append(list(o.values())[0][0, : len(ids0)])
        d6428 = float(np.abs(outs[0] - outs[1]).max())
        d1228 = float(np.abs(outs[2] - outs[1]).max())
        report[kind]["cross_sig_maxdiff"] = max(d6428, d1228)
        print(f"[{kind}] cross-signature 64/128/256 max|diff| "
              f"{max(d6428, d1228):.3e}")

        # fill-mask E2E
        fm = {}
        for lang, text in CLOZE:
            prompt = text.replace("[MASK]", tok.mask_token)
            ids = tok(prompt)["input_ids"]
            pos = ids.index(tok.mask_token_id)
            with torch.no_grad():
                tl = mlm(input_ids=torch.tensor([ids])).logits[0, pos]
            ref_top = [tok.decode([t]).strip() for t in tl.topk(5).indices.tolist()]
            x, m = pad_to(ids, 128)
            o = sigs["mlm_128"](input_ids=x, attention_mask=m)
            gl = list(o.values())[0][0, pos]
            got_top = [tok.decode([t]).strip() for t in np.argsort(-gl)[:5].tolist()]
            fm[lang] = {"ref": ref_top, "got": got_top,
                        "top1_match": ref_top[0] == got_top[0],
                        "top5_overlap": len(set(ref_top) & set(got_top))}
            print(f"[{kind}] fill-mask {lang}: ref {ref_top} | got {got_top}")
        report[kind]["fill_mask"] = fm

    out = os.path.join(OUT_DIR, "verify_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("report:", out)


if __name__ == "__main__":
    main()
