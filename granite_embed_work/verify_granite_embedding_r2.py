#!/usr/bin/env python3
"""Quality gates for the converted granite-embedding-*-multilingual-r2 artifacts.

    python verify_granite_embedding_r2.py <out_dir> [--gates A,B,C,D]

Same gate set as the Nemotron embedder (and it reuses that module's
model-agnostic metric + dataset loaders), but the model contract differs:
CLS pooling, 768 dims, pad id 0, and **no prompt prefixes** — this model's
`config_sentence_transformers.json` ships empty query/document prompts, unlike
the E5-style Nemotron where the prefix was load-bearing. Gate B therefore runs
bare text on purpose; the prefix A/B is included as a check that this really is
the right contract rather than an assumption carried over.

  A  card oracle   — the base card's documented 3x3 cross-lingual cosine matrix
  B  STS17         — multilingual + cross-lingual Spearman (graded)
  C  NanoSciFact   — retrieval nDCG@10 / recall@5
  D  mechanics     — cross-signature agreement, pad-content invariance, mask liveness
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# The STS17 / NanoSciFact loaders and the metric helpers are model-agnostic and
# already live with the Nemotron embedder converter — import them from there
# rather than keeping a second copy in sync.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))
from verify_nemotron3_embed import (
    load_nanoscifact, load_sts17, ndcg_at_k, spearman,
)

MODEL_DIR = os.environ.get("GRANITE_EMBED_MODEL",
                           "ibm-granite/granite-embedding-311m-multilingual-r2")
PAD_ID = 0
DIM = 768
SEQ_LENS = (64, 128, 256, 512)

QUERIES = [
    "What is the tallest mountain in Japan?",
    "Wer hat das Lied Achy Breaky Heart geschrieben?",
    "ドイツの首都はどこですか？",
]
PASSAGES = [
    "富士山は、静岡県と山梨県にまたがる活火山で、標高3776.12 mで日本最高峰の独立峰である。",
    "Achy Breaky Heart is a country song written by Don Von Tress. Originally titled Don't Tell My Heart and performed by The Marcy Brothers in 1991.",
    "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland. Die Stadt ist mit rund 3,7 Millionen Einwohnern die bevölkerungsreichste Kommune Deutschlands.",
]
CARD = np.array([
    [0.9393, 0.6899, 0.7627],
    [0.6780, 0.9598, 0.7062],
    [0.7818, 0.7342, 0.9172],
])


class TorchEncoder:
    name = "torch_fp32"

    def __init__(self, tok):
        from transformers import AutoModel
        from convert_granite_embedding_r2 import Embedder

        m = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
        self.mod = Embedder(m, m.config.sliding_window).eval()
        self.tok = tok

    def encode_ids(self, ids):
        t = torch.tensor([ids], dtype=torch.int32)
        m = torch.ones(1, len(ids), dtype=torch.int32)
        with torch.inference_mode():
            return self.mod(t, m).numpy()[0]


class TfliteEncoder:
    def __init__(self, path, name):
        from ai_edge_litert.interpreter import Interpreter

        self.name = name
        self.it = Interpreter(model_path=path, num_threads=os.cpu_count())
        self.runners = {}
        for S in SEQ_LENS:
            try:
                self.runners[S] = self.it.get_signature_runner(f"embed_{S}")
            except Exception:
                pass
        self.lens = sorted(self.runners)

    def encode_ids(self, ids):
        S = next((s for s in self.lens if len(ids) <= s), self.lens[-1])
        ids = ids[:S]
        n = len(ids)
        a = np.full((1, S), PAD_ID, dtype=np.int32)
        m = np.zeros((1, S), dtype=np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        return list(self.runners[S](input_ids=a, attention_mask=m).values())[0][0]


def encode_texts(enc, tok, texts, prefix="", max_tokens=512, log_every=0):
    import time
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(prefix + t, add_special_tokens=True)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s "
                  f"({el/(i+1)*1000:.0f} ms/text)", flush=True)
    return out


def gate_a(encoders, tok, report):
    print("\n=== GATE A: model-card oracle (external truth) ===")
    ref = None
    for enc in encoders:
        q = encode_texts(enc, tok, QUERIES)
        p = encode_texts(enc, tok, PASSAGES)
        s = q @ p.T
        row = {"max_abs_vs_card": float(np.abs(s - CARD).max()),
               "diagonal_is_argmax": bool((s.argmax(1) == np.arange(3)).all()),
               "diag": [round(float(x), 4) for x in np.diag(s)]}
        if ref is None:
            ref = s
        else:
            row["max_abs_vs_torch"] = float(np.abs(s - ref).max())
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} diag {row['diag']}  max|d| vs card "
              f"{row['max_abs_vs_card']:.4f}  argmax-diag {row['diagonal_is_argmax']}"
              + (f"  vs torch {row['max_abs_vs_torch']:.4f}"
                 if "max_abs_vs_torch" in row else ""))
    print("  card diag:", [round(float(x), 4) for x in np.diag(CARD)])


def gate_b(encoders, tok, report, limit, prefix_ab):
    print(f"\n=== GATE B: STS17 Spearman (<= {limit} pairs/lang, NO prefix) ===")
    data = load_sts17(limit)
    if not data:
        print("  no data — skipped")
        return
    for enc in encoders:
        per_lang = {}
        for pair, rows in data.items():
            a = encode_texts(enc, tok, [r[0] for r in rows])
            b = encode_texts(enc, tok, [r[1] for r in rows])
            per_lang[pair] = round(spearman((a * b).sum(1),
                                            np.array([r[2] for r in rows])), 4)
        report.setdefault("B_sts17", {})[enc.name] = {
            "per_lang": per_lang,
            "mean": round(float(np.mean(list(per_lang.values()))), 4)}
        print(f"  {enc.name:11s} mean {np.mean(list(per_lang.values())):.4f}  {per_lang}")

    if prefix_ab:
        # This model documents EMPTY prompts. Confirm that rather than assume it:
        # if a query:/passage: style prefix helped, the contract would be wrong.
        print("  -- prefix control (should NOT help; prompts are empty upstream) --")
        enc = encoders[-1]
        rows = data.get("en-en", [])[:100]
        gold = np.array([r[2] for r in rows])
        for tag, p in (("none", ""), ("query: ", "query: ")):
            a = encode_texts(enc, tok, [r[0] for r in rows], p)
            b = encode_texts(enc, tok, [r[1] for r in rows], p)
            print(f"     en-en with prefix {tag!r:10s} -> "
                  f"{spearman((a*b).sum(1), gold):+.4f}")


def gate_c(encoders, tok, report, n_docs):
    print(f"\n=== GATE C: NanoSciFact retrieval (corpus {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    print(f"  {len(queries)} queries, {len(docs)} docs")
    for enc in encoders:
        D = encode_texts(enc, tok, [t for _, t in docs], log_every=200)
        Q = encode_texts(enc, tok, [t for _, t in queries])
        sims = Q @ D.T
        doc_ids = [i for i, _ in docs]
        ndcgs, r5, r1 = [], [], []
        for qi, (qid, _) in enumerate(queries):
            g = gold.get(qid, set())
            if not g:
                continue
            order = np.argsort(-sims[qi])
            rel = [1.0 if doc_ids[j] in g else 0.0 for j in order[:10]]
            ndcgs.append(ndcg_at_k(rel, [1.0] * len(g), 10))
            r5.append(float(any(rel[:5])))
            r1.append(float(rel[0]))
        row = {"ndcg@10": round(float(np.mean(ndcgs)), 4),
               "recall@5": round(float(np.mean(r5)), 4),
               "hit@1": round(float(np.mean(r1)), 4), "n_queries": len(ndcgs)}
        report.setdefault("C_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_d(encoders, tok, report):
    print("\n=== GATE D: graph mechanics on the shipped artifacts ===")
    ids = tok(PASSAGES[1], add_special_tokens=True)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens")
    for enc in encoders:
        if not isinstance(enc, TfliteEncoder):
            continue
        vecs = {}
        for S in enc.lens:
            if S < n:
                continue
            a = np.full((1, S), PAD_ID, np.int32)
            m = np.zeros((1, S), np.int32)
            a[0, :n] = ids
            m[0, :n] = 1
            vecs[S] = list(enc.runners[S](input_ids=a,
                                          attention_mask=m).values())[0][0]
        base = vecs[min(vecs)]
        row = {"cross_sig_max_abs": {str(S): float(np.abs(v - base).max())
                                     for S, v in vecs.items()}}
        S = min(vecs)
        a = np.full((1, S), PAD_ID, np.int32)
        m = np.zeros((1, S), np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
        a2 = a.copy()
        a2[0, n:] = np.random.default_rng(1).integers(10, 260000, S - n)
        v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
        row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())
        m3 = m.copy()
        m3[0, n - 5:] = 0
        v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
        row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())

        # Short text in the LONGEST signature: with a 64-wide sliding window,
        # every pad row at q >= n_valid + 64 has an empty allowed set. That is
        # the common case in real use (a 10-token query padded to 512 leaves
        # 438 dead rows), and it is what produced NaN before the diagonal was
        # added to the mask. Gate it explicitly, and against the short-signature
        # answer for the same text.
        Smax, short_ids = max(enc.lens), ids[:12]
        sn = len(short_ids)
        outs = {}
        for S2 in (min(enc.lens), Smax):
            if S2 < sn:
                continue
            a4 = np.full((1, S2), PAD_ID, np.int32)
            m4 = np.zeros((1, S2), np.int32)
            a4[0, :sn] = short_ids
            m4[0, :sn] = 1
            outs[S2] = list(enc.runners[S2](input_ids=a4,
                                            attention_mask=m4).values())[0][0]
        row["short_in_long_finite"] = bool(np.isfinite(outs[Smax]).all())
        row["short_in_long_max_abs"] = float(
            np.abs(outs[Smax] - outs[min(outs)]).max())
        print(f"  {'':11s} short({sn} tok) in embed_{Smax}: finite="
              f"{row['short_in_long_finite']} vs embed_{min(outs)} "
              f"max|d| {row['short_in_long_max_abs']:.3e}")
        report.setdefault("D_mechanics", {})[enc.name] = row
        print(f"  {enc.name:11s} cross-sig {row['cross_sig_max_abs']}")
        print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
              f"(must be 0) | mask-live {row['mask_live_max_abs']:.3e} (must be >0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,B,C,D")
    ap.add_argument("--sts-limit", type=int, default=100)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--prefix-ab", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    encoders = []
    if not args.no_torch:
        print("loading torch reference ...")
        encoders.append(TorchEncoder(tok))
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"embed_{v}.tflite")
        if os.path.exists(p):
            print(f"loading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
            encoders.append(TfliteEncoder(p, v))
        else:
            print(f"(missing {p})")

    report = {"variants": [e.name for e in encoders]}
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, report)
    if "D" in gates:
        gate_d(encoders, tok, report)
    if "B" in gates:
        gate_b(encoders, tok, report, args.sts_limit, args.prefix_ab)
    if "C" in gates:
        gate_c(encoders, tok, report, args.docs)

    out = os.path.join(args.out_dir, "verify_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
