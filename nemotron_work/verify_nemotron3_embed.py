#!/usr/bin/env python3
"""Quality gates for the converted Nemotron-3-Embed-1B LiteRT artifacts.

    python verify_nemotron3_embed.py <out_dir> [--gates A,B,C,D]

Gates
  A  card oracle   — reproduce the model card's documented similarity matrix
                     (external truth, not our own harness)
  B  STS17         — multilingual + cross-lingual Spearman, 11 language pairs.
                     Graded metric: cannot saturate to a flattering floor.
  C  NanoSciFact   — retrieval nDCG@10 / recall@5 with the query:/passage:
                     prefixes, i.e. the task the model is actually for.
  D  mechanics     — cross-signature agreement, pad-content invariance and
                     mask liveness measured on the shipped artifacts.

Every gate runs on each variant (torch fp32 reference, fp32/wi8fc/fp16 tflite)
so the numbers are comparable; a variant is judged against the reference, never
against an absolute threshold.
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

MODEL_DIR = os.environ.get("NEMOTRON_EMBED_MODEL",
                           "nvidia/Nemotron-3-Embed-1B-BF16")
PAD_ID = 11
SEQ_LENS = (64, 128, 256, 512)

QUERIES = [
    "Write a Python function that counts the frequency of each element in a list of lists.",
    "Write a function that orders a dictionary with tuple keys by the product of each key's tuple values.",
    "What symptoms and common triggers help distinguish eczema from other inflammatory skin conditions?",
    "How can someone reduce exposure to pollen during allergy season?",
]
DOCUMENTS = [
    "def frequency_lists(list1):\n    flattened = [item for sublist in list1 for item in sublist]\n    counts = {}\n    for item in flattened:\n        if item in counts:\n            counts[item] += 1\n        else:\n            counts[item] = 1\n    return counts",
    "def sort_dict_item(test_dict):\n    return {key: test_dict[key] for key in sorted(test_dict.keys(), key=lambda ele: ele[0] * ele[1])}",
    "Eczema commonly causes itchy, dry, inflamed patches of skin. The affected areas may look red, scaly, cracked, or darker than the surrounding skin depending on skin tone. Symptoms can flare after exposure to irritants, allergens, stress, or changes in weather.",
    "People with pollen allergy can reduce exposure by staying indoors on dry, windy days, avoiding early-morning outdoor activity, and going outside after rain when pollen levels are lower. They should check pollen forecasts, close windows and doors when counts are high, and consider starting allergy medication before symptoms begin if high pollen is expected. After being outside, showering, changing clothes, avoiding outdoor laundry drying, and wearing a face mask for yard work can help limit pollen contact.",
]
CARD_TF = np.array([
    [0.8069, 0.0252, 0.0001, -0.0312],
    [0.0446, 0.6466, -0.0514, 0.0385],
    [-0.0098, -0.0410, 0.6450, 0.0998],
    [-0.0215, 0.0212, 0.1197, 0.7679],
])
STS17_PAIRS = ["en-en", "ar-ar", "es-es", "ko-ko", "nl-en", "en-ar", "en-de",
               "en-tr", "es-en", "fr-en", "it-en"]


# --------------------------------------------------------------------------
# encoders: one interface, four backends
# --------------------------------------------------------------------------
class TorchEncoder:
    name = "torch_fp32"

    def __init__(self, tok):
        from transformers import AutoModel
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from convert_nemotron3_embed import Embedder

        m = AutoModel.from_pretrained(
            MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        assert m.config.is_causal is False
        self.mod = Embedder(m).eval()
        self.tok = tok

    def encode_ids(self, ids):
        S = len(ids)
        t = torch.tensor([ids], dtype=torch.int32)
        m = torch.ones(1, S, dtype=torch.int32)
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

    def pick(self, n):
        for S in self.lens:
            if n <= S:
                return S
        return self.lens[-1]

    def encode_ids(self, ids):
        """Right-pad to the smallest signature that fits; truncate past the max."""
        S = self.pick(len(ids))
        ids = ids[:S]
        n = len(ids)
        a = np.full((1, S), PAD_ID, dtype=np.int32)
        m = np.zeros((1, S), dtype=np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        r = self.runners[S](input_ids=a, attention_mask=m)
        return list(r.values())[0][0]


def encode_texts(enc, tok, texts, prefix="", max_tokens=512, log_every=0):
    out = np.empty((len(texts), 2048), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(prefix + t, add_special_tokens=True)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s "
                  f"({el/(i+1)*1000:.0f} ms/text)", flush=True)
    return out


# --------------------------------------------------------------------------
def spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        x_sorted = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x_sorted[j + 1] == x_sorted[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def ndcg_at_k(ranked_rel, ideal_rel, k=10):
    def dcg(rels):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = dcg(sorted(ideal_rel, reverse=True))
    return dcg(ranked_rel) / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
def gate_a(encoders, tok, report):
    print("\n=== GATE A: model-card oracle (external truth) ===")
    ref = None
    for enc in encoders:
        q = encode_texts(enc, tok, QUERIES, "query: ")
        d = encode_texts(enc, tok, DOCUMENTS, "passage: ")
        s = q @ d.T
        vs_card = float(np.abs(s - CARD_TF).max())
        argmax_ok = bool((s.argmax(1) == np.arange(4)).all())
        row = {"max_abs_vs_card": vs_card, "diagonal_is_argmax": argmax_ok,
               "diag": [round(float(x), 4) for x in np.diag(s)]}
        if ref is None:
            ref = s
        else:
            row["max_abs_vs_torch"] = float(np.abs(s - ref).max())
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} diag {row['diag']}  max|d| vs card {vs_card:.4f}"
              f"  argmax-diagonal {argmax_ok}"
              + (f"  max|d| vs torch {row['max_abs_vs_torch']:.4f}"
                 if "max_abs_vs_torch" in row else ""))
    print("  card diag:", [round(float(x), 4) for x in np.diag(CARD_TF)])


def load_sts17(limit):
    from huggingface_hub import hf_hub_download

    data = {}
    for pair in STS17_PAIRS:
        try:
            p = hf_hub_download("mteb/sts17-crosslingual-sts",
                                f"test/{pair}.jsonl.gz", repo_type="dataset")
        except Exception as e:
            print(f"  (skip {pair}: {type(e).__name__})")
            continue
        rows = []
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["sentence1"], r["sentence2"], float(r["score"])))
        data[pair] = rows[:limit] if limit else rows
    return data


def gate_b(encoders, tok, report, limit):
    print(f"\n=== GATE B: STS17 multilingual/cross-lingual Spearman "
          f"(<= {limit} pairs/lang) ===")
    data = load_sts17(limit)
    if not data:
        print("  no data — skipped")
        return
    n = sum(len(v) for v in data.values())
    print(f"  {len(data)} language pairs, {n} sentence pairs, {2*n} encodes/variant")
    for enc in encoders:
        per_lang, all_p, all_g = {}, [], []
        for pair, rows in data.items():
            # `query: ` on BOTH sides. This model only ever saw a prefix, so raw
            # text is out of distribution: measured on 100 pairs of the int8
            # artifact, dropping the prefix costs en-en 0.865->0.636 and
            # collapses cross-lingual outright (en-ar 0.724->-0.029,
            # es-en 0.803->0.064). Note this contradicts the vendor's own
            # config_sentence_transformers.json `default_prompt_name: null`.
            s1 = encode_texts(enc, tok, [r[0] for r in rows], "query: ")
            s2 = encode_texts(enc, tok, [r[1] for r in rows], "query: ")
            pred = (s1 * s2).sum(1)
            gold = np.array([r[2] for r in rows])
            per_lang[pair] = round(spearman(pred, gold), 4)
            all_p.append(pred); all_g.append(gold)
        overall = spearman(np.concatenate(all_p), np.concatenate(all_g))
        report.setdefault("B_sts17", {})[enc.name] = {
            "per_lang": per_lang, "pooled": round(overall, 4),
            "mean": round(float(np.mean(list(per_lang.values()))), 4)}
        print(f"  {enc.name:11s} mean {np.mean(list(per_lang.values())):.4f} "
              f"pooled {overall:.4f}  {per_lang}")


def load_nanoscifact(n_docs):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    R = "zeta-alpha-ai/NanoSciFact"
    corpus = pq.read_table(hf_hub_download(R, "corpus/train-00000-of-00001.parquet",
                                           repo_type="dataset")).to_pydict()
    queries = pq.read_table(hf_hub_download(R, "queries/train-00000-of-00001.parquet",
                                            repo_type="dataset")).to_pydict()
    qrels = pq.read_table(hf_hub_download(R, "qrels/train-00000-of-00001.parquet",
                                          repo_type="dataset")).to_pydict()
    gold = {}
    for q, c in zip(qrels["query-id"], qrels["corpus-id"]):
        gold.setdefault(str(q), set()).add(str(c))
    ids = [str(i) for i in corpus["_id"]]
    texts = corpus["text"]
    keep_ids = sorted({c for s in gold.values() for c in s})
    rng = np.random.default_rng(0)
    rest = [i for i in ids if i not in set(keep_ids)]
    extra = list(rng.choice(rest, size=max(0, n_docs - len(keep_ids)),
                            replace=False)) if n_docs else rest
    sel = list(keep_ids) + [str(x) for x in extra]
    by_id = dict(zip(ids, texts))
    return ([(i, by_id[i]) for i in sel],
            list(zip([str(i) for i in queries["_id"]], queries["text"])), gold)


def gate_c(encoders, tok, report, n_docs):
    print(f"\n=== GATE C: NanoSciFact retrieval (corpus subsampled to {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    print(f"  {len(queries)} queries, {len(docs)} docs "
          f"({sum(len(v) for v in gold.values())} qrels)")
    for enc in encoders:
        D = encode_texts(enc, tok, [t for _, t in docs], "passage: ",
                         log_every=200)
        Q = encode_texts(enc, tok, [t for _, t in queries], "query: ")
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
    text = "passage: " + DOCUMENTS[2]
    ids = tok(text, add_special_tokens=True)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens")
    for enc in encoders:
        row = {}
        if isinstance(enc, TfliteEncoder):
            # cross-signature: the same text padded into each signature that fits
            vecs = {}
            for S in enc.lens:
                if S < n:
                    continue
                a = np.full((1, S), PAD_ID, dtype=np.int32)
                m = np.zeros((1, S), dtype=np.int32)
                a[0, :n] = ids; m[0, :n] = 1
                vecs[S] = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            base = vecs[min(vecs)]
            row["cross_sig_max_abs"] = {
                str(S): float(np.abs(v - base).max()) for S, v in vecs.items()}
            row["cross_sig_cos"] = {
                str(S): round(float(v @ base), 8) for S, v in vecs.items()}

            # pad-content invariance: scribble over the pad region
            S = min(vecs)
            a = np.full((1, S), PAD_ID, dtype=np.int32)
            m = np.zeros((1, S), dtype=np.int32)
            a[0, :n] = ids; m[0, :n] = 1
            v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            a2 = a.copy()
            a2[0, n:] = np.random.default_rng(1).integers(10, 120000, S - n)
            v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
            row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())

            # mask liveness: dropping real tokens must change the answer
            m3 = m.copy(); m3[0, n - 5:] = 0
            v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
            row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())

            print(f"  {enc.name:11s} cross-sig max|d| {row['cross_sig_max_abs']}")
            print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
                  f"(must be 0) | mask-live {row['mask_live_max_abs']:.3e} (must be >0)")
        report.setdefault("D_mechanics", {})[enc.name] = row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,B,C,D")
    ap.add_argument("--sts-limit", type=int, default=150)
    ap.add_argument("--docs", type=int, default=800)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
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
        gate_b(encoders, tok, report, args.sts_limit)
    if "C" in gates:
        gate_c(encoders, tok, report, args.docs)

    out = os.path.join(args.out_dir, "verify_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
