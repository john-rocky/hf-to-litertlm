#!/usr/bin/env python3
"""Quality gates for the converted voyage-4-nano LiteRT artifacts.

    python verify_voyage_embed.py <out_dir> [--gates A,D,B,C,E] [--prefix-ab]

Gates
  A  card example  — the README's own Red Planet example (1 query x 4 docs):
                     the Mars document must win; torch-vs-variant agreement is
                     the anchor (the card documents no exact scores).
  B  STS          — JSTS (JGLUE v1.3 valid) + STS17 subset, Spearman. NO prompt
                     on either side: config_sentence_transformers.json defines
                     only query/document prompts with default_prompt_name null,
                     so bare encode() is the vendor path for symmetric tasks.
  C  JSQuAD       — Japanese retrieval with the vendor prompts on BOTH sides
                     (query: "Represent the query for retrieving supporting
                     documents: ", document: "Represent the document for
                     retrieval: "). Also scored at MRL-truncated 256 dims
                     (slice + re-normalize — the README's documented use).
  D  mechanics     — cross-signature agreement, pad-content invariance, mask
                     liveness. For a MEAN pool, pad-invariance also proves the
                     divisor is sum(mask), not S.
  E  cross-variant — docs encoded by torch, queries by each variant. THE
                     production gate here: the model's selling point is the
                     shared embedding space with the voyage-4 cloud series
                     (device nano querying a cloud-built index), so quantized
                     queries must stay compatible with a full-precision index.
                     Also scored at 256 dims.

--prefix-ab additionally runs gate C queries WITHOUT the query prompt
(documents keep theirs) to document the prompt contract.

JGLUE fixtures cached in ./fixtures/ (CC BY-SA 4.0, eval-only).
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

MODEL_DIR = os.environ.get("VOYAGE_EMBED_MODEL", "voyageai/voyage-4-nano")
PAD_ID = 151643
DIM = 2048
MRL_DIM = 256
SEQ_LENS = (64, 128, 256, 512)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
JGLUE_RAW = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets"

QUERY_PREFIX = "Represent the query for retrieving supporting documents: "
DOC_PREFIX = "Represent the document for retrieval: "

CARD_QUERY = "Which planet is known as the Red Planet?"
CARD_DOCS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
CARD_GOLD = 1  # Mars
STS17_PAIRS = ["en-en", "es-en", "ko-ko", "en-ar"]


class TorchEncoder:
    name = "torch_fp32"

    def __init__(self, tok):
        from convert_voyage_embed import Embedder, load_model

        self.mod = Embedder(load_model(MODEL_DIR)).eval()
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
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(prefix + t)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s "
                  f"({el/(i+1)*1000:.0f} ms/text)", flush=True)
    return out


def mrl_truncate(vecs, dim=MRL_DIM):
    v = vecs[:, :dim].copy()
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
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


def fetch(name, rel):
    os.makedirs(FIXTURES, exist_ok=True)
    p = os.path.join(FIXTURES, name)
    if not os.path.exists(p):
        url = f"{JGLUE_RAW}/{rel}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, p)
    return p


def load_jsts(limit):
    rows = []
    with open(fetch("jsts_valid_v1.3.json", "jsts-v1.3/valid-v1.3.json")) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["sentence1"], r["sentence2"], float(r["label"])))
    return rows[:limit] if limit else rows


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


def load_jsquad(n_docs, n_queries):
    with open(fetch("jsquad_valid_v1.3.json", "jsquad-v1.3/valid-v1.3.json")) as f:
        data = json.load(f)["data"]
    ctx_id = {}
    docs = []
    queries = []
    for art in data:
        for para in art["paragraphs"]:
            c = para["context"]
            if c not in ctx_id:
                ctx_id[c] = len(docs)
                docs.append(c)
            for qa in para["qas"]:
                queries.append((qa["question"], ctx_id[c]))
    rng = np.random.default_rng(0)
    queries = [queries[i] for i in rng.choice(len(queries), n_queries, replace=False)]
    keep = sorted({g for _, g in queries})
    rest = [i for i in range(len(docs)) if i not in set(keep)]
    extra = list(rng.choice(rest, max(0, n_docs - len(keep)), replace=False))
    sel = keep + [int(x) for x in extra]
    remap = {old: new for new, old in enumerate(sel)}
    return ([docs[i] for i in sel],
            [(q, remap[g]) for q, g in queries])


def retrieval_scores(sims, queries):
    ndcgs, r5, r1 = [], [], []
    for qi, (_, gold) in enumerate(queries):
        order = np.argsort(-sims[qi])
        rel = [1.0 if j == gold else 0.0 for j in order[:10]]
        ndcgs.append(ndcg_at_k(rel, [1.0], 10))
        r5.append(float(any(rel[:5])))
        r1.append(float(rel[0]))
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4),
            "recall@5": round(float(np.mean(r5)), 4),
            "hit@1": round(float(np.mean(r1)), 4)}


def gate_a(encoders, tok, report):
    print("\n=== GATE A: README Red Planet example (vendor prompts) ===")
    ref = None
    for enc in encoders:
        q = encode_texts(enc, tok, [CARD_QUERY], prefix=QUERY_PREFIX)
        d = encode_texts(enc, tok, CARD_DOCS, prefix=DOC_PREFIX)
        s = (q @ d.T) * 100
        argmax_ok = int(s[0].argmax()) == CARD_GOLD
        row = {"scores_x100": [round(float(x), 2) for x in s[0]],
               "argmax_ok": argmax_ok}
        if ref is None:
            ref = s
        else:
            row["max_abs_vs_torch_x100"] = round(float(np.abs(s - ref).max()), 3)
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} {row['scores_x100']} mars-wins {argmax_ok}"
              + (f"  max|d| vs torch {row['max_abs_vs_torch_x100']}"
                 if "max_abs_vs_torch_x100" in row else ""))


def gate_b(encoders, tok, report, limit):
    print(f"\n=== GATE B: JSTS + STS17 Spearman (no prompt — vendor default "
          f"for symmetric tasks; <= {limit} pairs/set) ===")
    jsts = load_jsts(limit)
    sts17 = load_sts17(min(limit, 100))
    print(f"  JSTS {len(jsts)} pairs; STS17 {sorted(sts17)}")
    for enc in encoders:
        s1 = encode_texts(enc, tok, [r[0] for r in jsts], log_every=200)
        s2 = encode_texts(enc, tok, [r[1] for r in jsts])
        gold = np.array([r[2] for r in jsts])
        row = {"jsts": round(spearman((s1 * s2).sum(1), gold), 4)}
        per_lang = {}
        for pair, rows in sts17.items():
            a = encode_texts(enc, tok, [r[0] for r in rows])
            b = encode_texts(enc, tok, [r[1] for r in rows])
            per_lang[pair] = round(
                spearman((a * b).sum(1), np.array([r[2] for r in rows])), 4)
        row["sts17"] = per_lang
        report.setdefault("B_sts", {})[enc.name] = row
        print(f"  {enc.name:11s} JSTS {row['jsts']}  STS17 {per_lang}")


def gate_c(encoders, tok, report, n_docs, n_queries=150, prefix_ab=False):
    print(f"\n=== GATE C: JSQuAD retrieval (corpus {n_docs} docs, "
          f"{n_queries} questions; vendor prompts both sides) ===")
    docs, queries = load_jsquad(n_docs, n_queries)
    print(f"  {len(queries)} queries, {len(docs)} unique paragraphs")
    for enc in encoders:
        D = encode_texts(enc, tok, docs, prefix=DOC_PREFIX, log_every=200)
        Q = encode_texts(enc, tok, [q for q, _ in queries], prefix=QUERY_PREFIX)
        row = retrieval_scores(Q @ D.T, queries)
        row["mrl256"] = retrieval_scores(mrl_truncate(Q) @ mrl_truncate(D).T,
                                         queries)
        if prefix_ab:
            Qn = encode_texts(enc, tok, [q for q, _ in queries])
            row["no_query_prompt"] = retrieval_scores(Qn @ D.T, queries)
        report.setdefault("C_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f} "
              f"| mrl256 nDCG@10 {row['mrl256']['ndcg@10']:.4f}"
              + (f" | no-q-prompt nDCG@10 {row['no_query_prompt']['ndcg@10']:.4f}"
                 if prefix_ab else ""))


def gate_e(encoders, tok, report, n_docs, n_queries=150):
    print(f"\n=== GATE E: cross-variant retrieval (docs=torch_fp32, "
          f"queries=variant; corpus {n_docs}) — the shared-space gate ===")
    torch_enc = next((e for e in encoders if e.name == "torch_fp32"), None)
    if torch_enc is None:
        print("  needs the torch reference — skipped")
        return
    docs, queries = load_jsquad(n_docs, n_queries)
    D = encode_texts(torch_enc, tok, docs, prefix=DOC_PREFIX, log_every=200)
    for enc in encoders:
        Q = encode_texts(enc, tok, [q for q, _ in queries], prefix=QUERY_PREFIX)
        row = retrieval_scores(Q @ D.T, queries)
        row["mrl256"] = retrieval_scores(mrl_truncate(Q) @ mrl_truncate(D).T,
                                         queries)
        report.setdefault("E_cross_variant", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f} "
              f"| mrl256 nDCG@10 {row['mrl256']['ndcg@10']:.4f}")


def gate_d(encoders, tok, report):
    print("\n=== GATE D: graph mechanics on the shipped artifacts ===")
    text = DOC_PREFIX + "富士山は日本で一番高い山です。"
    ids = tok(text)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens (short by design)")
    for enc in encoders:
        row = {}
        if isinstance(enc, TfliteEncoder):
            vecs = {}
            for S in enc.lens:
                if S < n:
                    continue
                a = np.full((1, S), PAD_ID, dtype=np.int32)
                m = np.zeros((1, S), dtype=np.int32)
                a[0, :n] = ids; m[0, :n] = 1
                v = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
                assert np.isfinite(v).all(), f"{enc.name} embed_{S}: NaN/inf output"
                vecs[S] = v
            base = vecs[min(vecs)]
            row["cross_sig_max_abs"] = {
                str(S): float(np.abs(v - base).max()) for S, v in vecs.items()}

            S = max(vecs)
            a = np.full((1, S), PAD_ID, dtype=np.int32)
            m = np.zeros((1, S), dtype=np.int32)
            a[0, :n] = ids; m[0, :n] = 1
            v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            a2 = a.copy()
            a2[0, n:] = np.random.default_rng(1).integers(10, 151000, S - n)
            v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
            row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())

            # mask liveness: dropping the last real token must move a mean-pool
            # embedding (also proves the divisor tracks sum(mask)).
            m3 = m.copy(); m3[0, n - 1:] = 0
            v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
            row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())

            print(f"  {enc.name:11s} cross-sig max|d| {row['cross_sig_max_abs']}")
            print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
                  f"(must be 0) | drop-last-token {row['mask_live_max_abs']:.3e} "
                  f"(must be >0)")
        report.setdefault("D_mechanics", {})[enc.name] = row


def main():
    sys.stdout.reconfigure(line_buffering=True)  # result rows must not sit in a pipe buffer
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,D,B,C,E")
    ap.add_argument("--sts-limit", type=int, default=300)
    ap.add_argument("--docs", type=int, default=800)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--prefix-ab", action="store_true")
    ap.add_argument("--report", default="verify_report.json")
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
        gate_c(encoders, tok, report, args.docs, prefix_ab=args.prefix_ab)
    if "E" in gates:
        gate_e(encoders, tok, report, args.docs)

    out = os.path.join(args.out_dir, args.report)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
