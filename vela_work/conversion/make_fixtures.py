"""Step 1: the fixture texts of the Vela-1.0-Encoder-307M-Embedding conversion (fixtures/texts.json).

    python3   # transformers 4.57.6 + sentence-transformers 5.1.2 scripts/make_fixtures.py

Sources (downloaded once into fixtures/src/, sha256 in fixtures/src/SHA256SUMS, revisions in fixtures/src/REVISIONS):
  PAWS-X test split (google-research-datasets/paws-x @ 4cd8187c; licence text in pawsx_README.md: "The dataset may be
  freely used for any purpose, although acknowledgement of Google LLC as the data source would be appreciated"):
  en = the first 300 test pairs, es / ja / zh = the first 100 each (the card's languages minus ar, which PAWS-X lacks);
  every pair carries its paraphrase label (1 = paraphrase).
  Long rows (the L256 / L512 windows): the next en test pairs concatenated — sentence1s of k consecutive pairs joined
  with a space until the token count passes a target (and the matching sentence2s as the paired text), 40 rows per
  class: ~100 tok (L128), ~200 (L256), ~400 (L512); 10 rows pushed to the 512 window edge (truncated at 510 content tokens).
Every text is STRIPPED first (sentence-transformers' Transformer.tokenize strips each text before the tokenizer; a
trailing space would otherwise become a lone "▁" token 235248 before <eos>: 12 of the STS-B rows end in a space), then
tokenized with the repo's tokenizer.json (tokenizers; the post-processor adds <bos> … <eos>) and the row length (with the
specials) is recorded; the groups are keyed so every later gate can select by window. An app that feeds this graph must
strip the same way (memory metaspace-tokenizer-strip-trap).
"""
import json
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

W = Path(__file__).resolve().parents[1]
SRC = W / "fixtures" / "src"
HF = W / "hf"
N = {"en": 300, "es": 100, "ja": 100, "zh": 100}
LONG = [("long128", 100, 40), ("long256", 200, 40), ("long512", 400, 40), ("edge512", 10_000, 10)]
MAX_CONTENT = 510   # 512 - <bos> - <eos>


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    revs = dict(line.split() for line in (SRC / "REVISIONS").read_text().split("\n") if line.strip())
    sums = dict(reversed(line.split()) for line in (SRC / "SHA256SUMS").read_text().split("\n") if line.strip())
    rows, pairs = [], []

    def add(key, text, group, lang, src, idx, side, label=None):
        text = text.strip()
        enc = tok.encode(text)
        assert enc.ids[0] == 2 and enc.ids[-1] == 1, enc.ids[:3]
        rows.append(dict(key=key, text=text, group=group, lang=lang, source=src, row=idx, side=side, label=label,
                         ids=enc.ids, row_len=len(enc.ids), text_sha256=sha(text)))
        return enc.ids

    tables = {}
    for lang, n in N.items():
        t = pq.read_table(SRC / f"pawsx_{lang}_test.parquet").to_pylist()
        tables[lang] = t
        for i in range(n):
            r = t[i]
            k = f"pawsx_{lang}_{i:04d}"
            add(k + "_a", r["sentence1"], "pair", lang, f"pawsx_{lang}_test.parquet", i, "a", int(r["label"]))
            add(k + "_b", r["sentence2"], "pair", lang, f"pawsx_{lang}_test.parquet", i, "b", int(r["label"]))
            pairs.append(dict(key=k, lang=lang, label=int(r["label"]), a=k + "_a", b=k + "_b", group="pair"))
    # long rows from the en pairs after the first 300
    t = tables["en"]
    i = N["en"]
    for group, target, count in LONG:
        made = 0
        while made < count:
            a_parts, b_parts, used = [], [], []
            while True:
                r = t[i]; i += 1
                a_parts.append(r["sentence1"]); b_parts.append(r["sentence2"]); used.append(i - 1)
                a_text = " ".join(a_parts)
                n_tok = len(tok.encode(a_text).ids)
                if n_tok >= target + 2 or n_tok > MAX_CONTENT - 40:
                    break
            a_text, b_text = " ".join(a_parts), " ".join(b_parts)
            if group == "edge512":
                # cut both texts at 510 content tokens by decoding the first 510 content ids back to text
                for name in ("a_text", "b_text"):
                    ids = tok.encode(locals()[name]).ids[1:-1][:MAX_CONTENT]
                    txt = tok.decode(ids)
                    assert len(tok.encode(txt).ids) <= 512, len(tok.encode(txt).ids)
                    if name == "a_text":
                        a_text = txt
                    else:
                        b_text = txt
            k = f"pawsx_en_{group}_{made:03d}"
            add(k + "_a", a_text, group, "en", "pawsx_en_test.parquet", used, "a")
            add(k + "_b", b_text, group, "en", "pawsx_en_test.parquet", used, "b")
            pairs.append(dict(key=k, lang="en", label=None, a=k + "_a", b=k + "_b", group=group))
            made += 1
    # STS Benchmark v2 test (mteb/STSBenchmarkv2 @ 93b628c3 = the card's dataset_revision; licence unknown -> these rows
    # leave this repo only BY REFERENCE: dataset id + revision + row index + sha256, never the text)
    sts_file = SRC / "stsbv2_test.parquet"
    if sts_file.exists():
        t = pq.read_table(sts_file).to_pylist()
        for i, r in enumerate(t):
            k = f"stsb_{i:04d}"
            add(k + "_a", r["sentence1"], "stsb", "en", "stsbv2_test.parquet", i, "a")
            add(k + "_b", r["sentence2"], "stsb", "en", "stsbv2_test.parquet", i, "b")
            pairs.append(dict(key=k, lang="en", label=None, score=float(r["score"]), a=k + "_a", b=k + "_b", group="stsb"))
    lens = sorted(r["row_len"] for r in rows)
    by_group = {}
    for r in rows:
        g = by_group.setdefault(r["group"], dict(rows=0, min_len=10**9, max_len=0))
        g["rows"] += 1; g["min_len"] = min(g["min_len"], r["row_len"]); g["max_len"] = max(g["max_len"], r["row_len"])
    fits = {L: sum(1 for r in rows if r["row_len"] <= L) for L in (64, 128, 256, 512)}
    doc = dict(model="vllm-sr/Vela-1.0-Encoder-307M-Embedding", tokenizer_sha256=hashlib.sha256((HF / "tokenizer.json").read_bytes()).hexdigest(),
               sources=dict(pawsx=dict(repo="google-research-datasets/paws-x", revision=revs["pawsx_rev"], split="test",
                                       files={k: v for k, v in sums.items() if k.startswith("pawsx")},
                                       licence="freely used for any purpose; acknowledgement of Google LLC appreciated (pawsx_README.md, Licensing Information)")),
               stsb=dict(repo="mteb/STSBenchmarkv2", revision=revs.get("stsbv2_rev"), split="test", file=sums.get("stsbv2_test.parquet"),
                         licence="unknown (dataset card) -> by reference only outside this repo"),
               selection=dict(pairs=N, long=[dict(group=g, target_tokens=t, rows=c) for g, t, c in LONG]),
               counts=dict(rows=len(rows), pairs=len(pairs), by_group=by_group, fits=fits,
                           row_len=dict(min=lens[0], median=lens[len(lens) // 2], p90=lens[int(len(lens) * 0.9)], max=lens[-1])),
               rows=rows, pairs=pairs)
    (W / "fixtures" / "texts.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps(doc["counts"], indent=1))


if __name__ == "__main__":
    main()
