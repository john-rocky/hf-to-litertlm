#!/usr/bin/env python3
"""Corpus WER for r1, two formulas on the same normalized text.

J  = plain corpus WER: the leaderboard EnglishTextNormalizer on both
     sides, utterances with an empty normalized reference dropped, corpus WER by jiwer (S / D / I).
L  = the Open ASR Leaderboard's own scoring at huggingface/open_asr_leaderboard b2e6f049 (2026-09-23; the
     normalizer files are unchanged since 8c1e176e, 2026-09-14) = eval_utils.score_results: English ->
     EnglishTextNormalizer, other languages -> data_utils.ml_normalizer(text, lang=lang)
     (MultilingualNormalizer(remove_diacritics=False): BasicMultilingualTextNormalizer + num2words digits);
     multilingual sets -> normalize_compound_pairs; then kaldialign batch_error_rate(merge_compounds=True).
For a non-English language J uses ml_normalizer too (the leaderboard's multilingual normalizer).

wer.py --manifest <manifest.jsonl> --pred <pred.jsonl> [--lang en] [--ref-field text] [--mode M]
       [--subset <manifest-S32.jsonl>] [--details <out.jsonl>] [--json <out.json>]
--mode keeps pred lines whose "mode" equals M (s32_modes.jsonl); --subset scores only the ids of that manifest.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jiwer  # noqa: E402
import num2words  # noqa: E402
from kaldialign import batch_error_rate  # noqa: E402
from normalizer.eval_utils import normalize_compound_pairs  # noqa: E402
from normalizer.normalizer import BasicMultilingualTextNormalizer, EnglishTextNormalizer  # noqa: E402

# --- copied verbatim from open_asr_leaderboard normalizer/data_utils.py @ b2e6f049 (its module imports `datasets`)
FILLER_WORDS = {}


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    def __init__(self, remove_diacritics: bool = True):
        super().__init__(remove_diacritics)
        self._filler_patterns = {}
        base_normalize = super().__call__
        for lang, words in FILLER_WORDS.items():
            normalized_words = {base_normalize(w) for w in words}
            normalized_words.discard("")
            self._filler_patterns[lang] = re.compile(
                r"(?<!\S)(?:"
                + "|".join(re.escape(w) for w in sorted(normalized_words, key=len, reverse=True))
                + r")(?!\S)"
            )

    def _remove_fillers(self, text, lang):
        pattern = self._filler_patterns.get(lang)
        if pattern is None:
            return text
        text = pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_numbers(self, text, lang):
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        s = super().__call__(s)
        if lang is not None:
            s = self._remove_fillers(s, lang)
            s = self._normalize_numbers(s, lang)
        return s
# --- end of the copy


EN = EnglishTextNormalizer()
ML = MultilingualNormalizer(remove_diacritics=False)


def norm(text, lang):
    return EN(text) if lang == "en" else ML(text, lang=lang)


def score(refs_raw, hyps_raw, lang, ids=None):
    ids = ids or list(range(len(refs_raw)))
    R, H, keep, dropped = [], [], [], 0
    for i, r, h in zip(ids, refs_raw, hyps_raw):
        nr, nh = norm(r or "", lang), norm(h or "", lang)
        if not nr.strip():
            dropped += 1
            continue
        R.append(nr)
        H.append(nh)
        keep.append(i)
    o = jiwer.process_words(R, H)
    J = {"wer": round(100 * o.wer, 3), "S": o.substitutions, "D": o.deletions, "I": o.insertions,
         "ref_words": sum(len(r.split()) for r in R)}
    R2, H2 = (normalize_compound_pairs(R, H) if lang != "en" else (R, H))
    k = batch_error_rate([tuple(r.split()) for r in R2], [tuple(h.split()) for h in H2], merge_compounds=True)
    L = {"wer": round(100 * k["err_rate"], 3), "S": k["sub"], "D": k["del"], "I": k["ins"]}
    per = []
    for i, r, h in zip(keep, R, H):
        m = jiwer.process_words(r, h)
        per.append({"id": i, "ref": r, "hyp": h, "errors": m.substitutions + m.deletions + m.insertions,
                    "ref_words": len(r.split())})
    return {"utterances": len(R), "dropped_empty_ref": dropped, "J": J, "L": L, "empty_hyp": sum(1 for h in H if not h)}, per


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--ref-field", default="text")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--details", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    man = load(a.manifest)
    preds = [p for p in load(a.pred) if "id" in p and (a.mode is None or p.get("mode") == a.mode)]
    pm = {p["id"]: p for p in preds}
    if a.subset:
        keep = {r["id"] for r in load(a.subset)}
        man = [r for r in man if r["id"] in keep]
    missing = [r["id"] for r in man if r["id"] not in pm]
    ids = [r["id"] for r in man if r["id"] in pm]
    refs = [r[a.ref_field] for r in man if r["id"] in pm]
    hyps = [pm[i]["text"] for i in ids]
    res, per = score(refs, hyps, a.lang, ids)
    res.update({"manifest": a.manifest, "pred": a.pred, "lang": a.lang, "ref_field": a.ref_field, "mode": a.mode,
                "subset": a.subset, "missing_preds": len(missing)})
    print(json.dumps(res))
    if a.details:
        with open(a.details, "w") as f:
            for x in per:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
