#!/usr/bin/env python3
"""Score the FLEURS rows (pure python, any venv): funasr oracle (oracle_fleurs.json) and the runtime row
(mac_gate_<tag>.json, --mode fleurs) against the FLEURS references of the fixtures ($FUNASR_FLEURS_FIXTURES), plus the optional
text agreement of our oracle with a second, independent oracle_transcripts.json ($FUNASR_SHARED_ORACLE, if set).

  en_us        WER = word edit distance / reference words, words = common.norm_text (upper case, [A-Z0-9' ] only),
               reference = raw_transcription (the yardstick of the LibriSpeech rows).
  cmn_hans_cn, ja_jp
               CER = character edit distance / reference characters, both strings through
               NFKC -> lower() -> drop every character whose Unicode category is P* (punctuation) or Z* (separator)
               and every whitespace character; reference = raw_transcription.
Corpus numbers are sums over clips (sum of distances / sum of reference lengths). Writes fleurs_scores.json.
  python score_fleurs.py --runtime mac_gate_r8_main_fleurs_cpu_cpu.json
"""
import argparse
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.environ.get("FUNASR_FLEURS_FIXTURES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fleurs"))
SHARED = os.environ.get("FUNASR_SHARED_ORACLE", "")
CER_FORMULA = ("CER = sum(levenshtein(norm(ref), norm(hyp))) / sum(len(norm(ref))); norm(s) = NFKC(s).lower() with every "
               "char of Unicode category P* or Z* and every whitespace char removed; ref = FLEURS raw_transcription")
WER_FORMULA = ("WER = sum(levenshtein_words(norm_text(ref), norm_text(hyp))) / sum(len(norm_text(ref))); norm_text = "
               "upper(), '-' -> ' ', [^A-Z0-9' ] -> ' ', split; ref = FLEURS raw_transcription")


def norm_text(s):
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]+", " ", s).split()


def norm_chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return [ch for ch in s if not (unicodedata.category(ch)[0] in "PZ" or ch.isspace())]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def score(config, ref, hyp):
    if config == "en_us":
        r, h = norm_text(ref), norm_text(hyp)
    else:
        r, h = norm_chars(ref), norm_chars(hyp)
    return lev(r, h), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="")
    args = ap.parse_args()
    meta = {c["name"]: c for c in json.load(open(os.path.join(FIX, "meta.json")))["clips"] if c["path"].startswith("fleurs/")}
    oracle = {r["name"]: r for r in json.load(open(os.path.join(HERE, "oracle_fleurs.json")))["rows"]}
    runtime = {r["id"]: r for r in json.load(open(os.path.join(HERE, args.runtime)))["rows"]} if args.runtime else {}
    shared = json.load(open(SHARED))["transcripts"] if SHARED else {}
    rows, agg = [], {}
    for name, o in oracle.items():
        c = meta[name]
        cfg = c["source"]["config"]
        ref = c["reference_text"]["raw_transcription"]
        eo, n = score(cfg, ref, o["text"])
        row = {"name": name, "config": cfg, "ref": ref, "oracle_text": o["text"], "oracle_err": eo, "ref_len": n,
               "second_oracle_text": shared.get(name), "second_oracle_equal": shared.get(name) == o["text"]}
        a = agg.setdefault(cfg, {"n": 0, "ref_len": 0, "oracle_err": 0, "runtime_err": 0, "runtime_n": 0,
                                 "runtime_equal_oracle": 0, "second_oracle_equal": 0, "runtime_wall_s": 0.0, "audio_s": 0.0})
        a["n"] += 1
        a["ref_len"] += n
        a["oracle_err"] += eo
        a["second_oracle_equal"] += int(row["second_oracle_equal"])
        if name in runtime:
            rt = runtime[name]
            er, _ = score(cfg, ref, rt["text"])
            row.update({"runtime_text": rt["text"], "runtime_err": er, "runtime_equal_oracle": rt["text"] == o["text"],
                        "runtime_wall_s": rt["wall_s"]})
            a["runtime_err"] += er
            a["runtime_n"] += 1
            a["runtime_equal_oracle"] += int(row["runtime_equal_oracle"])
            a["runtime_wall_s"] += rt["wall_s"]
            a["audio_s"] += rt["audio_s"]
        rows.append(row)
    for cfg, a in agg.items():
        a["metric"] = "WER" if cfg == "en_us" else "CER"
        a["oracle_rate"] = a["oracle_err"] / a["ref_len"]
        if a["runtime_n"]:
            a["runtime_rate"] = a["runtime_err"] / a["ref_len"]
            a["runtime_rtf"] = a["runtime_wall_s"] / a["audio_s"]
    doc = {"formulas": {"en_us": WER_FORMULA, "cmn_hans_cn": CER_FORMULA, "ja_jp": CER_FORMULA},
           "fixtures": os.path.join(FIX, "meta.json"), "shared_oracle": SHARED, "runtime_json": args.runtime or None,
           "summary": agg, "rows": rows}
    with open(os.path.join(HERE, "fleurs_scores.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    for cfg, a in agg.items():
        print(cfg, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()}))


if __name__ == "__main__":
    main()
