"""Round 3 B, funasr oracle side: long_en_p1.wav and long_en_p2.wav (make_long_split.py: long_en cut at 483,840
samples = one runtime window) transcribed separately (one generate() each, same setup as oracle_funasr.py: fp32 CPU,
dither 0, greedy, itn true, no hotwords) and joined with a space; WER against the three references joined (143 words).
One discarded warm-up call first (oracle_extra.py pattern). Run with ~/venvs/funasr_oracle/bin/python.
Writes oracle_long_split.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import oracle_funasr as O  # noqa: E402
from oracle_extra import gen  # noqa: E402

PARTS = ["en_clip04", "en_clip09", "en_clip02"]


def main():
    meta = {r["id"]: r for r in C.load_meta()}
    m, facts = O.build()
    cap = O.instrument(m)
    warm = gen(m, cap, os.path.join(C.WORK, meta["example_zh"]["file"]), [])
    print("warmup", warm["text"], warm["embeds_dtype_at_prepare"], flush=True)
    doc = {"facts": {k: facts[k] for k in ["funasr_version", "torch_version", "frontend_dither", "ncpu", "torch_threads"]},
           "warmup": {"id": "example_zh", "text": warm["text"], "embeds_dtype_at_prepare": warm["embeds_dtype_at_prepare"]},
           "parts": []}
    for p in ["long_en_p1", "long_en_p2"]:
        r = gen(m, cap, os.path.join(C.OUT, "fixtures", f"{p}.wav"), [])
        r["part"] = p
        doc["parts"].append(r)
        print(p, r["L"], r["fake_token_len"], r["wall_s"], "|", r["text"], flush=True)
    ref = " ".join(meta[p]["text"] for p in PARTS)
    joined = " ".join(r["text"].strip() for r in doc["parts"])
    e, n = C.wer_counts(C.norm_text(ref), C.norm_text(joined))
    doc.update({"reference_concat": ref, "joined_text": joined, "wer": [e, n]})
    with open(os.path.join(C.WORK, "oracle_long_split.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({"wer": doc["wer"], "joined": joined}, ensure_ascii=False, indent=1))
    print("ORACLE_LONG_SPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
