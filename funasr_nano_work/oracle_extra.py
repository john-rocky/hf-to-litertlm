"""funasr oracle for the two extra Mac-gate rows (same setup as oracle_funasr.py: fp32 CPU, dither 0, greedy, itn true):

  long     out/fixtures/long_en.wav (make_long_fixture.py: en_clip04 + en_clip09 + en_clip02, ~60 s) in ONE generate()
           (funasr encodes the whole clip at once; the runtime cuts it into 504-frame chunks)
  hotword  en_clip04 with hotwords=["Birket Foster", "John Collier"] (+ the same clip without hotwords, re-run here)

One discarded warm-up call first (the first generate() of a process runs the adaptor rows through a bf16 embedding
tensor, round 1). Run with ~/venvs/funasr_oracle/bin/python. Writes oracle_extra.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import oracle_funasr as O  # noqa: E402

HOTWORDS = ["Birket Foster", "John Collier"]


def gen(m, cap, path, hotwords):
    cap.clear()
    t0 = time.perf_counter()
    res = m.generate(input=path, llm_kwargs=dict(O.LLM_KWARGS), itn=True, hotwords=list(hotwords), language=None,
                     max_length=512)
    wall = time.perf_counter() - t0
    r = res[0]
    L = int(cap["speech_lengths"].reshape(-1)[0])
    ftl = int(cap["fake_token_len"].reshape(-1)[0])
    src = cap["source_ids"][0].tolist()
    beg = int(cap["fbank_beg"].reshape(-1)[0])
    return {"file": os.path.relpath(path, C.WORK), "hotwords": list(hotwords), "L": L, "fake_token_len": ftl,
            "prefix_len": beg, "prefix_ids": src[:beg], "suffix_ids": src[beg + ftl:], "text": r["text"],
            "text_tn": r["text_tn"], "gen_ids": cap["gen_ids"], "n_gen": len(cap["gen_ids"]), "wall_s": round(wall, 3),
            "embeds_dtype_at_prepare": cap["embeds_dtype_at_prepare"],
            "llm_param_dtype_at_generate": cap["llm_param_dtype_at_generate"]}


def main():
    meta = {r["id"]: r for r in C.load_meta()}
    m, facts = O.build()
    cap = O.instrument(m)
    long_path = os.path.join(C.OUT, "fixtures", "long_en.wav")
    clip04 = os.path.join(C.WORK, meta["en_clip04"]["file"])
    warm = gen(m, cap, os.path.join(C.WORK, meta["example_zh"]["file"]), [])
    print("warmup", warm["text"], warm["embeds_dtype_at_prepare"], flush=True)
    doc = {"facts": {k: facts[k] for k in ["funasr_version", "torch_version", "frontend_dither", "ncpu", "torch_threads"]},
           "warmup": {"id": "example_zh", "text": warm["text"], "embeds_dtype_at_prepare": warm["embeds_dtype_at_prepare"]}}
    doc["long"] = gen(m, cap, long_path, [])
    print("long", doc["long"]["L"], doc["long"]["fake_token_len"], doc["long"]["wall_s"], "|", doc["long"]["text"], flush=True)
    doc["hotword"] = gen(m, cap, clip04, HOTWORDS)
    print("hotword", doc["hotword"]["prefix_len"], doc["hotword"]["wall_s"], "|", doc["hotword"]["text"], flush=True)
    doc["no_hotword_rerun"] = gen(m, cap, clip04, [])
    print("no hotword", doc["no_hotword_rerun"]["wall_s"], "|", doc["no_hotword_rerun"]["text"], flush=True)
    oracle = {r["id"]: r for r in json.load(open(os.path.join(C.WORK, "oracle_transcripts.json")))["rows"]}
    doc["no_hotword_rerun"]["equal_to_oracle_transcripts_json"] = doc["no_hotword_rerun"]["text"] == oracle["en_clip04"]["text"]
    parts = ["en_clip04", "en_clip09", "en_clip02"]
    ref = " ".join(meta[p]["text"] for p in parts)
    doc["long"]["reference_concat"] = ref
    doc["long"]["oracle_single_concat"] = " ".join(oracle[p]["text"] for p in parts)
    e, n = C.wer_counts(C.norm_text(ref), C.norm_text(doc["long"]["text"]))
    doc["long"]["wer"] = [e, n]
    e2, _ = C.wer_counts(C.norm_text(ref), C.norm_text(doc["long"]["oracle_single_concat"]))
    doc["long"]["oracle_single_concat_wer"] = [e2, n]
    for k in ["hotword", "no_hotword_rerun"]:
        e, n = C.wer_counts(C.norm_text(meta["en_clip04"]["text"]), C.norm_text(doc[k]["text"]))
        doc[k]["wer"] = [e, n]
    with open(os.path.join(C.WORK, "oracle_extra.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: doc[k]["text"] for k in ["long", "hotword", "no_hotword_rerun"]}, ensure_ascii=False, indent=1))
    print("ORACLE_EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
