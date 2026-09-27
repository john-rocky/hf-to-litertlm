"""funasr oracle on the FLEURS fixtures (google/fleurs test split, en_us / cmn_hans_cn / ja_jp x 50 clips, 16 kHz WAV + meta.json under $FUNASR_FLEURS_FIXTURES).

Same setup as oracle_funasr.py (fp32 CPU, dither 0, greedy, itn true, no hotwords, language None) with one discarded
warm-up call. Every wav is checked against the sha256 in the lane's meta.json before use.
Run with ~/venvs/funasr_oracle/bin/python. Writes oracle_fleurs.json (scored by score_fleurs.py).
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import oracle_funasr as O  # noqa: E402

FIX = os.environ.get("FUNASR_FLEURS_FIXTURES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fleurs"))
CONFIGS = ["en_us", "cmn_hans_cn", "ja_jp"]


def fleurs_clips():
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    clips = [c for c in meta["clips"] if c["path"].startswith("fleurs/") and c["source"]["config"] in CONFIGS]
    assert len(clips) == 150, len(clips)
    for c in clips:
        p = os.path.join(FIX, c["path"])
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == c["sha256"], (p, h, c["sha256"])
    return clips


def main():
    clips = fleurs_clips()
    m, facts = O.build()
    cap = O.instrument(m)
    warm = m.generate(input=os.path.join(C.WORK, "out", "fixtures", "example_zh.wav"), llm_kwargs=dict(O.LLM_KWARGS),
                      itn=True, hotwords=[], language=None, max_length=512)
    print("warmup", warm[0]["text"], flush=True)
    rows = []
    t_all = time.time()
    for c in clips:
        cap.clear()
        p = os.path.join(FIX, c["path"])
        t0 = time.perf_counter()
        res = m.generate(input=p, llm_kwargs=dict(O.LLM_KWARGS), itn=True, hotwords=[], language=None, max_length=512)
        wall = time.perf_counter() - t0
        rows.append({"name": c["name"], "config": c["source"]["config"], "path": c["path"], "duration_s": c["duration_s"],
                     "sha256": c["sha256"], "text": res[0]["text"], "gen_ids": cap.get("gen_ids"),
                     "fake_token_len": int(cap["fake_token_len"].reshape(-1)[0]), "wall_s": round(wall, 3),
                     "embeds_dtype_at_prepare": cap.get("embeds_dtype_at_prepare")})
        print(f"{c['name']:22s} {c['duration_s']:5.1f}s {wall:5.2f}s | {res[0]['text']}", flush=True)
    doc = {"facts": {k: facts[k] for k in ["funasr_version", "torch_version", "frontend_dither", "ncpu", "torch_threads"]},
           "fixtures_root": FIX, "n": len(rows), "total_s": round(time.time() - t_all, 1), "rows": rows}
    with open(os.path.join(C.WORK, "oracle_fleurs.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("ORACLE_FLEURS_DONE", flush=True)


if __name__ == "__main__":
    main()
