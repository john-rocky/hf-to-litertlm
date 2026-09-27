#!/usr/bin/env python3
"""LiteRT-LM runtime gate for the Fun-ASR-Nano-2512 bundle (litert-lm 0.17.1 python API). Pure python on purpose:
the run venv (~/venvs/lt0171run) has no numpy.

One engine per process; per clip a NEW conversation (greedy: top_k 1 / temp 0, max_output_tokens 512) and ONE user
message; the response text goes through funasr's own post-processing (`/sil` -> space, whitespace runs -> one space)
before it is compared with the oracle text.

Modes
  wav      the 25 fixtures (`fixtures/meta.json`), audio only -> template default instruction 语音转写：
  mp3      the 5 official example mp3s (out/hf_official/example/<lang>.mp3), decoded by the runtime (miniaudio)
  long     out/fixtures/long_en.wav (en_clip04 + en_clip09 + en_clip02, make_long_fixture.py) as one message
  hotword  en_clip04 with the funasr hotword prompt as a Text item in the same message
The rendered prompt of the first message is printed and asserted (audio-only render for wav / mp3 / long, the hotword
render for hotword). Writes mac_gate_<tag>.json.

  python mac_gate.py --bundle out/bundle/Fun-ASR-Nano-2512.litertlm --backend cpu --audio-backend cpu --tag X
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
EXPECTED_RENDER = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：<|AUDIO|>"
                   "<|im_end|>\n<|im_start|>assistant\n")
HOTWORDS = ["Birket Foster", "John Collier"]
HOTWORD_TEXT = ("请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
                "热词列表：[" + ", ".join(HOTWORDS) + "]\n语音转写：")
EXPECTED_RENDER_HOTWORD = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + HOTWORD_TEXT +
                           "<|AUDIO|><|im_end|>\n<|im_start|>assistant\n")
LONG_PARTS = ["en_clip04", "en_clip09", "en_clip02"]
FLEURS = os.environ.get("FUNASR_FLEURS_FIXTURES", os.path.join(HERE, "out", "fleurs"))  # google/fleurs test clips as 16 kHz WAV + meta.json
FLEURS_CONFIGS = ["en_us", "cmn_hans_cn", "ja_jp"]


def norm_text(s):  # = common.norm_text
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]+", " ", s).split()


def wer_counts(ref, hyp):  # = common.wer_counts (Levenshtein on words), pure python
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1], len(ref)


def postprocess(response):  # funasr FunASRNano.inference_llm `text`
    return re.sub(r"\s+", " ", response.replace("/sil", " "))


def loop_suspect(text):
    """A unit of >= 4 characters repeated >= 4 times back to back (degenerate decode)."""
    return bool(re.search(r"(.{4,}?)\1{3,}", text))


def resp_text(resp):
    if hasattr(resp, "contents"):
        return "".join(getattr(c, "text", "") for c in resp.contents.contents)
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else "".join(i.get("text", "") for i in c if isinstance(i, dict))
    return str(resp)


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_json(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def items_for(mode, meta_by_id, oracle, ids):
    """[(id, audio path, audio seconds, text item or None, oracle text or None, reference words or None)]"""
    rows = []
    if mode == "wav":
        for fid in ids:
            m = meta_by_id[fid]
            p = os.path.join(HERE, m["file"])
            rows.append((fid, p, wav_seconds(p), None, oracle[fid]["text"], m.get("text")))
    elif mode == "mp3":
        for fid in ids:
            lang = fid.split("_", 1)[1]
            p = os.path.join(OUT, "hf_official", "example", f"{lang}.mp3")
            rows.append((fid, p, meta_by_id[fid]["duration_s"], None, oracle[fid]["text"], meta_by_id[fid].get("text")))
    elif mode == "long":
        p = os.path.join(OUT, "fixtures", "long_en.wav")
        ref = " ".join(meta_by_id[i]["text"] for i in LONG_PARTS)
        extra = load_json("oracle_extra.json")
        o = extra["long"]["text"] if extra and "long" in extra else None
        rows.append(("long_en", p, wav_seconds(p), None, o, ref))
    elif mode == "fleurs":  # FLEURS fixtures (see oracle_fleurs.py); oracle = oracle_fleurs.json, scored by score_fleurs.py
        fl = {c["name"]: c for c in json.load(open(os.path.join(FLEURS, "meta.json")))["clips"]}
        of = load_json("oracle_fleurs.json")
        of = {r["name"]: r["text"] for r in of["rows"]} if of else {}
        for fid in ids:
            c = fl[fid]
            p = os.path.join(FLEURS, c["path"])
            rows.append((fid, p, wav_seconds(p), None, of.get(fid), None))
    elif mode == "hotword":
        fid = "en_clip04"
        p = os.path.join(HERE, meta_by_id[fid]["file"])
        extra = load_json("oracle_extra.json")
        o = extra["hotword"]["text"] if extra and "hotword" in extra else None
        rows.append((fid, p, wav_seconds(p), HOTWORD_TEXT, o, meta_by_id[fid].get("text")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--audio-backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--mode", default="wav", choices=["wav", "mp3", "long", "hotword", "fleurs"])
    ap.add_argument("--ids", default="", help="comma-separated fixture ids (default: all for the mode)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--cache-dir", default="", help="default out/runtime_cache/<bundle stem>")
    ap.add_argument("--compare", default="", help="another mac_gate JSON to diff texts against (same ids)")
    ap.add_argument("--act-f32", action="store_true",
                    help="engine activation_data_type=FLOAT32 (overrides the bundle's prefer_activation_type)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(HERE, "fixtures", "meta.json")))
    meta_by_id = {m["id"]: m for m in meta}
    oracle = {r["id"]: r for r in json.load(open(os.path.join(HERE, "oracle_transcripts.json")))["rows"]}
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.mode == "fleurs":
        ids = [c["name"] for c in json.load(open(os.path.join(FLEURS, "meta.json")))["clips"]
               if c["path"].startswith("fleurs/") and c["source"]["config"] in FLEURS_CONFIGS]
    elif args.mode == "mp3":
        ids = [m["id"] for m in meta if m["id"].startswith("example_")]
    else:
        ids = [m["id"] for m in meta]
    rows_in = items_for(args.mode, meta_by_id, oracle, ids)

    import litert_lm
    from litert_lm import Content, Contents, Message
    from litert_lm import interfaces as I
    from importlib.metadata import version

    def be(name):
        return I.CPU(thread_count=args.threads) if name == "cpu" else I.GPU()

    bundle = os.path.abspath(args.bundle)
    stem = os.path.splitext(os.path.basename(bundle))[0]
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(OUT, "runtime_cache", stem))
    os.makedirs(cache_dir, exist_ok=True)
    cache_before = sorted(os.listdir(cache_dir))
    print(f"litert-lm {version('litert-lm')} | bundle {bundle} ({os.path.getsize(bundle):,} B) | backend {args.backend} "
          f"audio {args.audio_backend} | act_f32 {args.act_f32} | mode {args.mode} | cache {cache_dir} "
          f"({len(cache_before)} files before)", flush=True)
    kw = {"activation_data_type": litert_lm.ActivationDataType.FLOAT32} if args.act_f32 else {}
    t0 = time.time()
    engine = litert_lm.Engine(bundle, backend=be(args.backend), audio_backend=be(args.audio_backend), cache_dir=cache_dir,
                              **kw)
    load_s = time.time() - t0
    print(f"engine loaded in {load_s:.2f} s", flush=True)
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

    compare = {r["id"]: r for r in json.load(open(args.compare))["rows"]} if args.compare else {}
    rows = []
    render_checked = None
    for k, (fid, path, secs, text_item, o_text, ref) in enumerate(rows_in):
        items = [Content.AudioFile(os.path.abspath(path))]
        if text_item is not None:
            items.append(Content.Text(text_item))  # after the audio on purpose: the template puts text first anyway
        msg = Message.user(Contents.of(items))
        conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
        err = None
        try:
            if k == 0:
                rendered = conv.render_message_to_string(msg)
                want = EXPECTED_RENDER_HOTWORD if args.mode == "hotword" else EXPECTED_RENDER
                print("RENDERED:", repr(rendered), flush=True)
                render_checked = {"rendered": rendered, "expected": want, "equal": rendered == want}
                assert rendered == want, (rendered, want)
            t1 = time.time()
            try:
                resp = conv.send_message(msg)
                raw = resp_text(resp)
            except Exception as e:  # noqa: BLE001  (recorded, not fixed: the GPU-audio row expects failures)
                raw, err = "", repr(e)
            wall = time.time() - t1
            tc = conv.token_count
        finally:
            conv.close()
        text = postprocess(raw)
        r = {"id": fid, "file": os.path.relpath(path, HERE), "audio_s": round(secs, 3), "text_raw": raw, "text": text,
             "wall_s": round(wall, 3), "token_count": tc, "oracle_text": o_text,
             "oracle_equal": (text == o_text) if o_text is not None else None,
             "oracle_equal_stripped": (text.strip() == o_text.strip()) if o_text is not None else None,
             "loop_suspect": loop_suspect(text)}
        if err:
            r["error"] = err
        if ref:
            e, n = wer_counts(norm_text(ref), norm_text(text))
            r["wer"] = [e, n]
            if o_text is not None:
                eo, _ = wer_counts(norm_text(ref), norm_text(o_text))
                r["oracle_wer"] = [eo, n]
        if fid in compare:
            r["compare_text"] = compare[fid]["text"]
            r["compare_equal"] = text == compare[fid]["text"]
        rows.append(r)
        flag = "==" if r["oracle_equal"] else ("!=" if r["oracle_equal"] is False else "..")
        print(f"[{fid:12s}] {secs:6.2f} s audio | {wall:6.2f} s | tok {tc} | {flag} {text!r}" + (f" | ERROR {err}" if err else "")
              + ("" if r["oracle_equal"] in (True, None) else f"\n      oracle: {o_text!r}"), flush=True)

    audio = sum(r["audio_s"] for r in rows)
    wall = sum(r["wall_s"] for r in rows)
    en = [r for r in rows if "wer" in r]
    summ = {"n": len(rows), "oracle_match": sum(1 for r in rows if r["oracle_equal"]),
            "oracle_match_stripped": sum(1 for r in rows if r["oracle_equal_stripped"]),
            "oracle_known": sum(1 for r in rows if r["oracle_equal"] is not None),
            "errors": sum(1 for r in rows if r.get("error")), "empty": sum(1 for r in rows if not r["text"].strip()),
            "loop_suspect": sum(1 for r in rows if r["loop_suspect"]),
            "wer_en": [sum(r["wer"][0] for r in en), sum(r["wer"][1] for r in en)] if en else None,
            "oracle_wer_en": [sum(r["oracle_wer"][0] for r in en if "oracle_wer" in r), sum(r["wer"][1] for r in en if "oracle_wer" in r)] if en else None,
            "audio_s": round(audio, 3), "wall_s": round(wall, 3), "rtf": round(wall / audio, 4) if audio else None,
            "rtf_excluding_first": round(sum(r["wall_s"] for r in rows[1:]) / sum(r["audio_s"] for r in rows[1:]), 4) if len(rows) > 1 else None,
            "first_clip_wall_s": rows[0]["wall_s"] if rows else None, "engine_load_s": round(load_s, 2)}
    if compare:
        summ["compare_equal"] = sum(1 for r in rows if r.get("compare_equal"))
        summ["compare_file"] = args.compare
    cache_after = sorted(os.listdir(cache_dir))
    doc = {"tag": args.tag, "mode": args.mode, "bundle": os.path.relpath(bundle, HERE), "bundle_bytes": os.path.getsize(bundle),
           "bundle_sha256": sha256(bundle), "backend": args.backend, "audio_backend": args.audio_backend,
           "act_f32": args.act_f32,
           "threads": args.threads, "max_output_tokens": args.max_out, "sampler": {"top_k": 1, "top_p": 1.0, "temperature": 0.0},
           "litert_lm": version("litert-lm"), "python": sys.version.split()[0], "cache_dir": os.path.relpath(cache_dir, HERE),
           "cache_files_before": cache_before, "cache_files_after": cache_after, "render_check": render_checked,
           "summary": summ, "rows": rows}
    if args.mode == "long":
        o_single = [oracle[i]["text"] for i in LONG_PARTS]
        doc["long"] = {"parts": LONG_PARTS, "oracle_single_concat": " ".join(o_single),
                       "reference_concat": " ".join(meta_by_id[i]["text"] for i in LONG_PARTS)}
        e, n = wer_counts(norm_text(doc["long"]["reference_concat"]), norm_text(doc["long"]["oracle_single_concat"]))
        doc["long"]["oracle_single_concat_wer"] = [e, n]
        if compare:
            pass
    with open(os.path.join(HERE, f"mac_gate_{args.tag}.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    print("GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
