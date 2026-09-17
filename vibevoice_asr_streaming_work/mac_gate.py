#!/usr/bin/env python3
"""LiteRT-LM runtime gate for the streaming bundle: one conversation per clip, one user turn per
26-frame window (22-frame advance, 4-frame lookahead kept by this script = the vendor's
`streaming_generate`), transcripts compared with the eager protocol run (eager_precheck.json).

  python mac_gate.py --bundle out/bundle/VibeVoice-ASR-Streaming-1.5B.litertlm [--backend cpu|gpu]
                     [--audio-backend cpu|gpu] [--limit N] [--tag T]
"""
import argparse
import json
import os
import struct
import sys
import time

import array
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BITNET = os.path.join(os.path.dirname(HERE), "vibevoice_asr_work")  # fixtures live there
# Pure python on purpose: the litert-lm-api run venvs have no numpy.

SR, HOP = 24000, 3200
CHUNK_FRAMES, LA_FRAMES = 22, 4  # preprocessor_config.json of the checkpoint


def load_wav(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", path
    i, sr, ch, bits, data = 12, None, None, None, None
    while i + 8 <= len(raw):
        cid, sz = raw[i:i + 4], struct.unpack("<I", raw[i + 4:i + 8])[0]
        body = raw[i + 8:i + 8 + sz]
        if cid == b"fmt ":
            _tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        i += 8 + sz + (sz & 1)
    assert sr == SR and ch == 1 and bits == 16, (path, sr, ch, bits)
    a = array.array("h")
    a.frombytes(data[:len(data) // 2 * 2])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def norm_text(s):
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]+", " ", s).split()


def wer_counts(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1], len(ref)


def write_wav(path, pcm16):  # array('h') -> 24 kHz mono PCM16 wav
    pcm = pcm16.tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
    with open(path, "wb") as f:
        f.write(hdr + b"data" + struct.pack("<I", len(pcm)) + pcm)


def windows_of(wav):
    chunk_s, win_s = CHUNK_FRAMES * HOP, (CHUNK_FRAMES + LA_FRAMES) * HOP
    out, s = [], 0
    while s < len(wav):
        seg = array.array("h", wav[s:min(s + win_s, len(wav))])
        if len(seg) < win_s:
            seg.extend([0] * (win_s - len(seg)))
        out.append(seg)
        s += chunk_s
    return out


def resp_text(resp):
    if hasattr(resp, "contents"):
        return "".join(getattr(c, "text", "") for c in resp.contents.contents)
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else "".join(i.get("text", "") for i in c if isinstance(i, dict))
    return str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--audio-backend", default="cpu")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--act-f32", action="store_true", help="activation_data_type=F32 (probe the fp16 hazard on GPU)")
    ap.add_argument("--wav-dir", default=os.path.join(HERE, "out", "windows"))
    args = ap.parse_args()
    os.makedirs(args.wav_dir, exist_ok=True)

    import litert_lm
    from litert_lm import Message, Contents, Content
    from litert_lm import interfaces as I

    def be(name):
        return I.CPU(thread_count=args.threads) if name == "cpu" else I.GPU()

    t0 = time.time()
    kw = {"activation_data_type": litert_lm.ActivationDataType.FLOAT32} if args.act_f32 else {}
    engine = litert_lm.Engine(args.bundle, backend=be(args.backend), audio_backend=be(args.audio_backend), **kw)
    load_s = time.time() - t0
    print(f"engine loaded in {load_s:.1f}s  backend={args.backend} audio_backend={args.audio_backend}")
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

    eager = {r["id"]: r for r in json.load(open(os.path.join(HERE, "eager_precheck.json")))}
    meta = json.load(open(os.path.join(BITNET, "fixtures", "meta.json")))[:args.limit]
    rows, errs, words, agree, n_chunks = [], 0, 0, 0, 0
    for i, m in enumerate(meta):
        wav = load_wav(os.path.join(BITNET, m["file"]))
        wins = windows_of(wav)
        conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
        texts, times, tokens = [], [], []
        try:
            for j, w in enumerate(wins):
                p = os.path.abspath(os.path.join(args.wav_dir, f"{m['id']}_w{j:02d}.wav"))
                write_wav(p, w)
                msg = Message.user(Contents.of([Content.AudioFile(p)]))
                if i == 0 and j < 2:
                    print(f"RENDERED (turn {j}):", repr(conv.render_message_to_string(msg)))
                t1 = time.time()
                resp = conv.send_message(msg)
                times.append(time.time() - t1)
                texts.append(resp_text(resp))
                tc = conv.token_count
                tokens.append(int(tc() if callable(tc) else tc))
        finally:
            conv.close()
        hyp = "".join(texts)
        hyp_plain = re.sub(r"^\s*(speaker\s*\d+|\[?[Ss]peaker[^:\]]*\]?)\s*:\s*", "", hyp)
        e, n = wer_counts(norm_text(m["text"]), norm_text(hyp_plain))
        errs, words = errs + e, words + n
        ref_chunks = eager.get(m["id"], {}).get("streaming_chunks", [])
        same = [a.strip() == b.strip() for a, b in zip(texts, ref_chunks)] if len(ref_chunks) == len(texts) else []
        agree += sum(same)
        n_chunks += len(texts)
        rows.append({"id": m["id"], "dur": round(len(wav) / SR, 2), "windows": len(wins), "chunks": texts,
                     "eager_chunks": ref_chunks, "chunk_agree": same, "hyp": hyp, "ref": m["text"],
                     "errs": e, "words": n, "sec_per_turn": [round(t, 2) for t in times], "token_count": tokens})
        print(f"[{m['id']}] {len(wins)} windows errs={e}/{n} agree={sum(same)}/{len(texts)} "
              f"turns={[round(t,1) for t in times]}s tokens={tokens} | {hyp!r}")
        sys.stdout.flush()
    print(f"corpus WER {errs}/{words} = {100*errs/max(words,1):.2f}%  chunk-level agreement with eager {agree}/{n_chunks}  load {load_s:.1f}s")
    tag = args.tag or f"{args.backend}_{args.audio_backend}{'_f32' if args.act_f32 else ''}"
    json.dump({"bundle": args.bundle, "backend": args.backend, "audio_backend": args.audio_backend,
               "wer_errs": errs, "ref_words": words, "chunk_agree": agree, "chunks": n_chunks,
               "load_s": round(load_s, 1), "rows": rows},
              open(os.path.join(HERE, f"mac_gate_{tag}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
