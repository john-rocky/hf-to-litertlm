#!/usr/bin/env python3
# Third-party code, Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0); details in THIRD_PARTY_NOTICES.md.
# - normalize_language_name, detect_and_fix_repetitions, parse_asr_output, _ASR_TEXT_TAG, _LANG_PREFIX: copied from
#   qwen-asr 0.0.6, qwen_asr/inference/utils.py (Copyright 2026 The Alibaba Qwen team; https://github.com/Qwen/Qwen3-ASR).
# - _EN2ZH_PUNCT, _ZH2EN_PUNCT, _ALL_PUNCT_PAT, _normalize_punct_by_context, parse_language_output: copied from
#   netease-youdao/Confucius4-R2T2 @26d55a54, r2t2/r2t2_asr.py l.34-130 (Copyright 2026 The NetEase Youdao team;
#   https://github.com/netease-youdao/Confucius4-R2T2); streaming_transcribe / finish_streaming_transcribe below follow
#   the same file's l.302-579 with the engine call replaced.
# Code copied verbatim, docstrings omitted.
"""The vendor's streaming procedure S1 with the engine call swapped, on the released runtime and on an eager
reference. Run in ~/venvs/ltmain0918 (transformers 5.14.1 for the tokenizer; the runtime backend is a worker
subprocess in ~/venvs/lt0171run = litert-lm-api 0.17.1).

S1 = Confucius4-R2T2 r2t2/r2t2_asr.py (vendor clone 26d55a54) streaming_transcribe l.302-490 + finish_streaming_transcribe
l.492-579 with init_streaming_state's defaults (unfixed_chunk_num 2, unfixed_token_num 5, chunk_size_sec 2.0,
rollback_punctuation False, no context, no forced language). Per chunk: audio_accum += chunk (never trimmed, no
padding), prefix = "" for the first 2 chunks, else the previous raw decode cut at '|' minus its last 5 tokens
(re-encoded with the vendor tokenizer, k grows while the decode has U+FFFD), prompt = prompt_raw + prefix, one greedy
call, raw = prefix + _normalize_punct_by_context(gen_text); then the vendor's language / text / fixed-text updates
(a chunk whose raw has no <asr_text> does not advance chunk_id). The tail (< 2 s) goes through the finish step.
Helpers below are copied from the vendor repository and qwen-asr 0.0.6 (code verbatim, docstrings omitted; see the
notice at the top of this file and THIRD_PARTY_NOTICES.md).

  --backend runtime   per call a new Conversation on the r3 bundle, user message [AudioFile(accum wav), Text(prefix)];
                      the bundle template renders prompt_raw + prefix (asserted on every call: render ==
                      VENDOR_PROMPT + prefix). The accumulated audio is written as PCM16 (the samples are int16 / 32768,
                      so the runtime decodes the same floats).
  --backend eager     transformers 5.14.1 Qwen3ASRForConditionalGeneration, out/hf_layout, fp32, CPU:
                      processor(text=prompt_raw + prefix, audio=audio_accum) + generate(greedy, max_new_tokens), new ids
                      decoded with skip_special_tokens=True (= vLLM SamplingParams(skip_special_tokens=True)).
Both backends cap a call at --max_new_tokens (default 128).

  ~/venvs/ltmain0918/bin/python r3_stream.py --backend runtime --bundle out/export/c4r3_30s_C/model.litertlm \
      --out out/r3_stream/runtime_s1.jsonl
  ~/venvs/ltmain0918/bin/python r3_stream.py --backend eager --out out/r3_stream/eager_s1.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from typing import Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
import common  # noqa: E402

VENDOR_PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                 "<|im_end|>\n<|im_start|>assistant\n")
PROMPTS = {"vendor": VENDOR_PROMPT, "litert": common.LITERT_TORCH_PROMPT}
PROMPT_RAW = VENDOR_PROMPT  # set from --prompt in main(); the bundle's jinja must render this + prefix
SAMPLE_RATE = 16000
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 2.0

# ---- verbatim: qwen-asr 0.0.6 qwen_asr/inference/utils.py ------------------------------------------------------
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break

                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # extract "language xxx" from meta
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()
# ---- end verbatim (qwen-asr) -------------------------------------------------------------------------------------


# ---- verbatim: vendor r2t2/r2t2_asr.py l.34-121 ------------------------------------------------------------------
_EN2ZH_PUNCT = {',': '，', '.': '。', '!': '！', '?': '？', ';': '；', ':': '：', '(': '（', ')': '）'}
_ZH2EN_PUNCT = {v: k for k, v in _EN2ZH_PUNCT.items()}
_ALL_PUNCT_PAT = re.compile(r'[,\.!?;:()，。！？；：（）]')


def _normalize_punct_by_context(text: str) -> str:
    def _replace(m):
        punct = m.group()
        pos = m.start()
        prev_char = ""
        for i in range(pos - 1, -1, -1):
            if not text[i].isspace():
                prev_char = text[i]
                break
        if not prev_char:
            return punct
        if '一' <= prev_char <= '鿿':
            return _EN2ZH_PUNCT.get(punct, punct)
        elif prev_char.isascii() and (prev_char.isalnum() or prev_char in '"\''):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct
    return _ALL_PUNCT_PAT.sub(_replace, text)


def parse_language_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    if user_language == "English":
        s = str(raw).rstrip()
    else:
        s = str(raw).strip()
    if not s:
        return "", ""

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # extract "language xxx" from meta
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()
# ---- end verbatim (vendor) ---------------------------------------------------------------------------------------


class State:
    def __init__(self):
        self.chunk_id = 0
        self.buffer = np.zeros((0,), dtype=np.float32)
        self.audio_accum = np.zeros((0,), dtype=np.float32)
        self.prompt_raw = PROMPT_RAW
        self.force_language = None
        self.language = ""
        self.text = ""
        self._raw_decoded = ""


def decode_rollback(tok, raw, k):
    """vendor l.393-411 / l.459-469: decode(ids[:len - k]), growing k while the decode contains U+FFFD."""
    cur_ids = tok.encode(raw)
    while True:
        end_idx = max(0, len(cur_ids) - k)
        out = tok.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
        if '�' not in out:
            return out
        if end_idx == 0:
            return ""
        k += 1


def streaming_transcribe(gen, tok, pcm, state, chunk_size_samples, log):
    """vendor streaming_transcribe l.350-490 (rollback_punctuation False); gen(audio, prefix) -> (gen_text, info)."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)
    fixed_text = ""
    while state.buffer.shape[0] >= chunk_size_samples:
        chunk = state.buffer[:chunk_size_samples]
        state.buffer = state.buffer[chunk_size_samples:]
        state.audio_accum = chunk if state.audio_accum.shape[0] == 0 else np.concatenate([state.audio_accum, chunk])
        prefix = ""
        if state.chunk_id < UNFIXED_CHUNK_NUM:
            prefix = ""
        else:
            state._raw_decoded = state._raw_decoded.split("|")[0]
            prefix = decode_rollback(tok, state._raw_decoded, int(UNFIXED_TOKEN_NUM))
        prefix = prefix.split("|")[0]
        gen_text, info = gen(state.audio_accum, prefix)
        gen_text = _normalize_punct_by_context(gen_text).replace('�', '')
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang = None
        if state.force_language is None:
            lang, _ = parse_language_output(state._raw_decoded, user_language=state.force_language)
        if state.force_language == "Chinese" or lang == "Chinese":
            state._raw_decoded = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', state._raw_decoded)
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        has_tag = "<asr_text>" in state._raw_decoded
        if has_tag:
            state._raw_decoded = state._raw_decoded.split("<asr_text>")[0] + "<asr_text>" + txt
        else:
            state._raw_decoded = txt
        state._raw_decoded = state._raw_decoded.split("|")[0]
        k = int(UNFIXED_TOKEN_NUM)
        has_tag = "<asr_text>" in state._raw_decoded
        if has_tag and state._raw_decoded.split('<asr_text>')[1] == "":
            k = 0
        fixed_text = decode_rollback(tok, state._raw_decoded, k)
        if "<asr_text>" in fixed_text:
            _meta, fixed_text = fixed_text.split("<asr_text>", 1)
        fixed_text = fixed_text.split("|")[0]
        has_tag = "<asr_text>" in state._raw_decoded
        entry = {"step": "chunk", "chunk_id": state.chunk_id, "accum_seconds": round(len(state.audio_accum) / 16000, 3),
                 "prefix": prefix, "gen_text": gen_text, "raw": state._raw_decoded, **info}
        if not has_tag and state.force_language is None:
            state.text = ""
            fixed_text = ""
            entry.update({"text": state.text, "fixed_text": fixed_text, "advanced": False})
            log.append(entry)
            continue
        state.language = lang
        state.text = txt.split("|")[0]
        state.chunk_id += 1
        entry.update({"text": state.text, "fixed_text": fixed_text, "language": lang, "advanced": True})
        log.append(entry)
    return state.text, fixed_text


def finish_streaming_transcribe(gen, tok, state, log):
    """vendor finish_streaming_transcribe l.532-579."""
    if state.buffer is None or state.buffer.shape[0] == 0:
        return state.text
    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)
    state.audio_accum = tail if state.audio_accum.shape[0] == 0 else np.concatenate([state.audio_accum, tail])
    prefix = ""
    if state.chunk_id < UNFIXED_CHUNK_NUM:
        prefix = ""
    else:
        cur_ids = tok.encode(state._raw_decoded)
        end_idx = max(1, len(cur_ids) - int(UNFIXED_TOKEN_NUM))
        prefix = tok.decode(cur_ids[:end_idx])
    prefix = prefix.split("|")[0]
    gen_text, info = gen(state.audio_accum, prefix)
    gen_text = _normalize_punct_by_context(gen_text).replace('�', '')
    state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
    state._raw_decoded = state._raw_decoded.split("|")[0]
    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = txt.split("|")[0]
    state.chunk_id += 1
    log.append({"step": "finish", "chunk_id": state.chunk_id - 1, "accum_seconds": round(len(state.audio_accum) / 16000, 3),
                "prefix": prefix, "gen_text": gen_text, "raw": state._raw_decoded, "text": state.text,
                "language": lang, **info})
    return state.text


def write_pcm16(path, x):
    ints = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(ints.tobytes())


class RuntimeBackend:
    def __init__(self, bundle, max_new_tokens, wav_dir, threads):
        self.max_new_tokens = max_new_tokens
        self.wav_dir = wav_dir
        os.makedirs(wav_dir, exist_ok=True)
        cache = os.path.join(OUT, "r3_rt_cache", os.path.basename(os.path.dirname(bundle)))
        self.p = subprocess.Popen(
            [os.path.expanduser("~/venvs/lt0171run/bin/python"), os.path.join(HERE, "r3_stream_worker.py"),
             "--bundle", bundle, "--cache_dir", cache, "--threads", str(threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(wav_dir, "worker.stderr.log"), "w"),
            text=True, bufsize=1)
        self.ready = self._read()
        self.n = 0

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            if line.startswith("R3JSON "):
                return json.loads(line[7:])

    def __call__(self, audio, prefix):
        self.n += 1
        wav = os.path.join(self.wav_dir, f"accum_{self.n:05d}.wav")
        write_pcm16(wav, audio)
        self.p.stdin.write(json.dumps({"wav": wav, "prefix": prefix, "max_output_tokens": self.max_new_tokens},
                                      ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        r = self._read()
        os.remove(wav)
        if "error" in r:
            raise RuntimeError(r["error"])
        assert r["render"] == PROMPT_RAW + prefix, (r["render"], prefix)
        return r["text"], {"seconds": round(r["seconds"], 4), "render_ok": True, "token_count": r["token_count"]}

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


class EagerBackend:
    def __init__(self, max_new_tokens, threads):
        import torch
        import transformers
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(
            common.HF_LAYOUT, dtype=torch.float32).eval()
        self.proc = transformers.AutoProcessor.from_pretrained(common.HF_LAYOUT)
        self.max_new_tokens = max_new_tokens

    def __call__(self, audio, prefix):
        inputs = self.proc(text=PROMPT_RAW + prefix, audio=audio, return_tensors="pt")
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens)
        dt = time.time() - t0
        text = self.proc.batch_decode(out[:, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text, {"seconds": round(dt, 4), "n_input_ids": int(n_in)}


def select_clips(n_per_lang):
    clips = common.load_clips()
    out = [c for c in clips if c["config"] is None]
    for cfg in ("en_us", "cmn_hans_cn", "ja_jp"):
        out += [c for c in clips if c["config"] == cfg][:n_per_lang]
    return out


def append_only_violations(log):
    """Times the confirmed text (fixed_text after the tag) of an advancing chunk does not start with the previous
    confirmed text."""
    prev, viol, events = "", 0, []
    for e in log:
        if e["step"] != "chunk" or not e.get("advanced"):
            continue
        cur = e["fixed_text"]
        if prev and not cur.startswith(prev):
            viol += 1
            events.append({"chunk_id": e["chunk_id"], "prev": prev, "cur": cur})
        prev = cur
    return viol, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["runtime", "eager"], required=True)
    ap.add_argument("--bundle", default=os.path.join(OUT, "export", "c4r3_30s_C", "model.litertlm"))
    ap.add_argument("--n_per_lang", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", choices=["vendor", "litert"], default="vendor",
                    help="prompt_raw: vendor _build_text_prompt string, or the litert-torch prompt the ship bundle bakes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global PROMPT_RAW
    PROMPT_RAW = PROMPTS[args.prompt]
    import transformers
    tok = transformers.AutoProcessor.from_pretrained(common.HF_LAYOUT).tokenizer
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    if args.backend == "runtime":
        gen = RuntimeBackend(args.bundle, args.max_new_tokens, os.path.join(os.path.dirname(args.out), "wav_runtime"),
                             args.threads)
        header_extra = {"bundle": os.path.realpath(args.bundle), "worker": gen.ready}
    else:
        gen = EagerBackend(args.max_new_tokens, args.threads)
        header_extra = {"stack": f"transformers {transformers.__version__}, out/hf_layout fp32, CPU"}
    load_s = time.time() - t0
    clips = select_clips(args.n_per_lang)
    if args.limit:
        clips = clips[:args.limit]
    chunk_size_samples = int(round(CHUNK_SIZE_SEC * SAMPLE_RATE))
    with open(args.out, "w") as f:
        f.write(json.dumps({"type": "header", "backend": args.backend, "procedure": "vendor S1 streaming_transcribe + "
                            "finish_streaming_transcribe (r2t2_asr.py l.302-579 @26d55a54)",
                            "unfixed_chunk_num": UNFIXED_CHUNK_NUM, "unfixed_token_num": UNFIXED_TOKEN_NUM,
                            "chunk_size_sec": CHUNK_SIZE_SEC, "max_new_tokens": args.max_new_tokens,
                            "prompt": args.prompt, "prompt_raw": PROMPT_RAW,
                            "threads": args.threads, "load_seconds": round(load_s, 2), **header_extra},
                           ensure_ascii=False) + "\n")
        for c in clips:
            wav = common.read_wav(c["path"])
            state = State()
            log = []
            streaming_transcribe(gen, tok, wav, state, chunk_size_samples, log)
            final = finish_streaming_transcribe(gen, tok, state, log)
            viol, events = append_only_violations(log)
            secs = [e["seconds"] for e in log]
            row = {"clip": c["name"], "config": c["config"], "audio_seconds": round(len(wav) / 16000, 3),
                   "text": final, "language": state.language, "raw": state._raw_decoded, "calls": len(log),
                   "append_only_violations": viol, "violation_events": events,
                   "call_seconds_total": round(sum(secs), 3), "call_seconds_max": round(max(secs), 3) if secs else None,
                   "log": log}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(c["name"], len(log), "calls", round(sum(secs), 1), "s, viol", viol, "|", final[:70], flush=True)
    if args.backend == "runtime":
        gen.close()


if __name__ == "__main__":
    main()
