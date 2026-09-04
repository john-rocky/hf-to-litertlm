"""Shared helpers for the VibeVoice-ASR-BitNet lane: fixture I/O, the vendor's audio
normalizer, the vendor's prompt layout (VibeASR.cpp utils/prompt_builder.h), WER."""
import json
import math
import os
import re
import struct

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
SR = 24000
HOP = 3200  # samples per latent frame (8*5*5*4*2*2) -> 7.5 frames/s
SYSTEM_PROMPT = ("You are a helpful assistant that transcribes audio input into text output "
                 "in JSON format.")
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643
SPEECH_START, SPEECH_END, SPEECH_PAD = 151646, 151647, 151648  # <|object_ref_start|> <|object_ref_end|> <|box_start|>


def load_wav(path):
    """Minimal RIFF walk: afconvert emits WAVE_FORMAT_EXTENSIBLE (tag 65534) which the
    stdlib wave module refuses; the data chunk is plain PCM16LE mono."""
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
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def normalize_audio(x: np.ndarray, target_db_fs: float = -25.0, eps: float = 1e-6):
    """vibevoice.processor.audio_utils.AudioNormalizer (== VibeASR.cpp audio_io.h)."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = x * (10 ** (target_db_fs / 20) / (rms + eps))
    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx > 1.0:
        y = y / (mx + eps)
    return y.astype(np.float32)


def pad_to_hop(x: np.ndarray):
    n = math.ceil(len(x) / HOP) * HOP
    return np.pad(x, (0, n - len(x))), n // HOP


def build_prompt_ids(tok, n_audio_tokens: int, duration_sec: float, variant: str = "exact"):
    """Token ids exactly as VibeASR.cpp builds them for the 1.5B ("text") model.
    variant: exact     = vendor layout, no generation prompt (model emits the header)
             genprompt = exact + '<|im_start|>assistant\\n'
             nodur     = suffix without the duration sentence
             wrongdur  = duration replaced by 1.00
    """
    def t(s):
        return tok.encode(s, add_special_tokens=False)
    dur = f"{duration_sec:.2f}"
    if variant == "nodur":
        suffix = "\nPlease transcribe it."
    elif variant == "wrongdur":
        suffix = "\nThis is a 1.00 seconds audio, please transcribe it."
    else:
        suffix = f"\nThis is a {dur} seconds audio, please transcribe it."
    ids = [IM_START] + t("system\n" + SYSTEM_PROMPT) + [IM_END] + t("\n")
    ids += [IM_START] + t("user\n") + [SPEECH_START] + [SPEECH_PAD] * n_audio_tokens + [SPEECH_END]
    ids += t(suffix) + [IM_END] + t("\n")
    if variant == "genprompt":
        ids += [IM_START] + t("assistant\n")
    return ids


def strip_header(tok, gen_ids):
    """Drop a model-emitted '<|im_start|>assistant\\n' header and stop tokens."""
    ids = list(gen_ids)
    if ids and ids[0] == IM_START:
        # header = <|im_start|> assistant \n
        j = 1
        while j < len(ids) and j < 4 and ids[j] not in (IM_END, ENDOFTEXT):
            piece = tok.decode([ids[j]])
            j += 1
            if piece.endswith("\n"):
                break
        ids = ids[j:]
    out = []
    for i in ids:
        if i in (IM_END, ENDOFTEXT):
            break
        out.append(i)
    return out


def norm_text(s: str):
    s = s.upper().replace("-", " ")
    s = re.sub(r"[^A-Z0-9' ]+", " ", s)
    return s.split()


def wer_counts(ref_words, hyp_words):
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref_words) + 1)
    d[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            c = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + c)
    return int(d[-1, -1]), len(ref_words)


def load_meta():
    return json.load(open(os.path.join(WORK, "fixtures", "meta.json")))
