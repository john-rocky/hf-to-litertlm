"""Shared constants and helpers for the granite-speech-5.0-2b scripts (no torch import here)."""
import hashlib
import json
import os
from pathlib import Path

REPO_ID = "ibm-granite/granite-speech-5.0-2b"
REVISION = "981601698173ac1a71fe6838b127d8e54f8cf682"

SR = 16000
HOP = 160
WIN_S = 36                       # audio encoder window (s): 12 s = lcm of the 4 s attention block and the 0.3 s
                                 # Q-Former window, so a window edge cuts neither; 36 s = three of them
WIN_FRAMES = WIN_S * SR // HOP   # 3600 runtime frames of 160 samples
WIN_SAMPLES = WIN_FRAMES * HOP   # 576000
OUT_TOKENS = WIN_FRAMES // 10    # 360 audio tokens per window (10 per second)
AUDIO_TOKEN = "<|audio|>"
AUDIO_ID = 100352
EOS_ID = 100257
PAD_ID = 100256

USER_TEXT = "transcribe the speech with proper punctuation and capitalization."
USER_CONTENT = AUDIO_TOKEN + USER_TEXT
# the repo's chat_template.jinja rendered for [{"role": "user", "content": USER_CONTENT}]
PROMPT = "USER: " + USER_CONTENT + "\n ASSISTANT:"
MAX_NEW = 256


def wav_chunks(path):
    """RIFF walk (afconvert writes WAVE_FORMAT_EXTENSIBLE): (format code, channels, rate, bits, data bytes)."""
    raw = Path(path).read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", path
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    code = int.from_bytes(fmt[0:2], "little")
    if code == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the format code
        code = int.from_bytes(fmt[24:26], "little")
    return (code, int.from_bytes(fmt[2:4], "little"), int.from_bytes(fmt[4:8], "little"),
            int.from_bytes(fmt[14:16], "little"), data)


def wav_float(path):
    """Mono 16 kHz WAV -> float32 samples as LiteRT-LM's miniaudio decoder and torchaudio / soundfile return them:
    PCM16 -> int16 / 32768; IEEE float32 (format 3, e.g. the repo's multilingual_sample.wav) -> the stored samples."""
    import numpy as np
    code, ch, rate, bits, data = wav_chunks(path)
    assert (ch, rate) == (1, SR), (path, ch, rate)
    if (code, bits) == (1, 16):
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    assert (code, bits) == (3, 32), (path, code, bits)
    return np.frombuffer(data, dtype="<f4").astype(np.float32).copy()


def num_audio_tokens(n_samples):
    """transformers GraniteSpeechFeatureExtractor._get_num_audio_features (5.22.1)."""
    mel = n_samples // HOP + 1
    enc = mel // 2
    nblocks = -(-enc // 15)
    return nblocks * 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_clips(path):
    """clips.json written by fetch_clips.py: [{id, path (relative to the json's dir or absolute), ...}]."""
    base = Path(path).resolve().parent
    clips = json.load(open(path))["clips"]
    for c in clips:
        c["abspath"] = c["path"] if os.path.isabs(c["path"]) else str(base / c["path"])
    return clips


def read_jsonl(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows
