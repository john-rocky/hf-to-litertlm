"""Shared helpers for the Qwen3-ASR-1.7B scripts in this directory (stdlib; numpy only inside read_wav, so every venv
used here can import it). Adapted from confucius4_r2t2_work/common.py: the checkpoint is already in the transformers
v5 -hf layout, so the model directory is the Hub snapshot itself (no layout conversion)."""
import json
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIX = os.environ.get("C4_FIXTURES", os.path.expanduser("~/code/coreai/_funasr_nano/fixtures"))
REPO = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
MODEL_DIR = os.environ.get("QWEN3_ASR17_DIR", os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/" + REVISION))
ASR_TEXT_TAG = "<asr_text>"
# litert-torch Qwen3AsrProcessor._PROMPT (litert_torch/generative/export_hf/model_ext/qwen3/qwen3_asr.py), the
# prompt the export bakes into the audio encoder as prefix ids [151644, 872, 151669] / postfix
# [151670, 151645, 151644, 77091, 198].
LITERT_TORCH_PROMPT = ("<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|>"
                       "<|im_start|>assistant\n")


def load_clips():
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    clips = []
    for c in meta["clips"]:
        cfg = c["source"].get("config") if c["path"].startswith("fleurs/") else None
        clips.append({"name": c["name"], "path": os.path.join(FIX, c["path"]), "config": cfg,
                      "duration_s": c["duration_s"], "num_samples": c["num_samples"],
                      "ref": (c["reference_text"] or {}).get("raw_transcription")})
    return clips


def read_wav(path):
    """PCM16 mono 16 kHz WAV -> float32 in [-1, 1) as int16 / 32768 (what soundfile returns for dtype='float32')."""
    import numpy as np  # lazy: the released-runtime venv (lt0171run) has no numpy and does not call this
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float32) / 32768.0


def split_raw(raw):
    """Same split as qwen_asr.inference.utils.parse_asr_output for the no-forced-language case:
    language = text between 'language ' and the tag (first line), text = after the tag, stripped."""
    s = (raw or "").strip()
    if ASR_TEXT_TAG not in s:
        return "", s
    meta, text = s.split(ASR_TEXT_TAG, 1)
    lang = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            lang = line[len("language "):].strip()
            break
    if "language none" in meta.lower():
        lang = ""
    return lang, text.strip()
