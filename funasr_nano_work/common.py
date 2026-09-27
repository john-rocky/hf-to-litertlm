"""Shared helpers for the Fun-ASR-Nano lane (no funasr / torch import here, so both venvs can use it).

norm_text / wer_counts are copied verbatim from vibevoice_asr_work/common.py so the WER yardstick is
the same as the lane precedent ("Mr." vs MISTER and digits vs words count as errors)."""
import json
import math
import os
import re

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out")
SR = 16000
FRAME = 960              # runtime frame = hop = one LFR step (6 x 160 samples)
WIN_FRAMES = 504         # 30.24 s window
WIN_SAMPLES = FRAME * WIN_FRAMES
OUT_TOKENS = 63          # features rows = WIN_FRAMES // 8

IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643
# funasr get_prompt(hotwords=[], language=None, itn=True) wrapped by data_load_speech (tokenized
# as two strings around the speech placeholder; no special token around the audio embeddings).
PREFIX_IDS = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
              105761, 46670, 61443, 5122]
SUFFIX_IDS = [151645, 198, 151644, 77091, 198]
PROMPT_ITN = "语音转写："
PROMPT_NOITN = "语音转写，不进行文本规整："


def load_meta():
    return json.load(open(os.path.join(WORK, "fixtures", "meta.json")))


def fbank_frames(n):
    """kaldi snip_edges frame count for n samples (25 ms / 10 ms at 16 kHz)."""
    return 1 + (n - 400) // 160 if n >= 400 else 0


def lfr_len(n):
    return math.ceil(fbank_frames(n) / 6)


def fake_token_len(L):
    """funasr FunASRNano.data_load_speech (use_low_frame_rate=True)."""
    olens = 1 + (L - 3 + 2) // 2
    olens = 1 + (olens - 3 + 2) // 2
    return (olens - 1) // 2 + 1


def postprocess(response):
    """funasr FunASRNano.inference_llm: text / text_tn from the decoded response."""
    text = re.sub(r"\s+", " ", response.replace("/sil", " "))
    text_tn = re.sub(r"[^\w\s　一-鿿]+", "", response)
    return text, text_tn


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


def corpus_wer(rows, meta_by_id, key="text"):
    errs = words = 0
    per = {}
    for r in rows:
        ref = meta_by_id[r["id"]].get("text")
        if not ref:
            continue
        e, w = wer_counts(norm_text(ref), norm_text(r[key]))
        per[r["id"]] = [e, w]
        errs += e
        words += w
    return {"errors": errs, "words": words, "wer": (errs / words) if words else None, "per_clip": per}
