"""Constants and host rules of the r1 streaming contract (numpy only; shared by the torch modules and the LiteRT host
loop in .venv-q). See graphs.py for the graphs."""
import numpy as np

WEIGHTS = "hf/nemotron/model.safetensors"
MELFB = "out/graphs/melfb.npy"

HID, HEADS, HDIM, FF, NLAYERS = 1024, 8, 128, 4096, 24
MEL, SUB_CH, PROJ = 128, 256, 640
CONV_K = 9
NUM_PROMPTS, PROMPT_FF = 128, 2048
LOOKAHEAD = 3
Q = LOOKAHEAD + 1                 # 4 encoder frames per chunk
CACHE = 56                        # sliding_window 57 - 1
KV = CACHE + Q                    # 60 keys
N_FIRST, T_FIRST = 4040, 25       # processor.num_samples_first_audio_chunk / num_mel_frames_first_audio_chunk
N_NEXT, T_NEXT = 5520, 32         # processor.num_samples_per_audio_chunk / num_mel_frames_per_audio_chunk
HOP, WIN, NFFT = 160, 400, 512
NBIN = NFFT // 2 + 1
LOG_GUARD = 2.0 ** -24
PREEMPH = 0.97
NEG = -30000.0                    # additive mask value of the empty cache slots (exp underflows to 0 like -inf)
DEC_HID, DEC_NL, VOCAB, BLANK, MAX_SYMS = 640, 2, 13088, 13087, 10



# --------------------------------------------------------------------------- host rules (CONTRACT)
def neg_mask(chunk_idx):
    """[1, 1, Q, KV]: NEG on the right-aligned cache slots not yet filled after chunk_idx chunks, else 0."""
    valid = min(CACHE, Q * chunk_idx)
    m = np.zeros((1, 1, Q, KV), np.float32)
    m[..., : CACHE - valid] = NEG
    return m


def one_hot(prompt_id):
    v = np.zeros((1, NUM_PROMPTS), np.float32)
    v[0, prompt_id] = 1.0
    return v


def pcm_chunks(audio):
    """The card's streaming loop on audio + N_NEXT zeros: first chunk = x[:4040]; then
    x[start:start + 5520] with start = mel_idx * 160 - 256 while start + 5520 < len(x)."""
    x = np.concatenate([audio.astype(np.float32), np.zeros(N_NEXT, np.float32)])
    out = [x[:N_FIRST]]
    mel_idx = T_FIRST
    start = mel_idx * HOP - NFFT // 2
    while start + N_NEXT < x.shape[0]:
        out.append(x[start:start + N_NEXT])
        mel_idx += T_NEXT
        start = mel_idx * HOP - NFFT // 2
    return out
