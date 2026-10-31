"""PyTorch reference U for parakeet-ultra r1 (CPU fp32, transformers ParakeetForTDT). Adapted from
parakeet_redux_work/scripts/redux_ref.py (r1 of that lane): the ternary expansion is gone, the weights are ultra's own.

U = ParakeetForTDT built from hf/ultra/config.json (byte-identical to the original nvidia/parakeet-tdt-0.6b-v3
config.json: same git blob), every tensor of hf/ultra/model.safetensors loaded strictly (F16 raised to fp32, F32 and the
I64 BatchNorm counters as stored); vad_head.* (6 tensors) is the only thing left out. Arithmetic fp32 on the CPU (memory
mac-cpu-bf16-gemm-slow). Feature extractor = the original repo's processor_config.json (ParakeetFeatureExtractor; ultra
ships none), generation config = the original's generation_config.json.

Text = the TDT decode: tokenizers.Tokenizer.decode(ids, skip_special_tokens=True) on the repo's tokenizer.json, no
grouping of repeated ids (ParakeetTokenizer.decode's default groups them = the CTC rule).

tdt_loop() is a plain greedy TDT loop with the rules of transformers' ParakeetTDTGenerationMixin (token = argmax over
the vocab, duration = argmax over the duration logits, blank with duration 0 advances 1, stop at the encoder valid
length, eos stops, no per-frame symbol cap) plus an optional NeMo-style cap; gaps=True also keeps each step's top-2
logit gap of the token head and of the duration head (the P0 noise rule of LANE §1b).
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K = os.path.dirname(HERE)
ORIG = os.path.join(K, "hf", "orig")
ULTRA = os.path.join(K, "hf", "ultra")
VAD_KEYS = {"vad_head.ctx.bias", "vad_head.ctx.weight", "vad_head.out.bias", "vad_head.out.weight",
            "vad_head.proj.bias", "vad_head.proj.weight"}
BLANK = 8192
SR = 16000
_TK = None


def set_threads(n):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(ORIG)


def read_safetensors(path):
    """-> {name: np.ndarray} as stored (F16 stays float16, I64 int64); 0-d tensors stay 0-d."""
    from safetensors.numpy import load_file
    return {k: np.array(v, copy=True) for k, v in load_file(path).items()}


def ultra_state_dict(path=None):
    """-> ({name: torch tensor}, {stored dtype: count}): F16 -> float32, F32 as is, I64 as is."""
    raw = read_safetensors(path or os.path.join(ULTRA, "model.safetensors"))
    sd, dt = {}, {}
    for k, v in raw.items():
        dt[str(v.dtype)] = dt.get(str(v.dtype), 0) + 1
        t = torch.from_numpy(v)
        sd[k] = t.to(torch.float32) if t.dtype == torch.float16 else t
    return sd, dt


def build(which="U"):
    """-> (model, load_report dict)"""
    from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    if which != "U":
        raise SystemExit(which)
    rep = {"which": which}
    cfg = ParakeetTDTConfig.from_pretrained(ULTRA)
    cfg.torch_dtype = torch.float32
    model = ParakeetForTDT(cfg)
    sd, dt = ultra_state_dict()
    rep["stored_dtypes"] = dt
    model_keys = set(model.state_dict().keys())
    unexpected = sorted(set(sd) - model_keys)
    rep["unexpected"] = unexpected
    if set(unexpected) != VAD_KEYS:
        raise SystemExit(f"unexpected keys other than vad_head: {sorted(set(unexpected) - VAD_KEYS)}")
    for k in unexpected:
        sd.pop(k)
    shape_bad = [(k, tuple(sd[k].shape), tuple(v.shape)) for k, v in model.state_dict().items()
                 if k in sd and tuple(sd[k].shape) != tuple(v.shape)]
    if shape_bad:
        raise SystemExit(f"shape mismatch: {shape_bad[:5]}")
    res = model.load_state_dict(sd, strict=True)
    rep["missing"] = list(res.missing_keys)
    rep["n_tensors_loaded"] = len(sd)
    model.generation_config = GenerationConfig.from_pretrained(ORIG)
    model = model.float().eval()
    rep["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    rep["n_params"] = int(sum(p.numel() for p in model.parameters()))
    emb = model.decoder.embedding.weight
    rep["blank_embedding_row_absmax"] = float(emb[BLANK].abs().max())
    rep["attn_implementation"] = model.config._attn_implementation
    return model, rep


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def features(proc, audio, pad_to=None, mask_padding=True):
    """-> (input_features [1,T,128], attention_mask [1,T] bool).
    pad_to=None: the audio alone (the processor's own call).
    pad_to=N, mask_padding=True: zeros to N samples, the extractor told the true length (stats over the valid frames,
    padded frames zeroed, mask = valid frames)."""
    fe = proc.feature_extractor
    if pad_to is None:
        b = fe(audio, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
    elif mask_padding:
        b = fe(audio, sampling_rate=SR, return_tensors="pt", return_attention_mask=True,
               padding="max_length", max_length=pad_to)
    else:
        x = np.zeros(pad_to, np.float32)
        x[:len(audio)] = audio
        b = fe(x, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
    return b["input_features"], b["attention_mask"]


@torch.no_grad()
def gen(model, feats, mask):
    """transformers generate(); mask=None -> no attention mask (all frames valid, decode walks every frame)."""
    kw = {"input_features": feats}
    if mask is not None:
        kw["attention_mask"] = mask
    out = model.generate(**kw, return_dict_in_generate=True)
    seq = out.sequences[0].tolist()
    dur = out.durations[0].tolist()
    return seq, dur


def steps_from_generate(seq, dur):
    """generate's (sequences, durations) -> [(frame, token, dur)] per step (the start token dropped)."""
    steps = []
    t = 0
    for tok, d in zip(seq[1:], dur[1:]):
        steps.append((t, tok, d))
        t += d
    return steps


@torch.no_grad()
def encode(model, feats, mask):
    """-> (enc_proj [T,640] fp32 torch, enc_hidden [T,1024], valid_T int)"""
    o = model.get_audio_features(input_features=feats, attention_mask=mask, output_attention_mask=True)
    T = o.pooler_output.shape[1]
    valid = int(o.attention_mask.sum()) if (mask is not None and o.attention_mask is not None) else T
    return o.pooler_output[0], o.last_hidden_state[0], valid


@torch.no_grad()
def tdt_loop(model, enc_proj, valid_T, cap=None, eos=3, max_steps=None, gaps=False):
    """Greedy TDT on one utterance. enc_proj: [T,640] torch fp32. cap=None: transformers' TDT rule (no per-frame
    cap); cap=10: after 10 non-blank emissions on one frame, advance 1 (NeMo greedy). gaps=True: each step also
    keeps (token top-2 gap, duration top-2 gap). -> dict"""
    dec = model.decoder
    joint = model.joint
    V = model.config.vocab_size
    T = enc_proj.shape[0]
    emb = dec.embedding.weight
    h = torch.zeros(dec.lstm.num_layers, 1, dec.lstm.hidden_size)
    c = torch.zeros_like(h)
    o, (h, c) = dec.lstm(emb[BLANK].view(1, 1, -1), (h, c))
    d_out = dec.decoder_projector(o)[0, 0]
    t = 0
    steps, gap_rows = [], []
    tokens, frames, durs = [], [], []
    on_frame = 0
    cap_hits = 0
    max_run = 0
    eos_hit = False
    if max_steps is None:
        max_steps = model.config.max_symbols_per_step * T
    n = 0
    while t < valid_T and n < max_steps:
        n += 1
        tt = min(t, T - 1)
        logits = joint.head(joint.activation(enc_proj[tt] + d_out))
        tl, dl = logits[:V], logits[V:]
        tok = int(torch.argmax(tl))
        d = int(torch.argmax(dl))
        if gaps:
            ts, ds = torch.topk(tl, 2).values, torch.topk(dl, 2).values
            gap_rows.append((float(ts[0] - ts[1]), float(ds[0] - ds[1])))
        if tok == BLANK and d == 0:
            d = 1
        steps.append((t, tok, d))
        if tok != BLANK:
            tokens.append(tok)
            frames.append(t)
            durs.append(d)
            if tok == eos:
                eos_hit = True
                break
            o, (h, c) = dec.lstm(emb[tok].view(1, 1, -1), (h, c))
            d_out = dec.decoder_projector(o)[0, 0]
        if d == 0:
            on_frame += 1 if tok != BLANK else 0
            max_run = max(max_run, on_frame)
            if cap is not None and on_frame >= cap:
                cap_hits += 1
                d = 1
        if d > 0:
            on_frame = 0
        t += d
    r = {"tokens": tokens, "frames": frames, "durs": durs, "steps": steps, "n_steps": n,
         "cap_hits": cap_hits, "max_run_same_frame": max_run, "eos_hit": eos_hit}
    if gaps:
        r["gaps"] = gap_rows
    return r


def tdt_text(ids):
    """TDT decode of one id list (no grouping of repeated ids)."""
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(os.path.join(ULTRA, "tokenizer.json"))
    return _TK.decode([i for i in ids if i != BLANK], skip_special_tokens=True)


def decode_text(proc, ids):
    """kept for the copied scripts' call sites: the TDT decode (proc unused)."""
    return tdt_text(ids)


def env_record():
    import platform
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "machine": platform.machine(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]
