"""PyTorch references for parakeet-redux r1 (CPU fp32, transformers ParakeetForTDT).

B = ParakeetForTDT built from the redux config.json, ternary modules expanded by unpack3.py, the redux dense
tensors (F16 raised to fp32, BatchNorm running stats included) loaded strictly; vad_head.* (6 tensors) is the only
thing left out. O = the original nvidia/parakeet-tdt-0.6b-v3 via from_pretrained (the same class and code).
Both use the original repo's processor_config.json (ParakeetFeatureExtractor), its tokenizer files (tokenizer.json
is byte-identical in both repos) and its generation_config.json.

tdt_loop() is a plain greedy TDT loop with the same rules as transformers' ParakeetTDTGenerationMixin
(token = argmax over the vocab, duration = argmax over the duration logits, blank with duration 0 advances 1,
stop at the encoder valid length, eos stops, no per-frame symbol cap) plus an optional NeMo-style cap; it takes
any encoder output (torch or LiteRT) and is checked against generate() token for token.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unpack3  # noqa: E402

K = os.path.dirname(HERE)
ORIG = os.path.join(K, "hf", "orig")
REDUX = os.path.join(K, "hf", "redux")
VAD_KEYS = {"vad_head.ctx.bias", "vad_head.ctx.weight", "vad_head.out.bias", "vad_head.out.weight",
            "vad_head.proj.bias", "vad_head.proj.weight"}
BLANK = 8192
SR = 16000


def set_threads(n):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(ORIG)


def build(which):
    """-> (model, load_report dict)"""
    from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    rep = {"which": which}
    if which == "B":
        cfg = ParakeetTDTConfig.from_pretrained(REDUX)
        cfg.torch_dtype = torch.float32
        model = ParakeetForTDT(cfg)
        sd = unpack3.load_redux_state_dict(os.path.join(REDUX, "model.safetensors"),
                                           os.path.join(REDUX, "ternary.json"))
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
        dtype_in = {k: str(v.dtype) for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=True)
        rep["missing"] = list(res.missing_keys)
        rep["n_tensors_loaded"] = len(sd)
        rep["dtypes_loaded"] = sorted(set(dtype_in.values()))
        model.generation_config = GenerationConfig.from_pretrained(ORIG)
    elif which == "O":
        model, info = ParakeetForTDT.from_pretrained(ORIG, dtype=torch.float32, output_loading_info=True)
        rep.update({k: sorted(v) if isinstance(v, (set, list)) else str(v) for k, v in info.items()})
    else:
        raise SystemExit(which)
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
    padded frames zeroed, mask = valid frames).
    pad_to=N, mask_padding=False: zeros to N samples treated as audio (stats over the whole window, mask all valid)."""
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
def tdt_loop(model, enc_proj, valid_T, cap=None, eos=3, max_steps=None):
    """Greedy TDT on one utterance. enc_proj: [T,640] torch fp32. cap=None: transformers' TDT rule (no per-frame
    cap); cap=10: after 10 non-blank emissions on one frame, advance 1 (NeMo greedy). -> dict"""
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
    steps = []
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
        tok = int(torch.argmax(logits[:V]))
        d = int(torch.argmax(logits[V:]))
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
    return {"tokens": tokens, "frames": frames, "durs": durs, "steps": steps, "n_steps": n,
            "cap_hits": cap_hits, "max_run_same_frame": max_run, "eos_hit": eos_hit}


def decode_text(proc, ids):
    return proc.tokenizer.decode([i for i in ids if i != BLANK], skip_special_tokens=True)


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
