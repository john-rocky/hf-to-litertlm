#!/usr/bin/env python3
"""PyTorch reference P for Phonon-2 (CPU fp32, transformers ParakeetForTDT; venv .venv-ref).

P = stock ParakeetForTDT built from the original repo's config.json (nvidia/parakeet-tdt-0.6b-v3), loaded strictly with
the tensors of FermionResearch/Phonon-2's container: five-value records expanded to sign x {lo, hi} (fp16), intN records
to q x row scale (fp32), fp16 records as stored, everything raised to fp32. This is the computation of the author's
reference_transformers.py `container_state_dict` + `load_model`, written here instead of run (that file is only read):
  - BatchNorm `num_batches_tracked`: the writer stores the int64 counter as fp16 (inf for counts >= 65,520); a non-finite
    value becomes 0, a finite one int(), as an int64 (BatchNorm in eval never reads it)
  - the kernel-1 pointwise convs were stored squeezed [O, I]; stock wants [O, I, 1]
The container reader is the author's fermion_container.py, loaded by file path (json + numpy only, no network, no pickle);
hf/phonon2 is never the cwd nor on sys.path. Run this file's users with `python -I`. Another model's folders: PHONON_HF_DIR (under hf/, default phonon2) and PHONON_UNPACK_DIR (under its unpacked/, default model_phonon2_c4c_int6).

Arithmetic fp32 on the CPU. Feature extractor = the original repo's processor_config.json,
generation config = the original's generation_config.json (the author's reference reads both from the base repo too).

Text = the TDT decode: tokenizers.Tokenizer.decode(ids, skip_special_tokens=True) on the original's tokenizer.json, no
grouping of repeated ids (ParakeetTokenizer.decode's default groups them = the CTC rule).

tdt_loop() is a plain greedy TDT loop with the rules of transformers' ParakeetTDTGenerationMixin (token = argmax over
the vocab, duration = argmax over the duration logits, blank with duration 0 advances 1, stop at the encoder valid
length, eos stops, no per-frame symbol cap) plus an optional NeMo-style cap; gaps=True also keeps each step's top-2
logit gap of the token head and of the duration head (the token check's noise rule).

CLI (the same-computation check):
  p1_ref.py check --n 3   the first n S32 WAVs: the author's call (processor([w], ..., padding=True) -> generate ->
                          processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()) against this file's
                          features() -> gen() -> tdt_text(); ids and text must be equal. -> results/r1/ref_check.json
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K = os.path.dirname(HERE)
ORIG = os.path.join(K, "hf", "orig")
READER = os.path.join(K, "hf", os.environ.get("PHONON_HF_DIR", "phonon2"), "fermion_container.py")
CONTAINER = os.path.join(os.path.dirname(READER), "unpacked", os.environ.get("PHONON_UNPACK_DIR", "model_phonon2_c4c_int6"), "model.fermion")
BLANK = 8192
SR = 16000
_TK = None


def set_threads(n):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(ORIG)


def load_reader():
    """the author's container reader, by file path (hf/phonon2 stays off sys.path)"""
    spec = importlib.util.spec_from_file_location("fermion_container_author", READER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("fermion_container_author", None)
    spec.loader.exec_module(mod)
    return mod


def container_state_dict(path=None):
    """-> ({name: torch tensor}, index): reference_transformers.container_state_dict's computation"""
    tensors, index = load_reader().read_container(path or CONTAINER, with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v.astype(np.float32))
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                arr = arr[:, :, None]
            sd[k] = torch.from_numpy(arr)
    return sd, index


def build(which="P"):
    """-> (model, load_report dict); strict load: any missing / unexpected key or shape difference raises"""
    from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    if which != "P":
        raise SystemExit(which)
    rep = {"which": which, "container": CONTAINER}
    cfg = ParakeetTDTConfig.from_pretrained(ORIG)
    cfg.torch_dtype = torch.float32
    model = ParakeetForTDT(cfg)
    sd, index = container_state_dict()
    rep["container_records"] = len(index)
    rep["state_dict_keys"] = len(sd)
    keys = set(model.state_dict().keys())
    rep["missing"] = sorted(keys - set(sd))
    rep["unexpected"] = sorted(set(sd) - keys)
    if rep["missing"] or rep["unexpected"]:
        raise SystemExit(f"load: missing {rep['missing'][:5]} unexpected {rep['unexpected'][:5]}")
    shape_bad = [(k, tuple(sd[k].shape), tuple(v.shape)) for k, v in model.state_dict().items()
                 if tuple(sd[k].shape) != tuple(v.shape)]
    if shape_bad:
        raise SystemExit(f"shape mismatch: {shape_bad[:5]}")
    model.load_state_dict(sd, strict=True)
    model.generation_config = GenerationConfig.from_pretrained(ORIG)
    model = model.float().eval()
    rep["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    rep["n_params"] = int(sum(p.numel() for p in model.parameters()))
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
        _TK = Tokenizer.from_file(os.path.join(ORIG, "tokenizer.json"))
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


def check(n):
    """the author's call against this file's path on the first n S32 WAVs"""
    set_threads(6)
    model, rep = build("P")
    proc = processor()
    rows = read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")[:n]
    out = []
    for r in rows:
        w = load_wav(r["path"])
        inp = proc([w], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            o = model.generate(input_features=inp["input_features"], attention_mask=inp.get("attention_mask"))
        seq = getattr(o, "sequences", o)
        author_text = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
        author_ids = [int(x) for x in seq[0].tolist()]
        feats, mask = features(proc, w)
        s2, d2 = gen(model, feats, mask)
        mine = [tk for (_, tk, _) in steps_from_generate(s2, d2) if tk != BLANK]
        my_text = tdt_text(mine)
        author_nb = [i for i in author_ids[1:] if i != BLANK]
        out.append({"id": r["id"], "author_text": author_text, "text": my_text, "text_equal": author_text == my_text,
                    "ids_equal": author_nb == mine, "n_ids": len(mine)})
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    res = {"load_report": rep, "env": env_record(), "rows": out, "all_equal": all(x["text_equal"] and x["ids_equal"]
                                                                                  for x in out)}
    json.dump(res, open("results/r1/ref_check.json", "w"), indent=1, ensure_ascii=False)
    print("all_equal", res["all_equal"], flush=True)
    return 0 if res["all_equal"] else 3


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        sys.exit(check(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3))
    raise SystemExit("usage: p1_ref.py check [--n 3]")
