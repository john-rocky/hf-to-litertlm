"""PyTorch fp32 oracle for Audio8-TTS-Preview-0.6b (transformers 4.57.x, vendor's version family).

Runs the vendor remote code unmodified (trust_remote_code) on CPU fp32, with a seeded
torch.Generator, and records everything the LiteRT gate needs:
  fixtures/ref_<lang>_44k.wav, ref_<lang>_codes.npy   reference audio (44.1 kHz) + codec codes [10,T]
  out/oracle/<case>.npz   prompt [11,P], codes [10,T], semantic [T], slow_logits [T+1,4097] (per step,
                          semantic range then eos), fast_hidden [T+1,896], fast_logits_first8 [8,9,4096],
                          tf_logits [P+T,4097] (single teacher-forced forward on prompt+generated)
  out/oracle/<case>.wav   44.1 kHz decoded waveform

  ~/parakeet-env/bin/python3 audio8_tts_work/oracle_ref.py [case_id ...]
"""
import os, sys, json, time
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
ORACLE_OUT = os.path.join(C.OUT, "oracle")
os.makedirs(ORACLE_OUT, exist_ok=True)
os.makedirs(C.FIX, exist_ok=True)

import transformers
from transformers import AutoModel, AutoProcessor
print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
assert transformers.__version__.startswith("4.57"), "oracle must run on the vendor's transformers 4.57.x"

processor = AutoProcessor.from_pretrained(C.SNAP, trust_remote_code=True)
model = AutoModel.from_pretrained(C.SNAP, trust_remote_code=True, dtype=torch.float32).eval()
# rope table is a non-persistent buffer computed in __init__: assert it is alive (tf 5.x meta-load trap)
assert model.freqs_cis.abs().max().item() > 0 and model.fast_freqs_cis.abs().max().item() > 0
print("model loaded, params", sum(p.numel() for p in model.parameters()), flush=True)

# ---------------- references: 16 kHz -> 44.1 kHz -> codec codes ----------------
ref_codes = {}
for key, r in C.REFS.items():
    wav44_path = os.path.join(C.FIX, f"ref_{key}_44k.wav")
    codes_path = os.path.join(C.FIX, f"ref_{key}_codes.npy")
    if not os.path.exists(wav44_path):
        a, sr = sf.read(r["wav16"], dtype="float32", always_2d=True)
        a = a.mean(1)
        assert sr == 16000
        a44 = resample_poly(a, 441, 160).astype(np.float32)
        sf.write(wav44_path, a44, C.SR, subtype="PCM_16")
    a44, sr = sf.read(wav44_path, dtype="float32")
    assert sr == C.SR
    if not os.path.exists(codes_path):
        with torch.inference_mode():
            codes, lens = model.encode_audio(torch.tensor(a44)[None, None], torch.tensor([a44.shape[0]]))
        codes = codes[0, :, : int(lens[0])].numpy().astype(np.int64)
        np.save(codes_path, codes)
    ref_codes[key] = np.load(codes_path)
    print(f"ref {key}: {a44.shape[0]/C.SR:.2f}s -> codes {ref_codes[key].shape}", flush=True)

# ---------------- hooks: record per-step slow logits / fast hidden / early fast logits ----------------
REC = {}
_orig_slow = model._slow_step
_orig_fast = model._fast_step


def rec_slow(input_ids, cache_position, position_ids, attention_mask):
    logits, fast_hidden = _orig_slow(input_ids, cache_position, position_ids, attention_mask)
    sl = torch.cat([logits[0, C.SEM_BEGIN:C.SEM_END + 1], logits[0, C.EOS:C.EOS + 1]]).numpy().astype(np.float32)
    REC["slow_logits"].append(sl)
    REC["fast_hidden"].append(fast_hidden[0, 0].numpy().astype(np.float32))
    return logits, fast_hidden


def rec_fast(hidden, position):
    scores = _orig_fast(hidden, position)
    if position >= 1 and len(REC["fast_logits"]) < 8 * 9:
        REC["fast_logits"].append(scores[0].numpy().astype(np.float32))
    return scores


model._slow_step = rec_slow
model._fast_step = rec_fast

wanted = set(sys.argv[1:])
for case_id, lang, ref_key, text, seed in C.cases():
    if wanted and case_id not in wanted:
        continue
    npz_path = os.path.join(ORACLE_OUT, f"{case_id}.npz")
    if os.path.exists(npz_path) and not wanted:
        print("skip", case_id); continue
    REC.clear(); REC.update(slow_logits=[], fast_hidden=[], fast_logits=[])
    if ref_key is None:
        inputs = processor(text=[text], return_tensors="pt")
    else:
        inputs = processor(text=[text], reference_text=[C.REFS[ref_key]["text"]],
                           reference_codes=[ref_codes[ref_key]], return_tensors="pt")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=C.MAX_NEW_TOKENS, temperature=C.TEMPERATURE,
                             top_p=C.TOP_P, top_k=C.TOP_K, do_sample=True,
                             return_dict_in_generate=True, generator=g)
        t_gen = time.time() - t0
        codes = out.codes[0].numpy().astype(np.int64)      # [10,T]
        T = int(out.code_lengths[0])
        codes = codes[:, :T]
        wav, wl = model.decode_audio(out.codes)
        wav = wav[0, : int(wl[0])].numpy().astype(np.float32)
        # rebuild the exact prompt the model saw (same code path as generate)
        prompt, pmask = model._prepare_prompt(**{k: v for k, v in inputs.items()})
        prompt = prompt[0].numpy().astype(np.int64)          # [11,P]
        assert pmask.all(), "batch-1 prompt must be unpadded"
        semantic = codes[0] + C.SEM_BEGIN
        full = np.concatenate([prompt, np.concatenate([semantic[None], codes], 0)], 1)  # [11,P+T]
        # teacher-forced single forward (no cache) -> logits for every position
        for layer in list(model.layers) + list(model.fast_layers):
            layer.attention.kv_cache = None  # generate() leaves the caches attached
        tf = model(input_ids=torch.tensor(full)[None], return_dict=True).logits[0]
        tf_logits = torch.cat([tf[:, C.SEM_BEGIN:C.SEM_END + 1], tf[:, C.EOS:C.EOS + 1]], 1).numpy().astype(np.float32)
    finished = bool(out.finished[0])
    slow_logits = np.stack(REC["slow_logits"])[: T + 1]
    fast_hidden = np.stack(REC["fast_hidden"])[: T + 1]
    fl = np.stack(REC["fast_logits"]) if REC["fast_logits"] else np.zeros((0, 4096), np.float32)
    fl = fl[: (len(fl) // 9) * 9].reshape(-1, 9, 4096)
    np.savez(npz_path, prompt=prompt, codes=codes, semantic=semantic, slow_logits=slow_logits,
             fast_hidden=fast_hidden, fast_logits_first8=fl, tf_logits=tf_logits, wav=wav,
             seed=seed, finished=finished, text=text, lang=lang, ref=ref_key or "")
    sf.write(os.path.join(ORACLE_OUT, f"{case_id}.wav"), wav, C.SR, subtype="PCM_16")
    print(f"{case_id}: prompt {prompt.shape[1]} frames {T} ({T*C.FRAME/C.SR:.2f}s) finished={finished} "
          f"gen {t_gen:.1f}s  tf-vs-step max|d| {np.abs(tf_logits[prompt.shape[1]-1:][:T+1]-slow_logits).max():.3e}", flush=True)
print("ORACLE_DONE")
