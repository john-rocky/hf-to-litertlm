"""Audio8-TTS on LiteRT: host-side end-to-end loop (slow AR prefill/decode + fast AR + codec decoder).

Mirrors the vendor's generate() call-for-call, including its sampler (top-k/top-p, RAS repetition
re-draw, exponential-race Gumbel sampling) with the SAME torch.Generator draw order, so an fp32 graph
reproduces the oracle's code sequence and quantized graphs can be compared frame-by-frame.

  SLOW=out/slow/slow_fp32_c2048.tflite FAST=out/fast/fast_fp32.tflite CODEC=out/codec/codec_decoder_fp32_T64.tflite \
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/hostloop_e2e.py [case_id ...]
Outputs: out/e2e/<tag>/<case>.wav + .npz (codes, semantic, timings), summary line per case.
"""
import os, sys, time, json
import numpy as np
import torch
import soundfile as sf
from tokenizers import Tokenizer
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

W = C.WORK
SLOW = os.environ.get("SLOW", f"{C.OUT}/slow/slow_fp32_c2048.tflite")
FAST = os.environ.get("FAST", f"{C.OUT}/fast/fast_fp32.tflite")
CODEC = os.environ.get("CODEC", f"{C.OUT}/codec/codec_decoder_fp32_T64.tflite")
TAG = os.environ.get("TAG", "fp32")
NTHREADS = int(os.environ.get("NTHREADS", "4"))
CODEC_CTX = int(os.environ.get("CODEC_CTX", "32"))   # left-context frames re-decoded per codec window
NEG = -1e9
OUTD = f"{C.OUT}/e2e/{TAG}"; os.makedirs(OUTD, exist_ok=True)

# ---------------- prompt (mirror of processing_arktts.py + _prepare_prompt) ----------------
tok = Tokenizer.from_file(os.path.join(C.SNAP, "tokenizer.json"))
def enc(text):
    return tok.encode(text, add_special_tokens=False).ids
def clean(t):
    return " ".join(str(t).strip().split())
def build_prompt(text, ref_text=None, ref_codes=None):
    target = clean(text)
    if ref_codes is None:
        parts = ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                 "<|im_start|>user\n", target, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
        row0 = sum((enc(p) for p in parts), [])
        prompt = np.zeros((C.NUM_CB + 1, len(row0)), np.int64); prompt[0] = row0
        return prompt
    rt = clean(ref_text)
    if "<|speaker:" not in rt:
        rt = "<|speaker:0|>" + rt
    prefix = sum((enc(p) for p in ["<|im_start|>system\n",
                 "convert the provided text to speech reference to the following:\n\nText:\n", rt, "\n\nSpeech:\n"]), [])
    suffix = sum((enc(p) for p in ["<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n",
                 "<|im_start|>assistant\n<|voice|>"]), [])
    L = ref_codes.shape[1]
    row0 = prefix + (ref_codes[0] + C.SEM_BEGIN).tolist() + suffix
    prompt = np.zeros((C.NUM_CB + 1, len(row0)), np.int64); prompt[0] = row0
    prompt[1:, len(prefix): len(prefix) + L] = ref_codes
    return prompt

# ---------------- interpreters ----------------
t0 = time.time()
it_s = Interpreter(model_path=SLOW, num_threads=NTHREADS)
sigs = it_s.get_signature_list()
prefill_sigs = sorted([int(k.split("_")[1]) for k in sigs if k.startswith("prefill_")])
pre = {P: it_s.get_signature_runner(f"prefill_{P}") for P in prefill_sigs}
dec = it_s.get_signature_runner("decode")
CACHE = dec.get_input_details()["mask"]["shape"][-1]
KV = [n for n in dec.get_input_details() if n[:2] in ("k_", "v_")]
it_f = Interpreter(model_path=FAST, num_threads=NTHREADS)
fast = it_f.get_signature_runner("step")
it_c = Interpreter(model_path=CODEC, num_threads=NTHREADS)
cod = it_c.get_signature_runner("decode")
CODEC_T = cod.get_input_details()["codes"]["shape"][-1]
print(f"[load] slow {os.path.basename(SLOW)} prefill {prefill_sigs} cache {CACHE} | fast {os.path.basename(FAST)} | "
      f"codec {os.path.basename(CODEC)} T{CODEC_T} ctx {CODEC_CTX} | {NTHREADS} threads | {time.time()-t0:.1f}s", flush=True)

FAST_MASKS = [np.where(np.arange(C.NUM_CB) <= p, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, -1) for p in range(C.NUM_CB)]
ALLOWED = np.concatenate([np.arange(C.SEM_BEGIN, C.SEM_END + 1), [C.EOS]])


def slow_prefill(prompt):
    """Prefill prompt[:, :-1] in right-padded chunks, then decode the last prompt token -> (logits, hidden, kv)."""
    kv = {n: np.zeros((1, C.KV_HEADS, CACHE, C.HEAD_DIM), np.float32) for n in KV}
    P = prompt.shape[1]
    s = 0
    while s < P - 1:
        rem = P - 1 - s
        Tn = next((p for p in prefill_sigs if p >= rem), prefill_sigs[-1])
        n = min(rem, Tn)
        buf = np.zeros((1, C.NUM_CB + 1, Tn), np.int32); buf[0, :, :n] = prompt[:, s:s + n]
        mask = np.full((1, 1, Tn, CACHE), NEG, np.float32)
        for i in range(Tn):
            mask[0, 0, i, : s + i + 1] = 0.0
        out = pre[Tn](codes=buf, input_pos=np.arange(s, s + Tn, dtype=np.int32), mask=mask, **kv)
        kv = {n_: out[n_] for n_ in KV}
        s += n
    return slow_decode(prompt[:, -1], P - 1, kv)


def slow_decode(column, pos, kv):
    mask = np.full((1, 1, 1, CACHE), NEG, np.float32); mask[..., : pos + 1] = 0.0
    out = dec(codes=column.reshape(1, C.NUM_CB + 1, 1).astype(np.int32), input_pos=np.array([pos], np.int32), mask=mask, **kv)
    return out["logits"][0], out["hidden"], {n: out[n] for n in KV}


def fast_frame(hidden, semantic, g, temperature, top_p, top_k):
    k = np.zeros((C.N_FAST, 1, C.KV_HEADS, C.NUM_CB, C.HEAD_DIM), np.float32); v = np.zeros_like(k)
    o = fast(hidden=hidden, token=np.zeros(1, np.int32), use_hidden=np.ones(1, np.float32), pos=np.zeros(1, np.int32),
             mask=FAST_MASKS[0], k_all=k, v_all=v)
    k, v = o["k_all"], o["v_all"]
    cur = int(np.clip(semantic - C.SEM_BEGIN, 0, C.CB_SIZE - 1))
    codes = [cur]
    for p in range(1, C.NUM_CB):
        o = fast(hidden=hidden, token=np.array([cur], np.int32), use_hidden=np.zeros(1, np.float32), pos=np.array([p], np.int32),
                 mask=FAST_MASKS[p], k_all=k, v_all=v)
        k, v = o["k_all"], o["v_all"]
        scores = processed(torch.tensor(o["logits"][0]), top_k, top_p, temperature)
        cur = sample(scores, torch.rand((1, C.CB_SIZE), generator=g)[0])
        codes.append(cur)
    return codes


# ---------------- sampler (vendor math, torch ops, same generator draw order) ----------------
def processed(scores, top_k, top_p, temperature):
    sorted_scores, sorted_idx = torch.sort(scores, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
    pos = torch.arange(scores.shape[-1])
    remove_sorted = (cum > torch.tensor(top_p, dtype=cum.dtype)) | (pos >= top_k)
    remove_sorted[0] = False
    remove = torch.zeros_like(remove_sorted).scatter(0, sorted_idx, remove_sorted)
    scores = scores.masked_fill(remove, float("-inf"))
    return scores / torch.tensor(temperature).clamp_min(1e-5)


def sample(scores, u):
    prob = torch.softmax(scores, dim=-1)
    return int(torch.argmax(prob / (-torch.log(u))))


def sample_semantic(logits4097, previous, g, temperature, top_p, top_k):
    lg = torch.tensor(logits4097)
    normal_scores = processed(lg, top_k, top_p, temperature)
    u = torch.rand((1, C.VOCAB), generator=g)[0, ALLOWED]      # the vendor draws over the full vocab
    normal = sample(normal_scores, u)
    high_scores = processed(lg, top_k, C.RAS_TOP_P, C.RAS_TEMP)
    u2 = torch.rand((1, C.VOCAB), generator=g)[0, ALLOWED]
    high = sample(high_scores, u2)
    normal_id, high_id = int(ALLOWED[normal]), int(ALLOWED[high])
    if previous is None:
        return normal_id
    if normal_id in previous and C.SEM_BEGIN <= normal_id <= C.SEM_END:
        return high_id
    return normal_id


# ---------------- codec: windowed decode with left context (causal graph) ----------------
def codec_decode(codes):
    N = codes.shape[1]
    wav = np.zeros(N * C.FRAME, np.float32)
    done = 0; calls = 0
    while done < N:
        s = 0 if done == 0 else max(0, done - CODEC_CTX)
        n = min(CODEC_T, N - s)
        buf = np.zeros((1, C.NUM_CB, CODEC_T), np.int32); buf[0, :, :n] = codes[:, s:s + n]
        out = cod(codes=buf)["wav"][0, 0]; calls += 1
        wav[done * C.FRAME:(s + n) * C.FRAME] = out[(done - s) * C.FRAME: n * C.FRAME]
        done = s + n
    return wav, calls


# ---------------- run cases ----------------
wanted = set(sys.argv[1:])
summary = []
for case_id, lang, ref_key, text, seed in C.cases():
    if wanted and case_id not in wanted:
        continue
    ref_codes = np.load(f"{C.FIX}/ref_{ref_key}_codes.npy") if ref_key else None
    prompt = build_prompt(text, C.REFS[ref_key]["text"] if ref_key else None, ref_codes)
    orc = np.load(f"{C.OUT}/oracle/{case_id}.npz")
    assert prompt.shape == orc["prompt"].shape and (prompt == orc["prompt"]).all(), f"prompt mismatch {case_id}"
    P = prompt.shape[1]
    g = torch.Generator().manual_seed(int(seed))
    max_new = min(C.MAX_NEW_TOKENS, C.MAX_SEQ - P)
    t_pre = time.perf_counter()
    logits, hidden, kv = slow_prefill(prompt)
    t_pre = time.perf_counter() - t_pre
    previous = None; frames = []; sems = []
    t_slow = t_fast = 0.0; t_loop = time.perf_counter()
    for step in range(max_new):
        sem = sample_semantic(logits, previous, g, C.TEMPERATURE, C.TOP_P, C.TOP_K)
        if sem == C.EOS:
            break
        t1 = time.perf_counter(); cb = fast_frame(hidden, sem, g, C.TEMPERATURE, C.TOP_P, C.TOP_K); t_fast += time.perf_counter() - t1
        frames.append(cb); sems.append(sem)
        if previous is None:
            previous = [0] * C.RAS_WINDOW
        else:
            previous = previous[1:] + [sem]
        col = np.array([sem] + cb, np.int64)
        t1 = time.perf_counter(); logits, hidden, kv = slow_decode(col, P + step, kv); t_slow += time.perf_counter() - t1
    t_loop = time.perf_counter() - t_loop
    codes = np.array(frames, np.int64).T if frames else np.zeros((C.NUM_CB, 0), np.int64)   # [10,T]
    T = codes.shape[1]
    t1 = time.perf_counter(); wav, ccalls = codec_decode(codes) if T else (np.zeros(0, np.float32), 0); t_codec = time.perf_counter() - t1
    oc = orc["codes"]; To = oc.shape[1]; n = min(T, To)
    sem_match = int((np.array(sems[:n]) == orc["semantic"][:n]).all()) if n else 0
    first_div = next((i for i in range(n) if not (codes[:, i] == oc[:, i]).all()), n)
    audio_s = T * C.FRAME / C.SR
    total = t_pre + t_loop + t_codec
    line = (f"{case_id}: frames {T} (oracle {To}) first-divergence {first_div}/{n} | prefill {t_pre*1e3:.0f}ms "
            f"slow {t_slow/max(T,1)*1e3:.1f}ms/f fast {t_fast/max(T,1)*1e3:.1f}ms/f codec {t_codec*1e3:.0f}ms ({ccalls} calls) "
            f"| audio {audio_s:.2f}s total {total:.2f}s RTF {total/max(audio_s,1e-6):.2f}")
    print(line, flush=True)
    sf.write(f"{OUTD}/{case_id}.wav", wav, C.SR, subtype="PCM_16")
    np.savez(f"{OUTD}/{case_id}.npz", codes=codes, semantic=np.array(sems), prompt=prompt, t_pre=t_pre, t_slow=t_slow,
             t_fast=t_fast, t_codec=t_codec, t_loop=t_loop, first_div=first_div, oracle_frames=To)
    summary.append(dict(case=case_id, frames=T, oracle_frames=To, first_div=first_div, t_pre=t_pre, t_slow=t_slow,
                        t_fast=t_fast, t_codec=t_codec, t_loop=t_loop, audio_s=audio_s, rtf=total / max(audio_s, 1e-6)))
json.dump(dict(slow=SLOW, fast=FAST, codec=CODEC, threads=NTHREADS, codec_ctx=CODEC_CTX, cases=summary),
          open(f"{OUTD}/summary.json", "w"), indent=1)
print("E2E_DONE")
