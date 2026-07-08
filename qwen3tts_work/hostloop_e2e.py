"""Qwen3-TTS on LiteRT: full host-side E2E pipeline (CompiledModel-style loop).

Graphs : talker prefill_32/decode (fp32 or int8), mtp_step, codec_decode_T64 (all tflite)
Host   : Qwen2 BPE tokenizer, embedding tables (numpy), dual-track embed aggregation,
         greedy sampling w/ suppress + repetition penalty + min-new-tokens,
         MTP 17-call inner loop, codec chunk decode.

Verifies every stage against qwen3tts_work/ref/e2e_ref.npz (PyTorch greedy reference).

  TALKER=out/qwen3tts-talker-fp32/model.tflite .venv/bin/python qwen3tts_work/hostloop_e2e.py
"""
import os, time
import numpy as np

DEFAULT_TEXT = "Hello! This is a small test of speech synthesis running on device."
TEXT = os.environ.get("TEXT", DEFAULT_TEXT)
IS_REF = TEXT == DEFAULT_TEXT  # ref-comparison asserts only apply to the ref sentence
DUMP_MTP = os.environ.get("DUMP_MTP")  # npz path: per-frame MTP inputs (calibration)
# SAMPLE=1: cb0 top-k50/T0.9 sampling (the model's production mode; greedy EOS
# is unreliable on many sentences even at fp32). Residual books stay greedy.
SAMPLE = os.environ.get("SAMPLE") == "1"
SEED = int(os.environ.get("SEED", "0"))
rng = np.random.default_rng(SEED)
SRC = "/Users/majimadaisuke/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc"
TALKER = os.environ.get("TALKER", "out/qwen3tts-talker-fp32/model.tflite")
HOSTD = "out/qwen3tts-host"
NEG = -1e9
CACHE = 1024
MAX_FRAMES = 512
EOS, THINK, THINK_BOS, THINK_EOS, PAD_ID, BOS_ID, LANG_EN = 2150, 2154, 2156, 2157, 2148, 2149, 2050
TTS_BOS, TTS_EOS, TTS_PAD = 151672, 151673, 151671
REP_PEN = 1.05

ref = np.load("qwen3tts_work/ref/e2e_ref.npz")

# ---------------- host constants ----------------
t0 = time.time()
codec_emb = np.load(f"{HOSTD}/codec_embedding.npy")            # [3072,1024]
text_emb = np.load(f"{HOSTD}/text_embedding.npy", mmap_mode="r")  # [151936,2048]
tp = np.load(f"{HOSTD}/text_projection.npz")
mtp_emb = np.load("out/qwen3tts-mtp/mtp_embeddings.npy")       # [15,2048,1024]
spk = ref["spk_emb"].astype(np.float32)                        # [1024] (enroll-time PyTorch)


def silu(x):
    return x / (1.0 + np.exp(-x))


def text_project(rows):
    h = rows @ tp["w1"].T + tp["b1"]
    return (silu(h) @ tp["w2"].T + tp["b2"]).astype(np.float32)


def embed_text(ids):
    return text_project(np.asarray(text_emb[ids], np.float32))

# ---------------- tokenize ----------------
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(SRC)
ids = tok(f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n",
          return_tensors="np")["input_ids"][0]
if IS_REF:
    assert np.array_equal(ids, ref["input_ids"][0]), "tokenizer mismatch"
    print(f"[ok] tokenizer: {len(ids)} ids identical to reference")

# ---------------- prefill construction (x-vector voice clone, streaming mode) ----------------
tts_bos, tts_eos, tts_pad = embed_text(np.array([TTS_BOS, TTS_EOS, TTS_PAD]))
codec_pre = codec_emb[[THINK, THINK_BOS, LANG_EN, THINK_EOS]]
codec_pre = np.concatenate([codec_pre, spk[None], codec_emb[[PAD_ID, BOS_ID]]], 0)  # [7,1024]
role = embed_text(ids[:3])                                     # [3,1024]
body = np.concatenate([np.repeat(tts_pad[None], 5, 0), tts_bos[None]], 0) + codec_pre[:-1]
first_text = embed_text(ids[3:4]) + codec_pre[-1:]
prefill = np.concatenate([role, body, first_text], 0)[None]    # [1,10,1024]
trailing = np.concatenate([embed_text(ids[4:-5]), tts_eos[None]], 0)  # [L,1024]

if IS_REF:
    pe = ref["prefill_embeds"]
    print(f"[ok] prefill embeds {prefill.shape} vs ref {pe.shape}: max|d| {np.abs(prefill-pe).max():.3e}")
    tr_ref = ref["trailing_text_hidden"]
    print(f"[ok] trailing {trailing.shape} vs ref {tr_ref.shape}: max|d| {np.abs(trailing-tr_ref[0]).max():.3e}")
    print(f"[ok] tts_pad vs ref: max|d| {np.abs(tts_pad-ref['tts_pad_embed'][0,0]).max():.3e}")

# ---------------- interpreters ----------------
from ai_edge_litert.interpreter import Interpreter

NTHREADS = int(os.environ.get("NTHREADS", str(os.cpu_count())))
MTP_THREADS = int(os.environ.get("MTP_THREADS", str(NTHREADS)))
it_t = Interpreter(model_path=TALKER, num_threads=NTHREADS)
pre32 = it_t.get_signature_runner("prefill_32")
dec = it_t.get_signature_runner("decode")
MTP_MODEL = os.environ.get("MTP_MODEL", "out/qwen3tts-mtp/mtp_step.tflite")
it_m = Interpreter(model_path=MTP_MODEL, num_threads=MTP_THREADS)
mtp = it_m.get_signature_runner()
# Codec: single fp32 graph, or the mixed-precision split (Part A transformer
# fp32 + Part B convnet forced fp16) when CODEC_SPLIT=1 — ~2.2x, ASR-identical.
CODEC_SPLIT = os.environ.get("CODEC_SPLIT") == "1"
if CODEC_SPLIT:
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.options import CpuOptions, Options

    def _cm(path, flags):
        return CompiledModel.from_file(
            path, options=Options(cpu_options=CpuOptions(
                num_threads=NTHREADS, xnnpack_flags=flags)))

    cmA = _cm("out/qwen3tts-codec/codec_partA_T64.tflite", None)
    cmB = _cm("out/qwen3tts-codec/codec_partB_T64.tflite", 4)  # 4 = FORCE_FP16
    _HN = 512 * 64  # partA hidden [1,512,64]

    def codec_decode(buf):
        ia = cmA.create_input_buffers(0); oa = cmA.create_output_buffers(0)
        ia[0].write(buf.reshape(-1).astype(np.int32))
        cmA.run_by_index(0, ia, oa)
        ib = cmB.create_input_buffers(0); ob = cmB.create_output_buffers(0)
        ib[0].write(np.array(oa[0].read(_HN, np.float32)))
        cmB.run_by_index(0, ib, ob)
        return np.array(ob[0].read(64 * 1920, np.float32))
else:
    it_c = Interpreter(model_path="out/qwen3tts-codec/codec_decode_T64.tflite", num_threads=NTHREADS)
    codec = it_c.get_signature_runner()

    def codec_decode(buf):
        return list(codec(args_0=buf).values())[0][0, 0]
kv_names = [n for n in dec.get_input_details() if n.startswith("kv_cache")]
print(f"[ok] interpreters loaded ({time.time()-t0:.1f}s incl tables), threads={NTHREADS}/mtp{MTP_THREADS}")

# ---------------- talker helpers ----------------
kv = {n: np.zeros(dec.get_input_details()[n]["shape"], np.float32) for n in kv_names}


def talker_prefill(kv, embeds):
    P = embeds.shape[1]
    buf = np.zeros((1, 32, 1024), np.float32)
    buf[:, :P] = embeds
    mask = np.full((1, 1, 32, CACHE), NEG, np.float32)
    for i in range(32):
        mask[0, 0, i, : min(i, P - 1) + 1] = 0.0
    out = pre32(embeddings=buf, input_pos=np.arange(32, dtype=np.int32), mask=mask, **kv)
    return out


def talker_decode(kv, embed, pos):
    mask = np.full((1, 1, 1, CACHE), NEG, np.float32)
    mask[..., : pos + 1] = 0.0
    out = dec(embeddings=embed.reshape(1, 1, 1024).astype(np.float32),
              input_pos=np.array([pos], np.int32), mask=mask, **kv)
    logits = out.pop("logits")[0, 0]
    return logits[:3072], logits[3072:], out


MTP_L, MTP_KV, MTP_HD, MTP_CACHE = 5, 8, 128, 17
# Folded single-invoke graph (export_mtp_folded.py) has 3 inputs
# (past_hidden, cb0_embed, noise); the step graph has 5.
MTP_FOLDED = len(mtp.get_input_details()) == 3
MTP_ZERO_NOISE = np.zeros((15, 2048), np.float32)
# A quantized (static int8/int16) folded graph exposes integer I/O; quantize
# the fp32 feeds with the graph's input scale/zero and read codes (int) direct.
_mtp_in = mtp.get_input_details() if MTP_FOLDED else {}


def _q(name, arr):
    d = _mtp_in[name]
    dt = d["dtype"]
    if dt == np.float32:
        return arr.astype(np.float32)
    s, z = d["quantization"]
    return np.clip(np.round(arr / s) + z, np.iinfo(dt).min,
                   np.iinfo(dt).max).astype(dt)


def mtp_frame(hidden, cb0):
    if MTP_FOLDED:
        out = mtp(args_0=_q("args_0", hidden.reshape(1, 1, 1024)),
                  args_1=_q("args_1", codec_emb[cb0].reshape(1, 1, 1024)),
                  args_2=_q("args_2", MTP_ZERO_NOISE))
        return [int(x) for x in out["output_0"]]
    k_all = np.zeros((MTP_L, 1, MTP_KV, MTP_CACHE, MTP_HD), np.float32)
    v_all = np.zeros_like(k_all)
    codes = []
    feeds = [hidden, codec_emb[cb0]]
    for t in range(16):
        embed = (feeds[t] if t < 2 else mtp_emb[t - 2][codes[-1]]).reshape(1, 1, 1024)
        mask = np.where(np.arange(MTP_CACHE) <= t, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, -1)
        out = mtp(args_0=embed.astype(np.float32), args_1=np.array([t], np.int32),
                  args_2=mask, args_3=k_all, args_4=v_all)
        k_all, v_all = out["output_1"], out["output_2"]
        if t >= 1:
            codes.append(int(out["output_0"][t - 1].argmax()))
    return codes  # 15 codes


# ---------------- decode loop ----------------
suppress = np.zeros(3072, np.float32)
suppress[2048:] = NEG
suppress[EOS] = 0.0

t_pre = time.time()
kv = talker_prefill(kv, prefill)
pos = prefill.shape[1] - 1
cb0_logits, hidden, kv = talker_decode(kv, prefill[0, -1], pos)
t_pre = time.time() - t_pre

frames = []
gen_hist = []
mtp_inputs = []  # (hidden, cb0) per frame, for DUMP_MTP calibration capture
t_talker = t_mtp = 0.0
t_loop = time.time()
while len(frames) < MAX_FRAMES:
    lg = cb0_logits + suppress
    if len(frames) < 2:
        lg[EOS] = NEG  # min_new_tokens=2
    for gid in set(gen_hist):
        lg[gid] = lg[gid] / REP_PEN if lg[gid] > 0 else lg[gid] * REP_PEN
    if SAMPLE:
        top = np.argpartition(lg, -50)[-50:]
        p = np.exp((lg[top] - lg[top].max()) / 0.9)
        p /= p.sum()
        cb0 = int(rng.choice(top, p=p))
    else:
        cb0 = int(lg.argmax())
    gen_hist.append(cb0)
    if cb0 == EOS:
        break
    tm = time.time()
    sub = mtp_frame(hidden, cb0)
    t_mtp += time.time() - tm
    if DUMP_MTP:
        mtp_inputs.append((hidden.copy(), cb0))
    frames.append([cb0] + sub)

    emb = codec_emb[cb0] + mtp_emb[np.arange(15), sub].sum(0)
    step = len(frames) - 1
    emb = emb + (trailing[step] if step < len(trailing) else tts_pad)
    pos += 1
    tm = time.time()
    cb0_logits, hidden, kv = talker_decode(kv, emb, pos)
    t_talker += time.time() - tm
t_loop = time.time() - t_loop

codes = np.array(frames, np.int32)  # [T,16]
print(f"[gen] {len(codes)} frames  prefill {t_pre*1e3:.0f}ms  talker {t_talker:.2f}s  mtp {t_mtp:.2f}s  loop {t_loop:.2f}s")

if DUMP_MTP:
    np.savez(DUMP_MTP,
             hiddens=np.stack([h for h, _ in mtp_inputs]).astype(np.float32),
             cb0s=np.array([c for _, c in mtp_inputs], np.int32))
    print(f"[dump] {len(mtp_inputs)} MTP input frames -> {DUMP_MTP}")

# ---------------- compare codes ----------------
rc = ref["codes"]
Tm = min(len(codes), len(rc))
cb0_match = (codes[:Tm, 0] == rc[:Tm, 0]).mean()
all_match = (codes[:Tm] == rc[:Tm]).mean()
print(f"[cmp] frames {len(codes)} vs ref {len(rc)}; cb0 match {cb0_match:.2%}, all-16 match {all_match:.2%}")

# ---------------- codec decode ----------------
t_cod = time.time()
T = len(codes)
wav_chunks = []
CH = 64
CTX = 25
i = 0
while i < T:
    ctx = min(CTX, i)
    # Window = left context + new frames must fit the fixed-T graph.
    j = min(i + CH - ctx, T)
    chunk = codes[i - ctx: j]
    buf = np.zeros((1, 16, CH), np.int32)
    buf[0, :, : len(chunk)] = chunk.T
    wav = codec_decode(buf)
    wav_chunks.append(wav[ctx * 1920: len(chunk) * 1920])
    i = j
wav = np.concatenate(wav_chunks)
t_cod = time.time() - t_cod

dur = len(wav) / 24000
wall = t_pre + t_loop + t_cod
print(f"[time] codec {t_cod:.2f}s | total {wall:.2f}s for {dur:.2f}s audio -> RTF {wall/dur:.2f}")

rw = ref["wav"]
n = min(len(wav), len(rw))
c = float(np.corrcoef(wav[:n], rw[:n])[0, 1])
print(f"[cmp] wav corr vs ref: {c:.6f} (len {len(wav)} vs {len(rw)})")

import soundfile as sf  # noqa: E402

sf.write("qwen3tts_work/ref/litert_e2e.wav", wav, 24000)
print("saved qwen3tts_work/ref/litert_e2e.wav")
print("HOSTLOOP_DONE")
