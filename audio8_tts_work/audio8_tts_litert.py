#!/usr/bin/env python3
"""Audio8 TTS Preview 0.6B on LiteRT -- reference host loop (CPU, ai-edge-litert Interpreter).

    python audio8_tts_litert.py --model-dir <repo> --text "Hello there." --out out.wav
    python audio8_tts_litert.py --model-dir <repo> --text "..." --ref-audio ref.wav --ref-text "exact transcript" --out out.wav
    python audio8_tts_litert.py --model-dir <repo> --register-voice ref.wav --ref-text "..." --voice-out voices/alice   # codes.npy + meta.json
    python audio8_tts_litert.py --model-dir <repo> --text "..." --voice voices/alice --out out.wav

Files (from the LiteRT repo): slow_ar_<q>.tflite (prefill_256 + decode), fast_ar_int8.tflite (step),
codec_decoder_fp16_T128.tflite / _T192.tflite (decode), codec_encoder_fp16_10s.tflite (encode, optional),
tokenizer.json. The sampler is the vendor's: top-k/top-p + temperature, RAS re-draw for repeated semantic tokens
(window 10, temperature 1.0 / top-p 0.9), exponential-race sampling, seedable.
"""
import argparse, json, os, time
import numpy as np

SEM_BEGIN, SEM_END, EOS, PAD = 151678, 155773, 151645, 151643
VOCAB, NUM_CB, CB_SIZE, DIM, KV_HEADS, HEAD_DIM, N_FAST, MAX_SEQ = 155776, 10, 4096, 896, 2, 64, 4, 2048
SR, FRAME = 44100, 2048
NEG = -1e9
ALLOWED = np.concatenate([np.arange(SEM_BEGIN, SEM_END + 1), [EOS]])


def clean(t):
    return " ".join(str(t).strip().split())


class Audio8LiteRT:
    def __init__(self, model_dir, slow="slow_ar_int8.tflite", fast="fast_ar_int8.tflite",
                 codec=("codec_decoder_fp16_T128.tflite", "codec_decoder_fp16_T192.tflite"),
                 encoder="codec_encoder_fp16_10s.tflite", threads=4, codec_ctx=128):
        from tokenizers import Tokenizer
        from ai_edge_litert.interpreter import Interpreter
        self.md = model_dir
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.it_slow = Interpreter(model_path=os.path.join(model_dir, slow), num_threads=threads)
        sigs = self.it_slow.get_signature_list()
        self.prefills = {int(k.split("_")[1]): self.it_slow.get_signature_runner(k) for k in sigs if k.startswith("prefill_")}
        self.dec = self.it_slow.get_signature_runner("decode")
        self.cache = int(self.dec.get_input_details()["mask"]["shape"][-1])
        self.kv_names = [n for n in self.dec.get_input_details() if n[:2] in ("k_", "v_")]
        self.it_fast = Interpreter(model_path=os.path.join(model_dir, fast), num_threads=threads)
        self.fast = self.it_fast.get_signature_runner("step")
        self.codecs = {}
        for f in codec:
            p = os.path.join(model_dir, f)
            if os.path.exists(p):
                r = Interpreter(model_path=p, num_threads=threads).get_signature_runner("decode")
                self.codecs[int(r.get_input_details()["codes"]["shape"][-1])] = r
        self.encoder_path = os.path.join(model_dir, encoder)
        self.codec_ctx = codec_ctx
        self.fast_masks = [np.where(np.arange(NUM_CB) <= p, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, -1) for p in range(NUM_CB)]

    # ---------------- prompt ----------------
    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def build_prompt(self, text, ref_text=None, ref_codes=None):
        target = clean(text)
        if ref_codes is None:
            parts = ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                     "<|im_start|>user\n", target, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
            row0 = sum((self._enc(p) for p in parts), [])
            prompt = np.zeros((NUM_CB + 1, len(row0)), np.int64); prompt[0] = row0
            return prompt
        rt = clean(ref_text)
        if "<|speaker:" not in rt:
            rt = "<|speaker:0|>" + rt
        prefix = sum((self._enc(p) for p in ["<|im_start|>system\n",
                     "convert the provided text to speech reference to the following:\n\nText:\n", rt, "\n\nSpeech:\n"]), [])
        suffix = sum((self._enc(p) for p in ["<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n",
                     "<|im_start|>assistant\n<|voice|>"]), [])
        L = ref_codes.shape[1]
        row0 = prefix + (ref_codes[0].astype(np.int64) + SEM_BEGIN).tolist() + suffix
        prompt = np.zeros((NUM_CB + 1, len(row0)), np.int64); prompt[0] = row0
        prompt[1:, len(prefix): len(prefix) + L] = ref_codes
        return prompt

    # ---------------- slow AR ----------------
    def _prefill(self, prompt):
        kv = {n: np.zeros((1, KV_HEADS, self.cache, HEAD_DIM), np.float32) for n in self.kv_names}
        P = prompt.shape[1]
        sizes = sorted(self.prefills)
        s = 0
        while s < P - 1:  # prefill prompt[:-1] in right-padded chunks; the last token goes through decode
            rem = P - 1 - s
            Tn = next((p for p in sizes if p >= rem), sizes[-1]); n = min(rem, Tn)
            buf = np.zeros((1, NUM_CB + 1, Tn), np.int32); buf[0, :, :n] = prompt[:, s:s + n]
            mask = np.full((1, 1, Tn, self.cache), NEG, np.float32)
            for i in range(Tn):
                mask[0, 0, i, : s + i + 1] = 0.0
            out = self.prefills[Tn](codes=buf, input_pos=np.arange(s, s + Tn, dtype=np.int32), mask=mask, **kv)
            kv = {k: out[k] for k in self.kv_names}; s += n
        return self._decode(prompt[:, -1], P - 1, kv)

    def _decode(self, column, pos, kv):
        mask = np.full((1, 1, 1, self.cache), NEG, np.float32); mask[..., : pos + 1] = 0.0
        out = self.dec(codes=column.reshape(1, NUM_CB + 1, 1).astype(np.int32), input_pos=np.array([pos], np.int32), mask=mask, **kv)
        return out["logits"][0], out["hidden"], {k: out[k] for k in self.kv_names}

    # ---------------- sampler (vendor math) ----------------
    @staticmethod
    def _processed(scores, top_k, top_p, temperature):
        order = np.argsort(-scores, kind="stable")
        s = scores[order]
        p = np.exp(s - s.max()); p /= p.sum()
        cum = np.cumsum(p)
        remove = (cum > top_p) | (np.arange(s.size) >= top_k); remove[0] = False
        out = scores.copy(); out[order[remove]] = -np.inf
        return out / max(temperature, 1e-5)

    @staticmethod
    def _sample(scores, u):
        p = np.exp(scores - scores.max()); p /= p.sum()
        return int(np.argmax(p / (-np.log(u))))

    def _sample_semantic(self, logits, previous, rng, temperature, top_p, top_k):
        normal = self._sample(self._processed(logits, top_k, top_p, temperature), rng.random(logits.size))
        high = self._sample(self._processed(logits, top_k, 0.9, 1.0), rng.random(logits.size))
        normal_id, high_id = int(ALLOWED[normal]), int(ALLOWED[high])
        if previous and normal_id in previous and SEM_BEGIN <= normal_id <= SEM_END:
            return high_id
        return normal_id

    # ---------------- fast AR ----------------
    def _fast_frame(self, hidden, semantic, rng, temperature, top_p, top_k):
        k = np.zeros((N_FAST, 1, KV_HEADS, NUM_CB, HEAD_DIM), np.float32); v = np.zeros_like(k)
        o = self.fast(hidden=hidden, token=np.zeros(1, np.int32), use_hidden=np.ones(1, np.float32),
                      pos=np.zeros(1, np.int32), mask=self.fast_masks[0], k_all=k, v_all=v)
        k, v = o["k_all"], o["v_all"]
        cur = int(np.clip(semantic - SEM_BEGIN, 0, CB_SIZE - 1)); codes = [cur]
        for p in range(1, NUM_CB):
            o = self.fast(hidden=hidden, token=np.array([cur], np.int32), use_hidden=np.zeros(1, np.float32),
                          pos=np.array([p], np.int32), mask=self.fast_masks[p], k_all=k, v_all=v)
            k, v = o["k_all"], o["v_all"]
            cur = self._sample(self._processed(o["logits"][0], top_k, top_p, temperature), rng.random(CB_SIZE))
            codes.append(cur)
        return codes

    # ---------------- generation ----------------
    def generate_codes(self, text, ref_text=None, ref_codes=None, max_new_tokens=512, temperature=0.7,
                       top_p=0.9, top_k=50, seed=42, on_frame=None):
        prompt = self.build_prompt(text, ref_text, ref_codes)
        P = prompt.shape[1]
        if P >= MAX_SEQ:
            raise ValueError(f"prompt length {P} must be smaller than {MAX_SEQ}")
        rng = np.random.default_rng(seed)
        logits, hidden, kv = self._prefill(prompt)
        previous, frames = [], []
        for step in range(min(max_new_tokens, MAX_SEQ - P)):
            sem = self._sample_semantic(logits, previous, rng, temperature, top_p, top_k)
            if sem == EOS:
                break
            cb = self._fast_frame(hidden, sem, rng, temperature, top_p, top_k)
            frames.append(cb); previous = (previous + [sem])[-10:]
            if on_frame:
                on_frame(len(frames))
            logits, hidden, kv = self._decode(np.array([sem] + cb, np.int64), P + step, kv)
        return np.array(frames, np.int64).T if frames else np.zeros((NUM_CB, 0), np.int64)

    def decode_audio(self, codes):
        """Causal codec: windows of T frames with `codec_ctx` frames of left context; each window is right-padded."""
        N = codes.shape[1]
        T = min([t for t in self.codecs if t >= N], default=max(self.codecs))
        run = self.codecs[T]
        wav = np.zeros(N * FRAME, np.float32); done = 0
        while done < N:
            s = 0 if done == 0 else max(0, done - self.codec_ctx); n = min(T, N - s)
            buf = np.zeros((1, NUM_CB, T), np.int32); buf[0, :, :n] = codes[:, s:s + n]
            out = run(codes=buf)["wav"][0, 0]
            wav[done * FRAME:(s + n) * FRAME] = out[(done - s) * FRAME: n * FRAME]; done = s + n
        return wav

    def encode_reference(self, wav44k, threads=4):
        from ai_edge_litert.interpreter import Interpreter
        run = Interpreter(model_path=self.encoder_path, num_threads=threads).get_signature_runner("encode")
        N = int(run.get_input_details()["audio"]["shape"][-1])
        a = np.asarray(wav44k, np.float32)[:N]
        buf = np.zeros((1, 1, N), np.float32); buf[0, 0, : a.size] = a
        codes = run(audio=buf)["codes"][0]
        n = int(np.ceil(a.size / FRAME))
        return codes[:, :n].astype(np.int64)


def load_audio_44k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != SR:
        g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text"); ap.add_argument("--out", default="out.wav")
    ap.add_argument("--ref-audio"); ap.add_argument("--ref-text"); ap.add_argument("--voice")
    ap.add_argument("--register-voice"); ap.add_argument("--voice-out")
    ap.add_argument("--slow", default="slow_ar_int8.tflite"); ap.add_argument("--fast", default="fast_ar_int8.tflite")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50); ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    t0 = time.time()
    m = Audio8LiteRT(a.model_dir, slow=a.slow, fast=a.fast, threads=a.threads)
    print(f"[load] {time.time()-t0:.1f}s")
    if a.register_voice:
        codes = m.encode_reference(load_audio_44k(a.register_voice))
        os.makedirs(a.voice_out, exist_ok=True)
        np.save(os.path.join(a.voice_out, "codes.npy"), codes.astype(np.uint16))
        json.dump({"reference_text": clean(a.ref_text or ""), "shape": list(codes.shape), "sample_rate": SR},
                  open(os.path.join(a.voice_out, "meta.json"), "w"), ensure_ascii=False, indent=2)
        print(f"[voice] {codes.shape[1]} frames -> {a.voice_out}")
        if not a.text:
            return
    ref_text = ref_codes = None
    if a.voice:
        ref_codes = np.load(os.path.join(a.voice, "codes.npy")).astype(np.int64)
        ref_text = json.load(open(os.path.join(a.voice, "meta.json")))["reference_text"]
    elif a.ref_audio:
        if not a.ref_text:
            raise SystemExit("--ref-text (exact transcript) is required with --ref-audio")
        ref_codes, ref_text = m.encode_reference(load_audio_44k(a.ref_audio)), a.ref_text
    t1 = time.time()
    codes = m.generate_codes(a.text, ref_text, ref_codes, a.max_new_tokens, a.temperature, a.top_p, a.top_k, a.seed)
    t2 = time.time()
    wav = m.decode_audio(codes)
    t3 = time.time()
    import soundfile as sf
    sf.write(a.out, wav, SR, subtype="PCM_16")
    dur = wav.size / SR
    print(f"[done] {codes.shape[1]} frames = {dur:.2f}s audio | AR {t2-t1:.2f}s codec {t3-t2:.2f}s | RTF {(t3-t1)/max(dur,1e-6):.2f} -> {a.out}")


if __name__ == "__main__":
    main()
