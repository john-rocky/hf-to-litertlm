#!/usr/bin/env python3
"""r2 LiteRT driver (venv .venv-q: ai-edge-litert 2.3.0): mel30 -> enc30_<form> -> dec loop, the host rules of
CONTRACT.md. mel30 and dec always run on the CPU (Interpreter + XNNPACK); the encoder on the CPU (Interpreter, `cpu`)
or on the Mac GPU (CompiledModel, Metal accelerator of the wheel, GPU|CPU, `gpu32` = enforce_f32, `gpu16` = default).

Host rules: an utterance is cut into consecutive 479,840-sample windows from the start; a window shorter than 320
samples is dropped; each window goes zero-padded into audio [1, 480000] with n_valid = its sample count; the loop runs
over the L3 = ceil((n_valid // 160) / 8) valid encoder frames: predictor primed with blank 8192 (zero LSTM state);
per step joint(enc[t], dec_out) -> token = argmax of the first 8,193 logits, duration = argmax of the last 5 (values
0..4); blank with duration 0 advances 1; a non-blank token runs the predictor (its state carries on) and eos (3) stops
the window; 10 non-blank tokens on one frame advance 1 (the per-frame cap); the window ends at t >= L3 or after
10 x 375 steps. Window texts are decoded separately and joined with a space (r1's mode ii), so the driver stores the
token ids per window and the text is made afterwards by r2_text.py (.venv-ref).

CLI (resumable P1 run; one JSON line per utterance, one pipeline for all the sets given):
  r2_drive.py run --form int4x --backend cpu --threads 3 --sets ls[,de,fr,ru] [--outdir results/r2]
                  [--deadline EPOCH] [--limit N]      -> <outdir>/p1_<form>_<backend>_<set>.jsonl (+ .meta.json, .DONE)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_AUDIO = 480000
W30 = 479840
MIN_SAMPLES = 320
HOP = 160
BLANK = 8192
VOCAB = 8193
EOS = 3
CAP = 10
T_ENC = 375
D = 640
FILES = {"mel": "out/r2/mel30.tflite", "dec": "out/r2/dec.tflite",
         "fp32": "out/r2/enc30_fp32.tflite", "int4x": "out/r2/enc30_int4x.tflite",
         "fp16w": "out/r2/enc30_fp16w.tflite", "r1": "out/r2/enc30_r1.tflite",
         "fp32_raw": "out/r2/enc30_fp32_raw.tflite"}
SETS = {"ls": ("data/sets/librispeech-test-clean/manifest.jsonl", "en"),
        "s4": ("data/sets/librispeech-test-clean/manifest-S4.jsonl", "en"),
        "s32": ("data/sets/librispeech-test-clean/manifest-S32.jsonl", "en"),
        "de": ("data/sets/fleurs-de/manifest.jsonl", "de"),
        "fr": ("data/sets/fleurs-fr/manifest.jsonl", "fr"),
        "ru": ("data/sets/fleurs-ru/manifest.jsonl", "ru")}


def windows(n_samples, size=W30, min_samples=MIN_SAMPLES):
    out = []
    for s in range(0, max(n_samples, 1), size):
        n = min(size, n_samples - s)
        if n >= min_samples:
            out.append((s, n))
    return out


def enc_valid_frames(n_valid):
    return ((n_valid // HOP) + 7) // 8


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000, (path, sr)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


class CpuGraph:
    """Interpreter + XNNPACK (the default delegate of ai-edge-litert's Python Interpreter); no weight cache."""

    def __init__(self, path, threads):
        from ai_edge_litert.interpreter import Interpreter
        t0 = time.perf_counter()
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.runners = {k: self.it.get_signature_runner(k) for k in self.it.get_signature_list()}
        self.load_s = time.perf_counter() - t0
        self.info = {"path": path, "backend": "cpu", "threads": threads, "load_s": round(self.load_s, 2),
                     "signatures": self.it.get_signature_list()}

    def __call__(self, key, **kw):
        return self.runners[key](**kw)


class GpuGraph:
    """CompiledModel on the Mac GPU (the wheel's Metal accelerator); ops the GPU refuses fall back to the CPU."""

    def __init__(self, path, f32, key="serving_default"):
        from ai_edge_litert.compiled_model import CompiledModel, GpuOptions, HardwareAccelerator, Options
        opts = Options(HardwareAccelerator.GPU | HardwareAccelerator.CPU, gpu_options=GpuOptions(enforce_f32=f32))
        t0 = time.perf_counter()
        self.m = CompiledModel.from_file(path, options=opts)
        self.load_s = time.perf_counter() - t0
        try:
            self.fully = bool(self.m.is_fully_accelerated())
        except Exception as e:  # noqa: BLE001
            self.fully = f"unavailable: {type(e).__name__}: {e}"
        self.key = key
        sigs = self.m.get_signature_list()
        self.in_names = list(sigs[key]["inputs"])
        self.out_names = list(sigs[key]["outputs"])
        self.inb = {n: self.m.create_input_buffer_by_name(key, n) for n in self.in_names}
        self.outb = {n: self.m.create_output_buffer_by_name(key, n) for n in self.out_names}
        det = self.m.get_output_tensor_details(key)
        known = {"enc": (1, T_ENC, D)}
        self.out_shapes = {}
        for n in self.out_names:
            try:
                self.out_shapes[n] = tuple(int(x) for x in det[n]["shape"])
            except Exception:  # noqa: BLE001
                self.out_shapes[n] = known[n]
        self.info = {"path": path, "backend": "gpu32" if f32 else "gpu16", "enforce_f32": f32,
                     "compile_s": round(self.load_s, 2), "fully_accelerated": self.fully, "signatures": sigs,
                     "out_shapes": {k: list(v) for k, v in self.out_shapes.items()}}

    def __call__(self, key, **kw):
        assert key == self.key
        for n, v in kw.items():
            self.inb[n].write(np.ascontiguousarray(v))
        self.m.run_by_name(key, self.inb, self.outb)
        return {n: self.outb[n].read(int(np.prod(s)), np.float32).reshape(s) for n, s in self.out_shapes.items()}


class Pipeline:
    def __init__(self, form, backend="cpu", threads=4, dec_threads=1, mel_threads=None):
        self.mel = CpuGraph(FILES["mel"], mel_threads or threads)
        if backend == "cpu":
            self.enc = CpuGraph(FILES[form], threads)
        elif backend in ("gpu32", "gpu16"):
            self.enc = GpuGraph(FILES[form], backend == "gpu32")
        else:
            raise SystemExit(backend)
        self.dec = CpuGraph(FILES["dec"], dec_threads)
        self.pred = self.dec.runners["predict"]
        self.joint = self.dec.runners["joint"]
        self.form, self.backend, self.threads = form, backend, threads

    def features(self, audio):
        n = len(audio)
        x = np.zeros((1, N_AUDIO), np.float32)
        x[0, :n] = audio
        return self.mel("serving_default", audio=x, n_valid=np.array([n], np.int32))

    def encode(self, features, mel_mask):
        # a copy: the next call of the same interpreter must not change an output already handed out
        return np.array(self.enc("serving_default", features=features, mel_mask=mel_mask)["enc"][0], copy=True)

    def loop(self, enc, valid_t, keep_steps=False):
        h = np.zeros((2, 1, D), np.float32)
        c = np.zeros((2, 1, D), np.float32)
        o = self.pred(token=np.array([[BLANK]], np.int32), h=h, c=c)
        dec_out, h, c = o["dec_out"], o["h_out"], o["c_out"]
        t = n = on = max_run = cap_hits = 0
        toks, steps = [], []
        eos = False
        while t < valid_t and n < CAP * T_ENC:
            n += 1
            lg = self.joint(enc_frame=enc[t:t + 1], dec_out=dec_out)["logits"][0]
            tok = int(np.argmax(lg[:VOCAB]))
            d = int(np.argmax(lg[VOCAB:]))
            if tok == BLANK and d == 0:
                d = 1
            if keep_steps:
                steps.append((t, tok, d))
            if tok != BLANK:
                toks.append(tok)
                if tok == EOS:
                    eos = True
                    break
                o = self.pred(token=np.array([[tok]], np.int32), h=h, c=c)
                dec_out, h, c = o["dec_out"], o["h_out"], o["c_out"]
            if d == 0:
                if tok != BLANK:
                    on += 1
                    max_run = max(max_run, on)
                if on >= CAP:
                    cap_hits += 1
                    d = 1
            if d > 0:
                on = 0
            t += d
        r = {"tokens": toks, "n_steps": n, "max_run_same_frame": max_run, "cap_hits": cap_hits, "eos": eos}
        if keep_steps:
            r["steps"] = steps
        return r

    def window(self, audio, keep_steps=False, keep_enc=False):
        t0 = time.perf_counter()
        mo = self.features(audio)
        t1 = time.perf_counter()
        enc = self.encode(mo["features"], mo["mel_mask"])
        t2 = time.perf_counter()
        vt = enc_valid_frames(len(audio))
        r = self.loop(enc, vt, keep_steps)
        t3 = time.perf_counter()
        r.update({"samples": len(audio), "valid_enc": vt, "frame_mask_sum": int(mo["frame_mask"].sum()),
                  "mel_ms": round(1e3 * (t1 - t0), 2), "enc_ms": round(1e3 * (t2 - t1), 2),
                  "dec_ms": round(1e3 * (t3 - t2), 2)})
        if keep_enc:
            r["_enc"] = enc
            r["_mel"] = mo
        return r

    def utterance(self, audio):
        t0 = time.perf_counter()
        wins = [self.window(audio[s:s + n]) for s, n in windows(len(audio))]
        dt = time.perf_counter() - t0
        return {"win_tokens": [w["tokens"] for w in wins], "n_windows": len(wins),
                "n_steps": sum(w["n_steps"] for w in wins),
                "max_run_same_frame": max([w["max_run_same_frame"] for w in wins], default=0),
                "cap_hits": sum(w["cap_hits"] for w in wins), "eos": any(w["eos"] for w in wins),
                "empty_windows": sum(1 for w in wins if not w["tokens"]),
                "frame_mask_ok": all(w["frame_mask_sum"] == w["valid_enc"] for w in wins),
                "mel_ms": round(sum(w["mel_ms"] for w in wins), 2), "enc_ms": round(sum(w["enc_ms"] for w in wins), 2),
                "dec_ms": round(sum(w["dec_ms"] for w in wins), 2), "proc_s": round(dt, 4)}


def env_record():
    import platform
    import ai_edge_litert
    return {"python": platform.python_version(), "ai_edge_litert": getattr(ai_edge_litert, "__version__", "?"),
            "numpy": np.__version__, "machine": platform.machine(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loadavg": os.getloadavg()}


def run_set(p, a, set_name, out):
    man, _ = SETS[set_name]
    rows = read_jsonl(man)
    if a.limit:
        rows = rows[:a.limit]
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        open(out + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
        print(f"{out}: nothing to do ({len(done)} done)", flush=True)
        return True
    meta = {"form": a.form, "backend": a.backend, "threads": a.threads, "set": set_name, "manifest": man,
            "mel": p.mel.info, "enc": p.enc.info, "dec": p.dec.info, "env": env_record()}
    with open(out + ".meta.json", "a") as f:
        f.write(json.dumps(meta, default=str) + "\n")
    n, per, t0 = 0, 0.0, time.time()
    with open(out, "a") as fo:
        for r in todo:
            if a.deadline and time.time() + max(3 * per, 5.0) > a.deadline:
                break
            audio = load_wav(r["path"])
            res = p.utterance(audio)
            per = res["proc_s"] if per == 0 else 0.8 * per + 0.2 * res["proc_s"]
            fo.write(json.dumps({"id": r["id"], "audio_s": len(audio) / 16000, **res}) + "\n")
            fo.flush()
            n += 1
    left = len(todo) - n
    print(f"{out}: +{n}, {left} left, {time.time() - t0:.0f} s", flush=True)
    if left == 0:
        open(out + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
    return left == 0


def cmd_run(a):
    sets = a.sets.split(",")
    outs = {s: os.path.join(a.outdir, f"p1_{a.form}_{a.backend}_{s}.jsonl") for s in sets}
    if all(os.path.exists(o + ".DONE") for o in outs.values()):
        print("all done", flush=True)
        return 0
    if a.deadline and time.time() > a.deadline - 120:
        print("deadline too close, not loading", flush=True)
        return 0
    os.makedirs(a.outdir, exist_ok=True)
    p = Pipeline(a.form, a.backend, a.threads)
    print(f"loaded {a.form}/{a.backend}: {json.dumps(p.enc.info, default=str)[:400]}", flush=True)
    for s in sets:
        if os.path.exists(outs[s] + ".DONE"):
            continue
        if not run_set(p, a, s, outs[s]):
            break
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--form", required=True, choices=sorted(FILES))
    r.add_argument("--backend", default="cpu", choices=["cpu", "gpu32", "gpu16"])
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--sets", required=True, help="comma list of " + ",".join(sorted(SETS)))
    r.add_argument("--outdir", default="results/r2")
    r.add_argument("--deadline", type=float, default=0.0)
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
