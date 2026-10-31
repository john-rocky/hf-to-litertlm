#!/usr/bin/env python3
"""parakeet-ultra r1 LiteRT driver (venv .venv-q: ai-edge-litert 2.3.0). Copied from parakeet_redux_work/scripts/
r4_drive.py (redux r2_drive.py with the window buckets of redux CONTRACT v3); changed: every file under out/r1, the
encoder forms of this lane, the decoder file as an option (dec = fp32, dec_int8 = dynamic-range int8).
mel<b> -> enc<b>_<form> -> dec loop, b = the bucket of each 30 s window: the smallest of the pipeline's buckets whose
audio input holds the window (10 s: n <= 160,000 samples, 20 s: n <= 320,000, else 30 s). `--buckets 30` is the
30 s-only pipeline of r2 (same code path, every window in the 30 s files). mel and dec always run on the CPU
(Interpreter + XNNPACK, no weight cache); the encoders on the CPU (`cpu`) or the Mac GPU (CompiledModel with the
wheel's Metal accelerator, GPU|CPU; `gpu32` = enforce_f32, `gpu16` = the default fp16 arithmetic).

Files: out/r1/mel{10,20,30}.tflite, out/r1/enc{10,20,30}_<form>.tflite (form = fp32, fp32s, fp16ws, int8, i4b32,
i4b128), out/r1/dec.tflite or out/r1/dec_int8.tflite.
The decode loop and the window cut are r2's (CONTRACT v2 rules 1-5; the step limit stays 10 x 375 in every bucket).

CLI (resumable P1 run; one JSON line per utterance):
  u1_drive.py run --form int8 --backend cpu --threads 3 --buckets 10,20,30 --dec dec_int8 --sets ls[,de,fr,ru]
                  [--shard 0/4] [--outdir results/r1] [--deadline EPOCH] [--limit N]
     -> <outdir>/p1_<form>_<backend>_<bkt|w30>_<dec>_<set>[.s<k>of<n>].jsonl (+ .meta.json, .DONE)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r2_drive import CpuGraph, SETS, env_record, load_wav, read_jsonl, windows  # noqa: E402,F401

HOP = 160
BLANK = 8192
VOCAB = 8193
EOS = 3
CAP = 10
MAX_STEPS = CAP * 375
D = 640
T_MEL = {10: 1000, 20: 2000, 30: 3000}
DECS = {"dec": "out/r1/dec.tflite", "dec_int8": "out/r1/dec_int8.tflite"}
FORMS = ["fp32", "fp32s", "fp16ws", "int8", "i4b32", "i4b128"]


def n_audio(b):
    return T_MEL[b] * HOP


def bucket_of(n, buckets):
    for b in sorted(buckets):
        if b != 30 and n <= n_audio(b):
            return b
    return 30


def mel_path(b):
    return f"out/r1/mel{b}.tflite"


def enc_path(b, form):
    return f"out/r1/enc{b}_{form}.tflite"


def enc_valid_frames(n_valid):
    return ((n_valid // HOP) + 7) // 8


def bk_tag(buckets):
    return "w30" if tuple(buckets) == (30,) else "bkt"


class GpuGraph:
    """CompiledModel on the Mac GPU (the wheel's Metal accelerator); ops the GPU refuses fall back to the CPU. Output
    shapes from the compiled model's details, else from `shapes`."""

    def __init__(self, path, f32, shapes=None, key="serving_default"):
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
        self.out_shapes = {}
        try:
            det = self.m.get_output_tensor_details(key)
        except Exception:  # noqa: BLE001
            det = {}
        for n in self.out_names:
            try:
                self.out_shapes[n] = tuple(int(x) for x in det[n]["shape"])
            except Exception:  # noqa: BLE001
                self.out_shapes[n] = tuple((shapes or {})[n])
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
    def __init__(self, form, backend="cpu", threads=4, buckets=(10, 20, 30), dec_threads=1, dec="dec"):
        self.buckets = tuple(sorted(buckets))
        self.mel, self.enc = {}, {}
        for b in self.buckets:
            self.mel[b] = CpuGraph(mel_path(b), threads)
            p = enc_path(b, form)
            if backend == "cpu" or os.environ.get("R4_FAKE_GPU") == "1":  # R4_FAKE_GPU: dry runs outside the window
                self.enc[b] = CpuGraph(p, threads)
            elif backend in ("gpu32", "gpu16"):
                self.enc[b] = GpuGraph(p, backend == "gpu32", {"enc": (1, T_MEL[b] // 8, D)})
            else:
                raise SystemExit(backend)
        self.dec = CpuGraph(DECS[dec], dec_threads)
        self.dec_name = dec
        self.pred = self.dec.runners["predict"]
        self.joint = self.dec.runners["joint"]
        self.form, self.backend, self.threads = form, backend, threads

    def info(self):
        return {"buckets": list(self.buckets), "mel": {b: g.info for b, g in self.mel.items()},
                "enc": {b: g.info for b, g in self.enc.items()}, "dec": self.dec.info}

    def features(self, audio, b):
        n = len(audio)
        x = np.zeros((1, n_audio(b)), np.float32)
        x[0, :n] = audio
        return self.mel[b]("serving_default", audio=x, n_valid=np.array([n], np.int32))

    def encode(self, features, mel_mask, b):
        return np.array(self.enc[b]("serving_default", features=features, mel_mask=mel_mask)["enc"][0], copy=True)

    def loop(self, enc, valid_t, keep_steps=False):
        h = np.zeros((2, 1, D), np.float32)
        c = np.zeros((2, 1, D), np.float32)
        o = self.pred(token=np.array([[BLANK]], np.int32), h=h, c=c)
        dec_out, h, c = o["dec_out"], o["h_out"], o["c_out"]
        t = n = on = max_run = cap_hits = 0
        toks, steps = [], []
        eos = False
        while t < valid_t and n < MAX_STEPS:
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
        b = bucket_of(len(audio), self.buckets)
        t0 = time.perf_counter()
        mo = self.features(audio, b)
        t1 = time.perf_counter()
        enc = self.encode(mo["features"], mo["mel_mask"], b)
        t2 = time.perf_counter()
        vt = enc_valid_frames(len(audio))
        r = self.loop(enc, vt, keep_steps)
        t3 = time.perf_counter()
        r.update({"bucket": b, "samples": len(audio), "valid_enc": vt, "frame_mask_sum": int(mo["frame_mask"].sum()),
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
                "buckets": [w["bucket"] for w in wins], "n_steps": sum(w["n_steps"] for w in wins),
                "max_run_same_frame": max([w["max_run_same_frame"] for w in wins], default=0),
                "cap_hits": sum(w["cap_hits"] for w in wins), "eos": any(w["eos"] for w in wins),
                "empty_windows": sum(1 for w in wins if not w["tokens"]),
                "frame_mask_ok": all(w["frame_mask_sum"] == w["valid_enc"] for w in wins),
                "mel_ms": round(sum(w["mel_ms"] for w in wins), 2), "enc_ms": round(sum(w["enc_ms"] for w in wins), 2),
                "dec_ms": round(sum(w["dec_ms"] for w in wins), 2), "proc_s": round(dt, 4)}


def run_set(p, a, set_name, out, shard):
    man, _ = SETS[set_name]
    rows = read_jsonl(man)
    if a.limit:
        rows = rows[:a.limit]
    k, n_sh = shard
    rows = rows[k::n_sh]
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        open(out, "a").close()
        open(out + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
        print(f"{out}: nothing to do ({len(done)} done)", flush=True)
        return True
    meta = {"form": a.form, "backend": a.backend, "threads": a.threads, "buckets": list(p.buckets), "dec": a.dec,
            "set": set_name,
            "shard": list(shard), "manifest": man, "pipeline": p.info(), "env": env_record()}
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


def out_name(outdir, form, backend, buckets, dec, s, shard):
    k, n = shard
    suf = f".s{k}of{n}" if n > 1 else ""
    return os.path.join(outdir, f"p1_{form}_{backend}_{bk_tag(buckets)}_{dec}_{s}{suf}.jsonl")


def cmd_run(a):
    sets = a.sets.split(",")
    buckets = tuple(sorted(int(x) for x in a.buckets.split(",")))
    shard = tuple(int(x) for x in a.shard.split("/"))
    outs = {s: out_name(a.outdir, a.form, a.backend, buckets, a.dec, s, shard) for s in sets}
    if all(os.path.exists(o + ".DONE") for o in outs.values()):
        print("all done", flush=True)
        return 0
    if a.deadline and time.time() > a.deadline - 120:
        print("deadline too close, not loading", flush=True)
        return 0
    os.makedirs(a.outdir, exist_ok=True)
    p = Pipeline(a.form, a.backend, a.threads, buckets, dec=a.dec)
    print(f"loaded {a.form}/{a.backend}/{buckets}: " + json.dumps({b: g.info.get("compile_s", g.info.get("load_s"))
                                                                    for b, g in p.enc.items()}), flush=True)
    for s in sets:
        if os.path.exists(outs[s] + ".DONE"):
            continue
        if not run_set(p, a, s, outs[s], shard):
            break
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--form", required=True, choices=FORMS)
    r.add_argument("--dec", default="dec", choices=sorted(DECS))
    r.add_argument("--backend", default="cpu", choices=["cpu", "gpu32", "gpu16"])
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--buckets", default="10,20,30")
    r.add_argument("--sets", required=True, help="comma list of " + ",".join(sorted(SETS)))
    r.add_argument("--shard", default="0/1")
    r.add_argument("--outdir", default="results/r1")
    r.add_argument("--deadline", type=float, default=0.0)
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
