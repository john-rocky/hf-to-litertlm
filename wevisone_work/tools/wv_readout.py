#!/usr/bin/env python3
"""Round-2 line 3: conversion parity of the WeVisOne-VL-2B graphs (no runtime) — the tflites driven the way LiteRT-LM
v0.18.1 drives them, teacher-forced on the fp32 HF references' own tokens.

Per sequence (KV states zeroed at the start):
  embeddings  text tokens through the EMBEDDER tflite ([1, 1, 8196]); the 256 <|image_pad|> slots from
              (graph) the VISION_ENCODER -> VISION_ADAPTER tflites on out/r2/png512/<stem>.png * (1 / 255), or
              (hf)    HF's visual outputs for the same PNG (out/r2/hf_vis) + flag 1 + zeros, which isolates the decoder
  positions   the runtime's 1-D run 0, 1, 2, ...; mask 0 on cache slots <= position, -0.7 * FLT_MAX elsewhere
  schedule    prompt[:-1] through prefill signatures that fit exactly (largest first, no padding), then the decode
              signature on the last prompt token (the runtime's pending token) and on each reference token
Compared with the fp32 HF teacher-forced logits on the same tokens (out/r2/hf_tf): softmax (T = 1, float64) of both,
max |dp| over the vocabulary, argmax agreement on the non-tie positions (tie = HF top-2 probability gap <= 1e-4,
decided before the numbers, the decider rule).

Pass line: image rows vs B: non-tie argmax all equal and max |dp| <= 0.02 over the 29 rows; text rows
vs T0 the same over the 10 rows (two_turn: both turns).

    ~/venvs/lt0100dev/bin/python -I tools/wv_readout.py --label fp32 --dec out/r2/dec_fp32/model.tflite \
        --emb out/r2/dec_fp32/embedder.tflite --enc out/r2/vision/vision_encoder_fp32.tflite \
        --ada out/r2/vision/vision_adapter_fp32.tflite --vision graph [--rows ...] [--text] [--threads 8]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
HFTF = K / "out" / "r2" / "hf_tf"
HFVIS = K / "out" / "r2" / "hf_vis"
PNG = K / "out" / "r2" / "png512"
IMAGE_PAD = 151655
LADDER = [1024, 256, 64, 16, 4, 1]
NEG = np.float32(-0.7 * np.finfo(np.float32).max)
TIE_P = 1e-4
BAR_DP = 0.02


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def compiled(path, threads, cache=None):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import CpuOptions, Options
    cpu = CpuOptions(num_threads=threads, xnnpack_weight_cache_path=str(cache) if cache else "")
    return CompiledModel.from_file(str(path), options=Options(hardware_accelerators=HardwareAccelerator.CPU,
                                                              cpu_options=cpu))


class Single:
    def __init__(self, path, threads):
        self.model = compiled(path, threads)
        sigs = self.model.get_signature_list()
        assert len(sigs) == 1, sigs
        self.key = next(iter(sigs))
        self.inp = self.model.get_input_tensor_details(self.key)
        self.out = self.model.get_output_tensor_details(self.key)

    def __call__(self, x):
        (iname, idet), = self.inp.items()
        (oname, odet), = self.out.items()
        ib = self.model.create_input_buffer_by_name(self.key, iname)
        ob = self.model.create_output_buffer_by_name(self.key, oname)
        try:
            ib.write(np.ascontiguousarray(x, dtype=idet["dtype"]).reshape(idet["shape"]))
            self.model.run_by_name(self.key, {iname: ib}, {oname: ob})
            y = ob.read(int(np.prod(odet["shape"])), odet["dtype"]).reshape(odet["shape"]).copy()
        finally:
            ib.destroy()
            ob.destroy()
        return y


class Decoder:
    def __init__(self, path, threads, cache):
        t0 = time.monotonic()
        self.model = compiled(path, threads, cache)
        self.load_s = time.monotonic() - t0
        self.sigs = sorted(self.model.get_signature_list())
        assert set(self.sigs) == {"decode", *(f"prefill_{n}" for n in LADDER)}, self.sigs
        self.inputs = {k: self.model.get_input_tensor_details(k) for k in self.sigs}
        self.outputs = {k: self.model.get_output_tensor_details(k) for k in self.sigs}
        self.states = [n for n in self.inputs["decode"] if n not in ("embeddings", "input_pos", "mask")]
        self.cache_len = int(self.inputs["decode"]["mask"]["shape"][-1])
        self.width = int(self.inputs["decode"]["embeddings"]["shape"][-1])
        for k in self.sigs:
            assert set(self.inputs[k]) == {"embeddings", "input_pos", "mask", *self.states}, k
            assert set(self.outputs[k]) == set(self.states) | ({"logits"} if k == "decode" else set()), \
                (k, sorted(self.outputs[k]))
        self.calls = 0

    def zero_states(self):
        return {n: np.zeros(self.inputs["decode"][n]["shape"], dtype=self.inputs["decode"][n]["dtype"])
                for n in self.states}

    def step(self, key, emb, positions, states):
        inp, out = self.inputs[key], self.outputs[key]
        allowed = np.arange(self.cache_len)[None, :] <= positions[:, None]
        mask = np.where(allowed, np.float32(0.0), NEG).astype(np.float32).reshape(inp["mask"]["shape"])
        feeds = dict(states, embeddings=np.ascontiguousarray(emb, np.float32).reshape(inp["embeddings"]["shape"]),
                     input_pos=positions.astype(inp["input_pos"]["dtype"]).reshape(inp["input_pos"]["shape"]),
                     mask=mask)
        ibuf, obuf = {}, {}
        try:
            for name, arr in feeds.items():
                b = self.model.create_input_buffer_by_name(key, name)
                ibuf[name] = b
                b.write(np.ascontiguousarray(arr))
            obuf = {name: self.model.create_output_buffer_by_name(key, name) for name in out}
            self.model.run_by_name(key, ibuf, obuf)
            vals = {name: obuf[name].read(int(np.prod(d["shape"])), d["dtype"]).reshape(d["shape"])
                    for name, d in out.items()}
        finally:
            for b in [*ibuf.values(), *obuf.values()]:
                b.destroy()
        self.calls += 1
        return {n: vals[n] for n in self.states}, vals.get("logits")


def chunk_plan(n):
    plan, p = [], 0
    for length in LADDER:
        while n - p >= length:
            plan.append((f"prefill_{length}", p, length))
            p += length
    assert p == n
    return plan


def softmax64(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Readout:
    def __init__(self, a):
        self.dec = Decoder(K / a.dec, a.threads, K / "out" / "r2" / "cache" / f"readout_{a.label}.xnnpack_cache")
        self.emb = Single(K / a.emb, a.threads)
        self.enc = Single(K / a.enc, a.threads) if a.enc else None
        self.ada = Single(K / a.ada, a.threads) if a.ada else None
        self.vision = a.vision
        self._emb = {}
        log(f"decoder loaded {self.dec.load_s:.1f} s, width {self.dec.width}, cache {self.dec.cache_len}, "
            f"{len(self.dec.states)} states")

    def embed(self, t):
        if t not in self._emb:
            self._emb[t] = self.emb(np.array([[t]], np.int32)).reshape(-1).copy()
        return self._emb[t]

    def image_rows(self, stem):
        if self.vision == "hf":
            w = np.load(HFVIS / f"{stem}.npz")["wide"]
            tail = np.zeros((w.shape[0], self.dec.width - w.shape[1]), np.float32)
            tail[:, 0] = 1.0
            return np.concatenate([w, tail], axis=1)
        from PIL import Image
        with Image.open(PNG / f"{stem}.png") as im:
            u8 = np.asarray(im.convert("RGB"))
        x = (u8.astype(np.float32) * np.float32(1.0 / 255.0))[None]
        return self.ada(self.enc(x))[0]                                       # [256, 8196]

    def forced(self, prompt_ids, ref_ids, stem=None):
        img = self.image_rows(stem) if stem else None
        rows, k = [], 0
        for t in prompt_ids:
            if t == IMAGE_PAD:
                rows.append(img[k])
                k += 1
            else:
                rows.append(self.embed(t))
        assert img is None or k == img.shape[0]
        emb = np.stack(rows)
        n = len(prompt_ids)
        states = self.dec.zero_states()
        for key, a, length in chunk_plan(n - 1):
            states, lg = self.dec.step(key, emb[a:a + length], np.arange(a, a + length), states)
        out = []
        cur, pos = emb[n - 1:n], n - 1
        for i, t in enumerate(ref_ids):
            states, lg = self.dec.step("decode", cur, np.array([pos]), states)
            out.append(lg.reshape(-1).astype(np.float32))
            if i + 1 < len(ref_ids):
                cur, pos = self.embed(t)[None], pos + 1
        sha = hashlib.sha256(np.ascontiguousarray(img, np.float32).tobytes()).hexdigest() if img is not None else None
        return np.stack(out), sha


def compare(test_logits, ref_npz):
    ref = np.load(ref_npz)
    rl = ref["logits"]
    assert rl.shape == test_logits.shape, (rl.shape, test_logits.shape)
    pr, pt = softmax64(rl), softmax64(test_logits)
    dp = np.abs(pr - pt).max(-1)
    top2 = np.sort(pr, axis=-1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    tie = gap <= TIE_P
    ra, ta = pr.argmax(-1), pt.argmax(-1)
    agree = ra == ta
    new_ids = ref["new_ids"]
    return {"n": int(len(dp)), "n_tie": int(tie.sum()), "nontie_agree": int((agree & ~tie).sum()),
            "n_nontie": int((~tie).sum()), "all_agree": int(agree.sum()), "max_dp": float(dp.max()),
            "mean_dp": float(dp.mean()), "argmax_ref_token_agree": int((ta == new_ids).sum()),
            "disagree_at": [int(i) for i in np.nonzero(~agree)[0]],
            "disagree_tie": [bool(tie[i]) for i in np.nonzero(~agree)[0]],
            "max_logit_abs": float(np.abs(rl - test_logits).max()),
            "worst_pos": int(dp.argmax()), "ref_gap_at_worst": float(gap[dp.argmax()])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--enc", default="")
    ap.add_argument("--ada", default="")
    ap.add_argument("--vision", choices=("graph", "hf"), default="graph")
    ap.add_argument("--rows", default="")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out_path = K / "r2" / "readout" / f"{a.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (K / "out" / "r2" / "cache").mkdir(parents=True, exist_ok=True)
    res = json.loads(out_path.read_text()) if out_path.exists() else {"image": {}, "text": {}}
    res.update({"label": a.label, "files": {k: getattr(a, k) for k in ("dec", "emb", "enc", "ada")},
                "vision": a.vision, "tie_p": TIE_P, "bar_dp": BAR_DP})
    ro = Readout(a)
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    want = None if a.rows in ("", "all") else set(a.rows.split(","))
    for row in rows if a.rows else []:
        if (want and row["id"] not in want) or row["id"] in res["image"]:
            continue
        b = json.loads((K / "ref" / "B" / f"{row['id']}.json").read_text())
        t0 = time.perf_counter()
        lg, sha = ro.forced(b["prompt_ids"], b["new_ids"], Path(row["image"]).stem)
        c = compare(lg, HFTF / "B" / f"{row['id']}.npz")
        c.update({"image_sha256": sha, "wall_s": round(time.perf_counter() - t0, 1)})
        res["image"][row["id"]] = c
        log(f"{row['id']}: nontie {c['nontie_agree']}/{c['n_nontie']} max|dp| {c['max_dp']:.4f} "
            f"({c['wall_s']} s)")
        out_path.write_text(json.dumps(res, indent=1) + "\n")
    if a.text:
        for p in sorted((K / "ref" / "T0").glob("*.json")):
            t0r = json.loads(p.read_text())
            for i, rec in enumerate(t0r["records"]):
                name = t0r["row"] if len(t0r["records"]) == 1 else f"{t0r['row']}_turn{i + 1}"
                if name in res["text"]:
                    continue
                lg, _ = ro.forced(rec["prompt_ids"], rec["new_ids"])
                res["text"][name] = compare(lg, HFTF / "T0" / f"{name}.npz")
                log(f"text {name}: nontie {res['text'][name]['nontie_agree']}/{res['text'][name]['n_nontie']} "
                    f"max|dp| {res['text'][name]['max_dp']:.4f}")
                out_path.write_text(json.dumps(res, indent=1) + "\n")

    def summ(d):
        if not d:
            return None
        return {"rows": len(d), "nontie_agree": sum(v["nontie_agree"] for v in d.values()),
                "n_nontie": sum(v["n_nontie"] for v in d.values()), "n_tie": sum(v["n_tie"] for v in d.values()),
                "max_dp": max(v["max_dp"] for v in d.values()),
                "rows_all_nontie_agree": sum(v["nontie_agree"] == v["n_nontie"] for v in d.values()),
                "pass": all(v["nontie_agree"] == v["n_nontie"] for v in d.values())
                and max(v["max_dp"] for v in d.values()) <= BAR_DP}
    res["summary"] = {"image": summ(res["image"]), "text": summ(res["text"]), "decoder_calls": ro.dec.calls}
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    log("READOUT", a.label, json.dumps(res["summary"]))


if __name__ == "__main__":
    sys.exit(main())
