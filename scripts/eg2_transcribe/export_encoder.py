#!/usr/bin/env python3
"""EmbeddingGemma 2's audio encoder (audio tower + embed_audio) -> one LiteRT graph per window, fp16-safe.

Signature audio_<S>s (S = 30 or 10): mel [1, 100 S, 128] float32 (the Gemma 4 feature extractor's log-mel rows, right
padding allowed) + mask [1, 100 S] float32 (1 = real frame) -> soft [1, 25 S, 512] (padding rows 0) + mask [1, 25 S].

Subcommands, in order (each loads the model in a fresh process; run them as scripts, see "Names" below):
  kmap    the rescale exponent k of every norm site, from the float32 ranges over the probe windows: every test-clean-20
          utterance as one 30 s window (right padding included) and 10 full windows spread over the long clip.
          k = the smallest k >= 0 with 4x headroom under fp16's 65,504 for both the row sum of x^2 and the largest x^2.
  export  load inside litert-torch's Gemma 4 patch (a context manager: only a load inside it builds the patched audio
          classes), keep the patched module's float32 outputs on every window as the reference, rewrite (below), check
          the rewrite against that reference bit for bit, convert eg2_audio_f16safe_w<S>.tflite, cast every
          FULLY_CONNECTED / CONV_2D / DEPTHWISE_CONV_2D weight to float16 (explicit DEQUANTIZE) ->
          eg2_audio_f16safe_fp16w_w<S>.tflite, then compare the module with the fixtures' soft tokens (whole utterances).
  parity  each converted file on the CPU (LiteRT Interpreter, XNNPACK) on every utterance cut into the file's window,
          valid tokens against the float32 reference: mean cosine >= 0.999 and minimum >= 0.99.

The rewrite, exact in float32 (checked bit for bit before converting):
  1. every RMSNorm (the 108 of the 12 conformer layers and embed_audio's pre-projection norm) -> F16SafeRMSNorm:
     x * 2^-k, mean of squares + eps * 4^-k, rsqrt clamped at 2^k / sqrt(eps) * (1 + 2^-20), times the weight. Powers of
     two commute with float32 rounding, so the result is x * rsqrt(mean(x^2) + eps) * w to the bit; the clamp is never
     reached in float32. In fp16, eps * 4^-k rounds to 0 for eps 1e-6 at k = 3, and an all-zero row (padding) would give
     0 * inf: the clamp keeps it at 0. The norms are plain ops, not composites (a composite has no place for the clamp).
  2. the attention's invalid-logit constant -1e9 -> -30000: -1e9 is -inf in fp16 and the mask is (1 - valid) * c,
     so 0 * -inf = NaN on every valid position. The softcap bounds the logits to +-50, so the masked exp is 0 in
     float32 with either constant.
  3. the output mask from the input mask by float ops (mask[:, ::2][:, ::2], the two stride-2 subsampling convolutions):
     the tower's own boolean mask on the way to a graph output makes the GPU's fp16 kernel build fail.
The two subsample LayerNorms stay as they are (k = 0 on these ranges).

Names: every tensor name in the converted file carries the Python path of its module classes ("export_nobool.
EG2AudioEncoderNoBool/...", "__main__.F16SafeRMSNorm_..."). The published files were converted with these names; run this
file as a script and keep the class names to get the same bytes. Other names give other bytes, not other numbers.

Do not load the model with attn_implementation="eager" under transformers 5.19.0 (the Gemma 4 audio attention expects a
boolean mask; the eager path passes a float one and the attention reads the masked positions). The patched graph builds
its own additive mask and does not depend on the attention implementation.

Usage:
  export_encoder.py kmap   --model <dir or repo id> [--revision R] --fixtures <reader/fixtures> --out <kmap.json>
  export_encoder.py export --model <dir or repo id> [--revision R] --fixtures <reader/fixtures> --kmap kmap.json
                           --out <dir> [--windows 30,10]
  export_encoder.py parity --fixtures <reader/fixtures> --out <dir> [--threads 4]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import litert_torch  # noqa: F401  (before transformers: it registers its torch ops)
from litert_torch.generative.export_hf.model_ext.gemma4 import patch as g4patch
import torch
import transformers
from transformers import AutoConfig
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2Model

import common

MASK_VALUE = -30000.0
F16_MAX = 65504.0
NORM_ATTRS = {"ff1.pre": ("feed_forward1", "pre_layer_norm"), "ff1.post": ("feed_forward1", "post_layer_norm"),
              "attn.pre": (None, "norm_pre_attn"), "attn.post": (None, "norm_post_attn"),
              "lconv.pre": ("lconv1d", "pre_layer_norm"), "lconv.convnorm": ("lconv1d", "conv_norm"),
              "ff2.pre": ("feed_forward2", "pre_layer_norm"), "ff2.post": ("feed_forward2", "post_layer_norm"),
              "out": (None, "norm_out")}


class EG2AudioEncoderNoBool(torch.nn.Module):
    """mel [1, F, 128] + mask [1, F] (float 0/1) -> soft [1, F/4, 512] (masked) + mask [1, F/4] (float, no bool)."""

    def __init__(self, tower, embed_audio):
        super().__init__()
        self.tower = tower
        self.embed_audio = embed_audio

    def forward(self, mel, mask):
        out = self.tower(mel, mask)
        m = mask[:, ::2][:, ::2]
        soft = self.embed_audio(out.last_hidden_state) * m.unsqueeze(-1)
        return {"soft": soft, "mask": m}


# The published files were converted with this class in a module named export_nobool (see "Names" above).
EG2AudioEncoderNoBool.__module__ = "export_nobool"


class F16SafeRMSNorm(torch.nn.Module):
    """x * rsqrt(mean(x^2) + eps) * w on x * 2^-k with eps * 4^-k, rsqrt clamped at the zero row's own value.
    pow_form=True takes the root as pow(., -0.5) (the form of embed_audio's norm), else torch.rsqrt (the tower's)."""

    def __init__(self, weight, eps, k, pow_form=False):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone(), requires_grad=False) if weight is not None else None
        self.k, self.eps, self.pow_form = int(k), float(eps), bool(pow_form)
        self.scale = 2.0 ** -self.k
        self.eps_k = self.eps * 4.0 ** -self.k
        self.r_max = (2.0 ** self.k) / math.sqrt(self.eps) * (1.0 + 2.0 ** -20)

    def forward(self, x):
        if self.k:
            x = x * self.scale
        ms = x.pow(2).mean(-1, keepdim=True) + self.eps_k
        r = torch.pow(ms, -0.5) if self.pow_form else torch.rsqrt(ms)
        r = torch.clamp(r, max=self.r_max)
        y = x * r
        return y * self.weight if self.weight is not None else y


def load(model, revision):
    kw = {} if os.path.isdir(model) else {"revision": revision}
    cfg = AutoConfig.from_pretrained(model, **kw)
    cfg.vision_config = None  # the vision tower is not built
    with g4patch.gemma4_litert_patch():
        m = EmbeddingGemma2Model.from_pretrained(model, config=cfg, dtype=torch.float32, **kw)
    m.eval()
    tower = m.audio_tower
    if not (isinstance(tower, g4patch.LiteRTGemma4AudioModel)
            and isinstance(tower.layers[0].self_attn, g4patch.LiteRTGemma4AudioAttention)
            and isinstance(tower.subsample_conv_projection.layer0, g4patch.LiteRTGemma4AudioSubSampleConvProjectionLayer)):
        raise SystemExit(f"the audio tower is not the patched one ({type(tower).__name__}): load it in a fresh process, "
                         "inside gemma4_litert_patch, before any other load of the model")
    return m


def norm_sites(model):
    """[(site name, parent module, attribute)] for the 109 RMSNorm sites."""
    out = []
    for li, L in enumerate(model.audio_tower.layers):
        for site, (sub, attr) in NORM_ATTRS.items():
            out.append((f"L{li:02d}.{site}", getattr(L, sub) if sub else L, attr))
    out.append(("embed.norm", model.embed_audio, "embedding_pre_projection_norm"))
    return out


def make_f16safe(model, kmap):
    """In place: replace the RMSNorms, set the mask constant. -> {site: k}."""
    done = {}
    for name, parent, attr in norm_sites(model):
        old = getattr(parent, attr)
        k = int(kmap[name])
        if name == "embed.norm":
            assert type(old).__name__ == "EmbeddingGemma2RMSNorm" and not old.with_scale, type(old)
            new = F16SafeRMSNorm(None, old.eps, k, pow_form=True)
        else:
            assert type(old).__name__ == "Gemma4RMSNorm" and type(old).__module__.startswith("litert_torch"), type(old)
            assert old.with_scale
            new = F16SafeRMSNorm(old.weight, old.variance_epsilon, k)
        setattr(parent, attr, new)
        done[name] = k
    model.audio_tower.config.attention_invalid_logits_value = MASK_VALUE
    return done


def utterance_windows(fixtures, ids, ws):
    """[(name, mel_w, mask_w)] every utterance cut into windows of ws seconds."""
    out = []
    for uid in ids:
        for wi, (mw, mk, _) in enumerate(common.windows_of(common.load_fixture(fixtures, uid, "features"), ws * common.FPS)):
            out.append((f"{uid}#w{wi}", mw, mk))
    return out


def run(module, wins):
    res = {}
    with torch.no_grad():
        for name, mw, mk in wins:
            o = module(torch.from_numpy(mw)[None], torch.from_numpy(mk)[None])
            res[name] = (o["soft"][0].numpy().copy(), o["mask"][0].numpy().copy())
    return res


def valid_rows(res, ids):
    """{uid: valid soft rows of all its windows back to back}."""
    out = {}
    for uid in ids:
        rows, wi = [], 0
        while f"{uid}#w{wi}" in res:
            s, m = res[f"{uid}#w{wi}"]
            v = int((m > 0.5).sum())
            assert (m[:v] > 0.5).all() and not (m[v:] > 0.5).any(), "the output mask is not a prefix"
            rows.append(s[:v])
            wi += 1
        out[uid] = np.concatenate(rows, 0)
    return out


def cmd_kmap(a):
    model = load(a.model, a.revision)
    module = EG2AudioEncoderNoBool(model.audio_tower, model.embed_audio).eval()
    stats = {}

    def hook(name, centered):
        def f(mod, args):
            x = args[0].detach().double()
            if centered:
                x = x - x.mean(-1, keepdim=True)
            rows = x.reshape(-1, x.shape[-1])
            s = stats.setdefault(name, {"max_sumsq": 0.0, "max_abs": 0.0})
            s["max_sumsq"] = max(s["max_sumsq"], float((rows * rows).sum(-1).max()))
            s["max_abs"] = max(s["max_abs"], float(rows.abs().amax(-1).max()))
        return f

    sub = model.audio_tower.subsample_conv_projection
    for i, lay in enumerate((sub.layer0, sub.layer1)):
        lay.norm.register_forward_pre_hook(hook(f"sub.l{i}.norm", True))
    for name, parent, attr in norm_sites(model):
        getattr(parent, attr).register_forward_pre_hook(hook(name, False))
    ids = common.fixture_ids(a.fixtures)
    wins = utterance_windows(a.fixtures, ids, 30) + common.long_windows(a.fixtures, 30 * common.FPS)
    run(module, wins)

    def k_for(v):
        return 0 if v <= 0 else max(0, math.ceil(math.log(4 * v / F16_MAX, 4)))

    k = {name: max(k_for(s["max_sumsq"]), k_for(s["max_abs"] ** 2)) for name, s in stats.items()}
    doc = {"rule": "k = min k >= 0 with 4 x max(row sum x^2, max x^2) * 4^-k <= 65504 (all rows of the probe windows)",
           "windows": f"{len(wins)} x 30 s: {len(ids)} test-clean-20 utterances (right padding) + 10 long-clip windows",
           "k": k, "ranges": stats}
    json.dump(doc, open(a.out, "w"), indent=1)
    hist = {str(v): sum(1 for x in k.values() if x == v) for v in sorted(set(k.values()))}
    print(f"KMAP_DONE {len(k)} sites, k histogram {hist} -> {a.out}", flush=True)


def cmd_export(a):
    from litert_torch._convert import interface as converter_utils
    from ai_edge_quantizer import quantizer
    torch.manual_seed(0)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    kmap = json.load(open(a.kmap))["k"]
    model = load(a.model, a.revision)
    module = EG2AudioEncoderNoBool(model.audio_tower, model.embed_audio).eval()
    ids = common.fixture_ids(a.fixtures)
    rep = {"litert_torch": getattr(litert_torch, "__version__", "?"), "transformers": transformers.__version__,
           "torch": torch.__version__, "model": a.model, "revision": a.revision, "mask_value": MASK_VALUE,
           "kmap": os.path.abspath(a.kmap), "check": {}, "files": {}}
    wss = [int(x) for x in a.windows.split(",")]
    wins, before = {}, {}
    for ws in wss:
        wins[ws] = utterance_windows(a.fixtures, ids, ws) + common.long_windows(a.fixtures, ws * common.FPS)
        before[ws] = run(module, wins[ws])
        ref = valid_rows(before[ws], ids)
        np.savez(os.path.join(a.out, f"torch_windows_w{ws}.npz"), **ref)
    sites = make_f16safe(model, kmap)
    rep["sites"] = sites
    rep["k_histogram"] = {str(k): sum(1 for v in sites.values() if v == k) for k in sorted(set(sites.values()))}
    print(f"rewrote {len(sites)} norm sites, k histogram {rep['k_histogram']}, mask constant {MASK_VALUE}", flush=True)
    for ws in wss:
        after = run(module, wins[ws])
        beq = all(np.array_equal(after[n][0], before[ws][n][0]) and np.array_equal(after[n][1], before[ws][n][1])
                  for n in after)
        dmax = max(float(np.abs(after[n][0].astype(np.float64) - before[ws][n][0]).max()) for n in after)
        rep["check"][f"w{ws}"] = {"windows": len(wins[ws]), "bit_equal_vs_unmodified": beq, "max_abs": dmax}
        print(f"w{ws}: rewritten vs unmodified on {len(wins[ws])} windows: max|d| {dmax:.3e}, bit-equal {beq}", flush=True)
        if not beq:
            raise SystemExit("the rewrite is not exact in float32: not converted")
    json.dump(rep, open(os.path.join(a.out, "export_report.json"), "w"), indent=1)
    for ws in wss:
        F = ws * common.FPS
        out = os.path.join(a.out, f"eg2_audio_f16safe_w{ws}.tflite")
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        t1 = time.time()
        conv = converter_utils.Converter()
        conv.add_signature(f"audio_{ws}s", module, sample_kwargs={"mel": torch.zeros(1, F, 128), "mask": torch.ones(1, F)})
        conv.convert(strict_export=False).export(out + ".tmp")
        os.replace(out + ".tmp", out)
        rep["files"][os.path.basename(out)] = {"signature": f"audio_{ws}s", "bytes": os.path.getsize(out),
                                              "sha256": common.sha256_file(out), "seconds": round(time.time() - t1, 1)}
        print(f"w{ws}: {out} {os.path.getsize(out):,} B in {time.time() - t1:.1f} s", flush=True)
        json.dump(rep, open(os.path.join(a.out, "export_report.json"), "w"), indent=1)
    cfg = {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}, "compute_precision": "FLOAT",
           "explicit_dequantize": True}
    recipe = [{"regex": ".*", "operation": op, "algorithm_key": "float_casting", "op_config": dict(cfg)}
              for op in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")]
    for ws in wss:
        src = os.path.join(a.out, f"eg2_audio_f16safe_w{ws}.tflite")
        out = os.path.join(a.out, f"eg2_audio_f16safe_fp16w_w{ws}.tflite")
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        t1 = time.time()
        qt = quantizer.Quantizer(src)
        qt.load_quantization_recipe(recipe)
        qt.quantize().export_model(out + ".tmp.tflite", overwrite=True)
        os.replace(out + ".tmp.tflite", out)
        rep["files"][os.path.basename(out)] = {"from": os.path.basename(src), "weights": "float16 (FC, CONV_2D, DEPTHWISE_CONV_2D)",
                                              "bytes": os.path.getsize(out), "sha256": common.sha256_file(out),
                                              "seconds": round(time.time() - t1, 1)}
        print(f"w{ws}: {out} {os.path.getsize(out):,} B in {time.time() - t1:.1f} s", flush=True)
        json.dump(rep, open(os.path.join(a.out, "export_report.json"), "w"), indent=1)
    # the converted module against the transformers reference (the fixtures' soft tokens): whole utterances
    md = 0.0
    with torch.no_grad():
        for uid in ids:
            mel = common.load_fixture(a.fixtures, uid, "features")
            ref = common.load_fixture(a.fixtures, uid, "soft_tokens")
            soft = module(torch.from_numpy(mel)[None], torch.ones(1, mel.shape[0]))["soft"][0].numpy()
            md = max(md, float(np.abs(soft[:len(ref)].astype(np.float64) - ref).max()))
    rep["module_vs_fixture_soft_tokens_max_abs"] = md
    print(f"the converted module vs the fixtures' soft tokens (whole utterances): max|d| {md:.3e}", flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.out, "export_report.json"), "w"), indent=1)
    for name, f in rep["files"].items():
        print(f"{f['sha256']}  {name}", flush=True)
    print("EXPORT_DONE", flush=True)


def cmd_parity(a):
    from ai_edge_litert.interpreter import Interpreter
    ids = common.fixture_ids(a.fixtures)
    rep = {"runtime": "ai_edge_litert Interpreter (XNNPACK)", "threads": a.threads,
           "bar": "valid tokens of every utterance: mean cosine >= 0.999 and minimum >= 0.99", "files": {}}
    for path in sorted(p for p in os.listdir(a.out) if p.startswith("eg2_audio_") and p.endswith(".tflite")):
        it = Interpreter(model_path=os.path.join(a.out, path), num_threads=a.threads)
        sig = list(it.get_signature_list())[0]
        ws = int(sig[len("audio_"):-1])
        runner = it.get_signature_runner(sig)
        ref = np.load(os.path.join(a.out, f"torch_windows_w{ws}.npz"))
        cs, md = [], 0.0
        for uid in ids:
            rows = []
            for mw, mk, _ in common.windows_of(common.load_fixture(a.fixtures, uid, "features"), ws * common.FPS):
                o = runner(mel=mw[None], mask=mk[None])
                v = int((np.asarray(o["mask"])[0] > 0.5).sum())
                rows.append(np.asarray(o["soft"])[0, :v].copy())
            got = np.concatenate(rows, 0)
            assert got.shape == ref[uid].shape, (uid, got.shape, ref[uid].shape)
            cs.append(common.rowwise_cos(got, ref[uid]))
            md = max(md, float(np.abs(got.astype(np.float64) - ref[uid]).max()))
        c = np.concatenate(cs)
        r = {"signature": sig, "tokens": int(c.size), "cos_mean": float(c.mean()), "cos_min": float(c.min()),
             "max_abs": md, "pass": bool(c.mean() >= 0.999 and c.min() >= 0.99)}
        rep["files"][path] = r
        print(f"{path} {sig}: {r['tokens']} tokens, cos mean {r['cos_mean']:.13f} min {r['cos_min']:.10f} "
              f"max|d| {md:.3e} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
        del runner, it
    json.dump(rep, open(os.path.join(a.out, "parity_cpu.json"), "w"), indent=1)
    print("PARITY_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["kmap", "export", "parity"])
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--revision", default="914f7f89142e33e77833254d9c9b90c3cef7303b")
    ap.add_argument("--fixtures", required=True, help="eg2-transcribe reader/fixtures (MANIFEST.json + rebuilt arrays)")
    ap.add_argument("--kmap", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmap.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="30,10")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    n = common.check_fixtures(a.fixtures, long=a.cmd != "parity")
    print(f"fixtures: {n} files equal to MANIFEST.json", flush=True)
    {"kmap": cmd_kmap, "export": cmd_export, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    main()
