"""D1Audio (the export form) against the provider's own `get_audio_features`, clip by clip, at every bucket.

The provider side is built from the snapshot's code without the 2B language model: its `Lfm3OmniEncoder` and
`Lfm3OmniProjector` with the checkpoint's tensors, d1's `lfm3_conv` patch applied as `Lfm3SystemOne.__init__` applies
it, the LFM2 attention under "sdpa" (what "d1" resolves to for a mask that is not a `Sees`: `attention.d1_attention`
calls `sdpa_attention_forward`, and the "d1" mask interface is `sdpa_mask`), and `Lfm3OmniModel.get_audio_features`
itself called on that pair. The features come from the provider's own feature extractor (`processor.feature_extractor`
after `media.read_audio` + `media.resampled`).

Measures per clip: provider fp32 vs D1Audio fp32 at the clip's bucket (and at every larger bucket: padding invariance),
and with --fp64: the provider in float64 (the LFM2 casts of the float64 reference) vs D1Audio in float64 (identity),
provider fp32 vs fp64 and D1Audio fp32 vs fp64.

  $REF conversion/audio_check.py [--fp64] [--clips a.wav b.npy ...] [--synthetic] [--save-feats DIR] \
      --out results/audio_check_<tag>.json

--save-feats DIR writes each clip's provider input_features [128, F] (float32) as DIR/<clip>.npy: the export's check
inputs (conversion/export_audio.py reads them; the export environment has no provider code).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "out/hf_home"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(WORK, "out/hf_modules"))


def provider_parts(snap, dtype):
    """(get_audio_features as a function of (input_features, mask), the processor, the provider package)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], snap)
    pkg = cls.__module__.rsplit(".", 1)[0]
    om = sys.modules[pkg + ".modeling_lfm3_omni"]
    lfm3 = sys.modules[pkg + ".lfm3"]
    config.audio_encoder_config.text_config._attn_implementation = "sdpa"
    holder = torch.nn.Module()
    holder.audio_encoder = om.Lfm3OmniEncoder(config.audio_encoder_config)
    holder.audio_projector = om.Lfm3OmniProjector(config)
    state = {}
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            for pre, name in (("model.audio_encoder.", "audio_encoder."), ("model.audio_projector.", "audio_projector.")):
                if k.startswith(pre):
                    state[name + k[len(pre):]] = f.get_tensor(k)
    res = holder.load_state_dict(state, strict=False)
    missing = [k for k in res.missing_keys if "rotary_emb" not in k]
    assert not missing and not res.unexpected_keys, (missing, res.unexpected_keys)
    holder = holder.to(dtype).eval()
    lfm3.lfm3_conv(holder)  # Lfm3SystemOne.__init__: the LFM2 convs as `depthwise`, the downsampler's as `pinned_conv`
    me = types.SimpleNamespace(config=config, audio_encoder=holder.audio_encoder, audio_projector=holder.audio_projector)

    def get(feats, mask):
        with torch.inference_mode():
            return om.Lfm3OmniModel.get_audio_features(me, feats, mask)

    proc = AutoProcessor.from_pretrained(snap, trust_remote_code=True)
    return get, proc, pkg


def synthetic_clips():
    """Deterministic clips at every bucket's edges: noise + tones + bursts, 16 kHz float32, (name, wave)."""
    import numpy as np

    rng = np.random.default_rng(7)
    out = []
    for sec in (0.3, 2.5, 5.0, 5.05, 7.7, 10.0, 13.3, 20.0, 26.6, 30.0):
        n = int(sec * 16000)
        t = np.arange(n) / 16000
        w = rng.normal(0, 0.01, n) + 0.2 * np.sin(2 * np.pi * (220 + 180 * np.sin(t)) * t) * (np.sin(3 * t) > 0)
        w[: n // 7] *= 0.01
        out.append((f"syn_{sec:g}s", w.astype(np.float32)))
    out.append(("syn_silence_1s", np.zeros(16000, np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--buckets", default="504,1000,2000,3000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--save-feats", default="")
    a = ap.parse_args()
    import numpy as np
    import torch

    import d1_audio as A

    torch.set_num_threads(a.threads)
    snap = A.SNAP_DEFAULT
    t0 = time.time()
    get32, proc, pkg = provider_parts(snap, torch.float32)
    media = sys.modules[pkg + ".media"]
    fe = proc.feature_extractor
    rate = fe.sampling_rate
    buckets = [int(b) for b in a.buckets.split(",")]
    W = A.load_audio_weights(snap)
    spec = A.Spec(snap)
    mods32 = {T: A.D1Audio(W, spec, T, torch.float32).eval() for T in buckets}
    if a.fp64:
        from transformers.models.lfm2 import modeling_lfm2 as m2

        orig = (m2.Lfm2RMSNorm.forward, m2.Lfm2RotaryEmbedding.forward)

        def rms64(self, v):  # = the float64 reference's Lfm2RMSNorm.forward
            return self.weight * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

        @torch.no_grad()
        def rope64(self, v, position_ids):  # = the float64 reference's Lfm2RotaryEmbedding.forward
            fr = position_ids[..., None].to(v.dtype) * self.inv_freq.to(v.dtype)
            emb = torch.cat((fr, fr), dim=-1)
            return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

        get64, _, _ = provider_parts(snap, torch.float64)
        W64 = {k: v.double() for k, v in W.items()}  # one float64 copy shared by the buckets
        mods64 = {T: A.D1Audio(W64, spec, T, torch.float64).eval() for T in buckets}
    clips = synthetic_clips() if a.synthetic else []
    for p in a.clips:
        wave, given = media.read_audio(p, rate)
        clips.append((os.path.basename(p), media.resampled(wave, given, rate)))
    rows = []
    for name, wave in clips:
        feats = fe(wave, sampling_rate=rate, return_tensors="pt")
        x, m = feats["input_features"], feats["feature_attention_mask"]
        frames = int(m.sum())
        n_tok = A.tokens_for(frames)
        if a.save_feats:
            os.makedirs(os.path.join(WORK, a.save_feats), exist_ok=True)
            np.save(os.path.join(WORK, a.save_feats, f"{name}.npy"), x[0, :, :frames].numpy().astype(np.float32))
        ref32 = get32(x, m).float().numpy()
        row = {"clip": name, "samples": int(len(wave)), "frames": frames, "tokens": n_tok,
               "provider_tokens": int(ref32.shape[0]), "ref_max_abs": float(np.abs(ref32).max()), "buckets": {}}
        if a.fp64:
            m2.Lfm2RMSNorm.forward, m2.Lfm2RotaryEmbedding.forward = rms64, rope64  # fp64_patch's LFM2 casts
            try:
                ref64 = get64(x, m).numpy()
            finally:
                m2.Lfm2RMSNorm.forward, m2.Lfm2RotaryEmbedding.forward = orig
            row["provider_fp32_vs_fp64"] = float(np.abs(ref32 - ref64).max())
        for T in buckets:
            if frames > T:
                continue
            mel, valid = A.make_inputs(x[0].numpy(), T)
            with torch.inference_mode():
                got = mods32[T](torch.from_numpy(mel), torch.from_numpy(valid))["features"][0, :n_tok].numpy()
            d = np.abs(got - ref32)
            cos = float(np.min((got * ref32).sum(-1) /
                               (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref32, axis=-1) + 1e-30)))
            b = {"fp32_vs_provider_fp32_max_abs": float(d.max()), "min_cos": cos}
            if a.fp64:
                with torch.inference_mode():
                    g64 = mods64[T](torch.from_numpy(mel).double(), torch.from_numpy(valid).double())["features"][0, :n_tok].numpy()
                b["fp64_vs_provider_fp64_max_abs"] = float(np.abs(g64 - ref64).max())
                b["own_fp32_vs_fp64_max_abs"] = float(np.abs(got - g64).max())
            row["buckets"][str(T)] = b
        rows.append(row)
        print(json.dumps({"clip": name, "frames": frames, "tokens": n_tok,
                          **{T: round(v["fp32_vs_provider_fp32_max_abs"], 7) for T, v in row["buckets"].items()},
                          **({"id64": max(v["fp64_vs_provider_fp64_max_abs"] for v in row["buckets"].values()),
                              "p32v64": row["provider_fp32_vs_fp64"]} if a.fp64 else {})}), flush=True)
    summary = {"clips": len(rows),
               "max_fp32_vs_provider": max(v["fp32_vs_provider_fp32_max_abs"] for r in rows for v in r["buckets"].values()),
               "min_cos": min(v["min_cos"] for r in rows for v in r["buckets"].values()),
               "token_count_equal": all(r["tokens"] == r["provider_tokens"] for r in rows)}
    if a.fp64:
        summary["max_identity_fp64"] = max(v["fp64_vs_provider_fp64_max_abs"] for r in rows for v in r["buckets"].values())
        summary["max_provider_fp32_vs_fp64"] = max(r["provider_fp32_vs_fp64"] for r in rows)
        summary["max_own_fp32_vs_fp64"] = max(v["own_fp32_vs_fp64_max_abs"] for r in rows for v in r["buckets"].values())
    doc = {"what": "D1Audio vs the provider's get_audio_features", "snapshot": snap, "buckets": buckets,
           "summary": summary, "rows": rows, "wall_s": round(time.time() - t0, 1), "torch": torch.__version__}
    json.dump(doc, open(os.path.join(WORK, a.out), "w"), indent=1)
    print("AUDIO_CHECK", json.dumps(summary))


if __name__ == "__main__":
    main()
