#!/usr/bin/env python3
"""Load microsoft/VibeVoice-ASR-BitNet (legacy `VibeVoiceForASRTraining` layout,
fp32 latent weights) into the transformers-native `VibeVoiceAsrForConditionalGeneration`
and save two checkpoints the rest of the lane consumes:

  hf_native/  full ASR model (acoustic + semantic encoders, projector, Qwen2 LM, lm_head)
  lm_native/  the Qwen2ForCausalLM alone (what export_hf converts to prefill/decode)

Two things happen on the way, both taken from VibeASR.cpp (the official runtime):

1. Rename.  The Hub checkpoint uses the `vibevoice` package layout
   (`model.acoustic_tokenizer.encoder.downsample_layers.N.0.conv.conv`,
   `stages.N.j.mixer.conv.conv.conv`, `acoustic_connector.fc1/norm/fc2`, ...); the
   native class uses `stem/conv_layers/head`, `mixer.conv`, `multi_modal_projector.
   acoustic_linear_1/acoustic_norm/acoustic_linear_2`.  The decoder (VAE synthesis half)
   is dropped: ASR only needs the encoders.  Every rename is a full-key rewrite and the
   result is loaded with strict=True (missing == unexpected == []).

2. Ternarize.  "BitNet" here means the 7 projection weights per LM layer (q/k/v/o,
   gate/up/down) were trained with the BitNet b1.58 fake-quant; the checkpoint stores
   the fp32 latent weights.  VibeASR.cpp's `convert_lm_to_gguf.py` quantizes them at
   conversion time with a per-TENSOR absmean scale:
       s = 1 / mean(|w|).clamp(min=1e-5);  w_t = round(w*s).clamp(-1, 1) / s
   We apply exactly that here, so the saved LM holds {-a, 0, +a} per tensor (a = mean|w|).
   Embedding, lm_head, norms, biases stay full precision (the runtime keeps embeddings at
   Q6_K and never ternarizes the head).

Usage:
  python load_bitnet.py [--src hf] [--out .]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BITNET_KEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
               "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def ternarize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    w = w.float()
    s = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * s).round().clamp(-1, 1) / s, float(1.0 / s)


def rename(key: str) -> str | None:
    """Legacy key -> native key (None = drop)."""
    m = re.match(r"model\.(acoustic|semantic)_tokenizer\.(encoder|decoder)\.(.*)", key)
    if m:
        kind, half, rest = m.groups()
        if half == "decoder":
            return None
        base = f"model.{kind}_tokenizer_encoder."
        mm = re.match(r"downsample_layers\.(\d+)\.0\.conv\.conv\.(weight|bias)$", rest)
        if mm:
            i, p = int(mm.group(1)), mm.group(2)
            return base + (f"stem.conv.conv.{p}" if i == 0 else f"conv_layers.{i-1}.conv.conv.{p}")
        mm = re.match(r"stages\.(\d+)\.(\d+)\.(.*)", rest)
        if mm:
            s, j, tail = int(mm.group(1)), int(mm.group(2)), mm.group(3)
            tail = tail.replace("mixer.conv.conv.conv.", "mixer.conv.")
            stage = "stem.stage" if s == 0 else f"conv_layers.{s-1}.stage"
            return base + f"{stage}.{j}.{tail}"
        mm = re.match(r"head\.conv\.conv\.(weight|bias)$", rest)
        if mm:
            return base + f"head.conv.{mm.group(1)}"
        raise KeyError(f"unmapped tokenizer key: {key}")
    m = re.match(r"model\.(acoustic|semantic)_connector\.(fc1|fc2|norm)\.(weight|bias)$", key)
    if m:
        kind, part, p = m.groups()
        part = {"fc1": f"{kind}_linear_1", "fc2": f"{kind}_linear_2", "norm": f"{kind}_norm"}[part]
        return f"model.multi_modal_projector.{part}.{p}"
    if key.startswith("model.language_model.") or key == "lm_head.weight":
        return key
    raise KeyError(f"unmapped key: {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hf")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    src = args.src
    cfg = json.load(open(os.path.join(src, "config.json")))

    from transformers import (Qwen2Config, VibeVoiceAsrConfig, VibeVoiceAsrForConditionalGeneration,
                              VibeVoiceAcousticTokenizerEncoderConfig, Qwen2ForCausalLM)

    def enc_cfg(legacy: dict) -> VibeVoiceAcousticTokenizerEncoderConfig:
        depths = [int(x) for x in legacy["encoder_depths"].split("-")]
        ratios = list(reversed(legacy["encoder_ratios"]))  # legacy lists them decoder-order
        assert legacy["layernorm"] == "RMSNorm" and legacy["mixer_layer"] == "depthwise_conv"
        assert legacy["causal"] is True and legacy["pad_mode"] == "constant"
        return VibeVoiceAcousticTokenizerEncoderConfig(
            channels=legacy["channels"], hidden_size=legacy["vae_dim"],
            num_filters=legacy["encoder_n_filters"], depths=depths, downsampling_ratios=ratios,
            kernel_size=7, rms_norm_eps=legacy["layernorm_eps"],
            layer_scale_init_value=legacy["layer_scale_init_value"], hidden_act="gelu",
            ffn_expansion=4, vae_std=legacy.get("fix_std", 0.0) or 0.0)

    acoustic = enc_cfg(cfg["acoustic_tokenizer_config"])
    semantic = enc_cfg(cfg["semantic_tokenizer_config"])
    dec = dict(cfg["decoder_config"])
    for k in ("dtype", "torch_dtype", "_attn_implementation_autoset"):
        dec.pop(k, None)
    assert dec.pop("model_type") == "qwen2"

    # ---- read every tensor once, rename, ternarize ----
    t0 = time.time()
    shards = sorted(f for f in os.listdir(src) if re.match(r"model-\d+-of-\d+\.safetensors$", f))
    assert shards, "no safetensors shards in " + src
    sd, tern_stats, dropped = {}, {}, 0
    for sh in shards:
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            for k in f.keys():
                nk = rename(k)
                if nk is None:
                    dropped += 1
                    continue
                t = f.get_tensor(k)
                assert t.dtype == torch.float32, (k, t.dtype)
                if nk.startswith("model.language_model.layers.") and nk.endswith(BITNET_KEYS):
                    t, alpha = ternarize(t)
                    u = torch.unique(t)
                    assert u.numel() <= 3, (nk, u.numel())
                    tern_stats[nk] = {"alpha": alpha, "zero_frac": float((t == 0).float().mean())}
                sd[nk] = t.contiguous()
    print(f"read {len(sd)} tensors (+{dropped} decoder tensors dropped), "
          f"{len(tern_stats)} ternarized, {time.time()-t0:.0f}s")

    # lm_head vs embed_tokens: the legacy config says tie_word_embeddings=true but the
    # checkpoint carries both; decide from the bytes, never from the flag.
    tied = torch.equal(sd["lm_head.weight"], sd["model.language_model.embed_tokens.weight"])
    print("lm_head == embed_tokens:", tied)
    dec["tie_word_embeddings"] = tied
    text = Qwen2Config(**dec)

    config = VibeVoiceAsrConfig(
        acoustic_tokenizer_encoder_config=acoustic, semantic_tokenizer_encoder_config=semantic,
        text_config=text, audio_token_id=151648, audio_bos_token_id=151646,
        audio_eos_token_id=151647, acoustic_tokenizer_chunk_size=1440000)
    # Direct construction (no meta device): __init__-computed buffers such as the
    # rotary inv_freq must be real, not to_empty() garbage.
    model = VibeVoiceAsrForConditionalGeneration(config)
    # Load lm_head EXPLICITLY even though it equals embed_tokens: the composite class's
    # tie_weights() keys off the top-level config (no tie flag there) and silently leaves a
    # randomly-initialised head — first run of this lane shipped a random lm_head into
    # hf_native/ and every transcript was token soup (flat top-5 ~0.002).
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print("missing:", missing, "unexpected:", unexpected)
    assert torch.equal(model.lm_head.weight, model.model.language_model.embed_tokens.weight) == tied
    model = model.float().eval()

    # ---- save full native model ----
    out_full = os.path.join(args.out, "hf_native")
    os.makedirs(out_full, exist_ok=True)
    model.save_pretrained(out_full, safe_serialization=True, max_shard_size="5GB")
    # Round-trip proof: the saved checkpoint must load with the SAME head bytes.
    chk = VibeVoiceAsrForConditionalGeneration.from_pretrained(out_full, dtype=torch.float32)
    assert torch.equal(chk.lm_head.weight, model.lm_head.weight), "lm_head did not round-trip"
    assert torch.equal(chk.model.language_model.layers[0].mlp.down_proj.weight,
                       model.model.language_model.layers[0].mlp.down_proj.weight)
    del chk
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            os.system(f"cp '{p}' '{out_full}/'")
    json.dump({"source": "microsoft/VibeVoice-ASR-BitNet", "tied_lm_head": tied,
               "ternarized": len(tern_stats), "alpha_min": min(v["alpha"] for v in tern_stats.values()),
               "alpha_max": max(v["alpha"] for v in tern_stats.values()),
               "zero_frac_mean": sum(v["zero_frac"] for v in tern_stats.values()) / len(tern_stats),
               "per_tensor": tern_stats},
              open(os.path.join(args.out, "ternarize_report.json"), "w"), indent=1)
    print("saved", out_full)

    # ---- save LM-only checkpoint for export_hf ----
    out_lm = os.path.join(args.out, "lm_native")
    os.makedirs(out_lm, exist_ok=True)
    lm = Qwen2ForCausalLM(text)
    lm_sd = {k.replace("model.language_model.", "model."): v for k, v in model.state_dict().items()
             if k.startswith("model.language_model.")}
    lm_sd["lm_head.weight"] = model.lm_head.weight.data
    if tied:
        lm_sd.pop("lm_head.weight")
    missing, unexpected = lm.load_state_dict(lm_sd, strict=False)
    if tied:
        missing = [m for m in missing if m != "lm_head.weight"]
        lm.tie_weights()
    assert not missing and not unexpected, (missing, unexpected)
    lm.generation_config.eos_token_id = [151645, 151643]
    lm.generation_config.pad_token_id = 151643
    lm.save_pretrained(out_lm, safe_serialization=True, max_shard_size="5GB")
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        os.system(f"cp '{os.path.join(src, fn)}' '{out_lm}/'")
    print("saved", out_lm)
    print("LOAD_DONE")


if __name__ == "__main__":
    main()
