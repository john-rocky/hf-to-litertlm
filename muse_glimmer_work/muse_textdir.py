#!/usr/bin/env python3
"""muse_textdir.py OUT_DIR --vlm-config CONFIG.json --files-from DIR [--weights-from DIR]
                  [--layers N | --pattern sliding,full,...] [--vocab V] [--random-ckpt SEED]

Writes a text-only model directory for muse_text_lm.py (MuseGlimmerTextForCausalLM):
config.json = the VLM config's text_config (+ architectures), and symlinks to the tokenizer
and template files of --files-from. Weights:
  --weights-from DIR   symlink the VLM checkpoint's shards and index (names already match);
  --random-ckpt SEED   save a randomly initialised checkpoint of the (truncated) config, for
                       a quick check: the export and the HF reference then read the same numbers.
--layers keeps the first N layers (types and per-layer rope theta follow); --pattern sets the
layer types directly (rope theta 0 = NoPE for "full_attention", as in the checkpoint).
--vocab overrides the vocabulary size (quick checks only; the tokenizer is not resized).
"""
import argparse
import json
import os
import shutil
import sys

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "generation_config.json"]


def prescale_config(out, tc):
    """Marks the text dir for the prescaled head: MuseGlimmerTextForCausalLM.from_pretrained multiplies lm_head.weight
    by c = output_multiplier / final_logit_softcapping (fp32) after loading, and the head is T * tanh(x W'^T), which is
    the VLM head T * tanh(x W^T * mult / T). Folding the two scalars into W is what the default export path does by
    itself; the lightweight path cannot fold a lazy weight and leaves MUL(W, c) in front of the FC (then the FC is not
    quantized and the multiply runs every step)."""
    import numpy as np
    c = np.float32(tc["output_multiplier"]) / np.float32(tc["final_logit_softcapping"])
    cfg_path = os.path.join(out, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["lm_head_prescaled"] = True
    cfg["lm_head_prescale_factor"] = float(c)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"lm_head prescale factor {float(c):.9g} (applied at load)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vlm-config", required=True)
    ap.add_argument("--files-from", required=True)
    ap.add_argument("--weights-from")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--pattern")
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--random-ckpt", type=int)
    ap.add_argument("--prescale-lm-head", action="store_true",
                    help="mark the config so lm_head.weight is scaled by output_multiplier / final_logit_softcapping "
                         "after loading (muse_text_lm.py)")
    a = ap.parse_args()
    if bool(a.weights_from) == (a.random_ckpt is not None):
        sys.exit("give exactly one of --weights-from / --random-ckpt")

    vlm = json.load(open(a.vlm_config))
    tc = dict(vlm["text_config"])
    rope_theta = tc["rope_parameters"]["rope_theta"]
    if a.pattern:
        types = a.pattern.split(",")
        tc["layer_types"] = types
        tc["layer_rope_theta"] = [rope_theta if t == "sliding_attention" else 0 for t in types]
        tc["num_hidden_layers"] = len(types)
    elif a.layers:
        tc["num_hidden_layers"] = a.layers
        tc["layer_types"] = tc["layer_types"][:a.layers]
        tc["layer_rope_theta"] = tc["layer_rope_theta"][:a.layers]
    if a.vocab:
        tc["vocab_size"] = a.vocab
    tc["architectures"] = ["MuseGlimmerTextForCausalLM"]
    tc["dtype"] = vlm.get("dtype", "bfloat16")
    tc["transformers_version"] = vlm.get("transformers_version")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(tc, f, indent=2)
    if a.weights_from:
        for name in sorted(os.listdir(a.weights_from)):
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                dst = os.path.join(a.out, name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(os.path.join(a.weights_from, name)), dst)
        if a.prescale_lm_head:
            prescale_config(a.out, tc)
    else:
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "muse_text_lm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "muse_text_lm.py"))
        muse_text_lm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(muse_text_lm)
        from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
        torch.manual_seed(a.random_ckpt)
        cfg = MuseGlimmerTextConfig(**{k: v for k, v in tc.items() if k != "architectures"})
        model = muse_text_lm.MuseGlimmerTextForCausalLM(cfg).float()
        # post_init leaves the centered norms at 0 (= scale 1); give every norm and weight
        # a nonzero random value so a wrong norm form or a dropped term shows in the logits.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 1:
                    p.copy_(torch.randn_like(p) * 0.1)
                else:
                    p.copy_(torch.randn_like(p) * (p.shape[-1] ** -0.5))
        model.save_pretrained(a.out, safe_serialization=True)
        # save_pretrained rewrote config.json from cfg; keep our architectures line
        with open(os.path.join(a.out, "config.json")) as f:
            saved = json.load(f)
        saved["architectures"] = ["MuseGlimmerTextForCausalLM"]
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(saved, f, indent=2)
    # Tokenizer and template files are copied, after the weights step: a symlink here once let
    # save_pretrained write a test model's generation_config.json through to the source copy.
    for name in TOKENIZER_FILES:
        src = os.path.join(a.files_from, name)
        dst = os.path.join(a.out, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            shutil.copy2(os.path.realpath(src), dst)
    print(f"text dir {a.out}: layers {tc['num_hidden_layers']} {tc['layer_types']} vocab {tc['vocab_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
