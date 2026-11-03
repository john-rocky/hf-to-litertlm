#!/usr/bin/env python3
"""v4_text_lm.py — the text decoder of MiniCPM-V-4 as a registered transformers class (model_type minicpmv4_text).

The vendor model is LlamaForCausalLM with rope_type "longrope": short_factor == long_factor, factor 1.0 and
original_max_position_embeddings 32786 inside rope_scaling, so for every sequence the rotary uses
inv_freq = 1 / (short_factor * theta ** (arange(0, d, 2) / d)) and attention scaling 1.0 (transformers 4.51
_compute_longrope_parameters; the long/short switch only fires past 32786 positions). litert-torch export_hf refuses
rope types "dynamic" / "longrope" (core/export_lib.py verify_model_compatibility), and longrope's per-call update is a
data-dependent branch. This class keeps a default-type rotary and writes the short-factor inv_freq into it after the
weights load (and at construction), which is the same arithmetic without the branch. The derived config carries the
factors as `rope_short_factor` (make_text_dir.py)."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


class V4TextConfig(LlamaConfig):
    model_type = "minicpmv4_text"

    def __init__(self, rope_short_factor=None, **kwargs):
        self.rope_short_factor = rope_short_factor
        super().__init__(**kwargs)


def short_factor_inv_freq(config):
    sf = torch.tensor(config.rope_short_factor, dtype=torch.float32, device="cpu")
    d = config.head_dim
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = config.rope_parameters["rope_theta"]
    return 1.0 / (sf * base ** (torch.arange(0, d, 2, dtype=torch.int64, device="cpu").float() / d))


class V4TextForCausalLM(LlamaForCausalLM):
    config_class = V4TextConfig

    def __init__(self, config):
        super().__init__(config)
        self._set_rope()

    def _set_rope(self):
        re = self.model.rotary_emb
        if re.inv_freq.device.type == "meta":
            return None  # constructed on the meta device inside from_pretrained: set after the load (below)
        inv = short_factor_inv_freq(self.config).to(re.inv_freq.device)
        re.register_buffer("inv_freq", inv, persistent=False)
        re.register_buffer("original_inv_freq", inv.clone(), persistent=False)
        re.rope_type = "default"
        re.attention_scaling = 1.0
        return inv

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        m = super().from_pretrained(*args, **kwargs)
        inv = m._set_rope()
        print(f"V4TextForCausalLM: short-factor inv_freq set after load (first {inv[0].item():.6g}, "
              f"last {inv[-1].item():.6g})", flush=True)
        return m


def register():
    AutoConfig.register("minicpmv4_text", V4TextConfig)
    AutoModelForCausalLM.register(V4TextConfig, V4TextForCausalLM)
