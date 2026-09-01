"""Load granite-speech-5.0-470m-turboctc into the repo's own remote-code module.

The Hub repo is two-faced (2026-09-01): config.json/model.safetensors were
re-exported for the STOCK transformers >=5.16.0.dev0 `GraniteSpeech5ForCTC`
(model_type granite_speech5_ctc, HF-standard tensor names, no auto_map), while
the shipped remote-code .py files are the older packaged `CtcConformerForCTC`
(model_type ctc_conformer, stock-3.3 granite_speech tensor names). Our venvs
carry transformers 5.14.1 = stock class unavailable, so we load through the
remote-code module with a full-key rename of the state dict.

Rename table (checkpoint name -> remote module name), applied per full key in
one pass (NO cascading — `conv.norm.*` (BatchNorm, 2048) and `norm_conv.*`
(LayerNorm, 1024) would collide under sequential prefix rewriting):

  self_attn.q_proj -> attn.to_q          norm_self_att   -> attn.pre_norm
  self_attn.k_proj -> attn.to_k          norm_feed_forward1 -> ff1.pre_norm
  self_attn.v_proj -> attn.to_v          norm_feed_forward2 -> ff2.pre_norm
  self_attn.o_proj -> attn.to_out        norm_conv       -> conv.norm
  self_attn.rel_pos_emb -> attn.rel_pos_emb
  feed_forward1.linear1 -> ff1.up_proj   feed_forward1.linear2 -> ff1.down_proj
  feed_forward2.linear1 -> ff2.up_proj   feed_forward2.linear2 -> ff2.down_proj
  conv.depthwise_conv -> conv.depth_conv.conv
  conv.norm -> conv.batch_norm           norm_out        -> post_norm
  (pointwise_lin1/2, input_linear, out, out_mid unchanged)

load_state_dict(strict=True) over all 550 tensors is the structural proof that
the mapping is complete; the WER gate is the functional proof.
"""

import json
import re
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).parent / "hf_model"

_RULES = [
    (r"\.self_attn\.q_proj\.", ".attn.to_q."),
    (r"\.self_attn\.k_proj\.", ".attn.to_k."),
    (r"\.self_attn\.v_proj\.", ".attn.to_v."),
    (r"\.self_attn\.o_proj\.", ".attn.to_out."),
    (r"\.self_attn\.rel_pos_emb\.", ".attn.rel_pos_emb."),
    (r"\.norm_self_att\.", ".attn.pre_norm."),
    (r"\.feed_forward1\.linear1\.", ".ff1.up_proj."),
    (r"\.feed_forward1\.linear2\.", ".ff1.down_proj."),
    (r"\.feed_forward2\.linear1\.", ".ff2.up_proj."),
    (r"\.feed_forward2\.linear2\.", ".ff2.down_proj."),
    (r"\.norm_feed_forward1\.", ".ff1.pre_norm."),
    (r"\.norm_feed_forward2\.", ".ff2.pre_norm."),
    (r"\.conv\.depthwise_conv\.", ".conv.depth_conv.conv."),
    (r"\.conv\.norm\.", ".conv.batch_norm."),
    (r"\.norm_conv\.", ".conv.norm."),
    (r"\.norm_out\.", ".post_norm."),
]


def _rename(key):
    for pat, rep in _RULES:
        new, n = re.subn(pat, rep, key)
        if n:
            return new  # first matching rule wins; exactly one applies per key
    return key


def _assert_config_matches_hub(cfg):
    """Bind the remote-code config defaults to the Hub's stock-format config.json."""
    hub = json.load(open(MODEL_DIR / "config.json"))
    enc = hub["encoder_config"]
    checks = {
        "hidden_dim": enc["hidden_size"],
        "num_layers": enc["num_hidden_layers"],
        "num_heads": enc["num_attention_heads"],
        "dim_head": enc["head_dim"],
        "feedforward_mult": enc["intermediate_size"] // enc["hidden_size"],
        "context_size": enc["context_size"],
        "max_pos_emb": enc["max_position_embeddings"],
        "conv_kernel_size": enc["conv_kernel_size"],
        "conv_expansion_factor": enc["conv_expansion_factor"],
        "output_dim": hub["vocab_size"],
        "n_mels": enc["num_mel_bins"],
        "subsample_layers": tuple(enc["subsample_layers"]),
    }
    for field, hub_val in checks.items():
        got = getattr(cfg, field)
        assert got == hub_val, f"config.{field}: remote default {got} != hub {hub_val}"


def load_eager(dtype=torch.float32):
    """Returns (model.eval() in `dtype`, processor). Offline, local files only."""
    from safetensors.torch import load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ref = str(MODEL_DIR)
    model_cls = get_class_from_dynamic_module(
        "modeling_ctc_conformer.CtcConformerForCTC", ref)
    config_cls = get_class_from_dynamic_module(
        "configuration_ctc_conformer.CtcConformerConfig", ref)
    proc_cls = get_class_from_dynamic_module(
        "processing_ctc_conformer.CtcConformerProcessor", ref)

    cfg = config_cls()  # defaults documented to mirror the trained model
    _assert_config_matches_hub(cfg)

    model = model_cls(cfg)
    sd = load_file(MODEL_DIR / "model.safetensors")
    renamed = {}
    for k, v in sd.items():
        nk = _rename(k)
        assert nk not in renamed, f"rename collision on {nk} (from {k})"
        renamed[nk] = v.to(dtype) if v.is_floating_point() else v
    model.load_state_dict(renamed, strict=True)
    model = model.to(dtype).eval()

    processor = proc_cls.from_pretrained(ref)
    return model, processor
