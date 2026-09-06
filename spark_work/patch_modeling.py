#!/usr/bin/env python3
"""Build an export-ready copy of a Spark-X2.5 checkpoint dir.

The vendor `modeling_spark.py` (XHToken/Spark-X2.5-*, model_type spark2_5, remote
code) computes attention through its own `eager_attention_forward` and never
looks at `config._attn_implementation`. litert-torch's export_hf path works by
setting `_attn_implementation = "lrt_transposed_attention"` and handing the model
its own KV cache (k/v come back from `past_key_values.update` in a transposed
layout that only the registered interface understands) plus per-call kwargs
(`k_ts_idx`, `v_ts_idx`) that must reach the attention call. So the vendor file
is patched, by exact-string replacement (each hit asserted unique), to:

  1. dispatch through `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`
     whenever a non-eager implementation is set (eager stays byte-identical to
     the vendor math -- parity_eager.py proves it), applying the per-head
     sigmoid output gate in the interface's [B, T, N, H] layout;
  2. thread `**kwargs` from ForCausalLM -> Model -> DecoderLayer -> Attention;
  3. declare `_supports_attention_backend` (transformers 5 refuses a custom
     attention interface without it), `_supports_sdpa`, `_can_compile_fullgraph`;
  4. mark sliding layers with `is_sliding` (read by the exporter's SDPA
     composites to pick the local/global kernel).

Everything else (rope, masks, MLP, tied lm_head) is the vendor code untouched:
the exporter passes `attention_mask` as the {"full_attention", "sliding_attention"}
dict the vendor forward already accepts, and rope is computed in-graph from
`cache_position`.

  0. (REF_FIX, applied in both modes) transformers-5 compatibility, two edits:
     `_tied_weights_keys` list -> mapping form (5.x `post_init` calls `.keys()`
     on it: `AttributeError: 'list' object has no attribute 'keys'`, the vendor
     model cannot even be constructed; the mapping is the 5.x spelling of the
     same tie, and the vendor forward uses `F.linear(h, embedding.weight)`
     anyway), and the mask-utility kwargs (`inputs_embeds`, no `cache_position`;
     the 4.5x names raise TypeError on the eager/generate path). The vendor file
     targets transformers 4.57.1; every venv here is 5.14.1.

    python3 spark_work/patch_modeling.py <vendor_ckpt_dir> <export_dir>            # full export patch
    python3 spark_work/patch_modeling.py <vendor_ckpt_dir> <ref_dir> --ref-only    # only REF_FIX: the parity reference

Weights (*.safetensors + index) are symlinked; configs/tokenizer files copied.
"""
import os
import shutil
import sys

SRC, DST = sys.argv[1], sys.argv[2]
REF_ONLY = "--ref-only" in sys.argv[3:]

# transformers 5.x compatibility (needed to construct / run the vendor model at all)
REF_FIX = [
    (
        "    _tied_weights_keys = [\"lm_head.weight\"]  # noqa: RUF012\n",
        "    _tied_weights_keys = {\"lm_head.weight\": \"model.embedding.weight\"}  # transformers 5 mapping form\n",
    ),
    # create_causal_mask / create_sliding_window_causal_mask in transformers 5.x take
    # `inputs_embeds` and no `cache_position` (the 4.5x names the vendor file uses raise
    # TypeError on the eager/generate path; the export path passes the mask dict and never
    # reaches this block).
    (
        "            mask_kwargs = {\n"
        "                \"config\": self.config,\n"
        "                \"input_embeds\": inputs_embeds,\n"
        "                \"attention_mask\": attention_mask,\n"
        "                \"cache_position\": cache_position,\n"
        "                \"past_key_values\": past_key_values,\n"
        "                \"position_ids\": position_ids,\n"
        "            }\n",
        "            mask_kwargs = {  # transformers 5.x signature\n"
        "                \"config\": self.config,\n"
        "                \"inputs_embeds\": inputs_embeds,\n"
        "                \"attention_mask\": attention_mask,\n"
        "                \"past_key_values\": past_key_values,\n"
        "                \"position_ids\": position_ids,\n"
        "            }\n",
    ),
]

REPLACEMENTS = [
    # 1. import the attention registry
    (
        "from transformers.modeling_utils import PreTrainedModel\n",
        "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel\n",
    ),
    # 2. keep the gate in [B, T, N, 1] (applied after attention in [B, T, N, H])
    (
        "        if gate_score is not None:\n"
        "            gate_score = gate_score.view(bsz, seq_len, self.num_heads, 1).transpose(1, 2)\n",
        "        if gate_score is not None:\n"
        "            # [B, T, N, 1]: applied after attention in the [B, T, N, H] layout.\n"
        "            gate_score = gate_score.view(bsz, seq_len, self.num_heads, 1)\n",
    ),
    # 3. attention dispatch + gate + merge heads
    (
        "        attn_output, attn_weights = eager_attention_forward(\n"
        "            self, q, k, v,\n"
        "            attention_mask=attention_mask,\n"
        "            scaling=self.scaling,\n"
        "            dropout=self.attention_dropout if self.training else 0.0,\n"
        "        )\n"
        "\n"
        "        if gate_score is not None:\n"
        "            if self.gate_attn_act_mode == \"sigmoid\":\n"
        "                gate = torch.sigmoid(gate_score.float())\n"
        "            elif self.gate_attn_act_mode == \"silu\":\n"
        "                gate = F.silu(gate_score.float())\n"
        "            else:\n"
        "                raise ValueError(f\"Unsupported gate_attn_act_mode: {self.gate_attn_act_mode}\")\n"
        "            gate = gate.to(attn_output.dtype)\n"
        "            attn_output = attn_output * gate\n"
        "\n"
        "        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)\n"
        "        attn_output = self.out_proj(attn_output)\n",
        "        attn_impl = getattr(self.config, \"_attn_implementation\", None) or \"eager\"\n"
        "        if attn_impl != \"eager\" and attn_impl in ALL_ATTENTION_FUNCTIONS:\n"
        "            # Registered attention interface (LiteRT export: lrt_transposed_attention;\n"
        "            # also sdpa). k/v arrive from the cache in the interface's own layout and\n"
        "            # the interface returns [B, T, N, H].\n"
        "            attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS[attn_impl](\n"
        "                self, q, k, v, attention_mask,\n"
        "                dropout=self.attention_dropout if self.training else 0.0,\n"
        "                scaling=self.scaling,\n"
        "                **kwargs,\n"
        "            )\n"
        "        else:\n"
        "            attn_output, attn_weights = eager_attention_forward(\n"
        "                self, q, k, v,\n"
        "                attention_mask=attention_mask,\n"
        "                scaling=self.scaling,\n"
        "                dropout=self.attention_dropout if self.training else 0.0,\n"
        "            )\n"
        "            attn_output = attn_output.transpose(1, 2)  # -> [B, T, N, H]\n"
        "\n"
        "        if gate_score is not None:\n"
        "            if self.gate_attn_act_mode == \"sigmoid\":\n"
        "                gate = torch.sigmoid(gate_score.float())\n"
        "            elif self.gate_attn_act_mode == \"silu\":\n"
        "                gate = F.silu(gate_score.float())\n"
        "            else:\n"
        "                raise ValueError(f\"Unsupported gate_attn_act_mode: {self.gate_attn_act_mode}\")\n"
        "            gate = gate.to(attn_output.dtype)\n"
        "            attn_output = attn_output * gate\n"
        "\n"
        "        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()\n"
        "        attn_output = self.out_proj(attn_output)\n",
    ),
    # 4. decoder layer: thread kwargs into attention, mark sliding layers
    (
        "            past_key_values=past_key_values,\n"
        "            cache_position=cache_position,\n"
        "            position_ids=position_ids,\n"
        "        )\n"
        "        hidden_states = residual + hidden_states\n",
        "            past_key_values=past_key_values,\n"
        "            cache_position=cache_position,\n"
        "            position_ids=position_ids,\n"
        "            **kwargs,\n"
        "        )\n"
        "        hidden_states = residual + hidden_states\n",
    ),
    (
        "        self.self_attn.partial_rotary_factor = config.get_partial_rotary_factor(self.layer_type)\n",
        "        self.self_attn.partial_rotary_factor = config.get_partial_rotary_factor(self.layer_type)\n"
        "        self.self_attn.is_sliding = self.layer_type == \"sliding_attention\"\n",
    ),
    # 5. model: thread kwargs into decoder layers
    (
        "                    attention_mask=layer_attention_mask,\n"
        "                    past_key_values=past_key_values,\n"
        "                    cache_position=cache_position,\n"
        "                    position_ids=position_ids,\n"
        "                )\n",
        "                    attention_mask=layer_attention_mask,\n"
        "                    past_key_values=past_key_values,\n"
        "                    cache_position=cache_position,\n"
        "                    position_ids=position_ids,\n"
        "                    **kwargs,\n"
        "                )\n",
    ),
    # 6. causal LM: thread kwargs into the model
    (
        "            inputs_embeds=inputs_embeds,\n"
        "            use_cache=use_cache,\n"
        "            cache_position=cache_position,\n"
        "        )\n",
        "            inputs_embeds=inputs_embeds,\n"
        "            use_cache=use_cache,\n"
        "            cache_position=cache_position,\n"
        "            **kwargs,\n"
        "        )\n",
    ),
    # 7. capability flags
    (
        "    _skip_keys_device_placement = [\"past_key_values\"]  # noqa: RUF012\n",
        "    _skip_keys_device_placement = [\"past_key_values\"]  # noqa: RUF012\n"
        "    _supports_attention_backend = True\n"
        "    _supports_sdpa = True\n"
        "    _supports_flash_attn = False\n"
        "    _can_compile_fullgraph = True\n",
    ),
]


def main():
  src = open(os.path.join(SRC, "modeling_spark.py"), encoding="utf-8").read()
  out = src
  edits = REF_FIX + ([] if REF_ONLY else REPLACEMENTS)
  for i, (old, new) in enumerate(edits, 1):
    n = out.count(old)
    if n == 0 and new in out:
      print(f"edit {i}: already applied, skipping")
      continue
    assert n == 1, f"replacement {i}: expected exactly 1 hit, got {n}"
    out = out.replace(old, new)
  os.makedirs(DST, exist_ok=True)
  if REF_ONLY:
    header = (
        "# REFERENCE copy made by spark_work/patch_modeling.py --ref-only: the vendor\n"
        "# file with ONLY the transformers-5 `_tied_weights_keys` mapping fix.\n"
    )
  else:
    header = (
        "# PATCHED for LiteRT export by spark_work/patch_modeling.py -- see that file\n"
        "# for the exact edits (attention-interface dispatch, kwargs threading,\n"
        "# capability flags, transformers-5 tie mapping). Math in eager mode is\n"
        "# identical to the vendor file.\n"
    )
  with open(os.path.join(DST, "modeling_spark.py"), "w", encoding="utf-8") as f:
    f.write(header + out)
  copied, linked = [], []
  for name in sorted(os.listdir(SRC)):
    if name == "modeling_spark.py" or name.startswith("."):
      continue
    s = os.path.join(SRC, name)
    d = os.path.join(DST, name)
    if os.path.isdir(s):
      continue
    if os.path.lexists(d):
      os.remove(d)
    if name.endswith(".safetensors"):
      os.symlink(os.path.abspath(s), d)
      linked.append(name)
    else:
      shutil.copy2(s, d)
      copied.append(name)
  print(f"patched modeling -> {DST}/modeling_spark.py ({len(edits)} edits, ref_only={REF_ONLY})")
  print(f"copied: {copied}")
  print(f"symlinked: {linked}")


if __name__ == "__main__":
  main()
