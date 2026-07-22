"""Fix for LFM2/2.5 ShortConv conv-state corruption on padded prefill chunks.

The LiteRT-LM runtime right-pads a prefill chunk whenever the prompt does not
exactly fill a prefill signature (pad tokens = 0, pad attention-mask rows left
at the all-blocked fill value: -0.7*FLT_MAX, or -45824 in f16 mode). Attention
masks the pads out, but a causal conv cannot: the stock exporter saves
`next_state = padded_input[:, :, -(L_cache-1):]` — the PAD columns — so the
state handed to the next chunk / decode step is garbage and the first generated
token of nearly every reply is corrupted (e.g. "to is a vast..." instead of
"The ocean is..."). Smaller LFMs may mask the symptom; LFM2.5-1.2B flips
visibly.

LFM2's conv layers receive the raw 4D additive mask during prefill (HF
Lfm2Model passes `attention_mask` through as `linear_attention`; it is None
during decode, where padding cannot occur). Real rows contain 0.0 entries; pad
rows stay at the fill value — so a row is valid iff its max is ~0. This patch
zeroes the pad columns and gathers the conv state from the last (L_cache - 1)
VALID columns. Decode keeps the stock fast path. With the patch, engine output
is token-identical to an exact per-token decode loop at every prompt length.

Usage: import and call `apply_patch()` BEFORE running the litert-torch export.
"""
import torch


def apply_patch():
    from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv
    from transformers.models.lfm2 import modeling_lfm2

    def forward(self, hidden_states, past_key_values=None, cache_position=None,
                attention_mask=None):
        x = modeling_lfm2.apply_mask_to_padding_states(
            hidden_states, attention_mask)
        b, c, x_proj = self.in_proj(x).chunk(3, dim=-1)
        conv_input = b * x_proj
        valid_len = None
        seq_len = conv_input.shape[1]
        if (attention_mask is not None and attention_mask.dim() == 4
                and attention_mask.shape[2] == seq_len):
            row_max = attention_mask[0, 0].amax(dim=-1)
            valid = row_max > -1000.0
            conv_input = conv_input * valid[None, :, None].to(conv_input.dtype)
            valid_len = valid.to(torch.int32).sum()
        conv_input_t = conv_input.transpose(1, 2)
        state = past_key_values.layers[self.layer_idx].conv_state
        padded_input = torch.cat([state, conv_input_t], dim=-1)
        if valid_len is None:
            next_state = padded_input[:, :, -(self.L_cache - 1):]
        else:
            state_idx = valid_len + torch.arange(
                self.L_cache - 1, dtype=torch.int32)
            next_state = padded_input.index_select(-1, state_idx)
        conv_out = self.conv(padded_input)
        conv_out = conv_out.transpose(1, 2)
        y = c * conv_out
        y = self.out_proj(y)
        past_key_values.layers[self.layer_idx].conv_state = next_state
        return y

    short_conv.Lfm2ShortConv.forward = forward
    print("LFM2 ShortConv padded-prefill state fix applied.")
