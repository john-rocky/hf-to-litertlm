"""Reference re-build of the stock granite_speech attention_dists buffer.

The stock GraniteSpeechCTCEncoder registers attention_dists with
persistent=False, i.e. it is NOT in the checkpoint and is rebuilt from config
at init — the exact buffer class that the meta-device load path can silently
zero (tf5x-meta-load-zeroes-init-buffers). This mirrors the formula from
transformers.models.granite_speech.modeling_granite_speech (read from the
installed 5.14.1 source) so eager_gate.py can assert the loaded model's buffer
equals a fresh build.
"""

import torch


def fresh_attention_dists(config):
    enc = config.to_encoder_config() if hasattr(config, "to_encoder_config") else config
    seq = torch.arange(enc.context_size)
    relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
    return torch.clamp(relpos_dist, -enc.context_size, enc.context_size) + enc.max_pos_emb
