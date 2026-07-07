"""Dumps talker reference activations for the cross-env equivalence check.

Feeds a fixed random embedding sequence through the raw talker (prefill
style, no cache) and stores logits + hidden states for verify_talker.py.

Usage (reference env):
    python dump_talker_ref.py
"""

import os

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main() -> None:
    torch.manual_seed(123)
    model = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu',
        dtype=torch.float32).model
    talker = model.talker

    x = torch.randn(1, 24, 1024, dtype=torch.float32) * 0.02
    with torch.no_grad():
        hidden = talker.model(inputs_embeds=x, use_cache=False)
        hidden = hidden.last_hidden_state
        logits = talker.codec_head(hidden)

    os.makedirs('ref', exist_ok=True)
    np.savez('ref/talker_equiv_ref.npz', x=x.numpy(), hidden=hidden.numpy(),
             logits=logits.numpy())
    print('dumped', x.shape, hidden.shape, logits.shape)


if __name__ == '__main__':
    main()
