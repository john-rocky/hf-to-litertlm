"""Tiny random LFM3: the exporter's module vs HF fp32 on every runtime path.

    PYTHONPATH=<lfm3_work> <venv>/bin/python -m lfm3_ext.tiny_parity_test <code_dir>

`code_dir` = any LFM3 repo dir (its `modeling_lfm3.py`). Six layers (conv,
conv, full, conv, sliding, conv), window 8, 41 tokens, chunk 16: the sliding
mask, the zero-tail partial rotary, the width-7 conv state across calls, a
padded prefill call whose state the decode reads, and (embeddings input) pad
slots holding noise. Prints one line per path and exits 1 on a miss.
"""

import copy
import sys

import torch

from . import harness
from . import loader


def tiny_model(code_dir, seed=0):
  config = loader.load_lfm3_config(
      code_dir,
      {
          "model_type": "lfm3",
          "vocab_size": 512,
          "hidden_size": 64,
          "intermediate_size": 96,
          "num_hidden_layers": 6,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "conv_L_cache": 7,
          "sliding_window": 8,
          "layer_types": [
              "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
          ],
          "max_position_embeddings": 4096,
          "pad_token_id": 3,
          "bos_token_id": 1,
          "eos_token_id": 2,
      },
  )
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  torch.manual_seed(seed)
  model_cls = loader._remote_class(code_dir, "modeling_lfm3.Lfm3ForCausalLM")  # pylint: disable=protected-access
  model = model_cls(config).float().eval()
  # Non-trivial norm and conv weights (init is ones / small).
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("norm.weight"):
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
      elif name.endswith("conv.conv.weight"):
        p.copy_(0.5 * torch.randn_like(p))
  return model


def main(code_dir):
  ref_model = tiny_model(code_dir)
  ids = torch.randint(4, 512, (41,), generator=torch.Generator().manual_seed(1)).tolist()
  with torch.no_grad():
    ref = ref_model(torch.tensor([ids])).logits[0].float()
  failed = False
  for embeddings in (False, True):
    model = copy.deepcopy(ref_model)
    config = harness.make_export_config(64, externalize_embedder=embeddings)
    with harness.patched(model, config):
      runner = harness.Runner(model, config, pad_token_id=3)
      paths = harness.run_paths(runner, ids, chunk=16, split=29)
    for name, logits in paths.items():
      r = harness.compare(ref, logits)
      ok = r["max_abs_dlogit"] <= 1e-4 and r["top1_agree"] == r["top1_of"] and r["finite"]
      failed |= not ok
      print(
          f"{'emb' if embeddings else 'tok'} {name:24s} max|d|={r['max_abs_dlogit']:.2e}"
          f" top1 {r['top1_agree']}/{r['top1_of']} (near-tie {r['excluded_near_tie']})"
          f" {'OK' if ok else 'FAIL'}"
      )
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1]))
