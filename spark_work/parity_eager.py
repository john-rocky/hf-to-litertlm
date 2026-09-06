#!/usr/bin/env python3
"""Prove the export patch is math-identical to the vendor modeling code.

Loads the VENDOR dir (eager, remote code) and the PATCHED export dir under
`attn_implementation` eager and sdpa, runs the same token ids (fp32, CPU),
and reports max |logit diff| + argmax agreement. Also checks lm_head tying.

    ~/venvs/ltconv040dev/bin/python spark_work/parity_eager.py <vendor_dir> <export_dir> [L]
"""
import sys

import numpy as np
import torch
import transformers

VENDOR, PATCHED = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 24
torch.manual_seed(0)


def load(path, impl):
  m = transformers.AutoModelForCausalLM.from_pretrained(
      path, dtype=torch.float32, trust_remote_code=True, attn_implementation=impl
  ).eval()
  return m


def logits(m, ids):
  with torch.no_grad():
    return m(torch.tensor(ids)[None, :], use_cache=False).logits[0].float().numpy()


ref = load(VENDOR, "eager")
vocab = ref.config.vocab_size
ids = np.random.default_rng(0).integers(0, vocab, size=(L,)).tolist()
print(f"vendor: {type(ref).__module__}  attn={ref.config._attn_implementation}")
ref_l = logits(ref, ids)
del ref

for impl in ("eager", "sdpa"):
  m = load(PATCHED, impl)
  tied = m.lm_head.weight.data_ptr() == m.model.embedding.weight.data_ptr()
  l = logits(m, ids)
  d = np.abs(l - ref_l).max()
  agree = (l.argmax(-1) == ref_l.argmax(-1)).mean()
  print(f"patched[{impl}]: module={type(m).__module__} attn={m.config._attn_implementation} "
        f"lm_head_tied={tied} max|dlogit|={d:.3e} argmax_agree={agree:.3f}")
  del m
