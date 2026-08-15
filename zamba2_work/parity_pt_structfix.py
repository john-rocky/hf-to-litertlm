#!/usr/bin/env python3
"""pt parity leg for Zamba2 num_mem_blocks>1 checkpoints.

Plain transformers 5.14.1 cannot even CONSTRUCT a num_mem_blocks=2 Zamba2
(post_init raises at tie-key expansion; get_layers assigns block_id by global
layer index while the checkpoint and the tie cycle follow hybrid occurrence
order). This wrapper applies the structure-only fix from
model_ext/zamba2/patch.py to the plain class — the reference scan stays the
UNPATCHED upstream torch_forward, so the parity reference is still upstream
math — then runs scripts/parity_logits_bigmodel.py's pt leg unchanged.

  PYTHONPATH=~/code/litert-torch .venv-092/bin/python \
      zamba2_work/parity_pt_structfix.py --hf Zyphra/Zamba2-2.7B-instruct \
      --n 48 --out zamba2_work/parity_pt_27b.npz
"""
import os
import sys

from litert_torch.generative.export_hf.model_ext.zamba2 import patch as zpatch
from transformers.models.zamba2 import modeling_zamba2

modeling_zamba2.Zamba2Model.get_layers = zpatch._get_layers_hybrid_order

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import parity_logits_bigmodel  # noqa: E402

sys.argv = ["parity_logits_bigmodel.py", "pt"] + sys.argv[1:]
parity_logits_bigmodel.main()
