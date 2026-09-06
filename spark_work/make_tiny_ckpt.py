#!/usr/bin/env python3
"""Make a tiny random-weight Spark2_5 checkpoint (vendor code + real tokenizer)
to de-risk the export path before the real weights land. Point it at the REF dir
(patch_modeling.py --ref-only): the raw vendor file cannot be constructed under
transformers 5 (`_tied_weights_keys` list).

    ~/venvs/ltconv040dev/bin/python spark_work/make_tiny_ckpt.py <ref_dir> <tiny_dir>
"""
import json
import os
import shutil
import sys

import torch
import transformers

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
for name in ("configuration_spark.py", "modeling_spark.py", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
             "merges.txt", "chat_template.jinja", "generation_config.json"):
  if os.path.exists(os.path.join(SRC, name)):
    shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))
  else:
    print("WARN missing in vendor dir (copy it in by hand):", name)
cfg = json.load(open(os.path.join(SRC, "config.json")))
cfg.update({
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "sliding_window": 16, "max_position_embeddings": 4096,
})
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
config = transformers.AutoConfig.from_pretrained(DST, trust_remote_code=True)
torch.manual_seed(0)
model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
model.save_pretrained(DST, safe_serialization=True)
print("tiny checkpoint:", DST, "params:", sum(p.numel() for p in model.parameters()))
