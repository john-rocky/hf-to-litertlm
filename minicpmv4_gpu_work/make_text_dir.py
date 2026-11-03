#!/usr/bin/env python3
"""make_text_dir.py SRC OUT [--prove REF_DIR]
Writes the text decoder of openbmb/MiniCPM-V-4 as its own checkpoint dir: llm.model.* -> model.*, llm.lm_head.weight ->
lm_head.weight (bf16 as published, no other change), config model_type minicpmv4_text (v4_text_lm.py; the
vendor's Llama fields, the rope as a default type plus rope_short_factor), the repo's tokenizer.json verbatim, and a
tokenizer_config.json without the remote-code auto_map. --prove loads OUT with the registered class and compares its
logits with the vendor's own llm on probe sequences saved from a PyTorch reference run (REF_DIR/probe_*.pt; that
reference script is not in this directory, and the option is not needed for the conversion)."""
import argparse
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod():
    spec = importlib.util.spec_from_file_location("v4_text_lm", os.path.join(HERE, "v4_text_lm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    rs = cfg["rope_scaling"]
    assert rs["rope_type"] == "longrope" and rs["short_factor"] == rs["long_factor"] and rs["factor"] == 1.0
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "rms_norm_eps", "rope_theta",
            "attention_bias", "attention_dropout", "mlp_bias", "tie_word_embeddings", "bos_token_id", "eos_token_id",
            "pad_token_id", "pretraining_tp", "initializer_range"]
    out = {k: cfg[k] for k in keep}
    out.update({"architectures": ["V4TextForCausalLM"], "model_type": "minicpmv4_text", "rope_scaling": None,
                "rope_short_factor": rs["short_factor"], "torch_dtype": "bfloat16",
                "_source": "openbmb/MiniCPM-V-4@6f69a8235885be89608f3a71ffd40695b379e1e7 (llm.*)"})
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for shard in sorted(set(idx.values())):
        with safe_open(os.path.join(a.src, shard), "pt") as f:
            for name in f.keys():
                if name.startswith("llm."):
                    tensors[name[len("llm."):]] = f.get_tensor(name)
    assert "lm_head.weight" in tensors and "model.embed_tokens.weight" in tensors, sorted(tensors)[:5]
    save_file(tensors, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {len(tensors)} tensors", flush=True)
    shutil.copyfile(os.path.join(a.src, "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
    tc = json.load(open(os.path.join(a.src, "tokenizer_config.json")))
    tc.pop("auto_map", None)
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(tc, open(os.path.join(a.out, "tokenizer_config.json"), "w"), indent=1, ensure_ascii=False)
    for f in ("special_tokens_map.json", "generation_config.json"):
        shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))


def prove(a):
    import torch
    mod = load_mod()
    mod.register()
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.out, torch_dtype=torch.float32).eval()
    res = {}
    for name in sorted(os.listdir(a.prove)):
        if not name.startswith("probe_"):
            continue
        p = torch.load(os.path.join(a.prove, name))
        with torch.no_grad():
            got = m(input_ids=p["ids"][None]).logits[0]
        want = p["logits"]
        d = (got - want).abs().max().item()
        top = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
        res[name] = {"positions": int(p["ids"].shape[0]), "max_abs_dlogit": d, "top1_agree": top}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(a.out, "prove.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--prove")
    ap.add_argument("--skip-write", action="store_true")
    a = ap.parse_args()
    if not a.skip_write:
        write(a)
    if a.prove:
        prove(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
