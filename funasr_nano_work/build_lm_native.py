"""Build out/lm_native/: a plain Qwen3ForCausalLM checkpoint from the -vllm model.safetensors `llm.*`.

config = out/hf_vllm/config.json minus the audio/vLLM keys, architectures = Qwen3ForCausalLM.
llm.lm_head.weight == llm.model.embed_tokens.weight bit-for-bit (inspect_weights.py) -> tie_word_embeddings
true and the head is not saved. Weights stay BF16. Tokenizer files + generation_config.json copied.
Round-trip: reload in fp32, assert no missing/unexpected keys, head tied and equal to the source bytes.
Run with ~/venvs/lt094dev/bin/python.
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "hf_vllm")
DST = os.path.join(HERE, "out", "lm_native")
DROP_KEYS = ["audio_encoder", "audio_encoder_conf", "max_source_positions", "decoder_start_token_id"]
COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]


def main():
    os.makedirs(DST, exist_ok=True)
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    for k in DROP_KEYS:
        cfg.pop(k, None)
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    sd = {}
    with safe_open(os.path.join(SRC, "model.safetensors"), framework="pt") as f:
        head = f.get_tensor("llm.lm_head.weight")
        emb = f.get_tensor("llm.model.embed_tokens.weight")
        tied = bool(torch.equal(head.view(torch.int16), emb.view(torch.int16)))
        for k in f.keys():
            if k.startswith("llm.") and not (tied and k == "llm.lm_head.weight"):
                sd[k[len("llm."):]] = f.get_tensor(k).contiguous()
    cfg["tie_word_embeddings"] = tied
    with open(os.path.join(DST, "config.json"), "w") as fo:
        json.dump(cfg, fo, indent=2)
    save_file(sd, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})
    for fn in COPY:
        shutil.copy2(os.path.join(SRC, fn), os.path.join(DST, fn))
    print(f"saved {len(sd)} tensors, tie_word_embeddings={tied}")

    from transformers import Qwen3ForCausalLM
    lm, info = Qwen3ForCausalLM.from_pretrained(DST, dtype=torch.float32, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    assert not bad, bad
    assert lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr() or not tied
    assert torch.equal(lm.lm_head.weight, head.float()), "lm_head differs from the source bytes"
    assert torch.equal(lm.model.embed_tokens.weight, emb.float())
    rep = {"n_tensors_saved": len(sd), "tie_word_embeddings": tied, "loading_info": {k: list(v) for k, v in info.items()},
           "config": cfg, "files": sorted(os.listdir(DST))}
    with open(os.path.join(HERE, "lm_native_report.json"), "w") as fo:
        json.dump(rep, fo, indent=1)
    print("round-trip OK:", {k: len(v) for k, v in info.items()})


if __name__ == "__main__":
    main()
