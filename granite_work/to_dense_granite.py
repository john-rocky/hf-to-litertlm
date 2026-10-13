#!/usr/bin/env python3
"""to_dense_granite.py SRC_DIR OUT_DIR REPORT.json - rewrite ibm-granite/granite-4.0-micro (a granitemoehybrid
checkpoint with 40 attention layers and 0 experts) as a dense GraniteForCausalLM checkpoint. Weights are moved, never
recomputed: every output tensor is a byte copy of a source tensor or of one half of it.

Why: litert-torch's generic export path loads the model with AutoModelForCausalLM; granite-4.1-3b / 4.2-3b shipped
through that path as `granite` (GraniteForCausalLM). granite-4.0-micro has the same numbers (40L, hidden 2560, GQA 40:8,
vocab 100352, tied, multipliers 0.015625 / 12 / 10 / 0.22, rope_theta 1e7) under another model_type and other weight
names. transformers 5.14.1 `modeling_granitemoehybrid.py`:
  - GraniteMoeHybridMLP.forward L793-797: `hidden_states = self.input_linear(hidden_states)`;
    `chunked_hidden_states = hidden_states.chunk(2, dim=-1)`;
    `hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]`; `self.output_linear(...)`.
    input_linear = nn.Linear(hidden, 2 * shared_intermediate_size) -> weight [2*8192, 2560]; output feature j < 8192
    comes from weight row j -> chunk [0] (the one through the activation) = rows [0, 8192) = gate_proj, rows
    [8192, 16384) = up_proj. modeling_granite.py GraniteMLP L215: down_proj(act_fn(gate_proj(x)) * up_proj(x)).
  - GraniteMoeHybridDecoderLayer L1000 / L1011-1012 / L1048-1052: num_local_experts == 0 -> block_sparse_moe None,
    has_experts False -> hidden = shared_mlp(hidden) only (no router, no expert term).
Renames: model.layers.N.shared_mlp.input_linear.weight [16384, 2560] -> model.layers.N.mlp.gate_proj.weight (rows
0..8191) + model.layers.N.mlp.up_proj.weight (rows 8192..16383); model.layers.N.shared_mlp.output_linear.weight ->
model.layers.N.mlp.down_proj.weight; every other key (embed_tokens, self_attn q/k/v/o, input_layernorm,
post_attention_layernorm, norm) unchanged. No lm_head in the source (tie_word_embeddings true) - none written.

config.json: the key set of ibm-granite/granite-4.1-3b's config.json (GraniteForCausalLM), values from the source;
refuses unless the source is what the rewrite assumes (all layer_types attention, 0 experts, rope, rmsnorm,
intermediate_size == shared_intermediate_size, no biases). Tokenizer / template / generation files are byte copies.
Shards keep the source split (one output shard per source shard, same names), index rewritten.
Exit 0 = written + every self-check passed; 1 = a check failed; 2 = usage / refused.
"""
import hashlib
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEEP = ["architectures", "attention_bias", "attention_dropout", "attention_multiplier", "bos_token_id",
        "embedding_multiplier", "eos_token_id", "hidden_act", "hidden_size", "initializer_range", "intermediate_size",
        "logits_scaling", "max_position_embeddings", "mlp_bias", "model_type", "num_attention_heads",
        "num_hidden_layers", "num_key_value_heads", "pad_token_id", "residual_multiplier", "rms_norm_eps",
        "rope_scaling", "rope_theta", "tie_word_embeddings", "torch_dtype", "transformers_version", "use_cache",
        "vocab_size"]  # = sorted keys of ibm-granite/granite-4.1-3b config.json (c0650403)
DROP_EXPECTED = ["layer_types", "mamba_chunk_size", "mamba_conv_bias", "mamba_d_conv", "mamba_d_head", "mamba_d_state",
                 "mamba_expand", "mamba_n_groups", "mamba_n_heads", "mamba_proj_bias", "normalization_function",
                 "num_experts_per_tok", "num_local_experts", "output_router_logits", "position_embedding_type",
                 "router_aux_loss_coef", "shared_intermediate_size"]
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "generation_config.json"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tsha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src, out, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if os.path.exists(out):
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    cfg = json.load(open(os.path.join(src, "config.json")))
    n = cfg["num_hidden_layers"]
    I = cfg["shared_intermediate_size"]
    refuse = []
    if cfg.get("model_type") != "granitemoehybrid":
        refuse.append(f"model_type {cfg.get('model_type')}")
    if cfg.get("layer_types") != ["attention"] * n:
        refuse.append("layer_types are not all 'attention'")
    if cfg.get("num_local_experts") != 0:
        refuse.append(f"num_local_experts {cfg.get('num_local_experts')}")
    if cfg.get("position_embedding_type") != "rope":
        refuse.append(f"position_embedding_type {cfg.get('position_embedding_type')}")
    if cfg.get("normalization_function") != "rmsnorm":
        refuse.append(f"normalization_function {cfg.get('normalization_function')}")
    if cfg.get("intermediate_size") != I:
        refuse.append(f"intermediate_size {cfg.get('intermediate_size')} != shared_intermediate_size {I}")
    if cfg.get("attention_bias") is not False:
        refuse.append("attention_bias")
    if cfg.get("rope_scaling") is not None:
        refuse.append(f"rope_scaling {cfg.get('rope_scaling')}")
    extra = sorted(set(cfg) - set(KEEP) - set(DROP_EXPECTED))
    if extra:
        refuse.append(f"unexpected config keys {extra}")
    if refuse:
        print("refusing: " + "; ".join(refuse), file=sys.stderr)
        return 2

    dense = {}
    for k in KEEP:
        if k == "architectures":
            dense[k] = ["GraniteForCausalLM"]
        elif k == "model_type":
            dense[k] = "granite"
        elif k == "mlp_bias":
            dense[k] = False  # GraniteMoeHybridMLP: nn.Linear(..., bias=False) for both linears
        elif k == "intermediate_size":
            dense[k] = I
        else:
            dense[k] = cfg[k]
    dropped = {k: cfg[k] for k in sorted(set(cfg) - set(KEEP))}
    added = sorted(set(KEEP) - set(cfg))

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    os.makedirs(out)
    new_map, renamed, kept, seen_src, rows = {}, [], [], set(), []
    total = 0
    checks_ok = True
    for shard in shards:
        tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                seen_src.add(k)
                t = f.get_tensor(k)
                if ".shared_mlp.input_linear.weight" in k:
                    base = k.replace(".shared_mlp.input_linear.weight", ".mlp.")
                    if tuple(t.shape) != (2 * I, cfg["hidden_size"]):
                        print(f"bad shape {k} {tuple(t.shape)}", file=sys.stderr)
                        return 1
                    gate = t[:I].clone().contiguous()
                    up = t[I:].clone().contiguous()
                    ok = torch.equal(torch.cat([gate, up], 0), t)
                    checks_ok &= ok
                    tensors[base + "gate_proj.weight"] = gate
                    tensors[base + "up_proj.weight"] = up
                    renamed.append({"src": k, "src_shape": list(t.shape), "src_sha256": tsha(t),
                                    "dst": [base + "gate_proj.weight", base + "up_proj.weight"],
                                    "dst_rows": [[0, I], [I, 2 * I]], "dst_sha256": [tsha(gate), tsha(up)],
                                    "concat_equals_src": ok})
                elif ".shared_mlp.output_linear.weight" in k:
                    nk = k.replace(".shared_mlp.output_linear.weight", ".mlp.down_proj.weight")
                    tensors[nk] = t.contiguous()
                    renamed.append({"src": k, "src_shape": list(t.shape), "src_sha256": tsha(t), "dst": [nk],
                                    "dst_sha256": [tsha(tensors[nk])]})
                else:
                    tensors[k] = t.contiguous()
                    kept.append(k)
        for k, t in tensors.items():
            new_map[k] = shard
            total += t.numel() * t.element_size()
        save_file(tensors, os.path.join(out, shard), metadata=meta or {"format": "pt"})
        rows.append({"shard": shard, "tensors": len(tensors), "sha256": sha256_file(os.path.join(out, shard)),
                     "size": os.path.getsize(os.path.join(out, shard))})
        del tensors
    if seen_src != set(idx["weight_map"]):
        print("source shards and index disagree", file=sys.stderr)
        checks_ok = False
    new_idx = {"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(new_idx, f, indent=2)
        f.write("\n")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dense, f, indent=2)
        f.write("\n")
    copied = []
    for fn in COPY_FILES:
        shutil.copyfile(os.path.join(src, fn), os.path.join(out, fn))
        a, b = sha256_file(os.path.join(src, fn)), sha256_file(os.path.join(out, fn))
        checks_ok &= a == b
        copied.append({"file": fn, "sha256": b, "identical": a == b})

    # The key set GraniteForCausalLM expects, from the class itself on the meta device.
    from transformers import AutoConfig, GraniteForCausalLM
    loaded_cfg = AutoConfig.from_pretrained(out)  # the written config.json, read the way the export reads it
    with torch.device("meta"):
        model = GraniteForCausalLM(loaded_cfg)
    expected = set(model.state_dict().keys())
    cfg_read = {"class": type(loaded_cfg).__name__, "rope_parameters": dict(loaded_cfg.rope_parameters or {}),
                "intermediate_size": loaded_cfg.intermediate_size, "mlp_bias": loaded_cfg.mlp_bias,
                "embedding_multiplier": loaded_cfg.embedding_multiplier, "logits_scaling": loaded_cfg.logits_scaling,
                "residual_multiplier": loaded_cfg.residual_multiplier,
                "attention_multiplier": loaded_cfg.attention_multiplier,
                "tie_word_embeddings": loaded_cfg.tie_word_embeddings}
    written = set(new_map)
    missing = sorted(expected - written)
    unexpected = sorted(written - expected)
    if missing != ["lm_head.weight"] or unexpected:
        checks_ok = False
    rep = {"src": os.path.abspath(src), "out": os.path.abspath(out),
           "config": {"kept_from_src": [k for k in KEEP if k in cfg and k not in ("architectures", "model_type")],
                      "set": {"architectures": dense["architectures"], "model_type": "granite", "mlp_bias": False,
                              "intermediate_size": I},
                      "added_keys": added, "dropped": dropped, "as_read_by_AutoConfig": cfg_read},
           "tensors": {"source": len(seen_src), "written": len(written), "renamed": renamed, "unchanged": len(kept)},
           "expected_by_GraniteForCausalLM": len(expected), "missing_vs_class": missing,
           "missing_note": "lm_head.weight is tied to model.embed_tokens.weight (tie_word_embeddings true)",
           "unexpected_vs_class": unexpected, "shards": rows, "copied_files": copied,
           "verdict": "PASS" if checks_ok else "FAIL"}
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"{rep['verdict']}: {len(seen_src)} source tensors -> {len(written)} written; renamed {len(renamed)}; "
          f"missing vs class {missing}; unexpected {unexpected}; dropped config keys {list(dropped)}")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
