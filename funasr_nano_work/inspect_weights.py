"""Inspect out/hf_vllm/model.safetensors against the round-1 premises (key families, shapes, dtypes,
counts) and measure llm.lm_head.weight vs llm.model.embed_tokens.weight. Writes weights_inspect.json.
Run with ~/venvs/lt094dev/bin/python (safetensors + torch)."""
import collections
import json
import os
import re

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, "out", "hf_vllm", "model.safetensors")


def main():
    out = {}
    with safe_open(ST, framework="pt") as f:
        keys = list(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
        dtypes = collections.Counter(f.get_slice(k).get_dtype() for k in keys)
        n_params = sum(int(torch.tensor(s).prod()) if s else 1 for s in shapes.values())
        fam = collections.Counter(k.split(".")[0] for k in keys)
        out["n_tensors"] = len(keys)
        out["n_params"] = n_params
        out["dtypes"] = dict(dtypes)
        out["families"] = dict(fam)
        enc = [k for k in keys if k.startswith("audio_encoder.")]
        out["encoder_blocks"] = {
            "encoders0": sorted({int(m.group(1)) for k in enc for m in [re.match(r"audio_encoder\.encoders0\.(\d+)\.", k)] if m}),
            "encoders": sorted({int(m.group(1)) for k in enc for m in [re.match(r"audio_encoder\.encoders\.(\d+)\.", k)] if m}),
            "tp_encoders": sorted({int(m.group(1)) for k in enc for m in [re.match(r"audio_encoder\.tp_encoders\.(\d+)\.", k)] if m}),
        }
        # per-layer suffix set of one representative layer of each family
        def suffixes(prefix):
            return sorted(k[len(prefix):] for k in keys if k.startswith(prefix))
        out["encoders0.0_suffixes"] = {s: shapes["audio_encoder.encoders0.0." + s] for s in suffixes("audio_encoder.encoders0.0.")}
        out["encoders.0_suffixes"] = {s: shapes["audio_encoder.encoders.0." + s] for s in suffixes("audio_encoder.encoders.0.")}
        out["tp_encoders.0_suffixes"] = {s: shapes["audio_encoder.tp_encoders.0." + s] for s in suffixes("audio_encoder.tp_encoders.0.")}
        out["encoder_other"] = {k: shapes[k] for k in enc if not re.match(r"audio_encoder\.(encoders0|encoders|tp_encoders)\.\d+\.", k)}
        out["adaptor"] = {k: shapes[k] for k in keys if k.startswith("audio_adaptor.")}
        out["llm_non_layer"] = {k: shapes[k] for k in keys if k.startswith("llm.") and ".layers." not in k}
        out["llm_layers"] = sorted({int(m.group(1)) for k in keys for m in [re.match(r"llm\.model\.layers\.(\d+)\.", k)] if m})
        out["llm_layer0_suffixes"] = {k[len("llm.model.layers.0."):]: shapes[k] for k in keys if k.startswith("llm.model.layers.0.")}
        out["ctc_keys"] = [k for k in keys if "ctc" in k]
        head = f.get_tensor("llm.lm_head.weight")
        emb = f.get_tensor("llm.model.embed_tokens.weight")
    eq = bool(torch.equal(head, emb))
    diff = (head.float() - emb.float()).abs()
    rows_diff = int((diff.amax(dim=1) > 0).sum())
    out["lm_head_vs_embed"] = {"shape": list(head.shape), "dtype": str(head.dtype), "torch_equal": eq,
                               "max_abs_diff": float(diff.max()), "rows_differing": rows_diff,
                               "bytes_equal": head.view(torch.int16).equal(emb.view(torch.int16))}
    with open(os.path.join(HERE, "weights_inspect.json"), "w") as fo:
        json.dump(out, fo, indent=1)
    print(json.dumps({k: out[k] for k in ["n_tensors", "n_params", "dtypes", "families", "ctc_keys", "lm_head_vs_embed"]}, indent=1))
    print("encoders0/encoders/tp_encoders counts:", {k: (len(v), v[:1], v[-1:]) for k, v in out["encoder_blocks"].items()})
    print("encoders0.0:", out["encoders0.0_suffixes"])
    print("encoders.0:", out["encoders.0_suffixes"])
    print("encoder_other:", out["encoder_other"])
    print("adaptor:", json.dumps(out["adaptor"]))
    print("llm_non_layer:", out["llm_non_layer"], "layers", len(out["llm_layers"]))
    print("llm layer0:", out["llm_layer0_suffixes"])


if __name__ == "__main__":
    main()
