"""Merge check: fold the LoRA independently in float32 and compare with peft's merge and with the author's full-weight
checkpoint (written by the author's scripts/merge_lora_checkpoint.py at kev-1.0 into merged/kev-<model>-v1.0).

0.8B: base = Qwen3_5ForConditionalGeneration.from_pretrained(Qwen/Qwen3.5-0.8B-Base@dc7cdfe2, dtype=float32) (a loader
independent of the author's). Own fold: for every (lora_A, lora_B) pair of the adapter, W_fp32 += (B.float() @
A.float()) * (lora_alpha / r), no rounding. The adapter was trained on the text model, so its keys
`base_model.model.layers.N.<mod>.lora_A.weight` name `model.language_model.layers.N.<mod>.weight` of the multimodal
class. peft: PeftModel.from_pretrained(<copy of the fp32 text model>, adapter).merge_and_unload() (wrapping the text
model, where the adapter's keys resolve; wrapping the multimodal class would leave every LoRA at its init silently).
Author: merged/kev-0.8b-v1.0/checkpoint/model*.safetensors (keys without prefix). -> results/merge_0.8b.json
--model 4b: the same three-way check on Kev-4B, streamed tensor by tensor (one fp32 model in memory) -> results/merge_4b.json"""
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

K = Path(__file__).resolve().parents[1]
BASE, BASE_REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
ADAPTER_DIR = K / "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d"
MERGED = K / "merged/kev-0.8b-v1.0"
TEXT_PREFIX = "model.language_model."


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration
    t0 = time.time()
    cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    assert cfg["use_rslora"] is False and cfg["r"] == 16 and cfg["lora_alpha"] == 32, cfg
    assert not cfg.get("use_dora") and not cfg.get("modules_to_save") and not cfg.get("trainable_token_indices")
    scale = cfg["lora_alpha"] / cfg["r"]
    assert scale == 2.0

    full = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, revision=BASE_REV, dtype=torch.float32)
    full.eval()
    sd = full.state_dict()
    text_names = sorted(k for k in sd if k.startswith(TEXT_PREFIX))
    base_text = {k[len(TEXT_PREFIX):]: sd[k].clone() for k in text_names}   # fp32 copies of the base text weights

    # own fold
    with safe_open(ADAPTER_DIR / "adapter_model.safetensors", "pt") as f:
        akeys = sorted(f.keys())
        adapter = {k: f.get_tensor(k) for k in akeys}
    assert len(akeys) == 372, len(akeys)
    stems = sorted({k[: -len(".lora_A.weight")] for k in akeys if k.endswith(".lora_A.weight")})
    assert len(stems) == 186 and all(f"{s}.lora_B.weight" in adapter for s in stems)
    kinds = {}
    own = dict(base_text)
    for stem in stems:
        a, b = adapter[f"{stem}.lora_A.weight"], adapter[f"{stem}.lora_B.weight"]
        assert a.dtype == torch.float32 and b.dtype == torch.float32 and a.shape[0] == 16 and b.shape[1] == 16
        name = stem.removeprefix("base_model.model.") + ".weight"
        assert name in own, name
        assert tuple((b @ a).shape) == tuple(own[name].shape), name
        own[name] = own[name] + (b.float() @ a.float()) * scale
        layer = int(name.split(".")[1])
        kind = "linear_attention" if ".linear_attn." in name else "full_attention" if ".self_attn." in name else "mlp"
        kinds.setdefault(layer, set()).add(name.split(".")[-2])
    adapted = sorted(s.removeprefix("base_model.model.") + ".weight" for s in stems)
    layer_types = full.config.text_config.layer_types
    per_layer = {i: len(kinds.get(i, ())) for i in range(len(layer_types))}
    assert all(per_layer[i] == (8 if t == "linear_attention" else 7) for i, t in enumerate(layer_types)), per_layer

    # peft on the fp32 text model
    text = copy.deepcopy(full.model.language_model)
    del full, sd
    peft_model = PeftModel.from_pretrained(text, str(ADAPTER_DIR), torch_device="cpu")
    lora_loaded = {n: m for n, m in peft_model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    assert len(lora_loaded) == 186, len(lora_loaded)
    merged_peft = peft_model.merge_and_unload().state_dict()
    assert set(merged_peft) == set(own), (len(merged_peft), len(own))

    # the author's checkpoint
    shards = sorted((MERGED / "checkpoint").glob("model*.safetensors"))
    author = {}
    for shard in shards:
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                author[k] = f.get_tensor(k)
    assert set(author) == set(own), (sorted(set(author) ^ set(own))[:4])
    assert all(t.dtype == torch.float32 for t in author.values())

    def compare(x, y, names):
        diffs = {n: maxabs(x[n], y[n]) for n in names}
        worst = max(diffs, key=diffs.get)
        return {"tensors": len(names), "max_abs": diffs[worst], "worst": worst, "nonzero": sum(v > 0 for v in diffs.values()),
                "bit_equal": sum(torch.equal(x[n], y[n]) for n in names)}

    unadapted = sorted(set(own) - set(adapted))
    report = {
        "base": f"{BASE}@{BASE_REV}", "base_loader": "transformers.Qwen3_5ForConditionalGeneration.from_pretrained(dtype=float32)",
        "adapter": str(ADAPTER_DIR), "adapter_sha256": sha256_file(ADAPTER_DIR / "adapter_model.safetensors"),
        "lora": {"r": cfg["r"], "alpha": cfg["lora_alpha"], "scale": scale, "use_rslora": cfg["use_rslora"],
                 "adapter_tensors": len(akeys), "pairs": len(stems),
                 "pairs_by_kind": {"linear_attention_layers": sum(t == "linear_attention" for t in layer_types),
                                   "full_attention_layers": sum(t == "full_attention" for t in layer_types),
                                   "targets_per_linear_attention_layer": 8, "targets_per_full_attention_layer": 7}},
        "key_map": "adapter base_model.model.<x>.lora_{A,B}.weight -> Qwen3_5ForConditionalGeneration model.language_model.<x>.weight "
                   "(the multimodal class nests the text model under language_model)",
        "text_tensors": len(own), "adapted_weights": len(adapted), "unadapted_weights": len(unadapted),
        "own_vs_peft_adapted": compare(own, merged_peft, adapted),
        "own_vs_peft_all": compare(own, merged_peft, sorted(own)),
        "own_vs_author_adapted": compare(own, author, adapted),
        "own_vs_author_unadapted": compare(own, author, unadapted),
        "own_vs_author_all": compare(own, author, sorted(own)),
        "delta_norms": {"max_abs_delta": max(maxabs(own[n], base_text[n]) for n in adapted)},
        "author_checkpoint": {
            "dir": str(MERGED / "checkpoint"),
            "shards": [{"file": s.name, "bytes": s.stat().st_size, "sha256": sha256_file(s)} for s in shards],
            "merge_json": json.loads((MERGED / "merge.json").read_text()),
            "config_dtype": json.loads((MERGED / "checkpoint/config.json").read_text()).get("dtype"),
            "head_pt_sha256": sha256_file(MERGED / "checkpoint/head.pt"),
            "du_bytes": sum(p.stat().st_size for p in (MERGED / "checkpoint").iterdir()),
        },
        "seconds": round(time.time() - t0, 1),
    }
    out = K / "results/merge_0.8b.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("text_tensors", "adapted_weights", "own_vs_peft_adapted", "own_vs_author_all", "seconds")}, indent=1))


BASE_4B, BASE_4B_REV = "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"
ADAPTER_DIR_4B = K / "hf/hub/models--jaredpalmer--kev-4b/snapshots/591dcb5bd6d05eb0b5131ea6608f93f10243335c"
MERGED_4B = K / "merged/kev-4b-v1.0"


def main_4b():
    """The same three-way check on Kev-4B, streamed tensor by tensor so the 4B fp32 weights are held once
    (the 0.8B path above holds four fp32 copies: base clones, own fold, peft text copy, author checkpoint ~ 67 GB at 4B).

    peft: Qwen3_5ForConditionalGeneration.from_pretrained(base@1001bb4d, dtype=float32) -> its text model wrapped by
    PeftModel.from_pretrained(adapter) -> merge_and_unload() (the one fp32 model in memory).
    own fold: per tensor, the base value read straight from the base safetensors shards (bf16/fp32 -> fp32, exact) plus
    (B.float() @ A.float()) * (alpha / r) for the 248 adapted weights; never rounded.
    author: merged/kev-4b-v1.0/checkpoint shards read lazily (safe_open), one tensor at a time.
    -> results/merge_4b.json"""
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    out = K / "results/merge_4b.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    t0 = time.time()
    cfg = json.loads((ADAPTER_DIR_4B / "adapter_config.json").read_text())
    assert cfg["use_rslora"] is False and cfg["r"] == 16 and cfg["lora_alpha"] == 32, cfg
    assert not cfg.get("use_dora") and not cfg.get("modules_to_save") and not cfg.get("trainable_token_indices")
    scale = cfg["lora_alpha"] / cfg["r"]
    assert scale == 2.0
    with safe_open(ADAPTER_DIR_4B / "adapter_model.safetensors", "pt") as f:
        akeys = sorted(f.keys())
        adapter = {k: f.get_tensor(k) for k in akeys}
    assert len(akeys) == 496, len(akeys)
    stems = sorted({k[: -len(".lora_A.weight")] for k in akeys if k.endswith(".lora_A.weight")})
    assert len(stems) == 248 and all(f"{s}.lora_B.weight" in adapter for s in stems)
    delta_of = {}
    for stem in stems:
        a, b = adapter[f"{stem}.lora_A.weight"], adapter[f"{stem}.lora_B.weight"]
        assert a.dtype == torch.float32 and b.dtype == torch.float32 and a.shape[0] == 16 and b.shape[1] == 16
        delta_of[stem.removeprefix("base_model.model.") + ".weight"] = (a, b)

    # base shards (the independent reader of the own fold)
    base_dir = Path(snapshot_download(BASE_4B, revision=BASE_4B_REV, allow_patterns=["*.json", "*.safetensors"]))
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    base_files = {}

    def base_tensor(name):
        key = TEXT_PREFIX + name
        shard = index[key]
        if shard not in base_files:
            base_files[shard] = safe_open(base_dir / shard, "pt")
        return base_files[shard].get_tensor(key)

    # peft on the fp32 text model of an independent loader
    t_load = time.time()
    full = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_4B, revision=BASE_4B_REV, dtype=torch.float32)
    full.eval()
    layer_types = full.config.text_config.layer_types
    text = full.model.language_model
    full.model.language_model = None
    del full
    load_seconds = round(time.time() - t_load, 1)
    text_names = sorted(text.state_dict())
    assert all(n in text_names for n in delta_of), [n for n in delta_of if n not in text_names][:3]
    kinds = {}
    for name in delta_of:
        layer = int(name.split(".")[1])
        kinds.setdefault(layer, set()).add(name.split(".")[-2])
    per_layer = {i: len(kinds.get(i, ())) for i in range(len(layer_types))}
    assert all(per_layer[i] == (8 if t == "linear_attention" else 7) for i, t in enumerate(layer_types)), per_layer
    peft_model = PeftModel.from_pretrained(text, str(ADAPTER_DIR_4B), torch_device="cpu")
    lora_loaded = {n for n, m in peft_model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    assert len(lora_loaded) == 248, len(lora_loaded)
    merged_peft = peft_model.merge_and_unload().state_dict()
    assert set(merged_peft) == set(text_names), (len(merged_peft), len(text_names))

    # the author's checkpoint, lazily
    shards = sorted((MERGED_4B / "checkpoint").glob("model*.safetensors"))
    author_files = [safe_open(sh, "pt") for sh in shards]
    author_where = {k: f for f in author_files for k in f.keys()}
    assert set(author_where) == set(text_names), sorted(set(author_where) ^ set(text_names))[:4]

    adapted = sorted(delta_of)
    unadapted = sorted(set(text_names) - set(adapted))
    stats = {k: {} for k in ("own_vs_peft", "own_vs_author")}
    bit = {k: {} for k in stats}
    max_delta = 0.0
    author_dtypes = set()
    for n in text_names:
        base = base_tensor(n).float()
        if n in delta_of:
            a, b = delta_of[n]
            own = base + (b.float() @ a.float()) * scale
            max_delta = max(max_delta, maxabs(own, base))
        else:
            own = base
        theirs = author_where[n].get_tensor(n)
        author_dtypes.add(str(theirs.dtype))
        for key, other in (("own_vs_peft", merged_peft[n]), ("own_vs_author", theirs)):
            assert tuple(other.shape) == tuple(own.shape), (key, n)
            stats[key][n] = maxabs(own, other)
            bit[key][n] = bool(torch.equal(own, other.float()) and other.dtype == torch.float32)
        del base, own, theirs

    def compare(key, names):
        worst = max(names, key=lambda n: stats[key][n])
        return {"tensors": len(names), "max_abs": stats[key][worst], "worst": worst,
                "nonzero": sum(stats[key][n] > 0 for n in names), "bit_equal": sum(bit[key][n] for n in names)}

    report = {
        "model": "4b", "base": f"{BASE_4B}@{BASE_4B_REV}",
        "base_loader": "transformers.Qwen3_5ForConditionalGeneration.from_pretrained(dtype=float32) (peft side); "
                       "safetensors safe_open of the base shards, bf16/fp32 -> fp32 (own-fold side)",
        "adapter": str(ADAPTER_DIR_4B.relative_to(K)), "adapter_sha256": sha256_file(ADAPTER_DIR_4B / "adapter_model.safetensors"),
        "lora": {"r": cfg["r"], "alpha": cfg["lora_alpha"], "scale": scale, "use_rslora": cfg["use_rslora"],
                 "adapter_tensors": len(akeys), "pairs": len(stems),
                 "pairs_by_kind": {"linear_attention_layers": sum(t == "linear_attention" for t in layer_types),
                                   "full_attention_layers": sum(t == "full_attention" for t in layer_types),
                                   "targets_per_linear_attention_layer": 8, "targets_per_full_attention_layer": 7}},
        "key_map": "adapter base_model.model.<x>.lora_{A,B}.weight -> base shard key model.language_model.<x>.weight "
                   "-> author checkpoint key <x>.weight",
        "text_tensors": len(text_names), "adapted_weights": len(adapted), "unadapted_weights": len(unadapted),
        "own_vs_peft_adapted": compare("own_vs_peft", adapted),
        "own_vs_peft_all": compare("own_vs_peft", text_names),
        "own_vs_author_adapted": compare("own_vs_author", adapted),
        "own_vs_author_unadapted": compare("own_vs_author", unadapted),
        "own_vs_author_all": compare("own_vs_author", text_names),
        "author_tensor_dtypes": sorted(author_dtypes),
        "delta_norms": {"max_abs_delta": max_delta},
        "author_checkpoint": {
            "dir": str((MERGED_4B / "checkpoint").relative_to(K)),
            "shards": [{"file": sh.name, "bytes": sh.stat().st_size, "sha256": sha256_file(sh)} for sh in shards],
            "merge_json": json.loads((MERGED_4B / "merge.json").read_text()),
            "config_dtype": json.loads((MERGED_4B / "checkpoint/config.json").read_text()).get("dtype"),
            "head_pt_sha256": sha256_file(MERGED_4B / "checkpoint/head.pt"),
            "du_bytes": sum(p.stat().st_size for p in (MERGED_4B / "checkpoint").iterdir()),
        },
        "peft_load_seconds": load_seconds,
    }
    report["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("text_tensors", "adapted_weights", "own_vs_peft_adapted", "own_vs_peft_all",
                                              "own_vs_author_all", "seconds")}, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["0.8b", "4b"], default="0.8b")
    if ap.parse_args().model == "4b":
        main_4b()
    else:
        main()
