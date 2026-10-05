"""How head/kev_<model>_pointer_head.{safetensors,json} are made from the checkpoint's head.pt.

    HF_HOME=$W/hf HF_HUB_OFFLINE=1 $ORACLE scripts/host_head_export.py                 (0.8B: check the existing files)
    HF_HOME=$W/hf HF_HUB_OFFLINE=1 $ORACLE scripts/host_head_export.py --model 4b --write   (4B: write, then check)

head.pt (jaredpalmer/kev-0.8b v1.0, sha256 f400bd12…; jaredpalmer/kev-4b v1.0, sha256 dd633435…) is read with
torch.load(weights_only=True); its "head" state dict (q.weight [256,d], q.bias, k.weight, k.bias, float32) is written
with safetensors.torch.save_file in the key order of the 0.8B file. The JSON beside it carries the temperature, the
delimiter and pad token ids, the readout formula and the graph I/O. The check compares tensors (bit for bit) with head.pt,
the temperature with the JSON, and reports whether a rewrite is byte-identical (it need not be: the metadata order of a
rewrite can differ)."""
import hashlib
import json
import tempfile
from pathlib import Path

K = Path(__file__).resolve().parents[1]
MODELS = {
    "0.8b": {"head_pt": K / "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/head.pt",
             "head_pt_sha256": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
             "shipped": K / "host/kev_0.8b_pointer_head.safetensors", "contract": K / "host/kev_0.8b_pointer_head.json"},
    "4b": {"head_pt": K / "hf/hub/models--jaredpalmer--kev-4b/snapshots/591dcb5bd6d05eb0b5131ea6608f93f10243335c/head.pt",
           "head_pt_sha256": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
           "shipped": K / "host/kev_4b_pointer_head.safetensors", "contract": K / "host/kev_4b_pointer_head.json",
           "model": "Kev-4B", "repo": "jaredpalmer/kev-4b", "tag_object": "591dcb5b",
           "commit": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
           "base": {"repo": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b", "hidden_size": 2560}},
}
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"


def write_4b(cfg, torch, save_file):
    """host/kev_4b_pointer_head.{safetensors,json}: the 0.8B files' layout with the 4B values (never overwrites)."""
    assert not cfg["shipped"].exists() and not cfg["contract"].exists(), "the 4B head files exist; checking only"
    d = torch.load(cfg["head_pt"], map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in d["head"].items()}
    from safetensors import safe_open
    with safe_open(str(MODELS["0.8b"]["shipped"]), "pt") as f:
        meta08, keys08 = f.metadata(), list(f.keys())
    assert sorted(sd) == sorted(keys08), (list(sd), keys08)
    metadata = {"format": meta08["format"], "dtype": "float32",
                "source": f"{cfg['repo']}@{cfg['tag_object']} (tag v1.0) head.pt sha256 {cfg['head_pt_sha256'][:8]}…"}
    assert set(metadata) == set(meta08)
    save_file({k: sd[k] for k in keys08}, str(cfg["shipped"]), metadata=metadata)
    c08 = json.loads(MODELS["0.8b"]["contract"].read_text())
    hidden = cfg["base"]["hidden_size"]
    c = dict(c08)
    c["model"] = cfg["model"]
    c["source"] = {"repo": cfg["repo"], "tag": "v1.0", "commit": cfg["commit"], "head_pt_sha256": cfg["head_pt_sha256"]}
    c["base"] = dict(cfg["base"])
    c["head"] = dict(c08["head"], file=cfg["shipped"].name,
                     tensors={k: list(sd[k].shape) for k in ("q.weight", "q.bias", "k.weight", "k.bias")})
    assert c["head"]["scale"] == 1 / int(d["head_dim"]) ** 0.5 and int(d["head_dim"]) == 256
    c["temperature"] = float(d["temperature"])
    c["tokenizer"] = ("jaredpalmer/kev-4b@v1.0 tokenizer.json (19,989,325 B, sha256 " + TOKENIZER_SHA256 + "; the same file "
                      "as jaredpalmer/kev-0.8b@v1.0) + tokenizer_config.json = the transformers Qwen2Tokenizer pipeline over the "
                      "Qwen3.5 base vocabulary (Qwen2 regex, 33 added tokens, len 248077); load with tokenizers.Tokenizer.from_file; "
                      "add_special_tokens=False; user text: <|name|> -> <¦name¦> before tokenizing. The base repo's raw "
                      "tokenizer.json differs on combining marks and on <think>/<tool_response>/<tts_pad> (measured 2026-10-03).")
    c["graph"] = dict(c08["graph"], output={"hidden": f"float32 [1,L,{hidden}] (after final RMSNorm)"},
                      gpu_precision="FP32 (the GPU checks ran with float32 activations; float16 activations were not "
                                    "measured on Kev-4B and gave non-finite hidden states on Kev-0.8B)")
    assert list(c) == list(c08), (list(c), list(c08))
    cfg["contract"].write_text(json.dumps(c, indent=1, ensure_ascii=False))
    return {"written": [str(cfg["shipped"].relative_to(K)), str(cfg["contract"].relative_to(K))]}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="0.8b")
    ap.add_argument("--write", action="store_true", help="4B: make the head files first (never overwrites)")
    a = ap.parse_args()
    cfg = MODELS[a.model]
    HEAD_PT, HEAD_PT_SHA256, SHIPPED, CONTRACT = cfg["head_pt"], cfg["head_pt_sha256"], cfg["shipped"], cfg["contract"]
    assert sha256_file(HEAD_PT) == HEAD_PT_SHA256
    written = None
    if a.write:
        assert a.model == "4b", "--write is for the 4B head (the 0.8B files exist)"
        written = write_4b(cfg, torch, save_file)
    d = torch.load(HEAD_PT, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in d["head"].items()}
    with safe_open(str(SHIPPED), "pt") as f:
        metadata, shipped_keys = f.metadata(), list(f.keys())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "head.safetensors"
        save_file({k: sd[k] for k in shipped_keys}, str(out), metadata=metadata)
        same_bytes = out.read_bytes() == SHIPPED.read_bytes()
    shipped = load_file(str(SHIPPED))
    tensors_equal = {k: bool(torch.equal(sd[k], shipped[k]) and sd[k].dtype == shipped[k].dtype) for k in sd}
    contract = json.loads(CONTRACT.read_text())
    doc = {"model": a.model, "written": written,
           "head_pt": str(HEAD_PT.relative_to(K)), "head_pt_sha256": HEAD_PT_SHA256, "keys": list(sd),
           "shapes": {k: list(v.shape) for k, v in sd.items()}, "dtypes": {k: str(v.dtype) for k, v in sd.items()},
           "temperature_head_pt": float(d["temperature"]), "temperature_json": contract["temperature"],
           "temperature_equal": float(d["temperature"]) == contract["temperature"], "head_dim": int(d["head_dim"]),
           "shipped_sha256": sha256_file(SHIPPED), "shipped_metadata": metadata, "shipped_key_order": shipped_keys,
           "tensors_bit_equal": tensors_equal, "rewritten_file_byte_identical": same_bytes}
    print(json.dumps(doc, indent=1))
    assert all(tensors_equal.values()) and doc["temperature_equal"]


if __name__ == "__main__":
    main()
