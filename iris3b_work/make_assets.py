"""Writes the host-side files of the published folder and copies its run configuration.

    python make_assets.py --out OUT [--iris DIR] [--text-encoder DIR] [--upstream DIR] [--published DIR]

  iris_tokenizer.json  {"files": {name: text}, "sha256": {name: sha256}} of qwen_vocab.txt (tokens by id),
                       qwen_merges.txt (merges by rank) and qwen_added.txt (added tokens by id), all from the text
                       encoder's tokenizer.json, and prompt_prefix_ids.txt / prompt_suffix_ids.txt: the chat template's
                       34-token prefix and 5-token suffix as upstream's Qwen3VLTextEncoder builds them (its __init__
                       runs with the Hugging Face model replaced by a stub, so no weight is loaded)
  iris_embed_bf16.f32  the checkpoint's embed_tokens [151936, 2560] as raw little-endian bfloat16 (the name is the app's)
  iris_host.f32        the float32 tensors the host reads, concatenated in HOST_ORDER: the text encoder's rotary tables
                       for positions 0..333 (transformers' Qwen3VLTextRotaryEmbedding, the tables the model builds for a
                       text-only prompt, on 8 threads as the published file was made) and OUT/host/*.bin of export_dit.py
  iris3b_config.json   copied from --published (the model repo's file: graph files, signatures, precision, host
                       offsets; the app and the Python host read it), then checked against the files written here
  SHA256SUMS           copied from --published

The tokenizer's text files and the rotary tables also stay in OUT/host/. Checks (a failure exits 4): every host
tensor's offset, count and shape against the config's host index, every graph's signature against its config entry,
the modulation row ranges against export_dit.py's. The host tensors whose sha256 differs from the config's
source_sha256 are listed, not failed (none for the published checkpoint; a fine-tune's tables differ).
OUT/model_assets.json records all of it.
Defaults: --iris OUT/weights/iris-3b/turbo, --text-encoder OUT/weights/Qwen3-VL-4B-Instruct, --upstream OUT/upstream,
--published OUT/published/model.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

import te_modules as T

TOKENIZER_FILES = ["qwen_vocab.txt", "qwen_merges.txt", "qwen_added.txt", "prompt_prefix_ids.txt",
                   "prompt_suffix_ids.txt"]
HOST_ORDER = ["te_rope_cos", "te_rope_sin", "y_pos", "cos_j", "sin_j", "cos_i", "sin_i", "pix_pos", "sigmas",
              "t_emb_s0", "t_emb_s1", "t_emb_s2", "t_emb_s3", "mods_s0", "mods_s1", "mods_s2", "mods_s3", "noise_pix"]
EMBED = "model.language_model.embed_tokens.weight"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class _StubDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=T.HIDDEN, num_hidden_layers=36)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        raise RuntimeError("the stub decoder only stands in for the weights at construction")


class _StubModel:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def get_decoder(self):
        return _StubDecoder()


def upstream_encoder(iris_dir):
    """Upstream's Qwen3VLTextEncoder, built from the checkpoint's config.yaml with a stub in place of the HF model."""
    import transformers
    from omegaconf import OmegaConf

    from iris3b.config import inference_config
    from iris3b.registry import TEXT_ENCODERS
    import iris3b.text  # noqa: F401  (registers the encoders)

    transformers.Qwen3VLForConditionalGeneration = _StubModel
    raw = OmegaConf.to_container(OmegaConf.load(iris_dir / "config.yaml"))
    cfg = inference_config(raw, [f"text_encoder.pretrained={T.TE_DIR}"])
    return TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device="cpu")


def tokenizer_assets(iris_dir, host):
    """The tokenizer's text files (in `host`); returns the prefix / suffix lengths read from upstream."""
    enc = upstream_encoder(iris_dir)
    tok = enc.tokenizer
    assert enc._prefix_len == T.PREFIX_LEN and enc._suffix_len == T.SUFFIX_LEN, (enc._prefix_len, enc._suffix_len)
    assert enc._pad_id == T.PAD_ID and enc.max_length == T.WINDOW, (enc._pad_id, enc.max_length)
    assert tuple(enc.hidden_layers) == T.TAPS, enc.hidden_layers
    spec = json.loads((T.TE_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = spec["model"]["vocab"]
    merges = spec["model"]["merges"]
    added = {a["content"]: a["id"] for a in spec["added_tokens"]}
    assert added == tok.get_added_vocab(), "added tokens differ between tokenizer.json and the HF tokenizer"
    for a in spec["added_tokens"]:
        assert not a["normalized"] and not a["lstrip"] and not a["rstrip"] and not a["single_word"], a
    size = max(max(vocab.values()), max(added.values())) + 1
    tokens = [None] * size
    for t, i in list(vocab.items()) + list(added.items()):
        tokens[i] = t
    assert all(t is not None for t in tokens), "the vocabulary has gaps"
    assert not any(any(c.isspace() for c in t) for t in tokens), "a token holds whitespace"
    (host / "qwen_vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    (host / "qwen_merges.txt").write_text(
        "\n".join(m if isinstance(m, str) else " ".join(m) for m in merges) + "\n", encoding="utf-8")
    (host / "qwen_added.txt").write_text(
        "\n".join(f"{c}\t{i}" for c, i in sorted(added.items(), key=lambda kv: kv[1])) + "\n", encoding="utf-8")
    (host / "prompt_prefix_ids.txt").write_text(",".join(map(str, enc._prefix_ids.tolist())) + "\n")
    (host / "prompt_suffix_ids.txt").write_text(",".join(map(str, enc._suffix_ids.tolist())) + "\n")
    return {"prefix_len": enc._prefix_len, "suffix_len": enc._suffix_len, "vocab_size": size, "merges": len(merges)}


def rope_tables(host):
    """The text encoder's rotary tables as the model builds them: rotary_emb(hidden, position_ids) of
    Qwen3VLTextModel.forward with no position_ids given (every MRoPE axis = arange(334))."""
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    rotary = Qwen3VLTextRotaryEmbedding(AutoConfig.from_pretrained(T.TE_DIR).text_config)
    hidden = torch.zeros(1, T.SEQ, T.HIDDEN, dtype=torch.float32)
    position_ids = torch.arange(T.SEQ).view(1, 1, -1).expand(4, 1, -1)[1:]
    cos, sin = rotary(hidden, position_ids)
    cos[0].numpy().astype("<f4").tofile(host / "te_rope_cos.bin")
    sin[0].numpy().astype("<f4").tofile(host / "te_rope_sin.bin")
    # the plain 1D formula (te_modules.host_rope) gives the same tables: text-only MRoPE is 1D RoPE
    my_cos, my_sin = T.host_rope()
    return {"vs_host_rope_max_abs": {"cos": float((cos[0] - my_cos[0, 0]).abs().max()),
                                     "sin": float((sin[0] - my_sin[0, 0]).abs().max())}}


def host_pack(host, model_dir):
    """iris_host.f32 and its index {name: offset, count, shape, sha256}."""
    dit = json.loads((host / "dit_manifest.json").read_text())
    entries = [("te_rope_cos", host / "te_rope_cos.bin", [1, 1, T.SEQ, T.HEAD_DIM]),
               ("te_rope_sin", host / "te_rope_sin.bin", [1, 1, T.SEQ, T.HEAD_DIM])]
    for n in HOST_ORDER[2:]:
        e = dit["files"][n]
        assert e["dtype"] == "float32", (n, e["dtype"])
        entries.append((n, host / e["file"], e["shape"]))
    index, offset = {}, 0
    for n, src, shape in entries:
        count = int(np.prod(shape))
        assert src.stat().st_size == 4 * count, (n, src.stat().st_size, shape)
        index[n] = {"offset": offset, "count": count, "shape": shape, "sha256": sha256(src)}
        offset += 4 * count
    tmp = model_dir / "iris_host.f32.part"
    with open(tmp, "wb") as f:
        for _, src, _ in entries:
            f.write(src.read_bytes())
    os.replace(tmp, model_dir / "iris_host.f32")
    return index, dit["mod_rows"]


def check_config(cfg, index, mod_rows, model_dir):
    """Structural mismatches (a failure) and the host tensors whose sha256 differs from the config's source_sha256
    (none for the published checkpoint; expected for a fine-tune, whose tables differ)."""
    from fp16_weights import signature

    problems = []
    tensors = cfg["host"]["tensors"]
    if cfg["host"]["file"] != "iris_host.f32" or list(tensors) != list(index):
        problems.append(f"host: file {cfg['host']['file']}, names {list(tensors)}")
    sha_differs = []
    for n, e in index.items():
        c = tensors.get(n, {})
        got = (c.get("offset"), c.get("count"), c.get("shape"))
        if got != (e["offset"], e["count"], e["shape"]):
            problems.append(f"host {n}: config {got}, written {e}")
        if c.get("source_sha256") != e["sha256"]:
            sha_differs.append(n)
    g = cfg["graphs"]
    specs = g["te"] + [g["adapter"]] + g["dual"] + g["single"] + [g["pixel"]]
    for spec in specs:
        sig = signature(model_dir / spec["file"])
        want = {"signature": spec["signature"], "inputs": spec["inputs"], "input_shapes": spec["input_shapes"],
                "outputs": spec["outputs"], "output_shapes": spec["output_shapes"]}
        have = {"signature": sig["signature"], "inputs": [t[0] for t in sig["inputs"]],
                "input_shapes": [t[1] for t in sig["inputs"]],
                "outputs": [[t[0], int(np.prod(t[1]))] for t in sig["outputs"]],
                "output_shapes": [t[1] for t in sig["outputs"]]}
        if want != have or any(t[2] != "float32" for t in sig["inputs"] + sig["outputs"]):
            problems.append(f"graph {spec['file']}: config {want}, file {sig}")
    for kind in ("dual", "single"):
        want = [mod_rows[f"dit_{kind}{i}"] for i in range(3)]
        if g[f"{kind}_mod_rows"] != want:
            problems.append(f"{kind}_mod_rows: config {g[f'{kind}_mod_rows']}, export {want}")
    if cfg["tokenizer"] != "iris_tokenizer.json" or cfg["embed"] != "iris_embed_bf16.f32":
        problems.append(f"tokenizer / embed names: {cfg['tokenizer']} {cfg['embed']}")
    if cfg["sources"]["embed_bytes"] != (model_dir / "iris_embed_bf16.f32").stat().st_size:
        problems.append(f"embed bytes: config {cfg['sources']['embed_bytes']}")
    return {"graphs": len(specs), "host_tensors": len(index), "problems": problems,
            "host_sha256_differs_from_config": sha_differs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output root (default: out)")
    ap.add_argument("--iris", default=None, help="iris-3b Turbo checkpoint directory (for config.yaml)")
    ap.add_argument("--text-encoder", default=None, help="Qwen3-VL-4B-Instruct directory")
    ap.add_argument("--upstream", default=None, help="speridlabs/iris-3b checkout")
    ap.add_argument("--published", default=None, help="directory with the model repo's iris3b_config.json and SHA256SUMS")
    args = ap.parse_args()
    root = Path(args.out)
    iris_dir = Path(args.iris) if args.iris else root / "weights" / "iris-3b" / "turbo"
    T.TE_DIR = Path(args.text_encoder) if args.text_encoder else root / "weights" / "Qwen3-VL-4B-Instruct"
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    published = Path(args.published) if args.published else root / "published" / "model"
    sys.path.insert(0, str(upstream / "src"))
    torch.set_num_threads(8)
    t0 = time.time()
    host = root / "host"
    model_dir = root / "model"
    host.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rec = {"threads": torch.get_num_threads(), "torch": torch.__version__}

    rec["tokenizer"] = tokenizer_assets(iris_dir, host)
    files = {n: (host / n).read_text(encoding="utf-8") for n in TOKENIZER_FILES}
    doc = {"files": files, "sha256": {n: sha256(host / n) for n in TOKENIZER_FILES}}
    (model_dir / "iris_tokenizer.json").write_text(json.dumps(doc, ensure_ascii=True))

    w = T.load_tensors([EMBED])[EMBED]
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (T.VOCAB_ROWS, T.HIDDEN), (w.dtype, w.shape)
    tmp = model_dir / "iris_embed_bf16.f32.part"
    w.contiguous().view(torch.int16).numpy().astype("<i2").tofile(tmp)
    os.replace(tmp, model_dir / "iris_embed_bf16.f32")
    del w

    rec["rope"] = rope_tables(host)
    index, mod_rows = host_pack(host, model_dir)
    rec["host_index"] = index

    for name in ("iris3b_config.json", "SHA256SUMS"):
        shutil.copyfile(published / name, model_dir / name)
    cfg = json.loads((model_dir / "iris3b_config.json").read_text())
    rec["config_check"] = check_config(cfg, index, mod_rows, model_dir)
    rec["seconds"] = round(time.time() - t0, 1)
    (root / "model_assets.json").write_text(json.dumps(rec, indent=1))
    print("ASSETS_DONE", json.dumps({"rope": rec["rope"], "tokenizer": rec["tokenizer"],
                                     "config_problems": rec["config_check"]["problems"],
                                     "host_sha256_differs_from_config":
                                         rec["config_check"]["host_sha256_differs_from_config"],
                                     "seconds": rec["seconds"]}), flush=True)
    if rec["config_check"]["problems"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
