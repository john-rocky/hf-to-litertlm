#!/usr/bin/env python3
"""Rewrite the Confucius4-R2T2 checkpoint (original `qwen_asr` layout) into the transformers v5
`Qwen3ASRForConditionalGeneration` (-hf) layout, by key rename only.

The tensor payload is copied byte for byte: only the safetensors JSON header is rewritten (key
names), every data offset and every byte after the header stays the same, so the values remain
the original bf16 numbers (no permute, no transpose, no cast).

  thinker.audio_tower.proj1.*  -> model.multi_modal_projector.linear_1.*
  thinker.audio_tower.proj2.*  -> model.multi_modal_projector.linear_2.*
  thinker.audio_tower.*        -> model.audio_tower.*
  thinker.model.*              -> model.language_model.*

Small files: config.json / generation_config.json / processor_config.json verbatim from
Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7 (that repo has no preprocessor_config.json; processor_config.json
is its feature-extractor + processor file), tokenizer files and chat_template.json verbatim from
Confucius4-R2T2 @ 185ce639.

Check (a): renamed name/shape/dtype set == HfApi.get_safetensors_metadata("Qwen/Qwen3-ASR-1.7B-hf").
Writes out/hf_layout/ and out/layout_check_a.json.

  ~/venvs/ltmain0918/bin/python convert_layout.py
"""
import hashlib
import json
import os
import shutil
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SRC = os.path.expanduser(
    "~/.cache/huggingface/hub/models--netease-youdao--Confucius4-R2T2/snapshots/"
    "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9")
HF17_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
HF17_REV = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
HF17_SMALL = os.path.join(OUT, "hf17_small")  # curl'ed from resolve/<rev>/<file>
DST = os.path.join(OUT, "hf_layout")

FROM_HF17 = ["config.json", "generation_config.json", "processor_config.json"]
FROM_C4 = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
           "special_tokens_map.json", "chat_template.json"]


def rename(key):
    for old, new in (("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
                     ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
                     ("thinker.audio_tower.", "model.audio_tower."),
                     ("thinker.model.", "model.language_model.")):
        if key.startswith(old):
            return new + key[len(old):]
    raise ValueError(f"unmapped key: {key}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(DST, exist_ok=True)
    src_st = os.path.join(SRC, "model.safetensors")
    with open(src_st, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data_start = 8 + n
    meta = header.pop("__metadata__", None)
    new_header = {}
    for k, v in header.items():
        nk = rename(k)
        assert nk not in new_header, nk
        new_header[nk] = v
    if meta is not None:
        new_header = {"__metadata__": meta, **new_header}
    hb = json.dumps(new_header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    dst_st = os.path.join(DST, "model.safetensors")
    with open(src_st, "rb") as fi, open(dst_st, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        fi.seek(data_start)
        shutil.copyfileobj(fi, fo, 1 << 24)
    src_payload = os.path.getsize(src_st) - data_start
    dst_payload = os.path.getsize(dst_st) - 8 - len(hb)
    assert src_payload == dst_payload, (src_payload, dst_payload)

    for fn in FROM_HF17:
        shutil.copyfile(os.path.join(HF17_SMALL, fn), os.path.join(DST, fn))
    for fn in FROM_C4:
        shutil.copyfile(os.path.realpath(os.path.join(SRC, fn)), os.path.join(DST, fn))

    # Check (a): name/shape/dtype set against the -hf repo's safetensors metadata (no weight download).
    from huggingface_hub import HfApi
    ref = HfApi().get_safetensors_metadata(HF17_REPO, revision=HF17_REV)
    ref_t = {}
    for fm in ref.files_metadata.values():
        for name, ti in fm.tensors.items():
            ref_t[name] = (ti.dtype, list(ti.shape))
    ours = {k: (v["dtype"], v["shape"]) for k, v in new_header.items() if k != "__metadata__"}
    only_ours = sorted(set(ours) - set(ref_t))
    only_ref = sorted(set(ref_t) - set(ours))
    shape_diff = sorted(k for k in set(ours) & set(ref_t) if list(ours[k][1]) != list(ref_t[k][1]))
    dtype_diff = sorted(k for k in set(ours) & set(ref_t) if ours[k][0] != ref_t[k][0])
    params = sum(eval("*".join(map(str, s))) if s else 1 for _, s in ours.values())
    res = {
        "src_safetensors": src_st, "src_sha256": sha256(src_st), "src_bytes": os.path.getsize(src_st),
        "dst_safetensors": dst_st, "dst_sha256": sha256(dst_st), "dst_bytes": os.path.getsize(dst_st),
        "payload_bytes": src_payload, "payload_copied_byte_for_byte": True,
        "n_src": len(header), "n_dst": len(ours), "n_ref": len(ref_t), "params": params,
        "ref_repo": HF17_REPO, "ref_rev": HF17_REV,
        "only_ours": only_ours, "only_ref": only_ref, "shape_diff": shape_diff, "dtype_diff": dtype_diff,
        "pass_a": not (only_ours or only_ref or shape_diff or dtype_diff) and len(ours) == len(ref_t) == 707,
        "small_files": {fn: sha256(os.path.join(DST, fn)) for fn in FROM_HF17 + FROM_C4},
        "small_files_source": {**{fn: f"{HF17_REPO}@{HF17_REV}" for fn in FROM_HF17},
                               **{fn: "netease-youdao/Confucius4-R2T2@185ce639" for fn in FROM_C4}},
    }
    with open(os.path.join(OUT, "layout_check_a.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "small_files"}, indent=1))


if __name__ == "__main__":
    main()
