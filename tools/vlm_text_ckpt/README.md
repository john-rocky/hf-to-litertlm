# The text tower of a VLM as a plain CausalLM checkpoint

Some VLMs put a language model the LLM rail already exports under a vision tower: MiniCPM-V-4.7-1B carries
Qwen3.5's `qwen3_5_text` decoder (24 layers, GatedDeltaNet + full attention, untied head) as
`model.language_model.*`. The exporter loads a `*ForCausalLM` checkpoint, not the VLM class, so the text
tower goes out as a derived checkpoint:

```
python tools/vlm_text_ckpt/make_text_ckpt.py <vlm snapshot> <out dir> \
    --prefix model.language_model. --model-type qwen3_5_text --architectures Qwen3_5ForCausalLM \
    --report <out>.json --prove
```

`make_text_ckpt.py` keeps every tensor under `--prefix` (renamed to `model.*`) and the untied head
(`--head lm_head.weight`; when the VLM has none the derived config keeps `tie_word_embeddings = true`),
drops the tensors under `--drop` (default: `model.vision_tower.`, `model.merger.`) and refuses on any other
key, so a new family's layout is checked rather than guessed. It writes `config.json` from the VLM's
`text_config` (`--text-config`) with the given `architectures` and `model_type`, one `model.safetensors`
in the stored dtype, and copies the tokenizer files it finds (`--copy`) byte for byte. Sharded sources
(`model.safetensors.index.json`) are read shard by shard. The derived directory is what the LLM rail
(`scripts/convert.py`, `scripts/export_simple_template.py`) takes as its model.

## Prove the remap before exporting

`--prove` loads the full VLM (`AutoModelForImageTextToText`, or `--full-class <Name>`, `--trust-remote-code`
when the model needs it) and the derived checkpoint in fp32 with eager attention and compares their logits
on one text-only input: `--prove-ids ids.json` (a JSON list) or `--prove-text` through the VLM's tokenizer,
cut to `--prove-max-tokens`. The remap is a rename, so equal kernels must give max |Δ| 0 and the same argmax
at every position; a difference at the 1e-5 scale comes from the attention kernel (sdpa against eager), not
from the remap. The report JSON carries the counts, the file sizes and sha256 sums, and the proof. Memory: the two
models load in fp32 one after the other; weights alone are four bytes a parameter (MiniCPM-V-4.7-1B: full model
3.1 GB bf16 → 6.2 GB, derived checkpoint 2.0 GB → 4.0 GB).

MiniCPM-V-4.7-1B (`openbmb/MiniCPM-V-4.7-1B`, revision 984a89aa; 2026-10-13, re-run 2026-10-14 with this
script on transformers 5.19.0 / torch 2.14.1): 780 source tensors → 321 kept (320 renamed + `lm_head.weight`), 453 vision-tower and 6 merger
tensors dropped, 1,006,209,856 parameters in bf16, `model.safetensors` 2,012,456,880 B sha256 `1d87e52a…`;
full model vs derived checkpoint on 48 chat-templated tokens: max |Δ| 0.0, argmax 48/48 (eager vs eager);
the same derived checkpoint under sdpa differed from the full eager model by 5.3e-5. The bundles built from
it are `litert-community/MiniCPM-V-4.7-1B` (REPRODUCE.md, 2026-10-13).
