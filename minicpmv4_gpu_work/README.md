# MiniCPM-V-4 → LiteRT-LM, GPU files

These scripts make the two GPU files of [litert-community/MiniCPM-V-4](https://huggingface.co/litert-community/MiniCPM-V-4) from [openbmb/MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4) at revision `6f69a823` (Apache-2.0):

- `MiniCPM-V-4-int8-gpu.litertlm` (4,578,480,128 B): the text decoder with int8 weights per channel; for a Mac GPU.
- `MiniCPM-V-4-mixed-int4-gpu.litertlm` (4,315,598,848 B): the text decoder with int4 blockwise-32 weights, the 32 MLP down projections with fp16 weights and the output head int8; for Android phone GPUs.

Both carry the token embedder (int8) in its own section, the SigLIP vision encoder and resampler (fp16 weights), the source tokenizer and a chat template, for LiteRT-LM's `minicpmv4` model type (the runtime slices each image into a thumbnail and up to 9 slices of 64 tokens). The repository's `MiniCPM-V-4-int8.litertlm` (CPU) is another conversion and is not made here. Speeds, answers and limits: the model card.

## Run

```bash
hf download openbmb/MiniCPM-V-4 --revision 6f69a8235885be89608f3a71ffd40695b379e1e7 --local-dir src
PY=/path/to/venv/bin/python3 LT=/path/to/litert-lm-venv/bin XR_TOOLS_PATH=/path/to/litert-tools \
  bash minicpmv4_gpu_work/reproduce.sh src work
```

Versions: Python 3.12.11, litert-torch 0.10.0, ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0, torch 2.14.1, transformers 5.20.2, litert-lm-builder 0.19.0; the litert-lm 0.19.0 CLI for the externalize step; LiteRT's externalize tools at `8761012b` (`tools/externalize/`).

## Steps

1. `make_text_dir.py`: the decoder's weights (bf16 as published) and a config of model type `minicpmv4_text` (`v4_text_lm.py`: a Llama decoder whose rotary embedding takes the checkpoint's short factors; its long and short factors are equal and the factor is 1.0, so this is the same arithmetic without the data-dependent branch `export_hf` refuses).
2. `text_export.py`: litert-torch `export_hf`, fp32, task `text_generation`, the embedder in its own graph (the decoder takes embeddings, so the image features land at the image tokens), prefill signatures 1,024 / 256 / 64 / 16, KV cache 4,096. Run with `PYTHONHASHSEED=0`.
3. `quant.py` and `quant_mixed.py`: ai-edge-quantizer. `wi8` = every supported op int8 per channel (dynamic); `emb8` = the embedding lookup int8 per channel; the mixed int4 form = every fully connected layer int4 blockwise-32 (min-max) with the output head int8 (the recipe of `minicpm5_work/quantize_minicpm5.py`), and `--fp16-fc 'Linear_down_proj;'` gives the 32 down projections fp16 weights instead. Int4 block scales are floored at 2^-14.
4. `tools/gpu_graph/gpu_graph_retrofit.py`: the GPU composites for the cache update and the attention matrix products, weights byte-equal.
5. `v4_vision.py export`: the navit SigLIP encoder and the resampler rewritten in plain PyTorch ops (one-hot lookups instead of gathers, the resampler's query side folded into one fully connected layer, masks as -1e4 additions), converted with litert-torch, signature `vision_1088` (the runtime pads each slice to 1,088 patches), fp16 weights.
6. `build_bundle.py` + `chat_template_v4.jinja`: the `.litertlm` (type `minicpmv4`, start `<s>`, stop ids 2 and 73440, `max_num_tokens` 4096, the source `tokenizer.json` verbatim, no backend constraint); then `tools/externalize/externalize_repo.sh` moves the decoder and embedder weights to TFLiteWeights sections, adds the ExecutorMetadata section and runs a CPU smoke on a photo.

## Checks

- Before the export, the text checkpoint was checked against the source model's own LLM on probe sequences (`make_text_dir.py --prove`; the probes come from a PyTorch reference run that is not in this directory).
- The vision rewrite in fp32 against the source resampler output per slice, then the TFLite form on the CPU and the GPU.
- The files: 12 questions (8 text, 4 photos) against the source model in PyTorch fp32 on the CPU (greedy), correct answers and teacher-forced top-1 agreement; 12 text questions about passages of 88 to 701 tokens; Benchmark 1,024 / 256 on a Mac M4 Max, a Galaxy S26 and a Pixel 11 Pro. Results: the model card.

## The shipped files

Each section's sha256 as stored in the shipped files, from the externalize step's report of each build:

| section | `MiniCPM-V-4-int8-gpu.litertlm` | `MiniCPM-V-4-mixed-int4-gpu.litertlm` |
|---|---|---|
| LlmMetadataProto (616 B) | `32cb0ac0…` | `32cb0ac0…` |
| ExecutorMetadataProto (4,339 B) | `c95f7e89…` | `c95f7e89…` |
| HF_Tokenizer_Zlib (1,099,309 B) | `ceaca8d6…` | `ceaca8d6…` |
| TFLiteModel prefill_decode | 65,826,864 B `2defdfdc…` | 1,837,176,000 B `200758a8…` |
| TFLiteModel embedder (4,416,416 B) | `707f65e5…` | `707f65e5…` |
| TFLiteModel vision_encoder (901,380,880 B) | `62dcb855…` | `62dcb855…` |
| TFLiteWeights prefill_decode | 3,417,640,960 B `0bd523bb…` | 1,383,403,520 B `d09b8d13…` |
| TFLiteWeights embedder (188,026,880 B) | `d55d9a1e…` | `d55d9a1e…` |
| whole file | `9de381c48eef28ebc304c7028f2ae41b99904cfa7130b8e55ff7de0eeebb50d9` | `227774abf34bd517bbb9d90a8f7f53154037db693568c1dacb1372e97aefcae4` |

The builder writes a new uuid and creation time into the container header at every build, so a rebuilt file's whole-file sha256 differs even when every section is the same.

These are the scripts that built the shipped files on 2026-11-02 and 2026-11-03, with comments, docstrings and paths changed for this repository. `reproduce.sh` chains the same commands; it has not been run end to end from this repository, so we make no claim that it gives the same section bytes.
