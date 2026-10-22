# iris-3b Turbo → LiteRT graphs (float16 weights, float32 activations)

These scripts convert [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) Turbo (Apache-2.0: a 3B
pixel-space DiT that draws an image in 4 steps, with the language stack of
[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as its text encoder) into the files
published as [litert-community/iris-3b-Turbo-LiteRT](https://huggingface.co/litert-community/iris-3b-Turbo-LiteRT) for
768 × 768 images: 20 LiteRT graphs (12 for the text encoder, the text adapter, 6 trunk chunks and the pixel
stage) with every FULLY_CONNECTED weight stored as float16 behind a DEQUANTIZE, the tokenizer, the bfloat16 embedding
table and the host tensors (`iris_host.f32`: precomputed inputs of the graphs and the sampler). Run as written below,
they write the published files byte for byte.

The files run in an Android app ([LiteRT-Models `iris3b/`](https://github.com/john-rocky/LiteRT-Models/tree/main/iris3b),
every graph on the phone's GPU at FP32 precision with LiteRT 2.2.0) and, on the CPU, in the Python host
`python/iris3b_litert.py` of the model repo.

## Reproduce

```sh
cd iris3b_work
OUT=out   # every file goes under OUT: about 55 GB (checkpoints 15 GB, float32 graphs 25.7 GB, the model folder 13.7 GB)
# Python 3.12.11 with every package pinned (macOS arm64)
uv venv -p 3.12.11 .venv && uv pip install -p .venv -r requirements.txt
# inputs at pinned revisions: the checkpoint, the text encoder, the model repo's run configuration and sums, the upstream code
.venv/bin/hf download speridlabs/iris-3b turbo/config.yaml turbo/distill.yaml turbo/model.safetensors --revision adc05b909fb05d4276d26aae5a52b36406b2ef26 --local-dir "$OUT/weights/iris-3b"
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --include "*.json" --include "*.txt" --include "*.safetensors" --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir "$OUT/weights/Qwen3-VL-4B-Instruct"
.venv/bin/hf download litert-community/iris-3b-Turbo-LiteRT model/iris3b_config.json model/SHA256SUMS --revision 743fd20acefd7f9ddcf3585f88bcda3257f7213a --local-dir "$OUT/published"
git clone -q https://github.com/speridlabs/iris-3b "$OUT/upstream" && git -C "$OUT/upstream" checkout -q 13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e
(cd "$OUT/weights" && shasum -a 256 -c) < inputs.sha256
# float32 graphs (text encoder, DiT), checked against SHA256SUMS.fp32, and the DiT's host tables
.venv/bin/python export_te.py --out "$OUT"
.venv/bin/python export_dit.py --out "$OUT"
(cd "$OUT/fp32" && shasum -a 256 -c) < SHA256SUMS.fp32
# float16 weights, the host files and the run configuration -> OUT/model, checked against the model repo's sums
.venv/bin/python fp16_weights.py --out "$OUT"
.venv/bin/python make_assets.py --out "$OUT"
(cd "$OUT/model" && shasum -a 256 -c SHA256SUMS)
```

Run as printed on 2026-10-22 on macOS 27.0 (Apple M4 Max, 128 GB; OUT on an external USB disk; the environment came
from uv's cache): the 23 files the scripts write came out byte for byte (sha256 equal to the model repo's
`SHA256SUMS`; the configuration is the copied file), and the 20 float32 graphs matched `SHA256SUMS.fp32`. Its lines
for 16 graphs are the original conversion's; for `dit_dual0`, `dit_dual1`, `dit_single0` and `dit_single1` they are what
`export_dit.py` wrote in three earlier processes, not the original conversion's bytes (see below). The downloads took
396 s (6.0 GB and 8.9 GB), the conversion 645 s (text encoder 265 s, DiT 254 s, float16 weights 113 s, host
files 13 s) and the three checks 121 s. The largest process, `export_dit.py`, peaked at 22.7 GB of memory, and OUT
held 54.4 GB at the end.

`OUT/model` is then the model repo's `model/` folder: `iris_te_c00_f16w.tflite` … `iris_te_c11_f16w.tflite`,
`iris_dit_{adapter,dual0,dual1,dual2,single0,single1,single2,pixel}_f16w_dit.tflite`, `iris_embed_bf16.f32`,
`iris_host.f32`, `iris_tokenizer.json`, `iris3b_config.json` and `SHA256SUMS`.

| File | What it does |
|---|---|
| `te_modules.py` | The text encoder re-authored for fixed-shape GPU graphs: one graph per hidden state the DiT reads |
| `dit_modules.py` | The DiT re-authored as chunk modules: adapter, dual-stream chunks of 3 + 3 + 2 blocks, single-stream chunks of 6 + 6 + 4, pixel stage; the host-side tables |
| `export_te.py`, `export_dit.py` | The float32 graphs (litert-torch), and the DiT's host tables |
| `fp16_weights.py` | Every FULLY_CONNECTED weight to float16 + DEQUANTIZE (ai-edge-quantizer), with per-file checks |
| `make_assets.py` | Tokenizer files, embedding table, the text encoder's rotary tables, `iris_host.f32`; copies and checks the run configuration |
| `requirements.txt` | Every package version (`uv pip freeze`) |
| `inputs.sha256`, `SHA256SUMS.fp32` | sha256 of the downloaded inputs and of the 20 float32 graphs |

## The non-obvious parts

- **The graphs' tensor names carry the Python module and class names** (`te_modules.TEChunk/te_modules.TELayer_0`,
  `dit_modules.AdapterGraph/dit_modules.LayerBlock_0/dit_modules.Lin_qkv`). Renaming `te_modules.py`,
  `dit_modules.py`, a class or a submodule attribute changes the bytes, not the computation. So does the order of
  operations in `forward`: normalizing and transposing q and then k writes a different file from normalizing both and
  then transposing, with the same numbers. The values a graph is traced with do not enter the file: these scripts
  trace with zeros, the published graphs were traced with the activations of a real prompt, and the bytes are the same.
- **One text-encoder graph per tap.** The DiT reads the hidden states after layers 2, 5, …, 35 (1-based) of the
  36-layer language stack, over a 34-token chat-template prefix and a 300-token window. Each graph returns exactly one
  of them, so 35 layers run in 12 graphs. With no position ids, Qwen3-VL's interleaved MRoPE gives every axis the same
  positions and equals plain 1D RoPE (bit-equal on 15 prompts and all 12 taps), so the rotary tables are graph inputs.
  The 32 query heads attend the 8 KV heads through a grouped reshape of q (no repeat of k and v), and the attention
  scale is folded into the q-norm weight.
- **The DiT rewrites are exact in real arithmetic.** The interleaved RoPE becomes a half-split rotation, with the
  even / odd de-interleave baked into the q and k projection rows and norm weights. The 20 query heads are reordered so
  that each group of 5 is a plain attention over the 5 KV heads (no repeat, no rank-5 tensor). RMSNorm first
  divides each row by its largest magnitude when that is above 1, so a sum of squares never exceeds the row width.
  `F.unfold`, `F.fold` and the pixel grouping are permutations done on the host into one patch-major layout. The pixel
  stage's attention runs over all 2,304 patch tokens (each patch's 256 pixels compressed to one token), so it
  cannot be split per patch.
- **What depends only on t is precomputed.** Turbo samples at four fixed sigmas (1.0, 0.938, 0.833, 0.625) without
  classifier-free guidance, so the timestep embedding and every block's adaLN rows are host tables (`t_emb_s<i>`,
  `mods_s<i>` in `iris_host.f32`), and the text adapter runs once per prompt.
- **Float16 weights, float32 activations.** The 20 graphs take 12,869,097,872 bytes instead of 25,733,449,508.
  Float16 holds the checkpoint's bfloat16 weights exactly except values below 2^-17 in magnitude (the largest change
  2.98e-8) and the query rows of the text adapter's two LayerBlocks, which carry the folded softmax scale 80^-0.5 and
  so are not bfloat16 values (13,107,200 weights; the largest change 1.50e-5, at most 4.48e-4 relative for weights of
  magnitude 2^-14 or more). No weight becomes inf. In the conversion, int8 weights in either group of trunk chunks
  and float16 compute in the dual chunks, the single chunks or the pixel stage each missed the bar of 35 dB PSNR
  and 0.98 SSIM against the all-float32 PyTorch image on a prompt with numbers and punctuation (upstream's own bfloat16
  text encoder scores 30.3 dB there), so the weights are float16 and the activations float32.
- **Four float32 graphs depend on what was converted before them.** The float32 bytes of `dit_dual0`, `dit_dual1`,
  `dit_single0` and `dit_single1` (1.8 GB each, the only float32 graphs whose weights the converter writes after the
  flatbuffer) change with the graphs converted earlier in the same process, but the float16 step writes the weights
  in buffer order, so the published files come out the same, and the last command checks them.
- **litert-torch 0.9.4 lowers the text adapter's attention output projections to a rank-2 FULLY_CONNECTED and a
  mixed-rank ADD** (`[1, 1, 3600, 2560] + [3600, 2560]`, 300 rows in the refiner), with or without their bias, while
  its MLP output projections, which also feed a lone ADD, stay rank 4. With int8 weights, the GPU delegate of LiteRT
  2.2.0 refused the adapter (`Unable to parse bc coord for BATCH axis`, on a Mac GPU and on a Galaxy S26), also with
  the nine FCs that feed an ADD kept in float (tried on the Mac), so the adapter stayed float.
- **The host tables are computed with the torch thread counts the published files were made with**: 8 for the
  text encoder's rotary tables (`make_assets.py`), 12 for the DiT's tables (`export_dit.py`). In the original
  conversion, the plain 1D formula (`te_modules.host_rope`) run in a later process differed from the published rotary
  tables in 24 of 85,504 values, by at most 5.96e-8 (the cause was not isolated); in this run, that formula and
  transformers' rotary embedding on 8 threads both gave the published values exactly.
- **The run configuration is copied, not generated.** `iris3b_config.json` names the graph files, their inputs and
  outputs, the precision and accelerator of each graph and the offsets of the host tensors. Its `source` and
  `source_sha256` fields record the files of the original conversion; neither the app nor the Python host reads them.
  `make_assets.py` checks every host offset, count and shape in it against the `iris_host.f32` it writes, and every
  graph signature against the graph files in `OUT/model`.

## A fine-tune

A fine-tune of iris-3b Turbo saved like the original (`config.yaml`, `distill.yaml`, `model.safetensors`) takes the
same commands in a new OUT (the export and float16 scripts skip graph files that are already there), with
`--iris <dir>` added to `export_dit.py` and `make_assets.py`, and `--text-encoder <dir>` added to `export_te.py` and
`make_assets.py` if its text encoder changed. The changed files then fail both sum checks; `make_assets.py` still
checks the structure against the copied configuration and lists the host tensors whose sha256 differs from its
`source_sha256`. This has not been run on a fine-tune here.
