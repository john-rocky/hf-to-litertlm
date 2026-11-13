# Qwen3.5-122B-A10B on LiteRT-LM: the hybrid MoE text decoder written with the Tensor API

[Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) (revision `dc4d3484`, Apache License 2.0) is
Qwen's hybrid mixture-of-experts model: 48 layers in the layout 12 × (3 × Gated DeltaNet + 1 × gated attention), each
followed by a mixture of 256 experts of which 8 routed and 1 shared are used for each token; 122B parameters, 10B of
them activated (source model card). This directory writes its text decoder straight from the bf16 checkpoint with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), one tensor at a time, and packs it into
the two `.litertlm` files of [litert-community/Qwen3.5-122B-A10B](https://huggingface.co/litert-community/Qwen3.5-122B-A10B):
`qwen35_122b_a10b_mixed_int4_gpu_ekv4096.litertlm` (70.3 GB) for the GPU and
`qwen35_122b_a10b_mixed_int4_ekv4096.litertlm` (81.1 GB) for the CPU. The vision encoder and the
multi-token-prediction layers of the checkpoint are not converted.

- Gated DeltaNet layers in ordinary ops: the projections, the causal convolution over a 3-row state, the l2-normalized
  q / k, the gated delta recurrence (one step per decoded token; chunks of 64 tokens in prefill) and the gated RMSNorm.
  Each layer keeps a recurrent state of 64 × 128 × 128 and its last 3 inputs of 12,288 channels; both are
  `kv_cache_r_<i>` / `kv_cache_c_<i>` inputs and outputs, which LiteRT-LM treats as linear-attention states. Prefill
  slots are real when their `input_pos` continues the first slot's; pad slots leave both states unchanged.
- Gated attention layers: q / k RMSNorm per head, rotary embedding on the first 64 of 256 dimensions, grouped-query
  attention over the KV cache, the output times sigmoid of the per-head gate.
- Each layer's 256 experts are one TFLite custom op `moe` with SiLU: only the 8 routed experts are computed for each
  token. The router (fp32), its softmax, top-8 and the renormalization of the top-8 weights, and the shared expert with
  its sigmoid gate are ordinary ops in the graph.
- Weights: experts int4 blockwise (one fp32 scale per block of 32 values in the CPU file and of 128 values in the GPU
  file, scales below 2^-14 raised to 2^-14), the Gated DeltaNet and attention projections, the shared expert and
  lm_head int8 per channel; the router, the shared expert's gate, the Gated DeltaNet's a / b projections, the norms and
  the token table fp32. One token table of 248,320 × 3,072 fp32 is 3.05 GB, and the WebGPU delegate of the runtime
  below crashed while it created that buffer. The CPU file holds it in the graph as two tables of 1,536 columns, looked
  up with the same ids and concatenated (the same values). The GPU file holds it as one table in a model of its own, in
  a section in front of the decoder, and the decoder's signatures take the embeddings in place of the token ids.
  Activations, the KV cache and both states are fp32; KV 4,096 tokens.
- Signatures `prefill_128`, `prefill_1024` and `decode`. Each file carries LiteRT-LM's own
  `models/qwen3_5/LlmMetadataProto.pbtext` at `dab2e708` verbatim, the checkpoint's own `tokenizer.json` bytes, and
  declares fp32 activations.

Why the Tensor API: the converter path would export the model in fp32 first (4 bytes per parameter: 488.4 GB for the
decoder's 122.1 billion parameters, counted from the pack) and quantize in a second step; here the pack reads one
tensor at a time (peak footprint 3.74 GB, the largest of its 49 parts) and the writer maps the packed weights (peak
footprint 3.41 GB).

## The runtime it needs

LiteRT-LM 0.19.1 (PyPI, macOS arm64) does not run these files. On the GPU it stops at engine creation with
`CUSTOM moe: moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu').` and `Failed to create engine`
(the CLI still exits 0); on the CPU it prints `ERROR: moe node #236 unsupported activation='silu'` for each `moe`
node and stops with `Failed to invoke the compiled model` (exit 0 as well). The experts use SiLU, and the `moe`
kernels of LiteRT-LM 0.19.1 accept only GELU. LiteRT-LM 0.19.2 from PyPI, run on 2026-11-13, stops with the same two
messages on a 4-layer test file with the same `moe` op. SiLU in the `moe` op is in two LiteRT pull requests, open on 2026-11-10 and
still open on 2026-11-14: [#11192](https://github.com/google-ai-edge/LiteRT/pull/11192) (the GPU delegate's kernel)
and [#11188](https://github.com/google-ai-edge/LiteRT/pull/11188) (the XNNPACK kernel). The checks below ran on
`litert_lm_advanced_main` built on 2026-10-28 by the steps of [`../qwen3_30b_a3b_work/`](../qwen3_30b_a3b_work/)
(LiteRT-LM `dab2e708`, LiteRT `dcb657e8` and ML Drift `76492bb9` with the WORKSPACE's patch commands, and the two pull
requests), the same files as there (sha256 `litert_lm_advanced_main` a545652b…, `libLiteRt.dylib` 5684d285…,
`libLiteRtWebGpuAccelerator.dylib` e08d12aa…); the logits comparisons ran through a small C++ runner that calls
the graph's signatures directly and writes the logits, linked against the same `libLiteRt.dylib` and, on the GPU,
the same `libLiteRtWebGpuAccelerator.dylib` (the runner is not in this directory). On 2026-11-10 `gh pr diff` gave the same two diffs that build applied
(sha256 `9b413402…` for #11192, `6a996ece…` for #11188), and LiteRT-LM 0.19.1 (2026-11-05) was the latest release.

Run the GPU file on the Mac's GPU (WebGPU), with the file's own sampler as in the first of its 3 sampled runs below
(`--sampler_top_k=1` makes it greedy):

```bash
./litert_lm_advanced_main --backend=gpu --model_path=qwen35_122b_a10b_mixed_int4_gpu_ekv4096.litertlm \
  --max_output_tokens=2048 --min_log_severity=2 --sampler_handles_input=false \
  --cache_dir=:nocache --input_prompt="What is 17 + 25? Answer briefly."
```

Its engine takes about 2 minutes to load, and its peak footprint in the checks below was at most 72.02 GB on the
128 GB Mac used here. Every GPU run used `--sampler_handles_input=false`. The CPU file on the GPU needs about 106 GB
(peak footprint up to 106.2 GB) and did not pass the 12-question check (below).

Run the CPU file on the Mac's CPU (the command of its 12-question check below):

```bash
./litert_lm_advanced_main --backend=cpu --model_path=qwen35_122b_a10b_mixed_int4_ekv4096.litertlm \
  --max_output_tokens=2048 --min_log_severity=2 --sampler_top_k=1 --num_cpu_threads=12 \
  --cache_dir=:nocache --input_prompt="What is 17 + 25? Answer briefly."
```

## Files

| file | what it does |
|---|---|
| `tapi_pack_q35moe.py` | reads the checkpoint one tensor at a time (one expert at a time for the fused expert tensors) and writes, per part (`--part globals`, `--part layer:<N>`, then `--part index`), `weights.bin` + `index.tsv`, then `parts.txt`, `config.tsv` and the two custom ops' options: experts stacked in the `moe` op's layout and quantized with ai-edge-quantizer's own functions, the zero-centered RMSNorm weights stored as 1 + w. |
| `graph/qwen35moe_writer.cc`, `graph/BUILD` | the Tensor API graph (see above); `--out_offset` writes it into a `.litertlm` from a byte offset on; `--external_embedder` and `--embedder_only --embedder_out` write the GPU file's decoder and its token table as two models (step 5). |
| `patches/*.diff`, `overlay.sh` | three patches to the Tensor API at LiteRT `dcb657e8` (blockwise scales held in a named tensor, custom op codes kept apart by name, saving at an offset) and the link of `graph/` into the checkout as `tensor/examples/qwen35moe`. |
| `make_bundle.py`, `template/qwen3_5_official_dab2e708.pbtext` | the `.litertlm`: the official Qwen3.5 metadata (sha256 `2bdb4bcb…`), the tokenizer (sha256 checked), the decoder, fp32 activations declared; in place for a graph too big to copy (`--plan-offset`, `--head-only`); `--embedder` adds the token table's model as a section in front of the decoder (step 5); `--ids-from` copies the header's uuid and creation time from another file. |

`../qwen3_30b_a3b_work/` has `scale_scan.py` (zero and non-finite scales, per part here), `bundle_sections.py` (a
`.litertlm`'s sections with their sha256) and `apply_ws_patch_cmds.py` (for the runtime build).

## Steps

The Python steps ran with Python 3.12.11, torch 2.14.1, safetensors 0.8.0, numpy 2.5.4, ai-edge-quantizer 0.10.0,
flatbuffers 25.12.19, protobuf 7.36.2 and litert-lm-builder 0.19.0; `requirements.txt` is that environment's full
freeze (`python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`). Disk: the checkpoint 250 GB; for
the CPU file the pack 81.1 GB and the bundle 81.1 GB; for the GPU file (step 5) the pack 70.2 GB, the embedder 3.05 GB
and the bundle 70.3 GB.

1. The checkpoint:

   ```bash
   hf download Qwen/Qwen3.5-122B-A10B --revision dc4d348443bc740c68e2d77492492c11606384d5 --local-dir SNAPSHOT
   ```

2. Pack the weights, one part per layer (the 48 layer parts are independent and can run in any order or at once),
   then check the scales:

   ```bash
   F="--layers all --experts xb4 --block 32 --fc int8 --lm-head int8 --embed fp32"
   python tapi_pack_q35moe.py SNAPSHOT PACK $F --part globals
   for n in $(seq 0 47); do python tapi_pack_q35moe.py SNAPSHOT PACK $F --part layer:$n; done
   python tapi_pack_q35moe.py SNAPSHOT PACK $F --part index
   for d in PACK/globals PACK/L??; do python ../qwen3_30b_a3b_work/scale_scan.py $d $d/scale_scan.json; done
   ```

3. Build the writer in a LiteRT checkout at `dcb657e8` (the LiteRT that LiteRT-LM 0.19.0 pins):

   ```bash
   git clone https://github.com/google-ai-edge/LiteRT.git && git -C LiteRT checkout dcb657e8a537b2dc944e53e14c7eb43bcd9ededa
   bash overlay.sh LiteRT
   (cd LiteRT && bazel build -c opt //tensor/examples/qwen35moe:qwen35moe_writer)
   ```

4. Write the decoder into the bundle in place (the graph is 81.1 GB; copying it into a bundle would need the space
   twice). `--moe_fusion_guard` keeps the GPU delegate from fusing the residual ADD into the moe kernel (it adds
   `+ moe * 0`, exact); `--split_embed_cols 2` holds the token table as two tables:

   ```bash
   OUT=qwen35_122b_a10b_mixed_int4_ekv4096.litertlm
   X=$(python make_bundle.py --plan-offset --tokenizer SNAPSHOT/tokenizer.json --out $OUT | awk '/^TFLITE_OFFSET/{print $2}')
   rm -f $OUT   # the writer refuses a file that already runs past the offset
   LiteRT/bazel-bin/tensor/examples/qwen35moe/qwen35moe_writer --weights_dir=PACK --out=$OUT --out_offset=$X \
     --prefill=128,1024 --cache=4096 --gdn plain --moe_fusion_guard --split_embed_cols 2
   python make_bundle.py --head-only --tflite-offset $X --tokenizer SNAPSHOT/tokenizer.json --out $OUT
   python ../qwen3_30b_a3b_work/bundle_sections.py $OUT
   ```

5. The GPU file, `qwen35_122b_a10b_mixed_int4_gpu_ekv4096.litertlm`: steps 2 and 4 with three changes. The experts'
   int4 blocks are 128 values long (`--block 128`, a pack of its own). The token table is a model of its own, written
   first (`--embedder_only`) and put in a section in front of the decoder (`make_bundle.py --embedder`). The decoder is
   written with `--external_embedder` (its signatures take the embeddings in place of the token ids, and it holds no
   token table) and without `--split_embed_cols`:

   ```bash
   F128="--layers all --experts xb4 --block 128 --fc int8 --lm-head int8 --embed fp32"
   python tapi_pack_q35moe.py SNAPSHOT PACK128 $F128 --part globals
   for n in $(seq 0 47); do python tapi_pack_q35moe.py SNAPSHOT PACK128 $F128 --part layer:$n; done
   python tapi_pack_q35moe.py SNAPSHOT PACK128 $F128 --part index
   for d in PACK128/globals PACK128/L??; do python ../qwen3_30b_a3b_work/scale_scan.py $d $d/scale_scan.json; done
   W=LiteRT/bazel-bin/tensor/examples/qwen35moe/qwen35moe_writer
   OUT=qwen35_122b_a10b_mixed_int4_gpu_ekv4096.litertlm
   $W --weights_dir=PACK128 --embedder_only --embedder_out=embedder.tflite
   X=$(python make_bundle.py --plan-offset --tokenizer SNAPSHOT/tokenizer.json --embedder embedder.tflite --out $OUT | awk '/^TFLITE_OFFSET/{print $2}')
   rm -f $OUT
   $W --weights_dir=PACK128 --out=$OUT --out_offset=$X \
     --prefill=128,1024 --cache=4096 --gdn plain --moe_fusion_guard --external_embedder
   python make_bundle.py --head-only --tflite-offset $X --tokenizer SNAPSHOT/tokenizer.json --embedder embedder.tflite --out $OUT
   python ../qwen3_30b_a3b_work/bundle_sections.py $OUT
   ```

## How it was checked

All on the Mac M4 Max (128 GB) with the runtime above: the tiny model and the CPU file on 2026-11-09 and 2026-11-10,
the GPU file on 2026-11-13, and the step 5 check from a clean copy on 2026-11-14.

- The graph on a tiny model of the same architecture (4 layers: 3 Gated DeltaNet and 1 gated attention, hidden size
  256, 16 experts of which 4 are used for each token, fp32 weights): the logits of the CPU and of WebGPU are within
  2.2e-6 of Transformers' at every position, for decode alone (40 positions) and for a 16-slot prefill of 12 tokens and
  4 pad slots followed by decode (28 positions); the top token is the same at every position.
- The CPU file against the author's weights: the first 24 positions of a fixed prompt (a GSM8K question in the chat
  template), fed one token at a time; at each position the file's top next token against the top token of
  Transformers' own layers run one layer at a time from the bf16 weights. WebGPU: 23 of 24. CPU: 20 of 24, the same in
  each of three processes (the logits are identical); the conversion asked for 22, so the CPU fails this check. The
  source model's top token is among the CPU's top five at every position. Against a second reference, the same layers
  with the pack's quantized weights turned back into fp32, WebGPU picks the same token at 24 of 24 (largest logit
  difference 6.96e-05) and the CPU at 21 of 24 (it differs at positions 0, 3 and 4). At position 17, where both
  backends differ from the bf16 reference, the quantized-weight reference picks the same token as the file.
- The CPU file, 12 questions and a 3-turn conversation, thinking on (the chat template's default), greedy
  (`--sampler_top_k=1`), up to 2,048 tokens per answer, each question in a new process; the pass mark is 11 of 12 right, every answer (the 12
  and the 3 turns) ending with the stop token, and the third turn recalling a number from the first. CPU (12
  threads): 12 of 12 right, 12 of 12 ended with the stop token, the 3 turns pass. WebGPU: questions 1 to 3 right and
  ended with the stop token; question 4's reply ran past the 10 minutes LiteRT-LM's CLI waits for a reply (about 700
  tokens in 775 s) and the process ended with a segmentation fault, so the GPU fails this check. During that run the
  engine's peak footprint was 105.9 to 106.2 GB on the 128 GB Mac.
- Steps 1 to 4 from a clean copy on 2026-11-10, on a part of the model: `requirements.txt` into a new venv (its pip
  freeze the same 74 package versions) and the writer built at LiteRT `dcb657e8`; step 2 for 3 of the 49 parts (`globals`,
  `layer:0`, a Gated DeltaNet layer, and `layer:3`, a gated-attention layer) wrote each part's `weights.bin` and
  `index.tsv` byte for byte as in the shipped file's pack; the writer built here and the writer that wrote the shipped
  file wrote the same bytes from a 2-layer pack (layers 0 and 3) with step 4's flags; step 4's bundle steps, with
  `make_bundle.py --ids-from` set to the shipped file, wrote its first 3,653,632 bytes (the header, the metadata, the
  tokenizer and the section table) byte for byte. The other 46 pack parts and the 48-layer write did not run again.
- The GPU file against the author's weights, the same 24 positions on WebGPU: the source model's top token at 24 of 24.
- The GPU file, the same 12 questions and 3-turn conversation on WebGPU. Greedy: 11 of 12 right and 11 of 12 ending
  with the stop token (question 2's reply reached the 2,048-token cap inside the thought), and the 3 turns pass. With
  the file's own sampler (type `TOP_P`: top_p 0.95, top_k 20, temperature 0.6), in each of 3 runs (the CLI's default
  seed, `--sampler_seed=2` and `--sampler_seed=3`): 12 of 12 right, 12 of 12 ending with the stop token, and the 3
  turns pass. So greedy decoding misses the pass mark above (question 2 did not end with the stop token), and the
  file's own sampler meets it in each of the 3 runs.
- Step 5 from a clean copy on 2026-11-14, on a part of the model: `requirements.txt` into a new venv (its pip freeze
  the same 74 package versions) and the writer built again at LiteRT `dcb657e8`; step 5's pack (`--block 128`) for the
  same 3 parts wrote each part's `weights.bin` and `index.tsv` byte for byte as in the GPU file's pack; the writer
  built here, with `--embedder_only`, wrote the GPU file's embedder section byte for byte; the writer built here and
  the writer that wrote the GPU file wrote the same bytes from a 4-layer pack (the GPU file's layers 0 to 3) with
  step 5's flags; step 5's bundle steps, with `make_bundle.py --ids-from` set to the GPU file, wrote its first
  3,055,026,176 bytes (the header, the metadata, the tokenizer, the embedder section and the section table) byte for
  byte. The other 46 pack parts and the 48-layer write did not run again.

Not shipped:

- The token table as one fp32 constant of 3,051,356,160 bytes in the decoder's graph: WebGPU stopped with a segmentation fault while it
  created the engine, on a 2-layer file with this model's weights and on one with synthetic weights in the same
  shapes; the synthetic file with the table split in two made an engine and decoded.
- The CPU file on WebGPU with the weights converted on the host (`--convert_weights_on_gpu=false`): the engine load
  passed 92.3 GB of footprint, above the 90 GB memory limit set for that run, and was stopped there.

Speeds and answers: the model card.
