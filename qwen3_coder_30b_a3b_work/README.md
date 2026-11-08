# Qwen3-Coder-30B-A3B-Instruct on LiteRT-LM: the Qwen3-30B-A3B decoder with a chat template for XML tool calls

[Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (revision `b2cff646`,
Apache License 2.0) is a coding model with tool calling. Its `config.json` equals that of
[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) in every key but `rope_theta` (10,000,000 against
1,000,000), `max_position_embeddings` (262,144 against 40,960) and `transformers_version`: 48 layers, 128 SiLU experts
per layer of which 8 are used for each token, 30.5B parameters of which 3.3B are active. So the pack, the writer and
the bundle step are those of [`../qwen3_30b_a3b_work/`](../qwen3_30b_a3b_work/) (its README), unchanged; the pack and
the writer read `rope_theta` from `config.json`. This directory holds what differs, the LiteRT-LM metadata:

- **Chat template.** The source's `chat_template.jinja` joins a message's content as a string. LiteRT-LM 0.18 and
  later pass the content as a list of typed parts, on which that template raises. `template/qwen3_coder_litertlm.jinja`
  is the source's template with these edits: every read of a content goes through a `format_content` macro (a string
  as is; the text of `text` parts; the `response` of `tool_response` parts, as JSON when it is a mapping or a list); a
  tool message with N `tool_response` parts renders N `<tool_response>` blocks, which is what the source's template
  renders for N tool messages; `messages[0]` is read only when there are messages; `loop.previtem` and
  `loop.nextitem` became index reads; variables set inside a loop got new names; every tag trims its whitespace.
  (The template's first comment calls this list `K/TEMPLATE.md`, a file of the conversion's notes that is not in this
  repository.)
- **Tool calls.** The model writes its calls as XML: `<tool_call>`, then `<function=NAME>` with one
  `<parameter=NAME>` block per argument. The metadata declares `supports_function_calling`, the `generic_model` type
  and a `tool_call_config` copied field for field from LiteRT-LM's own `models/qwen3_5/LlmMetadataProto.pbtext` at
  LiteRT-LM `dab2e708` (`template/qwen3_5_official_dab2e708.pbtext`), a configuration for the same XML form. LiteRT-LM
  accepts a `tool_call_config` only with `generic_model`
  (`runtime/conversation/model_data_processor/model_data_processor_factory.cc`).
- **Stops and sampler.** Stop tokens `<|im_end|>` (151645) and `<|endoftext|>` (151643), the source's
  `generation_config.json` `eos_token_id`; sampler `TOP_P` with top_k 20, top_p 0.8 and temperature 0.7 from the same
  file. Its `repetition_penalty` of 1.05 has no field in LiteRT-LM's metadata.
- No thinking: the source supports only non-thinking mode, so the metadata has no channel and no start token
  (`add_bos_token` is false).

## The runtime it needs

The same as Qwen3-30B-A3B's, for the same reason (SiLU in the `moe` op): see
[The runtime it needs](../qwen3_30b_a3b_work/README.md#the-runtime-it-needs). On 2026-11-09, LiteRT-LM 0.19.1, the
latest release (PyPI and GitHub), refuses these files on the CPU (`moe node #85 unsupported activation='silu'`) and on
the GPU (`moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`), and the two LiteRT pull
requests with SiLU, #11192 and #11188, were open. Every check below ran on the build described there.

## Files

| file | what it does |
|---|---|
| `template/qwen3_coder_litertlm.jinja` | the chat template the files carry (sha256 `3cdac6ef…`). |
| `make_pbtext.py` | writes `template/qwen3_coder_litertlm.pbtext` from the template, the values above and the `tool_call_config` of `template/qwen3_5_official_dab2e708.pbtext`, then parses it back and compares field by field (litert-lm-builder 0.19.0 or newer). |
| `template/qwen3_coder_litertlm.pbtext` | the LlmMetadata of both files as text (`max_num_tokens` 4096; `make_bundle.py --max-tokens` sets 2048 for the other file). |
| `template/qwen3_5_official_dab2e708.pbtext` | LiteRT-LM's `models/qwen3_5/LlmMetadataProto.pbtext` at `dab2e708`, the source of `tool_call_config`. |

## Steps

The Python steps ran with Python 3.12, torch 2.14.1, safetensors 0.8.0, numpy 2.5.4, ai-edge-quantizer 0.10.0,
flatbuffers 25.12.19, protobuf 7.36.2 and litert-lm-builder 0.19.0. `Q` is `../qwen3_30b_a3b_work`.

1. The checkpoint (61.1 GB):

   ```bash
   hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --local-dir SNAPSHOT
   ```

2. Pack the weights and check the scales, the writer, and the decoders: steps 2 to 4 of `Q/README.md` with this
   `SNAPSHOT`:

   ```bash
   python Q/tapi_pack_moe.py SNAPSHOT PACK --experts xb4 --block 32 --fc int8 --lm-head int8
   python Q/scale_scan.py PACK scale_scan.json
   W=LiteRT/bazel-bin/tensor/examples/qwen3moe/qwen3moe_writer    # built as in Q/README.md step 3
   $W --weights_dir=PACK --out=model_kv4096.tflite --prefill=128,1024 --cache=4096 --moe_fusion_guard
   $W --weights_dir=PACK --out=model_kv2048.tflite --prefill=128,1024 --cache=2048 --moe_fusion_guard
   ```

3. The bundles, with this directory's metadata and the checkpoint's `tokenizer.json` (`make_bundle.py` checks its
   sha256; the default is Qwen3-30B-A3B's):

   ```bash
   T=19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d
   python Q/make_bundle.py --tflite model_kv4096.tflite --tokenizer SNAPSHOT/tokenizer.json --tokenizer-sha256 $T \
     --pbtext template/qwen3_coder_litertlm.pbtext --max-tokens 4096 --out qwen3_coder_30b_a3b_mixed_int4_ekv4096.litertlm
   python Q/make_bundle.py --tflite model_kv2048.tflite --tokenizer SNAPSHOT/tokenizer.json --tokenizer-sha256 $T \
     --pbtext template/qwen3_coder_litertlm.pbtext --max-tokens 2048 --out qwen3_coder_30b_a3b_mixed_int4_ekv2048.litertlm
   python Q/bundle_sections.py qwen3_coder_30b_a3b_mixed_int4_ekv4096.litertlm
   ```

## How it was checked

Every model run used the runtime build above (`litert_lm_advanced_main`) on an Apple M4 Max (128 GB), greedy
(`--sampler_top_k=1`), and on the GPU (WebGPU) `--sampler_handles_input=false`.

- **The chat template, before any model run.** 8 conversations (one turn; a code prompt with a fenced block; a system
  message; three turns; tools; a system message and tools; a tool call and its result; two calls and two results),
  each with and without the generation prompt: the template gives the text that the source's template gives through
  transformers' `apply_chat_template`, with string content and with typed parts, and LiteRT-LM 0.19.1's own renderer
  gives the same text (16 of 16).
- **The real weights, first two layers.** Layers 0-1 + final norm + lm_head with fp32 experts, four prompts through
  the decode signature (162 positions) and both prefill signatures (83 positions each), against Transformers in fp32:
  the CPU picks Transformers' top-1 at every position with a largest logit difference of 2.5e-5, and the router picks
  the same 8 experts at every position of both layers; WebGPU (fp32), every position, 4.0e-5. Quantized as shipped,
  the two layers on WebGPU equal Transformers run on the dequantized weights at every position of the decode
  signature (4.2e-5).
- **The reference.** Transformers with the bf16 weights, computed in fp32 on the CPU, greedy, with
  `repetition_penalty` 1.0: the checkpoint's `generation_config.json` sets 1.05, transformers applies it in greedy
  decoding too, and LiteRT-LM's metadata has no such setting. Its greedy replies to three questions, teacher-forced
  through Transformers' own module, give the same top-1 at every position (largest logit difference 1.1e-4).
- **The whole model.** 12 questions, each with " Answer briefly." appended: the 8 short questions of this repository's
  `scripts/verify_quality.py` and four code tasks judged by running the reply's code (`is_palindrome` with edge
  cases, `fib(n)` and `fib(10)`, a bug fix in a given function, a SQL query run on a small table), and a 3-turn
  conversation whose third turn asks for a change to the function of the first answer (its code is run). Up to 2,048
  tokens per answer. Both files on WebGPU and the 4,096-token file on the CPU give Transformers' verdict on 12 of 12
  questions, end every answer with `<|im_end|>`, and pass the 3-turn check. On WebGPU the 2,048-token file's answers
  and turns are the 4,096-token file's byte for byte. The text differs in places between WebGPU, the CPU and
  Transformers (on the CPU the int8 layers' inputs are quantized on the fly); the verdicts do not.
- **Tool calls.** Two prompts rendered by the template with a tool list (the source card's `square_the_number`
  example, "square the number 1024", and a `get_weather` tool with a location and a unit), greedy: both files on WebGPU
  and the 4,096-token file on the CPU write the call that Transformers writes, byte for byte, and the
  `tool_call_config` patterns read the function name and the arguments out of it. LiteRT-LM's CLI takes no tools,
  so the rendered text was given as is (`--use_session`); the runtime's own parsing of a call was not run.
- **The default GPU sampler path.** Without `--sampler_handles_input=false` (the CLI's default, where the GPU sampler
  writes the next decode's inputs), the four code questions and the three turns ran twice on the 4,096-token file: the
  verdicts were the same, the second run's text was the flagged run's byte for byte, and in the first run the reply to
  the bug-fix question took another (working) fix. The flagged runs gave one text each time.

Speeds: the model card.
