# litert-community .litertlm audit

Generated 2026-11-02 by `tools/community_audit/audit.py` (version 1) from https://huggingface.co, anonymous reads of the bundle headers only (HTTP ranges, no weights).
Repos 155 (each read at the commit in its row; sha256 of the sorted repo@commit list `9dc91de0159ceb665810678edfbfaec9571a1504472623515fce9e14ab389f22`), files 332, headers read 305, failed 27, repos not read 0, transient read errors 0 (HTTP 429 / 5xx / network: a table with any is not complete; read again with --pin).
Downloads = the Hub's count when the repo list was read.

## Rules

- `rebuild:inline-weights` — LLM file without a TFLiteWeights section: every weight sits inside the TFLite model sections
- `rebuild:no-executor-metadata` — LLM file without an ExecutorMetadata section (state buffers, attention mask); LiteRT-LM v0.19.0 reads the section when it is there and runs without it
- `rebuild:cpu-only` — the backend_constraint of the main model section names cpu and not gpu
- `rebuild:tool-calls-as-text` — the template reads `tools`, the model type is generic_model and the file has no tool_call_config: LiteRT-LM v0.19.0's generic processor then has no tool-call fence (generic_data_processor.h CodeFenceStart), so a call arrives as text
- `rebuild:thinking-without-channel` — the template reads enable_thinking, the file declares no channel: with thinking on, the thought arrives inside the answer
- `warn:think-tag-without-channel` — the template writes <think> but does not read enable_thinking, the file declares no channel (an empty think block of a thinking-off template also counts)
- `warn:header-past-end-offset` — the header flatbuffer reads past the header end offset written in the file; the audit read it from the bytes after that offset
- `warn:sampler-top-k` — sampler type TOP_K: LiteRT-LM v0.19.0's sampler factory (runtime/components/sampler_factory.cc) has cases for TYPE_UNSPECIFIED and TOP_P only
- `warn:sampler-p0` — sampler TOP_P with p = 0: one token kept, greedy
- `warn:vision-over-2gib` — a tf_lite_vision_encoder section over 2 GiB: LiteRT-LM v0.19.0's ModelInfo and litert_lm_loaded_file_create refuse the file (the engine loads it)
- `warn:no-jinja-template` — no jinja template (prompt affixes only): no `tools`, no `enable_thinking`
- `warn:template-does-not-parse` — the jinja template does not parse in jinja2 (the runtime uses minijinja)
- `warn:source-eos-not-a-stop-token` — an eos_token_id of the base model's generation_config / config is not among the file's stop token ids
- `warn:source-template-has-tools` — the base model's chat template reads `tools`, the file's does not
- `warn:source-template-thinks` — the base model's chat template reads enable_thinking or writes <think>, the file declares no channel
- `warn:source-has-vision` — the base model's config has a vision_config, the file has no vision section
- `info:npu` — the main model section is constrained to npu (an ahead-of-time NPU file)

## Summary

| count | of |
|---:|---|
| 41 | embedding files |
| 243 | llm files |
| 2 | other files |
| 4 | speech2text files |
| 2 | text2image files |
| 13 | text2speech files |
| 41 | LLM files with external weights |
| 119 | LLM files with ExecutorMetadata |
| 56 | LLM files with a vision section |
| 18 | LLM files with an audio section |
| 121 | LLM files whose template reads `tools` |
| 111 | LLM files that declare a channel |
| 2 | `info:npu` |
| 1 | `rebuild:cpu-only` |
| 200 | `rebuild:inline-weights` |
| 124 | `rebuild:no-executor-metadata` |
| 9 | `rebuild:thinking-without-channel` |
| 38 | `rebuild:tool-calls-as-text` |
| 2 | `warn:header-past-end-offset` |
| 29 | `warn:no-jinja-template` |
| 3 | `warn:source-eos-not-a-stop-token` |
| 34 | `warn:source-has-vision` |
| 69 | `warn:source-template-has-tools` |
| 46 | `warn:source-template-thinks` |
| 23 | `warn:think-tag-without-channel` |

## Worked example: litert-community/MiniCPM-V-4

Commit `2d05ab8722280bd1f5e654c6a1a5e39c0e2bfc8c`; base_model: openbmb/MiniCPM-V-4.

### MiniCPM-V-4-int8.litertlm

- size 4,214,021,104 B, sha256 `d45ef0cd8141597d731d002b72a152ab1a0f8b9cda8d9a0f5e1bc3a4689dcbd0`
- sections: LlmMetadataProto 0.00 GB; HF_Tokenizer_Zlib 0.00 GB; TFLiteModel tf_lite_prefill_decode [backend_constraint cpu] 3.45 GB; TFLiteModel tf_lite_embedder [backend_constraint cpu] 0.19 GB; TFLiteModel tf_lite_vision_encoder [backend_constraint cpu] 0.58 GB
- weights: inline; executor metadata: none
- model type minicpmv4 {}; max_num_tokens 1024; min_runtime_version —
- template: {"form": "jinja", "bytes": 425, "parses": true, "reads_tools": false, "reads_enable_thinking": false}; channels []; thinking control never (no thought channel)
- sampler null; stop tokens [{"token_ids": {"ids": [2]}}, {"token_ids": {"ids": [73440]}}]
- flags: rebuild:inline-weights, rebuild:no-executor-metadata, rebuild:cpu-only
- run, cpu (Mac, LiteRT-LM 0.19.0): engine pass, text pass, image pass, tool fail (no call: the template does not read tools, so the model gets no tool list)
- run, gpu (Mac, LiteRT-LM 0.19.0): engine error (RuntimeError: litert_lm_engine_create failed with status INVALID_ARGUMENT (3): INVALID_ARGUMENT: Main backend constraint mismatch. Model requires one of [cpu] but Main backend is GPU); 3 checks skipped (no engine)
- run, gpu_vcpu (Mac, LiteRT-LM 0.19.0): engine error (RuntimeError: litert_lm_engine_create failed with status INVALID_ARGUMENT (3): INVALID_ARGUMENT: Main backend constraint mismatch. Model requires one of [cpu] but Main backend is GPU); 3 checks skipped (no engine)

Base model openbmb/MiniCPM-V-4: {"sha": "6f69a8235885be89608f3a71ffd40695b379e1e7", "status": "ok", "eos_token_ids": [2, 73440], "max_position_embeddings": 32768, "has_chat_template": true, "template_reads_tools": false, "template_thinks": false, "has_vision_config": true}

## Run check (Mac, LiteRT-LM 0.19.0)

Files: the top 8 LLM repos by Hub downloads in audit.json (gated repos and files whose header did not read left out), one file per repo (not NPU-constrained; a file without a backend constraint first, then the smallest), plus litert-community/MiniCPM-V-4, litert-community/Qwen3.5-4B:Qwen3.5-4B_mixed_int4.litertlm by name. Each check in a new process. Host: Darwin arm64.

| repo | file | backend | engine | text | image | tool | think_on | think_off |
|---|---|---|---|---|---|---|---|---|
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it.litertlm | cpu | pass | pass | pass | pass | pass | pass |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it.litertlm | gpu | pass | pass | pass | pass | pass | pass |
| litert-community/gemma-4-E4B-it-litert-lm | gemma-4-E4B-it.litertlm | cpu | pass | pass | pass | pass | pass | pass |
| litert-community/gemma-4-E4B-it-litert-lm | gemma-4-E4B-it.litertlm | gpu | pass | pass | pass | pass | pass | pass |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm | cpu | pass | pass | — | pass | — | — |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm | gpu | pass | pass | — | pass | — | — |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm | cpu | pass | pass | — | fail | — | — |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm | gpu | pass | pass | — | fail | — | — |
| litert-community/Qwen3-0.6B | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm | cpu | pass | pass | — | pass | pass | pass |
| litert-community/Qwen3-0.6B | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm | gpu | pass | pass | — | pass | pass | pass |
| litert-community/Phi-4-mini-instruct | Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm | cpu | pass | pass | — | fail | — | — |
| litert-community/Phi-4-mini-instruct | Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm | gpu | pass | pass | — | fail | — | — |
| litert-community/Qwen3-1.7B | Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm | cpu | pass | pass | — | pass | pass | pass |
| litert-community/Qwen3-1.7B | Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm | gpu | pass | pass | — | pass | pass | pass |
| litert-community/gemma-4-12B-it-litert-lm | gemma-4-12B-it.litertlm | cpu | pass | pass | pass | pass | pass | pass |
| litert-community/gemma-4-12B-it-litert-lm | gemma-4-12B-it.litertlm | gpu | pass | pass | pass | pass | pass | pass |
| litert-community/MiniCPM-V-4 | MiniCPM-V-4-int8.litertlm | cpu | pass | pass | pass | fail | — | — |
| litert-community/MiniCPM-V-4 | MiniCPM-V-4-int8.litertlm | gpu | error | skip | skip | skip | — | — |
| litert-community/MiniCPM-V-4 | MiniCPM-V-4-int8.litertlm | gpu_vcpu | error | skip | skip | skip | — | — |
| litert-community/Qwen3.5-4B | Qwen3.5-4B_mixed_int4.litertlm | cpu | pass | pass | — | fail | pass | pass |
| litert-community/Qwen3.5-4B | Qwen3.5-4B_mixed_int4.litertlm | gpu | pass | pass | — | fail | pass | pass |

Header against run: each engine or check that did not pass, with what the header says about it, and each LiteRT-LM ModelInfo field (input_modalities vision and audio, max_context_tokens) that differs from the header:

- litert-community/DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm) cpu tool: fail — header: template reads tools: False, model type not set; run: 'no call: the template does not read tools, so the model gets no tool list'
- litert-community/DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm) gpu tool: fail — header: template reads tools: False, model type not set; run: 'no call: the template does not read tools, so the model gets no tool list'
- litert-community/MiniCPM-V-4 (MiniCPM-V-4-int8.litertlm) cpu tool: fail — header: template reads tools: False, model type minicpmv4; run: 'no call: the template does not read tools, so the model gets no tool list'
- litert-community/MiniCPM-V-4 (MiniCPM-V-4-int8.litertlm) gpu: no engine — header: backend constraint cpu; run: RuntimeError: litert_lm_engine_create failed with status INVALID_ARGUMENT (3): INVALID_ARGUMENT: Main backend constraint mismatch. Model requires one of [cpu] but Main backend is GPU
- litert-community/MiniCPM-V-4 (MiniCPM-V-4-int8.litertlm) gpu_vcpu: no engine — header: backend constraint cpu; run: RuntimeError: litert_lm_engine_create failed with status INVALID_ARGUMENT (3): INVALID_ARGUMENT: Main backend constraint mismatch. Model requires one of [cpu] but Main backend is GPU
- litert-community/Phi-4-mini-instruct (Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm) cpu tool: fail — header: template reads tools: False, model type not set; run: 'no call: the template does not read tools, so the model gets no tool list'
- litert-community/Phi-4-mini-instruct (Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm) gpu tool: fail — header: template reads tools: False, model type not set; run: 'no call: the template does not read tools, so the model gets no tool list'
- litert-community/Qwen3.5-4B (Qwen3.5-4B_mixed_int4.litertlm) cpu tool: fail — header: template reads tools: True, model type generic_model; run: 'the call arrived as text'
- litert-community/Qwen3.5-4B (Qwen3.5-4B_mixed_int4.litertlm) gpu tool: fail — header: template reads tools: True, model type generic_model; run: 'the call arrived as text'

## Rebuild list (LLM files with a `rebuild:` flag)

| repo | file | why |
|---|---|---|
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it-web.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_Google_Tensor_G5.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_Google_Tensor_G6.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_NPU_Uncompiled.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_Qualcomm_QCS8275.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_intel_LNL.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm | gemma-4-E2B-it_intel_PTL.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E4B-it-litert-lm | gemma-4-E4B-it-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E4B-it-litert-lm | gemma-4-E4B-it-web.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-E4B-it-litert-lm | gemma-4-E4B-it.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_f32_ekv4096.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm | inline-weights, no-executor-metadata |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-0.6B | Qwen3-0.6B.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-0.6B | Qwen3-0.6B.mediatek.mt6993.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-0.6B | qwen3_0_6b_mixed_int4.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/Phi-4-mini-instruct | Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-1.7B | Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm | inline-weights |
| litert-community/gemma-4-12B-it-litert-lm | gemma-4-12B-it-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-12B-it-litert-lm | gemma-4-12B-it-web.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-12B-it-litert-lm | gemma-4-12B-it.litertlm | inline-weights |
| litert-community/gemma-4-12B-it-litert-lm | gemma4-12B-it-Intel_PTL_NPU.litertlm | inline-weights |
| litert-community/gemma-4-31B-it-litert-lm | gemma-4-31B-it-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-31B-it-litert-lm | gemma-4-31B-it-web.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-4B | qwen3_4b_channelwise_int8_float32kv.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-1.7B | bonsai-1.7b-int2pc-32k-crashfix.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-1.7B | bonsai-1.7b-int2pc-32k-mp-crashfix.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-1.7B | bonsai-1.7b-int2pc-4k-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-1.7B | bonsai-1.7b-v2max-sdpa-fp16-int2all-32k-fixed2.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-1.7B | bonsai-1.7b-v2sdpa-int2all-32k-fixed2.litertlm | inline-weights, no-executor-metadata |
| litert-community/SmolLM2-360M-Instruct | SmolLM2_360M_instruct.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3.5-2B | Qwen3.5-2B-VL_int8.litertlm | inline-weights |
| litert-community/Qwen3.5-2B | Qwen3.5-2B_int8.litertlm | inline-weights |
| litert-community/Qwen3.5-4B | Qwen3.5-4B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Qwen3.5-4B | Qwen3.5-4B_mixed_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/gemma-4-26B-A4B-it-litert-lm | gemma-4-26B-A4B-it-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-26B-A4B-it-litert-lm | gemma-4-26B-A4B-it-web.litertlm | inline-weights, no-executor-metadata |
| litert-community/gemma-4-26B-A4B-it-litert-lm | gemma4-26B-A4B-it-Intel_PTL_NPU.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int4.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int4_gpu.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int8.litertlm | inline-weights |
| litert-community/Qwen3-0.6B-int4 | qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-0.6B-int4 | qwen3_0.6b_q4_block32_ekv1280.litertlm | inline-weights, no-executor-metadata, tool-calls-as-text, thinking-without-channel |
| litert-community/Qwen2-VL-2B | Qwen2-VL-2B.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3.5-0.8B | Qwen3.5-0.8B-VL_int8.litertlm | inline-weights |
| litert-community/Qwen3.5-0.8B | Qwen3.5-0.8B_int8.litertlm | inline-weights |
| litert-community/Qwen3-4B-Instruct-2507 | qwen3_4b_instruct_2507_mixed_int4.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen2.5-Coder-3B-Instruct | Qwen2.5_Coder_3B_It.litertlm | inline-weights, no-executor-metadata |
| litert-community/SmolLM2-135M-Instruct | SmolLM2_135M_Instruct.litertlm | inline-weights, no-executor-metadata |
| litert-community/SmolLM3-3B | SmolLM3-3B.litertlm | no-executor-metadata |
| litert-community/SmolLM3-3B | SmolLM3-3B_q4_block32_ekv4096.litertlm | no-executor-metadata |
| litert-community/Qwen2-0.5B-Instruct | Qwen2_0.5B_Instruct.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-14B | qwen3_14b_channelwise_int8_float32kv.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/Qwen3-14B | qwen3_14b_mixed_int4.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/Ministral-3-3B-Instruct-2512 | Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm | no-executor-metadata |
| litert-community/Qwen3-8B | qwen3_8b_channelwise_int8_float32kv.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/Qwen3-8B | qwen3_8b_mixed_int4.litertlm | inline-weights, no-executor-metadata, thinking-without-channel |
| litert-community/Gemma3-1B-IT-Tensor-NPU | Gemma3-1B-IT_int4hadamard_aot_ekv4096_G3.litertlm | inline-weights |
| litert-community/Gemma3-1B-IT-Tensor-NPU | Gemma3-1B-IT_int4hadamard_aot_ekv4096_G4.litertlm | inline-weights |
| litert-community/DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-7B_q4_block32_ekv4096.litertlm | no-executor-metadata |
| litert-community/VibeThinker-3B | model.litertlm | no-executor-metadata |
| litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int4.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int4_gpu.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int8.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int8_gpu.litertlm | inline-weights |
| litert-community/FastVLM-0.5B | FastVLM-0.5B.litertlm | inline-weights, no-executor-metadata |
| litert-community/FastVLM-0.5B | FastVLM-0.5B.qualcomm.sm8750.litertlm | inline-weights, no-executor-metadata |
| litert-community/FastVLM-0.5B | FastVLM-0.5B.qualcomm.sm8850.litertlm | inline-weights, no-executor-metadata |
| litert-community/Spark-X2.5-4B | Spark-X2.5-4B_int4.litertlm | no-executor-metadata |
| litert-community/Spark-X2.5-4B | Spark-X2.5-4B_int8.litertlm | no-executor-metadata |
| litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int4.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int4_gpu.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int8.litertlm | inline-weights |
| litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int8_gpu.litertlm | inline-weights |
| litert-community/Qwen3-ASR-0.6B | qwen3_asr_0.6b_5s_i8.litertlm | inline-weights |
| litert-community/SmolVLM2-500M | SmolVLM2-500M.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3-4B-Thinking-2507 | model.litertlm | no-executor-metadata |
| litert-community/Hy-MT2-1.8B | Hy-MT2-1.8B_int8.litertlm | no-executor-metadata |
| litert-community/Qwen3-ASR-1.7B | Qwen3-ASR-1.7B-15s.litertlm | inline-weights |
| litert-community/Qwen3-ASR-1.7B | Qwen3-ASR-1.7B.litertlm | inline-weights |
| litert-community/LFM2.5-230M | LFM2.5-230M_int4.litertlm | inline-weights |
| litert-community/LFM2.5-230M | LFM2.5-230M_int8.litertlm | inline-weights |
| litert-community/LFM2.5-2.6B | LFM2.5-2.6B_int4.litertlm | inline-weights |
| litert-community/Phi-4-mini-reasoning | model.litertlm | no-executor-metadata |
| litert-community/Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct_int4.litertlm | no-executor-metadata |
| litert-community/granite-4.0-350m-litert-lm | granite-4.0-350m_q8_ekv1280.litertlm | inline-weights, no-executor-metadata, tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B | NVIDIA-Nemotron-Labs-Dory1.0-4B_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B | NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm | inline-weights, tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B | NVIDIA-Nemotron-Labs-Dory1.0-4B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/LFM2.5-VL-450M | LFM2.5-VL-450M_int4.litertlm | inline-weights |
| litert-community/LFM2.5-VL-450M | LFM2.5-VL-450M_int4_fixB.litertlm | inline-weights |
| litert-community/LFM3-1.7B-LiteRT | LFM3-1.7B_int8_ekv2048.litertlm | inline-weights, tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT | LFM3-1.7B_int8_ekv4096.litertlm | inline-weights, tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT | LFM3-1.7B_mixed_int4_ekv2048.litertlm | inline-weights, tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT | LFM3-1.7B_mixed_int4_ekv4096.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Shieldstral-1.0-3B | Shieldstral-1.0-3B-vision_int4.litertlm | inline-weights, no-executor-metadata |
| litert-community/Shieldstral-1.0-3B | Shieldstral-1.0-3B-vision_int4_gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/Shieldstral-1.0-3B | Shieldstral-1.0-3B_int4.litertlm | no-executor-metadata |
| litert-community/Shieldstral-1.0-3B | Shieldstral-1.0-3B_int8.litertlm | no-executor-metadata |
| litert-community/granite-4.0-h-350m | granite-4.0-h-350m_fp16.litertlm | inline-weights |
| litert-community/granite-4.0-h-350m | granite-4.0-h-350m_int8.litertlm | inline-weights |
| litert-community/granite-4.0-h-350m | granite-4.0-h-350m_int8_gpu.litertlm | inline-weights |
| litert-community/Falcon-H1-3B-Instruct | Falcon-H1-3B-Instruct_int8.litertlm | inline-weights |
| litert-community/LFM2.5-VL-1.6B | LFM2.5-VL-1.6B_int4.litertlm | inline-weights |
| litert-community/LFM2.5-VL-1.6B | LFM2.5-VL-1.6B_int4_fixB.litertlm | inline-weights |
| litert-community/LFM2.5-VL-1.6B | LFM2.5-VL-1.6B_int8.litertlm | inline-weights |
| litert-community/Spark-X2.5-1.7B | Spark-X2.5-1.7B_int4.litertlm | no-executor-metadata |
| litert-community/Spark-X2.5-1.7B | Spark-X2.5-1.7B_int8.litertlm | no-executor-metadata |
| litert-community/Nemotron-3-Nano-4B | Nemotron-3-Nano-4B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/decider-2b-vision-LiteRT | decider-2b-vision_fp16-int8vocab.litertlm | inline-weights |
| litert-community/decider-2b-vision-LiteRT | decider-2b-vision_fp16.litertlm | inline-weights |
| litert-community/decider-2b-vision-LiteRT | decider-2b-vision_int8.litertlm | inline-weights |
| litert-community/granite-4.2-3b | granite-4.2-3b_int4.litertlm | no-executor-metadata, tool-calls-as-text |
| litert-community/granite-4.2-3b | granite-4.2-3b_int8.litertlm | no-executor-metadata, tool-calls-as-text |
| litert-community/MiniCPM-V-4.7-1B | MiniCPM-V-4.7-1B-VL_int8.litertlm | inline-weights |
| litert-community/MiniCPM-V-4.7-1B | MiniCPM-V-4.7-1B-VL_int8vis.litertlm | inline-weights |
| litert-community/MiniCPM-V-4.7-1B | MiniCPM-V-4.7-1B_int8.litertlm | inline-weights |
| litert-community/LFM2.5-VL-3B | LFM2.5-VL-3B_int4.litertlm | inline-weights |
| litert-community/LFM2.5-VL-3B | LFM2.5-VL-3B_int4_fixB.litertlm | inline-weights |
| litert-community/LFM2.5-VL-3B | LFM2.5-VL-3B_int8.litertlm | inline-weights |
| litert-community/LLaVA-OneVision-0.5B | LLaVA-OneVision-0.5B.litertlm | inline-weights, no-executor-metadata |
| litert-community/Nanbeige4.2-3B | model.litertlm | inline-weights, no-executor-metadata |
| litert-community/Nanbeige4.2-3B | model_fp32act.litertlm | no-executor-metadata |
| litert-community/SmolLM2-1.7B-Instruct | SmolLM2-1_7B-Instruct_dynamic_wi8_afp32.litertlm | inline-weights, no-executor-metadata |
| litert-community/PaddleOCR-VL-1.6 | PaddleOCR-VL-1.6.litertlm | inline-weights, no-executor-metadata |
| litert-community/sarashina2.2-0.5b-instruct-v0.1 | sarashina2.2-0.5b-instruct-v0.1_int4.litertlm | no-executor-metadata |
| litert-community/sarashina2.2-0.5b-instruct-v0.1 | sarashina2.2-0.5b-instruct-v0.1_int8.litertlm | no-executor-metadata |
| litert-community/North-Micro-Vision-Instruct | North-Micro-Vision-Instruct_int4.litertlm | inline-weights, no-executor-metadata |
| litert-community/North-Micro-Vision-Instruct | North-Micro-Vision-Instruct_wi8.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen3.5-9B | Qwen3.5-9B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Qwen3.5-9B | Qwen3.5-9B_mixed_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Agents-A1-4B | Agents-A1-4B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Agents-A1-4B | Agents-A1-4B_mixed_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Jan-nano | model.litertlm | no-executor-metadata |
| litert-community/Jan-nano | model_block32.litertlm | no-executor-metadata |
| litert-community/FastContext-1.0-4B-SFT | model.litertlm | no-executor-metadata |
| litert-community/FastContext-1.0-4B-SFT | model_block128.litertlm | no-executor-metadata |
| litert-community/sarashina2.2-1b-instruct-v0.1 | sarashina2.2-1b-instruct-v0.1_int4.litertlm | no-executor-metadata |
| litert-community/sarashina2.2-1b-instruct-v0.1 | sarashina2.2-1b-instruct-v0.1_int8.litertlm | no-executor-metadata |
| litert-community/InternVL3_5-1B | model.litertlm | inline-weights, no-executor-metadata |
| litert-community/granite-4.1-3b | granite-4.1-3b_int4.litertlm | no-executor-metadata |
| litert-community/granite-4.1-3b | granite-4.1-3b_int8.litertlm | no-executor-metadata |
| litert-community/Ovis2.5-2B | Ovis2.5-2B.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-8B | bonsai-8b-int2pc-32k-crashfix.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-8B | bonsai-8b-int2pc-4k-gpu.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ternary-Bonsai-8B | bonsai-8b-sdpa-fp16-int2all-32k-fixed2.litertlm | inline-weights, no-executor-metadata |
| litert-community/InternVL3-2B | InternVL3-2B.litertlm | inline-weights, no-executor-metadata |
| litert-community/VibeVoice-ASR-Streaming-1.5B | VibeVoice-ASR-Streaming-1.5B.litertlm | inline-weights, no-executor-metadata |
| litert-community/VibeVoice-ASR-Streaming-1.5B | VibeVoice-ASR-Streaming-1.5B_int8.litertlm | inline-weights, no-executor-metadata |
| litert-community/OLMo-2-1B-Instruct | OLMo-2-1B-Instruct_q4_block32_ekv4096.litertlm | no-executor-metadata |
| litert-community/LFM2.5-350M | LFM2.5-350M_int4.litertlm | inline-weights |
| litert-community/LFM2.5-350M | LFM2.5-350M_int8.litertlm | inline-weights |
| litert-community/InternVL3-1B | InternVL3-1B.litertlm | inline-weights, no-executor-metadata |
| litert-community/LFM2-700M | LFM2-700M_int4.litertlm | inline-weights |
| litert-community/granite-4.0-h-1b | granite-4.0-h-1b_int8.litertlm | inline-weights |
| litert-community/decider-0.8b-LiteRT | decider-0.8b_fp16.litertlm | inline-weights |
| litert-community/decider-0.8b-LiteRT | decider-0.8b_int8.litertlm | inline-weights |
| litert-community/Mage-VL | Mage-VL.litertlm | inline-weights, no-executor-metadata |
| litert-community/VibeVoice-ASR-BitNet | VibeVoice-ASR-BitNet.litertlm | inline-weights, no-executor-metadata |
| litert-community/SmolVLM2-2.2B | SmolVLM2-2.2B.litertlm | inline-weights, no-executor-metadata |
| litert-community/Fun-ASR-Nano-2512 | Fun-ASR-Nano-2512.litertlm | no-executor-metadata |
| litert-community/VibeThinker-1.5B | VibeThinker-1.5B.litertlm | inline-weights, no-executor-metadata |
| litert-community/NVIDIA-Nemotron-Nano-9B-v2 | NVIDIA-Nemotron-Nano-9B-v2_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Nano-9B-v2 | NVIDIA-Nemotron-Nano-9B-v2_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/InternVL3_5-2B | model.litertlm | inline-weights, no-executor-metadata |
| litert-community/InternVL3_5-4B | model.litertlm | inline-weights, no-executor-metadata |
| litert-community/G9v3-3B | G9v3-3B_int4.litertlm | tool-calls-as-text |
| litert-community/G9v3-3B | G9v3-3B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Nanbeige4.1-3B | model.litertlm | no-executor-metadata |
| litert-community/Ministral-3-3B-Reasoning-2512 | model.litertlm | no-executor-metadata |
| litert-community/Polaris-4B-Preview | model.litertlm | no-executor-metadata |
| litert-community/TinySwallow-1.5B-Instruct | TinySwallow-1.5B-Instruct.litertlm | inline-weights, no-executor-metadata |
| litert-community/LFM2-2.6B | LFM2-2.6B_int4.litertlm | inline-weights |
| litert-community/LFM2-2.6B | LFM2-2.6B_int8.litertlm | inline-weights |
| litert-community/Ministral-3-8B-Instruct-2512 | Ministral-3-8B-Instruct-2512_int4.litertlm | inline-weights |
| litert-community/Ministral-3-8B-Instruct-2512 | Ministral-3-8B-Instruct-2512_int8.litertlm | inline-weights |
| litert-community/LFM3-Audio-2B-LiteRT | lfm3-audio-2b_int4_b128_c2048_h30.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Qwen2-1.5B-Instruct | Qwen2_1.5B_Instruct.litertlm | inline-weights, no-executor-metadata |
| litert-community/granite-4.0-micro | granite-4.0-micro_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/granite-4.0-micro | granite-4.0-micro_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Falcon-H1-Tiny-R-0.6B | Falcon-H1-Tiny-R-0.6B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/granite-docling-258M | granite-docling-258M.litertlm | inline-weights, no-executor-metadata |
| litert-community/Zamba2-1.2B-instruct | Zamba2-1.2B-instruct_int8.litertlm | inline-weights |
| litert-community/MiniCPM-OCR-1.0 | MiniCPM-OCR-1.0-VL_int8.litertlm | inline-weights |
| litert-community/MiniCPM-OCR-1.0 | MiniCPM-OCR-1.0-VL_int8vis.litertlm | inline-weights |
| litert-community/WeVisOne-VL-2B | WeVisOne-VL-2B_f16.litertlm | inline-weights |
| litert-community/WeVisOne-VL-2B | WeVisOne-VL-2B_int8.litertlm | inline-weights |
| litert-community/granite-4.1-8b | granite-4.1-8b_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/granite-4.1-8b | granite-4.1-8b_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/LFM2-2.6B-Exp | LFM2-2.6B-Exp_int4.litertlm | inline-weights |
| litert-community/LFM2-2.6B-Exp | LFM2-2.6B-Exp_int8.litertlm | inline-weights |
| litert-community/Falcon-H1-0.5B-Instruct | Falcon-H1-0.5B-Instruct_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B | NVIDIA-Nemotron-3.5-Bolt-E4B_int4.litertlm | inline-weights |
| litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B | NVIDIA-Nemotron-3.5-Bolt-E4B_qualcomm_sm8850.litertlm | inline-weights, no-executor-metadata |
| litert-community/LFM3-VL-2B-LiteRT | LFM3-VL-2B_f16.litertlm | inline-weights |
| litert-community/LFM3-VL-2B-LiteRT | LFM3-VL-2B_int4.litertlm | inline-weights |
| litert-community/LFM3-VL-2B-LiteRT | LFM3-VL-2B_int4_ctx4096.litertlm | inline-weights |
| litert-community/Falcon-H1-1.5B-Deep-Instruct | Falcon-H1-1.5B-Deep-Instruct_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Falcon-H1-1.5B-Instruct | Falcon-H1-1.5B-Instruct_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Olmo-3-7B-Think | Olmo-3-7B-Think_int4.litertlm | inline-weights |
| litert-community/Olmo-3-7B-Think | Olmo-3-7B-Think_int4_c2048.litertlm | inline-weights |
| litert-community/Olmo-3-7B-Think | Olmo-3-7B-Think_int8.litertlm | inline-weights |
| litert-community/Falcon-H1R-7B | Falcon-H1R-7B_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Falcon-H1R-7B | Falcon-H1R-7B_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Spark-X2.6-4B | Spark-X2.6-4B_int4.litertlm | inline-weights, no-executor-metadata |
| litert-community/Spark-X2.6-4B | Spark-X2.6-4B_int8-kv2048.litertlm | inline-weights, no-executor-metadata |
| litert-community/Spark-X2.6-4B | Spark-X2.6-4B_int8.litertlm | inline-weights, no-executor-metadata |
| litert-community/LFM3-Omni-2B-LiteRT | LFM3-Omni-2B_int8.litertlm | inline-weights, tool-calls-as-text, thinking-without-channel |
| litert-community/Zamba2-2.7B-instruct | Zamba2-2.7B-instruct_int8.litertlm | inline-weights |
| litert-community/Ornith-1.5-9B | Ornith-1.5-9B_int8.litertlm | inline-weights |
| litert-community/Ornith-1.5-9B | Ornith-1.5-9B_mixed_int4.litertlm | inline-weights |
| litert-community/Qwen3.8-27B | Qwen3.8-27B_mixed_int4.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Phi-3-mini-4k-instruct | phi3_mini_4k_instruct_int8.litertlm | inline-weights, no-executor-metadata |
| litert-community/Ornith-1.0-9B | Ornith-1.0-9B_int8.litertlm | inline-weights |
| litert-community/Ornith-1.0-9B | Ornith-1.0-9B_mixed_int4.litertlm | inline-weights |
| litert-community/Spark-X2.6-1.7B | Spark-X2.6-1.7B_int4-mlp.litertlm | inline-weights, no-executor-metadata |
| litert-community/Spark-X2.6-1.7B | Spark-X2.6-1.7B_int8.litertlm | inline-weights, no-executor-metadata |
| litert-community/1Bit-Bonsai-1.7B-PoC | bonsai-1.7b-int1-REAL.litertlm | inline-weights, no-executor-metadata |
| litert-community/Gemma3-270m-q8_S_NPU | gemma3-270m-qat-Q8_S_sm8550.litertlm | inline-weights, no-executor-metadata |
| litert-community/granite-3.0-1b-a400m-instruct | granite-3.0-1b-a400m-instruct_int8.litertlm | inline-weights, tool-calls-as-text |
| litert-community/Muse-Glimmer-30B-LiteRT | Muse-Glimmer-30B_q4_ekv2048.litertlm | inline-weights, no-executor-metadata |
| litert-community/Qwen2.5-VL-72B-Instruct-LiteRT | Qwen2.5-VL-72B-Instruct_q4_ekv2048.litertlm | inline-weights, no-executor-metadata |
| litert-community/MiniCPM-V-4 | MiniCPM-V-4-int8.litertlm | inline-weights, no-executor-metadata, cpu-only |
| litert-community/MiniCPM5-1B | MiniCPM5-1B_dynamic_wi8_afp32.litertlm | inline-weights |
| litert-community/MiniCPM5-1B | minicpm_wi4b32_wi8_afp32.litertlm | inline-weights, no-executor-metadata |
| litert-community/MiniCPM5-1B | minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm | inline-weights, no-executor-metadata |
| litert-community/MiniCPM5-2B | MiniCPM5-2B_int4.litertlm | inline-weights, no-executor-metadata |
| litert-community/MiniCPM5-2B | MiniCPM5-2B_int8.litertlm | inline-weights |
| litert-community/MiniCPM5-2B | minicpm_wi4c_wi8_afp32.litertlm | inline-weights, tool-calls-as-text |
| litert-community/MiniCPM5-2B | minicpm_wi8_afp32.litertlm | inline-weights, tool-calls-as-text |
| litert-community/codegemma-7b-it-int4-litertlm | codegemma-7b-it-int4-litertlm.litertlm | inline-weights, no-executor-metadata |
| litert-community/functiongemma-mobile-actions_q8_ekv1024.litertlm | mobile-actions_q8_ekv1024.litertlm | inline-weights, no-executor-metadata |

## All files

| repo @ commit | downloads | file | kind | weights | executor metadata | V/A | model type | template | thinking | sampler | max tokens | flags |
|---|---:|---|---|---|---|---|---|---|---|---|---:|---|
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it-gpu.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it-web.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_Google_Tensor_G5.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | in template | — | — | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_Google_Tensor_G6.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | in template | — | — | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_NPU_Uncompiled.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_Qualcomm_QCS8275.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_intel_LNL.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/gemma-4-E2B-it-litert-lm @ 1caa527c | 1379018 | gemma-4-E2B-it_intel_PTL.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/gemma-4-E4B-it-litert-lm @ e6a3892a | 464993 | gemma-4-E4B-it-gpu.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-E4B-it-litert-lm @ e6a3892a | 464993 | gemma-4-E4B-it-web.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-E4B-it-litert-lm @ e6a3892a | 464993 | gemma-4-E4B-it.litertlm | llm | inline | — | VA | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m.litertlm | embedding | external (6 sec, 0.45 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Google_Tensor_G5.litertlm | embedding | inline | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Google_Tensor_G6.litertlm | embedding | inline | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Intel_PTL.litertlm | embedding | inline | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_MediaTek_MT6991.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_MediaTek_MT6993.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_NPU_Uncompiled.litertlm | embedding | inline | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Qualcomm_QCS8275.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Qualcomm_SM8550.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Qualcomm_SM8650.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Qualcomm_SM8750.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-740m-litert-lm @ 94408f31 | 168437 | embeddinggemma-2-740m_Qualcomm_SM8850.litertlm | embedding | external (4 sec, 0.16 GB) | — | VA | — | — | — | — | — | — |
| litert-community/Qwen2.5-1.5B-Instruct @ 2f23bacf | 89411 | Qwen2.5-1.5B-Instruct_multi-prefill-seq_f32_ekv4096.litertlm | llm | inline | — | — | — | affixes | — | k40 p0.95 T1.0 | — | inline-weights no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/Qwen2.5-1.5B-Instruct @ 2f23bacf | 89411 | Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm | llm | inline | — | — | qwen2p5 | jinja+tools | — | TOP_P k40 p0.95 T1.0 | 4096 | inline-weights no-executor-metadata |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B @ 9fd87554 | 70513 | DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm | llm | inline | — | — | — | affixes | — | k64 p0.95 T1.0 | — | inline-weights no-executor-metadata no-jinja-template source-template-thinks |
| litert-community/Qwen3-0.6B @ 2670e142 | 57670 | Qwen3-0.6B.litertlm | llm | inline | — | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | inline-weights no-executor-metadata |
| litert-community/Qwen3-0.6B @ 2670e142 | 57670 | Qwen3-0.6B.mediatek.mt6993.litertlm | llm | inline | — | — | — | affixes | — | — | 4096 | inline-weights no-executor-metadata no-jinja-template source-template-has-tools source-template-thinks |
| litert-community/Qwen3-0.6B @ 2670e142 | 57670 | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm | llm | external (1 sec, 0.30 GB) | max_history=32771 state:global_key_cachex28,global_value_cachex28 | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | — |
| litert-community/Qwen3-0.6B @ 2670e142 | 57670 | qwen3_0_6b_mixed_int4.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.95 T0.6 | 2048 | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/Phi-4-mini-instruct @ edafb144 | 20101 | Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm | llm | inline | — | — | — | affixes | — | k40 p0.95 T1.0 | — | inline-weights no-executor-metadata no-jinja-template source-eos-not-a-stop-token source-template-has-tools |
| litert-community/Qwen3-1.7B @ ebb4eef4 | 17070 | Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm | llm | inline | max_history=32771 state:global_key_cachex28,global_value_cachex28 | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | inline-weights |
| litert-community/Qwen3-1.7B @ ebb4eef4 | 17070 | Qwen3_1.7B.litertlm | llm | external (1 sec, 1.72 GB) | max_history=32771 state:global_key_cachex28,global_value_cachex28 | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m.litertlm | embedding | external (2 sec, 0.14 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Google_Tensor_G5.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Google_Tensor_G6.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Intel_PTL.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_MediaTek_MT6991.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_MediaTek_MT6993.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Qualcomm_QCS8275.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Qualcomm_SM8550.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Qualcomm_SM8650.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Qualcomm_SM8750.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-270m-litert-lm @ 9be6e8b9 | 16991 | embeddinggemma-2-text-270m_Qualcomm_SM8850.litertlm | embedding | external (1 sec, 0.07 GB) | — | — | — | — | — | — | — | — |
| litert-community/gemma-4-12B-it-litert-lm @ ac25717e | 14901 | gemma-4-12B-it-gpu.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-12B-it-litert-lm @ ac25717e | 14901 | gemma-4-12B-it-web.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-12B-it-litert-lm @ ac25717e | 14901 | gemma-4-12B-it.litertlm | llm | inline | max_history=131071 state:global_key_cachex8,global_value_cachex8,local_key_cachex40,local_value_cachex40 mask:type=causal,local_type=vision_bidirectional,sliding_window_size=1024 | VA | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | 131071 | inline-weights |
| litert-community/gemma-4-12B-it-litert-lm @ ac25717e | 14901 | gemma4-12B-it-Intel_PTL_NPU.litertlm | llm | inline | max_history=4096 state:global_key_cachex8,global_value_cachex8,local_key_cachex40,local_value_cachex40 | — | gemma4 | jinja+tools | thought/switch | — | 32766 | inline-weights source-has-vision |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m.litertlm | embedding | external (4 sec, 0.35 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Google_Tensor_G5.litertlm | embedding | inline | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Google_Tensor_G6.litertlm | embedding | inline | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Intel_PTL.litertlm | embedding | inline | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_MediaTek_MT6991.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_MediaTek_MT6993.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Qualcomm_QCS8275.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Qualcomm_SM8550.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Qualcomm_SM8650.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Qualcomm_SM8750.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/embeddinggemma-2-text-vision-440m-litert-lm @ e301f74d | 14412 | embeddinggemma-2-text-vision-440m_Qualcomm_SM8850.litertlm | embedding | external (2 sec, 0.07 GB) | — | V | — | — | — | — | — | — |
| litert-community/gemma-4-31B-it-litert-lm @ 99617231 | 12443 | gemma-4-31B-it-gpu.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-31B-it-litert-lm @ 99617231 | 12443 | gemma-4-31B-it-web.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/FLUX.2-klein-4B-LiteRT @ 223f4c98 | 12221 | flux2-klein-4b.litertlm | text2image | inline | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_mt6991.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_mt6993.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_sm8550.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_sm8650.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q4_ekv1280_sm8850.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G5.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G6.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Gemma3-1B-IT @ d448835d | 11362 | gemma3-1b-it-int4.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Qwen3-4B @ 965e091f | 10894 | qwen3_4b_channelwise_int8_float32kv.litertlm | llm | inline | — | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | inline-weights no-executor-metadata |
| litert-community/Qwen3-4B @ 965e091f | 10894 | qwen3_4b_mixed_int4.litertlm | llm | external (1 sec, 2.01 GB) | max_history=32771 state:global_key_cachex36,global_value_cachex36 | — | qwen3 | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.6 | 4096 | — |
| litert-community/parakeet-tdt-0.6b-v3 @ 1f78e5eb | 10349 | parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm | speech2text | inline | — | — | — | — | — | — | — | — |
| litert-community/Ternary-Bonsai-1.7B @ 990a1e0a | 7959 | bonsai-1.7b-int2pc-32k-crashfix.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-1.7B @ 990a1e0a | 7959 | bonsai-1.7b-int2pc-32k-mp-crashfix.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-1.7B @ 990a1e0a | 7959 | bonsai-1.7b-int2pc-4k-gpu.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 4096 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-1.7B @ 990a1e0a | 7959 | bonsai-1.7b-v2max-sdpa-fp16-int2all-32k-fixed2.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-1.7B @ 990a1e0a | 7959 | bonsai-1.7b-v2sdpa-int2all-32k-fixed2.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/SmolLM2-360M-Instruct @ 507c99cf | 6901 | SmolLM2_360M_instruct.litertlm | llm | inline | — | — | — | affixes | — | — | 8192 | inline-weights no-executor-metadata no-jinja-template |
| litert-community/whisper-tiny @ d0662bbc | 6237 | whisper_tiny_30s_i8.litertlm | speech2text | inline | — | — | — | — | — | — | — | — |
| litert-community/Bonsai-Image-ternary-4B @ 317d577f | 5938 | bonsai-flux2-4b.litertlm | text2image | inline | — | — | — | — | — | — | — | — |
| litert-community/Qwen3.5-2B @ a2877bbd | 5134 | Qwen3.5-2B-VL_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/Qwen3.5-2B @ a2877bbd | 5134 | Qwen3.5-2B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks source-has-vision |
| litert-community/Kokoro-82M @ 33ad69fe | 5117 | kokoro_int8_fp16.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/Kokoro-82M @ 33ad69fe | 5117 | kokoro_int8_gpu_fp16.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/Qwen3.5-4B @ 3e4011ec | 5088 | Qwen3.5-4B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/Qwen3.5-4B @ 3e4011ec | 5088 | Qwen3.5-4B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/gemma-4-26B-A4B-it-litert-lm @ 8584bbd2 | 5003 | gemma-4-26B-A4B-it-gpu.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | TOP_P k64 p0.95 T1.0 | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-26B-A4B-it-litert-lm @ 8584bbd2 | 5003 | gemma-4-26B-A4B-it-web.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | — | inline-weights no-executor-metadata source-has-vision |
| litert-community/gemma-4-26B-A4B-it-litert-lm @ 8584bbd2 | 5003 | gemma4-26B-A4B-it-Intel_PTL_NPU.litertlm | llm | inline | max_history=32768 state:global_key_cachex5,global_value_cachex5,local_key_cachex25,local_value_cachex25 | — | gemma4 | jinja+tools | thought/switch | — | 32768 | inline-weights source-has-vision |
| litert-community/LFM2.5-1.2B-Instruct @ a7a19073 | 4955 | LFM2.5-1.2B-Instruct_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-Instruct @ a7a19073 | 4955 | LFM2.5-1.2B-Instruct_int4_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-Instruct @ a7a19073 | 4955 | LFM2.5-1.2B-Instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/Qwen3-0.6B-int4 @ 256f3304 | 4357 | qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm | llm | inline | — | — | generic_model | affixes | — | — | 1280 | inline-weights no-executor-metadata no-jinja-template source-template-has-tools source-template-thinks |
| litert-community/Qwen3-0.6B-int4 @ 256f3304 | 4357 | qwen3_0.6b_q4_block32_ekv1280.litertlm | llm | inline | — | — | generic_model | jinja+tools | in template | — | 1280 | inline-weights no-executor-metadata tool-calls-as-text thinking-without-channel source-template-thinks |
| litert-community/moonshine-tiny @ eb73d789 | 4281 | moonshine_tiny_5s_i8.litertlm | speech2text | inline | — | — | — | — | — | — | — | — |
| litert-community/Qwen2-VL-2B @ 26697042 | 4206 | Qwen2-VL-2B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Qwen3.5-0.8B @ 0edf7e8c | 4012 | Qwen3.5-0.8B-VL_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/Qwen3.5-0.8B @ 0edf7e8c | 4012 | Qwen3.5-0.8B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks source-has-vision |
| litert-community/Qwen3-4B-Instruct-2507 @ 4d409771 | 3450 | qwen3_4b_instruct_2507_mixed_int4.litertlm | llm | inline | — | — | qwen3 | jinja+tools | — | TOP_P k20 p0.8 T0.7 | 2048 | inline-weights no-executor-metadata |
| litert-community/Qwen2.5-Coder-3B-Instruct @ f02778f1 | 3283 | Qwen2.5_Coder_3B_It.litertlm | llm | inline | — | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/SmolLM2-135M-Instruct @ 8111e0a6 | 3149 | SmolLM2_135M_Instruct.litertlm | llm | inline | — | — | generic_model | affixes | — | — | 4096 | inline-weights no-executor-metadata no-jinja-template |
| litert-community/SmolLM3-3B @ 7a5ec4ff | 3015 | SmolLM3-3B.litertlm | llm | external (1 sec, 3.07 GB) | — | — | — | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/SmolLM3-3B @ 7a5ec4ff | 3015 | SmolLM3-3B_q4_block32_ekv4096.litertlm | llm | external (2 sec, 1.80 GB) | — | — | generic_model | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Qwen2-0.5B-Instruct @ 13aab3e5 | 2930 | Qwen2_0.5B_Instruct.litertlm | llm | inline | — | — | qwen2p5 | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Qwen3-14B @ e4122fd3 | 2670 | qwen3_14b_channelwise_int8_float32kv.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.95 T0.6 | 4096 | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/Qwen3-14B @ e4122fd3 | 2670 | qwen3_14b_mixed_int4.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.95 T0.6 | 2048 | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/Ministral-3-3B-Instruct-2512 @ c26821f1 | 2449 | Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm | llm | external (2 sec, 2.12 GB) | — | — | generic_model | jinja | — | — | 4096 | no-executor-metadata source-template-has-tools source-has-vision |
| litert-community/Qwen3-8B @ 71ff7055 | 2242 | qwen3_8b_channelwise_int8_float32kv.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.95 T0.6 | 4096 | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/Qwen3-8B @ 71ff7055 | 2242 | qwen3_8b_mixed_int4.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.95 T0.6 | 2048 | inline-weights no-executor-metadata thinking-without-channel source-template-thinks |
| litert-community/Gemma3-1B-IT-Tensor-NPU @ c8d7806d | 2065 | Gemma3-1B-IT_int4hadamard_aot_ekv4096_G3.litertlm | llm | inline | max_history=4096 state:global_key_cachex26,global_value_cachex26 | — | generic_model | jinja | — | — | 4096 | inline-weights |
| litert-community/Gemma3-1B-IT-Tensor-NPU @ c8d7806d | 2065 | Gemma3-1B-IT_int4hadamard_aot_ekv4096_G4.litertlm | llm | inline | max_history=4096 state:global_key_cachex26,global_value_cachex26 | — | generic_model | jinja | — | — | 4096 | inline-weights |
| litert-community/Gemma3-1B-IT-Tensor-NPU @ c8d7806d | 2065 | Gemma3-1B-IT_int8_aot_ekv4096_G5.litertlm | llm | inline | max_history=4096 state:global_key_cachex26,global_value_cachex26 | — | generic_model | jinja | — | — | 4096 | — |
| litert-community/Gemma3-1B-IT-Tensor-NPU @ c8d7806d | 2065 | Gemma3-1B-IT_int8_aot_ekv4096_G6.litertlm | llm | inline | max_history=4096 state:global_key_cachex26,global_value_cachex26 | — | generic_model | jinja | — | — | 4096 | — |
| litert-community/DeepSeek-R1-Distill-Qwen-7B @ c0d6ee9f | 2055 | DeepSeek-R1-Distill-Qwen-7B_q4_block32_ekv4096.litertlm | llm | external (2 sec, 4.08 GB) | — | — | qwen2p5 | affixes | thought/unswitched | — | 4096 | no-executor-metadata no-jinja-template |
| litert-community/VibeThinker-3B @ ae95dbfd | 1813 | model.litertlm | llm | external (2 sec, 1.85 GB) | — | — | qwen2p5 | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/LFM2.5-1.2B-Thinking @ 3bc5d67a | 1749 | LFM2.5-1.2B-Thinking_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking @ 3bc5d67a | 1749 | LFM2.5-1.2B-Thinking_int4_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking @ 3bc5d67a | 1749 | LFM2.5-1.2B-Thinking_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-Thinking @ 3bc5d67a | 1749 | LFM2.5-1.2B-Thinking_int8_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/functiongemma-270m-ft-mobile-actions @ 6d7eb4d4 | 1547 | functiongemma-270m-ft-mobile-actions_Google_Tensor_G5.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/functiongemma-270m-ft-mobile-actions @ 6d7eb4d4 | 1547 | functiongemma-270m-ft-mobile-actions_Google_Tensor_G6.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/functiongemma-270m-ft-mobile-actions @ 6d7eb4d4 | 1547 | mobile_actions_q8_ekv1024.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/FastVLM-0.5B @ 46001324 | 1525 | FastVLM-0.5B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 1280 | inline-weights no-executor-metadata |
| litert-community/FastVLM-0.5B @ 46001324 | 1525 | FastVLM-0.5B.qualcomm.sm8750.litertlm | llm | inline | — | V | fast_vlm | affixes | — | — | 1280 | inline-weights no-executor-metadata no-jinja-template |
| litert-community/FastVLM-0.5B @ 46001324 | 1525 | FastVLM-0.5B.qualcomm.sm8850.litertlm | llm | inline | — | V | fast_vlm | affixes | — | — | 1280 | inline-weights no-executor-metadata no-jinja-template |
| litert-community/Spark-X2.5-4B @ b610483b | 1338 | Spark-X2.5-4B_int4.litertlm | llm | external (2 sec, 2.39 GB) | — | — | generic_model | jinja | thought/switch | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Spark-X2.5-4B @ b610483b | 1338 | Spark-X2.5-4B_int8.litertlm | llm | external (1 sec, 4.11 GB) | — | — | generic_model | jinja | thought/switch | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/LFM2.5-1.2B-JP @ acfb19d1 | 1333 | LFM2.5-1.2B-JP_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-JP @ acfb19d1 | 1333 | LFM2.5-1.2B-JP_int4_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-JP @ acfb19d1 | 1333 | LFM2.5-1.2B-JP_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-1.2B-JP @ acfb19d1 | 1333 | LFM2.5-1.2B-JP_int8_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/parakeet-ctc-0.6b @ e3c5b655 | 1330 | parakeet_ctc_0.6b_5s_i8.litertlm | speech2text | inline | — | A | — | — | — | — | — | — |
| litert-community/Qwen3-ASR-0.6B @ 80384dfb | 1297 | qwen3_asr_0.6b_5s_i8.litertlm | llm | inline | max_history=512 state:global_key_cachex28,global_value_cachex28 | A | qwen3 | jinja | — | — | 512 | inline-weights |
| litert-community/SmolVLM2-500M @ 53203d98 | 1205 | SmolVLM2-500M.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.mediatek.mt6991.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.mediatek.mt6993.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.qualcomm.sm8550.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.qualcomm.sm8650.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.qualcomm.sm8750.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/gemma-3-270m-it @ 9d209327 | 1189 | gemma3-270m-it-q8.qualcomm.sm8850.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Qwen3-4B-Thinking-2507 @ 73ef9c9e | 1118 | Qwen3_4b_thinking_dynamic_wi4b32_afp32.litertlm | llm | external (1 sec, 2.01 GB) | max_history=32771 state:global_key_cachex36,global_value_cachex36 | — | qwen3 | jinja+tools | thought/always | TOP_P k20 p0.95 T0.6 | 4096 | — |
| litert-community/Qwen3-4B-Thinking-2507 @ 73ef9c9e | 1118 | model.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | thought/always | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/Hy-MT2-1.8B @ 8911c3f3 | 1094 | Hy-MT2-1.8B_int8.litertlm | llm | external (1 sec, 1.79 GB) | — | — | generic_model | jinja | — | — | 4096 | no-executor-metadata |
| litert-community/Qwen3-ASR-1.7B @ 42ea558e | 1047 | Qwen3-ASR-1.7B-15s.litertlm | llm | inline | max_history=1024 state:global_key_cachex28,global_value_cachex28 | A | generic_model | jinja | — | — | 1024 | inline-weights |
| litert-community/Qwen3-ASR-1.7B @ 42ea558e | 1047 | Qwen3-ASR-1.7B.litertlm | llm | inline | max_history=1024 state:global_key_cachex28,global_value_cachex28 | A | generic_model | jinja | — | — | 1024 | inline-weights |
| litert-community/LFM2.5-230M @ cf0f724f | 1042 | LFM2.5-230M_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx8 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-230M @ cf0f724f | 1042 | LFM2.5-230M_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx8 | — | lfm2 | jinja+tools | thought/unswitched | — | 4096 | inline-weights |
| litert-community/LFM2.5-2.6B @ aefceae5 | 970 | LFM2.5-2.6B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/always | — | 4096 | inline-weights |
| litert-community/LFM2.5-2.6B @ aefceae5 | 970 | LFM2.5-2.6B_int8.litertlm | llm | external (1 sec, 2.70 GB) | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/always | — | 4096 | — |
| litert-community/Phi-4-mini-reasoning @ 884c748f | 945 | model.litertlm | llm | external (2 sec, 2.53 GB) | — | — | generic_model | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Qwen3-TTS-12Hz-0.6B-Base @ d6ba751e | 945 | qwen3_tts_int4.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/Qwen3-TTS-12Hz-0.6B-Base @ d6ba751e | 945 | qwen3_tts_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/Qwen2.5-Coder-1.5B-Instruct @ 1e5fdc46 | 925 | Qwen2.5-Coder-1.5B-Instruct_int4.litertlm | llm | external (2 sec, 1.01 GB) | — | — | qwen2p5 | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/granite-4.0-350m-litert-lm @ 6f1e9bce | 916 | granite-4.0-350m_q8_ekv1280.litertlm | llm | inline | — | — | generic_model | jinja+tools | — | — | 1280 | inline-weights no-executor-metadata tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B @ 75962da2 | 864 | NVIDIA-Nemotron-Labs-Dory1.0-4B_int4.litertlm | llm | inline | max_history=2048 state:global_key_cachex56,global_value_cachex56 | — | generic_model | jinja+tools | thought/switch | — | 2048 | inline-weights tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B @ 75962da2 | 864 | NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm | llm | inline | max_history=2048 state:global_key_cachex36,global_value_cachex36 | — | generic_model | jinja+tools | thought/switch | — | 2048 | inline-weights tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B @ 75962da2 | 864 | NVIDIA-Nemotron-Labs-Dory1.0-4B_int8.litertlm | llm | inline | max_history=2048 state:global_key_cachex56,global_value_cachex56 | — | generic_model | jinja+tools | thought/switch | — | 2048 | inline-weights tool-calls-as-text |
| litert-community/LFM2.5-VL-450M @ 17581fc1 | 863 | LFM2.5-VL-450M_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/LFM2.5-VL-450M @ 17581fc1 | 863 | LFM2.5-VL-450M_int4_fixB.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/LFM2.5-VL-450M @ 17581fc1 | 863 | LFM2.5-VL-450M_int8.litertlm | llm | external (4 sec, 0.52 GB) | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | source-template-has-tools source-template-thinks |
| litert-community/LFM3-1.7B-LiteRT @ 9bbda264 | 851 | LFM3-1.7B_int8_ekv2048.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | — | generic_model | jinja+tools | thought/switch | TOP_P k50 p1.0 T0.1 | 2048 | inline-weights tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT @ 9bbda264 | 851 | LFM3-1.7B_int8_ekv4096.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | — | generic_model | jinja+tools | thought/switch | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT @ 9bbda264 | 851 | LFM3-1.7B_mixed_int4_ekv2048.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | — | generic_model | jinja+tools | thought/switch | TOP_P k50 p1.0 T0.1 | 2048 | inline-weights tool-calls-as-text |
| litert-community/LFM3-1.7B-LiteRT @ 9bbda264 | 851 | LFM3-1.7B_mixed_int4_ekv4096.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | — | generic_model | jinja+tools | thought/switch | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights tool-calls-as-text |
| litert-community/Shieldstral-1.0-3B @ e19ad2c8 | 845 | Shieldstral-1.0-3B-vision_int4.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Shieldstral-1.0-3B @ e19ad2c8 | 845 | Shieldstral-1.0-3B-vision_int4_gpu.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Shieldstral-1.0-3B @ e19ad2c8 | 845 | Shieldstral-1.0-3B_int4.litertlm | llm | external (2 sec, 2.12 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-has-vision |
| litert-community/Shieldstral-1.0-3B @ e19ad2c8 | 845 | Shieldstral-1.0-3B_int8.litertlm | llm | external (2 sec, 3.83 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-has-vision |
| litert-community/granite-4.0-h-350m @ 17d2d76d | 828 | granite-4.0-h-350m_fp16.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx56 | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights |
| litert-community/granite-4.0-h-350m @ 17d2d76d | 828 | granite-4.0-h-350m_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx56 | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights |
| litert-community/granite-4.0-h-350m @ 17d2d76d | 828 | granite-4.0-h-350m_int8_gpu.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx56 | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights |
| litert-community/Falcon-H1-3B-Instruct @ 8f515999 | 823 | Falcon-H1-3B-Instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex32,global_value_cachex32,linear_attentionx64 | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights |
| litert-community/LFM2.5-VL-1.6B @ 9f72335a | 793 | LFM2.5-VL-1.6B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools |
| litert-community/LFM2.5-VL-1.6B @ 9f72335a | 793 | LFM2.5-VL-1.6B_int4_fixB.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools |
| litert-community/LFM2.5-VL-1.6B @ 9f72335a | 793 | LFM2.5-VL-1.6B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools |
| litert-community/Spark-X2.5-1.7B @ 8ed49e63 | 723 | Spark-X2.5-1.7B_int4.litertlm | llm | external (2 sec, 1.12 GB) | — | — | generic_model | jinja | thought/switch | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Spark-X2.5-1.7B @ 8ed49e63 | 723 | Spark-X2.5-1.7B_int8.litertlm | llm | external (1 sec, 1.71 GB) | — | — | generic_model | jinja | thought/switch | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Nemotron-3-Nano-4B @ 91d59fd2 | 665 | Nemotron-3-Nano-4B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx42 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/decider-2b-vision-LiteRT @ 28796a64 | 653 | decider-2b-vision_fp16-int8vocab.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/decider-2b-vision-LiteRT @ 28796a64 | 653 | decider-2b-vision_fp16.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/decider-2b-vision-LiteRT @ 28796a64 | 653 | decider-2b-vision_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/granite-4.2-3b @ 093d6054 | 639 | granite-4.2-3b_int4.litertlm | llm | external (2 sec, 1.96 GB) | — | — | generic_model | jinja+tools | thought/switch | — | 4096 | no-executor-metadata tool-calls-as-text |
| litert-community/granite-4.2-3b @ 093d6054 | 639 | granite-4.2-3b_int8.litertlm | llm | external (2 sec, 3.66 GB) | — | — | generic_model | jinja+tools | thought/switch | — | 4096 | no-executor-metadata tool-calls-as-text |
| litert-community/MiniCPM-V-4.7-1B @ 1de9c49f | 602 | MiniCPM-V-4.7-1B-VL_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/MiniCPM-V-4.7-1B @ 1de9c49f | 602 | MiniCPM-V-4.7-1B-VL_int8vis.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/MiniCPM-V-4.7-1B @ 1de9c49f | 602 | MiniCPM-V-4.7-1B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks source-has-vision |
| litert-community/LFM2.5-VL-3B @ b907fab1 | 556 | LFM2.5-VL-3B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/LFM2.5-VL-3B @ b907fab1 | 556 | LFM2.5-VL-3B_int4_fixB.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/LFM2.5-VL-3B @ b907fab1 | 556 | LFM2.5-VL-3B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | V | lfm2 | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/LLaVA-OneVision-0.5B @ d83d931f | 533 | LLaVA-OneVision-0.5B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/Nanbeige4.2-3B @ 2ff063fb | 509 | model.litertlm | llm | inline | — | — | generic_model | jinja | thought/always | — | 4096 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/Nanbeige4.2-3B @ 2ff063fb | 509 | model_fp32act.litertlm | llm | external (2 sec, 2.34 GB) | — | — | generic_model | jinja | thought/always | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/SmolLM2-1.7B-Instruct @ c0ee765a | 503 | SmolLM2-1_7B-Instruct_dynamic_wi8_afp32.litertlm | llm | inline | — | — | generic_model | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/PaddleOCR-VL-1.6 @ 51dc0859 | 484 | PaddleOCR-VL-1.6.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/sarashina2.2-0.5b-instruct-v0.1 @ 3cfba6dd | 479 | sarashina2.2-0.5b-instruct-v0.1_int4.litertlm | llm | external (1 sec, 0.46 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/sarashina2.2-0.5b-instruct-v0.1 @ 3cfba6dd | 479 | sarashina2.2-0.5b-instruct-v0.1_int8.litertlm | llm | external (1 sec, 0.79 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/North-Micro-Vision-Instruct @ 539d4e77 | 477 | North-Micro-Vision-Instruct_int4.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/North-Micro-Vision-Instruct @ 539d4e77 | 477 | North-Micro-Vision-Instruct_wi8.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Qwen3.5-9B @ 93903ee7 | 474 | Qwen3.5-9B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/Qwen3.5-9B @ 93903ee7 | 474 | Qwen3.5-9B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/Agents-A1-4B @ 6c44e56f | 460 | Agents-A1-4B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.85 | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/Agents-A1-4B @ 6c44e56f | 460 | Agents-A1-4B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja+tools | thought/switch | TOP_P k20 p0.95 T0.85 | 4096 | inline-weights tool-calls-as-text source-has-vision |
| litert-community/Jan-nano @ 636e011d | 456 | model.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | thought/unswitched | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/Jan-nano @ 636e011d | 456 | model_block32.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | thought/unswitched | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/FastContext-1.0-4B-SFT @ 1412f113 | 442 | model.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | — | — | 4096 | no-executor-metadata no-jinja-template |
| litert-community/FastContext-1.0-4B-SFT @ 1412f113 | 442 | model_block128.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | — | — | 4096 | no-executor-metadata no-jinja-template |
| litert-community/sarashina2.2-1b-instruct-v0.1 @ 6b6fb5e2 | 429 | sarashina2.2-1b-instruct-v0.1_int4.litertlm | llm | external (1 sec, 0.80 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/sarashina2.2-1b-instruct-v0.1 @ 6b6fb5e2 | 429 | sarashina2.2-1b-instruct-v0.1_int8.litertlm | llm | external (1 sec, 1.41 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/InternVL3_5-1B @ a51f2187 | 421 | model.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/granite-4.1-3b @ 89f95d39 | 392 | granite-4.1-3b_int4.litertlm | llm | external (2 sec, 1.96 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/granite-4.1-3b @ 89f95d39 | 392 | granite-4.1-3b_int8.litertlm | llm | external (2 sec, 3.66 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/Ovis2.5-2B @ fed7cb51 | 388 | Ovis2.5-2B.litertlm | llm | inline | — | V | fast_vlm | jinja | thought/unswitched | — | 2048 | inline-weights no-executor-metadata |
| litert-community/Ternary-Bonsai-8B @ 4260ac3c | 374 | bonsai-8b-int2pc-32k-crashfix.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-8B @ 4260ac3c | 374 | bonsai-8b-int2pc-4k-gpu.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 4096 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Ternary-Bonsai-8B @ 4260ac3c | 374 | bonsai-8b-sdpa-fp16-int2all-32k-fixed2.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 32768 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/InternVL3-2B @ 4f65e5a4 | 368 | InternVL3-2B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/VibeVoice-ASR-Streaming-1.5B @ 24a0c9ba | 365 | VibeVoice-ASR-Streaming-1.5B.litertlm | llm | inline | — | A | generic_model | jinja | — | — | 2048 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/VibeVoice-ASR-Streaming-1.5B @ 24a0c9ba | 365 | VibeVoice-ASR-Streaming-1.5B_int8.litertlm | llm | inline | — | A | generic_model | jinja | — | — | 2048 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/OLMo-2-1B-Instruct @ 8ea10f12 | 364 | OLMo-2-1B-Instruct_q4_block32_ekv4096.litertlm | llm | external (1 sec, 0.85 GB) | — | — | generic_model | affixes | — | — | 4096 | no-executor-metadata no-jinja-template |
| litert-community/LFM2.5-350M @ 61a0218b | 352 | LFM2.5-350M_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/LFM2.5-350M @ 61a0218b | 352 | LFM2.5-350M_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/InternVL3-1B @ 875247a4 | 342 | InternVL3-1B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/LFM2-700M @ 67e65f1a | 330 | LFM2-700M_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/LFM2-700M @ 67e65f1a | 330 | LFM2-700M_int8.litertlm | llm | external (1 sec, 0.74 GB) | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx10 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | — |
| litert-community/granite-4.0-h-1b @ eebcee3c | 315 | granite-4.0-h-1b_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx72 | — | qwen2p5 | jinja+tools | — | — | 4096 | inline-weights |
| litert-community/decider-0.8b-LiteRT @ 463fe22e | 313 | decider-0.8b_fp16.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | — | generic_model | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/decider-0.8b-LiteRT @ 463fe22e | 313 | decider-0.8b_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | — | generic_model | jinja | — | — | 4096 | inline-weights source-template-has-tools source-template-thinks |
| litert-community/Mage-VL @ 97e8e2b2 | 301 | Mage-VL.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/VibeVoice-ASR-BitNet @ 193b2d57 | 301 | VibeVoice-ASR-BitNet.litertlm | llm | inline | — | A | generic_model | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/SmolVLM2-2.2B @ 4c87fe05 | 297 | SmolVLM2-2.2B.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/Fun-ASR-Nano-2512 @ a0fbd419 | 268 | Fun-ASR-Nano-2512.litertlm | llm | external (2 sec, 0.75 GB) | — | A | generic_model | jinja | — | — | 2048 | no-executor-metadata |
| litert-community/VibeThinker-1.5B @ 24367f44 | 254 | VibeThinker-1.5B.litertlm | llm | inline | — | — | generic_model | affixes | — | — | 4096 | inline-weights no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/NVIDIA-Nemotron-Nano-9B-v2 @ e37062ec | 251 | NVIDIA-Nemotron-Nano-9B-v2_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx54 | — | generic_model | jinja+tools | thought/always | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-Nano-9B-v2 @ e37062ec | 251 | NVIDIA-Nemotron-Nano-9B-v2_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex4,global_value_cachex4,linear_attentionx54 | — | generic_model | jinja+tools | thought/always | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Llama-3.2-1B @ fd0e16d3 | 246 | llama3_2_1b_mixed_int4_gpu.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/InternVL3_5-2B @ b62b8ba3 | 240 | model.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/InternVL3_5-4B @ d2c10148 | 237 | model.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 2048 | inline-weights no-executor-metadata |
| litert-community/G9v3-3B @ a830f828 | 221 | G9v3-3B_int4.litertlm | llm | external (2 sec, 1.76 GB) | max_history=4096 state:global_key_cachex52,global_value_cachex52 | — | generic_model | jinja+tools | thought/switch | — | 4096 | tool-calls-as-text |
| litert-community/G9v3-3B @ a830f828 | 221 | G9v3-3B_int8.litertlm | llm | inline | max_history=4096 state:global_key_cachex52,global_value_cachex52 | — | generic_model | jinja+tools | thought/switch | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Nanbeige4.1-3B @ 50eee364 | 219 | model.litertlm | llm | external (2 sec, 2.18 GB) | — | — | generic_model | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools |
| litert-community/Ministral-3-3B-Reasoning-2512 @ d6e02745 | 217 | model.litertlm | llm | external (2 sec, 2.12 GB) | — | — | generic_model | jinja | thought/unswitched | — | 4096 | no-executor-metadata source-template-has-tools source-has-vision |
| litert-community/Polaris-4B-Preview @ 8906d918 | 216 | model.litertlm | llm | external (2 sec, 2.40 GB) | — | — | qwen3 | affixes | thought/unswitched | — | 4096 | no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/TinySwallow-1.5B-Instruct @ cf67d4d7 | 205 | TinySwallow-1.5B-Instruct.litertlm | llm | inline | — | — | generic_model | affixes | — | — | 4096 | inline-weights no-executor-metadata no-jinja-template source-template-has-tools |
| litert-community/LFM2-2.6B @ 50a87e39 | 203 | LFM2-2.6B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/LFM2-2.6B @ 50a87e39 | 203 | LFM2-2.6B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/Ministral-3-8B-Instruct-2512 @ 8cd920a7 | 202 | Ministral-3-8B-Instruct-2512_int4.litertlm | llm | inline | max_history=4096 state:global_key_cachex34,global_value_cachex34 | — | generic_model | jinja | — | — | 4096 | inline-weights source-template-has-tools source-has-vision |
| litert-community/Ministral-3-8B-Instruct-2512 @ 8cd920a7 | 202 | Ministral-3-8B-Instruct-2512_int8.litertlm | llm | inline | max_history=4096 state:global_key_cachex34,global_value_cachex34 | — | generic_model | jinja | — | — | 4096 | inline-weights source-template-has-tools source-has-vision |
| litert-community/LFM3-Audio-2B-LiteRT @ 8e14ffaa | 200 | lfm3-audio-2b_int4_b128_c2048_h30.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | A | generic_model | jinja+tools | thought/unswitched | — | 2048 | inline-weights tool-calls-as-text |
| litert-community/Llama-3.2-3B @ 40173512 | 199 | llama3_2_3b_mixed_int4_gpu.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/functiongemma-270m-ft-tiny-garden @ aca35636 | 198 | tiny_garden_q8_ekv1024.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/EmbeddingGemma-300M-Tensor-G4-NPU @ 18d2d8d8 | 195 | EmbeddingGemma-300M_seq256_Google_Tensor_G4.litertlm | other | inline | — | — | — | — | — | — | — | header-past-end-offset |
| litert-community/EmbeddingGemma-300M-Tensor-G4-NPU @ 18d2d8d8 | 195 | EmbeddingGemma-300M_seq512_Google_Tensor_G4.litertlm | other | inline | — | — | — | — | — | — | — | header-past-end-offset |
| litert-community/Qwen2-1.5B-Instruct @ 0292a162 | 195 | Qwen2_1.5B_Instruct.litertlm | llm | inline | — | — | qwen2p5 | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/granite-4.0-micro @ f2d39fb2 | 189 | granite-4.0-micro_int4.litertlm | llm | inline | max_history=4096 state:global_key_cachex40,global_value_cachex40 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/granite-4.0-micro @ f2d39fb2 | 189 | granite-4.0-micro_int8.litertlm | llm | inline | max_history=4096 state:global_key_cachex40,global_value_cachex40 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/LFM2.5-Embedding-350M @ 040bda13 | 184 | LFM2.5-Embedding-350M_fp16.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/LFM2.5-Embedding-350M @ 040bda13 | 184 | LFM2.5-Embedding-350M_wi8fc.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/Falcon-H1-Tiny-R-0.6B @ a0998082 | 177 | Falcon-H1-Tiny-R-0.6B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex44,global_value_cachex44,linear_attentionx88 | — | generic_model | jinja+tools | thought/unswitched | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/granite-docling-258M @ 402edc68 | 175 | granite-docling-258M.litertlm | llm | inline | — | V | fast_vlm | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Zamba2-1.2B-instruct @ c2a2b0d1 | 174 | Zamba2-1.2B-instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx76 | — | generic_model | jinja | — | — | 4096 | inline-weights |
| litert-community/MiniCPM-OCR-1.0 @ b3110eae | 167 | MiniCPM-OCR-1.0-VL_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/MiniCPM-OCR-1.0 @ b3110eae | 167 | MiniCPM-OCR-1.0-VL_int8vis.litertlm | llm | inline | max_history=0 state:global_key_cachex6,global_value_cachex6,linear_attentionx36 | V | fast_vlm | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks |
| litert-community/WeVisOne-VL-2B @ 54c673c8 | 161 | WeVisOne-VL-2B_f16.litertlm | llm | inline | max_history=0 state:global_key_cachex28,global_value_cachex28 | V | fast_vlm | jinja | — | — | 2048 | inline-weights source-template-has-tools |
| litert-community/WeVisOne-VL-2B @ 54c673c8 | 161 | WeVisOne-VL-2B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex28,global_value_cachex28 | V | fast_vlm | jinja | — | — | 2048 | inline-weights source-template-has-tools |
| litert-community/granite-4.1-8b @ 6dac5357 | 161 | granite-4.1-8b_int4.litertlm | llm | inline | max_history=4096 state:global_key_cachex40,global_value_cachex40 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/granite-4.1-8b @ 6dac5357 | 161 | granite-4.1-8b_int8.litertlm | llm | inline | max_history=4096 state:global_key_cachex40,global_value_cachex40 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/LFM2-2.6B-Exp @ 3b4b9796 | 159 | LFM2-2.6B-Exp_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/LFM2-2.6B-Exp @ 3b4b9796 | 159 | LFM2-2.6B-Exp_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx22 | — | lfm2 | jinja+tools | thought/unswitched | TOP_P k50 p1.0 T0.1 | 4096 | inline-weights |
| litert-community/Falcon-H1-0.5B-Instruct @ 21894494 | 156 | Falcon-H1-0.5B-Instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex36,global_value_cachex36,linear_attentionx72 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B @ a98def82 | 156 | NVIDIA-Nemotron-3.5-Bolt-E4B_int4.litertlm | llm | inline | max_history=2048 state:global_key_cachex24,global_value_cachex24 | — | gemma4 | jinja+tools | thought/switch | — | 2048 | inline-weights source-has-vision |
| litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B @ a98def82 | 156 | NVIDIA-Nemotron-3.5-Bolt-E4B_qualcomm_sm8850.litertlm | llm | inline | — | — | gemma4 | jinja+tools | thought/switch | — | 2048 | inline-weights no-executor-metadata source-has-vision |
| litert-community/LFM3-VL-2B-LiteRT @ 58085927 | 151 | LFM3-VL-2B_f16.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | V | fast_vlm | jinja+tools | thought/switch | — | 2048 | inline-weights |
| litert-community/LFM3-VL-2B-LiteRT @ 58085927 | 151 | LFM3-VL-2B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | V | fast_vlm | jinja+tools | thought/switch | — | 2048 | inline-weights |
| litert-community/LFM3-VL-2B-LiteRT @ 58085927 | 151 | LFM3-VL-2B_int4_ctx4096.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | V | fast_vlm | jinja+tools | thought/switch | — | 4096 | inline-weights |
| litert-community/MedGemma-1.5-4B-IT @ abba4da1 | 146 | medgemma-1.5-4b-it_q4_block32_ekv2048.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/MedGemma-1.5-4B-IT @ abba4da1 | 146 | medgemma-1.5-4b-it_q4_block32_vision_ekv2048.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/MedGemma-1.5-4B-IT @ abba4da1 | 146 | medgemma-1.5-4b-it_q8_gpu_ekv2048.litertlm | not read | header read: HTTP 401 (gated repo: an anonymous read is refused) | — | — | — | — | — | — | — | — |
| litert-community/Falcon-H1-1.5B-Deep-Instruct @ 92317a24 | 144 | Falcon-H1-1.5B-Deep-Instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex66,global_value_cachex66,linear_attentionx132 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Falcon-H1-1.5B-Instruct @ f9dea07b | 144 | Falcon-H1-1.5B-Instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex24,global_value_cachex24,linear_attentionx48 | — | generic_model | jinja+tools | — | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Olmo-3-7B-Think @ d3e41240 | 144 | Olmo-3-7B-Think_int4.litertlm | llm | inline | max_history=4096 state:global_key_cachex32,global_value_cachex32 | — | generic_model | jinja | thought/always | — | 4096 | inline-weights |
| litert-community/Olmo-3-7B-Think @ d3e41240 | 144 | Olmo-3-7B-Think_int4_c2048.litertlm | llm | inline | max_history=2048 state:global_key_cachex32,global_value_cachex32 | — | generic_model | jinja | thought/always | — | 2048 | inline-weights |
| litert-community/Olmo-3-7B-Think @ d3e41240 | 144 | Olmo-3-7B-Think_int8.litertlm | llm | inline | max_history=4096 state:global_key_cachex32,global_value_cachex32 | — | generic_model | jinja | thought/always | — | 4096 | inline-weights |
| litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT @ d32b05f4 | 143 | Vela-1.0-Encoder-307M-Embedding_fp16.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/Falcon-H1R-7B @ fc7cab37 | 123 | Falcon-H1R-7B_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex44,global_value_cachex44,linear_attentionx88 | — | generic_model | jinja+tools | thought/unswitched | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Falcon-H1R-7B @ fc7cab37 | 123 | Falcon-H1R-7B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex44,global_value_cachex44,linear_attentionx88 | — | generic_model | jinja+tools | thought/unswitched | — | 4096 | inline-weights tool-calls-as-text |
| litert-community/Spark-X2.6-4B @ f8da7265 | 123 | Spark-X2.6-4B_int4.litertlm | llm | inline | — | — | generic_model | jinja | thought/switch | — | 4096 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/Spark-X2.6-4B @ f8da7265 | 123 | Spark-X2.6-4B_int8-kv2048.litertlm | llm | inline | — | — | generic_model | jinja | thought/switch | — | 2048 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/Spark-X2.6-4B @ f8da7265 | 123 | Spark-X2.6-4B_int8.litertlm | llm | inline | — | — | generic_model | jinja | thought/switch | — | 4096 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/LFM3-Omni-2B-LiteRT @ a287b6e6 | 118 | LFM3-Omni-2B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex11,global_value_cachex11,linear_attentionx13 | VA | generic_model | jinja+tools | in template | — | 4096 | inline-weights tool-calls-as-text thinking-without-channel source-template-thinks |
| litert-community/Zamba2-2.7B-instruct @ 04cacdd4 | 115 | Zamba2-2.7B-instruct_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex9,global_value_cachex9,linear_attentionx108 | — | generic_model | jinja | — | — | 4096 | inline-weights |
| litert-community/Ornith-1.5-9B @ 1e4ad751 | 114 | Ornith-1.5-9B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-eos-not-a-stop-token source-template-has-tools source-template-thinks source-has-vision |
| litert-community/Ornith-1.5-9B @ 1e4ad751 | 114 | Ornith-1.5-9B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-eos-not-a-stop-token source-template-has-tools source-template-thinks source-has-vision |
| litert-community/Qwen3.8-27B @ 309a955f | 114 | Qwen3.8-27B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex16,global_value_cachex16,linear_attentionx96 | — | generic_model | jinja+tools | in template | — | 4096 | inline-weights tool-calls-as-text think-tag-without-channel source-template-thinks source-has-vision |
| litert-community/Nemotron-3-Embed-1B @ 9dd8d53a | 101 | Nemotron-3-Embed-1B_fp16.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/Nemotron-3-Embed-1B @ 9dd8d53a | 101 | Nemotron-3-Embed-1B_wi8fc.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/Phi-3-mini-4k-instruct @ f69f3c22 | 88 | phi3_mini_4k_instruct_int8.litertlm | llm | inline | — | — | generic_model | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/Ornith-1.0-9B @ 25002c9a | 87 | Ornith-1.0-9B_int8.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks source-has-vision |
| litert-community/Ornith-1.0-9B @ 25002c9a | 87 | Ornith-1.0-9B_mixed_int4.litertlm | llm | inline | max_history=0 state:global_key_cachex8,global_value_cachex8,linear_attentionx48 | — | generic_model | jinja | in template | — | 4096 | inline-weights think-tag-without-channel source-template-has-tools source-template-thinks source-has-vision |
| litert-community/granite-embedding-311m-multilingual-r2 @ 1b10683d | 80 | granite-embedding-311m-r2_fp16.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/granite-embedding-311m-multilingual-r2 @ 1b10683d | 80 | granite-embedding-311m-r2_wi8fc.litertlm | embedding | inline | — | — | — | — | — | — | — | — |
| litert-community/Spark-X2.6-1.7B @ bc1ef142 | 68 | Spark-X2.6-1.7B_int4-mlp.litertlm | llm | inline | — | — | generic_model | jinja | thought/switch | — | 4096 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/Spark-X2.6-1.7B @ bc1ef142 | 68 | Spark-X2.6-1.7B_int8.litertlm | llm | inline | — | — | generic_model | jinja | thought/switch | — | 4096 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/1Bit-Bonsai-1.7B-PoC @ 538befef | 58 | bonsai-1.7b-int1-REAL.litertlm | llm | inline | — | — | qwen3 | jinja+tools | in template | TOP_P k20 p0.85 T0.5 | 4096 | inline-weights no-executor-metadata think-tag-without-channel source-template-thinks |
| litert-community/Gemma3-270m-q8_S_NPU @ 78da9274 | 50 | gemma3-270m-qat-Q8_S_sm8550.litertlm | llm | inline | — | — | generic_model | jinja | — | — | 1280 | inline-weights no-executor-metadata |
| litert-community/granite-3.0-1b-a400m-instruct @ 9571313d | 48 | granite-3.0-1b-a400m-instruct_int8.litertlm | llm | inline | max_history=2048 state:global_key_cachex24,global_value_cachex24 | — | generic_model | jinja+tools | — | — | 2048 | inline-weights tool-calls-as-text |
| litert-community/Muse-Glimmer-30B-LiteRT @ 131810b8 | 31 | Muse-Glimmer-30B_q4_ekv2048.litertlm | llm | inline | — | V | fast_vlm | jinja | — | TOP_P k64 p0.95 T1.0 | 2048 | inline-weights no-executor-metadata source-template-has-tools |
| litert-community/Qwen2.5-VL-72B-Instruct-LiteRT @ 7205693f | 27 | Qwen2.5-VL-72B-Instruct_q4_ekv2048.litertlm | llm | inline | — | V | fast_vlm | jinja | — | TOP_P k1 p1.0 T1.0 | 2048 | inline-weights no-executor-metadata |
| litert-community/MiniCPM-V-4 @ 2d05ab87 | 0 | MiniCPM-V-4-int8.litertlm | llm | inline | — | V | minicpmv4 | jinja | — | — | 1024 | inline-weights no-executor-metadata cpu-only |
| litert-community/MiniCPM5-1B @ 4f39a7a6 | 0 | MiniCPM5-1B_dynamic_wi8_afp32.litertlm | llm | inline | max_history=32771 state:global_key_cachex24,global_value_cachex24 | — | minicpm5 | jinja+tools | thought/switch | — | 4096 | inline-weights |
| litert-community/MiniCPM5-1B @ 4f39a7a6 | 0 | minicpm_wi4b32_wi8_afp32.litertlm | llm | inline | — | — | minicpm5 | jinja+tools | thought/switch | — | 1024 | inline-weights no-executor-metadata |
| litert-community/MiniCPM5-1B @ 4f39a7a6 | 0 | minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm | llm | inline | — | — | minicpm5 | jinja+tools | thought/switch | — | 1024 | inline-weights no-executor-metadata |
| litert-community/MiniCPM5-2B @ 2db9e214 | 0 | MiniCPM5-2B_int4.litertlm | llm | inline | — | — | minicpm5 | jinja+tools | thought/switch | — | 4096 | inline-weights no-executor-metadata |
| litert-community/MiniCPM5-2B @ 2db9e214 | 0 | MiniCPM5-2B_int8.litertlm | llm | inline | max_history=32771 state:global_key_cachex42,global_value_cachex42 | — | minicpm5 | jinja+tools | thought/switch | — | 4096 | inline-weights |
| litert-community/MiniCPM5-2B @ 2db9e214 | 0 | minicpm_wi4c_wi8_afp32.litertlm | llm | inline | max_history=8192 state:global_key_cachex42,global_value_cachex42 | — | generic_model | jinja+tools | thought/switch | — | 8192 | inline-weights tool-calls-as-text |
| litert-community/MiniCPM5-2B @ 2db9e214 | 0 | minicpm_wi8_afp32.litertlm | llm | inline | max_history=8192 state:global_key_cachex42,global_value_cachex42 | — | generic_model | jinja+tools | thought/switch | — | 8192 | inline-weights tool-calls-as-text |
| litert-community/codegemma-7b-it-int4-litertlm @ aac11862 | 0 | codegemma-7b-it-int4-litertlm.litertlm | llm | inline | — | — | generic_model | jinja | — | — | 4096 | inline-weights no-executor-metadata |
| litert-community/functiongemma-mobile-actions_q8_ekv1024.litertlm @ 56836180 | 0 | mobile-actions_q8_ekv1024.litertlm | llm | inline | — | — | function_gemma | none | — | — | 1024 | inline-weights no-executor-metadata no-jinja-template |
| litert-community/phonon-31m-en @ 63a61708 | 0 | phonon_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_de_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_en_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_es_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_fr_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_it_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_nl_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |
| litert-community/pocket-tts @ 03f949ec | 0 | pocket_tts_pt_int8.litertlm | text2speech | inline | — | — | — | — | — | — | — | — |

## Failed reads

| repo | file | stage | reason |
|---|---|---|---|
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_mt6991.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_mt6993.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_sm8550.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_sm8650.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q4_ekv1280_sm8850.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G5.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | Gemma3-1B-IT_q8_ekv1280_Google_Tensor_G6.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Gemma3-1B-IT | gemma3-1b-it-int4.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Llama-3.2-1B | llama3_2_1b_mixed_int4_gpu.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/Llama-3.2-3B | llama3_2_3b_mixed_int4_gpu.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/MedGemma-1.5-4B-IT | medgemma-1.5-4b-it_q4_block32_ekv2048.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/MedGemma-1.5-4B-IT | medgemma-1.5-4b-it_q4_block32_vision_ekv2048.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/MedGemma-1.5-4B-IT | medgemma-1.5-4b-it_q8_gpu_ekv2048.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/functiongemma-270m-ft-mobile-actions | functiongemma-270m-ft-mobile-actions_Google_Tensor_G5.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/functiongemma-270m-ft-mobile-actions | functiongemma-270m-ft-mobile-actions_Google_Tensor_G6.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/functiongemma-270m-ft-mobile-actions | mobile_actions_q8_ekv1024.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/functiongemma-270m-ft-tiny-garden | tiny_garden_q8_ekv1024.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.mediatek.mt6991.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.mediatek.mt6993.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.qualcomm.sm8550.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.qualcomm.sm8650.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.qualcomm.sm8750.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
| litert-community/gemma-3-270m-it | gemma3-270m-it-q8.qualcomm.sm8850.litertlm | header read | HTTP 401 (gated repo: an anonymous read is refused) |
