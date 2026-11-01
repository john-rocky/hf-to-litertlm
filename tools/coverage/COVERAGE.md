# Popular open models and their LiteRT-LM conversions in litert-community

Fetched from the Hugging Face Hub API at **2026-11-01T20:03:04Z** (UTC) by `tools/coverage/make_coverage_table.py`; rebuild with `python3 tools/coverage/make_coverage_table.py`.

The top 100 models per pipeline tag (`text-generation`, `image-text-to-text`) by the Hub's 30-day download count, ties ordered by repo id. Left out of the ranking and counted here:
- `text-generation` (167 listed): 8 gguf / mlx repo, 55 quantized re-upload (base_model:quantized), 3 under 10 M parameters
- `image-text-to-text` (291 listed): 7 gguf / mlx repo, 179 quantized re-upload (base_model:quantized), 5 under 10 M parameters

litert-community has 425 repos; 155 hold a `.litertlm` file and 93 ship `litertlm_manifest.json`.

| | `text-generation` | `image-text-to-text` |
|---|---:|---:|
| joined, no gap | 4 | 0 |
| joined, with gaps | 16 | 16 |
| not joined | 80 | 84 |

A model is joined to the repos whose card `base_model` names it; when none does, to a repo whose card declares no `org/name` base_model and whose name matches (marked *(by name)*). GPU = a file whose manifest `backends` lists gpu (verified to generate). Ext. weights = a file with a TFLiteWeights section. Image / audio = the manifest's input flags. A dash = no manifest to read.

## Joined, no gap

| Rank | Model | Downloads (30 d) | Likes | Params (B) | litert-community repo | .litertlm files | GPU | Ext. weights | Image / audio |
|---|---|---:|---:|---:|---|---:|---|---|---|
| text #75 | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 611,561 | 1,072 | 3.075 | [SmolLM3-3B](https://huggingface.co/litert-community/SmolLM3-3B) | 2 | yes | yes | text only |
| text #79 | [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) | 558,905 | 119 | 3.403 | [granite-4.1-3b](https://huggingface.co/litert-community/granite-4.1-3b) | 2 | yes | yes | text only |
| text #93 | [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | 457,095 | 629 | 4.022 | [Qwen3-4B-Thinking-2507](https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507) | 2 | yes | yes | text only |
| text #99 | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 431,969 | 167 | 1.544 | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct) | 1 | yes | yes | text only |

## Joined, with gaps

| Rank | Model | Downloads (30 d) | Likes | Params (B) | litert-community repo | .litertlm files | GPU | Ext. weights | Image / audio | Gaps |
|---|---|---:|---:|---:|---|---:|---|---|---|---|
| text #2 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 23,444,902 | 1,867 | 0.752 | [Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B)<br>[Qwen3-0.6B-int4](https://huggingface.co/litert-community/Qwen3-0.6B-int4) | 6 | — | — | — | no manifest |
| text #3 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 13,002,507 | 529 | 14.768 | [Qwen3-14B](https://huggingface.co/litert-community/Qwen3-14B) | 2 | — | — | — | no manifest |
| text #4 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 8,424,710 | 778 | 4.022 | [Qwen3-4B](https://huggingface.co/litert-community/Qwen3-4B) | 2 | — | — | — | no manifest |
| image+text #1 | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 8,301,351 | 4,262 | 31.273 | [gemma-4-31B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-31B-it-litert-lm) | 2 | — | — | — | no manifest |
| image+text #2 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 7,892,866 | 18,912 | 27.781 | [Qwen3.8-27B](https://huggingface.co/litert-community/Qwen3.8-27B) | 1 | yes | no | text only | no external weights, no image input |
| text #5 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 7,660,335 | 2,160 | 8.191 | [Qwen3-8B](https://huggingface.co/litert-community/Qwen3-8B) | 2 | — | — | — | no manifest |
| image+text #4 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 6,588,471 | 1,175 | 4.66 | [Qwen3.5-4B](https://huggingface.co/litert-community/Qwen3.5-4B) | 2 | yes | no | text only | no external weights, no image input |
| image+text #5 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 6,471,053 | 2,288 | 9.653 | [Qwen3.5-9B](https://huggingface.co/litert-community/Qwen3.5-9B) | 2 | yes | no | text only | no external weights, no image input |
| image+text #7 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 6,032,823 | 1,725 | 25.806 | [gemma-4-26B-A4B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-26B-A4B-it-litert-lm) | 3 | — | — | — | no manifest |
| text #9 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 4,960,320 | 724 | 0.494 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | 0 | — | — | — | no .litertlm file (.task only) |
| text #11 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 4,554,130 | 923 | 1.544 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) | 2 | — | — | — | no manifest |
| text #14 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 4,199,333 | 607 | 2.032 | [Qwen3-1.7B](https://huggingface.co/litert-community/Qwen3-1.7B) | 2 | — | — | — | no manifest |
| image+text #14 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 3,730,405 | 793 | 0.873 | [Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B) | 2 | yes | no | image | no external weights |
| image+text #15 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 3,431,121 | 464 | 2.274 | [Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B) | 2 | yes | no | image | no external weights |
| text #19 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 2,610,894 | 1,041 | 4.022 | [Qwen3-4B-Instruct-2507](https://huggingface.co/litert-community/Qwen3-4B-Instruct-2507) | 1 | — | — | — | no manifest |
| text #25 | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | 1,883,615 | 142 | 3.974 | [Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B) | 1 | yes | no | text only | no external weights |
| text #46 | [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 1,083,600 | 1,831 | 2.517 | [MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) | 4 | yes | no | text only | no external weights |
| image+text #29 | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | 1,033,262 | 529 | 2.209 | [Qwen2-VL-2B](https://huggingface.co/litert-community/Qwen2-VL-2B) | 1 | yes | no | image | no external weights |
| image+text #32 | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 955,800 | 1,612 | 4.3 | [Gemma3-4B-IT](https://huggingface.co/litert-community/Gemma3-4B-IT) | 0 | — | — | — | no .litertlm file (.task only) |
| text #60 | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 929,720 | 1,895 | 1.1 | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/litert-community/TinyLlama-1.1B-Chat-v1.0) | 0 | — | — | — | no .litertlm file (.task only) |
| text #65 | [ornith-ai/Ornith-1.0-9B](https://huggingface.co/ornith-ai/Ornith-1.0-9B) | 800,231 | 566 | 9.41 | [Ornith-1.0-9B](https://huggingface.co/litert-community/Ornith-1.0-9B) | 2 | yes | no | text only | no external weights |
| text #66 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 784,646 | 1,654 | 1.777 | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-1.5B) | 1 | — | — | — | no manifest |
| text #84 | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 483,882 | 2,741 | 1.236 | [Llama-3.2-1B](https://huggingface.co/litert-community/Llama-3.2-1B) | 1 | — | — | — | no manifest |
| text #86 | [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) | 476,015 | 404 | 9.653 | [Ornith-1.5-9B](https://huggingface.co/litert-community/Ornith-1.5-9B) | 2 | yes | no | text only | no external weights |
| text #97 | [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | 438,315 | 153 | 3.086 | [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-3B-Instruct) (by name) | 1 | — | — | — | no manifest |
| image+text #57 | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 355,733 | 899 | 12.187 | [Gemma3-12B-IT](https://huggingface.co/litert-community/Gemma3-12B-IT) | 0 | — | — | — | no .litertlm file (.task only) |
| image+text #58 | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 351,339 | 2,061 | 27.432 | [Gemma3-27B-IT](https://huggingface.co/litert-community/Gemma3-27B-IT) | 0 | — | — | — | no .litertlm file (.task only) |
| image+text #75 | [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) | 222,564 | 411 | 1.554 | [MiniCPM-V-4.7-1B](https://huggingface.co/litert-community/MiniCPM-V-4.7-1B) | 3 | yes | no | image | no external weights |
| image+text #80 | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 195,811 | 2,107 | 29.777 | [Muse-Glimmer-30B-LiteRT](https://huggingface.co/litert-community/Muse-Glimmer-30B-LiteRT) | 1 | yes | no | image | no external weights |
| image+text #86 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 179,086 | 1,092 | 4.3 | [MedGemma-1.5-4B-IT](https://huggingface.co/litert-community/MedGemma-1.5-4B-IT) | 3 | — | — | — | no manifest |
| image+text #93 | [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) | 137,686 | 61 | 0.894 | [LLaVA-OneVision-0.5B](https://huggingface.co/litert-community/LLaVA-OneVision-0.5B) | 1 | yes | no | image | no external weights |
| image+text #98 | [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | 125,090 | 1,292 | 0.258 | [granite-docling-258M](https://huggingface.co/litert-community/granite-docling-258M) | 1 | no | no | image | no GPU row, no external weights |

## Not joined

| Rank | Model | Downloads (30 d) | Likes | Params (B) |
|---|---|---:|---:|---:|
| text #1 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 28,989,033 | 3,909 | 0.137 |
| image+text #3 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 6,902,450 | 1,235 | 8.767 |
| text #6 | [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | 6,580,429 | 1,590 | 685.355 |
| image+text #6 | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 6,329,724 | 3,012 | 321.323 |
| text #7 | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 6,046,280 | 5,260 | 20.915 |
| image+text #8 | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 5,838,999 | 1,762 | 8.292 |
| text #8 | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 5,601,971 | 2,483 | 7.616 |
| image+text #9 | [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | 5,241,149 | 433 | 0.232 |
| text #10 | [dphn/dolphin-2.9.1-yi-1.5-34b](https://huggingface.co/dphn/dolphin-2.9.1-yi-1.5-34b) | 4,778,759 | 90 | 34.389 |
| image+text #10 | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 4,622,082 | 3,064 | 35.952 |
| text #12 | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 4,501,381 | 8,737 | 8.03 |
| text #13 | [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) | 4,444,035 | 333 | ? |
| image+text #11 | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 4,122,291 | 4,936 | 3.336 |
| image+text #12 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4,062,866 | 520 | 4.438 |
| image+text #13 | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 4,029,520 | 6,689 | 180.0 |
| text #15 | [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) | 3,406,696 | 66 | 0.213 |
| text #16 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 3,168,654 | 4,094 | 304.18 |
| text #17 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3,112,986 | 628 | 3.086 |
| text #18 | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 2,934,052 | 791 | 32.762 |
| image+text #16 | [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) | 2,780,133 | 2,950 | 27.781 |
| text #20 | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 2,549,284 | 5,475 | 116.829 |
| text #21 | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | 2,473,851 | 2,361 | 753.33 |
| image+text #17 | [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | 2,429,212 | 1,450 | 1.927 |
| text #22 | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 2,372,787 | 266 | 0.135 |
| image+text #18 | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 2,306,643 | 2,342 | 27.781 |
| image+text #19 | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 2,127,615 | 4,833 | 763.205 |
| image+text #20 | [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) | 2,084,207 | 1,832 | 952.378 |
| image+text #21 | [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 2,055,501 | 3,461 | 3.336 |
| text #23 | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 1,967,786 | 210 | 14.77 |
| text #24 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1,899,093 | 1,931 | 1.236 |
| image+text #22 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 1,812,990 | 1,081 | 27.781 |
| text #26 | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1,763,961 | 840 | 31.578 |
| text #27 | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 1,741,602 | 3,277 | 7.242 |
| image+text #23 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 1,720,099 | 1,550 | 35.952 |
| text #28 | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | 1,712,192 | 14,422 | 684.49 |
| text #29 | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 1,657,256 | 4,461 | 684.49 |
| text #30 | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | 1,615,751 | 2,141 | 753.864 |
| text #31 | [distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2) | 1,554,891 | 666 | 0.088 |
| image+text #24 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 1,541,027 | 742 | 3.755 |
| text #32 | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 1,499,654 | 856 | 7.616 |
| text #33 | [ornith-ai/Ornith-1.0-35B](https://huggingface.co/ornith-ai/Ornith-1.0-35B) | 1,455,824 | 519 | 35.107 |
| text #34 | [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 1,429,742 | 381 | 560.525 |
| image+text #25 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 1,425,456 | 510 | 2.128 |
| image+text #26 | [datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2) | 1,413,672 | 593 | 5.296 |
| image+text #27 | [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) | 1,411,262 | 435 | 265.956 |
| text #35 | [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) | 1,358,450 | 1,529 | 228.69 |
| text #36 | [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | 1,340,948 | 112 | 1.236 |
| text #37 | [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 1,331,157 | 990 | 298.786 |
| text #38 | [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) | 1,328,974 | 649 | 110.469 |
| text #39 | [PrimeIntellect/Qwen3-0.6B](https://huggingface.co/PrimeIntellect/Qwen3-0.6B) | 1,310,251 | 1 | 0.752 |
| text #40 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 1,210,292 | 488 | 0.494 |
| image+text #28 | [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) | 1,191,094 | 144 | 0.686 |
| text #41 | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 1,177,164 | 258 | 32.913 |
| text #42 | [kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) | 1,167,475 | 26 | ? |
| text #43 | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 1,153,587 | 440 | 123.611 |
| text #44 | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) | 1,153,554 | 239 | 30.534 |
| text #45 | [ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) | 1,087,112 | 69 | 19.529 |
| text #47 | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 1,056,691 | 1,314 | 30.532 |
| text #48 | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 1,053,046 | 852 | 30.532 |
| text #49 | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 1,051,869 | 977 | 30.532 |
| text #50 | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 1,051,332 | 2,362 | 290.945 |
| text #51 | [ornith-ai/Ornith-1.5-9B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-9B-NVFP4) | 1,042,536 | 17 | 6.729 |
| text #52 | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 1,031,607 | 2,922 | 3.213 |
| image+text #30 | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 1,026,056 | 652 | 125.086 |
| text #53 | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | 1,018,095 | 488 | 17.82 |
| text #54 | [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) | 1,015,489 | 39 | 0.096 |
| text #55 | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 1,012,092 | 375 | 32.764 |
| text #56 | [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) | 998,572 | 1,852 | 753.864 |
| image+text #31 | [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 980,257 | 1,581 | 403.398 |
| text #57 | [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) | 977,907 | 320 | 33.443 |
| text #58 | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 968,875 | 1,891 | 31.221 |
| text #59 | [poolside/Laguna-M.1](https://huggingface.co/poolside/Laguna-M.1) | 944,336 | 149 | 225.796 |
| text #61 | [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) | 892,465 | 56 | 0.324 |
| text #62 | [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) | 866,019 | 149 | 0.559 |
| text #63 | [openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large) | 844,700 | 371 | 0.812 |
| image+text #33 | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | 841,775 | 510 | 33.453 |
| image+text #34 | [OpenGVLab/InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B) | 828,325 | 84 | 0.938 |
| image+text #35 | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 813,134 | 11,905 | 2779.932 |
| text #64 | [poolside/Laguna-XS-2.1](https://huggingface.co/poolside/Laguna-XS-2.1) | 802,751 | 263 | 33.443 |
| image+text #36 | [Infomaniak-AI/vllm-translategemma-4b-it](https://huggingface.co/Infomaniak-AI/vllm-translategemma-4b-it) | 799,476 | 15 | 4.3 |
| text #67 | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 782,597 | 380 | 14.77 |
| text #68 | [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2) | 761,896 | 36 | ? |
| image+text #37 | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 755,070 | 850 | 1.006 |
| image+text #38 | [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | 724,991 | 843 | 1.12 |
| text #69 | [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) | 702,413 | 80 | 2.249 |
| text #70 | [farbodtavakkoli/OTel-2.0-LLM-31B-IT-QLoRA](https://huggingface.co/farbodtavakkoli/OTel-2.0-LLM-31B-IT-QLoRA) | 688,003 | 3 | 31.273 |
| image+text #39 | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | 681,877 | 134 | 0.873 |
| text #71 | [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) | 654,391 | 1,076 | 117.562 |
| image+text #40 | [OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF) | 649,740 | 10 | 21.233 |
| image+text #41 | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 632,361 | 1,319 | 25.824 |
| text #72 | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | 630,848 | 248 | 1.544 |
| text #73 | [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) | 626,102 | 212 | 79.674 |
| text #74 | [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | 620,279 | 97 | 1.721 |
| image+text #42 | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | 618,901 | 383 | 7.063 |
| image+text #43 | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | 612,504 | 123 | 4.66 |
| image+text #44 | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 609,237 | 1,297 | 8.291 |
| text #76 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 605,719 | 219 | 0.596 |
| image+text #45 | [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | 602,406 | 354 | 2.439 |
| image+text #46 | [raxcore-dev/Rax-4.5](https://huggingface.co/raxcore-dev/Rax-4.5) | 579,589 | 5 | 2.274 |
| text #77 | [Bahushruth/Qwen3.6-35B-A3B-abliterated-v4](https://huggingface.co/Bahushruth/Qwen3.6-35B-A3B-abliterated-v4) | 579,362 | 13 | 34.661 |
| image+text #47 | [Qwen/Qwen3.5-122B-A10B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4) | 561,803 | 61 | 125.086 |
| text #78 | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 560,703 | 2,740 | 8.03 |
| text #80 | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) | 537,269 | 341 | 7.616 |
| text #81 | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 526,033 | 149 | 8.191 |
| text #82 | [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 513,412 | 2,202 | 32.764 |
| text #83 | [ornith-ai/Ornith-1.5-35B-A3B-FP8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8) | 496,905 | 34 | 35.952 |
| text #85 | [meta-llama/Llama-3.1-405B-FP8](https://huggingface.co/meta-llama/Llama-3.1-405B-FP8) | 477,560 | 124 | 405.853 |
| text #87 | [unsloth/Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) | 475,526 | 99 | 3.213 |
| text #88 | [EleutherAI/gpt-neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m) | 472,429 | 231 | 0.15 |
| text #89 | [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | 469,417 | 118 | 4.022 |
| text #90 | [farbodtavakkoli/OTel-2.0-LLM-31B-IT](https://huggingface.co/farbodtavakkoli/OTel-2.0-LLM-31B-IT) | 462,695 | 31 | 31.273 |
| text #91 | [ornith-ai/Ornith-1.5-397B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-397B-NVFP4) | 459,168 | 24 | 209.238 |
| text #92 | [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) | 457,699 | 542 | 8.03 |
| text #94 | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | 453,951 | 456 | 67.229 |
| image+text #48 | [dots-studio/dots.ocr](https://huggingface.co/dots-studio/dots.ocr) | 447,614 | 1,347 | 3.039 |
| text #95 | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 447,605 | 4,229 | 7.242 |
| text #96 | [shibing624/macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese) | 440,593 | 121 | 0.102 |
| image+text #49 | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | 435,648 | 627 | 31.071 |
| text #98 | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 434,134 | 1,725 | 79.674 |
| text #100 | [openbmb/MiniCPM5-2B-MLX](https://huggingface.co/openbmb/MiniCPM5-2B-MLX) | 426,079 | 57 | 2.517 |
| image+text #50 | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) | 422,545 | 1,107 | 35.213 |
| image+text #51 | [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 419,321 | 1,150 | 3.389 |
| image+text #52 | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | 409,850 | 1,132 | 4.3 |
| image+text #53 | [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) | 406,938 | 448 | 3.745 |
| image+text #54 | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 406,716 | 2,898 | 1026.879 |
| image+text #55 | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | 392,354 | 1,885 | 0.777 |
| image+text #56 | [kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4) | 362,241 | 3 | ? |
| image+text #59 | [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) | 349,180 | 3,160 | 3.831 |
| image+text #60 | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | 343,084 | 108 | 2.274 |
| image+text #61 | [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 335,643 | 517 | 3.755 |
| image+text #62 | [Zyphra/ZAYA1-VL-8B](https://huggingface.co/Zyphra/ZAYA1-VL-8B) | 333,629 | 42 | 9.722 |
| image+text #63 | [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | 317,760 | 749 | 4.147 |
| image+text #64 | [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | 304,059 | 139 | 9.653 |
| image+text #65 | [huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated) | 298,926 | 64 | 8.292 |
| image+text #66 | [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) | 298,296 | 429 | 235.67 |
| image+text #67 | [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) | 295,401 | 172 | 4.3 |
| image+text #68 | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | 286,468 | 689 | 2.923 |
| image+text #69 | [zenosai/MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) | 272,998 | 17 | 0.878 |
| image+text #70 | [nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD) | 272,176 | 41 | 7.699 |
| image+text #71 | [Alibaba-DAMO-Academy/RynnBrain1.1-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-2B) | 254,516 | 12 | 2.213 |
| image+text #72 | [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | 249,755 | 1,640 | 1026.879 |
| image+text #73 | [OpenGVLab/InternVL2_5-4B-MPO](https://huggingface.co/OpenGVLab/InternVL2_5-4B-MPO) | 241,506 | 18 | 3.713 |
| image+text #74 | [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | 223,258 | 8 | 0.777 |
| image+text #76 | [meta-models/Muse-Glimmer-30B-assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) | 221,337 | 80 | 2.556 |
| image+text #77 | [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | 215,777 | 962 | 304.647 |
| image+text #78 | [mistral-experimental/pixtral-12b](https://huggingface.co/mistral-experimental/pixtral-12b) | 204,474 | 106 | 12.683 |
| image+text #79 | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 198,424 | 1,248 | 1.3 |
| image+text #81 | [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | 195,416 | 1,600 | 427.04 |
| image+text #82 | [dots-studio/dots.mocr](https://huggingface.co/dots-studio/dots.mocr) | 193,159 | 192 | 3.039 |
| image+text #83 | [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B) | 190,152 | 34 | 10.017 |
| image+text #84 | [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | 189,150 | 315 | 7.567 |
| image+text #85 | [microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) | 185,055 | 154 | 0.232 |
| image+text #87 | [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) | 172,815 | 1,577 | 0.716 |
| image+text #88 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 170,353 | 4,245 | 763.205 |
| image+text #89 | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | 163,424 | 255 | 33.357 |
| image+text #90 | [unsloth/Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B) | 160,309 | 29 | 4.66 |
| image+text #91 | [NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct) | 155,211 | 22 | 2.209 |
| image+text #92 | [unsloth/gemma-3n-E4B-it](https://huggingface.co/unsloth/gemma-3n-E4B-it) | 154,014 | 10 | 7.85 |
| image+text #94 | [OpenGVLab/InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B) | 136,887 | 83 | 2.206 |
| image+text #95 | [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | 125,917 | 234 | 8.767 |
| image+text #96 | [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) | 125,604 | 78 | 321.323 |
| image+text #97 | [ibm-granite/granite-vision-4.1-4b](https://huggingface.co/ibm-granite/granite-vision-4.1-4b) | 125,548 | 114 | 3.997 |
| image+text #99 | [opendatalab/MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | 124,293 | 109 | 1.156 |
| image+text #100 | [nvidia/NVIDIA-Nemotron-Parse-v1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) | 123,959 | 175 | 0.957 |
