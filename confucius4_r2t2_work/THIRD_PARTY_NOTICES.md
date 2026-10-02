# Third-party notices

Two scripts in this directory contain code from other projects. All three projects are licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). The copied code is unchanged apart from omitted docstrings; the copyright lines below are the ones in the source files.

| Project | Version | Source file | Copied into | What was copied | Copyright |
|---|---|---|---|---|---|
| [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) | commit 26d55a54 | `r2t2/r2t2_asr.py` l.34-130 | `r3_stream.py` | `_EN2ZH_PUNCT`, `_ZH2EN_PUNCT`, `_ALL_PUNCT_PAT`, `_normalize_punct_by_context`, `parse_language_output`; `streaming_transcribe` / `finish_streaming_transcribe` (l.302-579) are re-implemented step by step with the engine call replaced | Copyright 2026 The NetEase Youdao team |
| [qwen-asr](https://github.com/Qwen/Qwen3-ASR) (PyPI `qwen-asr`) | 0.0.6 | `qwen_asr/inference/utils.py` | `r3_stream.py` | `_ASR_TEXT_TAG`, `_LANG_PREFIX`, `normalize_language_name`, `detect_and_fix_repetitions`, `parse_asr_output` | Copyright 2026 The Alibaba Qwen team |
| [litert-torch](https://github.com/google-ai-edge/litert-torch) | commit 731ef0a | `litert_torch/generative/export_hf/model_ext/qwen3/qwen3_asr.py` l.106-134 | `export_bundle_r3.py` | the body of `Qwen3AsrEncoder.forward`, as `AudioEncoderR3.encoder_body`, without the prompt-embedding concat | Copyright 2026 The LiteRT Torch Authors |

The model weights are not covered here: the converted model is distributed under the NetEase Youdao Model Use License Agreement (see the model card).
